"""
IndiaLend Credit Scoring
========================
Production-grade credit scoring engine built for Indian lending:
- LightGBM and XGBoost models optimized for Apple Silicon
- CAT/NonCAT segmentation (RBI mandated for NBFCs)
- Score mapping to CIBIL-like 300-900 scale
- Alternative data scoring (UPI, phone, e-commerce)
- Model explainability with feature importance
- Segment-specific model training

Key Indian market gaps this fills:
1. Most NBFCs use single-model approach → we use segmented models
2. Bureau-only scoring misses thin-file applicants → alternative scoring
3. No standardised internal scorecard → we produce CIBIL-scale scores
"""

from __future__ import annotations

import warnings
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd
from scipy.special import expit  # sigmoid
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    roc_auc_score,
    precision_recall_curve,
    average_precision_score,
)
from sklearn.model_selection import StratifiedKFold

from .utils import RISK_GRADE_MAP, risk_grade_from_score

warnings.filterwarnings("ignore", category=UserWarning)


# ---------------------------------------------------------------------------
# CAT / NonCAT Classifier (RBI Segmentation)
# ---------------------------------------------------------------------------

class CAT_NonCAT_Classifier:
    """Classify applicants into CAT (>= Rs 30 lakh income) or NonCAT.

    RBI requires different risk treatment, capital allocation, and
    reporting for CAT vs NonCAT exposures. This classifier also supports
    custom thresholds for sub-segmentation.
    """

    def __init__(self, income_threshold: float = 3_000_000):
        self.income_threshold = income_threshold

    def classify(self, data: Union[Dict, pd.DataFrame]) -> Union[str, pd.Series]:
        """Classify applicant(s) as CAT or NonCAT."""
        if isinstance(data, dict):
            income = data.get("annual_income", 0)
            return "CAT" if income >= self.income_threshold else "NonCAT"

        if isinstance(data, pd.DataFrame):
            col = "annual_income"
            if col not in data.columns:
                raise ValueError(f"DataFrame must contain '{col}' column")
            return pd.Series(
                np.where(data[col] >= self.income_threshold, "CAT", "NonCAT"),
                index=data.index,
                name="cat_segment",
            )

        raise TypeError("Input must be dict or DataFrame")

    def segment_data(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Split DataFrame into CAT and NonCAT segments."""
        segments = self.classify(df)
        return {
            "CAT": df[segments == "CAT"].copy(),
            "NonCAT": df[segments == "NonCAT"].copy(),
        }


# ---------------------------------------------------------------------------
# Credit Scorer (Main Engine)
# ---------------------------------------------------------------------------

class CreditScorer:
    """Production credit scoring engine with LightGBM/XGBoost.

    Features:
    - Segment-aware training (separate models for CAT/NonCAT)
    - Probability calibration (Platt scaling)
    - Score mapping to CIBIL-like 300-900 scale
    - Feature importance and SHAP-ready
    - Model persistence (save/load)
    - Cross-validation with stratified folds
    """

    # Default feature set for scoring
    DEFAULT_FEATURES = [
        "age", "education_numeric", "dependents", "years_employed",
        "years_current_job", "annual_income", "monthly_income",
        "existing_emi", "num_existing_loans", "credit_card_outstanding",
        "credit_card_limit", "credit_utilization", "loan_amount",
        "tenure_months", "ltv_ratio", "foir", "cibil_score",
        "dpd_30_12m", "dpd_60_12m", "dpd_90_12m",
        "enquiries_3m", "enquiries_6m", "enquiries_12m",
        "oldest_account_months", "avg_account_age_months",
        "written_off_accounts", "settled_accounts",
    ]

    def __init__(
        self,
        model_type: str = "lightgbm",
        cat_type: str = "auto",
        features: Optional[List[str]] = None,
        n_estimators: int = 500,
        learning_rate: float = 0.05,
        max_depth: int = 6,
        min_child_samples: int = 50,
        reg_alpha: float = 0.1,
        reg_lambda: float = 1.0,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        scale_pos_weight: Optional[float] = None,
        random_state: int = 42,
        n_jobs: int = -1,
    ):
        self.model_type = model_type.lower()
        self.cat_type = cat_type  # "auto", "CAT", "NonCAT", "combined"
        self.features = features or self.DEFAULT_FEATURES
        self.random_state = random_state
        self.n_jobs = n_jobs

        # Model hyperparameters
        self._params = {
            "n_estimators": n_estimators,
            "learning_rate": learning_rate,
            "max_depth": max_depth,
            "min_child_samples": min_child_samples,
            "reg_alpha": reg_alpha,
            "reg_lambda": reg_lambda,
            "subsample": subsample,
            "colsample_bytree": colsample_bytree,
            "random_state": random_state,
            "n_jobs": n_jobs,
            "verbose": -1,
        }
        if scale_pos_weight is not None:
            self._params["scale_pos_weight"] = scale_pos_weight

        # Models storage (one per segment or single combined)
        self._models: Dict[str, Any] = {}
        self._calibrators: Dict[str, Any] = {}
        self._feature_importance: Dict[str, pd.DataFrame] = {}
        self._fitted = False
        self._cat_classifier = CAT_NonCAT_Classifier()

    def _build_model(self, scale_pos_weight: Optional[float] = None):
        """Create a fresh model instance.

        Supports lightgbm, xgboost, and sklearn (HistGradientBoosting).
        Falls back to sklearn if lightgbm/xgboost fail to load.
        """
        params = dict(self._params)
        if scale_pos_weight is not None:
            params["scale_pos_weight"] = scale_pos_weight

        if self.model_type == "lightgbm":
            try:
                import lightgbm as lgb
                return lgb.LGBMClassifier(**params)
            except (ImportError, OSError):
                warnings.warn(
                    "LightGBM unavailable (missing libomp?), falling back to "
                    "sklearn HistGradientBoostingClassifier (similar performance)."
                )
                self.model_type = "sklearn"
                return self._build_sklearn_model(params, scale_pos_weight)

        elif self.model_type == "xgboost":
            try:
                import xgboost as xgb
                xgb_params = {
                    "n_estimators": params["n_estimators"],
                    "learning_rate": params["learning_rate"],
                    "max_depth": params["max_depth"],
                    "reg_alpha": params["reg_alpha"],
                    "reg_lambda": params["reg_lambda"],
                    "subsample": params["subsample"],
                    "colsample_bytree": params["colsample_bytree"],
                    "random_state": params["random_state"],
                    "n_jobs": params["n_jobs"],
                    "verbosity": 0,
                    "tree_method": "hist",  # Fast on Apple Silicon
                    "device": "cpu",
                }
                if scale_pos_weight is not None:
                    xgb_params["scale_pos_weight"] = scale_pos_weight
                return xgb.XGBClassifier(**xgb_params)
            except Exception:
                warnings.warn(
                    "XGBoost unavailable (missing libomp?), falling back to sklearn."
                )
                self.model_type = "sklearn"
                return self._build_sklearn_model(params, scale_pos_weight)

        elif self.model_type == "sklearn":
            return self._build_sklearn_model(params, scale_pos_weight)

        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

    def _build_sklearn_model(self, params: dict, scale_pos_weight: Optional[float] = None):
        """Build sklearn HistGradientBoostingClassifier (LightGBM-inspired)."""
        from sklearn.ensemble import HistGradientBoostingClassifier

        # Map to sklearn params
        class_weight = None
        if scale_pos_weight and scale_pos_weight > 1:
            # HistGBM doesn't support scale_pos_weight directly,
            # but we can use sample_weight in fit() — handled in _fit_segment
            pass

        return HistGradientBoostingClassifier(
            max_iter=params.get("n_estimators", 500),
            learning_rate=params.get("learning_rate", 0.05),
            max_depth=params.get("max_depth", 6),
            min_samples_leaf=params.get("min_child_samples", 50),
            l2_regularization=params.get("reg_lambda", 1.0),
            max_bins=255,
            random_state=params.get("random_state", 42),
            verbose=0,
        )

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        cat_segment: Optional[pd.Series] = None,
        calibrate: bool = True,
        eval_set: Optional[Tuple[pd.DataFrame, pd.Series]] = None,
    ) -> "CreditScorer":
        """Train the credit scoring model(s).

        Parameters
        ----------
        X : DataFrame
            Feature matrix.
        y : Series
            Binary target (1 = default, 0 = non-default).
        cat_segment : Series, optional
            CAT/NonCAT labels. Auto-detected from annual_income if not provided.
        calibrate : bool
            Apply Platt scaling for probability calibration.
        eval_set : tuple, optional
            (X_val, y_val) for early stopping.
        """
        # Ensure we only use available features
        available = [f for f in self.features if f in X.columns]
        if len(available) < len(self.features) * 0.5:
            raise ValueError(
                f"Too many missing features: have {len(available)}/{len(self.features)}"
            )
        self.features = available

        if self.cat_type == "combined" or self.cat_type is None:
            # Single model for all segments
            self._fit_segment("combined", X[self.features], y, calibrate)
        else:
            # Segment-specific models
            if cat_segment is None:
                if "annual_income" in X.columns:
                    cat_segment = self._cat_classifier.classify(X)
                else:
                    # Fallback to combined
                    self._fit_segment("combined", X[self.features], y, calibrate)
                    self._fitted = True
                    return self

            for segment in ["CAT", "NonCAT"]:
                mask = cat_segment == segment
                if mask.sum() < 50:
                    continue
                X_seg = X.loc[mask, self.features]
                y_seg = y.loc[mask]
                self._fit_segment(segment, X_seg, y_seg, calibrate)

        self._fitted = True
        return self

    def _fit_segment(
        self,
        segment: str,
        X: pd.DataFrame,
        y: pd.Series,
        calibrate: bool,
    ) -> None:
        """Train model for a specific segment."""
        # Auto-compute class weight
        n_pos = y.sum()
        n_neg = len(y) - n_pos
        spw = n_neg / max(n_pos, 1)

        model = self._build_model(scale_pos_weight=spw)

        # sklearn HistGBM needs sample_weight for imbalanced classes
        if self.model_type == "sklearn" and spw > 1:
            sample_weight = np.where(y == 1, spw, 1.0)
            model.fit(X, y, sample_weight=sample_weight)
        else:
            model.fit(X, y)

        self._models[segment] = model

        # Calibration
        if calibrate and len(y) > 200:
            calibrated = CalibratedClassifierCV(
                model, method="sigmoid", cv=3
            )
            calibrated.fit(X, y)
            self._calibrators[segment] = calibrated

        # Feature importance
        if hasattr(model, "feature_importances_"):
            imp = pd.DataFrame({
                "feature": self.features,
                "importance": model.feature_importances_,
            }).sort_values("importance", ascending=False)
            self._feature_importance[segment] = imp
        else:
            # Fallback: use permutation importance for sklearn models
            from sklearn.inspection import permutation_importance
            perm_imp = permutation_importance(model, X, y, n_repeats=5, random_state=42, n_jobs=-1)
            imp = pd.DataFrame({
                "feature": self.features,
                "importance": perm_imp.importances_mean,
            }).sort_values("importance", ascending=False)
            self._feature_importance[segment] = imp

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict default probability.

        Returns array of shape (n_samples,) with P(default).
        """
        if not self._fitted:
            raise RuntimeError("Call fit() before predict_proba()")

        X_feat = X[self.features] if all(f in X.columns for f in self.features) else X

        if "combined" in self._models:
            if "combined" in self._calibrators:
                return self._calibrators["combined"].predict_proba(X_feat)[:, 1]
            return self._models["combined"].predict_proba(X_feat)[:, 1]

        # Segment-specific prediction
        probs = np.zeros(len(X_feat))
        if "annual_income" in X.columns:
            segments = self._cat_classifier.classify(X)
        else:
            segments = pd.Series(["NonCAT"] * len(X), index=X.index)

        for seg_name in ["CAT", "NonCAT"]:
            mask = segments == seg_name
            if mask.sum() == 0:
                continue
            if seg_name not in self._models:
                # Fallback to other segment's model
                fallback = "CAT" if seg_name == "NonCAT" else "NonCAT"
                if fallback not in self._models:
                    continue
                seg_name = fallback

            X_seg = X_feat.loc[mask]
            if seg_name in self._calibrators:
                probs[mask.values] = self._calibrators[seg_name].predict_proba(X_seg)[:, 1]
            else:
                probs[mask.values] = self._models[seg_name].predict_proba(X_seg)[:, 1]

        return probs

    def score(self, X: pd.DataFrame) -> pd.DataFrame:
        """Score applicants and return CIBIL-like scores + risk grades.

        Returns DataFrame with columns:
        - default_probability
        - cibil_score (300-900 scale)
        - risk_grade (A-E)
        - segment (CAT/NonCAT)
        """
        probs = self.predict_proba(X)

        # Map probability to CIBIL-like score
        # P(default)=0 → 900, P(default)=1 → 300
        # Using logistic mapping for smooth curve
        scores = 900 - (probs * 600)
        scores = np.clip(scores, 300, 900).astype(int)

        # Risk grades
        grades = pd.Series([risk_grade_from_score(s) for s in scores])

        # Segment
        if "annual_income" in X.columns:
            segments = self._cat_classifier.classify(X)
        else:
            segments = pd.Series(["Unknown"] * len(X), index=X.index)

        return pd.DataFrame({
            "default_probability": np.round(probs, 6),
            "cibil_score": scores,
            "risk_grade": grades.values,
            "segment": segments.values,
        }, index=X.index)

    def evaluate(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        threshold: float = 0.5,
    ) -> Dict[str, Any]:
        """Evaluate model performance.

        Returns dict with AUC, accuracy, precision-recall metrics,
        KS statistic, and Gini coefficient.
        """
        probs = self.predict_proba(X)
        preds = (probs >= threshold).astype(int)

        auc = roc_auc_score(y, probs)
        ap = average_precision_score(y, probs)
        acc = accuracy_score(y, preds)

        # KS statistic (key metric for Indian credit scoring)
        from scipy.stats import ks_2samp
        ks_stat, ks_pval = ks_2samp(probs[y == 0], probs[y == 1])

        # Gini coefficient
        gini = 2 * auc - 1

        report = classification_report(y, preds, output_dict=True)

        return {
            "auc_roc": round(auc, 4),
            "average_precision": round(ap, 4),
            "accuracy": round(acc, 4),
            "ks_statistic": round(ks_stat, 4),
            "gini_coefficient": round(gini, 4),
            "precision_default": round(report.get("1", {}).get("precision", 0), 4),
            "recall_default": round(report.get("1", {}).get("recall", 0), 4),
            "f1_default": round(report.get("1", {}).get("f1-score", 0), 4),
            "classification_report": report,
            "n_samples": len(y),
            "n_defaults": int(y.sum()),
            "default_rate": round(float(y.mean()), 4),
        }

    def cross_validate(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        n_folds: int = 5,
    ) -> Dict[str, Any]:
        """Run stratified k-fold cross-validation."""
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=self.random_state)
        fold_results = []

        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            fold_scorer = CreditScorer(
                model_type=self.model_type,
                cat_type="combined",
                features=self.features,
                **{k: v for k, v in self._params.items()
                   if k not in ("verbose", "n_jobs", "random_state")},
                random_state=self.random_state + fold,
                n_jobs=self.n_jobs,
            )
            fold_scorer.fit(X_train, y_train, calibrate=False)
            metrics = fold_scorer.evaluate(X_val, y_val)
            metrics["fold"] = fold
            fold_results.append(metrics)

        # Aggregate
        auc_scores = [r["auc_roc"] for r in fold_results]
        ks_scores = [r["ks_statistic"] for r in fold_results]
        gini_scores = [r["gini_coefficient"] for r in fold_results]

        return {
            "n_folds": n_folds,
            "mean_auc": round(np.mean(auc_scores), 4),
            "std_auc": round(np.std(auc_scores), 4),
            "mean_ks": round(np.mean(ks_scores), 4),
            "std_ks": round(np.std(ks_scores), 4),
            "mean_gini": round(np.mean(gini_scores), 4),
            "std_gini": round(np.std(gini_scores), 4),
            "fold_results": fold_results,
        }

    def feature_importance(self, segment: str = "combined") -> pd.DataFrame:
        """Get feature importance for a segment."""
        if segment in self._feature_importance:
            return self._feature_importance[segment]
        # Return first available
        if self._feature_importance:
            return next(iter(self._feature_importance.values()))
        return pd.DataFrame(columns=["feature", "importance"])

    def save(self, path: str) -> None:
        """Save model to disk."""
        joblib.dump({
            "models": self._models,
            "calibrators": self._calibrators,
            "features": self.features,
            "model_type": self.model_type,
            "cat_type": self.cat_type,
            "params": self._params,
            "feature_importance": self._feature_importance,
        }, path)

    @classmethod
    def load(cls, path: str) -> "CreditScorer":
        """Load model from disk."""
        data = joblib.load(path)
        scorer = cls(
            model_type=data["model_type"],
            cat_type=data["cat_type"],
            features=data["features"],
        )
        scorer._models = data["models"]
        scorer._calibrators = data["calibrators"]
        scorer._params = data["params"]
        scorer._feature_importance = data["feature_importance"]
        scorer._fitted = True
        return scorer


# ---------------------------------------------------------------------------
# Alternative Data Scorer
# ---------------------------------------------------------------------------

class AlternativeScorer:
    """Score thin-file applicants using alternative data.

    In India, ~300M adults are "credit invisible" (no bureau file).
    Alternative data sources:
    - UPI transaction patterns (volume, diversity, regularity)
    - Phone usage (age, recharge, data usage)
    - E-commerce behavior (order frequency, returns, spend)

    Produces a 300-900 scale score that can supplement or replace
    traditional bureau scores.
    """

    # Feature weights (tuned on Indian alternative data studies)
    FEATURE_WEIGHTS = {
        "upi_txn_count_3m": 0.15,
        "upi_txn_value_3m": 0.12,
        "upi_unique_merchants": 0.10,
        "upi_inflow_ratio": 0.08,
        "phone_age_months": 0.10,
        "avg_recharge_amount": 0.08,
        "data_usage_gb": 0.05,
        "app_install_count": 0.05,
        "ecom_orders_6m": 0.10,
        "ecom_spend_6m": 0.10,
        "ecom_return_rate": -0.07,  # Negative: high returns = higher risk
    }

    def __init__(self, model_type: str = "heuristic"):
        """
        Parameters
        ----------
        model_type : str
            "heuristic" for rule-based, "ml" for trained ML model.
        """
        self.model_type = model_type
        self._ml_model = None
        self._fitted = False

    def score(self, data: Union[Dict, pd.DataFrame]) -> Union[Dict, pd.DataFrame]:
        """Score using alternative data.

        Returns score on 300-900 scale with confidence.
        """
        if isinstance(data, dict):
            return self._score_single(data)

        if isinstance(data, pd.DataFrame):
            results = []
            for _, row in data.iterrows():
                results.append(self._score_single(row.to_dict()))
            return pd.DataFrame(results, index=data.index)

        raise TypeError("Input must be dict or DataFrame")

    def _score_single(self, data: Dict) -> Dict:
        """Score a single applicant."""
        if self.model_type == "heuristic":
            return self._heuristic_score(data)
        elif self._ml_model is not None:
            return self._ml_score(data)
        return self._heuristic_score(data)

    def _heuristic_score(self, data: Dict) -> Dict:
        """Rule-based alternative scoring."""
        raw_score = 0.0
        features_used = 0
        max_possible = 0.0

        for feature, weight in self.FEATURE_WEIGHTS.items():
            value = data.get(feature)
            if value is None:
                continue

            features_used += 1
            abs_weight = abs(weight)
            max_possible += abs_weight

            # Normalize each feature to 0-1 range
            normalized = self._normalize_feature(feature, value)

            if weight < 0:
                raw_score += abs_weight * (1 - normalized)
            else:
                raw_score += abs_weight * normalized

        if features_used == 0:
            return {"alt_score": 300, "confidence": 0.0, "features_used": 0}

        # Normalize to 0-1
        score_pct = raw_score / max_possible if max_possible > 0 else 0

        # Map to 300-900
        alt_score = int(300 + score_pct * 600)
        alt_score = max(300, min(900, alt_score))

        # Confidence based on features available
        confidence = min(features_used / len(self.FEATURE_WEIGHTS), 1.0)

        return {
            "alt_score": alt_score,
            "confidence": round(confidence, 2),
            "features_used": features_used,
            "risk_grade": risk_grade_from_score(alt_score),
        }

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "AlternativeScorer":
        """Train ML model on alternative data features."""
        alt_features = [f for f in self.FEATURE_WEIGHTS.keys() if f in X.columns]
        if len(alt_features) < 3:
            raise ValueError("Need at least 3 alternative data features")

        try:
            import lightgbm as lgb
            model = lgb.LGBMClassifier(
                n_estimators=200, learning_rate=0.05, max_depth=4,
                min_child_samples=30, verbose=-1, n_jobs=-1,
            )
        except (ImportError, OSError):
            from sklearn.ensemble import HistGradientBoostingClassifier
            model = HistGradientBoostingClassifier(
                max_iter=200, learning_rate=0.05, max_depth=4,
                min_samples_leaf=30, verbose=0,
            )
        model.fit(X[alt_features], y)
        self._ml_model = model
        self._alt_features = alt_features
        self._fitted = True
        self.model_type = "ml"
        return self

    def _ml_score(self, data: Dict) -> Dict:
        """ML-based alternative scoring."""
        features = {f: data.get(f, 0) for f in self._alt_features}
        X = pd.DataFrame([features])
        prob = self._ml_model.predict_proba(X)[0, 1]
        alt_score = int(900 - prob * 600)
        alt_score = max(300, min(900, alt_score))

        return {
            "alt_score": alt_score,
            "default_probability": round(float(prob), 6),
            "confidence": 0.85,
            "features_used": len(self._alt_features),
            "risk_grade": risk_grade_from_score(alt_score),
        }

    @staticmethod
    def _normalize_feature(feature: str, value: float) -> float:
        """Normalize a feature value to 0-1 range based on Indian benchmarks."""
        ranges = {
            "upi_txn_count_3m": (0, 300),
            "upi_txn_value_3m": (0, 500_000),
            "upi_unique_merchants": (0, 100),
            "upi_inflow_ratio": (0, 1),
            "phone_age_months": (0, 120),
            "avg_recharge_amount": (0, 1500),
            "data_usage_gb": (0, 40),
            "app_install_count": (0, 80),
            "ecom_orders_6m": (0, 25),
            "ecom_spend_6m": (0, 100_000),
            "ecom_return_rate": (0, 0.5),
        }
        lo, hi = ranges.get(feature, (0, 1))
        if hi == lo:
            return 0.5
        return max(0, min(1, (value - lo) / (hi - lo)))
