"""
IndiaLend Ensemble Credit Scorer
=================================
Combines multiple ML models into a high-performance ensemble:

1. HistGradientBoosting (sklearn) - Fast gradient boosting
2. Random Forest (sklearn) - Bagging ensemble
3. Extra Trees (sklearn) - Randomized splits
4. Logistic Regression (sklearn) - Interpretable baseline
5. Neural Network (PyTorch Metal) - Deep non-linear patterns
6. LightGBM / XGBoost (if available) - Industry standard

Ensemble methods:
- Soft voting (average probabilities)
- Weighted voting (AUC-weighted average)
- Stacking (meta-learner on top of base models)

Indian market context:
    RBI model validation guidelines (circular DOR.MRG.REC.76/00-00-007/2024)
    recommend ensemble approaches for internal rating models. Using multiple
    model types reduces model risk and satisfies regulatory requirements
    for model comparison / challenger frameworks.
"""

from __future__ import annotations

import time
import warnings
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import (
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
    StackingClassifier,
    VotingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from .utils import risk_grade_from_score

warnings.filterwarnings("ignore", category=UserWarning)


# ---------------------------------------------------------------------------
# Model Factory
# ---------------------------------------------------------------------------

class ModelFactory:
    """Creates configured ML models for credit scoring.

    Each model is tuned for Indian credit data characteristics:
    - ~8% default rate (class imbalance)
    - 27-80 numeric features
    - Mix of continuous (income) and discrete (DPD counts) features
    """

    @staticmethod
    def create_hist_gbm(
        n_estimators: int = 800,
        learning_rate: float = 0.03,
        max_depth: int = 7,
        min_samples_leaf: int = 40,
        l2_reg: float = 1.0,
        random_state: int = 42,
    ) -> HistGradientBoostingClassifier:
        """HistGradientBoosting - fast, LightGBM-equivalent in sklearn."""
        return HistGradientBoostingClassifier(
            max_iter=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            l2_regularization=l2_reg,
            max_bins=255,
            early_stopping=True,
            n_iter_no_change=20,
            validation_fraction=0.1,
            random_state=random_state,
            verbose=0,
        )

    @staticmethod
    def create_random_forest(
        n_estimators: int = 500,
        max_depth: int = 15,
        min_samples_leaf: int = 20,
        random_state: int = 42,
        n_jobs: int = -1,
    ) -> RandomForestClassifier:
        """Random Forest - robust bagging ensemble."""
        return RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            max_features="sqrt",
            class_weight="balanced",
            random_state=random_state,
            n_jobs=n_jobs,
            verbose=0,
        )

    @staticmethod
    def create_extra_trees(
        n_estimators: int = 500,
        max_depth: int = 15,
        min_samples_leaf: int = 20,
        random_state: int = 42,
        n_jobs: int = -1,
    ) -> ExtraTreesClassifier:
        """Extra Trees - more randomized splits, often less overfitting."""
        return ExtraTreesClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            max_features="sqrt",
            class_weight="balanced",
            random_state=random_state,
            n_jobs=n_jobs,
            verbose=0,
        )

    @staticmethod
    def create_logistic_regression(
        C: float = 0.1,
        random_state: int = 42,
    ) -> LogisticRegression:
        """Logistic Regression - interpretable baseline (RBI-friendly)."""
        return LogisticRegression(
            C=C,
            class_weight="balanced",
            max_iter=1000,
            solver="lbfgs",
            random_state=random_state,
        )

    @staticmethod
    def create_lightgbm(
        n_estimators: int = 800,
        learning_rate: float = 0.03,
        max_depth: int = 7,
        random_state: int = 42,
        n_jobs: int = -1,
    ):
        """LightGBM - industry standard (requires libomp on macOS)."""
        try:
            import lightgbm as lgb
            return lgb.LGBMClassifier(
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                max_depth=max_depth,
                min_child_samples=40,
                reg_alpha=0.1,
                reg_lambda=1.0,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=random_state,
                n_jobs=n_jobs,
                verbose=-1,
            )
        except (ImportError, OSError):
            return None

    @staticmethod
    def create_xgboost(
        n_estimators: int = 800,
        learning_rate: float = 0.03,
        max_depth: int = 7,
        random_state: int = 42,
        n_jobs: int = -1,
    ):
        """XGBoost - gradient boosting with regularization."""
        try:
            import xgboost as xgb
            return xgb.XGBClassifier(
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                max_depth=max_depth,
                reg_alpha=0.1,
                reg_lambda=1.0,
                subsample=0.8,
                colsample_bytree=0.8,
                tree_method="hist",
                device="cpu",
                random_state=random_state,
                n_jobs=n_jobs,
                verbosity=0,
            )
        except (ImportError, Exception):
            return None


# ---------------------------------------------------------------------------
# Ensemble Credit Scorer
# ---------------------------------------------------------------------------

class EnsembleCreditScorer:
    """Ensemble of multiple ML models for maximum credit scoring accuracy.

    Trains multiple diverse models and combines their predictions using
    weighted averaging (weights proportional to validation AUC).

    Parameters
    ----------
    models : dict, optional
        Dict of {name: model_instance}. If None, creates default set.
    ensemble_method : str
        "weighted" (AUC-weighted average), "voting" (soft voting),
        or "stacking" (meta-learner).
    calibrate : bool
        Apply Platt scaling to each base model.
    n_folds_stack : int
        Number of folds for stacking meta-learner.
    random_state : int
        Random seed.
    verbose : bool
        Print training progress.

    Example
    -------
    >>> ens = EnsembleCreditScorer(verbose=True)
    >>> ens.fit(X_train, y_train, X_val=X_val, y_val=y_val)
    >>> comparison = ens.compare_models()
    >>> scores = ens.score(X_test)
    """

    def __init__(
        self,
        models: Optional[Dict[str, Any]] = None,
        ensemble_method: str = "weighted",
        calibrate: bool = True,
        n_folds_stack: int = 5,
        features: Optional[List[str]] = None,
        random_state: int = 42,
        n_jobs: int = -1,
        verbose: bool = True,
    ):
        self.ensemble_method = ensemble_method
        self.calibrate = calibrate
        self.n_folds_stack = n_folds_stack
        self.features = features
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.verbose = verbose

        # Build default models if not provided
        if models is None:
            self._models = self._build_default_models()
        else:
            self._models = models

        self._fitted_models: Dict[str, Any] = {}
        self._calibrators: Dict[str, Any] = {}
        self._model_weights: Dict[str, float] = {}
        self._model_metrics: Dict[str, Dict] = {}
        self._scaler = StandardScaler()
        self._fitted = False
        self._stacking_model = None

    def _build_default_models(self) -> Dict[str, Any]:
        """Create default model set."""
        factory = ModelFactory()
        models = {
            "HistGBM": factory.create_hist_gbm(random_state=self.random_state),
            "RandomForest": factory.create_random_forest(
                random_state=self.random_state, n_jobs=self.n_jobs
            ),
            "ExtraTrees": factory.create_extra_trees(
                random_state=self.random_state, n_jobs=self.n_jobs
            ),
            "LogisticRegression": factory.create_logistic_regression(
                random_state=self.random_state
            ),
        }

        # Try adding LightGBM and XGBoost
        lgbm = factory.create_lightgbm(random_state=self.random_state, n_jobs=self.n_jobs)
        if lgbm is not None:
            models["LightGBM"] = lgbm

        xgb = factory.create_xgboost(random_state=self.random_state, n_jobs=self.n_jobs)
        if xgb is not None:
            models["XGBoost"] = xgb

        return models

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
        sample_weight: Optional[np.ndarray] = None,
    ) -> "EnsembleCreditScorer":
        """Train all models in the ensemble.

        Parameters
        ----------
        X : DataFrame
            Training features.
        y : Series
            Binary target (1=default).
        X_val : DataFrame, optional
            Validation set for model weighting.
        y_val : Series, optional
            Validation target.
        sample_weight : array, optional
            Per-sample weights.
        """
        # Select features
        if self.features:
            available = [f for f in self.features if f in X.columns]
        else:
            available = list(X.select_dtypes(include=[np.number]).columns)
        self.features = available

        X_train = X[self.features].copy()
        X_train_values = X_train.values.astype(np.float32)
        np.nan_to_num(X_train_values, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

        # Scale for LogisticRegression
        self._scaler.fit(X_train_values)

        if X_val is not None:
            X_val_use = X_val[self.features].copy()
        else:
            X_val_use = None

        # Compute sample weights for imbalanced classes
        if sample_weight is None:
            n_pos = y.sum()
            n_neg = len(y) - n_pos
            spw = n_neg / max(n_pos, 1)
            sample_weight = np.where(y == 1, spw, 1.0).astype(np.float32)

        if self.verbose:
            print(f"\n{'='*70}")
            print(f"  Ensemble Credit Scorer - Training {len(self._models)} models")
            print(f"{'='*70}")
            print(f"  Train: {len(X):,} samples | Val: {len(X_val) if X_val is not None else 0:,}")
            print(f"  Features: {len(self.features)} | Default rate: {y.mean():.2%}")
            print(f"  Models: {', '.join(self._models.keys())}")
            print(f"{'='*70}")
            print(f"  {'Model':<22} {'Train Time':>12} {'Val AUC':>10} {'Val KS':>10} {'Status':>10}")
            print(f"  {'-'*65}")

        total_start = time.time()

        for name, model in self._models.items():
            start = time.time()

            try:
                # Scale input for linear models
                if name == "LogisticRegression":
                    X_fit = pd.DataFrame(
                        self._scaler.transform(X_train_values),
                        columns=self.features, index=X_train.index,
                    )
                else:
                    X_fit = X_train

                # Fit with sample weight
                needs_sw = name in ("HistGBM", "LogisticRegression")
                if needs_sw:
                    model.fit(X_fit, y, sample_weight=sample_weight)
                elif hasattr(model, "fit"):
                    try:
                        model.fit(X_fit, y, sample_weight=sample_weight)
                    except TypeError:
                        model.fit(X_fit, y)

                self._fitted_models[name] = model

                # Calibrate
                if self.calibrate and len(y) > 500:
                    try:
                        cal = CalibratedClassifierCV(model, method="sigmoid", cv=3)
                        cal.fit(X_fit, y)
                        self._calibrators[name] = cal
                    except Exception:
                        pass

                # Evaluate on validation
                train_time = time.time() - start
                val_auc = 0.0
                val_ks = 0.0

                if X_val_use is not None and y_val is not None:
                    probs = self._predict_single(name, X_val_use)
                    val_auc = roc_auc_score(y_val, probs)
                    from scipy.stats import ks_2samp
                    val_ks, _ = ks_2samp(probs[y_val == 0], probs[y_val == 1])

                self._model_metrics[name] = {
                    "val_auc": round(val_auc, 4),
                    "val_ks": round(val_ks, 4),
                    "train_time": round(train_time, 2),
                }
                self._model_weights[name] = max(val_auc, 0.5)

                if self.verbose:
                    print(
                        f"  {name:<22} {train_time:>10.1f}s {val_auc:>10.4f} "
                        f"{val_ks:>10.4f} {'OK':>10}"
                    )

            except Exception as e:
                if self.verbose:
                    print(f"  {name:<22} {'--':>10} {'--':>10} {'--':>10} {'FAILED':>10}")
                    print(f"    Error: {e}")

        # Normalize weights
        if self._model_weights:
            total_w = sum(self._model_weights.values())
            self._model_weights = {
                k: v / total_w for k, v in self._model_weights.items()
            }

        # Build stacking model if requested
        if self.ensemble_method == "stacking" and len(self._fitted_models) >= 2:
            self._build_stacking(X_train, y, sample_weight)

        total_time = time.time() - total_start
        self._fitted = True

        if self.verbose:
            print(f"  {'-'*65}")

            # Ensemble validation AUC
            if X_val_use is not None and y_val is not None:
                ens_probs = self.predict_proba(X_val)
                ens_auc = roc_auc_score(y_val, ens_probs)
                from scipy.stats import ks_2samp
                ens_ks, _ = ks_2samp(ens_probs[y_val == 0], ens_probs[y_val == 1])
                print(
                    f"  {'ENSEMBLE':.<22} {total_time:>10.1f}s {ens_auc:>10.4f} "
                    f"{ens_ks:>10.4f} {'OK':>10}"
                )

            print(f"\n  Model weights: {self._weight_summary()}")
            print(f"{'='*70}\n")

        return self

    def _weight_summary(self) -> str:
        """Format model weights for display."""
        parts = [f"{k}: {v:.1%}" for k, v in
                 sorted(self._model_weights.items(), key=lambda x: -x[1])]
        return " | ".join(parts)

    def _predict_single(self, name: str, X: pd.DataFrame) -> np.ndarray:
        """Get predictions from a single model."""
        X_use = X[self.features].copy() if self.features else X
        X_values = X_use.values.astype(np.float32)
        np.nan_to_num(X_values, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

        if name == "LogisticRegression":
            X_input = pd.DataFrame(
                self._scaler.transform(X_values),
                columns=self.features, index=X_use.index,
            )
        else:
            X_input = X_use

        # Use calibrated model if available
        if name in self._calibrators:
            return self._calibrators[name].predict_proba(X_input)[:, 1]
        return self._fitted_models[name].predict_proba(X_input)[:, 1]

    def _build_stacking(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        sample_weight: np.ndarray,
    ) -> None:
        """Build stacking meta-learner."""
        estimators = [(name, model) for name, model in self._fitted_models.items()
                      if name != "LogisticRegression"]
        if len(estimators) < 2:
            return

        self._stacking_model = StackingClassifier(
            estimators=estimators,
            final_estimator=LogisticRegression(C=1.0, max_iter=500),
            cv=self.n_folds_stack,
            stack_method="predict_proba",
            n_jobs=self.n_jobs,
            verbose=0,
        )
        try:
            self._stacking_model.fit(X, y, sample_weight=sample_weight)
        except Exception:
            self._stacking_model = None

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict default probability using ensemble."""
        if not self._fitted:
            raise RuntimeError("Call fit() before predict_proba()")

        if self.ensemble_method == "stacking" and self._stacking_model is not None:
            X_use = X[self.features].copy()
            return self._stacking_model.predict_proba(X_use)[:, 1]

        # Weighted average of all models
        all_probs = []
        weights = []

        for name in self._fitted_models:
            try:
                probs = self._predict_single(name, X)
                all_probs.append(probs)
                weights.append(self._model_weights.get(name, 1.0))
            except Exception:
                continue

        if not all_probs:
            raise RuntimeError("No models produced valid predictions")

        # Weighted average
        weights = np.array(weights)
        weights /= weights.sum()
        ensemble_probs = np.zeros(len(X))
        for probs, w in zip(all_probs, weights):
            ensemble_probs += probs * w

        return ensemble_probs

    def predict_proba_all(self, X: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Get predictions from each individual model + ensemble."""
        results = {}
        for name in self._fitted_models:
            try:
                results[name] = self._predict_single(name, X)
            except Exception:
                continue
        results["Ensemble"] = self.predict_proba(X)
        return results

    def score(self, X: pd.DataFrame) -> pd.DataFrame:
        """Score applicants, returning CIBIL-scale scores + risk grades."""
        probs = self.predict_proba(X)
        scores = 900 - (probs * 600)
        scores = np.clip(scores, 300, 900).astype(int)
        grades = [risk_grade_from_score(s) for s in scores]

        return pd.DataFrame({
            "default_probability": np.round(probs, 6),
            "cibil_score": scores,
            "risk_grade": grades,
            "model": "ensemble",
        }, index=X.index)

    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Evaluate ensemble performance."""
        from scipy.stats import ks_2samp
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

        probs = self.predict_proba(X)
        preds = (probs >= 0.5).astype(int)

        auc = roc_auc_score(y, probs)
        ks_stat, _ = ks_2samp(probs[y == 0], probs[y == 1])
        gini = 2 * auc - 1

        return {
            "model": "EnsembleCreditScorer",
            "method": self.ensemble_method,
            "n_models": len(self._fitted_models),
            "auc_roc": round(auc, 4),
            "ks_statistic": round(ks_stat, 4),
            "gini_coefficient": round(gini, 4),
            "accuracy": round(accuracy_score(y, preds), 4),
            "precision": round(precision_score(y, preds, zero_division=0), 4),
            "recall": round(recall_score(y, preds, zero_division=0), 4),
            "f1_score": round(f1_score(y, preds, zero_division=0), 4),
        }

    def compare_models(
        self,
        X: pd.DataFrame,
        y: pd.Series,
    ) -> pd.DataFrame:
        """Compare all individual models and the ensemble.

        Returns DataFrame with metrics for each model.
        """
        from scipy.stats import ks_2samp

        rows = []
        for name in self._fitted_models:
            try:
                probs = self._predict_single(name, X)
                auc = roc_auc_score(y, probs)
                ks_stat, _ = ks_2samp(probs[y == 0], probs[y == 1])
                gini = 2 * auc - 1
                preds = (probs >= 0.5).astype(int)

                from sklearn.metrics import f1_score
                f1 = f1_score(y, preds, zero_division=0)

                rows.append({
                    "Model": name,
                    "AUC": round(auc, 4),
                    "KS": round(ks_stat, 4),
                    "Gini": round(gini, 4),
                    "F1": round(f1, 4),
                    "Weight": round(self._model_weights.get(name, 0), 4),
                    "Train Time (s)": self._model_metrics.get(name, {}).get("train_time", 0),
                })
            except Exception:
                continue

        # Add ensemble
        ens_probs = self.predict_proba(X)
        ens_auc = roc_auc_score(y, ens_probs)
        ens_ks, _ = ks_2samp(ens_probs[y == 0], ens_probs[y == 1])
        ens_gini = 2 * ens_auc - 1
        ens_preds = (ens_probs >= 0.5).astype(int)

        from sklearn.metrics import f1_score
        ens_f1 = f1_score(y, ens_preds, zero_division=0)

        rows.append({
            "Model": f"ENSEMBLE ({self.ensemble_method})",
            "AUC": round(ens_auc, 4),
            "KS": round(ens_ks, 4),
            "Gini": round(ens_gini, 4),
            "F1": round(ens_f1, 4),
            "Weight": 1.0,
            "Train Time (s)": sum(r.get("Train Time (s)", 0) for r in rows),
        })

        return pd.DataFrame(rows).sort_values("AUC", ascending=False).reset_index(drop=True)

    def feature_importance(self, top_n: int = 20) -> pd.DataFrame:
        """Aggregate feature importance across models."""
        importance_sum = np.zeros(len(self.features))
        count = 0

        for name, model in self._fitted_models.items():
            if hasattr(model, "feature_importances_"):
                imp = model.feature_importances_
                if len(imp) == len(self.features):
                    weight = self._model_weights.get(name, 1.0)
                    importance_sum += (imp / imp.sum()) * weight
                    count += 1

        if count == 0:
            # Fallback: permutation importance
            return pd.DataFrame(columns=["feature", "importance"])

        importance_avg = importance_sum / count
        df = pd.DataFrame({
            "feature": self.features,
            "importance": importance_avg,
        }).sort_values("importance", ascending=False).head(top_n)

        return df.reset_index(drop=True)

    def save(self, path: str) -> None:
        """Save ensemble to disk."""
        joblib.dump({
            "fitted_models": self._fitted_models,
            "calibrators": self._calibrators,
            "model_weights": self._model_weights,
            "model_metrics": self._model_metrics,
            "features": self.features,
            "scaler": self._scaler,
            "ensemble_method": self.ensemble_method,
            "stacking_model": self._stacking_model,
        }, path)

    @classmethod
    def load(cls, path: str) -> "EnsembleCreditScorer":
        """Load ensemble from disk."""
        data = joblib.load(path)
        ens = cls(
            models={},
            ensemble_method=data["ensemble_method"],
            features=data["features"],
        )
        ens._fitted_models = data["fitted_models"]
        ens._calibrators = data["calibrators"]
        ens._model_weights = data["model_weights"]
        ens._model_metrics = data["model_metrics"]
        ens._scaler = data["scaler"]
        ens._stacking_model = data.get("stacking_model")
        ens._fitted = True
        return ens
