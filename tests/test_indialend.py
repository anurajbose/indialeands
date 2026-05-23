"""
IndiaLend Test Suite
====================
Comprehensive tests for all modules.
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import pytest

# ============================================================
# Utils
# ============================================================

class TestValidators:
    def test_pan_valid(self):
        from indialend import pan_validator
        assert pan_validator("ABCPP1234K") is True
        assert pan_validator("AAAHP0001A") is True

    def test_pan_invalid(self):
        from indialend import pan_validator
        assert pan_validator("123456789X") is False
        assert pan_validator("") is False
        assert pan_validator("SHORT") is False

    def test_aadhaar_valid(self):
        from indialend import aadhaar_validator
        assert aadhaar_validator("234567890123") is True
        assert aadhaar_validator("9999 8888 7777") is True

    def test_aadhaar_invalid(self):
        from indialend import aadhaar_validator
        assert aadhaar_validator("123456789012") is False  # starts with 1
        assert aadhaar_validator("0000") is False

    def test_phone_valid(self):
        from indialend import indian_phone_validator
        assert indian_phone_validator("9876543210") is True
        assert indian_phone_validator("+919876543210") is True

    def test_phone_invalid(self):
        from indialend import indian_phone_validator
        assert indian_phone_validator("1234567890") is False
        assert indian_phone_validator("12345") is False

    def test_risk_grade(self):
        from indialend import risk_grade_from_score
        assert risk_grade_from_score(800) == "A"
        assert risk_grade_from_score(720) == "B"
        assert risk_grade_from_score(680) == "C"
        assert risk_grade_from_score(620) == "D"
        assert risk_grade_from_score(500) == "E"

    def test_mask_pan(self):
        from indialend import mask_pan
        assert mask_pan("ABCPP1234K") == "ABCPP****K"

    def test_mask_aadhaar(self):
        from indialend import mask_aadhaar
        assert mask_aadhaar("234567890123") == "XXXX-XXXX-0123"


class TestSyntheticData:
    def test_generate_default_size(self):
        from indialend import generate_synthetic_data
        df = generate_synthetic_data(n_samples=100, seed=42)
        assert len(df) == 100
        assert "default_flag" in df.columns
        assert df["cibil_score"].min() >= 300
        assert df["cibil_score"].max() <= 900

    def test_generate_with_alternatives(self):
        from indialend import generate_synthetic_data
        df = generate_synthetic_data(n_samples=50, include_alternative=True)
        assert "upi_txn_count_3m" in df.columns
        assert "ecom_orders_6m" in df.columns

    def test_generate_reproducible(self):
        from indialend import generate_synthetic_data
        df1 = generate_synthetic_data(n_samples=50, seed=123)
        df2 = generate_synthetic_data(n_samples=50, seed=123)
        pd.testing.assert_frame_equal(df1, df2)

    def test_load_sample_data(self):
        from indialend import load_sample_data
        X_train, X_test, y_train, y_test = load_sample_data(n_samples=200)
        assert len(X_train) + len(X_test) == 200
        assert y_train.isin([0, 1]).all()

    def test_transaction_history(self):
        from indialend import generate_transaction_history
        df = generate_transaction_history(n_accounts=10, months=6)
        assert len(df) == 60
        assert "dpd" in df.columns
        assert "status" in df.columns


# ============================================================
# Data Management
# ============================================================

class TestDataValidator:
    def test_valid_applicant(self):
        from indialend import DataValidator
        v = DataValidator()
        result = v.validate({
            "applicant_id": "APP001",
            "pan": "ABCPP1234K",
            "name": "Raj Kumar",
            "age": 35,
            "phone": "9876543210",
            "employment_type": "Salaried",
            "annual_income": 600000,
            "loan_amount": 400000,
        })
        assert result["valid"] is True

    def test_missing_fields(self):
        from indialend import DataValidator
        v = DataValidator()
        result = v.validate({"name": "Test"})
        assert result["valid"] is False
        assert len(result["errors"]) > 0

    def test_invalid_pan(self):
        from indialend import DataValidator
        v = DataValidator()
        result = v.validate({
            "applicant_id": "APP001",
            "pan": "INVALID",
            "name": "Test",
            "age": 30,
            "phone": "9876543210",
            "employment_type": "Salaried",
            "annual_income": 500000,
            "loan_amount": 200000,
        })
        assert result["valid"] is False


class TestKYCValidator:
    def test_complete_kyc(self):
        from indialend import KYC_Validator
        kyc = KYC_Validator()
        result = kyc.validate_documents({
            "pan": True, "aadhaar": True,
            "salary_slip": True, "photo": True,
        })
        assert result["complete"] is True

    def test_incomplete_kyc(self):
        from indialend import KYC_Validator
        kyc = KYC_Validator()
        result = kyc.validate_documents({"photo": True})
        assert result["complete"] is False
        assert len(result["missing"]) > 0


class TestDuplicateDetection:
    def test_detect_duplicate(self):
        from indialend import DuplicateDetection
        dd = DuplicateDetection()
        dd.register("APP001", {"pan": "ABCPP1234K", "phone": "9876543210"})
        result = dd.check({"pan": "ABCPP1234K", "phone": "9876543210"})
        assert result["is_duplicate"] is True
        assert result["confidence"] >= 0.80

    def test_no_duplicate(self):
        from indialend import DuplicateDetection
        dd = DuplicateDetection()
        dd.register("APP001", {"pan": "ABCPP1234K"})
        result = dd.check({"pan": "XYZPP5678M"})
        assert result["is_duplicate"] is False


# ============================================================
# Credit Scoring
# ============================================================

class TestCreditScoring:
    @pytest.fixture
    def trained_scorer(self):
        from indialend import CreditScorer, load_sample_data
        X_train, X_test, y_train, y_test = load_sample_data(n_samples=500)
        scorer = CreditScorer(model_type="sklearn", cat_type="combined", n_estimators=50)
        scorer.fit(X_train, y_train, calibrate=False)
        return scorer, X_test, y_test

    def test_fit_and_score(self, trained_scorer):
        scorer, X_test, _ = trained_scorer
        scores = scorer.score(X_test)
        assert "cibil_score" in scores.columns
        assert "risk_grade" in scores.columns
        assert scores["cibil_score"].min() >= 300
        assert scores["cibil_score"].max() <= 900

    def test_predict_proba(self, trained_scorer):
        scorer, X_test, _ = trained_scorer
        probs = scorer.predict_proba(X_test)
        assert len(probs) == len(X_test)
        assert probs.min() >= 0
        assert probs.max() <= 1

    def test_evaluate(self, trained_scorer):
        scorer, X_test, y_test = trained_scorer
        metrics = scorer.evaluate(X_test, y_test)
        assert "auc_roc" in metrics
        assert 0 <= metrics["auc_roc"] <= 1
        assert "ks_statistic" in metrics

    def test_save_load(self, trained_scorer, tmp_path):
        scorer, X_test, _ = trained_scorer
        path = str(tmp_path / "model.joblib")
        scorer.save(path)
        from indialend import CreditScorer
        loaded = CreditScorer.load(path)
        scores = loaded.score(X_test)
        assert len(scores) == len(X_test)


class TestCATClassifier:
    def test_classify_cat(self):
        from indialend import CAT_NonCAT_Classifier
        c = CAT_NonCAT_Classifier()
        assert c.classify({"annual_income": 5_000_000}) == "CAT"
        assert c.classify({"annual_income": 500_000}) == "NonCAT"

    def test_classify_dataframe(self):
        from indialend import CAT_NonCAT_Classifier
        c = CAT_NonCAT_Classifier()
        df = pd.DataFrame({"annual_income": [5_000_000, 500_000, 3_000_000]})
        result = c.classify(df)
        assert result.tolist() == ["CAT", "NonCAT", "CAT"]


class TestAlternativeScorer:
    def test_heuristic_score(self):
        from indialend import AlternativeScorer
        scorer = AlternativeScorer()
        result = scorer.score({
            "upi_txn_count_3m": 100,
            "phone_age_months": 48,
            "ecom_orders_6m": 10,
        })
        assert 300 <= result["alt_score"] <= 900
        assert result["confidence"] > 0


# ============================================================
# Bureau Integration
# ============================================================

class TestBureauManager:
    def test_get_score(self):
        from indialend import BureauManager
        bm = BureauManager()
        resp = bm.get_bureau_score({"pan": "ABCPP1234K", "annual_income": 600000, "age": 35})
        assert resp.success is True
        assert 300 <= resp.cibil_score <= 900

    def test_cache(self):
        from indialend import BureauManager
        bm = BureauManager()
        r1 = bm.get_bureau_score({"pan": "TESTPAN001"})
        r2 = bm.get_bureau_score({"pan": "TESTPAN001"})
        assert r1.cibil_score == r2.cibil_score

    def test_to_dict(self):
        from indialend import BureauManager
        bm = BureauManager()
        resp = bm.get_bureau_score({"pan": "ABCPP1234K"})
        d = resp.to_dict()
        assert "cibil_score" in d
        assert "risk_grade" in d


class TestConsentTracker:
    def test_consent_flow(self):
        from indialend import ConsentTracker
        from datetime import datetime
        ct = ConsentTracker()
        ct.record_consent("ABCPP1234K", datetime.now(), validity_period_days=30)
        check = ct.check_consent("ABCPP1234K")
        assert check["has_consent"] is True
        assert check["days_remaining"] >= 29

    def test_no_consent(self):
        from indialend import ConsentTracker
        ct = ConsentTracker()
        check = ct.check_consent("UNKNOWN_PAN")
        assert check["has_consent"] is False


class TestBureauOptimizer:
    def test_should_call(self):
        from indialend import BureauCallOptimizer
        opt = BureauCallOptimizer()
        result = opt.should_call_bureau("NEWPAN12345")
        assert result["should_pull"] is True


# ============================================================
# Income Verification
# ============================================================

class TestIncomeVerifier:
    def test_itr_verification(self):
        from indialend import IncomeVerifier
        v = IncomeVerifier()
        result = v.verify({
            "itr": [{"gross_income": 600000, "taxable_income": 450000, "tax_paid": 15000}]
        })
        assert result["verified_income"] > 0
        assert result["confidence_score"] > 0

    def test_salary_verification(self):
        from indialend import IncomeVerifier
        v = IncomeVerifier()
        result = v.verify({
            "salary_slips": [
                {"gross_salary": 50000, "net_salary": 42000},
                {"gross_salary": 50000, "net_salary": 42000},
                {"gross_salary": 50000, "net_salary": 42000},
            ]
        })
        assert result["verified_income"] > 0

    def test_no_documents(self):
        from indialend import IncomeVerifier
        v = IncomeVerifier()
        result = v.verify({})
        assert result["verified_income"] == 0


# ============================================================
# Policy Engine
# ============================================================

class TestPolicyEngine:
    def test_approval(self):
        from indialend import PolicyEngine
        pe = PolicyEngine()
        result = pe.evaluate({
            "age": 35, "annual_income": 800000, "employment_type": "Salaried",
            "loan_amount": 300000, "loan_product": "Personal Loan", "tenure_months": 36,
            "cibil_score": 750, "foir": 0.30, "dpd_30_12m": 0, "dpd_90_12m": 0,
            "enquiries_3m": 1, "oldest_account_months": 60,
            "written_off_accounts": 0, "existing_emi": 5000,
        })
        assert result["decision"] == "APPROVED"
        assert result["loan_amount"] > 0
        assert result["interest_rate"] > 0

    def test_rejection(self):
        from indialend import PolicyEngine
        pe = PolicyEngine()
        result = pe.evaluate({
            "age": 17, "annual_income": 50000, "employment_type": "Salaried",
            "loan_amount": 100000, "tenure_months": 12,
            "cibil_score": 400, "foir": 0.80, "dpd_90_12m": 3,
            "written_off_accounts": 2, "existing_emi": 10000,
        })
        assert result["decision"] == "REJECTED"

    def test_decision_rules_update(self):
        from indialend import DecisionRules
        original = DecisionRules.get_rule("approval", "min_bureau_score")
        DecisionRules.update_rule("approval", "min_bureau_score", 700)
        assert DecisionRules.get_rule("approval", "min_bureau_score") == 700
        DecisionRules.update_rule("approval", "min_bureau_score", original)


# ============================================================
# Risk Assessment
# ============================================================

class TestRiskAssessment:
    def test_risk_metrics(self):
        from indialend import RiskAssessment
        ra = RiskAssessment()
        result = ra.calculate_risk_metrics(
            {"loan_amount": 500000, "tenure_months": 36, "loan_product": "Personal Loan"},
            bureau_data={"cibil_score": 720, "dpd_30_12m": 0, "dpd_90_12m": 0},
        )
        assert "probability_of_default" in result
        assert "expected_loss" in result
        assert result["expected_loss"] >= 0

    def test_risk_rating(self):
        from indialend import RiskAssessment
        ra = RiskAssessment()
        r1 = ra.calculate_risk_metrics(
            {"loan_amount": 100000, "loan_product": "Personal Loan"},
            bureau_data={"cibil_score": 800},
        )
        r2 = ra.calculate_risk_metrics(
            {"loan_amount": 100000, "loan_product": "Personal Loan"},
            bureau_data={"cibil_score": 500, "dpd_90_12m": 2, "written_off_accounts": 1},
        )
        assert r1["probability_of_default"] < r2["probability_of_default"]


class TestPortfolioMonitor:
    def test_add_and_metrics(self):
        from indialend import PortfolioMonitor
        pm = PortfolioMonitor()
        pm.add_loan({"loan_amount": 500000, "dpd": 0, "expected_loss": 1000})
        pm.add_loan({"loan_amount": 300000, "dpd": 95, "expected_loss": 15000})
        metrics = pm.calculate_portfolio_metrics()
        assert metrics["total_loans"] == 2
        assert metrics["npl_count"] == 1


class TestStressTest:
    def test_stress_scenario(self):
        from indialend import StressTest
        st = StressTest()
        portfolio = pd.DataFrame([
            {"loan_amount": 500000, "probability_of_default": 0.05, "loss_given_default": 0.45, "dpd": 0},
            {"loan_amount": 300000, "probability_of_default": 0.10, "loss_given_default": 0.50, "dpd": 30},
        ])
        result = st.run_stress_test(portfolio, scenario="economic_slowdown")
        assert result["stressed_expected_loss"] > result["base_expected_loss"]


class TestEarlyWarning:
    def test_healthy_loan(self):
        from indialend import EarlyWarningSystem
        ews = EarlyWarningSystem()
        result = ews.check_loan_health({"dpd": 0, "enquiries_3m": 0, "credit_utilization": 0.3})
        assert result["risk_level"] == "HEALTHY"

    def test_stressed_loan(self):
        from indialend import EarlyWarningSystem
        ews = EarlyWarningSystem()
        result = ews.check_loan_health({
            "dpd": 45, "enquiries_3m": 5, "credit_utilization": 0.95,
        })
        assert result["risk_level"] in ("HIGH", "MEDIUM")
        assert len(result["warnings"]) > 0


# ============================================================
# Compliance
# ============================================================

class TestNPAClassifier:
    def test_standard(self):
        from indialend import NPA_Classifier
        result = NPA_Classifier.classify(dpd=0)
        assert result["category"] == "Standard"
        assert result["is_npa"] is False

    def test_sma1(self):
        from indialend import NPA_Classifier
        result = NPA_Classifier.classify(dpd=45)
        assert result["category"] == "SMA-1"

    def test_npa(self):
        from indialend import NPA_Classifier
        result = NPA_Classifier.classify(dpd=100)
        assert result["is_npa"] is True
        assert result["category"] == "Sub-Standard"


class TestDivergenceTracker:
    def test_track_divergence(self):
        from indialend import DivergenceTracker
        dt = DivergenceTracker()
        dt.track("APP1", "APPROVED", "REJECTED")
        dt.track("APP2", "APPROVED", "APPROVED")
        report = dt.get_divergence_report()
        assert report["divergent_count"] == 1
        assert report["total_decisions"] == 2


class TestFairnessAuditor:
    def test_fair_decisions(self):
        from indialend import FairnessAuditor
        fa = FairnessAuditor()
        df = pd.DataFrame({
            "decision": ["APPROVED"] * 50 + ["REJECTED"] * 50,
            "gender": ["M"] * 50 + ["F"] * 50,
        })
        result = fa.audit(df, sensitive_attributes=["gender"])
        assert "gender" in result["attribute_results"]


class TestRBICompliance:
    def test_divergence_check(self):
        from indialend import RBI_Compliance
        c = RBI_Compliance()
        result = c.check_divergence(
            bureau_decision=True,
            internal_decision=False,
            applicant_id="APP001",
        )
        assert result["is_divergent"] is True

    def test_audit_report(self):
        from indialend import RBI_Compliance
        c = RBI_Compliance()
        c.check_divergence("APPROVED", "APPROVED", "APP1")
        report = c.generate_audit_report()
        assert "compliance_status" in report


# ============================================================
# Neural Credit Scorer (PyTorch + Apple Metal GPU)
# ============================================================

class TestNeuralCreditScorer:
    """Tests for PyTorch neural network with Metal GPU acceleration."""

    def test_torch_available(self):
        """Torch should be installed."""
        try:
            import torch
            assert torch.__version__
        except ImportError:
            pytest.skip("PyTorch not installed")

    def test_device_detection(self):
        try:
            from indialend import get_device
            device = get_device()
            assert device.type in ("mps", "cuda", "cpu")
        except ImportError:
            pytest.skip("PyTorch not installed")

    def test_neural_scorer_init(self):
        try:
            from indialend import NeuralCreditScorer
            scorer = NeuralCreditScorer(
                hidden_dims=[64, 32], epochs=3, batch_size=256, verbose=False,
            )
            assert scorer.epochs == 3
            assert "GPU" in scorer.device_name or "CPU" in scorer.device_name
        except ImportError:
            pytest.skip("PyTorch not installed")

    def test_neural_scorer_fit_predict(self):
        try:
            from indialend import NeuralCreditScorer, load_sample_data
            from sklearn.metrics import roc_auc_score

            X_train, X_test, y_train, y_test = load_sample_data(n_samples=1000)

            scorer = NeuralCreditScorer(
                hidden_dims=[32, 16],
                epochs=5,
                batch_size=128,
                patience=5,
                verbose=False,
            )
            scorer.fit(X_train, y_train, X_val=X_test, y_val=y_test)

            # Prediction
            probs = scorer.predict_proba(X_test)
            assert len(probs) == len(X_test)
            assert (probs >= 0).all() and (probs <= 1).all()

            # Evaluation
            metrics = scorer.evaluate(X_test, y_test)
            assert "auc_roc" in metrics
            assert "device" in metrics
            assert metrics["auc_roc"] > 0.5  # at least better than random

            # Score output
            scores = scorer.score(X_test)
            assert "cibil_score" in scores.columns
            assert scores["cibil_score"].between(300, 900).all()
        except ImportError:
            pytest.skip("PyTorch not installed")

    def test_neural_scorer_save_load(self, tmp_path):
        try:
            from indialend import NeuralCreditScorer, load_sample_data

            X_train, X_test, y_train, y_test = load_sample_data(n_samples=500)

            scorer = NeuralCreditScorer(
                hidden_dims=[16, 8], epochs=3, batch_size=128, verbose=False,
            )
            scorer.fit(X_train, y_train, X_val=X_test, y_val=y_test)

            path = str(tmp_path / "neural.joblib")
            scorer.save(path)

            loaded = NeuralCreditScorer.load(path)
            probs_orig = scorer.predict_proba(X_test)
            probs_loaded = loaded.predict_proba(X_test)
            np.testing.assert_allclose(probs_orig, probs_loaded, rtol=1e-4)
        except ImportError:
            pytest.skip("PyTorch not installed")

    def test_focal_loss(self):
        try:
            import torch
            from indialend.neural_scorer import FocalLoss

            loss_fn = FocalLoss(alpha=0.25, gamma=2.0)
            inputs = torch.tensor([0.1, 0.9, 0.3, 0.7])
            targets = torch.tensor([0.0, 1.0, 0.0, 1.0])
            loss = loss_fn(inputs, targets)
            assert loss.item() >= 0
        except ImportError:
            pytest.skip("PyTorch not installed")


# ============================================================
# Ensemble Credit Scorer
# ============================================================

class TestEnsembleCreditScorer:
    """Tests for multi-model ensemble."""

    def test_model_factory(self):
        from indialend import ModelFactory
        factory = ModelFactory()
        assert factory.create_hist_gbm() is not None
        assert factory.create_random_forest() is not None
        assert factory.create_extra_trees() is not None
        assert factory.create_logistic_regression() is not None

    def test_ensemble_init_default_models(self):
        from indialend import EnsembleCreditScorer
        ens = EnsembleCreditScorer(verbose=False)
        # Should have at least 4 sklearn models
        assert len(ens._models) >= 4
        assert "HistGBM" in ens._models
        assert "RandomForest" in ens._models
        assert "ExtraTrees" in ens._models
        assert "LogisticRegression" in ens._models

    def test_ensemble_fit_predict(self):
        from indialend import EnsembleCreditScorer, load_sample_data

        X_train, X_test, y_train, y_test = load_sample_data(n_samples=1000)

        ens = EnsembleCreditScorer(verbose=False, calibrate=False)
        ens.fit(X_train, y_train, X_val=X_test, y_val=y_test)

        assert ens._fitted is True
        assert len(ens._fitted_models) >= 4

        # Predictions
        probs = ens.predict_proba(X_test)
        assert len(probs) == len(X_test)
        assert (probs >= 0).all() and (probs <= 1).all()

        # Evaluation
        metrics = ens.evaluate(X_test, y_test)
        assert metrics["auc_roc"] > 0.7

        # Score output
        scores = ens.score(X_test)
        assert "cibil_score" in scores.columns
        assert scores["cibil_score"].between(300, 900).all()

    def test_ensemble_compare_models(self):
        from indialend import EnsembleCreditScorer, load_sample_data

        X_train, X_test, y_train, y_test = load_sample_data(n_samples=800)

        ens = EnsembleCreditScorer(verbose=False, calibrate=False)
        ens.fit(X_train, y_train, X_val=X_test, y_val=y_test)

        comparison = ens.compare_models(X_test, y_test)
        assert "Model" in comparison.columns
        assert "AUC" in comparison.columns
        assert len(comparison) >= 5  # base models + ensemble

    def test_ensemble_save_load(self, tmp_path):
        from indialend import EnsembleCreditScorer, load_sample_data

        X_train, X_test, y_train, y_test = load_sample_data(n_samples=500)

        ens = EnsembleCreditScorer(verbose=False, calibrate=False)
        ens.fit(X_train, y_train, X_val=X_test, y_val=y_test)

        path = str(tmp_path / "ensemble.joblib")
        ens.save(path)

        loaded = EnsembleCreditScorer.load(path)
        probs_orig = ens.predict_proba(X_test)
        probs_loaded = loaded.predict_proba(X_test)
        np.testing.assert_allclose(probs_orig, probs_loaded, rtol=1e-4)

    def test_ensemble_predict_all_models(self):
        from indialend import EnsembleCreditScorer, load_sample_data

        X_train, X_test, y_train, y_test = load_sample_data(n_samples=500)

        ens = EnsembleCreditScorer(verbose=False, calibrate=False)
        ens.fit(X_train, y_train, X_val=X_test, y_val=y_test)

        all_probs = ens.predict_proba_all(X_test)
        assert "Ensemble" in all_probs
        assert len(all_probs) >= 5

    def test_ensemble_feature_importance(self):
        from indialend import EnsembleCreditScorer, load_sample_data

        X_train, X_test, y_train, y_test = load_sample_data(n_samples=500)

        ens = EnsembleCreditScorer(verbose=False, calibrate=False)
        ens.fit(X_train, y_train, X_val=X_test, y_val=y_test)

        importance = ens.feature_importance(top_n=10)
        # Tree models should have importance (may be empty if only LR was available)
        if len(importance) > 0:
            assert "feature" in importance.columns
            assert "importance" in importance.columns


# ============================================================
# Large-scale data generation
# ============================================================

class TestLargeDataGeneration:
    """Verify chunked generation produces valid large datasets."""

    def test_chunked_generation(self):
        from indialend import generate_synthetic_data

        # 60K triggers chunking (chunk_size=50K)
        df = generate_synthetic_data(n_samples=60_000, chunk_size=50_000)
        assert len(df) == 60_000
        # Applicant IDs should be unique
        assert df["applicant_id"].nunique() == 60_000
        # Default rate should be approximately correct
        assert 0.04 < df["default_flag"].mean() < 0.12

