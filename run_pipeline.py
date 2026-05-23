#!/usr/bin/env python3
"""
IndiaLend - Full End-to-End Pipeline
=====================================
Generates synthetic data, trains models on Apple Silicon M4,
runs the complete decision pipeline, and validates all modules.
"""

import os
import sys
import time
import warnings

warnings.filterwarnings("ignore")

# Add project to path
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd

print("=" * 70)
print("  IndiaLend - Unified Credit Decision Engine")
print("  Full Pipeline Execution on Apple Silicon M4")
print("=" * 70)


# =========================================================================
# STEP 1: Generate Synthetic Data (10,000 applications)
# =========================================================================
print("\n[1/8] Generating synthetic Indian credit data...")
t0 = time.time()

from indialend import generate_synthetic_data, load_sample_data

# Full dataset for training
df_full = generate_synthetic_data(
    n_samples=10_000,
    seed=42,
    default_rate=0.08,
    include_alternative=True,
    include_bureau_history=True,
)

print(f"  Generated {len(df_full):,} applications in {time.time()-t0:.2f}s")
print(f"  Features: {len(df_full.columns)} columns")
print(f"  Default rate: {df_full['default_flag'].mean():.1%}")
print(f"  CAT segment: {(df_full['cat_segment'] == 'CAT').mean():.1%}")
print(f"  Income range: Rs {df_full['annual_income'].min():,.0f} - Rs {df_full['annual_income'].max():,.0f}")
print(f"  CIBIL range: {df_full['cibil_score'].min()} - {df_full['cibil_score'].max()}")
print(f"  Products: {df_full['loan_product'].nunique()} types")

# Save data
os.makedirs("data", exist_ok=True)
df_full.to_csv("data/synthetic_credit_data.csv", index=False)
print(f"  Saved to data/synthetic_credit_data.csv")


# =========================================================================
# STEP 2: Data Validation & KYC
# =========================================================================
print("\n[2/8] Running data validation & duplicate detection...")
t0 = time.time()

from indialend import DataValidator, KYC_Validator, DuplicateDetection

validator = DataValidator()
sample = df_full.iloc[0].to_dict()
result = validator.validate(sample)
print(f"  Validation result: valid={result['valid']}, errors={len(result['errors'])}, warnings={len(result['warnings'])}")

kyc = KYC_Validator()
kyc_result = kyc.validate_documents(
    {"pan": "ABCDE1234F", "aadhaar": "234567890123", "salary_slip": True, "photo": True},
    employment_type="Salaried",
)
print(f"  KYC check: complete={kyc_result['complete']}, score={kyc_result['score']}")

dup_detector = DuplicateDetection()
dup_results = dup_detector.find_duplicates(df_full.head(500))
dup_count = dup_results["is_duplicate"].sum()
print(f"  Duplicate scan (500 apps): {dup_count} duplicates found")
print(f"  Completed in {time.time()-t0:.2f}s")


# =========================================================================
# STEP 3: Train Credit Scoring Models (LightGBM on M4)
# =========================================================================
print("\n[3/8] Training credit scoring models (LightGBM)...")
t0 = time.time()

from indialend import CreditScorer, AlternativeScorer, CAT_NonCAT_Classifier

# Prepare features
X_train, X_test, y_train, y_test = load_sample_data(n_samples=10_000, seed=42)
print(f"  Train: {len(X_train):,} | Test: {len(X_test):,}")
print(f"  Train default rate: {y_train.mean():.1%} | Test: {y_test.mean():.1%}")

# Train LightGBM model (segment-aware)
scorer = CreditScorer(
    model_type="lightgbm",
    cat_type="combined",
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
)
scorer.fit(X_train, y_train)
print(f"  LightGBM trained in {time.time()-t0:.2f}s")

# Evaluate
metrics = scorer.evaluate(X_test, y_test)
print(f"  AUC-ROC: {metrics['auc_roc']:.4f}")
print(f"  KS Statistic: {metrics['ks_statistic']:.4f}")
print(f"  Gini: {metrics['gini_coefficient']:.4f}")
print(f"  Precision (default): {metrics['precision_default']:.4f}")
print(f"  Recall (default): {metrics['recall_default']:.4f}")

# Cross-validation
print("  Running 5-fold cross-validation...")
cv_results = scorer.cross_validate(X_train, y_train, n_folds=5)
print(f"  CV AUC: {cv_results['mean_auc']:.4f} ± {cv_results['std_auc']:.4f}")
print(f"  CV KS:  {cv_results['mean_ks']:.4f} ± {cv_results['std_ks']:.4f}")

# Feature importance
fi = scorer.feature_importance()
print(f"  Top 5 features:")
for _, row in fi.head(5).iterrows():
    print(f"    {row['feature']}: {row['importance']:.0f}")

# Train second model for comparison (XGBoost or sklearn)
print("  Training comparison model...")
t1 = time.time()
scorer_xgb = CreditScorer(
    model_type="xgboost",
    cat_type="combined",
    n_estimators=500,
    learning_rate=0.05,
)
scorer_xgb.fit(X_train, y_train)
xgb_metrics = scorer_xgb.evaluate(X_test, y_test)
actual_type = scorer_xgb.model_type.upper()
print(f"  {actual_type} AUC: {xgb_metrics['auc_roc']:.4f} (trained in {time.time()-t1:.2f}s)")

# Save best model
os.makedirs("models", exist_ok=True)
scorer.save("models/credit_scorer_lgbm.joblib")
print(f"  Model saved to models/credit_scorer_lgbm.joblib")

# Score full dataset
scores_df = scorer.score(X_test)
print(f"\n  Score distribution:")
print(f"    Mean: {scores_df['cibil_score'].mean():.0f}")
print(f"    Median: {scores_df['cibil_score'].median():.0f}")
print(f"    Grade A: {(scores_df['risk_grade'] == 'A').mean():.1%}")
print(f"    Grade B: {(scores_df['risk_grade'] == 'B').mean():.1%}")
print(f"    Grade C: {(scores_df['risk_grade'] == 'C').mean():.1%}")
print(f"    Grade D: {(scores_df['risk_grade'] == 'D').mean():.1%}")
print(f"    Grade E: {(scores_df['risk_grade'] == 'E').mean():.1%}")

# Alternative scorer
print("\n  Training alternative data scorer...")
alt_features = [c for c in df_full.columns if c.startswith(("upi_", "phone_", "ecom_", "app_", "avg_recharge", "data_usage"))]
alt_scorer = AlternativeScorer()
if alt_features:
    alt_result = alt_scorer.score(df_full.iloc[0].to_dict())
    print(f"  Alt score (heuristic): {alt_result['alt_score']} ({alt_result['risk_grade']}), confidence: {alt_result['confidence']}")


# =========================================================================
# STEP 4: Bureau Integration
# =========================================================================
print("\n[4/8] Testing bureau integration...")
t0 = time.time()

from indialend import BureauManager, ConsentTracker, BureauCallOptimizer

bureau = BureauManager()

# Get bureau score for a sample applicant
sample_applicant = {
    "pan": "ABCPP1234K",
    "name": "Raj Kumar",
    "age": 35,
    "annual_income": 600_000,
    "employment_type": "Salaried",
    "num_existing_loans": 1,
    "existing_emi": 8000,
}

bureau_response = bureau.get_bureau_score(sample_applicant, soft_pull=True)
print(f"  Bureau: {bureau_response.bureau} | Score: {bureau_response.cibil_score} ({bureau_response.risk_grade})")
print(f"  Total accounts: {bureau_response.total_accounts} | DPD 30+: {bureau_response.dpd_30_12m}")
print(f"  NPA flagged: {bureau_response.is_npa_flagged}")

# Consent tracking
from datetime import datetime
consent = ConsentTracker()
consent.record_consent("ABCPP1234K", datetime.now(), validity_period_days=30)
consent_check = consent.check_consent("ABCPP1234K")
print(f"  Consent: valid={consent_check['has_consent']}, days_remaining={consent_check['days_remaining']}")

# Call optimizer
optimizer = BureauCallOptimizer()
opt_result = optimizer.should_call_bureau("ABCPP1234K")
print(f"  Optimizer: should_pull={opt_result['should_pull']}, cost=Rs {opt_result['estimated_cost']}")
print(f"  Completed in {time.time()-t0:.2f}s")


# =========================================================================
# STEP 5: Income Verification
# =========================================================================
print("\n[5/8] Testing income verification...")
t0 = time.time()

from indialend import IncomeVerifier

verifier = IncomeVerifier()

# Test with ITR data
income_result = verifier.verify(
    documents={
        "itr": [
            {"gross_income": 540_000, "deductions": 150_000, "taxable_income": 390_000,
             "tax_paid": 12_000, "assessment_year": "2023-24", "itr_form": "ITR-1"},
            {"gross_income": 510_000, "deductions": 140_000, "taxable_income": 370_000,
             "tax_paid": 10_000, "assessment_year": "2022-23", "itr_form": "ITR-1"},
        ],
        "salary_slips": [
            {"gross_salary": 45_000, "basic": 22_500, "hra": 9_000, "da": 4_500,
             "special_allowance": 9_000, "pf_deduction": 2_700, "tds": 1_000,
             "net_salary": 41_300, "month": "2024-01"},
            {"gross_salary": 45_000, "basic": 22_500, "hra": 9_000, "da": 4_500,
             "special_allowance": 9_000, "pf_deduction": 2_700, "tds": 1_000,
             "net_salary": 41_300, "month": "2024-02"},
            {"gross_salary": 45_000, "basic": 22_500, "hra": 9_000, "da": 4_500,
             "special_allowance": 9_000, "pf_deduction": 2_700, "tds": 1_000,
             "net_salary": 41_300, "month": "2024-03"},
        ],
    },
    employment_type="Salaried",
    declared_income=540_000,
)
print(f"  Verified income: Rs {income_result['verified_income']:,.0f}")
print(f"  Confidence: {income_result['confidence_score']:.0%}")
print(f"  Sources: {income_result['sources']}")
print(f"  Primary: {income_result['primary_source']}")
print(f"  Discrepancy: {income_result['discrepancy_flag']}")
print(f"  Completed in {time.time()-t0:.2f}s")


# =========================================================================
# STEP 6: Policy Engine Decision
# =========================================================================
print("\n[6/8] Running policy engine...")
t0 = time.time()

from indialend import PolicyEngine

policy = PolicyEngine()

# Evaluate the sample applicant
decision = policy.evaluate(
    applicant_data={
        "applicant_id": "APP001",
        "age": 35,
        "annual_income": 540_000,
        "employment_type": "Salaried",
        "loan_amount": 400_000,
        "loan_product": "Auto Loan",
        "tenure_months": 60,
        "cibil_score": bureau_response.cibil_score,
        "foir": 0.35,
        "dpd_30_12m": bureau_response.dpd_30_12m,
        "dpd_90_12m": bureau_response.dpd_90_12m,
        "enquiries_3m": bureau_response.enquiries_3m,
        "oldest_account_months": bureau_response.oldest_account_months,
        "written_off_accounts": bureau_response.written_off_accounts,
        "asset_value": 600_000,
        "existing_emi": 8000,
    },
    bureau_response=bureau_response,
    income_verification=income_result,
)

print(f"  Decision: {decision['decision']}")
print(f"  Loan amount: Rs {decision['loan_amount']:,.0f}")
print(f"  Interest rate: {decision['interest_rate']:.1f}%")
print(f"  Tenure: {decision['tenure']} months")
print(f"  EMI: Rs {decision['emi']:,.0f}")
print(f"  Rules passed: {decision['rules_passed']}/{decision['rules_passed'] + decision['rules_failed']}")
if decision['conditions']:
    print(f"  Conditions: {decision['conditions']}")
print(f"  Completed in {time.time()-t0:.2f}s")

# Batch evaluation
print("  Running batch evaluation (100 applicants)...")
batch_df = df_full.head(100).copy()
batch_results = policy.batch_evaluate(batch_df)
print(f"  Approval rate: {batch_results.attrs.get('approval_rate', 0):.1%}")
decision_counts = batch_results["decision"].value_counts()
for dec, count in decision_counts.items():
    print(f"    {dec}: {count}")


# =========================================================================
# STEP 7: Risk Assessment & Portfolio Monitoring
# =========================================================================
print("\n[7/8] Running risk assessment & portfolio monitoring...")
t0 = time.time()

from indialend import (
    RiskAssessment, PortfolioMonitor, DefaultPredictor,
    EarlyWarningSystem, StressTest
)
from indialend import generate_transaction_history

# Individual risk assessment
risk_engine = RiskAssessment()
risk_metrics = risk_engine.calculate_risk_metrics(
    loan_data={
        "loan_amount": decision["loan_amount"],
        "tenure_months": decision["tenure"],
        "employment_type": "Salaried",
        "loan_product": "Auto Loan",
        "ltv_ratio": 0.67,
        "risk_grade": "B",
    },
    bureau_data=bureau_response.to_dict(),
)
print(f"  PD: {risk_metrics['probability_of_default']:.2%}")
print(f"  LGD: {risk_metrics['loss_given_default']:.2%}")
print(f"  EAD: Rs {risk_metrics['exposure_at_default']:,.0f}")
print(f"  Expected Loss: Rs {risk_metrics['expected_loss']:,.0f}")
print(f"  Risk Rating: {risk_metrics['risk_rating']}")
print(f"  Capital Requirement: Rs {risk_metrics['capital_requirement']:,.0f}")

# Portfolio monitoring
monitor = PortfolioMonitor()
for _, row in df_full.head(200).iterrows():
    risk = risk_engine.calculate_risk_metrics(
        {"loan_amount": row["loan_amount"], "tenure_months": row["tenure_months"],
         "loan_product": row["loan_product"], "employment_type": row["employment_type"],
         "risk_grade": "C", "ltv_ratio": row.get("ltv_ratio", 0.8)},
        bureau_data={"cibil_score": row["cibil_score"], "dpd_30_12m": row["dpd_30_12m"],
                     "dpd_90_12m": row["dpd_90_12m"]},
    )
    loan = {**row.to_dict(), **risk}
    loan["dpd"] = np.random.choice([0, 0, 0, 0, 5, 15, 30, 60, 95], p=[0.6, 0.1, 0.05, 0.05, 0.05, 0.05, 0.04, 0.03, 0.03])
    monitor.add_loan(loan)

portfolio_metrics = monitor.calculate_portfolio_metrics()
print(f"\n  Portfolio ({monitor.portfolio_size} loans):")
print(f"    Total exposure: Rs {portfolio_metrics['total_exposure']:,.0f}")
print(f"    NPL count: {portfolio_metrics['npl_count']} ({portfolio_metrics['npl_ratio']:.1%})")
print(f"    SMA-0: {portfolio_metrics['sma0_count']} | SMA-1: {portfolio_metrics['sma1_count']} | SMA-2: {portfolio_metrics['sma2_count']}")
print(f"    Total expected loss: Rs {portfolio_metrics['total_expected_loss']:,.0f}")
print(f"    Portfolio health: {portfolio_metrics['portfolio_health']}")

# Early Warning System
ews = EarlyWarningSystem()
txn_history = generate_transaction_history(n_accounts=50, months=12, seed=42)
sample_loan = monitor._loans[0]
health = ews.check_loan_health(sample_loan)
print(f"\n  Early Warning Sample:")
print(f"    Risk level: {health['risk_level']} (score: {health['risk_score']})")
if health['warnings']:
    for w in health['warnings'][:3]:
        print(f"    Warning: {w}")

# Default Prediction
predictor = DefaultPredictor()
active_df = pd.DataFrame(monitor._loans[:50])
predictions = predictor.predict_defaults(active_df)
high_risk = predictions[predictions["risk_flag"] == "HIGH"]
print(f"\n  Default Predictions (50 loans):")
print(f"    HIGH risk: {len(high_risk)}")
print(f"    MEDIUM risk: {(predictions['risk_flag'] == 'MEDIUM').sum()}")
print(f"    WATCH: {(predictions['risk_flag'] == 'WATCH').sum()}")
print(f"    LOW risk: {(predictions['risk_flag'] == 'LOW').sum()}")

# Stress Testing
stress = StressTest()
stress_result = stress.run_stress_test(
    pd.DataFrame(monitor._loans), scenario="economic_slowdown"
)
print(f"\n  Stress Test (Economic Slowdown):")
print(f"    Stressed EL: Rs {stress_result['stressed_expected_loss']:,.0f}")
print(f"    Incremental loss: Rs {stress_result['incremental_loss']:,.0f} ({stress_result['loss_increase_pct']:.0f}%)")
print(f"    Estimated new NPAs: {stress_result['estimated_npl_increase']}")

# Run all scenarios
all_scenarios = stress.run_all_scenarios(pd.DataFrame(monitor._loans))
print(f"\n  All Stress Scenarios:")
for _, s in all_scenarios.iterrows():
    print(f"    {s['scenario']:25s} | EL: Rs {s['stressed_expected_loss']:>12,.0f} | NPL: {s['estimated_npl_ratio']:.1%}")

print(f"\n  Completed in {time.time()-t0:.2f}s")


# =========================================================================
# STEP 8: RBI Compliance
# =========================================================================
print("\n[8/8] Running compliance checks...")
t0 = time.time()

from indialend import RBI_Compliance, FairnessAuditor, NPA_Classifier

compliance = RBI_Compliance()

# Track divergence
compliance.check_divergence(
    bureau_decision=bureau_response.cibil_score >= 700,
    internal_decision=decision["decision"] == "APPROVED",
    applicant_id="APP001",
    bureau_score=bureau_response.cibil_score,
)

# Batch divergence tracking
for _, row in batch_results.head(50).iterrows():
    compliance.check_divergence(
        bureau_decision="APPROVED" if np.random.random() > 0.3 else "REJECTED",
        internal_decision=row["decision"],
        applicant_id=row.get("applicant_id", ""),
    )

# Generate audit report
audit_report = compliance.generate_audit_report("2024-04")
print(f"  Compliance status: {audit_report['compliance_status']}")
print(f"  Divergence rate: {audit_report['divergence_summary']['divergence_rate']:.1%}")
print(f"  Total decisions tracked: {audit_report['divergence_summary']['total_decisions']}")

# NPA classification
npa_result = NPA_Classifier.classify(dpd=45)
print(f"\n  NPA Classification (DPD=45):")
print(f"    Category: {npa_result['category']}")
print(f"    Provision rate: {npa_result['provision_rate']:.0%}")
print(f"    Action: {npa_result['action_required']}")

# Fairness audit
auditor = FairnessAuditor()
# Prepare decisions with demographic data
audit_data = batch_results.copy()
audit_data["gender"] = np.random.choice(["M", "F"], size=len(audit_data), p=[0.68, 0.32])
audit_data["age"] = np.random.randint(21, 60, size=len(audit_data))
audit_data["city_tier"] = np.random.choice(["metro", "tier1", "tier2", "rural"], size=len(audit_data))

fairness = auditor.audit(audit_data, sensitive_attributes=["gender", "city_tier"])
print(f"\n  Fairness Audit:")
print(f"    Overall fair: {fairness['overall_fair']}")
for attr, result in fairness["attribute_results"].items():
    print(f"    {attr}: 4/5ths rule {'PASS' if result['passes_4_5_rule'] else 'FAIL'} "
          f"(disparity: {result['max_disparity']:.1%})")

print(f"  Completed in {time.time()-t0:.2f}s")


# =========================================================================
# SUMMARY
# =========================================================================
print("\n" + "=" * 70)
print("  PIPELINE COMPLETE - SUMMARY")
print("=" * 70)
print(f"""
  Data Generated:      {len(df_full):,} applications, {len(df_full.columns)} features
  Model Performance:   AUC={metrics['auc_roc']:.4f}, KS={metrics['ks_statistic']:.4f}, Gini={metrics['gini_coefficient']:.4f}
  CV Performance:      AUC={cv_results['mean_auc']:.4f} ± {cv_results['std_auc']:.4f}
  Bureau Integration:  CIBIL score={bureau_response.cibil_score} ({bureau_response.risk_grade})
  Income Verified:     Rs {income_result['verified_income']:,.0f} ({income_result['confidence_score']:.0%} confidence)
  Decision:            {decision['decision']} @ {decision['interest_rate']:.1f}% for Rs {decision['loan_amount']:,.0f}
  Risk Assessment:     PD={risk_metrics['probability_of_default']:.2%}, EL=Rs {risk_metrics['expected_loss']:,.0f}
  Portfolio:           {monitor.portfolio_size} loans, NPL={portfolio_metrics['npl_ratio']:.1%}
  Compliance:          {audit_report['compliance_status']}
  Fairness:            {'PASS' if fairness['overall_fair'] else 'NEEDS REVIEW'}
""")
print("  Model saved:  models/credit_scorer_lgbm.joblib")
print("  Data saved:   data/synthetic_credit_data.csv")
print("=" * 70)
print("  IndiaLend is ready for production deployment!")
print("=" * 70)
