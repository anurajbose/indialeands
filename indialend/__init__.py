"""
IndiaLend - Unified Credit Decision Engine for Indian Lending
=============================================================

One-stop solution for Indian NBFC credit decisioning:
credit scoring, bureau integration, income verification,
policy engine, risk assessment, and RBI compliance.

Quick Start
-----------
>>> from indialend import CreditScorer, load_sample_data
>>> X_train, X_test, y_train, y_test = load_sample_data()
>>> scorer = CreditScorer()
>>> scorer.fit(X_train, y_train)
>>> scores = scorer.score(X_test)
"""

__version__ = "1.0.0"
__author__ = "IndiaLend Team"

# --- Credit Scoring ---
from .credit_scoring import (
    CreditScorer,
    AlternativeScorer,
    CAT_NonCAT_Classifier,
)

# --- Bureau Integration ---
from .bureau_integration import (
    BureauManager,
    BureauResponse,
    BureauType,
    ConsentTracker,
    BureauCallOptimizer,
)

# --- Income Verification ---
from .income_verification import (
    IncomeVerifier,
    IncomeResult,
    ITR_Parser,
    GST_Parser,
    SalarySlipParser,
    BankStatementAnalyzer,
)

# --- Policy Engine ---
from .policy_engine import (
    PolicyEngine,
    BRE_Evaluator,
    DecisionRules,
)

# --- Risk Assessment ---
from .risk_assessment import (
    RiskAssessment,
    PortfolioMonitor,
    DefaultPredictor,
    EarlyWarningSystem,
    StressTest,
)

# --- Compliance ---
from .compliance import (
    RBI_Compliance,
    NPA_Classifier,
    DivergenceTracker,
    FairnessAuditor,
)

# --- Data Management ---
from .data_management import (
    DataValidator,
    KYC_Validator,
    DuplicateDetection,
    DataPreprocessor,
    ApplicantProfile,
    LoanApplication,
)

# --- Neural Scorer (PyTorch + Apple Metal) ---
from .neural_scorer import NeuralCreditScorer, CreditNet, FocalLoss, get_device

# --- Ensemble Scorer ---
from .ensemble_scorer import EnsembleCreditScorer, ModelFactory

# --- Utilities ---
from .utils import (
    generate_synthetic_data,
    load_sample_data,
    generate_transaction_history,
    set_seed,
    pan_validator,
    aadhaar_validator,
    indian_phone_validator,
    gstin_validator,
    ifsc_validator,
    risk_grade_from_score,
    mask_pan,
    mask_aadhaar,
    hash_pii,
)

__all__ = [
    # Credit Scoring
    "CreditScorer", "AlternativeScorer", "CAT_NonCAT_Classifier",
    # Bureau
    "BureauManager", "BureauResponse", "BureauType",
    "ConsentTracker", "BureauCallOptimizer",
    # Income
    "IncomeVerifier", "IncomeResult",
    "ITR_Parser", "GST_Parser", "SalarySlipParser", "BankStatementAnalyzer",
    # Policy
    "PolicyEngine", "BRE_Evaluator", "DecisionRules",
    # Risk
    "RiskAssessment", "PortfolioMonitor", "DefaultPredictor",
    "EarlyWarningSystem", "StressTest",
    # Compliance
    "RBI_Compliance", "NPA_Classifier", "DivergenceTracker", "FairnessAuditor",
    # Data
    "DataValidator", "KYC_Validator", "DuplicateDetection",
    "DataPreprocessor", "ApplicantProfile", "LoanApplication",
    # Neural Scorer
    "NeuralCreditScorer", "CreditNet", "FocalLoss", "get_device",
    # Ensemble
    "EnsembleCreditScorer", "ModelFactory",
    # Utils
    "generate_synthetic_data", "load_sample_data",
    "generate_transaction_history", "set_seed",
    "pan_validator", "aadhaar_validator", "indian_phone_validator",
    "gstin_validator", "ifsc_validator", "risk_grade_from_score",
    "mask_pan", "mask_aadhaar", "hash_pii",
]
