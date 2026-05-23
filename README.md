# IndiaLend

**Unified Credit Decision Engine for Indian NBFC Lending**

[![PyPI version](https://badge.fury.io/py/indialend.svg)](https://pypi.org/project/indialend/)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

One-stop Python library for Indian fintech credit decisioning: credit scoring, bureau integration, income verification, policy engine, risk assessment, and RBI compliance.

## Why IndiaLend?

Indian NBFCs and fintechs face unique challenges that global credit scoring libraries don't address:

- **RBI CAT/NonCAT segmentation** - Regulatory requirement for different risk treatment based on income threshold (Rs 30 lakh)
- **Bureau integration** - CIBIL, Equifax, CRIF with consent tracking and smart caching
- **CIBIL-scale scoring** - 300-900 scale that Indian lenders understand
- **Indian document validation** - PAN, Aadhaar, GSTIN, IFSC validators
- **NPA classification** - SMA-0/1/2, Sub-Standard, Doubtful, Loss per RBI norms
- **Alternative scoring** - UPI, phone, e-commerce data for thin-file applicants (~300M credit-invisible Indians)

## Features

### Credit Scoring
- Multi-model ensemble (HistGBM, Random Forest, Extra Trees, Logistic Regression)
- PyTorch neural network with **Apple Silicon Metal GPU** acceleration
- CAT/NonCAT segment-specific models
- Probability calibration (Platt scaling)
- Alternative data scoring for thin-file applicants

### Bureau Integration
- CIBIL, Equifax, CRIF bureau simulation with deterministic responses
- Consent tracking (RBI mandate)
- Smart caching and fallback ordering
- Bureau call cost optimization

### Income Verification
- Priority-based: ITR > GST > Salary Slip > Bank Statement
- Tax estimation (new regime slabs)
- GST turnover analysis
- Salary consistency checking

### Policy Engine
- Business Rules Engine (BRE) with eligibility knockout + approval criteria
- Risk-based dynamic pricing
- EMI calculation
- Batch evaluation

### Risk Assessment
- PD/LGD/EAD/Expected Loss (Basel III IRB approach)
- Portfolio monitoring with NPL tracking
- Default prediction (6/12/24 month horizons)
- Early Warning System
- Stress testing (5 scenarios)

### RBI Compliance
- NPA classification per RBI norms
- Divergence tracking (bureau vs internal)
- Fairness auditing (4/5ths rule)
- Provisioning rate calculation

## Installation

```bash
pip install indialend
```

With optional dependencies:

```bash
# Apple Silicon Metal GPU support (neural network)
pip install indialend[gpu]

# All optional dependencies
pip install indialend[all]
```

## Quick Start

```python
from indialend import CreditScorer, load_sample_data

# Load sample data (1000 synthetic Indian credit applications)
X_train, X_test, y_train, y_test = load_sample_data()

# Train credit scoring model
scorer = CreditScorer()
scorer.fit(X_train, y_train)

# Score applicants (CIBIL-like 300-900 scale)
scores = scorer.score(X_test)
print(scores.head())
#    default_probability  cibil_score risk_grade segment
# 0             0.023145          886          A  NonCAT
# 1             0.156789          806          A     CAT
# 2             0.892345          365          E  NonCAT
```

### Neural Network with Metal GPU

```python
from indialend import NeuralCreditScorer

# Train on Apple Silicon Metal GPU
neural = NeuralCreditScorer(
    epochs=100,
    batch_size=2048,
    device="auto",  # auto-detects MPS/CUDA/CPU
    verbose=True,
)
neural.fit(X_train, y_train, X_val=X_test, y_val=y_test)
scores = neural.score(X_test)
```

### Multi-Model Ensemble

```python
from indialend import EnsembleCreditScorer

# Train 6+ models and combine with AUC-weighted averaging
ensemble = EnsembleCreditScorer(verbose=True)
ensemble.fit(X_train, y_train, X_val=X_test, y_val=y_test)

# Compare all models
comparison = ensemble.compare_models(X_test, y_test)
print(comparison)
```

### Full Training Pipeline

```bash
# Train all models on 200K synthetic data with Metal GPU
python train_models.py

# Quick test run
python train_models.py --quick

# Large dataset with more epochs
python train_models.py --samples 500000 --epochs 200
```

### Bureau + Income + Policy Decision

```python
from indialend import BureauManager, IncomeVerifier, PolicyEngine

# Bureau check
bureau = BureauManager()
report = bureau.fetch("ABCDE1234F", bureau_type="CIBIL")

# Income verification
verifier = IncomeVerifier()
income = verifier.verify({"itr_income": 800000, "salary_monthly": 65000})

# Policy decision
engine = PolicyEngine()
decision = engine.evaluate({
    "annual_income": 800000,
    "cibil_score": 720,
    "loan_amount": 500000,
    "employment_type": "Salaried",
    "age": 32,
})
print(decision["decision"])  # "APPROVED" / "REJECTED" / "MANUAL_REVIEW"
```

## Synthetic Data

Generate realistic Indian credit application data:

```python
from indialend import generate_synthetic_data

# 200K applications with 69 correlated features
df = generate_synthetic_data(n_samples=200_000)
print(f"Shape: {df.shape}")
print(f"Default rate: {df['default_flag'].mean():.2%}")
print(f"CAT segment: {(df['cat_segment'] == 'CAT').mean():.1%}")
```

Features include demographics, geography (4-tier cities), employment, income, bureau history, alternative data (UPI, phone, e-commerce), and realistic correlations.

## Architecture

```
indialend/
  credit_scoring.py    - CreditScorer, AlternativeScorer, CAT/NonCAT
  neural_scorer.py     - NeuralCreditScorer (PyTorch + Metal GPU)
  ensemble_scorer.py   - EnsembleCreditScorer, ModelFactory
  bureau_integration.py - BureauManager, ConsentTracker
  income_verification.py - IncomeVerifier, ITR/GST/Salary parsers
  policy_engine.py     - PolicyEngine, BRE, DecisionRules
  risk_assessment.py   - PD/LGD/EAD, PortfolioMonitor, StressTest
  compliance.py        - NPA classification, fairness auditing
  data_management.py   - Validators, KYC, deduplication
  utils.py             - Synthetic data, PAN/Aadhaar validators
```

## Hardware Requirements

- **Minimum**: Any machine with Python 3.9+ and 4GB RAM
- **Recommended**: Apple Silicon Mac (M1/M2/M3/M4) for Metal GPU acceleration
- **Training 200K samples**: ~16GB RAM, ~5 minutes on M4

## License

MIT License. See [LICENSE](LICENSE) for details.
