# Changelog

All notable changes to IndiaLend will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2026-04-05

### Added
- Initial production release of IndiaLend
- **Credit Scoring Module**
  - `CreditScorer` with LightGBM/XGBoost/sklearn backends
  - `CAT_NonCAT_Classifier` for RBI-mandated income segmentation
  - `AlternativeScorer` for thin-file applicants (UPI, phone, e-commerce data)
  - CIBIL-scale scoring (300-900) with risk grades (A-E)
  - Probability calibration via Platt scaling
- **Neural Credit Scorer (PyTorch + Apple Metal GPU)**
  - Deep residual network with BatchNorm + SiLU activation
  - Focal loss for class imbalance (~8% default rate)
  - MPS (Metal Performance Shaders) backend for Apple Silicon
  - Early stopping, cosine annealing LR, gradient clipping
- **Ensemble Credit Scorer**
  - 6+ model ensemble: HistGBM, Random Forest, Extra Trees, Logistic Regression, LightGBM, XGBoost
  - Weighted voting (AUC-weighted), soft voting, and stacking
  - Model comparison utilities
- **Bureau Integration**
  - `BureauManager` simulating CIBIL, Equifax, CRIF responses
  - `ConsentTracker` for RBI consent compliance
  - `BureauCallOptimizer` for cost reduction
- **Income Verification**
  - Priority-based verification: ITR > GST > Salary > Bank Statements
  - ITR parser with tax estimation (new regime slabs)
  - GST turnover analyzer
  - Salary slip consistency checker
- **Policy Engine**
  - Business Rules Engine with eligibility knockout + approval criteria
  - Risk-based dynamic pricing
  - EMI calculation and FOIR checking
  - Batch evaluation support
- **Risk Assessment**
  - PD/LGD/EAD/Expected Loss (Basel III IRB)
  - Portfolio monitoring with NPL tracking
  - Default prediction (6/12/24 month horizons)
  - Early Warning System with multi-signal detection
  - Stress testing with 5 scenarios (base, mild, slowdown, rate hike, sector crisis)
- **RBI Compliance**
  - `NPA_Classifier` (SMA-0/1/2, Sub-Standard, Doubtful, Loss) with provisioning
  - `DivergenceTracker` for bureau vs internal decisions
  - `FairnessAuditor` (4/5ths rule for disparate impact)
- **Data Management**
  - `DataValidator`, `KYC_Validator`, `DuplicateDetection`
  - Pydantic models for applicant and loan application
  - Indian document validators (PAN, Aadhaar, GSTIN, IFSC, IFSC)
- **Synthetic Data Generation**
  - 69+ correlated features with realistic Indian distributions
  - Chunked generation for 200K+ samples
  - CAT/NonCAT segmentation, bureau history, alternative data
- **Training Pipeline**
  - `train_models.py` - comprehensive multi-model training script
  - Hardware detection (Apple Silicon, Metal GPU, CPU cores)
  - Automatic fallbacks (libomp missing -> sklearn)
- **Testing**
  - 53+ pytest tests covering all modules

### Technical Highlights
- Apple Silicon M-series optimized (Metal GPU via PyTorch MPS)
- Scales to 500K+ samples on 16GB RAM
- Cross-validation, feature importance, model persistence
- Indian credit scoring metrics (KS statistic, Gini coefficient)
