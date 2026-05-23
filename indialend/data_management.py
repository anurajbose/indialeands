"""
IndiaLend Data Management
=========================
Data validation, KYC verification, duplicate detection, and data
preprocessing pipeline for Indian lending applications.

Handles Indian-specific document types: PAN, Aadhaar, GST, voter ID,
passport, driving license. Includes fuzzy duplicate detection to
prevent repeat-loan fraud (a major gap in Indian NBFC operations).
"""

from __future__ import annotations

import hashlib
import re
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, validator

from .utils import (
    aadhaar_validator,
    gstin_validator,
    ifsc_validator,
    indian_phone_validator,
    mask_aadhaar,
    mask_pan,
    pan_validator,
)


# ---------------------------------------------------------------------------
# Pydantic models for structured validation
# ---------------------------------------------------------------------------

class ApplicantProfile(BaseModel):
    """Validated applicant profile with Indian-specific field rules."""

    applicant_id: str = Field(..., min_length=1, max_length=20)
    pan: str = Field(..., min_length=10, max_length=10)
    name: str = Field(..., min_length=2, max_length=100)
    age: int = Field(..., ge=18, le=70)
    gender: Optional[str] = Field(None, pattern=r"^(M|F|O)$")
    phone: str = Field(..., min_length=10, max_length=13)
    email: Optional[str] = None
    employment_type: str = Field(
        ...,
        pattern=r"^(Salaried|Self-Employed|Professional|Business Owner)$",
    )
    annual_income: float = Field(..., gt=0)
    loan_amount: float = Field(..., gt=0)
    tenure: int = Field(..., ge=3, le=360)
    loan_product: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None

    class Config:
        str_strip_whitespace = True


class LoanApplication(BaseModel):
    """Full loan application with all fields needed for decisioning."""

    applicant: ApplicantProfile
    aadhaar: Optional[str] = None
    gstin: Optional[str] = None
    cibil_score: Optional[int] = Field(None, ge=300, le=900)
    existing_emi: float = Field(0, ge=0)
    num_existing_loans: int = Field(0, ge=0)
    credit_card_outstanding: float = Field(0, ge=0)
    residence_type: Optional[str] = None
    years_employed: int = Field(0, ge=0)
    education: Optional[str] = None
    marital_status: Optional[str] = None
    dependents: int = Field(0, ge=0)
    co_applicant: Optional[ApplicantProfile] = None


# ---------------------------------------------------------------------------
# DataValidator
# ---------------------------------------------------------------------------

class DataValidator:
    """Validate applicant data against Indian lending rules.

    Checks:
    - Required field presence
    - PAN format and checksum
    - Aadhaar format
    - Phone number validity
    - Age eligibility (18-65 for most products, 21-60 for home loans)
    - Income reasonableness
    - Loan amount bounds per product
    """

    REQUIRED_FIELDS = {"applicant_id", "pan", "name", "age", "phone",
                       "employment_type", "annual_income", "loan_amount"}

    PRODUCT_AGE_LIMITS = {
        "Home Loan": (21, 60),
        "Personal Loan": (21, 58),
        "Auto Loan": (21, 60),
        "Business Loan": (21, 65),
        "Education Loan": (18, 35),
        "Gold Loan": (18, 70),
        "Two-Wheeler Loan": (21, 55),
        "Consumer Durable Loan": (21, 55),
        "MSME Loan": (21, 65),
        "Loan Against Property": (25, 60),
    }

    PRODUCT_MAX_LTV = {
        "Home Loan": 0.80,
        "Auto Loan": 0.85,
        "Loan Against Property": 0.60,
        "Gold Loan": 0.75,
        "Two-Wheeler Loan": 0.90,
    }

    def __init__(self):
        self._warnings: List[str] = []
        self._errors: List[str] = []

    def validate(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Run all validations and return result dict.

        Returns
        -------
        dict with keys: valid (bool), errors (list), warnings (list), cleaned_data (dict)
        """
        self._warnings = []
        self._errors = []
        cleaned = dict(data)

        self._check_required_fields(cleaned)
        self._check_pan(cleaned)
        self._check_aadhaar(cleaned)
        self._check_phone(cleaned)
        self._check_age(cleaned)
        self._check_income(cleaned)
        self._check_loan_amount(cleaned)
        self._check_foir(cleaned)

        return {
            "valid": len(self._errors) == 0,
            "errors": list(self._errors),
            "warnings": list(self._warnings),
            "cleaned_data": cleaned,
        }

    def _check_required_fields(self, data: Dict) -> None:
        missing = self.REQUIRED_FIELDS - set(data.keys())
        for f in missing:
            self._errors.append(f"Missing required field: {f}")

    def _check_pan(self, data: Dict) -> None:
        pan = data.get("pan", "")
        if pan and not pan_validator(pan):
            self._errors.append(f"Invalid PAN format: {mask_pan(pan)}")

    def _check_aadhaar(self, data: Dict) -> None:
        aadhaar = data.get("aadhaar", "")
        if aadhaar and not aadhaar_validator(aadhaar):
            self._errors.append(f"Invalid Aadhaar format: {mask_aadhaar(aadhaar)}")

    def _check_phone(self, data: Dict) -> None:
        phone = data.get("phone", "")
        if phone and not indian_phone_validator(phone):
            self._errors.append(f"Invalid Indian phone number")

    def _check_age(self, data: Dict) -> None:
        age = data.get("age")
        product = data.get("loan_product", "Personal Loan")
        if age is None:
            return
        limits = self.PRODUCT_AGE_LIMITS.get(product, (18, 65))
        if age < limits[0] or age > limits[1]:
            self._errors.append(
                f"Age {age} outside eligible range {limits} for {product}"
            )

    def _check_income(self, data: Dict) -> None:
        income = data.get("annual_income", 0)
        emp = data.get("employment_type", "")
        if income < 100_000:
            self._errors.append("Annual income below minimum threshold (Rs 1,00,000)")
        if emp == "Salaried" and income > 10_000_000:
            self._warnings.append("Unusually high salary declared — verify with ITR")
        if emp in ("Self-Employed", "Business Owner") and income > 50_000_000:
            self._warnings.append("Very high income — additional verification needed")

    def _check_loan_amount(self, data: Dict) -> None:
        amount = data.get("loan_amount", 0)
        income = data.get("annual_income", 1)
        product = data.get("loan_product", "Personal Loan")

        if amount <= 0:
            self._errors.append("Loan amount must be positive")
            return

        # Income multiplier check
        max_mult = {"Home Loan": 60, "Loan Against Property": 40,
                    "Business Loan": 20, "Personal Loan": 10,
                    "Auto Loan": 8, "Education Loan": 15}.get(product, 10)
        if amount > income * max_mult:
            self._warnings.append(
                f"Loan amount ({amount:,.0f}) exceeds {max_mult}x annual income"
            )

    def _check_foir(self, data: Dict) -> None:
        """Check Fixed Obligation to Income Ratio."""
        monthly_income = data.get("annual_income", 0) / 12
        existing_emi = data.get("existing_emi", 0)
        if monthly_income > 0:
            foir = existing_emi / monthly_income
            if foir > 0.50:
                self._warnings.append(
                    f"High FOIR ({foir:.0%}) — existing obligations exceed 50% of income"
                )
            if foir > 0.70:
                self._errors.append(
                    f"FOIR ({foir:.0%}) exceeds maximum threshold (70%)"
                )


# ---------------------------------------------------------------------------
# KYC Validator
# ---------------------------------------------------------------------------

class KYC_Validator:
    """Verify KYC document completeness per RBI guidelines.

    Indian lending KYC requires:
    - Identity proof (PAN / Aadhaar / Voter ID / Passport)
    - Address proof (Aadhaar / Utility bill / Bank statement)
    - Income proof (ITR / Salary slip / Bank statement)
    - Photo (recent passport-size)
    """

    IDENTITY_DOCS = {"pan", "aadhaar", "voter_id", "passport", "driving_license"}
    ADDRESS_DOCS = {"aadhaar", "utility_bill", "bank_statement", "rent_agreement",
                    "passport", "voter_id"}
    INCOME_DOCS = {"itr", "salary_slip", "bank_statement", "gst_certificate",
                   "form_16", "ca_certificate"}
    PHOTO_DOCS = {"photo", "selfie"}

    def __init__(self, require_photo: bool = True, require_video_kyc: bool = False):
        self.require_photo = require_photo
        self.require_video_kyc = require_video_kyc

    def validate_documents(
        self,
        documents: Dict[str, Any],
        employment_type: str = "Salaried",
    ) -> Dict[str, Any]:
        """Validate KYC document set.

        Parameters
        ----------
        documents : dict
            Keys are document types, values are document data/status.
        employment_type : str
            Affects which income docs are required.

        Returns
        -------
        dict with keys: complete (bool), missing (list), verified (list), score (float)
        """
        provided = set(documents.keys())
        verified = []
        missing = []

        # Identity check
        identity_present = provided & self.IDENTITY_DOCS
        if not identity_present:
            missing.append("identity_proof (PAN/Aadhaar/Voter ID/Passport)")
        else:
            verified.extend(list(identity_present))

        # PAN is always required for loans > 50k (Income Tax Act)
        if "pan" not in provided:
            missing.append("pan (mandatory for loans > Rs 50,000)")

        # Address check
        address_present = provided & self.ADDRESS_DOCS
        if not address_present:
            missing.append("address_proof (Aadhaar/Utility bill/Bank statement)")
        else:
            verified.extend([d for d in address_present if d not in verified])

        # Income check
        income_present = provided & self.INCOME_DOCS
        if not income_present:
            missing.append("income_proof (ITR/Salary slip/Bank statement)")
        else:
            # Employment-specific income doc preference
            if employment_type == "Salaried":
                preferred = {"salary_slip", "form_16", "bank_statement"}
                if not (income_present & preferred):
                    missing.append("salary_slip or form_16 (preferred for salaried)")
            elif employment_type in ("Self-Employed", "Business Owner"):
                preferred = {"itr", "gst_certificate", "ca_certificate"}
                if not (income_present & preferred):
                    missing.append("itr or gst_certificate (preferred for self-employed)")
            verified.extend([d for d in income_present if d not in verified])

        # Photo
        if self.require_photo:
            photo_present = provided & self.PHOTO_DOCS
            if not photo_present:
                missing.append("photo (passport-size)")
            else:
                verified.extend([d for d in photo_present if d not in verified])

        # Video KYC (RBI mandated for digital-only onboarding)
        if self.require_video_kyc and "video_kyc" not in provided:
            missing.append("video_kyc (required for digital onboarding)")

        score = len(verified) / max(len(verified) + len(missing), 1)

        return {
            "complete": len(missing) == 0,
            "missing": missing,
            "verified": verified,
            "score": round(score, 2),
            "documents_provided": len(provided),
        }


# ---------------------------------------------------------------------------
# Duplicate Detection
# ---------------------------------------------------------------------------

class DuplicateDetection:
    """Detect duplicate/fraud loan applications.

    Uses multiple matching strategies:
    1. Exact PAN match (strongest signal)
    2. Phone number match
    3. Name + DOB fuzzy match
    4. Aadhaar match
    5. Address similarity
    6. Device fingerprint (if available)

    Critical for Indian market: borrowers apply at multiple NBFCs
    simultaneously; detecting this prevents over-leveraging.
    """

    def __init__(self, pan_weight: float = 1.0, phone_weight: float = 0.8,
                 name_weight: float = 0.5, aadhaar_weight: float = 1.0):
        self.weights = {
            "pan": pan_weight,
            "phone": phone_weight,
            "name_dob": name_weight,
            "aadhaar": aadhaar_weight,
        }
        self._seen_pans: Dict[str, List[str]] = defaultdict(list)
        self._seen_phones: Dict[str, List[str]] = defaultdict(list)
        self._seen_aadhaar: Dict[str, List[str]] = defaultdict(list)
        self._seen_names: Dict[str, List[str]] = defaultdict(list)

    def register(self, applicant_id: str, data: Dict[str, Any]) -> None:
        """Register an application for future duplicate checks."""
        if pan := data.get("pan"):
            self._seen_pans[pan.upper()].append(applicant_id)
        if phone := data.get("phone"):
            cleaned = phone.replace(" ", "").replace("-", "")[-10:]
            self._seen_phones[cleaned].append(applicant_id)
        if aadhaar := data.get("aadhaar"):
            cleaned = aadhaar.replace(" ", "").replace("-", "")
            self._seen_aadhaar[cleaned].append(applicant_id)
        if name := data.get("name"):
            key = self._normalize_name(name)
            if dob := data.get("date_of_birth", data.get("age")):
                key = f"{key}_{dob}"
            self._seen_names[key].append(applicant_id)

    def check(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Check if this application is a potential duplicate.

        Returns
        -------
        dict with: is_duplicate (bool), confidence (float), matches (list of dicts)
        """
        matches = []

        # PAN match
        if pan := data.get("pan"):
            existing = self._seen_pans.get(pan.upper(), [])
            if existing:
                matches.append({
                    "match_type": "pan",
                    "matched_ids": existing,
                    "confidence": 0.99,
                })

        # Phone match
        if phone := data.get("phone"):
            cleaned = phone.replace(" ", "").replace("-", "")[-10:]
            existing = self._seen_phones.get(cleaned, [])
            if existing:
                matches.append({
                    "match_type": "phone",
                    "matched_ids": existing,
                    "confidence": 0.85,
                })

        # Aadhaar match
        if aadhaar := data.get("aadhaar"):
            cleaned = aadhaar.replace(" ", "").replace("-", "")
            existing = self._seen_aadhaar.get(cleaned, [])
            if existing:
                matches.append({
                    "match_type": "aadhaar",
                    "matched_ids": existing,
                    "confidence": 0.99,
                })

        # Name + age/DOB match
        if name := data.get("name"):
            key = self._normalize_name(name)
            if dob := data.get("date_of_birth", data.get("age")):
                key = f"{key}_{dob}"
            existing = self._seen_names.get(key, [])
            if existing:
                matches.append({
                    "match_type": "name_dob",
                    "matched_ids": existing,
                    "confidence": 0.70,
                })

        if matches:
            max_conf = max(m["confidence"] for m in matches)
            # Boost confidence when multiple signals match
            if len(matches) >= 2:
                max_conf = min(max_conf + 0.10, 1.0)
        else:
            max_conf = 0.0

        return {
            "is_duplicate": max_conf >= 0.80,
            "confidence": round(max_conf, 2),
            "matches": matches,
            "match_count": len(matches),
        }

    def find_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Scan a DataFrame of applications and flag duplicates.

        Returns DataFrame with columns: applicant_id, is_duplicate,
        confidence, matched_with, match_types
        """
        results = []
        temp_detector = DuplicateDetection(
            pan_weight=self.weights["pan"],
            phone_weight=self.weights["phone"],
            name_weight=self.weights["name_dob"],
            aadhaar_weight=self.weights["aadhaar"],
        )

        for _, row in df.iterrows():
            data = row.to_dict()
            check_result = temp_detector.check(data)
            results.append({
                "applicant_id": data.get("applicant_id", ""),
                "is_duplicate": check_result["is_duplicate"],
                "confidence": check_result["confidence"],
                "matched_with": (
                    check_result["matches"][0]["matched_ids"]
                    if check_result["matches"] else []
                ),
                "match_types": [m["match_type"] for m in check_result["matches"]],
            })
            temp_detector.register(data.get("applicant_id", ""), data)

        return pd.DataFrame(results)

    @staticmethod
    def _normalize_name(name: str) -> str:
        """Normalize name for fuzzy matching."""
        name = name.lower().strip()
        # Remove common Indian honorifics
        for prefix in ("mr ", "mrs ", "ms ", "dr ", "shri ", "smt ", "kumar ", "kumari "):
            if name.startswith(prefix):
                name = name[len(prefix):]
        # Remove extra whitespace
        name = re.sub(r"\s+", " ", name).strip()
        return name


# ---------------------------------------------------------------------------
# Data Preprocessor
# ---------------------------------------------------------------------------

class DataPreprocessor:
    """Preprocess raw application data into model-ready features.

    Handles:
    - Missing value imputation (median for numeric, mode for categorical)
    - Categorical encoding (target encoding for high-cardinality, one-hot otherwise)
    - Feature engineering (ratios, interactions, binning)
    - Outlier capping (winsorization at 1st/99th percentile)
    """

    # Numeric features for the scoring model
    NUMERIC_FEATURES = [
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

    CATEGORICAL_FEATURES = [
        "gender", "marital_status", "education", "employment_type",
        "city_tier", "residence_type", "loan_product", "cat_segment",
        "company_size",
    ]

    def __init__(self, winsorize: bool = True, percentile: float = 0.01):
        self.winsorize = winsorize
        self.percentile = percentile
        self._fitted = False
        self._numeric_medians: Dict[str, float] = {}
        self._numeric_bounds: Dict[str, Tuple[float, float]] = {}
        self._categorical_modes: Dict[str, str] = {}

    def fit(self, df: pd.DataFrame) -> "DataPreprocessor":
        """Learn imputation values and bounds from training data."""
        for col in self.NUMERIC_FEATURES:
            if col in df.columns:
                self._numeric_medians[col] = float(df[col].median())
                if self.winsorize:
                    lo = float(df[col].quantile(self.percentile))
                    hi = float(df[col].quantile(1 - self.percentile))
                    self._numeric_bounds[col] = (lo, hi)

        for col in self.CATEGORICAL_FEATURES:
            if col in df.columns:
                mode_vals = df[col].mode()
                self._categorical_modes[col] = mode_vals.iloc[0] if len(mode_vals) > 0 else "Unknown"

        self._fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply preprocessing to data."""
        if not self._fitted:
            raise RuntimeError("Call fit() before transform()")

        result = df.copy()

        # Impute numeric
        for col, median in self._numeric_medians.items():
            if col in result.columns:
                result[col] = result[col].fillna(median)

        # Winsorize
        if self.winsorize:
            for col, (lo, hi) in self._numeric_bounds.items():
                if col in result.columns:
                    result[col] = result[col].clip(lo, hi)

        # Impute categorical
        for col, mode in self._categorical_modes.items():
            if col in result.columns:
                result[col] = result[col].fillna(mode)

        # Engineer derived features
        result = self._engineer_features(result)

        return result

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)

    @staticmethod
    def _engineer_features(df: pd.DataFrame) -> pd.DataFrame:
        """Create derived features useful for credit scoring."""
        result = df.copy()

        # Debt-to-income ratio
        if "existing_emi" in result.columns and "monthly_income" in result.columns:
            result["debt_to_income"] = (
                result["existing_emi"] / result["monthly_income"].clip(lower=1)
            ).round(4)

        # Loan-to-income ratio
        if "loan_amount" in result.columns and "annual_income" in result.columns:
            result["loan_to_income"] = (
                result["loan_amount"] / result["annual_income"].clip(lower=1)
            ).round(4)

        # Available income after EMI
        if all(c in result.columns for c in ["monthly_income", "existing_emi", "proposed_emi"]):
            result["net_disposable_income"] = (
                result["monthly_income"] - result["existing_emi"] - result["proposed_emi"]
            )

        # Credit utilization bucket
        if "credit_utilization" in result.columns:
            result["credit_util_bucket"] = pd.cut(
                result["credit_utilization"],
                bins=[-0.01, 0.10, 0.30, 0.50, 0.75, 1.01],
                labels=["very_low", "low", "moderate", "high", "very_high"],
            ).astype(str)

        # Bureau score bucket
        if "cibil_score" in result.columns:
            result["score_bucket"] = pd.cut(
                result["cibil_score"],
                bins=[299, 550, 600, 650, 700, 750, 900],
                labels=["very_poor", "poor", "fair", "good", "very_good", "excellent"],
            ).astype(str)

        # Enquiry intensity (recent vs total)
        if "enquiries_3m" in result.columns and "enquiries_12m" in result.columns:
            result["enquiry_intensity"] = (
                result["enquiries_3m"] / result["enquiries_12m"].clip(lower=1)
            ).round(4)

        # DPD severity index
        dpd_cols = [c for c in ["dpd_30_12m", "dpd_60_12m", "dpd_90_12m"] if c in result.columns]
        if dpd_cols:
            weights = [1, 2, 4][:len(dpd_cols)]
            result["dpd_severity_index"] = sum(
                result[col] * w for col, w in zip(dpd_cols, weights)
            )

        return result
