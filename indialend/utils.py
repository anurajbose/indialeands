"""
IndiaLend Utilities
==================
Synthetic data generation (realistic Indian credit profiles),
PAN/Aadhaar/phone validators, and common helpers.

Designed to produce data that mirrors real Indian NBFC portfolios:
- Salaried vs Self-Employed distributions
- Metro/Tier-1/Tier-2/Rural geography mix
- Realistic income, loan-amount, and tenure ranges
- Correlated credit features (income<->score, employment<->risk)
"""

from __future__ import annotations

import hashlib
import re
import string
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constants – Indian-market specific
# ---------------------------------------------------------------------------

INDIAN_STATES = [
    "Maharashtra", "Karnataka", "Tamil Nadu", "Delhi", "Gujarat",
    "Telangana", "Uttar Pradesh", "West Bengal", "Rajasthan",
    "Madhya Pradesh", "Kerala", "Punjab", "Haryana", "Bihar",
    "Andhra Pradesh", "Odisha", "Jharkhand", "Assam", "Chhattisgarh",
    "Goa",
]

CITY_TIERS: Dict[str, List[str]] = {
    "metro": ["Mumbai", "Delhi", "Bangalore", "Hyderabad", "Chennai", "Kolkata", "Pune"],
    "tier1": ["Ahmedabad", "Jaipur", "Lucknow", "Chandigarh", "Indore", "Nagpur",
              "Coimbatore", "Kochi", "Visakhapatnam", "Bhopal"],
    "tier2": ["Nashik", "Vadodara", "Mysore", "Jodhpur", "Raipur", "Dehradun",
              "Aurangabad", "Ranchi", "Gwalior", "Jabalpur"],
    "rural": ["Latur", "Sangli", "Satara", "Wardha", "Nandurbar", "Jalna",
              "Buldhana", "Osmanabad", "Hingoli", "Washim"],
}

EMPLOYMENT_TYPES = ["Salaried", "Self-Employed", "Professional", "Business Owner"]

LOAN_PRODUCTS = [
    "Personal Loan", "Auto Loan", "Home Loan", "Business Loan",
    "Education Loan", "Gold Loan", "Loan Against Property",
    "Two-Wheeler Loan", "Consumer Durable Loan", "MSME Loan",
]

EDUCATION_LEVELS = [
    "Below 10th", "10th Pass", "12th Pass", "Graduate",
    "Post Graduate", "Professional Degree", "Doctorate",
]

MARITAL_STATUS = ["Single", "Married", "Divorced", "Widowed"]

RESIDENCE_TYPES = ["Owned", "Rented", "Company Provided", "Parental", "PG/Hostel"]

# Risk-grade cutoffs (CIBIL-like 300-900 scale)
RISK_GRADE_MAP = {
    "A": (750, 900),
    "B": (700, 749),
    "C": (650, 699),
    "D": (600, 649),
    "E": (300, 599),
}


# ---------------------------------------------------------------------------
# Validators
# ---------------------------------------------------------------------------

def pan_validator(pan: str) -> bool:
    """Validate Indian PAN (Permanent Account Number).

    Format: ABCDE1234F  (5 letters, 4 digits, 1 letter)
    4th char encodes entity type: C=Company, P=Person, H=HUF, etc.
    """
    if not isinstance(pan, str):
        return False
    return bool(re.match(r"^[A-Z]{3}[ABCFGHLJPT][A-Z]\d{4}[A-Z]$", pan.upper()))


def aadhaar_validator(aadhaar: str) -> bool:
    """Validate Indian Aadhaar number (12 digits, not starting with 0/1)."""
    if not isinstance(aadhaar, str):
        return False
    cleaned = aadhaar.replace(" ", "").replace("-", "")
    return bool(re.match(r"^[2-9]\d{11}$", cleaned))


def indian_phone_validator(phone: str) -> bool:
    """Validate Indian mobile number (10 digits starting with 6-9)."""
    if not isinstance(phone, str):
        return False
    cleaned = phone.replace(" ", "").replace("-", "").replace("+91", "").lstrip("0")
    return bool(re.match(r"^[6-9]\d{9}$", cleaned))


def gstin_validator(gstin: str) -> bool:
    """Validate Indian GSTIN (15 chars: 2-digit state + PAN + entity + Z + check)."""
    if not isinstance(gstin, str):
        return False
    return bool(re.match(
        r"^\d{2}[A-Z]{3}[ABCFGHLJPT][A-Z]\d{4}[A-Z]\d[Z][A-Z\d]$",
        gstin.upper(),
    ))


def ifsc_validator(ifsc: str) -> bool:
    """Validate Indian IFSC code (4 letters + 0 + 6 alphanumeric)."""
    if not isinstance(ifsc, str):
        return False
    return bool(re.match(r"^[A-Z]{4}0[A-Z0-9]{6}$", ifsc.upper()))


def risk_grade_from_score(score: float) -> str:
    """Map a CIBIL-scale score (300-900) to risk grade A-E."""
    for grade, (lo, hi) in RISK_GRADE_MAP.items():
        if lo <= score <= hi:
            return grade
    return "E"


# ---------------------------------------------------------------------------
# Synthetic PAN / Aadhaar generators
# ---------------------------------------------------------------------------

def _random_pan(rng: np.random.Generator, entity: str = "P") -> str:
    letters = string.ascii_uppercase
    first3 = "".join(rng.choice(list(letters)) for _ in range(3))
    fifth = rng.choice(list(letters))
    digits = "".join(str(rng.integers(0, 10)) for _ in range(4))
    last = rng.choice(list(letters))
    return f"{first3}{entity}{fifth}{digits}{last}"


def _random_aadhaar(rng: np.random.Generator) -> str:
    first = str(rng.integers(2, 10))
    rest = "".join(str(rng.integers(0, 10)) for _ in range(11))
    return first + rest


def _random_phone(rng: np.random.Generator) -> str:
    first = str(rng.integers(6, 10))
    rest = "".join(str(rng.integers(0, 10)) for _ in range(9))
    return first + rest


# ---------------------------------------------------------------------------
# Synthetic Data Generator
# ---------------------------------------------------------------------------

def set_seed(seed: int = 42) -> np.random.Generator:
    """Return a seeded NumPy random generator for reproducibility."""
    return np.random.default_rng(seed)


def generate_synthetic_data(
    n_samples: int = 10_000,
    seed: int = 42,
    default_rate: float = 0.08,
    include_alternative: bool = True,
    include_bureau_history: bool = True,
    chunk_size: int = 50_000,
) -> pd.DataFrame:
    """Generate realistic Indian credit application data.

    Produces correlated features that reflect real-world patterns:
    - Higher income → higher CIBIL score → lower default
    - Self-employed → higher income variance → slightly higher risk
    - Metro applicants → higher income but also higher loan amounts
    - Age/experience correlates with stability

    Parameters
    ----------
    n_samples : int
        Number of applications to generate. Supports 200K+ efficiently.
    seed : int
        Random seed for reproducibility.
    default_rate : float
        Target overall default rate (will be ~approximate).
    include_alternative : bool
        Include alternative data features (UPI, phone usage).
    include_bureau_history : bool
        Include detailed bureau trade-line history features.
    chunk_size : int
        For large datasets (>50K), generate in chunks to manage memory.

    Returns
    -------
    pd.DataFrame
        Synthetic dataset with ~60-80 features per applicant.
    """
    # For very large datasets, generate in chunks and concatenate
    if n_samples > chunk_size:
        chunks = []
        generated = 0
        chunk_id = 0
        while generated < n_samples:
            this_chunk = min(chunk_size, n_samples - generated)
            chunk_df = generate_synthetic_data(
                n_samples=this_chunk,
                seed=seed + chunk_id,
                default_rate=default_rate,
                include_alternative=include_alternative,
                include_bureau_history=include_bureau_history,
                chunk_size=chunk_size,
            )
            # Fix applicant IDs for uniqueness
            chunk_df["applicant_id"] = [
                f"APP{str(generated + i + 1).zfill(7)}" for i in range(len(chunk_df))
            ]
            chunks.append(chunk_df)
            generated += this_chunk
            chunk_id += 1
        return pd.concat(chunks, ignore_index=True)

    rng = np.random.default_rng(seed)

    # --- Demographics ---
    ages = rng.integers(21, 65, size=n_samples)
    genders = rng.choice(["M", "F"], size=n_samples, p=[0.68, 0.32])
    marital = rng.choice(MARITAL_STATUS, size=n_samples, p=[0.25, 0.60, 0.10, 0.05])
    education = rng.choice(
        EDUCATION_LEVELS, size=n_samples,
        p=[0.03, 0.07, 0.15, 0.40, 0.20, 0.12, 0.03],
    )
    dependents = rng.integers(0, 5, size=n_samples)
    residence = rng.choice(RESIDENCE_TYPES, size=n_samples, p=[0.35, 0.35, 0.05, 0.20, 0.05])

    # Education numeric for correlation
    edu_map = {e: i for i, e in enumerate(EDUCATION_LEVELS)}
    edu_numeric = np.array([edu_map[e] for e in education])

    # --- Geography ---
    tier_probs = [0.40, 0.25, 0.20, 0.15]
    tiers = rng.choice(["metro", "tier1", "tier2", "rural"], size=n_samples, p=tier_probs)
    cities = np.array([rng.choice(CITY_TIERS[t]) for t in tiers])
    states = rng.choice(INDIAN_STATES, size=n_samples)

    # Geography income multiplier
    geo_mult = np.where(tiers == "metro", 1.4,
               np.where(tiers == "tier1", 1.15,
               np.where(tiers == "tier2", 0.95, 0.75)))

    # --- Employment ---
    employment = rng.choice(
        EMPLOYMENT_TYPES, size=n_samples,
        p=[0.50, 0.20, 0.15, 0.15],
    )
    years_employed = np.clip(
        rng.normal(ages - 22, 3, size=n_samples).astype(int), 0, 40
    )
    years_current_job = np.clip(
        rng.exponential(3.5, size=n_samples).astype(int), 0, years_employed
    )
    company_size = rng.choice(
        ["Startup", "SME", "Mid-size", "Large Corp", "MNC", "Government"],
        size=n_samples,
        p=[0.10, 0.20, 0.20, 0.25, 0.15, 0.10],
    )

    # --- Income (correlated with education, employment, geography, age) ---
    base_income = np.exp(
        rng.normal(12.2, 0.7, size=n_samples)  # log-normal, median ~200k
        + edu_numeric * 0.12
        + (ages - 30) * 0.015
    ) * geo_mult

    # Self-employed: higher variance, slightly higher mean
    se_mask = np.isin(employment, ["Self-Employed", "Business Owner"])
    base_income[se_mask] *= rng.uniform(0.6, 2.0, size=se_mask.sum())

    annual_income = np.clip(base_income, 100_000, 50_000_000).astype(int)
    monthly_income = annual_income // 12

    # CAT/NonCAT segmentation (RBI: CAT >= 30 lakh)
    cat_segment = np.where(annual_income >= 3_000_000, "CAT", "NonCAT")

    # --- Existing obligations ---
    existing_emi = (monthly_income * rng.uniform(0, 0.45, size=n_samples)).astype(int)
    num_existing_loans = rng.integers(0, 6, size=n_samples)
    credit_card_outstanding = (
        monthly_income * rng.exponential(0.8, size=n_samples)
    ).astype(int)
    credit_card_limit = np.clip(
        credit_card_outstanding * rng.uniform(1.5, 5.0, size=n_samples),
        50_000, 10_000_000,
    ).astype(int)
    credit_utilization = np.clip(
        credit_card_outstanding / np.maximum(credit_card_limit, 1), 0, 1
    )

    # --- Loan request ---
    loan_product = rng.choice(LOAN_PRODUCTS, size=n_samples, p=[
        0.25, 0.15, 0.15, 0.10, 0.05, 0.05, 0.08, 0.07, 0.05, 0.05,
    ])
    # Loan amount correlates with income and product type
    product_mult = {
        "Personal Loan": 3, "Auto Loan": 5, "Home Loan": 20,
        "Business Loan": 8, "Education Loan": 4, "Gold Loan": 2,
        "Loan Against Property": 15, "Two-Wheeler Loan": 1,
        "Consumer Durable Loan": 0.5, "MSME Loan": 6,
    }
    loan_amount = np.array([
        int(np.clip(
            annual_income[i] * product_mult.get(loan_product[i], 3)
            * rng.uniform(0.3, 1.2),
            50_000, 100_000_000,
        ))
        for i in range(n_samples)
    ])
    tenure_months = np.array([
        rng.choice([12, 24, 36, 48, 60, 84, 120, 180, 240])
        if loan_product[i] in ("Home Loan", "Loan Against Property")
        else rng.choice([6, 12, 18, 24, 36, 48, 60])
        for i in range(n_samples)
    ])

    # LTV (Loan-to-Value)
    asset_value = (loan_amount * rng.uniform(1.0, 2.5, size=n_samples)).astype(int)
    ltv_ratio = loan_amount / np.maximum(asset_value, 1)

    # FOIR (Fixed Obligation to Income Ratio)
    proposed_emi = loan_amount / np.maximum(tenure_months, 1) * 1.1  # rough EMI
    foir = (existing_emi + proposed_emi) / np.maximum(monthly_income, 1)

    # --- Bureau / Credit History ---
    # CIBIL score: correlated with income, employment stability, credit utilization
    score_noise = rng.normal(0, 50, size=n_samples)
    raw_score = (
        550
        + np.log(annual_income + 1) * 15
        + years_employed * 2
        - credit_utilization * 100
        - num_existing_loans * 8
        + edu_numeric * 10
        + score_noise
    )
    cibil_score = np.clip(raw_score, 300, 900).astype(int)

    # DPD (Days Past Due) history – correlated with score
    dpd_30_12m = np.clip(
        rng.poisson(np.where(cibil_score < 600, 3, np.where(cibil_score < 700, 1, 0.2)),
                     size=n_samples),
        0, 12,
    )
    dpd_60_12m = np.clip(dpd_30_12m - rng.integers(0, 3, size=n_samples), 0, 6)
    dpd_90_12m = np.clip(dpd_60_12m - rng.integers(0, 2, size=n_samples), 0, 3)

    # Enquiries
    enquiries_3m = rng.integers(0, 6, size=n_samples)
    enquiries_6m = enquiries_3m + rng.integers(0, 4, size=n_samples)
    enquiries_12m = enquiries_6m + rng.integers(0, 6, size=n_samples)

    # Account age
    oldest_account_months = np.clip(
        (years_employed * 12 * rng.uniform(0.3, 0.9, size=n_samples)).astype(int),
        0, 480,
    )
    newest_account_months = np.clip(
        rng.integers(1, np.maximum(oldest_account_months, 2)),
        0, oldest_account_months,
    )
    avg_account_age_months = ((oldest_account_months + newest_account_months) // 2)

    # Write-offs and settlements
    written_off_accounts = np.where(
        cibil_score < 550,
        rng.integers(0, 3, size=n_samples),
        np.where(cibil_score < 650, rng.integers(0, 2, size=n_samples), 0),
    )
    settled_accounts = np.where(
        cibil_score < 600,
        rng.integers(0, 2, size=n_samples),
        0,
    )

    # --- KYC / Identity ---
    pans = np.array([_random_pan(rng) for _ in range(n_samples)])
    aadhaar_numbers = np.array([_random_aadhaar(rng) for _ in range(n_samples)])
    phone_numbers = np.array([_random_phone(rng) for _ in range(n_samples)])
    has_email = rng.choice([True, False], size=n_samples, p=[0.75, 0.25])

    # --- Applicant IDs ---
    applicant_ids = np.array([f"APP{str(i+1).zfill(7)}" for i in range(n_samples)])

    # --- Application date ---
    base_date = datetime(2024, 1, 1)
    app_dates = np.array([
        (base_date + timedelta(days=int(rng.integers(0, 365)))).strftime("%Y-%m-%d")
        for _ in range(n_samples)
    ])

    # --- Default target (correlated with risk factors) ---
    default_logit = (
        -3.0
        - (cibil_score - 650) * 0.008
        + credit_utilization * 1.5
        + dpd_30_12m * 0.3
        + dpd_90_12m * 0.8
        + enquiries_3m * 0.15
        - years_employed * 0.05
        + foir * 1.2
        + written_off_accounts * 1.0
        + np.where(se_mask, 0.3, 0.0)
        - edu_numeric * 0.1
        + rng.normal(0, 0.5, size=n_samples)
    )
    default_prob = 1 / (1 + np.exp(-default_logit))
    # Calibrate to target default rate
    threshold = np.quantile(default_prob, 1 - default_rate)
    default_flag = (default_prob >= threshold).astype(int)

    # --- Build DataFrame ---
    df = pd.DataFrame({
        # Identity
        "applicant_id": applicant_ids,
        "pan": pans,
        "aadhaar": aadhaar_numbers,
        "phone": phone_numbers,
        "has_email": has_email,
        "application_date": pd.to_datetime(app_dates),

        # Demographics
        "age": ages,
        "gender": genders,
        "marital_status": marital,
        "education": education,
        "education_numeric": edu_numeric,
        "dependents": dependents,
        "residence_type": residence,

        # Geography
        "city": cities,
        "city_tier": tiers,
        "state": states,

        # Employment
        "employment_type": employment,
        "years_employed": years_employed,
        "years_current_job": years_current_job,
        "company_size": company_size,

        # Income
        "annual_income": annual_income,
        "monthly_income": monthly_income,
        "cat_segment": cat_segment,

        # Existing obligations
        "existing_emi": existing_emi,
        "num_existing_loans": num_existing_loans,
        "credit_card_outstanding": credit_card_outstanding,
        "credit_card_limit": credit_card_limit,
        "credit_utilization": np.round(credit_utilization, 4),

        # Loan request
        "loan_product": loan_product,
        "loan_amount": loan_amount,
        "tenure_months": tenure_months,
        "asset_value": asset_value,
        "ltv_ratio": np.round(ltv_ratio, 4),
        "proposed_emi": proposed_emi.astype(int),
        "foir": np.round(foir, 4),

        # Bureau / Credit history
        "cibil_score": cibil_score,
        "dpd_30_12m": dpd_30_12m,
        "dpd_60_12m": dpd_60_12m,
        "dpd_90_12m": dpd_90_12m,
        "enquiries_3m": enquiries_3m,
        "enquiries_6m": enquiries_6m,
        "enquiries_12m": enquiries_12m,
        "oldest_account_months": oldest_account_months,
        "newest_account_months": newest_account_months,
        "avg_account_age_months": avg_account_age_months,
        "written_off_accounts": written_off_accounts,
        "settled_accounts": settled_accounts,

        # Target
        "default_flag": default_flag,
        "default_probability": np.round(default_prob, 6),
    })

    # --- Alternative data (UPI, phone, e-commerce) ---
    if include_alternative:
        # UPI transaction patterns
        upi_txn_count_3m = np.clip(
            rng.poisson(
                np.where(annual_income > 1_000_000, 90, np.where(annual_income > 500_000, 50, 25)),
                size=n_samples,
            ), 0, 500,
        )
        upi_txn_value_3m = (
            upi_txn_count_3m * rng.uniform(200, 5000, size=n_samples)
        ).astype(int)
        upi_unique_merchants = np.clip(
            (upi_txn_count_3m * rng.uniform(0.1, 0.5, size=n_samples)).astype(int), 1, 200
        )
        upi_inflow_ratio = np.round(rng.uniform(0.3, 0.9, size=n_samples), 4)

        # Phone usage
        phone_age_months = rng.integers(6, 120, size=n_samples)
        avg_recharge_amount = np.clip(
            rng.normal(400, 150, size=n_samples), 100, 2000
        ).astype(int)
        data_usage_gb = np.round(rng.uniform(1, 50, size=n_samples), 1)
        app_install_count = rng.integers(10, 100, size=n_samples)

        # E-commerce
        ecom_orders_6m = rng.integers(0, 30, size=n_samples)
        ecom_spend_6m = (ecom_orders_6m * rng.uniform(500, 5000, size=n_samples)).astype(int)
        ecom_return_rate = np.round(
            np.clip(rng.beta(2, 10, size=n_samples), 0, 0.5), 4
        )

        alt_cols = {
            "upi_txn_count_3m": upi_txn_count_3m,
            "upi_txn_value_3m": upi_txn_value_3m,
            "upi_unique_merchants": upi_unique_merchants,
            "upi_inflow_ratio": upi_inflow_ratio,
            "phone_age_months": phone_age_months,
            "avg_recharge_amount": avg_recharge_amount,
            "data_usage_gb": data_usage_gb,
            "app_install_count": app_install_count,
            "ecom_orders_6m": ecom_orders_6m,
            "ecom_spend_6m": ecom_spend_6m,
            "ecom_return_rate": ecom_return_rate,
        }
        for col, vals in alt_cols.items():
            df[col] = vals

    # --- Detailed bureau trade-line history ---
    if include_bureau_history:
        total_accounts = num_existing_loans + rng.integers(0, 5, size=n_samples)
        active_accounts = np.clip(num_existing_loans, 0, total_accounts)
        closed_accounts = total_accounts - active_accounts
        secured_accounts = np.clip(
            (total_accounts * rng.uniform(0.2, 0.6, size=n_samples)).astype(int),
            0, total_accounts,
        )
        unsecured_accounts = total_accounts - secured_accounts
        total_outstanding = (
            loan_amount * rng.uniform(0.3, 1.5, size=n_samples) + credit_card_outstanding
        ).astype(int)
        total_sanctioned = (total_outstanding * rng.uniform(1.0, 2.0, size=n_samples)).astype(int)
        max_dpd_ever = np.clip(
            dpd_90_12m * rng.integers(30, 120, size=n_samples), 0, 365
        )
        suit_filed = (max_dpd_ever > 180).astype(int)

        bureau_cols = {
            "total_accounts": total_accounts,
            "active_accounts": active_accounts,
            "closed_accounts": closed_accounts,
            "secured_accounts": secured_accounts,
            "unsecured_accounts": unsecured_accounts,
            "total_outstanding": total_outstanding,
            "total_sanctioned": total_sanctioned,
            "max_dpd_ever": max_dpd_ever,
            "suit_filed": suit_filed,
        }
        for col, vals in bureau_cols.items():
            df[col] = vals

    return df


def load_sample_data(
    n_samples: int = 1000,
    seed: int = 42,
    test_size: float = 0.2,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Generate sample data and return train/test split.

    Returns
    -------
    X_train, X_test, y_train, y_test
    """
    from sklearn.model_selection import train_test_split

    df = generate_synthetic_data(n_samples=n_samples, seed=seed)
    feature_cols = [
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
    X = df[feature_cols]
    y = df["default_flag"]

    return train_test_split(X, y, test_size=test_size, random_state=seed, stratify=y)


def generate_transaction_history(
    n_accounts: int = 100,
    months: int = 12,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate synthetic monthly payment/transaction history for portfolio monitoring.

    Returns DataFrame with columns:
    account_id, month, payment_due, payment_made, dpd, balance, status
    """
    rng = np.random.default_rng(seed)
    records = []

    for acct in range(n_accounts):
        acct_id = f"LOAN{str(acct + 1).zfill(6)}"
        emi = int(rng.uniform(5000, 80000))
        balance = int(emi * rng.uniform(12, 60))
        # Risk profile for this account
        risk_level = rng.choice(["low", "medium", "high"], p=[0.6, 0.25, 0.15])
        late_prob = {"low": 0.05, "medium": 0.20, "high": 0.45}[risk_level]

        for m in range(months):
            month_date = (datetime(2024, 1, 1) + timedelta(days=30 * m)).strftime("%Y-%m")
            is_late = rng.random() < late_prob
            if is_late:
                dpd = int(rng.choice([5, 15, 30, 60, 90], p=[0.3, 0.25, 0.25, 0.12, 0.08]))
                payment = int(emi * rng.uniform(0.0, 0.95))
            else:
                dpd = 0
                payment = emi

            balance = max(0, balance - payment)
            status = (
                "current" if dpd == 0
                else "sma-0" if dpd <= 30
                else "sma-1" if dpd <= 60
                else "sma-2" if dpd <= 90
                else "npa"
            )
            records.append({
                "account_id": acct_id,
                "month": month_date,
                "payment_due": emi,
                "payment_made": payment,
                "dpd": dpd,
                "balance": balance,
                "status": status,
                "risk_level": risk_level,
            })

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Hashing / Masking helpers (for PII protection)
# ---------------------------------------------------------------------------

def mask_pan(pan: str) -> str:
    """Mask PAN for display: ABCDE****F"""
    if len(pan) != 10:
        return pan
    return pan[:5] + "****" + pan[-1]


def mask_aadhaar(aadhaar: str) -> str:
    """Mask Aadhaar: XXXX-XXXX-1234"""
    cleaned = aadhaar.replace(" ", "").replace("-", "")
    if len(cleaned) != 12:
        return aadhaar
    return f"XXXX-XXXX-{cleaned[-4:]}"


def hash_pii(value: str, salt: str = "indialend") -> str:
    """SHA-256 hash for PII de-identification."""
    return hashlib.sha256(f"{salt}:{value}".encode()).hexdigest()[:16]
