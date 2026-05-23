"""
IndiaLend Risk Assessment
=========================
Comprehensive risk quantification and portfolio monitoring:
- PD (Probability of Default) estimation
- LGD (Loss Given Default) by product type
- EAD (Exposure at Default) calculation
- Expected Loss computation
- Portfolio-level risk metrics
- Default prediction (6/12/24 month horizons)
- Early Warning System for pre-NPA detection
- Stress testing under macro scenarios

Key Indian market gaps:
1. NBFCs detect NPAs reactively (after 90 DPD) → we predict 30-60 days early
2. Portfolio risk is siloed → we aggregate across products
3. No stress testing → we model recession/rate-hike scenarios
4. RBI expects IRB-approach risk metrics → we compute PD/LGD/EAD natively
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy.stats import norm


# ---------------------------------------------------------------------------
# Risk Assessment (Individual Loan)
# ---------------------------------------------------------------------------

class RiskAssessment:
    """Calculate risk metrics for individual loans.

    Implements simplified IRB (Internal Ratings Based) approach
    per RBI Basel III guidelines for NBFCs.
    """

    # LGD assumptions by product (Indian market benchmarks)
    LGD_BY_PRODUCT = {
        "Personal Loan": 0.60,
        "Auto Loan": 0.35,
        "Home Loan": 0.20,
        "Business Loan": 0.55,
        "Education Loan": 0.50,
        "Gold Loan": 0.15,
        "Loan Against Property": 0.25,
        "Two-Wheeler Loan": 0.45,
        "Consumer Durable Loan": 0.70,
        "MSME Loan": 0.55,
    }

    # PD mapping from risk grade (calibrated to Indian default rates)
    PD_BY_GRADE = {
        "A": 0.005,   # 0.5%
        "B": 0.015,   # 1.5%
        "C": 0.040,   # 4.0%
        "D": 0.080,   # 8.0%
        "E": 0.200,   # 20.0%
    }

    def calculate_risk_metrics(
        self,
        loan_data: Dict[str, Any],
        bureau_data: Optional[Dict] = None,
        model_pd: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Calculate PD, LGD, EAD, Expected Loss, and risk rating.

        Parameters
        ----------
        loan_data : dict
            Loan details: loan_amount, tenure, employment_type, loan_product
        bureau_data : dict, optional
            Bureau response data for PD estimation.
        model_pd : float, optional
            Pre-computed PD from credit scoring model.
        """
        loan_amount = loan_data.get("loan_amount", 0)
        tenure = loan_data.get("tenure_months", loan_data.get("tenure", 36))
        product = loan_data.get("loan_product", "Personal Loan")
        employment = loan_data.get("employment_type", "Salaried")

        # --- PD (Probability of Default) ---
        if model_pd is not None:
            pd_value = model_pd
        elif bureau_data:
            pd_value = self._estimate_pd(bureau_data)
        else:
            # Fallback: use risk grade from data
            grade = loan_data.get("risk_grade", "C")
            pd_value = self.PD_BY_GRADE.get(grade, 0.04)

        # Employment adjustment
        emp_adj = {
            "Salaried": 0.8,
            "Professional": 0.9,
            "Self-Employed": 1.2,
            "Business Owner": 1.3,
        }.get(employment, 1.0)
        pd_value *= emp_adj
        pd_value = min(pd_value, 1.0)

        # --- LGD (Loss Given Default) ---
        lgd = self.LGD_BY_PRODUCT.get(product, 0.45)

        # Adjust LGD for collateral
        ltv = loan_data.get("ltv_ratio", 0.8)
        if ltv < 0.5:
            lgd *= 0.7  # Well-collateralised
        elif ltv > 0.9:
            lgd *= 1.2  # High LTV

        lgd = min(lgd, 1.0)

        # --- EAD (Exposure at Default) ---
        # Assumes default mid-way through loan
        months_to_default = min(tenure // 2, 24)
        if tenure > 0:
            amortization_factor = 1 - (months_to_default / tenure) * 0.5
        else:
            amortization_factor = 1.0
        ead = loan_amount * amortization_factor

        # --- Expected Loss ---
        expected_loss = pd_value * lgd * ead

        # --- Risk Rating ---
        if pd_value < 0.01:
            risk_rating = "A"
        elif pd_value < 0.03:
            risk_rating = "B"
        elif pd_value < 0.06:
            risk_rating = "C"
        elif pd_value < 0.12:
            risk_rating = "D"
        else:
            risk_rating = "E"

        # --- Risk-Weighted Assets (RWA) for capital adequacy ---
        # Basel III correlation formula (simplified)
        correlation = 0.12 * (1 - np.exp(-50 * pd_value)) / (1 - np.exp(-50))
        correlation += 0.24 * (1 - (1 - np.exp(-50 * pd_value)) / (1 - np.exp(-50)))

        if pd_value > 0 and pd_value < 1:
            z = norm.ppf(pd_value)
            z_conf = norm.ppf(0.999)
            conditional_pd = norm.cdf(
                (z + np.sqrt(correlation) * z_conf) / np.sqrt(1 - correlation)
            )
            rwa = lgd * conditional_pd * ead * 1.06  # Maturity adjustment
        else:
            rwa = lgd * pd_value * ead

        # --- Capital requirement ---
        capital_requirement = rwa * 0.15  # 15% CRAR for NBFCs

        return {
            "probability_of_default": round(pd_value, 6),
            "loss_given_default": round(lgd, 4),
            "exposure_at_default": round(ead, 2),
            "expected_loss": round(expected_loss, 2),
            "unexpected_loss": round(rwa - expected_loss, 2) if rwa > expected_loss else 0,
            "risk_rating": risk_rating,
            "risk_weighted_assets": round(rwa, 2),
            "capital_requirement": round(capital_requirement, 2),
            "loan_amount": loan_amount,
            "product": product,
        }

    def _estimate_pd(self, bureau_data: Dict) -> float:
        """Estimate PD from bureau data."""
        score = bureau_data.get("cibil_score", 650)
        dpd_30 = bureau_data.get("dpd_30_12m", 0)
        dpd_90 = bureau_data.get("dpd_90_12m", 0)
        writeoffs = bureau_data.get("written_off_accounts", 0)
        enquiries = bureau_data.get("enquiries_3m", 0)

        # Score-based PD
        if score >= 750:
            base_pd = 0.005
        elif score >= 700:
            base_pd = 0.015
        elif score >= 650:
            base_pd = 0.035
        elif score >= 600:
            base_pd = 0.070
        elif score >= 550:
            base_pd = 0.120
        else:
            base_pd = 0.250

        # Adjustments
        base_pd += dpd_30 * 0.02
        base_pd += dpd_90 * 0.08
        base_pd += writeoffs * 0.15
        base_pd += max(0, enquiries - 3) * 0.01

        return min(base_pd, 1.0)


# ---------------------------------------------------------------------------
# Portfolio Monitor
# ---------------------------------------------------------------------------

class PortfolioMonitor:
    """Monitor portfolio-level risk metrics.

    Tracks:
    - NPL ratio (Non-Performing Loans)
    - Portfolio expected loss
    - Concentration risk (by product, geography, risk grade)
    - Vintage analysis
    - Collection efficiency
    """

    def __init__(self):
        self._loans: List[Dict] = []

    def add_loan(self, loan_data: Dict[str, Any]) -> None:
        """Add a loan to the portfolio."""
        loan = dict(loan_data)
        loan.setdefault("status", "current")
        loan.setdefault("dpd", 0)
        loan.setdefault("disbursement_date", datetime.now().isoformat())
        self._loans.append(loan)

    def add_loans(self, loans: List[Dict]) -> None:
        """Add multiple loans."""
        for loan in loans:
            self.add_loan(loan)

    @property
    def portfolio_size(self) -> int:
        return len(self._loans)

    def calculate_portfolio_metrics(self) -> Dict[str, Any]:
        """Calculate comprehensive portfolio metrics."""
        if not self._loans:
            return {"error": "No loans in portfolio"}

        df = pd.DataFrame(self._loans)

        total_exposure = df.get("loan_amount", pd.Series([0])).sum()
        total_loans = len(df)

        # NPL classification (RBI norms)
        # SMA-0: 1-30 DPD, SMA-1: 31-60 DPD, SMA-2: 61-90 DPD, NPA: 90+ DPD
        dpd = df.get("dpd", pd.Series([0] * total_loans))
        npa_mask = dpd > 90
        sma2_mask = (dpd > 60) & (dpd <= 90)
        sma1_mask = (dpd > 30) & (dpd <= 60)
        sma0_mask = (dpd > 0) & (dpd <= 30)

        npa_count = npa_mask.sum()
        npl_amount = df.loc[npa_mask, "loan_amount"].sum() if "loan_amount" in df.columns else 0

        # Expected loss
        el_values = df.get("expected_loss", pd.Series([0] * total_loans))
        total_el = el_values.sum()

        # Concentration by product
        product_conc = {}
        if "loan_product" in df.columns:
            product_conc = (
                df.groupby("loan_product")["loan_amount"]
                .agg(["sum", "count"])
                .rename(columns={"sum": "exposure", "count": "count"})
                .to_dict("index")
            )

        # Concentration by risk grade
        grade_conc = {}
        if "risk_rating" in df.columns:
            grade_conc = (
                df.groupby("risk_rating")["loan_amount"]
                .agg(["sum", "count"])
                .rename(columns={"sum": "exposure", "count": "count"})
                .to_dict("index")
            )

        # Average risk metrics
        avg_pd = df.get("probability_of_default", pd.Series([0])).mean()

        return {
            "total_loans": total_loans,
            "total_exposure": round(total_exposure, 2),
            "npl_count": int(npa_count),
            "npl_amount": round(npl_amount, 2),
            "npl_ratio": round(npa_count / max(total_loans, 1), 4),
            "npl_amount_ratio": round(npl_amount / max(total_exposure, 1), 4),
            "sma0_count": int(sma0_mask.sum()),
            "sma1_count": int(sma1_mask.sum()),
            "sma2_count": int(sma2_mask.sum()),
            "total_expected_loss": round(total_el, 2),
            "avg_probability_of_default": round(avg_pd, 6),
            "product_concentration": product_conc,
            "risk_grade_concentration": grade_conc,
            "portfolio_health": (
                "healthy" if npa_count / max(total_loans, 1) < 0.02
                else "watch" if npa_count / max(total_loans, 1) < 0.05
                else "stressed" if npa_count / max(total_loans, 1) < 0.10
                else "critical"
            ),
        }

    def vintage_analysis(self) -> pd.DataFrame:
        """Vintage analysis: default rates by disbursement cohort."""
        if not self._loans:
            return pd.DataFrame()

        df = pd.DataFrame(self._loans)
        if "disbursement_date" not in df.columns:
            return pd.DataFrame()

        df["disbursement_date"] = pd.to_datetime(df["disbursement_date"])
        df["vintage_month"] = df["disbursement_date"].dt.to_period("M")
        df["is_default"] = df.get("dpd", 0) > 90

        vintage = df.groupby("vintage_month").agg(
            total_loans=("loan_amount", "count"),
            total_amount=("loan_amount", "sum"),
            defaults=("is_default", "sum"),
            default_amount=("loan_amount", lambda x: x[df.loc[x.index, "is_default"]].sum()
                            if "is_default" in df.columns else 0),
        ).reset_index()

        vintage["default_rate"] = vintage["defaults"] / vintage["total_loans"].clip(lower=1)

        return vintage


# ---------------------------------------------------------------------------
# Default Predictor
# ---------------------------------------------------------------------------

class DefaultPredictor:
    """Predict loan defaults at 6, 12, and 24 month horizons.

    Uses a combination of:
    - Payment behavior trends (DPD trajectory)
    - Bureau score changes
    - Income stress signals
    - Macro factors
    """

    def predict_defaults(
        self,
        active_loans: pd.DataFrame,
        transaction_history: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """Predict default probability for active loans.

        Parameters
        ----------
        active_loans : DataFrame
            Active loan portfolio with risk metrics.
        transaction_history : DataFrame, optional
            Monthly payment history for trend analysis.

        Returns
        -------
        DataFrame with columns: account_id, pd_6m, pd_12m, pd_24m,
        risk_flag, recommendation
        """
        results = []

        for _, loan in active_loans.iterrows():
            account_id = loan.get("account_id", loan.get("applicant_id", ""))
            current_dpd = loan.get("dpd", 0)
            score = loan.get("cibil_score", 650)
            pd_base = loan.get("probability_of_default", 0.05)

            # DPD trend factor
            dpd_factor = 1.0
            if current_dpd > 0:
                dpd_factor = 1.0 + (current_dpd / 30) * 0.5

            # Score-based adjustment
            score_factor = max(0.3, (900 - score) / 300)

            # 6-month PD
            pd_6m = min(pd_base * dpd_factor * 1.5, 1.0)

            # 12-month PD (higher horizon = higher uncertainty)
            pd_12m = min(pd_base * dpd_factor * score_factor * 2.0, 1.0)

            # 24-month PD
            pd_24m = min(pd_base * dpd_factor * score_factor * 3.0, 1.0)

            # Risk flag
            if pd_6m > 0.20:
                risk_flag = "HIGH"
                recommendation = "Immediate collection action; consider restructuring"
            elif pd_6m > 0.10:
                risk_flag = "MEDIUM"
                recommendation = "Proactive outreach; payment reminder"
            elif pd_12m > 0.15:
                risk_flag = "WATCH"
                recommendation = "Monitor closely; quarterly review"
            else:
                risk_flag = "LOW"
                recommendation = "Standard monitoring"

            results.append({
                "account_id": account_id,
                "current_dpd": current_dpd,
                "pd_6m": round(pd_6m, 4),
                "pd_12m": round(pd_12m, 4),
                "pd_24m": round(pd_24m, 4),
                "risk_flag": risk_flag,
                "recommendation": recommendation,
            })

        return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# Early Warning System
# ---------------------------------------------------------------------------

class EarlyWarningSystem:
    """Detect loan stress 30-60 days before NPA classification.

    Monitors multiple signals:
    1. Payment deterioration (partial payments, increasing DPD)
    2. Bureau score decline
    3. Increased credit enquiries (seeking more debt)
    4. Bank balance deterioration
    5. Employment change signals
    """

    WARNING_THRESHOLDS = {
        "dpd_increase": 15,           # DPD jumped by 15+ days
        "partial_payment_ratio": 0.80, # Paying less than 80% of EMI
        "consecutive_late": 2,         # 2+ consecutive late payments
        "score_decline": 50,           # Score dropped 50+ points
        "enquiry_spike": 3,            # 3+ enquiries in 3 months
        "utilization_spike": 0.85,     # Credit utilization > 85%
    }

    def check_loan_health(
        self,
        loan: Dict[str, Any],
        transaction_history: Optional[List[Dict]] = None,
        previous_bureau: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """Check health of a single loan.

        Returns warning level, signals detected, and recommendation.
        """
        warnings = []
        risk_score = 0  # 0-100, higher = worse

        current_dpd = loan.get("dpd", 0)
        emi = loan.get("emi", loan.get("proposed_emi", 0))

        # --- DPD analysis ---
        if current_dpd > 0:
            risk_score += min(current_dpd, 30)
            if current_dpd > 30:
                warnings.append(f"DPD at {current_dpd} days — approaching SMA-1")
            elif current_dpd > 15:
                warnings.append(f"DPD at {current_dpd} days — early stress signal")

        # --- Payment history analysis ---
        if transaction_history:
            recent = transaction_history[-6:]  # Last 6 months
            late_count = sum(1 for t in recent if t.get("dpd", 0) > 0)
            partial_count = sum(
                1 for t in recent
                if t.get("payment_made", 0) < t.get("payment_due", 0) * 0.8
            )

            if late_count >= self.WARNING_THRESHOLDS["consecutive_late"]:
                warnings.append(f"{late_count} late payments in last 6 months")
                risk_score += late_count * 8

            if partial_count >= 2:
                warnings.append(f"{partial_count} partial payments in last 6 months")
                risk_score += partial_count * 10

            # Payment trend (decreasing payments)
            payments = [t.get("payment_made", 0) for t in recent if t.get("payment_made", 0) > 0]
            if len(payments) >= 3:
                trend = np.polyfit(range(len(payments)), payments, 1)[0]
                if trend < -emi * 0.05:  # Declining by >5% of EMI per month
                    warnings.append("Payment amounts declining over time")
                    risk_score += 15

        # --- Bureau score change ---
        if previous_bureau:
            score_change = (
                loan.get("cibil_score", 0) - previous_bureau.get("cibil_score", 0)
            )
            if score_change < -self.WARNING_THRESHOLDS["score_decline"]:
                warnings.append(f"Bureau score declined by {abs(score_change)} points")
                risk_score += 20

        # --- Enquiry spike ---
        enquiries = loan.get("enquiries_3m", 0)
        if enquiries >= self.WARNING_THRESHOLDS["enquiry_spike"]:
            warnings.append(f"{enquiries} bureau enquiries in 3 months — seeking more credit")
            risk_score += 15

        # --- Credit utilization ---
        utilization = loan.get("credit_utilization", 0)
        if utilization > self.WARNING_THRESHOLDS["utilization_spike"]:
            warnings.append(f"Credit utilization at {utilization:.0%} — maxing out credit")
            risk_score += 15

        # --- Determine risk level ---
        risk_score = min(risk_score, 100)
        if risk_score >= 60:
            risk_level = "HIGH"
            recommendation = (
                "URGENT: Contact borrower immediately. Consider restructuring "
                "or enhanced collection. Likely to become NPA within 30 days."
            )
        elif risk_score >= 35:
            risk_level = "MEDIUM"
            recommendation = (
                "Proactive outreach needed. Schedule call with borrower. "
                "Review payment plan. Consider sending early warning letter."
            )
        elif risk_score >= 15:
            risk_level = "WATCH"
            recommendation = (
                "Monitor closely. Increase review frequency to weekly. "
                "Check for any adverse bureau changes."
            )
        else:
            risk_level = "HEALTHY"
            recommendation = "Standard monitoring. No action needed."

        return {
            "risk_level": risk_level,
            "risk_score": risk_score,
            "warnings": warnings,
            "warning_count": len(warnings),
            "recommendation": recommendation,
            "current_dpd": current_dpd,
            "timestamp": datetime.now().isoformat(),
        }

    def scan_portfolio(
        self,
        loans: List[Dict],
        transaction_histories: Optional[Dict[str, List[Dict]]] = None,
    ) -> pd.DataFrame:
        """Scan entire portfolio for early warning signals.

        Returns DataFrame sorted by risk_score (highest first).
        """
        results = []

        for loan in loans:
            acct_id = loan.get("account_id", loan.get("applicant_id", ""))
            txn_history = (
                transaction_histories.get(acct_id, [])
                if transaction_histories else None
            )
            health = self.check_loan_health(loan, txn_history)
            health["account_id"] = acct_id
            results.append(health)

        df = pd.DataFrame(results)
        if len(df) > 0:
            df = df.sort_values("risk_score", ascending=False)
        return df


# ---------------------------------------------------------------------------
# Stress Testing
# ---------------------------------------------------------------------------

class StressTest:
    """Portfolio stress testing under adverse scenarios.

    Scenarios calibrated to Indian market:
    - Base case: Normal conditions
    - Mild stress: Moderate slowdown (GDP growth drops 2%)
    - Economic slowdown: Sharp contraction
    - Rate hike: RBI rate increase (impacts EMIs, defaults)
    - Sector crisis: Specific industry downturn
    """

    SCENARIOS = {
        "base": {
            "name": "Base Case",
            "pd_multiplier": 1.0,
            "lgd_multiplier": 1.0,
            "recovery_rate": 0.40,
            "description": "Normal economic conditions",
        },
        "mild_stress": {
            "name": "Mild Stress",
            "pd_multiplier": 1.5,
            "lgd_multiplier": 1.1,
            "recovery_rate": 0.35,
            "description": "GDP growth drops by 2%, moderate credit tightening",
        },
        "economic_slowdown": {
            "name": "Economic Slowdown",
            "pd_multiplier": 2.5,
            "lgd_multiplier": 1.3,
            "recovery_rate": 0.25,
            "description": "Sharp GDP contraction, rising unemployment",
        },
        "rate_hike": {
            "name": "Rate Hike",
            "pd_multiplier": 1.8,
            "lgd_multiplier": 1.15,
            "recovery_rate": 0.30,
            "description": "RBI raises repo rate by 200bps, EMI burden increases",
        },
        "sector_crisis": {
            "name": "Sector Crisis",
            "pd_multiplier": 3.0,
            "lgd_multiplier": 1.5,
            "recovery_rate": 0.15,
            "description": "Major industry downturn (real estate, NBFC liquidity crisis)",
        },
    }

    def run_stress_test(
        self,
        portfolio: Union[pd.DataFrame, List[Dict]],
        scenario: str = "economic_slowdown",
        custom_params: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """Run stress test on portfolio.

        Parameters
        ----------
        portfolio : DataFrame or list of dict
            Loan portfolio with risk metrics.
        scenario : str
            Scenario name from SCENARIOS.
        custom_params : dict, optional
            Override scenario parameters.
        """
        if isinstance(portfolio, list):
            portfolio = pd.DataFrame(portfolio)

        params = dict(self.SCENARIOS.get(scenario, self.SCENARIOS["base"]))
        if custom_params:
            params.update(custom_params)

        total_loans = len(portfolio)
        total_exposure = portfolio.get("loan_amount", pd.Series([0])).sum()

        # Apply stress multipliers
        base_pd = portfolio.get("probability_of_default", pd.Series([0.05] * total_loans))
        stressed_pd = (base_pd * params["pd_multiplier"]).clip(upper=1.0)

        base_lgd = portfolio.get("loss_given_default", pd.Series([0.45] * total_loans))
        stressed_lgd = (base_lgd * params["lgd_multiplier"]).clip(upper=1.0)

        ead = portfolio.get("exposure_at_default", portfolio.get("loan_amount", pd.Series([0])))

        # Stressed expected loss
        stressed_el = stressed_pd * stressed_lgd * ead
        base_el = base_pd * base_lgd * ead

        # Estimate new NPAs
        current_npa = (portfolio.get("dpd", pd.Series([0])) > 90).sum()
        estimated_new_npa = int(
            (stressed_pd > 0.15).sum() - current_npa
        )
        estimated_new_npa = max(0, estimated_new_npa)

        # Capital impact
        additional_provisions = float(stressed_el.sum() - base_el.sum())

        return {
            "scenario": params["name"],
            "description": params["description"],
            "pd_multiplier": params["pd_multiplier"],
            "lgd_multiplier": params["lgd_multiplier"],
            "total_loans": total_loans,
            "total_exposure": round(float(total_exposure), 2),
            "base_expected_loss": round(float(base_el.sum()), 2),
            "stressed_expected_loss": round(float(stressed_el.sum()), 2),
            "incremental_loss": round(float(additional_provisions), 2),
            "loss_increase_pct": round(
                float(additional_provisions / max(base_el.sum(), 1) * 100), 1
            ),
            "current_npa_count": int(current_npa),
            "estimated_npl_increase": estimated_new_npa,
            "estimated_npl_total": int(current_npa + estimated_new_npa),
            "estimated_npl_ratio": round(
                (current_npa + estimated_new_npa) / max(total_loans, 1), 4
            ),
            "additional_provisions_needed": round(float(additional_provisions), 2),
            "avg_stressed_pd": round(float(stressed_pd.mean()), 4),
            "max_stressed_pd": round(float(stressed_pd.max()), 4),
            "recovery_rate": params["recovery_rate"],
        }

    def run_all_scenarios(
        self,
        portfolio: Union[pd.DataFrame, List[Dict]],
    ) -> pd.DataFrame:
        """Run all stress test scenarios and return comparison."""
        results = []
        for scenario_name in self.SCENARIOS:
            result = self.run_stress_test(portfolio, scenario=scenario_name)
            results.append(result)
        return pd.DataFrame(results)
