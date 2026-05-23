"""
IndiaLend Income Verification
==============================
Automated income verification from multiple Indian document sources:
- ITR (Income Tax Returns) - highest confidence
- GST (Goods & Services Tax) - for business income
- Salary slips / Form 16 - for salaried employees
- Bank statements - deposit pattern analysis

Key Indian market gaps this fills:
1. Manual income verification takes 1-2 days → automated in seconds
2. No cross-validation across sources → we triangulate multiple sources
3. Income inflation by applicants → statistical fraud detection
4. Self-employed income estimation → GST + bank statement analysis
"""

from __future__ import annotations

import hashlib
import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Income Verification Result
# ---------------------------------------------------------------------------

class IncomeResult:
    """Standardised income verification result."""

    def __init__(
        self,
        verified_income: float,
        confidence_score: float,
        sources: List[str],
        method: str,
        details: Optional[Dict] = None,
    ):
        self.verified_income = verified_income
        self.confidence_score = min(max(confidence_score, 0), 1.0)
        self.sources = sources
        self.method = method
        self.details = details or {}
        self.timestamp = datetime.now()
        self.monthly_income = verified_income / 12

    def to_dict(self) -> Dict[str, Any]:
        return {
            "verified_income": self.verified_income,
            "monthly_income": round(self.monthly_income, 2),
            "confidence_score": round(self.confidence_score, 4),
            "sources": self.sources,
            "method": self.method,
            "details": self.details,
            "timestamp": self.timestamp.isoformat(),
        }


# ---------------------------------------------------------------------------
# ITR Parser
# ---------------------------------------------------------------------------

class ITR_Parser:
    """Parse Indian Income Tax Return data for income verification.

    Supports ITR-1 (salaried), ITR-3 (business), ITR-4 (presumptive).
    Extracts: gross income, deductions, taxable income, tax paid.

    In production, this would parse actual ITR XML/JSON from the
    Income Tax portal. Here we process structured data.
    """

    ITR_CONFIDENCE = {
        "ITR-1": 0.95,  # Salaried, most reliable
        "ITR-2": 0.92,  # Capital gains
        "ITR-3": 0.88,  # Business income
        "ITR-4": 0.85,  # Presumptive income
    }

    def parse(self, itr_data: Union[Dict, List[Dict]]) -> Dict[str, Any]:
        """Parse ITR data and extract verified income.

        Parameters
        ----------
        itr_data : dict or list of dict
            Single year or multiple years of ITR data.
            Expected keys: gross_income, deductions, taxable_income,
            tax_paid, assessment_year, itr_form
        """
        if isinstance(itr_data, dict):
            itr_data = [itr_data]

        if not itr_data:
            return {"verified_income": 0, "confidence": 0, "years": 0}

        yearly_incomes = []
        forms = []

        for itr in itr_data:
            gross = itr.get("gross_income", 0)
            deductions = itr.get("deductions", 0)
            taxable = itr.get("taxable_income", gross - deductions)
            tax_paid = itr.get("tax_paid", 0)
            form = itr.get("itr_form", "ITR-1")
            ay = itr.get("assessment_year", "")

            # Cross-check: tax paid should be reasonable for declared income
            expected_tax = self._estimate_tax(taxable)
            tax_ratio = tax_paid / max(expected_tax, 1) if expected_tax > 0 else 1

            yearly_incomes.append({
                "assessment_year": ay,
                "gross_income": gross,
                "deductions": deductions,
                "taxable_income": taxable,
                "tax_paid": tax_paid,
                "expected_tax": expected_tax,
                "tax_consistency": min(tax_ratio, 2.0),
                "form": form,
            })
            forms.append(form)

        # Average income across years (weighted: recent years more)
        weights = list(range(1, len(yearly_incomes) + 1))
        incomes = [y["gross_income"] for y in yearly_incomes]
        weighted_income = np.average(incomes, weights=weights)

        # Confidence based on form type and consistency
        form_confidence = np.mean([
            self.ITR_CONFIDENCE.get(f, 0.80) for f in forms
        ])

        # Check income trend (growing is good, volatile is risky)
        if len(incomes) >= 2:
            growth_rate = (incomes[-1] - incomes[0]) / max(incomes[0], 1)
            cv = np.std(incomes) / max(np.mean(incomes), 1)  # Coefficient of variation
            trend_factor = 1.0 if growth_rate >= 0 else 0.9
            stability_factor = max(0.7, 1.0 - cv * 0.3)
        else:
            trend_factor = 0.95
            stability_factor = 0.90

        confidence = form_confidence * trend_factor * stability_factor

        return {
            "verified_income": round(weighted_income, 2),
            "confidence": round(confidence, 4),
            "years": len(yearly_incomes),
            "yearly_details": yearly_incomes,
            "income_trend": "growing" if len(incomes) >= 2 and incomes[-1] > incomes[0] else "stable",
            "forms_filed": forms,
        }

    @staticmethod
    def _estimate_tax(taxable_income: float) -> float:
        """Estimate tax under new regime (FY 2024-25 slabs)."""
        slabs = [
            (300_000, 0.00),
            (600_000, 0.05),
            (900_000, 0.10),
            (1_200_000, 0.15),
            (1_500_000, 0.20),
            (float("inf"), 0.30),
        ]
        tax = 0
        prev_limit = 0
        for limit, rate in slabs:
            taxable_in_slab = min(taxable_income, limit) - prev_limit
            if taxable_in_slab <= 0:
                break
            tax += taxable_in_slab * rate
            prev_limit = limit
        # Add cess (4%)
        tax *= 1.04
        return round(tax, 2)


# ---------------------------------------------------------------------------
# GST Parser
# ---------------------------------------------------------------------------

class GST_Parser:
    """Parse GST data for business income verification.

    GSTN filings reveal business turnover. For self-employed and
    business applicants, GST data is often more reliable than ITR.
    """

    def parse(self, gst_data: Union[Dict, List[Dict]]) -> Dict[str, Any]:
        """Parse GST data.

        Parameters
        ----------
        gst_data : dict or list of dict
            Monthly/quarterly GST filing data.
            Expected keys: period, turnover, tax_paid, filing_type (GSTR-1/3B)
        """
        if isinstance(gst_data, dict):
            gst_data = [gst_data]

        if not gst_data:
            return {"verified_income": 0, "confidence": 0, "periods": 0}

        turnovers = []
        for filing in gst_data:
            turnover = filing.get("turnover", 0)
            turnovers.append(turnover)

        # Estimate annual turnover
        periods = len(gst_data)
        if periods >= 12:
            annual_turnover = sum(turnovers[-12:])
        elif periods >= 4:
            # Quarterly data
            annual_turnover = sum(turnovers) * (12 / periods)
        else:
            annual_turnover = sum(turnovers) * (12 / max(periods, 1))

        # Estimate income as % of turnover (varies by industry)
        # Conservative: 15-30% margin for Indian SMEs
        estimated_income = annual_turnover * 0.20

        # Confidence
        if periods >= 12:
            confidence = 0.90
        elif periods >= 6:
            confidence = 0.85
        elif periods >= 3:
            confidence = 0.75
        else:
            confidence = 0.60

        # Check filing regularity
        filing_regularity = periods / 12  # Assume 12 months expected

        return {
            "verified_income": round(estimated_income, 2),
            "annual_turnover": round(annual_turnover, 2),
            "confidence": round(confidence * min(filing_regularity, 1.0), 4),
            "periods": periods,
            "avg_monthly_turnover": round(annual_turnover / 12, 2),
            "margin_assumed": 0.20,
        }


# ---------------------------------------------------------------------------
# Salary Slip Parser
# ---------------------------------------------------------------------------

class SalarySlipParser:
    """Parse salary slips for income verification.

    Handles common Indian salary structures:
    - Basic + HRA + DA + Special Allowance
    - Deductions: PF, ESI, Professional Tax, TDS
    """

    def parse(self, salary_data: Union[Dict, List[Dict]]) -> Dict[str, Any]:
        """Parse salary slip data.

        Parameters
        ----------
        salary_data : dict or list of dict
            Monthly salary slip data.
            Expected keys: month, gross_salary, basic, hra, da,
            special_allowance, pf_deduction, tds, net_salary
        """
        if isinstance(salary_data, dict):
            salary_data = [salary_data]

        if not salary_data:
            return {"verified_income": 0, "confidence": 0, "months": 0}

        gross_salaries = []
        net_salaries = []

        for slip in salary_data:
            gross = slip.get("gross_salary", 0)
            net = slip.get("net_salary", gross)

            # If only components are provided, calculate gross
            if gross == 0:
                gross = (
                    slip.get("basic", 0)
                    + slip.get("hra", 0)
                    + slip.get("da", 0)
                    + slip.get("special_allowance", 0)
                    + slip.get("other_allowances", 0)
                )

            gross_salaries.append(gross)
            net_salaries.append(net)

        avg_monthly_gross = np.mean(gross_salaries)
        annual_income = avg_monthly_gross * 12

        # Confidence based on number of slips and consistency
        months = len(salary_data)
        if months >= 6:
            confidence = 0.90
        elif months >= 3:
            confidence = 0.85
        else:
            confidence = 0.75

        # Check salary consistency (coefficient of variation)
        if len(gross_salaries) >= 2:
            cv = np.std(gross_salaries) / max(np.mean(gross_salaries), 1)
            if cv > 0.15:
                confidence *= 0.85  # Inconsistent salary
                # Could indicate variable pay, bonus months, or fraud

        # Verify PF/TDS deductions are reasonable
        if salary_data[0].get("pf_deduction"):
            pf_rate = salary_data[0]["pf_deduction"] / max(salary_data[0].get("basic", 1), 1)
            if 0.10 <= pf_rate <= 0.14:
                confidence = min(confidence + 0.03, 1.0)  # PF matches expected rate

        return {
            "verified_income": round(annual_income, 2),
            "monthly_gross": round(avg_monthly_gross, 2),
            "monthly_net": round(np.mean(net_salaries), 2),
            "confidence": round(confidence, 4),
            "months": months,
            "salary_consistency": "consistent" if cv <= 0.10 else "variable" if 'cv' in dir() else "insufficient_data",
        }


# ---------------------------------------------------------------------------
# Bank Statement Analyzer
# ---------------------------------------------------------------------------

class BankStatementAnalyzer:
    """Analyze bank statements for income estimation.

    Last resort for income verification. Analyzes:
    - Regular deposit patterns (salary credits)
    - Business credit patterns (for self-employed)
    - Average Monthly Balance (AMB)
    - Cash flow analysis
    - Bounce/return patterns (negative signal)
    """

    def analyze(
        self,
        transactions: Union[pd.DataFrame, List[Dict]],
        account_type: str = "savings",
    ) -> Dict[str, Any]:
        """Analyze bank statement transactions.

        Parameters
        ----------
        transactions : DataFrame or list of dict
            Transaction data with columns: date, amount, type (credit/debit),
            description, balance
        account_type : str
            "savings" or "current"
        """
        if isinstance(transactions, list):
            transactions = pd.DataFrame(transactions)

        if transactions.empty:
            return {"verified_income": 0, "confidence": 0, "months": 0}

        # Ensure date column
        if "date" in transactions.columns:
            transactions["date"] = pd.to_datetime(transactions["date"])
        elif "transaction_date" in transactions.columns:
            transactions["date"] = pd.to_datetime(transactions["transaction_date"])

        # Separate credits and debits
        credits = transactions[transactions["type"].str.lower() == "credit"]
        debits = transactions[transactions["type"].str.lower() == "debit"]

        # Identify salary credits (regular, similar amounts)
        salary_credits = self._identify_salary_credits(credits)

        # Total credits analysis
        total_credits = credits["amount"].sum()
        total_debits = debits["amount"].sum()
        months = max(1, (transactions["date"].max() - transactions["date"].min()).days / 30)

        avg_monthly_credits = total_credits / months
        avg_monthly_debits = total_debits / months

        # Income estimation
        if salary_credits is not None and len(salary_credits) > 0:
            estimated_monthly = salary_credits["amount"].mean()
            income_method = "salary_credit_identification"
            confidence = 0.80
        else:
            # Fallback: use top credit percentile as income proxy
            if len(credits) > 0:
                p75_credit = credits["amount"].quantile(0.75)
                estimated_monthly = avg_monthly_credits * 0.6  # Conservative estimate
                income_method = "credit_pattern_analysis"
                confidence = 0.65
            else:
                estimated_monthly = 0
                income_method = "insufficient_data"
                confidence = 0.0

        annual_income = estimated_monthly * 12

        # Bounce analysis
        bounces = transactions[
            transactions["description"].str.lower().str.contains(
                "bounce|return|dishono", na=False
            )
        ] if "description" in transactions.columns else pd.DataFrame()

        # Average Monthly Balance
        if "balance" in transactions.columns:
            amb = transactions.groupby(transactions["date"].dt.to_period("M"))["balance"].mean().mean()
        else:
            amb = 0

        return {
            "verified_income": round(annual_income, 2),
            "monthly_income_estimate": round(estimated_monthly, 2),
            "confidence": round(confidence, 4),
            "method": income_method,
            "months_analyzed": round(months, 1),
            "avg_monthly_credits": round(avg_monthly_credits, 2),
            "avg_monthly_debits": round(avg_monthly_debits, 2),
            "avg_monthly_balance": round(amb, 2),
            "bounce_count": len(bounces),
            "credit_to_debit_ratio": round(
                total_credits / max(total_debits, 1), 4
            ),
            "total_transactions": len(transactions),
        }

    @staticmethod
    def _identify_salary_credits(credits: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Identify regular salary credits from transaction list.

        Salary credits are:
        - Similar amounts (within 10% variation)
        - Regular interval (monthly, ~30 days)
        - From employer (NEFT/RTGS, specific descriptions)
        """
        if credits.empty or len(credits) < 2:
            return None

        # Sort by date
        credits = credits.sort_values("date")

        # Look for regular credits with similar amounts
        # Group by approximate amount (within 10% bands)
        amounts = credits["amount"].values
        median_amount = np.median(amounts)

        # Filter credits within 15% of median
        mask = (amounts >= median_amount * 0.85) & (amounts <= median_amount * 1.15)
        regular_credits = credits[mask]

        if len(regular_credits) < 2:
            return None

        # Check regularity (intervals should be ~28-33 days)
        dates = regular_credits["date"].sort_values()
        intervals = dates.diff().dt.days.dropna()

        if len(intervals) == 0:
            return None

        avg_interval = intervals.mean()
        if 25 <= avg_interval <= 35:  # Monthly
            return regular_credits

        return None


# ---------------------------------------------------------------------------
# Master Income Verifier
# ---------------------------------------------------------------------------

class IncomeVerifier:
    """Master income verification engine.

    Priority-based verification:
    1. ITR (Tax filing) — 95% confidence
    2. GST (Business turnover) — 90% confidence
    3. Salary slips — 85% confidence
    4. Bank statements — 70% confidence

    Cross-validates across sources and flags inconsistencies.
    """

    SOURCE_PRIORITY = ["itr", "gst", "salary_slips", "bank_statements"]
    SOURCE_CONFIDENCE = {
        "itr": 0.95,
        "gst": 0.90,
        "salary_slips": 0.85,
        "bank_statements": 0.70,
    }

    def __init__(self):
        self.itr_parser = ITR_Parser()
        self.gst_parser = GST_Parser()
        self.salary_parser = SalarySlipParser()
        self.bank_analyzer = BankStatementAnalyzer()

    def verify(
        self,
        documents: Dict[str, Any],
        employment_type: str = "Salaried",
        declared_income: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Verify income from multiple document sources.

        Parameters
        ----------
        documents : dict
            Keys: 'itr', 'gst', 'salary_slips', 'bank_statements'
            Values: document data (dict or list of dicts)
        employment_type : str
            Affects source priority and confidence weighting.
        declared_income : float, optional
            Income declared by applicant (for discrepancy detection).

        Returns
        -------
        dict with: verified_income, confidence_score, sources, method,
        discrepancy_flag, details
        """
        results = {}
        sources_used = []

        # Parse all available sources
        if "itr" in documents and documents["itr"]:
            results["itr"] = self.itr_parser.parse(documents["itr"])
            sources_used.append("itr")

        if "gst" in documents and documents["gst"]:
            results["gst"] = self.gst_parser.parse(documents["gst"])
            sources_used.append("gst")

        if "salary_slips" in documents and documents["salary_slips"]:
            results["salary_slips"] = self.salary_parser.parse(documents["salary_slips"])
            sources_used.append("salary_slips")

        if "bank_statements" in documents and documents["bank_statements"]:
            results["bank_statements"] = self.bank_analyzer.analyze(documents["bank_statements"])
            sources_used.append("bank_statements")

        if not sources_used:
            return {
                "verified_income": 0,
                "confidence_score": 0.0,
                "sources": [],
                "method": "no_documents",
                "discrepancy_flag": False,
                "details": {},
            }

        # Select best source based on priority and employment type
        if employment_type in ("Self-Employed", "Business Owner"):
            priority = ["gst", "itr", "bank_statements", "salary_slips"]
        elif employment_type == "Salaried":
            priority = ["itr", "salary_slips", "bank_statements", "gst"]
        else:
            priority = self.SOURCE_PRIORITY

        # Get primary income estimate
        primary_source = None
        primary_income = 0
        primary_confidence = 0

        for source in priority:
            if source in results and results[source].get("verified_income", 0) > 0:
                primary_source = source
                primary_income = results[source]["verified_income"]
                primary_confidence = results[source].get("confidence", 0.5)
                break

        # Cross-validate with other sources
        all_incomes = {
            s: r.get("verified_income", 0)
            for s, r in results.items()
            if r.get("verified_income", 0) > 0
        }

        discrepancy_flag = False
        discrepancy_details = []

        if len(all_incomes) >= 2:
            values = list(all_incomes.values())
            max_income = max(values)
            min_income = min(values)
            variance_ratio = (max_income - min_income) / max(max_income, 1)

            if variance_ratio > 0.30:
                discrepancy_flag = True
                discrepancy_details.append(
                    f"Income variance across sources: {variance_ratio:.0%} "
                    f"(range: Rs {min_income:,.0f} - Rs {max_income:,.0f})"
                )
                # Use weighted average when discrepancy exists
                weights = [self.SOURCE_CONFIDENCE.get(s, 0.5) for s in all_incomes.keys()]
                primary_income = float(np.average(list(all_incomes.values()), weights=weights))
                primary_confidence *= 0.80  # Reduce confidence

        # Check against declared income
        if declared_income and declared_income > 0:
            ratio = primary_income / declared_income
            if ratio < 0.70:
                discrepancy_flag = True
                discrepancy_details.append(
                    f"Verified income ({primary_income:,.0f}) significantly below "
                    f"declared income ({declared_income:,.0f})"
                )
            elif ratio > 1.50:
                discrepancy_details.append(
                    f"Verified income ({primary_income:,.0f}) significantly above "
                    f"declared income ({declared_income:,.0f})"
                )

        # Final confidence
        confidence = min(primary_confidence * self.SOURCE_CONFIDENCE.get(primary_source, 0.5), 1.0)
        if len(sources_used) >= 2 and not discrepancy_flag:
            confidence = min(confidence + 0.05, 1.0)  # Boost for cross-validation

        return {
            "verified_income": round(primary_income, 2),
            "monthly_income": round(primary_income / 12, 2),
            "confidence_score": round(confidence, 4),
            "sources": sources_used,
            "primary_source": primary_source,
            "method": f"priority_based_{primary_source}",
            "discrepancy_flag": discrepancy_flag,
            "discrepancy_details": discrepancy_details,
            "all_income_estimates": all_incomes,
            "details": results,
        }
