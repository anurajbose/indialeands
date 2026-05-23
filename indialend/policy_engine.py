"""
IndiaLend Policy Engine
========================
Business Rules Engine (BRE) for automated credit decisioning.

Replaces hardcoded if/else chains with a configurable, auditable
rule system. Supports:
- Eligibility screening (knockout rules)
- Credit approval with conditional terms
- Risk-based dynamic pricing
- Product-specific policy sets
- A/B testing of rule variants
- Full decision audit trail

Key Indian market gaps:
1. Most NBFCs hardcode rules in application code → brittle, untestable
2. Rate changes require code deploys → we allow runtime rule updates
3. No audit trail → we log every rule evaluation
4. Single approval path → we support conditional approvals with conditions
"""

from __future__ import annotations

import copy
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Decision Rules Database
# ---------------------------------------------------------------------------

class DecisionRules:
    """Configurable rule database for credit decisioning.

    Rules are organised by category:
    - eligibility: Knockout rules (must-pass)
    - approval: Approval criteria
    - pricing: Interest rate determination
    - limits: Loan amount and tenure limits
    - product: Product-specific overrides
    """

    _rules: Dict[str, Dict[str, Any]] = {
        # --- Eligibility (knockout) ---
        "eligibility": {
            "min_age": 21,
            "max_age": 60,
            "min_income": 200_000,  # Rs 2 lakh p.a.
            "min_bureau_score": 550,
            "max_dpd_90_12m": 0,     # Zero 90+ DPD in last 12 months
            "max_written_off": 0,     # Zero write-offs
            "max_suit_filed": 0,      # No active suits
            "max_foir": 0.65,         # Max 65% FOIR
            "min_employment_years": 1,
            "required_kyc": True,
        },
        # --- Approval thresholds ---
        "approval": {
            "min_bureau_score": 650,
            "min_income": 300_000,
            "max_foir": 0.55,
            "max_dpd_30_12m": 1,
            "max_enquiries_3m": 4,
            "min_account_age_months": 12,
            "conditional_score_range": (550, 649),  # Conditional approval zone
        },
        # --- Pricing ---
        "pricing": {
            "base_rate": 12.0,  # Base interest rate %
            "risk_premium": {
                "A": 0.0,    # 750-900
                "B": 1.5,    # 700-749
                "C": 3.0,    # 650-699
                "D": 5.0,    # 600-649
                "E": 8.0,    # <600
            },
            "employment_adjustment": {
                "Salaried": -0.5,
                "Professional": 0.0,
                "Self-Employed": 0.5,
                "Business Owner": 1.0,
            },
            "tenure_adjustment": {
                (0, 12): 0.5,
                (13, 36): 0.0,
                (37, 60): -0.25,
                (61, 120): -0.50,
                (121, 360): 0.50,  # Long tenure = higher rate
            },
            "min_rate": 9.0,
            "max_rate": 24.0,
        },
        # --- Loan limits ---
        "limits": {
            "min_loan_amount": 25_000,
            "max_income_multiplier": {
                "Personal Loan": 10,
                "Auto Loan": 8,
                "Home Loan": 50,
                "Business Loan": 15,
                "Education Loan": 12,
                "Gold Loan": 5,
                "Loan Against Property": 30,
                "Two-Wheeler Loan": 4,
                "Consumer Durable Loan": 3,
                "MSME Loan": 12,
            },
            "max_ltv": {
                "Home Loan": 0.80,
                "Auto Loan": 0.85,
                "Loan Against Property": 0.60,
                "Gold Loan": 0.75,
                "Two-Wheeler Loan": 0.90,
            },
            "max_tenure_months": {
                "Personal Loan": 60,
                "Auto Loan": 84,
                "Home Loan": 360,
                "Business Loan": 60,
                "Education Loan": 180,
                "Gold Loan": 36,
                "Loan Against Property": 240,
                "Two-Wheeler Loan": 48,
                "Consumer Durable Loan": 24,
                "MSME Loan": 60,
            },
        },
    }

    @classmethod
    def all_rules(cls) -> Dict:
        """Get all current rules."""
        return copy.deepcopy(cls._rules)

    @classmethod
    def get_rule(cls, category: str, key: str) -> Any:
        """Get a specific rule value."""
        return cls._rules.get(category, {}).get(key)

    @classmethod
    def update_rule(cls, category: str, key: str, value: Any) -> None:
        """Update a rule value (for A/B testing, policy changes)."""
        if category not in cls._rules:
            cls._rules[category] = {}
        cls._rules[category][key] = value

    @classmethod
    def reset_rules(cls) -> None:
        """Reset rules to defaults (useful after A/B tests)."""
        cls._rules = copy.deepcopy(DecisionRules._rules)


# ---------------------------------------------------------------------------
# BRE Evaluator (Business Rules Engine)
# ---------------------------------------------------------------------------

class BRE_Evaluator:
    """Evaluate a single applicant against a rule set.

    Each rule evaluation returns:
    - passed (bool)
    - rule_name (str)
    - actual_value
    - threshold_value
    - message (str)
    """

    def evaluate_rule(
        self,
        rule_name: str,
        actual: Any,
        threshold: Any,
        operator: str = ">=",
    ) -> Dict[str, Any]:
        """Evaluate a single rule.

        Parameters
        ----------
        operator : str
            ">=", "<=", ">", "<", "==", "!=", "in", "not_in"
        """
        ops = {
            ">=": lambda a, t: a >= t,
            "<=": lambda a, t: a <= t,
            ">": lambda a, t: a > t,
            "<": lambda a, t: a < t,
            "==": lambda a, t: a == t,
            "!=": lambda a, t: a != t,
            "in": lambda a, t: a in t,
            "not_in": lambda a, t: a not in t,
        }

        op_func = ops.get(operator, ops[">="])
        passed = op_func(actual, threshold)

        return {
            "rule_name": rule_name,
            "passed": passed,
            "actual_value": actual,
            "threshold_value": threshold,
            "operator": operator,
            "message": (
                f"PASS: {rule_name}" if passed
                else f"FAIL: {rule_name} — actual={actual}, required {operator} {threshold}"
            ),
        }


# ---------------------------------------------------------------------------
# Policy Engine
# ---------------------------------------------------------------------------

class PolicyEngine:
    """Main policy engine for credit decisioning.

    Evaluates applicant against all policy rules and produces a
    structured decision:
    - APPROVED: All criteria met
    - CONDITIONAL_APPROVAL: Near-miss on some criteria, approvable with conditions
    - REJECTED: Failed knockout or multiple criteria
    - REFER_TO_MANUAL: Edge cases needing human review

    Also determines:
    - Approved loan amount (may differ from requested)
    - Interest rate (risk-based pricing)
    - Tenure
    - Conditions (if conditional approval)
    """

    DECISION_APPROVED = "APPROVED"
    DECISION_CONDITIONAL = "CONDITIONAL_APPROVAL"
    DECISION_REJECTED = "REJECTED"
    DECISION_REFER = "REFER_TO_MANUAL"

    def __init__(self, rules: Optional[Dict] = None):
        self.rules = rules or DecisionRules.all_rules()
        self.bre = BRE_Evaluator()

    def evaluate(
        self,
        applicant_data: Dict[str, Any],
        bureau_response: Any = None,
        income_verification: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """Run full policy evaluation.

        Parameters
        ----------
        applicant_data : dict
            Applicant profile data.
        bureau_response : BureauResponse or dict, optional
            Bureau data (converted to dict if BureauResponse).
        income_verification : dict, optional
            Income verification result.

        Returns
        -------
        dict with: decision, loan_amount, interest_rate, tenure,
        conditions, rule_results, reasons
        """
        # Normalise inputs
        data = dict(applicant_data)
        if bureau_response is not None:
            if hasattr(bureau_response, "to_dict"):
                bureau_data = bureau_response.to_dict()
            elif isinstance(bureau_response, dict):
                bureau_data = bureau_response
            else:
                bureau_data = {}
            data.update(bureau_data)

        if income_verification:
            data["verified_income"] = income_verification.get("verified_income", data.get("annual_income", 0))
            data["income_confidence"] = income_verification.get("confidence_score", 0)
            data["income_discrepancy"] = income_verification.get("discrepancy_flag", False)

        # Use verified income if available, else declared
        effective_income = data.get("verified_income", data.get("annual_income", 0))
        data["effective_income"] = effective_income

        # Run all checks
        eligibility_results = self._check_eligibility(data)
        approval_results = self._check_approval(data)
        all_results = eligibility_results + approval_results

        # Determine decision
        eligibility_passed = all(r["passed"] for r in eligibility_results)
        approval_passed = all(r["passed"] for r in approval_results)
        failed_rules = [r for r in all_results if not r["passed"]]

        if not eligibility_passed:
            decision = self.DECISION_REJECTED
            reasons = [r["message"] for r in eligibility_results if not r["passed"]]
        elif approval_passed:
            decision = self.DECISION_APPROVED
            reasons = ["All criteria met"]
        elif len(failed_rules) <= 2:
            # Check if conditional approval is possible
            bureau_score = data.get("cibil_score", 0)
            cond_range = self.rules["approval"].get("conditional_score_range", (550, 649))
            if cond_range[0] <= bureau_score <= cond_range[1]:
                decision = self.DECISION_CONDITIONAL
                reasons = [r["message"] for r in failed_rules]
            else:
                decision = self.DECISION_REFER
                reasons = [r["message"] for r in failed_rules]
        else:
            decision = self.DECISION_REJECTED
            reasons = [r["message"] for r in failed_rules]

        # Calculate loan terms
        loan_amount, tenure, interest_rate, conditions = self._calculate_terms(
            data, decision
        )

        # EMI calculation
        if interest_rate > 0 and tenure > 0:
            monthly_rate = interest_rate / 100 / 12
            emi = loan_amount * monthly_rate * (1 + monthly_rate) ** tenure / (
                (1 + monthly_rate) ** tenure - 1
            )
        else:
            emi = 0

        return {
            "decision": decision,
            "loan_amount": int(loan_amount),
            "tenure": int(tenure),
            "interest_rate": round(interest_rate, 2),
            "emi": int(emi),
            "conditions": conditions,
            "reasons": reasons,
            "rule_results": all_results,
            "rules_passed": sum(1 for r in all_results if r["passed"]),
            "rules_failed": sum(1 for r in all_results if not r["passed"]),
            "risk_grade": data.get("risk_grade", ""),
            "effective_income": int(effective_income),
            "timestamp": datetime.now().isoformat(),
        }

    def _check_eligibility(self, data: Dict) -> List[Dict]:
        """Check knockout eligibility rules."""
        rules = self.rules.get("eligibility", {})
        results = []

        # Age
        results.append(self.bre.evaluate_rule(
            "min_age", data.get("age", 0), rules.get("min_age", 21), ">="
        ))
        results.append(self.bre.evaluate_rule(
            "max_age", data.get("age", 100), rules.get("max_age", 60), "<="
        ))

        # Income
        income = data.get("effective_income", data.get("annual_income", 0))
        results.append(self.bre.evaluate_rule(
            "min_income", income, rules.get("min_income", 200_000), ">="
        ))

        # Bureau score
        score = data.get("cibil_score", 0)
        results.append(self.bre.evaluate_rule(
            "min_bureau_score", score, rules.get("min_bureau_score", 550), ">="
        ))

        # DPD 90+ (knockout)
        results.append(self.bre.evaluate_rule(
            "max_dpd_90",
            data.get("dpd_90_12m", 0),
            rules.get("max_dpd_90_12m", 0),
            "<=",
        ))

        # Write-offs
        results.append(self.bre.evaluate_rule(
            "max_written_off",
            data.get("written_off_accounts", 0),
            rules.get("max_written_off", 0),
            "<=",
        ))

        # FOIR
        foir = data.get("foir", 0)
        if foir == 0 and data.get("existing_emi", 0) > 0 and income > 0:
            foir = (data["existing_emi"] * 12) / income
        results.append(self.bre.evaluate_rule(
            "max_foir", foir, rules.get("max_foir", 0.65), "<="
        ))

        return results

    def _check_approval(self, data: Dict) -> List[Dict]:
        """Check approval criteria (non-knockout)."""
        rules = self.rules.get("approval", {})
        results = []

        # Bureau score (approval threshold is higher than eligibility)
        results.append(self.bre.evaluate_rule(
            "approval_bureau_score",
            data.get("cibil_score", 0),
            rules.get("min_bureau_score", 650),
            ">=",
        ))

        # Income (approval threshold)
        results.append(self.bre.evaluate_rule(
            "approval_income",
            data.get("effective_income", 0),
            rules.get("min_income", 300_000),
            ">=",
        ))

        # FOIR (tighter for approval)
        foir = data.get("foir", 0)
        results.append(self.bre.evaluate_rule(
            "approval_foir", foir, rules.get("max_foir", 0.55), "<="
        ))

        # DPD 30+ count
        results.append(self.bre.evaluate_rule(
            "max_dpd_30",
            data.get("dpd_30_12m", 0),
            rules.get("max_dpd_30_12m", 1),
            "<=",
        ))

        # Enquiry intensity
        results.append(self.bre.evaluate_rule(
            "max_enquiries_3m",
            data.get("enquiries_3m", 0),
            rules.get("max_enquiries_3m", 4),
            "<=",
        ))

        # Credit history length
        results.append(self.bre.evaluate_rule(
            "min_account_age",
            data.get("oldest_account_months", 0),
            rules.get("min_account_age_months", 12),
            ">=",
        ))

        return results

    def _calculate_terms(
        self,
        data: Dict,
        decision: str,
    ) -> Tuple[float, int, float, List[str]]:
        """Calculate loan amount, tenure, interest rate, and conditions."""
        conditions: List[str] = []

        if decision == self.DECISION_REJECTED:
            return 0, 0, 0.0, ["Application rejected"]

        product = data.get("loan_product", "Personal Loan")
        requested_amount = data.get("loan_amount", 0)
        requested_tenure = data.get("tenure_months", data.get("tenure", 36))
        income = data.get("effective_income", 0)
        score = data.get("cibil_score", 0)

        # --- Loan amount ---
        max_mult = self.rules["limits"]["max_income_multiplier"].get(product, 10)
        max_amount = income * max_mult

        # LTV check
        max_ltv = self.rules["limits"]["max_ltv"].get(product, 1.0)
        asset_value = data.get("asset_value", requested_amount * 2)
        ltv_max_amount = asset_value * max_ltv

        approved_amount = min(requested_amount, max_amount, ltv_max_amount)

        # For conditional approval, reduce amount by 20%
        if decision == self.DECISION_CONDITIONAL:
            approved_amount *= 0.80
            conditions.append("Loan amount reduced by 20% due to conditional approval")

        # --- Tenure ---
        max_tenure = self.rules["limits"]["max_tenure_months"].get(product, 60)
        approved_tenure = min(requested_tenure, max_tenure)

        # --- Interest rate (risk-based pricing) ---
        pricing = self.rules["pricing"]
        base_rate = pricing["base_rate"]

        # Risk grade premium
        risk_grade = data.get("risk_grade", "C")
        if not risk_grade:
            if score >= 750:
                risk_grade = "A"
            elif score >= 700:
                risk_grade = "B"
            elif score >= 650:
                risk_grade = "C"
            elif score >= 600:
                risk_grade = "D"
            else:
                risk_grade = "E"
        risk_premium = pricing["risk_premium"].get(risk_grade, 3.0)

        # Employment adjustment
        emp_type = data.get("employment_type", "Salaried")
        emp_adj = pricing["employment_adjustment"].get(emp_type, 0)

        # Tenure adjustment
        tenure_adj = 0
        for (lo, hi), adj in pricing["tenure_adjustment"].items():
            if lo <= approved_tenure <= hi:
                tenure_adj = adj
                break

        # Conditional approval surcharge
        cond_adj = 1.5 if decision == self.DECISION_CONDITIONAL else 0

        interest_rate = base_rate + risk_premium + emp_adj + tenure_adj + cond_adj
        interest_rate = np.clip(interest_rate, pricing["min_rate"], pricing["max_rate"])

        # --- Conditions ---
        if decision == self.DECISION_CONDITIONAL:
            if score < 650:
                conditions.append("Provide additional income proof")
            if data.get("foir", 0) > 0.50:
                conditions.append("Reduce existing obligations or provide co-applicant")
            if data.get("enquiries_3m", 0) > 3:
                conditions.append("Explain recent credit enquiries")
            if not conditions:
                conditions.append("Additional documentation required")

        if decision == self.DECISION_REFER:
            conditions.append("Manual review by credit committee required")

        return approved_amount, approved_tenure, interest_rate, conditions

    def batch_evaluate(
        self,
        applicants_df: pd.DataFrame,
        bureau_data: Optional[Dict[str, Dict]] = None,
    ) -> pd.DataFrame:
        """Evaluate a batch of applicants.

        Parameters
        ----------
        applicants_df : DataFrame
            Applicant data.
        bureau_data : dict, optional
            Dict mapping applicant_id to bureau response dict.

        Returns
        -------
        DataFrame with decision results for each applicant.
        """
        results = []

        for _, row in applicants_df.iterrows():
            data = row.to_dict()
            app_id = data.get("applicant_id", "")
            bureau = bureau_data.get(app_id, {}) if bureau_data else {}

            decision = self.evaluate(data, bureau_response=bureau)
            decision["applicant_id"] = app_id
            results.append(decision)

        df = pd.DataFrame(results)

        # Summary stats
        if len(df) > 0:
            df.attrs["approval_rate"] = (
                df["decision"].isin([self.DECISION_APPROVED, self.DECISION_CONDITIONAL]).mean()
            )
            df.attrs["avg_interest_rate"] = df.loc[
                df["decision"] != self.DECISION_REJECTED, "interest_rate"
            ].mean()

        return df
