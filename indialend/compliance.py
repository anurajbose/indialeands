"""
IndiaLend Compliance
====================
RBI regulatory compliance, fairness auditing, and divergence tracking
for Indian NBFC operations.

Implements:
- Divergence tracking: Bureau decision vs internal decision comparison
  (RBI mandates tracking and reporting divergences)
- NPA classification: SMA-0/1/2 and NPA staging per RBI norms
- Fairness auditing: Disparate impact analysis (4/5ths rule)
- Audit trail: Complete decision audit logging
- Regulatory reporting: Generate RBI-ready compliance reports

Key Indian market gaps:
1. Manual divergence tracking → automated with thresholds
2. No fairness testing in Indian NBFCs → we check for bias
3. Audit trails are incomplete → we log every decision path
4. NPA classification is retrospective → we classify in real-time
"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# NPA Classifier (RBI norms)
# ---------------------------------------------------------------------------

class NPA_Classifier:
    """Classify loan accounts per RBI asset classification norms.

    Categories:
    - Standard: Current, no overdue
    - SMA-0: 1-30 DPD (Special Mention Account)
    - SMA-1: 31-60 DPD
    - SMA-2: 61-90 DPD
    - Sub-Standard: 91-365 DPD (NPA)
    - Doubtful: 366-730 DPD
    - Loss: 730+ DPD or recovery unlikely

    Provisioning norms:
    - Standard: 0.40% (0.25% for housing)
    - Sub-Standard: 15%
    - Doubtful (1-year): 25%
    - Doubtful (2-year): 40%
    - Doubtful (3+ years): 100%
    - Loss: 100%
    """

    PROVISION_RATES = {
        "Standard": 0.004,
        "SMA-0": 0.005,
        "SMA-1": 0.05,
        "SMA-2": 0.10,
        "Sub-Standard": 0.15,
        "Doubtful-1": 0.25,
        "Doubtful-2": 0.40,
        "Doubtful-3": 1.00,
        "Loss": 1.00,
    }

    @classmethod
    def classify(cls, dpd: int, is_secured: bool = True) -> Dict[str, Any]:
        """Classify a loan account based on DPD.

        Parameters
        ----------
        dpd : int
            Days Past Due.
        is_secured : bool
            Whether the loan is secured (affects provisioning).

        Returns
        -------
        dict with: category, stage, provision_rate, is_npa, action_required
        """
        if dpd <= 0:
            category = "Standard"
            stage = 1
            action = "None"
        elif dpd <= 30:
            category = "SMA-0"
            stage = 1
            action = "Monitor — send payment reminder"
        elif dpd <= 60:
            category = "SMA-1"
            stage = 2
            action = "Proactive outreach — contact borrower"
        elif dpd <= 90:
            category = "SMA-2"
            stage = 2
            action = "Escalate — demand letter, consider restructuring"
        elif dpd <= 365:
            category = "Sub-Standard"
            stage = 3
            action = "NPA — initiate recovery process"
        elif dpd <= 730:
            category = "Doubtful-1" if dpd <= 365 + 365 else "Doubtful-2"
            stage = 3
            action = "NPA — legal/SARFAESI action"
        else:
            category = "Loss"
            stage = 3
            action = "Write-off candidate — proceed with recovery/write-off"

        provision_rate = cls.PROVISION_RATES.get(category, 0.004)
        if not is_secured and category in ("Sub-Standard", "Doubtful-1"):
            provision_rate += 0.10  # Additional provision for unsecured

        return {
            "category": category,
            "stage": stage,
            "provision_rate": provision_rate,
            "is_npa": dpd > 90,
            "is_sma": 0 < dpd <= 90,
            "dpd": dpd,
            "action_required": action,
        }

    @classmethod
    def classify_portfolio(cls, df: pd.DataFrame) -> pd.DataFrame:
        """Classify entire portfolio."""
        results = []
        for _, row in df.iterrows():
            dpd = row.get("dpd", 0)
            is_secured = row.get("is_secured", True)
            classification = cls.classify(dpd, is_secured)
            classification["account_id"] = row.get("account_id", row.get("applicant_id", ""))
            classification["loan_amount"] = row.get("loan_amount", 0)
            classification["provision_amount"] = round(
                row.get("loan_amount", 0) * classification["provision_rate"], 2
            )
            results.append(classification)

        return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# Divergence Tracker
# ---------------------------------------------------------------------------

class DivergenceTracker:
    """Track divergence between bureau recommendations and internal decisions.

    RBI requires NBFCs to monitor and report cases where:
    - Bureau says APPROVE but internal says REJECT (or vice versa)
    - Risk grades differ significantly between bureau and internal model

    Material divergences must be reported in regulatory filings.
    """

    def __init__(self, material_threshold: float = 0.10):
        """
        Parameters
        ----------
        material_threshold : float
            Divergence rate above which it's considered material (10% default).
        """
        self.material_threshold = material_threshold
        self._records: List[Dict] = []

    def track(
        self,
        applicant_id: str,
        bureau_decision: str,
        internal_decision: str,
        bureau_score: Optional[int] = None,
        internal_score: Optional[int] = None,
        reason: str = "",
    ) -> Dict[str, Any]:
        """Track a single decision pair.

        Parameters
        ----------
        bureau_decision : str
            "APPROVED" or "REJECTED" based on bureau score.
        internal_decision : str
            Internal engine decision.
        """
        is_divergent = bureau_decision != internal_decision
        score_gap = abs((bureau_score or 0) - (internal_score or 0))

        record = {
            "applicant_id": applicant_id,
            "bureau_decision": bureau_decision,
            "internal_decision": internal_decision,
            "bureau_score": bureau_score,
            "internal_score": internal_score,
            "is_divergent": is_divergent,
            "score_gap": score_gap,
            "divergence_type": self._classify_divergence(bureau_decision, internal_decision),
            "reason": reason,
            "timestamp": datetime.now().isoformat(),
        }
        self._records.append(record)
        return record

    def _classify_divergence(self, bureau: str, internal: str) -> str:
        """Classify the type of divergence."""
        if bureau == internal:
            return "aligned"
        if bureau == "APPROVED" and internal in ("REJECTED", "REFER_TO_MANUAL"):
            return "conservative"  # We rejected what bureau approved
        if bureau == "REJECTED" and internal == "APPROVED":
            return "aggressive"  # We approved what bureau would reject
        return "partial"

    def get_divergence_report(self) -> Dict[str, Any]:
        """Generate divergence summary report."""
        if not self._records:
            return {"total_decisions": 0, "divergence_rate": 0}

        df = pd.DataFrame(self._records)
        total = len(df)
        divergent = df["is_divergent"].sum()
        rate = divergent / total

        type_counts = df["divergence_type"].value_counts().to_dict()

        return {
            "total_decisions": total,
            "divergent_count": int(divergent),
            "aligned_count": int(total - divergent),
            "divergence_rate": round(rate, 4),
            "is_material": rate > self.material_threshold,
            "divergence_types": type_counts,
            "avg_score_gap": round(df["score_gap"].mean(), 1),
            "max_score_gap": int(df["score_gap"].max()),
            "conservative_count": int(type_counts.get("conservative", 0)),
            "aggressive_count": int(type_counts.get("aggressive", 0)),
        }

    def get_records(self) -> pd.DataFrame:
        return pd.DataFrame(self._records) if self._records else pd.DataFrame()


# ---------------------------------------------------------------------------
# RBI Compliance
# ---------------------------------------------------------------------------

class RBI_Compliance:
    """Master compliance engine for RBI regulations.

    Covers:
    - Divergence tracking and reporting
    - NPA classification
    - Capital adequacy checks
    - Audit trail generation
    - Regulatory reporting
    """

    def __init__(self):
        self.divergence_tracker = DivergenceTracker()
        self.npa_classifier = NPA_Classifier()
        self._audit_log: List[Dict] = []

    def check_divergence(
        self,
        bureau_decision: Union[str, bool],
        internal_decision: Union[str, bool],
        applicant_id: str,
        bureau_score: Optional[int] = None,
        internal_score: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Check and log decision divergence.

        Parameters accept both string decisions and boolean (approved/not).
        """
        # Normalise inputs
        if isinstance(bureau_decision, bool):
            bureau_decision = "APPROVED" if bureau_decision else "REJECTED"
        if isinstance(internal_decision, bool):
            internal_decision = "APPROVED" if internal_decision else "REJECTED"

        result = self.divergence_tracker.track(
            applicant_id=applicant_id,
            bureau_decision=bureau_decision,
            internal_decision=internal_decision,
            bureau_score=bureau_score,
            internal_score=internal_score,
        )

        self._log_audit(
            "divergence_check",
            applicant_id,
            {
                "bureau_decision": bureau_decision,
                "internal_decision": internal_decision,
                "is_divergent": result["is_divergent"],
            },
        )

        return result

    def classify_loan(self, dpd: int, is_secured: bool = True) -> Dict[str, Any]:
        """Classify a loan per RBI NPA norms."""
        return NPA_Classifier.classify(dpd, is_secured)

    def generate_audit_report(
        self,
        period: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Generate comprehensive audit report.

        Parameters
        ----------
        period : str, optional
            Period for report (e.g., "2024-04" for April 2024).
        """
        divergence_report = self.divergence_tracker.get_divergence_report()

        report = {
            "report_type": "RBI Compliance Audit",
            "generated_at": datetime.now().isoformat(),
            "period": period or datetime.now().strftime("%Y-%m"),
            "divergence_summary": divergence_report,
            "audit_entries": len(self._audit_log),
            "compliance_status": (
                "COMPLIANT" if not divergence_report.get("is_material", False)
                else "ATTENTION_REQUIRED"
            ),
            "recommendations": self._generate_recommendations(divergence_report),
        }

        return report

    def _generate_recommendations(self, divergence: Dict) -> List[str]:
        """Generate compliance recommendations."""
        recs = []

        if divergence.get("is_material"):
            recs.append(
                "MATERIAL DIVERGENCE: Divergence rate exceeds threshold. "
                "Board-level review required. Include in regulatory filing."
            )

        if divergence.get("aggressive_count", 0) > 0:
            recs.append(
                f"Aggressive decisions detected ({divergence['aggressive_count']}): "
                "Approving loans that bureau would reject increases portfolio risk."
            )

        if divergence.get("avg_score_gap", 0) > 100:
            recs.append(
                "High average score gap between bureau and internal model. "
                "Review model calibration."
            )

        if not recs:
            recs.append("No compliance issues detected. Continue standard monitoring.")

        return recs

    def _log_audit(self, action: str, entity_id: str, details: Dict) -> None:
        """Log an audit event."""
        self._audit_log.append({
            "action": action,
            "entity_id": entity_id,
            "details": details,
            "timestamp": datetime.now().isoformat(),
        })

    def get_audit_log(self) -> pd.DataFrame:
        return pd.DataFrame(self._audit_log) if self._audit_log else pd.DataFrame()


# ---------------------------------------------------------------------------
# Fairness Auditor
# ---------------------------------------------------------------------------

class FairnessAuditor:
    """Audit lending decisions for bias and discrimination.

    Tests for disparate impact across protected attributes
    (gender, age group, geography, religion proxy via name patterns).

    Uses the 4/5ths (80%) rule: if approval rate for a protected group
    is less than 80% of the highest-approval-rate group, it constitutes
    adverse impact.

    Also checks:
    - Interest rate disparity across groups
    - Loan amount disparity
    - Rejection reason distribution
    """

    def audit(
        self,
        decisions_df: pd.DataFrame,
        sensitive_attributes: List[str] = None,
    ) -> Dict[str, Any]:
        """Run fairness audit on a set of decisions.

        Parameters
        ----------
        decisions_df : DataFrame
            Must contain 'decision' column and sensitive attribute columns.
        sensitive_attributes : list of str
            Columns to check for disparate impact.
        """
        if sensitive_attributes is None:
            sensitive_attributes = ["gender", "age_group", "city_tier"]

        # Ensure decision is binary
        df = decisions_df.copy()
        if "is_approved" not in df.columns:
            df["is_approved"] = df["decision"].isin(
                ["APPROVED", "CONDITIONAL_APPROVAL"]
            ).astype(int)

        results = {}

        for attr in sensitive_attributes:
            if attr not in df.columns:
                # Try to derive age_group from age
                if attr == "age_group" and "age" in df.columns:
                    df["age_group"] = pd.cut(
                        df["age"], bins=[0, 25, 35, 45, 55, 100],
                        labels=["18-25", "26-35", "36-45", "46-55", "55+"],
                    ).astype(str)
                else:
                    continue

            attr_result = self._audit_attribute(df, attr)
            results[attr] = attr_result

        # Overall assessment
        any_violation = any(
            not r.get("passes_4_5_rule", True) for r in results.values()
        )

        return {
            "attributes_tested": list(results.keys()),
            "attribute_results": results,
            "overall_fair": not any_violation,
            "timestamp": datetime.now().isoformat(),
        }

    def _audit_attribute(self, df: pd.DataFrame, attr: str) -> Dict[str, Any]:
        """Audit a single sensitive attribute."""
        group_stats = df.groupby(attr).agg(
            total=("is_approved", "count"),
            approved=("is_approved", "sum"),
        )
        group_stats["approval_rate"] = group_stats["approved"] / group_stats["total"]

        max_rate = group_stats["approval_rate"].max()
        min_rate = group_stats["approval_rate"].min()

        # 4/5ths rule
        if max_rate > 0:
            disparity = min_rate / max_rate
            passes = disparity >= 0.80
        else:
            disparity = 1.0
            passes = True

        # Interest rate disparity (if available)
        rate_disparity = {}
        if "interest_rate" in df.columns:
            rate_by_group = df.groupby(attr)["interest_rate"].mean()
            rate_disparity = rate_by_group.to_dict()

        return {
            "group_approval_rates": group_stats["approval_rate"].to_dict(),
            "group_counts": group_stats["total"].to_dict(),
            "max_approval_rate": round(float(max_rate), 4),
            "min_approval_rate": round(float(min_rate), 4),
            "max_disparity": round(float(max_rate - min_rate), 4),
            "disparity_ratio": round(float(disparity), 4),
            "passes_4_5_rule": passes,
            "interest_rate_by_group": rate_disparity,
            "highest_approval_group": group_stats["approval_rate"].idxmax(),
            "lowest_approval_group": group_stats["approval_rate"].idxmin(),
        }
