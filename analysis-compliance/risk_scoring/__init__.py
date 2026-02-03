"""
Risk Scoring Module
Multi-factor risk assessment and scoring
"""

from .composite_risk import (
    CompositeRiskScorer,
    OperationalRiskAssessor,
    RegulatoryRiskAssessor,
)

try:
    from .compliance_risk import ComplianceRiskScorer
    from .financial_risk import FinancialRiskAssessor

    HAS_SPECIALIZED_RISK = True
except ImportError:
    HAS_SPECIALIZED_RISK = False

__all__ = ["CompositeRiskScorer", "OperationalRiskAssessor", "RegulatoryRiskAssessor"]

if HAS_SPECIALIZED_RISK:
    __all__.extend(["ComplianceRiskScorer", "FinancialRiskAssessor"])
