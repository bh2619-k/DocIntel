"""
Real-Time Document Intelligence & Compliance Engine
Risk Scoring Engine
Version: 1.0
Date: 2026-02-03

This module provides comprehensive risk assessment with 5 risk scoring features:
1. Compliance Risk Scoring
2. Financial Risk Assessment
3. Operational Risk Evaluation
4. Regulatory Risk Analysis
5. Composite Risk Scoring
"""

import logging
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Set
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import json
from collections import defaultdict
import uuid
import math

logger = logging.getLogger(__name__)


class RiskLevel(Enum):
    """Risk severity levels"""

    MINIMAL = "minimal"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class RiskCategory(Enum):
    """Risk categories"""

    COMPLIANCE = "compliance"
    FINANCIAL = "financial"
    OPERATIONAL = "operational"
    REGULATORY = "regulatory"
    COMPOSITE = "composite"


@dataclass
class RiskFactor:
    """Individual risk factor"""

    factor_id: str
    name: str
    description: str
    weight: float  # 0.0-1.0
    value: float  # 0.0-1.0
    confidence: float  # 0.0-1.0
    evidence: List[str]
    source: str


@dataclass
class RiskAssessmentResult:
    """Risk assessment result"""

    assessment_id: str
    document_id: str
    risk_category: RiskCategory
    risk_level: RiskLevel
    risk_score: float  # 0.0-1.0
    confidence_score: float  # 0.0-1.0
    assessed_at: datetime
    risk_factors: List[RiskFactor]
    business_impact: Dict[str, float]
    mitigation_recommendations: List[str]
    context: Dict[str, Any]
    expires_at: Optional[datetime] = None


class ComplianceRiskScorer:
    """Compliance risk scoring system (Feature 1)"""

    def __init__(
        self,
        compliance_frameworks: List[str] = None,
        risk_thresholds: Dict[str, float] = None,
    ):
        """
        Initialize compliance risk scorer

        Args:
            compliance_frameworks: Supported compliance frameworks
            risk_thresholds: Risk level thresholds
        """
        self.compliance_frameworks = compliance_frameworks or [
            "SOX",
            "GDPR",
            "HIPAA",
            "PCI-DSS",
            "SOC2",
            "ISO27001",
            "CCPA",
        ]

        self.risk_thresholds = risk_thresholds or {
            "minimal": 0.2,
            "low": 0.4,
            "medium": 0.6,
            "high": 0.8,
            "critical": 1.0,
        }

        # Compliance rule weights by framework
        self.framework_weights = {
            "SOX": {"data_accuracy": 0.4, "audit_trail": 0.3, "access_control": 0.3},
            "GDPR": {
                "data_privacy": 0.4,
                "consent_management": 0.3,
                "data_retention": 0.3,
            },
            "HIPAA": {"data_security": 0.4, "access_logging": 0.3, "encryption": 0.3},
            "PCI-DSS": {
                "payment_security": 0.5,
                "network_security": 0.3,
                "monitoring": 0.2,
            },
        }

        # Compliance violation history
        self.violation_history: Dict[str, List[Tuple[datetime, str, float]]] = (
            defaultdict(list)
        )

    def assess_compliance_risk(
        self,
        document_data: Dict[str, Any],
        compliance_rules: List[Dict[str, Any]],
        framework: str = "SOX",
    ) -> RiskAssessmentResult:
        """
        Assess compliance risk for document

        Args:
            document_data: Document analysis data
            compliance_rules: Applied compliance rules
            framework: Compliance framework to evaluate

        Returns:
            Compliance risk assessment result
        """
        try:
            current_time = datetime.now()
            document_id = document_data.get("document_id", "unknown")

            risk_factors = []

            # 1. Rule violation risk
            violation_risk = self._assess_rule_violations(compliance_rules, framework)
            if violation_risk:
                risk_factors.append(violation_risk)

            # 2. Data quality risk
            quality_risk = self._assess_data_quality_risk(document_data, framework)
            if quality_risk:
                risk_factors.append(quality_risk)

            # 3. Control gap risk
            control_risk = self._assess_control_gaps(document_data, framework)
            if control_risk:
                risk_factors.append(control_risk)

            # 4. Historical compliance risk
            historical_risk = self._assess_historical_compliance(document_id, framework)
            if historical_risk:
                risk_factors.append(historical_risk)

            # 5. Framework-specific risk
            framework_risk = self._assess_framework_specific_risk(
                document_data, framework
            )
            if framework_risk:
                risk_factors.append(framework_risk)

            # Calculate overall compliance risk score
            risk_score = self._calculate_compliance_risk_score(risk_factors, framework)
            risk_level = self._get_risk_level(risk_score)

            # Calculate confidence
            confidence_score = self._calculate_compliance_confidence(risk_factors)

            # Business impact assessment
            business_impact = self._assess_compliance_business_impact(
                risk_score, framework
            )

            # Generate mitigation recommendations
            recommendations = self._generate_compliance_recommendations(
                risk_factors, framework
            )

            result = RiskAssessmentResult(
                assessment_id=str(uuid.uuid4()),
                document_id=document_id,
                risk_category=RiskCategory.COMPLIANCE,
                risk_level=risk_level,
                risk_score=risk_score,
                confidence_score=confidence_score,
                assessed_at=current_time,
                risk_factors=risk_factors,
                business_impact=business_impact,
                mitigation_recommendations=recommendations,
                context={
                    "framework": framework,
                    "total_rules": len(compliance_rules),
                    "violated_rules": len(
                        [r for r in compliance_rules if not r.get("passed", True)]
                    ),
                },
            )

            # Store in violation history if high risk
            if risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL]:
                self.violation_history[document_id].append(
                    (current_time, framework, risk_score)
                )

            return result

        except Exception as e:
            logger.error(f"Compliance risk assessment failed: {e}")
            return self._create_default_assessment(
                document_data, RiskCategory.COMPLIANCE
            )

    def _assess_rule_violations(
        self, compliance_rules: List[Dict[str, Any]], framework: str
    ) -> Optional[RiskFactor]:
        """Assess risk from rule violations"""
        try:
            if not compliance_rules:
                return None

            total_rules = len(compliance_rules)
            violated_rules = [r for r in compliance_rules if not r.get("passed", True)]
            critical_violations = [
                r for r in violated_rules if r.get("severity", "").lower() == "critical"
            ]

            if not violated_rules:
                violation_risk = 0.1  # Minimal risk with no violations
            else:
                # Calculate violation risk based on severity and count
                violation_rate = len(violated_rules) / total_rules
                critical_rate = (
                    len(critical_violations) / total_rules if total_rules > 0 else 0
                )

                violation_risk = violation_rate * 0.6 + critical_rate * 0.4

            evidence = [
                f"Total rules evaluated: {total_rules}",
                f"Rules violated: {len(violated_rules)}",
                f"Critical violations: {len(critical_violations)}",
            ]

            return RiskFactor(
                factor_id=f"rule_violations_{framework}",
                name="Rule Violations",
                description=f"Risk from {framework} compliance rule violations",
                weight=0.3,
                value=violation_risk,
                confidence=0.9 if total_rules > 10 else 0.7,
                evidence=evidence,
                source="compliance_rules_engine",
            )

        except Exception as e:
            logger.warning(f"Rule violation assessment failed: {e}")
            return None

    def _assess_data_quality_risk(
        self, document_data: Dict[str, Any], framework: str
    ) -> Optional[RiskFactor]:
        """Assess risk from data quality issues"""
        try:
            quality_metrics = document_data.get("quality_metrics", {})

            if not quality_metrics:
                return None

            # Key quality indicators
            completeness = quality_metrics.get("completeness", 1.0)
            accuracy = quality_metrics.get("accuracy", 1.0)
            consistency = quality_metrics.get("consistency", 1.0)
            validity = quality_metrics.get("validity", 1.0)

            # Calculate data quality risk (inverse of quality score)
            avg_quality = np.mean([completeness, accuracy, consistency, validity])
            quality_risk = 1.0 - avg_quality

            evidence = [
                f"Data completeness: {completeness:.2f}",
                f"Data accuracy: {accuracy:.2f}",
                f"Data consistency: {consistency:.2f}",
                f"Data validity: {validity:.2f}",
            ]

            return RiskFactor(
                factor_id=f"data_quality_{framework}",
                name="Data Quality Risk",
                description="Risk from poor data quality affecting compliance",
                weight=0.25,
                value=quality_risk,
                confidence=0.8,
                evidence=evidence,
                source="data_quality_analyzer",
            )

        except Exception as e:
            logger.warning(f"Data quality risk assessment failed: {e}")
            return None

    def _assess_control_gaps(
        self, document_data: Dict[str, Any], framework: str
    ) -> Optional[RiskFactor]:
        """Assess risk from control gaps"""
        try:
            controls = document_data.get("controls", {})

            required_controls = self._get_required_controls(framework)
            implemented_controls = set(controls.keys())

            missing_controls = set(required_controls) - implemented_controls
            control_gap_rate = (
                len(missing_controls) / len(required_controls)
                if required_controls
                else 0
            )

            # Check control effectiveness
            ineffective_controls = [
                name
                for name, effectiveness in controls.items()
                if isinstance(effectiveness, (int, float)) and effectiveness < 0.7
            ]

            effectiveness_gap = (
                len(ineffective_controls) / len(controls) if controls else 0
            )

            overall_control_risk = control_gap_rate * 0.6 + effectiveness_gap * 0.4

            evidence = [
                f"Required controls: {len(required_controls)}",
                f"Missing controls: {len(missing_controls)}",
                f"Ineffective controls: {len(ineffective_controls)}",
            ]

            return RiskFactor(
                factor_id=f"control_gaps_{framework}",
                name="Control Gaps",
                description="Risk from missing or ineffective controls",
                weight=0.2,
                value=overall_control_risk,
                confidence=0.75,
                evidence=evidence,
                source="control_assessment",
            )

        except Exception as e:
            logger.warning(f"Control gap assessment failed: {e}")
            return None

    def _assess_historical_compliance(
        self, document_id: str, framework: str
    ) -> Optional[RiskFactor]:
        """Assess risk based on historical compliance"""
        try:
            if document_id not in self.violation_history:
                return None

            # Get recent violation history (last 90 days)
            cutoff_date = datetime.now() - timedelta(days=90)
            recent_violations = [
                (ts, fw, score)
                for ts, fw, score in self.violation_history[document_id]
                if ts >= cutoff_date and fw == framework
            ]

            if not recent_violations:
                historical_risk = 0.1  # Low risk with clean history
            else:
                # Calculate risk based on frequency and severity of recent violations
                avg_violation_score = np.mean(
                    [score for _, _, score in recent_violations]
                )
                violation_frequency = len(recent_violations) / 90  # violations per day

                historical_risk = min(
                    1.0, avg_violation_score * 0.7 + violation_frequency * 10 * 0.3
                )

            evidence = [
                f"Historical violations (90 days): {len(recent_violations)}",
                f"Average violation score: {avg_violation_score:.2f}"
                if recent_violations
                else "No recent violations",
            ]

            return RiskFactor(
                factor_id=f"historical_compliance_{framework}",
                name="Historical Compliance",
                description="Risk based on historical compliance violations",
                weight=0.15,
                value=historical_risk,
                confidence=0.8 if recent_violations else 0.6,
                evidence=evidence,
                source="violation_history",
            )

        except Exception as e:
            logger.warning(f"Historical compliance assessment failed: {e}")
            return None

    def _assess_framework_specific_risk(
        self, document_data: Dict[str, Any], framework: str
    ) -> Optional[RiskFactor]:
        """Assess framework-specific risks"""
        try:
            framework_risk = 0.0
            evidence = []

            if framework == "SOX":
                # SOX-specific risks: financial accuracy, audit trail
                financial_accuracy = document_data.get("financial_accuracy", 1.0)
                audit_completeness = document_data.get("audit_trail_completeness", 1.0)

                framework_risk = 1.0 - ((financial_accuracy + audit_completeness) / 2)
                evidence = [
                    f"Financial accuracy: {financial_accuracy:.2f}",
                    f"Audit completeness: {audit_completeness:.2f}",
                ]

            elif framework == "GDPR":
                # GDPR-specific risks: privacy, consent, data retention
                privacy_compliance = document_data.get("privacy_compliance", 1.0)
                consent_validity = document_data.get("consent_validity", 1.0)

                framework_risk = 1.0 - ((privacy_compliance + consent_validity) / 2)
                evidence = [
                    f"Privacy compliance: {privacy_compliance:.2f}",
                    f"Consent validity: {consent_validity:.2f}",
                ]

            elif framework == "HIPAA":
                # HIPAA-specific risks: health data security, access controls
                security_compliance = document_data.get("security_compliance", 1.0)
                access_control_strength = document_data.get(
                    "access_control_strength", 1.0
                )

                framework_risk = 1.0 - (
                    (security_compliance + access_control_strength) / 2
                )
                evidence = [
                    f"Security compliance: {security_compliance:.2f}",
                    f"Access control strength: {access_control_strength:.2f}",
                ]

            if framework_risk > 0:
                return RiskFactor(
                    factor_id=f"framework_specific_{framework}",
                    name=f"{framework} Specific Risk",
                    description=f"Risk specific to {framework} requirements",
                    weight=0.1,
                    value=framework_risk,
                    confidence=0.7,
                    evidence=evidence,
                    source=f"{framework.lower()}_assessment",
                )

            return None

        except Exception as e:
            logger.warning(f"Framework-specific risk assessment failed: {e}")
            return None

    def _get_required_controls(self, framework: str) -> List[str]:
        """Get required controls for framework"""
        control_map = {
            "SOX": [
                "access_control",
                "audit_logging",
                "data_validation",
                "segregation_of_duties",
            ],
            "GDPR": [
                "consent_management",
                "data_encryption",
                "access_rights",
                "breach_notification",
            ],
            "HIPAA": ["access_control", "audit_logs", "encryption", "risk_assessment"],
            "PCI-DSS": [
                "network_security",
                "access_control",
                "monitoring",
                "encryption",
            ],
        }
        return control_map.get(framework, [])

    def _calculate_compliance_risk_score(
        self, risk_factors: List[RiskFactor], framework: str
    ) -> float:
        """Calculate overall compliance risk score"""
        try:
            if not risk_factors:
                return 0.0

            # Weighted average of risk factors
            total_weight = sum(factor.weight for factor in risk_factors)
            if total_weight == 0:
                return 0.0

            weighted_score = (
                sum(factor.value * factor.weight for factor in risk_factors)
                / total_weight
            )

            # Framework-specific adjustments
            if framework in ["SOX", "HIPAA"]:
                # More stringent frameworks - increase risk for any violations
                weighted_score *= 1.1
            elif framework == "GDPR":
                # High penalty framework - increase risk significantly
                weighted_score *= 1.2

            return min(1.0, weighted_score)

        except Exception:
            return 0.5

    def _calculate_compliance_confidence(self, risk_factors: List[RiskFactor]) -> float:
        """Calculate confidence in compliance risk assessment"""
        try:
            if not risk_factors:
                return 0.5

            # Average confidence weighted by factor importance
            total_weight = sum(factor.weight for factor in risk_factors)
            if total_weight == 0:
                return 0.5

            weighted_confidence = (
                sum(factor.confidence * factor.weight for factor in risk_factors)
                / total_weight
            )
            return weighted_confidence

        except Exception:
            return 0.5

    def _assess_compliance_business_impact(
        self, risk_score: float, framework: str
    ) -> Dict[str, float]:
        """Assess business impact of compliance risk"""
        try:
            base_impact = risk_score

            # Framework-specific impact multipliers
            impact_multipliers = {
                "SOX": {"financial": 1.5, "reputation": 1.3, "legal": 1.4},
                "GDPR": {"financial": 2.0, "reputation": 1.8, "legal": 1.9},
                "HIPAA": {"financial": 1.4, "reputation": 1.6, "legal": 1.5},
                "PCI-DSS": {"financial": 1.3, "reputation": 1.2, "legal": 1.1},
            }

            multipliers = impact_multipliers.get(
                framework, {"financial": 1.0, "reputation": 1.0, "legal": 1.0}
            )

            return {
                "financial_impact": min(1.0, base_impact * multipliers["financial"]),
                "reputation_impact": min(1.0, base_impact * multipliers["reputation"]),
                "legal_impact": min(1.0, base_impact * multipliers["legal"]),
                "operational_impact": min(1.0, base_impact * 0.8),
            }

        except Exception:
            return {
                "financial_impact": 0.5,
                "reputation_impact": 0.5,
                "legal_impact": 0.5,
                "operational_impact": 0.5,
            }

    def _generate_compliance_recommendations(
        self, risk_factors: List[RiskFactor], framework: str
    ) -> List[str]:
        """Generate compliance risk mitigation recommendations"""
        recommendations = []

        try:
            for factor in risk_factors:
                if factor.value > 0.6:  # High risk factors
                    if "violation" in factor.name.lower():
                        recommendations.append(
                            f"Address {framework} rule violations immediately"
                        )
                        recommendations.append(
                            "Implement automated compliance monitoring"
                        )
                    elif "quality" in factor.name.lower():
                        recommendations.append(
                            "Improve data quality processes and validation"
                        )
                        recommendations.append(
                            "Implement data quality monitoring dashboards"
                        )
                    elif "control" in factor.name.lower():
                        recommendations.append(
                            "Implement missing controls and improve effectiveness"
                        )
                        recommendations.append("Regular control testing and validation")
                    elif "historical" in factor.name.lower():
                        recommendations.append(
                            "Strengthen compliance processes to prevent recurrence"
                        )
                        recommendations.append(
                            "Implement proactive compliance monitoring"
                        )

            # Framework-specific recommendations
            if framework == "GDPR":
                recommendations.append(
                    "Ensure proper consent management and data subject rights"
                )
            elif framework == "SOX":
                recommendations.append("Strengthen financial controls and audit trails")
            elif framework == "HIPAA":
                recommendations.append(
                    "Enhance health data security and access controls"
                )

            return list(set(recommendations))  # Remove duplicates

        except Exception:
            return [
                "Implement comprehensive compliance monitoring",
                "Regular compliance assessments",
            ]

    def _get_risk_level(self, risk_score: float) -> RiskLevel:
        """Convert risk score to risk level"""
        if risk_score >= self.risk_thresholds["critical"]:
            return RiskLevel.CRITICAL
        elif risk_score >= self.risk_thresholds["high"]:
            return RiskLevel.HIGH
        elif risk_score >= self.risk_thresholds["medium"]:
            return RiskLevel.MEDIUM
        elif risk_score >= self.risk_thresholds["low"]:
            return RiskLevel.LOW
        else:
            return RiskLevel.MINIMAL

    def _create_default_assessment(
        self, document_data: Dict[str, Any], category: RiskCategory
    ) -> RiskAssessmentResult:
        """Create default assessment when processing fails"""
        return RiskAssessmentResult(
            assessment_id=str(uuid.uuid4()),
            document_id=document_data.get("document_id", "unknown"),
            risk_category=category,
            risk_level=RiskLevel.MEDIUM,
            risk_score=0.5,
            confidence_score=0.3,
            assessed_at=datetime.now(),
            risk_factors=[],
            business_impact={
                "financial_impact": 0.5,
                "reputation_impact": 0.5,
                "legal_impact": 0.5,
                "operational_impact": 0.5,
            },
            mitigation_recommendations=[
                "Manual risk assessment required",
                "Insufficient data for automated assessment",
            ],
            context={"error": "Assessment processing failed"},
        )
