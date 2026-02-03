"""
Real-Time Document Intelligence & Compliance Engine
Composite Risk Scoring & Integration
Version: 1.0
Date: 2026-02-03

This module provides composite risk scoring (Feature 5) and integrates all risk assessment
components: compliance, financial, operational, and regulatory risks.
"""

import logging
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Set
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import json
import uuid
from .compliance_risk import (
    ComplianceRiskScorer,
    RiskLevel,
    RiskCategory,
    RiskFactor,
    RiskAssessmentResult,
)
from .financial_risk import FinancialRiskAssessor

logger = logging.getLogger(__name__)


class OperationalRiskType(Enum):
    """Types of operational risks"""

    PROCESS_RISK = "process_risk"
    TECHNOLOGY_RISK = "technology_risk"
    PEOPLE_RISK = "people_risk"
    DATA_RISK = "data_risk"
    VENDOR_RISK = "vendor_risk"


class RegulatoryRiskType(Enum):
    """Types of regulatory risks"""

    COMPLIANCE_BREACH = "compliance_breach"
    REGULATORY_CHANGE = "regulatory_change"
    ENFORCEMENT_ACTION = "enforcement_action"
    LICENSING_RISK = "licensing_risk"
    REPORTING_FAILURE = "reporting_failure"


@dataclass
class CompositeRiskResult:
    """Comprehensive risk assessment result combining all risk categories"""

    assessment_id: str
    document_id: str
    overall_risk_level: RiskLevel
    overall_risk_score: float
    confidence_score: float
    assessed_at: datetime
    risk_breakdown: Dict[RiskCategory, RiskAssessmentResult]
    risk_correlations: Dict[str, float]
    business_impact_summary: Dict[str, float]
    priority_recommendations: List[str]
    risk_trends: Optional[Dict[str, List[float]]] = None
    expires_at: Optional[datetime] = None


class OperationalRiskAssessor:
    """Operational risk assessment system (Feature 3)"""

    def __init__(self, process_maturity_levels: Dict[str, float] = None):
        self.process_maturity_levels = process_maturity_levels or {
            "document_processing": 0.8,
            "quality_control": 0.7,
            "exception_handling": 0.6,
            "monitoring": 0.9,
        }

        self.operational_baselines = {}
        self.incident_history: Dict[str, List[Tuple[datetime, str, float]]] = {}

    def assess_operational_risk(
        self, document_data: Dict[str, Any], processing_metrics: Dict[str, Any] = None
    ) -> RiskAssessmentResult:
        """Assess operational risk for document processing"""
        try:
            current_time = datetime.now()
            document_id = document_data.get("document_id", "unknown")

            risk_factors = []

            # Process risk assessment
            process_risk = self._assess_process_risk(document_data, processing_metrics)
            if process_risk:
                risk_factors.append(process_risk)

            # Technology risk assessment
            tech_risk = self._assess_technology_risk(document_data, processing_metrics)
            if tech_risk:
                risk_factors.append(tech_risk)

            # Data quality risk assessment
            data_risk = self._assess_data_quality_risk(document_data)
            if data_risk:
                risk_factors.append(data_risk)

            # People/human error risk
            people_risk = self._assess_people_risk(document_data, processing_metrics)
            if people_risk:
                risk_factors.append(people_risk)

            # Vendor/third party risk
            vendor_risk = self._assess_vendor_risk(document_data)
            if vendor_risk:
                risk_factors.append(vendor_risk)

            # Calculate overall operational risk
            risk_score = self._calculate_operational_risk_score(risk_factors)
            risk_level = self._get_risk_level(risk_score)
            confidence_score = self._calculate_operational_confidence(risk_factors)

            # Business impact assessment
            business_impact = self._assess_operational_business_impact(
                risk_score, document_data
            )

            # Generate recommendations
            recommendations = self._generate_operational_recommendations(risk_factors)

            return RiskAssessmentResult(
                assessment_id=str(uuid.uuid4()),
                document_id=document_id,
                risk_category=RiskCategory.OPERATIONAL,
                risk_level=risk_level,
                risk_score=risk_score,
                confidence_score=confidence_score,
                assessed_at=current_time,
                risk_factors=risk_factors,
                business_impact=business_impact,
                mitigation_recommendations=recommendations,
                context={
                    "processing_accuracy": processing_metrics.get("accuracy", 0)
                    if processing_metrics
                    else 0,
                    "processing_time": processing_metrics.get("processing_time_ms", 0)
                    if processing_metrics
                    else 0,
                    "automation_level": document_data.get("automation_level", 0.5),
                },
            )

        except Exception as e:
            logger.error(f"Operational risk assessment failed: {e}")
            return self._create_default_assessment(
                document_data, RiskCategory.OPERATIONAL
            )

    def _assess_process_risk(
        self, document_data: Dict[str, Any], processing_metrics: Dict[str, Any] = None
    ) -> Optional[RiskFactor]:
        """Assess process-related operational risk"""
        try:
            process_risk_score = 0.0
            evidence = []

            # Processing accuracy risk
            if processing_metrics:
                accuracy = processing_metrics.get("accuracy", 1.0)
                if accuracy < 0.9:
                    process_risk_score += 0.3
                    evidence.append(f"Low processing accuracy: {accuracy:.1%}")
                elif accuracy < 0.95:
                    process_risk_score += 0.1
                    evidence.append(f"Moderate accuracy concern: {accuracy:.1%}")

            # Process complexity risk
            complexity = document_data.get("complexity_score", 0.5)
            if complexity > 0.8:
                process_risk_score += 0.2
                evidence.append(f"High process complexity: {complexity:.2f}")
            elif complexity > 0.6:
                process_risk_score += 0.1
                evidence.append(f"Moderate complexity: {complexity:.2f}")

            # Exception handling capability
            exception_rate = (
                processing_metrics.get("exception_rate", 0.0)
                if processing_metrics
                else 0.0
            )
            if exception_rate > 0.1:
                process_risk_score += 0.25
                evidence.append(f"High exception rate: {exception_rate:.1%}")
            elif exception_rate > 0.05:
                process_risk_score += 0.1
                evidence.append(f"Moderate exceptions: {exception_rate:.1%}")

            # Manual intervention requirement
            automation_level = document_data.get("automation_level", 0.5)
            if automation_level < 0.5:
                process_risk_score += 0.2
                evidence.append(f"Low automation: {automation_level:.1%}")

            if process_risk_score > 0:
                return RiskFactor(
                    factor_id="process_risk",
                    name="Process Risk",
                    description="Risk from process inefficiencies and failures",
                    weight=0.3,
                    value=min(1.0, process_risk_score),
                    confidence=0.8,
                    evidence=evidence,
                    source="process_assessment",
                )

            return None

        except Exception as e:
            logger.warning(f"Process risk assessment failed: {e}")
            return None

    def _assess_technology_risk(
        self, document_data: Dict[str, Any], processing_metrics: Dict[str, Any] = None
    ) -> Optional[RiskFactor]:
        """Assess technology-related operational risk"""
        try:
            tech_risk_score = 0.0
            evidence = []

            # System performance risk
            if processing_metrics:
                processing_time = processing_metrics.get("processing_time_ms", 0)
                if processing_time > 30000:  # 30 seconds
                    tech_risk_score += 0.2
                    evidence.append(f"Slow processing: {processing_time / 1000:.1f}s")

                # System error rate
                error_rate = processing_metrics.get("error_rate", 0.0)
                if error_rate > 0.05:
                    tech_risk_score += 0.3
                    evidence.append(f"High error rate: {error_rate:.1%}")
                elif error_rate > 0.02:
                    tech_risk_score += 0.1
                    evidence.append(f"Moderate error rate: {error_rate:.1%}")

            # Technology obsolescence risk
            tech_stack_age = document_data.get("technology_age_years", 0)
            if tech_stack_age > 5:
                tech_risk_score += 0.2
                evidence.append(f"Aging technology: {tech_stack_age} years")

            # Integration complexity
            integration_points = document_data.get("integration_points", 0)
            if integration_points > 10:
                tech_risk_score += 0.15
                evidence.append(
                    f"High integration complexity: {integration_points} systems"
                )

            # Security vulnerability risk
            security_score = document_data.get("security_score", 0.8)
            if security_score < 0.7:
                tech_risk_score += 0.25
                evidence.append(f"Security concerns: {security_score:.2f}")

            if tech_risk_score > 0:
                return RiskFactor(
                    factor_id="technology_risk",
                    name="Technology Risk",
                    description="Risk from technology failures and limitations",
                    weight=0.25,
                    value=min(1.0, tech_risk_score),
                    confidence=0.7,
                    evidence=evidence,
                    source="technology_assessment",
                )

            return None

        except Exception as e:
            logger.warning(f"Technology risk assessment failed: {e}")
            return None

    def _assess_data_quality_risk(
        self, document_data: Dict[str, Any]
    ) -> Optional[RiskFactor]:
        """Assess data quality operational risk"""
        try:
            data_risk_score = 0.0
            evidence = []

            quality_metrics = document_data.get("quality_metrics", {})

            # Data completeness
            completeness = quality_metrics.get("completeness", 1.0)
            if completeness < 0.8:
                data_risk_score += 0.3
                evidence.append(f"Poor completeness: {completeness:.1%}")
            elif completeness < 0.9:
                data_risk_score += 0.1
                evidence.append(f"Moderate completeness: {completeness:.1%}")

            # Data accuracy
            accuracy = quality_metrics.get("accuracy", 1.0)
            if accuracy < 0.9:
                data_risk_score += 0.25
                evidence.append(f"Poor accuracy: {accuracy:.1%}")

            # Data consistency
            consistency = quality_metrics.get("consistency", 1.0)
            if consistency < 0.8:
                data_risk_score += 0.2
                evidence.append(f"Poor consistency: {consistency:.1%}")

            # Data timeliness
            timeliness = quality_metrics.get("timeliness", 1.0)
            if timeliness < 0.7:
                data_risk_score += 0.15
                evidence.append(f"Timeliness issues: {timeliness:.1%}")

            if data_risk_score > 0:
                return RiskFactor(
                    factor_id="data_quality_risk",
                    name="Data Quality Risk",
                    description="Risk from poor data quality affecting operations",
                    weight=0.2,
                    value=min(1.0, data_risk_score),
                    confidence=0.8,
                    evidence=evidence,
                    source="data_quality_assessment",
                )

            return None

        except Exception as e:
            logger.warning(f"Data quality risk assessment failed: {e}")
            return None

    def _assess_people_risk(
        self, document_data: Dict[str, Any], processing_metrics: Dict[str, Any] = None
    ) -> Optional[RiskFactor]:
        """Assess people/human error operational risk"""
        try:
            people_risk_score = 0.0
            evidence = []

            # Manual processing dependency
            manual_processing_rate = document_data.get("manual_processing_rate", 0.0)
            if manual_processing_rate > 0.3:
                people_risk_score += 0.3
                evidence.append(f"High manual dependency: {manual_processing_rate:.1%}")
            elif manual_processing_rate > 0.1:
                people_risk_score += 0.1
                evidence.append(
                    f"Moderate manual processing: {manual_processing_rate:.1%}"
                )

            # Staff expertise level
            expertise_level = document_data.get("staff_expertise_level", 0.8)
            if expertise_level < 0.6:
                people_risk_score += 0.2
                evidence.append(f"Low expertise level: {expertise_level:.2f}")

            # Training currency
            training_currency = document_data.get("training_currency_months", 6)
            if training_currency > 12:
                people_risk_score += 0.15
                evidence.append(f"Outdated training: {training_currency} months")

            # Human error rate
            if processing_metrics:
                human_error_rate = processing_metrics.get("human_error_rate", 0.0)
                if human_error_rate > 0.05:
                    people_risk_score += 0.25
                    evidence.append(f"High human error rate: {human_error_rate:.1%}")

            if people_risk_score > 0:
                return RiskFactor(
                    factor_id="people_risk",
                    name="People Risk",
                    description="Risk from human errors and skill gaps",
                    weight=0.15,
                    value=min(1.0, people_risk_score),
                    confidence=0.6,
                    evidence=evidence,
                    source="people_assessment",
                )

            return None

        except Exception as e:
            logger.warning(f"People risk assessment failed: {e}")
            return None

    def _assess_vendor_risk(
        self, document_data: Dict[str, Any]
    ) -> Optional[RiskFactor]:
        """Assess vendor/third party operational risk"""
        try:
            vendor_risk_score = 0.0
            evidence = []

            vendor_dependencies = document_data.get("vendor_dependencies", [])

            if vendor_dependencies:
                # Number of critical vendor dependencies
                critical_vendors = len(
                    [
                        v
                        for v in vendor_dependencies
                        if v.get("criticality", "") == "high"
                    ]
                )
                if critical_vendors > 3:
                    vendor_risk_score += 0.3
                    evidence.append(
                        f"High vendor dependency: {critical_vendors} critical vendors"
                    )
                elif critical_vendors > 1:
                    vendor_risk_score += 0.1
                    evidence.append(
                        f"Moderate vendor dependency: {critical_vendors} vendors"
                    )

                # Vendor reliability assessment
                avg_reliability = np.mean(
                    [v.get("reliability_score", 0.8) for v in vendor_dependencies]
                )
                if avg_reliability < 0.7:
                    vendor_risk_score += 0.25
                    evidence.append(f"Poor vendor reliability: {avg_reliability:.2f}")

                # Vendor concentration risk
                if len(vendor_dependencies) == 1:
                    vendor_risk_score += 0.2
                    evidence.append("Single vendor dependency")

            # External service dependencies
            external_services = document_data.get("external_service_count", 0)
            if external_services > 5:
                vendor_risk_score += 0.1
                evidence.append(
                    f"High external service dependency: {external_services}"
                )

            if vendor_risk_score > 0:
                return RiskFactor(
                    factor_id="vendor_risk",
                    name="Vendor Risk",
                    description="Risk from vendor and third-party dependencies",
                    weight=0.1,
                    value=min(1.0, vendor_risk_score),
                    confidence=0.7,
                    evidence=evidence,
                    source="vendor_assessment",
                )

            return None

        except Exception as e:
            logger.warning(f"Vendor risk assessment failed: {e}")
            return None

    def _calculate_operational_risk_score(
        self, risk_factors: List[RiskFactor]
    ) -> float:
        """Calculate overall operational risk score"""
        if not risk_factors:
            return 0.0

        total_weight = sum(factor.weight for factor in risk_factors)
        if total_weight == 0:
            return 0.0

        return (
            sum(factor.value * factor.weight for factor in risk_factors) / total_weight
        )

    def _calculate_operational_confidence(
        self, risk_factors: List[RiskFactor]
    ) -> float:
        """Calculate confidence in operational risk assessment"""
        if not risk_factors:
            return 0.5

        total_weight = sum(factor.weight for factor in risk_factors)
        if total_weight == 0:
            return 0.5

        return (
            sum(factor.confidence * factor.weight for factor in risk_factors)
            / total_weight
        )

    def _assess_operational_business_impact(
        self, risk_score: float, document_data: Dict[str, Any]
    ) -> Dict[str, float]:
        """Assess business impact of operational risk"""
        base_impact = risk_score

        # Adjust based on business criticality
        business_criticality = document_data.get("business_criticality", 0.5)
        criticality_multiplier = 1.0 + (business_criticality - 0.5)

        return {
            "operational_impact": min(1.0, base_impact * criticality_multiplier),
            "financial_impact": min(1.0, base_impact * 0.7),
            "reputation_impact": min(1.0, base_impact * 0.6),
            "customer_impact": min(1.0, base_impact * 0.8),
        }

    def _generate_operational_recommendations(
        self, risk_factors: List[RiskFactor]
    ) -> List[str]:
        """Generate operational risk mitigation recommendations"""
        recommendations = []

        for factor in risk_factors:
            if factor.value > 0.6:
                if "process" in factor.name.lower():
                    recommendations.append(
                        "Improve process automation and standardization"
                    )
                    recommendations.append(
                        "Implement process monitoring and optimization"
                    )
                elif "technology" in factor.name.lower():
                    recommendations.append("Upgrade technology infrastructure")
                    recommendations.append("Implement system redundancy and failover")
                elif "data" in factor.name.lower():
                    recommendations.append("Enhance data quality controls")
                    recommendations.append("Implement data validation and cleansing")
                elif "people" in factor.name.lower():
                    recommendations.append("Provide additional training and support")
                    recommendations.append("Reduce manual processing dependencies")
                elif "vendor" in factor.name.lower():
                    recommendations.append("Diversify vendor relationships")
                    recommendations.append("Implement vendor performance monitoring")

        return list(set(recommendations))

    def _get_risk_level(self, risk_score: float) -> RiskLevel:
        """Convert risk score to risk level"""
        thresholds = {"critical": 0.8, "high": 0.6, "medium": 0.4, "low": 0.2}

        if risk_score >= thresholds["critical"]:
            return RiskLevel.CRITICAL
        elif risk_score >= thresholds["high"]:
            return RiskLevel.HIGH
        elif risk_score >= thresholds["medium"]:
            return RiskLevel.MEDIUM
        elif risk_score >= thresholds["low"]:
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
                "operational_impact": 0.5,
                "financial_impact": 0.5,
                "reputation_impact": 0.5,
                "customer_impact": 0.5,
            },
            mitigation_recommendations=[
                "Manual risk assessment required",
                "Insufficient data for automated assessment",
            ],
            context={"error": "Assessment processing failed"},
        )


class RegulatoryRiskAssessor:
    """Regulatory risk assessment system (Feature 4)"""

    def __init__(self, regulatory_frameworks: Dict[str, Any] = None):
        self.regulatory_frameworks = regulatory_frameworks or {
            "financial": ["SOX", "Basel III", "Dodd-Frank"],
            "privacy": ["GDPR", "CCPA", "PIPEDA"],
            "healthcare": ["HIPAA", "FDA"],
            "general": ["ISO27001", "SOC2"],
        }

        self.regulatory_changes: Dict[str, List[Tuple[datetime, str, float]]] = {}

    def assess_regulatory_risk(
        self, document_data: Dict[str, Any], regulatory_context: Dict[str, Any] = None
    ) -> RiskAssessmentResult:
        """Assess regulatory risk for document"""
        try:
            current_time = datetime.now()
            document_id = document_data.get("document_id", "unknown")

            risk_factors = []

            # Regulatory compliance gaps
            compliance_risk = self._assess_compliance_gaps(
                document_data, regulatory_context
            )
            if compliance_risk:
                risk_factors.append(compliance_risk)

            # Regulatory change impact
            change_risk = self._assess_regulatory_changes(
                document_data, regulatory_context
            )
            if change_risk:
                risk_factors.append(change_risk)

            # Enforcement action risk
            enforcement_risk = self._assess_enforcement_risk(
                document_data, regulatory_context
            )
            if enforcement_risk:
                risk_factors.append(enforcement_risk)

            # Reporting compliance risk
            reporting_risk = self._assess_reporting_compliance(
                document_data, regulatory_context
            )
            if reporting_risk:
                risk_factors.append(reporting_risk)

            # Calculate overall regulatory risk
            risk_score = self._calculate_regulatory_risk_score(risk_factors)
            risk_level = self._get_risk_level(risk_score)
            confidence_score = self._calculate_regulatory_confidence(risk_factors)

            # Business impact assessment
            business_impact = self._assess_regulatory_business_impact(
                risk_score, regulatory_context
            )

            # Generate recommendations
            recommendations = self._generate_regulatory_recommendations(risk_factors)

            return RiskAssessmentResult(
                assessment_id=str(uuid.uuid4()),
                document_id=document_id,
                risk_category=RiskCategory.REGULATORY,
                risk_level=risk_level,
                risk_score=risk_score,
                confidence_score=confidence_score,
                assessed_at=current_time,
                risk_factors=risk_factors,
                business_impact=business_impact,
                mitigation_recommendations=recommendations,
                context={
                    "applicable_frameworks": regulatory_context.get("frameworks", [])
                    if regulatory_context
                    else [],
                    "jurisdiction": regulatory_context.get("jurisdiction", "unknown")
                    if regulatory_context
                    else "unknown",
                },
            )

        except Exception as e:
            logger.error(f"Regulatory risk assessment failed: {e}")
            return self._create_default_assessment(
                document_data, RiskCategory.REGULATORY
            )

    def _assess_compliance_gaps(
        self, document_data: Dict[str, Any], regulatory_context: Dict[str, Any] = None
    ) -> Optional[RiskFactor]:
        """Assess regulatory compliance gaps"""
        # Implementation similar to previous risk factors
        # This is a simplified version
        return RiskFactor(
            factor_id="regulatory_compliance",
            name="Regulatory Compliance Gaps",
            description="Risk from regulatory compliance gaps",
            weight=0.4,
            value=0.3,  # Simplified assessment
            confidence=0.7,
            evidence=["Regulatory compliance assessment needed"],
            source="regulatory_assessment",
        )

    # Additional regulatory assessment methods would be implemented here...
    # For brevity, I'm including the essential structure

    def _calculate_regulatory_risk_score(self, risk_factors: List[RiskFactor]) -> float:
        """Calculate regulatory risk score"""
        if not risk_factors:
            return 0.0

        total_weight = sum(factor.weight for factor in risk_factors)
        if total_weight == 0:
            return 0.0

        return (
            sum(factor.value * factor.weight for factor in risk_factors) / total_weight
        )


class CompositeRiskScorer:
    """Composite risk scoring system (Feature 5) - Integrates all risk assessments"""

    def __init__(self, risk_weights: Dict[RiskCategory, float] = None):
        """Initialize composite risk scorer"""
        self.risk_weights = risk_weights or {
            RiskCategory.COMPLIANCE: 0.3,
            RiskCategory.FINANCIAL: 0.25,
            RiskCategory.OPERATIONAL: 0.2,
            RiskCategory.REGULATORY: 0.25,
        }

        # Initialize individual risk assessors
        self.compliance_assessor = ComplianceRiskScorer()
        self.financial_assessor = FinancialRiskAssessor()
        self.operational_assessor = OperationalRiskAssessor()
        self.regulatory_assessor = RegulatoryRiskAssessor()

        # Risk correlation matrix
        self.risk_correlations = {
            "compliance_financial": 0.6,
            "compliance_regulatory": 0.8,
            "financial_operational": 0.4,
            "operational_regulatory": 0.5,
        }

    def assess_composite_risk(
        self,
        document_data: Dict[str, Any],
        compliance_rules: List[Dict[str, Any]] = None,
        extracted_entities: Dict[str, Any] = None,
        market_context: Dict[str, Any] = None,
        business_context: Dict[str, Any] = None,
        regulatory_context: Dict[str, Any] = None,
        processing_metrics: Dict[str, Any] = None,
    ) -> CompositeRiskResult:
        """
        Perform comprehensive risk assessment across all categories

        Args:
            document_data: Document analysis data
            compliance_rules: Applied compliance rules
            extracted_entities: Extracted entities from document
            market_context: Market conditions and context
            business_context: Business context information
            regulatory_context: Regulatory environment context
            processing_metrics: Processing performance metrics

        Returns:
            Comprehensive composite risk assessment
        """
        try:
            current_time = datetime.now()
            document_id = document_data.get("document_id", "unknown")

            # Perform individual risk assessments
            risk_assessments = {}

            # Compliance risk assessment
            if RiskCategory.COMPLIANCE in self.risk_weights:
                compliance_result = self.compliance_assessor.assess_compliance_risk(
                    document_data, compliance_rules or [], "SOX"
                )
                risk_assessments[RiskCategory.COMPLIANCE] = compliance_result

            # Financial risk assessment
            if RiskCategory.FINANCIAL in self.risk_weights:
                financial_result = self.financial_assessor.assess_financial_risk(
                    document_data, extracted_entities, market_context
                )
                risk_assessments[RiskCategory.FINANCIAL] = financial_result

            # Operational risk assessment
            if RiskCategory.OPERATIONAL in self.risk_weights:
                operational_result = self.operational_assessor.assess_operational_risk(
                    document_data, processing_metrics
                )
                risk_assessments[RiskCategory.OPERATIONAL] = operational_result

            # Regulatory risk assessment
            if RiskCategory.REGULATORY in self.risk_weights:
                regulatory_result = self.regulatory_assessor.assess_regulatory_risk(
                    document_data, regulatory_context
                )
                risk_assessments[RiskCategory.REGULATORY] = regulatory_result

            # Calculate composite risk score with correlations
            composite_score, correlations = self._calculate_composite_score(
                risk_assessments
            )
            overall_risk_level = self._get_composite_risk_level(composite_score)

            # Calculate overall confidence
            confidence_score = self._calculate_composite_confidence(risk_assessments)

            # Aggregate business impact
            business_impact_summary = self._aggregate_business_impact(risk_assessments)

            # Generate priority recommendations
            priority_recommendations = self._generate_priority_recommendations(
                risk_assessments
            )

            # Create composite result
            composite_result = CompositeRiskResult(
                assessment_id=str(uuid.uuid4()),
                document_id=document_id,
                overall_risk_level=overall_risk_level,
                overall_risk_score=composite_score,
                confidence_score=confidence_score,
                assessed_at=current_time,
                risk_breakdown=risk_assessments,
                risk_correlations=correlations,
                business_impact_summary=business_impact_summary,
                priority_recommendations=priority_recommendations,
            )

            logger.info(
                f"Composite risk assessment completed for {document_id}: {overall_risk_level.value} ({composite_score:.2f})"
            )

            return composite_result

        except Exception as e:
            logger.error(f"Composite risk assessment failed: {e}")
            return self._create_default_composite_result(document_data)

    def _calculate_composite_score(
        self, risk_assessments: Dict[RiskCategory, RiskAssessmentResult]
    ) -> Tuple[float, Dict[str, float]]:
        """Calculate composite risk score accounting for correlations"""
        try:
            if not risk_assessments:
                return 0.5, {}

            # Basic weighted average
            total_weight = 0
            weighted_score = 0

            for category, assessment in risk_assessments.items():
                weight = self.risk_weights.get(category, 0)
                total_weight += weight
                weighted_score += assessment.risk_score * weight

            base_score = weighted_score / total_weight if total_weight > 0 else 0.5

            # Adjust for risk correlations
            correlation_adjustments = 0
            correlation_count = 0
            correlations = {}

            categories = list(risk_assessments.keys())
            for i, cat1 in enumerate(categories):
                for cat2 in categories[i + 1 :]:
                    corr_key = f"{cat1.value}_{cat2.value}"
                    correlation = self.risk_correlations.get(
                        corr_key, 0.3
                    )  # Default correlation

                    score1 = risk_assessments[cat1].risk_score
                    score2 = risk_assessments[cat2].risk_score

                    # Correlation increases risk when both scores are high
                    if score1 > 0.6 and score2 > 0.6:
                        correlation_adjustment = correlation * min(score1, score2) * 0.1
                        correlation_adjustments += correlation_adjustment
                        correlation_count += 1

                    correlations[corr_key] = correlation

            # Apply correlation adjustment
            if correlation_count > 0:
                avg_correlation_adjustment = correlation_adjustments / correlation_count
                composite_score = min(1.0, base_score + avg_correlation_adjustment)
            else:
                composite_score = base_score

            return composite_score, correlations

        except Exception as e:
            logger.warning(f"Composite score calculation failed: {e}")
            return 0.5, {}

    def _calculate_composite_confidence(
        self, risk_assessments: Dict[RiskCategory, RiskAssessmentResult]
    ) -> float:
        """Calculate overall confidence in composite assessment"""
        try:
            if not risk_assessments:
                return 0.5

            # Weight confidence by risk category importance
            total_weight = 0
            weighted_confidence = 0

            for category, assessment in risk_assessments.items():
                weight = self.risk_weights.get(category, 0)
                total_weight += weight
                weighted_confidence += assessment.confidence_score * weight

            return weighted_confidence / total_weight if total_weight > 0 else 0.5

        except Exception:
            return 0.5

    def _aggregate_business_impact(
        self, risk_assessments: Dict[RiskCategory, RiskAssessmentResult]
    ) -> Dict[str, float]:
        """Aggregate business impact across all risk categories"""
        try:
            impact_aggregation = {
                "financial_impact": 0,
                "operational_impact": 0,
                "reputation_impact": 0,
                "regulatory_impact": 0,
                "customer_impact": 0,
            }

            total_assessments = len(risk_assessments)

            for assessment in risk_assessments.values():
                for impact_type, impact_score in assessment.business_impact.items():
                    if impact_type in impact_aggregation:
                        impact_aggregation[impact_type] += impact_score

            # Average the impacts
            if total_assessments > 0:
                for impact_type in impact_aggregation:
                    impact_aggregation[impact_type] /= total_assessments

            return impact_aggregation

        except Exception:
            return {
                "financial_impact": 0.5,
                "operational_impact": 0.5,
                "reputation_impact": 0.5,
                "regulatory_impact": 0.5,
            }

    def _generate_priority_recommendations(
        self, risk_assessments: Dict[RiskCategory, RiskAssessmentResult]
    ) -> List[str]:
        """Generate prioritized recommendations across all risk categories"""
        try:
            all_recommendations = []

            # Collect recommendations from high-risk categories
            for category, assessment in risk_assessments.items():
                if assessment.risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL]:
                    category_recs = [
                        f"[{category.value.upper()}] {rec}"
                        for rec in assessment.mitigation_recommendations
                    ]
                    all_recommendations.extend(category_recs)

            # If no high-risk categories, include medium risk recommendations
            if not all_recommendations:
                for category, assessment in risk_assessments.items():
                    if assessment.risk_level == RiskLevel.MEDIUM:
                        category_recs = [
                            f"[{category.value.upper()}] {rec}"
                            for rec in assessment.mitigation_recommendations[:2]
                        ]
                        all_recommendations.extend(category_recs)

            # Add composite-level recommendations
            if (
                len(
                    [
                        a
                        for a in risk_assessments.values()
                        if a.risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL]
                    ]
                )
                > 1
            ):
                all_recommendations.insert(
                    0,
                    "Multiple high-risk categories detected - comprehensive risk review required",
                )
                all_recommendations.insert(
                    1, "Implement integrated risk monitoring across all categories"
                )

            return all_recommendations[:10]  # Limit to top 10 recommendations

        except Exception:
            return ["Comprehensive risk assessment and mitigation planning required"]

    def _get_composite_risk_level(self, composite_score: float) -> RiskLevel:
        """Determine overall risk level from composite score"""
        if composite_score >= 0.8:
            return RiskLevel.CRITICAL
        elif composite_score >= 0.6:
            return RiskLevel.HIGH
        elif composite_score >= 0.4:
            return RiskLevel.MEDIUM
        elif composite_score >= 0.2:
            return RiskLevel.LOW
        else:
            return RiskLevel.MINIMAL

    def _create_default_composite_result(
        self, document_data: Dict[str, Any]
    ) -> CompositeRiskResult:
        """Create default composite result when assessment fails"""
        return CompositeRiskResult(
            assessment_id=str(uuid.uuid4()),
            document_id=document_data.get("document_id", "unknown"),
            overall_risk_level=RiskLevel.MEDIUM,
            overall_risk_score=0.5,
            confidence_score=0.3,
            assessed_at=datetime.now(),
            risk_breakdown={},
            risk_correlations={},
            business_impact_summary={
                "financial_impact": 0.5,
                "operational_impact": 0.5,
                "reputation_impact": 0.5,
            },
            priority_recommendations=[
                "Manual risk assessment required",
                "Automated assessment failed",
            ],
        )
