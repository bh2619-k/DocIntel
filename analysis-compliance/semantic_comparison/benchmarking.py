"""
Real-Time Document Intelligence & Compliance Engine
Contract Benchmarking & Deviation Analysis System
Version: 1.0
Date: 2026-02-03

This module provides contract benchmarking against industry standards,
deviation detection, and performance analysis using semantic embeddings.
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
from dataclasses import dataclass
from enum import Enum
from collections import defaultdict
import json
import uuid
import statistics

try:
    import numpy as np

    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

    # Mock numpy functionality for basic operations
    class MockNumpy:
        def mean(self, data):
            return statistics.mean(data) if data else 0.0

        def min(self, data):
            return min(data) if data else 0.0

        def max(self, data):
            return max(data) if data else 0.0

        def array(self, data):
            return data

    np = MockNumpy()

logger = logging.getLogger(__name__)


class BenchmarkType(Enum):
    """Types of benchmarks"""

    INDUSTRY_STANDARD = "industry_standard"
    COMPANY_TEMPLATE = "company_template"
    REGULATORY_BASELINE = "regulatory_baseline"
    MARKET_BEST_PRACTICE = "market_best_practice"
    HISTORICAL_AVERAGE = "historical_average"


class DeviationType(Enum):
    """Types of contract deviations"""

    MISSING_CLAUSE = "missing_clause"
    MODIFIED_CLAUSE = "modified_clause"
    EXTRA_CLAUSE = "extra_clause"
    FORMATTING_ISSUE = "formatting_issue"


@dataclass
class ContractDeviation:
    """Contract deviation from standard"""

    deviation_id: str
    deviation_type: DeviationType
    clause_type: str
    description: str
    severity: str  # LOW, MEDIUM, HIGH, CRITICAL
    impact_score: float
    business_risk: float
    legal_risk: float
    financial_impact: Optional[float]
    recommendations: List[str]
    evidence: Dict[str, Any]


@dataclass
class BenchmarkStandard:
    """Benchmark standard definition"""

    benchmark_id: str
    name: str
    description: str
    benchmark_type: BenchmarkType
    industry: str
    jurisdiction: str
    version: str
    created_at: datetime
    clause_templates: Dict[str, Dict[str, Any]]
    performance_targets: Dict[str, float]
    risk_thresholds: Dict[str, float]
    metadata: Dict[str, Any]


@dataclass
class BenchmarkResult:
    """Contract benchmarking result"""

    benchmark_id: str
    contract_id: str
    benchmark_standard_id: str
    overall_compliance: float
    confidence_score: float
    benchmarked_at: datetime
    clause_level_results: List[Dict[str, Any]]
    performance_metrics: Dict[str, float]
    identified_gaps: List[ContractDeviation]
    improvement_recommendations: List[str]
    risk_assessment: Dict[str, float]
    metadata: Dict[str, Any]


class PerformanceMetric(Enum):
    """Contract performance metrics"""

    NEGOTIATION_SPEED = "negotiation_speed"
    CLAUSE_COMPLETENESS = "clause_completeness"
    RISK_SCORE = "risk_score"
    COMPLIANCE_SCORE = "compliance_score"
    STANDARDIZATION_LEVEL = "standardization_level"


@dataclass
class BenchmarkStandard:
    """Benchmark standard definition"""

    benchmark_id: str
    name: str
    description: str
    benchmark_type: BenchmarkType
    industry: str
    jurisdiction: str
    version: str
    created_at: datetime
    clause_templates: Dict[str, Dict[str, Any]]
    performance_targets: Dict[str, float]
    risk_thresholds: Dict[str, float]
    metadata: Dict[str, Any]


@dataclass
class BenchmarkResult:
    """Contract benchmarking result"""

    benchmark_id: str
    contract_id: str
    benchmark_standard_id: str
    overall_compliance: float
    confidence_score: float
    benchmarked_at: datetime
    clause_level_results: List[Dict[str, Any]]
    performance_metrics: Dict[str, float]
    identified_gaps: List[ContractDeviation]
    improvement_recommendations: List[str]
    risk_assessment: Dict[str, float]
    metadata: Dict[str, Any]


@dataclass
class DeviationAnalysis:
    """Comprehensive deviation analysis"""

    analysis_id: str
    contract_id: str
    analysis_type: str
    total_deviations: int
    critical_deviations: int
    deviation_categories: Dict[str, int]
    impact_assessment: Dict[str, float]
    trend_analysis: Optional[Dict[str, List[float]]]
    recommendations: List[str]
    analyzed_at: datetime


class ContractBenchmarker:
    """Contract benchmarking system against industry standards"""

    def __init__(self):
        """Initialize contract benchmarker"""
        # Benchmark standards storage
        self.benchmark_standards: Dict[str, BenchmarkStandard] = {}

        # Benchmarking history for trend analysis
        self.benchmarking_history: Dict[str, List[BenchmarkResult]] = defaultdict(list)

        # Industry benchmarks and statistics
        self.industry_statistics: Dict[str, Dict[str, Any]] = {}

        # Performance baselines
        self.performance_baselines = {
            "clause_completeness": {
                "excellent": 0.95,
                "good": 0.85,
                "acceptable": 0.75,
                "poor": 0.60,
            },
            "risk_score": {"low": 0.2, "medium": 0.5, "high": 0.7, "critical": 0.9},
            "compliance_score": {
                "excellent": 0.95,
                "good": 0.85,
                "acceptable": 0.75,
                "poor": 0.60,
            },
            "standardization": {"high": 0.9, "medium": 0.7, "low": 0.5, "custom": 0.3},
        }

    def add_benchmark_standard(self, benchmark_standard: BenchmarkStandard) -> bool:
        """Add a new benchmark standard"""
        try:
            self.benchmark_standards[benchmark_standard.benchmark_id] = (
                benchmark_standard
            )

            # Add clause embeddings for benchmark templates
            for clause_type, clause_data in benchmark_standard.clause_templates.items():
                if "embedding" in clause_data and "text" in clause_data:
                    clause_id = f"{benchmark_standard.benchmark_id}_{clause_type}"
                    embedding = np.array(clause_data["embedding"])

                    self.embedding_manager.add_clause_embedding(
                        clause_id=clause_id,
                        embedding=embedding,
                        clause_type=clause_type,
                        clause_text=clause_data["text"],
                        metadata={
                            "benchmark_id": benchmark_standard.benchmark_id,
                            "is_template": True,
                        },
                    )

            logger.info(f"Added benchmark standard: {benchmark_standard.name}")
            return True

        except Exception as e:
            logger.error(f"Failed to add benchmark standard: {e}")
            return False

    def benchmark_contract(
        self,
        contract_data: Dict[str, Any],
        benchmark_standard_id: str,
        industry: str = None,
    ) -> BenchmarkResult:
        """
        Benchmark contract against standard

        Args:
            contract_data: Contract data with clauses and embeddings
            benchmark_standard_id: ID of benchmark standard to use
            industry: Industry context for benchmarking

        Returns:
            Benchmarking result
        """
        try:
            current_time = datetime.now()
            contract_id = contract_data.get("contract_id", "unknown")

            if benchmark_standard_id not in self.benchmark_standards:
                logger.error(f"Benchmark standard not found: {benchmark_standard_id}")
                return self._create_default_benchmark_result(
                    contract_data, benchmark_standard_id
                )

            benchmark_standard = self.benchmark_standards[benchmark_standard_id]

            # Perform clause-level benchmarking
            clause_results = self._benchmark_clauses(contract_data, benchmark_standard)

            # Calculate overall compliance
            overall_compliance = self._calculate_overall_compliance(clause_results)

            # Calculate performance metrics
            performance_metrics = self._calculate_performance_metrics(
                contract_data, clause_results
            )

            # Identify gaps and deviations
            gaps = self._identify_benchmark_gaps(clause_results, benchmark_standard)

            # Generate improvement recommendations
            recommendations = self._generate_improvement_recommendations(
                gaps, performance_metrics
            )

            # Assess risks
            risk_assessment = self._assess_benchmark_risks(gaps, performance_metrics)

            # Calculate confidence
            confidence_score = self._calculate_benchmark_confidence(clause_results)

            result = BenchmarkResult(
                benchmark_id=str(uuid.uuid4()),
                contract_id=contract_id,
                benchmark_standard_id=benchmark_standard_id,
                overall_compliance=overall_compliance,
                confidence_score=confidence_score,
                benchmarked_at=current_time,
                clause_level_results=clause_results,
                performance_metrics=performance_metrics,
                identified_gaps=gaps,
                improvement_recommendations=recommendations,
                risk_assessment=risk_assessment,
                metadata={
                    "benchmark_name": benchmark_standard.name,
                    "industry": industry or benchmark_standard.industry,
                    "total_clauses_evaluated": len(clause_results),
                },
            )

            # Store in history
            self.benchmarking_history[contract_id].append(result)

            logger.info(
                f"Contract benchmarked: {contract_id} against {benchmark_standard.name} - {overall_compliance:.1%} compliance"
            )

            return result

        except Exception as e:
            logger.error(f"Contract benchmarking failed: {e}")
            return self._create_default_benchmark_result(
                contract_data, benchmark_standard_id
            )

    def _benchmark_clauses(
        self, contract_data: Dict[str, Any], benchmark_standard: BenchmarkStandard
    ) -> List[Dict[str, Any]]:
        """Benchmark individual clauses against standard templates"""
        try:
            clause_results = []
            contract_clauses = contract_data.get("clauses", [])

            for clause in contract_clauses:
                clause_id = clause.get("clause_id", "")
                clause_type = clause.get("clause_type", "")

                # Find matching benchmark template
                template_clause_id = f"{benchmark_standard.benchmark_id}_{clause_type}"

                if template_clause_id in self.embedding_manager.clause_embeddings:
                    # Calculate similarity to template
                    similarity = self.embedding_manager.calculate_clause_similarity(
                        clause_id, template_clause_id
                    )

                    if similarity is not None:
                        # Assess compliance level
                        compliance_level = self._assess_clause_compliance(
                            similarity, clause_type
                        )

                        # Check for required elements
                        required_elements = benchmark_standard.clause_templates.get(
                            clause_type, {}
                        ).get("required_elements", [])
                        element_compliance = self._check_required_elements(
                            clause, required_elements
                        )

                        clause_result = {
                            "clause_id": clause_id,
                            "clause_type": clause_type,
                            "similarity_to_template": similarity,
                            "compliance_level": compliance_level,
                            "element_compliance": element_compliance,
                            "overall_score": (
                                similarity * 0.7 + element_compliance * 0.3
                            ),
                            "gaps_identified": self._identify_clause_gaps(
                                clause,
                                benchmark_standard.clause_templates.get(
                                    clause_type, {}
                                ),
                            ),
                            "risk_level": self._assess_clause_benchmark_risk(
                                similarity, clause_type
                            ),
                        }
                        clause_results.append(clause_result)
                else:
                    # Clause type not in benchmark
                    clause_result = {
                        "clause_id": clause_id,
                        "clause_type": clause_type,
                        "similarity_to_template": 0.0,
                        "compliance_level": "non_standard",
                        "element_compliance": 0.0,
                        "overall_score": 0.0,
                        "gaps_identified": [
                            f"Clause type '{clause_type}' not in benchmark standard"
                        ],
                        "risk_level": "medium",
                    }
                    clause_results.append(clause_result)

            # Check for missing required clauses
            required_clause_types = set(benchmark_standard.clause_templates.keys())
            present_clause_types = set(
                clause.get("clause_type", "") for clause in contract_clauses
            )
            missing_clause_types = required_clause_types - present_clause_types

            for missing_type in missing_clause_types:
                clause_result = {
                    "clause_id": f"missing_{missing_type}",
                    "clause_type": missing_type,
                    "similarity_to_template": 0.0,
                    "compliance_level": "missing",
                    "element_compliance": 0.0,
                    "overall_score": 0.0,
                    "gaps_identified": [
                        f"Required clause type '{missing_type}' is missing"
                    ],
                    "risk_level": self._assess_missing_clause_risk(
                        missing_type, benchmark_standard
                    ),
                }
                clause_results.append(clause_result)

            return clause_results

        except Exception as e:
            logger.error(f"Clause benchmarking failed: {e}")
            return []

    def _assess_clause_compliance(self, similarity: float, clause_type: str) -> str:
        """Assess compliance level based on similarity"""
        if similarity >= 0.9:
            return "excellent"
        elif similarity >= 0.8:
            return "good"
        elif similarity >= 0.7:
            return "acceptable"
        elif similarity >= 0.5:
            return "needs_improvement"
        else:
            return "non_compliant"

    def _check_required_elements(
        self, clause: Dict[str, Any], required_elements: List[str]
    ) -> float:
        """Check if clause contains required elements"""
        try:
            if not required_elements:
                return 1.0

            clause_text = clause.get("text", "").lower()
            clause_entities = clause.get("entities", {})

            elements_found = 0

            for element in required_elements:
                element_lower = element.lower()

                # Check in text
                if element_lower in clause_text:
                    elements_found += 1
                # Check in extracted entities
                elif any(
                    element_lower in str(entity_list).lower()
                    for entity_list in clause_entities.values()
                ):
                    elements_found += 1

            return elements_found / len(required_elements)

        except Exception:
            return 0.0

    def _identify_clause_gaps(
        self, clause: Dict[str, Any], template: Dict[str, Any]
    ) -> List[str]:
        """Identify gaps in clause compared to template"""
        gaps = []

        try:
            # Check required elements
            required_elements = template.get("required_elements", [])
            clause_text = clause.get("text", "").lower()

            for element in required_elements:
                if element.lower() not in clause_text:
                    gaps.append(f"Missing required element: {element}")

            # Check minimum length requirement
            min_length = template.get("min_length", 0)
            if len(clause_text) < min_length:
                gaps.append(
                    f"Clause too short: {len(clause_text)} chars (minimum: {min_length})"
                )

            # Check for prohibited terms
            prohibited_terms = template.get("prohibited_terms", [])
            for term in prohibited_terms:
                if term.lower() in clause_text:
                    gaps.append(f"Contains prohibited term: {term}")

            return gaps

        except Exception:
            return ["Unable to identify gaps"]

    def _assess_clause_benchmark_risk(self, similarity: float, clause_type: str) -> str:
        """Assess risk level of clause deviation from benchmark"""
        high_risk_clauses = [
            "liability",
            "indemnification",
            "termination",
            "governing_law",
        ]
        medium_risk_clauses = ["payment", "delivery", "warranty", "confidentiality"]

        risk_multiplier = 1.0
        if any(risk_type in clause_type.lower() for risk_type in high_risk_clauses):
            risk_multiplier = 1.5
        elif any(risk_type in clause_type.lower() for risk_type in medium_risk_clauses):
            risk_multiplier = 1.2

        adjusted_risk_score = (1.0 - similarity) * risk_multiplier

        if adjusted_risk_score >= 0.8:
            return "critical"
        elif adjusted_risk_score >= 0.6:
            return "high"
        elif adjusted_risk_score >= 0.4:
            return "medium"
        else:
            return "low"

    def _assess_missing_clause_risk(
        self, clause_type: str, benchmark_standard: BenchmarkStandard
    ) -> str:
        """Assess risk of missing clause"""
        template = benchmark_standard.clause_templates.get(clause_type, {})
        importance = template.get("importance", "medium")

        importance_risk_map = {
            "critical": "critical",
            "high": "high",
            "medium": "medium",
            "low": "low",
        }

        return importance_risk_map.get(importance, "medium")

    def _calculate_overall_compliance(
        self, clause_results: List[Dict[str, Any]]
    ) -> float:
        """Calculate overall compliance score"""
        try:
            if not clause_results:
                return 0.0

            # Weight by clause importance and type
            total_weight = 0
            weighted_score = 0

            for result in clause_results:
                clause_type = result.get("clause_type", "")
                score = result.get("overall_score", 0)

                # Assign weights based on clause importance
                weight = self._get_clause_weight(clause_type)
                total_weight += weight
                weighted_score += score * weight

            return weighted_score / total_weight if total_weight > 0 else 0.0

        except Exception:
            return 0.0

    def _get_clause_weight(self, clause_type: str) -> float:
        """Get importance weight for clause type"""
        high_importance = ["liability", "termination", "governing_law", "payment"]
        medium_importance = [
            "warranty",
            "confidentiality",
            "delivery",
            "indemnification",
        ]

        if any(important in clause_type.lower() for important in high_importance):
            return 1.5
        elif any(important in clause_type.lower() for important in medium_importance):
            return 1.2
        else:
            return 1.0

    def _calculate_performance_metrics(
        self, contract_data: Dict[str, Any], clause_results: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """Calculate contract performance metrics"""
        try:
            metrics = {}

            # Clause completeness
            total_expected_clauses = len(
                [r for r in clause_results if not r["clause_id"].startswith("missing_")]
            )
            present_clauses = len(
                [r for r in clause_results if r["compliance_level"] != "missing"]
            )
            metrics["clause_completeness"] = present_clauses / max(
                1, total_expected_clauses
            )

            # Average compliance score
            compliance_scores = [
                r["overall_score"]
                for r in clause_results
                if r["compliance_level"] != "missing"
            ]
            metrics["compliance_score"] = (
                np.mean(compliance_scores) if compliance_scores else 0.0
            )

            # Risk score (inverse of compliance)
            risk_scores = []
            for result in clause_results:
                risk_level = result.get("risk_level", "low")
                risk_score_map = {
                    "low": 0.2,
                    "medium": 0.5,
                    "high": 0.7,
                    "critical": 0.9,
                }
                risk_scores.append(risk_score_map.get(risk_level, 0.5))

            metrics["risk_score"] = np.mean(risk_scores) if risk_scores else 0.5

            # Standardization level
            standardization_scores = [
                r["similarity_to_template"]
                for r in clause_results
                if r["similarity_to_template"] > 0
            ]
            metrics["standardization_level"] = (
                np.mean(standardization_scores) if standardization_scores else 0.0
            )

            return metrics

        except Exception:
            return {
                "clause_completeness": 0.5,
                "compliance_score": 0.5,
                "risk_score": 0.5,
                "standardization_level": 0.5,
            }

    def _identify_benchmark_gaps(
        self,
        clause_results: List[Dict[str, Any]],
        benchmark_standard: BenchmarkStandard,
    ) -> List[ContractDeviation]:
        """Identify gaps between contract and benchmark"""
        gaps = []

        try:
            for result in clause_results:
                clause_type = result.get("clause_type", "")
                compliance_level = result.get("compliance_level", "")
                overall_score = result.get("overall_score", 0)
                risk_level = result.get("risk_level", "low")

                if compliance_level == "missing":
                    gap = ContractDeviation(
                        deviation_id=str(uuid.uuid4()),
                        deviation_type=DeviationType.MISSING_CLAUSE,
                        clause_type=clause_type,
                        description=f"Missing required clause: {clause_type}",
                        severity=risk_level.upper(),
                        impact_score=0.8,
                        business_risk=0.7,
                        legal_risk=0.8,
                        financial_impact=None,
                        recommendations=[
                            f"Add {clause_type} clause per benchmark standard"
                        ],
                        evidence={"benchmark_gap": result},
                    )
                    gaps.append(gap)

                elif overall_score < 0.6:  # Significant deviation
                    gap = ContractDeviation(
                        deviation_id=str(uuid.uuid4()),
                        deviation_type=DeviationType.MODIFIED_CLAUSE,
                        clause_type=clause_type,
                        description=f"Clause deviates significantly from benchmark: {clause_type}",
                        severity="HIGH" if overall_score < 0.4 else "MEDIUM",
                        impact_score=1.0 - overall_score,
                        business_risk=0.6,
                        legal_risk=0.7,
                        financial_impact=None,
                        recommendations=result.get(
                            "gaps_identified", ["Review clause against benchmark"]
                        ),
                        evidence={"benchmark_gap": result},
                    )
                    gaps.append(gap)

            return gaps

        except Exception as e:
            logger.error(f"Gap identification failed: {e}")
            return []

    def _generate_improvement_recommendations(
        self, gaps: List[ContractDeviation], performance_metrics: Dict[str, float]
    ) -> List[str]:
        """Generate improvement recommendations"""
        recommendations = []

        try:
            # Address critical gaps first
            critical_gaps = [gap for gap in gaps if gap.severity == "CRITICAL"]
            if critical_gaps:
                recommendations.append(
                    f"Address {len(critical_gaps)} critical gaps immediately"
                )

            # Performance-based recommendations
            if performance_metrics.get("clause_completeness", 1) < 0.8:
                recommendations.append(
                    "Improve clause completeness - add missing required clauses"
                )

            if performance_metrics.get("standardization_level", 1) < 0.7:
                recommendations.append(
                    "Increase standardization - align clauses with benchmark templates"
                )

            if performance_metrics.get("risk_score", 0) > 0.6:
                recommendations.append(
                    "Review high-risk deviations and implement risk mitigation measures"
                )

            # Gap-specific recommendations
            missing_clauses = [
                gap
                for gap in gaps
                if gap.deviation_type == DeviationType.MISSING_CLAUSE
            ]
            if missing_clauses:
                recommendations.append(
                    "Implement clause library with required templates"
                )

            modified_clauses = [
                gap
                for gap in gaps
                if gap.deviation_type == DeviationType.MODIFIED_CLAUSE
            ]
            if len(modified_clauses) > 3:
                recommendations.append(
                    "Establish clause approval process for deviations from standard"
                )

            # General recommendations
            recommendations.append("Regular benchmarking against industry standards")
            recommendations.append(
                "Training on contract standardization best practices"
            )

            return recommendations[:10]  # Limit to top 10

        except Exception:
            return ["Comprehensive contract review recommended"]

    def _assess_benchmark_risks(
        self, gaps: List[ContractDeviation], performance_metrics: Dict[str, float]
    ) -> Dict[str, float]:
        """Assess risks from benchmark deviations"""
        try:
            risk_assessment = {
                "overall_risk": 0.0,
                "legal_risk": 0.0,
                "financial_risk": 0.0,
                "compliance_risk": 0.0,
                "operational_risk": 0.0,
            }

            # Aggregate gap risks
            if gaps:
                legal_risks = [gap.legal_risk for gap in gaps]
                business_risks = [gap.business_risk for gap in gaps]

                risk_assessment["legal_risk"] = min(1.0, np.mean(legal_risks) * 1.2)
                risk_assessment["financial_risk"] = min(1.0, np.mean(business_risks))
                risk_assessment["operational_risk"] = min(
                    1.0, np.mean(business_risks) * 0.8
                )

            # Performance-based risks
            compliance_score = performance_metrics.get("compliance_score", 1.0)

            risk_assessment["compliance_risk"] = 1.0 - compliance_score

            # Calculate overall risk
            risk_assessment["overall_risk"] = np.mean(
                [
                    risk_assessment["legal_risk"],
                    risk_assessment["financial_risk"],
                    risk_assessment["compliance_risk"],
                    risk_assessment["operational_risk"],
                ]
            )

            return risk_assessment

        except Exception:
            return {
                "overall_risk": 0.5,
                "legal_risk": 0.5,
                "financial_risk": 0.5,
                "compliance_risk": 0.5,
                "operational_risk": 0.5,
            }

    def _calculate_benchmark_confidence(
        self, clause_results: List[Dict[str, Any]]
    ) -> float:
        """Calculate confidence in benchmarking results"""
        try:
            if not clause_results:
                return 0.3

            # Factor in number of clauses evaluated
            coverage_factor = min(1.0, len(clause_results) / 10.0)

            # Factor in similarity scores (higher similarity = higher confidence)
            similarities = [
                r["similarity_to_template"]
                for r in clause_results
                if r["similarity_to_template"] > 0
            ]
            similarity_factor = np.mean(similarities) if similarities else 0.5

            # Factor in completeness
            missing_count = len(
                [r for r in clause_results if r["compliance_level"] == "missing"]
            )
            completeness_factor = (
                1.0 - (missing_count / len(clause_results)) if clause_results else 0.5
            )

            confidence = (
                coverage_factor * 0.3
                + similarity_factor * 0.4
                + completeness_factor * 0.3
            )

            return min(1.0, max(0.1, confidence))

        except Exception:
            return 0.5

    def _create_default_benchmark_result(
        self, contract_data: Dict[str, Any], benchmark_standard_id: str
    ) -> BenchmarkResult:
        """Create default benchmark result when processing fails"""
        return BenchmarkResult(
            benchmark_id=str(uuid.uuid4()),
            contract_id=contract_data.get("contract_id", "unknown"),
            benchmark_standard_id=benchmark_standard_id,
            overall_compliance=0.5,
            confidence_score=0.3,
            benchmarked_at=datetime.now(),
            clause_level_results=[],
            performance_metrics={
                "clause_completeness": 0.5,
                "compliance_score": 0.5,
                "risk_score": 0.5,
                "standardization_level": 0.5,
            },
            identified_gaps=[],
            improvement_recommendations=[
                "Manual benchmarking required",
                "Automated benchmarking failed",
            ],
            risk_assessment={
                "overall_risk": 0.5,
                "legal_risk": 0.5,
                "financial_risk": 0.5,
                "compliance_risk": 0.5,
                "operational_risk": 0.5,
            },
            metadata={"error": "Benchmarking processing failed"},
        )

    def get_benchmarking_statistics(self) -> Dict[str, Any]:
        """Get benchmarking system statistics"""
        try:
            stats = {
                "total_benchmarks": len(self.benchmark_standards),
                "total_contracts_benchmarked": len(self.benchmarking_history),
                "benchmark_types": {},
                "industry_coverage": {},
                "average_compliance": 0.0,
                "common_gaps": [],
            }

            # Benchmark type distribution
            for benchmark in self.benchmark_standards.values():
                benchmark_type = benchmark.benchmark_type.value
                stats["benchmark_types"][benchmark_type] = (
                    stats["benchmark_types"].get(benchmark_type, 0) + 1
                )

                industry = benchmark.industry
                stats["industry_coverage"][industry] = (
                    stats["industry_coverage"].get(industry, 0) + 1
                )

            # Average compliance across all benchmarked contracts
            all_results = []
            for contract_results in self.benchmarking_history.values():
                all_results.extend(contract_results)

            if all_results:
                avg_compliance = np.mean(
                    [result.overall_compliance for result in all_results]
                )
                stats["average_compliance"] = avg_compliance

                # Common gaps analysis
                gap_types = defaultdict(int)
                for result in all_results:
                    for gap in result.identified_gaps:
                        gap_types[gap.clause_type] += 1

                stats["common_gaps"] = sorted(
                    gap_types.items(), key=lambda x: x[1], reverse=True
                )[:5]

            return stats

        except Exception as e:
            logger.error(f"Failed to get benchmarking statistics: {e}")
            return {}


class DeviationAnalyzer:
    """Analyzes contract deviations and trends"""

    def __init__(self, contract_benchmarker: ContractBenchmarker):
        self.contract_benchmarker = contract_benchmarker
        self.deviation_history: Dict[str, List[DeviationAnalysis]] = defaultdict(list)

    def analyze_deviations(
        self, contract_id: str, analysis_type: str = "comprehensive"
    ) -> DeviationAnalysis:
        """
        Analyze deviations for contract across multiple benchmarks

        Args:
            contract_id: Contract to analyze
            analysis_type: Type of analysis (comprehensive, trend, gap_analysis)

        Returns:
            Deviation analysis result
        """
        try:
            current_time = datetime.now()

            # Get benchmarking history for contract
            benchmark_results = self.contract_benchmarker.benchmarking_history.get(
                contract_id, []
            )

            if not benchmark_results:
                return self._create_default_deviation_analysis(
                    contract_id, analysis_type
                )

            # Aggregate deviations across all benchmarks
            all_deviations = []
            for result in benchmark_results:
                all_deviations.extend(result.identified_gaps)

            # Categorize deviations
            deviation_categories = self._categorize_deviations(all_deviations)

            # Count critical deviations
            critical_deviations = len(
                [dev for dev in all_deviations if dev.severity == "CRITICAL"]
            )

            # Assess impact
            impact_assessment = self._assess_deviation_impact(all_deviations)

            # Trend analysis if historical data available
            trend_analysis = None
            if len(benchmark_results) > 1:
                trend_analysis = self._analyze_deviation_trends(benchmark_results)

            # Generate recommendations
            recommendations = self._generate_deviation_recommendations(
                all_deviations, deviation_categories
            )

            analysis = DeviationAnalysis(
                analysis_id=str(uuid.uuid4()),
                contract_id=contract_id,
                analysis_type=analysis_type,
                total_deviations=len(all_deviations),
                critical_deviations=critical_deviations,
                deviation_categories=deviation_categories,
                impact_assessment=impact_assessment,
                trend_analysis=trend_analysis,
                recommendations=recommendations,
                analyzed_at=current_time,
            )

            # Store in history
            self.deviation_history[contract_id].append(analysis)

            return analysis

        except Exception as e:
            logger.error(f"Deviation analysis failed: {e}")
            return self._create_default_deviation_analysis(contract_id, analysis_type)

    def _categorize_deviations(
        self, deviations: List[ContractDeviation]
    ) -> Dict[str, int]:
        """Categorize deviations by type and clause type"""
        categories = defaultdict(int)

        for deviation in deviations:
            categories[deviation.deviation_type.value] += 1
            categories[f"clause_{deviation.clause_type}"] += 1
            categories[f"severity_{deviation.severity.lower()}"] += 1

        return dict(categories)

    def _assess_deviation_impact(
        self, deviations: List[ContractDeviation]
    ) -> Dict[str, float]:
        """Assess overall impact of deviations"""
        if not deviations:
            return {
                "legal_impact": 0,
                "business_impact": 0,
                "financial_impact": 0,
                "compliance_impact": 0,
            }

        legal_impacts = [dev.legal_risk for dev in deviations]
        business_impacts = [dev.business_risk for dev in deviations]
        impact_scores = [dev.impact_score for dev in deviations]

        return {
            "legal_impact": np.mean(legal_impacts),
            "business_impact": np.mean(business_impacts),
            "financial_impact": np.mean(impact_scores),
            "compliance_impact": len(
                [dev for dev in deviations if dev.severity in ["HIGH", "CRITICAL"]]
            )
            / len(deviations),
        }

    def _analyze_deviation_trends(
        self, benchmark_results: List[BenchmarkResult]
    ) -> Dict[str, List[float]]:
        """Analyze trends in deviations over time"""
        trends = {
            "total_deviations": [],
            "critical_deviations": [],
            "compliance_scores": [],
            "risk_scores": [],
        }

        for result in sorted(benchmark_results, key=lambda x: x.benchmarked_at):
            trends["total_deviations"].append(len(result.identified_gaps))
            trends["critical_deviations"].append(
                len(
                    [
                        gap
                        for gap in result.identified_gaps
                        if gap.severity == "CRITICAL"
                    ]
                )
            )
            trends["compliance_scores"].append(result.overall_compliance)
            trends["risk_scores"].append(
                result.risk_assessment.get("overall_risk", 0.5)
            )

        return trends

    def _generate_deviation_recommendations(
        self, deviations: List[ContractDeviation], categories: Dict[str, int]
    ) -> List[str]:
        """Generate recommendations for addressing deviations"""
        recommendations = []

        # Critical deviations
        critical_count = categories.get("severity_critical", 0)
        if critical_count > 0:
            recommendations.append(
                f"Address {critical_count} critical deviations immediately"
            )

        # Common deviation types
        if categories.get("missing_clause", 0) > 3:
            recommendations.append(
                "Implement comprehensive clause library and templates"
            )

        if categories.get("modified_clause", 0) > 5:
            recommendations.append(
                "Establish deviation approval process and guidelines"
            )

        # Clause-specific recommendations
        high_frequency_clauses = [
            k.replace("clause_", "")
            for k, v in categories.items()
            if k.startswith("clause_") and v > 2
        ]
        if high_frequency_clauses:
            recommendations.append(
                f"Focus on standardizing {', '.join(high_frequency_clauses[:3])} clauses"
            )

        # Process improvements
        recommendations.append("Regular contract review and standardization training")
        recommendations.append("Implement automated deviation detection and alerts")

        return recommendations[:8]

    def _create_default_deviation_analysis(
        self, contract_id: str, analysis_type: str
    ) -> DeviationAnalysis:
        """Create default deviation analysis when processing fails"""
        return DeviationAnalysis(
            analysis_id=str(uuid.uuid4()),
            contract_id=contract_id,
            analysis_type=analysis_type,
            total_deviations=0,
            critical_deviations=0,
            deviation_categories={},
            impact_assessment={
                "legal_impact": 0.5,
                "business_impact": 0.5,
                "financial_impact": 0.5,
                "compliance_impact": 0.5,
            },
            trend_analysis=None,
            recommendations=[
                "Manual deviation analysis required",
                "Insufficient benchmarking data",
            ],
            analyzed_at=datetime.now(),
        )
