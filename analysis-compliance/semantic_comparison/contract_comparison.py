"""
Real-Time Document Intelligence & Compliance Engine
Semantic Contract Comparison System
Version: 1.0
Date: 2026-02-03

This module provides clause embedding-based contract comparison, similarity analysis,
deviation detection, and benchmarking using AI-extracted clause embeddings from Module 4.
"""

import logging
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Set
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import json
import uuid
from collections import defaultdict
import math
from scipy.spatial.distance import cosine, euclidean
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

logger = logging.getLogger(__name__)


class ComparisonType(Enum):
    """Types of contract comparisons"""

    CLAUSE_SIMILARITY = "clause_similarity"
    CONTRACT_SIMILARITY = "contract_similarity"
    DEVIATION_ANALYSIS = "deviation_analysis"
    BENCHMARK_ANALYSIS = "benchmark_analysis"
    RISK_COMPARISON = "risk_comparison"


class DeviationType(Enum):
    """Types of contract deviations"""

    MISSING_CLAUSE = "missing_clause"
    ADDITIONAL_CLAUSE = "additional_clause"
    MODIFIED_CLAUSE = "modified_clause"
    VALUE_DEVIATION = "value_deviation"
    TERM_DEVIATION = "term_deviation"


@dataclass
class ClauseComparison:
    """Individual clause comparison result"""

    clause_id_1: str
    clause_id_2: str
    clause_type_1: str
    clause_type_2: str
    similarity_score: float
    confidence_score: float
    embedding_distance: float
    semantic_alignment: float
    key_differences: List[str]
    risk_implications: List[str]
    comparison_metadata: Dict[str, Any]


@dataclass
class ContractDeviation:
    """Contract deviation from baseline or benchmark"""

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
class SemanticComparisonResult:
    """Comprehensive semantic comparison result"""

    comparison_id: str
    comparison_type: ComparisonType
    contract_id_1: str
    contract_id_2: Optional[str]  # None for benchmark comparisons
    overall_similarity: float
    confidence_score: float
    comparison_timestamp: datetime
    clause_comparisons: List[ClauseComparison]
    identified_deviations: List[ContractDeviation]
    risk_analysis: Dict[str, float]
    business_implications: List[str]
    legal_recommendations: List[str]
    metadata: Dict[str, Any]


class ClauseEmbeddingManager:
    """Manages clause embeddings and similarity calculations"""

    def __init__(self, embedding_dimension: int = 768):
        """
        Initialize clause embedding manager

        Args:
            embedding_dimension: Dimension of clause embeddings
        """
        self.embedding_dimension = embedding_dimension

        # Storage for clause embeddings
        self.clause_embeddings: Dict[str, np.ndarray] = {}
        self.clause_metadata: Dict[str, Dict[str, Any]] = {}

        # Clause type embeddings for semantic alignment
        self.clause_type_embeddings: Dict[str, np.ndarray] = {}

        # Similarity thresholds
        self.similarity_thresholds = {
            "high_similarity": 0.85,
            "moderate_similarity": 0.70,
            "low_similarity": 0.50,
            "dissimilar": 0.30,
        }

    def add_clause_embedding(
        self,
        clause_id: str,
        embedding: np.ndarray,
        clause_type: str,
        clause_text: str,
        metadata: Dict[str, Any] = None,
    ):
        """
        Add clause embedding to the system

        Args:
            clause_id: Unique identifier for clause
            embedding: Clause embedding vector
            clause_type: Type/category of clause
            clause_text: Original clause text
            metadata: Additional metadata
        """
        try:
            if embedding.shape[0] != self.embedding_dimension:
                logger.warning(
                    f"Embedding dimension mismatch: {embedding.shape[0]} vs {self.embedding_dimension}"
                )
                return False

            # Normalize embedding
            normalized_embedding = embedding / (np.linalg.norm(embedding) + 1e-8)

            self.clause_embeddings[clause_id] = normalized_embedding
            self.clause_metadata[clause_id] = {
                "clause_type": clause_type,
                "clause_text": clause_text,
                "added_at": datetime.now(),
                "metadata": metadata or {},
            }

            # Update clause type embedding (average of all clauses of this type)
            self._update_clause_type_embedding(clause_type, normalized_embedding)

            return True

        except Exception as e:
            logger.error(f"Failed to add clause embedding: {e}")
            return False

    def _update_clause_type_embedding(self, clause_type: str, embedding: np.ndarray):
        """Update the average embedding for a clause type"""
        try:
            if clause_type in self.clause_type_embeddings:
                # Update running average
                current_embedding = self.clause_type_embeddings[clause_type]
                # Simple update - in practice, you'd track count for proper averaging
                self.clause_type_embeddings[clause_type] = (
                    current_embedding + embedding
                ) / 2
            else:
                self.clause_type_embeddings[clause_type] = embedding.copy()

        except Exception as e:
            logger.warning(f"Failed to update clause type embedding: {e}")

    def calculate_clause_similarity(
        self, clause_id_1: str, clause_id_2: str
    ) -> Optional[float]:
        """Calculate semantic similarity between two clauses"""
        try:
            if (
                clause_id_1 not in self.clause_embeddings
                or clause_id_2 not in self.clause_embeddings
            ):
                return None

            emb1 = self.clause_embeddings[clause_id_1]
            emb2 = self.clause_embeddings[clause_id_2]

            # Cosine similarity
            similarity = np.dot(emb1, emb2) / (
                np.linalg.norm(emb1) * np.linalg.norm(emb2) + 1e-8
            )

            return max(0.0, min(1.0, similarity))  # Clamp to [0,1]

        except Exception as e:
            logger.error(f"Clause similarity calculation failed: {e}")
            return None

    def find_similar_clauses(
        self, clause_id: str, top_k: int = 10, clause_type_filter: str = None
    ) -> List[Tuple[str, float]]:
        """
        Find most similar clauses to given clause

        Args:
            clause_id: Reference clause ID
            top_k: Number of similar clauses to return
            clause_type_filter: Optional filter by clause type

        Returns:
            List of (clause_id, similarity_score) tuples
        """
        try:
            if clause_id not in self.clause_embeddings:
                return []

            reference_embedding = self.clause_embeddings[clause_id]
            similarities = []

            for other_id, other_embedding in self.clause_embeddings.items():
                if other_id == clause_id:
                    continue

                # Apply clause type filter if specified
                if clause_type_filter:
                    other_clause_type = self.clause_metadata[other_id]["clause_type"]
                    if other_clause_type != clause_type_filter:
                        continue

                similarity = np.dot(reference_embedding, other_embedding)
                similarities.append((other_id, similarity))

            # Sort by similarity and return top k
            similarities.sort(key=lambda x: x[1], reverse=True)
            return similarities[:top_k]

        except Exception as e:
            logger.error(f"Similar clauses search failed: {e}")
            return []

    def calculate_semantic_alignment(
        self, clause_id: str, expected_clause_type: str
    ) -> float:
        """Calculate how well a clause aligns with its expected type"""
        try:
            if clause_id not in self.clause_embeddings:
                return 0.0

            if expected_clause_type not in self.clause_type_embeddings:
                return 0.5  # Neutral alignment for unknown types

            clause_embedding = self.clause_embeddings[clause_id]
            type_embedding = self.clause_type_embeddings[expected_clause_type]

            alignment = np.dot(clause_embedding, type_embedding)
            return max(0.0, min(1.0, alignment))

        except Exception as e:
            logger.warning(f"Semantic alignment calculation failed: {e}")
            return 0.0

    def get_embedding_statistics(self) -> Dict[str, Any]:
        """Get statistics about stored embeddings"""
        try:
            stats = {
                "total_clauses": len(self.clause_embeddings),
                "clause_types": len(self.clause_type_embeddings),
                "clause_type_distribution": {},
                "embedding_dimension": self.embedding_dimension,
            }

            # Count clauses by type
            for clause_data in self.clause_metadata.values():
                clause_type = clause_data["clause_type"]
                stats["clause_type_distribution"][clause_type] = (
                    stats["clause_type_distribution"].get(clause_type, 0) + 1
                )

            return stats

        except Exception as e:
            logger.error(f"Failed to get embedding statistics: {e}")
            return {}


class ContractComparator:
    """Performs contract-level semantic comparisons"""

    def __init__(self, embedding_manager: ClauseEmbeddingManager):
        """
        Initialize contract comparator

        Args:
            embedding_manager: Clause embedding manager instance
        """
        self.embedding_manager = embedding_manager

        # Comparison weights for different aspects
        self.comparison_weights = {
            "semantic_similarity": 0.4,
            "structural_similarity": 0.3,
            "clause_coverage": 0.2,
            "value_alignment": 0.1,
        }

        # Deviation severity thresholds
        self.deviation_thresholds = {
            "critical": 0.9,
            "high": 0.7,
            "medium": 0.5,
            "low": 0.3,
        }

    def compare_contracts(
        self, contract_1: Dict[str, Any], contract_2: Dict[str, Any]
    ) -> SemanticComparisonResult:
        """
        Perform comprehensive contract comparison

        Args:
            contract_1: First contract data with clause embeddings
            contract_2: Second contract data with clause embeddings

        Returns:
            Comprehensive comparison result
        """
        try:
            comparison_id = str(uuid.uuid4())
            comparison_timestamp = datetime.now()

            contract_id_1 = contract_1.get("contract_id", "contract_1")
            contract_id_2 = contract_2.get("contract_id", "contract_2")

            # Extract clauses from contracts
            clauses_1 = contract_1.get("clauses", [])
            clauses_2 = contract_2.get("clauses", [])

            # Perform clause-level comparisons
            clause_comparisons = self._compare_clause_sets(clauses_1, clauses_2)

            # Calculate overall similarity
            overall_similarity = self._calculate_contract_similarity(
                clause_comparisons, clauses_1, clauses_2
            )

            # Identify deviations
            deviations = self._identify_contract_deviations(
                clause_comparisons, clauses_1, clauses_2
            )

            # Analyze risks
            risk_analysis = self._analyze_comparison_risks(
                clause_comparisons, deviations
            )

            # Generate business implications
            business_implications = self._generate_business_implications(
                deviations, risk_analysis
            )

            # Generate legal recommendations
            legal_recommendations = self._generate_legal_recommendations(
                deviations, clause_comparisons
            )

            # Calculate confidence
            confidence_score = self._calculate_comparison_confidence(clause_comparisons)

            return SemanticComparisonResult(
                comparison_id=comparison_id,
                comparison_type=ComparisonType.CONTRACT_SIMILARITY,
                contract_id_1=contract_id_1,
                contract_id_2=contract_id_2,
                overall_similarity=overall_similarity,
                confidence_score=confidence_score,
                comparison_timestamp=comparison_timestamp,
                clause_comparisons=clause_comparisons,
                identified_deviations=deviations,
                risk_analysis=risk_analysis,
                business_implications=business_implications,
                legal_recommendations=legal_recommendations,
                metadata={
                    "clauses_1_count": len(clauses_1),
                    "clauses_2_count": len(clauses_2),
                    "matched_clauses": len(
                        [c for c in clause_comparisons if c.similarity_score > 0.7]
                    ),
                    "comparison_weights": self.comparison_weights,
                },
            )

        except Exception as e:
            logger.error(f"Contract comparison failed: {e}")
            return self._create_default_comparison_result(contract_1, contract_2)

    def _compare_clause_sets(
        self, clauses_1: List[Dict[str, Any]], clauses_2: List[Dict[str, Any]]
    ) -> List[ClauseComparison]:
        """Compare two sets of clauses and find best matches"""
        try:
            clause_comparisons = []

            # Create similarity matrix
            similarity_matrix = np.zeros((len(clauses_1), len(clauses_2)))

            for i, clause_1 in enumerate(clauses_1):
                for j, clause_2 in enumerate(clauses_2):
                    clause_id_1 = clause_1.get("clause_id", f"clause_1_{i}")
                    clause_id_2 = clause_2.get("clause_id", f"clause_2_{j}")

                    similarity = self.embedding_manager.calculate_clause_similarity(
                        clause_id_1, clause_id_2
                    )
                    if similarity is not None:
                        similarity_matrix[i, j] = similarity

            # Find optimal matching using greedy approach
            used_j = set()

            for i, clause_1 in enumerate(clauses_1):
                best_j = -1
                best_similarity = 0.0

                # Find best match for clause_1[i]
                for j in range(len(clauses_2)):
                    if j in used_j:
                        continue

                    if similarity_matrix[i, j] > best_similarity:
                        best_similarity = similarity_matrix[i, j]
                        best_j = j

                # Create comparison if we found a reasonable match
                if (
                    best_j >= 0 and best_similarity > 0.3
                ):  # Minimum similarity threshold
                    clause_2 = clauses_2[best_j]
                    used_j.add(best_j)

                    comparison = self._create_clause_comparison(
                        clause_1, clause_2, best_similarity
                    )
                    clause_comparisons.append(comparison)

            return clause_comparisons

        except Exception as e:
            logger.error(f"Clause set comparison failed: {e}")
            return []

    def _create_clause_comparison(
        self, clause_1: Dict[str, Any], clause_2: Dict[str, Any], similarity: float
    ) -> ClauseComparison:
        """Create detailed clause comparison"""
        try:
            clause_id_1 = clause_1.get("clause_id", "unknown")
            clause_id_2 = clause_2.get("clause_id", "unknown")
            clause_type_1 = clause_1.get("clause_type", "unknown")
            clause_type_2 = clause_2.get("clause_type", "unknown")

            # Calculate embedding distance
            emb1 = self.embedding_manager.clause_embeddings.get(clause_id_1)
            emb2 = self.embedding_manager.clause_embeddings.get(clause_id_2)

            embedding_distance = 0.0
            if emb1 is not None and emb2 is not None:
                embedding_distance = euclidean(emb1, emb2)

            # Calculate semantic alignment
            semantic_alignment_1 = self.embedding_manager.calculate_semantic_alignment(
                clause_id_1, clause_type_1
            )
            semantic_alignment_2 = self.embedding_manager.calculate_semantic_alignment(
                clause_id_2, clause_type_2
            )
            semantic_alignment = (semantic_alignment_1 + semantic_alignment_2) / 2

            # Identify key differences
            key_differences = self._identify_clause_differences(clause_1, clause_2)

            # Assess risk implications
            risk_implications = self._assess_clause_risk_implications(
                clause_1, clause_2, similarity
            )

            # Calculate confidence
            confidence_score = min(1.0, similarity + semantic_alignment * 0.2)

            return ClauseComparison(
                clause_id_1=clause_id_1,
                clause_id_2=clause_id_2,
                clause_type_1=clause_type_1,
                clause_type_2=clause_type_2,
                similarity_score=similarity,
                confidence_score=confidence_score,
                embedding_distance=embedding_distance,
                semantic_alignment=semantic_alignment,
                key_differences=key_differences,
                risk_implications=risk_implications,
                comparison_metadata={
                    "clause_1_length": len(clause_1.get("text", "")),
                    "clause_2_length": len(clause_2.get("text", "")),
                    "type_match": clause_type_1 == clause_type_2,
                },
            )

        except Exception as e:
            logger.error(f"Clause comparison creation failed: {e}")
            return ClauseComparison(
                clause_id_1="error",
                clause_id_2="error",
                clause_type_1="error",
                clause_type_2="error",
                similarity_score=0.0,
                confidence_score=0.0,
                embedding_distance=999.0,
                semantic_alignment=0.0,
                key_differences=["Comparison failed"],
                risk_implications=["Unable to assess risk"],
                comparison_metadata={},
            )

    def _identify_clause_differences(
        self, clause_1: Dict[str, Any], clause_2: Dict[str, Any]
    ) -> List[str]:
        """Identify key differences between clauses"""
        differences = []

        try:
            # Check for type differences
            type_1 = clause_1.get("clause_type", "")
            type_2 = clause_2.get("clause_type", "")
            if type_1 != type_2:
                differences.append(f"Clause type mismatch: {type_1} vs {type_2}")

            # Check for significant length differences
            text_1 = clause_1.get("text", "")
            text_2 = clause_2.get("text", "")
            length_ratio = len(text_2) / (len(text_1) + 1)
            if length_ratio > 1.5 or length_ratio < 0.67:
                differences.append(
                    f"Significant length difference: {len(text_1)} vs {len(text_2)} characters"
                )

            # Check for entity differences
            entities_1 = set(clause_1.get("entities", {}).keys())
            entities_2 = set(clause_2.get("entities", {}).keys())

            missing_entities = entities_1 - entities_2
            additional_entities = entities_2 - entities_1

            if missing_entities:
                differences.append(
                    f"Missing entities in clause 2: {', '.join(missing_entities)}"
                )
            if additional_entities:
                differences.append(
                    f"Additional entities in clause 2: {', '.join(additional_entities)}"
                )

            # Check for value differences in monetary amounts
            values_1 = clause_1.get("extracted_values", {})
            values_2 = clause_2.get("extracted_values", {})

            for key in values_1.keys() & values_2.keys():
                val_1 = values_1[key]
                val_2 = values_2[key]
                if isinstance(val_1, (int, float)) and isinstance(val_2, (int, float)):
                    if abs(val_1 - val_2) / (abs(val_1) + 1) > 0.1:  # 10% difference
                        differences.append(
                            f"Value difference in {key}: {val_1} vs {val_2}"
                        )

            return differences

        except Exception as e:
            logger.warning(f"Difference identification failed: {e}")
            return ["Unable to identify differences"]

    def _assess_clause_risk_implications(
        self, clause_1: Dict[str, Any], clause_2: Dict[str, Any], similarity: float
    ) -> List[str]:
        """Assess risk implications of clause differences"""
        risk_implications = []

        try:
            clause_type = clause_1.get("clause_type", "").lower()

            # High-risk clause types with low similarity
            if similarity < 0.7:
                high_risk_types = [
                    "liability",
                    "termination",
                    "penalty",
                    "indemnification",
                    "warranty",
                ]
                if any(risk_type in clause_type for risk_type in high_risk_types):
                    risk_implications.append(
                        f"High risk: Significant differences in {clause_type} clause"
                    )

                # Financial implications
                if "payment" in clause_type or "price" in clause_type:
                    risk_implications.append(
                        "Financial risk: Payment terms differ significantly"
                    )

                # Legal implications
                if "governing" in clause_type or "jurisdiction" in clause_type:
                    risk_implications.append(
                        "Legal risk: Governing law or jurisdiction differences"
                    )

                # Compliance implications
                if "compliance" in clause_type or "regulatory" in clause_type:
                    risk_implications.append(
                        "Compliance risk: Regulatory requirements may differ"
                    )

            # Check for missing critical clauses
            if similarity < 0.3:
                risk_implications.append(
                    "Critical: Clauses may be fundamentally different or missing"
                )

            return risk_implications

        except Exception:
            return ["Unable to assess risk implications"]

    def _calculate_contract_similarity(
        self,
        clause_comparisons: List[ClauseComparison],
        clauses_1: List[Dict[str, Any]],
        clauses_2: List[Dict[str, Any]],
    ) -> float:
        """Calculate overall contract similarity"""
        try:
            if not clause_comparisons:
                return 0.0

            # Semantic similarity component
            avg_semantic_similarity = np.mean(
                [comp.similarity_score for comp in clause_comparisons]
            )

            # Structural similarity component (clause coverage)
            total_clauses = len(clauses_1) + len(clauses_2)
            matched_clauses = len(clause_comparisons)
            structural_similarity = (
                (2 * matched_clauses) / total_clauses if total_clauses > 0 else 0
            )

            # Clause type coverage
            types_1 = set(clause.get("clause_type", "") for clause in clauses_1)
            types_2 = set(clause.get("clause_type", "") for clause in clauses_2)

            common_types = types_1 & types_2
            total_types = types_1 | types_2
            clause_coverage = len(common_types) / len(total_types) if total_types else 0

            # Weight the components
            weights = self.comparison_weights
            overall_similarity = (
                avg_semantic_similarity * weights["semantic_similarity"]
                + structural_similarity * weights["structural_similarity"]
                + clause_coverage * weights["clause_coverage"]
                + 0.8 * weights["value_alignment"]  # Default value alignment
            )

            return min(1.0, max(0.0, overall_similarity))

        except Exception as e:
            logger.error(f"Contract similarity calculation failed: {e}")
            return 0.0

    def _identify_contract_deviations(
        self,
        clause_comparisons: List[ClauseComparison],
        clauses_1: List[Dict[str, Any]],
        clauses_2: List[Dict[str, Any]],
    ) -> List[ContractDeviation]:
        """Identify deviations between contracts"""
        deviations = []

        try:
            # Find matched clause IDs
            matched_clause_ids_1 = set(comp.clause_id_1 for comp in clause_comparisons)
            matched_clause_ids_2 = set(comp.clause_id_2 for comp in clause_comparisons)

            # Missing clauses in contract 2
            for clause_1 in clauses_1:
                clause_id_1 = clause_1.get("clause_id", "")
                if clause_id_1 not in matched_clause_ids_1:
                    deviation = ContractDeviation(
                        deviation_id=str(uuid.uuid4()),
                        deviation_type=DeviationType.MISSING_CLAUSE,
                        clause_type=clause_1.get("clause_type", "unknown"),
                        description=f"Clause missing in contract 2: {clause_1.get('clause_type', 'unknown')}",
                        severity=self._assess_missing_clause_severity(clause_1),
                        impact_score=0.7,
                        business_risk=0.6,
                        legal_risk=0.8,
                        financial_impact=None,
                        recommendations=["Review need for this clause in contract 2"],
                        evidence={"missing_clause": clause_1},
                    )
                    deviations.append(deviation)

            # Additional clauses in contract 2
            for clause_2 in clauses_2:
                clause_id_2 = clause_2.get("clause_id", "")
                if clause_id_2 not in matched_clause_ids_2:
                    deviation = ContractDeviation(
                        deviation_id=str(uuid.uuid4()),
                        deviation_type=DeviationType.ADDITIONAL_CLAUSE,
                        clause_type=clause_2.get("clause_type", "unknown"),
                        description=f"Additional clause in contract 2: {clause_2.get('clause_type', 'unknown')}",
                        severity="MEDIUM",
                        impact_score=0.5,
                        business_risk=0.4,
                        legal_risk=0.6,
                        financial_impact=None,
                        recommendations=[
                            "Review purpose and necessity of additional clause"
                        ],
                        evidence={"additional_clause": clause_2},
                    )
                    deviations.append(deviation)

            # Modified clauses (low similarity)
            for comparison in clause_comparisons:
                if comparison.similarity_score < 0.5:
                    severity = "HIGH" if comparison.similarity_score < 0.3 else "MEDIUM"
                    deviation = ContractDeviation(
                        deviation_id=str(uuid.uuid4()),
                        deviation_type=DeviationType.MODIFIED_CLAUSE,
                        clause_type=comparison.clause_type_1,
                        description=f"Significant modification in {comparison.clause_type_1} clause",
                        severity=severity,
                        impact_score=1.0 - comparison.similarity_score,
                        business_risk=0.8 if severity == "HIGH" else 0.5,
                        legal_risk=0.9 if severity == "HIGH" else 0.6,
                        financial_impact=None,
                        recommendations=comparison.risk_implications,
                        evidence={"comparison": comparison},
                    )
                    deviations.append(deviation)

            return deviations

        except Exception as e:
            logger.error(f"Contract deviation identification failed: {e}")
            return []

    def _assess_missing_clause_severity(self, clause: Dict[str, Any]) -> str:
        """Assess severity of missing clause"""
        clause_type = clause.get("clause_type", "").lower()

        critical_clauses = [
            "liability",
            "termination",
            "indemnification",
            "governing_law",
        ]
        high_importance_clauses = ["payment", "delivery", "warranty", "confidentiality"]

        if any(critical in clause_type for critical in critical_clauses):
            return "CRITICAL"
        elif any(important in clause_type for important in high_importance_clauses):
            return "HIGH"
        else:
            return "MEDIUM"

    def _analyze_comparison_risks(
        self,
        clause_comparisons: List[ClauseComparison],
        deviations: List[ContractDeviation],
    ) -> Dict[str, float]:
        """Analyze overall risks from contract comparison"""
        try:
            risk_analysis = {
                "overall_risk": 0.0,
                "legal_risk": 0.0,
                "financial_risk": 0.0,
                "operational_risk": 0.0,
                "compliance_risk": 0.0,
            }

            # Aggregate risks from deviations
            if deviations:
                legal_risks = [dev.legal_risk for dev in deviations]
                business_risks = [dev.business_risk for dev in deviations]

                risk_analysis["legal_risk"] = min(1.0, np.mean(legal_risks) * 1.2)
                risk_analysis["financial_risk"] = min(1.0, np.mean(business_risks))
                risk_analysis["operational_risk"] = min(
                    1.0, np.mean(business_risks) * 0.8
                )

            # Consider low-similarity clauses
            low_similarity_comparisons = [
                comp for comp in clause_comparisons if comp.similarity_score < 0.6
            ]
            if low_similarity_comparisons:
                similarity_risk = 1.0 - np.mean(
                    [comp.similarity_score for comp in low_similarity_comparisons]
                )
                risk_analysis["compliance_risk"] = min(1.0, similarity_risk)

            # Calculate overall risk
            risk_analysis["overall_risk"] = np.mean(
                [
                    risk_analysis["legal_risk"],
                    risk_analysis["financial_risk"],
                    risk_analysis["operational_risk"],
                    risk_analysis["compliance_risk"],
                ]
            )

            return risk_analysis

        except Exception as e:
            logger.error(f"Risk analysis failed: {e}")
            return {
                "overall_risk": 0.5,
                "legal_risk": 0.5,
                "financial_risk": 0.5,
                "operational_risk": 0.5,
                "compliance_risk": 0.5,
            }

    def _generate_business_implications(
        self, deviations: List[ContractDeviation], risk_analysis: Dict[str, float]
    ) -> List[str]:
        """Generate business implications from comparison"""
        implications = []

        try:
            # High-level risk implications
            if risk_analysis.get("overall_risk", 0) > 0.7:
                implications.append(
                    "High-risk contract differences detected - detailed legal review recommended"
                )

            # Specific deviation implications
            critical_deviations = [
                dev for dev in deviations if dev.severity == "CRITICAL"
            ]
            if critical_deviations:
                implications.append(
                    f"{len(critical_deviations)} critical deviations found - immediate attention required"
                )

            missing_clauses = [
                dev
                for dev in deviations
                if dev.deviation_type == DeviationType.MISSING_CLAUSE
            ]
            if missing_clauses:
                implications.append(
                    f"Missing clauses may create legal gaps and exposure"
                )

            modified_clauses = [
                dev
                for dev in deviations
                if dev.deviation_type == DeviationType.MODIFIED_CLAUSE
            ]
            if modified_clauses:
                implications.append(
                    "Modified clauses may change risk profile and obligations"
                )

            # Financial implications
            if risk_analysis.get("financial_risk", 0) > 0.6:
                implications.append(
                    "Contract differences may have significant financial implications"
                )

            return implications[:10]  # Limit to top 10

        except Exception:
            return ["Unable to assess business implications"]

    def _generate_legal_recommendations(
        self,
        deviations: List[ContractDeviation],
        clause_comparisons: List[ClauseComparison],
    ) -> List[str]:
        """Generate legal recommendations from comparison"""
        recommendations = []

        try:
            # Critical deviations
            critical_deviations = [
                dev for dev in deviations if dev.severity == "CRITICAL"
            ]
            if critical_deviations:
                recommendations.append(
                    "Engage legal counsel for critical clause deviations"
                )

            # Low similarity clauses
            low_similarity = [
                comp for comp in clause_comparisons if comp.similarity_score < 0.4
            ]
            if low_similarity:
                recommendations.append(
                    "Legal review required for substantially different clauses"
                )

            # Missing high-risk clauses
            missing_critical = [
                dev
                for dev in deviations
                if dev.deviation_type == DeviationType.MISSING_CLAUSE
                and dev.severity in ["CRITICAL", "HIGH"]
            ]
            if missing_critical:
                recommendations.append(
                    "Add missing critical clauses to ensure legal protection"
                )

            # General recommendations
            if len(deviations) > 5:
                recommendations.append(
                    "Comprehensive legal review recommended due to multiple deviations"
                )

            recommendations.append(
                "Document all approved deviations with business justification"
            )
            recommendations.append(
                "Establish deviation approval process for future contracts"
            )

            return recommendations[:8]  # Limit to top 8

        except Exception:
            return ["Legal review recommended for contract differences"]

    def _calculate_comparison_confidence(
        self, clause_comparisons: List[ClauseComparison]
    ) -> float:
        """Calculate confidence in comparison results"""
        try:
            if not clause_comparisons:
                return 0.3

            # Average confidence from clause comparisons
            clause_confidences = [comp.confidence_score for comp in clause_comparisons]
            avg_confidence = np.mean(clause_confidences)

            # Adjust for number of comparisons (more comparisons = higher confidence)
            comparison_factor = min(1.0, len(clause_comparisons) / 10.0)

            # Adjust for semantic alignment
            alignment_scores = [comp.semantic_alignment for comp in clause_comparisons]
            avg_alignment = np.mean(alignment_scores) if alignment_scores else 0.5

            overall_confidence = (
                avg_confidence * 0.6 + comparison_factor * 0.2 + avg_alignment * 0.2
            )

            return min(1.0, max(0.1, overall_confidence))

        except Exception:
            return 0.5

    def _create_default_comparison_result(
        self, contract_1: Dict[str, Any], contract_2: Dict[str, Any]
    ) -> SemanticComparisonResult:
        """Create default comparison result when processing fails"""
        return SemanticComparisonResult(
            comparison_id=str(uuid.uuid4()),
            comparison_type=ComparisonType.CONTRACT_SIMILARITY,
            contract_id_1=contract_1.get("contract_id", "unknown"),
            contract_id_2=contract_2.get("contract_id", "unknown"),
            overall_similarity=0.5,
            confidence_score=0.3,
            comparison_timestamp=datetime.now(),
            clause_comparisons=[],
            identified_deviations=[],
            risk_analysis={
                "overall_risk": 0.5,
                "legal_risk": 0.5,
                "financial_risk": 0.5,
                "operational_risk": 0.5,
                "compliance_risk": 0.5,
            },
            business_implications=[
                "Manual comparison required",
                "Automated comparison failed",
            ],
            legal_recommendations=[
                "Legal review recommended",
                "Automated analysis unavailable",
            ],
            metadata={"error": "Comparison processing failed"},
        )
