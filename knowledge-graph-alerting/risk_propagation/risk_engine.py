"""
Real-Time Document Intelligence & Compliance Engine
Risk Propagation Engine
Version: 1.0
Date: 2026-02-03

This module implements comprehensive risk propagation through entity relationships
in the knowledge graph, calculating cascading risk impacts and identifying risk clusters.
"""

import logging
from typing import Dict, Any, List, Optional, Set, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import json
import uuid
import math
from concurrent.futures import ThreadPoolExecutor
import threading
import heapq
from collections import defaultdict, deque

try:
    import numpy as np

    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

logger = logging.getLogger(__name__)


class RiskType(Enum):
    """Types of risks in the system"""

    COMPLIANCE_VIOLATION = "compliance_violation"
    FINANCIAL_RISK = "financial_risk"
    OPERATIONAL_RISK = "operational_risk"
    REPUTATION_RISK = "reputation_risk"
    LEGAL_RISK = "legal_risk"
    MARKET_RISK = "market_risk"
    CREDIT_RISK = "credit_risk"
    LIQUIDITY_RISK = "liquidity_risk"
    CYBERSECURITY_RISK = "cybersecurity_risk"
    REGULATORY_RISK = "regulatory_risk"


class RiskSeverity(Enum):
    """Risk severity levels"""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    NEGLIGIBLE = "negligible"


class PropagationDirection(Enum):
    """Risk propagation directions"""

    FORWARD = "forward"  # Following relationship direction
    BACKWARD = "backward"  # Against relationship direction
    BIDIRECTIONAL = "bidirectional"  # Both directions


@dataclass
class RiskEvent:
    """Represents a risk event in the system"""

    risk_id: str
    source_entity_id: str
    risk_type: RiskType
    severity: RiskSeverity
    description: str
    impact_score: float  # 0.0 to 1.0
    probability: float  # 0.0 to 1.0
    risk_score: float  # Combined impact * probability
    metadata: Dict[str, Any]
    detected_at: datetime
    created_at: datetime

    def __post_init__(self):
        """Calculate risk score if not provided"""
        if self.risk_score == 0:
            self.risk_score = self.impact_score * self.probability


@dataclass
class RiskPropagation:
    """Represents risk propagation along a relationship"""

    propagation_id: str
    source_risk_id: str
    source_entity_id: str
    target_entity_id: str
    relationship_type: str
    propagated_risk_score: float
    attenuation_factor: float  # How much risk is reduced during propagation
    propagation_path: List[str]  # Path of entity IDs
    confidence: float
    created_at: datetime


@dataclass
class RiskCluster:
    """Represents a cluster of related risks"""

    cluster_id: str
    entity_ids: Set[str]
    risk_events: List[RiskEvent]
    total_risk_score: float
    dominant_risk_type: RiskType
    cluster_size: int
    created_at: datetime
    updated_at: datetime


class RiskPropagationEngine:
    """Comprehensive risk propagation and analysis engine"""

    def __init__(
        self,
        neo4j_manager=None,
        max_propagation_depth: int = 5,
        default_attenuation: float = 0.8,
        risk_threshold: float = 0.1,
    ):
        """
        Initialize risk propagation engine

        Args:
            neo4j_manager: Neo4j graph manager instance
            max_propagation_depth: Maximum depth for risk propagation
            default_attenuation: Default risk attenuation factor
            risk_threshold: Minimum risk score to consider
        """
        self.neo4j_manager = neo4j_manager
        self.max_propagation_depth = max_propagation_depth
        self.default_attenuation = default_attenuation
        self.risk_threshold = risk_threshold

        # Thread safety
        self._lock = threading.RLock()
        self._executor = ThreadPoolExecutor(max_workers=8)

        # Risk storage
        self.risk_events: Dict[str, RiskEvent] = {}
        self.propagations: Dict[str, List[RiskPropagation]] = defaultdict(list)
        self.risk_clusters: Dict[str, RiskCluster] = {}

        # Relationship-specific attenuation factors
        self.relationship_attenuations = {
            "OWNS": 0.9,
            "CONTROLS": 0.9,
            "EMPLOYED_BY": 0.7,
            "SUBSIDIARY_OF": 0.85,
            "PARTNER_WITH": 0.6,
            "PARTY_TO": 0.8,
            "SIGNATORY": 0.75,
            "BENEFICIARY": 0.7,
            "GUARANTOR": 0.95,
            "RELATED_TO": 0.5,
            "MENTIONS": 0.3,
            "REFERENCES": 0.4,
            "CONTAINS": 0.2,
        }

        # Risk type propagation rules
        self.propagation_rules = {
            RiskType.COMPLIANCE_VIOLATION: {
                "OWNS": 0.9,
                "CONTROLS": 0.95,
                "EMPLOYED_BY": 0.7,
                "SUBSIDIARY_OF": 0.85,
                "PARTY_TO": 0.8,
            },
            RiskType.FINANCIAL_RISK: {
                "OWNS": 0.95,
                "CONTROLS": 0.9,
                "GUARANTOR": 0.98,
                "SUBSIDIARY_OF": 0.8,
                "PARTNER_WITH": 0.6,
            },
            RiskType.OPERATIONAL_RISK: {
                "EMPLOYED_BY": 0.8,
                "CONTROLS": 0.7,
                "SUBSIDIARY_OF": 0.75,
                "PARTNER_WITH": 0.5,
            },
            RiskType.REPUTATION_RISK: {
                "OWNS": 0.8,
                "EMPLOYED_BY": 0.6,
                "SUBSIDIARY_OF": 0.7,
                "PARTNER_WITH": 0.65,
                "RELATED_TO": 0.4,
            },
        }

        # Statistics
        self.stats = {
            "risks_processed": 0,
            "propagations_calculated": 0,
            "clusters_identified": 0,
            "total_entities_analyzed": 0,
        }

    def register_risk_event(self, risk_event: RiskEvent) -> bool:
        """
        Register a new risk event in the system

        Args:
            risk_event: Risk event to register

        Returns:
            True if successful
        """
        try:
            with self._lock:
                self.risk_events[risk_event.risk_id] = risk_event
                self.stats["risks_processed"] += 1

            logger.info(
                f"Registered risk event: {risk_event.risk_id} ({risk_event.risk_type.value})"
            )
            return True

        except Exception as e:
            logger.error(f"Failed to register risk event: {e}")
            return False

    def propagate_risk(
        self,
        risk_id: str,
        max_depth: int = None,
        direction: PropagationDirection = PropagationDirection.FORWARD,
    ) -> List[RiskPropagation]:
        """
        Propagate risk through the knowledge graph

        Args:
            risk_id: ID of risk event to propagate
            max_depth: Maximum propagation depth
            direction: Propagation direction

        Returns:
            List of risk propagations
        """
        try:
            if risk_id not in self.risk_events:
                logger.warning(f"Risk event not found: {risk_id}")
                return []

            risk_event = self.risk_events[risk_id]
            max_depth = max_depth or self.max_propagation_depth

            propagations = []
            visited = set()

            # Initialize propagation queue with (entity_id, risk_score, depth, path)
            queue = deque(
                [
                    (
                        risk_event.source_entity_id,
                        risk_event.risk_score,
                        0,
                        [risk_event.source_entity_id],
                    )
                ]
            )
            visited.add(risk_event.source_entity_id)

            while queue and len(propagations) < 1000:  # Limit to prevent infinite loops
                current_entity, current_risk_score, depth, path = queue.popleft()

                if depth >= max_depth or current_risk_score < self.risk_threshold:
                    continue

                # Get relationships for current entity
                relationships = self._get_entity_relationships(
                    current_entity, direction
                )

                for relationship in relationships:
                    target_entity = relationship.get("target_id") or relationship.get(
                        "other_id"
                    )
                    rel_type = relationship.get("type", "RELATED_TO")

                    if target_entity and target_entity not in visited:
                        # Calculate propagated risk score
                        propagated_score = self._calculate_propagated_risk(
                            current_risk_score, risk_event.risk_type, rel_type, depth
                        )

                        if propagated_score >= self.risk_threshold:
                            # Create propagation record
                            propagation = RiskPropagation(
                                propagation_id=str(uuid.uuid4()),
                                source_risk_id=risk_id,
                                source_entity_id=current_entity,
                                target_entity_id=target_entity,
                                relationship_type=rel_type,
                                propagated_risk_score=propagated_score,
                                attenuation_factor=self._get_attenuation_factor(
                                    risk_event.risk_type, rel_type
                                ),
                                propagation_path=path + [target_entity],
                                confidence=relationship.get("confidence", 1.0),
                                created_at=datetime.now(),
                            )

                            propagations.append(propagation)

                            # Add to queue for further propagation
                            queue.append(
                                (
                                    target_entity,
                                    propagated_score,
                                    depth + 1,
                                    path + [target_entity],
                                )
                            )
                            visited.add(target_entity)

            # Store propagations
            with self._lock:
                self.propagations[risk_id].extend(propagations)
                self.stats["propagations_calculated"] += len(propagations)

            logger.info(
                f"Propagated risk {risk_id}: {len(propagations)} propagations found"
            )
            return propagations

        except Exception as e:
            logger.error(f"Risk propagation failed for {risk_id}: {e}")
            return []

    def _get_entity_relationships(
        self, entity_id: str, direction: PropagationDirection
    ) -> List[Dict[str, Any]]:
        """Get relationships for an entity from the knowledge graph"""
        try:
            if not self.neo4j_manager:
                return []

            if direction == PropagationDirection.FORWARD:
                return self.neo4j_manager.get_relationships(entity_id, "outgoing")
            elif direction == PropagationDirection.BACKWARD:
                return self.neo4j_manager.get_relationships(entity_id, "incoming")
            else:  # BIDIRECTIONAL
                return self.neo4j_manager.get_relationships(entity_id, "both")

        except Exception as e:
            logger.error(f"Failed to get relationships for {entity_id}: {e}")
            return []

    def _calculate_propagated_risk(
        self,
        base_risk_score: float,
        risk_type: RiskType,
        relationship_type: str,
        depth: int,
    ) -> float:
        """Calculate propagated risk score"""
        try:
            # Get base attenuation factor
            attenuation = self._get_attenuation_factor(risk_type, relationship_type)

            # Apply depth decay
            depth_factor = math.pow(0.9, depth)

            # Calculate propagated score
            propagated_score = base_risk_score * attenuation * depth_factor

            return max(0.0, min(1.0, propagated_score))

        except Exception as e:
            logger.error(f"Risk calculation failed: {e}")
            return 0.0

    def _get_attenuation_factor(
        self, risk_type: RiskType, relationship_type: str
    ) -> float:
        """Get attenuation factor for risk type and relationship"""
        try:
            # Check risk-specific rules first
            if risk_type in self.propagation_rules:
                type_rules = self.propagation_rules[risk_type]
                if relationship_type in type_rules:
                    return type_rules[relationship_type]

            # Fall back to general relationship attenuation
            return self.relationship_attenuations.get(
                relationship_type, self.default_attenuation
            )

        except Exception as e:
            logger.error(f"Attenuation calculation failed: {e}")
            return self.default_attenuation

    def calculate_entity_risk_score(self, entity_id: str) -> Dict[str, Any]:
        """
        Calculate total risk score for an entity

        Args:
            entity_id: Entity to analyze

        Returns:
            Risk analysis results
        """
        try:
            total_risk = 0.0
            risk_types = defaultdict(float)
            direct_risks = []
            propagated_risks = []

            # Find direct risks
            for risk_event in self.risk_events.values():
                if risk_event.source_entity_id == entity_id:
                    direct_risks.append(risk_event)
                    total_risk += risk_event.risk_score
                    risk_types[risk_event.risk_type] += risk_event.risk_score

            # Find propagated risks
            for risk_id, propagations in self.propagations.items():
                for propagation in propagations:
                    if propagation.target_entity_id == entity_id:
                        propagated_risks.append(propagation)
                        total_risk += propagation.propagated_risk_score

                        # Get original risk type
                        if risk_id in self.risk_events:
                            original_risk = self.risk_events[risk_id]
                            risk_types[original_risk.risk_type] += (
                                propagation.propagated_risk_score
                            )

            # Find dominant risk type
            dominant_risk_type = None
            max_risk_value = 0.0
            for risk_type, value in risk_types.items():
                if value > max_risk_value:
                    max_risk_value = value
                    dominant_risk_type = risk_type

            return {
                "entity_id": entity_id,
                "total_risk_score": min(1.0, total_risk),  # Cap at 1.0
                "direct_risk_count": len(direct_risks),
                "propagated_risk_count": len(propagated_risks),
                "risk_type_breakdown": dict(risk_types),
                "dominant_risk_type": dominant_risk_type.value
                if dominant_risk_type
                else None,
                "direct_risks": [r.risk_id for r in direct_risks],
                "propagated_risks": [p.propagation_id for p in propagated_risks],
                "calculated_at": datetime.now(),
            }

        except Exception as e:
            logger.error(f"Entity risk calculation failed for {entity_id}: {e}")
            return {"entity_id": entity_id, "total_risk_score": 0.0, "error": str(e)}

    def identify_risk_clusters(
        self, min_cluster_size: int = 3, similarity_threshold: float = 0.7
    ) -> List[RiskCluster]:
        """
        Identify clusters of related high-risk entities

        Args:
            min_cluster_size: Minimum entities per cluster
            similarity_threshold: Minimum similarity for clustering

        Returns:
            List of identified risk clusters
        """
        try:
            clusters = []

            # Get all entities with significant risk
            high_risk_entities = []
            for risk_id, propagations in self.propagations.items():
                for propagation in propagations:
                    if propagation.propagated_risk_score >= self.risk_threshold:
                        high_risk_entities.append(propagation.target_entity_id)

            # Add entities with direct risks
            for risk_event in self.risk_events.values():
                if risk_event.risk_score >= self.risk_threshold:
                    high_risk_entities.append(risk_event.source_entity_id)

            # Remove duplicates
            unique_entities = list(set(high_risk_entities))

            if len(unique_entities) < min_cluster_size:
                return clusters

            # Build entity similarity matrix based on shared risks and connections
            similarity_matrix = self._build_entity_similarity_matrix(unique_entities)

            # Apply clustering algorithm (simple connected components)
            clusters_raw = self._cluster_entities(
                unique_entities, similarity_matrix, similarity_threshold
            )

            # Create cluster objects
            for i, cluster_entities in enumerate(clusters_raw):
                if len(cluster_entities) >= min_cluster_size:
                    cluster_risks = []
                    total_risk_score = 0.0
                    risk_type_counts = defaultdict(int)

                    # Collect risks for cluster entities
                    for entity_id in cluster_entities:
                        entity_risk = self.calculate_entity_risk_score(entity_id)
                        total_risk_score += entity_risk["total_risk_score"]

                        # Count risk types
                        for risk_type, score in entity_risk.get(
                            "risk_type_breakdown", {}
                        ).items():
                            if score > 0:
                                risk_type_counts[risk_type] += 1

                    # Find dominant risk type
                    dominant_risk_type = (
                        max(risk_type_counts.items(), key=lambda x: x[1])[0]
                        if risk_type_counts
                        else RiskType.OPERATIONAL_RISK
                    )

                    cluster = RiskCluster(
                        cluster_id=str(uuid.uuid4()),
                        entity_ids=set(cluster_entities),
                        risk_events=[],  # Will be populated if needed
                        total_risk_score=total_risk_score
                        / len(cluster_entities),  # Average
                        dominant_risk_type=RiskType(dominant_risk_type)
                        if isinstance(dominant_risk_type, str)
                        else dominant_risk_type,
                        cluster_size=len(cluster_entities),
                        created_at=datetime.now(),
                        updated_at=datetime.now(),
                    )

                    clusters.append(cluster)

            # Store clusters
            with self._lock:
                for cluster in clusters:
                    self.risk_clusters[cluster.cluster_id] = cluster
                self.stats["clusters_identified"] += len(clusters)

            logger.info(f"Identified {len(clusters)} risk clusters")
            return clusters

        except Exception as e:
            logger.error(f"Risk clustering failed: {e}")
            return []

    def _build_entity_similarity_matrix(
        self, entities: List[str]
    ) -> Dict[Tuple[str, str], float]:
        """Build similarity matrix between entities based on shared risks and connections"""
        try:
            similarity_matrix = {}

            for i, entity1 in enumerate(entities):
                for entity2 in entities[i + 1 :]:
                    similarity = 0.0

                    # Calculate similarity based on shared risks
                    entity1_risks = self._get_entity_risk_types(entity1)
                    entity2_risks = self._get_entity_risk_types(entity2)

                    if entity1_risks and entity2_risks:
                        shared_risks = entity1_risks.intersection(entity2_risks)
                        total_risks = entity1_risks.union(entity2_risks)

                        if total_risks:
                            risk_similarity = len(shared_risks) / len(total_risks)
                            similarity += risk_similarity * 0.7

                    # Calculate similarity based on graph connections
                    if self.neo4j_manager:
                        path = self.neo4j_manager.find_shortest_path(
                            entity1, entity2, max_length=3
                        )
                        if path and path["length"] <= 2:
                            connection_similarity = 1.0 / (path["length"] + 1)
                            similarity += connection_similarity * 0.3

                    similarity_matrix[(entity1, entity2)] = similarity

            return similarity_matrix

        except Exception as e:
            logger.error(f"Similarity matrix calculation failed: {e}")
            return {}

    def _get_entity_risk_types(self, entity_id: str) -> Set[RiskType]:
        """Get risk types associated with an entity"""
        risk_types = set()

        # Direct risks
        for risk_event in self.risk_events.values():
            if risk_event.source_entity_id == entity_id:
                risk_types.add(risk_event.risk_type)

        # Propagated risks
        for risk_id, propagations in self.propagations.items():
            for propagation in propagations:
                if (
                    propagation.target_entity_id == entity_id
                    and risk_id in self.risk_events
                ):
                    risk_types.add(self.risk_events[risk_id].risk_type)

        return risk_types

    def _cluster_entities(
        self,
        entities: List[str],
        similarity_matrix: Dict[Tuple[str, str], float],
        threshold: float,
    ) -> List[List[str]]:
        """Cluster entities using connected components approach"""
        try:
            # Build adjacency graph
            graph = defaultdict(set)

            for (entity1, entity2), similarity in similarity_matrix.items():
                if similarity >= threshold:
                    graph[entity1].add(entity2)
                    graph[entity2].add(entity1)

            # Find connected components
            visited = set()
            clusters = []

            for entity in entities:
                if entity not in visited:
                    cluster = []
                    stack = [entity]

                    while stack:
                        current = stack.pop()
                        if current not in visited:
                            visited.add(current)
                            cluster.append(current)

                            # Add connected entities
                            for neighbor in graph[current]:
                                if neighbor not in visited:
                                    stack.append(neighbor)

                    if cluster:
                        clusters.append(cluster)

            return clusters

        except Exception as e:
            logger.error(f"Entity clustering failed: {e}")
            return []

    def analyze_risk_paths(
        self, source_entity: str, target_entity: str
    ) -> Dict[str, Any]:
        """
        Analyze risk propagation paths between two entities

        Args:
            source_entity: Source entity ID
            target_entity: Target entity ID

        Returns:
            Path analysis results
        """
        try:
            if not self.neo4j_manager:
                return {"error": "Neo4j manager not available"}

            # Find shortest path
            path = self.neo4j_manager.find_shortest_path(
                source_entity, target_entity, max_length=self.max_propagation_depth
            )

            if not path:
                return {
                    "source_entity": source_entity,
                    "target_entity": target_entity,
                    "path_exists": False,
                    "message": "No path found between entities",
                }

            # Calculate risk propagation along path
            path_nodes = [node["node_id"] for node in path["nodes"]]
            path_relationships = path["relationships"]

            # Get risks at source
            source_risks = []
            for risk_event in self.risk_events.values():
                if risk_event.source_entity_id == source_entity:
                    source_risks.append(risk_event)

            # Calculate risk at each step
            path_analysis = []
            current_risk_scores = {
                risk.risk_type: risk.risk_score for risk in source_risks
            }

            for i, relationship in enumerate(path_relationships):
                rel_type = relationship.get("type", "RELATED_TO")
                target_node = path_nodes[i + 1]

                # Calculate propagated risks
                propagated_scores = {}
                for risk_type, score in current_risk_scores.items():
                    attenuation = self._get_attenuation_factor(risk_type, rel_type)
                    propagated_scores[risk_type] = score * attenuation

                path_analysis.append(
                    {
                        "step": i + 1,
                        "relationship_type": rel_type,
                        "target_entity": target_node,
                        "risk_scores": propagated_scores.copy(),
                        "total_risk": sum(propagated_scores.values()),
                    }
                )

                current_risk_scores = propagated_scores

            return {
                "source_entity": source_entity,
                "target_entity": target_entity,
                "path_exists": True,
                "path_length": len(path_relationships),
                "path_analysis": path_analysis,
                "final_risk_scores": current_risk_scores,
                "total_final_risk": sum(current_risk_scores.values()),
                "analyzed_at": datetime.now(),
            }

        except Exception as e:
            logger.error(f"Risk path analysis failed: {e}")
            return {"error": str(e)}

    def get_high_risk_entities(
        self, threshold: float = 0.5, limit: int = 50
    ) -> List[Dict[str, Any]]:
        """Get entities with highest risk scores"""
        try:
            # Collect all entities that appear in risks or propagations
            all_entities = set()

            # Add entities with direct risks
            for risk_event in self.risk_events.values():
                all_entities.add(risk_event.source_entity_id)

            # Add entities with propagated risks
            for propagations in self.propagations.values():
                for propagation in propagations:
                    all_entities.add(propagation.target_entity_id)

            # Calculate risk scores for all entities
            entity_risks = []
            for entity_id in all_entities:
                risk_analysis = self.calculate_entity_risk_score(entity_id)
                if risk_analysis["total_risk_score"] >= threshold:
                    entity_risks.append(risk_analysis)

            # Sort by risk score (descending)
            entity_risks.sort(key=lambda x: x["total_risk_score"], reverse=True)

            return entity_risks[:limit]

        except Exception as e:
            logger.error(f"High risk entity analysis failed: {e}")
            return []

    def get_propagation_statistics(self) -> Dict[str, Any]:
        """Get comprehensive propagation statistics"""
        try:
            stats = self.stats.copy()

            # Calculate additional metrics
            total_propagations = sum(len(props) for props in self.propagations.values())
            avg_propagations_per_risk = (
                total_propagations / len(self.risk_events) if self.risk_events else 0
            )

            # Risk type distribution
            risk_type_counts = defaultdict(int)
            for risk_event in self.risk_events.values():
                risk_type_counts[risk_event.risk_type.value] += 1

            # Severity distribution
            severity_counts = defaultdict(int)
            for risk_event in self.risk_events.values():
                severity_counts[risk_event.severity.value] += 1

            # Cluster statistics
            cluster_sizes = [
                cluster.cluster_size for cluster in self.risk_clusters.values()
            ]
            avg_cluster_size = (
                sum(cluster_sizes) / len(cluster_sizes) if cluster_sizes else 0
            )

            stats.update(
                {
                    "total_risk_events": len(self.risk_events),
                    "total_propagations": total_propagations,
                    "total_clusters": len(self.risk_clusters),
                    "avg_propagations_per_risk": avg_propagations_per_risk,
                    "avg_cluster_size": avg_cluster_size,
                    "risk_type_distribution": dict(risk_type_counts),
                    "severity_distribution": dict(severity_counts),
                    "max_propagation_depth": self.max_propagation_depth,
                    "risk_threshold": self.risk_threshold,
                }
            )

            return stats

        except Exception as e:
            logger.error(f"Statistics calculation failed: {e}")
            return self.stats.copy()

    def clear_expired_risks(self, retention_days: int = 30):
        """Clear old risk events and propagations"""
        try:
            cutoff_date = datetime.now() - timedelta(days=retention_days)

            # Clear expired risk events
            expired_risks = []
            for risk_id, risk_event in self.risk_events.items():
                if risk_event.created_at < cutoff_date:
                    expired_risks.append(risk_id)

            for risk_id in expired_risks:
                del self.risk_events[risk_id]
                if risk_id in self.propagations:
                    del self.propagations[risk_id]

            logger.info(f"Cleared {len(expired_risks)} expired risk events")

        except Exception as e:
            logger.error(f"Risk cleanup failed: {e}")

    def close(self):
        """Close the risk propagation engine and cleanup resources"""
        try:
            if hasattr(self, "_executor") and self._executor:
                self._executor.shutdown(wait=True)

            logger.info("Risk propagation engine closed")

        except Exception as e:
            logger.error(f"Error closing risk propagation engine: {e}")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
