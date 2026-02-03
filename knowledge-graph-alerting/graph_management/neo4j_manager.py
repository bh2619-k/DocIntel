"""
Real-Time Document Intelligence & Compliance Engine
Neo4j Knowledge Graph Management
Version: 1.0
Date: 2026-02-03

This module provides comprehensive Neo4j knowledge graph management including
node/edge operations, relationship inference, and graph analytics.
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
from dataclasses import dataclass
from enum import Enum
import json
import uuid
from concurrent.futures import ThreadPoolExecutor
import threading

try:
    from neo4j import GraphDatabase

    HAS_NEO4J = True
except ImportError:
    HAS_NEO4J = False

    # Mock implementations for testing
    class MockDriver:
        def session(self, **kwargs):
            return MockSession()

        def close(self):
            pass

    class MockSession:
        def run(self, query, parameters=None):
            return MockResult()

        def close(self):
            pass

    class MockResult:
        def data(self):
            return []

        def single(self):
            return None


logger = logging.getLogger(__name__)


class NodeType(Enum):
    """Types of nodes in the knowledge graph"""

    DOCUMENT = "Document"
    PERSON = "Person"
    ORGANIZATION = "Organization"
    CONTRACT = "Contract"
    FINANCIAL_INSTRUMENT = "FinancialInstrument"
    RISK_EVENT = "RiskEvent"
    COMPLIANCE_RULE = "ComplianceRule"
    LOCATION = "Location"
    DATE = "Date"
    AMOUNT = "Amount"
    LEGAL_ENTITY = "LegalEntity"
    PRODUCT = "Product"
    ALERT = "Alert"


class RelationshipType(Enum):
    """Types of relationships in the knowledge graph"""

    # Document relationships
    CONTAINS = "CONTAINS"
    MENTIONS = "MENTIONS"
    REFERENCES = "REFERENCES"

    # Entity relationships
    EMPLOYED_BY = "EMPLOYED_BY"
    OWNS = "OWNS"
    CONTROLS = "CONTROLS"
    SUBSIDIARY_OF = "SUBSIDIARY_OF"
    PARTNER_WITH = "PARTNER_WITH"
    RELATED_TO = "RELATED_TO"

    # Contract relationships
    PARTY_TO = "PARTY_TO"
    SIGNATORY = "SIGNATORY"
    BENEFICIARY = "BENEFICIARY"
    GUARANTOR = "GUARANTOR"

    # Risk relationships
    CAUSES = "CAUSES"
    MITIGATES = "MITIGATES"
    AFFECTS = "AFFECTS"
    TRIGGERS = "TRIGGERS"

    # Compliance relationships
    VIOLATES = "VIOLATES"
    COMPLIES_WITH = "COMPLIES_WITH"
    SUBJECT_TO = "SUBJECT_TO"

    # Temporal relationships
    BEFORE = "BEFORE"
    AFTER = "AFTER"
    DURING = "DURING"

    # Location relationships
    LOCATED_AT = "LOCATED_AT"
    OPERATES_IN = "OPERATES_IN"


@dataclass
class GraphNode:
    """Represents a node in the knowledge graph"""

    node_id: str
    node_type: NodeType
    properties: Dict[str, Any]
    labels: List[str]
    created_at: datetime
    updated_at: datetime

    def to_cypher_properties(self) -> str:
        """Convert properties to Cypher format"""
        props = self.properties.copy()
        props.update(
            {
                "node_id": self.node_id,
                "created_at": self.created_at.isoformat(),
                "updated_at": self.updated_at.isoformat(),
            }
        )

        # Format properties for Cypher
        prop_strings = []
        for key, value in props.items():
            if isinstance(value, str):
                prop_strings.append(f"{key}: '{value.replace('"', '\\"')}'")
            elif isinstance(value, (int, float, bool)):
                prop_strings.append(f"{key}: {json.dumps(value)}")
            elif isinstance(value, (list, dict)):
                prop_strings.append(f"{key}: {json.dumps(value)}")
            else:
                prop_strings.append(f"{key}: '{str(value)}'")

        return "{" + ", ".join(prop_strings) + "}"


@dataclass
class GraphRelationship:
    """Represents a relationship in the knowledge graph"""

    relationship_id: str
    source_node_id: str
    target_node_id: str
    relationship_type: RelationshipType
    properties: Dict[str, Any]
    confidence: float
    created_at: datetime
    updated_at: datetime

    def to_cypher_properties(self) -> str:
        """Convert properties to Cypher format"""
        props = self.properties.copy()
        props.update(
            {
                "relationship_id": self.relationship_id,
                "confidence": self.confidence,
                "created_at": self.created_at.isoformat(),
                "updated_at": self.updated_at.isoformat(),
            }
        )

        # Format properties for Cypher
        prop_strings = []
        for key, value in props.items():
            if isinstance(value, str):
                prop_strings.append(f"{key}: '{value.replace('"', '\\"')}'")
            elif isinstance(value, (int, float, bool)):
                prop_strings.append(f"{key}: {json.dumps(value)}")
            elif isinstance(value, (list, dict)):
                prop_strings.append(f"{key}: {json.dumps(value)}")
            else:
                prop_strings.append(f"{key}: '{str(value)}'")

        return "{" + ", ".join(prop_strings) + "}"


@dataclass
class GraphUpdate:
    """Represents a batch update to the knowledge graph"""

    update_id: str
    nodes_to_create: List[GraphNode]
    nodes_to_update: List[GraphNode]
    nodes_to_delete: List[str]
    relationships_to_create: List[GraphRelationship]
    relationships_to_update: List[GraphRelationship]
    relationships_to_delete: List[str]
    created_at: datetime


class Neo4jGraphManager:
    """Comprehensive Neo4j knowledge graph management system"""

    def __init__(
        self,
        neo4j_uri: str,
        neo4j_user: str,
        neo4j_password: str,
        database: str = "neo4j",
        max_connection_pool_size: int = 50,
    ):
        """
        Initialize Neo4j graph manager

        Args:
            neo4j_uri: Neo4j database URI
            neo4j_user: Database username
            neo4j_password: Database password
            database: Database name
            max_connection_pool_size: Maximum connection pool size
        """
        self.neo4j_uri = neo4j_uri
        self.neo4j_user = neo4j_user
        self.database = database

        # Connection and thread safety
        self._lock = threading.RLock()
        self._executor = ThreadPoolExecutor(max_workers=10)

        # Initialize driver
        if HAS_NEO4J:
            try:
                self.driver = GraphDatabase.driver(
                    neo4j_uri,
                    auth=(neo4j_user, neo4j_password),
                    max_connection_pool_size=max_connection_pool_size,
                )
                # Test connection
                with self.driver.session(database=database) as session:
                    session.run("RETURN 1")
                logger.info(f"Connected to Neo4j database: {neo4j_uri}")
            except Exception as e:
                logger.error(f"Failed to connect to Neo4j: {e}")
                self.driver = MockDriver()
        else:
            logger.warning("Neo4j driver not available, using mock implementation")
            self.driver = MockDriver()

        # Caches and indexes
        self.node_cache: Dict[str, GraphNode] = {}
        self.relationship_cache: Dict[str, GraphRelationship] = {}

        # Statistics
        self.stats = {
            "nodes_created": 0,
            "nodes_updated": 0,
            "nodes_deleted": 0,
            "relationships_created": 0,
            "relationships_updated": 0,
            "relationships_deleted": 0,
            "queries_executed": 0,
            "batch_updates": 0,
        }

        # Initialize graph schema
        self._initialize_schema()

    def _initialize_schema(self):
        """Initialize Neo4j graph schema with constraints and indexes"""
        try:
            with self.driver.session(database=self.database) as session:
                # Create uniqueness constraints
                constraints = [
                    "CREATE CONSTRAINT IF NOT EXISTS FOR (n:Document) REQUIRE n.node_id IS UNIQUE",
                    "CREATE CONSTRAINT IF NOT EXISTS FOR (n:Person) REQUIRE n.node_id IS UNIQUE",
                    "CREATE CONSTRAINT IF NOT EXISTS FOR (n:Organization) REQUIRE n.node_id IS UNIQUE",
                    "CREATE CONSTRAINT IF NOT EXISTS FOR (n:Contract) REQUIRE n.node_id IS UNIQUE",
                    "CREATE CONSTRAINT IF NOT EXISTS FOR (n:FinancialInstrument) REQUIRE n.node_id IS UNIQUE",
                    "CREATE CONSTRAINT IF NOT EXISTS FOR (n:RiskEvent) REQUIRE n.node_id IS UNIQUE",
                    "CREATE CONSTRAINT IF NOT EXISTS FOR (n:ComplianceRule) REQUIRE n.node_id IS UNIQUE",
                    "CREATE CONSTRAINT IF NOT EXISTS FOR (n:Alert) REQUIRE n.node_id IS UNIQUE",
                ]

                for constraint in constraints:
                    try:
                        session.run(constraint)
                    except Exception as e:
                        # Constraint might already exist
                        logger.debug(f"Constraint creation skipped: {e}")

                # Create performance indexes
                indexes = [
                    "CREATE INDEX IF NOT EXISTS FOR (n:Document) ON (n.document_type)",
                    "CREATE INDEX IF NOT EXISTS FOR (n:Person) ON (n.name)",
                    "CREATE INDEX IF NOT EXISTS FOR (n:Organization) ON (n.name)",
                    "CREATE INDEX IF NOT EXISTS FOR (n:Contract) ON (n.contract_type)",
                    "CREATE INDEX IF NOT EXISTS FOR (n) ON (n.created_at)",
                    "CREATE INDEX IF NOT EXISTS FOR (n) ON (n.updated_at)",
                ]

                for index in indexes:
                    try:
                        session.run(index)
                    except Exception as e:
                        logger.debug(f"Index creation skipped: {e}")

                logger.info("Neo4j schema initialization completed")

        except Exception as e:
            logger.error(f"Schema initialization failed: {e}")

    def create_node(
        self, node_type: NodeType, properties: Dict[str, Any], node_id: str = None
    ) -> str:
        """
        Create a new node in the knowledge graph

        Args:
            node_type: Type of node to create
            properties: Node properties
            node_id: Optional custom node ID

        Returns:
            Created node ID
        """
        try:
            if node_id is None:
                node_id = str(uuid.uuid4())

            current_time = datetime.now()

            # Create node object
            node = GraphNode(
                node_id=node_id,
                node_type=node_type,
                properties=properties,
                labels=[node_type.value],
                created_at=current_time,
                updated_at=current_time,
            )

            # Execute Cypher query
            with self.driver.session(database=self.database) as session:
                cypher_props = node.to_cypher_properties()
                query = f"""
                CREATE (n:{node_type.value} {cypher_props})
                RETURN n.node_id as node_id
                """

                result = session.run(query)
                record = result.single()

                if record:
                    created_node_id = record["node_id"]

                    # Update cache
                    with self._lock:
                        self.node_cache[created_node_id] = node
                        self.stats["nodes_created"] += 1
                        self.stats["queries_executed"] += 1

                    logger.debug(f"Created node: {created_node_id} ({node_type.value})")
                    return created_node_id
                else:
                    raise Exception("Node creation returned no result")

        except Exception as e:
            logger.error(f"Failed to create node: {e}")
            raise

    def update_node(self, node_id: str, properties: Dict[str, Any]) -> bool:
        """
        Update an existing node's properties

        Args:
            node_id: ID of node to update
            properties: New properties to set

        Returns:
            True if update successful
        """
        try:
            current_time = datetime.now()

            # Prepare property updates
            set_clauses = []
            for key, value in properties.items():
                if isinstance(value, str):
                    set_clauses.append(f"n.{key} = '{value.replace('"', '\\"')}'")
                elif isinstance(value, (int, float, bool)):
                    set_clauses.append(f"n.{key} = {json.dumps(value)}")
                elif isinstance(value, (list, dict)):
                    set_clauses.append(f"n.{key} = {json.dumps(value)}")
                else:
                    set_clauses.append(f"n.{key} = '{str(value)}'")

            set_clauses.append(f"n.updated_at = '{current_time.isoformat()}'")

            with self.driver.session(database=self.database) as session:
                query = f"""
                MATCH (n {{node_id: '{node_id}'}})
                SET {", ".join(set_clauses)}
                RETURN n.node_id as node_id
                """

                result = session.run(query)
                record = result.single()

                if record:
                    # Update cache if node exists
                    with self._lock:
                        if node_id in self.node_cache:
                            self.node_cache[node_id].properties.update(properties)
                            self.node_cache[node_id].updated_at = current_time

                        self.stats["nodes_updated"] += 1
                        self.stats["queries_executed"] += 1

                    logger.debug(f"Updated node: {node_id}")
                    return True
                else:
                    logger.warning(f"Node not found for update: {node_id}")
                    return False

        except Exception as e:
            logger.error(f"Failed to update node {node_id}: {e}")
            return False

    def delete_node(self, node_id: str, cascade: bool = True) -> bool:
        """
        Delete a node from the knowledge graph

        Args:
            node_id: ID of node to delete
            cascade: Whether to delete related relationships

        Returns:
            True if deletion successful
        """
        try:
            with self.driver.session(database=self.database) as session:
                if cascade:
                    # Delete node and all relationships
                    query = f"""
                    MATCH (n {{node_id: '{node_id}'}})
                    DETACH DELETE n
                    """
                else:
                    # Delete only the node (will fail if relationships exist)
                    query = f"""
                    MATCH (n {{node_id: '{node_id}'}})
                    DELETE n
                    """

                session.run(query)

                # Remove from cache
                with self._lock:
                    if node_id in self.node_cache:
                        del self.node_cache[node_id]

                    self.stats["nodes_deleted"] += 1
                    self.stats["queries_executed"] += 1

                logger.debug(f"Deleted node: {node_id}")
                return True

        except Exception as e:
            logger.error(f"Failed to delete node {node_id}: {e}")
            return False

    def create_relationship(
        self,
        source_node_id: str,
        target_node_id: str,
        relationship_type: RelationshipType,
        properties: Dict[str, Any] = None,
        confidence: float = 1.0,
        relationship_id: str = None,
    ) -> str:
        """
        Create a relationship between two nodes

        Args:
            source_node_id: Source node ID
            target_node_id: Target node ID
            relationship_type: Type of relationship
            properties: Relationship properties
            confidence: Confidence score
            relationship_id: Optional custom relationship ID

        Returns:
            Created relationship ID
        """
        try:
            if relationship_id is None:
                relationship_id = str(uuid.uuid4())

            if properties is None:
                properties = {}

            current_time = datetime.now()

            # Create relationship object
            relationship = GraphRelationship(
                relationship_id=relationship_id,
                source_node_id=source_node_id,
                target_node_id=target_node_id,
                relationship_type=relationship_type,
                properties=properties,
                confidence=confidence,
                created_at=current_time,
                updated_at=current_time,
            )

            with self.driver.session(database=self.database) as session:
                cypher_props = relationship.to_cypher_properties()
                query = f"""
                MATCH (source {{node_id: '{source_node_id}'}})
                MATCH (target {{node_id: '{target_node_id}'}})
                CREATE (source)-[r:{relationship_type.value} {cypher_props}]->(target)
                RETURN r.relationship_id as relationship_id
                """

                result = session.run(query)
                record = result.single()

                if record:
                    created_rel_id = record["relationship_id"]

                    # Update cache
                    with self._lock:
                        self.relationship_cache[created_rel_id] = relationship
                        self.stats["relationships_created"] += 1
                        self.stats["queries_executed"] += 1

                    logger.debug(
                        f"Created relationship: {created_rel_id} ({relationship_type.value})"
                    )
                    return created_rel_id
                else:
                    raise Exception("Relationship creation returned no result")

        except Exception as e:
            logger.error(f"Failed to create relationship: {e}")
            raise

    def update_relationship(
        self, relationship_id: str, properties: Dict[str, Any]
    ) -> bool:
        """
        Update relationship properties

        Args:
            relationship_id: ID of relationship to update
            properties: New properties to set

        Returns:
            True if update successful
        """
        try:
            current_time = datetime.now()

            # Prepare property updates
            set_clauses = []
            for key, value in properties.items():
                if isinstance(value, str):
                    set_clauses.append(f"r.{key} = '{value.replace('"', '\\"')}'")
                elif isinstance(value, (int, float, bool)):
                    set_clauses.append(f"r.{key} = {json.dumps(value)}")
                elif isinstance(value, (list, dict)):
                    set_clauses.append(f"r.{key} = {json.dumps(value)}")
                else:
                    set_clauses.append(f"r.{key} = '{str(value)}'")

            set_clauses.append(f"r.updated_at = '{current_time.isoformat()}'")

            with self.driver.session(database=self.database) as session:
                query = f"""
                MATCH ()-[r {{relationship_id: '{relationship_id}'}}]->()
                SET {", ".join(set_clauses)}
                RETURN r.relationship_id as relationship_id
                """

                result = session.run(query)
                record = result.single()

                if record:
                    # Update cache
                    with self._lock:
                        if relationship_id in self.relationship_cache:
                            self.relationship_cache[relationship_id].properties.update(
                                properties
                            )
                            self.relationship_cache[
                                relationship_id
                            ].updated_at = current_time

                        self.stats["relationships_updated"] += 1
                        self.stats["queries_executed"] += 1

                    logger.debug(f"Updated relationship: {relationship_id}")
                    return True
                else:
                    logger.warning(
                        f"Relationship not found for update: {relationship_id}"
                    )
                    return False

        except Exception as e:
            logger.error(f"Failed to update relationship {relationship_id}: {e}")
            return False

    def delete_relationship(self, relationship_id: str) -> bool:
        """
        Delete a relationship from the knowledge graph

        Args:
            relationship_id: ID of relationship to delete

        Returns:
            True if deletion successful
        """
        try:
            with self.driver.session(database=self.database) as session:
                query = f"""
                MATCH ()-[r {{relationship_id: '{relationship_id}'}}]->()
                DELETE r
                """

                session.run(query)

                # Remove from cache
                with self._lock:
                    if relationship_id in self.relationship_cache:
                        del self.relationship_cache[relationship_id]

                    self.stats["relationships_deleted"] += 1
                    self.stats["queries_executed"] += 1

                logger.debug(f"Deleted relationship: {relationship_id}")
                return True

        except Exception as e:
            logger.error(f"Failed to delete relationship {relationship_id}: {e}")
            return False

    def batch_update(self, graph_update: GraphUpdate) -> Dict[str, Any]:
        """
        Execute a batch update to the knowledge graph

        Args:
            graph_update: Batch update specification

        Returns:
            Update results summary
        """
        try:
            results = {
                "nodes_created": 0,
                "nodes_updated": 0,
                "nodes_deleted": 0,
                "relationships_created": 0,
                "relationships_updated": 0,
                "relationships_deleted": 0,
                "errors": [],
            }

            with self.driver.session(database=self.database) as session:
                # Use transaction for consistency
                with session.begin_transaction() as tx:
                    # Delete relationships first (to avoid constraint violations)
                    for rel_id in graph_update.relationships_to_delete:
                        try:
                            tx.run(
                                f"MATCH ()-[r {{relationship_id: '{rel_id}'}}]->() DELETE r"
                            )
                            results["relationships_deleted"] += 1
                        except Exception as e:
                            results["errors"].append(
                                f"Failed to delete relationship {rel_id}: {e}"
                            )

                    # Delete nodes
                    for node_id in graph_update.nodes_to_delete:
                        try:
                            tx.run(
                                f"MATCH (n {{node_id: '{node_id}'}}) DETACH DELETE n"
                            )
                            results["nodes_deleted"] += 1
                        except Exception as e:
                            results["errors"].append(
                                f"Failed to delete node {node_id}: {e}"
                            )

                    # Create nodes
                    for node in graph_update.nodes_to_create:
                        try:
                            cypher_props = node.to_cypher_properties()
                            query = f"CREATE (n:{node.node_type.value} {cypher_props})"
                            tx.run(query)
                            results["nodes_created"] += 1
                        except Exception as e:
                            results["errors"].append(
                                f"Failed to create node {node.node_id}: {e}"
                            )

                    # Update nodes
                    for node in graph_update.nodes_to_update:
                        try:
                            set_clauses = []
                            for key, value in node.properties.items():
                                if isinstance(value, str):
                                    set_clauses.append(
                                        f"n.{key} = '{value.replace('"', '\\"')}'"
                                    )
                                else:
                                    set_clauses.append(f"n.{key} = {json.dumps(value)}")

                            set_clauses.append(
                                f"n.updated_at = '{node.updated_at.isoformat()}'"
                            )

                            query = f"""
                            MATCH (n {{node_id: '{node.node_id}'}})
                            SET {", ".join(set_clauses)}
                            """
                            tx.run(query)
                            results["nodes_updated"] += 1
                        except Exception as e:
                            results["errors"].append(
                                f"Failed to update node {node.node_id}: {e}"
                            )

                    # Create relationships
                    for relationship in graph_update.relationships_to_create:
                        try:
                            cypher_props = relationship.to_cypher_properties()
                            query = f"""
                            MATCH (source {{node_id: '{relationship.source_node_id}'}})
                            MATCH (target {{node_id: '{relationship.target_node_id}'}})
                            CREATE (source)-[r:{relationship.relationship_type.value} {cypher_props}]->(target)
                            """
                            tx.run(query)
                            results["relationships_created"] += 1
                        except Exception as e:
                            results["errors"].append(
                                f"Failed to create relationship {relationship.relationship_id}: {e}"
                            )

                    # Update relationships
                    for relationship in graph_update.relationships_to_update:
                        try:
                            set_clauses = []
                            for key, value in relationship.properties.items():
                                if isinstance(value, str):
                                    set_clauses.append(
                                        f"r.{key} = '{value.replace('"', '\\"')}'"
                                    )
                                else:
                                    set_clauses.append(f"r.{key} = {json.dumps(value)}")

                            set_clauses.append(
                                f"r.updated_at = '{relationship.updated_at.isoformat()}'"
                            )

                            query = f"""
                            MATCH ()-[r {{relationship_id: '{relationship.relationship_id}'}}]->()
                            SET {", ".join(set_clauses)}
                            """
                            tx.run(query)
                            results["relationships_updated"] += 1
                        except Exception as e:
                            results["errors"].append(
                                f"Failed to update relationship {relationship.relationship_id}: {e}"
                            )

            # Update statistics
            with self._lock:
                self.stats["nodes_created"] += results["nodes_created"]
                self.stats["nodes_updated"] += results["nodes_updated"]
                self.stats["nodes_deleted"] += results["nodes_deleted"]
                self.stats["relationships_created"] += results["relationships_created"]
                self.stats["relationships_updated"] += results["relationships_updated"]
                self.stats["relationships_deleted"] += results["relationships_deleted"]
                self.stats["batch_updates"] += 1
                self.stats["queries_executed"] += (
                    len(graph_update.nodes_to_create)
                    + len(graph_update.nodes_to_update)
                    + len(graph_update.nodes_to_delete)
                    + len(graph_update.relationships_to_create)
                    + len(graph_update.relationships_to_update)
                    + len(graph_update.relationships_to_delete)
                )

            logger.info(f"Batch update completed: {graph_update.update_id}")
            return results

        except Exception as e:
            logger.error(f"Batch update failed: {e}")
            return {"error": str(e)}

    def get_node(self, node_id: str) -> Optional[Dict[str, Any]]:
        """Get node by ID"""
        try:
            with self.driver.session(database=self.database) as session:
                query = f"MATCH (n {{node_id: '{node_id}'}}) RETURN n"
                result = session.run(query)
                record = result.single()

                if record:
                    node = record["n"]
                    return dict(node)
                else:
                    return None

        except Exception as e:
            logger.error(f"Failed to get node {node_id}: {e}")
            return None

    def get_relationships(
        self, node_id: str, direction: str = "both"
    ) -> List[Dict[str, Any]]:
        """
        Get relationships for a node

        Args:
            node_id: Node ID
            direction: "incoming", "outgoing", or "both"

        Returns:
            List of relationship data
        """
        try:
            with self.driver.session(database=self.database) as session:
                if direction == "outgoing":
                    query = f"""
                    MATCH (n {{node_id: '{node_id}'}})-[r]->(target)
                    RETURN r, target.node_id as target_id
                    """
                elif direction == "incoming":
                    query = f"""
                    MATCH (source)-[r]->(n {{node_id: '{node_id}'}})
                    RETURN r, source.node_id as source_id
                    """
                else:  # both
                    query = f"""
                    MATCH (n {{node_id: '{node_id}'}})-[r]-(other)
                    RETURN r, other.node_id as other_id
                    """

                result = session.run(query)
                relationships = []

                for record in result:
                    rel_data = dict(record["r"])
                    if direction == "outgoing":
                        rel_data["target_id"] = record["target_id"]
                    elif direction == "incoming":
                        rel_data["source_id"] = record["source_id"]
                    else:
                        rel_data["other_id"] = record["other_id"]

                    relationships.append(rel_data)

                return relationships

        except Exception as e:
            logger.error(f"Failed to get relationships for {node_id}: {e}")
            return []

    def find_shortest_path(
        self, source_id: str, target_id: str, max_length: int = 5
    ) -> Optional[List[Dict[str, Any]]]:
        """
        Find shortest path between two nodes

        Args:
            source_id: Source node ID
            target_id: Target node ID
            max_length: Maximum path length

        Returns:
            Path as list of nodes and relationships
        """
        try:
            with self.driver.session(database=self.database) as session:
                query = f"""
                MATCH path = shortestPath((source {{node_id: '{source_id}'}})-[*1..{max_length}]-(target {{node_id: '{target_id}'}}))
                RETURN nodes(path) as nodes, relationships(path) as relationships
                """

                result = session.run(query)
                record = result.single()

                if record:
                    nodes = [dict(node) for node in record["nodes"]]
                    relationships = [dict(rel) for rel in record["relationships"]]

                    return {
                        "nodes": nodes,
                        "relationships": relationships,
                        "length": len(relationships),
                    }
                else:
                    return None

        except Exception as e:
            logger.error(f"Failed to find shortest path: {e}")
            return None

    def execute_cypher(
        self, query: str, parameters: Dict[str, Any] = None
    ) -> List[Dict[str, Any]]:
        """
        Execute custom Cypher query

        Args:
            query: Cypher query string
            parameters: Query parameters

        Returns:
            Query results
        """
        try:
            with self.driver.session(database=self.database) as session:
                result = session.run(query, parameters or {})
                return result.data()

        except Exception as e:
            logger.error(f"Cypher query failed: {e}")
            return []

    def get_graph_statistics(self) -> Dict[str, Any]:
        """Get comprehensive graph statistics"""
        try:
            stats = self.stats.copy()

            # Get current graph state
            with self.driver.session(database=self.database) as session:
                # Count nodes by type
                node_counts = {}
                for node_type in NodeType:
                    query = f"MATCH (n:{node_type.value}) RETURN count(n) as count"
                    result = session.run(query)
                    record = result.single()
                    node_counts[node_type.value] = record["count"] if record else 0

                # Count relationships by type
                relationship_counts = {}
                for rel_type in RelationshipType:
                    query = (
                        f"MATCH ()-[r:{rel_type.value}]->() RETURN count(r) as count"
                    )
                    result = session.run(query)
                    record = result.single()
                    relationship_counts[rel_type.value] = (
                        record["count"] if record else 0
                    )

                # Get total counts
                total_nodes_result = session.run("MATCH (n) RETURN count(n) as count")
                total_relationships_result = session.run(
                    "MATCH ()-[r]->() RETURN count(r) as count"
                )

                total_nodes = total_nodes_result.single()["count"]
                total_relationships = total_relationships_result.single()["count"]

                stats.update(
                    {
                        "current_nodes": total_nodes,
                        "current_relationships": total_relationships,
                        "node_type_distribution": node_counts,
                        "relationship_type_distribution": relationship_counts,
                        "cache_size": len(self.node_cache)
                        + len(self.relationship_cache),
                    }
                )

            return stats

        except Exception as e:
            logger.error(f"Failed to get graph statistics: {e}")
            return self.stats.copy()

    def infer_relationships(self, document_id: str) -> List[GraphRelationship]:
        """
        Infer new relationships based on document analysis

        Args:
            document_id: Document to analyze for relationships

        Returns:
            List of inferred relationships
        """
        try:
            inferred_relationships = []
            current_time = datetime.now()

            # Get all entities mentioned in the document
            with self.driver.session(database=self.database) as session:
                query = f"""
                MATCH (doc:Document {{node_id: '{document_id}'}})-[:MENTIONS]->(entity)
                RETURN entity.node_id as entity_id, labels(entity) as labels, entity
                """

                result = session.run(query)
                entities = result.data()

                # Analyze entity relationships based on patterns
                for i, entity1 in enumerate(entities):
                    for entity2 in entities[i + 1 :]:
                        # Infer relationships based on entity types
                        inferred_rel = self._infer_entity_relationship(
                            entity1, entity2, document_id
                        )

                        if inferred_rel:
                            inferred_relationships.append(inferred_rel)

            return inferred_relationships

        except Exception as e:
            logger.error(f"Relationship inference failed: {e}")
            return []

    def _infer_entity_relationship(
        self, entity1: Dict, entity2: Dict, document_id: str
    ) -> Optional[GraphRelationship]:
        """Infer relationship between two entities"""
        try:
            labels1 = entity1["labels"]
            labels2 = entity2["labels"]

            # Person-Organization relationships
            if "Person" in labels1 and "Organization" in labels2:
                return GraphRelationship(
                    relationship_id=str(uuid.uuid4()),
                    source_node_id=entity1["entity_id"],
                    target_node_id=entity2["entity_id"],
                    relationship_type=RelationshipType.EMPLOYED_BY,
                    properties={
                        "inferred_from": document_id,
                        "inference_type": "person_organization",
                    },
                    confidence=0.7,
                    created_at=datetime.now(),
                    updated_at=datetime.now(),
                )

            # Contract-Party relationships
            if "Contract" in labels1 and (
                "Person" in labels2 or "Organization" in labels2
            ):
                return GraphRelationship(
                    relationship_id=str(uuid.uuid4()),
                    source_node_id=entity2["entity_id"],
                    target_node_id=entity1["entity_id"],
                    relationship_type=RelationshipType.PARTY_TO,
                    properties={
                        "inferred_from": document_id,
                        "inference_type": "contract_party",
                    },
                    confidence=0.8,
                    created_at=datetime.now(),
                    updated_at=datetime.now(),
                )

            # Organization-Organization relationships (subsidiaries, partnerships)
            if "Organization" in labels1 and "Organization" in labels2:
                # Simple heuristic based on name similarity
                name1 = entity1.get("entity", {}).get("name", "")
                name2 = entity2.get("entity", {}).get("name", "")

                if name1 and name2:
                    # Check for subsidiary patterns
                    if (
                        any(
                            word in name1.lower()
                            for word in ["subsidiary", "division", "unit"]
                        )
                        and name2.lower() in name1.lower()
                    ):
                        return GraphRelationship(
                            relationship_id=str(uuid.uuid4()),
                            source_node_id=entity1["entity_id"],
                            target_node_id=entity2["entity_id"],
                            relationship_type=RelationshipType.SUBSIDIARY_OF,
                            properties={
                                "inferred_from": document_id,
                                "inference_type": "subsidiary_pattern",
                            },
                            confidence=0.6,
                            created_at=datetime.now(),
                            updated_at=datetime.now(),
                        )

            return None

        except Exception as e:
            logger.error(f"Entity relationship inference failed: {e}")
            return None

    def close(self):
        """Close Neo4j driver and cleanup resources"""
        try:
            if hasattr(self, "driver") and self.driver:
                self.driver.close()

            if hasattr(self, "_executor") and self._executor:
                self._executor.shutdown(wait=True)

            logger.info("Neo4j graph manager closed")

        except Exception as e:
            logger.error(f"Error closing graph manager: {e}")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
