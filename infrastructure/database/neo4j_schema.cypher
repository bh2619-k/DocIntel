// Real-Time Document Intelligence & Compliance Engine
// Neo4j Knowledge Graph Schema Definition
// Version: 1.0
// Date: 2026-02-03

// =============================================================================
// NODE DEFINITIONS
// =============================================================================

// Document Node
// Represents processed documents in the system
// Properties: document_id, tenant_id, type, status, created_at, metadata
CREATE CONSTRAINT document_id_unique IF NOT EXISTS FOR (d:Document) REQUIRE d.document_id IS UNIQUE;
CREATE CONSTRAINT document_tenant_composite IF NOT EXISTS FOR (d:Document) REQUIRE (d.tenant_id, d.document_id) IS UNIQUE;
CREATE INDEX document_tenant_idx IF NOT EXISTS FOR (d:Document) ON (d.tenant_id);
CREATE INDEX document_type_idx IF NOT EXISTS FOR (d:Document) ON (d.type);
CREATE INDEX document_created_at_idx IF NOT EXISTS FOR (d:Document) ON (d.created_at);

// Clause Node
// Represents extracted legal/contractual clauses
// Properties: clause_id, tenant_id, type, content, confidence, risk_score, page_numbers
CREATE CONSTRAINT clause_id_unique IF NOT EXISTS FOR (c:Clause) REQUIRE c.clause_id IS UNIQUE;
CREATE INDEX clause_tenant_idx IF NOT EXISTS FOR (c:Clause) ON (c.tenant_id);
CREATE INDEX clause_type_idx IF NOT EXISTS FOR (c:Clause) ON (c.type);
CREATE INDEX clause_risk_score_idx IF NOT EXISTS FOR (c:Clause) ON (c.risk_score);
CREATE INDEX clause_confidence_idx IF NOT EXISTS FOR (c:Clause) ON (c.confidence);

// Entity Node
// Represents named entities extracted from documents
// Properties: entity_id, tenant_id, type, value, normalized_value, confidence
CREATE CONSTRAINT entity_id_unique IF NOT EXISTS FOR (e:Entity) REQUIRE e.entity_id IS UNIQUE;
CREATE INDEX entity_tenant_idx IF NOT EXISTS FOR (e:Entity) ON (e.tenant_id);
CREATE INDEX entity_type_idx IF NOT EXISTS FOR (e:Entity) ON (e.type);
CREATE INDEX entity_normalized_value_idx IF NOT EXISTS FOR (e:Entity) ON (e.normalized_value);
CREATE INDEX entity_confidence_idx IF NOT EXISTS FOR (e:Entity) ON (e.confidence);

// Risk Node
// Represents identified risks and compliance violations
// Properties: risk_id, tenant_id, type, score, severity, status, created_at
CREATE CONSTRAINT risk_id_unique IF NOT EXISTS FOR (r:Risk) REQUIRE r.risk_id IS UNIQUE;
CREATE INDEX risk_tenant_idx IF NOT EXISTS FOR (r:Risk) ON (r.tenant_id);
CREATE INDEX risk_type_idx IF NOT EXISTS FOR (r:Risk) ON (r.type);
CREATE INDEX risk_score_idx IF NOT EXISTS FOR (r:Risk) ON (r.score);
CREATE INDEX risk_severity_idx IF NOT EXISTS FOR (r:Risk) ON (r.severity);
CREATE INDEX risk_status_idx IF NOT EXISTS FOR (r:Risk) ON (r.status);

// Version Node
// Represents document versions and amendments
// Properties: version_id, tenant_id, version_number, change_type, created_at
CREATE CONSTRAINT version_id_unique IF NOT EXISTS FOR (v:Version) REQUIRE v.version_id IS UNIQUE;
CREATE INDEX version_tenant_idx IF NOT EXISTS FOR (v:Version) ON (v.tenant_id);
CREATE INDEX version_number_idx IF NOT EXISTS FOR (v:Version) ON (v.version_number);
CREATE INDEX version_created_at_idx IF NOT EXISTS FOR (v:Version) ON (v.created_at);

// =============================================================================
// RELATIONSHIP DEFINITIONS
// =============================================================================

// CONTAINS Relationship
// Document -[CONTAINS]-> Clause
// Document -[CONTAINS]-> Entity
// Properties: span_start, span_end, page_number, extracted_at
CREATE INDEX contains_span_idx IF NOT EXISTS FOR ()-[r:CONTAINS]-() ON (r.span_start, r.span_end);
CREATE INDEX contains_page_idx IF NOT EXISTS FOR ()-[r:CONTAINS]-() ON (r.page_number);
CREATE INDEX contains_extracted_at_idx IF NOT EXISTS FOR ()-[r:CONTAINS]-() ON (r.extracted_at);

// REFERENCES Relationship
// Clause -[REFERENCES]-> Entity (clause mentions entity)
// Entity -[REFERENCES]-> Entity (entity refers to another entity)
// Document -[REFERENCES]-> Document (document references another document)
// Properties: reference_type, confidence, context
CREATE INDEX references_type_idx IF NOT EXISTS FOR ()-[r:REFERENCES]-() ON (r.reference_type);
CREATE INDEX references_confidence_idx IF NOT EXISTS FOR ()-[r:REFERENCES]-() ON (r.confidence);

// AMENDS Relationship
// Version -[AMENDS]-> Document
// Document -[AMENDS]-> Document (newer document amends older)
// Properties: amendment_type, effective_date, change_summary
CREATE INDEX amends_type_idx IF NOT EXISTS FOR ()-[r:AMENDS]-() ON (r.amendment_type);
CREATE INDEX amends_effective_date_idx IF NOT EXISTS FOR ()-[r:AMENDS]-() ON (r.effective_date);

// VIOLATES Relationship
// Clause -[VIOLATES]-> Risk
// Entity -[VIOLATES]-> Risk
// Document -[VIOLATES]-> Risk
// Properties: violation_type, severity, rule_id, detected_at, confidence
CREATE INDEX violates_type_idx IF NOT EXISTS FOR ()-[r:VIOLATES]-() ON (r.violation_type);
CREATE INDEX violates_severity_idx IF NOT EXISTS FOR ()-[r:VIOLATES]-() ON (r.severity);
CREATE INDEX violates_rule_id_idx IF NOT EXISTS FOR ()-[r:VIOLATES]-() ON (r.rule_id);
CREATE INDEX violates_detected_at_idx IF NOT EXISTS FOR ()-[r:VIOLATES]-() ON (r.detected_at);

// RELATED_TO Relationship
// Entity -[RELATED_TO]-> Entity (semantic/business relationship)
// Clause -[RELATED_TO]-> Clause (related clauses across documents)
// Document -[RELATED_TO]-> Document (business relationship between documents)
// Properties: relationship_type, strength, discovered_at, source
CREATE INDEX related_to_type_idx IF NOT EXISTS FOR ()-[r:RELATED_TO]-() ON (r.relationship_type);
CREATE INDEX related_to_strength_idx IF NOT EXISTS FOR ()-[r:RELATED_TO]-() ON (r.strength);
CREATE INDEX related_to_source_idx IF NOT EXISTS FOR ()-[r:RELATED_TO]-() ON (r.source);

// =============================================================================
// CONSTRAINT DEFINITIONS
// =============================================================================

// Ensure tenant isolation at the relationship level
// All relationships must connect nodes within the same tenant
CREATE CONSTRAINT tenant_consistency_contains IF NOT EXISTS
FOR ()-[r:CONTAINS]-()
REQUIRE EXISTS {
    MATCH (a)-[r]-(b)
    WHERE a.tenant_id = b.tenant_id
};

CREATE CONSTRAINT tenant_consistency_references IF NOT EXISTS
FOR ()-[r:REFERENCES]-()
REQUIRE EXISTS {
    MATCH (a)-[r]-(b)
    WHERE a.tenant_id = b.tenant_id
};

CREATE CONSTRAINT tenant_consistency_amends IF NOT EXISTS
FOR ()-[r:AMENDS]-()
REQUIRE EXISTS {
    MATCH (a)-[r]-(b)
    WHERE a.tenant_id = b.tenant_id
};

CREATE CONSTRAINT tenant_consistency_violates IF NOT EXISTS
FOR ()-[r:VIOLATES]-()
REQUIRE EXISTS {
    MATCH (a)-[r]-(b)
    WHERE a.tenant_id = b.tenant_id
};

CREATE CONSTRAINT tenant_consistency_related_to IF NOT EXISTS
FOR ()-[r:RELATED_TO]-()
REQUIRE EXISTS {
    MATCH (a)-[r]-(b)
    WHERE a.tenant_id = b.tenant_id
};

// =============================================================================
// COMPOSITE INDEXES FOR COMMON QUERIES
// =============================================================================

// Multi-tenant document queries
CREATE INDEX document_tenant_type_status IF NOT EXISTS
FOR (d:Document) ON (d.tenant_id, d.type, d.status);

// Risk analysis queries
CREATE INDEX risk_tenant_severity_status IF NOT EXISTS
FOR (r:Risk) ON (r.tenant_id, r.severity, r.status);

// Entity resolution queries
CREATE INDEX entity_tenant_type_normalized IF NOT EXISTS
FOR (e:Entity) ON (e.tenant_id, e.type, e.normalized_value);

// Compliance tracking queries
CREATE INDEX clause_tenant_type_risk IF NOT EXISTS
FOR (c:Clause) ON (c.tenant_id, c.type, c.risk_score);

// Version tracking queries
CREATE INDEX version_tenant_document_number IF NOT EXISTS
FOR (v:Version) ON (v.tenant_id, v.original_document_id, v.version_number);

// =============================================================================
// UTILITY PROCEDURES
// =============================================================================

// Procedure to create a new document with validation
CREATE OR REPLACE PROCEDURE createDocumentNode(
    documentId STRING,
    tenantId STRING,
    documentType STRING,
    status STRING,
    metadata MAP
)
CALL {
    CREATE (d:Document {
        document_id: documentId,
        tenant_id: tenantId,
        type: documentType,
        status: status,
        created_at: datetime(),
        updated_at: datetime(),
        metadata: metadata
    })
    RETURN d
};

// Procedure to establish entity resolution
CREATE OR REPLACE PROCEDURE resolveEntities(
    tenantId STRING,
    entityType STRING,
    similarityThreshold FLOAT
)
CALL {
    MATCH (e1:Entity {tenant_id: tenantId, type: entityType})
    MATCH (e2:Entity {tenant_id: tenantId, type: entityType})
    WHERE e1.entity_id <> e2.entity_id
    AND apoc.text.jaroWinkler(e1.normalized_value, e2.normalized_value) > similarityThreshold
    MERGE (e1)-[r:RELATED_TO]-(e2)
    SET r.relationship_type = 'DUPLICATE_CANDIDATE',
        r.strength = apoc.text.jaroWinkler(e1.normalized_value, e2.normalized_value),
        r.discovered_at = datetime(),
        r.source = 'AUTO_RESOLUTION'
};

// Procedure to propagate risk scores
CREATE OR REPLACE PROCEDURE propagateRiskScores(
    tenantId STRING,
    maxDepth INTEGER
)
CALL {
    MATCH (r:Risk {tenant_id: tenantId, status: 'ACTIVE'})
    MATCH (r)<-[:VIOLATES]-(source)
    MATCH (source)-[:RELATED_TO*1..maxDepth]-(target)
    WHERE target.tenant_id = tenantId
    WITH r, target,
         r.score * (0.8 ^ length([]-[:RELATED_TO*1..maxDepth]-()) ) as propagated_score
    WHERE propagated_score > 0.1
    MERGE (target)-[:VIOLATES]->(propagated_risk:Risk {
        risk_id: apoc.create.uuid(),
        tenant_id: tenantId,
        type: 'PROPAGATED_' + r.type,
        score: propagated_score,
        severity: CASE
            WHEN propagated_score > 0.8 THEN 'HIGH'
            WHEN propagated_score > 0.5 THEN 'MEDIUM'
            ELSE 'LOW'
        END,
        status: 'ACTIVE',
        created_at: datetime(),
        source_risk_id: r.risk_id
    })
};

// =============================================================================
// SAMPLE QUERIES FOR VALIDATION
// =============================================================================

// Query 1: Find all high-risk clauses for a tenant
// MATCH (d:Document {tenant_id: $tenantId})-[:CONTAINS]->(c:Clause)
// WHERE c.risk_score > 0.7
// RETURN d.document_id, c.clause_id, c.type, c.risk_score
// ORDER BY c.risk_score DESC;

// Query 2: Find documents with compliance violations
// MATCH (d:Document {tenant_id: $tenantId})-[:VIOLATES]->(r:Risk {type: 'COMPLIANCE'})
// WHERE r.status = 'ACTIVE'
// RETURN d.document_id, r.risk_id, r.score, r.severity;

// Query 3: Entity relationship discovery
// MATCH (e1:Entity {tenant_id: $tenantId})-[:RELATED_TO]-(e2:Entity)
// WHERE e1.type = $entityType
// RETURN e1.value, e2.value,
//        [(e1)-[r:RELATED_TO]-(e2) | r.relationship_type] as relationships;

// Query 4: Document amendment chain
// MATCH path = (original:Document)-[:AMENDS*]->(latest:Document)
// WHERE original.tenant_id = $tenantId
//   AND original.document_id = $originalDocId
// RETURN [node in nodes(path) | {id: node.document_id, version: node.version_number}] as amendment_chain;

// Query 5: Risk propagation analysis
// MATCH (source)-[:VIOLATES]->(r:Risk {tenant_id: $tenantId})
// MATCH (source)-[:RELATED_TO*1..3]-(target)
// WHERE r.score > $minRiskScore
// RETURN source, target, r.type, r.score,
//        length([]-[:RELATED_TO*1..3]-()) as propagation_distance;

// =============================================================================
// TENANT ISOLATION VALIDATION
// =============================================================================

// Function to validate tenant isolation
CREATE OR REPLACE FUNCTION validateTenantIsolation(tenantId STRING)
RETURNS BOOLEAN
LANGUAGE javascript AS '
    var query = `
        MATCH (n) WHERE n.tenant_id <> $tenantId
        MATCH (m) WHERE m.tenant_id = $tenantId
        MATCH (n)-[r]-(m)
        RETURN count(r) as cross_tenant_relationships
    `;
    var result = neo4j.run(query, {tenantId: tenantId});
    return result.records[0].get("cross_tenant_relationships") === 0;
';

// =============================================================================
// PERFORMANCE MONITORING QUERIES
// =============================================================================

// Query to monitor graph size by tenant
// MATCH (n) WHERE n.tenant_id = $tenantId
// RETURN labels(n)[0] as node_type, count(n) as count
// UNION
// MATCH ()-[r]-() WHERE startNode(r).tenant_id = $tenantId
// RETURN type(r) as relationship_type, count(r) as count;

// Query to find high-degree nodes (potential bottlenecks)
// MATCH (n {tenant_id: $tenantId})
// WITH n, [(n)-[]->() | 1] as out_degree, [(n)<-[]-() | 1] as in_degree
// WHERE size(out_degree) + size(in_degree) > 100
// RETURN n, size(out_degree) + size(in_degree) as total_degree
// ORDER BY total_degree DESC;

// End of Neo4j Knowledge Graph Schema
