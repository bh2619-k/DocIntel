-- Real-Time Document Intelligence & Compliance Engine
-- PostgreSQL Schema Definition
-- Version: 1.0
-- Date: 2026-02-03

-- Enable necessary extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";
CREATE EXTENSION IF NOT EXISTS "btree_gin";

-- Create custom types
CREATE TYPE document_status AS ENUM ('QUEUED', 'PREPROCESSING', 'AI_PROCESSING', 'ANALYSIS', 'GRAPH_UPDATE', 'COMPLETED', 'FAILED', 'RETRY');
CREATE TYPE document_type AS ENUM ('CONTRACT', 'INVOICE', 'COMPLIANCE', 'AMENDMENT', 'OTHER');
CREATE TYPE entity_type AS ENUM ('PERSON', 'ORGANIZATION', 'DATE', 'AMOUNT', 'ADDRESS', 'CLAUSE_REFERENCE', 'LEGAL_ENTITY');
CREATE TYPE alert_severity AS ENUM ('LOW', 'MEDIUM', 'HIGH', 'CRITICAL');
CREATE TYPE alert_status AS ENUM ('ACTIVE', 'ACKNOWLEDGED', 'RESOLVED', 'SUPPRESSED');
CREATE TYPE user_role AS ENUM ('LEGAL', 'FINANCE', 'ADMIN');
CREATE TYPE compliance_rule_status AS ENUM ('ACTIVE', 'DRAFT', 'DEPRECATED');
CREATE TYPE checkpoint_stage AS ENUM ('PREPROCESSING', 'AI_EXTRACTION', 'ANALYSIS', 'GRAPH_UPDATE', 'ALERT_GENERATION');

-- Tenants table (multi-tenant isolation)
CREATE TABLE tenants (
    tenant_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(255) NOT NULL,
    subdomain VARCHAR(100) UNIQUE NOT NULL,
    plan_type VARCHAR(50) NOT NULL DEFAULT 'standard',
    max_monthly_documents INTEGER DEFAULT 10000,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    is_active BOOLEAN DEFAULT true,
    encryption_key_id VARCHAR(255), -- For customer-managed keys
    settings JSONB DEFAULT '{}'::jsonb
);

-- Users table
CREATE TABLE users (
    user_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    tenant_id UUID NOT NULL REFERENCES tenants(tenant_id) ON DELETE CASCADE,
    email VARCHAR(255) NOT NULL,
    name VARCHAR(255) NOT NULL,
    role user_role NOT NULL,
    is_active BOOLEAN DEFAULT true,
    last_login_at TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    preferences JSONB DEFAULT '{}'::jsonb,
    UNIQUE(tenant_id, email)
);

-- Documents table (main entity)
CREATE TABLE documents (
    document_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    tenant_id UUID NOT NULL REFERENCES tenants(tenant_id) ON DELETE CASCADE,
    uploaded_by UUID REFERENCES users(user_id),
    original_filename VARCHAR(500) NOT NULL,
    file_size_bytes BIGINT NOT NULL,
    mime_type VARCHAR(100) NOT NULL,
    document_type document_type,
    language_code VARCHAR(10) DEFAULT 'en',
    page_count INTEGER,
    status document_status DEFAULT 'QUEUED',
    priority INTEGER DEFAULT 1, -- 1=Normal, 2=High, 3=Rush
    storage_path VARCHAR(1000) NOT NULL, -- S3/MinIO path
    processing_started_at TIMESTAMP WITH TIME ZONE,
    processing_completed_at TIMESTAMP WITH TIME ZONE,
    processing_time_seconds INTEGER,
    ocr_required BOOLEAN DEFAULT true,
    digital_text_extracted BOOLEAN DEFAULT false,
    error_message TEXT,
    retry_count INTEGER DEFAULT 0,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    metadata JSONB DEFAULT '{}'::jsonb
);

-- Entities table (extracted from documents)
CREATE TABLE entities (
    entity_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    tenant_id UUID NOT NULL REFERENCES tenants(tenant_id) ON DELETE CASCADE,
    document_id UUID NOT NULL REFERENCES documents(document_id) ON DELETE CASCADE,
    type entity_type NOT NULL,
    value TEXT NOT NULL,
    normalized_value TEXT, -- Standardized form for matching
    confidence DECIMAL(5,4) NOT NULL CHECK (confidence >= 0 AND confidence <= 1),
    page_number INTEGER,
    span_start INTEGER, -- Character offset in document
    span_end INTEGER,   -- Character offset in document
    extraction_method VARCHAR(50) NOT NULL, -- 'NER', 'RULE_BASED', 'MANUAL'
    extracted_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    reviewed_by UUID REFERENCES users(user_id),
    is_manually_corrected BOOLEAN DEFAULT false,
    context_snippet TEXT, -- Surrounding text for human review
    INDEX(tenant_id, document_id),
    INDEX(tenant_id, type, normalized_value),
    INDEX(confidence),
    INDEX(extracted_at)
);

-- Clauses table (legal/contract clauses)
CREATE TABLE clauses (
    clause_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    tenant_id UUID NOT NULL REFERENCES tenants(tenant_id) ON DELETE CASCADE,
    document_id UUID NOT NULL REFERENCES documents(document_id) ON DELETE CASCADE,
    clause_type VARCHAR(100) NOT NULL, -- 'TERMINATION', 'PAYMENT', 'LIABILITY', etc.
    content TEXT NOT NULL,
    normalized_content TEXT, -- Cleaned version for comparison
    page_numbers INTEGER[], -- Pages where clause spans
    span_start INTEGER,
    span_end INTEGER,
    confidence DECIMAL(5,4) NOT NULL CHECK (confidence >= 0 AND confidence <= 1),
    embedding_vector_id VARCHAR(255), -- Reference to vector store
    extracted_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    reviewed_by UUID REFERENCES users(user_id),
    is_standard_clause BOOLEAN DEFAULT false,
    risk_score DECIMAL(3,2) DEFAULT 0.0,
    INDEX(tenant_id, document_id),
    INDEX(tenant_id, clause_type),
    INDEX(confidence),
    INDEX(risk_score DESC)
);

-- Processing checkpoints (for resumption after failure)
CREATE TABLE processing_checkpoints (
    checkpoint_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    tenant_id UUID NOT NULL REFERENCES tenants(tenant_id) ON DELETE CASCADE,
    document_id UUID NOT NULL REFERENCES documents(document_id) ON DELETE CASCADE,
    stage checkpoint_stage NOT NULL,
    status VARCHAR(20) NOT NULL DEFAULT 'PENDING', -- PENDING, COMPLETED, FAILED
    started_at TIMESTAMP WITH TIME ZONE,
    completed_at TIMESTAMP WITH TIME ZONE,
    error_message TEXT,
    output_metadata JSONB DEFAULT '{}'::jsonb,
    UNIQUE(document_id, stage)
);

-- Compliance rules (versioned rule engine)
CREATE TABLE compliance_rules (
    rule_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    tenant_id UUID NOT NULL REFERENCES tenants(tenant_id) ON DELETE CASCADE,
    name VARCHAR(255) NOT NULL,
    description TEXT,
    rule_type VARCHAR(50) NOT NULL, -- 'MANDATORY_CLAUSE', 'PROHIBITED_TERM', 'FORMAT_CHECK'
    document_types document_type[] NOT NULL,
    rule_definition JSONB NOT NULL, -- JSON rule structure
    severity alert_severity NOT NULL,
    status compliance_rule_status DEFAULT 'DRAFT',
    effective_date DATE NOT NULL,
    expiry_date DATE,
    version INTEGER NOT NULL DEFAULT 1,
    created_by UUID NOT NULL REFERENCES users(user_id),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    approved_by UUID REFERENCES users(user_id),
    approved_at TIMESTAMP WITH TIME ZONE,
    INDEX(tenant_id, status, effective_date),
    INDEX(document_types),
    UNIQUE(tenant_id, name, version)
);

-- Alerts table
CREATE TABLE alerts (
    alert_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    tenant_id UUID NOT NULL REFERENCES tenants(tenant_id) ON DELETE CASCADE,
    document_id UUID NOT NULL REFERENCES documents(document_id) ON DELETE CASCADE,
    entity_id UUID REFERENCES entities(entity_id),
    clause_id UUID REFERENCES clauses(clause_id),
    compliance_rule_id UUID REFERENCES compliance_rules(rule_id),
    alert_type VARCHAR(50) NOT NULL, -- 'COMPLIANCE_VIOLATION', 'ANOMALY', 'RISK_FLAG'
    severity alert_severity NOT NULL,
    status alert_status DEFAULT 'ACTIVE',
    title VARCHAR(255) NOT NULL,
    message TEXT NOT NULL,
    explanation TEXT NOT NULL, -- Human-readable justification
    confidence DECIMAL(5,4) NOT NULL,
    span_start INTEGER, -- Document span that triggered alert
    span_end INTEGER,
    model_version VARCHAR(50), -- Model/rule version that generated alert
    assigned_to_role user_role,
    assigned_to_user UUID REFERENCES users(user_id),
    acknowledged_by UUID REFERENCES users(user_id),
    acknowledged_at TIMESTAMP WITH TIME ZONE,
    resolved_by UUID REFERENCES users(user_id),
    resolved_at TIMESTAMP WITH TIME ZONE,
    resolution_notes TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    cooldown_until TIMESTAMP WITH TIME ZONE, -- Prevents duplicate alerts
    INDEX(tenant_id, status, created_at),
    INDEX(tenant_id, assigned_to_role),
    INDEX(severity, status),
    INDEX(cooldown_until)
);

-- Document versions (for contract amendments and tracking)
CREATE TABLE document_versions (
    version_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    tenant_id UUID NOT NULL REFERENCES tenants(tenant_id) ON DELETE CASCADE,
    original_document_id UUID NOT NULL REFERENCES documents(document_id) ON DELETE CASCADE,
    amended_document_id UUID REFERENCES documents(document_id) ON DELETE SET NULL,
    version_number INTEGER NOT NULL,
    change_summary TEXT,
    diff_analysis JSONB, -- Structured diff information
    created_by UUID REFERENCES users(user_id),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    INDEX(tenant_id, original_document_id, version_number),
    UNIQUE(original_document_id, version_number)
);

-- Risk scores (for entities and relationships)
CREATE TABLE risk_scores (
    risk_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    tenant_id UUID NOT NULL REFERENCES tenants(tenant_id) ON DELETE CASCADE,
    document_id UUID NOT NULL REFERENCES documents(document_id) ON DELETE CASCADE,
    entity_id UUID REFERENCES entities(entity_id),
    clause_id UUID REFERENCES clauses(clause_id),
    risk_type VARCHAR(50) NOT NULL, -- 'COMPLIANCE', 'FINANCIAL', 'LEGAL', 'OPERATIONAL'
    score DECIMAL(5,4) NOT NULL CHECK (score >= 0 AND score <= 1),
    factors JSONB NOT NULL, -- Contributing risk factors
    calculated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    model_version VARCHAR(50) NOT NULL,
    INDEX(tenant_id, risk_type, score DESC),
    INDEX(calculated_at),
    UNIQUE(document_id, entity_id, clause_id, risk_type)
);

-- Audit log for all changes
CREATE TABLE audit_logs (
    log_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    tenant_id UUID NOT NULL REFERENCES tenants(tenant_id) ON DELETE CASCADE,
    user_id UUID REFERENCES users(user_id),
    action VARCHAR(50) NOT NULL,
    resource_type VARCHAR(50) NOT NULL,
    resource_id UUID NOT NULL,
    old_values JSONB,
    new_values JSONB,
    ip_address INET,
    user_agent TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    INDEX(tenant_id, created_at),
    INDEX(resource_type, resource_id),
    INDEX(user_id, created_at)
);

-- Create indexes for performance
CREATE INDEX CONCURRENTLY idx_documents_tenant_status ON documents(tenant_id, status);
CREATE INDEX CONCURRENTLY idx_documents_tenant_type ON documents(tenant_id, document_type);
CREATE INDEX CONCURRENTLY idx_documents_created_at ON documents(created_at);
CREATE INDEX CONCURRENTLY idx_entities_tenant_doc ON entities(tenant_id, document_id);
CREATE INDEX CONCURRENTLY idx_entities_type_value ON entities(type, value);
CREATE INDEX CONCURRENTLY idx_clauses_tenant_doc ON clauses(tenant_id, document_id);
CREATE INDEX CONCURRENTLY idx_alerts_tenant_status ON alerts(tenant_id, status);
CREATE INDEX CONCURRENTLY idx_processing_checkpoints_doc ON processing_checkpoints(document_id);

-- Create GIN indexes for JSONB columns
CREATE INDEX CONCURRENTLY idx_documents_metadata_gin ON documents USING GIN(metadata);
CREATE INDEX CONCURRENTLY idx_compliance_rules_definition_gin ON compliance_rules USING GIN(rule_definition);
CREATE INDEX CONCURRENTLY idx_risk_scores_factors_gin ON risk_scores USING GIN(factors);

-- Row Level Security (RLS) for multi-tenant isolation
ALTER TABLE tenants ENABLE ROW LEVEL SECURITY;
ALTER TABLE users ENABLE ROW LEVEL SECURITY;
ALTER TABLE documents ENABLE ROW LEVEL SECURITY;
ALTER TABLE entities ENABLE ROW LEVEL SECURITY;
ALTER TABLE clauses ENABLE ROW LEVEL SECURITY;
ALTER TABLE processing_checkpoints ENABLE ROW LEVEL SECURITY;
ALTER TABLE compliance_rules ENABLE ROW LEVEL SECURITY;
ALTER TABLE alerts ENABLE ROW LEVEL SECURITY;
ALTER TABLE document_versions ENABLE ROW LEVEL SECURITY;
ALTER TABLE risk_scores ENABLE ROW LEVEL SECURITY;
ALTER TABLE audit_logs ENABLE ROW LEVEL SECURITY;

-- RLS Policies (to be activated with proper role setup)
-- Example policy for documents table:
-- CREATE POLICY tenant_isolation_documents ON documents
--     FOR ALL TO application_user
--     USING (tenant_id = current_setting('app.tenant_id')::UUID);

-- Functions and triggers
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Apply updated_at triggers
CREATE TRIGGER update_tenants_updated_at BEFORE UPDATE ON tenants FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_users_updated_at BEFORE UPDATE ON users FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_documents_updated_at BEFORE UPDATE ON documents FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Function to calculate document processing time
CREATE OR REPLACE FUNCTION calculate_processing_time()
RETURNS TRIGGER AS $$
BEGIN
    IF NEW.status = 'COMPLETED' AND OLD.status != 'COMPLETED' AND NEW.processing_started_at IS NOT NULL THEN
        NEW.processing_completed_at = CURRENT_TIMESTAMP;
        NEW.processing_time_seconds = EXTRACT(EPOCH FROM (NEW.processing_completed_at - NEW.processing_started_at));
    END IF;
    RETURN NEW;
END;
$$ language 'plpgsql';

CREATE TRIGGER calculate_document_processing_time
    BEFORE UPDATE ON documents
    FOR EACH ROW
    EXECUTE FUNCTION calculate_processing_time();

-- Views for common queries
CREATE VIEW document_processing_summary AS
SELECT
    d.tenant_id,
    d.document_type,
    d.status,
    COUNT(*) as count,
    AVG(d.processing_time_seconds) as avg_processing_time,
    MAX(d.processing_time_seconds) as max_processing_time,
    DATE_TRUNC('hour', d.created_at) as hour_bucket
FROM documents d
WHERE d.created_at >= CURRENT_DATE - INTERVAL '7 days'
GROUP BY d.tenant_id, d.document_type, d.status, DATE_TRUNC('hour', d.created_at);

CREATE VIEW active_alerts_by_severity AS
SELECT
    a.tenant_id,
    a.severity,
    a.alert_type,
    a.assigned_to_role,
    COUNT(*) as alert_count,
    MIN(a.created_at) as oldest_alert,
    MAX(a.created_at) as newest_alert
FROM alerts a
WHERE a.status = 'ACTIVE'
GROUP BY a.tenant_id, a.severity, a.alert_type, a.assigned_to_role;

-- Grant permissions (to be customized per deployment)
-- GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO application_user;
-- GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO application_user;
-- GRANT EXECUTE ON ALL FUNCTIONS IN SCHEMA public TO application_user;
