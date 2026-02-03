# Infrastructure Module - Core Data & Storage

This module contains the foundational data and storage components for the Real-Time Document Intelligence & Compliance Engine.

## Overview

The Infrastructure module provides the core data storage and configuration management capabilities that all other modules depend on. It implements a multi-tenant, multi-deployment architecture supporting both cloud and on-premise environments.

## Components

### 1. Database Layer (`database/`)

#### PostgreSQL Schema (`postgresql_schema.sql`)

- **Purpose**: Primary relational database for documents, entities, checkpoints, compliance rules, and user management
- **Key Features**:
  - Multi-tenant isolation with Row-Level Security (RLS)
  - Comprehensive indexing for performance
  - Audit logging and change tracking
  - Processing checkpoints for fault tolerance
  - Compliance rule versioning
- **Tables**:
  - `tenants`, `users`, `documents`, `entities`, `clauses`
  - `processing_checkpoints`, `compliance_rules`, `alerts`
  - `document_versions`, `risk_scores`, `audit_logs`

#### Neo4j Knowledge Graph (`neo4j_schema.cypher`)

- **Purpose**: 5-node, 5-edge knowledge graph for entity relationships and risk propagation
- **Schema**:
  - **Nodes**: Document, Clause, Entity, Risk, Version
  - **Relationships**: CONTAINS, REFERENCES, AMENDS, VIOLATES, RELATED_TO
- **Features**:
  - Tenant isolation constraints
  - Risk propagation procedures
  - Entity resolution utilities
  - Performance optimization indexes

### 2. Cache Layer (`cache/`)

#### Redis Configuration (`redis.conf`, `redis_manager.py`)

- **Purpose**: Priority queues, session management, and pub/sub messaging
- **Features**:
  - Stream-based priority queues (High, Normal, Retry)
  - Session and cache management
  - Real-time event pub/sub
  - Processing checkpoints
  - Rate limiting
  - Dead letter queue handling

### 3. Storage Layer (`storage/`)

#### Object Storage (`storage_manager.py`)

- **Purpose**: Unified S3/MinIO interface for document storage
- **Buckets**:
  - `documents-raw`: Original uploaded files
  - `documents-processed`: OCR outputs and preprocessed text
  - `documents-checkpoints`: Processing stage artifacts
  - `models`: ML model artifacts
  - `backups`: System backups
  - `temp`: Temporary files (auto-cleanup)
- **Features**:
  - Multi-tenant path isolation
  - Encryption at rest
  - Lifecycle policies
  - Presigned URLs
  - Metadata management

#### Vector Store (`vector_store.py`)

- **Purpose**: Clause-level embeddings for semantic search
- **Backends**: Pinecone (cloud) and Weaviate (on-premise)
- **Features**:
  - Tenant-isolated indexes/classes
  - Clause-level embedding storage
  - Semantic similarity search
  - Metadata filtering
  - Batch operations

### 4. Docker Infrastructure (`docker/`)

#### Local Development Environment (`docker-compose.yml`)

- **Services**: PostgreSQL, Neo4j, Redis, MinIO, Weaviate
- **Monitoring**: Flower (Celery), Redis Commander, pgAdmin
- **Networking**: Isolated bridge network
- **Features**:
  - Health checks for all services
  - Automatic bucket initialization
  - Development-friendly configuration
  - Volume persistence

### 5. Configuration Management (`config.py`)

#### Centralized Settings

- **Purpose**: Environment-aware configuration with tenant overrides
- **Features**:
  - Multi-environment support (dev, staging, prod, on-premise)
  - Type-safe configuration objects
  - Feature flags
  - Tenant-specific overrides
  - Security settings management

## Quick Start

### 1. Environment Setup

```bash
# Copy environment template
cp .env.example .env

# Edit environment variables
nano .env
```

### 2. Start Infrastructure

```bash
# Start all services
cd infrastructure/docker
docker-compose up -d

# Check service health
docker-compose ps
```

### 3. Initialize Databases

```bash
# PostgreSQL schema is auto-loaded
# Neo4j schema needs manual execution
docker exec -it docintell-neo4j cypher-shell -u neo4j -p dev_password_123 -f /var/lib/neo4j/import/schema.cypher

# MinIO buckets are auto-created
```

### 4. Verify Setup

```bash
# Check database connectivity
python -c "from infrastructure.config import get_config; print(get_config().database.connection_url)"

# Test Redis connection
docker exec -it docintell-redis redis-cli ping

# Verify MinIO buckets
docker exec -it docintell-minio mc ls minio
```

## Service URLs (Development)

| Service         | URL                   | Credentials                              |
| --------------- | --------------------- | ---------------------------------------- |
| PostgreSQL      | `localhost:5432`      | `docintell_user:dev_password_123`        |
| Neo4j Web UI    | http://localhost:7474 | `neo4j:dev_password_123`                 |
| Redis           | `localhost:6379`      | `:dev_password_123`                      |
| MinIO Console   | http://localhost:9001 | `admin:dev_password_123`                 |
| Weaviate        | http://localhost:8080 | No auth (dev only)                       |
| pgAdmin         | http://localhost:8082 | `admin@docintell.local:dev_password_123` |
| Redis Commander | http://localhost:8081 | Auto-configured                          |

## Configuration

### Environment Variables

Key environment variables (see `.env.example` for full list):

```bash
# Deployment
DEPLOYMENT_MODE=development
ENVIRONMENT=local

# Databases
POSTGRES_PASSWORD=dev_password_123
NEO4J_PASSWORD=dev_password_123
REDIS_PASSWORD=dev_password_123

# Storage
STORAGE_BACKEND=minio
MINIO_ROOT_USER=admin
MINIO_ROOT_PASSWORD=dev_password_123

# Vector Store
VECTOR_STORE_BACKEND=weaviate
WEAVIATE_URL=http://localhost:8080

# Security
JWT_SECRET_KEY=your-super-secret-jwt-key-change-in-production
```

### Multi-Tenant Configuration

Tenant-specific overrides use the format:

```bash
TENANT_{TENANT_ID}_{SETTING_NAME}=value
```

Example:

```bash
TENANT_acme_corp_MAX_MONTHLY_DOCUMENTS=50000
```

## Data Flow

1. **Document Upload** → MinIO raw storage
2. **Processing Jobs** → Redis priority queues
3. **Checkpoints** → PostgreSQL + Redis
4. **Entities/Clauses** → PostgreSQL + Neo4j
5. **Embeddings** → Weaviate/Pinecone
6. **Alerts** → Redis pub/sub
7. **Audit Trail** → PostgreSQL audit_logs

## Security Features

- **Multi-tenant isolation** at database, storage, and vector store levels
- **Encryption at rest** for all storage (configurable KMS)
- **Row-level security** in PostgreSQL
- **Tenant-scoped** Neo4j constraints
- **Namespaced** vector store indexes
- **Audit logging** for all changes
- **Rate limiting** at API and Redis levels

## Monitoring

- **Health checks** for all services
- **Prometheus metrics** (when enabled)
- **Structured logging** with JSON format
- **Service dashboards** via web UIs
- **Queue monitoring** via Flower

## Backup Strategy

- **PostgreSQL**: WAL shipping + daily snapshots
- **Neo4j**: Causal cluster replication + snapshots
- **Redis**: AOF persistence + RDB snapshots
- **MinIO**: Cross-region replication (production)
- **Vector Store**: Platform-specific backup (Pinecone managed, Weaviate snapshots)

## Scaling Considerations

### Horizontal Scaling

- **PostgreSQL**: Read replicas + connection pooling
- **Neo4j**: Causal cluster (3+ nodes)
- **Redis**: Sentinel + clustering
- **MinIO**: Distributed mode (4+ nodes)
- **Weaviate**: Multi-node cluster

### Performance Tuning

- Database connection pooling
- Redis pipelining for batch operations
- MinIO multipart uploads
- Vector store batch operations
- Proper indexing strategies

## Troubleshooting

### Common Issues

1. **Database Connection Errors**

   ```bash
   # Check PostgreSQL logs
   docker logs docintell-postgres

   # Test connection
   docker exec -it docintell-postgres psql -U docintell_user -d document_intelligence
   ```

2. **Redis Connection Issues**

   ```bash
   # Check Redis logs
   docker logs docintell-redis

   # Test connection
   docker exec -it docintell-redis redis-cli ping
   ```

3. **MinIO Access Issues**

   ```bash
   # Check MinIO logs
   docker logs docintell-minio

   # List buckets
   docker exec -it docintell-minio-init mc ls minio
   ```

4. **Neo4j Memory Issues**
   ```bash
   # Increase heap size in docker-compose.yml
   NEO4J_dbms_memory_heap_max__size: 4G
   ```

### Health Checks

```python
from infrastructure.config import get_config
from infrastructure.cache.redis_manager import RedisManager
from infrastructure.storage.storage_manager import StorageManager

# Test configuration loading
config = get_config()
print(f"Deployment mode: {config.deployment_mode}")

# Test Redis connectivity
redis_mgr = RedisManager()
health = redis_mgr.health_check()
print(f"Redis health: {health['status']}")
```

## Dependencies

See `requirements.txt` for full Python dependency list. Key dependencies:

- `psycopg[binary]` - PostgreSQL driver
- `neo4j` - Neo4j driver
- `redis` - Redis client
- `boto3` - AWS S3 client
- `minio` - MinIO client
- `pinecone-client` - Pinecone vector store
- `weaviate-client` - Weaviate vector store

## Next Steps

After completing the infrastructure setup:

1. **Module 2**: Document Processing Pipeline (OCR, preprocessing)
2. **Module 3**: AI/ML Processing (NER, clause extraction)
3. **Module 4**: Compliance & Analysis Engine
4. **Module 5**: Knowledge Graph Operations
5. **Module 6**: Alert & Workflow Engine
6. **Module 7**: API Gateway & Integration Layer
7. **Module 8**: Observability & Operations

Each subsequent module builds upon this infrastructure foundation.
