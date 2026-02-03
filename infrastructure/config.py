"""
Real-Time Document Intelligence & Compliance Engine
Configuration Management Module
Version: 1.0
Date: 2026-02-03

This module provides centralized configuration management for all system components
with support for multiple deployment environments and tenant-specific overrides.
"""

import os
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class DeploymentMode(Enum):
    """Deployment environment modes"""

    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    ON_PREMISE = "on-premise"


class StorageBackend(Enum):
    """Storage backend options"""

    S3 = "s3"
    MINIO = "minio"


class VectorBackend(Enum):
    """Vector store backend options"""

    PINECONE = "pinecone"
    WEAVIATE = "weaviate"


@dataclass
class DatabaseConfig:
    """Database configuration settings"""

    host: str
    port: int
    database: str
    username: str
    password: str
    max_connections: int = 100
    pool_size: int = 20

    @property
    def connection_url(self) -> str:
        return f"postgresql://{self.username}:{self.password}@{self.host}:{self.port}/{self.database}"


@dataclass
class RedisConfig:
    """Redis configuration settings"""

    host: str
    port: int
    password: str
    db: int = 0
    max_connections: int = 50

    @property
    def connection_url(self) -> str:
        return f"redis://:{self.password}@{self.host}:{self.port}/{self.db}"


@dataclass
class Neo4jConfig:
    """Neo4j configuration settings"""

    uri: str
    username: str
    password: str
    database: str = "neo4j"
    max_connection_pool_size: int = 50


@dataclass
class StorageConfig:
    """Storage configuration settings"""

    backend: StorageBackend
    bucket_prefix: str
    region: str = "us-east-1"

    # S3-specific settings
    aws_access_key_id: Optional[str] = None
    aws_secret_access_key: Optional[str] = None
    kms_key_id: Optional[str] = None

    # MinIO-specific settings
    minio_endpoint: Optional[str] = None
    minio_access_key: Optional[str] = None
    minio_secret_key: Optional[str] = None
    minio_use_ssl: bool = False


@dataclass
class VectorStoreConfig:
    """Vector store configuration settings"""

    backend: VectorBackend
    dimension: int = 384

    # Pinecone settings
    pinecone_api_key: Optional[str] = None
    pinecone_environment: Optional[str] = None
    pinecone_index_name: str = "document-clauses"

    # Weaviate settings
    weaviate_url: Optional[str] = None
    weaviate_api_key: Optional[str] = None


@dataclass
class ProcessingConfig:
    """Document processing configuration"""

    max_file_size_mb: int = 100
    max_pages_per_document: int = 500
    max_concurrent_uploads: int = 500
    processing_timeout_seconds: int = 1800
    ocr_timeout_seconds: int = 300
    max_retries: int = 3
    retry_delay_seconds: int = 300


@dataclass
class MLConfig:
    """Machine learning model configuration"""

    # Model paths
    document_classifier_model: str = "microsoft/layoutlmv3-base"
    ner_model: str = "dbmdz/bert-large-cased-finetuned-conll03-english"
    clause_extraction_model: str = "google/flan-t5-large"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    # Processing parameters
    ner_confidence_threshold: float = 0.85
    ner_batch_size: int = 32
    clause_extraction_max_length: int = 512
    clause_extraction_batch_size: int = 8
    embedding_batch_size: int = 64

    # Cache settings
    hf_cache_dir: str = "/tmp/docintell/huggingface_cache"
    transformers_cache: str = "/tmp/docintell/transformers_cache"


@dataclass
class SecurityConfig:
    """Security and authentication configuration"""

    jwt_secret_key: str
    jwt_algorithm: str = "HS256"
    jwt_access_token_expire_minutes: int = 60
    jwt_refresh_token_expire_days: int = 30

    # API Security
    api_rate_limit_per_minute: int = 60
    api_rate_limit_per_hour: int = 1000
    api_rate_limit_per_day: int = 10000
    cors_allowed_origins: List[str] = None

    # Data encryption
    encryption_algorithm: str = "AES256"
    encryption_key_rotation_days: int = 90
    data_retention_days: int = 2555  # 7 years


@dataclass
class MonitoringConfig:
    """Monitoring and observability configuration"""

    log_level: str = "INFO"
    log_format: str = "json"
    enable_prometheus_metrics: bool = True
    prometheus_port: int = 8000
    enable_request_tracing: bool = True
    trace_sampling_rate: float = 0.1

    # Alerting
    enable_slack_alerts: bool = False
    slack_webhook_url: Optional[str] = None
    enable_email_alerts: bool = True
    alert_email_from: str = "alerts@docintell.local"
    alert_email_to: str = "admin@docintell.local"


class ConfigManager:
    """
    Central configuration manager for the document intelligence system
    Handles environment-specific settings and tenant overrides
    """

    def __init__(self, env_file: Optional[str] = None):
        """Initialize configuration manager"""
        self.env_file = env_file or ".env"
        self._load_environment()

        # Initialize configuration objects
        self.deployment_mode = DeploymentMode(
            self._get_env("DEPLOYMENT_MODE", "development")
        )
        self.environment = self._get_env("ENVIRONMENT", "local")

        self.database = self._load_database_config()
        self.redis = self._load_redis_config()
        self.neo4j = self._load_neo4j_config()
        self.storage = self._load_storage_config()
        self.vector_store = self._load_vector_store_config()
        self.processing = self._load_processing_config()
        self.ml = self._load_ml_config()
        self.security = self._load_security_config()
        self.monitoring = self._load_monitoring_config()

        # Feature flags
        self.feature_flags = self._load_feature_flags()

        logger.info(
            f"Configuration loaded for {self.deployment_mode.value} environment"
        )

    def _load_environment(self):
        """Load environment variables from .env file"""
        try:
            from dotenv import load_dotenv

            if os.path.exists(self.env_file):
                load_dotenv(self.env_file)
                logger.info(f"Loaded environment from {self.env_file}")
        except ImportError:
            logger.warning(
                "python-dotenv not installed, using system environment variables"
            )

    def _get_env(self, key: str, default: Any = None, required: bool = False) -> Any:
        """Get environment variable with type conversion"""
        value = os.getenv(key, default)

        if required and value is None:
            raise ValueError(f"Required environment variable {key} is not set")

        # Convert string values to appropriate types
        if isinstance(value, str):
            if value.lower() in ("true", "false"):
                return value.lower() == "true"
            elif value.isdigit():
                return int(value)
            elif value.replace(".", "").isdigit():
                return float(value)

        return value

    def _get_env_list(
        self, key: str, default: List[str] = None, separator: str = ","
    ) -> List[str]:
        """Get environment variable as a list"""
        value = self._get_env(key)
        if value is None:
            return default or []

        if isinstance(value, str):
            return [item.strip() for item in value.split(separator) if item.strip()]

        return value

    def _load_database_config(self) -> DatabaseConfig:
        """Load database configuration"""
        return DatabaseConfig(
            host=self._get_env("POSTGRES_HOST", "localhost"),
            port=self._get_env("POSTGRES_PORT", 5432),
            database=self._get_env("POSTGRES_DB", "document_intelligence"),
            username=self._get_env("POSTGRES_USER", "docintell_user"),
            password=self._get_env("POSTGRES_PASSWORD", required=True),
            max_connections=self._get_env("POSTGRES_MAX_CONNECTIONS", 100),
            pool_size=self._get_env("POSTGRES_POOL_SIZE", 20),
        )

    def _load_redis_config(self) -> RedisConfig:
        """Load Redis configuration"""
        return RedisConfig(
            host=self._get_env("REDIS_HOST", "localhost"),
            port=self._get_env("REDIS_PORT", 6379),
            password=self._get_env("REDIS_PASSWORD", required=True),
            db=self._get_env("REDIS_DB", 0),
            max_connections=self._get_env("REDIS_MAX_CONNECTIONS", 50),
        )

    def _load_neo4j_config(self) -> Neo4jConfig:
        """Load Neo4j configuration"""
        return Neo4jConfig(
            uri=self._get_env("NEO4J_URI", "bolt://localhost:7687"),
            username=self._get_env("NEO4J_USER", "neo4j"),
            password=self._get_env("NEO4J_PASSWORD", required=True),
            database=self._get_env("NEO4J_DATABASE", "neo4j"),
            max_connection_pool_size=self._get_env(
                "NEO4J_MAX_CONNECTION_POOL_SIZE", 50
            ),
        )

    def _load_storage_config(self) -> StorageConfig:
        """Load storage configuration"""
        backend = StorageBackend(self._get_env("STORAGE_BACKEND", "minio"))

        return StorageConfig(
            backend=backend,
            bucket_prefix=self._get_env("S3_BUCKET_PREFIX", "docintell"),
            region=self._get_env("AWS_REGION", "us-east-1"),
            aws_access_key_id=self._get_env("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=self._get_env("AWS_SECRET_ACCESS_KEY"),
            kms_key_id=self._get_env("S3_KMS_KEY_ID"),
            minio_endpoint=self._get_env("MINIO_ENDPOINT"),
            minio_access_key=self._get_env("MINIO_ROOT_USER"),
            minio_secret_key=self._get_env("MINIO_ROOT_PASSWORD"),
            minio_use_ssl=self._get_env("MINIO_USE_SSL", False),
        )

    def _load_vector_store_config(self) -> VectorStoreConfig:
        """Load vector store configuration"""
        backend = VectorBackend(self._get_env("VECTOR_STORE_BACKEND", "weaviate"))

        return VectorStoreConfig(
            backend=backend,
            dimension=self._get_env("EMBEDDING_DIMENSION", 384),
            pinecone_api_key=self._get_env("PINECONE_API_KEY"),
            pinecone_environment=self._get_env("PINECONE_ENVIRONMENT"),
            pinecone_index_name=self._get_env(
                "PINECONE_INDEX_NAME", "document-clauses"
            ),
            weaviate_url=self._get_env("WEAVIATE_URL", "http://localhost:8080"),
            weaviate_api_key=self._get_env("WEAVIATE_API_KEY"),
        )

    def _load_processing_config(self) -> ProcessingConfig:
        """Load processing configuration"""
        return ProcessingConfig(
            max_file_size_mb=self._get_env("MAX_FILE_SIZE_MB", 100),
            max_pages_per_document=self._get_env("MAX_PAGES_PER_DOCUMENT", 500),
            max_concurrent_uploads=self._get_env("MAX_CONCURRENT_UPLOADS", 500),
            processing_timeout_seconds=self._get_env(
                "PROCESSING_TIMEOUT_SECONDS", 1800
            ),
            ocr_timeout_seconds=self._get_env("OCR_TIMEOUT_SECONDS", 300),
            max_retries=self._get_env("MAX_RETRIES", 3),
            retry_delay_seconds=self._get_env("RETRY_DELAY", 300),
        )

    def _load_ml_config(self) -> MLConfig:
        """Load ML configuration"""
        return MLConfig(
            document_classifier_model=self._get_env(
                "DOCUMENT_CLASSIFIER_MODEL", "microsoft/layoutlmv3-base"
            ),
            ner_model=self._get_env(
                "NER_MODEL", "dbmdz/bert-large-cased-finetuned-conll03-english"
            ),
            clause_extraction_model=self._get_env(
                "CLAUSE_EXTRACTION_MODEL", "google/flan-t5-large"
            ),
            embedding_model=self._get_env(
                "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
            ),
            ner_confidence_threshold=self._get_env("NER_CONFIDENCE_THRESHOLD", 0.85),
            ner_batch_size=self._get_env("NER_BATCH_SIZE", 32),
            clause_extraction_max_length=self._get_env(
                "CLAUSE_EXTRACTION_MAX_LENGTH", 512
            ),
            clause_extraction_batch_size=self._get_env(
                "CLAUSE_EXTRACTION_BATCH_SIZE", 8
            ),
            embedding_batch_size=self._get_env("EMBEDDING_BATCH_SIZE", 64),
            hf_cache_dir=self._get_env(
                "HF_CACHE_DIR", "/tmp/docintell/huggingface_cache"
            ),
            transformers_cache=self._get_env(
                "TRANSFORMERS_CACHE", "/tmp/docintell/transformers_cache"
            ),
        )

    def _load_security_config(self) -> SecurityConfig:
        """Load security configuration"""
        return SecurityConfig(
            jwt_secret_key=self._get_env("JWT_SECRET_KEY", required=True),
            jwt_algorithm=self._get_env("JWT_ALGORITHM", "HS256"),
            jwt_access_token_expire_minutes=self._get_env(
                "JWT_ACCESS_TOKEN_EXPIRE_MINUTES", 60
            ),
            jwt_refresh_token_expire_days=self._get_env(
                "JWT_REFRESH_TOKEN_EXPIRE_DAYS", 30
            ),
            api_rate_limit_per_minute=self._get_env("API_RATE_LIMIT_PER_MINUTE", 60),
            api_rate_limit_per_hour=self._get_env("API_RATE_LIMIT_PER_HOUR", 1000),
            api_rate_limit_per_day=self._get_env("API_RATE_LIMIT_PER_DAY", 10000),
            cors_allowed_origins=self._get_env_list("CORS_ALLOWED_ORIGINS"),
            encryption_algorithm=self._get_env("ENCRYPTION_ALGORITHM", "AES256"),
            encryption_key_rotation_days=self._get_env(
                "ENCRYPTION_KEY_ROTATION_DAYS", 90
            ),
            data_retention_days=self._get_env("DATA_RETENTION_DAYS", 2555),
        )

    def _load_monitoring_config(self) -> MonitoringConfig:
        """Load monitoring configuration"""
        return MonitoringConfig(
            log_level=self._get_env("LOG_LEVEL", "INFO"),
            log_format=self._get_env("LOG_FORMAT", "json"),
            enable_prometheus_metrics=self._get_env("ENABLE_PROMETHEUS_METRICS", True),
            prometheus_port=self._get_env("PROMETHEUS_PORT", 8000),
            enable_request_tracing=self._get_env("ENABLE_REQUEST_TRACING", True),
            trace_sampling_rate=self._get_env("TRACE_SAMPLING_RATE", 0.1),
            enable_slack_alerts=self._get_env("ENABLE_SLACK_ALERTS", False),
            slack_webhook_url=self._get_env("SLACK_WEBHOOK_URL"),
            enable_email_alerts=self._get_env("ENABLE_EMAIL_ALERTS", True),
            alert_email_from=self._get_env(
                "ALERT_EMAIL_FROM", "alerts@docintell.local"
            ),
            alert_email_to=self._get_env("ALERT_EMAIL_TO", "admin@docintell.local"),
        )

    def _load_feature_flags(self) -> Dict[str, bool]:
        """Load feature flags"""
        return {
            # Processing features
            "enable_document_splitting": self._get_env(
                "ENABLE_DOCUMENT_SPLITTING", True
            ),
            "enable_digital_pdf_detection": self._get_env(
                "ENABLE_DIGITAL_PDF_DETECTION", True
            ),
            "enable_multi_language_support": self._get_env(
                "ENABLE_MULTI_LANGUAGE_SUPPORT", True
            ),
            "enable_active_learning": self._get_env("ENABLE_ACTIVE_LEARNING", True),
            # Compliance features
            "enable_gdpr_compliance": self._get_env("ENABLE_GDPR_COMPLIANCE", True),
            "enable_hipaa_compliance": self._get_env("ENABLE_HIPAA_COMPLIANCE", False),
            "enable_data_anonymization": self._get_env(
                "ENABLE_DATA_ANONYMIZATION", True
            ),
            # Integration features
            "enable_salesforce_integration": self._get_env(
                "ENABLE_SALESFORCE_INTEGRATION", False
            ),
            "enable_docusign_integration": self._get_env(
                "ENABLE_DOCUSIGN_INTEGRATION", False
            ),
            "enable_webhook_notifications": self._get_env(
                "ENABLE_WEBHOOK_NOTIFICATIONS", True
            ),
            # Development features
            "debug": self._get_env("DEBUG", False),
            "enable_swagger_ui": self._get_env("ENABLE_SWAGGER_UI", True),
            "mock_external_services": self._get_env("MOCK_EXTERNAL_SERVICES", False),
        }

    def get_tenant_config(
        self, tenant_id: str, setting_name: str, default: Any = None
    ) -> Any:
        """Get tenant-specific configuration override"""
        tenant_key = f"TENANT_{tenant_id.upper()}_{setting_name.upper()}"
        return self._get_env(tenant_key, default)

    def is_development(self) -> bool:
        """Check if running in development mode"""
        return self.deployment_mode == DeploymentMode.DEVELOPMENT

    def is_production(self) -> bool:
        """Check if running in production mode"""
        return self.deployment_mode == DeploymentMode.PRODUCTION

    def is_on_premise(self) -> bool:
        """Check if running in on-premise mode"""
        return self.deployment_mode == DeploymentMode.ON_PREMISE

    def feature_enabled(self, feature_name: str) -> bool:
        """Check if a feature flag is enabled"""
        return self.feature_flags.get(feature_name, False)

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary (for serialization)"""
        return {
            "deployment_mode": self.deployment_mode.value,
            "environment": self.environment,
            "database": self.database.__dict__,
            "redis": self.redis.__dict__,
            "neo4j": self.neo4j.__dict__,
            "storage": self.storage.__dict__,
            "vector_store": self.vector_store.__dict__,
            "processing": self.processing.__dict__,
            "ml": self.ml.__dict__,
            "security": {
                k: v
                for k, v in self.security.__dict__.items()
                if not k.endswith("_key")
            },
            "monitoring": self.monitoring.__dict__,
            "feature_flags": self.feature_flags,
        }


# Global configuration instance
_config: Optional[ConfigManager] = None


def get_config() -> ConfigManager:
    """Get global configuration instance"""
    global _config
    if _config is None:
        _config = ConfigManager()
    return _config


def initialize_config(env_file: Optional[str] = None) -> ConfigManager:
    """Initialize global configuration with custom env file"""
    global _config
    _config = ConfigManager(env_file)
    return _config
