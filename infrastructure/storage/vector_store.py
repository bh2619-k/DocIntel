"""
Real-Time Document Intelligence & Compliance Engine
Vector Store Manager for Pinecone/Weaviate Operations
Version: 1.0
Date: 2026-02-03

This module provides a unified interface for vector operations supporting
both Pinecone (cloud) and Weaviate (on-premise) deployments for clause-level
embeddings and semantic search.
"""

import logging
import time
from datetime import datetime
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass
from enum import Enum
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class VectorBackend(Enum):
    """Supported vector store backends"""

    PINECONE = "pinecone"
    WEAVIATE = "weaviate"


@dataclass
class VectorConfig:
    """Vector store configuration"""

    backend: VectorBackend
    api_key: str = ""
    environment: str = ""  # Pinecone environment
    index_name: str = "document-clauses"
    dimension: int = 384  # Sentence-BERT embedding size
    metric: str = "cosine"
    replicas: int = 1
    pods: int = 1
    pod_type: str = "p1.x1"

    # Weaviate-specific settings
    weaviate_url: str = "http://localhost:8080"
    weaviate_auth_api_key: str = ""

    # Multi-tenancy settings
    tenant_isolation: bool = True


@dataclass
class EmbeddingMetadata:
    """Metadata for embeddings"""

    clause_id: str
    document_id: str
    tenant_id: str
    clause_type: str
    content_snippet: str  # First 500 chars of clause
    confidence: float
    page_numbers: List[int]
    created_at: datetime
    model_version: str = "sentence-transformers/all-MiniLM-L6-v2"


class VectorStoreInterface(ABC):
    """Abstract interface for vector store operations"""

    @abstractmethod
    def create_index(self, config: VectorConfig) -> bool:
        """Create vector index"""
        pass

    @abstractmethod
    def upsert_embeddings(
        self, embeddings: List[Tuple[str, List[float], Dict]]
    ) -> bool:
        """Insert or update embeddings"""
        pass

    @abstractmethod
    def query_similar(
        self, query_vector: List[float], top_k: int = 10, filter_metadata: Dict = None
    ) -> List[Dict]:
        """Query for similar vectors"""
        pass

    @abstractmethod
    def delete_embeddings(self, ids: List[str]) -> bool:
        """Delete embeddings by IDs"""
        pass

    @abstractmethod
    def get_index_stats(self) -> Dict:
        """Get index statistics"""
        pass


class PineconeVectorStore(VectorStoreInterface):
    """Pinecone vector store implementation"""

    def __init__(self, config: VectorConfig):
        """Initialize Pinecone client"""
        try:
            import pinecone

            self.pinecone = pinecone

            # Initialize Pinecone
            pinecone.init(api_key=config.api_key, environment=config.environment)

            self.config = config
            self.index = None
            self._connect_to_index()

        except ImportError:
            raise ImportError("pinecone-client package required for Pinecone backend")
        except Exception as e:
            logger.error(f"Failed to initialize Pinecone: {e}")
            raise

    def _get_index_name(self, tenant_id: str = None) -> str:
        """Get tenant-specific index name"""
        if self.config.tenant_isolation and tenant_id:
            return f"{self.config.index_name}-{tenant_id}"
        return self.config.index_name

    def _connect_to_index(self, tenant_id: str = None):
        """Connect to Pinecone index"""
        index_name = self._get_index_name(tenant_id)

        if index_name in self.pinecone.list_indexes():
            self.index = self.pinecone.Index(index_name)
        else:
            logger.warning(f"Index {index_name} not found. Call create_index() first.")

    def create_index(self, config: VectorConfig, tenant_id: str = None) -> bool:
        """Create Pinecone index"""
        try:
            index_name = self._get_index_name(tenant_id)

            # Check if index already exists
            if index_name in self.pinecone.list_indexes():
                logger.info(f"Index {index_name} already exists")
                self.index = self.pinecone.Index(index_name)
                return True

            # Create new index
            self.pinecone.create_index(
                name=index_name,
                dimension=config.dimension,
                metric=config.metric,
                replicas=config.replicas,
                pods=config.pods,
                pod_type=config.pod_type,
            )

            # Wait for index to be ready
            max_wait_time = 300  # 5 minutes
            wait_time = 0

            while wait_time < max_wait_time:
                if index_name in self.pinecone.list_indexes():
                    self.index = self.pinecone.Index(index_name)
                    logger.info(f"Created Pinecone index {index_name}")
                    return True
                time.sleep(10)
                wait_time += 10

            raise TimeoutError(f"Index {index_name} creation timed out")

        except Exception as e:
            logger.error(f"Failed to create Pinecone index: {e}")
            return False

    def upsert_embeddings(
        self, embeddings: List[Tuple[str, List[float], Dict]], tenant_id: str = None
    ) -> bool:
        """Upsert embeddings to Pinecone"""
        try:
            if not self.index:
                self._connect_to_index(tenant_id)

            # Convert to Pinecone format
            vectors = []
            for embedding_id, vector, metadata in embeddings:
                vectors.append(
                    {"id": embedding_id, "values": vector, "metadata": metadata}
                )

            # Batch upsert (max 100 vectors per batch)
            batch_size = 100
            for i in range(0, len(vectors), batch_size):
                batch = vectors[i : i + batch_size]
                self.index.upsert(vectors=batch)

            logger.info(f"Upserted {len(embeddings)} embeddings to Pinecone")
            return True

        except Exception as e:
            logger.error(f"Failed to upsert embeddings: {e}")
            return False

    def query_similar(
        self,
        query_vector: List[float],
        top_k: int = 10,
        filter_metadata: Dict = None,
        tenant_id: str = None,
    ) -> List[Dict]:
        """Query similar embeddings from Pinecone"""
        try:
            if not self.index:
                self._connect_to_index(tenant_id)

            # Add tenant filter if using tenant isolation
            if self.config.tenant_isolation and tenant_id and not filter_metadata:
                filter_metadata = {"tenant_id": tenant_id}
            elif self.config.tenant_isolation and tenant_id and filter_metadata:
                filter_metadata["tenant_id"] = tenant_id

            # Query Pinecone
            response = self.index.query(
                vector=query_vector,
                top_k=top_k,
                filter=filter_metadata,
                include_metadata=True,
                include_values=False,
            )

            # Convert response format
            results = []
            for match in response["matches"]:
                results.append(
                    {
                        "id": match["id"],
                        "score": match["score"],
                        "metadata": match["metadata"],
                    }
                )

            return results

        except Exception as e:
            logger.error(f"Failed to query embeddings: {e}")
            return []

    def delete_embeddings(self, ids: List[str], tenant_id: str = None) -> bool:
        """Delete embeddings from Pinecone"""
        try:
            if not self.index:
                self._connect_to_index(tenant_id)

            # Batch delete
            batch_size = 1000
            for i in range(0, len(ids), batch_size):
                batch = ids[i : i + batch_size]
                self.index.delete(ids=batch)

            logger.info(f"Deleted {len(ids)} embeddings from Pinecone")
            return True

        except Exception as e:
            logger.error(f"Failed to delete embeddings: {e}")
            return False

    def get_index_stats(self, tenant_id: str = None) -> Dict:
        """Get Pinecone index statistics"""
        try:
            if not self.index:
                self._connect_to_index(tenant_id)

            stats = self.index.describe_index_stats()
            return {
                "total_vector_count": stats.get("total_vector_count", 0),
                "dimension": stats.get("dimension", 0),
                "index_fullness": stats.get("index_fullness", 0.0),
                "namespaces": stats.get("namespaces", {}),
            }

        except Exception as e:
            logger.error(f"Failed to get index stats: {e}")
            return {}


class WeaviateVectorStore(VectorStoreInterface):
    """Weaviate vector store implementation"""

    def __init__(self, config: VectorConfig):
        """Initialize Weaviate client"""
        try:
            import weaviate

            # Configure authentication
            auth_config = None
            if config.weaviate_auth_api_key:
                auth_config = weaviate.AuthApiKey(api_key=config.weaviate_auth_api_key)

            # Initialize client
            self.client = weaviate.Client(
                url=config.weaviate_url, auth_client_secret=auth_config
            )

            self.config = config
            self._validate_connection()

        except ImportError:
            raise ImportError("weaviate-client package required for Weaviate backend")
        except Exception as e:
            logger.error(f"Failed to initialize Weaviate: {e}")
            raise

    def _validate_connection(self):
        """Validate Weaviate connection"""
        if not self.client.is_ready():
            raise ConnectionError("Weaviate server is not ready")

    def _get_class_name(self, tenant_id: str = None) -> str:
        """Get tenant-specific class name"""
        base_name = "DocumentClause"
        if self.config.tenant_isolation and tenant_id:
            # Sanitize tenant ID for class name
            safe_tenant_id = tenant_id.replace("-", "_").replace(" ", "_")
            return f"{base_name}_{safe_tenant_id}"
        return base_name

    def create_index(self, config: VectorConfig, tenant_id: str = None) -> bool:
        """Create Weaviate class (equivalent to index)"""
        try:
            class_name = self._get_class_name(tenant_id)

            # Check if class already exists
            try:
                self.client.schema.get(class_name)
                logger.info(f"Weaviate class {class_name} already exists")
                return True
            except Exception:
                pass  # Class doesn't exist, continue to create

            # Define class schema
            class_schema = {
                "class": class_name,
                "description": "Document clause embeddings for semantic search",
                "vectorizer": "none",  # We provide our own vectors
                "properties": [
                    {
                        "name": "clause_id",
                        "dataType": ["string"],
                        "description": "Unique clause identifier",
                    },
                    {
                        "name": "document_id",
                        "dataType": ["string"],
                        "description": "Parent document identifier",
                    },
                    {
                        "name": "tenant_id",
                        "dataType": ["string"],
                        "description": "Tenant identifier for isolation",
                    },
                    {
                        "name": "clause_type",
                        "dataType": ["string"],
                        "description": "Type of legal clause",
                    },
                    {
                        "name": "content_snippet",
                        "dataType": ["text"],
                        "description": "First 500 characters of clause content",
                    },
                    {
                        "name": "confidence",
                        "dataType": ["number"],
                        "description": "Extraction confidence score",
                    },
                    {
                        "name": "page_numbers",
                        "dataType": ["int[]"],
                        "description": "Pages where clause appears",
                    },
                    {
                        "name": "created_at",
                        "dataType": ["date"],
                        "description": "When embedding was created",
                    },
                    {
                        "name": "model_version",
                        "dataType": ["string"],
                        "description": "Embedding model version",
                    },
                ],
            }

            # Create class
            self.client.schema.create_class(class_schema)
            logger.info(f"Created Weaviate class {class_name}")
            return True

        except Exception as e:
            logger.error(f"Failed to create Weaviate class: {e}")
            return False

    def upsert_embeddings(
        self, embeddings: List[Tuple[str, List[float], Dict]], tenant_id: str = None
    ) -> bool:
        """Upsert embeddings to Weaviate"""
        try:
            class_name = self._get_class_name(tenant_id)

            # Batch upsert
            with self.client.batch as batch:
                batch.batch_size = 100
                batch.dynamic = True

                for embedding_id, vector, metadata in embeddings:
                    # Add object with vector
                    batch.add_data_object(
                        uuid=embedding_id,
                        data_object=metadata,
                        class_name=class_name,
                        vector=vector,
                    )

            logger.info(f"Upserted {len(embeddings)} embeddings to Weaviate")
            return True

        except Exception as e:
            logger.error(f"Failed to upsert embeddings: {e}")
            return False

    def query_similar(
        self,
        query_vector: List[float],
        top_k: int = 10,
        filter_metadata: Dict = None,
        tenant_id: str = None,
    ) -> List[Dict]:
        """Query similar embeddings from Weaviate"""
        try:
            class_name = self._get_class_name(tenant_id)

            # Build query
            query_builder = (
                self.client.query.get(
                    class_name,
                    [
                        "clause_id",
                        "document_id",
                        "tenant_id",
                        "clause_type",
                        "content_snippet",
                        "confidence",
                        "page_numbers",
                        "created_at",
                        "model_version",
                    ],
                )
                .with_near_vector({"vector": query_vector, "certainty": 0.7})
                .with_limit(top_k)
                .with_additional(["certainty", "distance"])
            )

            # Add filters if provided
            if filter_metadata:
                # Build where filter
                where_filter = {"operator": "And", "operands": []}

                for key, value in filter_metadata.items():
                    where_filter["operands"].append(
                        {
                            "path": [key],
                            "operator": "Equal",
                            "valueString"
                            if isinstance(value, str)
                            else "valueNumber": value,
                        }
                    )

                query_builder = query_builder.with_where(where_filter)

            # Execute query
            response = query_builder.do()

            # Convert response format
            results = []
            if "data" in response and "Get" in response["data"]:
                objects = response["data"]["Get"][class_name]

                for obj in objects:
                    # Extract additional info
                    additional = obj.get("_additional", {})

                    results.append(
                        {
                            "id": obj.get("clause_id"),
                            "score": additional.get("certainty", 0.0),
                            "distance": additional.get("distance", 1.0),
                            "metadata": {
                                k: v for k, v in obj.items() if not k.startswith("_")
                            },
                        }
                    )

            return results

        except Exception as e:
            logger.error(f"Failed to query embeddings: {e}")
            return []

    def delete_embeddings(self, ids: List[str], tenant_id: str = None) -> bool:
        """Delete embeddings from Weaviate"""
        try:
            class_name = self._get_class_name(tenant_id)

            # Delete objects by UUID
            for embedding_id in ids:
                self.client.data_object.delete(uuid=embedding_id, class_name=class_name)

            logger.info(f"Deleted {len(ids)} embeddings from Weaviate")
            return True

        except Exception as e:
            logger.error(f"Failed to delete embeddings: {e}")
            return False

    def get_index_stats(self, tenant_id: str = None) -> Dict:
        """Get Weaviate class statistics"""
        try:
            class_name = self._get_class_name(tenant_id)

            # Get object count
            response = self.client.query.aggregate(class_name).with_meta_count().do()

            object_count = 0
            if "data" in response and "Aggregate" in response["data"]:
                agg_data = response["data"]["Aggregate"][class_name]
                if agg_data and len(agg_data) > 0:
                    object_count = agg_data[0].get("meta", {}).get("count", 0)

            return {
                "total_vector_count": object_count,
                "dimension": self.config.dimension,
                "class_name": class_name,
            }

        except Exception as e:
            logger.error(f"Failed to get index stats: {e}")
            return {}


class VectorStoreManager:
    """
    Unified vector store manager
    Handles clause-level embeddings with multi-tenant isolation
    """

    def __init__(self, config: VectorConfig):
        """Initialize vector store manager"""
        self.config = config

        if config.backend == VectorBackend.PINECONE:
            self.store = PineconeVectorStore(config)
        elif config.backend == VectorBackend.WEAVIATE:
            self.store = WeaviateVectorStore(config)
        else:
            raise ValueError(f"Unsupported vector backend: {config.backend}")

    def initialize_tenant_index(self, tenant_id: str) -> bool:
        """Initialize index for a specific tenant"""
        return self.store.create_index(self.config, tenant_id)

    def store_clause_embedding(
        self, metadata: EmbeddingMetadata, embedding_vector: List[float]
    ) -> bool:
        """Store a single clause embedding"""
        # Convert metadata to dict
        metadata_dict = {
            "clause_id": metadata.clause_id,
            "document_id": metadata.document_id,
            "tenant_id": metadata.tenant_id,
            "clause_type": metadata.clause_type,
            "content_snippet": metadata.content_snippet,
            "confidence": metadata.confidence,
            "page_numbers": metadata.page_numbers,
            "created_at": metadata.created_at.isoformat(),
            "model_version": metadata.model_version,
        }

        # Generate embedding ID
        embedding_id = f"{metadata.tenant_id}:{metadata.clause_id}"

        return self.store.upsert_embeddings(
            [(embedding_id, embedding_vector, metadata_dict)], metadata.tenant_id
        )

    def store_clause_embeddings_batch(
        self, embeddings_data: List[Tuple[EmbeddingMetadata, List[float]]]
    ) -> bool:
        """Store multiple clause embeddings in batch"""
        # Convert to format expected by store
        embeddings = []
        tenant_id = None

        for metadata, vector in embeddings_data:
            if tenant_id is None:
                tenant_id = metadata.tenant_id
            elif tenant_id != metadata.tenant_id:
                raise ValueError("All embeddings in batch must be from same tenant")

            metadata_dict = {
                "clause_id": metadata.clause_id,
                "document_id": metadata.document_id,
                "tenant_id": metadata.tenant_id,
                "clause_type": metadata.clause_type,
                "content_snippet": metadata.content_snippet,
                "confidence": metadata.confidence,
                "page_numbers": metadata.page_numbers,
                "created_at": metadata.created_at.isoformat(),
                "model_version": metadata.model_version,
            }

            embedding_id = f"{metadata.tenant_id}:{metadata.clause_id}"
            embeddings.append((embedding_id, vector, metadata_dict))

        return self.store.upsert_embeddings(embeddings, tenant_id)

    def find_similar_clauses(
        self,
        query_vector: List[float],
        tenant_id: str,
        clause_type: str = None,
        document_id: str = None,
        top_k: int = 10,
        min_confidence: float = 0.0,
    ) -> List[Dict]:
        """Find similar clauses with optional filters"""
        # Build filter
        filter_metadata = {}
        if clause_type:
            filter_metadata["clause_type"] = clause_type
        if document_id:
            filter_metadata["document_id"] = document_id
        if min_confidence > 0:
            filter_metadata["confidence"] = {"$gte": min_confidence}

        # Query vector store
        results = self.store.query_similar(
            query_vector=query_vector,
            top_k=top_k,
            filter_metadata=filter_metadata,
            tenant_id=tenant_id,
        )

        return results

    def delete_document_clauses(self, document_id: str, tenant_id: str) -> bool:
        """Delete all clause embeddings for a document"""
        # First query to get all clause IDs for the document
        results = self.store.query_similar(
            query_vector=[0.0] * self.config.dimension,  # Dummy vector
            top_k=10000,  # Large number to get all
            filter_metadata={"document_id": document_id},
            tenant_id=tenant_id,
        )

        if not results:
            return True

        # Extract embedding IDs
        embedding_ids = [result["id"] for result in results]

        return self.store.delete_embeddings(embedding_ids, tenant_id)

    def get_tenant_stats(self, tenant_id: str) -> Dict:
        """Get statistics for tenant's embeddings"""
        return self.store.get_index_stats(tenant_id)

    def health_check(self) -> Dict[str, Any]:
        """Perform vector store health check"""
        try:
            # Test basic connectivity and get stats
            stats = self.store.get_index_stats()

            return {
                "status": "healthy",
                "backend": self.config.backend.value,
                "stats": stats,
            }

        except Exception as e:
            return {
                "status": "unhealthy",
                "backend": self.config.backend.value,
                "error": str(e),
            }
