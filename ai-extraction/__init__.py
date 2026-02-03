"""
Real-Time Document Intelligence & Compliance Engine
AI Extraction & Vectorization Module - Integration Interface
Version: 1.0
Date: 2026-02-03

This module provides the main integration interface for the AI Extraction & Vectorization Module,
orchestrating all components together for end-to-end document intelligence processing.
"""

import logging
import asyncio
from typing import Dict, List, Any, Optional
from datetime import datetime

from .encoders.bert_encoder import SharedBERTEncoder, BERTVariant, PoolingStrategy
from .ner.document_ner import MultiDocumentNERManager, DocumentType
from .clause_extraction.t5_extractor import (
    T5ClauseExtractor,
    ExtractionMode,
    ClauseType,
)
from .embeddings.clause_embeddings import ClauseEmbeddingGenerator, EmbeddingStrategy
from .vector_store.vector_operations import (
    VectorStoreManager,
    VectorStoreType,
    BatchStrategy,
)
from .versioning.model_versioning import ModelVersioningInterface

logger = logging.getLogger(__name__)


class AIExtractionPipeline:
    """
    Main AI Extraction & Vectorization pipeline orchestrating all components
    """

    def __init__(
        self,
        bert_variant: BERTVariant = BERTVariant.BERT_BASE,
        extraction_mode: ExtractionMode = ExtractionMode.HYBRID,
        embedding_strategy: EmbeddingStrategy = EmbeddingStrategy.HIERARCHICAL,
        primary_vector_store: VectorStoreType = VectorStoreType.PINECONE,
        enable_versioning: bool = True,
    ):
        """
        Initialize AI extraction pipeline

        Args:
            bert_variant: BERT model variant to use
            extraction_mode: T5 clause extraction mode
            embedding_strategy: Embedding generation strategy
            primary_vector_store: Primary vector store type
            enable_versioning: Enable model versioning
        """
        self.bert_variant = bert_variant
        self.extraction_mode = extraction_mode
        self.embedding_strategy = embedding_strategy
        self.primary_vector_store = primary_vector_store
        self.enable_versioning = enable_versioning

        # Initialize components
        self.bert_encoder = None
        self.ner_manager = None
        self.clause_extractor = None
        self.embedding_generator = None
        self.vector_store_manager = None
        self.versioning_interface = None

        # Pipeline state
        self.is_initialized = False
        self.pipeline_metrics = {
            "documents_processed": 0,
            "entities_extracted": 0,
            "clauses_extracted": 0,
            "embeddings_generated": 0,
            "vectors_stored": 0,
            "average_processing_time": 0.0,
        }

        # Initialize pipeline
        self._initialize_pipeline()

        logger.info("AI Extraction Pipeline initialized successfully")

    def _initialize_pipeline(self):
        """Initialize all pipeline components"""
        try:
            # 1. Initialize shared BERT encoder
            self.bert_encoder = SharedBERTEncoder(
                model_variant=self.bert_variant,
                pooling_strategy=PoolingStrategy.ATTENTION_WEIGHTED,
                enable_fine_tuning=True,
            )

            # 2. Initialize NER manager
            self.ner_manager = MultiDocumentNERManager(
                bert_encoder=self.bert_encoder,
                supported_document_types=[
                    DocumentType.LEGAL,
                    DocumentType.FINANCIAL,
                    DocumentType.GENERAL,
                ],
            )

            # 3. Initialize T5 clause extractor
            self.clause_extractor = T5ClauseExtractor(
                extraction_mode=self.extraction_mode,
                max_input_length=1024,
                max_output_length=256,
            )

            # 4. Initialize embedding generator
            self.embedding_generator = ClauseEmbeddingGenerator(
                bert_encoder=self.bert_encoder,
                embedding_strategy=self.embedding_strategy,
                enable_dimensionality_reduction=True,
                reduced_dim=256,
            )

            # 5. Initialize vector store manager
            self.vector_store_manager = VectorStoreManager(
                primary_store_type=self.primary_vector_store,
                batch_strategy=BatchStrategy.ADAPTIVE,
                enable_backup=True,
            )

            # 6. Initialize model versioning (if enabled)
            if self.enable_versioning:
                self.versioning_interface = ModelVersioningInterface(
                    enable_auto_rollback=True, max_versions_per_model=10
                )

            self.is_initialized = True

        except Exception as e:
            logger.error(f"Pipeline initialization failed: {e}")
            raise

    async def process_document(
        self,
        document_text: str,
        document_metadata: Optional[Dict[str, Any]] = None,
        enable_ner: bool = True,
        enable_clause_extraction: bool = True,
        enable_vectorization: bool = True,
        namespace: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Process a document through the complete AI extraction pipeline

        Args:
            document_text: Document text to process
            document_metadata: Additional document metadata
            enable_ner: Enable named entity recognition
            enable_clause_extraction: Enable clause extraction
            enable_vectorization: Enable vectorization and storage
            namespace: Vector store namespace

        Returns:
            Complete processing results
        """
        try:
            processing_start = datetime.now()

            result = {
                "status": "success",
                "document_id": document_metadata.get("document_id", "unknown")
                if document_metadata
                else "unknown",
                "processing_steps": {},
                "extracted_entities": {},
                "extracted_clauses": [],
                "embeddings": {},
                "vector_storage": {},
                "processing_metadata": {
                    "total_processing_time_ms": 0,
                    "pipeline_version": "1.0",
                    "components_used": [],
                },
            }

            if not document_text.strip():
                result["status"] = "failed"
                result["error"] = "Empty document text"
                return result

            # Step 1: Named Entity Recognition
            if enable_ner:
                logger.info("Starting NER processing")
                ner_start = datetime.now()

                ner_result = await self.ner_manager.extract_entities_async(
                    document_text, document_metadata
                )

                ner_time = (datetime.now() - ner_start).total_seconds() * 1000

                result["extracted_entities"] = ner_result
                result["processing_steps"]["ner"] = {
                    "status": "completed",
                    "processing_time_ms": ner_time,
                    "entities_found": sum(
                        len(entities)
                        for entities in ner_result.get("entities_by_type", {}).values()
                    ),
                }
                result["processing_metadata"]["components_used"].append("NER")

                # Update metrics
                self.pipeline_metrics["entities_extracted"] += result[
                    "processing_steps"
                ]["ner"]["entities_found"]

            # Step 2: Clause Extraction
            if enable_clause_extraction:
                logger.info("Starting clause extraction")
                clause_start = datetime.now()

                clause_result = self.clause_extractor.extract_clauses(
                    document_text,
                    clause_types=list(ClauseType),
                    document_context=document_metadata,
                )

                clause_time = (datetime.now() - clause_start).total_seconds() * 1000

                result["extracted_clauses"] = clause_result.get("extracted_clauses", [])
                result["processing_steps"]["clause_extraction"] = {
                    "status": "completed",
                    "processing_time_ms": clause_time,
                    "clauses_found": len(result["extracted_clauses"]),
                }
                result["processing_metadata"]["components_used"].append("T5_EXTRACTOR")

                # Update metrics
                self.pipeline_metrics["clauses_extracted"] += result[
                    "processing_steps"
                ]["clause_extraction"]["clauses_found"]

            # Step 3: Embedding Generation and Vectorization
            if enable_vectorization and result["extracted_clauses"]:
                logger.info("Starting embedding generation")
                embedding_start = datetime.now()

                # Generate clause embeddings
                embedding_result = self.embedding_generator.generate_clause_embeddings(
                    result["extracted_clauses"], return_attention_weights=False
                )

                embedding_time = (
                    datetime.now() - embedding_start
                ).total_seconds() * 1000

                result["embeddings"] = embedding_result
                result["processing_steps"]["embedding_generation"] = {
                    "status": "completed",
                    "processing_time_ms": embedding_time,
                    "embeddings_generated": len(embedding_result.get("embeddings", [])),
                }
                result["processing_metadata"]["components_used"].append(
                    "EMBEDDING_GENERATOR"
                )

                # Update metrics
                self.pipeline_metrics["embeddings_generated"] += result[
                    "processing_steps"
                ]["embedding_generation"]["embeddings_generated"]

                # Step 4: Vector Store Upsert
                if embedding_result.get("embeddings"):
                    logger.info("Starting vector storage")
                    storage_start = datetime.now()

                    storage_result = (
                        await self.vector_store_manager.upsert_clause_embeddings(
                            embedding_result, document_metadata, namespace
                        )
                    )

                    storage_time = (
                        datetime.now() - storage_start
                    ).total_seconds() * 1000

                    result["vector_storage"] = storage_result
                    result["processing_steps"]["vector_storage"] = {
                        "status": storage_result.get("status", "failed"),
                        "processing_time_ms": storage_time,
                        "vectors_stored": storage_result.get("vectors_processed", 0),
                    }
                    result["processing_metadata"]["components_used"].append(
                        "VECTOR_STORE"
                    )

                    # Update metrics
                    self.pipeline_metrics["vectors_stored"] += result[
                        "processing_steps"
                    ]["vector_storage"]["vectors_stored"]

            # Calculate total processing time
            total_processing_time = (
                datetime.now() - processing_start
            ).total_seconds() * 1000
            result["processing_metadata"]["total_processing_time_ms"] = (
                total_processing_time
            )

            # Update pipeline metrics
            self.pipeline_metrics["documents_processed"] += 1
            current_avg = self.pipeline_metrics["average_processing_time"]
            total_docs = self.pipeline_metrics["documents_processed"]
            new_avg = (
                (current_avg * (total_docs - 1)) + total_processing_time
            ) / total_docs
            self.pipeline_metrics["average_processing_time"] = new_avg

            logger.info(
                f"Document processing completed in {total_processing_time:.0f}ms"
            )

            return result

        except Exception as e:
            logger.error(f"Document processing failed: {e}")
            return {
                "status": "failed",
                "error": str(e),
                "document_id": document_metadata.get("document_id", "unknown")
                if document_metadata
                else "unknown",
            }

    async def process_batch_documents(
        self,
        documents: List[Dict[str, Any]],
        batch_size: int = 10,
        namespace: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Process multiple documents in batches

        Args:
            documents: List of documents with 'text' and 'metadata'
            batch_size: Batch size for processing
            namespace: Vector store namespace

        Returns:
            Batch processing results
        """
        try:
            batch_start = datetime.now()

            result = {
                "status": "success",
                "total_documents": len(documents),
                "processed_documents": 0,
                "failed_documents": 0,
                "batch_results": [],
                "batch_summary": {
                    "total_entities": 0,
                    "total_clauses": 0,
                    "total_embeddings": 0,
                    "total_vectors": 0,
                },
                "processing_time_ms": 0,
            }

            # Process documents in batches
            for i in range(0, len(documents), batch_size):
                batch = documents[i : i + batch_size]

                logger.info(
                    f"Processing batch {i // batch_size + 1} ({len(batch)} documents)"
                )

                # Process batch concurrently
                batch_tasks = []
                for doc in batch:
                    task = self.process_document(
                        doc.get("text", ""),
                        doc.get("metadata", {}),
                        namespace=namespace,
                    )
                    batch_tasks.append(task)

                # Wait for batch completion
                batch_results = await asyncio.gather(
                    *batch_tasks, return_exceptions=True
                )

                # Process batch results
                for doc_result in batch_results:
                    if isinstance(doc_result, Exception):
                        result["failed_documents"] += 1
                        logger.error(f"Document processing failed: {doc_result}")
                    else:
                        if doc_result.get("status") == "success":
                            result["processed_documents"] += 1

                            # Update summary
                            if "extracted_entities" in doc_result:
                                entities_count = sum(
                                    len(entities)
                                    for entities in doc_result["extracted_entities"]
                                    .get("entities_by_type", {})
                                    .values()
                                )
                                result["batch_summary"]["total_entities"] += (
                                    entities_count
                                )

                            if "extracted_clauses" in doc_result:
                                result["batch_summary"]["total_clauses"] += len(
                                    doc_result["extracted_clauses"]
                                )

                            if "embeddings" in doc_result:
                                embeddings_count = len(
                                    doc_result["embeddings"].get("embeddings", [])
                                )
                                result["batch_summary"]["total_embeddings"] += (
                                    embeddings_count
                                )

                            if "vector_storage" in doc_result:
                                vectors_count = doc_result["vector_storage"].get(
                                    "vectors_processed", 0
                                )
                                result["batch_summary"]["total_vectors"] += (
                                    vectors_count
                                )
                        else:
                            result["failed_documents"] += 1

                        result["batch_results"].append(doc_result)

            # Calculate total processing time
            total_processing_time = (
                datetime.now() - batch_start
            ).total_seconds() * 1000
            result["processing_time_ms"] = total_processing_time

            logger.info(
                f"Batch processing completed: {result['processed_documents']}/{result['total_documents']} successful"
            )

            return result

        except Exception as e:
            logger.error(f"Batch processing failed: {e}")
            return {
                "status": "failed",
                "error": str(e),
                "total_documents": len(documents),
                "processed_documents": 0,
            }

    async def search_similar_content(
        self,
        query_text: str,
        search_type: str = "clauses",
        top_k: int = 10,
        namespace: Optional[str] = None,
        filter_conditions: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Search for similar content using the AI extraction pipeline

        Args:
            query_text: Query text
            search_type: Type of search ("clauses", "entities", "hybrid")
            top_k: Number of results to return
            namespace: Search namespace
            filter_conditions: Additional filter conditions

        Returns:
            Search results
        """
        try:
            search_start = datetime.now()

            result = {
                "status": "success",
                "query": query_text,
                "search_type": search_type,
                "results": [],
                "search_metadata": {
                    "total_results": 0,
                    "search_time_ms": 0,
                    "query_processing_time_ms": 0,
                },
            }

            # Process query text to generate embeddings
            query_start = datetime.now()

            if search_type in ["clauses", "hybrid"]:
                # Extract clauses from query
                query_clauses = self.clause_extractor.extract_clauses(
                    query_text, clause_types=list(ClauseType)
                )

                if query_clauses.get("extracted_clauses"):
                    # Generate embeddings for query clauses
                    query_embeddings = (
                        self.embedding_generator.generate_clause_embeddings(
                            query_clauses["extracted_clauses"]
                        )
                    )

                    # Search vector store
                    if query_embeddings.get("embeddings"):
                        for embedding in query_embeddings["embeddings"]:
                            search_result = (
                                await self.vector_store_manager.search_similar_clauses(
                                    embedding,
                                    top_k=top_k,
                                    filter_conditions=filter_conditions,
                                    namespace=namespace,
                                )
                            )

                            result["results"].extend(search_result.get("results", []))

            elif search_type == "entities":
                # Extract entities from query
                query_entities = await self.ner_manager.extract_entities_async(
                    query_text
                )

                # Use entities for filtering in vector search
                if query_entities.get("entities_by_type"):
                    entity_filters = {}
                    for entity_type, entities in query_entities[
                        "entities_by_type"
                    ].items():
                        if entities:
                            entity_filters[f"entity_{entity_type}"] = [
                                e["text"] for e in entities
                            ]

                    # Simple text similarity search using BERT encoder
                    query_encoding = self.bert_encoder.encode_batch([query_text])
                    if query_encoding.get("embeddings"):
                        query_embedding = (
                            query_encoding["embeddings"][0].mean(dim=0).tolist()
                        )

                        search_result = (
                            await self.vector_store_manager.search_similar_clauses(
                                query_embedding,
                                top_k=top_k,
                                filter_conditions={
                                    **filter_conditions,
                                    **entity_filters,
                                }
                                if filter_conditions
                                else entity_filters,
                                namespace=namespace,
                            )
                        )

                        result["results"] = search_result.get("results", [])

            query_processing_time = (
                datetime.now() - query_start
            ).total_seconds() * 1000

            # Deduplicate and rank results
            if result["results"]:
                result["results"] = self._deduplicate_and_rank_results(
                    result["results"]
                )
                result["results"] = result["results"][:top_k]

            # Update metadata
            result["search_metadata"]["total_results"] = len(result["results"])
            result["search_metadata"]["query_processing_time_ms"] = (
                query_processing_time
            )

            total_search_time = (datetime.now() - search_start).total_seconds() * 1000
            result["search_metadata"]["search_time_ms"] = total_search_time

            logger.info(
                f"Search completed: {len(result['results'])} results in {total_search_time:.0f}ms"
            )

            return result

        except Exception as e:
            logger.error(f"Search failed: {e}")
            return {
                "status": "failed",
                "error": str(e),
                "query": query_text,
                "results": [],
            }

    def _deduplicate_and_rank_results(
        self, results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Deduplicate and rank search results"""
        try:
            # Simple deduplication based on vector ID
            seen_ids = set()
            unique_results = []

            for result in results:
                result_id = result.get("id") or result.get("metadata", {}).get(
                    "vector_id"
                )
                if result_id and result_id not in seen_ids:
                    seen_ids.add(result_id)
                    unique_results.append(result)
                elif not result_id:
                    unique_results.append(result)

            # Sort by score (descending)
            unique_results.sort(key=lambda x: x.get("score", 0), reverse=True)

            return unique_results

        except Exception:
            return results

    def get_pipeline_status(self) -> Dict[str, Any]:
        """Get comprehensive pipeline status"""
        try:
            status = {
                "pipeline_info": {
                    "is_initialized": self.is_initialized,
                    "bert_variant": self.bert_variant.value,
                    "extraction_mode": self.extraction_mode.value,
                    "embedding_strategy": self.embedding_strategy.value,
                    "primary_vector_store": self.primary_vector_store.value,
                    "versioning_enabled": self.enable_versioning,
                },
                "component_status": {},
                "performance_metrics": self.pipeline_metrics,
                "health_status": "healthy",
            }

            # Get component statuses
            if self.bert_encoder:
                status["component_status"]["bert_encoder"] = (
                    self.bert_encoder.get_encoder_info()
                )

            if self.ner_manager:
                status["component_status"]["ner_manager"] = (
                    self.ner_manager.get_manager_info()
                )

            if self.clause_extractor:
                status["component_status"]["clause_extractor"] = (
                    self.clause_extractor.get_extractor_info()
                )

            if self.embedding_generator:
                status["component_status"]["embedding_generator"] = (
                    self.embedding_generator.get_embedding_info()
                )

            if self.vector_store_manager:
                status["component_status"]["vector_store_manager"] = (
                    self.vector_store_manager.get_manager_info()
                )

            if self.versioning_interface:
                status["component_status"]["versioning_interface"] = (
                    self.versioning_interface.get_system_overview()
                )

            # Determine overall health
            component_health = []
            for component_name, component_info in status["component_status"].items():
                if isinstance(component_info, dict) and "error" in component_info:
                    component_health.append("unhealthy")
                else:
                    component_health.append("healthy")

            if "unhealthy" in component_health:
                status["health_status"] = "degraded"
            elif not component_health:
                status["health_status"] = "initializing"

            return status

        except Exception as e:
            logger.error(f"Failed to get pipeline status: {e}")
            return {"health_status": "unhealthy", "error": str(e)}

    async def shutdown_pipeline(self):
        """Gracefully shutdown the pipeline"""
        try:
            logger.info("Shutting down AI extraction pipeline")

            # Stop monitoring and cleanup resources
            if self.versioning_interface:
                # Save any pending versions or experiments
                pass

            if self.vector_store_manager:
                # Flush any pending vector operations
                pass

            # Reset state
            self.is_initialized = False

            logger.info("Pipeline shutdown completed")

        except Exception as e:
            logger.error(f"Pipeline shutdown failed: {e}")


# Module-level convenience functions
def create_ai_extraction_pipeline(
    config: Optional[Dict[str, Any]] = None,
) -> AIExtractionPipeline:
    """
    Create an AI extraction pipeline with configuration

    Args:
        config: Pipeline configuration

    Returns:
        Configured AI extraction pipeline
    """
    if config is None:
        config = {}

    return AIExtractionPipeline(
        bert_variant=BERTVariant(config.get("bert_variant", "BERT_BASE")),
        extraction_mode=ExtractionMode(config.get("extraction_mode", "HYBRID")),
        embedding_strategy=EmbeddingStrategy(
            config.get("embedding_strategy", "HIERARCHICAL")
        ),
        primary_vector_store=VectorStoreType(
            config.get("primary_vector_store", "PINECONE")
        ),
        enable_versioning=config.get("enable_versioning", True),
    )


async def process_document_batch(
    documents: List[Dict[str, Any]], config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Convenience function to process a batch of documents

    Args:
        documents: List of documents to process
        config: Pipeline configuration

    Returns:
        Batch processing results
    """
    pipeline = create_ai_extraction_pipeline(config)

    try:
        result = await pipeline.process_batch_documents(documents)
        await pipeline.shutdown_pipeline()
        return result
    except Exception:
        await pipeline.shutdown_pipeline()
        raise


# Export main classes and functions
__all__ = [
    "AIExtractionPipeline",
    "create_ai_extraction_pipeline",
    "process_document_batch",
]
