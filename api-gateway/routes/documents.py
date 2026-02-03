"""
Real-Time Document Intelligence & Compliance Engine
REST API Routes - Documents
Version: 1.0
Date: 2026-02-03

This module provides REST API endpoints for document upload, status queries,
result retrieval, and document management operations.
"""

import os
import mimetypes
from datetime import datetime
from typing import List, Optional, Dict, Any
import logging

try:
    from fastapi import (
        APIRouter,
        Depends,
        HTTPException,
        status,
        UploadFile,
        File,
        Form,
        Query,
    )
    from pydantic import BaseModel
except ImportError:
    # For development - these will be available at runtime
    pass

from ..auth.authentication import UserContext, require_permission

try:
    from ..schemas.api_schemas import (
        DocumentUploadResponse,
        DocumentStatusResponse,
        DocumentQueryResponse,
        DocumentListResponse,
        ProcessingStatsResponse,
    )
except ImportError:
    # For development - these will be available at runtime
    pass

logger = logging.getLogger(__name__)


class DocumentUploadRequest(BaseModel):
    """Document upload request parameters"""

    document_type: Optional[str] = None
    priority: Optional[str] = "normal"  # normal, high, rush
    language: Optional[str] = "en"
    metadata: Optional[Dict[str, Any]] = {}
    enable_ocr: Optional[bool] = True
    enable_classification: Optional[bool] = True


class DocumentQueryRequest(BaseModel):
    """Document query request"""

    query: str
    document_types: Optional[List[str]] = None
    date_from: Optional[datetime] = None
    date_to: Optional[datetime] = None
    limit: Optional[int] = 10
    offset: Optional[int] = 0


def create_document_router(
    storage_manager, queue_manager, database_manager, websocket_manager, auth_dependency
) -> APIRouter:
    """Create document API router with dependencies"""

    router = APIRouter(prefix="/api/v1/documents", tags=["documents"])

    @router.post(
        "/upload",
        response_model=DocumentUploadResponse,
        status_code=status.HTTP_201_CREATED,
    )
    @require_permission("documents:upload")
    async def upload_document(
        file: UploadFile = File(...),
        document_type: Optional[str] = Form(None),
        priority: str = Form("normal"),
        language: str = Form("en"),
        enable_ocr: bool = Form(True),
        enable_classification: bool = Form(True),
        user_context: UserContext = Depends(auth_dependency),
    ):
        """
        Upload a new document for processing

        - **file**: Document file (PDF, DOCX, images)
        - **document_type**: Override automatic document classification
        - **priority**: Processing priority (normal, high, rush)
        - **language**: Document language for OCR optimization
        - **enable_ocr**: Whether to perform OCR on the document
        - **enable_classification**: Whether to classify document type
        """
        try:
            # Validate file
            await _validate_upload_file(file, user_context.tenant_id)

            # Generate document ID
            import uuid

            document_id = str(uuid.uuid4())

            # Determine MIME type
            mime_type = file.content_type or mimetypes.guess_type(file.filename)[0]

            # Upload file to storage
            storage_path = await storage_manager.upload_document(
                tenant_id=user_context.tenant_id,
                document_id=document_id,
                file_data=file.file,
                filename=file.filename,
                content_type=mime_type,
            )

            # Create database record
            await database_manager.create_document(
                tenant_id=user_context.tenant_id,
                document_id=document_id,
                uploaded_by=user_context.user_id,
                original_filename=file.filename,
                file_size_bytes=file.size,
                mime_type=mime_type,
                document_type=document_type,
                language_code=language,
                storage_path=storage_path,
                priority=_map_priority_to_int(priority),
                metadata={
                    "enable_ocr": enable_ocr,
                    "enable_classification": enable_classification,
                    "uploaded_via": "api",
                },
            )

            # Create processing job
            job_id = await queue_manager.create_document_job(
                document_id=document_id,
                tenant_id=user_context.tenant_id,
                priority=priority,
                document_type=document_type,
                storage_path=storage_path,
                processing_options={
                    "enable_ocr": enable_ocr,
                    "enable_classification": enable_classification,
                    "language": language,
                },
            )

            # Notify via WebSocket
            if websocket_manager:
                await websocket_manager.notify_document_status_change(
                    tenant_id=user_context.tenant_id,
                    document_id=document_id,
                    status="QUEUED",
                    stage="PREPROCESSING",
                )

            logger.info(f"Document {document_id} uploaded by {user_context.email}")

            return DocumentUploadResponse(
                document_id=document_id,
                job_id=job_id,
                filename=file.filename,
                file_size=file.size,
                status="QUEUED",
                storage_path=storage_path,
                estimated_processing_time=_estimate_processing_time(
                    file.size, enable_ocr
                ),
                message="Document uploaded successfully and queued for processing",
            )

        except Exception as e:
            logger.error(f"Document upload failed: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Document upload failed: {str(e)}",
            )

    @router.get("/{document_id}/status", response_model=DocumentStatusResponse)
    @require_permission("documents:read")
    async def get_document_status(
        document_id: str, user_context: UserContext = Depends(auth_dependency)
    ):
        """
        Get current processing status of a document

        Returns detailed status including current stage, progress,
        and any errors encountered during processing.
        """
        try:
            # Get document from database
            document = await database_manager.get_document(
                document_id=document_id, tenant_id=user_context.tenant_id
            )

            if not document:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND, detail="Document not found"
                )

            # Get processing checkpoints
            checkpoints = await database_manager.get_document_checkpoints(
                document_id=document_id, tenant_id=user_context.tenant_id
            )

            # Get current queue position if still queued
            queue_position = None
            if document["status"] == "QUEUED":
                queue_position = await queue_manager.get_queue_position(
                    document_id=document_id, tenant_id=user_context.tenant_id
                )

            return DocumentStatusResponse(
                document_id=document_id,
                status=document["status"],
                current_stage=_get_current_stage(checkpoints),
                progress_percentage=_calculate_progress_percentage(
                    document["status"], checkpoints
                ),
                queue_position=queue_position,
                processing_started_at=document.get("processing_started_at"),
                processing_completed_at=document.get("processing_completed_at"),
                processing_time_seconds=document.get("processing_time_seconds"),
                error_message=document.get("error_message"),
                retry_count=document.get("retry_count", 0),
                checkpoints=checkpoints,
            )

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error getting document status {document_id}: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to get document status",
            )

    @router.get("/{document_id}", response_model=DocumentQueryResponse)
    @require_permission("documents:read")
    async def get_document(
        document_id: str,
        include_entities: bool = Query(True),
        include_clauses: bool = Query(True),
        include_alerts: bool = Query(True),
        user_context: UserContext = Depends(auth_dependency),
    ):
        """
        Get complete document information including extracted data

        Returns document metadata, extracted entities, clauses,
        compliance status, and any alerts generated.
        """
        try:
            # Get document
            document = await database_manager.get_document(
                document_id=document_id, tenant_id=user_context.tenant_id
            )

            if not document:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND, detail="Document not found"
                )

            # Check if processing is complete
            if document["status"] not in ["COMPLETED", "FAILED"]:
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail="Document is still being processed",
                )

            result = {
                "document_id": document_id,
                "filename": document["original_filename"],
                "document_type": document["document_type"],
                "language": document["language_code"],
                "status": document["status"],
                "created_at": document["created_at"],
                "processing_completed_at": document["processing_completed_at"],
                "processing_time_seconds": document["processing_time_seconds"],
                "metadata": document.get("metadata", {}),
            }

            # Get extracted entities
            if include_entities and document["status"] == "COMPLETED":
                entities = await database_manager.get_document_entities(
                    document_id=document_id, tenant_id=user_context.tenant_id
                )
                result["entities"] = entities

            # Get extracted clauses
            if include_clauses and document["status"] == "COMPLETED":
                clauses = await database_manager.get_document_clauses(
                    document_id=document_id, tenant_id=user_context.tenant_id
                )
                result["clauses"] = clauses

            # Get alerts
            if include_alerts:
                alerts = await database_manager.get_document_alerts(
                    document_id=document_id, tenant_id=user_context.tenant_id
                )
                result["alerts"] = alerts

            return DocumentQueryResponse(**result)

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error getting document {document_id}: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to get document",
            )

    @router.get("/", response_model=DocumentListResponse)
    @require_permission("documents:read")
    async def list_documents(
        status: Optional[str] = Query(None),
        document_type: Optional[str] = Query(None),
        uploaded_by: Optional[str] = Query(None),
        date_from: Optional[datetime] = Query(None),
        date_to: Optional[datetime] = Query(None),
        limit: int = Query(50, le=100),
        offset: int = Query(0, ge=0),
        user_context: UserContext = Depends(auth_dependency),
    ):
        """
        List documents with filtering and pagination

        Supports filtering by status, type, uploader, and date range.
        Results are paginated and sorted by creation date (newest first).
        """
        try:
            filters = {
                "tenant_id": user_context.tenant_id,
                "status": status,
                "document_type": document_type,
                "uploaded_by": uploaded_by,
                "date_from": date_from,
                "date_to": date_to,
            }

            # Remove None values
            filters = {k: v for k, v in filters.items() if v is not None}

            documents, total_count = await database_manager.list_documents(
                filters=filters, limit=limit, offset=offset
            )

            return DocumentListResponse(
                documents=documents,
                total_count=total_count,
                limit=limit,
                offset=offset,
                has_more=offset + len(documents) < total_count,
            )

        except Exception as e:
            logger.error(f"Error listing documents: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to list documents",
            )

    @router.delete("/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
    @require_permission("documents:delete")
    async def delete_document(
        document_id: str, user_context: UserContext = Depends(auth_dependency)
    ):
        """
        Delete a document and all associated data

        This will remove the document from storage, database,
        and vector store. This action cannot be undone.
        """
        try:
            # Check if document exists
            document = await database_manager.get_document(
                document_id=document_id, tenant_id=user_context.tenant_id
            )

            if not document:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND, detail="Document not found"
                )

            # Delete from storage
            await storage_manager.delete_document(
                tenant_id=user_context.tenant_id, storage_path=document["storage_path"]
            )

            # Delete from vector store
            try:
                await storage_manager.delete_document_embeddings(
                    document_id=document_id, tenant_id=user_context.tenant_id
                )
            except Exception as e:
                logger.warning(f"Failed to delete embeddings for {document_id}: {e}")

            # Delete from database (cascades to related data)
            await database_manager.delete_document(
                document_id=document_id, tenant_id=user_context.tenant_id
            )

            logger.info(f"Document {document_id} deleted by {user_context.email}")

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error deleting document {document_id}: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to delete document",
            )

    @router.post("/query", response_model=List[DocumentQueryResponse])
    @require_permission("documents:read")
    async def search_documents(
        query_request: DocumentQueryRequest,
        user_context: UserContext = Depends(auth_dependency),
    ):
        """
        Search documents using semantic search

        Uses vector similarity search on clause embeddings to find
        relevant documents based on natural language queries.
        """
        try:
            # Perform semantic search
            search_results = await storage_manager.semantic_search_documents(
                tenant_id=user_context.tenant_id,
                query=query_request.query,
                document_types=query_request.document_types,
                date_from=query_request.date_from,
                date_to=query_request.date_to,
                limit=query_request.limit,
                offset=query_request.offset,
            )

            # Enrich results with document data
            enriched_results = []
            for result in search_results:
                document = await database_manager.get_document(
                    document_id=result["document_id"], tenant_id=user_context.tenant_id
                )

                if document and document["status"] == "COMPLETED":
                    enriched_results.append(
                        DocumentQueryResponse(
                            document_id=result["document_id"],
                            filename=document["original_filename"],
                            document_type=document["document_type"],
                            language=document["language_code"],
                            status=document["status"],
                            created_at=document["created_at"],
                            processing_completed_at=document["processing_completed_at"],
                            processing_time_seconds=document["processing_time_seconds"],
                            relevance_score=result.get("score", 0.0),
                            matching_clauses=result.get("matching_clauses", []),
                            metadata=document.get("metadata", {}),
                        )
                    )

            return enriched_results

        except Exception as e:
            logger.error(f"Error searching documents: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Search failed",
            )

    @router.get("/stats/processing", response_model=ProcessingStatsResponse)
    @require_permission("system:read")
    async def get_processing_stats(
        user_context: UserContext = Depends(auth_dependency),
    ):
        """
        Get document processing statistics for the tenant

        Returns processing volume, timing, and performance metrics.
        """
        try:
            stats = await database_manager.get_processing_stats(
                tenant_id=user_context.tenant_id
            )

            queue_stats = await queue_manager.get_queue_stats(
                tenant_id=user_context.tenant_id
            )

            return ProcessingStatsResponse(**stats, queue_stats=queue_stats)

        except Exception as e:
            logger.error(f"Error getting processing stats: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to get processing statistics",
            )

    return router


# Helper functions


async def _validate_upload_file(file: UploadFile, tenant_id: str):
    """Validate uploaded file"""
    # Check file size
    max_size = 100 * 1024 * 1024  # 100MB
    if file.size > max_size:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File size {file.size} exceeds maximum {max_size} bytes",
        )

    # Check file type
    allowed_extensions = {
        ".pdf",
        ".docx",
        ".doc",
        ".png",
        ".jpg",
        ".jpeg",
        ".tiff",
        ".tif",
    }
    file_extension = os.path.splitext(file.filename)[1].lower()

    if file_extension not in allowed_extensions:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"File type {file_extension} not supported",
        )

    # Check filename
    if not file.filename or len(file.filename) > 255:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid filename"
        )


def _map_priority_to_int(priority: str) -> int:
    """Map priority string to integer"""
    priority_map = {"normal": 1, "high": 2, "rush": 3}
    return priority_map.get(priority.lower(), 1)


def _estimate_processing_time(file_size: int, enable_ocr: bool) -> int:
    """Estimate processing time in seconds"""
    base_time = 30  # 30 seconds base
    size_factor = file_size / (1024 * 1024)  # Size in MB
    ocr_multiplier = 3 if enable_ocr else 1

    return int(base_time + (size_factor * 10 * ocr_multiplier))


def _get_current_stage(checkpoints: Dict[str, str]) -> Optional[str]:
    """Get current processing stage from checkpoints"""
    stages = [
        "PREPROCESSING",
        "AI_EXTRACTION",
        "ANALYSIS",
        "GRAPH_UPDATE",
        "ALERT_GENERATION",
    ]

    for stage in stages:
        if checkpoints.get(stage) in ["IN_PROGRESS", "FAILED"]:
            return stage
        elif checkpoints.get(stage) != "COMPLETED":
            return stage

    return "COMPLETED"


def _calculate_progress_percentage(status: str, checkpoints: Dict[str, str]) -> int:
    """Calculate processing progress percentage"""
    if status == "COMPLETED":
        return 100
    elif status == "FAILED":
        return 0
    elif status == "QUEUED":
        return 0

    stages = [
        "PREPROCESSING",
        "AI_EXTRACTION",
        "ANALYSIS",
        "GRAPH_UPDATE",
        "ALERT_GENERATION",
    ]
    completed_stages = sum(
        1 for stage in stages if checkpoints.get(stage) == "COMPLETED"
    )

    return int((completed_stages / len(stages)) * 100)
