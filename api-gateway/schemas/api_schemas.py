"""
Real-Time Document Intelligence & Compliance Engine
API Response Schemas
Version: 1.0
Date: 2026-02-03

This module defines Pydantic models for API request/response schemas.
"""

from datetime import datetime
from typing import List, Optional, Dict, Any

try:
    from pydantic import BaseModel, Field
except ImportError:
    # For development - these will be available at runtime
    BaseModel = object

    def Field(*args, **kwargs):
        return None


class DocumentUploadResponse(BaseModel):
    """Response for document upload"""

    document_id: str = Field(..., description="Unique document identifier")
    job_id: str = Field(..., description="Processing job identifier")
    filename: str = Field(..., description="Original filename")
    file_size: int = Field(..., description="File size in bytes")
    status: str = Field(..., description="Current processing status")
    storage_path: str = Field(..., description="Storage location")
    estimated_processing_time: int = Field(
        ..., description="Estimated processing time in seconds"
    )
    message: str = Field(..., description="Success message")


class DocumentStatusResponse(BaseModel):
    """Response for document status query"""

    document_id: str = Field(..., description="Document identifier")
    status: str = Field(..., description="Current status")
    current_stage: Optional[str] = Field(None, description="Current processing stage")
    progress_percentage: int = Field(..., description="Progress percentage (0-100)")
    queue_position: Optional[int] = Field(
        None, description="Position in processing queue"
    )
    processing_started_at: Optional[datetime] = Field(
        None, description="Processing start time"
    )
    processing_completed_at: Optional[datetime] = Field(
        None, description="Processing completion time"
    )
    processing_time_seconds: Optional[int] = Field(
        None, description="Total processing time"
    )
    error_message: Optional[str] = Field(None, description="Error message if failed")
    retry_count: int = Field(0, description="Number of retry attempts")
    checkpoints: Dict[str, str] = Field(
        default_factory=dict, description="Processing stage checkpoints"
    )


class EntityResponse(BaseModel):
    """Response for extracted entity"""

    entity_id: str = Field(..., description="Entity identifier")
    entity_type: str = Field(..., description="Type of entity")
    entity_value: str = Field(..., description="Entity text value")
    confidence_score: float = Field(..., description="Extraction confidence (0-1)")
    start_char: Optional[int] = Field(None, description="Start character position")
    end_char: Optional[int] = Field(None, description="End character position")
    page_number: Optional[int] = Field(None, description="Page number")
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata"
    )


class ClauseResponse(BaseModel):
    """Response for extracted clause"""

    clause_id: str = Field(..., description="Clause identifier")
    clause_type: str = Field(..., description="Type of clause")
    clause_text: str = Field(..., description="Clause content")
    start_char: int = Field(..., description="Start character position")
    end_char: int = Field(..., description="End character position")
    page_number: Optional[int] = Field(None, description="Page number")
    confidence_score: float = Field(..., description="Classification confidence")
    risk_score: float = Field(..., description="Risk assessment score")
    entities: List[EntityResponse] = Field(
        default_factory=list, description="Related entities"
    )


class AlertResponse(BaseModel):
    """Response for compliance alert"""

    alert_id: str = Field(..., description="Alert identifier")
    alert_type: str = Field(..., description="Type of alert")
    severity: str = Field(..., description="Alert severity level")
    title: str = Field(..., description="Alert title")
    description: str = Field(..., description="Detailed description")
    triggered_by_rule: str = Field(
        ..., description="Compliance rule that triggered alert"
    )
    related_clauses: List[str] = Field(
        default_factory=list, description="Related clause IDs"
    )
    created_at: datetime = Field(..., description="Alert creation time")
    acknowledged_at: Optional[datetime] = Field(None, description="Acknowledgment time")


class DocumentQueryResponse(BaseModel):
    """Response for document query"""

    document_id: str = Field(..., description="Document identifier")
    filename: str = Field(..., description="Original filename")
    document_type: str = Field(..., description="Document type classification")
    language: str = Field(..., description="Document language")
    status: str = Field(..., description="Processing status")
    created_at: datetime = Field(..., description="Upload timestamp")
    processing_completed_at: Optional[datetime] = Field(
        None, description="Processing completion time"
    )
    processing_time_seconds: Optional[int] = Field(
        None, description="Processing duration"
    )
    relevance_score: Optional[float] = Field(None, description="Search relevance score")
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Document metadata"
    )
    entities: Optional[List[EntityResponse]] = Field(
        None, description="Extracted entities"
    )
    clauses: Optional[List[ClauseResponse]] = Field(
        None, description="Extracted clauses"
    )
    alerts: Optional[List[AlertResponse]] = Field(None, description="Generated alerts")
    matching_clauses: Optional[List[str]] = Field(
        None, description="Clauses matching search"
    )


class DocumentSummary(BaseModel):
    """Summary information for document list"""

    document_id: str = Field(..., description="Document identifier")
    filename: str = Field(..., description="Original filename")
    document_type: str = Field(..., description="Document type")
    status: str = Field(..., description="Processing status")
    created_at: datetime = Field(..., description="Upload timestamp")
    uploaded_by: str = Field(..., description="Uploader user ID")
    file_size_bytes: int = Field(..., description="File size")
    processing_time_seconds: Optional[int] = Field(
        None, description="Processing duration"
    )


class DocumentListResponse(BaseModel):
    """Response for document list query"""

    documents: List[DocumentSummary] = Field(..., description="List of documents")
    total_count: int = Field(..., description="Total number of documents")
    limit: int = Field(..., description="Query limit")
    offset: int = Field(..., description="Query offset")
    has_more: bool = Field(..., description="Whether more results exist")


class QueueStats(BaseModel):
    """Processing queue statistics"""

    pending_jobs: int = Field(..., description="Number of pending jobs")
    processing_jobs: int = Field(..., description="Number of jobs being processed")
    failed_jobs: int = Field(..., description="Number of failed jobs")
    average_wait_time_seconds: float = Field(..., description="Average queue wait time")
    average_processing_time_seconds: float = Field(
        ..., description="Average processing time"
    )


class ProcessingStatsResponse(BaseModel):
    """Response for processing statistics"""

    total_documents: int = Field(..., description="Total documents processed")
    completed_documents: int = Field(
        ..., description="Successfully processed documents"
    )
    failed_documents: int = Field(..., description="Failed processing documents")
    processing_documents: int = Field(..., description="Currently processing documents")
    queued_documents: int = Field(..., description="Documents in queue")
    average_processing_time_seconds: float = Field(
        ..., description="Average processing time"
    )
    total_processing_time_hours: float = Field(..., description="Total processing time")
    documents_processed_today: int = Field(..., description="Documents processed today")
    documents_processed_this_week: int = Field(
        ..., description="Documents processed this week"
    )
    documents_processed_this_month: int = Field(
        ..., description="Documents processed this month"
    )
    queue_stats: QueueStats = Field(..., description="Current queue statistics")


class ErrorResponse(BaseModel):
    """Error response schema"""

    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Detailed error information")
    timestamp: datetime = Field(
        default_factory=datetime.utcnow, description="Error timestamp"
    )
    request_id: Optional[str] = Field(None, description="Request identifier")


class WebSocketMessage(BaseModel):
    """WebSocket message schema"""

    type: str = Field(..., description="Message type")
    timestamp: datetime = Field(
        default_factory=datetime.utcnow, description="Message timestamp"
    )
    tenant_id: str = Field(..., description="Tenant identifier")
    data: Dict[str, Any] = Field(..., description="Message payload")


class DocumentStatusUpdate(WebSocketMessage):
    """Document status update message"""

    type: str = Field("document_status", description="Message type")
    document_id: str = Field(..., description="Document identifier")
    status: str = Field(..., description="New status")
    stage: Optional[str] = Field(None, description="Current processing stage")
    progress: Optional[int] = Field(None, description="Progress percentage")
    error: Optional[str] = Field(None, description="Error message")


class AlertNotification(WebSocketMessage):
    """Alert notification message"""

    type: str = Field("alert", description="Message type")
    alert_id: str = Field(..., description="Alert identifier")
    document_id: str = Field(..., description="Related document")
    severity: str = Field(..., description="Alert severity")
    title: str = Field(..., description="Alert title")
    description: str = Field(..., description="Alert description")
