"""
Real-Time Document Intelligence & Compliance Engine
API Documentation Generator
Version: 1.0
Date: 2026-02-03

This module generates comprehensive OpenAPI/Swagger documentation
for all API endpoints in the document intelligence system.
"""

import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)


def get_openapi_schema() -> Dict[str, Any]:
    """
    Generate OpenAPI 3.0.0 schema for the Document Intelligence API

    Returns:
        Complete OpenAPI schema dictionary
    """
    return {
        "openapi": "3.0.0",
        "info": {
            "title": "Real-Time Document Intelligence & Compliance Engine API",
            "version": "1.0.0",
            "description": """
            A comprehensive API for intelligent document processing, compliance checking,
            and real-time alert generation for legal and financial organizations.

            ## Features
            - Multi-tenant document upload and processing
            - OCR and AI-powered text extraction
            - Entity and clause extraction
            - Compliance rule checking
            - Real-time processing status updates
            - Advanced document search and analytics
            - WebSocket support for live notifications

            ## Authentication
            All endpoints require Bearer token authentication using JWT tokens.
            Include the token in the Authorization header: `Bearer <your-token>`

            ## Rate Limiting
            API requests are rate limited per tenant and user role.
            Rate limit headers are included in all responses.

            ## WebSocket Connection
            Connect to `/ws` endpoint for real-time updates on document processing,
            alerts, and system notifications.
            """,
            "termsOfService": "https://api.documentintelligence.ai/terms",
            "contact": {
                "name": "API Support",
                "email": "api-support@documentintelligence.ai",
                "url": "https://docs.documentintelligence.ai",
            },
            "license": {
                "name": "Commercial License",
                "url": "https://api.documentintelligence.ai/license",
            },
        },
        "servers": [
            {
                "url": "https://api.documentintelligence.ai/v1",
                "description": "Production server",
            },
            {
                "url": "https://staging-api.documentintelligence.ai/v1",
                "description": "Staging server",
            },
            {
                "url": "http://localhost:8000/api/v1",
                "description": "Development server",
            },
        ],
        "security": [{"BearerAuth": []}],
        "components": {
            "securitySchemes": {
                "BearerAuth": {
                    "type": "http",
                    "scheme": "bearer",
                    "bearerFormat": "JWT",
                    "description": "JWT Bearer token authentication",
                }
            },
            "schemas": {
                "DocumentUploadResponse": {
                    "type": "object",
                    "required": [
                        "document_id",
                        "job_id",
                        "filename",
                        "file_size",
                        "status",
                        "storage_path",
                        "estimated_processing_time",
                        "message",
                    ],
                    "properties": {
                        "document_id": {
                            "type": "string",
                            "format": "uuid",
                            "description": "Unique document identifier",
                            "example": "550e8400-e29b-41d4-a716-446655440000",
                        },
                        "job_id": {
                            "type": "string",
                            "description": "Processing job identifier",
                            "example": "job_123e4567-e89b-12d3-a456-426614174000",
                        },
                        "filename": {
                            "type": "string",
                            "description": "Original filename",
                            "example": "contract_agreement.pdf",
                        },
                        "file_size": {
                            "type": "integer",
                            "description": "File size in bytes",
                            "example": 2048576,
                        },
                        "status": {
                            "type": "string",
                            "enum": ["QUEUED", "PROCESSING", "COMPLETED", "FAILED"],
                            "description": "Current processing status",
                            "example": "QUEUED",
                        },
                        "storage_path": {
                            "type": "string",
                            "description": "Storage location path",
                            "example": "tenant_123/documents/550e8400-e29b-41d4-a716-446655440000.pdf",
                        },
                        "estimated_processing_time": {
                            "type": "integer",
                            "description": "Estimated processing time in seconds",
                            "example": 300,
                        },
                        "message": {
                            "type": "string",
                            "description": "Success message",
                            "example": "Document uploaded successfully and queued for processing",
                        },
                    },
                },
                "DocumentStatusResponse": {
                    "type": "object",
                    "required": ["document_id", "status", "progress_percentage"],
                    "properties": {
                        "document_id": {
                            "type": "string",
                            "format": "uuid",
                            "description": "Document identifier",
                        },
                        "status": {
                            "type": "string",
                            "enum": [
                                "QUEUED",
                                "PROCESSING",
                                "COMPLETED",
                                "FAILED",
                                "CANCELLED",
                            ],
                            "description": "Current status",
                        },
                        "current_stage": {
                            "type": "string",
                            "enum": [
                                "PREPROCESSING",
                                "OCR_EXTRACTION",
                                "AI_EXTRACTION",
                                "ANALYSIS",
                                "GRAPH_UPDATE",
                                "ALERT_GENERATION",
                            ],
                            "description": "Current processing stage",
                            "nullable": True,
                        },
                        "progress_percentage": {
                            "type": "integer",
                            "minimum": 0,
                            "maximum": 100,
                            "description": "Progress percentage",
                        },
                        "queue_position": {
                            "type": "integer",
                            "description": "Position in processing queue",
                            "nullable": True,
                        },
                        "processing_started_at": {
                            "type": "string",
                            "format": "date-time",
                            "description": "Processing start time",
                            "nullable": True,
                        },
                        "processing_completed_at": {
                            "type": "string",
                            "format": "date-time",
                            "description": "Processing completion time",
                            "nullable": True,
                        },
                        "processing_time_seconds": {
                            "type": "integer",
                            "description": "Total processing time",
                            "nullable": True,
                        },
                        "error_message": {
                            "type": "string",
                            "description": "Error message if failed",
                            "nullable": True,
                        },
                        "retry_count": {
                            "type": "integer",
                            "description": "Number of retry attempts",
                            "default": 0,
                        },
                        "checkpoints": {
                            "type": "object",
                            "additionalProperties": {"type": "string"},
                            "description": "Processing stage checkpoints",
                        },
                    },
                },
                "EntityResponse": {
                    "type": "object",
                    "required": [
                        "entity_id",
                        "entity_type",
                        "entity_value",
                        "confidence_score",
                    ],
                    "properties": {
                        "entity_id": {
                            "type": "string",
                            "description": "Entity identifier",
                        },
                        "entity_type": {
                            "type": "string",
                            "enum": [
                                "PERSON",
                                "ORGANIZATION",
                                "DATE",
                                "MONEY",
                                "LOCATION",
                                "LEGAL_ENTITY",
                                "CONTRACT_TERM",
                            ],
                            "description": "Type of entity",
                        },
                        "entity_value": {
                            "type": "string",
                            "description": "Entity text value",
                        },
                        "confidence_score": {
                            "type": "number",
                            "minimum": 0,
                            "maximum": 1,
                            "description": "Extraction confidence",
                        },
                        "start_char": {
                            "type": "integer",
                            "description": "Start character position",
                            "nullable": True,
                        },
                        "end_char": {
                            "type": "integer",
                            "description": "End character position",
                            "nullable": True,
                        },
                        "page_number": {
                            "type": "integer",
                            "description": "Page number",
                            "nullable": True,
                        },
                        "metadata": {
                            "type": "object",
                            "additionalProperties": True,
                            "description": "Additional metadata",
                        },
                    },
                },
                "ClauseResponse": {
                    "type": "object",
                    "required": [
                        "clause_id",
                        "clause_type",
                        "clause_text",
                        "start_char",
                        "end_char",
                        "confidence_score",
                        "risk_score",
                    ],
                    "properties": {
                        "clause_id": {
                            "type": "string",
                            "description": "Clause identifier",
                        },
                        "clause_type": {
                            "type": "string",
                            "enum": [
                                "PAYMENT_TERMS",
                                "TERMINATION",
                                "CONFIDENTIALITY",
                                "LIABILITY",
                                "GOVERNING_LAW",
                                "FORCE_MAJEURE",
                            ],
                            "description": "Type of clause",
                        },
                        "clause_text": {
                            "type": "string",
                            "description": "Clause content",
                        },
                        "start_char": {
                            "type": "integer",
                            "description": "Start character position",
                        },
                        "end_char": {
                            "type": "integer",
                            "description": "End character position",
                        },
                        "page_number": {
                            "type": "integer",
                            "description": "Page number",
                            "nullable": True,
                        },
                        "confidence_score": {
                            "type": "number",
                            "minimum": 0,
                            "maximum": 1,
                            "description": "Classification confidence",
                        },
                        "risk_score": {
                            "type": "number",
                            "minimum": 0,
                            "maximum": 1,
                            "description": "Risk assessment score",
                        },
                        "entities": {
                            "type": "array",
                            "items": {"$ref": "#/components/schemas/EntityResponse"},
                            "description": "Related entities",
                        },
                    },
                },
                "AlertResponse": {
                    "type": "object",
                    "required": [
                        "alert_id",
                        "alert_type",
                        "severity",
                        "title",
                        "description",
                        "triggered_by_rule",
                        "created_at",
                    ],
                    "properties": {
                        "alert_id": {
                            "type": "string",
                            "description": "Alert identifier",
                        },
                        "alert_type": {
                            "type": "string",
                            "enum": [
                                "COMPLIANCE_VIOLATION",
                                "HIGH_RISK_CLAUSE",
                                "MISSING_CLAUSE",
                                "REGULATORY_ISSUE",
                            ],
                            "description": "Type of alert",
                        },
                        "severity": {
                            "type": "string",
                            "enum": ["LOW", "MEDIUM", "HIGH", "CRITICAL"],
                            "description": "Alert severity level",
                        },
                        "title": {"type": "string", "description": "Alert title"},
                        "description": {
                            "type": "string",
                            "description": "Detailed description",
                        },
                        "triggered_by_rule": {
                            "type": "string",
                            "description": "Compliance rule that triggered alert",
                        },
                        "related_clauses": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Related clause IDs",
                        },
                        "created_at": {
                            "type": "string",
                            "format": "date-time",
                            "description": "Alert creation time",
                        },
                        "acknowledged_at": {
                            "type": "string",
                            "format": "date-time",
                            "description": "Acknowledgment time",
                            "nullable": True,
                        },
                    },
                },
                "DocumentQueryResponse": {
                    "type": "object",
                    "required": [
                        "document_id",
                        "filename",
                        "document_type",
                        "language",
                        "status",
                        "created_at",
                    ],
                    "properties": {
                        "document_id": {
                            "type": "string",
                            "format": "uuid",
                            "description": "Document identifier",
                        },
                        "filename": {
                            "type": "string",
                            "description": "Original filename",
                        },
                        "document_type": {
                            "type": "string",
                            "enum": [
                                "CONTRACT",
                                "INVOICE",
                                "LEGAL_BRIEF",
                                "COMPLIANCE_REPORT",
                                "OTHER",
                            ],
                            "description": "Document type classification",
                        },
                        "language": {
                            "type": "string",
                            "description": "Document language code",
                        },
                        "status": {
                            "type": "string",
                            "enum": ["QUEUED", "PROCESSING", "COMPLETED", "FAILED"],
                            "description": "Processing status",
                        },
                        "created_at": {
                            "type": "string",
                            "format": "date-time",
                            "description": "Upload timestamp",
                        },
                        "processing_completed_at": {
                            "type": "string",
                            "format": "date-time",
                            "description": "Processing completion time",
                            "nullable": True,
                        },
                        "processing_time_seconds": {
                            "type": "integer",
                            "description": "Processing duration",
                            "nullable": True,
                        },
                        "relevance_score": {
                            "type": "number",
                            "minimum": 0,
                            "maximum": 1,
                            "description": "Search relevance score",
                            "nullable": True,
                        },
                        "metadata": {
                            "type": "object",
                            "additionalProperties": True,
                            "description": "Document metadata",
                        },
                        "entities": {
                            "type": "array",
                            "items": {"$ref": "#/components/schemas/EntityResponse"},
                            "description": "Extracted entities",
                            "nullable": True,
                        },
                        "clauses": {
                            "type": "array",
                            "items": {"$ref": "#/components/schemas/ClauseResponse"},
                            "description": "Extracted clauses",
                            "nullable": True,
                        },
                        "alerts": {
                            "type": "array",
                            "items": {"$ref": "#/components/schemas/AlertResponse"},
                            "description": "Generated alerts",
                            "nullable": True,
                        },
                        "matching_clauses": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Clauses matching search",
                            "nullable": True,
                        },
                    },
                },
                "ErrorResponse": {
                    "type": "object",
                    "required": ["error", "message", "timestamp"],
                    "properties": {
                        "error": {
                            "type": "string",
                            "description": "Error type",
                            "example": "ValidationError",
                        },
                        "message": {
                            "type": "string",
                            "description": "Error message",
                            "example": "File size exceeds maximum allowed limit",
                        },
                        "detail": {
                            "type": "string",
                            "description": "Detailed error information",
                            "nullable": True,
                        },
                        "timestamp": {
                            "type": "string",
                            "format": "date-time",
                            "description": "Error timestamp",
                        },
                        "request_id": {
                            "type": "string",
                            "description": "Request identifier",
                            "nullable": True,
                        },
                    },
                },
            },
            "parameters": {
                "DocumentId": {
                    "name": "document_id",
                    "in": "path",
                    "required": True,
                    "schema": {"type": "string", "format": "uuid"},
                    "description": "Unique document identifier",
                },
                "TenantId": {
                    "name": "x-tenant-id",
                    "in": "header",
                    "required": False,
                    "schema": {"type": "string"},
                    "description": "Tenant identifier (extracted from JWT if not provided)",
                },
                "Limit": {
                    "name": "limit",
                    "in": "query",
                    "required": False,
                    "schema": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 100,
                        "default": 50,
                    },
                    "description": "Maximum number of results to return",
                },
                "Offset": {
                    "name": "offset",
                    "in": "query",
                    "required": False,
                    "schema": {"type": "integer", "minimum": 0, "default": 0},
                    "description": "Number of results to skip",
                },
            },
            "responses": {
                "ValidationError": {
                    "description": "Validation error",
                    "content": {
                        "application/json": {
                            "schema": {"$ref": "#/components/schemas/ErrorResponse"},
                            "example": {
                                "error": "ValidationError",
                                "message": "File size exceeds maximum allowed limit",
                                "detail": "File size 104857600 bytes exceeds maximum 100000000 bytes",
                                "timestamp": "2026-02-03T10:30:00Z",
                                "request_id": "req_123456789",
                            },
                        }
                    },
                },
                "AuthenticationError": {
                    "description": "Authentication required",
                    "content": {
                        "application/json": {
                            "schema": {"$ref": "#/components/schemas/ErrorResponse"},
                            "example": {
                                "error": "AuthenticationError",
                                "message": "Invalid or expired token",
                                "timestamp": "2026-02-03T10:30:00Z",
                            },
                        }
                    },
                },
                "AuthorizationError": {
                    "description": "Insufficient permissions",
                    "content": {
                        "application/json": {
                            "schema": {"$ref": "#/components/schemas/ErrorResponse"},
                            "example": {
                                "error": "AuthorizationError",
                                "message": "Insufficient permissions for this operation",
                                "timestamp": "2026-02-03T10:30:00Z",
                            },
                        }
                    },
                },
                "NotFoundError": {
                    "description": "Resource not found",
                    "content": {
                        "application/json": {
                            "schema": {"$ref": "#/components/schemas/ErrorResponse"},
                            "example": {
                                "error": "NotFoundError",
                                "message": "Document not found",
                                "timestamp": "2026-02-03T10:30:00Z",
                            },
                        }
                    },
                },
                "RateLimitError": {
                    "description": "Rate limit exceeded",
                    "headers": {
                        "X-RateLimit-Limit": {
                            "schema": {"type": "integer"},
                            "description": "Request limit per time window",
                        },
                        "X-RateLimit-Remaining": {
                            "schema": {"type": "integer"},
                            "description": "Requests remaining in current window",
                        },
                        "X-RateLimit-Reset": {
                            "schema": {"type": "integer"},
                            "description": "Time when rate limit resets (Unix timestamp)",
                        },
                    },
                    "content": {
                        "application/json": {
                            "schema": {"$ref": "#/components/schemas/ErrorResponse"},
                            "example": {
                                "error": "RateLimitError",
                                "message": "Rate limit exceeded. Try again later.",
                                "timestamp": "2026-02-03T10:30:00Z",
                            },
                        }
                    },
                },
            },
        },
        "paths": {
            "/documents/upload": {
                "post": {
                    "summary": "Upload a document for processing",
                    "description": "Upload a new document file for AI-powered processing including OCR, entity extraction, clause analysis, and compliance checking.",
                    "tags": ["Documents"],
                    "security": [{"BearerAuth": []}],
                    "requestBody": {
                        "required": True,
                        "content": {
                            "multipart/form-data": {
                                "schema": {
                                    "type": "object",
                                    "required": ["file"],
                                    "properties": {
                                        "file": {
                                            "type": "string",
                                            "format": "binary",
                                            "description": "Document file to upload (PDF, DOCX, images)",
                                        },
                                        "document_type": {
                                            "type": "string",
                                            "enum": [
                                                "CONTRACT",
                                                "INVOICE",
                                                "LEGAL_BRIEF",
                                                "COMPLIANCE_REPORT",
                                                "OTHER",
                                            ],
                                            "description": "Override automatic document classification",
                                        },
                                        "priority": {
                                            "type": "string",
                                            "enum": ["normal", "high", "rush"],
                                            "default": "normal",
                                            "description": "Processing priority level",
                                        },
                                        "language": {
                                            "type": "string",
                                            "default": "en",
                                            "description": "Document language for OCR optimization",
                                        },
                                        "enable_ocr": {
                                            "type": "boolean",
                                            "default": True,
                                            "description": "Whether to perform OCR on the document",
                                        },
                                        "enable_classification": {
                                            "type": "boolean",
                                            "default": True,
                                            "description": "Whether to classify document type",
                                        },
                                    },
                                }
                            }
                        },
                    },
                    "responses": {
                        "201": {
                            "description": "Document uploaded successfully",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "$ref": "#/components/schemas/DocumentUploadResponse"
                                    }
                                }
                            },
                        },
                        "400": {"$ref": "#/components/responses/ValidationError"},
                        "401": {"$ref": "#/components/responses/AuthenticationError"},
                        "403": {"$ref": "#/components/responses/AuthorizationError"},
                        "413": {
                            "description": "File too large",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "$ref": "#/components/schemas/ErrorResponse"
                                    }
                                }
                            },
                        },
                        "429": {"$ref": "#/components/responses/RateLimitError"},
                        "500": {
                            "description": "Internal server error",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "$ref": "#/components/schemas/ErrorResponse"
                                    }
                                }
                            },
                        },
                    },
                }
            },
            "/documents/{document_id}/status": {
                "get": {
                    "summary": "Get document processing status",
                    "description": "Retrieve the current processing status, progress, and any errors for a specific document.",
                    "tags": ["Documents"],
                    "security": [{"BearerAuth": []}],
                    "parameters": [{"$ref": "#/components/parameters/DocumentId"}],
                    "responses": {
                        "200": {
                            "description": "Document status retrieved successfully",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "$ref": "#/components/schemas/DocumentStatusResponse"
                                    }
                                }
                            },
                        },
                        "401": {"$ref": "#/components/responses/AuthenticationError"},
                        "403": {"$ref": "#/components/responses/AuthorizationError"},
                        "404": {"$ref": "#/components/responses/NotFoundError"},
                        "429": {"$ref": "#/components/responses/RateLimitError"},
                    },
                }
            },
            "/documents/{document_id}": {
                "get": {
                    "summary": "Get complete document information",
                    "description": "Retrieve complete document information including extracted entities, clauses, compliance status, and alerts.",
                    "tags": ["Documents"],
                    "security": [{"BearerAuth": []}],
                    "parameters": [
                        {"$ref": "#/components/parameters/DocumentId"},
                        {
                            "name": "include_entities",
                            "in": "query",
                            "schema": {"type": "boolean", "default": True},
                            "description": "Include extracted entities in response",
                        },
                        {
                            "name": "include_clauses",
                            "in": "query",
                            "schema": {"type": "boolean", "default": True},
                            "description": "Include extracted clauses in response",
                        },
                        {
                            "name": "include_alerts",
                            "in": "query",
                            "schema": {"type": "boolean", "default": True},
                            "description": "Include generated alerts in response",
                        },
                    ],
                    "responses": {
                        "200": {
                            "description": "Document information retrieved successfully",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "$ref": "#/components/schemas/DocumentQueryResponse"
                                    }
                                }
                            },
                        },
                        "401": {"$ref": "#/components/responses/AuthenticationError"},
                        "403": {"$ref": "#/components/responses/AuthorizationError"},
                        "404": {"$ref": "#/components/responses/NotFoundError"},
                        "409": {
                            "description": "Document still being processed",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "$ref": "#/components/schemas/ErrorResponse"
                                    }
                                }
                            },
                        },
                        "429": {"$ref": "#/components/responses/RateLimitError"},
                    },
                },
                "delete": {
                    "summary": "Delete a document",
                    "description": "Permanently delete a document and all associated data including storage files, database records, and vector embeddings.",
                    "tags": ["Documents"],
                    "security": [{"BearerAuth": []}],
                    "parameters": [{"$ref": "#/components/parameters/DocumentId"}],
                    "responses": {
                        "204": {"description": "Document deleted successfully"},
                        "401": {"$ref": "#/components/responses/AuthenticationError"},
                        "403": {"$ref": "#/components/responses/AuthorizationError"},
                        "404": {"$ref": "#/components/responses/NotFoundError"},
                        "429": {"$ref": "#/components/responses/RateLimitError"},
                    },
                },
            },
            "/documents": {
                "get": {
                    "summary": "List documents with filtering",
                    "description": "Retrieve a paginated list of documents with optional filtering by status, type, uploader, and date range.",
                    "tags": ["Documents"],
                    "security": [{"BearerAuth": []}],
                    "parameters": [
                        {
                            "name": "status",
                            "in": "query",
                            "schema": {
                                "type": "string",
                                "enum": ["QUEUED", "PROCESSING", "COMPLETED", "FAILED"],
                            },
                            "description": "Filter by processing status",
                        },
                        {
                            "name": "document_type",
                            "in": "query",
                            "schema": {
                                "type": "string",
                                "enum": [
                                    "CONTRACT",
                                    "INVOICE",
                                    "LEGAL_BRIEF",
                                    "COMPLIANCE_REPORT",
                                    "OTHER",
                                ],
                            },
                            "description": "Filter by document type",
                        },
                        {
                            "name": "uploaded_by",
                            "in": "query",
                            "schema": {"type": "string"},
                            "description": "Filter by uploader user ID",
                        },
                        {
                            "name": "date_from",
                            "in": "query",
                            "schema": {"type": "string", "format": "date-time"},
                            "description": "Filter documents uploaded from this date",
                        },
                        {
                            "name": "date_to",
                            "in": "query",
                            "schema": {"type": "string", "format": "date-time"},
                            "description": "Filter documents uploaded until this date",
                        },
                        {"$ref": "#/components/parameters/Limit"},
                        {"$ref": "#/components/parameters/Offset"},
                    ],
                    "responses": {
                        "200": {
                            "description": "Documents retrieved successfully",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "properties": {
                                            "documents": {
                                                "type": "array",
                                                "items": {
                                                    "$ref": "#/components/schemas/DocumentQueryResponse"
                                                },
                                            },
                                            "total_count": {"type": "integer"},
                                            "limit": {"type": "integer"},
                                            "offset": {"type": "integer"},
                                            "has_more": {"type": "boolean"},
                                        },
                                    }
                                }
                            },
                        },
                        "401": {"$ref": "#/components/responses/AuthenticationError"},
                        "403": {"$ref": "#/components/responses/AuthorizationError"},
                        "429": {"$ref": "#/components/responses/RateLimitError"},
                    },
                }
            },
            "/documents/query": {
                "post": {
                    "summary": "Search documents using semantic search",
                    "description": "Perform semantic search across documents using natural language queries to find relevant content based on clause embeddings.",
                    "tags": ["Documents"],
                    "security": [{"BearerAuth": []}],
                    "requestBody": {
                        "required": True,
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "required": ["query"],
                                    "properties": {
                                        "query": {
                                            "type": "string",
                                            "description": "Natural language search query",
                                            "example": "payment terms and late fees",
                                        },
                                        "document_types": {
                                            "type": "array",
                                            "items": {"type": "string"},
                                            "description": "Filter by document types",
                                        },
                                        "date_from": {
                                            "type": "string",
                                            "format": "date-time",
                                            "description": "Filter documents from this date",
                                        },
                                        "date_to": {
                                            "type": "string",
                                            "format": "date-time",
                                            "description": "Filter documents until this date",
                                        },
                                        "limit": {
                                            "type": "integer",
                                            "minimum": 1,
                                            "maximum": 100,
                                            "default": 10,
                                            "description": "Maximum number of results",
                                        },
                                        "offset": {
                                            "type": "integer",
                                            "minimum": 0,
                                            "default": 0,
                                            "description": "Results offset",
                                        },
                                    },
                                }
                            }
                        },
                    },
                    "responses": {
                        "200": {
                            "description": "Search completed successfully",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "array",
                                        "items": {
                                            "$ref": "#/components/schemas/DocumentQueryResponse"
                                        },
                                    }
                                }
                            },
                        },
                        "400": {"$ref": "#/components/responses/ValidationError"},
                        "401": {"$ref": "#/components/responses/AuthenticationError"},
                        "403": {"$ref": "#/components/responses/AuthorizationError"},
                        "429": {"$ref": "#/components/responses/RateLimitError"},
                    },
                }
            },
            "/documents/stats/processing": {
                "get": {
                    "summary": "Get processing statistics",
                    "description": "Retrieve document processing statistics including volume, timing, performance metrics, and queue status.",
                    "tags": ["Analytics"],
                    "security": [{"BearerAuth": []}],
                    "responses": {
                        "200": {
                            "description": "Statistics retrieved successfully",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "properties": {
                                            "total_documents": {"type": "integer"},
                                            "completed_documents": {"type": "integer"},
                                            "failed_documents": {"type": "integer"},
                                            "processing_documents": {"type": "integer"},
                                            "queued_documents": {"type": "integer"},
                                            "average_processing_time_seconds": {
                                                "type": "number"
                                            },
                                            "total_processing_time_hours": {
                                                "type": "number"
                                            },
                                            "documents_processed_today": {
                                                "type": "integer"
                                            },
                                            "documents_processed_this_week": {
                                                "type": "integer"
                                            },
                                            "documents_processed_this_month": {
                                                "type": "integer"
                                            },
                                            "queue_stats": {
                                                "type": "object",
                                                "properties": {
                                                    "pending_jobs": {"type": "integer"},
                                                    "processing_jobs": {
                                                        "type": "integer"
                                                    },
                                                    "failed_jobs": {"type": "integer"},
                                                    "average_wait_time_seconds": {
                                                        "type": "number"
                                                    },
                                                    "average_processing_time_seconds": {
                                                        "type": "number"
                                                    },
                                                },
                                            },
                                        },
                                    }
                                }
                            },
                        },
                        "401": {"$ref": "#/components/responses/AuthenticationError"},
                        "403": {"$ref": "#/components/responses/AuthorizationError"},
                        "429": {"$ref": "#/components/responses/RateLimitError"},
                    },
                }
            },
            "/ws": {
                "get": {
                    "summary": "WebSocket connection for real-time updates",
                    "description": "Establish a WebSocket connection to receive real-time updates on document processing status, alerts, and system notifications.",
                    "tags": ["WebSocket"],
                    "parameters": [
                        {
                            "name": "token",
                            "in": "query",
                            "required": True,
                            "schema": {"type": "string"},
                            "description": "JWT authentication token",
                        }
                    ],
                    "responses": {
                        "101": {"description": "WebSocket connection established"},
                        "401": {"$ref": "#/components/responses/AuthenticationError"},
                        "403": {"$ref": "#/components/responses/AuthorizationError"},
                    },
                }
            },
        },
        "tags": [
            {
                "name": "Documents",
                "description": "Document upload, processing, and management operations",
            },
            {
                "name": "Analytics",
                "description": "Processing statistics and analytics endpoints",
            },
            {"name": "WebSocket", "description": "Real-time communication endpoints"},
        ],
    }


def generate_api_documentation_html() -> str:
    """
    Generate HTML documentation page with embedded Swagger UI

    Returns:
        HTML string for API documentation page
    """
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Document Intelligence API Documentation</title>
        <link rel="stylesheet" type="text/css" href="https://unpkg.com/swagger-ui-dist@4.15.5/swagger-ui.css" />
        <style>
            html { box-sizing: border-box; overflow: -moz-scrollbars-vertical; overflow-y: scroll; }
            *, *:before, *:after { box-sizing: inherit; }
            body { margin:0; background: #fafafa; }
        </style>
    </head>
    <body>
        <div id="swagger-ui"></div>
        <script src="https://unpkg.com/swagger-ui-dist@4.15.5/swagger-ui-bundle.js"></script>
        <script src="https://unpkg.com/swagger-ui-dist@4.15.5/swagger-ui-standalone-preset.js"></script>
        <script>
            window.onload = function() {
                const ui = SwaggerUIBundle({
                    url: '/api/v1/openapi.json',
                    dom_id: '#swagger-ui',
                    deepLinking: true,
                    presets: [
                        SwaggerUIBundle.presets.apis,
                        SwaggerUIStandalonePreset
                    ],
                    plugins: [
                        SwaggerUIBundle.plugins.DownloadUrl
                    ],
                    layout: "StandaloneLayout",
                    tryItOutEnabled: true,
                    requestInterceptor: function(request) {
                        // Auto-add authorization header if token exists in localStorage
                        const token = localStorage.getItem('api_token');
                        if (token && !request.headers.Authorization) {
                            request.headers.Authorization = 'Bearer ' + token;
                        }
                        return request;
                    }
                });
            };
        </script>
    </body>
    </html>
    """
