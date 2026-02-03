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
    schema = {
        "openapi": "3.0.0",
        "info": {
            "title": "Real-Time Document Intelligence & Compliance Engine API",
            "version": "1.0.0",
            "description": """
            A comprehensive API for intelligent document processing, compliance checking,
            and real-time alert generation for legal and financial organizations.
            """,
            "contact": {
                "name": "API Support",
                "email": "api-support@documentintelligence.ai",
            },
        },
        "servers": [
            {
                "url": "https://api.documentintelligence.ai/v1",
                "description": "Production server",
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
                    ],
                    "properties": {
                        "document_id": {
                            "type": "string",
                            "format": "uuid",
                            "description": "Unique document identifier",
                        },
                        "job_id": {
                            "type": "string",
                            "description": "Processing job identifier",
                        },
                        "filename": {
                            "type": "string",
                            "description": "Original filename",
                        },
                        "file_size": {
                            "type": "integer",
                            "description": "File size in bytes",
                        },
                        "status": {
                            "type": "string",
                            "enum": ["QUEUED", "PROCESSING", "COMPLETED", "FAILED"],
                            "description": "Current processing status",
                        },
                    },
                }
            },
        },
        "paths": {
            "/documents/upload": {
                "post": {
                    "summary": "Upload a document for processing",
                    "description": "Upload a new document file for AI-powered processing",
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
                                            "description": "Document file to upload",
                                        }
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
                        }
                    },
                }
            }
        },
        "tags": [
            {
                "name": "Documents",
                "description": "Document upload, processing, and management operations",
            }
        ],
    }

    return schema


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
    </head>
    <body>
        <div id="swagger-ui"></div>
        <script src="https://unpkg.com/swagger-ui-dist@4.15.5/swagger-ui-bundle.js"></script>
        <script>
            window.onload = function() {
                SwaggerUIBundle({
                    url: '/api/v1/openapi.json',
                    dom_id: '#swagger-ui',
                    deepLinking: true,
                    layout: "StandaloneLayout"
                });
            };
        </script>
    </body>
    </html>
    """
