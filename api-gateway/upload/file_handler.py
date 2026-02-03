"""
Real-Time Document Intelligence & Compliance Engine
File Upload Handler
Version: 1.0
Date: 2026-02-03

This module handles secure file uploads with validation, virus scanning,
and storage management for the document intelligence system.
"""

import os
import hashlib
import tempfile
import logging
from typing import Optional, Dict, Any, BinaryIO
from datetime import datetime

logger = logging.getLogger(__name__)


class FileUploadHandler:
    """
    Handles secure file uploads with comprehensive validation and security checks
    """

    def __init__(
        self,
        storage_manager,
        virus_scanner=None,
        max_file_size: int = 100 * 1024 * 1024,  # 100MB
        allowed_extensions: set = None,
    ):
        """
        Initialize file upload handler

        Args:
            storage_manager: Storage manager instance
            virus_scanner: Optional virus scanning service
            max_file_size: Maximum file size in bytes
            allowed_extensions: Set of allowed file extensions
        """
        self.storage_manager = storage_manager
        self.virus_scanner = virus_scanner
        self.max_file_size = max_file_size
        self.allowed_extensions = allowed_extensions or {
            ".pdf",
            ".docx",
            ".doc",
            ".png",
            ".jpg",
            ".jpeg",
            ".tiff",
            ".tif",
            ".bmp",
            ".gif",
            ".txt",
            ".rtf",
        }

        # Supported MIME types
        self.allowed_mime_types = {
            "application/pdf",
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            "application/msword",
            "image/png",
            "image/jpeg",
            "image/tiff",
            "image/bmp",
            "image/gif",
            "text/plain",
            "application/rtf",
        }

    async def validate_and_upload(
        self,
        tenant_id: str,
        document_id: str,
        file_data: BinaryIO,
        filename: str,
        content_type: Optional[str] = None,
        uploader_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Validate and upload file with comprehensive security checks

        Args:
            tenant_id: Tenant identifier
            document_id: Document identifier
            file_data: File binary data
            filename: Original filename
            content_type: MIME content type
            uploader_id: User ID of uploader

        Returns:
            Dict containing upload results and metadata

        Raises:
            ValueError: For validation errors
            SecurityError: For security violations
            RuntimeError: For processing errors
        """
        try:
            # Reset file pointer
            file_data.seek(0)

            # Read file content
            file_content = file_data.read()
            file_size = len(file_content)

            # Reset for further processing
            file_data.seek(0)

            # Validate file size
            self._validate_file_size(file_size)

            # Validate filename
            self._validate_filename(filename)

            # Validate file extension
            file_extension = self._validate_file_extension(filename)

            # Validate MIME type
            detected_mime_type = await self._detect_mime_type(file_content, filename)
            self._validate_mime_type(detected_mime_type, content_type)

            # Generate file hash for deduplication
            file_hash = self._generate_file_hash(file_content)

            # Check for duplicates
            duplicate_info = await self._check_for_duplicates(
                tenant_id, file_hash, filename
            )

            # Perform virus scan if available
            scan_result = await self._scan_for_viruses(file_content, filename)

            # Validate file structure
            structure_validation = await self._validate_file_structure(
                file_content, file_extension, detected_mime_type
            )

            # Extract metadata
            metadata = await self._extract_file_metadata(
                file_content, filename, detected_mime_type
            )

            # Upload to storage
            storage_path = await self.storage_manager.upload_document(
                tenant_id=tenant_id,
                document_id=document_id,
                file_data=file_data,
                filename=filename,
                content_type=detected_mime_type,
            )

            # Compile upload results
            upload_result = {
                "document_id": document_id,
                "storage_path": storage_path,
                "original_filename": filename,
                "file_size_bytes": file_size,
                "file_extension": file_extension,
                "mime_type": detected_mime_type,
                "file_hash": file_hash,
                "upload_timestamp": datetime.utcnow(),
                "uploader_id": uploader_id,
                "validation": {
                    "virus_scan": scan_result,
                    "structure_validation": structure_validation,
                    "duplicate_check": duplicate_info,
                },
                "metadata": metadata,
            }

            logger.info(f"File uploaded successfully: {document_id} ({filename})")
            return upload_result

        except Exception as e:
            logger.error(f"File upload failed for {filename}: {e}")
            raise

    def _validate_file_size(self, file_size: int) -> None:
        """Validate file size against limits"""
        if file_size == 0:
            raise ValueError("File is empty")

        if file_size > self.max_file_size:
            raise ValueError(
                f"File size ({file_size} bytes) exceeds maximum allowed size "
                f"({self.max_file_size} bytes)"
            )

    def _validate_filename(self, filename: str) -> None:
        """Validate filename for security and format"""
        if not filename:
            raise ValueError("Filename cannot be empty")

        if len(filename) > 255:
            raise ValueError("Filename too long (max 255 characters)")

        # Check for dangerous characters
        dangerous_chars = ["<", ">", ":", '"', "|", "?", "*", "\0"]
        if any(char in filename for char in dangerous_chars):
            raise ValueError("Filename contains invalid characters")

        # Check for path traversal
        if ".." in filename or filename.startswith("/") or filename.startswith("\\"):
            raise ValueError("Filename contains invalid path sequences")

        # Check for reserved names (Windows)
        reserved_names = {
            "CON",
            "PRN",
            "AUX",
            "NUL",
            "COM1",
            "COM2",
            "COM3",
            "COM4",
            "COM5",
            "COM6",
            "COM7",
            "COM8",
            "COM9",
            "LPT1",
            "LPT2",
            "LPT3",
            "LPT4",
            "LPT5",
            "LPT6",
            "LPT7",
            "LPT8",
            "LPT9",
        }
        base_name = os.path.splitext(filename)[0].upper()
        if base_name in reserved_names:
            raise ValueError("Filename uses reserved system name")

    def _validate_file_extension(self, filename: str) -> str:
        """Validate and return file extension"""
        file_extension = os.path.splitext(filename)[1].lower()

        if not file_extension:
            raise ValueError("File must have an extension")

        if file_extension not in self.allowed_extensions:
            raise ValueError(
                f"File extension '{file_extension}' not allowed. "
                f"Allowed extensions: {sorted(self.allowed_extensions)}"
            )

        return file_extension

    async def _detect_mime_type(self, file_content: bytes, filename: str) -> str:
        """Detect MIME type from file content"""
        try:
            # Try to use python-magic if available
            try:
                import magic

                mime_type = magic.from_buffer(file_content, mime=True)
                return mime_type
            except ImportError:
                pass

            # Fallback to mimetypes module
            import mimetypes

            mime_type, _ = mimetypes.guess_type(filename)

            if mime_type:
                return mime_type

            # Basic detection based on file signatures
            if file_content.startswith(b"%PDF"):
                return "application/pdf"
            elif file_content.startswith(b"PK\x03\x04"):
                if filename.lower().endswith(".docx"):
                    return "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
            elif file_content.startswith((b"\x89PNG", b"\xff\xd8\xff")):
                if file_content.startswith(b"\x89PNG"):
                    return "image/png"
                else:
                    return "image/jpeg"

            return "application/octet-stream"

        except Exception as e:
            logger.warning(f"MIME type detection failed: {e}")
            return "application/octet-stream"

    def _validate_mime_type(
        self, detected_mime_type: str, provided_mime_type: Optional[str]
    ) -> None:
        """Validate MIME type against allowed types"""
        if detected_mime_type not in self.allowed_mime_types:
            raise ValueError(
                f"MIME type '{detected_mime_type}' not allowed. "
                f"Allowed types: {sorted(self.allowed_mime_types)}"
            )

        # Check for MIME type spoofing
        if provided_mime_type and provided_mime_type != detected_mime_type:
            logger.warning(
                f"MIME type mismatch: provided '{provided_mime_type}', "
                f"detected '{detected_mime_type}'"
            )

    def _generate_file_hash(self, file_content: bytes) -> str:
        """Generate SHA-256 hash of file content"""
        return hashlib.sha256(file_content).hexdigest()

    async def _check_for_duplicates(
        self, tenant_id: str, file_hash: str, filename: str
    ) -> Dict[str, Any]:
        """Check for duplicate files"""
        try:
            # This would check the database for existing files with the same hash
            # For now, return a placeholder
            return {
                "is_duplicate": False,
                "existing_document_id": None,
                "hash_matches": 0,
            }
        except Exception as e:
            logger.warning(f"Duplicate check failed: {e}")
            return {"is_duplicate": False, "error": str(e)}

    async def _scan_for_viruses(
        self, file_content: bytes, filename: str
    ) -> Dict[str, Any]:
        """Scan file for viruses and malware"""
        if not self.virus_scanner:
            return {"scan_performed": False, "clean": True, "scanner": None}

        try:
            # Write to temporary file for scanning
            with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                temp_file.write(file_content)
                temp_file.flush()

                # Perform scan
                scan_result = await self.virus_scanner.scan_file(temp_file.name)

                # Clean up
                os.unlink(temp_file.name)

                if not scan_result.get("clean", True):
                    raise SecurityError(
                        f"Virus detected: {scan_result.get('threat', 'Unknown threat')}"
                    )

                return {
                    "scan_performed": True,
                    "clean": scan_result.get("clean", True),
                    "scanner": scan_result.get("scanner", "unknown"),
                    "scan_time": scan_result.get("scan_time"),
                }

        except Exception as e:
            logger.error(f"Virus scan failed: {e}")
            # Fail secure - reject if scan fails
            raise SecurityError(f"Security scan failed: {e}")

    async def _validate_file_structure(
        self, file_content: bytes, file_extension: str, mime_type: str
    ) -> Dict[str, Any]:
        """Validate internal file structure"""
        validation_result = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "structure_type": "unknown",
        }

        try:
            if mime_type == "application/pdf":
                validation_result = await self._validate_pdf_structure(file_content)
            elif "image/" in mime_type:
                validation_result = await self._validate_image_structure(
                    file_content, mime_type
                )
            elif "word" in mime_type:
                validation_result = await self._validate_word_structure(file_content)

            if not validation_result.get("valid", True):
                raise ValueError(
                    f"File structure validation failed: {validation_result.get('errors')}"
                )

        except Exception as e:
            logger.warning(f"Structure validation failed: {e}")
            validation_result["errors"].append(str(e))

        return validation_result

    async def _validate_pdf_structure(self, file_content: bytes) -> Dict[str, Any]:
        """Validate PDF file structure"""
        try:
            # Basic PDF header validation
            if not file_content.startswith(b"%PDF"):
                return {"valid": False, "errors": ["Invalid PDF header"]}

            # Check for EOF marker
            if b"%%EOF" not in file_content[-1024:]:
                return {
                    "valid": True,
                    "warnings": ["PDF EOF marker not found in expected location"],
                }

            return {
                "valid": True,
                "structure_type": "pdf",
                "errors": [],
                "warnings": [],
            }

        except Exception as e:
            return {"valid": False, "errors": [f"PDF validation error: {e}"]}

    async def _validate_image_structure(
        self, file_content: bytes, mime_type: str
    ) -> Dict[str, Any]:
        """Validate image file structure"""
        try:
            if mime_type == "image/png":
                # PNG signature validation
                if not file_content.startswith(b"\x89PNG\r\n\x1a\n"):
                    return {"valid": False, "errors": ["Invalid PNG signature"]}
            elif mime_type == "image/jpeg":
                # JPEG signature validation
                if not file_content.startswith(b"\xff\xd8\xff"):
                    return {"valid": False, "errors": ["Invalid JPEG signature"]}

            return {
                "valid": True,
                "structure_type": "image",
                "errors": [],
                "warnings": [],
            }

        except Exception as e:
            return {"valid": False, "errors": [f"Image validation error: {e}"]}

    async def _validate_word_structure(self, file_content: bytes) -> Dict[str, Any]:
        """Validate Word document structure"""
        try:
            # DOCX files are ZIP archives
            if file_content.startswith(b"PK\x03\x04"):
                return {
                    "valid": True,
                    "structure_type": "docx",
                    "errors": [],
                    "warnings": [],
                }

            # DOC files have different signature
            if file_content.startswith(
                (b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1", b"\x0d\x44\x4f\x43")
            ):
                return {
                    "valid": True,
                    "structure_type": "doc",
                    "errors": [],
                    "warnings": [],
                }

            return {"valid": False, "errors": ["Unrecognized Word document format"]}

        except Exception as e:
            return {"valid": False, "errors": [f"Word document validation error: {e}"]}

    async def _extract_file_metadata(
        self, file_content: bytes, filename: str, mime_type: str
    ) -> Dict[str, Any]:
        """Extract file metadata"""
        metadata = {
            "filename": filename,
            "mime_type": mime_type,
            "size_bytes": len(file_content),
            "extraction_timestamp": datetime.utcnow().isoformat(),
        }

        try:
            # Extract basic metadata based on file type
            if mime_type == "application/pdf":
                metadata.update(await self._extract_pdf_metadata(file_content))
            elif "image/" in mime_type:
                metadata.update(await self._extract_image_metadata(file_content))

        except Exception as e:
            logger.warning(f"Metadata extraction failed: {e}")
            metadata["extraction_error"] = str(e)

        return metadata

    async def _extract_pdf_metadata(self, file_content: bytes) -> Dict[str, Any]:
        """Extract PDF metadata"""
        # This is a placeholder - would use PyPDF2 or similar
        return {
            "type": "pdf",
            "estimated_pages": file_content.count(b"/Page "),
            "has_text_layer": b"/Font" in file_content,
            "potentially_encrypted": b"/Encrypt" in file_content,
        }

    async def _extract_image_metadata(self, file_content: bytes) -> Dict[str, Any]:
        """Extract image metadata"""
        # This is a placeholder - would use Pillow or similar
        return {
            "type": "image",
            "estimated_resolution": "unknown",
            "color_depth": "unknown",
        }


class SecurityError(Exception):
    """Raised when security validation fails"""

    pass
