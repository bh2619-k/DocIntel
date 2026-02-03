"""
Real-Time Document Intelligence & Compliance Engine
Storage Manager for S3/MinIO Operations
Version: 1.0
Date: 2026-02-03

This module provides a unified interface for document storage operations
supporting both AWS S3 (cloud) and MinIO (on-premise) deployments.
"""

import boto3
import hashlib
import mimetypes
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, BinaryIO
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class StorageBackend(Enum):
    """Supported storage backends"""

    S3 = "s3"
    MINIO = "minio"


class StorageBucket(Enum):
    """Storage bucket types"""

    DOCUMENTS_RAW = "documents-raw"
    DOCUMENTS_PROCESSED = "documents-processed"
    DOCUMENTS_CHECKPOINTS = "documents-checkpoints"
    MODELS = "models"
    BACKUPS = "backups"
    TEMP = "temp"


@dataclass
class StorageConfig:
    """Storage configuration settings"""

    backend: StorageBackend
    endpoint_url: Optional[str] = None  # Required for MinIO
    region: str = "us-east-1"
    access_key_id: str = ""
    secret_access_key: str = ""
    bucket_prefix: str = "docintell"
    kms_key_id: Optional[str] = None
    encryption: bool = True
    versioning: bool = True


@dataclass
class FileMetadata:
    """File metadata structure"""

    file_path: str
    file_size: int
    content_type: str
    checksum: str
    tenant_id: str
    document_id: str
    uploaded_at: datetime
    tags: Dict[str, str] = None


class StorageManager:
    """
    Unified storage manager for document intelligence system
    Supports both S3 and MinIO with tenant isolation and security
    """

    def __init__(self, config: StorageConfig):
        """Initialize storage manager with configuration"""
        self.config = config
        self.s3_client = self._create_s3_client()
        self.bucket_names = self._get_bucket_names()

    def _create_s3_client(self):
        """Create S3 client for AWS or MinIO"""
        client_config = {
            "aws_access_key_id": self.config.access_key_id,
            "aws_secret_access_key": self.config.secret_access_key,
            "region_name": self.config.region,
        }

        # MinIO-specific configuration
        if self.config.backend == StorageBackend.MINIO:
            client_config["endpoint_url"] = self.config.endpoint_url
            client_config["config"] = boto3.session.Config(
                signature_version="s3v4", s3={"addressing_style": "path"}
            )

        return boto3.client("s3", **client_config)

    def _get_bucket_names(self) -> Dict[StorageBucket, str]:
        """Get actual bucket names with prefix"""
        return {
            bucket: f"{self.config.bucket_prefix}-{bucket.value}"
            for bucket in StorageBucket
        }

    def _generate_object_key(
        self,
        tenant_id: str,
        document_id: str,
        filename: str,
        document_type: str = "unknown",
    ) -> str:
        """
        Generate tenant-isolated object key
        Format: {tenant_id}/{document_type}/{year}/{month}/{day}/{document_id}/{filename}
        """
        now = datetime.utcnow()
        return (
            f"{tenant_id}/{document_type}/"
            f"{now.year:04d}/{now.month:02d}/{now.day:02d}/"
            f"{document_id}/{filename}"
        )

    def _calculate_checksum(self, file_obj: BinaryIO) -> str:
        """Calculate SHA-256 checksum of file"""
        hash_sha256 = hashlib.sha256()

        # Save current position
        current_pos = file_obj.tell()
        file_obj.seek(0)

        # Calculate hash
        for chunk in iter(lambda: file_obj.read(4096), b""):
            hash_sha256.update(chunk)

        # Restore position
        file_obj.seek(current_pos)

        return hash_sha256.hexdigest()

    def _get_content_type(self, filename: str) -> str:
        """Get MIME type for file"""
        content_type, _ = mimetypes.guess_type(filename)
        return content_type or "application/octet-stream"

    # =============================================================================
    # BUCKET MANAGEMENT
    # =============================================================================

    def create_buckets(self):
        """Create all required storage buckets"""
        for bucket_type, bucket_name in self.bucket_names.items():
            try:
                # Check if bucket exists
                self.s3_client.head_bucket(Bucket=bucket_name)
                logger.info(f"Bucket {bucket_name} already exists")

            except self.s3_client.exceptions.NoSuchBucket:
                # Create bucket
                if self.config.region != "us-east-1":
                    self.s3_client.create_bucket(
                        Bucket=bucket_name,
                        CreateBucketConfiguration={
                            "LocationConstraint": self.config.region
                        },
                    )
                else:
                    self.s3_client.create_bucket(Bucket=bucket_name)

                logger.info(f"Created bucket {bucket_name}")

                # Configure bucket settings
                self._configure_bucket(bucket_name, bucket_type)

            except Exception as e:
                logger.error(f"Error creating bucket {bucket_name}: {e}")
                raise

    def _configure_bucket(self, bucket_name: str, bucket_type: StorageBucket):
        """Configure bucket security and lifecycle policies"""
        try:
            # Enable versioning for non-temp buckets
            if bucket_type != StorageBucket.TEMP and self.config.versioning:
                self.s3_client.put_bucket_versioning(
                    Bucket=bucket_name, VersioningConfiguration={"Status": "Enabled"}
                )

            # Enable encryption
            if self.config.encryption:
                encryption_config = {
                    "Rules": [
                        {
                            "ApplyServerSideEncryptionByDefault": {
                                "SSEAlgorithm": "AES256"
                            }
                        }
                    ]
                }

                # Use KMS if configured
                if self.config.kms_key_id:
                    encryption_config["Rules"][0][
                        "ApplyServerSideEncryptionByDefault"
                    ] = {
                        "SSEAlgorithm": "aws:kms",
                        "KMSMasterKeyID": self.config.kms_key_id,
                    }

                self.s3_client.put_bucket_encryption(
                    Bucket=bucket_name,
                    ServerSideEncryptionConfiguration=encryption_config,
                )

            # Configure lifecycle rules
            self._set_lifecycle_policy(bucket_name, bucket_type)

            # Enable access logging (except for temp bucket)
            if bucket_type != StorageBucket.TEMP:
                self._enable_access_logging(bucket_name)

        except Exception as e:
            logger.error(f"Error configuring bucket {bucket_name}: {e}")

    def _set_lifecycle_policy(self, bucket_name: str, bucket_type: StorageBucket):
        """Set lifecycle policies for automatic cleanup and cost optimization"""
        lifecycle_rules = []

        if bucket_type == StorageBucket.TEMP:
            # Delete temp files after 1 day
            lifecycle_rules.append(
                {
                    "ID": "DeleteTempFiles",
                    "Status": "Enabled",
                    "Filter": {"Prefix": ""},
                    "Expiration": {"Days": 1},
                }
            )

        elif bucket_type == StorageBucket.DOCUMENTS_CHECKPOINTS:
            # Move checkpoints to IA after 30 days, delete after 90 days
            lifecycle_rules.append(
                {
                    "ID": "CheckpointLifecycle",
                    "Status": "Enabled",
                    "Filter": {"Prefix": ""},
                    "Transitions": [{"Days": 30, "StorageClass": "STANDARD_IA"}],
                    "Expiration": {"Days": 90},
                }
            )

        elif bucket_type == StorageBucket.DOCUMENTS_PROCESSED:
            # Move to IA after 90 days, Glacier after 365 days
            lifecycle_rules.append(
                {
                    "ID": "ProcessedDocumentLifecycle",
                    "Status": "Enabled",
                    "Filter": {"Prefix": ""},
                    "Transitions": [
                        {"Days": 90, "StorageClass": "STANDARD_IA"},
                        {"Days": 365, "StorageClass": "GLACIER"},
                    ],
                }
            )

        if lifecycle_rules:
            self.s3_client.put_bucket_lifecycle_configuration(
                Bucket=bucket_name, LifecycleConfiguration={"Rules": lifecycle_rules}
            )

    def _enable_access_logging(self, bucket_name: str):
        """Enable access logging for audit trails"""
        # Note: In production, you'd configure this to log to a separate audit bucket
        pass

    # =============================================================================
    # FILE UPLOAD OPERATIONS
    # =============================================================================

    def upload_raw_document(
        self,
        file_obj: BinaryIO,
        filename: str,
        tenant_id: str,
        document_id: str,
        document_type: str = "unknown",
        tags: Dict[str, str] = None,
    ) -> FileMetadata:
        """Upload raw document file"""
        return self._upload_file(
            file_obj=file_obj,
            filename=filename,
            bucket=StorageBucket.DOCUMENTS_RAW,
            tenant_id=tenant_id,
            document_id=document_id,
            document_type=document_type,
            tags=tags,
        )

    def upload_processed_document(
        self,
        file_obj: BinaryIO,
        filename: str,
        tenant_id: str,
        document_id: str,
        processing_stage: str,
        tags: Dict[str, str] = None,
    ) -> FileMetadata:
        """Upload processed document artifact"""
        if not tags:
            tags = {}
        tags["processing_stage"] = processing_stage

        return self._upload_file(
            file_obj=file_obj,
            filename=filename,
            bucket=StorageBucket.DOCUMENTS_PROCESSED,
            tenant_id=tenant_id,
            document_id=document_id,
            document_type="processed",
            tags=tags,
        )

    def upload_checkpoint(
        self,
        file_obj: BinaryIO,
        filename: str,
        tenant_id: str,
        document_id: str,
        stage: str,
        tags: Dict[str, str] = None,
    ) -> FileMetadata:
        """Upload processing checkpoint artifact"""
        if not tags:
            tags = {}
        tags["checkpoint_stage"] = stage

        return self._upload_file(
            file_obj=file_obj,
            filename=filename,
            bucket=StorageBucket.DOCUMENTS_CHECKPOINTS,
            tenant_id=tenant_id,
            document_id=document_id,
            document_type="checkpoint",
            tags=tags,
        )

    def _upload_file(
        self,
        file_obj: BinaryIO,
        filename: str,
        bucket: StorageBucket,
        tenant_id: str,
        document_id: str,
        document_type: str,
        tags: Dict[str, str] = None,
    ) -> FileMetadata:
        """Internal method for file uploads"""
        # Generate object key
        object_key = self._generate_object_key(
            tenant_id, document_id, filename, document_type
        )

        # Calculate file metadata
        file_obj.seek(0, 2)  # Seek to end
        file_size = file_obj.tell()
        file_obj.seek(0)  # Seek back to beginning

        checksum = self._calculate_checksum(file_obj)
        content_type = self._get_content_type(filename)

        # Prepare upload parameters
        bucket_name = self.bucket_names[bucket]
        upload_args = {
            "Bucket": bucket_name,
            "Key": object_key,
            "Body": file_obj,
            "ContentType": content_type,
            "Metadata": {
                "tenant-id": tenant_id,
                "document-id": document_id,
                "checksum": checksum,
                "original-filename": filename,
            },
        }

        # Add encryption
        if self.config.encryption:
            if self.config.kms_key_id:
                upload_args["ServerSideEncryption"] = "aws:kms"
                upload_args["SSEKMSKeyId"] = self.config.kms_key_id
            else:
                upload_args["ServerSideEncryption"] = "AES256"

        # Add tags
        if tags:
            upload_args["Tagging"] = "&".join([f"{k}={v}" for k, v in tags.items()])

        try:
            # Upload file
            self.s3_client.put_object(**upload_args)

            # Create metadata object
            metadata = FileMetadata(
                file_path=f"s3://{bucket_name}/{object_key}",
                file_size=file_size,
                content_type=content_type,
                checksum=checksum,
                tenant_id=tenant_id,
                document_id=document_id,
                uploaded_at=datetime.utcnow(),
                tags=tags,
            )

            logger.info(f"Uploaded file {filename} to {metadata.file_path}")
            return metadata

        except Exception as e:
            logger.error(f"Failed to upload file {filename}: {e}")
            raise

    # =============================================================================
    # FILE DOWNLOAD OPERATIONS
    # =============================================================================

    def download_file(
        self, file_path: str, local_path: str = None
    ) -> Union[str, bytes]:
        """Download file from storage"""
        # Parse S3 path
        if not file_path.startswith("s3://"):
            raise ValueError("File path must be S3 URI (s3://bucket/key)")

        path_parts = file_path[5:].split("/", 1)
        bucket_name = path_parts[0]
        object_key = path_parts[1]

        try:
            if local_path:
                # Download to local file
                self.s3_client.download_file(bucket_name, object_key, local_path)
                return local_path
            else:
                # Download to memory
                response = self.s3_client.get_object(Bucket=bucket_name, Key=object_key)
                return response["Body"].read()

        except Exception as e:
            logger.error(f"Failed to download file {file_path}: {e}")
            raise

    def get_file_metadata(self, file_path: str) -> Dict:
        """Get file metadata from storage"""
        # Parse S3 path
        path_parts = file_path[5:].split("/", 1)
        bucket_name = path_parts[0]
        object_key = path_parts[1]

        try:
            response = self.s3_client.head_object(Bucket=bucket_name, Key=object_key)
            return {
                "size": response["ContentLength"],
                "last_modified": response["LastModified"],
                "content_type": response["ContentType"],
                "metadata": response.get("Metadata", {}),
                "checksum": response.get("ETag", "").strip('"'),
            }
        except Exception as e:
            logger.error(f"Failed to get metadata for {file_path}: {e}")
            raise

    # =============================================================================
    # FILE MANAGEMENT OPERATIONS
    # =============================================================================

    def delete_file(self, file_path: str):
        """Delete file from storage"""
        # Parse S3 path
        path_parts = file_path[5:].split("/", 1)
        bucket_name = path_parts[0]
        object_key = path_parts[1]

        try:
            self.s3_client.delete_object(Bucket=bucket_name, Key=object_key)
            logger.info(f"Deleted file {file_path}")
        except Exception as e:
            logger.error(f"Failed to delete file {file_path}: {e}")
            raise

    def list_tenant_files(
        self, tenant_id: str, bucket: StorageBucket, limit: int = 1000
    ) -> List[Dict]:
        """List files for a specific tenant"""
        bucket_name = self.bucket_names[bucket]
        prefix = f"{tenant_id}/"

        try:
            response = self.s3_client.list_objects_v2(
                Bucket=bucket_name, Prefix=prefix, MaxKeys=limit
            )

            files = []
            for obj in response.get("Contents", []):
                files.append(
                    {
                        "key": obj["Key"],
                        "size": obj["Size"],
                        "last_modified": obj["LastModified"],
                        "storage_class": obj.get("StorageClass", "STANDARD"),
                    }
                )

            return files

        except Exception as e:
            logger.error(f"Failed to list files for tenant {tenant_id}: {e}")
            raise

    def generate_presigned_url(
        self, file_path: str, expiration: int = 3600, method: str = "GET"
    ) -> str:
        """Generate presigned URL for file access"""
        # Parse S3 path
        path_parts = file_path[5:].split("/", 1)
        bucket_name = path_parts[0]
        object_key = path_parts[1]

        try:
            if method.upper() == "GET":
                url = self.s3_client.generate_presigned_url(
                    "get_object",
                    Params={"Bucket": bucket_name, "Key": object_key},
                    ExpiresIn=expiration,
                )
            elif method.upper() == "PUT":
                url = self.s3_client.generate_presigned_url(
                    "put_object",
                    Params={"Bucket": bucket_name, "Key": object_key},
                    ExpiresIn=expiration,
                )
            else:
                raise ValueError(f"Unsupported method: {method}")

            return url

        except Exception as e:
            logger.error(f"Failed to generate presigned URL for {file_path}: {e}")
            raise

    # =============================================================================
    # CLEANUP AND MAINTENANCE
    # =============================================================================

    def cleanup_temp_files(self, older_than_hours: int = 24):
        """Cleanup temporary files older than specified hours"""
        bucket_name = self.bucket_names[StorageBucket.TEMP]
        cutoff_time = datetime.utcnow() - timedelta(hours=older_than_hours)

        try:
            response = self.s3_client.list_objects_v2(Bucket=bucket_name)

            deleted_count = 0
            for obj in response.get("Contents", []):
                if obj["LastModified"].replace(tzinfo=None) < cutoff_time:
                    self.s3_client.delete_object(Bucket=bucket_name, Key=obj["Key"])
                    deleted_count += 1

            logger.info(f"Cleaned up {deleted_count} temporary files")
            return deleted_count

        except Exception as e:
            logger.error(f"Failed to cleanup temp files: {e}")
            raise

    def get_storage_usage(self, tenant_id: str = None) -> Dict[str, int]:
        """Get storage usage statistics"""
        usage_stats = {}

        for bucket_type, bucket_name in self.bucket_names.items():
            try:
                prefix = f"{tenant_id}/" if tenant_id else ""

                response = self.s3_client.list_objects_v2(
                    Bucket=bucket_name, Prefix=prefix
                )

                total_size = sum(obj["Size"] for obj in response.get("Contents", []))
                file_count = len(response.get("Contents", []))

                usage_stats[bucket_type.value] = {
                    "size_bytes": total_size,
                    "size_human": self._human_readable_size(total_size),
                    "file_count": file_count,
                }

            except Exception as e:
                logger.error(f"Failed to get usage for bucket {bucket_name}: {e}")
                usage_stats[bucket_type.value] = {
                    "size_bytes": 0,
                    "size_human": "0 B",
                    "file_count": 0,
                }

        return usage_stats

    def _human_readable_size(self, size_bytes: int) -> str:
        """Convert bytes to human readable format"""
        if size_bytes == 0:
            return "0 B"

        size_names = ["B", "KB", "MB", "GB", "TB"]
        i = 0
        while size_bytes >= 1024 and i < len(size_names) - 1:
            size_bytes /= 1024.0
            i += 1

        return f"{size_bytes:.1f} {size_names[i]}"
