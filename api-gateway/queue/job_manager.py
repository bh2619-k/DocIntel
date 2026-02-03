"""
Real-Time Document Intelligence & Compliance Engine
Job Creation & Queue Management
Version: 1.0
Date: 2026-02-03

This module handles document processing job creation, priority routing,
and queue management with tenant isolation and resource optimization.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
from enum import Enum

logger = logging.getLogger(__name__)


class JobPriority(Enum):
    """Job priority levels"""

    LOW = 1
    NORMAL = 2
    HIGH = 3
    RUSH = 4
    CRITICAL = 5


class JobStatus(Enum):
    """Job status values"""

    CREATED = "CREATED"
    QUEUED = "QUEUED"
    ASSIGNED = "ASSIGNED"
    PROCESSING = "PROCESSING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"
    RETRY = "RETRY"


class ProcessingStage(Enum):
    """Document processing stages"""

    PREPROCESSING = "PREPROCESSING"
    OCR_EXTRACTION = "OCR_EXTRACTION"
    AI_EXTRACTION = "AI_EXTRACTION"
    CLASSIFICATION = "CLASSIFICATION"
    ANALYSIS = "ANALYSIS"
    GRAPH_UPDATE = "GRAPH_UPDATE"
    ALERT_GENERATION = "ALERT_GENERATION"
    POSTPROCESSING = "POSTPROCESSING"


class DocumentJobManager:
    """
    Manages document processing jobs with priority queuing and resource allocation
    """

    def __init__(self, redis_manager, database_manager, websocket_manager=None):
        """
        Initialize job manager

        Args:
            redis_manager: Redis connection manager
            database_manager: Database connection manager
            websocket_manager: WebSocket manager for real-time updates
        """
        self.redis = redis_manager
        self.database = database_manager
        self.websocket = websocket_manager

        # Queue configuration
        self.queue_configs = {
            JobPriority.CRITICAL: {
                "queue_name": "critical_queue",
                "max_workers": 4,
                "timeout_minutes": 5,
                "retry_attempts": 3,
            },
            JobPriority.RUSH: {
                "queue_name": "rush_queue",
                "max_workers": 6,
                "timeout_minutes": 15,
                "retry_attempts": 3,
            },
            JobPriority.HIGH: {
                "queue_name": "high_queue",
                "max_workers": 8,
                "timeout_minutes": 30,
                "retry_attempts": 2,
            },
            JobPriority.NORMAL: {
                "queue_name": "normal_queue",
                "max_workers": 12,
                "timeout_minutes": 60,
                "retry_attempts": 2,
            },
            JobPriority.LOW: {
                "queue_name": "low_queue",
                "max_workers": 4,
                "timeout_minutes": 120,
                "retry_attempts": 1,
            },
        }

    async def create_document_job(
        self,
        document_id: str,
        tenant_id: str,
        storage_path: str,
        priority: str = "normal",
        document_type: Optional[str] = None,
        processing_options: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Create a new document processing job

        Args:
            document_id: Unique document identifier
            tenant_id: Tenant identifier for isolation
            storage_path: Location of document in storage
            priority: Processing priority level
            document_type: Override document type classification
            processing_options: Processing configuration options
            metadata: Additional job metadata

        Returns:
            Job ID string
        """
        try:
            # Generate unique job ID
            import uuid

            job_id = f"job_{uuid.uuid4()}"

            # Parse priority
            job_priority = self._parse_priority(priority)
            queue_config = self.queue_configs[job_priority]

            # Default processing options
            default_options = {
                "enable_ocr": True,
                "enable_classification": True,
                "enable_entity_extraction": True,
                "enable_clause_extraction": True,
                "enable_compliance_check": True,
                "enable_risk_assessment": True,
                "language": "en",
                "ocr_confidence_threshold": 0.8,
                "entity_confidence_threshold": 0.7,
                "classification_confidence_threshold": 0.6,
            }

            if processing_options:
                default_options.update(processing_options)

            # Estimate processing time and resources
            processing_estimate = await self._estimate_processing_requirements(
                document_id, tenant_id, storage_path, default_options
            )

            # Create job record
            job_data = {
                "job_id": job_id,
                "document_id": document_id,
                "tenant_id": tenant_id,
                "storage_path": storage_path,
                "priority": job_priority.value,
                "priority_name": job_priority.name,
                "document_type": document_type,
                "processing_options": default_options,
                "metadata": metadata or {},
                "status": JobStatus.CREATED.value,
                "current_stage": None,
                "created_at": datetime.utcnow().isoformat(),
                "estimated_completion_time": processing_estimate[
                    "estimated_completion"
                ],
                "estimated_processing_minutes": processing_estimate[
                    "processing_minutes"
                ],
                "resource_requirements": processing_estimate["resources"],
                "queue_name": queue_config["queue_name"],
                "timeout_minutes": queue_config["timeout_minutes"],
                "max_retries": queue_config["retry_attempts"],
                "retry_count": 0,
                "assigned_worker": None,
                "processing_started_at": None,
                "processing_completed_at": None,
                "error_message": None,
                "checkpoints": {},
            }

            # Store job in database
            await self.database.create_processing_job(job_data)

            # Add to appropriate priority queue
            await self._enqueue_job(job_data)

            # Update document status
            await self.database.update_document_status(
                document_id=document_id,
                tenant_id=tenant_id,
                status="QUEUED",
                metadata={"job_id": job_id, "queue_name": queue_config["queue_name"]},
            )

            # Send real-time notification
            if self.websocket:
                await self.websocket.notify_document_status_change(
                    tenant_id=tenant_id,
                    document_id=document_id,
                    status="QUEUED",
                    stage="PREPROCESSING",
                    queue_position=await self._get_queue_position(job_id, job_priority),
                )

            logger.info(f"Created processing job {job_id} for document {document_id}")
            return job_id

        except Exception as e:
            logger.error(f"Failed to create job for document {document_id}: {e}")
            raise

    async def get_job_status(
        self, job_id: str, tenant_id: str
    ) -> Optional[Dict[str, Any]]:
        """
        Get current job status and progress

        Args:
            job_id: Job identifier
            tenant_id: Tenant identifier for security

        Returns:
            Job status dictionary or None if not found
        """
        try:
            job_data = await self.database.get_processing_job(job_id, tenant_id)

            if not job_data:
                return None

            # Enrich with real-time queue information
            if job_data["status"] == JobStatus.QUEUED.value:
                priority = JobPriority(job_data["priority"])
                queue_position = await self._get_queue_position(job_id, priority)
                estimated_start_time = await self._estimate_start_time(job_id, priority)

                job_data.update(
                    {
                        "queue_position": queue_position,
                        "estimated_start_time": estimated_start_time,
                    }
                )

            return job_data

        except Exception as e:
            logger.error(f"Failed to get job status {job_id}: {e}")
            return None

    async def update_job_progress(
        self,
        job_id: str,
        tenant_id: str,
        stage: ProcessingStage,
        progress_percentage: int,
        status: JobStatus = JobStatus.PROCESSING,
        checkpoint_data: Optional[Dict[str, Any]] = None,
        error_message: Optional[str] = None,
    ) -> bool:
        """
        Update job progress and stage

        Args:
            job_id: Job identifier
            tenant_id: Tenant identifier
            stage: Current processing stage
            progress_percentage: Progress percentage (0-100)
            status: Current job status
            checkpoint_data: Stage-specific checkpoint data
            error_message: Error message if applicable

        Returns:
            Success boolean
        """
        try:
            # Update job in database
            update_data = {
                "status": status.value,
                "current_stage": stage.value,
                "progress_percentage": progress_percentage,
                "last_updated_at": datetime.utcnow().isoformat(),
            }

            if error_message:
                update_data["error_message"] = error_message

            if status == JobStatus.PROCESSING and not await self._get_job_field(
                job_id, "processing_started_at"
            ):
                update_data["processing_started_at"] = datetime.utcnow().isoformat()

            if status == JobStatus.COMPLETED:
                update_data["processing_completed_at"] = datetime.utcnow().isoformat()

            await self.database.update_processing_job(job_id, tenant_id, update_data)

            # Update checkpoint
            if checkpoint_data:
                await self.database.update_job_checkpoint(
                    job_id=job_id,
                    tenant_id=tenant_id,
                    stage=stage.value,
                    checkpoint_data=checkpoint_data,
                )

            # Get document ID for notifications
            document_id = await self._get_job_field(job_id, "document_id")

            # Update document status
            if document_id:
                await self.database.update_document_status(
                    document_id=document_id,
                    tenant_id=tenant_id,
                    status=status.value,
                    current_stage=stage.value,
                    progress_percentage=progress_percentage,
                )

                # Send real-time notification
                if self.websocket:
                    await self.websocket.notify_document_status_change(
                        tenant_id=tenant_id,
                        document_id=document_id,
                        status=status.value,
                        stage=stage.value,
                        progress=progress_percentage,
                        error=error_message,
                    )

            return True

        except Exception as e:
            logger.error(f"Failed to update job progress {job_id}: {e}")
            return False

    async def fail_job(
        self,
        job_id: str,
        tenant_id: str,
        error_message: str,
        stage: Optional[ProcessingStage] = None,
        retry: bool = True,
    ) -> bool:
        """
        Mark job as failed and optionally retry

        Args:
            job_id: Job identifier
            tenant_id: Tenant identifier
            error_message: Failure reason
            stage: Stage where failure occurred
            retry: Whether to attempt retry

        Returns:
            Success boolean
        """
        try:
            job_data = await self.database.get_processing_job(job_id, tenant_id)
            if not job_data:
                return False

            retry_count = job_data.get("retry_count", 0)
            max_retries = job_data.get("max_retries", 0)

            # Determine if retry is possible and appropriate
            should_retry = (
                retry
                and retry_count < max_retries
                and not self._is_permanent_failure(error_message)
            )

            if should_retry:
                # Schedule retry
                retry_delay = self._calculate_retry_delay(retry_count)
                retry_time = datetime.utcnow() + timedelta(seconds=retry_delay)

                update_data = {
                    "status": JobStatus.RETRY.value,
                    "retry_count": retry_count + 1,
                    "error_message": error_message,
                    "retry_scheduled_at": retry_time.isoformat(),
                    "last_updated_at": datetime.utcnow().isoformat(),
                }

                if stage:
                    update_data["failed_stage"] = stage.value

                await self.database.update_processing_job(
                    job_id, tenant_id, update_data
                )

                # Schedule retry job
                await self._schedule_retry(job_data, retry_time)

                logger.info(f"Job {job_id} scheduled for retry #{retry_count + 1}")

            else:
                # Permanent failure
                update_data = {
                    "status": JobStatus.FAILED.value,
                    "error_message": error_message,
                    "processing_completed_at": datetime.utcnow().isoformat(),
                    "last_updated_at": datetime.utcnow().isoformat(),
                }

                if stage:
                    update_data["failed_stage"] = stage.value

                await self.database.update_processing_job(
                    job_id, tenant_id, update_data
                )

                # Update document status
                document_id = job_data.get("document_id")
                if document_id:
                    await self.database.update_document_status(
                        document_id=document_id,
                        tenant_id=tenant_id,
                        status="FAILED",
                        error_message=error_message,
                    )

                    # Send notification
                    if self.websocket:
                        await self.websocket.notify_document_status_change(
                            tenant_id=tenant_id,
                            document_id=document_id,
                            status="FAILED",
                            stage=stage.value if stage else None,
                            error=error_message,
                        )

                logger.error(f"Job {job_id} permanently failed: {error_message}")

            return True

        except Exception as e:
            logger.error(f"Failed to handle job failure {job_id}: {e}")
            return False

    async def cancel_job(
        self, job_id: str, tenant_id: str, reason: str = "User cancelled"
    ) -> bool:
        """
        Cancel a pending or processing job

        Args:
            job_id: Job identifier
            tenant_id: Tenant identifier
            reason: Cancellation reason

        Returns:
            Success boolean
        """
        try:
            job_data = await self.database.get_processing_job(job_id, tenant_id)
            if not job_data:
                return False

            current_status = job_data.get("status")
            if current_status in [
                JobStatus.COMPLETED.value,
                JobStatus.FAILED.value,
                JobStatus.CANCELLED.value,
            ]:
                return False  # Cannot cancel completed jobs

            # Remove from queue if still queued
            if current_status == JobStatus.QUEUED.value:
                priority = JobPriority(job_data["priority"])
                await self._dequeue_job(job_id, priority)

            # Update job status
            update_data = {
                "status": JobStatus.CANCELLED.value,
                "error_message": reason,
                "processing_completed_at": datetime.utcnow().isoformat(),
                "last_updated_at": datetime.utcnow().isoformat(),
            }

            await self.database.update_processing_job(job_id, tenant_id, update_data)

            # Update document status
            document_id = job_data.get("document_id")
            if document_id:
                await self.database.update_document_status(
                    document_id=document_id,
                    tenant_id=tenant_id,
                    status="CANCELLED",
                    error_message=reason,
                )

                # Send notification
                if self.websocket:
                    await self.websocket.notify_document_status_change(
                        tenant_id=tenant_id,
                        document_id=document_id,
                        status="CANCELLED",
                        error=reason,
                    )

            logger.info(f"Job {job_id} cancelled: {reason}")
            return True

        except Exception as e:
            logger.error(f"Failed to cancel job {job_id}: {e}")
            return False

    async def get_queue_stats(self, tenant_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Get processing queue statistics

        Args:
            tenant_id: Optional tenant filter

        Returns:
            Queue statistics dictionary
        """
        try:
            stats = {}

            for priority, config in self.queue_configs.items():
                queue_name = config["queue_name"]
                queue_length = await self.redis.get_queue_length(queue_name, tenant_id)
                processing_count = await self.database.count_processing_jobs_by_status(
                    JobStatus.PROCESSING.value, priority.value, tenant_id
                )

                stats[priority.name.lower()] = {
                    "queue_length": queue_length,
                    "processing_count": processing_count,
                    "max_workers": config["max_workers"],
                    "available_workers": max(
                        0, config["max_workers"] - processing_count
                    ),
                }

            # Overall statistics
            total_queued = sum(stat["queue_length"] for stat in stats.values())
            total_processing = sum(stat["processing_count"] for stat in stats.values())

            stats["overall"] = {
                "total_queued": total_queued,
                "total_processing": total_processing,
                "total_capacity": sum(
                    config["max_workers"] for config in self.queue_configs.values()
                ),
                "utilization_percentage": round(
                    (
                        total_processing
                        / sum(
                            config["max_workers"]
                            for config in self.queue_configs.values()
                        )
                    )
                    * 100,
                    2,
                ),
            }

            return stats

        except Exception as e:
            logger.error(f"Failed to get queue stats: {e}")
            return {}

    # Helper methods

    def _parse_priority(self, priority_str: str) -> JobPriority:
        """Parse priority string to enum"""
        priority_mapping = {
            "low": JobPriority.LOW,
            "normal": JobPriority.NORMAL,
            "high": JobPriority.HIGH,
            "rush": JobPriority.RUSH,
            "critical": JobPriority.CRITICAL,
        }
        return priority_mapping.get(priority_str.lower(), JobPriority.NORMAL)

    async def _estimate_processing_requirements(
        self,
        document_id: str,
        tenant_id: str,
        storage_path: str,
        processing_options: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Estimate processing time and resource requirements"""
        # This would analyze the document and estimate requirements
        # For now, return conservative estimates

        base_time_minutes = 5

        # Adjust based on processing options
        if processing_options.get("enable_ocr", False):
            base_time_minutes += 10

        if processing_options.get("enable_entity_extraction", False):
            base_time_minutes += 5

        if processing_options.get("enable_clause_extraction", False):
            base_time_minutes += 8

        if processing_options.get("enable_compliance_check", False):
            base_time_minutes += 3

        estimated_completion = datetime.utcnow() + timedelta(minutes=base_time_minutes)

        return {
            "processing_minutes": base_time_minutes,
            "estimated_completion": estimated_completion.isoformat(),
            "resources": {
                "cpu_intensive": processing_options.get("enable_ocr", False),
                "memory_intensive": processing_options.get(
                    "enable_entity_extraction", False
                ),
                "gpu_beneficial": processing_options.get("enable_ai_extraction", False),
            },
        }

    async def _enqueue_job(self, job_data: Dict[str, Any]) -> None:
        """Add job to appropriate priority queue"""
        queue_name = job_data["queue_name"]
        priority = job_data["priority"]

        # Add to Redis queue with priority score
        await self.redis.add_to_priority_queue(
            queue_name=queue_name,
            job_id=job_data["job_id"],
            priority_score=priority,
            job_data=job_data,
        )

    async def _dequeue_job(self, job_id: str, priority: JobPriority) -> bool:
        """Remove job from queue"""
        queue_name = self.queue_configs[priority]["queue_name"]
        return await self.redis.remove_from_queue(queue_name, job_id)

    async def _get_queue_position(
        self, job_id: str, priority: JobPriority
    ) -> Optional[int]:
        """Get job position in queue"""
        queue_name = self.queue_configs[priority]["queue_name"]
        return await self.redis.get_queue_position(queue_name, job_id)

    async def _estimate_start_time(
        self, job_id: str, priority: JobPriority
    ) -> Optional[str]:
        """Estimate when job processing will start"""
        try:
            position = await self._get_queue_position(job_id, priority)
            if position is None:
                return None

            # Estimate based on average processing time and queue position
            avg_processing_minutes = 15  # Conservative estimate
            estimated_minutes = position * avg_processing_minutes

            start_time = datetime.utcnow() + timedelta(minutes=estimated_minutes)
            return start_time.isoformat()

        except Exception:
            return None

    async def _get_job_field(self, job_id: str, field_name: str) -> Any:
        """Get specific field from job data"""
        try:
            job_data = await self.database.get_processing_job_field(job_id, field_name)
            return job_data.get(field_name) if job_data else None
        except Exception:
            return None

    def _is_permanent_failure(self, error_message: str) -> bool:
        """Determine if error is permanent (no retry)"""
        permanent_errors = [
            "file corrupted",
            "unsupported format",
            "invalid file structure",
            "virus detected",
            "permission denied",
        ]

        error_lower = error_message.lower()
        return any(perm_error in error_lower for perm_error in permanent_errors)

    def _calculate_retry_delay(self, retry_count: int) -> int:
        """Calculate exponential backoff delay for retries"""
        # Exponential backoff: 30s, 60s, 120s, 240s, 480s
        base_delay = 30
        return min(base_delay * (2**retry_count), 480)

    async def _schedule_retry(
        self, job_data: Dict[str, Any], retry_time: datetime
    ) -> None:
        """Schedule job for retry"""
        # This would use a delayed queue or scheduler
        # For now, just log the intention
        job_id = job_data["job_id"]
        logger.info(f"Job {job_id} scheduled for retry at {retry_time}")

        # Could implement with Redis delayed queues or similar mechanism
