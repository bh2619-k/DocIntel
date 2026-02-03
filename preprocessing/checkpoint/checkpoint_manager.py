"""
Real-Time Document Intelligence & Compliance Engine
Checkpoint Management Module
Version: 1.0
Date: 2026-02-03

This module provides comprehensive checkpoint system for tracking preprocessing
progress and enabling resume functionality after interruptions with state persistence.
"""

import logging
import json
import hashlib
from typing import Dict, List, Any, Optional
from datetime import datetime, timezone, timedelta
from enum import Enum
from pathlib import Path

logger = logging.getLogger(__name__)


class ProcessingStage(Enum):
    """Document processing stages"""

    UPLOADED = "UPLOADED"
    DETECTED = "DETECTED"
    OCR_STARTED = "OCR_STARTED"
    OCR_COMPLETED = "OCR_COMPLETED"
    LAYOUT_ANALYZED = "LAYOUT_ANALYZED"
    TEXT_CORRECTED = "TEXT_CORRECTED"
    LANGUAGE_PROCESSED = "LANGUAGE_PROCESSED"
    PREPROCESSING_COMPLETED = "PREPROCESSING_COMPLETED"
    ERROR = "ERROR"
    CANCELLED = "CANCELLED"


class CheckpointType(Enum):
    """Types of checkpoints"""

    AUTOMATIC = "AUTOMATIC"  # System-generated checkpoints
    MANUAL = "MANUAL"  # User-triggered checkpoints
    ERROR_RECOVERY = "ERROR_RECOVERY"  # Error recovery checkpoints
    MILESTONE = "MILESTONE"  # Important processing milestones


class CheckpointStatus(Enum):
    """Checkpoint status"""

    ACTIVE = "ACTIVE"  # Currently processing
    COMPLETED = "COMPLETED"  # Successfully completed
    FAILED = "FAILED"  # Failed with error
    CANCELLED = "CANCELLED"  # User cancelled
    RESUMABLE = "RESUMABLE"  # Can be resumed


class DocumentCheckpoint:
    """
    Individual document processing checkpoint
    """

    def __init__(
        self,
        document_id: str,
        stage: ProcessingStage,
        checkpoint_type: CheckpointType = CheckpointType.AUTOMATIC,
    ):
        """
        Initialize document checkpoint

        Args:
            document_id: Unique document identifier
            stage: Current processing stage
            checkpoint_type: Type of checkpoint
        """
        self.checkpoint_id = self._generate_checkpoint_id(document_id, stage)
        self.document_id = document_id
        self.stage = stage
        self.checkpoint_type = checkpoint_type
        self.status = CheckpointStatus.ACTIVE
        self.created_at = datetime.now(timezone.utc)
        self.updated_at = self.created_at
        self.completion_percentage = 0.0

        # Processing state data
        self.processing_state = {}
        self.intermediate_results = {}
        self.error_info = None
        self.resource_usage = {}

        # Dependencies and relationships
        self.previous_checkpoint_id = None
        self.next_checkpoint_id = None
        self.dependent_checkpoints = []

        logger.debug(
            f"Created checkpoint {self.checkpoint_id} for document {document_id} at stage {stage.value}"
        )

    def _generate_checkpoint_id(self, document_id: str, stage: ProcessingStage) -> str:
        """Generate unique checkpoint ID"""
        timestamp = datetime.now(timezone.utc).isoformat()
        data = f"{document_id}_{stage.value}_{timestamp}"
        return hashlib.sha256(data.encode()).hexdigest()[:16]

    def update_progress(self, percentage: float, state_data: Dict[str, Any] = None):
        """Update checkpoint progress"""
        self.completion_percentage = max(0, min(100, percentage))
        self.updated_at = datetime.now(timezone.utc)

        if state_data:
            self.processing_state.update(state_data)

        logger.debug(
            f"Checkpoint {self.checkpoint_id} progress: {self.completion_percentage}%"
        )

    def set_intermediate_result(self, key: str, result: Any):
        """Store intermediate processing result"""
        self.intermediate_results[key] = {
            "data": result,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "size_bytes": len(json.dumps(result, default=str))
            if isinstance(result, (dict, list))
            else None,
        }

        logger.debug(
            f"Stored intermediate result '{key}' in checkpoint {self.checkpoint_id}"
        )

    def complete(self, final_results: Dict[str, Any] = None):
        """Mark checkpoint as completed"""
        self.status = CheckpointStatus.COMPLETED
        self.completion_percentage = 100.0
        self.updated_at = datetime.now(timezone.utc)

        if final_results:
            self.intermediate_results.update(final_results)

        logger.info(
            f"Checkpoint {self.checkpoint_id} completed for stage {self.stage.value}"
        )

    def fail(self, error: Exception, error_context: Dict[str, Any] = None):
        """Mark checkpoint as failed"""
        self.status = CheckpointStatus.FAILED
        self.updated_at = datetime.now(timezone.utc)

        self.error_info = {
            "error_type": type(error).__name__,
            "error_message": str(error),
            "error_context": error_context or {},
            "timestamp": self.updated_at.isoformat(),
            "is_recoverable": self._is_recoverable_error(error),
        }

        logger.error(f"Checkpoint {self.checkpoint_id} failed: {error}")

    def _is_recoverable_error(self, error: Exception) -> bool:
        """Determine if error is recoverable"""
        recoverable_errors = [
            "NetworkError",
            "TimeoutError",
            "TemporaryFailure",
            "RateLimitError",
            "ServiceUnavailable",
        ]
        return type(error).__name__ in recoverable_errors

    def to_dict(self) -> Dict[str, Any]:
        """Convert checkpoint to dictionary for serialization"""
        return {
            "checkpoint_id": self.checkpoint_id,
            "document_id": self.document_id,
            "stage": self.stage.value,
            "checkpoint_type": self.checkpoint_type.value,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "completion_percentage": self.completion_percentage,
            "processing_state": self.processing_state,
            "intermediate_results": self._serialize_results(),
            "error_info": self.error_info,
            "resource_usage": self.resource_usage,
            "previous_checkpoint_id": self.previous_checkpoint_id,
            "next_checkpoint_id": self.next_checkpoint_id,
            "dependent_checkpoints": self.dependent_checkpoints,
        }

    def _serialize_results(self) -> Dict[str, Any]:
        """Serialize intermediate results for storage"""
        try:
            # Serialize complex objects to JSON-safe format
            serialized = {}
            for key, value in self.intermediate_results.items():
                if isinstance(value, dict) and "data" in value:
                    serialized[key] = {
                        "timestamp": value["timestamp"],
                        "size_bytes": value["size_bytes"],
                        "data_type": type(value["data"]).__name__,
                        "data_summary": str(value["data"])[:200] + "..."
                        if len(str(value["data"])) > 200
                        else str(value["data"]),
                    }
                else:
                    serialized[key] = value
            return serialized
        except Exception as e:
            logger.warning(f"Failed to serialize results: {e}")
            return {}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DocumentCheckpoint":
        """Create checkpoint from dictionary"""
        checkpoint = cls.__new__(cls)
        checkpoint.checkpoint_id = data["checkpoint_id"]
        checkpoint.document_id = data["document_id"]
        checkpoint.stage = ProcessingStage(data["stage"])
        checkpoint.checkpoint_type = CheckpointType(data["checkpoint_type"])
        checkpoint.status = CheckpointStatus(data["status"])
        checkpoint.created_at = datetime.fromisoformat(data["created_at"])
        checkpoint.updated_at = datetime.fromisoformat(data["updated_at"])
        checkpoint.completion_percentage = data["completion_percentage"]
        checkpoint.processing_state = data.get("processing_state", {})
        checkpoint.intermediate_results = data.get("intermediate_results", {})
        checkpoint.error_info = data.get("error_info")
        checkpoint.resource_usage = data.get("resource_usage", {})
        checkpoint.previous_checkpoint_id = data.get("previous_checkpoint_id")
        checkpoint.next_checkpoint_id = data.get("next_checkpoint_id")
        checkpoint.dependent_checkpoints = data.get("dependent_checkpoints", [])

        return checkpoint


class CheckpointManager:
    """
    Comprehensive checkpoint management system
    """

    def __init__(self, storage_path: str = "./checkpoints"):
        """
        Initialize checkpoint manager

        Args:
            storage_path: Directory path for storing checkpoint data
        """
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)

        # In-memory checkpoint registry
        self.active_checkpoints: Dict[str, DocumentCheckpoint] = {}
        self.checkpoint_history: Dict[str, List[DocumentCheckpoint]] = {}

        # Configuration
        self.config = {
            "max_checkpoints_per_document": 10,
            "checkpoint_retention_hours": 72,
            "auto_cleanup_enabled": True,
            "compression_enabled": True,
            "backup_enabled": True,
        }

        # Processing statistics
        self.stats = {
            "total_checkpoints": 0,
            "completed_checkpoints": 0,
            "failed_checkpoints": 0,
            "resumed_documents": 0,
        }

        # Initialize storage and load existing checkpoints
        self._initialize_storage()

        logger.info(f"Checkpoint manager initialized with storage path: {storage_path}")

    def _initialize_storage(self):
        """Initialize checkpoint storage and load existing checkpoints"""
        try:
            # Create storage structure
            (self.storage_path / "active").mkdir(exist_ok=True)
            (self.storage_path / "completed").mkdir(exist_ok=True)
            (self.storage_path / "failed").mkdir(exist_ok=True)
            (self.storage_path / "backups").mkdir(exist_ok=True)

            # Load existing active checkpoints
            self._load_active_checkpoints()

            # Load statistics
            self._load_statistics()

        except Exception as e:
            logger.error(f"Failed to initialize checkpoint storage: {e}")

    def _load_active_checkpoints(self):
        """Load active checkpoints from storage"""
        try:
            active_dir = self.storage_path / "active"

            for checkpoint_file in active_dir.glob("*.json"):
                try:
                    with open(checkpoint_file, "r", encoding="utf-8") as f:
                        checkpoint_data = json.load(f)

                    checkpoint = DocumentCheckpoint.from_dict(checkpoint_data)
                    self.active_checkpoints[checkpoint.checkpoint_id] = checkpoint

                    # Add to history
                    doc_id = checkpoint.document_id
                    if doc_id not in self.checkpoint_history:
                        self.checkpoint_history[doc_id] = []
                    self.checkpoint_history[doc_id].append(checkpoint)

                except Exception as e:
                    logger.warning(
                        f"Failed to load checkpoint from {checkpoint_file}: {e}"
                    )

            logger.info(f"Loaded {len(self.active_checkpoints)} active checkpoints")

        except Exception as e:
            logger.error(f"Failed to load active checkpoints: {e}")

    def _load_statistics(self):
        """Load checkpoint statistics from storage"""
        try:
            stats_file = self.storage_path / "stats.json"
            if stats_file.exists():
                with open(stats_file, "r", encoding="utf-8") as f:
                    saved_stats = json.load(f)
                self.stats.update(saved_stats)

        except Exception as e:
            logger.warning(f"Failed to load statistics: {e}")

    async def create_checkpoint(
        self,
        document_id: str,
        stage: ProcessingStage,
        initial_state: Dict[str, Any] = None,
        checkpoint_type: CheckpointType = CheckpointType.AUTOMATIC,
    ) -> DocumentCheckpoint:
        """
        Create a new checkpoint for document processing

        Args:
            document_id: Unique document identifier
            stage: Processing stage
            initial_state: Initial processing state data
            checkpoint_type: Type of checkpoint

        Returns:
            Created checkpoint
        """
        try:
            # Check if we need to cleanup old checkpoints
            await self._cleanup_old_checkpoints(document_id)

            # Create new checkpoint
            checkpoint = DocumentCheckpoint(document_id, stage, checkpoint_type)

            if initial_state:
                checkpoint.processing_state.update(initial_state)

            # Link to previous checkpoint if exists
            previous_checkpoint = self._get_latest_checkpoint(document_id)
            if previous_checkpoint:
                checkpoint.previous_checkpoint_id = previous_checkpoint.checkpoint_id
                previous_checkpoint.next_checkpoint_id = checkpoint.checkpoint_id

            # Register checkpoint
            self.active_checkpoints[checkpoint.checkpoint_id] = checkpoint

            if document_id not in self.checkpoint_history:
                self.checkpoint_history[document_id] = []
            self.checkpoint_history[document_id].append(checkpoint)

            # Persist to storage
            await self._save_checkpoint(checkpoint)

            # Update statistics
            self.stats["total_checkpoints"] += 1
            await self._save_statistics()

            logger.info(
                f"Created checkpoint {checkpoint.checkpoint_id} for document {document_id} at stage {stage.value}"
            )

            return checkpoint

        except Exception as e:
            logger.error(f"Failed to create checkpoint: {e}")
            raise

    def _get_latest_checkpoint(self, document_id: str) -> Optional[DocumentCheckpoint]:
        """Get the latest checkpoint for a document"""
        try:
            if document_id in self.checkpoint_history:
                checkpoints = self.checkpoint_history[document_id]
                if checkpoints:
                    return max(checkpoints, key=lambda x: x.created_at)
            return None

        except Exception as e:
            logger.warning(f"Failed to get latest checkpoint: {e}")
            return None

    async def update_checkpoint(
        self,
        checkpoint_id: str,
        progress: float = None,
        state_update: Dict[str, Any] = None,
        intermediate_results: Dict[str, Any] = None,
    ) -> bool:
        """
        Update checkpoint progress and state

        Args:
            checkpoint_id: Checkpoint identifier
            progress: Progress percentage (0-100)
            state_update: State data to update
            intermediate_results: Intermediate processing results

        Returns:
            Success status
        """
        try:
            checkpoint = self.active_checkpoints.get(checkpoint_id)
            if not checkpoint:
                logger.warning(f"Checkpoint {checkpoint_id} not found")
                return False

            # Update progress
            if progress is not None:
                checkpoint.update_progress(progress, state_update)
            elif state_update:
                checkpoint.processing_state.update(state_update)
                checkpoint.updated_at = datetime.now(timezone.utc)

            # Store intermediate results
            if intermediate_results:
                for key, value in intermediate_results.items():
                    checkpoint.set_intermediate_result(key, value)

            # Persist changes
            await self._save_checkpoint(checkpoint)

            logger.debug(f"Updated checkpoint {checkpoint_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to update checkpoint {checkpoint_id}: {e}")
            return False

    async def complete_checkpoint(
        self, checkpoint_id: str, final_results: Dict[str, Any] = None
    ) -> bool:
        """
        Mark checkpoint as completed

        Args:
            checkpoint_id: Checkpoint identifier
            final_results: Final processing results

        Returns:
            Success status
        """
        try:
            checkpoint = self.active_checkpoints.get(checkpoint_id)
            if not checkpoint:
                logger.warning(f"Checkpoint {checkpoint_id} not found")
                return False

            # Complete checkpoint
            checkpoint.complete(final_results)

            # Move to completed storage
            await self._move_to_completed(checkpoint)

            # Remove from active checkpoints
            del self.active_checkpoints[checkpoint_id]

            # Update statistics
            self.stats["completed_checkpoints"] += 1
            await self._save_statistics()

            logger.info(f"Completed checkpoint {checkpoint_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to complete checkpoint {checkpoint_id}: {e}")
            return False

    async def fail_checkpoint(
        self, checkpoint_id: str, error: Exception, error_context: Dict[str, Any] = None
    ) -> bool:
        """
        Mark checkpoint as failed

        Args:
            checkpoint_id: Checkpoint identifier
            error: Error that caused failure
            error_context: Additional error context

        Returns:
            Success status
        """
        try:
            checkpoint = self.active_checkpoints.get(checkpoint_id)
            if not checkpoint:
                logger.warning(f"Checkpoint {checkpoint_id} not found")
                return False

            # Fail checkpoint
            checkpoint.fail(error, error_context)

            # Move to failed storage if not recoverable
            if not checkpoint.error_info.get("is_recoverable", False):
                await self._move_to_failed(checkpoint)
                del self.active_checkpoints[checkpoint_id]
            else:
                # Keep in active for potential recovery
                checkpoint.status = CheckpointStatus.RESUMABLE
                await self._save_checkpoint(checkpoint)

            # Update statistics
            self.stats["failed_checkpoints"] += 1
            await self._save_statistics()

            logger.error(f"Failed checkpoint {checkpoint_id}: {error}")
            return True

        except Exception as e:
            logger.error(f"Failed to handle checkpoint failure {checkpoint_id}: {e}")
            return False

    async def resume_document_processing(
        self, document_id: str
    ) -> Optional[DocumentCheckpoint]:
        """
        Resume document processing from last valid checkpoint

        Args:
            document_id: Document identifier

        Returns:
            Resumable checkpoint or None
        """
        try:
            # Find last resumable checkpoint
            resumable_checkpoint = None

            if document_id in self.checkpoint_history:
                for checkpoint in reversed(self.checkpoint_history[document_id]):
                    if checkpoint.status in [
                        CheckpointStatus.RESUMABLE,
                        CheckpointStatus.FAILED,
                    ]:
                        if checkpoint.error_info and checkpoint.error_info.get(
                            "is_recoverable", False
                        ):
                            resumable_checkpoint = checkpoint
                            break
                    elif checkpoint.status == CheckpointStatus.COMPLETED:
                        # Find the next stage to resume from
                        next_stage = self._get_next_processing_stage(checkpoint.stage)
                        if next_stage:
                            # Create new checkpoint for next stage
                            resumable_checkpoint = await self.create_checkpoint(
                                document_id,
                                next_stage,
                                checkpoint.processing_state,
                                CheckpointType.ERROR_RECOVERY,
                            )
                        break

            if resumable_checkpoint:
                # Reset status to active
                resumable_checkpoint.status = CheckpointStatus.ACTIVE
                resumable_checkpoint.updated_at = datetime.now(timezone.utc)
                await self._save_checkpoint(resumable_checkpoint)

                self.stats["resumed_documents"] += 1
                await self._save_statistics()

                logger.info(
                    f"Resumed document {document_id} from checkpoint {resumable_checkpoint.checkpoint_id}"
                )

            return resumable_checkpoint

        except Exception as e:
            logger.error(f"Failed to resume document processing {document_id}: {e}")
            return None

    def _get_next_processing_stage(
        self, current_stage: ProcessingStage
    ) -> Optional[ProcessingStage]:
        """Get next processing stage after current stage"""
        stage_order = [
            ProcessingStage.UPLOADED,
            ProcessingStage.DETECTED,
            ProcessingStage.OCR_STARTED,
            ProcessingStage.OCR_COMPLETED,
            ProcessingStage.LAYOUT_ANALYZED,
            ProcessingStage.TEXT_CORRECTED,
            ProcessingStage.LANGUAGE_PROCESSED,
            ProcessingStage.PREPROCESSING_COMPLETED,
        ]

        try:
            current_index = stage_order.index(current_stage)
            if current_index < len(stage_order) - 1:
                return stage_order[current_index + 1]
        except (ValueError, IndexError):
            pass

        return None

    async def get_document_progress(self, document_id: str) -> Dict[str, Any]:
        """
        Get comprehensive progress information for document

        Args:
            document_id: Document identifier

        Returns:
            Document progress information
        """
        try:
            progress_info = {
                "document_id": document_id,
                "current_stage": "unknown",
                "overall_progress": 0.0,
                "status": "unknown",
                "checkpoints": [],
                "estimated_completion": None,
                "error_info": None,
            }

            if document_id not in self.checkpoint_history:
                return progress_info

            checkpoints = self.checkpoint_history[document_id]
            latest_checkpoint = max(checkpoints, key=lambda x: x.created_at)

            progress_info.update(
                {
                    "current_stage": latest_checkpoint.stage.value,
                    "status": latest_checkpoint.status.value,
                    "overall_progress": self._calculate_overall_progress(checkpoints),
                    "checkpoints": [
                        {
                            "checkpoint_id": cp.checkpoint_id,
                            "stage": cp.stage.value,
                            "status": cp.status.value,
                            "progress": cp.completion_percentage,
                            "created_at": cp.created_at.isoformat(),
                            "updated_at": cp.updated_at.isoformat(),
                        }
                        for cp in checkpoints
                    ],
                    "error_info": latest_checkpoint.error_info,
                }
            )

            # Estimate completion time
            if latest_checkpoint.status == CheckpointStatus.ACTIVE:
                progress_info["estimated_completion"] = self._estimate_completion_time(
                    checkpoints
                )

            return progress_info

        except Exception as e:
            logger.error(f"Failed to get document progress: {e}")
            return {"document_id": document_id, "error": str(e)}

    def _calculate_overall_progress(
        self, checkpoints: List[DocumentCheckpoint]
    ) -> float:
        """Calculate overall document processing progress"""
        try:
            total_stages = len(ProcessingStage) - 2  # Exclude ERROR and CANCELLED
            completed_stages = 0
            current_stage_progress = 0

            for checkpoint in checkpoints:
                if checkpoint.status == CheckpointStatus.COMPLETED:
                    completed_stages += 1
                elif checkpoint.status == CheckpointStatus.ACTIVE:
                    current_stage_progress = checkpoint.completion_percentage / 100.0

            overall_progress = (
                completed_stages + current_stage_progress
            ) / total_stages
            return min(100.0, overall_progress * 100.0)

        except Exception:
            return 0.0

    def _estimate_completion_time(
        self, checkpoints: List[DocumentCheckpoint]
    ) -> Optional[str]:
        """Estimate completion time based on checkpoint history"""
        try:
            completed_checkpoints = [
                cp for cp in checkpoints if cp.status == CheckpointStatus.COMPLETED
            ]
            if len(completed_checkpoints) < 2:
                return None

            # Calculate average processing time per stage
            total_time = 0
            for i in range(1, len(completed_checkpoints)):
                prev_cp = completed_checkpoints[i - 1]
                curr_cp = completed_checkpoints[i]
                stage_time = (curr_cp.updated_at - prev_cp.created_at).total_seconds()
                total_time += stage_time

            avg_time_per_stage = total_time / (len(completed_checkpoints) - 1)

            # Estimate remaining stages
            total_stages = len(ProcessingStage) - 2
            remaining_stages = total_stages - len(completed_checkpoints)

            if remaining_stages > 0:
                estimated_seconds = remaining_stages * avg_time_per_stage
                completion_time = datetime.now(timezone.utc) + timedelta(
                    seconds=estimated_seconds
                )
                return completion_time.isoformat()

            return None

        except Exception:
            return None

    async def _save_checkpoint(self, checkpoint: DocumentCheckpoint):
        """Save checkpoint to persistent storage"""
        try:
            checkpoint_file = (
                self.storage_path / "active" / f"{checkpoint.checkpoint_id}.json"
            )
            checkpoint_data = checkpoint.to_dict()

            with open(checkpoint_file, "w", encoding="utf-8") as f:
                json.dump(checkpoint_data, f, indent=2, ensure_ascii=False)

        except Exception as e:
            logger.error(f"Failed to save checkpoint {checkpoint.checkpoint_id}: {e}")

    async def _move_to_completed(self, checkpoint: DocumentCheckpoint):
        """Move checkpoint to completed storage"""
        try:
            # Save to completed directory
            completed_file = (
                self.storage_path / "completed" / f"{checkpoint.checkpoint_id}.json"
            )
            checkpoint_data = checkpoint.to_dict()

            with open(completed_file, "w", encoding="utf-8") as f:
                json.dump(checkpoint_data, f, indent=2, ensure_ascii=False)

            # Remove from active directory
            active_file = (
                self.storage_path / "active" / f"{checkpoint.checkpoint_id}.json"
            )
            if active_file.exists():
                active_file.unlink()

        except Exception as e:
            logger.error(f"Failed to move checkpoint to completed: {e}")

    async def _move_to_failed(self, checkpoint: DocumentCheckpoint):
        """Move checkpoint to failed storage"""
        try:
            # Save to failed directory
            failed_file = (
                self.storage_path / "failed" / f"{checkpoint.checkpoint_id}.json"
            )
            checkpoint_data = checkpoint.to_dict()

            with open(failed_file, "w", encoding="utf-8") as f:
                json.dump(checkpoint_data, f, indent=2, ensure_ascii=False)

            # Remove from active directory
            active_file = (
                self.storage_path / "active" / f"{checkpoint.checkpoint_id}.json"
            )
            if active_file.exists():
                active_file.unlink()

        except Exception as e:
            logger.error(f"Failed to move checkpoint to failed: {e}")

    async def _cleanup_old_checkpoints(self, document_id: str):
        """Cleanup old checkpoints for document"""
        try:
            if document_id not in self.checkpoint_history:
                return

            checkpoints = self.checkpoint_history[document_id]
            max_checkpoints = self.config["max_checkpoints_per_document"]

            if len(checkpoints) >= max_checkpoints:
                # Remove oldest completed checkpoints
                completed_checkpoints = [
                    cp for cp in checkpoints if cp.status == CheckpointStatus.COMPLETED
                ]
                completed_checkpoints.sort(key=lambda x: x.created_at)

                to_remove = len(checkpoints) - max_checkpoints + 1

                for checkpoint in completed_checkpoints[:to_remove]:
                    # Remove from memory
                    if checkpoint.checkpoint_id in self.active_checkpoints:
                        del self.active_checkpoints[checkpoint.checkpoint_id]

                    checkpoints.remove(checkpoint)

                    # Remove from storage
                    completed_file = (
                        self.storage_path
                        / "completed"
                        / f"{checkpoint.checkpoint_id}.json"
                    )
                    if completed_file.exists():
                        completed_file.unlink()

        except Exception as e:
            logger.warning(f"Failed to cleanup old checkpoints: {e}")

    async def _save_statistics(self):
        """Save checkpoint statistics to storage"""
        try:
            stats_file = self.storage_path / "stats.json"
            with open(stats_file, "w", encoding="utf-8") as f:
                json.dump(self.stats, f, indent=2)

        except Exception as e:
            logger.warning(f"Failed to save statistics: {e}")

    def get_system_statistics(self) -> Dict[str, Any]:
        """Get comprehensive system checkpoint statistics"""
        try:
            active_count = len(self.active_checkpoints)
            total_documents = len(self.checkpoint_history)

            # Calculate success rate
            total_attempts = (
                self.stats["completed_checkpoints"] + self.stats["failed_checkpoints"]
            )
            success_rate = (
                (self.stats["completed_checkpoints"] / total_attempts * 100)
                if total_attempts > 0
                else 0
            )

            return {
                "total_checkpoints": self.stats["total_checkpoints"],
                "active_checkpoints": active_count,
                "completed_checkpoints": self.stats["completed_checkpoints"],
                "failed_checkpoints": self.stats["failed_checkpoints"],
                "total_documents": total_documents,
                "resumed_documents": self.stats["resumed_documents"],
                "success_rate_percent": round(success_rate, 2),
                "storage_path": str(self.storage_path),
                "configuration": self.config,
            }

        except Exception as e:
            logger.error(f"Failed to get system statistics: {e}")
            return {"error": str(e)}
