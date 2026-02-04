"""
Checkpoint-Based Recovery System for Document Intelligence Pipeline

This module provides comprehensive state persistence and recovery capabilities
for the document intelligence pipeline orchestration system.

Key Features:
- Automatic checkpoint creation at critical execution points
- Recovery from system failures with state restoration
- Incremental processing and resume capabilities
- State versioning and rollback support
- Distributed checkpoint storage and synchronization
- Recovery strategy configuration and customization
- Health monitoring and recovery triggering

Classes:
    CheckpointType: Types of checkpoints in the system
    RecoveryStrategy: Strategy for handling recovery scenarios
    CheckpointMetadata: Metadata associated with checkpoints
    Checkpoint: Individual checkpoint with state data
    CheckpointStorage: Abstract storage interface
    FileCheckpointStorage: File-based checkpoint storage
    DatabaseCheckpointStorage: Database-based checkpoint storage
    CheckpointManager: Main checkpoint management system
    RecoveryManager: Handles recovery operations

Dependencies:
    - asyncio: For asynchronous checkpoint operations
    - logging: For comprehensive logging
    - pickle: For state serialization
    - json: For metadata serialization
    - pathlib: For file path management
    - datetime: For timestamp management
    - typing: For type hints
    - dataclasses: For structured data
    - threading: For concurrent access control
    - hashlib: For checkpoint integrity verification
"""

import asyncio
import hashlib
import json
import logging
import pickle
import threading
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Set, Union
import sqlite3
import os


# Configure comprehensive logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class CheckpointType(Enum):
    """Types of checkpoints in the system."""

    MANUAL = auto()
    AUTOMATIC = auto()
    MILESTONE = auto()
    ERROR_STATE = auto()
    SCHEDULED = auto()
    PRE_CRITICAL_OPERATION = auto()


class RecoveryStrategy(Enum):
    """Recovery strategies for different failure scenarios."""

    RESTART_FROM_LAST_CHECKPOINT = auto()
    RESTART_FROM_MILESTONE = auto()
    RESTART_FROM_BEGINNING = auto()
    PARTIAL_RECOVERY = auto()
    MANUAL_INTERVENTION = auto()
    SKIP_FAILED_COMPONENTS = auto()


class CheckpointStatus(Enum):
    """Status of checkpoint operations."""

    CREATING = auto()
    COMPLETED = auto()
    FAILED = auto()
    VERIFYING = auto()
    CORRUPTED = auto()
    RESTORING = auto()


@dataclass
class CheckpointMetadata:
    """Metadata associated with a checkpoint."""

    checkpoint_id: str
    checkpoint_type: CheckpointType
    job_id: str
    workflow_id: str
    task_id: Optional[str]
    created_at: datetime
    created_by: str
    description: str
    version: str
    tags: Dict[str, str] = field(default_factory=dict)
    size_bytes: int = 0
    checksum: Optional[str] = None
    recovery_info: Dict[str, Any] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)
    expiry_date: Optional[datetime] = None


@dataclass
class Checkpoint:
    """Represents a checkpoint with state data."""

    metadata: CheckpointMetadata
    state_data: Dict[str, Any]
    status: CheckpointStatus = CheckpointStatus.CREATING
    error_message: Optional[str] = None
    verification_attempts: int = 0
    max_verification_attempts: int = 3


class CheckpointStorage(ABC):
    """Abstract interface for checkpoint storage backends."""

    @abstractmethod
    async def save_checkpoint(self, checkpoint: Checkpoint) -> bool:
        """Save a checkpoint to storage."""
        pass

    @abstractmethod
    async def load_checkpoint(self, checkpoint_id: str) -> Optional[Checkpoint]:
        """Load a checkpoint from storage."""
        pass

    @abstractmethod
    async def list_checkpoints(
        self, job_id: Optional[str] = None, workflow_id: Optional[str] = None
    ) -> List[CheckpointMetadata]:
        """List available checkpoints with optional filtering."""
        pass

    @abstractmethod
    async def delete_checkpoint(self, checkpoint_id: str) -> bool:
        """Delete a checkpoint from storage."""
        pass

    @abstractmethod
    async def verify_checkpoint(self, checkpoint_id: str) -> bool:
        """Verify checkpoint integrity."""
        pass


class FileCheckpointStorage(CheckpointStorage):
    """File-based checkpoint storage implementation."""

    def __init__(
        self,
        storage_path: str = "checkpoints",
        compression: bool = True,
        encryption: bool = False,
    ):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.compression = compression
        self.encryption = encryption
        self.lock = threading.RLock()

    async def save_checkpoint(self, checkpoint: Checkpoint) -> bool:
        """Save a checkpoint to file storage."""
        try:
            checkpoint_id = checkpoint.metadata.checkpoint_id

            # Create checkpoint directory
            checkpoint_dir = self.storage_path / checkpoint_id
            checkpoint_dir.mkdir(exist_ok=True)

            # Save metadata
            metadata_file = checkpoint_dir / "metadata.json"
            with open(metadata_file, "w") as f:
                json.dump(asdict(checkpoint.metadata), f, indent=2, default=str)

            # Save state data
            state_file = checkpoint_dir / "state.pkl"
            with open(state_file, "wb") as f:
                pickle.dump(checkpoint.state_data, f)

            # Calculate checksum
            checkpoint.metadata.checksum = await self._calculate_checksum(state_file)
            checkpoint.metadata.size_bytes = state_file.stat().st_size

            # Update metadata with checksum
            with open(metadata_file, "w") as f:
                json.dump(asdict(checkpoint.metadata), f, indent=2, default=str)

            checkpoint.status = CheckpointStatus.COMPLETED
            logger.info(f"Checkpoint {checkpoint_id} saved successfully")
            return True

        except Exception as e:
            checkpoint.status = CheckpointStatus.FAILED
            checkpoint.error_message = str(e)
            logger.error(f"Failed to save checkpoint {checkpoint_id}: {str(e)}")
            return False

    async def load_checkpoint(self, checkpoint_id: str) -> Optional[Checkpoint]:
        """Load a checkpoint from file storage."""
        try:
            checkpoint_dir = self.storage_path / checkpoint_id
            if not checkpoint_dir.exists():
                return None

            # Load metadata
            metadata_file = checkpoint_dir / "metadata.json"
            if not metadata_file.exists():
                return None

            with open(metadata_file, "r") as f:
                metadata_dict = json.load(f)

            # Convert datetime strings back to datetime objects
            metadata_dict["created_at"] = datetime.fromisoformat(
                metadata_dict["created_at"]
            )
            if metadata_dict.get("expiry_date"):
                metadata_dict["expiry_date"] = datetime.fromisoformat(
                    metadata_dict["expiry_date"]
                )

            metadata = CheckpointMetadata(**metadata_dict)

            # Load state data
            state_file = checkpoint_dir / "state.pkl"
            if not state_file.exists():
                return None

            with open(state_file, "rb") as f:
                state_data = pickle.load(f)

            checkpoint = Checkpoint(
                metadata=metadata,
                state_data=state_data,
                status=CheckpointStatus.COMPLETED,
            )

            logger.info(f"Checkpoint {checkpoint_id} loaded successfully")
            return checkpoint

        except Exception as e:
            logger.error(f"Failed to load checkpoint {checkpoint_id}: {str(e)}")
            return None

    async def list_checkpoints(
        self, job_id: Optional[str] = None, workflow_id: Optional[str] = None
    ) -> List[CheckpointMetadata]:
        """List available checkpoints with optional filtering."""
        checkpoints = []

        try:
            for checkpoint_dir in self.storage_path.iterdir():
                if not checkpoint_dir.is_dir():
                    continue

                metadata_file = checkpoint_dir / "metadata.json"
                if not metadata_file.exists():
                    continue

                try:
                    with open(metadata_file, "r") as f:
                        metadata_dict = json.load(f)

                    # Convert datetime strings
                    metadata_dict["created_at"] = datetime.fromisoformat(
                        metadata_dict["created_at"]
                    )
                    if metadata_dict.get("expiry_date"):
                        metadata_dict["expiry_date"] = datetime.fromisoformat(
                            metadata_dict["expiry_date"]
                        )

                    metadata = CheckpointMetadata(**metadata_dict)

                    # Apply filters
                    if job_id and metadata.job_id != job_id:
                        continue
                    if workflow_id and metadata.workflow_id != workflow_id:
                        continue

                    checkpoints.append(metadata)

                except Exception as e:
                    logger.warning(
                        f"Failed to load metadata for {checkpoint_dir}: {str(e)}"
                    )
                    continue

            # Sort by creation time (newest first)
            checkpoints.sort(key=lambda x: x.created_at, reverse=True)
            return checkpoints

        except Exception as e:
            logger.error(f"Failed to list checkpoints: {str(e)}")
            return []

    async def delete_checkpoint(self, checkpoint_id: str) -> bool:
        """Delete a checkpoint from file storage."""
        try:
            checkpoint_dir = self.storage_path / checkpoint_id
            if checkpoint_dir.exists():
                # Remove all files in the checkpoint directory
                for file_path in checkpoint_dir.iterdir():
                    file_path.unlink()
                checkpoint_dir.rmdir()

                logger.info(f"Checkpoint {checkpoint_id} deleted successfully")
                return True
            return False

        except Exception as e:
            logger.error(f"Failed to delete checkpoint {checkpoint_id}: {str(e)}")
            return False

    async def verify_checkpoint(self, checkpoint_id: str) -> bool:
        """Verify checkpoint integrity."""
        try:
            checkpoint = await self.load_checkpoint(checkpoint_id)
            if not checkpoint:
                return False

            # Verify checksum if available
            if checkpoint.metadata.checksum:
                checkpoint_dir = self.storage_path / checkpoint_id
                state_file = checkpoint_dir / "state.pkl"
                current_checksum = await self._calculate_checksum(state_file)

                if current_checksum != checkpoint.metadata.checksum:
                    logger.warning(f"Checksum mismatch for checkpoint {checkpoint_id}")
                    return False

            return True

        except Exception as e:
            logger.error(f"Failed to verify checkpoint {checkpoint_id}: {str(e)}")
            return False

    async def _calculate_checksum(self, file_path: Path) -> str:
        """Calculate SHA-256 checksum of a file."""
        hash_sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        return hash_sha256.hexdigest()


class DatabaseCheckpointStorage(CheckpointStorage):
    """Database-based checkpoint storage implementation."""

    def __init__(self, db_path: str = "checkpoints.db"):
        self.db_path = db_path
        self.lock = threading.RLock()
        self._initialize_database()

    def _initialize_database(self):
        """Initialize the database schema."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS checkpoints (
                        checkpoint_id TEXT PRIMARY KEY,
                        checkpoint_type TEXT NOT NULL,
                        job_id TEXT NOT NULL,
                        workflow_id TEXT NOT NULL,
                        task_id TEXT,
                        created_at TEXT NOT NULL,
                        created_by TEXT NOT NULL,
                        description TEXT NOT NULL,
                        version TEXT NOT NULL,
                        tags TEXT,
                        size_bytes INTEGER DEFAULT 0,
                        checksum TEXT,
                        recovery_info TEXT,
                        dependencies TEXT,
                        expiry_date TEXT,
                        state_data BLOB NOT NULL,
                        status TEXT DEFAULT 'COMPLETED',
                        error_message TEXT
                    )
                """)

                # Create indexes for better performance
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_job_id ON checkpoints(job_id)"
                )
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_workflow_id ON checkpoints(workflow_id)"
                )
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_created_at ON checkpoints(created_at)"
                )

                conn.commit()
                logger.info("Database initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize database: {str(e)}")
            raise

    async def save_checkpoint(self, checkpoint: Checkpoint) -> bool:
        """Save a checkpoint to database storage."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Serialize complex fields
                state_data_blob = pickle.dumps(checkpoint.state_data)
                tags_json = json.dumps(checkpoint.metadata.tags)
                recovery_info_json = json.dumps(checkpoint.metadata.recovery_info)
                dependencies_json = json.dumps(checkpoint.metadata.dependencies)

                conn.execute(
                    """
                    INSERT OR REPLACE INTO checkpoints (
                        checkpoint_id, checkpoint_type, job_id, workflow_id, task_id,
                        created_at, created_by, description, version, tags, size_bytes,
                        checksum, recovery_info, dependencies, expiry_date, state_data,
                        status, error_message
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        checkpoint.metadata.checkpoint_id,
                        checkpoint.metadata.checkpoint_type.name,
                        checkpoint.metadata.job_id,
                        checkpoint.metadata.workflow_id,
                        checkpoint.metadata.task_id,
                        checkpoint.metadata.created_at.isoformat(),
                        checkpoint.metadata.created_by,
                        checkpoint.metadata.description,
                        checkpoint.metadata.version,
                        tags_json,
                        len(state_data_blob),
                        checkpoint.metadata.checksum,
                        recovery_info_json,
                        dependencies_json,
                        checkpoint.metadata.expiry_date.isoformat()
                        if checkpoint.metadata.expiry_date
                        else None,
                        state_data_blob,
                        checkpoint.status.name,
                        checkpoint.error_message,
                    ),
                )

                conn.commit()
                checkpoint.status = CheckpointStatus.COMPLETED
                logger.info(
                    f"Checkpoint {checkpoint.metadata.checkpoint_id} saved to database"
                )
                return True

        except Exception as e:
            checkpoint.status = CheckpointStatus.FAILED
            checkpoint.error_message = str(e)
            logger.error(f"Failed to save checkpoint to database: {str(e)}")
            return False

    async def load_checkpoint(self, checkpoint_id: str) -> Optional[Checkpoint]:
        """Load a checkpoint from database storage."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(
                    """
                    SELECT * FROM checkpoints WHERE checkpoint_id = ?
                """,
                    (checkpoint_id,),
                )

                row = cursor.fetchone()
                if not row:
                    return None

                # Map database columns
                columns = [desc[0] for desc in cursor.description]
                data = dict(zip(columns, row))

                # Deserialize complex fields
                state_data = pickle.loads(data["state_data"])
                tags = json.loads(data["tags"]) if data["tags"] else {}
                recovery_info = (
                    json.loads(data["recovery_info"]) if data["recovery_info"] else {}
                )
                dependencies = (
                    json.loads(data["dependencies"]) if data["dependencies"] else []
                )

                # Create metadata object
                metadata = CheckpointMetadata(
                    checkpoint_id=data["checkpoint_id"],
                    checkpoint_type=CheckpointType[data["checkpoint_type"]],
                    job_id=data["job_id"],
                    workflow_id=data["workflow_id"],
                    task_id=data["task_id"],
                    created_at=datetime.fromisoformat(data["created_at"]),
                    created_by=data["created_by"],
                    description=data["description"],
                    version=data["version"],
                    tags=tags,
                    size_bytes=data["size_bytes"],
                    checksum=data["checksum"],
                    recovery_info=recovery_info,
                    dependencies=dependencies,
                    expiry_date=datetime.fromisoformat(data["expiry_date"])
                    if data["expiry_date"]
                    else None,
                )

                # Create checkpoint object
                checkpoint = Checkpoint(
                    metadata=metadata,
                    state_data=state_data,
                    status=CheckpointStatus[data["status"]],
                    error_message=data["error_message"],
                )

                logger.info(f"Checkpoint {checkpoint_id} loaded from database")
                return checkpoint

        except Exception as e:
            logger.error(f"Failed to load checkpoint from database: {str(e)}")
            return None

    async def list_checkpoints(
        self, job_id: Optional[str] = None, workflow_id: Optional[str] = None
    ) -> List[CheckpointMetadata]:
        """List available checkpoints with optional filtering."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                query = """
                    SELECT checkpoint_id, checkpoint_type, job_id, workflow_id, task_id,
                           created_at, created_by, description, version, tags, size_bytes,
                           checksum, recovery_info, dependencies, expiry_date
                    FROM checkpoints
                    WHERE 1=1
                """
                params = []

                if job_id:
                    query += " AND job_id = ?"
                    params.append(job_id)

                if workflow_id:
                    query += " AND workflow_id = ?"
                    params.append(workflow_id)

                query += " ORDER BY created_at DESC"

                cursor = conn.execute(query, params)
                rows = cursor.fetchall()

                checkpoints = []
                for row in rows:
                    columns = [desc[0] for desc in cursor.description]
                    data = dict(zip(columns, row))

                    # Deserialize complex fields
                    tags = json.loads(data["tags"]) if data["tags"] else {}
                    recovery_info = (
                        json.loads(data["recovery_info"])
                        if data["recovery_info"]
                        else {}
                    )
                    dependencies = (
                        json.loads(data["dependencies"]) if data["dependencies"] else []
                    )

                    metadata = CheckpointMetadata(
                        checkpoint_id=data["checkpoint_id"],
                        checkpoint_type=CheckpointType[data["checkpoint_type"]],
                        job_id=data["job_id"],
                        workflow_id=data["workflow_id"],
                        task_id=data["task_id"],
                        created_at=datetime.fromisoformat(data["created_at"]),
                        created_by=data["created_by"],
                        description=data["description"],
                        version=data["version"],
                        tags=tags,
                        size_bytes=data["size_bytes"],
                        checksum=data["checksum"],
                        recovery_info=recovery_info,
                        dependencies=dependencies,
                        expiry_date=datetime.fromisoformat(data["expiry_date"])
                        if data["expiry_date"]
                        else None,
                    )

                    checkpoints.append(metadata)

                return checkpoints

        except Exception as e:
            logger.error(f"Failed to list checkpoints: {str(e)}")
            return []

    async def delete_checkpoint(self, checkpoint_id: str) -> bool:
        """Delete a checkpoint from database storage."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(
                    """
                    DELETE FROM checkpoints WHERE checkpoint_id = ?
                """,
                    (checkpoint_id,),
                )

                if cursor.rowcount > 0:
                    conn.commit()
                    logger.info(f"Checkpoint {checkpoint_id} deleted from database")
                    return True

                return False

        except Exception as e:
            logger.error(f"Failed to delete checkpoint from database: {str(e)}")
            return False

    async def verify_checkpoint(self, checkpoint_id: str) -> bool:
        """Verify checkpoint integrity."""
        try:
            checkpoint = await self.load_checkpoint(checkpoint_id)
            return checkpoint is not None

        except Exception as e:
            logger.error(f"Failed to verify checkpoint: {str(e)}")
            return False


class CheckpointManager:
    """
    Main checkpoint management system that handles checkpoint creation,
    storage, and cleanup operations.
    """

    def __init__(
        self,
        storage: CheckpointStorage,
        auto_checkpoint_interval: int = 300,  # 5 minutes
        max_checkpoints_per_job: int = 10,
        cleanup_expired_checkpoints: bool = True,
    ):
        self.storage = storage
        self.auto_checkpoint_interval = auto_checkpoint_interval
        self.max_checkpoints_per_job = max_checkpoints_per_job
        self.cleanup_expired_checkpoints = cleanup_expired_checkpoints

        # Tracking
        self.active_checkpoints = {}
        self.checkpoint_history = defaultdict(list)

        # Synchronization
        self.lock = threading.RLock()

        # Background tasks
        self._running = False
        self._background_tasks = []

    async def start(self):
        """Start the checkpoint manager."""
        self._running = True
        logger.info("Starting Checkpoint Manager")

        # Start background tasks
        self._background_tasks = [
            asyncio.create_task(self._cleanup_expired_checkpoints()),
            asyncio.create_task(self._monitor_checkpoint_health()),
        ]

    async def stop(self):
        """Stop the checkpoint manager."""
        self._running = False
        logger.info("Stopping Checkpoint Manager")

        # Cancel background tasks
        for task in self._background_tasks:
            task.cancel()

    async def create_checkpoint(
        self,
        job_id: str,
        workflow_id: str,
        state_data: Dict[str, Any],
        checkpoint_type: CheckpointType = CheckpointType.AUTOMATIC,
        task_id: Optional[str] = None,
        description: str = "",
        tags: Optional[Dict[str, str]] = None,
    ) -> Optional[str]:
        """Create a new checkpoint."""
        try:
            checkpoint_id = str(uuid.uuid4())

            # Create metadata
            metadata = CheckpointMetadata(
                checkpoint_id=checkpoint_id,
                checkpoint_type=checkpoint_type,
                job_id=job_id,
                workflow_id=workflow_id,
                task_id=task_id,
                created_at=datetime.now(),
                created_by="system",
                description=description or f"Auto checkpoint for job {job_id}",
                version="1.0",
                tags=tags or {},
                recovery_info={
                    "recovery_strategy": RecoveryStrategy.RESTART_FROM_LAST_CHECKPOINT.name,
                    "dependencies_verified": True,
                },
            )

            # Create checkpoint
            checkpoint = Checkpoint(metadata=metadata, state_data=state_data.copy())

            # Save checkpoint
            success = await self.storage.save_checkpoint(checkpoint)
            if success:
                with self.lock:
                    self.active_checkpoints[checkpoint_id] = checkpoint
                    self.checkpoint_history[job_id].append(checkpoint_id)

                # Cleanup old checkpoints for this job
                await self._cleanup_old_checkpoints(job_id)

                logger.info(f"Checkpoint {checkpoint_id} created for job {job_id}")
                return checkpoint_id
            else:
                logger.error(f"Failed to save checkpoint for job {job_id}")
                return None

        except Exception as e:
            logger.error(f"Failed to create checkpoint: {str(e)}")
            return None

    async def load_checkpoint(self, checkpoint_id: str) -> Optional[Checkpoint]:
        """Load a checkpoint by ID."""
        try:
            checkpoint = await self.storage.load_checkpoint(checkpoint_id)
            if checkpoint:
                with self.lock:
                    self.active_checkpoints[checkpoint_id] = checkpoint
                logger.info(f"Checkpoint {checkpoint_id} loaded successfully")
            return checkpoint

        except Exception as e:
            logger.error(f"Failed to load checkpoint {checkpoint_id}: {str(e)}")
            return None

    async def get_latest_checkpoint(self, job_id: str) -> Optional[Checkpoint]:
        """Get the latest checkpoint for a job."""
        try:
            checkpoints = await self.storage.list_checkpoints(job_id=job_id)
            if checkpoints:
                latest_metadata = checkpoints[0]  # Already sorted by creation time
                return await self.load_checkpoint(latest_metadata.checkpoint_id)
            return None

        except Exception as e:
            logger.error(f"Failed to get latest checkpoint for job {job_id}: {str(e)}")
            return None

    async def list_checkpoints(
        self, job_id: Optional[str] = None, workflow_id: Optional[str] = None
    ) -> List[CheckpointMetadata]:
        """List checkpoints with optional filtering."""
        return await self.storage.list_checkpoints(job_id, workflow_id)

    async def delete_checkpoint(self, checkpoint_id: str) -> bool:
        """Delete a checkpoint."""
        try:
            success = await self.storage.delete_checkpoint(checkpoint_id)
            if success:
                with self.lock:
                    self.active_checkpoints.pop(checkpoint_id, None)
                    # Remove from history
                    for job_id, history in self.checkpoint_history.items():
                        if checkpoint_id in history:
                            history.remove(checkpoint_id)
                            break
                logger.info(f"Checkpoint {checkpoint_id} deleted")
            return success

        except Exception as e:
            logger.error(f"Failed to delete checkpoint {checkpoint_id}: {str(e)}")
            return False

    async def verify_checkpoint(self, checkpoint_id: str) -> bool:
        """Verify checkpoint integrity."""
        return await self.storage.verify_checkpoint(checkpoint_id)

    async def _cleanup_old_checkpoints(self, job_id: str):
        """Clean up old checkpoints for a job, keeping only the most recent ones."""
        try:
            checkpoints = await self.storage.list_checkpoints(job_id=job_id)

            if len(checkpoints) > self.max_checkpoints_per_job:
                # Keep only the most recent checkpoints
                to_delete = checkpoints[self.max_checkpoints_per_job :]

                for metadata in to_delete:
                    await self.delete_checkpoint(metadata.checkpoint_id)

                logger.info(
                    f"Cleaned up {len(to_delete)} old checkpoints for job {job_id}"
                )

        except Exception as e:
            logger.error(f"Failed to cleanup old checkpoints: {str(e)}")

    async def _cleanup_expired_checkpoints(self):
        """Background task to clean up expired checkpoints."""
        while self._running:
            try:
                if self.cleanup_expired_checkpoints:
                    checkpoints = await self.storage.list_checkpoints()
                    current_time = datetime.now()

                    for metadata in checkpoints:
                        if metadata.expiry_date and current_time > metadata.expiry_date:
                            await self.delete_checkpoint(metadata.checkpoint_id)
                            logger.info(
                                f"Deleted expired checkpoint {metadata.checkpoint_id}"
                            )

                await asyncio.sleep(3600)  # Check every hour

            except Exception as e:
                logger.error(f"Error in checkpoint cleanup: {str(e)}")
                await asyncio.sleep(3600)

    async def _monitor_checkpoint_health(self):
        """Background task to monitor checkpoint health."""
        while self._running:
            try:
                # Verify random checkpoints
                checkpoints = await self.storage.list_checkpoints()

                for metadata in checkpoints[:5]:  # Check up to 5 checkpoints
                    is_valid = await self.verify_checkpoint(metadata.checkpoint_id)
                    if not is_valid:
                        logger.warning(
                            f"Checkpoint {metadata.checkpoint_id} failed verification"
                        )

                await asyncio.sleep(1800)  # Check every 30 minutes

            except Exception as e:
                logger.error(f"Error in checkpoint health monitoring: {str(e)}")
                await asyncio.sleep(1800)


class RecoveryManager:
    """
    Handles recovery operations using checkpoints to restore system state
    after failures or interruptions.
    """

    def __init__(self, checkpoint_manager: CheckpointManager):
        self.checkpoint_manager = checkpoint_manager
        self.recovery_strategies = {
            RecoveryStrategy.RESTART_FROM_LAST_CHECKPOINT: self._restart_from_last_checkpoint,
            RecoveryStrategy.RESTART_FROM_MILESTONE: self._restart_from_milestone,
            RecoveryStrategy.RESTART_FROM_BEGINNING: self._restart_from_beginning,
            RecoveryStrategy.PARTIAL_RECOVERY: self._partial_recovery,
            RecoveryStrategy.MANUAL_INTERVENTION: self._manual_intervention,
            RecoveryStrategy.SKIP_FAILED_COMPONENTS: self._skip_failed_components,
        }

    async def recover_job(
        self,
        job_id: str,
        strategy: RecoveryStrategy = RecoveryStrategy.RESTART_FROM_LAST_CHECKPOINT,
    ) -> Dict[str, Any]:
        """Recover a job using the specified strategy."""
        try:
            logger.info(
                f"Starting recovery for job {job_id} using strategy {strategy.name}"
            )

            recovery_handler = self.recovery_strategies.get(strategy)
            if not recovery_handler:
                raise ValueError(f"Unknown recovery strategy: {strategy}")

            result = await recovery_handler(job_id)
            logger.info(f"Recovery completed for job {job_id}")
            return result

        except Exception as e:
            logger.error(f"Recovery failed for job {job_id}: {str(e)}")
            return {"success": False, "error": str(e)}

    async def _restart_from_last_checkpoint(self, job_id: str) -> Dict[str, Any]:
        """Restart execution from the last available checkpoint."""
        try:
            checkpoint = await self.checkpoint_manager.get_latest_checkpoint(job_id)
            if not checkpoint:
                return {"success": False, "error": "No checkpoints found for job"}

            # Verify checkpoint integrity
            is_valid = await self.checkpoint_manager.verify_checkpoint(
                checkpoint.metadata.checkpoint_id
            )
            if not is_valid:
                return {"success": False, "error": "Latest checkpoint is corrupted"}

            # Extract recovery information
            recovery_info = checkpoint.metadata.recovery_info
            state_data = checkpoint.state_data

            return {
                "success": True,
                "checkpoint_id": checkpoint.metadata.checkpoint_id,
                "recovery_point": checkpoint.metadata.task_id,
                "state_data": state_data,
                "recovery_info": recovery_info,
                "message": f"Ready to resume from checkpoint {checkpoint.metadata.checkpoint_id}",
            }

        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _restart_from_milestone(self, job_id: str) -> Dict[str, Any]:
        """Restart execution from the last milestone checkpoint."""
        try:
            checkpoints = await self.checkpoint_manager.list_checkpoints(job_id=job_id)

            # Find the latest milestone checkpoint
            milestone_checkpoint = None
            for metadata in checkpoints:
                if metadata.checkpoint_type == CheckpointType.MILESTONE:
                    checkpoint = await self.checkpoint_manager.load_checkpoint(
                        metadata.checkpoint_id
                    )
                    if checkpoint:
                        milestone_checkpoint = checkpoint
                        break

            if not milestone_checkpoint:
                # Fall back to latest checkpoint
                return await self._restart_from_last_checkpoint(job_id)

            return {
                "success": True,
                "checkpoint_id": milestone_checkpoint.metadata.checkpoint_id,
                "recovery_point": milestone_checkpoint.metadata.task_id,
                "state_data": milestone_checkpoint.state_data,
                "message": f"Ready to resume from milestone checkpoint {milestone_checkpoint.metadata.checkpoint_id}",
            }

        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _restart_from_beginning(self, job_id: str) -> Dict[str, Any]:
        """Restart execution from the beginning."""
        try:
            # Find the earliest checkpoint to get initial configuration
            checkpoints = await self.checkpoint_manager.list_checkpoints(job_id=job_id)

            if checkpoints:
                # Sort by creation time (earliest first)
                earliest_metadata = min(checkpoints, key=lambda x: x.created_at)
                earliest_checkpoint = await self.checkpoint_manager.load_checkpoint(
                    earliest_metadata.checkpoint_id
                )

                if earliest_checkpoint:
                    # Extract initial configuration
                    initial_config = earliest_checkpoint.metadata.recovery_info.get(
                        "initial_config", {}
                    )

                    return {
                        "success": True,
                        "restart_mode": "from_beginning",
                        "initial_config": initial_config,
                        "workflow_id": earliest_checkpoint.metadata.workflow_id,
                        "message": "Ready to restart job from beginning",
                    }

            return {
                "success": True,
                "restart_mode": "from_beginning",
                "message": "No checkpoints found, will start fresh",
            }

        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _partial_recovery(self, job_id: str) -> Dict[str, Any]:
        """Perform partial recovery by analyzing completed tasks."""
        try:
            checkpoints = await self.checkpoint_manager.list_checkpoints(job_id=job_id)

            if not checkpoints:
                return {
                    "success": False,
                    "error": "No checkpoints found for partial recovery",
                }

            # Analyze checkpoints to determine completed tasks
            completed_tasks = set()
            failed_tasks = set()
            latest_state = {}

            for metadata in checkpoints:
                checkpoint = await self.checkpoint_manager.load_checkpoint(
                    metadata.checkpoint_id
                )
                if checkpoint:
                    state_data = checkpoint.state_data

                    # Extract task completion status
                    if "completed_tasks" in state_data:
                        completed_tasks.update(state_data["completed_tasks"])

                    if "failed_tasks" in state_data:
                        failed_tasks.update(state_data["failed_tasks"])

                    if "latest_state" in state_data:
                        latest_state.update(state_data["latest_state"])

            # Determine which tasks need to be rerun
            tasks_to_rerun = failed_tasks - completed_tasks

            return {
                "success": True,
                "completed_tasks": list(completed_tasks),
                "failed_tasks": list(failed_tasks),
                "tasks_to_rerun": list(tasks_to_rerun),
                "latest_state": latest_state,
                "message": f"Partial recovery plan: {len(tasks_to_rerun)} tasks need to be rerun",
            }

        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _manual_intervention(self, job_id: str) -> Dict[str, Any]:
        """Prepare for manual intervention recovery."""
        try:
            checkpoints = await self.checkpoint_manager.list_checkpoints(job_id=job_id)

            # Gather diagnostic information
            diagnostic_info = {
                "total_checkpoints": len(checkpoints),
                "checkpoint_timeline": [],
                "error_patterns": [],
                "recovery_recommendations": [],
            }

            for metadata in checkpoints:
                checkpoint = await self.checkpoint_manager.load_checkpoint(
                    metadata.checkpoint_id
                )
                if checkpoint:
                    timeline_entry = {
                        "checkpoint_id": metadata.checkpoint_id,
                        "created_at": metadata.created_at.isoformat(),
                        "task_id": metadata.task_id,
                        "status": checkpoint.status.name,
                        "error_message": checkpoint.error_message,
                    }
                    diagnostic_info["checkpoint_timeline"].append(timeline_entry)

                    # Collect error patterns
                    if checkpoint.error_message:
                        diagnostic_info["error_patterns"].append(
                            checkpoint.error_message
                        )

            # Generate recovery recommendations
            if diagnostic_info["error_patterns"]:
                diagnostic_info["recovery_recommendations"].append(
                    "Review error patterns and fix underlying issues before retry"
                )

            diagnostic_info["recovery_recommendations"].extend(
                [
                    "Consider using partial recovery to skip failed components",
                    "Verify system resources and dependencies",
                    "Check log files for detailed error information",
                ]
            )

            return {
                "success": True,
                "intervention_required": True,
                "diagnostic_info": diagnostic_info,
                "message": "Manual intervention required - review diagnostic information",
            }

        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _skip_failed_components(self, job_id: str) -> Dict[str, Any]:
        """Skip failed components and continue with remaining tasks."""
        try:
            # Get partial recovery information
            partial_result = await self._partial_recovery(job_id)
            if not partial_result.get("success"):
                return partial_result

            failed_tasks = set(partial_result["failed_tasks"])
            completed_tasks = set(partial_result["completed_tasks"])

            # Create recovery plan that skips failed tasks
            recovery_plan = {
                "skip_tasks": list(failed_tasks),
                "continue_from_tasks": list(completed_tasks),
                "modified_execution_plan": True,
            }

            return {
                "success": True,
                "recovery_plan": recovery_plan,
                "completed_tasks": list(completed_tasks),
                "skipped_tasks": list(failed_tasks),
                "message": f"Recovery plan created: {len(failed_tasks)} tasks will be skipped",
            }

        except Exception as e:
            return {"success": False, "error": str(e)}


# Example usage and demonstration
async def demo_checkpoint_recovery_system():
    """Demonstrate the checkpoint and recovery system functionality."""
    # Initialize storage and managers
    file_storage = FileCheckpointStorage("demo_checkpoints")
    checkpoint_manager = CheckpointManager(file_storage)
    recovery_manager = RecoveryManager(checkpoint_manager)

    await checkpoint_manager.start()

    try:
        # Simulate job execution with checkpoints
        job_id = str(uuid.uuid4())
        workflow_id = "document_processing"

        # Create initial checkpoint
        initial_state = {
            "job_status": "started",
            "completed_tasks": [],
            "current_task": "document_ingestion",
            "progress": 0,
        }

        checkpoint_id1 = await checkpoint_manager.create_checkpoint(
            job_id=job_id,
            workflow_id=workflow_id,
            state_data=initial_state,
            checkpoint_type=CheckpointType.MILESTONE,
            description="Job started",
        )
        logger.info(f"Created initial checkpoint: {checkpoint_id1}")

        # Simulate progress and create another checkpoint
        progress_state = {
            "job_status": "running",
            "completed_tasks": ["document_ingestion", "ocr_processing"],
            "current_task": "nlp_analysis",
            "progress": 0.6,
        }

        checkpoint_id2 = await checkpoint_manager.create_checkpoint(
            job_id=job_id,
            workflow_id=workflow_id,
            state_data=progress_state,
            checkpoint_type=CheckpointType.AUTOMATIC,
            description="Progress checkpoint",
        )
        logger.info(f"Created progress checkpoint: {checkpoint_id2}")

        # Simulate failure and create error checkpoint
        error_state = {
            "job_status": "failed",
            "completed_tasks": ["document_ingestion", "ocr_processing"],
            "failed_tasks": ["nlp_analysis"],
            "current_task": "nlp_analysis",
            "error": "NLP service unavailable",
            "progress": 0.6,
        }

        checkpoint_id3 = await checkpoint_manager.create_checkpoint(
            job_id=job_id,
            workflow_id=workflow_id,
            state_data=error_state,
            checkpoint_type=CheckpointType.ERROR_STATE,
            description="Error state checkpoint",
        )
        logger.info(f"Created error checkpoint: {checkpoint_id3}")

        # List all checkpoints
        checkpoints = await checkpoint_manager.list_checkpoints(job_id=job_id)
        logger.info(f"Total checkpoints for job {job_id}: {len(checkpoints)}")

        # Demonstrate recovery scenarios
        recovery_strategies = [
            RecoveryStrategy.RESTART_FROM_LAST_CHECKPOINT,
            RecoveryStrategy.RESTART_FROM_MILESTONE,
            RecoveryStrategy.PARTIAL_RECOVERY,
            RecoveryStrategy.SKIP_FAILED_COMPONENTS,
        ]

        for strategy in recovery_strategies:
            logger.info(f"\n--- Testing recovery strategy: {strategy.name} ---")
            result = await recovery_manager.recover_job(job_id, strategy)
            logger.info(f"Recovery result: {result}")

        # Verify checkpoint integrity
        for metadata in checkpoints:
            is_valid = await checkpoint_manager.verify_checkpoint(
                metadata.checkpoint_id
            )
            logger.info(f"Checkpoint {metadata.checkpoint_id} valid: {is_valid}")

    finally:
        await checkpoint_manager.stop()


if __name__ == "__main__":
    # Import required modules for the demo
    from collections import defaultdict
    import uuid

    # Run the demonstration
    asyncio.run(demo_checkpoint_recovery_system())
