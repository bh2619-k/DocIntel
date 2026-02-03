"""
Real-Time Document Intelligence & Compliance Engine
Redis Queue and Stream Configuration
Version: 1.0
Date: 2026-02-03

This module defines the Redis data structures and operations for:
1. Priority queues for document processing (High, Normal, Retry)
2. Session and cache management
3. Pub/sub for internal event routing
4. Stream configuration for checkpoint management
"""

import redis
import json
import uuid
import time
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum


class QueuePriority(Enum):
    """Document processing queue priorities"""

    HIGH = "documents:queue:high"
    NORMAL = "documents:queue:normal"
    RETRY = "documents:queue:retry"


class EventChannel(Enum):
    """Pub/sub channels for internal events"""

    DOCUMENT_STATUS = "events:document:status"
    ALERTS = "events:alerts"
    COMPLIANCE_VIOLATIONS = "events:compliance"
    SYSTEM_HEALTH = "events:system:health"


@dataclass
class DocumentJob:
    """Document processing job structure"""

    job_id: str
    document_id: str
    tenant_id: str
    priority: QueuePriority
    document_type: str
    file_path: str
    created_at: datetime
    retry_count: int = 0
    metadata: Dict[str, Any] = None


class RedisManager:
    """
    Redis manager for document intelligence system
    Handles streams, queues, pub/sub, and caching
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        password: str = None,
        db: int = 0,
        tenant_id: str = None,
    ):
        """Initialize Redis connection with tenant isolation"""
        self.redis_client = redis.Redis(
            host=host, port=port, password=password, db=db, decode_responses=True
        )
        self.tenant_id = tenant_id

    def get_tenant_key(self, key: str) -> str:
        """Generate tenant-isolated key"""
        if self.tenant_id:
            return f"tenant:{self.tenant_id}:{key}"
        return key

    # =============================================================================
    # STREAM OPERATIONS FOR DOCUMENT PROCESSING
    # =============================================================================

    def add_document_job(self, job: DocumentJob) -> str:
        """Add document to processing queue stream"""
        stream_name = self.get_tenant_key(job.priority.value)

        job_data = {
            "job_id": job.job_id,
            "document_id": job.document_id,
            "tenant_id": job.tenant_id,
            "document_type": job.document_type,
            "file_path": job.file_path,
            "created_at": job.created_at.isoformat(),
            "retry_count": str(job.retry_count),
            "metadata": json.dumps(job.metadata or {}),
        }

        # Add to stream with auto-generated ID
        stream_id = self.redis_client.xadd(stream_name, job_data, maxlen=10000)

        # Track job in lookup table for status queries
        job_key = self.get_tenant_key(f"jobs:{job.job_id}")
        self.redis_client.hset(
            job_key,
            mapping={
                "stream_name": stream_name,
                "stream_id": stream_id,
                "status": "QUEUED",
                "created_at": job.created_at.isoformat(),
            },
        )
        self.redis_client.expire(job_key, 86400)  # 24 hour TTL

        return stream_id

    def get_next_job(
        self, queue_priority: QueuePriority, consumer_group: str, consumer_name: str
    ) -> Optional[Dict]:
        """Get next job from priority queue using consumer groups"""
        stream_name = self.get_tenant_key(queue_priority.value)

        try:
            # Ensure consumer group exists
            try:
                self.redis_client.xgroup_create(
                    stream_name, consumer_group, id="0", mkstream=True
                )
            except redis.ResponseError as e:
                if "BUSYGROUP" not in str(e):
                    raise

            # Read from stream
            messages = self.redis_client.xreadgroup(
                consumer_group,
                consumer_name,
                {stream_name: ">"},
                count=1,
                block=1000,  # 1 second timeout
            )

            if messages:
                stream, msgs = messages[0]
                if msgs:
                    stream_id, fields = msgs[0]
                    return {"stream_id": stream_id, "stream_name": stream, **fields}

        except Exception as e:
            print(f"Error reading from queue {queue_priority}: {e}")

        return None

    def acknowledge_job(
        self, queue_priority: QueuePriority, consumer_group: str, stream_id: str
    ):
        """Acknowledge job completion"""
        stream_name = self.get_tenant_key(queue_priority.value)
        self.redis_client.xack(stream_name, consumer_group, stream_id)

    def move_to_retry_queue(self, job: Dict, error_message: str):
        """Move failed job to retry queue"""
        retry_count = int(job.get("retry_count", 0)) + 1

        # Create retry job
        retry_job = DocumentJob(
            job_id=job["job_id"],
            document_id=job["document_id"],
            tenant_id=job["tenant_id"],
            priority=QueuePriority.RETRY,
            document_type=job["document_type"],
            file_path=job["file_path"],
            created_at=datetime.now(),
            retry_count=retry_count,
            metadata=json.loads(job.get("metadata", "{}")),
        )

        # Add retry metadata
        if not retry_job.metadata:
            retry_job.metadata = {}
        retry_job.metadata["error_message"] = error_message
        retry_job.metadata["retry_scheduled_at"] = datetime.now().isoformat()

        # Add to retry queue if under max retries (3)
        if retry_count <= 3:
            self.add_document_job(retry_job)
        else:
            # Move to dead letter queue
            self.move_to_dead_letter(retry_job, "Max retries exceeded")

    def move_to_dead_letter(self, job: DocumentJob, reason: str):
        """Move job to dead letter queue for manual intervention"""
        dead_letter_key = self.get_tenant_key("dead_letter:documents")

        dead_letter_entry = {
            "job_id": job.job_id,
            "document_id": job.document_id,
            "tenant_id": job.tenant_id,
            "failed_at": datetime.now().isoformat(),
            "reason": reason,
            "job_data": json.dumps(
                {
                    "document_type": job.document_type,
                    "file_path": job.file_path,
                    "retry_count": job.retry_count,
                    "metadata": job.metadata,
                }
            ),
        }

        self.redis_client.lpush(dead_letter_key, json.dumps(dead_letter_entry))

        # Publish alert for dead letter entry
        self.publish_event(
            EventChannel.SYSTEM_HEALTH,
            {
                "type": "DEAD_LETTER_ENTRY",
                "job_id": job.job_id,
                "document_id": job.document_id,
                "tenant_id": job.tenant_id,
                "reason": reason,
            },
        )

    # =============================================================================
    # CHECKPOINT MANAGEMENT
    # =============================================================================

    def set_checkpoint(
        self, document_id: str, stage: str, status: str, metadata: Dict = None
    ):
        """Set processing checkpoint for document"""
        checkpoint_key = self.get_tenant_key(f"checkpoints:{document_id}:{stage}")

        checkpoint_data = {
            "document_id": document_id,
            "stage": stage,
            "status": status,
            "timestamp": datetime.now().isoformat(),
            "metadata": json.dumps(metadata or {}),
        }

        self.redis_client.hset(checkpoint_key, mapping=checkpoint_data)
        self.redis_client.expire(checkpoint_key, 86400)  # 24 hour TTL

        # Update overall document status
        doc_status_key = self.get_tenant_key(f"document_status:{document_id}")
        self.redis_client.hset(doc_status_key, stage, status)
        self.redis_client.expire(doc_status_key, 86400)

    def get_checkpoint(self, document_id: str, stage: str) -> Optional[Dict]:
        """Get checkpoint for document stage"""
        checkpoint_key = self.get_tenant_key(f"checkpoints:{document_id}:{stage}")
        data = self.redis_client.hgetall(checkpoint_key)

        if data:
            data["metadata"] = json.loads(data.get("metadata", "{}"))
            return data
        return None

    def get_document_checkpoints(self, document_id: str) -> Dict[str, str]:
        """Get all checkpoints for a document"""
        doc_status_key = self.get_tenant_key(f"document_status:{document_id}")
        return self.redis_client.hgetall(doc_status_key)

    # =============================================================================
    # PUB/SUB FOR REAL-TIME EVENTS
    # =============================================================================

    def publish_event(self, channel: EventChannel, data: Dict):
        """Publish event to channel"""
        channel_name = self.get_tenant_key(channel.value)

        event_data = {
            "event_id": str(uuid.uuid4()),
            "timestamp": datetime.now().isoformat(),
            "tenant_id": self.tenant_id,
            **data,
        }

        self.redis_client.publish(channel_name, json.dumps(event_data))

    def subscribe_to_events(self, channels: List[EventChannel]) -> redis.client.PubSub:
        """Subscribe to event channels"""
        pubsub = self.redis_client.pubsub()

        channel_names = [self.get_tenant_key(channel.value) for channel in channels]
        pubsub.subscribe(channel_names)

        return pubsub

    # =============================================================================
    # SESSION AND CACHE MANAGEMENT
    # =============================================================================

    def set_session(self, session_id: str, user_data: Dict, ttl: int = 3600):
        """Store user session data"""
        session_key = self.get_tenant_key(f"sessions:{session_id}")
        self.redis_client.setex(session_key, ttl, json.dumps(user_data))

    def get_session(self, session_id: str) -> Optional[Dict]:
        """Get user session data"""
        session_key = self.get_tenant_key(f"sessions:{session_id}")
        data = self.redis_client.get(session_key)

        if data:
            return json.loads(data)
        return None

    def delete_session(self, session_id: str):
        """Delete user session"""
        session_key = self.get_tenant_key(f"sessions:{session_id}")
        self.redis_client.delete(session_key)

    def cache_set(self, key: str, value: Any, ttl: int = 3600):
        """Set cached value with TTL"""
        cache_key = self.get_tenant_key(f"cache:{key}")
        self.redis_client.setex(cache_key, ttl, json.dumps(value))

    def cache_get(self, key: str) -> Optional[Any]:
        """Get cached value"""
        cache_key = self.get_tenant_key(f"cache:{key}")
        data = self.redis_client.get(cache_key)

        if data:
            return json.loads(data)
        return None

    def cache_delete(self, key: str):
        """Delete cached value"""
        cache_key = self.get_tenant_key(f"cache:{key}")
        self.redis_client.delete(cache_key)

    # =============================================================================
    # RATE LIMITING
    # =============================================================================

    def rate_limit_check(self, identifier: str, limit: int, window: int) -> bool:
        """
        Check if rate limit is exceeded
        Returns True if under limit, False if over limit
        """
        rate_key = self.get_tenant_key(f"rate_limit:{identifier}")

        current_time = int(datetime.now().timestamp())
        window_start = current_time - window

        # Remove old entries
        self.redis_client.zremrangebyscore(rate_key, 0, window_start)

        # Count current requests
        current_count = self.redis_client.zcard(rate_key)

        if current_count >= limit:
            return False

        # Add current request
        self.redis_client.zadd(rate_key, {str(current_time): current_time})
        self.redis_client.expire(rate_key, window)

        return True

    # =============================================================================
    # MONITORING AND HEALTH CHECKS
    # =============================================================================

    def get_queue_stats(self) -> Dict[str, int]:
        """Get queue length statistics"""
        stats = {}

        for priority in QueuePriority:
            stream_name = self.get_tenant_key(priority.value)
            try:
                info = self.redis_client.xinfo_stream(stream_name)
                stats[priority.name.lower()] = info.get("length", 0)
            except redis.ResponseError:
                stats[priority.name.lower()] = 0

        return stats

    def health_check(self) -> Dict[str, Any]:
        """Perform Redis health check"""
        try:
            # Ping test
            ping_result = self.redis_client.ping()

            # Connection count and memory info
            info = self.redis_client.info()

            return {
                "status": "healthy",
                "ping": ping_result,
                "connected_clients": info.get("connected_clients", 0),
                "used_memory": info.get("used_memory", 0),
                "used_memory_human": info.get("used_memory_human", "0B"),
                "queue_stats": self.get_queue_stats(),
            }

        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}


# =============================================================================
# REDIS STREAM CONSUMER EXAMPLE
# =============================================================================


class DocumentProcessor:
    """Example consumer for document processing streams"""

    def __init__(
        self, redis_manager: RedisManager, consumer_group: str, consumer_name: str
    ):
        self.redis_manager = redis_manager
        self.consumer_group = consumer_group
        self.consumer_name = consumer_name

    def process_documents(self):
        """Main processing loop"""
        while True:
            # Process high priority first
            job = self.redis_manager.get_next_job(
                QueuePriority.HIGH, self.consumer_group, self.consumer_name
            )

            if not job:
                # Process normal priority
                job = self.redis_manager.get_next_job(
                    QueuePriority.NORMAL, self.consumer_group, self.consumer_name
                )

            if not job:
                # Process retry queue
                job = self.redis_manager.get_next_job(
                    QueuePriority.RETRY, self.consumer_group, self.consumer_name
                )

            if job:
                try:
                    # Process the document
                    self.process_single_document(job)

                    # Acknowledge successful processing
                    queue_priority = (
                        QueuePriority.HIGH
                        if "high" in job["stream_name"]
                        else QueuePriority.NORMAL
                        if "normal" in job["stream_name"]
                        else QueuePriority.RETRY
                    )

                    self.redis_manager.acknowledge_job(
                        queue_priority, self.consumer_group, job["stream_id"]
                    )

                except Exception as e:
                    # Handle processing error
                    self.redis_manager.move_to_retry_queue(job, str(e))

            else:
                # No jobs available, short sleep
                time.sleep(1)

    def process_single_document(self, job: Dict):
        """Process a single document job"""
        document_id = job["document_id"]

        # Set processing checkpoint
        self.redis_manager.set_checkpoint(document_id, "PREPROCESSING", "IN_PROGRESS")

        # Publish status update
        self.redis_manager.publish_event(
            EventChannel.DOCUMENT_STATUS,
            {
                "document_id": document_id,
                "status": "PROCESSING_STARTED",
                "stage": "PREPROCESSING",
            },
        )

        # Simulate processing...
        # (This would call the actual document processing pipeline)

        # Mark checkpoint complete
        self.redis_manager.set_checkpoint(document_id, "PREPROCESSING", "COMPLETED")

        print(f"Processed document {document_id}")
