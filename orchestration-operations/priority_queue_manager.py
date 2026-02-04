"""
Priority Queue Management System for Document Intelligence Pipeline

This module provides comprehensive task scheduling, resource allocation, and
load balancing capabilities for the document intelligence pipeline orchestration.

Key Features:
- Multi-level priority queue with customizable priority schemes
- Resource-aware task scheduling and allocation
- Load balancing across available compute resources
- Dynamic priority adjustment based on SLA and urgency
- Queue monitoring and performance optimization
- Fair scheduling algorithms to prevent starvation
- Resource pool management with elasticity

Classes:
    QueuePriority: Enumeration of queue priority levels
    ResourceType: Types of resources in the system
    ResourcePool: Pool of available resources
    TaskSchedulingPolicy: Scheduling policy configurations
    QueuedTask: Task representation in the queue
    ResourceAllocation: Resource allocation for tasks
    PriorityQueueManager: Main queue management system
    LoadBalancer: Load balancing coordinator
    ResourceManager: Resource allocation and monitoring

Dependencies:
    - asyncio: For asynchronous task processing
    - heapq: For efficient priority queue operations
    - threading: For concurrent access control
    - dataclasses: For structured data
    - logging: For comprehensive logging
    - datetime: For timing and scheduling
    - typing: For type hints
    - enum: For enumeration types
"""

import asyncio
import heapq
import logging
import threading
import time
import uuid
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Set, Union
import psutil
import json
import os


# Configure comprehensive logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class QueuePriority(Enum):
    """Priority levels for tasks in the queue."""

    CRITICAL = 1
    HIGH = 2
    NORMAL = 3
    LOW = 4
    BACKGROUND = 5


class ResourceType(Enum):
    """Types of computational resources."""

    CPU = "cpu"
    MEMORY = "memory"
    GPU = "gpu"
    STORAGE = "storage"
    NETWORK = "network"
    CUSTOM = "custom"


class SchedulingPolicy(Enum):
    """Task scheduling policies."""

    FIFO = "first_in_first_out"
    PRIORITY_FIRST = "priority_first"
    SHORTEST_JOB_FIRST = "shortest_job_first"
    ROUND_ROBIN = "round_robin"
    FAIR_SHARE = "fair_share"
    DEADLINE_SENSITIVE = "deadline_sensitive"


class TaskState(Enum):
    """States of tasks in the queue."""

    QUEUED = auto()
    SCHEDULED = auto()
    RUNNING = auto()
    COMPLETED = auto()
    FAILED = auto()
    CANCELLED = auto()
    TIMEOUT = auto()
    WAITING_RESOURCES = auto()


@dataclass
class ResourceRequirement:
    """Resource requirement specification for a task."""

    resource_type: ResourceType
    amount: float
    unit: str
    is_required: bool = True
    max_wait_time: Optional[int] = None  # seconds


@dataclass
class ResourceCapacity:
    """Resource capacity specification."""

    resource_type: ResourceType
    total_capacity: float
    available_capacity: float
    unit: str
    utilization_threshold: float = 0.8


@dataclass
class QueuedTask:
    """Represents a task in the priority queue."""

    task_id: str
    job_id: str
    priority: QueuePriority
    submission_time: datetime
    estimated_duration: Optional[int] = None
    deadline: Optional[datetime] = None
    resource_requirements: List[ResourceRequirement] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    state: TaskState = TaskState.QUEUED
    retry_count: int = 0
    max_retries: int = 3
    owner: str = "system"
    tags: Dict[str, str] = field(default_factory=dict)

    def __lt__(self, other):
        """Enable priority queue comparison."""
        # Primary sort by priority value (lower number = higher priority)
        if self.priority.value != other.priority.value:
            return self.priority.value < other.priority.value

        # Secondary sort by submission time (earlier = higher priority)
        return self.submission_time < other.submission_time

    @property
    def wait_time(self) -> float:
        """Calculate current wait time in seconds."""
        return (datetime.now() - self.submission_time).total_seconds()

    @property
    def is_deadline_critical(self) -> bool:
        """Check if task is approaching deadline."""
        if not self.deadline:
            return False
        remaining = (self.deadline - datetime.now()).total_seconds()
        return remaining < (self.estimated_duration or 300)  # 5 minutes default


@dataclass
class ResourceAllocation:
    """Represents an allocation of resources to a task."""

    allocation_id: str
    task_id: str
    resource_allocations: Dict[ResourceType, float]
    allocation_time: datetime
    expected_release_time: datetime
    actual_release_time: Optional[datetime] = None
    status: str = "active"


class ResourcePool:
    """Manages a pool of computational resources."""

    def __init__(self, initial_capacities: Dict[ResourceType, ResourceCapacity]):
        self.capacities = initial_capacities
        self.allocations = {}
        self.allocation_history = deque(maxlen=1000)
        self.lock = threading.RLock()
        self._monitor_system_resources()

    def _monitor_system_resources(self):
        """Update resource capacities based on system status."""
        try:
            # CPU resources
            cpu_count = psutil.cpu_count()
            cpu_usage = psutil.cpu_percent(interval=1)
            cpu_available = cpu_count * (1 - cpu_usage / 100)

            if ResourceType.CPU in self.capacities:
                self.capacities[ResourceType.CPU].available_capacity = max(
                    0, cpu_available
                )

            # Memory resources
            memory = psutil.virtual_memory()
            memory_available_gb = memory.available / (1024**3)

            if ResourceType.MEMORY in self.capacities:
                self.capacities[
                    ResourceType.MEMORY
                ].available_capacity = memory_available_gb

            # Storage resources
            disk_usage = psutil.disk_usage("/")
            disk_available_gb = disk_usage.free / (1024**3)

            if ResourceType.STORAGE in self.capacities:
                self.capacities[
                    ResourceType.STORAGE
                ].available_capacity = disk_available_gb

        except Exception as e:
            logger.warning(f"Failed to monitor system resources: {str(e)}")

    def can_allocate(self, requirements: List[ResourceRequirement]) -> bool:
        """Check if resources can be allocated for the given requirements."""
        with self.lock:
            self._monitor_system_resources()

            for req in requirements:
                if req.is_required:
                    capacity = self.capacities.get(req.resource_type)
                    if not capacity or capacity.available_capacity < req.amount:
                        return False
            return True

    def allocate_resources(
        self,
        task_id: str,
        requirements: List[ResourceRequirement],
        duration_estimate: Optional[int] = None,
    ) -> Optional[ResourceAllocation]:
        """Allocate resources for a task."""
        with self.lock:
            if not self.can_allocate(requirements):
                return None

            allocation_id = str(uuid.uuid4())
            allocations = {}

            # Perform the allocation
            for req in requirements:
                if req.is_required:
                    capacity = self.capacities[req.resource_type]
                    capacity.available_capacity -= req.amount
                    allocations[req.resource_type] = req.amount

            # Create allocation record
            allocation = ResourceAllocation(
                allocation_id=allocation_id,
                task_id=task_id,
                resource_allocations=allocations,
                allocation_time=datetime.now(),
                expected_release_time=datetime.now()
                + timedelta(seconds=duration_estimate or 300),
            )

            self.allocations[allocation_id] = allocation
            self.allocation_history.append(allocation)

            logger.info(f"Allocated resources for task {task_id}: {allocations}")
            return allocation

    def release_resources(self, allocation_id: str):
        """Release allocated resources."""
        with self.lock:
            allocation = self.allocations.pop(allocation_id, None)
            if allocation:
                allocation.actual_release_time = datetime.now()
                allocation.status = "released"

                # Return resources to pool
                for resource_type, amount in allocation.resource_allocations.items():
                    capacity = self.capacities.get(resource_type)
                    if capacity:
                        capacity.available_capacity += amount

                logger.info(f"Released resources for task {allocation.task_id}")

    def get_resource_utilization(self) -> Dict[ResourceType, float]:
        """Get current resource utilization percentages."""
        with self.lock:
            utilization = {}
            for resource_type, capacity in self.capacities.items():
                total = capacity.total_capacity
                available = capacity.available_capacity
                utilized = total - available
                utilization[resource_type] = (utilized / total) if total > 0 else 0
            return utilization

    def get_capacity_info(self) -> Dict[ResourceType, Dict[str, Any]]:
        """Get detailed capacity information for all resources."""
        with self.lock:
            info = {}
            for resource_type, capacity in self.capacities.items():
                info[resource_type] = {
                    "total": capacity.total_capacity,
                    "available": capacity.available_capacity,
                    "used": capacity.total_capacity - capacity.available_capacity,
                    "utilization": (
                        capacity.total_capacity - capacity.available_capacity
                    )
                    / capacity.total_capacity,
                    "unit": capacity.unit,
                    "threshold": capacity.utilization_threshold,
                }
            return info


class TaskSchedulingPolicy:
    """Configurable task scheduling policy."""

    def __init__(self, policy_type: SchedulingPolicy, config: Dict[str, Any] = None):
        self.policy_type = policy_type
        self.config = config or {}
        self.user_quotas = defaultdict(lambda: {"used": 0, "limit": float("inf")})

    def calculate_effective_priority(self, task: QueuedTask) -> float:
        """Calculate effective priority considering various factors."""
        base_priority = task.priority.value

        # Deadline pressure adjustment
        deadline_factor = 1.0
        if task.is_deadline_critical:
            deadline_factor = 0.1  # Boost priority significantly

        # Wait time adjustment (aging)
        wait_time_hours = task.wait_time / 3600
        aging_factor = max(0.1, 1.0 - (wait_time_hours * 0.1))

        # User quota consideration
        user_quota = self.user_quotas[task.owner]
        quota_factor = 1.0
        if user_quota["used"] > user_quota["limit"]:
            quota_factor = 2.0  # Lower priority for users over quota

        effective_priority = (
            base_priority * deadline_factor * aging_factor * quota_factor
        )
        return effective_priority

    def should_schedule_task(self, task: QueuedTask, current_load: float) -> bool:
        """Determine if a task should be scheduled based on current conditions."""
        # Check resource availability
        if current_load > self.config.get("max_load_threshold", 0.9):
            return task.priority.value <= 2  # Only critical and high priority

        # Check user quotas
        user_quota = self.user_quotas[task.owner]
        if user_quota["used"] >= user_quota["limit"]:
            return task.priority == QueuePriority.CRITICAL

        return True

    def update_user_usage(self, owner: str, resource_usage: float):
        """Update user resource usage tracking."""
        self.user_quotas[owner]["used"] += resource_usage

    def reset_user_quotas(self):
        """Reset user quotas (typically done periodically)."""
        for user_data in self.user_quotas.values():
            user_data["used"] = 0


class PriorityQueueManager:
    """
    Main priority queue management system that handles task scheduling,
    resource allocation, and load balancing.
    """

    def __init__(
        self,
        resource_pool: ResourcePool,
        scheduling_policy: TaskSchedulingPolicy,
        max_concurrent_tasks: int = 50,
        queue_monitoring_interval: int = 5,
    ):
        self.resource_pool = resource_pool
        self.scheduling_policy = scheduling_policy
        self.max_concurrent_tasks = max_concurrent_tasks
        self.queue_monitoring_interval = queue_monitoring_interval

        # Queue management
        self.task_queues = {priority: [] for priority in QueuePriority}
        self.running_tasks = {}
        self.completed_tasks = deque(maxlen=1000)
        self.failed_tasks = deque(maxlen=1000)

        # Synchronization
        self.queue_lock = threading.RLock()
        self.execution_semaphore = asyncio.Semaphore(max_concurrent_tasks)

        # Monitoring
        self.metrics = {
            "tasks_submitted": 0,
            "tasks_completed": 0,
            "tasks_failed": 0,
            "total_wait_time": 0,
            "average_wait_time": 0,
            "queue_length": 0,
            "resource_utilization": {},
        }

        # Background tasks
        self._running = False
        self._background_tasks = []

    async def start(self):
        """Start the priority queue manager."""
        self._running = True
        logger.info("Starting Priority Queue Manager")

        # Start background monitoring tasks
        self._background_tasks = [
            asyncio.create_task(self._queue_processor()),
            asyncio.create_task(self._queue_monitor()),
            asyncio.create_task(self._resource_monitor()),
            asyncio.create_task(self._metrics_collector()),
        ]

    async def stop(self):
        """Stop the priority queue manager."""
        self._running = False
        logger.info("Stopping Priority Queue Manager")

        # Cancel background tasks
        for task in self._background_tasks:
            task.cancel()

        # Wait for running tasks to complete
        while self.running_tasks:
            logger.info(
                f"Waiting for {len(self.running_tasks)} running tasks to complete"
            )
            await asyncio.sleep(1)

    def submit_task(
        self,
        task_id: str,
        job_id: str,
        priority: QueuePriority,
        resource_requirements: List[ResourceRequirement] = None,
        estimated_duration: int = None,
        deadline: datetime = None,
        metadata: Dict[str, Any] = None,
        owner: str = "system",
    ) -> bool:
        """Submit a task to the priority queue."""
        try:
            task = QueuedTask(
                task_id=task_id,
                job_id=job_id,
                priority=priority,
                submission_time=datetime.now(),
                estimated_duration=estimated_duration,
                deadline=deadline,
                resource_requirements=resource_requirements or [],
                metadata=metadata or {},
                owner=owner,
            )

            with self.queue_lock:
                heapq.heappush(self.task_queues[priority], task)
                self.metrics["tasks_submitted"] += 1
                self._update_queue_metrics()

            logger.info(f"Submitted task {task_id} with priority {priority.name}")
            return True

        except Exception as e:
            logger.error(f"Failed to submit task {task_id}: {str(e)}")
            return False

    async def _queue_processor(self):
        """Background task to process queued tasks."""
        while self._running:
            try:
                # Get next task to schedule
                next_task = self._get_next_schedulable_task()

                if next_task and len(self.running_tasks) < self.max_concurrent_tasks:
                    # Check if resources are available
                    if self.resource_pool.can_allocate(next_task.resource_requirements):
                        async with self.execution_semaphore:
                            asyncio.create_task(self._execute_task(next_task))
                    else:
                        # Move task to waiting for resources
                        next_task.state = TaskState.WAITING_RESOURCES
                        logger.debug(f"Task {next_task.task_id} waiting for resources")

                await asyncio.sleep(0.1)  # Small delay to prevent busy waiting

            except Exception as e:
                logger.error(f"Error in queue processor: {str(e)}")
                await asyncio.sleep(1)

    def _get_next_schedulable_task(self) -> Optional[QueuedTask]:
        """Get the next task that should be scheduled."""
        with self.queue_lock:
            # Check all priority levels
            for priority in QueuePriority:
                queue = self.task_queues[priority]

                # Find schedulable tasks in this priority level
                schedulable_tasks = []
                remaining_tasks = []

                while queue:
                    task = heapq.heappop(queue)

                    if task.state == TaskState.QUEUED:
                        current_load = (
                            len(self.running_tasks) / self.max_concurrent_tasks
                        )

                        if self.scheduling_policy.should_schedule_task(
                            task, current_load
                        ):
                            schedulable_tasks.append(task)
                        else:
                            remaining_tasks.append(task)
                    else:
                        remaining_tasks.append(task)

                # Put non-schedulable tasks back
                for task in remaining_tasks:
                    heapq.heappush(queue, task)

                # Return the best schedulable task
                if schedulable_tasks:
                    # Sort by effective priority
                    best_task = min(
                        schedulable_tasks,
                        key=lambda t: self.scheduling_policy.calculate_effective_priority(
                            t
                        ),
                    )

                    # Put other schedulable tasks back
                    for task in schedulable_tasks:
                        if task != best_task:
                            heapq.heappush(queue, task)

                    return best_task

            return None

    async def _execute_task(self, task: QueuedTask):
        """Execute a single task."""
        task_id = task.task_id

        try:
            # Allocate resources
            allocation = self.resource_pool.allocate_resources(
                task_id, task.resource_requirements, task.estimated_duration
            )

            if not allocation:
                logger.warning(f"Failed to allocate resources for task {task_id}")
                task.state = TaskState.WAITING_RESOURCES
                return

            # Update task state
            task.state = TaskState.RUNNING
            start_time = datetime.now()

            with self.queue_lock:
                self.running_tasks[task_id] = task

            logger.info(f"Executing task {task_id}")

            # Simulate task execution (replace with actual task execution logic)
            execution_duration = task.estimated_duration or 5  # Default 5 seconds
            await asyncio.sleep(execution_duration)

            # Task completed successfully
            task.state = TaskState.COMPLETED
            completion_time = datetime.now()
            wait_time = (start_time - task.submission_time).total_seconds()

            # Update metrics
            with self.queue_lock:
                self.running_tasks.pop(task_id, None)
                self.completed_tasks.append(task)
                self.metrics["tasks_completed"] += 1
                self.metrics["total_wait_time"] += wait_time
                self._update_average_wait_time()

            # Release resources
            if allocation:
                self.resource_pool.release_resources(allocation.allocation_id)

            # Update user usage
            resource_usage = (
                sum(allocation.resource_allocations.values()) if allocation else 0
            )
            self.scheduling_policy.update_user_usage(task.owner, resource_usage)

            logger.info(
                f"Task {task_id} completed successfully (wait time: {wait_time:.2f}s)"
            )

        except Exception as e:
            # Task failed
            task.state = TaskState.FAILED
            logger.error(f"Task {task_id} failed: {str(e)}")

            with self.queue_lock:
                self.running_tasks.pop(task_id, None)
                self.failed_tasks.append(task)
                self.metrics["tasks_failed"] += 1

            # Handle retries
            if task.retry_count < task.max_retries:
                task.retry_count += 1
                task.state = TaskState.QUEUED
                await asyncio.sleep(2**task.retry_count)  # Exponential backoff

                # Resubmit for retry
                with self.queue_lock:
                    heapq.heappush(self.task_queues[task.priority], task)

                logger.info(f"Retrying task {task_id} (attempt {task.retry_count})")

            # Release resources if allocated
            if "allocation" in locals() and allocation:
                self.resource_pool.release_resources(allocation.allocation_id)

    async def _queue_monitor(self):
        """Monitor queue status and performance."""
        while self._running:
            try:
                with self.queue_lock:
                    # Calculate queue statistics
                    total_queued = sum(
                        len(queue) for queue in self.task_queues.values()
                    )
                    running_count = len(self.running_tasks)

                    # Check for stalled tasks
                    current_time = datetime.now()
                    stalled_tasks = []

                    for task in self.running_tasks.values():
                        if task.estimated_duration:
                            runtime = (
                                current_time - task.submission_time
                            ).total_seconds()
                            if (
                                runtime > task.estimated_duration * 2
                            ):  # 2x expected duration
                                stalled_tasks.append(task)

                    if stalled_tasks:
                        logger.warning(
                            f"Detected {len(stalled_tasks)} potentially stalled tasks"
                        )

                    # Log queue status
                    if total_queued > 0 or running_count > 0:
                        logger.info(
                            f"Queue status - Queued: {total_queued}, Running: {running_count}"
                        )

                await asyncio.sleep(self.queue_monitoring_interval)

            except Exception as e:
                logger.error(f"Error in queue monitor: {str(e)}")
                await asyncio.sleep(self.queue_monitoring_interval)

    async def _resource_monitor(self):
        """Monitor resource utilization and availability."""
        while self._running:
            try:
                utilization = self.resource_pool.get_resource_utilization()
                capacity_info = self.resource_pool.get_capacity_info()

                # Check for resource pressure
                high_utilization = {
                    res: util for res, util in utilization.items() if util > 0.8
                }

                if high_utilization:
                    logger.warning(
                        f"High resource utilization detected: {high_utilization}"
                    )

                # Update metrics
                self.metrics["resource_utilization"] = utilization

                # Log resource status periodically
                logger.debug(f"Resource utilization: {utilization}")

                await asyncio.sleep(30)  # Check every 30 seconds

            except Exception as e:
                logger.error(f"Error in resource monitor: {str(e)}")
                await asyncio.sleep(30)

    async def _metrics_collector(self):
        """Collect and update system metrics."""
        while self._running:
            try:
                with self.queue_lock:
                    self._update_queue_metrics()

                await asyncio.sleep(60)  # Update every minute

            except Exception as e:
                logger.error(f"Error in metrics collector: {str(e)}")
                await asyncio.sleep(60)

    def _update_queue_metrics(self):
        """Update queue-related metrics."""
        total_queued = sum(len(queue) for queue in self.task_queues.values())
        self.metrics["queue_length"] = total_queued

    def _update_average_wait_time(self):
        """Update average wait time metric."""
        completed_count = self.metrics["tasks_completed"]
        if completed_count > 0:
            self.metrics["average_wait_time"] = (
                self.metrics["total_wait_time"] / completed_count
            )

    # Public API methods
    def get_queue_status(self) -> Dict[str, Any]:
        """Get current queue status."""
        with self.queue_lock:
            status = {
                "queue_lengths": {
                    priority.name: len(queue)
                    for priority, queue in self.task_queues.items()
                },
                "running_tasks": len(self.running_tasks),
                "total_queued": sum(len(queue) for queue in self.task_queues.values()),
                "metrics": self.metrics.copy(),
            }
            return status

    def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a specific task."""
        # Check running tasks
        with self.queue_lock:
            if task_id in self.running_tasks:
                task = self.running_tasks[task_id]
                return {
                    "task_id": task_id,
                    "state": task.state.name,
                    "priority": task.priority.name,
                    "wait_time": task.wait_time,
                    "submission_time": task.submission_time.isoformat(),
                }

            # Check completed tasks
            for task in self.completed_tasks:
                if task.task_id == task_id:
                    return {
                        "task_id": task_id,
                        "state": task.state.name,
                        "priority": task.priority.name,
                        "completed": True,
                    }

            # Check failed tasks
            for task in self.failed_tasks:
                if task.task_id == task_id:
                    return {
                        "task_id": task_id,
                        "state": task.state.name,
                        "priority": task.priority.name,
                        "retry_count": task.retry_count,
                        "failed": True,
                    }

        return None

    def cancel_task(self, task_id: str) -> bool:
        """Cancel a queued or running task."""
        with self.queue_lock:
            # Check running tasks
            if task_id in self.running_tasks:
                task = self.running_tasks[task_id]
                task.state = TaskState.CANCELLED
                logger.info(f"Cancelled running task {task_id}")
                return True

            # Check queued tasks
            for priority, queue in self.task_queues.items():
                for i, task in enumerate(queue):
                    if task.task_id == task_id:
                        task.state = TaskState.CANCELLED
                        queue.pop(i)
                        heapq.heapify(queue)  # Restore heap property
                        logger.info(f"Cancelled queued task {task_id}")
                        return True

        return False

    def get_resource_status(self) -> Dict[str, Any]:
        """Get current resource status."""
        return {
            "utilization": self.resource_pool.get_resource_utilization(),
            "capacity": self.resource_pool.get_capacity_info(),
        }


class LoadBalancer:
    """Load balancing coordinator for distributing tasks across resources."""

    def __init__(self, queue_managers: List[PriorityQueueManager]):
        self.queue_managers = queue_managers
        self.load_balancing_strategy = "round_robin"
        self.current_manager_index = 0

    def submit_task_with_balancing(
        self, task_id: str, job_id: str, priority: QueuePriority, **kwargs
    ) -> bool:
        """Submit a task using load balancing to distribute across managers."""
        if self.load_balancing_strategy == "round_robin":
            manager = self.queue_managers[self.current_manager_index]
            self.current_manager_index = (self.current_manager_index + 1) % len(
                self.queue_managers
            )

        elif self.load_balancing_strategy == "least_loaded":
            manager = min(
                self.queue_managers, key=lambda m: m.get_queue_status()["total_queued"]
            )

        elif self.load_balancing_strategy == "resource_aware":
            manager = self._select_manager_by_resources(
                kwargs.get("resource_requirements", [])
            )

        else:
            manager = self.queue_managers[0]  # Default to first manager

        return manager.submit_task(task_id, job_id, priority, **kwargs)

    def _select_manager_by_resources(
        self, requirements: List[ResourceRequirement]
    ) -> PriorityQueueManager:
        """Select manager based on resource availability."""
        best_manager = self.queue_managers[0]
        best_score = 0

        for manager in self.queue_managers:
            score = 0
            resource_status = manager.get_resource_status()

            for req in requirements:
                capacity_info = resource_status["capacity"].get(req.resource_type)
                if capacity_info and capacity_info["available"] >= req.amount:
                    score += capacity_info["available"] / capacity_info["total"]

            if score > best_score:
                best_score = score
                best_manager = manager

        return best_manager

    def get_overall_status(self) -> Dict[str, Any]:
        """Get status across all managed queue managers."""
        overall_status = {
            "total_managers": len(self.queue_managers),
            "total_queued": 0,
            "total_running": 0,
            "manager_status": [],
        }

        for i, manager in enumerate(self.queue_managers):
            status = manager.get_queue_status()
            overall_status["total_queued"] += status["total_queued"]
            overall_status["total_running"] += status["running_tasks"]
            overall_status["manager_status"].append({"manager_id": i, "status": status})

        return overall_status


# Example usage and demonstration
async def demo_priority_queue_system():
    """Demonstrate the priority queue management system functionality."""
    # Initialize resource pool
    resource_capacities = {
        ResourceType.CPU: ResourceCapacity(ResourceType.CPU, 8.0, 8.0, "cores"),
        ResourceType.MEMORY: ResourceCapacity(ResourceType.MEMORY, 16.0, 16.0, "GB"),
        ResourceType.GPU: ResourceCapacity(ResourceType.GPU, 2.0, 2.0, "units"),
    }

    resource_pool = ResourcePool(resource_capacities)

    # Initialize scheduling policy
    scheduling_policy = TaskSchedulingPolicy(SchedulingPolicy.PRIORITY_FIRST)

    # Initialize queue manager
    queue_manager = PriorityQueueManager(
        resource_pool=resource_pool,
        scheduling_policy=scheduling_policy,
        max_concurrent_tasks=5,
    )

    await queue_manager.start()

    try:
        # Submit various tasks with different priorities
        task_configs = [
            {"priority": QueuePriority.CRITICAL, "duration": 2},
            {"priority": QueuePriority.HIGH, "duration": 3},
            {"priority": QueuePriority.NORMAL, "duration": 4},
            {"priority": QueuePriority.LOW, "duration": 5},
            {"priority": QueuePriority.BACKGROUND, "duration": 6},
        ]

        submitted_tasks = []
        for i, config in enumerate(task_configs):
            task_id = f"task_{i + 1}"
            job_id = f"job_{(i % 3) + 1}"  # Distribute across 3 jobs

            resource_requirements = [
                ResourceRequirement(ResourceType.CPU, 1.0, "cores"),
                ResourceRequirement(ResourceType.MEMORY, 2.0, "GB"),
            ]

            success = queue_manager.submit_task(
                task_id=task_id,
                job_id=job_id,
                priority=config["priority"],
                resource_requirements=resource_requirements,
                estimated_duration=config["duration"],
            )

            if success:
                submitted_tasks.append(task_id)
                logger.info(
                    f"Submitted {task_id} with priority {config['priority'].name}"
                )

        # Monitor execution
        for _ in range(20):  # Monitor for 20 seconds
            status = queue_manager.get_queue_status()
            resource_status = queue_manager.get_resource_status()

            logger.info(
                f"Queue Status: {status['queue_lengths']}, Running: {status['running_tasks']}"
            )
            logger.info(f"Resource Utilization: {resource_status['utilization']}")

            await asyncio.sleep(1)

        # Get final metrics
        final_status = queue_manager.get_queue_status()
        logger.info("Final Metrics:")
        for key, value in final_status["metrics"].items():
            logger.info(f"  {key}: {value}")

    finally:
        await queue_manager.stop()


if __name__ == "__main__":
    # Run the demonstration
    asyncio.run(demo_priority_queue_system())
