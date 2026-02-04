"""
Job State Machine for Document Intelligence Pipeline Orchestration

This module provides comprehensive workflow management with job state tracking,
task dependencies, and state transitions for the document intelligence pipeline.

Key Features:
- Workflow state machine with configurable transitions
- Task dependency resolution and execution ordering
- State persistence and recovery mechanisms
- Parallel execution coordination with resource limits
- Job priority and scheduling management
- Comprehensive logging and monitoring integration

Classes:
    JobState: Enumeration of all possible job states
    TaskDependency: Dependency relationship between tasks
    Job: Individual job with metadata and execution context
    WorkflowDefinition: Configurable workflow template
    JobStateMachine: Main orchestration engine for job execution

Dependencies:
    - asyncio: For asynchronous job execution
    - enum: For state enumeration
    - datetime: For timing and scheduling
    - typing: For type hints
    - dataclasses: For structured data
    - logging: For comprehensive logging
    - threading: For concurrent execution
    - queue: For job queuing mechanisms
"""

import asyncio
import logging
import threading
import time
import uuid
from collections import defaultdict, deque
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from queue import PriorityQueue, Queue
from typing import Any, Callable, Dict, List, Optional, Set, Union
import json
import os
from pathlib import Path


# Configure comprehensive logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class JobState(Enum):
    """Enumeration of all possible job states in the workflow."""

    PENDING = auto()
    QUEUED = auto()
    RUNNING = auto()
    PAUSED = auto()
    COMPLETED = auto()
    FAILED = auto()
    CANCELLED = auto()
    RETRYING = auto()
    TIMEOUT = auto()
    SCHEDULED = auto()


class JobPriority(Enum):
    """Job priority levels for scheduling."""

    CRITICAL = 1
    HIGH = 2
    NORMAL = 3
    LOW = 4
    BACKGROUND = 5


class TaskType(Enum):
    """Types of tasks in the document intelligence pipeline."""

    DOCUMENT_INGESTION = "document_ingestion"
    OCR_PROCESSING = "ocr_processing"
    NLP_ANALYSIS = "nlp_analysis"
    KNOWLEDGE_GRAPH_UPDATE = "knowledge_graph_update"
    ALERT_GENERATION = "alert_generation"
    WEBHOOK_NOTIFICATION = "webhook_notification"
    MODEL_TRAINING = "model_training"
    DRIFT_MONITORING = "drift_monitoring"


@dataclass
class TaskDependency:
    """Represents a dependency relationship between tasks."""

    task_id: str
    depends_on: str
    dependency_type: str = "completion"  # completion, data, resource
    wait_timeout: Optional[int] = None
    failure_handling: str = "fail"  # fail, skip, retry


@dataclass
class JobMetadata:
    """Metadata associated with a job."""

    job_id: str
    job_name: str
    job_type: str
    priority: JobPriority
    created_at: datetime
    updated_at: datetime
    owner: str
    tags: Dict[str, str] = field(default_factory=dict)
    timeout: Optional[int] = None
    retry_count: int = 0
    max_retries: int = 3


@dataclass
class TaskDefinition:
    """Definition of a task within a workflow."""

    task_id: str
    task_name: str
    task_type: TaskType
    handler: str  # Handler function name or class
    parameters: Dict[str, Any] = field(default_factory=dict)
    dependencies: List[TaskDependency] = field(default_factory=list)
    timeout: Optional[int] = None
    retry_policy: Dict[str, Any] = field(default_factory=dict)
    resource_requirements: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Job:
    """Represents a job in the orchestration system."""

    metadata: JobMetadata
    workflow_id: str
    tasks: List[TaskDefinition]
    state: JobState = JobState.PENDING
    current_task: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    results: Dict[str, Any] = field(default_factory=dict)
    execution_context: Dict[str, Any] = field(default_factory=dict)

    def __lt__(self, other):
        """Enable priority queue comparison based on job priority."""
        return self.metadata.priority.value < other.metadata.priority.value


@dataclass
class WorkflowDefinition:
    """Definition of a workflow template."""

    workflow_id: str
    workflow_name: str
    description: str
    tasks: List[TaskDefinition]
    default_parameters: Dict[str, Any] = field(default_factory=dict)
    parallel_execution: bool = False
    max_parallel_tasks: int = 5
    failure_policy: str = "stop"  # stop, continue, retry
    created_at: datetime = field(default_factory=datetime.now)


class TaskExecutor:
    """Handles the execution of individual tasks."""

    def __init__(self):
        self.handlers = {}
        self.register_default_handlers()

    def register_default_handlers(self):
        """Register default task handlers."""
        self.handlers.update(
            {
                TaskType.DOCUMENT_INGESTION.value: self._document_ingestion_handler,
                TaskType.OCR_PROCESSING.value: self._ocr_processing_handler,
                TaskType.NLP_ANALYSIS.value: self._nlp_analysis_handler,
                TaskType.KNOWLEDGE_GRAPH_UPDATE.value: self._kg_update_handler,
                TaskType.ALERT_GENERATION.value: self._alert_generation_handler,
                TaskType.WEBHOOK_NOTIFICATION.value: self._webhook_notification_handler,
                TaskType.MODEL_TRAINING.value: self._model_training_handler,
                TaskType.DRIFT_MONITORING.value: self._drift_monitoring_handler,
            }
        )

    def register_handler(self, task_type: str, handler: Callable):
        """Register a custom task handler."""
        self.handlers[task_type] = handler

    async def execute_task(
        self, task: TaskDefinition, context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute a single task and return results."""
        try:
            handler = self.handlers.get(task.task_type.value)
            if not handler:
                raise ValueError(f"No handler found for task type: {task.task_type}")

            logger.info(f"Executing task {task.task_id} of type {task.task_type}")

            # Apply timeout if specified
            if task.timeout:
                result = await asyncio.wait_for(
                    handler(task, context), timeout=task.timeout
                )
            else:
                result = await handler(task, context)

            logger.info(f"Task {task.task_id} completed successfully")
            return result

        except asyncio.TimeoutError:
            logger.error(f"Task {task.task_id} timed out")
            raise
        except Exception as e:
            logger.error(f"Task {task.task_id} failed: {str(e)}")
            raise

    # Default task handlers
    async def _document_ingestion_handler(
        self, task: TaskDefinition, context: Dict[str, Any]
    ):
        """Handle document ingestion tasks."""
        await asyncio.sleep(1)  # Simulate processing
        return {"status": "completed", "documents_processed": 1}

    async def _ocr_processing_handler(
        self, task: TaskDefinition, context: Dict[str, Any]
    ):
        """Handle OCR processing tasks."""
        await asyncio.sleep(2)  # Simulate processing
        return {"status": "completed", "text_extracted": True}

    async def _nlp_analysis_handler(
        self, task: TaskDefinition, context: Dict[str, Any]
    ):
        """Handle NLP analysis tasks."""
        await asyncio.sleep(1.5)  # Simulate processing
        return {"status": "completed", "entities_extracted": 5}

    async def _kg_update_handler(self, task: TaskDefinition, context: Dict[str, Any]):
        """Handle knowledge graph update tasks."""
        await asyncio.sleep(1)  # Simulate processing
        return {"status": "completed", "nodes_updated": 3}

    async def _alert_generation_handler(
        self, task: TaskDefinition, context: Dict[str, Any]
    ):
        """Handle alert generation tasks."""
        await asyncio.sleep(0.5)  # Simulate processing
        return {"status": "completed", "alerts_generated": 2}

    async def _webhook_notification_handler(
        self, task: TaskDefinition, context: Dict[str, Any]
    ):
        """Handle webhook notification tasks."""
        await asyncio.sleep(0.3)  # Simulate processing
        return {"status": "completed", "notifications_sent": 1}

    async def _model_training_handler(
        self, task: TaskDefinition, context: Dict[str, Any]
    ):
        """Handle model training tasks."""
        await asyncio.sleep(5)  # Simulate processing
        return {"status": "completed", "model_accuracy": 0.95}

    async def _drift_monitoring_handler(
        self, task: TaskDefinition, context: Dict[str, Any]
    ):
        """Handle drift monitoring tasks."""
        await asyncio.sleep(1)  # Simulate processing
        return {"status": "completed", "drift_detected": False}


class DependencyResolver:
    """Resolves task dependencies and determines execution order."""

    def __init__(self):
        self.dependency_graph = defaultdict(list)
        self.reverse_graph = defaultdict(list)

    def build_dependency_graph(self, tasks: List[TaskDefinition]):
        """Build dependency graph from task definitions."""
        self.dependency_graph.clear()
        self.reverse_graph.clear()

        for task in tasks:
            for dependency in task.dependencies:
                self.dependency_graph[dependency.depends_on].append(task.task_id)
                self.reverse_graph[task.task_id].append(dependency.depends_on)

    def get_execution_order(self, tasks: List[TaskDefinition]) -> List[List[str]]:
        """Get topologically sorted execution order with parallelization."""
        self.build_dependency_graph(tasks)

        # Track in-degree for each task
        in_degree = defaultdict(int)
        all_tasks = {task.task_id for task in tasks}

        for task_id in all_tasks:
            in_degree[task_id] = len(self.reverse_graph[task_id])

        execution_levels = []
        queue = deque([task_id for task_id in all_tasks if in_degree[task_id] == 0])

        while queue:
            current_level = []
            level_size = len(queue)

            for _ in range(level_size):
                task_id = queue.popleft()
                current_level.append(task_id)

                # Reduce in-degree for dependent tasks
                for dependent in self.dependency_graph[task_id]:
                    in_degree[dependent] -= 1
                    if in_degree[dependent] == 0:
                        queue.append(dependent)

            if current_level:
                execution_levels.append(current_level)

        return execution_levels

    def check_circular_dependencies(self, tasks: List[TaskDefinition]) -> bool:
        """Check for circular dependencies in the task graph."""
        visited = set()
        rec_stack = set()

        def has_cycle(task_id: str) -> bool:
            if task_id in rec_stack:
                return True
            if task_id in visited:
                return False

            visited.add(task_id)
            rec_stack.add(task_id)

            for dependent in self.dependency_graph[task_id]:
                if has_cycle(dependent):
                    return True

            rec_stack.remove(task_id)
            return False

        self.build_dependency_graph(tasks)
        all_tasks = {task.task_id for task in tasks}

        for task_id in all_tasks:
            if task_id not in visited and has_cycle(task_id):
                return True

        return False


class JobStateMachine:
    """
    Main orchestration engine for job execution with state management,
    dependency resolution, and parallel execution coordination.
    """

    def __init__(
        self, max_concurrent_jobs: int = 10, checkpoint_dir: str = "checkpoints"
    ):
        self.max_concurrent_jobs = max_concurrent_jobs
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)

        # Core components
        self.job_queue = PriorityQueue()
        self.active_jobs = {}
        self.completed_jobs = {}
        self.workflows = {}
        self.job_states = {}

        # Execution components
        self.task_executor = TaskExecutor()
        self.dependency_resolver = DependencyResolver()

        # Synchronization
        self.state_lock = threading.RLock()
        self.execution_semaphore = asyncio.Semaphore(max_concurrent_jobs)

        # Monitoring
        self.metrics = {
            "jobs_submitted": 0,
            "jobs_completed": 0,
            "jobs_failed": 0,
            "average_execution_time": 0,
            "total_execution_time": 0,
        }

        # Background tasks
        self._running = False
        self._background_tasks = []

    async def start(self):
        """Start the job state machine."""
        self._running = True
        logger.info("Starting Job State Machine")

        # Start background monitoring tasks
        self._background_tasks = [
            asyncio.create_task(self._job_processor()),
            asyncio.create_task(self._state_monitor()),
            asyncio.create_task(self._checkpoint_manager()),
        ]

    async def stop(self):
        """Stop the job state machine."""
        self._running = False
        logger.info("Stopping Job State Machine")

        # Cancel background tasks
        for task in self._background_tasks:
            task.cancel()

        # Wait for active jobs to complete
        await self._wait_for_active_jobs()

    def register_workflow(self, workflow: WorkflowDefinition):
        """Register a workflow definition."""
        with self.state_lock:
            self.workflows[workflow.workflow_id] = workflow
            logger.info(f"Registered workflow: {workflow.workflow_name}")

    def submit_job(
        self,
        workflow_id: str,
        job_name: str,
        parameters: Dict[str, Any] = None,
        priority: JobPriority = JobPriority.NORMAL,
        owner: str = "system",
    ) -> str:
        """Submit a job for execution."""
        if workflow_id not in self.workflows:
            raise ValueError(f"Unknown workflow: {workflow_id}")

        workflow = self.workflows[workflow_id]
        job_id = str(uuid.uuid4())

        # Create job metadata
        metadata = JobMetadata(
            job_id=job_id,
            job_name=job_name,
            job_type=workflow.workflow_name,
            priority=priority,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            owner=owner,
        )

        # Create job with workflow tasks
        job = Job(
            metadata=metadata,
            workflow_id=workflow_id,
            tasks=workflow.tasks.copy(),
            execution_context=parameters or {},
        )

        with self.state_lock:
            self.job_queue.put(job)
            self.job_states[job_id] = JobState.QUEUED
            self.metrics["jobs_submitted"] += 1

        logger.info(f"Submitted job {job_id}: {job_name}")
        return job_id

    async def _job_processor(self):
        """Background task to process jobs from the queue."""
        while self._running:
            try:
                if (
                    not self.job_queue.empty()
                    and len(self.active_jobs) < self.max_concurrent_jobs
                ):
                    job = self.job_queue.get_nowait()

                    async with self.execution_semaphore:
                        asyncio.create_task(self._execute_job(job))

                await asyncio.sleep(0.1)  # Small delay to prevent busy waiting

            except Exception as e:
                logger.error(f"Error in job processor: {str(e)}")
                await asyncio.sleep(1)

    async def _execute_job(self, job: Job):
        """Execute a single job with all its tasks."""
        job_id = job.metadata.job_id

        try:
            with self.state_lock:
                self.active_jobs[job_id] = job
                self._update_job_state(job, JobState.RUNNING)

            job.started_at = datetime.now()
            logger.info(f"Starting execution of job {job_id}")

            # Check for circular dependencies
            if self.dependency_resolver.check_circular_dependencies(job.tasks):
                raise ValueError("Circular dependency detected in workflow")

            # Get execution order
            execution_levels = self.dependency_resolver.get_execution_order(job.tasks)

            # Execute tasks level by level
            for level in execution_levels:
                await self._execute_task_level(job, level)

            # Job completed successfully
            job.completed_at = datetime.now()
            self._update_job_state(job, JobState.COMPLETED)
            self._update_metrics(job, success=True)

        except Exception as e:
            logger.error(f"Job {job_id} failed: {str(e)}")
            job.error_message = str(e)
            self._update_job_state(job, JobState.FAILED)
            self._update_metrics(job, success=False)

        finally:
            with self.state_lock:
                if job_id in self.active_jobs:
                    completed_job = self.active_jobs.pop(job_id)
                    self.completed_jobs[job_id] = completed_job

            # Save checkpoint
            await self._save_job_checkpoint(job)

    async def _execute_task_level(self, job: Job, task_ids: List[str]):
        """Execute a level of tasks that can run in parallel."""
        tasks_to_run = [task for task in job.tasks if task.task_id in task_ids]

        if not tasks_to_run:
            return

        # Create coroutines for all tasks in this level
        task_coroutines = []
        for task in tasks_to_run:
            coroutine = self._execute_single_task(job, task)
            task_coroutines.append(coroutine)

        # Execute tasks in parallel
        try:
            results = await asyncio.gather(*task_coroutines, return_exceptions=True)

            # Process results
            for task, result in zip(tasks_to_run, results):
                if isinstance(result, Exception):
                    raise result
                job.results[task.task_id] = result

        except Exception as e:
            logger.error(f"Task level execution failed: {str(e)}")
            raise

    async def _execute_single_task(
        self, job: Job, task: TaskDefinition
    ) -> Dict[str, Any]:
        """Execute a single task within a job."""
        job.current_task = task.task_id
        logger.info(f"Executing task {task.task_id} for job {job.metadata.job_id}")

        try:
            # Check dependencies are satisfied
            await self._wait_for_dependencies(job, task)

            # Execute the task
            result = await self.task_executor.execute_task(task, job.execution_context)

            logger.info(f"Task {task.task_id} completed for job {job.metadata.job_id}")
            return result

        except Exception as e:
            logger.error(
                f"Task {task.task_id} failed for job {job.metadata.job_id}: {str(e)}"
            )

            # Handle retry logic
            if (
                task.retry_policy
                and job.metadata.retry_count < job.metadata.max_retries
            ):
                await self._retry_task(job, task, e)
            else:
                raise

    async def _wait_for_dependencies(self, job: Job, task: TaskDefinition):
        """Wait for task dependencies to be satisfied."""
        for dependency in task.dependencies:
            depends_on = dependency.depends_on

            # Check if dependency is completed
            if depends_on not in job.results:
                timeout = dependency.wait_timeout or 300  # 5 minutes default
                start_time = time.time()

                while (
                    depends_on not in job.results
                    and (time.time() - start_time) < timeout
                ):
                    await asyncio.sleep(0.5)

                if depends_on not in job.results:
                    if dependency.failure_handling == "skip":
                        logger.warning(
                            f"Skipping task {task.task_id} due to missing dependency {depends_on}"
                        )
                        return
                    else:
                        raise TimeoutError(
                            f"Dependency {depends_on} not satisfied within timeout"
                        )

    async def _retry_task(self, job: Job, task: TaskDefinition, error: Exception):
        """Retry a failed task based on retry policy."""
        job.metadata.retry_count += 1
        retry_delay = task.retry_policy.get("delay", 1)

        logger.info(
            f"Retrying task {task.task_id} (attempt {job.metadata.retry_count})"
        )
        await asyncio.sleep(retry_delay)

        # Re-execute the task
        return await self._execute_single_task(job, task)

    def _update_job_state(self, job: Job, new_state: JobState):
        """Update job state and metadata."""
        job.state = new_state
        job.metadata.updated_at = datetime.now()
        self.job_states[job.metadata.job_id] = new_state

        logger.info(f"Job {job.metadata.job_id} state changed to {new_state.name}")

    def _update_metrics(self, job: Job, success: bool):
        """Update execution metrics."""
        if success:
            self.metrics["jobs_completed"] += 1
        else:
            self.metrics["jobs_failed"] += 1

        if job.started_at and job.completed_at:
            execution_time = (job.completed_at - job.started_at).total_seconds()
            total_time = self.metrics["total_execution_time"] + execution_time
            completed_jobs = self.metrics["jobs_completed"]

            self.metrics["total_execution_time"] = total_time
            if completed_jobs > 0:
                self.metrics["average_execution_time"] = total_time / completed_jobs

    async def _state_monitor(self):
        """Monitor job states and handle timeouts."""
        while self._running:
            try:
                current_time = datetime.now()

                with self.state_lock:
                    for job_id, job in list(self.active_jobs.items()):
                        # Check for job timeout
                        if job.metadata.timeout:
                            elapsed = (
                                (current_time - job.started_at).total_seconds()
                                if job.started_at
                                else 0
                            )
                            if elapsed > job.metadata.timeout:
                                logger.warning(f"Job {job_id} timed out")
                                self._update_job_state(job, JobState.TIMEOUT)

                await asyncio.sleep(5)  # Check every 5 seconds

            except Exception as e:
                logger.error(f"Error in state monitor: {str(e)}")
                await asyncio.sleep(5)

    async def _checkpoint_manager(self):
        """Manage job checkpoints for recovery."""
        while self._running:
            try:
                # Save checkpoints for active jobs
                with self.state_lock:
                    for job in self.active_jobs.values():
                        await self._save_job_checkpoint(job)

                await asyncio.sleep(30)  # Checkpoint every 30 seconds

            except Exception as e:
                logger.error(f"Error in checkpoint manager: {str(e)}")
                await asyncio.sleep(30)

    async def _save_job_checkpoint(self, job: Job):
        """Save job checkpoint to disk."""
        try:
            checkpoint_file = self.checkpoint_dir / f"{job.metadata.job_id}.json"

            checkpoint_data = {
                "job": asdict(job),
                "timestamp": datetime.now().isoformat(),
            }

            with open(checkpoint_file, "w") as f:
                json.dump(checkpoint_data, f, indent=2, default=str)

        except Exception as e:
            logger.error(
                f"Failed to save checkpoint for job {job.metadata.job_id}: {str(e)}"
            )

    async def _wait_for_active_jobs(self):
        """Wait for all active jobs to complete."""
        while self.active_jobs:
            logger.info(f"Waiting for {len(self.active_jobs)} active jobs to complete")
            await asyncio.sleep(1)

    # Public API methods
    def get_job_status(self, job_id: str) -> Optional[JobState]:
        """Get the current status of a job."""
        return self.job_states.get(job_id)

    def get_job_details(self, job_id: str) -> Optional[Job]:
        """Get detailed information about a job."""
        with self.state_lock:
            return self.active_jobs.get(job_id) or self.completed_jobs.get(job_id)

    def cancel_job(self, job_id: str) -> bool:
        """Cancel a running job."""
        with self.state_lock:
            if job_id in self.active_jobs:
                job = self.active_jobs[job_id]
                self._update_job_state(job, JobState.CANCELLED)
                return True
        return False

    def get_metrics(self) -> Dict[str, Any]:
        """Get execution metrics."""
        with self.state_lock:
            return self.metrics.copy()

    def list_workflows(self) -> List[WorkflowDefinition]:
        """List all registered workflows."""
        with self.state_lock:
            return list(self.workflows.values())


# Example usage and demonstration
async def demo_job_state_machine():
    """Demonstrate the job state machine functionality."""
    # Initialize the state machine
    state_machine = JobStateMachine(max_concurrent_jobs=5)

    # Create a sample workflow
    workflow = WorkflowDefinition(
        workflow_id="document_processing",
        workflow_name="Document Processing Pipeline",
        description="Complete document processing workflow",
        tasks=[
            TaskDefinition(
                task_id="ingest",
                task_name="Document Ingestion",
                task_type=TaskType.DOCUMENT_INGESTION,
                handler="document_ingestion_handler",
            ),
            TaskDefinition(
                task_id="ocr",
                task_name="OCR Processing",
                task_type=TaskType.OCR_PROCESSING,
                handler="ocr_processing_handler",
                dependencies=[TaskDependency(task_id="ocr", depends_on="ingest")],
            ),
            TaskDefinition(
                task_id="nlp",
                task_name="NLP Analysis",
                task_type=TaskType.NLP_ANALYSIS,
                handler="nlp_analysis_handler",
                dependencies=[TaskDependency(task_id="nlp", depends_on="ocr")],
            ),
            TaskDefinition(
                task_id="kg_update",
                task_name="Knowledge Graph Update",
                task_type=TaskType.KNOWLEDGE_GRAPH_UPDATE,
                handler="kg_update_handler",
                dependencies=[TaskDependency(task_id="kg_update", depends_on="nlp")],
            ),
        ],
    )

    # Register workflow and start state machine
    state_machine.register_workflow(workflow)
    await state_machine.start()

    try:
        # Submit multiple jobs
        job_ids = []
        for i in range(3):
            job_id = state_machine.submit_job(
                workflow_id="document_processing",
                job_name=f"Document Processing Job {i + 1}",
                priority=JobPriority.HIGH if i == 0 else JobPriority.NORMAL,
            )
            job_ids.append(job_id)
            logger.info(f"Submitted job {job_id}")

        # Monitor job execution
        while True:
            all_completed = True
            for job_id in job_ids:
                status = state_machine.get_job_status(job_id)
                if status not in [
                    JobState.COMPLETED,
                    JobState.FAILED,
                    JobState.CANCELLED,
                ]:
                    all_completed = False
                logger.info(f"Job {job_id}: {status.name if status else 'UNKNOWN'}")

            if all_completed:
                break

            await asyncio.sleep(2)

        # Print final metrics
        metrics = state_machine.get_metrics()
        logger.info("Final Metrics:")
        for key, value in metrics.items():
            logger.info(f"  {key}: {value}")

    finally:
        await state_machine.stop()


if __name__ == "__main__":
    # Run the demonstration
    asyncio.run(demo_job_state_machine())
