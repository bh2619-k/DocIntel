"""
Module 7 - Orchestration & Operations Module

This module provides comprehensive orchestration and operations management
for the document intelligence pipeline system.

Components:
- job_state_machine.py: Workflow management with job state tracking
- checkpoint_recovery.py: State persistence and failure recovery
- priority_queue_manager.py: Task scheduling and resource allocation
- pipeline_sequencer.py: Pipeline stage coordination
- observability_monitor.py: Performance monitoring and SLA tracking
- model_drift_monitor.py: Model performance and drift detection
- active_learning_loop.py: Continuous improvement feedback
- deployment_configs.py: Infrastructure management

This orchestration module coordinates all previous modules (1-6) to provide:
- Complete workflow orchestration
- Robust error handling and recovery
- Efficient resource utilization
- Comprehensive monitoring and alerting
- Continuous model improvement
- Scalable deployment management
"""

from .job_state_machine import (
    JobStateMachine,
    JobState,
    TaskDefinition,
    WorkflowDefinition,
    Job,
)

from .checkpoint_recovery import (
    CheckpointManager,
    RecoveryManager,
    Checkpoint,
    CheckpointType,
    RecoveryStrategy,
)

from .priority_queue_manager import (
    PriorityQueueManager,
    ResourcePool,
    QueuePriority,
    TaskSchedulingPolicy,
    LoadBalancer,
)

__all__ = [
    # Job State Machine
    "JobStateMachine",
    "JobState",
    "TaskDefinition",
    "WorkflowDefinition",
    "Job",
    # Checkpoint Recovery
    "CheckpointManager",
    "RecoveryManager",
    "Checkpoint",
    "CheckpointType",
    "RecoveryStrategy",
    # Priority Queue Management
    "PriorityQueueManager",
    "ResourcePool",
    "QueuePriority",
    "TaskSchedulingPolicy",
    "LoadBalancer",
]

__version__ = "1.0.0"
__author__ = "Document Intelligence System"
__description__ = "Orchestration & Operations Module for Document Intelligence Pipeline"
