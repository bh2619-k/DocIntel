"""
Real-Time Document Intelligence & Compliance Engine
Model Versioning Interface Module
Version: 1.0
Date: 2026-02-03

This module provides comprehensive model versioning system with A/B testing capabilities,
rollback mechanisms, performance monitoring, model comparison, and automated deployment.
"""

import logging
import uuid
import asyncio
from typing import Dict, List, Any, Optional, Tuple, Callable
from datetime import datetime, timedelta
from enum import Enum
import json
from abc import ABC, abstractmethod
import hashlib
import pickle

logger = logging.getLogger(__name__)


class ModelType(Enum):
    """Types of models in the system"""

    BERT_ENCODER = "BERT_ENCODER"
    NER_MODEL = "NER_MODEL"
    T5_EXTRACTOR = "T5_EXTRACTOR"
    EMBEDDING_GENERATOR = "EMBEDDING_GENERATOR"
    ENSEMBLE = "ENSEMBLE"


class VersionStatus(Enum):
    """Version deployment status"""

    DEVELOPMENT = "DEVELOPMENT"  # In development
    TESTING = "TESTING"  # Under testing
    STAGING = "STAGING"  # In staging environment
    PRODUCTION = "PRODUCTION"  # In production
    DEPRECATED = "DEPRECATED"  # Deprecated/retired
    ROLLBACK = "ROLLBACK"  # Rolled back


class DeploymentStrategy(Enum):
    """Model deployment strategies"""

    BLUE_GREEN = "BLUE_GREEN"  # Blue-green deployment
    CANARY = "CANARY"  # Canary deployment
    ROLLING = "ROLLING"  # Rolling deployment
    A_B_TEST = "A_B_TEST"  # A/B testing deployment
    IMMEDIATE = "IMMEDIATE"  # Immediate replacement


class PerformanceMetric(Enum):
    """Performance metrics for model evaluation"""

    ACCURACY = "ACCURACY"
    PRECISION = "PRECISION"
    RECALL = "RECALL"
    F1_SCORE = "F1_SCORE"
    LATENCY = "LATENCY"
    THROUGHPUT = "THROUGHPUT"
    MEMORY_USAGE = "MEMORY_USAGE"
    ERROR_RATE = "ERROR_RATE"
    CONFIDENCE_SCORE = "CONFIDENCE_SCORE"


class ModelVersion:
    """Represents a specific version of a model"""

    def __init__(
        self,
        version_id: str,
        model_type: ModelType,
        version_number: str,
        model_artifact_path: str,
        metadata: Optional[Dict[str, Any]] = None,
        configuration: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize model version

        Args:
            version_id: Unique version identifier
            model_type: Type of model
            version_number: Version number (e.g., "1.2.3")
            model_artifact_path: Path to model artifacts
            metadata: Additional metadata
            configuration: Model configuration
        """
        self.version_id = version_id
        self.model_type = model_type
        self.version_number = version_number
        self.model_artifact_path = model_artifact_path
        self.metadata = metadata or {}
        self.configuration = configuration or {}

        # Status tracking
        self.status = VersionStatus.DEVELOPMENT
        self.created_at = datetime.now()
        self.deployed_at = None
        self.rollback_at = None

        # Performance tracking
        self.performance_metrics = {}
        self.test_results = {}
        self.production_metrics = {}

        # Deployment info
        self.deployment_strategy = None
        self.traffic_percentage = 0.0
        self.parent_version_id = None
        self.child_version_ids = []

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return {
            "version_id": self.version_id,
            "model_type": self.model_type.value,
            "version_number": self.version_number,
            "model_artifact_path": self.model_artifact_path,
            "metadata": self.metadata,
            "configuration": self.configuration,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "deployed_at": self.deployed_at.isoformat() if self.deployed_at else None,
            "rollback_at": self.rollback_at.isoformat() if self.rollback_at else None,
            "performance_metrics": self.performance_metrics,
            "test_results": self.test_results,
            "production_metrics": self.production_metrics,
            "deployment_strategy": self.deployment_strategy.value
            if self.deployment_strategy
            else None,
            "traffic_percentage": self.traffic_percentage,
            "parent_version_id": self.parent_version_id,
            "child_version_ids": self.child_version_ids,
        }


class ABTestExperiment:
    """Represents an A/B testing experiment"""

    def __init__(
        self,
        experiment_id: str,
        experiment_name: str,
        control_version_id: str,
        treatment_version_id: str,
        traffic_split: Tuple[float, float] = (0.5, 0.5),
        success_metrics: List[PerformanceMetric] = None,
        duration_hours: int = 24,
    ):
        """
        Initialize A/B test experiment

        Args:
            experiment_id: Unique experiment identifier
            experiment_name: Human-readable experiment name
            control_version_id: Control version (baseline)
            treatment_version_id: Treatment version (new version)
            traffic_split: Traffic split between control and treatment
            success_metrics: Metrics to evaluate success
            duration_hours: Experiment duration in hours
        """
        self.experiment_id = experiment_id
        self.experiment_name = experiment_name
        self.control_version_id = control_version_id
        self.treatment_version_id = treatment_version_id
        self.traffic_split = traffic_split
        self.success_metrics = success_metrics or [PerformanceMetric.ACCURACY]
        self.duration_hours = duration_hours

        # Status tracking
        self.status = "PLANNED"
        self.start_time = None
        self.end_time = None
        self.created_at = datetime.now()

        # Results tracking
        self.control_metrics = {}
        self.treatment_metrics = {}
        self.statistical_significance = {}
        self.recommendation = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return {
            "experiment_id": self.experiment_id,
            "experiment_name": self.experiment_name,
            "control_version_id": self.control_version_id,
            "treatment_version_id": self.treatment_version_id,
            "traffic_split": self.traffic_split,
            "success_metrics": [metric.value for metric in self.success_metrics],
            "duration_hours": self.duration_hours,
            "status": self.status,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "created_at": self.created_at.isoformat(),
            "control_metrics": self.control_metrics,
            "treatment_metrics": self.treatment_metrics,
            "statistical_significance": self.statistical_significance,
            "recommendation": self.recommendation,
        }


class ModelVersioningInterface:
    """
    Comprehensive model versioning system with A/B testing and deployment management
    """

    def __init__(
        self,
        storage_backend: str = "local",
        enable_auto_rollback: bool = True,
        max_versions_per_model: int = 10,
        performance_threshold: Dict[str, float] = None,
        rollback_conditions: Dict[str, Any] = None,
    ):
        """
        Initialize model versioning interface

        Args:
            storage_backend: Storage backend for model artifacts
            enable_auto_rollback: Enable automatic rollback on performance degradation
            max_versions_per_model: Maximum versions to keep per model type
            performance_threshold: Performance thresholds for deployments
            rollback_conditions: Conditions that trigger automatic rollback
        """
        self.storage_backend = storage_backend
        self.enable_auto_rollback = enable_auto_rollback
        self.max_versions_per_model = max_versions_per_model
        self.performance_threshold = performance_threshold or {
            "accuracy": 0.85,
            "latency_ms": 1000,
            "error_rate": 0.05,
        }
        self.rollback_conditions = rollback_conditions or {
            "error_rate_threshold": 0.1,
            "latency_threshold_ms": 2000,
            "min_accuracy": 0.8,
        }

        # Version registry
        self.versions: Dict[str, ModelVersion] = {}
        self.model_type_versions: Dict[ModelType, List[str]] = {
            model_type: [] for model_type in ModelType
        }

        # Active deployments
        self.active_deployments: Dict[ModelType, str] = {}  # model_type -> version_id
        self.deployment_history: List[Dict[str, Any]] = []

        # A/B testing
        self.active_experiments: Dict[str, ABTestExperiment] = {}
        self.experiment_history: List[Dict[str, Any]] = []

        # Performance monitoring
        self.performance_monitor = PerformanceMonitor()
        self.deployment_manager = DeploymentManager()

        # Callbacks for model lifecycle events
        self.callbacks: Dict[str, List[Callable]] = {
            "on_version_created": [],
            "on_deployment": [],
            "on_rollback": [],
            "on_experiment_started": [],
            "on_experiment_completed": [],
        }

        logger.info("Model versioning interface initialized")

    def create_version(
        self,
        model_type: ModelType,
        version_number: str,
        model_artifact_path: str,
        metadata: Optional[Dict[str, Any]] = None,
        configuration: Optional[Dict[str, Any]] = None,
        parent_version_id: Optional[str] = None,
    ) -> str:
        """
        Create a new model version

        Args:
            model_type: Type of model
            version_number: Version number
            model_artifact_path: Path to model artifacts
            metadata: Additional metadata
            configuration: Model configuration
            parent_version_id: Parent version for lineage tracking

        Returns:
            Version ID of the created version
        """
        try:
            # Generate unique version ID
            version_id = str(uuid.uuid4())

            # Create model version
            version = ModelVersion(
                version_id=version_id,
                model_type=model_type,
                version_number=version_number,
                model_artifact_path=model_artifact_path,
                metadata=metadata or {},
                configuration=configuration or {},
            )

            # Set parent relationship
            if parent_version_id and parent_version_id in self.versions:
                version.parent_version_id = parent_version_id
                self.versions[parent_version_id].child_version_ids.append(version_id)

            # Store version
            self.versions[version_id] = version
            self.model_type_versions[model_type].append(version_id)

            # Cleanup old versions if needed
            self._cleanup_old_versions(model_type)

            # Trigger callbacks
            self._trigger_callbacks("on_version_created", version)

            logger.info(
                f"Created version {version_number} for {model_type.value}: {version_id}"
            )

            return version_id

        except Exception as e:
            logger.error(f"Failed to create model version: {e}")
            raise

    def deploy_version(
        self,
        version_id: str,
        strategy: DeploymentStrategy = DeploymentStrategy.BLUE_GREEN,
        traffic_percentage: float = 100.0,
        force_deploy: bool = False,
    ) -> Dict[str, Any]:
        """
        Deploy a model version

        Args:
            version_id: Version ID to deploy
            strategy: Deployment strategy
            traffic_percentage: Percentage of traffic to route to new version
            force_deploy: Force deployment even if performance checks fail

        Returns:
            Deployment result
        """
        try:
            if version_id not in self.versions:
                raise ValueError(f"Version {version_id} not found")

            version = self.versions[version_id]

            # Pre-deployment validation
            if not force_deploy:
                validation_result = self._validate_deployment(version)
                if not validation_result["valid"]:
                    return {
                        "status": "failed",
                        "error": "Deployment validation failed",
                        "validation_result": validation_result,
                    }

            # Execute deployment
            deployment_result = self.deployment_manager.deploy_version(
                version, strategy, traffic_percentage
            )

            if deployment_result["status"] == "success":
                # Update version status
                version.status = VersionStatus.PRODUCTION
                version.deployed_at = datetime.now()
                version.deployment_strategy = strategy
                version.traffic_percentage = traffic_percentage

                # Update active deployment
                self.active_deployments[version.model_type] = version_id

                # Record deployment history
                self.deployment_history.append(
                    {
                        "timestamp": datetime.now().isoformat(),
                        "version_id": version_id,
                        "model_type": version.model_type.value,
                        "strategy": strategy.value,
                        "traffic_percentage": traffic_percentage,
                        "status": "deployed",
                    }
                )

                # Start performance monitoring
                self.performance_monitor.start_monitoring(version_id)

                # Trigger callbacks
                self._trigger_callbacks("on_deployment", version)

                logger.info(f"Successfully deployed version {version_id}")

            return deployment_result

        except Exception as e:
            logger.error(f"Deployment failed for version {version_id}: {e}")
            return {"status": "failed", "error": str(e), "version_id": version_id}

    def rollback_version(
        self,
        model_type: ModelType,
        target_version_id: Optional[str] = None,
        reason: str = "Manual rollback",
    ) -> Dict[str, Any]:
        """
        Rollback to a previous version

        Args:
            model_type: Type of model to rollback
            target_version_id: Specific version to rollback to (if None, uses previous stable)
            reason: Reason for rollback

        Returns:
            Rollback result
        """
        try:
            current_version_id = self.active_deployments.get(model_type)
            if not current_version_id:
                return {
                    "status": "failed",
                    "error": f"No active deployment for {model_type.value}",
                }

            # Determine target version
            if target_version_id is None:
                target_version_id = self._get_previous_stable_version(
                    model_type, current_version_id
                )

            if target_version_id is None:
                return {
                    "status": "failed",
                    "error": "No stable version available for rollback",
                }

            target_version = self.versions[target_version_id]

            # Execute rollback
            rollback_result = self.deployment_manager.rollback_to_version(
                current_version_id, target_version_id
            )

            if rollback_result["status"] == "success":
                # Update version statuses
                current_version = self.versions[current_version_id]
                current_version.status = VersionStatus.ROLLBACK
                current_version.rollback_at = datetime.now()

                target_version.status = VersionStatus.PRODUCTION
                target_version.traffic_percentage = 100.0

                # Update active deployment
                self.active_deployments[model_type] = target_version_id

                # Record rollback history
                self.deployment_history.append(
                    {
                        "timestamp": datetime.now().isoformat(),
                        "version_id": current_version_id,
                        "target_version_id": target_version_id,
                        "model_type": model_type.value,
                        "action": "rollback",
                        "reason": reason,
                        "status": "completed",
                    }
                )

                # Trigger callbacks
                self._trigger_callbacks(
                    "on_rollback",
                    {
                        "from_version": current_version,
                        "to_version": target_version,
                        "reason": reason,
                    },
                )

                logger.info(
                    f"Successfully rolled back {model_type.value} from {current_version_id} to {target_version_id}"
                )

            return rollback_result

        except Exception as e:
            logger.error(f"Rollback failed for {model_type.value}: {e}")
            return {"status": "failed", "error": str(e)}

    def create_ab_test(
        self,
        experiment_name: str,
        control_version_id: str,
        treatment_version_id: str,
        traffic_split: Tuple[float, float] = (0.5, 0.5),
        success_metrics: List[PerformanceMetric] = None,
        duration_hours: int = 24,
    ) -> str:
        """
        Create an A/B testing experiment

        Args:
            experiment_name: Name of the experiment
            control_version_id: Control version (baseline)
            treatment_version_id: Treatment version (new version)
            traffic_split: Traffic split between control and treatment
            success_metrics: Metrics to evaluate success
            duration_hours: Experiment duration

        Returns:
            Experiment ID
        """
        try:
            # Validate versions
            if control_version_id not in self.versions:
                raise ValueError(f"Control version {control_version_id} not found")
            if treatment_version_id not in self.versions:
                raise ValueError(f"Treatment version {treatment_version_id} not found")

            # Ensure both versions are of the same model type
            control_type = self.versions[control_version_id].model_type
            treatment_type = self.versions[treatment_version_id].model_type
            if control_type != treatment_type:
                raise ValueError(
                    "Control and treatment versions must be of the same model type"
                )

            # Create experiment
            experiment_id = str(uuid.uuid4())
            experiment = ABTestExperiment(
                experiment_id=experiment_id,
                experiment_name=experiment_name,
                control_version_id=control_version_id,
                treatment_version_id=treatment_version_id,
                traffic_split=traffic_split,
                success_metrics=success_metrics,
                duration_hours=duration_hours,
            )

            # Store experiment
            self.active_experiments[experiment_id] = experiment

            logger.info(
                f"Created A/B test experiment: {experiment_name} ({experiment_id})"
            )

            return experiment_id

        except Exception as e:
            logger.error(f"Failed to create A/B test: {e}")
            raise

    async def start_ab_test(self, experiment_id: str) -> Dict[str, Any]:
        """
        Start an A/B testing experiment

        Args:
            experiment_id: Experiment ID to start

        Returns:
            Start result
        """
        try:
            if experiment_id not in self.active_experiments:
                raise ValueError(f"Experiment {experiment_id} not found")

            experiment = self.active_experiments[experiment_id]

            # Deploy both versions with traffic split
            control_deploy = self.deploy_version(
                experiment.control_version_id,
                DeploymentStrategy.A_B_TEST,
                experiment.traffic_split[0] * 100,
            )

            treatment_deploy = self.deploy_version(
                experiment.treatment_version_id,
                DeploymentStrategy.A_B_TEST,
                experiment.traffic_split[1] * 100,
            )

            if (
                control_deploy["status"] == "success"
                and treatment_deploy["status"] == "success"
            ):
                # Update experiment status
                experiment.status = "RUNNING"
                experiment.start_time = datetime.now()
                experiment.end_time = experiment.start_time + timedelta(
                    hours=experiment.duration_hours
                )

                # Start monitoring
                self.performance_monitor.start_ab_test_monitoring(experiment)

                # Schedule automatic completion
                asyncio.create_task(self._auto_complete_experiment(experiment_id))

                # Trigger callbacks
                self._trigger_callbacks("on_experiment_started", experiment)

                logger.info(f"Started A/B test experiment {experiment_id}")

                return {
                    "status": "success",
                    "experiment_id": experiment_id,
                    "start_time": experiment.start_time.isoformat(),
                    "end_time": experiment.end_time.isoformat(),
                }
            else:
                return {
                    "status": "failed",
                    "error": "Failed to deploy versions for A/B test",
                    "control_deploy": control_deploy,
                    "treatment_deploy": treatment_deploy,
                }

        except Exception as e:
            logger.error(f"Failed to start A/B test {experiment_id}: {e}")
            return {"status": "failed", "error": str(e)}

    async def complete_ab_test(
        self, experiment_id: str, force_complete: bool = False
    ) -> Dict[str, Any]:
        """
        Complete an A/B testing experiment and analyze results

        Args:
            experiment_id: Experiment ID to complete
            force_complete: Force completion even if duration hasn't elapsed

        Returns:
            Completion result with recommendations
        """
        try:
            if experiment_id not in self.active_experiments:
                raise ValueError(f"Experiment {experiment_id} not found")

            experiment = self.active_experiments[experiment_id]

            if experiment.status != "RUNNING":
                return {
                    "status": "failed",
                    "error": f"Experiment is not running (status: {experiment.status})",
                }

            # Check if experiment duration has elapsed
            if not force_complete and datetime.now() < experiment.end_time:
                return {
                    "status": "failed",
                    "error": "Experiment duration has not elapsed",
                }

            # Collect performance metrics
            control_metrics = self.performance_monitor.get_version_metrics(
                experiment.control_version_id
            )
            treatment_metrics = self.performance_monitor.get_version_metrics(
                experiment.treatment_version_id
            )

            # Analyze results
            analysis_result = self._analyze_ab_test_results(
                control_metrics, treatment_metrics, experiment.success_metrics
            )

            # Update experiment
            experiment.status = "COMPLETED"
            experiment.control_metrics = control_metrics
            experiment.treatment_metrics = treatment_metrics
            experiment.statistical_significance = analysis_result[
                "statistical_significance"
            ]
            experiment.recommendation = analysis_result["recommendation"]

            # Move to history
            self.experiment_history.append(experiment.to_dict())
            del self.active_experiments[experiment_id]

            # Execute recommendation
            if analysis_result["recommendation"] == "deploy_treatment":
                # Deploy treatment version fully
                self.deploy_version(
                    experiment.treatment_version_id, traffic_percentage=100.0
                )
                # Deprecate control version
                self.versions[
                    experiment.control_version_id
                ].status = VersionStatus.DEPRECATED
            elif analysis_result["recommendation"] == "keep_control":
                # Keep control version, deprecate treatment
                self.deploy_version(
                    experiment.control_version_id, traffic_percentage=100.0
                )
                self.versions[
                    experiment.treatment_version_id
                ].status = VersionStatus.DEPRECATED

            # Trigger callbacks
            self._trigger_callbacks("on_experiment_completed", experiment)

            logger.info(
                f"Completed A/B test experiment {experiment_id}: {analysis_result['recommendation']}"
            )

            return {
                "status": "success",
                "experiment_id": experiment_id,
                "recommendation": analysis_result["recommendation"],
                "control_metrics": control_metrics,
                "treatment_metrics": treatment_metrics,
                "statistical_significance": analysis_result["statistical_significance"],
            }

        except Exception as e:
            logger.error(f"Failed to complete A/B test {experiment_id}: {e}")
            return {"status": "failed", "error": str(e)}

    def compare_versions(
        self,
        version_id1: str,
        version_id2: str,
        metrics: List[PerformanceMetric] = None,
    ) -> Dict[str, Any]:
        """
        Compare two model versions

        Args:
            version_id1: First version ID
            version_id2: Second version ID
            metrics: Metrics to compare

        Returns:
            Comparison results
        """
        try:
            if version_id1 not in self.versions or version_id2 not in self.versions:
                raise ValueError("One or both versions not found")

            version1 = self.versions[version_id1]
            version2 = self.versions[version_id2]

            if metrics is None:
                metrics = list(PerformanceMetric)

            # Get performance metrics for both versions
            metrics1 = self.performance_monitor.get_version_metrics(version_id1)
            metrics2 = self.performance_monitor.get_version_metrics(version_id2)

            # Compare configurations
            config_diff = self._compare_configurations(
                version1.configuration, version2.configuration
            )

            # Compare performance metrics
            performance_comparison = {}
            for metric in metrics:
                metric_name = metric.value.lower()
                if metric_name in metrics1 and metric_name in metrics2:
                    val1 = metrics1[metric_name]
                    val2 = metrics2[metric_name]

                    performance_comparison[metric_name] = {
                        "version1": val1,
                        "version2": val2,
                        "difference": val2 - val1,
                        "percentage_change": ((val2 - val1) / val1 * 100)
                        if val1 != 0
                        else 0,
                        "better_version": version_id2
                        if self._is_metric_better(metric, val2, val1)
                        else version_id1,
                    }

            return {
                "version1": {
                    "version_id": version_id1,
                    "version_number": version1.version_number,
                    "model_type": version1.model_type.value,
                    "status": version1.status.value,
                },
                "version2": {
                    "version_id": version_id2,
                    "version_number": version2.version_number,
                    "model_type": version2.model_type.value,
                    "status": version2.status.value,
                },
                "configuration_diff": config_diff,
                "performance_comparison": performance_comparison,
                "recommendation": self._generate_comparison_recommendation(
                    performance_comparison
                ),
            }

        except Exception as e:
            logger.error(f"Version comparison failed: {e}")
            return {"status": "failed", "error": str(e)}

    def get_version_lineage(self, version_id: str) -> Dict[str, Any]:
        """
        Get the lineage tree for a model version

        Args:
            version_id: Version ID to get lineage for

        Returns:
            Lineage information
        """
        try:
            if version_id not in self.versions:
                raise ValueError(f"Version {version_id} not found")

            version = self.versions[version_id]

            # Build lineage tree
            lineage = {
                "target_version": version.to_dict(),
                "ancestors": [],
                "descendants": [],
            }

            # Get ancestors
            current_id = version.parent_version_id
            while current_id:
                if current_id in self.versions:
                    ancestor = self.versions[current_id]
                    lineage["ancestors"].append(
                        {
                            "version_id": ancestor.version_id,
                            "version_number": ancestor.version_number,
                            "created_at": ancestor.created_at.isoformat(),
                            "status": ancestor.status.value,
                        }
                    )
                    current_id = ancestor.parent_version_id
                else:
                    break

            # Get descendants
            def collect_descendants(vid: str):
                if vid in self.versions:
                    version_obj = self.versions[vid]
                    for child_id in version_obj.child_version_ids:
                        if child_id in self.versions:
                            child = self.versions[child_id]
                            lineage["descendants"].append(
                                {
                                    "version_id": child.version_id,
                                    "version_number": child.version_number,
                                    "created_at": child.created_at.isoformat(),
                                    "status": child.status.value,
                                }
                            )
                            collect_descendants(child_id)

            collect_descendants(version_id)

            return lineage

        except Exception as e:
            logger.error(f"Failed to get version lineage: {e}")
            return {"status": "failed", "error": str(e)}

    def get_system_overview(self) -> Dict[str, Any]:
        """Get comprehensive overview of the versioning system"""
        try:
            overview = {
                "summary": {
                    "total_versions": len(self.versions),
                    "active_deployments": len(self.active_deployments),
                    "active_experiments": len(self.active_experiments),
                    "deployment_history_count": len(self.deployment_history),
                },
                "models_by_type": {},
                "deployment_status": {},
                "recent_activity": [],
                "performance_summary": {},
            }

            # Models by type
            for model_type in ModelType:
                versions = self.model_type_versions[model_type]
                overview["models_by_type"][model_type.value] = {
                    "total_versions": len(versions),
                    "latest_version": versions[-1] if versions else None,
                    "active_version": self.active_deployments.get(model_type),
                }

            # Deployment status
            for model_type, version_id in self.active_deployments.items():
                if version_id in self.versions:
                    version = self.versions[version_id]
                    overview["deployment_status"][model_type.value] = {
                        "version_id": version_id,
                        "version_number": version.version_number,
                        "deployed_at": version.deployed_at.isoformat()
                        if version.deployed_at
                        else None,
                        "traffic_percentage": version.traffic_percentage,
                    }

            # Recent activity (last 10 items)
            overview["recent_activity"] = self.deployment_history[-10:]

            # Performance summary
            overview["performance_summary"] = (
                self.performance_monitor.get_system_performance_summary()
            )

            return overview

        except Exception as e:
            logger.error(f"Failed to get system overview: {e}")
            return {"status": "failed", "error": str(e)}

    # Helper methods
    def _validate_deployment(self, version: ModelVersion) -> Dict[str, Any]:
        """Validate version before deployment"""
        try:
            validation_result = {"valid": True, "warnings": [], "errors": []}

            # Check if model artifacts exist
            # In real implementation, would check file system
            if not version.model_artifact_path:
                validation_result["errors"].append("Model artifact path not specified")
                validation_result["valid"] = False

            # Check performance metrics if available
            if version.test_results:
                for metric, threshold in self.performance_threshold.items():
                    if metric in version.test_results:
                        value = version.test_results[metric]
                        if value < threshold:
                            validation_result["warnings"].append(
                                f"{metric} ({value}) is below threshold ({threshold})"
                            )

            return validation_result

        except Exception:
            return {"valid": False, "errors": ["Validation failed"]}

    def _cleanup_old_versions(self, model_type: ModelType):
        """Clean up old versions to maintain version limit"""
        try:
            versions = self.model_type_versions[model_type]
            if len(versions) > self.max_versions_per_model:
                # Keep the latest versions and active/production versions
                versions_to_remove = []

                for version_id in versions[: -self.max_versions_per_model]:
                    version = self.versions[version_id]
                    if (
                        version.status
                        not in [VersionStatus.PRODUCTION, VersionStatus.STAGING]
                        and version_id not in self.active_deployments.values()
                    ):
                        versions_to_remove.append(version_id)

                for version_id in versions_to_remove:
                    del self.versions[version_id]
                    self.model_type_versions[model_type].remove(version_id)

                logger.info(
                    f"Cleaned up {len(versions_to_remove)} old versions for {model_type.value}"
                )

        except Exception as e:
            logger.warning(f"Version cleanup failed: {e}")

    def _get_previous_stable_version(
        self, model_type: ModelType, current_version_id: str
    ) -> Optional[str]:
        """Get the previous stable version for rollback"""
        try:
            versions = self.model_type_versions[model_type]

            # Find versions that were successfully deployed before current
            stable_versions = []
            for version_id in versions:
                if version_id != current_version_id:
                    version = self.versions[version_id]
                    if (
                        version.status
                        in [VersionStatus.PRODUCTION, VersionStatus.STAGING]
                        and version.deployed_at
                    ):
                        stable_versions.append((version_id, version.deployed_at))

            if stable_versions:
                # Return most recent stable version
                stable_versions.sort(key=lambda x: x[1], reverse=True)
                return stable_versions[0][0]

            return None

        except Exception:
            return None

    def _compare_configurations(
        self, config1: Dict[str, Any], config2: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Compare two configuration dictionaries"""
        try:
            diff = {"added": {}, "removed": {}, "changed": {}, "unchanged": {}}

            all_keys = set(config1.keys()) | set(config2.keys())

            for key in all_keys:
                if key in config1 and key in config2:
                    if config1[key] != config2[key]:
                        diff["changed"][key] = {
                            "old": config1[key],
                            "new": config2[key],
                        }
                    else:
                        diff["unchanged"][key] = config1[key]
                elif key in config1:
                    diff["removed"][key] = config1[key]
                else:
                    diff["added"][key] = config2[key]

            return diff

        except Exception:
            return {"error": "Configuration comparison failed"}

    def _is_metric_better(
        self, metric: PerformanceMetric, value1: float, value2: float
    ) -> bool:
        """Determine if value1 is better than value2 for given metric"""
        # Higher is better for these metrics
        higher_better = [
            PerformanceMetric.ACCURACY,
            PerformanceMetric.PRECISION,
            PerformanceMetric.RECALL,
            PerformanceMetric.F1_SCORE,
            PerformanceMetric.THROUGHPUT,
            PerformanceMetric.CONFIDENCE_SCORE,
        ]

        # Lower is better for these metrics
        lower_better = [
            PerformanceMetric.LATENCY,
            PerformanceMetric.MEMORY_USAGE,
            PerformanceMetric.ERROR_RATE,
        ]

        if metric in higher_better:
            return value1 > value2
        elif metric in lower_better:
            return value1 < value2
        else:
            return value1 > value2  # Default: higher is better

    def _generate_comparison_recommendation(
        self, performance_comparison: Dict[str, Any]
    ) -> str:
        """Generate recommendation based on performance comparison"""
        try:
            better_count = {"version1": 0, "version2": 0}

            for metric_data in performance_comparison.values():
                better_version = metric_data.get("better_version")
                if "version1" in better_version:
                    better_count["version1"] += 1
                elif "version2" in better_version:
                    better_count["version2"] += 1

            if better_count["version2"] > better_count["version1"]:
                return "Consider upgrading to version2 for better performance"
            elif better_count["version1"] > better_count["version2"]:
                return "Version1 performs better overall"
            else:
                return "Both versions have similar performance characteristics"

        except Exception:
            return "Unable to generate recommendation"

    async def _auto_complete_experiment(self, experiment_id: str):
        """Automatically complete experiment after duration"""
        try:
            experiment = self.active_experiments[experiment_id]

            # Wait for experiment duration
            await asyncio.sleep(experiment.duration_hours * 3600)

            # Complete experiment
            await self.complete_ab_test(experiment_id)

        except Exception as e:
            logger.error(f"Auto-completion failed for experiment {experiment_id}: {e}")

    def _analyze_ab_test_results(
        self,
        control_metrics: Dict[str, float],
        treatment_metrics: Dict[str, float],
        success_metrics: List[PerformanceMetric],
    ) -> Dict[str, Any]:
        """Analyze A/B test results and generate recommendation"""
        try:
            # Simplified statistical analysis
            significant_improvements = 0
            total_comparisons = 0

            statistical_significance = {}

            for metric in success_metrics:
                metric_name = metric.value.lower()

                if metric_name in control_metrics and metric_name in treatment_metrics:
                    control_val = control_metrics[metric_name]
                    treatment_val = treatment_metrics[metric_name]

                    # Simple significance test (in real implementation, would use proper statistical tests)
                    improvement = (
                        ((treatment_val - control_val) / control_val * 100)
                        if control_val != 0
                        else 0
                    )
                    is_significant = abs(improvement) > 5  # 5% threshold
                    is_better = self._is_metric_better(
                        metric, treatment_val, control_val
                    )

                    statistical_significance[metric_name] = {
                        "control_value": control_val,
                        "treatment_value": treatment_val,
                        "improvement_percentage": improvement,
                        "is_significant": is_significant,
                        "treatment_better": is_better,
                    }

                    total_comparisons += 1
                    if is_significant and is_better:
                        significant_improvements += 1

            # Generate recommendation
            if (
                significant_improvements > 0
                and significant_improvements >= total_comparisons * 0.6
            ):
                recommendation = "deploy_treatment"
            elif significant_improvements == 0:
                recommendation = "keep_control"
            else:
                recommendation = "inconclusive"

            return {
                "statistical_significance": statistical_significance,
                "recommendation": recommendation,
                "summary": {
                    "significant_improvements": significant_improvements,
                    "total_comparisons": total_comparisons,
                    "improvement_rate": (
                        significant_improvements / total_comparisons * 100
                    )
                    if total_comparisons > 0
                    else 0,
                },
            }

        except Exception as e:
            logger.error(f"A/B test analysis failed: {e}")
            return {
                "statistical_significance": {},
                "recommendation": "error",
                "error": str(e),
            }

    def _trigger_callbacks(self, event: str, data: Any):
        """Trigger registered callbacks for lifecycle events"""
        try:
            if event in self.callbacks:
                for callback in self.callbacks[event]:
                    try:
                        callback(data)
                    except Exception as e:
                        logger.warning(f"Callback failed for event {event}: {e}")
        except Exception:
            pass

    def register_callback(self, event: str, callback: Callable):
        """Register callback for lifecycle events"""
        if event in self.callbacks:
            self.callbacks[event].append(callback)
        else:
            logger.warning(f"Unknown event type: {event}")


# Mock performance monitor and deployment manager
class PerformanceMonitor:
    """Mock performance monitoring system"""

    def __init__(self):
        self.version_metrics = {}
        self.monitoring_active = set()

    def start_monitoring(self, version_id: str):
        """Start monitoring a version"""
        self.monitoring_active.add(version_id)
        # Initialize with mock metrics
        self.version_metrics[version_id] = {
            "accuracy": 0.85 + (hash(version_id) % 100) / 1000,
            "latency_ms": 500 + (hash(version_id) % 200),
            "throughput": 1000 + (hash(version_id) % 500),
            "error_rate": 0.05 + (hash(version_id) % 50) / 10000,
            "memory_usage": 1024 + (hash(version_id) % 512),
        }

    def start_ab_test_monitoring(self, experiment: ABTestExperiment):
        """Start monitoring for A/B test"""
        self.start_monitoring(experiment.control_version_id)
        self.start_monitoring(experiment.treatment_version_id)

    def get_version_metrics(self, version_id: str) -> Dict[str, float]:
        """Get performance metrics for a version"""
        return self.version_metrics.get(version_id, {})

    def get_system_performance_summary(self) -> Dict[str, Any]:
        """Get overall system performance summary"""
        if not self.version_metrics:
            return {}

        all_metrics = list(self.version_metrics.values())
        avg_accuracy = sum(m.get("accuracy", 0) for m in all_metrics) / len(all_metrics)
        avg_latency = sum(m.get("latency_ms", 0) for m in all_metrics) / len(
            all_metrics
        )

        return {
            "average_accuracy": avg_accuracy,
            "average_latency_ms": avg_latency,
            "monitored_versions": len(self.version_metrics),
        }


class DeploymentManager:
    """Mock deployment management system"""

    def deploy_version(
        self,
        version: ModelVersion,
        strategy: DeploymentStrategy,
        traffic_percentage: float,
    ) -> Dict[str, Any]:
        """Deploy a version using specified strategy"""
        # Mock deployment logic
        return {
            "status": "success",
            "deployment_id": str(uuid.uuid4()),
            "strategy": strategy.value,
            "traffic_percentage": traffic_percentage,
        }

    def rollback_to_version(
        self, current_version_id: str, target_version_id: str
    ) -> Dict[str, Any]:
        """Rollback to a target version"""
        # Mock rollback logic
        return {
            "status": "success",
            "rollback_id": str(uuid.uuid4()),
            "from_version": current_version_id,
            "to_version": target_version_id,
        }
