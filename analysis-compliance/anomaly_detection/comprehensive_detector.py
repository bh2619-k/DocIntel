"""
Real-Time Document Intelligence & Compliance Engine
Complete Anomaly Detection Engine Integration
Version: 1.0
Date: 2026-02-03

This module integrates all 7 features of the anomaly detection system:
1. Statistical Analysis (Feature 1)
2. Pattern Recognition (Feature 2)
3. Outlier Detection (Feature 3)
4. Temporal Analysis (Feature 4)
5. Contextual Analysis (Feature 5)
6. Severity Scoring (Feature 6)
7. Alert Generation (Feature 7)
"""

import logging
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime, timedelta
from dataclasses import dataclass
import json
import asyncio
from concurrent.futures import ThreadPoolExecutor
import threading

from .anomaly_engine import (
    AnomalyDetectionResult,
    AnomalyType,
    AnomalySeverity,
    DetectionMethod,
    StatisticalAnalyzer,
    PatternRecognitionAnalyzer,
    OutlierDetector,
)
from .temporal_contextual import TemporalAnalyzer, ContextualAnalyzer
from .severity_alerting import SeverityScorer, AlertManager, Alert, AlertRule

logger = logging.getLogger(__name__)


@dataclass
class AnomalyDetectionConfig:
    """Configuration for anomaly detection system"""

    # Statistical analysis config
    statistical_enabled: bool = True
    statistical_z_threshold: float = 2.5
    statistical_iqr_multiplier: float = 1.5

    # Pattern recognition config
    pattern_enabled: bool = True
    pattern_similarity_threshold: float = 0.8
    pattern_min_sequence_length: int = 3

    # Outlier detection config
    outlier_enabled: bool = True
    outlier_contamination: float = 0.1
    outlier_ensemble_threshold: float = 0.6

    # Temporal analysis config
    temporal_enabled: bool = True
    temporal_time_window_minutes: int = 60
    temporal_trend_threshold: float = 0.05

    # Contextual analysis config
    contextual_enabled: bool = True
    contextual_similarity_threshold: float = 0.7
    contextual_min_samples: int = 10

    # Severity scoring config
    severity_enabled: bool = True
    severity_business_impact_weight: float = 0.3

    # Alert generation config
    alerts_enabled: bool = True
    alert_history_days: int = 30

    # Performance config
    max_concurrent_detections: int = 5
    detection_timeout_seconds: int = 30


class ComprehensiveAnomalyDetector:
    """Complete anomaly detection engine with all 7 features"""

    def __init__(self, config: AnomalyDetectionConfig = None):
        """
        Initialize comprehensive anomaly detector

        Args:
            config: Detection configuration
        """
        self.config = config or AnomalyDetectionConfig()

        # Initialize all analyzers
        self.statistical_analyzer = (
            StatisticalAnalyzer(
                z_threshold=self.config.statistical_z_threshold,
                iqr_multiplier=self.config.statistical_iqr_multiplier,
            )
            if self.config.statistical_enabled
            else None
        )

        self.pattern_analyzer = (
            PatternRecognitionAnalyzer(
                similarity_threshold=self.config.pattern_similarity_threshold,
                min_sequence_length=self.config.pattern_min_sequence_length,
            )
            if self.config.pattern_enabled
            else None
        )

        self.outlier_detector = (
            OutlierDetector(
                contamination=self.config.outlier_contamination,
                ensemble_threshold=self.config.outlier_ensemble_threshold,
            )
            if self.config.outlier_enabled
            else None
        )

        self.temporal_analyzer = (
            TemporalAnalyzer(
                time_window_minutes=self.config.temporal_time_window_minutes,
                trend_threshold=self.config.temporal_trend_threshold,
            )
            if self.config.temporal_enabled
            else None
        )

        self.contextual_analyzer = (
            ContextualAnalyzer(
                context_similarity_threshold=self.config.contextual_similarity_threshold,
                min_context_samples=self.config.contextual_min_samples,
            )
            if self.config.contextual_enabled
            else None
        )

        self.severity_scorer = (
            SeverityScorer() if self.config.severity_enabled else None
        )

        self.alert_manager = (
            AlertManager(alert_history_days=self.config.alert_history_days)
            if self.config.alerts_enabled
            else None
        )

        # Threading for concurrent detection
        self.thread_pool = ThreadPoolExecutor(
            max_workers=self.config.max_concurrent_detections
        )
        self.detection_lock = threading.Lock()

        # Detection metrics
        self.detection_metrics = {
            "total_detections": 0,
            "anomalies_found": 0,
            "false_positives": 0,
            "processing_times": [],
            "feature_performance": {
                "statistical": {"detections": 0, "anomalies": 0},
                "pattern": {"detections": 0, "anomalies": 0},
                "outlier": {"detections": 0, "anomalies": 0},
                "temporal": {"detections": 0, "anomalies": 0},
                "contextual": {"detections": 0, "anomalies": 0},
            },
        }

    async def detect_anomalies(
        self,
        data: Dict[str, Any],
        context_data: Dict[str, Any] = None,
        business_context: Dict[str, Any] = None,
        timestamp: Optional[datetime] = None,
    ) -> Tuple[List[AnomalyDetectionResult], List[Alert]]:
        """
        Comprehensive anomaly detection using all enabled features

        Args:
            data: Data point to analyze
            context_data: Contextual information
            business_context: Business context for severity scoring
            timestamp: Timestamp for temporal analysis

        Returns:
            Tuple of (detected_anomalies, generated_alerts)
        """
        try:
            start_time = datetime.now()
            if timestamp is None:
                timestamp = start_time

            detected_anomalies = []
            generated_alerts = []

            # Run all detection methods concurrently
            detection_tasks = []

            # Statistical analysis
            if self.statistical_analyzer:
                task = self._run_statistical_detection(data)
                detection_tasks.append(("statistical", task))

            # Pattern recognition
            if self.pattern_analyzer:
                task = self._run_pattern_detection(data)
                detection_tasks.append(("pattern", task))

            # Outlier detection
            if self.outlier_detector:
                task = self._run_outlier_detection(data)
                detection_tasks.append(("outlier", task))

            # Temporal analysis
            if self.temporal_analyzer:
                temporal_keys = [
                    k for k, v in data.items() if isinstance(v, (int, float))
                ]
                task = self._run_temporal_detection(data, temporal_keys, timestamp)
                detection_tasks.append(("temporal", task))

            # Contextual analysis
            if self.contextual_analyzer and context_data:
                task = self._run_contextual_detection(data, context_data)
                detection_tasks.append(("contextual", task))

            # Wait for all detections to complete with timeout
            detection_results = await asyncio.wait_for(
                asyncio.gather(
                    *[task[1] for task in detection_tasks], return_exceptions=True
                ),
                timeout=self.config.detection_timeout_seconds,
            )

            # Process detection results
            for i, (feature_name, result) in enumerate(
                zip([task[0] for task in detection_tasks], detection_results)
            ):
                self.detection_metrics["feature_performance"][feature_name][
                    "detections"
                ] += 1

                if isinstance(result, Exception):
                    logger.error(f"{feature_name} detection failed: {result}")
                    continue

                if isinstance(result, list):
                    detected_anomalies.extend(result)
                    self.detection_metrics["feature_performance"][feature_name][
                        "anomalies"
                    ] += len(result)
                elif result is not None:
                    detected_anomalies.append(result)
                    self.detection_metrics["feature_performance"][feature_name][
                        "anomalies"
                    ] += 1

            # Remove duplicates and combine similar anomalies
            detected_anomalies = self._deduplicate_anomalies(detected_anomalies)

            # Enhanced severity scoring and alert generation
            if detected_anomalies:
                for anomaly in detected_anomalies:
                    # Enhanced severity scoring
                    if self.severity_scorer:
                        severity_score, severity_factors = (
                            self.severity_scorer.calculate_severity_score(
                                anomaly, business_context
                            )
                        )
                        enhanced_severity = self.severity_scorer.get_severity_level(
                            severity_score
                        )
                        anomaly.severity = enhanced_severity

                    # Generate alerts
                    if self.alert_manager:
                        alerts = self.alert_manager.process_anomaly(
                            anomaly, business_context
                        )
                        generated_alerts.extend(alerts)

            # Update metrics
            processing_time = (datetime.now() - start_time).total_seconds()
            with self.detection_lock:
                self.detection_metrics["total_detections"] += 1
                self.detection_metrics["anomalies_found"] += len(detected_anomalies)
                self.detection_metrics["processing_times"].append(processing_time)

                # Keep only last 1000 processing times
                if len(self.detection_metrics["processing_times"]) > 1000:
                    self.detection_metrics["processing_times"] = self.detection_metrics[
                        "processing_times"
                    ][-1000:]

            logger.info(
                f"Anomaly detection completed: {len(detected_anomalies)} anomalies, {len(generated_alerts)} alerts in {processing_time:.2f}s"
            )

            return detected_anomalies, generated_alerts

        except asyncio.TimeoutError:
            logger.error("Anomaly detection timed out")
            return [], []
        except Exception as e:
            logger.error(f"Comprehensive anomaly detection failed: {e}")
            return [], []

    async def _run_statistical_detection(
        self, data: Dict[str, Any]
    ) -> List[AnomalyDetectionResult]:
        """Run statistical analysis"""
        loop = asyncio.get_event_loop()
        numeric_data = {k: v for k, v in data.items() if isinstance(v, (int, float))}
        return await loop.run_in_executor(
            self.thread_pool,
            self.statistical_analyzer.detect_statistical_anomalies,
            numeric_data,
        )

    async def _run_pattern_detection(
        self, data: Dict[str, Any]
    ) -> List[AnomalyDetectionResult]:
        """Run pattern recognition"""
        loop = asyncio.get_event_loop()
        sequence_data = {
            k: v
            for k, v in data.items()
            if isinstance(v, (list, tuple, str)) and len(str(v)) >= 3
        }
        return await loop.run_in_executor(
            self.thread_pool,
            self.pattern_analyzer.detect_pattern_anomalies,
            sequence_data,
        )

    async def _run_outlier_detection(
        self, data: Dict[str, Any]
    ) -> List[AnomalyDetectionResult]:
        """Run outlier detection"""
        loop = asyncio.get_event_loop()
        numeric_data = {k: v for k, v in data.items() if isinstance(v, (int, float))}
        return await loop.run_in_executor(
            self.thread_pool,
            self.outlier_detector.detect_outlier_anomalies,
            numeric_data,
        )

    async def _run_temporal_detection(
        self, data: Dict[str, Any], temporal_keys: List[str], timestamp: datetime
    ) -> List[AnomalyDetectionResult]:
        """Run temporal analysis"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.thread_pool,
            self.temporal_analyzer.detect_temporal_anomalies,
            data,
            temporal_keys,
            timestamp,
        )

    async def _run_contextual_detection(
        self, data: Dict[str, Any], context_data: Dict[str, Any]
    ) -> List[AnomalyDetectionResult]:
        """Run contextual analysis"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.thread_pool,
            self.contextual_analyzer.detect_contextual_anomalies,
            data,
            context_data,
        )

    def _deduplicate_anomalies(
        self, anomalies: List[AnomalyDetectionResult]
    ) -> List[AnomalyDetectionResult]:
        """Remove duplicate and similar anomalies"""
        try:
            if len(anomalies) <= 1:
                return anomalies

            deduplicated = []
            processed_signatures = set()

            for anomaly in anomalies:
                # Create signature for anomaly
                signature = f"{anomaly.anomaly_type.value}_{anomaly.detection_method.value}_{hash(str(sorted(anomaly.data_point.items())))}"

                if signature not in processed_signatures:
                    deduplicated.append(anomaly)
                    processed_signatures.add(signature)
                else:
                    # Find existing anomaly and combine if better
                    for existing in deduplicated:
                        existing_sig = f"{existing.anomaly_type.value}_{existing.detection_method.value}_{hash(str(sorted(existing.data_point.items())))}"
                        if existing_sig == signature:
                            # Keep the one with higher confidence
                            if anomaly.confidence_score > existing.confidence_score:
                                deduplicated.remove(existing)
                                deduplicated.append(anomaly)
                            break

            # Sort by severity and confidence
            severity_order = {
                AnomalySeverity.CRITICAL: 4,
                AnomalySeverity.HIGH: 3,
                AnomalySeverity.MEDIUM: 2,
                AnomalySeverity.LOW: 1,
            }

            deduplicated.sort(
                key=lambda x: (severity_order.get(x.severity, 0), x.confidence_score),
                reverse=True,
            )

            return deduplicated

        except Exception as e:
            logger.warning(f"Anomaly deduplication failed: {e}")
            return anomalies

    def add_alert_rule(self, rule: AlertRule) -> bool:
        """Add custom alert rule"""
        try:
            if self.alert_manager:
                self.alert_manager.alert_rules[rule.rule_id] = rule
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to add alert rule: {e}")
            return False

    def update_configuration(self, config: AnomalyDetectionConfig):
        """Update detection configuration"""
        try:
            self.config = config

            # Update analyzer configurations
            if self.statistical_analyzer and config.statistical_enabled:
                self.statistical_analyzer.z_threshold = config.statistical_z_threshold
                self.statistical_analyzer.iqr_multiplier = (
                    config.statistical_iqr_multiplier
                )

            if self.pattern_analyzer and config.pattern_enabled:
                self.pattern_analyzer.similarity_threshold = (
                    config.pattern_similarity_threshold
                )
                self.pattern_analyzer.min_sequence_length = (
                    config.pattern_min_sequence_length
                )

            if self.outlier_detector and config.outlier_enabled:
                self.outlier_detector.contamination = config.outlier_contamination
                self.outlier_detector.ensemble_threshold = (
                    config.outlier_ensemble_threshold
                )

            if self.temporal_analyzer and config.temporal_enabled:
                self.temporal_analyzer.time_window_minutes = (
                    config.temporal_time_window_minutes
                )
                self.temporal_analyzer.trend_threshold = config.temporal_trend_threshold

            if self.contextual_analyzer and config.contextual_enabled:
                self.contextual_analyzer.context_similarity_threshold = (
                    config.contextual_similarity_threshold
                )
                self.contextual_analyzer.min_context_samples = (
                    config.contextual_min_samples
                )

            logger.info("Anomaly detection configuration updated")

        except Exception as e:
            logger.error(f"Configuration update failed: {e}")

    def get_detection_metrics(self) -> Dict[str, Any]:
        """Get comprehensive detection metrics"""
        try:
            with self.detection_lock:
                metrics = self.detection_metrics.copy()

            # Calculate additional metrics
            if metrics["processing_times"]:
                metrics["avg_processing_time"] = np.mean(metrics["processing_times"])
                metrics["max_processing_time"] = max(metrics["processing_times"])
                metrics["min_processing_time"] = min(metrics["processing_times"])

            # Calculate anomaly rate
            if metrics["total_detections"] > 0:
                metrics["anomaly_rate"] = (
                    metrics["anomalies_found"] / metrics["total_detections"]
                )
            else:
                metrics["anomaly_rate"] = 0

            # Add alert metrics if available
            if self.alert_manager:
                alert_stats = self.alert_manager.get_alert_statistics()
                metrics["alert_stats"] = alert_stats

            return metrics

        except Exception as e:
            logger.error(f"Failed to get detection metrics: {e}")
            return {}

    def reset_metrics(self):
        """Reset detection metrics"""
        try:
            with self.detection_lock:
                self.detection_metrics = {
                    "total_detections": 0,
                    "anomalies_found": 0,
                    "false_positives": 0,
                    "processing_times": [],
                    "feature_performance": {
                        "statistical": {"detections": 0, "anomalies": 0},
                        "pattern": {"detections": 0, "anomalies": 0},
                        "outlier": {"detections": 0, "anomalies": 0},
                        "temporal": {"detections": 0, "anomalies": 0},
                        "contextual": {"detections": 0, "anomalies": 0},
                    },
                }
            logger.info("Detection metrics reset")
        except Exception as e:
            logger.error(f"Failed to reset metrics: {e}")

    def health_check(self) -> Dict[str, Any]:
        """Perform system health check"""
        health_status = {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "features": {},
            "issues": [],
        }

        try:
            # Check each feature
            if self.config.statistical_enabled:
                health_status["features"]["statistical"] = (
                    self.statistical_analyzer is not None
                )

            if self.config.pattern_enabled:
                health_status["features"]["pattern"] = self.pattern_analyzer is not None

            if self.config.outlier_enabled:
                health_status["features"]["outlier"] = self.outlier_detector is not None

            if self.config.temporal_enabled:
                health_status["features"]["temporal"] = (
                    self.temporal_analyzer is not None
                )

            if self.config.contextual_enabled:
                health_status["features"]["contextual"] = (
                    self.contextual_analyzer is not None
                )

            if self.config.severity_enabled:
                health_status["features"]["severity"] = self.severity_scorer is not None

            if self.config.alerts_enabled:
                health_status["features"]["alerts"] = self.alert_manager is not None

            # Check for issues
            if not any(health_status["features"].values()):
                health_status["issues"].append("No detection features are enabled")
                health_status["status"] = "warning"

            # Check thread pool health
            if self.thread_pool._shutdown:
                health_status["issues"].append("Thread pool is shutdown")
                health_status["status"] = "error"

            return health_status

        except Exception as e:
            health_status["status"] = "error"
            health_status["issues"].append(f"Health check failed: {e}")
            return health_status

    def __del__(self):
        """Cleanup resources"""
        try:
            if hasattr(self, "thread_pool"):
                self.thread_pool.shutdown(wait=True)
        except Exception:
            pass
