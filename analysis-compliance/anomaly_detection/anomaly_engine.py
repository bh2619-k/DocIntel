"""
Real-Time Document Intelligence & Compliance Engine
Anomaly Detection System
Version: 1.0
Date: 2026-02-03

This module provides comprehensive anomaly detection with 7 features:
statistical analysis, pattern recognition, outlier detection, temporal analysis,
contextual analysis, severity scoring, and alert generation.
"""

import logging
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime, timedelta
from enum import Enum
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
import json
from collections import defaultdict, deque
import statistics
import uuid

logger = logging.getLogger(__name__)


class AnomalyType(Enum):
    """Types of anomalies that can be detected"""

    STATISTICAL_OUTLIER = "STATISTICAL_OUTLIER"
    PATTERN_DEVIATION = "PATTERN_DEVIATION"
    TEMPORAL_ANOMALY = "TEMPORAL_ANOMALY"
    CONTEXTUAL_ANOMALY = "CONTEXTUAL_ANOMALY"
    THRESHOLD_VIOLATION = "THRESHOLD_VIOLATION"
    BEHAVIORAL_ANOMALY = "BEHAVIORAL_ANOMALY"
    COMPLIANCE_ANOMALY = "COMPLIANCE_ANOMALY"


class AnomalySeverity(Enum):
    """Anomaly severity levels"""

    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    INFO = "INFO"


class DetectionMethod(Enum):
    """Anomaly detection methods"""

    STATISTICAL = "STATISTICAL"
    MACHINE_LEARNING = "MACHINE_LEARNING"
    RULE_BASED = "RULE_BASED"
    PATTERN_MATCHING = "PATTERN_MATCHING"
    THRESHOLD_BASED = "THRESHOLD_BASED"
    CONTEXTUAL = "CONTEXTUAL"
    TEMPORAL = "TEMPORAL"


@dataclass
class AnomalyDetectionResult:
    """Result of anomaly detection"""

    anomaly_id: str
    anomaly_type: AnomalyType
    severity: AnomalySeverity
    detection_method: DetectionMethod
    confidence_score: float
    detected_at: datetime
    data_point: Dict[str, Any]
    baseline_stats: Dict[str, Any]
    deviation_metrics: Dict[str, Any]
    context: Dict[str, Any]
    recommendations: List[str]
    alert_required: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "anomaly_id": self.anomaly_id,
            "anomaly_type": self.anomaly_type.value,
            "severity": self.severity.value,
            "detection_method": self.detection_method.value,
            "confidence_score": self.confidence_score,
            "detected_at": self.detected_at.isoformat(),
            "data_point": self.data_point,
            "baseline_stats": self.baseline_stats,
            "deviation_metrics": self.deviation_metrics,
            "context": self.context,
            "recommendations": self.recommendations,
            "alert_required": self.alert_required,
        }


class StatisticalAnalyzer:
    """Statistical anomaly detection analyzer (Feature 1)"""

    def __init__(
        self,
        z_score_threshold: float = 3.0,
        iqr_multiplier: float = 1.5,
        window_size: int = 100,
        confidence_level: float = 0.95,
    ):
        """
        Initialize statistical analyzer

        Args:
            z_score_threshold: Z-score threshold for outlier detection
            iqr_multiplier: IQR multiplier for outlier detection
            window_size: Window size for rolling statistics
            confidence_level: Confidence level for statistical tests
        """
        self.z_score_threshold = z_score_threshold
        self.iqr_multiplier = iqr_multiplier
        self.window_size = window_size
        self.confidence_level = confidence_level

        # Historical data storage
        self.data_windows: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=window_size)
        )
        self.baseline_stats: Dict[str, Dict[str, float]] = {}

    def detect_statistical_anomalies(
        self, data: Dict[str, Any], metric_keys: List[str]
    ) -> List[AnomalyDetectionResult]:
        """
        Detect statistical anomalies in numerical data

        Args:
            data: Data point to analyze
            metric_keys: Keys for numerical metrics to analyze

        Returns:
            List of detected anomalies
        """
        try:
            anomalies = []
            current_time = datetime.now()

            for key in metric_keys:
                if key not in data:
                    continue

                value = data[key]
                if not isinstance(value, (int, float)):
                    continue

                # Add to historical data
                self.data_windows[key].append(value)

                # Need sufficient data for analysis
                if len(self.data_windows[key]) < 10:
                    continue

                # Calculate statistics
                window_data = list(self.data_windows[key])
                stats = self._calculate_statistics(window_data)
                self.baseline_stats[key] = stats

                # Z-score anomaly detection
                z_score_anomaly = self._detect_z_score_anomaly(value, stats)
                if z_score_anomaly:
                    anomaly = AnomalyDetectionResult(
                        anomaly_id=str(uuid.uuid4()),
                        anomaly_type=AnomalyType.STATISTICAL_OUTLIER,
                        severity=self._calculate_statistical_severity(
                            z_score_anomaly["z_score"]
                        ),
                        detection_method=DetectionMethod.STATISTICAL,
                        confidence_score=min(
                            1.0, abs(z_score_anomaly["z_score"]) / 10.0
                        ),
                        detected_at=current_time,
                        data_point={key: value},
                        baseline_stats=stats,
                        deviation_metrics=z_score_anomaly,
                        context={"metric_key": key, "window_size": len(window_data)},
                        recommendations=self._generate_statistical_recommendations(
                            z_score_anomaly
                        ),
                    )
                    anomalies.append(anomaly)

                # IQR anomaly detection
                iqr_anomaly = self._detect_iqr_anomaly(value, stats)
                if iqr_anomaly and not z_score_anomaly:  # Avoid duplicate detection
                    anomaly = AnomalyDetectionResult(
                        anomaly_id=str(uuid.uuid4()),
                        anomaly_type=AnomalyType.STATISTICAL_OUTLIER,
                        severity=self._calculate_iqr_severity(
                            iqr_anomaly["distance_from_iqr"]
                        ),
                        detection_method=DetectionMethod.STATISTICAL,
                        confidence_score=min(
                            1.0, iqr_anomaly["distance_from_iqr"] / 5.0
                        ),
                        detected_at=current_time,
                        data_point={key: value},
                        baseline_stats=stats,
                        deviation_metrics=iqr_anomaly,
                        context={"metric_key": key, "detection_method": "IQR"},
                        recommendations=self._generate_iqr_recommendations(iqr_anomaly),
                    )
                    anomalies.append(anomaly)

                # Modified Z-score (robust to outliers)
                modified_z_anomaly = self._detect_modified_z_score_anomaly(
                    value, window_data
                )
                if modified_z_anomaly and not z_score_anomaly and not iqr_anomaly:
                    anomaly = AnomalyDetectionResult(
                        anomaly_id=str(uuid.uuid4()),
                        anomaly_type=AnomalyType.STATISTICAL_OUTLIER,
                        severity=self._calculate_statistical_severity(
                            modified_z_anomaly["modified_z_score"]
                        ),
                        detection_method=DetectionMethod.STATISTICAL,
                        confidence_score=min(
                            1.0, abs(modified_z_anomaly["modified_z_score"]) / 8.0
                        ),
                        detected_at=current_time,
                        data_point={key: value},
                        baseline_stats=stats,
                        deviation_metrics=modified_z_anomaly,
                        context={
                            "metric_key": key,
                            "detection_method": "Modified Z-Score",
                        },
                        recommendations=self._generate_statistical_recommendations(
                            modified_z_anomaly
                        ),
                    )
                    anomalies.append(anomaly)

            return anomalies

        except Exception as e:
            logger.error(f"Statistical anomaly detection failed: {e}")
            return []

    def _calculate_statistics(self, data: List[float]) -> Dict[str, float]:
        """Calculate comprehensive statistics"""
        try:
            return {
                "mean": statistics.mean(data),
                "median": statistics.median(data),
                "std_dev": statistics.stdev(data) if len(data) > 1 else 0,
                "variance": statistics.variance(data) if len(data) > 1 else 0,
                "min": min(data),
                "max": max(data),
                "q1": np.percentile(data, 25),
                "q3": np.percentile(data, 75),
                "iqr": np.percentile(data, 75) - np.percentile(data, 25),
                "skewness": self._calculate_skewness(data),
                "kurtosis": self._calculate_kurtosis(data),
            }
        except Exception:
            return {}

    def _calculate_skewness(self, data: List[float]) -> float:
        """Calculate skewness"""
        try:
            n = len(data)
            if n < 3:
                return 0
            mean = statistics.mean(data)
            std_dev = statistics.stdev(data)
            if std_dev == 0:
                return 0
            skew = sum(((x - mean) / std_dev) ** 3 for x in data) / n
            return skew
        except Exception:
            return 0

    def _calculate_kurtosis(self, data: List[float]) -> float:
        """Calculate kurtosis"""
        try:
            n = len(data)
            if n < 4:
                return 0
            mean = statistics.mean(data)
            std_dev = statistics.stdev(data)
            if std_dev == 0:
                return 0
            kurt = sum(((x - mean) / std_dev) ** 4 for x in data) / n - 3
            return kurt
        except Exception:
            return 0

    def _detect_z_score_anomaly(
        self, value: float, stats: Dict[str, float]
    ) -> Optional[Dict[str, Any]]:
        """Detect Z-score based anomaly"""
        try:
            if stats["std_dev"] == 0:
                return None

            z_score = abs(value - stats["mean"]) / stats["std_dev"]

            if z_score > self.z_score_threshold:
                return {
                    "z_score": z_score,
                    "threshold": self.z_score_threshold,
                    "distance_from_mean": abs(value - stats["mean"]),
                    "direction": "above" if value > stats["mean"] else "below",
                }

            return None
        except Exception:
            return None

    def _detect_iqr_anomaly(
        self, value: float, stats: Dict[str, float]
    ) -> Optional[Dict[str, Any]]:
        """Detect IQR-based anomaly"""
        try:
            q1, q3, iqr = stats["q1"], stats["q3"], stats["iqr"]

            lower_bound = q1 - self.iqr_multiplier * iqr
            upper_bound = q3 + self.iqr_multiplier * iqr

            if value < lower_bound or value > upper_bound:
                if value < lower_bound:
                    distance = lower_bound - value
                    direction = "below"
                else:
                    distance = value - upper_bound
                    direction = "above"

                return {
                    "distance_from_iqr": distance / iqr if iqr > 0 else distance,
                    "lower_bound": lower_bound,
                    "upper_bound": upper_bound,
                    "direction": direction,
                }

            return None
        except Exception:
            return None

    def _detect_modified_z_score_anomaly(
        self, value: float, data: List[float]
    ) -> Optional[Dict[str, Any]]:
        """Detect anomaly using modified Z-score (robust to outliers)"""
        try:
            median = statistics.median(data)
            mad = statistics.median(
                [abs(x - median) for x in data]
            )  # Median Absolute Deviation

            if mad == 0:
                return None

            modified_z_score = 0.6745 * (value - median) / mad

            if abs(modified_z_score) > 3.5:  # Common threshold for modified Z-score
                return {
                    "modified_z_score": modified_z_score,
                    "threshold": 3.5,
                    "median": median,
                    "mad": mad,
                }

            return None
        except Exception:
            return None

    def _calculate_statistical_severity(self, z_score: float) -> AnomalySeverity:
        """Calculate severity based on Z-score"""
        abs_z = abs(z_score)
        if abs_z > 5:
            return AnomalySeverity.CRITICAL
        elif abs_z > 4:
            return AnomalySeverity.HIGH
        elif abs_z > 3:
            return AnomalySeverity.MEDIUM
        else:
            return AnomalySeverity.LOW

    def _calculate_iqr_severity(self, distance: float) -> AnomalySeverity:
        """Calculate severity based on IQR distance"""
        if distance > 5:
            return AnomalySeverity.CRITICAL
        elif distance > 3:
            return AnomalySeverity.HIGH
        elif distance > 2:
            return AnomalySeverity.MEDIUM
        else:
            return AnomalySeverity.LOW

    def _generate_statistical_recommendations(
        self, anomaly_data: Dict[str, Any]
    ) -> List[str]:
        """Generate recommendations for statistical anomalies"""
        recommendations = []

        if "z_score" in anomaly_data:
            z_score = anomaly_data["z_score"]
            if z_score > 4:
                recommendations.append(
                    "Investigate data collection process for potential errors"
                )
                recommendations.append("Verify measurement accuracy and calibration")
            elif z_score > 3:
                recommendations.append(
                    "Review data point for potential data quality issues"
                )
                recommendations.append("Consider adjusting baseline statistics window")

        if "direction" in anomaly_data:
            if anomaly_data["direction"] == "above":
                recommendations.append(
                    "Monitor for trend continuation - possible system overload"
                )
            else:
                recommendations.append(
                    "Check for system degradation or resource constraints"
                )

        recommendations.append("Increase monitoring frequency for this metric")

        return recommendations

    def _generate_iqr_recommendations(self, anomaly_data: Dict[str, Any]) -> List[str]:
        """Generate recommendations for IQR anomalies"""
        return [
            "Review data point against business context",
            "Consider if this represents a legitimate business event",
            "Evaluate need for threshold adjustment",
            "Monitor for pattern emergence",
        ]


class PatternRecognitionAnalyzer:
    """Pattern recognition anomaly detection analyzer (Feature 2)"""

    def __init__(
        self,
        sequence_length: int = 10,
        similarity_threshold: float = 0.8,
        pattern_cache_size: int = 1000,
    ):
        """
        Initialize pattern recognition analyzer

        Args:
            sequence_length: Length of sequences to analyze
            similarity_threshold: Threshold for pattern similarity
            pattern_cache_size: Maximum number of patterns to cache
        """
        self.sequence_length = sequence_length
        self.similarity_threshold = similarity_threshold
        self.pattern_cache_size = pattern_cache_size

        # Pattern storage
        self.known_patterns: Dict[str, List[List[float]]] = defaultdict(list)
        self.sequence_buffers: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=sequence_length)
        )
        self.pattern_frequencies: Dict[str, Dict[tuple, int]] = defaultdict(
            lambda: defaultdict(int)
        )

    def detect_pattern_anomalies(
        self, data: Dict[str, Any], sequence_keys: List[str]
    ) -> List[AnomalyDetectionResult]:
        """
        Detect pattern-based anomalies

        Args:
            data: Current data point
            sequence_keys: Keys for sequence data to analyze

        Returns:
            List of detected pattern anomalies
        """
        try:
            anomalies = []
            current_time = datetime.now()

            for key in sequence_keys:
                if key not in data:
                    continue

                value = data[key]
                if not isinstance(value, (int, float)):
                    continue

                # Add to sequence buffer
                self.sequence_buffers[key].append(value)

                # Need full sequence for analysis
                if len(self.sequence_buffers[key]) < self.sequence_length:
                    continue

                current_sequence = list(self.sequence_buffers[key])

                # Pattern matching analysis
                pattern_anomaly = self._detect_pattern_deviation(key, current_sequence)
                if pattern_anomaly:
                    anomaly = AnomalyDetectionResult(
                        anomaly_id=str(uuid.uuid4()),
                        anomaly_type=AnomalyType.PATTERN_DEVIATION,
                        severity=pattern_anomaly["severity"],
                        detection_method=DetectionMethod.PATTERN_MATCHING,
                        confidence_score=pattern_anomaly["confidence"],
                        detected_at=current_time,
                        data_point={key: current_sequence},
                        baseline_stats=pattern_anomaly["baseline_patterns"],
                        deviation_metrics=pattern_anomaly["deviation_metrics"],
                        context={
                            "metric_key": key,
                            "sequence_length": self.sequence_length,
                        },
                        recommendations=self._generate_pattern_recommendations(
                            pattern_anomaly
                        ),
                    )
                    anomalies.append(anomaly)

                # Update known patterns
                self._update_pattern_knowledge(key, current_sequence)

            return anomalies

        except Exception as e:
            logger.error(f"Pattern anomaly detection failed: {e}")
            return []

    def _detect_pattern_deviation(
        self, key: str, sequence: List[float]
    ) -> Optional[Dict[str, Any]]:
        """Detect deviations from known patterns"""
        try:
            if key not in self.known_patterns or not self.known_patterns[key]:
                return None

            # Calculate similarity to known patterns
            similarities = []
            for known_pattern in self.known_patterns[key]:
                similarity = self._calculate_sequence_similarity(
                    sequence, known_pattern
                )
                similarities.append(similarity)

            max_similarity = max(similarities) if similarities else 0

            # Check if current sequence deviates significantly
            if max_similarity < self.similarity_threshold:
                # Calculate pattern characteristics
                pattern_stats = self._analyze_pattern_characteristics(sequence)
                baseline_stats = self._get_baseline_pattern_stats(key)

                deviation_score = 1.0 - max_similarity
                severity = self._calculate_pattern_severity(
                    deviation_score, pattern_stats
                )

                return {
                    "deviation_score": deviation_score,
                    "max_similarity": max_similarity,
                    "threshold": self.similarity_threshold,
                    "pattern_characteristics": pattern_stats,
                    "baseline_patterns": baseline_stats,
                    "deviation_metrics": {
                        "similarity_deficit": self.similarity_threshold
                        - max_similarity,
                        "pattern_complexity": pattern_stats.get("complexity", 0),
                        "trend_deviation": pattern_stats.get("trend_strength", 0),
                    },
                    "severity": severity,
                    "confidence": min(1.0, deviation_score * 2),
                }

            return None

        except Exception as e:
            logger.error(f"Pattern deviation detection failed: {e}")
            return None

    def _calculate_sequence_similarity(
        self, seq1: List[float], seq2: List[float]
    ) -> float:
        """Calculate similarity between two sequences"""
        try:
            if len(seq1) != len(seq2):
                return 0.0

            # Normalize sequences
            norm_seq1 = self._normalize_sequence(seq1)
            norm_seq2 = self._normalize_sequence(seq2)

            # Calculate correlation coefficient
            correlation = np.corrcoef(norm_seq1, norm_seq2)[0, 1]

            # Handle NaN correlation
            if np.isnan(correlation):
                return 0.0

            # Convert to similarity (0-1 range)
            similarity = (correlation + 1) / 2

            return similarity

        except Exception:
            return 0.0

    def _normalize_sequence(self, sequence: List[float]) -> List[float]:
        """Normalize sequence to [0, 1] range"""
        try:
            if not sequence:
                return []

            min_val = min(sequence)
            max_val = max(sequence)

            if max_val == min_val:
                return [0.5] * len(sequence)

            return [(x - min_val) / (max_val - min_val) for x in sequence]

        except Exception:
            return sequence

    def _analyze_pattern_characteristics(
        self, sequence: List[float]
    ) -> Dict[str, float]:
        """Analyze characteristics of a pattern"""
        try:
            characteristics = {}

            # Trend strength
            if len(sequence) > 1:
                differences = [
                    sequence[i + 1] - sequence[i] for i in range(len(sequence) - 1)
                ]
                trend_strength = (
                    abs(sum(differences)) / len(differences) if differences else 0
                )
                characteristics["trend_strength"] = trend_strength

            # Volatility (standard deviation)
            characteristics["volatility"] = np.std(sequence) if len(sequence) > 1 else 0

            # Pattern complexity (number of direction changes)
            direction_changes = 0
            if len(sequence) > 2:
                for i in range(1, len(sequence) - 1):
                    if (
                        sequence[i] > sequence[i - 1] and sequence[i + 1] < sequence[i]
                    ) or (
                        sequence[i] < sequence[i - 1] and sequence[i + 1] > sequence[i]
                    ):
                        direction_changes += 1

            characteristics["complexity"] = direction_changes / max(
                1, len(sequence) - 2
            )

            # Periodicity (simplified autocorrelation check)
            characteristics["periodicity"] = self._detect_periodicity(sequence)

            return characteristics

        except Exception:
            return {}

    def _detect_periodicity(self, sequence: List[float]) -> float:
        """Detect periodicity in sequence"""
        try:
            if len(sequence) < 4:
                return 0.0

            # Check for simple periodicities
            max_period = len(sequence) // 2
            best_correlation = 0.0

            for period in range(2, max_period + 1):
                if len(sequence) >= 2 * period:
                    segment1 = sequence[:period]
                    segment2 = sequence[period : 2 * period]

                    correlation = self._calculate_sequence_similarity(
                        segment1, segment2
                    )
                    best_correlation = max(best_correlation, correlation)

            return best_correlation

        except Exception:
            return 0.0

    def _get_baseline_pattern_stats(self, key: str) -> Dict[str, Any]:
        """Get baseline statistics for known patterns"""
        try:
            if key not in self.known_patterns:
                return {}

            patterns = self.known_patterns[key]
            if not patterns:
                return {}

            # Calculate average characteristics
            all_characteristics = [
                self._analyze_pattern_characteristics(pattern) for pattern in patterns
            ]

            baseline = {}
            if all_characteristics:
                for char_key in all_characteristics[0].keys():
                    values = [chars.get(char_key, 0) for chars in all_characteristics]
                    baseline[f"avg_{char_key}"] = np.mean(values)
                    baseline[f"std_{char_key}"] = np.std(values)

            baseline["num_patterns"] = len(patterns)

            return baseline

        except Exception:
            return {}

    def _calculate_pattern_severity(
        self, deviation_score: float, pattern_stats: Dict[str, float]
    ) -> AnomalySeverity:
        """Calculate severity based on pattern deviation"""
        try:
            # Base severity on deviation score
            if deviation_score > 0.8:
                base_severity = AnomalySeverity.CRITICAL
            elif deviation_score > 0.6:
                base_severity = AnomalySeverity.HIGH
            elif deviation_score > 0.4:
                base_severity = AnomalySeverity.MEDIUM
            else:
                base_severity = AnomalySeverity.LOW

            # Adjust based on pattern complexity
            complexity = pattern_stats.get("complexity", 0)
            if complexity > 0.7:  # Very complex pattern
                # Upgrade severity for highly complex deviations
                severity_order = [
                    AnomalySeverity.LOW,
                    AnomalySeverity.MEDIUM,
                    AnomalySeverity.HIGH,
                    AnomalySeverity.CRITICAL,
                ]
                current_index = severity_order.index(base_severity)
                if current_index < len(severity_order) - 1:
                    return severity_order[current_index + 1]

            return base_severity

        except Exception:
            return AnomalySeverity.MEDIUM

    def _update_pattern_knowledge(self, key: str, sequence: List[float]):
        """Update knowledge base with new patterns"""
        try:
            # Add to known patterns if not too similar to existing ones
            is_novel = True

            for existing_pattern in self.known_patterns[key]:
                similarity = self._calculate_sequence_similarity(
                    sequence, existing_pattern
                )
                if similarity > 0.9:  # Very similar pattern already exists
                    is_novel = False
                    break

            if is_novel:
                self.known_patterns[key].append(sequence.copy())

                # Limit pattern cache size
                if len(self.known_patterns[key]) > self.pattern_cache_size:
                    self.known_patterns[key].pop(0)  # Remove oldest

        except Exception:
            pass

    def _generate_pattern_recommendations(
        self, pattern_data: Dict[str, Any]
    ) -> List[str]:
        """Generate recommendations for pattern anomalies"""
        recommendations = []

        deviation_score = pattern_data.get("deviation_score", 0)
        pattern_chars = pattern_data.get("pattern_characteristics", {})

        if deviation_score > 0.7:
            recommendations.append("Investigate potential system behavior change")
            recommendations.append("Review recent configuration or environment changes")

        if pattern_chars.get("complexity", 0) > 0.5:
            recommendations.append("Monitor for unstable system behavior")
            recommendations.append("Consider smoothing or filtering noisy data")

        if pattern_chars.get("trend_strength", 0) > 0.7:
            recommendations.append(
                "Strong trend detected - monitor for potential capacity issues"
            )

        recommendations.append(
            "Update pattern baseline if this represents new normal behavior"
        )

        return recommendations


class OutlierDetector:
    """Outlier detection analyzer (Feature 3)"""

    def __init__(
        self,
        isolation_forest_contamination: float = 0.1,
        lof_n_neighbors: int = 20,
        ensemble_threshold: float = 0.5,
    ):
        """
        Initialize outlier detector

        Args:
            isolation_forest_contamination: Contamination parameter for Isolation Forest
            lof_n_neighbors: Number of neighbors for Local Outlier Factor
            ensemble_threshold: Threshold for ensemble outlier detection
        """
        self.isolation_forest_contamination = isolation_forest_contamination
        self.lof_n_neighbors = lof_n_neighbors
        self.ensemble_threshold = ensemble_threshold

        # Historical data for outlier detection
        self.feature_history: Dict[str, List[List[float]]] = defaultdict(list)
        self.outlier_scores: Dict[str, List[float]] = defaultdict(list)

    def detect_outliers(
        self, data: Dict[str, Any], feature_keys: List[str]
    ) -> List[AnomalyDetectionResult]:
        """
        Detect outliers using multiple algorithms

        Args:
            data: Current data point
            feature_keys: Keys for features to analyze

        Returns:
            List of detected outlier anomalies
        """
        try:
            anomalies = []
            current_time = datetime.now()

            # Extract feature vector
            feature_vector = []
            valid_keys = []

            for key in feature_keys:
                if key in data and isinstance(data[key], (int, float)):
                    feature_vector.append(float(data[key]))
                    valid_keys.append(key)

            if len(feature_vector) < 2:  # Need at least 2 features
                return anomalies

            # Store in history
            feature_signature = "_".join(valid_keys)
            self.feature_history[feature_signature].append(feature_vector.copy())

            # Limit history size
            if len(self.feature_history[feature_signature]) > 1000:
                self.feature_history[feature_signature] = self.feature_history[
                    feature_signature
                ][-1000:]

            # Need sufficient history for outlier detection
            if len(self.feature_history[feature_signature]) < 50:
                return anomalies

            # Ensemble outlier detection
            outlier_result = self._detect_ensemble_outliers(
                feature_vector,
                self.feature_history[feature_signature],
                feature_signature,
            )

            if outlier_result["is_outlier"]:
                anomaly = AnomalyDetectionResult(
                    anomaly_id=str(uuid.uuid4()),
                    anomaly_type=AnomalyType.STATISTICAL_OUTLIER,
                    severity=outlier_result["severity"],
                    detection_method=DetectionMethod.MACHINE_LEARNING,
                    confidence_score=outlier_result["confidence"],
                    detected_at=current_time,
                    data_point=dict(zip(valid_keys, feature_vector)),
                    baseline_stats=outlier_result["baseline_stats"],
                    deviation_metrics=outlier_result["outlier_metrics"],
                    context={
                        "feature_keys": valid_keys,
                        "detection_algorithms": outlier_result["algorithms_triggered"],
                    },
                    recommendations=self._generate_outlier_recommendations(
                        outlier_result
                    ),
                )
                anomalies.append(anomaly)

            return anomalies

        except Exception as e:
            logger.error(f"Outlier detection failed: {e}")
            return []

    def _detect_ensemble_outliers(
        self, feature_vector: List[float], history: List[List[float]], signature: str
    ) -> Dict[str, Any]:
        """Detect outliers using ensemble of algorithms"""
        try:
            # Prepare data matrix
            data_matrix = np.array(history + [feature_vector])
            current_point = np.array(feature_vector).reshape(1, -1)

            outlier_scores = {}
            algorithms_triggered = []

            # 1. Isolation Forest (simplified implementation)
            isolation_score = self._isolation_forest_score(current_point, data_matrix)
            outlier_scores["isolation_forest"] = isolation_score
            if isolation_score > 0.6:
                algorithms_triggered.append("isolation_forest")

            # 2. Local Outlier Factor (simplified implementation)
            lof_score = self._local_outlier_factor_score(
                current_point, data_matrix[:-1]
            )
            outlier_scores["lof"] = lof_score
            if lof_score > 1.5:
                algorithms_triggered.append("lof")

            # 3. Mahalanobis Distance
            mahalanobis_score = self._mahalanobis_distance_score(
                current_point, data_matrix[:-1]
            )
            outlier_scores["mahalanobis"] = mahalanobis_score
            if mahalanobis_score > 3.0:
                algorithms_triggered.append("mahalanobis")

            # 4. One-Class SVM (simplified)
            svm_score = self._one_class_svm_score(current_point, data_matrix[:-1])
            outlier_scores["one_class_svm"] = svm_score
            if svm_score < -0.5:
                algorithms_triggered.append("one_class_svm")

            # Ensemble decision
            ensemble_score = self._calculate_ensemble_score(outlier_scores)
            is_outlier = ensemble_score > self.ensemble_threshold

            # Calculate baseline statistics
            baseline_stats = {
                "mean_vector": np.mean(data_matrix[:-1], axis=0).tolist(),
                "std_vector": np.std(data_matrix[:-1], axis=0).tolist(),
                "data_points": len(data_matrix) - 1,
            }

            # Determine severity
            severity = self._calculate_outlier_severity(
                ensemble_score, algorithms_triggered
            )

            return {
                "is_outlier": is_outlier,
                "ensemble_score": ensemble_score,
                "individual_scores": outlier_scores,
                "algorithms_triggered": algorithms_triggered,
                "confidence": min(1.0, ensemble_score * 2),
                "severity": severity,
                "baseline_stats": baseline_stats,
                "outlier_metrics": {
                    "ensemble_score": ensemble_score,
                    "threshold": self.ensemble_threshold,
                    "algorithms_count": len(algorithms_triggered),
                },
            }

        except Exception as e:
            logger.error(f"Ensemble outlier detection failed: {e}")
            return {"is_outlier": False, "error": str(e)}

    def _isolation_forest_score(self, point: np.ndarray, data: np.ndarray) -> float:
        """Simplified Isolation Forest scoring"""
        try:
            # Mock implementation - in practice would use sklearn.ensemble.IsolationForest
            # Calculate average path length to isolate the point
            n_samples = len(data)

            # Simplified scoring based on distance to data center
            data_center = np.mean(data, axis=0)
            distances = [np.linalg.norm(point - row) for row in data]
            avg_distance = np.mean(distances)

            # Normalize score (higher = more anomalous)
            max_distance = np.max(distances) if distances else 1
            score = avg_distance / max_distance if max_distance > 0 else 0

            return score

        except Exception:
            return 0.0

    def _local_outlier_factor_score(self, point: np.ndarray, data: np.ndarray) -> float:
        """Simplified Local Outlier Factor scoring"""
        try:
            if len(data) < self.lof_n_neighbors:
                return 1.0

            # Calculate k-nearest neighbors
            distances = [np.linalg.norm(point - row) for row in data]
            k_distances = sorted(distances)[: self.lof_n_neighbors]

            # Simplified LOF calculation
            avg_k_distance = np.mean(k_distances)

            # Compare with average k-distance of neighbors
            neighbor_avg_distances = []
            for i, row in enumerate(data):
                row_distances = [
                    np.linalg.norm(row - other_row)
                    for j, other_row in enumerate(data)
                    if i != j
                ]
                if len(row_distances) >= self.lof_n_neighbors:
                    neighbor_k_distances = sorted(row_distances)[: self.lof_n_neighbors]
                    neighbor_avg_distances.append(np.mean(neighbor_k_distances))

            if neighbor_avg_distances:
                neighbor_avg = np.mean(neighbor_avg_distances)
                lof_score = avg_k_distance / neighbor_avg if neighbor_avg > 0 else 1.0
            else:
                lof_score = 1.0

            return lof_score

        except Exception:
            return 1.0

    def _mahalanobis_distance_score(self, point: np.ndarray, data: np.ndarray) -> float:
        """Calculate Mahalanobis distance score"""
        try:
            if len(data) < 2:
                return 0.0

            # Calculate covariance matrix
            cov_matrix = np.cov(data.T)

            # Handle singular covariance matrix
            try:
                inv_cov = np.linalg.inv(cov_matrix)
            except np.linalg.LinAlgError:
                # Use pseudo-inverse for singular matrices
                inv_cov = np.linalg.pinv(cov_matrix)

            # Calculate mean
            mean_vector = np.mean(data, axis=0)

            # Calculate Mahalanobis distance
            diff = point.flatten() - mean_vector
            mahalanobis_dist = np.sqrt(np.dot(np.dot(diff, inv_cov), diff))

            return mahalanobis_dist

        except Exception:
            return 0.0

    def _one_class_svm_score(self, point: np.ndarray, data: np.ndarray) -> float:
        """Simplified One-Class SVM scoring"""
        try:
            # Mock implementation - simplified distance-based scoring
            # In practice would use sklearn.svm.OneClassSVM

            data_center = np.mean(data, axis=0)
            data_std = np.std(data, axis=0)

            # Normalize point
            normalized_point = (point.flatten() - data_center) / (data_std + 1e-8)

            # Simple scoring based on normalized distance
            score = -np.linalg.norm(normalized_point)

            return score

        except Exception:
            return 0.0

    def _calculate_ensemble_score(self, scores: Dict[str, float]) -> float:
        """Calculate ensemble score from individual algorithm scores"""
        try:
            # Weighted ensemble
            weights = {
                "isolation_forest": 0.3,
                "lof": 0.3,
                "mahalanobis": 0.25,
                "one_class_svm": 0.15,
            }

            # Normalize scores to [0, 1] range
            normalized_scores = {}
            for algo, score in scores.items():
                if algo == "isolation_forest":
                    normalized_scores[algo] = min(1.0, max(0.0, score))
                elif algo == "lof":
                    normalized_scores[algo] = min(
                        1.0, max(0.0, (score - 1.0))
                    )  # LOF > 1 indicates outlier
                elif algo == "mahalanobis":
                    normalized_scores[algo] = min(
                        1.0, max(0.0, score / 5.0)
                    )  # Scale Mahalanobis
                elif algo == "one_class_svm":
                    normalized_scores[algo] = min(
                        1.0, max(0.0, (-score))
                    )  # SVM negative scores

            # Calculate weighted average
            ensemble_score = sum(
                weights[algo] * normalized_scores.get(algo, 0)
                for algo in weights.keys()
            )

            return ensemble_score

        except Exception:
            return 0.0

    def _calculate_outlier_severity(
        self, ensemble_score: float, algorithms_triggered: List[str]
    ) -> AnomalySeverity:
        """Calculate severity based on ensemble score and number of algorithms triggered"""
        try:
            num_triggered = len(algorithms_triggered)

            if ensemble_score > 0.8 or num_triggered >= 3:
                return AnomalySeverity.CRITICAL
            elif ensemble_score > 0.6 or num_triggered >= 2:
                return AnomalySeverity.HIGH
            elif ensemble_score > 0.4 or num_triggered >= 1:
                return AnomalySeverity.MEDIUM
            else:
                return AnomalySeverity.LOW

        except Exception:
            return AnomalySeverity.MEDIUM

    def _generate_outlier_recommendations(
        self, outlier_data: Dict[str, Any]
    ) -> List[str]:
        """Generate recommendations for outlier anomalies"""
        recommendations = []

        algorithms_triggered = outlier_data.get("algorithms_triggered", [])
        ensemble_score = outlier_data.get("ensemble_score", 0)

        if "isolation_forest" in algorithms_triggered:
            recommendations.append(
                "Data point is isolated from normal distribution - investigate data source"
            )

        if "lof" in algorithms_triggered:
            recommendations.append(
                "Local density anomaly detected - check for localized issues"
            )

        if "mahalanobis" in algorithms_triggered:
            recommendations.append(
                "Correlation pattern anomaly - review multivariate relationships"
            )

        if ensemble_score > 0.7:
            recommendations.append(
                "High confidence outlier - immediate investigation recommended"
            )
            recommendations.append(
                "Consider if this represents a new operational state"
            )

        recommendations.append("Monitor subsequent data points for pattern emergence")

        return recommendations
