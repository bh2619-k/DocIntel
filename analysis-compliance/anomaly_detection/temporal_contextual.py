"""
Real-Time Document Intelligence & Compliance Engine
Temporal & Contextual Anomaly Analysis
Version: 1.0
Date: 2026-02-03

This module provides temporal analysis (Feature 4) and contextual analysis (Feature 5)
for comprehensive anomaly detection capabilities.
"""

import logging
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from collections import defaultdict, deque
import json
from dataclasses import dataclass
from .anomaly_engine import (
    AnomalyDetectionResult,
    AnomalyType,
    AnomalySeverity,
    DetectionMethod,
)
import uuid

logger = logging.getLogger(__name__)


class TemporalAnalyzer:
    """Temporal anomaly detection analyzer (Feature 4)"""

    def __init__(
        self,
        time_window_minutes: int = 60,
        seasonal_periods: List[int] = None,
        trend_threshold: float = 0.05,
        seasonality_threshold: float = 0.3,
    ):
        """
        Initialize temporal analyzer

        Args:
            time_window_minutes: Time window for temporal analysis
            seasonal_periods: Seasonal periods to detect (in minutes)
            trend_threshold: Threshold for trend detection
            seasonality_threshold: Threshold for seasonality detection
        """
        self.time_window_minutes = time_window_minutes
        self.seasonal_periods = seasonal_periods or [
            60,
            1440,
            10080,
        ]  # 1hr, 1day, 1week in minutes
        self.trend_threshold = trend_threshold
        self.seasonality_threshold = seasonality_threshold

        # Temporal data storage
        self.time_series_data: Dict[str, List[Tuple[datetime, float]]] = defaultdict(
            list
        )
        self.temporal_baselines: Dict[str, Dict[str, Any]] = {}
        self.trend_history: Dict[str, List[float]] = defaultdict(list)

        # Temporal patterns
        self.hourly_patterns: Dict[str, List[float]] = defaultdict(lambda: [0.0] * 24)
        self.daily_patterns: Dict[str, List[float]] = defaultdict(lambda: [0.0] * 7)
        self.weekly_patterns: Dict[str, List[float]] = defaultdict(list)

    def detect_temporal_anomalies(
        self,
        data: Dict[str, Any],
        temporal_keys: List[str],
        timestamp: Optional[datetime] = None,
    ) -> List[AnomalyDetectionResult]:
        """
        Detect temporal anomalies in time series data

        Args:
            data: Current data point
            temporal_keys: Keys for temporal metrics to analyze
            timestamp: Timestamp for the data point

        Returns:
            List of detected temporal anomalies
        """
        try:
            if timestamp is None:
                timestamp = datetime.now()

            anomalies = []

            for key in temporal_keys:
                if key not in data or not isinstance(data[key], (int, float)):
                    continue

                value = float(data[key])

                # Add to time series
                self.time_series_data[key].append((timestamp, value))

                # Maintain window size
                cutoff_time = timestamp - timedelta(minutes=self.time_window_minutes)
                self.time_series_data[key] = [
                    (ts, val)
                    for ts, val in self.time_series_data[key]
                    if ts >= cutoff_time
                ]

                # Need sufficient data for temporal analysis
                if len(self.time_series_data[key]) < 10:
                    continue

                # Update temporal patterns
                self._update_temporal_patterns(key, value, timestamp)

                # Detect various temporal anomalies
                temporal_anomalies = []

                # 1. Trend anomaly detection
                trend_anomaly = self._detect_trend_anomaly(key, timestamp, value)
                if trend_anomaly:
                    temporal_anomalies.append(trend_anomaly)

                # 2. Seasonal anomaly detection
                seasonal_anomaly = self._detect_seasonal_anomaly(key, timestamp, value)
                if seasonal_anomaly:
                    temporal_anomalies.append(seasonal_anomaly)

                # 3. Cyclic pattern anomaly detection
                cyclic_anomaly = self._detect_cyclic_anomaly(key, timestamp, value)
                if cyclic_anomaly:
                    temporal_anomalies.append(cyclic_anomaly)

                # 4. Rate of change anomaly detection
                rate_anomaly = self._detect_rate_change_anomaly(key, timestamp, value)
                if rate_anomaly:
                    temporal_anomalies.append(rate_anomaly)

                # Convert to AnomalyDetectionResult
                for temporal_anomaly in temporal_anomalies:
                    anomaly = AnomalyDetectionResult(
                        anomaly_id=str(uuid.uuid4()),
                        anomaly_type=AnomalyType.TEMPORAL_ANOMALY,
                        severity=temporal_anomaly["severity"],
                        detection_method=DetectionMethod.TEMPORAL,
                        confidence_score=temporal_anomaly["confidence"],
                        detected_at=timestamp,
                        data_point={key: value, "timestamp": timestamp.isoformat()},
                        baseline_stats=temporal_anomaly["baseline_stats"],
                        deviation_metrics=temporal_anomaly["deviation_metrics"],
                        context={
                            "metric_key": key,
                            "temporal_type": temporal_anomaly["temporal_type"],
                            "time_window": self.time_window_minutes,
                        },
                        recommendations=temporal_anomaly["recommendations"],
                    )
                    anomalies.append(anomaly)

            return anomalies

        except Exception as e:
            logger.error(f"Temporal anomaly detection failed: {e}")
            return []

    def _update_temporal_patterns(self, key: str, value: float, timestamp: datetime):
        """Update temporal pattern baselines"""
        try:
            hour = timestamp.hour
            day_of_week = timestamp.weekday()

            # Update hourly pattern
            current_hourly = self.hourly_patterns[key][hour]
            self.hourly_patterns[key][hour] = current_hourly * 0.9 + value * 0.1

            # Update daily pattern
            current_daily = self.daily_patterns[key][day_of_week]
            self.daily_patterns[key][day_of_week] = current_daily * 0.9 + value * 0.1

            # Update baseline statistics
            if key not in self.temporal_baselines:
                self.temporal_baselines[key] = {
                    "hourly_mean": 0,
                    "daily_mean": 0,
                    "overall_mean": 0,
                    "overall_std": 0,
                }

            # Calculate overall statistics
            values = [val for _, val in self.time_series_data[key]]
            if values:
                self.temporal_baselines[key]["overall_mean"] = np.mean(values)
                self.temporal_baselines[key]["overall_std"] = np.std(values)
                self.temporal_baselines[key]["hourly_mean"] = np.mean(
                    self.hourly_patterns[key]
                )
                self.temporal_baselines[key]["daily_mean"] = np.mean(
                    self.daily_patterns[key]
                )

        except Exception as e:
            logger.warning(f"Failed to update temporal patterns: {e}")

    def _detect_trend_anomaly(
        self, key: str, timestamp: datetime, value: float
    ) -> Optional[Dict[str, Any]]:
        """Detect trend anomalies"""
        try:
            if len(self.time_series_data[key]) < 20:
                return None

            # Get recent values
            recent_values = [val for _, val in self.time_series_data[key][-20:]]

            # Calculate trend using linear regression
            x_values = list(range(len(recent_values)))

            # Simple linear regression
            n = len(recent_values)
            sum_x = sum(x_values)
            sum_y = sum(recent_values)
            sum_xy = sum(x * y for x, y in zip(x_values, recent_values))
            sum_x2 = sum(x * x for x in x_values)

            # Calculate slope (trend)
            denominator = n * sum_x2 - sum_x * sum_x
            if denominator == 0:
                return None

            slope = (n * sum_xy - sum_x * sum_y) / denominator

            # Store trend history
            self.trend_history[key].append(slope)
            if len(self.trend_history[key]) > 100:
                self.trend_history[key].pop(0)

            # Detect trend anomaly
            if len(self.trend_history[key]) < 10:
                return None

            historical_slopes = self.trend_history[key][:-1]
            mean_slope = np.mean(historical_slopes)
            std_slope = np.std(historical_slopes)

            if std_slope == 0:
                return None

            z_score = abs(slope - mean_slope) / std_slope

            if z_score > 2.0:  # Significant trend change
                severity = (
                    AnomalySeverity.HIGH if z_score > 3.0 else AnomalySeverity.MEDIUM
                )

                return {
                    "temporal_type": "trend_anomaly",
                    "severity": severity,
                    "confidence": min(1.0, z_score / 4.0),
                    "baseline_stats": {
                        "mean_slope": mean_slope,
                        "std_slope": std_slope,
                        "historical_trends": len(historical_slopes),
                    },
                    "deviation_metrics": {
                        "current_slope": slope,
                        "z_score": z_score,
                        "threshold": 2.0,
                    },
                    "recommendations": [
                        f"Significant trend change detected (slope: {slope:.4f})",
                        "Monitor for continued trend deviation",
                        "Consider adjusting capacity or resources",
                    ],
                }

            return None

        except Exception as e:
            logger.warning(f"Trend anomaly detection failed: {e}")
            return None

    def _detect_seasonal_anomaly(
        self, key: str, timestamp: datetime, value: float
    ) -> Optional[Dict[str, Any]]:
        """Detect seasonal anomalies"""
        try:
            hour = timestamp.hour
            day_of_week = timestamp.weekday()

            # Get expected values based on time
            expected_hourly = self.hourly_patterns[key][hour]
            expected_daily = self.daily_patterns[key][day_of_week]

            # Calculate deviations
            hourly_deviation = abs(value - expected_hourly) / (expected_hourly + 1e-8)
            daily_deviation = abs(value - expected_daily) / (expected_daily + 1e-8)

            # Check for significant seasonal deviations
            max_deviation = max(hourly_deviation, daily_deviation)

            if max_deviation > self.seasonality_threshold:
                severity = (
                    AnomalySeverity.HIGH
                    if max_deviation > 0.5
                    else AnomalySeverity.MEDIUM
                )

                return {
                    "temporal_type": "seasonal_anomaly",
                    "severity": severity,
                    "confidence": min(1.0, max_deviation),
                    "baseline_stats": {
                        "expected_hourly": expected_hourly,
                        "expected_daily": expected_daily,
                        "hour": hour,
                        "day_of_week": day_of_week,
                    },
                    "deviation_metrics": {
                        "hourly_deviation": hourly_deviation,
                        "daily_deviation": daily_deviation,
                        "max_deviation": max_deviation,
                        "threshold": self.seasonality_threshold,
                    },
                    "recommendations": [
                        "Value deviates from seasonal pattern",
                        "Check for unusual activity during this time period",
                        "Consider updating seasonal baselines",
                    ],
                }

            return None

        except Exception as e:
            logger.warning(f"Seasonal anomaly detection failed: {e}")
            return None

    def _detect_cyclic_anomaly(
        self, key: str, timestamp: datetime, value: float
    ) -> Optional[Dict[str, Any]]:
        """Detect cyclic pattern anomalies"""
        try:
            if len(self.time_series_data[key]) < 50:
                return None

            values = [val for _, val in self.time_series_data[key]]

            # Detect cycles using autocorrelation
            autocorrelations = []
            for lag in self.seasonal_periods:
                if lag < len(values):
                    autocorr = self._calculate_autocorrelation(values, lag)
                    autocorrelations.append(autocorr)

            # Check if current value breaks expected cyclic pattern
            if autocorrelations:
                max_autocorr = max(autocorrelations)

                # If we have strong cyclical patterns, check for deviations
                if max_autocorr > 0.5:
                    # Find the period with maximum autocorrelation
                    best_period = self.seasonal_periods[
                        autocorrelations.index(max_autocorr)
                    ]

                    if best_period < len(values):
                        expected_value = values[-best_period]
                        deviation = abs(value - expected_value) / (
                            expected_value + 1e-8
                        )

                        if deviation > 0.3:  # 30% deviation from cyclic pattern
                            severity = (
                                AnomalySeverity.MEDIUM
                                if deviation > 0.5
                                else AnomalySeverity.LOW
                            )

                            return {
                                "temporal_type": "cyclic_anomaly",
                                "severity": severity,
                                "confidence": min(1.0, deviation),
                                "baseline_stats": {
                                    "cycle_period": best_period,
                                    "autocorrelation": max_autocorr,
                                    "expected_value": expected_value,
                                },
                                "deviation_metrics": {
                                    "cyclic_deviation": deviation,
                                    "threshold": 0.3,
                                },
                                "recommendations": [
                                    f"Deviation from {best_period}-period cycle detected",
                                    "Monitor for cycle disruption",
                                    "Check for external factors affecting pattern",
                                ],
                            }

            return None

        except Exception as e:
            logger.warning(f"Cyclic anomaly detection failed: {e}")
            return None

    def _detect_rate_change_anomaly(
        self, key: str, timestamp: datetime, value: float
    ) -> Optional[Dict[str, Any]]:
        """Detect rate of change anomalies"""
        try:
            if len(self.time_series_data[key]) < 5:
                return None

            # Calculate recent rates of change
            recent_data = self.time_series_data[key][-5:]
            rates = []

            for i in range(1, len(recent_data)):
                prev_time, prev_value = recent_data[i - 1]
                curr_time, curr_value = recent_data[i]

                time_diff = (curr_time - prev_time).total_seconds()
                if time_diff > 0:
                    rate = (curr_value - prev_value) / time_diff
                    rates.append(rate)

            if not rates:
                return None

            current_rate = rates[-1]

            # Compare with historical rates
            if len(rates) > 1:
                historical_rates = rates[:-1]
                mean_rate = np.mean(historical_rates)
                std_rate = np.std(historical_rates)

                if std_rate > 0:
                    z_score = abs(current_rate - mean_rate) / std_rate

                    if z_score > 2.0:  # Significant rate change
                        severity = (
                            AnomalySeverity.HIGH
                            if z_score > 3.0
                            else AnomalySeverity.MEDIUM
                        )

                        return {
                            "temporal_type": "rate_change_anomaly",
                            "severity": severity,
                            "confidence": min(1.0, z_score / 4.0),
                            "baseline_stats": {
                                "mean_rate": mean_rate,
                                "std_rate": std_rate,
                                "rate_samples": len(historical_rates),
                            },
                            "deviation_metrics": {
                                "current_rate": current_rate,
                                "rate_z_score": z_score,
                                "threshold": 2.0,
                            },
                            "recommendations": [
                                f"Unusual rate of change detected: {current_rate:.4f}/sec",
                                "Monitor for acceleration or deceleration patterns",
                                "Check for sudden load changes",
                            ],
                        }

            return None

        except Exception as e:
            logger.warning(f"Rate change anomaly detection failed: {e}")
            return None

    def _calculate_autocorrelation(self, values: List[float], lag: int) -> float:
        """Calculate autocorrelation at given lag"""
        try:
            if lag >= len(values):
                return 0.0

            n = len(values) - lag
            if n <= 0:
                return 0.0

            mean_val = np.mean(values)

            # Calculate autocorrelation
            numerator = sum(
                (values[i] - mean_val) * (values[i + lag] - mean_val) for i in range(n)
            )
            denominator = sum((values[i] - mean_val) ** 2 for i in range(len(values)))

            if denominator == 0:
                return 0.0

            autocorr = numerator / denominator
            return autocorr

        except Exception:
            return 0.0


class ContextualAnalyzer:
    """Contextual anomaly detection analyzer (Feature 5)"""

    def __init__(
        self,
        context_similarity_threshold: float = 0.7,
        min_context_samples: int = 10,
        context_features: List[str] = None,
    ):
        """
        Initialize contextual analyzer

        Args:
            context_similarity_threshold: Threshold for context similarity
            min_context_samples: Minimum samples needed for context analysis
            context_features: Features to use for context analysis
        """
        self.context_similarity_threshold = context_similarity_threshold
        self.min_context_samples = min_context_samples
        self.context_features = context_features or [
            "user_type",
            "document_type",
            "time_of_day",
            "day_of_week",
            "processing_mode",
            "document_size",
            "complexity_score",
        ]

        # Context storage
        self.context_profiles: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self.context_statistics: Dict[str, Dict[str, Any]] = {}

    def detect_contextual_anomalies(
        self, data: Dict[str, Any], context_data: Dict[str, Any]
    ) -> List[AnomalyDetectionResult]:
        """
        Detect contextual anomalies based on situational context

        Args:
            data: Current data point to analyze
            context_data: Contextual information

        Returns:
            List of detected contextual anomalies
        """
        try:
            anomalies = []
            current_time = datetime.now()

            # Extract context signature
            context_signature = self._extract_context_signature(context_data)

            if not context_signature:
                return anomalies

            # Store current context and data
            context_record = {
                "data": data.copy(),
                "context": context_data.copy(),
                "timestamp": current_time,
            }
            self.context_profiles[context_signature].append(context_record)

            # Limit context history
            if len(self.context_profiles[context_signature]) > 1000:
                self.context_profiles[context_signature] = self.context_profiles[
                    context_signature
                ][-1000:]

            # Need sufficient samples for analysis
            if len(self.context_profiles[context_signature]) < self.min_context_samples:
                return anomalies

            # Update context statistics
            self._update_context_statistics(context_signature)

            # Detect contextual anomalies
            contextual_anomalies = []

            # 1. Value-in-context anomaly
            value_anomaly = self._detect_value_in_context_anomaly(
                data, context_signature
            )
            if value_anomaly:
                contextual_anomalies.append(value_anomaly)

            # 2. Behavior-in-context anomaly
            behavior_anomaly = self._detect_behavior_in_context_anomaly(
                data, context_data, context_signature
            )
            if behavior_anomaly:
                contextual_anomalies.append(behavior_anomaly)

            # 3. Cross-context comparison anomaly
            cross_context_anomaly = self._detect_cross_context_anomaly(
                data, context_data
            )
            if cross_context_anomaly:
                contextual_anomalies.append(cross_context_anomaly)

            # Convert to AnomalyDetectionResult
            for contextual_anomaly in contextual_anomalies:
                anomaly = AnomalyDetectionResult(
                    anomaly_id=str(uuid.uuid4()),
                    anomaly_type=AnomalyType.CONTEXTUAL_ANOMALY,
                    severity=contextual_anomaly["severity"],
                    detection_method=DetectionMethod.CONTEXTUAL,
                    confidence_score=contextual_anomaly["confidence"],
                    detected_at=current_time,
                    data_point=data,
                    baseline_stats=contextual_anomaly["baseline_stats"],
                    deviation_metrics=contextual_anomaly["deviation_metrics"],
                    context={
                        "context_signature": context_signature,
                        "contextual_type": contextual_anomaly["contextual_type"],
                        "context_data": context_data,
                    },
                    recommendations=contextual_anomaly["recommendations"],
                )
                anomalies.append(anomaly)

            return anomalies

        except Exception as e:
            logger.error(f"Contextual anomaly detection failed: {e}")
            return []

    def _extract_context_signature(self, context_data: Dict[str, Any]) -> Optional[str]:
        """Extract a signature representing the context"""
        try:
            signature_parts = []

            for feature in self.context_features:
                if feature in context_data:
                    value = context_data[feature]

                    # Normalize value for signature
                    if isinstance(value, str):
                        signature_parts.append(f"{feature}:{value}")
                    elif isinstance(value, (int, float)):
                        # Bin numerical values
                        if feature == "time_of_day":
                            # Convert to hour bins
                            hour_bin = int(value) // 4  # 6-hour bins
                            signature_parts.append(f"{feature}:{hour_bin}")
                        elif feature == "document_size":
                            # Size bins: small, medium, large
                            if value < 1000:
                                size_bin = "small"
                            elif value < 10000:
                                size_bin = "medium"
                            else:
                                size_bin = "large"
                            signature_parts.append(f"{feature}:{size_bin}")
                        else:
                            # General numerical binning
                            bin_val = int(value * 10) / 10  # Round to 1 decimal
                            signature_parts.append(f"{feature}:{bin_val}")

            if signature_parts:
                return "|".join(sorted(signature_parts))

            return None

        except Exception:
            return None

    def _update_context_statistics(self, context_signature: str):
        """Update statistics for the given context"""
        try:
            if context_signature not in self.context_statistics:
                self.context_statistics[context_signature] = {}

            context_records = self.context_profiles[context_signature]

            # Calculate statistics for each data field
            data_fields = set()
            for record in context_records:
                data_fields.update(record["data"].keys())

            stats = {}
            for field in data_fields:
                values = []
                for record in context_records:
                    if field in record["data"] and isinstance(
                        record["data"][field], (int, float)
                    ):
                        values.append(float(record["data"][field]))

                if values:
                    stats[field] = {
                        "mean": np.mean(values),
                        "std": np.std(values),
                        "min": min(values),
                        "max": max(values),
                        "count": len(values),
                    }

            self.context_statistics[context_signature] = stats

        except Exception as e:
            logger.warning(f"Failed to update context statistics: {e}")

    def _detect_value_in_context_anomaly(
        self, data: Dict[str, Any], context_signature: str
    ) -> Optional[Dict[str, Any]]:
        """Detect when values are anomalous given the context"""
        try:
            if context_signature not in self.context_statistics:
                return None

            stats = self.context_statistics[context_signature]
            anomaly_scores = []
            anomalous_fields = []

            for field, value in data.items():
                if (
                    isinstance(value, (int, float))
                    and field in stats
                    and stats[field]["std"] > 0
                ):
                    z_score = abs(value - stats[field]["mean"]) / stats[field]["std"]

                    if z_score > 2.5:  # Anomalous in this context
                        anomaly_scores.append(z_score)
                        anomalous_fields.append(
                            {
                                "field": field,
                                "value": value,
                                "z_score": z_score,
                                "context_mean": stats[field]["mean"],
                                "context_std": stats[field]["std"],
                            }
                        )

            if anomalous_fields:
                max_z_score = max(anomaly_scores)
                severity = (
                    AnomalySeverity.HIGH
                    if max_z_score > 4.0
                    else AnomalySeverity.MEDIUM
                )

                return {
                    "contextual_type": "value_in_context_anomaly",
                    "severity": severity,
                    "confidence": min(1.0, max_z_score / 5.0),
                    "baseline_stats": stats,
                    "deviation_metrics": {
                        "anomalous_fields": anomalous_fields,
                        "max_z_score": max_z_score,
                        "threshold": 2.5,
                    },
                    "recommendations": [
                        f"Values anomalous for context: {', '.join(f['field'] for f in anomalous_fields)}",
                        "Review data collection in this context",
                        "Consider context-specific thresholds",
                    ],
                }

            return None

        except Exception as e:
            logger.warning(f"Value-in-context anomaly detection failed: {e}")
            return None

    def _detect_behavior_in_context_anomaly(
        self, data: Dict[str, Any], context_data: Dict[str, Any], context_signature: str
    ) -> Optional[Dict[str, Any]]:
        """Detect behavioral anomalies in context"""
        try:
            # Analyze behavioral patterns in this context
            context_records = self.context_profiles[context_signature]

            if len(context_records) < 20:
                return None

            # Look for behavioral indicators
            behavior_indicators = {
                "processing_time": data.get("processing_time_ms", 0),
                "success_rate": data.get("success_rate", 1.0),
                "error_count": data.get("error_count", 0),
                "resource_usage": data.get("memory_usage_mb", 0),
            }

            # Compare with historical behavior in this context
            historical_behaviors = []
            for record in context_records[:-1]:  # Exclude current record
                record_behavior = {
                    "processing_time": record["data"].get("processing_time_ms", 0),
                    "success_rate": record["data"].get("success_rate", 1.0),
                    "error_count": record["data"].get("error_count", 0),
                    "resource_usage": record["data"].get("memory_usage_mb", 0),
                }
                historical_behaviors.append(record_behavior)

            # Calculate behavior deviation
            behavior_deviations = []
            for indicator, current_value in behavior_indicators.items():
                if isinstance(current_value, (int, float)):
                    historical_values = [
                        b[indicator]
                        for b in historical_behaviors
                        if isinstance(b[indicator], (int, float))
                    ]

                    if len(historical_values) > 5:
                        mean_val = np.mean(historical_values)
                        std_val = np.std(historical_values)

                        if std_val > 0:
                            deviation = abs(current_value - mean_val) / std_val
                            if deviation > 2.0:
                                behavior_deviations.append(
                                    {
                                        "indicator": indicator,
                                        "current_value": current_value,
                                        "expected_value": mean_val,
                                        "deviation": deviation,
                                    }
                                )

            if behavior_deviations:
                max_deviation = max(d["deviation"] for d in behavior_deviations)
                severity = (
                    AnomalySeverity.HIGH
                    if max_deviation > 3.0
                    else AnomalySeverity.MEDIUM
                )

                return {
                    "contextual_type": "behavior_in_context_anomaly",
                    "severity": severity,
                    "confidence": min(1.0, max_deviation / 4.0),
                    "baseline_stats": {
                        "context_samples": len(historical_behaviors),
                        "context_signature": context_signature,
                    },
                    "deviation_metrics": {
                        "behavior_deviations": behavior_deviations,
                        "max_deviation": max_deviation,
                        "threshold": 2.0,
                    },
                    "recommendations": [
                        "Behavioral pattern anomaly detected in context",
                        f"Unusual behavior in indicators: {', '.join(d['indicator'] for d in behavior_deviations)}",
                        "Monitor for context-specific performance issues",
                    ],
                }

            return None

        except Exception as e:
            logger.warning(f"Behavior-in-context anomaly detection failed: {e}")
            return None

    def _detect_cross_context_anomaly(
        self, data: Dict[str, Any], context_data: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Detect anomalies by comparing across different contexts"""
        try:
            # Compare current data with similar contexts
            similar_contexts = self._find_similar_contexts(context_data)

            if len(similar_contexts) < 3:
                return None

            cross_context_deviations = []

            for field, value in data.items():
                if isinstance(value, (int, float)):
                    # Collect values from similar contexts
                    similar_values = []
                    for ctx_sig in similar_contexts:
                        for record in self.context_profiles[ctx_sig]:
                            if field in record["data"] and isinstance(
                                record["data"][field], (int, float)
                            ):
                                similar_values.append(record["data"][field])

                    if len(similar_values) > 10:
                        mean_val = np.mean(similar_values)
                        std_val = np.std(similar_values)

                        if std_val > 0:
                            z_score = abs(value - mean_val) / std_val

                            if z_score > 2.0:
                                cross_context_deviations.append(
                                    {
                                        "field": field,
                                        "value": value,
                                        "cross_context_mean": mean_val,
                                        "cross_context_std": std_val,
                                        "z_score": z_score,
                                    }
                                )

            if cross_context_deviations:
                max_z_score = max(d["z_score"] for d in cross_context_deviations)
                severity = (
                    AnomalySeverity.MEDIUM if max_z_score > 3.0 else AnomalySeverity.LOW
                )

                return {
                    "contextual_type": "cross_context_anomaly",
                    "severity": severity,
                    "confidence": min(1.0, max_z_score / 4.0),
                    "baseline_stats": {
                        "similar_contexts": len(similar_contexts),
                        "cross_context_samples": len(similar_values)
                        if similar_values
                        else 0,
                    },
                    "deviation_metrics": {
                        "cross_context_deviations": cross_context_deviations,
                        "max_z_score": max_z_score,
                        "threshold": 2.0,
                    },
                    "recommendations": [
                        "Value anomalous compared to similar contexts",
                        "Context may have unique characteristics",
                        "Consider context-specific baselines",
                    ],
                }

            return None

        except Exception as e:
            logger.warning(f"Cross-context anomaly detection failed: {e}")
            return None

    def _find_similar_contexts(self, context_data: Dict[str, Any]) -> List[str]:
        """Find contexts similar to the current one"""
        try:
            similar_contexts = []

            for ctx_sig in self.context_profiles.keys():
                similarity = self._calculate_context_similarity(context_data, ctx_sig)

                if similarity >= self.context_similarity_threshold:
                    similar_contexts.append(ctx_sig)

            return similar_contexts

        except Exception:
            return []

    def _calculate_context_similarity(
        self, context_data: Dict[str, Any], context_signature: str
    ) -> float:
        """Calculate similarity between current context and a stored context"""
        try:
            # Extract features from context signature
            sig_features = {}
            for part in context_signature.split("|"):
                if ":" in part:
                    key, value = part.split(":", 1)
                    sig_features[key] = value

            # Calculate similarity based on matching features
            matches = 0
            total_features = len(self.context_features)

            for feature in self.context_features:
                if feature in context_data and feature in sig_features:
                    current_value = context_data[feature]
                    sig_value = sig_features[feature]

                    if isinstance(current_value, str):
                        if current_value == sig_value:
                            matches += 1
                    elif isinstance(current_value, (int, float)):
                        # For numerical values, check if they fall in same bin
                        current_sig = self._extract_context_signature(
                            {feature: current_value}
                        )
                        if current_sig and f"{feature}:{sig_value}" in current_sig:
                            matches += 1

            similarity = matches / total_features if total_features > 0 else 0
            return similarity

        except Exception:
            return 0.0
