"""
Real-Time Document Intelligence & Compliance Engine
Severity Scoring & Alert Generation System
Version: 1.0
Date: 2026-02-03

This module provides severity scoring (Feature 6) and alert generation (Feature 7)
for comprehensive anomaly response capabilities.
"""

import logging
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Set
from datetime import datetime, timedelta
from collections import defaultdict, deque
import json
from dataclasses import dataclass
from enum import Enum
from .anomaly_engine import (
    AnomalyDetectionResult,
    AnomalyType,
    AnomalySeverity,
    DetectionMethod,
)
import uuid

logger = logging.getLogger(__name__)


@dataclass
class SeverityFactors:
    """Factors contributing to severity calculation"""

    business_impact: float  # 0.0-1.0
    urgency_factor: float  # 0.0-1.0
    scope_factor: float  # 0.0-1.0
    frequency_factor: float  # 0.0-1.0
    confidence_factor: float  # 0.0-1.0
    historical_factor: float  # 0.0-1.0


@dataclass
class AlertRule:
    """Alert configuration rule"""

    rule_id: str
    name: str
    severity_threshold: AnomalySeverity
    anomaly_types: List[AnomalyType]
    frequency_limit: int  # Max alerts per time window
    time_window_minutes: int
    escalation_enabled: bool
    escalation_delay_minutes: int
    notification_channels: List[str]
    conditions: Dict[str, Any]
    active: bool = True


@dataclass
class Alert:
    """Alert instance"""

    alert_id: str
    rule_id: str
    anomaly_id: str
    severity: AnomalySeverity
    title: str
    description: str
    created_at: datetime
    escalated: bool = False
    acknowledged: bool = False
    resolved: bool = False
    metadata: Dict[str, Any] = None


class SeverityScorer:
    """Advanced severity scoring system (Feature 6)"""

    def __init__(
        self,
        business_impact_weights: Dict[str, float] = None,
        severity_thresholds: Dict[str, float] = None,
    ):
        """
        Initialize severity scorer

        Args:
            business_impact_weights: Weights for business impact factors
            severity_thresholds: Thresholds for severity levels
        """
        self.business_impact_weights = business_impact_weights or {
            "data_integrity": 0.25,
            "compliance_risk": 0.30,
            "operational_impact": 0.20,
            "financial_risk": 0.15,
            "reputation_risk": 0.10,
        }

        self.severity_thresholds = severity_thresholds or {
            "low": 0.3,
            "medium": 0.6,
            "high": 0.8,
            "critical": 0.95,
        }

        # Historical severity tracking
        self.severity_history: Dict[str, List[Tuple[datetime, float]]] = defaultdict(
            list
        )
        self.anomaly_frequency: Dict[AnomalyType, List[datetime]] = defaultdict(list)

        # Business context mapping
        self.business_context_map = {
            "contract_execution": {"urgency": 0.9, "impact": 0.8},
            "compliance_audit": {"urgency": 0.95, "impact": 0.9},
            "financial_reporting": {"urgency": 0.8, "impact": 0.85},
            "risk_assessment": {"urgency": 0.7, "impact": 0.75},
            "routine_processing": {"urgency": 0.3, "impact": 0.4},
        }

    def calculate_severity_score(
        self, anomaly: AnomalyDetectionResult, business_context: Dict[str, Any] = None
    ) -> Tuple[float, SeverityFactors]:
        """
        Calculate comprehensive severity score for anomaly

        Args:
            anomaly: Detected anomaly result
            business_context: Business context information

        Returns:
            Tuple of (severity_score, severity_factors)
        """
        try:
            current_time = datetime.now()

            # Calculate individual severity factors
            business_impact = self._calculate_business_impact(anomaly, business_context)
            urgency_factor = self._calculate_urgency_factor(anomaly, business_context)
            scope_factor = self._calculate_scope_factor(anomaly)
            frequency_factor = self._calculate_frequency_factor(anomaly, current_time)
            confidence_factor = self._normalize_confidence(anomaly.confidence_score)
            historical_factor = self._calculate_historical_factor(anomaly, current_time)

            factors = SeverityFactors(
                business_impact=business_impact,
                urgency_factor=urgency_factor,
                scope_factor=scope_factor,
                frequency_factor=frequency_factor,
                confidence_factor=confidence_factor,
                historical_factor=historical_factor,
            )

            # Calculate weighted severity score
            severity_score = self._compute_weighted_severity(factors)

            # Store in history
            anomaly_key = (
                f"{anomaly.anomaly_type.value}_{anomaly.detection_method.value}"
            )
            self.severity_history[anomaly_key].append((current_time, severity_score))

            # Limit history size
            if len(self.severity_history[anomaly_key]) > 1000:
                self.severity_history[anomaly_key] = self.severity_history[anomaly_key][
                    -1000:
                ]

            return severity_score, factors

        except Exception as e:
            logger.error(f"Severity score calculation failed: {e}")
            return 0.5, SeverityFactors(0.5, 0.5, 0.5, 0.5, 0.5, 0.5)

    def _calculate_business_impact(
        self, anomaly: AnomalyDetectionResult, business_context: Dict[str, Any] = None
    ) -> float:
        """Calculate business impact factor"""
        try:
            impact_score = 0.0

            # Base impact from anomaly type
            type_impact_map = {
                AnomalyType.STATISTICAL_ANOMALY: 0.4,
                AnomalyType.PATTERN_ANOMALY: 0.5,
                AnomalyType.OUTLIER_ANOMALY: 0.6,
                AnomalyType.TEMPORAL_ANOMALY: 0.7,
                AnomalyType.CONTEXTUAL_ANOMALY: 0.8,
            }
            impact_score += type_impact_map.get(anomaly.anomaly_type, 0.5) * 0.3

            # Severity-based impact
            severity_impact_map = {
                AnomalySeverity.LOW: 0.3,
                AnomalySeverity.MEDIUM: 0.6,
                AnomalySeverity.HIGH: 0.8,
                AnomalySeverity.CRITICAL: 1.0,
            }
            impact_score += severity_impact_map.get(anomaly.severity, 0.5) * 0.4

            # Business context impact
            if business_context:
                context_type = business_context.get(
                    "process_type", "routine_processing"
                )
                context_impact = self.business_context_map.get(
                    context_type, {"impact": 0.5}
                )["impact"]
                impact_score += context_impact * 0.3
            else:
                impact_score += 0.5 * 0.3

            return min(1.0, impact_score)

        except Exception:
            return 0.5

    def _calculate_urgency_factor(
        self, anomaly: AnomalyDetectionResult, business_context: Dict[str, Any] = None
    ) -> float:
        """Calculate urgency factor"""
        try:
            urgency_score = 0.0

            # Time-based urgency
            time_since_detection = (
                datetime.now() - anomaly.detected_at
            ).total_seconds() / 3600
            if time_since_detection < 1:  # Within 1 hour
                urgency_score += 0.8
            elif time_since_detection < 4:  # Within 4 hours
                urgency_score += 0.6
            else:
                urgency_score += 0.3

            urgency_score *= 0.4

            # Business context urgency
            if business_context:
                context_type = business_context.get(
                    "process_type", "routine_processing"
                )
                context_urgency = self.business_context_map.get(
                    context_type, {"urgency": 0.5}
                )["urgency"]
                urgency_score += context_urgency * 0.4
            else:
                urgency_score += 0.5 * 0.4

            # Detection confidence urgency
            confidence_urgency = anomaly.confidence_score * 0.2
            urgency_score += confidence_urgency

            return min(1.0, urgency_score)

        except Exception:
            return 0.5

    def _calculate_scope_factor(self, anomaly: AnomalyDetectionResult) -> float:
        """Calculate scope/reach factor"""
        try:
            scope_score = 0.5  # Base scope

            # Check if anomaly affects multiple systems/processes
            context = anomaly.context or {}

            # Multi-metric anomalies have higher scope
            if isinstance(anomaly.data_point, dict) and len(anomaly.data_point) > 1:
                scope_score += 0.2

            # Temporal anomalies often have broader scope
            if anomaly.anomaly_type == AnomalyType.TEMPORAL_ANOMALY:
                scope_score += 0.15

            # Contextual anomalies may indicate systemic issues
            if anomaly.anomaly_type == AnomalyType.CONTEXTUAL_ANOMALY:
                scope_score += 0.25

            # High deviation indicates broader impact
            deviation_metrics = anomaly.deviation_metrics or {}
            max_deviation = 0
            for key, value in deviation_metrics.items():
                if isinstance(value, (int, float)) and "deviation" in key.lower():
                    max_deviation = max(max_deviation, abs(value))

            if max_deviation > 3:  # High deviation
                scope_score += 0.1

            return min(1.0, scope_score)

        except Exception:
            return 0.5

    def _calculate_frequency_factor(
        self, anomaly: AnomalyDetectionResult, current_time: datetime
    ) -> float:
        """Calculate frequency factor"""
        try:
            # Track anomaly frequency
            self.anomaly_frequency[anomaly.anomaly_type].append(current_time)

            # Clean old entries (keep last 24 hours)
            cutoff_time = current_time - timedelta(hours=24)
            self.anomaly_frequency[anomaly.anomaly_type] = [
                ts
                for ts in self.anomaly_frequency[anomaly.anomaly_type]
                if ts >= cutoff_time
            ]

            # Calculate frequency score
            recent_count = len(self.anomaly_frequency[anomaly.anomaly_type])

            if recent_count <= 1:
                return 0.3  # Rare anomaly - lower urgency
            elif recent_count <= 5:
                return 0.5  # Occasional anomaly
            elif recent_count <= 15:
                return 0.7  # Frequent anomaly - higher urgency
            else:
                return 0.9  # Very frequent - potential systemic issue

        except Exception:
            return 0.5

    def _normalize_confidence(self, confidence: float) -> float:
        """Normalize confidence score"""
        return max(0.0, min(1.0, confidence))

    def _calculate_historical_factor(
        self, anomaly: AnomalyDetectionResult, current_time: datetime
    ) -> float:
        """Calculate historical severity factor"""
        try:
            anomaly_key = (
                f"{anomaly.anomaly_type.value}_{anomaly.detection_method.value}"
            )

            if anomaly_key not in self.severity_history:
                return 0.5

            # Get recent severity history
            cutoff_time = current_time - timedelta(days=7)
            recent_severities = [
                score
                for ts, score in self.severity_history[anomaly_key]
                if ts >= cutoff_time
            ]

            if not recent_severities:
                return 0.5

            # Calculate trend
            if len(recent_severities) >= 3:
                # Check if severity is trending upward
                recent_trend = np.mean(recent_severities[-3:]) - np.mean(
                    recent_severities[:-3]
                )
                if recent_trend > 0.1:  # Increasing severity trend
                    return 0.8
                elif recent_trend < -0.1:  # Decreasing severity trend
                    return 0.3

            # Return average historical severity
            return np.mean(recent_severities)

        except Exception:
            return 0.5

    def _compute_weighted_severity(self, factors: SeverityFactors) -> float:
        """Compute final weighted severity score"""
        try:
            weights = {
                "business_impact": 0.25,
                "urgency_factor": 0.20,
                "scope_factor": 0.15,
                "frequency_factor": 0.15,
                "confidence_factor": 0.15,
                "historical_factor": 0.10,
            }

            severity_score = (
                factors.business_impact * weights["business_impact"]
                + factors.urgency_factor * weights["urgency_factor"]
                + factors.scope_factor * weights["scope_factor"]
                + factors.frequency_factor * weights["frequency_factor"]
                + factors.confidence_factor * weights["confidence_factor"]
                + factors.historical_factor * weights["historical_factor"]
            )

            return max(0.0, min(1.0, severity_score))

        except Exception:
            return 0.5

    def get_severity_level(self, severity_score: float) -> AnomalySeverity:
        """Convert severity score to severity level"""
        if severity_score >= self.severity_thresholds["critical"]:
            return AnomalySeverity.CRITICAL
        elif severity_score >= self.severity_thresholds["high"]:
            return AnomalySeverity.HIGH
        elif severity_score >= self.severity_thresholds["medium"]:
            return AnomalySeverity.MEDIUM
        else:
            return AnomalySeverity.LOW


class AlertManager:
    """Intelligent alert generation and management system (Feature 7)"""

    def __init__(
        self, default_rules: List[AlertRule] = None, alert_history_days: int = 30
    ):
        """
        Initialize alert manager

        Args:
            default_rules: Default alert rules
            alert_history_days: Days to keep alert history
        """
        self.alert_rules: Dict[str, AlertRule] = {}
        self.alert_history: List[Alert] = []
        self.alert_frequency_tracking: Dict[str, List[datetime]] = defaultdict(list)
        self.alert_history_days = alert_history_days

        # Initialize with default rules
        if default_rules:
            for rule in default_rules:
                self.alert_rules[rule.rule_id] = rule
        else:
            self._create_default_rules()

        self.severity_scorer = SeverityScorer()

        # Notification channels
        self.notification_channels = {
            "email": self._send_email_notification,
            "slack": self._send_slack_notification,
            "webhook": self._send_webhook_notification,
            "dashboard": self._send_dashboard_notification,
        }

    def _create_default_rules(self):
        """Create default alert rules"""
        default_rules = [
            AlertRule(
                rule_id="critical_anomaly",
                name="Critical Anomaly Alert",
                severity_threshold=AnomalySeverity.CRITICAL,
                anomaly_types=list(AnomalyType),
                frequency_limit=1,
                time_window_minutes=60,
                escalation_enabled=True,
                escalation_delay_minutes=15,
                notification_channels=["email", "slack", "webhook"],
                conditions={"confidence_threshold": 0.8},
            ),
            AlertRule(
                rule_id="high_severity_batch",
                name="High Severity Batch Alert",
                severity_threshold=AnomalySeverity.HIGH,
                anomaly_types=list(AnomalyType),
                frequency_limit=3,
                time_window_minutes=30,
                escalation_enabled=True,
                escalation_delay_minutes=30,
                notification_channels=["email", "slack"],
                conditions={"batch_size": 3, "confidence_threshold": 0.7},
            ),
            AlertRule(
                rule_id="temporal_pattern_alert",
                name="Temporal Pattern Anomaly",
                severity_threshold=AnomalySeverity.MEDIUM,
                anomaly_types=[AnomalyType.TEMPORAL_ANOMALY],
                frequency_limit=5,
                time_window_minutes=120,
                escalation_enabled=False,
                escalation_delay_minutes=0,
                notification_channels=["dashboard"],
                conditions={"pattern_type": "trend_anomaly"},
            ),
        ]

        for rule in default_rules:
            self.alert_rules[rule.rule_id] = rule

    def process_anomaly(
        self, anomaly: AnomalyDetectionResult, business_context: Dict[str, Any] = None
    ) -> List[Alert]:
        """
        Process anomaly and generate alerts based on rules

        Args:
            anomaly: Detected anomaly
            business_context: Business context for severity calculation

        Returns:
            List of generated alerts
        """
        try:
            current_time = datetime.now()
            generated_alerts = []

            # Calculate enhanced severity
            severity_score, severity_factors = (
                self.severity_scorer.calculate_severity_score(anomaly, business_context)
            )

            # Update anomaly severity if needed
            enhanced_severity = self.severity_scorer.get_severity_level(severity_score)

            # Check each alert rule
            for rule in self.alert_rules.values():
                if not rule.active:
                    continue

                # Check if rule applies to this anomaly
                if self._should_trigger_alert(
                    anomaly, enhanced_severity, rule, current_time
                ):
                    alert = self._create_alert(
                        anomaly,
                        rule,
                        enhanced_severity,
                        severity_score,
                        severity_factors,
                    )
                    generated_alerts.append(alert)

                    # Send notifications
                    self._send_alert_notifications(alert, rule)

                    # Track alert frequency
                    self.alert_frequency_tracking[rule.rule_id].append(current_time)

            # Clean up old tracking data
            self._cleanup_alert_tracking(current_time)

            # Store alerts in history
            self.alert_history.extend(generated_alerts)

            return generated_alerts

        except Exception as e:
            logger.error(f"Alert processing failed: {e}")
            return []

    def _should_trigger_alert(
        self,
        anomaly: AnomalyDetectionResult,
        severity: AnomalySeverity,
        rule: AlertRule,
        current_time: datetime,
    ) -> bool:
        """Determine if alert rule should trigger"""
        try:
            # Check severity threshold
            severity_order = {
                AnomalySeverity.LOW: 1,
                AnomalySeverity.MEDIUM: 2,
                AnomalySeverity.HIGH: 3,
                AnomalySeverity.CRITICAL: 4,
            }

            if severity_order.get(severity, 1) < severity_order.get(
                rule.severity_threshold, 1
            ):
                return False

            # Check anomaly type
            if anomaly.anomaly_type not in rule.anomaly_types:
                return False

            # Check frequency limits
            cutoff_time = current_time - timedelta(minutes=rule.time_window_minutes)
            recent_alerts = [
                ts
                for ts in self.alert_frequency_tracking[rule.rule_id]
                if ts >= cutoff_time
            ]

            if len(recent_alerts) >= rule.frequency_limit:
                return False

            # Check custom conditions
            if not self._check_rule_conditions(anomaly, rule):
                return False

            return True

        except Exception:
            return False

    def _check_rule_conditions(
        self, anomaly: AnomalyDetectionResult, rule: AlertRule
    ) -> bool:
        """Check custom rule conditions"""
        try:
            conditions = rule.conditions or {}

            # Check confidence threshold
            if "confidence_threshold" in conditions:
                if anomaly.confidence_score < conditions["confidence_threshold"]:
                    return False

            # Check pattern type for temporal anomalies
            if (
                "pattern_type" in conditions
                and anomaly.anomaly_type == AnomalyType.TEMPORAL_ANOMALY
            ):
                context = anomaly.context or {}
                if context.get("temporal_type") != conditions["pattern_type"]:
                    return False

            # Additional custom conditions can be added here

            return True

        except Exception:
            return True

    def _create_alert(
        self,
        anomaly: AnomalyDetectionResult,
        rule: AlertRule,
        severity: AnomalySeverity,
        severity_score: float,
        severity_factors: SeverityFactors,
    ) -> Alert:
        """Create alert instance"""
        alert = Alert(
            alert_id=str(uuid.uuid4()),
            rule_id=rule.rule_id,
            anomaly_id=anomaly.anomaly_id,
            severity=severity,
            title=self._generate_alert_title(anomaly, rule),
            description=self._generate_alert_description(
                anomaly, severity_score, severity_factors
            ),
            created_at=datetime.now(),
            metadata={
                "anomaly_type": anomaly.anomaly_type.value,
                "detection_method": anomaly.detection_method.value,
                "confidence_score": anomaly.confidence_score,
                "severity_score": severity_score,
                "severity_factors": {
                    "business_impact": severity_factors.business_impact,
                    "urgency_factor": severity_factors.urgency_factor,
                    "scope_factor": severity_factors.scope_factor,
                    "frequency_factor": severity_factors.frequency_factor,
                    "confidence_factor": severity_factors.confidence_factor,
                    "historical_factor": severity_factors.historical_factor,
                },
                "baseline_stats": anomaly.baseline_stats,
                "deviation_metrics": anomaly.deviation_metrics,
                "context": anomaly.context,
            },
        )

        return alert

    def _generate_alert_title(
        self, anomaly: AnomalyDetectionResult, rule: AlertRule
    ) -> str:
        """Generate alert title"""
        return f"{rule.name}: {anomaly.anomaly_type.value.replace('_', ' ').title()} Detected"

    def _generate_alert_description(
        self,
        anomaly: AnomalyDetectionResult,
        severity_score: float,
        severity_factors: SeverityFactors,
    ) -> str:
        """Generate detailed alert description"""
        description = f"""
Anomaly Detection Alert

Type: {anomaly.anomaly_type.value.replace("_", " ").title()}
Method: {anomaly.detection_method.value.replace("_", " ").title()}
Confidence: {anomaly.confidence_score:.2f}
Severity Score: {severity_score:.2f}

Severity Factors:
- Business Impact: {severity_factors.business_impact:.2f}
- Urgency: {severity_factors.urgency_factor:.2f}
- Scope: {severity_factors.scope_factor:.2f}
- Frequency: {severity_factors.frequency_factor:.2f}
- Confidence: {severity_factors.confidence_factor:.2f}
- Historical: {severity_factors.historical_factor:.2f}

Detected At: {anomaly.detected_at.strftime("%Y-%m-%d %H:%M:%S")}

Recommendations:
{chr(10).join(f"- {rec}" for rec in anomaly.recommendations)}
        """.strip()

        return description

    def _send_alert_notifications(self, alert: Alert, rule: AlertRule):
        """Send alert notifications through configured channels"""
        try:
            for channel in rule.notification_channels:
                if channel in self.notification_channels:
                    self.notification_channels[channel](alert, rule)

        except Exception as e:
            logger.error(f"Alert notification failed: {e}")

    def _send_email_notification(self, alert: Alert, rule: AlertRule):
        """Send email notification (placeholder)"""
        logger.info(f"EMAIL ALERT: {alert.title} (Severity: {alert.severity.value})")

    def _send_slack_notification(self, alert: Alert, rule: AlertRule):
        """Send Slack notification (placeholder)"""
        logger.info(f"SLACK ALERT: {alert.title} (Severity: {alert.severity.value})")

    def _send_webhook_notification(self, alert: Alert, rule: AlertRule):
        """Send webhook notification (placeholder)"""
        logger.info(f"WEBHOOK ALERT: {alert.title} (Severity: {alert.severity.value})")

    def _send_dashboard_notification(self, alert: Alert, rule: AlertRule):
        """Send dashboard notification (placeholder)"""
        logger.info(
            f"DASHBOARD ALERT: {alert.title} (Severity: {alert.severity.value})"
        )

    def _cleanup_alert_tracking(self, current_time: datetime):
        """Clean up old alert tracking data"""
        try:
            cutoff_time = current_time - timedelta(days=self.alert_history_days)

            # Clean frequency tracking
            for rule_id in self.alert_frequency_tracking:
                self.alert_frequency_tracking[rule_id] = [
                    ts
                    for ts in self.alert_frequency_tracking[rule_id]
                    if ts >= cutoff_time
                ]

            # Clean alert history
            self.alert_history = [
                alert for alert in self.alert_history if alert.created_at >= cutoff_time
            ]

        except Exception as e:
            logger.warning(f"Alert cleanup failed: {e}")

    def get_alert_statistics(self) -> Dict[str, Any]:
        """Get alert system statistics"""
        try:
            current_time = datetime.now()

            # Count alerts by severity in last 24 hours
            cutoff_24h = current_time - timedelta(hours=24)
            recent_alerts = [
                alert for alert in self.alert_history if alert.created_at >= cutoff_24h
            ]

            severity_counts = defaultdict(int)
            for alert in recent_alerts:
                severity_counts[alert.severity.value] += 1

            # Calculate resolution metrics
            resolved_alerts = [alert for alert in self.alert_history if alert.resolved]
            resolution_rate = (
                len(resolved_alerts) / len(self.alert_history)
                if self.alert_history
                else 0
            )

            return {
                "total_alerts": len(self.alert_history),
                "alerts_24h": len(recent_alerts),
                "severity_distribution": dict(severity_counts),
                "resolution_rate": resolution_rate,
                "active_rules": len(
                    [rule for rule in self.alert_rules.values() if rule.active]
                ),
                "notification_channels": list(self.notification_channels.keys()),
            }

        except Exception as e:
            logger.error(f"Failed to get alert statistics: {e}")
            return {}
