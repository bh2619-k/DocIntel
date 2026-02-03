"""
Real-Time Document Intelligence & Compliance Engine
Alert Generation System
Version: 1.0
Date: 2026-02-03

This module implements intelligent alert generation with idempotent processing,
cooldown periods, severity-based routing, and configurable thresholds.
"""

import logging
from typing import Dict, Any, List, Optional, Set, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import json
import uuid
import hashlib
import threading
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)


class AlertType(Enum):
    """Types of alerts in the system"""

    COMPLIANCE_VIOLATION = "compliance_violation"
    HIGH_RISK_DETECTION = "high_risk_detection"
    ANOMALY_DETECTION = "anomaly_detection"
    THRESHOLD_BREACH = "threshold_breach"
    SYSTEM_ERROR = "system_error"
    DOCUMENT_PROCESSED = "document_processed"
    ENTITY_RESOLVED = "entity_resolved"
    RISK_PROPAGATED = "risk_propagated"
    CLUSTER_IDENTIFIED = "cluster_identified"


class AlertSeverity(Enum):
    """Alert severity levels"""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class AlertStatus(Enum):
    """Alert status values"""

    PENDING = "pending"
    SENT = "sent"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"
    SUPPRESSED = "suppressed"
    EXPIRED = "expired"


class DeliveryChannel(Enum):
    """Alert delivery channels"""

    EMAIL = "email"
    SMS = "sms"
    WEBHOOK = "webhook"
    WEBSOCKET = "websocket"
    SLACK = "slack"
    TEAMS = "teams"
    IN_APP = "in_app"


@dataclass
class AlertRule:
    """Defines conditions and actions for alert generation"""

    rule_id: str
    name: str
    description: str
    alert_type: AlertType
    severity: AlertSeverity
    conditions: Dict[str, Any]  # Conditions that trigger the alert
    cooldown_minutes: int  # Minimum time between identical alerts
    enabled: bool
    channels: List[DeliveryChannel]
    recipients: List[str]  # User IDs or roles
    template: str  # Alert message template
    metadata: Dict[str, Any]
    created_at: datetime
    updated_at: datetime


@dataclass
class Alert:
    """Represents a generated alert"""

    alert_id: str
    rule_id: str
    alert_type: AlertType
    severity: AlertSeverity
    title: str
    message: str
    status: AlertStatus
    source_entity_id: Optional[str]
    source_document_id: Optional[str]
    risk_score: Optional[float]
    fingerprint: str  # For deduplication
    metadata: Dict[str, Any]
    channels: List[DeliveryChannel]
    recipients: List[str]
    delivery_attempts: List[Dict[str, Any]]
    created_at: datetime
    updated_at: datetime
    acknowledged_at: Optional[datetime] = None
    acknowledged_by: Optional[str] = None
    resolved_at: Optional[datetime] = None
    resolved_by: Optional[str] = None


@dataclass
class AlertDelivery:
    """Tracks alert delivery attempts"""

    delivery_id: str
    alert_id: str
    channel: DeliveryChannel
    recipient: str
    status: str  # 'pending', 'sent', 'delivered', 'failed'
    attempts: int
    last_attempt: datetime
    error_message: Optional[str]
    metadata: Dict[str, Any]


class AlertGenerationEngine:
    """Comprehensive alert generation and management system"""

    def __init__(
        self,
        max_alerts_per_hour: int = 1000,
        default_cooldown_minutes: int = 15,
        max_delivery_attempts: int = 3,
    ):
        """
        Initialize alert generation engine

        Args:
            max_alerts_per_hour: Rate limit for alert generation
            default_cooldown_minutes: Default cooldown between duplicate alerts
            max_delivery_attempts: Maximum delivery retry attempts
        """
        self.max_alerts_per_hour = max_alerts_per_hour
        self.default_cooldown_minutes = default_cooldown_minutes
        self.max_delivery_attempts = max_delivery_attempts

        # Thread safety
        self._lock = threading.RLock()
        self._executor = ThreadPoolExecutor(max_workers=10)

        # Storage
        self.alert_rules: Dict[str, AlertRule] = {}
        self.alerts: Dict[str, Alert] = {}
        self.alert_fingerprints: Dict[str, datetime] = {}  # For cooldown tracking
        self.delivery_queue: deque = deque()

        # Rate limiting
        self.alert_timestamps: deque = deque()  # Track alerts generated in last hour

        # Delivery handlers
        self.delivery_handlers: Dict[DeliveryChannel, Callable] = {}

        # Statistics
        self.stats = {
            "alerts_generated": 0,
            "alerts_sent": 0,
            "alerts_failed": 0,
            "rules_triggered": 0,
            "duplicates_suppressed": 0,
            "rate_limited": 0,
        }

        # Initialize default alert rules
        self._initialize_default_rules()

    def _initialize_default_rules(self):
        """Initialize default alert rules"""
        try:
            # High-risk entity detection
            self.register_alert_rule(
                AlertRule(
                    rule_id="high_risk_entity",
                    name="High-Risk Entity Detected",
                    description="Alert when entity risk score exceeds threshold",
                    alert_type=AlertType.HIGH_RISK_DETECTION,
                    severity=AlertSeverity.HIGH,
                    conditions={"risk_score": {"min": 0.8}},
                    cooldown_minutes=60,
                    enabled=True,
                    channels=[DeliveryChannel.EMAIL, DeliveryChannel.IN_APP],
                    recipients=["compliance_team", "risk_managers"],
                    template="High-risk entity detected: {entity_name} with risk score {risk_score:.2f}",
                    metadata={"auto_generated": True},
                    created_at=datetime.now(),
                    updated_at=datetime.now(),
                )
            )

            # Compliance violation
            self.register_alert_rule(
                AlertRule(
                    rule_id="compliance_violation",
                    name="Compliance Violation",
                    description="Alert on compliance rule violations",
                    alert_type=AlertType.COMPLIANCE_VIOLATION,
                    severity=AlertSeverity.CRITICAL,
                    conditions={"violation_type": {"any": True}},
                    cooldown_minutes=30,
                    enabled=True,
                    channels=[
                        DeliveryChannel.EMAIL,
                        DeliveryChannel.WEBHOOK,
                        DeliveryChannel.IN_APP,
                    ],
                    recipients=["compliance_officers", "legal_team"],
                    template="Compliance violation detected: {violation_description}",
                    metadata={"auto_generated": True},
                    created_at=datetime.now(),
                    updated_at=datetime.now(),
                )
            )

            # Risk cluster identification
            self.register_alert_rule(
                AlertRule(
                    rule_id="risk_cluster",
                    name="Risk Cluster Identified",
                    description="Alert when risk clusters are identified",
                    alert_type=AlertType.CLUSTER_IDENTIFIED,
                    severity=AlertSeverity.MEDIUM,
                    conditions={
                        "cluster_size": {"min": 5},
                        "total_risk_score": {"min": 2.0},
                    },
                    cooldown_minutes=120,
                    enabled=True,
                    channels=[DeliveryChannel.EMAIL, DeliveryChannel.IN_APP],
                    recipients=["risk_analysts"],
                    template="Risk cluster identified with {cluster_size} entities and total risk score {total_risk_score:.2f}",
                    metadata={"auto_generated": True},
                    created_at=datetime.now(),
                    updated_at=datetime.now(),
                )
            )

            logger.info("Default alert rules initialized")

        except Exception as e:
            logger.error(f"Failed to initialize default rules: {e}")

    def register_alert_rule(self, alert_rule: AlertRule) -> bool:
        """
        Register a new alert rule

        Args:
            alert_rule: Alert rule to register

        Returns:
            True if successful
        """
        try:
            with self._lock:
                self.alert_rules[alert_rule.rule_id] = alert_rule

            logger.info(f"Registered alert rule: {alert_rule.rule_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to register alert rule: {e}")
            return False

    def evaluate_conditions(self, event_data: Dict[str, Any]) -> List[str]:
        """
        Evaluate event data against all alert rules

        Args:
            event_data: Event data to evaluate

        Returns:
            List of triggered rule IDs
        """
        try:
            triggered_rules = []

            for rule_id, rule in self.alert_rules.items():
                if not rule.enabled:
                    continue

                if self._check_rule_conditions(rule, event_data):
                    triggered_rules.append(rule_id)

            return triggered_rules

        except Exception as e:
            logger.error(f"Condition evaluation failed: {e}")
            return []

    def _check_rule_conditions(
        self, rule: AlertRule, event_data: Dict[str, Any]
    ) -> bool:
        """Check if event data matches rule conditions"""
        try:
            conditions = rule.conditions

            for field, criteria in conditions.items():
                event_value = event_data.get(field)

                if event_value is None and criteria.get("required", False):
                    return False

                # Check minimum value
                if "min" in criteria and event_value is not None:
                    if (
                        isinstance(event_value, (int, float))
                        and event_value < criteria["min"]
                    ):
                        return False

                # Check maximum value
                if "max" in criteria and event_value is not None:
                    if (
                        isinstance(event_value, (int, float))
                        and event_value > criteria["max"]
                    ):
                        return False

                # Check exact match
                if "equals" in criteria and event_value != criteria["equals"]:
                    return False

                # Check contains
                if "contains" in criteria and isinstance(event_value, str):
                    if criteria["contains"] not in event_value:
                        return False

                # Check any (for presence of field)
                if "any" in criteria and criteria["any"] and event_value is None:
                    return False

            return True

        except Exception as e:
            logger.error(f"Rule condition check failed: {e}")
            return False

    def generate_alert(
        self,
        event_data: Dict[str, Any],
        rule_id: str = None,
        override_cooldown: bool = False,
    ) -> Optional[str]:
        """
        Generate an alert based on event data

        Args:
            event_data: Event data that triggered the alert
            rule_id: Specific rule to use (if None, evaluates all rules)
            override_cooldown: Whether to bypass cooldown period

        Returns:
            Alert ID if generated, None if suppressed
        """
        try:
            # Rate limiting check
            if not self._check_rate_limit():
                with self._lock:
                    self.stats["rate_limited"] += 1
                return None

            # Determine which rules to trigger
            if rule_id:
                triggered_rules = [rule_id] if rule_id in self.alert_rules else []
            else:
                triggered_rules = self.evaluate_conditions(event_data)

            if not triggered_rules:
                return None

            # Process each triggered rule
            alerts_generated = []

            for rule_id in triggered_rules:
                rule = self.alert_rules[rule_id]

                # Generate alert fingerprint for deduplication
                fingerprint = self._generate_fingerprint(rule_id, event_data)

                # Check cooldown period
                if not override_cooldown and self._is_in_cooldown(
                    fingerprint, rule.cooldown_minutes
                ):
                    with self._lock:
                        self.stats["duplicates_suppressed"] += 1
                    continue

                # Create alert
                alert_id = str(uuid.uuid4())
                alert = self._create_alert(alert_id, rule, event_data, fingerprint)

                # Store alert
                with self._lock:
                    self.alerts[alert_id] = alert
                    self.alert_fingerprints[fingerprint] = datetime.now()
                    self.stats["alerts_generated"] += 1
                    self.stats["rules_triggered"] += 1

                alerts_generated.append(alert_id)

                # Queue for delivery
                self._queue_alert_delivery(alert)

                logger.info(f"Generated alert: {alert_id} (rule: {rule_id})")

            return alerts_generated[0] if alerts_generated else None

        except Exception as e:
            logger.error(f"Alert generation failed: {e}")
            return None

    def _check_rate_limit(self) -> bool:
        """Check if within rate limits"""
        try:
            current_time = datetime.now()
            hour_ago = current_time - timedelta(hours=1)

            # Remove old timestamps
            while self.alert_timestamps and self.alert_timestamps[0] < hour_ago:
                self.alert_timestamps.popleft()

            # Check current rate
            if len(self.alert_timestamps) >= self.max_alerts_per_hour:
                return False

            # Add current timestamp
            self.alert_timestamps.append(current_time)
            return True

        except Exception as e:
            logger.error(f"Rate limit check failed: {e}")
            return False

    def _generate_fingerprint(self, rule_id: str, event_data: Dict[str, Any]) -> str:
        """Generate fingerprint for alert deduplication"""
        try:
            # Include rule ID and relevant event data fields
            fingerprint_data = {
                "rule_id": rule_id,
                "entity_id": event_data.get("entity_id"),
                "document_id": event_data.get("document_id"),
                "risk_type": event_data.get("risk_type"),
                "violation_type": event_data.get("violation_type"),
            }

            # Remove None values
            fingerprint_data = {
                k: v for k, v in fingerprint_data.items() if v is not None
            }

            # Create hash
            fingerprint_str = json.dumps(fingerprint_data, sort_keys=True)
            return hashlib.md5(fingerprint_str.encode()).hexdigest()

        except Exception as e:
            logger.error(f"Fingerprint generation failed: {e}")
            return str(uuid.uuid4())

    def _is_in_cooldown(self, fingerprint: str, cooldown_minutes: int) -> bool:
        """Check if fingerprint is in cooldown period"""
        try:
            if fingerprint not in self.alert_fingerprints:
                return False

            last_generated = self.alert_fingerprints[fingerprint]
            cooldown_period = timedelta(minutes=cooldown_minutes)

            return datetime.now() - last_generated < cooldown_period

        except Exception as e:
            logger.error(f"Cooldown check failed: {e}")
            return False

    def _create_alert(
        self,
        alert_id: str,
        rule: AlertRule,
        event_data: Dict[str, Any],
        fingerprint: str,
    ) -> Alert:
        """Create alert object from rule and event data"""
        try:
            # Format message using template
            message = self._format_alert_message(rule.template, event_data)

            # Generate title
            title = f"{rule.alert_type.value.replace('_', ' ').title()}: {event_data.get('entity_name', 'Unknown Entity')}"

            alert = Alert(
                alert_id=alert_id,
                rule_id=rule.rule_id,
                alert_type=rule.alert_type,
                severity=rule.severity,
                title=title,
                message=message,
                status=AlertStatus.PENDING,
                source_entity_id=event_data.get("entity_id"),
                source_document_id=event_data.get("document_id"),
                risk_score=event_data.get("risk_score"),
                fingerprint=fingerprint,
                metadata=event_data.copy(),
                channels=rule.channels.copy(),
                recipients=rule.recipients.copy(),
                delivery_attempts=[],
                created_at=datetime.now(),
                updated_at=datetime.now(),
            )

            return alert

        except Exception as e:
            logger.error(f"Alert creation failed: {e}")
            raise

    def _format_alert_message(self, template: str, event_data: Dict[str, Any]) -> str:
        """Format alert message using template and event data"""
        try:
            # Simple template formatting
            formatted_message = template

            for key, value in event_data.items():
                placeholder = "{" + key + "}"
                if placeholder in formatted_message:
                    formatted_message = formatted_message.replace(
                        placeholder, str(value)
                    )

            return formatted_message

        except Exception as e:
            logger.error(f"Message formatting failed: {e}")
            return template

    def _queue_alert_delivery(self, alert: Alert):
        """Queue alert for delivery"""
        try:
            for channel in alert.channels:
                for recipient in alert.recipients:
                    delivery = AlertDelivery(
                        delivery_id=str(uuid.uuid4()),
                        alert_id=alert.alert_id,
                        channel=channel,
                        recipient=recipient,
                        status="pending",
                        attempts=0,
                        last_attempt=datetime.now(),
                        error_message=None,
                        metadata={},
                    )

                    with self._lock:
                        self.delivery_queue.append(delivery)

        except Exception as e:
            logger.error(f"Alert delivery queueing failed: {e}")

    def register_delivery_handler(self, channel: DeliveryChannel, handler: Callable):
        """Register a delivery handler for a specific channel"""
        try:
            self.delivery_handlers[channel] = handler
            logger.info(f"Registered delivery handler for {channel.value}")

        except Exception as e:
            logger.error(f"Failed to register delivery handler: {e}")

    def process_delivery_queue(self, batch_size: int = 50):
        """Process pending alert deliveries"""
        try:
            deliveries_to_process = []

            # Get batch of deliveries
            with self._lock:
                for _ in range(min(batch_size, len(self.delivery_queue))):
                    if self.delivery_queue:
                        deliveries_to_process.append(self.delivery_queue.popleft())

            if not deliveries_to_process:
                return

            # Process deliveries
            for delivery in deliveries_to_process:
                try:
                    if delivery.channel in self.delivery_handlers:
                        handler = self.delivery_handlers[delivery.channel]

                        # Get alert details
                        alert = self.alerts.get(delivery.alert_id)
                        if not alert:
                            continue

                        # Attempt delivery
                        success = handler(alert, delivery)

                        # Update delivery status
                        delivery.attempts += 1
                        delivery.last_attempt = datetime.now()

                        if success:
                            delivery.status = "sent"
                            with self._lock:
                                self.stats["alerts_sent"] += 1
                        else:
                            delivery.status = "failed"
                            with self._lock:
                                self.stats["alerts_failed"] += 1

                            # Retry if under limit
                            if delivery.attempts < self.max_delivery_attempts:
                                with self._lock:
                                    self.delivery_queue.append(delivery)

                        # Update alert
                        alert.delivery_attempts.append(
                            {
                                "delivery_id": delivery.delivery_id,
                                "channel": delivery.channel.value,
                                "recipient": delivery.recipient,
                                "status": delivery.status,
                                "timestamp": delivery.last_attempt.isoformat(),
                            }
                        )
                        alert.updated_at = datetime.now()

                        if success and alert.status == AlertStatus.PENDING:
                            alert.status = AlertStatus.SENT

                    else:
                        logger.warning(
                            f"No handler registered for channel: {delivery.channel}"
                        )
                        delivery.status = "failed"
                        delivery.error_message = (
                            f"No handler for {delivery.channel.value}"
                        )

                except Exception as e:
                    logger.error(f"Delivery processing failed: {e}")
                    delivery.status = "failed"
                    delivery.error_message = str(e)

            logger.debug(f"Processed {len(deliveries_to_process)} alert deliveries")

        except Exception as e:
            logger.error(f"Delivery queue processing failed: {e}")

    def acknowledge_alert(self, alert_id: str, user_id: str) -> bool:
        """Acknowledge an alert"""
        try:
            if alert_id not in self.alerts:
                return False

            alert = self.alerts[alert_id]
            alert.status = AlertStatus.ACKNOWLEDGED
            alert.acknowledged_at = datetime.now()
            alert.acknowledged_by = user_id
            alert.updated_at = datetime.now()

            logger.info(f"Alert acknowledged: {alert_id} by {user_id}")
            return True

        except Exception as e:
            logger.error(f"Alert acknowledgment failed: {e}")
            return False

    def resolve_alert(self, alert_id: str, user_id: str) -> bool:
        """Resolve an alert"""
        try:
            if alert_id not in self.alerts:
                return False

            alert = self.alerts[alert_id]
            alert.status = AlertStatus.RESOLVED
            alert.resolved_at = datetime.now()
            alert.resolved_by = user_id
            alert.updated_at = datetime.now()

            logger.info(f"Alert resolved: {alert_id} by {user_id}")
            return True

        except Exception as e:
            logger.error(f"Alert resolution failed: {e}")
            return False

    def get_active_alerts(
        self, severity: AlertSeverity = None, limit: int = 100
    ) -> List[Alert]:
        """Get active (non-resolved) alerts"""
        try:
            active_alerts = []

            for alert in self.alerts.values():
                if alert.status not in [AlertStatus.RESOLVED, AlertStatus.EXPIRED]:
                    if severity is None or alert.severity == severity:
                        active_alerts.append(alert)

            # Sort by severity and creation time
            severity_order = {
                AlertSeverity.CRITICAL: 0,
                AlertSeverity.HIGH: 1,
                AlertSeverity.MEDIUM: 2,
                AlertSeverity.LOW: 3,
                AlertSeverity.INFO: 4,
            }

            active_alerts.sort(
                key=lambda x: (severity_order[x.severity], x.created_at), reverse=True
            )

            return active_alerts[:limit]

        except Exception as e:
            logger.error(f"Failed to get active alerts: {e}")
            return []

    def get_alert_statistics(self) -> Dict[str, Any]:
        """Get comprehensive alert statistics"""
        try:
            stats = self.stats.copy()

            # Alert status distribution
            status_counts = defaultdict(int)
            severity_counts = defaultdict(int)
            type_counts = defaultdict(int)

            for alert in self.alerts.values():
                status_counts[alert.status.value] += 1
                severity_counts[alert.severity.value] += 1
                type_counts[alert.alert_type.value] += 1

            # Rule effectiveness
            rule_effectiveness = {}
            for rule_id, rule in self.alert_rules.items():
                rule_alerts = [a for a in self.alerts.values() if a.rule_id == rule_id]
                rule_effectiveness[rule_id] = {
                    "total_alerts": len(rule_alerts),
                    "acknowledged": len(
                        [a for a in rule_alerts if a.status == AlertStatus.ACKNOWLEDGED]
                    ),
                    "resolved": len(
                        [a for a in rule_alerts if a.status == AlertStatus.RESOLVED]
                    ),
                }

            stats.update(
                {
                    "total_alerts": len(self.alerts),
                    "total_rules": len(self.alert_rules),
                    "active_alerts": len(self.get_active_alerts()),
                    "pending_deliveries": len(self.delivery_queue),
                    "status_distribution": dict(status_counts),
                    "severity_distribution": dict(severity_counts),
                    "type_distribution": dict(type_counts),
                    "rule_effectiveness": rule_effectiveness,
                }
            )

            return stats

        except Exception as e:
            logger.error(f"Statistics calculation failed: {e}")
            return self.stats.copy()

    def cleanup_expired_alerts(self, retention_days: int = 30):
        """Clean up old resolved alerts"""
        try:
            cutoff_date = datetime.now() - timedelta(days=retention_days)
            expired_alerts = []

            for alert_id, alert in self.alerts.items():
                if (
                    alert.status == AlertStatus.RESOLVED
                    and alert.resolved_at
                    and alert.resolved_at < cutoff_date
                ):
                    expired_alerts.append(alert_id)
                elif alert.created_at < cutoff_date:
                    expired_alerts.append(alert_id)

            # Remove expired alerts
            for alert_id in expired_alerts:
                del self.alerts[alert_id]

            # Clean up fingerprints
            expired_fingerprints = []
            for fingerprint, timestamp in self.alert_fingerprints.items():
                if timestamp < cutoff_date:
                    expired_fingerprints.append(fingerprint)

            for fingerprint in expired_fingerprints:
                del self.alert_fingerprints[fingerprint]

            logger.info(
                f"Cleaned up {len(expired_alerts)} expired alerts and {len(expired_fingerprints)} fingerprints"
            )

        except Exception as e:
            logger.error(f"Alert cleanup failed: {e}")

    def close(self):
        """Close the alert generation engine and cleanup resources"""
        try:
            if hasattr(self, "_executor") and self._executor:
                self._executor.shutdown(wait=True)

            logger.info("Alert generation engine closed")

        except Exception as e:
            logger.error(f"Error closing alert generation engine: {e}")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
