"""
Real-Time Document Intelligence & Compliance Engine
Role-Based Alert Routing System
Version: 1.0
Date: 2026-02-03

This module implements user preference management, escalation rules, and delivery
channel routing based on roles and alert types.
"""

import logging
from typing import Dict, Any, List, Optional, Set
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import json
import uuid
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)


class UserRole(Enum):
    """User roles in the system"""

    ADMIN = "admin"
    COMPLIANCE_OFFICER = "compliance_officer"
    RISK_MANAGER = "risk_manager"
    LEGAL_COUNSEL = "legal_counsel"
    AUDITOR = "auditor"
    ANALYST = "analyst"
    MANAGER = "manager"
    EXECUTIVE = "executive"
    IT_SUPPORT = "it_support"
    USER = "user"


class NotificationPreference(Enum):
    """Notification preferences"""

    IMMEDIATE = "immediate"
    HOURLY_DIGEST = "hourly_digest"
    DAILY_DIGEST = "daily_digest"
    WEEKLY_DIGEST = "weekly_digest"
    DISABLED = "disabled"


class EscalationAction(Enum):
    """Escalation actions"""

    NOTIFY_MANAGER = "notify_manager"
    ESCALATE_TO_ROLE = "escalate_to_role"
    INCREASE_FREQUENCY = "increase_frequency"
    ADD_CHANNELS = "add_channels"
    CREATE_TICKET = "create_ticket"
    SEND_URGENT = "send_urgent"


@dataclass
class User:
    """User profile with preferences and roles"""

    user_id: str
    username: str
    email: str
    phone: str
    roles: List[UserRole]
    department: str
    manager_id: Optional[str]
    timezone: str
    active: bool
    preferences: Dict[str, Any]
    created_at: datetime
    updated_at: datetime


@dataclass
class AlertPreference:
    """User alert preferences for specific alert types"""

    preference_id: str
    user_id: str
    alert_type: str
    severity_levels: List[str]  # Which severities to receive
    channels: List[str]  # Preferred delivery channels
    notification_preference: NotificationPreference
    quiet_hours_start: Optional[str]  # HH:MM format
    quiet_hours_end: Optional[str]  # HH:MM format
    enabled: bool
    created_at: datetime
    updated_at: datetime


@dataclass
class EscalationRule:
    """Escalation rule configuration"""

    rule_id: str
    name: str
    description: str
    alert_types: List[str]
    severity_levels: List[str]
    conditions: Dict[str, Any]
    escalation_delay_minutes: int
    escalation_actions: List[EscalationAction]
    target_roles: List[UserRole]
    target_users: List[str]
    enabled: bool
    created_at: datetime
    updated_at: datetime


@dataclass
class RoutingRule:
    """Alert routing rule"""

    rule_id: str
    name: str
    priority: int  # Higher number = higher priority
    conditions: Dict[str, Any]
    target_roles: List[UserRole]
    target_users: List[str]
    channels: List[str]
    enabled: bool
    created_at: datetime


@dataclass
class AlertRouting:
    """Alert routing decision"""

    routing_id: str
    alert_id: str
    user_id: str
    channels: List[str]
    delivery_preference: NotificationPreference
    estimated_delivery_time: datetime
    routing_rules_applied: List[str]
    escalation_rules: List[str]
    created_at: datetime


class RoleBasedAlertRouter:
    """Comprehensive role-based alert routing system"""

    def __init__(self):
        """Initialize the alert routing system"""

        # Thread safety
        self._lock = threading.RLock()
        self._executor = ThreadPoolExecutor(max_workers=5)

        # Storage
        self.users: Dict[str, User] = {}
        self.alert_preferences: Dict[str, List[AlertPreference]] = defaultdict(list)
        self.escalation_rules: Dict[str, EscalationRule] = {}
        self.routing_rules: Dict[str, RoutingRule] = {}
        self.active_escalations: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

        # Role hierarchy for escalation
        self.role_hierarchy = {
            UserRole.USER: [UserRole.ANALYST],
            UserRole.ANALYST: [UserRole.MANAGER],
            UserRole.MANAGER: [UserRole.RISK_MANAGER, UserRole.COMPLIANCE_OFFICER],
            UserRole.RISK_MANAGER: [UserRole.EXECUTIVE],
            UserRole.COMPLIANCE_OFFICER: [UserRole.LEGAL_COUNSEL, UserRole.EXECUTIVE],
            UserRole.LEGAL_COUNSEL: [UserRole.EXECUTIVE],
            UserRole.AUDITOR: [UserRole.COMPLIANCE_OFFICER],
            UserRole.IT_SUPPORT: [UserRole.ADMIN],
            UserRole.ADMIN: [UserRole.EXECUTIVE],
        }

        # Default alert routing by role
        self.default_role_routing = {
            UserRole.COMPLIANCE_OFFICER: ["compliance_violation", "regulatory_risk"],
            UserRole.RISK_MANAGER: [
                "high_risk_detection",
                "risk_propagated",
                "cluster_identified",
            ],
            UserRole.LEGAL_COUNSEL: ["compliance_violation", "legal_risk"],
            UserRole.AUDITOR: ["anomaly_detection", "compliance_violation"],
            UserRole.EXECUTIVE: ["critical_alerts"],
            UserRole.IT_SUPPORT: ["system_error", "technical_issues"],
            UserRole.ADMIN: ["system_error", "user_management"],
        }

        # Statistics
        self.stats = {
            "alerts_routed": 0,
            "escalations_triggered": 0,
            "preferences_applied": 0,
            "users_notified": 0,
            "routing_rules_executed": 0,
        }

        # Initialize default configurations
        self._initialize_default_configurations()

    def _initialize_default_configurations(self):
        """Initialize default routing and escalation rules"""
        try:
            # Default escalation rule for critical alerts
            self.register_escalation_rule(
                EscalationRule(
                    rule_id="critical_escalation",
                    name="Critical Alert Escalation",
                    description="Escalate critical alerts if not acknowledged within 15 minutes",
                    alert_types=["compliance_violation", "high_risk_detection"],
                    severity_levels=["critical", "high"],
                    conditions={"acknowledgment_timeout_minutes": 15},
                    escalation_delay_minutes=15,
                    escalation_actions=[
                        EscalationAction.NOTIFY_MANAGER,
                        EscalationAction.INCREASE_FREQUENCY,
                    ],
                    target_roles=[UserRole.MANAGER, UserRole.EXECUTIVE],
                    target_users=[],
                    enabled=True,
                    created_at=datetime.now(),
                    updated_at=datetime.now(),
                )
            )

            # Default routing rule for compliance alerts
            self.register_routing_rule(
                RoutingRule(
                    rule_id="compliance_routing",
                    name="Compliance Alert Routing",
                    priority=100,
                    conditions={"alert_type": "compliance_violation"},
                    target_roles=[UserRole.COMPLIANCE_OFFICER, UserRole.LEGAL_COUNSEL],
                    target_users=[],
                    channels=["email", "webhook", "in_app"],
                    enabled=True,
                    created_at=datetime.now(),
                )
            )

            # High-risk entity routing
            self.register_routing_rule(
                RoutingRule(
                    rule_id="high_risk_routing",
                    name="High-Risk Entity Routing",
                    priority=90,
                    conditions={
                        "alert_type": "high_risk_detection",
                        "severity": ["high", "critical"],
                    },
                    target_roles=[UserRole.RISK_MANAGER, UserRole.COMPLIANCE_OFFICER],
                    target_users=[],
                    channels=["email", "sms", "in_app"],
                    enabled=True,
                    created_at=datetime.now(),
                )
            )

            logger.info("Default routing configurations initialized")

        except Exception as e:
            logger.error(f"Failed to initialize default configurations: {e}")

    def register_user(self, user: User) -> bool:
        """Register a new user in the system"""
        try:
            with self._lock:
                self.users[user.user_id] = user

            logger.info(f"Registered user: {user.username} ({user.user_id})")
            return True

        except Exception as e:
            logger.error(f"Failed to register user: {e}")
            return False

    def update_user_preferences(
        self, user_id: str, alert_preference: AlertPreference
    ) -> bool:
        """Update user alert preferences"""
        try:
            if user_id not in self.users:
                logger.warning(f"User not found: {user_id}")
                return False

            with self._lock:
                # Remove existing preference for the same alert type
                existing_prefs = self.alert_preferences[user_id]
                self.alert_preferences[user_id] = [
                    pref
                    for pref in existing_prefs
                    if pref.alert_type != alert_preference.alert_type
                ]

                # Add new preference
                self.alert_preferences[user_id].append(alert_preference)
                self.stats["preferences_applied"] += 1

            logger.info(
                f"Updated preferences for user {user_id}: {alert_preference.alert_type}"
            )
            return True

        except Exception as e:
            logger.error(f"Failed to update user preferences: {e}")
            return False

    def register_escalation_rule(self, escalation_rule: EscalationRule) -> bool:
        """Register an escalation rule"""
        try:
            with self._lock:
                self.escalation_rules[escalation_rule.rule_id] = escalation_rule

            logger.info(f"Registered escalation rule: {escalation_rule.name}")
            return True

        except Exception as e:
            logger.error(f"Failed to register escalation rule: {e}")
            return False

    def register_routing_rule(self, routing_rule: RoutingRule) -> bool:
        """Register a routing rule"""
        try:
            with self._lock:
                self.routing_rules[routing_rule.rule_id] = routing_rule

            logger.info(f"Registered routing rule: {routing_rule.name}")
            return True

        except Exception as e:
            logger.error(f"Failed to register routing rule: {e}")
            return False

    def route_alert(self, alert_data: Dict[str, Any]) -> List[AlertRouting]:
        """Route an alert to appropriate users based on rules and preferences"""
        try:
            alert_id = alert_data.get("alert_id", str(uuid.uuid4()))
            alert_type = alert_data.get("alert_type", "unknown")
            severity = alert_data.get("severity", "medium")

            routings = []
            targeted_users = set()

            # Apply routing rules (sorted by priority)
            applicable_rules = self._get_applicable_routing_rules(alert_data)

            for rule in applicable_rules:
                # Get users from target roles
                role_users = self._get_users_by_roles(rule.target_roles)

                # Add specific target users
                for user_id in rule.target_users:
                    if user_id in self.users:
                        role_users.add(user_id)

                # Create routing for each user
                for user_id in role_users:
                    if user_id in targeted_users:
                        continue  # Avoid duplicate routing

                    routing = self._create_user_routing(
                        user_id, alert_id, alert_data, rule
                    )
                    if routing:
                        routings.append(routing)
                        targeted_users.add(user_id)

            # If no specific rules matched, use default role-based routing
            if not routings:
                default_users = self._get_default_recipients(alert_type, severity)
                for user_id in default_users:
                    if user_id not in targeted_users:
                        routing = self._create_default_routing(
                            user_id, alert_id, alert_data
                        )
                        if routing:
                            routings.append(routing)
                            targeted_users.add(user_id)

            # Schedule escalations if needed
            self._schedule_escalations(alert_id, alert_data, routings)

            # Update statistics
            with self._lock:
                self.stats["alerts_routed"] += 1
                self.stats["users_notified"] += len(routings)
                self.stats["routing_rules_executed"] += len(applicable_rules)

            logger.info(f"Routed alert {alert_id} to {len(routings)} users")
            return routings

        except Exception as e:
            logger.error(f"Alert routing failed: {e}")
            return []

    def _get_applicable_routing_rules(
        self, alert_data: Dict[str, Any]
    ) -> List[RoutingRule]:
        """Get routing rules applicable to the alert"""
        try:
            applicable_rules = []

            for rule in self.routing_rules.values():
                if not rule.enabled:
                    continue

                if self._check_routing_conditions(rule, alert_data):
                    applicable_rules.append(rule)

            # Sort by priority (descending)
            applicable_rules.sort(key=lambda r: r.priority, reverse=True)
            return applicable_rules

        except Exception as e:
            logger.error(f"Failed to get applicable routing rules: {e}")
            return []

    def _check_routing_conditions(
        self, rule: RoutingRule, alert_data: Dict[str, Any]
    ) -> bool:
        """Check if alert data matches routing rule conditions"""
        try:
            conditions = rule.conditions

            for field, criteria in conditions.items():
                alert_value = alert_data.get(field)

                if isinstance(criteria, str):
                    if alert_value != criteria:
                        return False
                elif isinstance(criteria, list):
                    if alert_value not in criteria:
                        return False
                elif isinstance(criteria, dict):
                    if "equals" in criteria and alert_value != criteria["equals"]:
                        return False
                    if "contains" in criteria and isinstance(alert_value, str):
                        if criteria["contains"] not in alert_value:
                            return False
                    if "min" in criteria and isinstance(alert_value, (int, float)):
                        if alert_value < criteria["min"]:
                            return False
                    if "max" in criteria and isinstance(alert_value, (int, float)):
                        if alert_value > criteria["max"]:
                            return False

            return True

        except Exception as e:
            logger.error(f"Routing condition check failed: {e}")
            return False

    def _get_users_by_roles(self, roles: List[UserRole]) -> Set[str]:
        """Get all active users with specified roles"""
        try:
            users = set()

            for user_id, user in self.users.items():
                if user.active:
                    for role in roles:
                        if role in user.roles:
                            users.add(user_id)
                            break

            return users

        except Exception as e:
            logger.error(f"Failed to get users by roles: {e}")
            return set()

    def _create_user_routing(
        self, user_id: str, alert_id: str, alert_data: Dict[str, Any], rule: RoutingRule
    ) -> Optional[AlertRouting]:
        """Create routing for a specific user"""
        try:
            user = self.users.get(user_id)
            if not user or not user.active:
                return None

            # Get user preferences
            user_preferences = self._get_user_preferences(
                user_id, alert_data["alert_type"]
            )

            # Determine channels
            channels = rule.channels.copy()
            if user_preferences and user_preferences.channels:
                channels = user_preferences.channels

            # Determine delivery preference
            delivery_preference = NotificationPreference.IMMEDIATE
            if user_preferences:
                delivery_preference = user_preferences.notification_preference

            # Check quiet hours
            if self._is_in_quiet_hours(user, user_preferences):
                if delivery_preference == NotificationPreference.IMMEDIATE:
                    delivery_preference = NotificationPreference.HOURLY_DIGEST

            # Calculate estimated delivery time
            estimated_delivery = self._calculate_delivery_time(delivery_preference)

            routing = AlertRouting(
                routing_id=str(uuid.uuid4()),
                alert_id=alert_id,
                user_id=user_id,
                channels=channels,
                delivery_preference=delivery_preference,
                estimated_delivery_time=estimated_delivery,
                routing_rules_applied=[rule.rule_id],
                escalation_rules=[],
                created_at=datetime.now(),
            )

            return routing

        except Exception as e:
            logger.error(f"Failed to create user routing: {e}")
            return None

    def _get_user_preferences(
        self, user_id: str, alert_type: str
    ) -> Optional[AlertPreference]:
        """Get user preferences for specific alert type"""
        try:
            user_prefs = self.alert_preferences.get(user_id, [])

            for pref in user_prefs:
                if pref.alert_type == alert_type and pref.enabled:
                    return pref

            return None

        except Exception as e:
            logger.error(f"Failed to get user preferences: {e}")
            return None

    def _is_in_quiet_hours(
        self, user: User, preferences: Optional[AlertPreference]
    ) -> bool:
        """Check if current time is within user's quiet hours"""
        try:
            if (
                not preferences
                or not preferences.quiet_hours_start
                or not preferences.quiet_hours_end
            ):
                return False

            current_time = datetime.now().time()
            start_time = datetime.strptime(
                preferences.quiet_hours_start, "%H:%M"
            ).time()
            end_time = datetime.strptime(preferences.quiet_hours_end, "%H:%M").time()

            if start_time <= end_time:
                return start_time <= current_time <= end_time
            else:
                # Quiet hours span midnight
                return current_time >= start_time or current_time <= end_time

        except Exception as e:
            logger.error(f"Quiet hours check failed: {e}")
            return False

    def _calculate_delivery_time(self, preference: NotificationPreference) -> datetime:
        """Calculate estimated delivery time based on preference"""
        current_time = datetime.now()

        if preference == NotificationPreference.IMMEDIATE:
            return current_time
        elif preference == NotificationPreference.HOURLY_DIGEST:
            # Next hour boundary
            return current_time.replace(minute=0, second=0, microsecond=0) + timedelta(
                hours=1
            )
        elif preference == NotificationPreference.DAILY_DIGEST:
            # Next day at 9 AM
            next_day = current_time.date() + timedelta(days=1)
            return datetime.combine(next_day, datetime.min.time()) + timedelta(hours=9)
        elif preference == NotificationPreference.WEEKLY_DIGEST:
            # Next Monday at 9 AM
            days_ahead = 7 - current_time.weekday()
            if days_ahead <= 0:
                days_ahead += 7
            next_monday = current_time.date() + timedelta(days=days_ahead)
            return datetime.combine(next_monday, datetime.min.time()) + timedelta(
                hours=9
            )
        else:
            return current_time

    def _get_default_recipients(self, alert_type: str, severity: str) -> Set[str]:
        """Get default recipients based on alert type and severity"""
        try:
            recipients = set()

            # Get roles that should receive this alert type
            target_roles = []
            for role, alert_types in self.default_role_routing.items():
                if alert_type in alert_types:
                    target_roles.append(role)

            # Add executives for critical alerts
            if severity in ["critical", "high"]:
                target_roles.append(UserRole.EXECUTIVE)

            # Get users with target roles
            recipients = self._get_users_by_roles(target_roles)

            return recipients

        except Exception as e:
            logger.error(f"Failed to get default recipients: {e}")
            return set()

    def _create_default_routing(
        self, user_id: str, alert_id: str, alert_data: Dict[str, Any]
    ) -> Optional[AlertRouting]:
        """Create default routing for a user"""
        try:
            user = self.users.get(user_id)
            if not user or not user.active:
                return None

            # Use default channels
            channels = ["email", "in_app"]
            severity = alert_data.get("severity", "medium")

            # Add SMS for high/critical alerts
            if severity in ["high", "critical"]:
                channels.append("sms")

            routing = AlertRouting(
                routing_id=str(uuid.uuid4()),
                alert_id=alert_id,
                user_id=user_id,
                channels=channels,
                delivery_preference=NotificationPreference.IMMEDIATE,
                estimated_delivery_time=datetime.now(),
                routing_rules_applied=["default"],
                escalation_rules=[],
                created_at=datetime.now(),
            )

            return routing

        except Exception as e:
            logger.error(f"Failed to create default routing: {e}")
            return None

    def _schedule_escalations(
        self, alert_id: str, alert_data: Dict[str, Any], routings: List[AlertRouting]
    ):
        """Schedule escalations for an alert"""
        try:
            alert_type = alert_data.get("alert_type", "unknown")
            severity = alert_data.get("severity", "medium")

            # Find applicable escalation rules
            applicable_escalations = []
            for rule in self.escalation_rules.values():
                if not rule.enabled:
                    continue

                if alert_type in rule.alert_types and severity in rule.severity_levels:
                    applicable_escalations.append(rule)

            # Schedule each escalation
            for rule in applicable_escalations:
                escalation_time = datetime.now() + timedelta(
                    minutes=rule.escalation_delay_minutes
                )

                escalation_info = {
                    "rule_id": rule.rule_id,
                    "alert_id": alert_id,
                    "escalation_time": escalation_time,
                    "actions": rule.escalation_actions,
                    "target_roles": rule.target_roles,
                    "target_users": rule.target_users,
                    "created_at": datetime.now(),
                }

                with self._lock:
                    self.active_escalations[alert_id].append(escalation_info)

        except Exception as e:
            logger.error(f"Escalation scheduling failed: {e}")

    def process_escalations(self) -> List[Dict[str, Any]]:
        """Process pending escalations"""
        try:
            current_time = datetime.now()
            triggered_escalations = []

            with self._lock:
                alerts_to_remove = []

                for alert_id, escalations in self.active_escalations.items():
                    remaining_escalations = []

                    for escalation in escalations:
                        if current_time >= escalation["escalation_time"]:
                            # Trigger escalation
                            triggered_escalations.append(escalation)
                            self.stats["escalations_triggered"] += 1
                        else:
                            remaining_escalations.append(escalation)

                    if remaining_escalations:
                        self.active_escalations[alert_id] = remaining_escalations
                    else:
                        alerts_to_remove.append(alert_id)

                # Clean up completed escalations
                for alert_id in alerts_to_remove:
                    del self.active_escalations[alert_id]

            # Execute escalation actions
            for escalation in triggered_escalations:
                self._execute_escalation_actions(escalation)

            return triggered_escalations

        except Exception as e:
            logger.error(f"Escalation processing failed: {e}")
            return []

    def _execute_escalation_actions(self, escalation: Dict[str, Any]):
        """Execute escalation actions"""
        try:
            alert_id = escalation["alert_id"]
            actions = escalation["actions"]

            for action in actions:
                if action == EscalationAction.NOTIFY_MANAGER:
                    self._notify_managers(alert_id, escalation["target_users"])
                elif action == EscalationAction.ESCALATE_TO_ROLE:
                    self._escalate_to_roles(alert_id, escalation["target_roles"])
                elif action == EscalationAction.INCREASE_FREQUENCY:
                    self._increase_notification_frequency(alert_id)
                elif action == EscalationAction.ADD_CHANNELS:
                    self._add_escalation_channels(alert_id)
                elif action == EscalationAction.SEND_URGENT:
                    self._send_urgent_notification(alert_id)

            logger.info(
                f"Executed escalation actions for alert {alert_id}: {[a.value for a in actions]}"
            )

        except Exception as e:
            logger.error(f"Escalation action execution failed: {e}")

    def _notify_managers(self, alert_id: str, user_ids: List[str]):
        """Notify managers of specified users"""
        try:
            managers = set()

            # Get managers of specified users
            for user_id in user_ids:
                user = self.users.get(user_id)
                if user and user.manager_id:
                    managers.add(user.manager_id)

            # If no specific users, get all managers
            if not user_ids:
                for user in self.users.values():
                    if user.manager_id:
                        managers.add(user.manager_id)

            # Create urgent notifications for managers
            for manager_id in managers:
                if manager_id in self.users:
                    # This would integrate with the alert generation system
                    logger.info(f"Escalating alert {alert_id} to manager {manager_id}")

        except Exception as e:
            logger.error(f"Manager notification failed: {e}")

    def _escalate_to_roles(self, alert_id: str, target_roles: List[UserRole]):
        """Escalate alert to users with specific roles"""
        try:
            target_users = self._get_users_by_roles(target_roles)

            for user_id in target_users:
                # Create urgent notification
                logger.info(f"Escalating alert {alert_id} to role user {user_id}")

        except Exception as e:
            logger.error(f"Role escalation failed: {e}")

    def _increase_notification_frequency(self, alert_id: str):
        """Increase notification frequency for an alert"""
        try:
            # This would modify existing routings to increase frequency
            logger.info(f"Increasing notification frequency for alert {alert_id}")

        except Exception as e:
            logger.error(f"Frequency increase failed: {e}")

    def _add_escalation_channels(self, alert_id: str):
        """Add additional notification channels for escalation"""
        try:
            # Add SMS and webhook channels for escalated alerts
            logger.info(f"Adding escalation channels for alert {alert_id}")

        except Exception as e:
            logger.error(f"Channel addition failed: {e}")

    def _send_urgent_notification(self, alert_id: str):
        """Send urgent notification for escalated alert"""
        try:
            # Send immediate high-priority notification
            logger.info(f"Sending urgent notification for alert {alert_id}")

        except Exception as e:
            logger.error(f"Urgent notification failed: {e}")

    def get_routing_statistics(self) -> Dict[str, Any]:
        """Get comprehensive routing statistics"""
        try:
            stats = self.stats.copy()

            # User statistics
            total_users = len(self.users)
            active_users = len([u for u in self.users.values() if u.active])

            # Role distribution
            role_counts = defaultdict(int)
            for user in self.users.values():
                for role in user.roles:
                    role_counts[role.value] += 1

            # Preference statistics
            total_preferences = sum(
                len(prefs) for prefs in self.alert_preferences.values()
            )

            # Rule statistics
            active_routing_rules = len(
                [r for r in self.routing_rules.values() if r.enabled]
            )
            active_escalation_rules = len(
                [r for r in self.escalation_rules.values() if r.enabled]
            )

            # Escalation statistics
            pending_escalations = sum(
                len(esc) for esc in self.active_escalations.values()
            )

            stats.update(
                {
                    "total_users": total_users,
                    "active_users": active_users,
                    "role_distribution": dict(role_counts),
                    "total_preferences": total_preferences,
                    "active_routing_rules": active_routing_rules,
                    "active_escalation_rules": active_escalation_rules,
                    "pending_escalations": pending_escalations,
                    "total_routing_rules": len(self.routing_rules),
                    "total_escalation_rules": len(self.escalation_rules),
                }
            )

            return stats

        except Exception as e:
            logger.error(f"Statistics calculation failed: {e}")
            return self.stats.copy()

    def cleanup_expired_data(self, retention_days: int = 30):
        """Clean up old routing data"""
        try:
            cutoff_date = datetime.now() - timedelta(days=retention_days)

            # Clean up old escalations
            with self._lock:
                alerts_to_remove = []
                for alert_id, escalations in self.active_escalations.items():
                    remaining_escalations = []
                    for escalation in escalations:
                        if escalation["created_at"] > cutoff_date:
                            remaining_escalations.append(escalation)

                    if remaining_escalations:
                        self.active_escalations[alert_id] = remaining_escalations
                    else:
                        alerts_to_remove.append(alert_id)

                for alert_id in alerts_to_remove:
                    del self.active_escalations[alert_id]

            logger.info(f"Cleaned up {len(alerts_to_remove)} expired escalations")

        except Exception as e:
            logger.error(f"Data cleanup failed: {e}")

    def close(self):
        """Close the alert router and cleanup resources"""
        try:
            if hasattr(self, "_executor") and self._executor:
                self._executor.shutdown(wait=True)

            logger.info("Role-based alert router closed")

        except Exception as e:
            logger.error(f"Error closing alert router: {e}")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
