"""
Real-Time Document Intelligence & Compliance Engine
Deterministic Rule Engine Module
Version: 1.0
Date: 2026-02-03

This module provides a deterministic rule engine with versioned rules for compliance checking,
rule validation, conflict detection, and performance optimization.
"""

import logging
import uuid
from typing import Dict, List, Any, Optional, Union, Callable, Set, Tuple
from datetime import datetime, timedelta
from enum import Enum
from abc import ABC, abstractmethod
import json
import re
from dataclasses import dataclass, asdict
from copy import deepcopy

logger = logging.getLogger(__name__)


class RuleType(Enum):
    """Types of compliance rules"""

    COMPLIANCE_CHECK = "COMPLIANCE_CHECK"
    VALIDATION_RULE = "VALIDATION_RULE"
    BUSINESS_RULE = "BUSINESS_RULE"
    REGULATORY_RULE = "REGULATORY_RULE"
    APPROVAL_RULE = "APPROVAL_RULE"
    THRESHOLD_RULE = "THRESHOLD_RULE"
    PATTERN_RULE = "PATTERN_RULE"
    CONDITIONAL_RULE = "CONDITIONAL_RULE"


class RuleStatus(Enum):
    """Rule status states"""

    DRAFT = "DRAFT"
    ACTIVE = "ACTIVE"
    INACTIVE = "INACTIVE"
    DEPRECATED = "DEPRECATED"
    TESTING = "TESTING"


class Severity(Enum):
    """Rule violation severity levels"""

    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    INFO = "INFO"


class OperatorType(Enum):
    """Rule condition operators"""

    EQUALS = "EQUALS"
    NOT_EQUALS = "NOT_EQUALS"
    GREATER_THAN = "GREATER_THAN"
    LESS_THAN = "LESS_THAN"
    GREATER_EQUAL = "GREATER_EQUAL"
    LESS_EQUAL = "LESS_EQUAL"
    CONTAINS = "CONTAINS"
    NOT_CONTAINS = "NOT_CONTAINS"
    STARTS_WITH = "STARTS_WITH"
    ENDS_WITH = "ENDS_WITH"
    REGEX_MATCH = "REGEX_MATCH"
    IN_LIST = "IN_LIST"
    NOT_IN_LIST = "NOT_IN_LIST"
    IS_NULL = "IS_NULL"
    IS_NOT_NULL = "IS_NOT_NULL"


@dataclass
class RuleCondition:
    """Individual rule condition"""

    field_path: str
    operator: OperatorType
    value: Any
    condition_id: str = None

    def __post_init__(self):
        if self.condition_id is None:
            self.condition_id = str(uuid.uuid4())


@dataclass
class RuleAction:
    """Action to take when rule is triggered"""

    action_type: str
    parameters: Dict[str, Any]
    action_id: str = None

    def __post_init__(self):
        if self.action_id is None:
            self.action_id = str(uuid.uuid4())


@dataclass
class RuleMetadata:
    """Rule metadata and configuration"""

    created_by: str
    created_at: datetime
    updated_by: Optional[str] = None
    updated_at: Optional[datetime] = None
    tags: List[str] = None
    categories: List[str] = None
    compliance_frameworks: List[str] = None
    business_impact: str = None

    def __post_init__(self):
        if self.tags is None:
            self.tags = []
        if self.categories is None:
            self.categories = []
        if self.compliance_frameworks is None:
            self.compliance_frameworks = []


class ComplianceRule:
    """Comprehensive compliance rule definition"""

    def __init__(
        self,
        rule_id: str,
        name: str,
        description: str,
        rule_type: RuleType,
        conditions: List[RuleCondition],
        actions: List[RuleAction],
        severity: Severity = Severity.MEDIUM,
        status: RuleStatus = RuleStatus.DRAFT,
        metadata: Optional[RuleMetadata] = None,
        logical_operator: str = "AND",
        priority: int = 100,
        enabled: bool = True,
    ):
        """
        Initialize compliance rule

        Args:
            rule_id: Unique rule identifier
            name: Rule name
            description: Rule description
            rule_type: Type of rule
            conditions: List of rule conditions
            actions: List of actions to execute
            severity: Rule violation severity
            status: Rule status
            metadata: Additional metadata
            logical_operator: Operator for combining conditions (AND/OR)
            priority: Rule execution priority (lower = higher priority)
            enabled: Whether rule is enabled
        """
        self.rule_id = rule_id
        self.name = name
        self.description = description
        self.rule_type = rule_type
        self.conditions = conditions
        self.actions = actions
        self.severity = severity
        self.status = status
        self.metadata = metadata
        self.logical_operator = logical_operator
        self.priority = priority
        self.enabled = enabled

        # Version tracking
        self.version = "1.0.0"
        self.parent_rule_id = None
        self.child_rule_ids = []

        # Performance tracking
        self.execution_count = 0
        self.success_count = 0
        self.failure_count = 0
        self.average_execution_time = 0.0
        self.last_executed = None

        # Validation
        self._validate_rule()

    def _validate_rule(self):
        """Validate rule definition"""
        if not self.rule_id or not self.name:
            raise ValueError("Rule ID and name are required")

        if not self.conditions:
            raise ValueError("At least one condition is required")

        if not self.actions:
            raise ValueError("At least one action is required")

        if self.logical_operator not in ["AND", "OR"]:
            raise ValueError("Logical operator must be AND or OR")

    def evaluate(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate rule against provided data

        Args:
            data: Data to evaluate against rule conditions

        Returns:
            Evaluation result with details
        """
        try:
            start_time = datetime.now()

            if not self.enabled or self.status not in [
                RuleStatus.ACTIVE,
                RuleStatus.TESTING,
            ]:
                return {
                    "rule_id": self.rule_id,
                    "triggered": False,
                    "reason": f"Rule is {self.status.value} or disabled",
                }

            # Evaluate all conditions
            condition_results = []
            for condition in self.conditions:
                result = self._evaluate_condition(condition, data)
                condition_results.append(result)

            # Apply logical operator
            if self.logical_operator == "AND":
                rule_triggered = all(
                    result["satisfied"] for result in condition_results
                )
            else:  # OR
                rule_triggered = any(
                    result["satisfied"] for result in condition_results
                )

            # Calculate execution time
            execution_time = (datetime.now() - start_time).total_seconds()
            self._update_performance_metrics(execution_time, True)

            evaluation_result = {
                "rule_id": self.rule_id,
                "rule_name": self.name,
                "rule_type": self.rule_type.value,
                "triggered": rule_triggered,
                "severity": self.severity.value,
                "condition_results": condition_results,
                "logical_operator": self.logical_operator,
                "execution_time_ms": execution_time * 1000,
                "evaluated_at": datetime.now().isoformat(),
            }

            # Add actions if rule is triggered
            if rule_triggered:
                evaluation_result["actions"] = [
                    asdict(action) for action in self.actions
                ]

            return evaluation_result

        except Exception as e:
            self._update_performance_metrics(0, False)
            logger.error(f"Rule evaluation failed for {self.rule_id}: {e}")
            return {
                "rule_id": self.rule_id,
                "triggered": False,
                "error": str(e),
                "evaluated_at": datetime.now().isoformat(),
            }

    def _evaluate_condition(
        self, condition: RuleCondition, data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Evaluate individual condition"""
        try:
            # Extract field value using path
            field_value = self._extract_field_value(data, condition.field_path)

            # Apply operator
            satisfied = self._apply_operator(
                field_value, condition.operator, condition.value
            )

            return {
                "condition_id": condition.condition_id,
                "field_path": condition.field_path,
                "operator": condition.operator.value,
                "expected_value": condition.value,
                "actual_value": field_value,
                "satisfied": satisfied,
            }

        except Exception as e:
            return {
                "condition_id": condition.condition_id,
                "field_path": condition.field_path,
                "satisfied": False,
                "error": str(e),
            }

    def _extract_field_value(self, data: Dict[str, Any], field_path: str) -> Any:
        """Extract field value using dot notation path"""
        try:
            current_data = data
            path_parts = field_path.split(".")

            for part in path_parts:
                if isinstance(current_data, dict):
                    current_data = current_data.get(part)
                elif isinstance(current_data, list) and part.isdigit():
                    index = int(part)
                    current_data = (
                        current_data[index] if index < len(current_data) else None
                    )
                else:
                    return None

            return current_data

        except Exception:
            return None

    def _apply_operator(
        self, field_value: Any, operator: OperatorType, expected_value: Any
    ) -> bool:
        """Apply operator to compare field value with expected value"""
        try:
            if operator == OperatorType.EQUALS:
                return field_value == expected_value
            elif operator == OperatorType.NOT_EQUALS:
                return field_value != expected_value
            elif operator == OperatorType.GREATER_THAN:
                return field_value > expected_value
            elif operator == OperatorType.LESS_THAN:
                return field_value < expected_value
            elif operator == OperatorType.GREATER_EQUAL:
                return field_value >= expected_value
            elif operator == OperatorType.LESS_EQUAL:
                return field_value <= expected_value
            elif operator == OperatorType.CONTAINS:
                return (
                    expected_value in str(field_value)
                    if field_value is not None
                    else False
                )
            elif operator == OperatorType.NOT_CONTAINS:
                return (
                    expected_value not in str(field_value)
                    if field_value is not None
                    else True
                )
            elif operator == OperatorType.STARTS_WITH:
                return (
                    str(field_value).startswith(str(expected_value))
                    if field_value is not None
                    else False
                )
            elif operator == OperatorType.ENDS_WITH:
                return (
                    str(field_value).endswith(str(expected_value))
                    if field_value is not None
                    else False
                )
            elif operator == OperatorType.REGEX_MATCH:
                return (
                    bool(re.search(str(expected_value), str(field_value)))
                    if field_value is not None
                    else False
                )
            elif operator == OperatorType.IN_LIST:
                return (
                    field_value in expected_value
                    if isinstance(expected_value, list)
                    else False
                )
            elif operator == OperatorType.NOT_IN_LIST:
                return (
                    field_value not in expected_value
                    if isinstance(expected_value, list)
                    else True
                )
            elif operator == OperatorType.IS_NULL:
                return field_value is None
            elif operator == OperatorType.IS_NOT_NULL:
                return field_value is not None
            else:
                return False

        except Exception:
            return False

    def _update_performance_metrics(self, execution_time: float, success: bool):
        """Update rule performance metrics"""
        try:
            self.execution_count += 1
            self.last_executed = datetime.now()

            if success:
                self.success_count += 1
            else:
                self.failure_count += 1

            # Update average execution time
            current_avg = self.average_execution_time
            new_avg = (
                (current_avg * (self.execution_count - 1)) + execution_time
            ) / self.execution_count
            self.average_execution_time = new_avg

        except Exception:
            pass

    def to_dict(self) -> Dict[str, Any]:
        """Convert rule to dictionary representation"""
        return {
            "rule_id": self.rule_id,
            "name": self.name,
            "description": self.description,
            "rule_type": self.rule_type.value,
            "conditions": [asdict(condition) for condition in self.conditions],
            "actions": [asdict(action) for action in self.actions],
            "severity": self.severity.value,
            "status": self.status.value,
            "metadata": asdict(self.metadata) if self.metadata else None,
            "logical_operator": self.logical_operator,
            "priority": self.priority,
            "enabled": self.enabled,
            "version": self.version,
            "performance_metrics": {
                "execution_count": self.execution_count,
                "success_count": self.success_count,
                "failure_count": self.failure_count,
                "average_execution_time": self.average_execution_time,
                "last_executed": self.last_executed.isoformat()
                if self.last_executed
                else None,
            },
        }


class RuleEngine:
    """
    Deterministic rule engine with versioning and optimization
    """

    def __init__(
        self,
        enable_versioning: bool = True,
        enable_performance_optimization: bool = True,
        rule_cache_size: int = 1000,
        execution_timeout_seconds: int = 30,
    ):
        """
        Initialize rule engine

        Args:
            enable_versioning: Enable rule versioning
            enable_performance_optimization: Enable performance optimizations
            rule_cache_size: Maximum number of rules to cache
            execution_timeout_seconds: Maximum execution time per rule
        """
        self.enable_versioning = enable_versioning
        self.enable_performance_optimization = enable_performance_optimization
        self.rule_cache_size = rule_cache_size
        self.execution_timeout = execution_timeout_seconds

        # Rule storage
        self.rules: Dict[str, ComplianceRule] = {}
        self.rule_versions: Dict[str, List[str]] = {}  # rule_name -> [rule_ids]
        self.active_rules: Dict[str, str] = {}  # rule_name -> active_rule_id

        # Rule organization
        self.rules_by_type: Dict[RuleType, List[str]] = {
            rule_type: [] for rule_type in RuleType
        }
        self.rules_by_category: Dict[str, List[str]] = {}
        self.rule_dependencies: Dict[str, Set[str]] = {}

        # Performance optimization
        self.rule_cache = {}
        self.execution_stats = {
            "total_evaluations": 0,
            "total_execution_time": 0,
            "cache_hits": 0,
            "cache_misses": 0,
        }

        # Conflict detection
        self.rule_conflicts: List[Dict[str, Any]] = []

        logger.info("Rule engine initialized")

    def add_rule(self, rule: ComplianceRule) -> bool:
        """
        Add rule to the engine

        Args:
            rule: ComplianceRule instance

        Returns:
            True if rule was added successfully
        """
        try:
            # Validate rule
            if rule.rule_id in self.rules:
                if not self.enable_versioning:
                    raise ValueError(f"Rule {rule.rule_id} already exists")

                # Handle versioning
                self._handle_rule_versioning(rule)

            # Store rule
            self.rules[rule.rule_id] = rule

            # Update indices
            self._update_rule_indices(rule)

            # Detect conflicts
            self._detect_rule_conflicts(rule)

            # Clear cache if performance optimization is enabled
            if self.enable_performance_optimization:
                self._clear_affected_cache(rule)

            logger.info(f"Added rule: {rule.name} ({rule.rule_id})")

            return True

        except Exception as e:
            logger.error(f"Failed to add rule {rule.rule_id}: {e}")
            return False

    def remove_rule(self, rule_id: str) -> bool:
        """Remove rule from the engine"""
        try:
            if rule_id not in self.rules:
                return False

            rule = self.rules[rule_id]

            # Remove from storage
            del self.rules[rule_id]

            # Update indices
            self._remove_from_indices(rule)

            # Clear cache
            if self.enable_performance_optimization:
                self._clear_affected_cache(rule)

            logger.info(f"Removed rule: {rule.name} ({rule_id})")

            return True

        except Exception as e:
            logger.error(f"Failed to remove rule {rule_id}: {e}")
            return False

    def evaluate_rules(
        self,
        data: Dict[str, Any],
        rule_types: Optional[List[RuleType]] = None,
        categories: Optional[List[str]] = None,
        max_priority: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Evaluate rules against provided data

        Args:
            data: Data to evaluate
            rule_types: Specific rule types to evaluate (None = all)
            categories: Specific categories to evaluate (None = all)
            max_priority: Maximum priority level to evaluate (None = all)

        Returns:
            Evaluation results
        """
        try:
            evaluation_start = datetime.now()

            # Get applicable rules
            applicable_rules = self._get_applicable_rules(
                rule_types, categories, max_priority
            )

            # Sort by priority
            applicable_rules.sort(key=lambda r: r.priority)

            # Evaluate rules
            results = {
                "evaluation_id": str(uuid.uuid4()),
                "evaluated_at": evaluation_start.isoformat(),
                "rule_results": [],
                "triggered_rules": [],
                "violations": [],
                "summary": {
                    "total_rules_evaluated": len(applicable_rules),
                    "rules_triggered": 0,
                    "critical_violations": 0,
                    "high_violations": 0,
                    "medium_violations": 0,
                    "low_violations": 0,
                },
            }

            for rule in applicable_rules:
                try:
                    # Check cache first
                    cache_key = self._generate_cache_key(rule, data)
                    cached_result = None

                    if self.enable_performance_optimization:
                        cached_result = self.rule_cache.get(cache_key)

                        if cached_result:
                            self.execution_stats["cache_hits"] += 1
                            rule_result = cached_result
                        else:
                            self.execution_stats["cache_misses"] += 1

                    if not cached_result:
                        # Evaluate rule
                        rule_result = rule.evaluate(data)

                        # Cache result
                        if self.enable_performance_optimization:
                            self._cache_result(cache_key, rule_result)

                    results["rule_results"].append(rule_result)

                    # Track triggered rules
                    if rule_result.get("triggered", False):
                        results["triggered_rules"].append(rule_result)
                        results["summary"]["rules_triggered"] += 1

                        # Create violation record
                        violation = {
                            "rule_id": rule.rule_id,
                            "rule_name": rule.name,
                            "severity": rule.severity.value,
                            "rule_type": rule.rule_type.value,
                            "description": rule.description,
                            "actions": rule_result.get("actions", []),
                            "condition_results": rule_result.get(
                                "condition_results", []
                            ),
                        }
                        results["violations"].append(violation)

                        # Update severity counts
                        severity_key = f"{rule.severity.value.lower()}_violations"
                        if severity_key in results["summary"]:
                            results["summary"][severity_key] += 1

                except Exception as e:
                    logger.error(f"Rule evaluation failed for {rule.rule_id}: {e}")
                    results["rule_results"].append(
                        {"rule_id": rule.rule_id, "triggered": False, "error": str(e)}
                    )

            # Calculate total execution time
            execution_time = (datetime.now() - evaluation_start).total_seconds()
            results["execution_time_ms"] = execution_time * 1000

            # Update engine stats
            self._update_engine_stats(execution_time, len(applicable_rules))

            return results

        except Exception as e:
            logger.error(f"Rule evaluation failed: {e}")
            return {
                "evaluation_id": str(uuid.uuid4()),
                "error": str(e),
                "rule_results": [],
                "triggered_rules": [],
                "violations": [],
            }

    def validate_rule_set(self) -> Dict[str, Any]:
        """Validate entire rule set for conflicts and issues"""
        try:
            validation_result = {
                "is_valid": True,
                "total_rules": len(self.rules),
                "active_rules": len(
                    [r for r in self.rules.values() if r.status == RuleStatus.ACTIVE]
                ),
                "conflicts": [],
                "issues": [],
                "performance_warnings": [],
            }

            # Check for rule conflicts
            conflicts = self._detect_all_rule_conflicts()
            validation_result["conflicts"] = conflicts

            if conflicts:
                validation_result["is_valid"] = False

            # Check for performance issues
            performance_issues = self._analyze_rule_performance()
            validation_result["performance_warnings"] = performance_issues

            # Check for rule issues
            for rule in self.rules.values():
                issues = self._validate_individual_rule(rule)
                if issues:
                    validation_result["issues"].extend(issues)
                    validation_result["is_valid"] = False

            return validation_result

        except Exception as e:
            logger.error(f"Rule set validation failed: {e}")
            return {"is_valid": False, "error": str(e)}

    def get_rule_statistics(self) -> Dict[str, Any]:
        """Get comprehensive rule engine statistics"""
        try:
            stats = {
                "rule_counts": {
                    "total": len(self.rules),
                    "by_status": {},
                    "by_type": {},
                    "by_severity": {},
                },
                "performance_metrics": {
                    "total_evaluations": self.execution_stats["total_evaluations"],
                    "average_execution_time": (
                        self.execution_stats["total_execution_time"]
                        / max(1, self.execution_stats["total_evaluations"])
                    ),
                    "cache_hit_rate": (
                        self.execution_stats["cache_hits"]
                        / max(
                            1,
                            self.execution_stats["cache_hits"]
                            + self.execution_stats["cache_misses"],
                        )
                    ),
                },
                "rule_health": {
                    "conflicts_detected": len(self.rule_conflicts),
                    "rules_with_errors": 0,
                    "average_rule_execution_time": 0,
                },
            }

            # Count by status
            for status in RuleStatus:
                count = len([r for r in self.rules.values() if r.status == status])
                stats["rule_counts"]["by_status"][status.value] = count

            # Count by type
            for rule_type in RuleType:
                count = len(self.rules_by_type[rule_type])
                stats["rule_counts"]["by_type"][rule_type.value] = count

            # Count by severity
            for severity in Severity:
                count = len([r for r in self.rules.values() if r.severity == severity])
                stats["rule_counts"]["by_severity"][severity.value] = count

            # Calculate average rule execution time
            if self.rules:
                total_exec_time = sum(
                    r.average_execution_time for r in self.rules.values()
                )
                stats["rule_health"]["average_rule_execution_time"] = (
                    total_exec_time / len(self.rules)
                )

            # Count rules with errors
            stats["rule_health"]["rules_with_errors"] = len(
                [r for r in self.rules.values() if r.failure_count > 0]
            )

            return stats

        except Exception as e:
            logger.error(f"Failed to get rule statistics: {e}")
            return {"error": str(e)}

    # Helper methods
    def _handle_rule_versioning(self, rule: ComplianceRule):
        """Handle rule versioning"""
        try:
            # Find existing rules with same name
            existing_versions = []
            for existing_rule in self.rules.values():
                if existing_rule.name == rule.name:
                    existing_versions.append(existing_rule)

            if existing_versions:
                # Increment version
                latest_version = max(existing_versions, key=lambda r: r.version)
                version_parts = latest_version.version.split(".")
                major, minor, patch = (
                    int(version_parts[0]),
                    int(version_parts[1]),
                    int(version_parts[2]),
                )

                # Increment patch version
                rule.version = f"{major}.{minor}.{patch + 1}"
                rule.parent_rule_id = latest_version.rule_id
                latest_version.child_rule_ids.append(rule.rule_id)

                # Update version tracking
                if rule.name not in self.rule_versions:
                    self.rule_versions[rule.name] = []
                self.rule_versions[rule.name].append(rule.rule_id)

                # Update active rule
                if rule.status == RuleStatus.ACTIVE:
                    # Deactivate previous version
                    latest_version.status = RuleStatus.DEPRECATED
                    self.active_rules[rule.name] = rule.rule_id

        except Exception as e:
            logger.warning(f"Rule versioning failed: {e}")

    def _update_rule_indices(self, rule: ComplianceRule):
        """Update rule indices"""
        try:
            # Update type index
            if rule.rule_id not in self.rules_by_type[rule.rule_type]:
                self.rules_by_type[rule.rule_type].append(rule.rule_id)

            # Update category indices
            if rule.metadata and rule.metadata.categories:
                for category in rule.metadata.categories:
                    if category not in self.rules_by_category:
                        self.rules_by_category[category] = []
                    if rule.rule_id not in self.rules_by_category[category]:
                        self.rules_by_category[category].append(rule.rule_id)

        except Exception as e:
            logger.warning(f"Failed to update rule indices: {e}")

    def _remove_from_indices(self, rule: ComplianceRule):
        """Remove rule from indices"""
        try:
            # Remove from type index
            if rule.rule_id in self.rules_by_type[rule.rule_type]:
                self.rules_by_type[rule.rule_type].remove(rule.rule_id)

            # Remove from category indices
            if rule.metadata and rule.metadata.categories:
                for category in rule.metadata.categories:
                    if (
                        category in self.rules_by_category
                        and rule.rule_id in self.rules_by_category[category]
                    ):
                        self.rules_by_category[category].remove(rule.rule_id)

        except Exception as e:
            logger.warning(f"Failed to remove rule from indices: {e}")

    def _get_applicable_rules(
        self,
        rule_types: Optional[List[RuleType]],
        categories: Optional[List[str]],
        max_priority: Optional[int],
    ) -> List[ComplianceRule]:
        """Get rules applicable for evaluation"""
        applicable_rules = []

        for rule in self.rules.values():
            # Check if rule is enabled and active
            if not rule.enabled or rule.status != RuleStatus.ACTIVE:
                continue

            # Check rule type filter
            if rule_types and rule.rule_type not in rule_types:
                continue

            # Check category filter
            if categories and rule.metadata and rule.metadata.categories:
                if not any(cat in rule.metadata.categories for cat in categories):
                    continue

            # Check priority filter
            if max_priority and rule.priority > max_priority:
                continue

            applicable_rules.append(rule)

        return applicable_rules

    def _detect_rule_conflicts(self, new_rule: ComplianceRule):
        """Detect conflicts with existing rules"""
        try:
            conflicts = []

            for existing_rule in self.rules.values():
                if existing_rule.rule_id == new_rule.rule_id:
                    continue

                # Check for conflicting conditions or actions
                conflict = self._analyze_rule_conflict(new_rule, existing_rule)
                if conflict:
                    conflicts.append(conflict)

            if conflicts:
                self.rule_conflicts.extend(conflicts)

        except Exception as e:
            logger.warning(f"Conflict detection failed: {e}")

    def _analyze_rule_conflict(
        self, rule1: ComplianceRule, rule2: ComplianceRule
    ) -> Optional[Dict[str, Any]]:
        """Analyze potential conflict between two rules"""
        try:
            # Simple conflict detection - same field paths with contradictory conditions
            rule1_fields = set(condition.field_path for condition in rule1.conditions)
            rule2_fields = set(condition.field_path for condition in rule2.conditions)

            common_fields = rule1_fields & rule2_fields

            if common_fields:
                # Check for contradictory operators
                for field in common_fields:
                    rule1_conditions = [
                        c for c in rule1.conditions if c.field_path == field
                    ]
                    rule2_conditions = [
                        c for c in rule2.conditions if c.field_path == field
                    ]

                    for c1 in rule1_conditions:
                        for c2 in rule2_conditions:
                            if self._are_conditions_conflicting(c1, c2):
                                return {
                                    "conflict_type": "contradictory_conditions",
                                    "rule1_id": rule1.rule_id,
                                    "rule2_id": rule2.rule_id,
                                    "conflicting_field": field,
                                    "condition1": asdict(c1),
                                    "condition2": asdict(c2),
                                }

            return None

        except Exception:
            return None

    def _are_conditions_conflicting(
        self, condition1: RuleCondition, condition2: RuleCondition
    ) -> bool:
        """Check if two conditions are conflicting"""
        # Simplified conflict detection
        conflicting_operators = [
            (OperatorType.EQUALS, OperatorType.NOT_EQUALS),
            (OperatorType.GREATER_THAN, OperatorType.LESS_EQUAL),
            (OperatorType.LESS_THAN, OperatorType.GREATER_EQUAL),
            (OperatorType.CONTAINS, OperatorType.NOT_CONTAINS),
            (OperatorType.IN_LIST, OperatorType.NOT_IN_LIST),
            (OperatorType.IS_NULL, OperatorType.IS_NOT_NULL),
        ]

        for op1, op2 in conflicting_operators:
            if (condition1.operator == op1 and condition2.operator == op2) or (
                condition1.operator == op2 and condition2.operator == op1
            ):
                return condition1.value == condition2.value

        return False

    def _detect_all_rule_conflicts(self) -> List[Dict[str, Any]]:
        """Detect all rule conflicts in the system"""
        return self.rule_conflicts.copy()

    def _analyze_rule_performance(self) -> List[Dict[str, Any]]:
        """Analyze rule performance for warnings"""
        warnings = []

        for rule in self.rules.values():
            # Check for slow rules
            if rule.average_execution_time > 1.0:  # 1 second threshold
                warnings.append(
                    {
                        "type": "slow_execution",
                        "rule_id": rule.rule_id,
                        "rule_name": rule.name,
                        "average_time": rule.average_execution_time,
                        "recommendation": "Consider optimizing rule conditions",
                    }
                )

            # Check for high failure rate
            if rule.execution_count > 10:
                failure_rate = rule.failure_count / rule.execution_count
                if failure_rate > 0.1:  # 10% failure rate threshold
                    warnings.append(
                        {
                            "type": "high_failure_rate",
                            "rule_id": rule.rule_id,
                            "rule_name": rule.name,
                            "failure_rate": failure_rate,
                            "recommendation": "Review rule conditions for errors",
                        }
                    )

        return warnings

    def _validate_individual_rule(self, rule: ComplianceRule) -> List[Dict[str, Any]]:
        """Validate individual rule for issues"""
        issues = []

        try:
            # Check for empty conditions
            if not rule.conditions:
                issues.append(
                    {
                        "type": "empty_conditions",
                        "rule_id": rule.rule_id,
                        "message": "Rule has no conditions",
                    }
                )

            # Check for invalid field paths
            for condition in rule.conditions:
                if not condition.field_path:
                    issues.append(
                        {
                            "type": "invalid_field_path",
                            "rule_id": rule.rule_id,
                            "condition_id": condition.condition_id,
                            "message": "Condition has empty field path",
                        }
                    )

            # Check for empty actions
            if not rule.actions:
                issues.append(
                    {
                        "type": "empty_actions",
                        "rule_id": rule.rule_id,
                        "message": "Rule has no actions",
                    }
                )

        except Exception:
            issues.append(
                {
                    "type": "validation_error",
                    "rule_id": rule.rule_id,
                    "message": "Rule validation failed",
                }
            )

        return issues

    def _generate_cache_key(self, rule: ComplianceRule, data: Dict[str, Any]) -> str:
        """Generate cache key for rule evaluation"""
        try:
            # Create hash from rule ID and relevant data
            relevant_data = {}
            for condition in rule.conditions:
                field_value = self._extract_field_value(data, condition.field_path)
                relevant_data[condition.field_path] = field_value

            cache_data = {
                "rule_id": rule.rule_id,
                "rule_version": rule.version,
                "data": relevant_data,
            }

            return str(hash(json.dumps(cache_data, sort_keys=True)))

        except Exception:
            return f"{rule.rule_id}_{hash(str(data))}"

    def _cache_result(self, cache_key: str, result: Dict[str, Any]):
        """Cache evaluation result"""
        try:
            if len(self.rule_cache) >= self.rule_cache_size:
                # Remove oldest entry
                oldest_key = next(iter(self.rule_cache))
                del self.rule_cache[oldest_key]

            self.rule_cache[cache_key] = result

        except Exception:
            pass

    def _clear_affected_cache(self, rule: ComplianceRule):
        """Clear cache entries affected by rule changes"""
        try:
            # Clear all cache entries for this rule
            keys_to_remove = [
                key for key in self.rule_cache.keys() if rule.rule_id in key
            ]
            for key in keys_to_remove:
                del self.rule_cache[key]

        except Exception:
            pass

    def _update_engine_stats(self, execution_time: float, rules_evaluated: int):
        """Update engine statistics"""
        try:
            self.execution_stats["total_evaluations"] += 1
            self.execution_stats["total_execution_time"] += execution_time

        except Exception:
            pass
