"""
Real-Time Document Intelligence & Compliance Engine
Compliance Audit Trail System
Version: 1.0
Date: 2026-02-03

This module provides comprehensive audit logging, evidence collection,
regulatory reporting, and compliance documentation generation with
tamper-proof logging and regulatory compliance features.
"""

import logging
from typing import Dict, Any, List, Optional, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import json
import hashlib
import uuid
from collections import defaultdict
import os
import asyncio
from concurrent.futures import ThreadPoolExecutor
import threading

logger = logging.getLogger(__name__)


class AuditEventType(Enum):
    """Types of audit events"""

    DOCUMENT_PROCESSED = "document_processed"
    RULE_VIOLATION = "rule_violation"
    ANOMALY_DETECTED = "anomaly_detected"
    RISK_ASSESSMENT = "risk_assessment"
    COMPLIANCE_CHECK = "compliance_check"
    USER_ACTION = "user_action"
    SYSTEM_ERROR = "system_error"
    DATA_ACCESS = "data_access"
    POLICY_CHANGE = "policy_change"
    ALERT_GENERATED = "alert_generated"


class ComplianceFramework(Enum):
    """Compliance frameworks"""

    SOX = "sarbanes_oxley"
    GDPR = "general_data_protection_regulation"
    HIPAA = "health_insurance_portability_accountability"
    PCI_DSS = "payment_card_industry_data_security"
    ISO_27001 = "iso_27001"
    COSO = "committee_of_sponsoring_organizations"
    BASEL_III = "basel_iii"
    MiFID_II = "markets_in_financial_instruments_directive"


class AuditLevel(Enum):
    """Audit logging levels"""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFORMATIONAL = "informational"


@dataclass
class AuditEvent:
    """Individual audit event record"""

    event_id: str
    event_type: AuditEventType
    timestamp: datetime
    user_id: str
    session_id: str
    document_id: Optional[str]
    event_description: str
    event_data: Dict[str, Any]
    compliance_frameworks: List[ComplianceFramework]
    audit_level: AuditLevel
    source_component: str
    ip_address: Optional[str]
    user_agent: Optional[str]
    outcome: str  # SUCCESS, FAILURE, WARNING
    evidence: Dict[str, Any]
    hash_value: str = field(init=False)

    def __post_init__(self):
        """Calculate hash for tamper detection"""
        self.hash_value = self._calculate_hash()

    def _calculate_hash(self) -> str:
        """Calculate SHA-256 hash of event data for integrity"""
        hash_data = {
            "event_id": self.event_id,
            "event_type": self.event_type.value,
            "timestamp": self.timestamp.isoformat(),
            "user_id": self.user_id,
            "session_id": self.session_id,
            "document_id": self.document_id,
            "event_description": self.event_description,
            "outcome": self.outcome,
        }

        hash_string = json.dumps(hash_data, sort_keys=True)
        return hashlib.sha256(hash_string.encode()).hexdigest()

    def verify_integrity(self) -> bool:
        """Verify event integrity"""
        return self.hash_value == self._calculate_hash()


@dataclass
class AuditChain:
    """Blockchain-style audit chain for tamper-proof logging"""

    chain_id: str
    previous_hash: Optional[str]
    events: List[AuditEvent]
    block_timestamp: datetime
    block_hash: str = field(init=False)

    def __post_init__(self):
        """Calculate block hash"""
        self.block_hash = self._calculate_block_hash()

    def _calculate_block_hash(self) -> str:
        """Calculate block hash including previous hash"""
        block_data = {
            "chain_id": self.chain_id,
            "previous_hash": self.previous_hash,
            "event_hashes": [event.hash_value for event in self.events],
            "block_timestamp": self.block_timestamp.isoformat(),
        }

        hash_string = json.dumps(block_data, sort_keys=True)
        return hashlib.sha256(hash_string.encode()).hexdigest()

    def verify_chain_integrity(self) -> bool:
        """Verify chain integrity"""
        # Verify block hash
        if self.block_hash != self._calculate_block_hash():
            return False

        # Verify all events
        for event in self.events:
            if not event.verify_integrity():
                return False

        return True


@dataclass
class ComplianceReport:
    """Compliance report for regulatory purposes"""

    report_id: str
    report_type: str
    framework: ComplianceFramework
    reporting_period_start: datetime
    reporting_period_end: datetime
    generated_at: datetime
    generated_by: str

    # Report content
    executive_summary: str
    compliance_status: str
    total_events: int
    violations_count: int
    anomalies_count: int
    risk_events_count: int

    # Detailed sections
    event_summary: Dict[str, int]
    violation_analysis: List[Dict[str, Any]]
    risk_assessment: Dict[str, Any]
    remediation_actions: List[str]
    control_effectiveness: Dict[str, float]

    # Evidence and attachments
    supporting_evidence: List[str]
    audit_trail_references: List[str]

    metadata: Dict[str, Any]


class AuditTrailManager:
    """Comprehensive audit trail management system"""

    def __init__(self, storage_path: str = "./audit_logs"):
        """
        Initialize audit trail manager

        Args:
            storage_path: Path for audit log storage
        """
        self.storage_path = storage_path
        self.active_sessions: Dict[str, Dict[str, Any]] = {}
        self.audit_chains: Dict[str, List[AuditChain]] = defaultdict(list)
        self.compliance_events: Dict[ComplianceFramework, List[AuditEvent]] = (
            defaultdict(list)
        )

        # Thread safety
        self._lock = threading.RLock()
        self._executor = ThreadPoolExecutor(max_workers=4)

        # Initialize storage
        os.makedirs(storage_path, exist_ok=True)

        # Configuration
        self.retention_policies = {
            ComplianceFramework.SOX: 2555,  # 7 years in days
            ComplianceFramework.GDPR: 2190,  # 6 years
            ComplianceFramework.HIPAA: 2190,  # 6 years
            ComplianceFramework.PCI_DSS: 365,  # 1 year
            ComplianceFramework.ISO_27001: 1095,  # 3 years
        }

        # Event filters for different compliance frameworks
        self.framework_filters = {
            ComplianceFramework.SOX: {
                "required_events": [
                    AuditEventType.DOCUMENT_PROCESSED,
                    AuditEventType.USER_ACTION,
                    AuditEventType.DATA_ACCESS,
                ],
                "audit_levels": [
                    AuditLevel.CRITICAL,
                    AuditLevel.HIGH,
                    AuditLevel.MEDIUM,
                ],
            },
            ComplianceFramework.GDPR: {
                "required_events": [
                    AuditEventType.DATA_ACCESS,
                    AuditEventType.USER_ACTION,
                    AuditEventType.DOCUMENT_PROCESSED,
                ],
                "audit_levels": [AuditLevel.CRITICAL, AuditLevel.HIGH],
            },
            ComplianceFramework.HIPAA: {
                "required_events": [
                    AuditEventType.DATA_ACCESS,
                    AuditEventType.DOCUMENT_PROCESSED,
                ],
                "audit_levels": [
                    AuditLevel.CRITICAL,
                    AuditLevel.HIGH,
                    AuditLevel.MEDIUM,
                ],
            },
        }

        logger.info(
            f"Audit trail manager initialized with storage path: {storage_path}"
        )

    def log_event(
        self,
        event_type: AuditEventType,
        user_id: str,
        session_id: str,
        description: str,
        event_data: Dict[str, Any] = None,
        document_id: str = None,
        compliance_frameworks: List[ComplianceFramework] = None,
        audit_level: AuditLevel = AuditLevel.MEDIUM,
        source_component: str = "system",
        outcome: str = "SUCCESS",
        evidence: Dict[str, Any] = None,
        ip_address: str = None,
        user_agent: str = None,
    ) -> str:
        """
        Log an audit event

        Returns:
            Event ID
        """
        try:
            event_id = str(uuid.uuid4())
            current_time = datetime.now()

            # Default values
            if event_data is None:
                event_data = {}
            if compliance_frameworks is None:
                compliance_frameworks = []
            if evidence is None:
                evidence = {}

            # Create audit event
            audit_event = AuditEvent(
                event_id=event_id,
                event_type=event_type,
                timestamp=current_time,
                user_id=user_id,
                session_id=session_id,
                document_id=document_id,
                event_description=description,
                event_data=event_data,
                compliance_frameworks=compliance_frameworks,
                audit_level=audit_level,
                source_component=source_component,
                ip_address=ip_address,
                user_agent=user_agent,
                outcome=outcome,
                evidence=evidence,
            )

            # Store event with thread safety
            with self._lock:
                self._store_event(audit_event)

                # Add to compliance framework events
                for framework in compliance_frameworks:
                    self.compliance_events[framework].append(audit_event)

            # Async processing
            self._executor.submit(self._process_event_async, audit_event)

            logger.debug(f"Audit event logged: {event_id}")
            return event_id

        except Exception as e:
            logger.error(f"Failed to log audit event: {e}")
            raise

    def _store_event(self, event: AuditEvent) -> None:
        """Store audit event in audit chain"""
        try:
            # Get or create audit chain for session
            session_chains = self.audit_chains[event.session_id]

            # Get previous hash
            previous_hash = None
            if session_chains:
                previous_hash = session_chains[-1].block_hash

            # Create new chain block with single event
            chain_block = AuditChain(
                chain_id=str(uuid.uuid4()),
                previous_hash=previous_hash,
                events=[event],
                block_timestamp=datetime.now(),
            )

            # Add to chains
            session_chains.append(chain_block)

        except Exception as e:
            logger.error(f"Failed to store audit event: {e}")
            raise

    def _process_event_async(self, event: AuditEvent) -> None:
        """Process audit event asynchronously"""
        try:
            # Write to persistent storage
            self._persist_event(event)

            # Check for compliance violations
            self._check_compliance_violations(event)

            # Update session tracking
            self._update_session_tracking(event)

        except Exception as e:
            logger.error(f"Async event processing failed: {e}")

    def _persist_event(self, event: AuditEvent) -> None:
        """Persist event to storage"""
        try:
            # Create daily log file
            date_str = event.timestamp.strftime("%Y-%m-%d")
            log_file = os.path.join(self.storage_path, f"audit_{date_str}.jsonl")

            # Event data for storage
            event_record = {
                "event_id": event.event_id,
                "event_type": event.event_type.value,
                "timestamp": event.timestamp.isoformat(),
                "user_id": event.user_id,
                "session_id": event.session_id,
                "document_id": event.document_id,
                "event_description": event.event_description,
                "event_data": event.event_data,
                "compliance_frameworks": [f.value for f in event.compliance_frameworks],
                "audit_level": event.audit_level.value,
                "source_component": event.source_component,
                "ip_address": event.ip_address,
                "user_agent": event.user_agent,
                "outcome": event.outcome,
                "evidence": event.evidence,
                "hash_value": event.hash_value,
            }

            # Append to log file
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(event_record) + "\n")

        except Exception as e:
            logger.error(f"Failed to persist event: {e}")

    def _check_compliance_violations(self, event: AuditEvent) -> None:
        """Check for potential compliance violations"""
        try:
            violations = []

            # Check for critical events without proper documentation
            if event.audit_level == AuditLevel.CRITICAL and not event.evidence:
                violations.append("Critical event lacks supporting evidence")

            # Check for suspicious user activity
            if event.event_type == AuditEventType.DATA_ACCESS:
                session_events = self._get_session_events(event.session_id)
                if len(session_events) > 100:  # Threshold for excessive activity
                    violations.append("Excessive data access activity detected")

            # Log violations
            if violations:
                violation_event = AuditEvent(
                    event_id=str(uuid.uuid4()),
                    event_type=AuditEventType.RULE_VIOLATION,
                    timestamp=datetime.now(),
                    user_id="system",
                    session_id=event.session_id,
                    document_id=None,
                    event_description=f"Compliance violations detected: {'; '.join(violations)}",
                    event_data={
                        "original_event": event.event_id,
                        "violations": violations,
                    },
                    compliance_frameworks=event.compliance_frameworks,
                    audit_level=AuditLevel.HIGH,
                    source_component="audit_monitor",
                    ip_address=None,
                    user_agent=None,
                    outcome="WARNING",
                    evidence={"violation_analysis": violations},
                )

                self._store_event(violation_event)

        except Exception as e:
            logger.error(f"Compliance violation check failed: {e}")

    def _update_session_tracking(self, event: AuditEvent) -> None:
        """Update session tracking information"""
        try:
            session_id = event.session_id

            if session_id not in self.active_sessions:
                self.active_sessions[session_id] = {
                    "start_time": event.timestamp,
                    "last_activity": event.timestamp,
                    "user_id": event.user_id,
                    "event_count": 0,
                    "documents_accessed": set(),
                    "ip_addresses": set(),
                }

            session_info = self.active_sessions[session_id]
            session_info["last_activity"] = event.timestamp
            session_info["event_count"] += 1

            if event.document_id:
                session_info["documents_accessed"].add(event.document_id)

            if event.ip_address:
                session_info["ip_addresses"].add(event.ip_address)

        except Exception as e:
            logger.error(f"Session tracking update failed: {e}")

    def _get_session_events(self, session_id: str) -> List[AuditEvent]:
        """Get all events for a session"""
        session_events = []

        for chain_block in self.audit_chains.get(session_id, []):
            session_events.extend(chain_block.events)

        return session_events

    def generate_compliance_report(
        self,
        framework: ComplianceFramework,
        start_date: datetime,
        end_date: datetime,
        report_type: str = "standard",
    ) -> ComplianceReport:
        """
        Generate compliance report for specified framework and period

        Args:
            framework: Compliance framework
            start_date: Report period start
            end_date: Report period end
            report_type: Type of report (standard, detailed, executive)

        Returns:
            Compliance report
        """
        try:
            report_id = str(uuid.uuid4())
            current_time = datetime.now()

            # Filter events for framework and period
            framework_events = [
                event
                for event in self.compliance_events[framework]
                if start_date <= event.timestamp <= end_date
            ]

            # Analyze events
            violations = [
                event
                for event in framework_events
                if event.event_type == AuditEventType.RULE_VIOLATION
            ]
            anomalies = [
                event
                for event in framework_events
                if event.event_type == AuditEventType.ANOMALY_DETECTED
            ]
            risk_events = [
                event
                for event in framework_events
                if event.event_type == AuditEventType.RISK_ASSESSMENT
            ]

            # Event summary
            event_summary = defaultdict(int)
            for event in framework_events:
                event_summary[event.event_type.value] += 1

            # Compliance status
            compliance_status = self._determine_compliance_status(
                framework, violations, anomalies
            )

            # Executive summary
            executive_summary = self._generate_executive_summary(
                framework,
                len(framework_events),
                len(violations),
                len(anomalies),
                compliance_status,
            )

            # Violation analysis
            violation_analysis = self._analyze_violations(violations)

            # Risk assessment
            risk_assessment = self._assess_compliance_risks(framework_events)

            # Control effectiveness
            control_effectiveness = self._evaluate_control_effectiveness(
                framework, framework_events
            )

            # Remediation actions
            remediation_actions = self._generate_remediation_actions(
                violations, anomalies
            )

            report = ComplianceReport(
                report_id=report_id,
                report_type=report_type,
                framework=framework,
                reporting_period_start=start_date,
                reporting_period_end=end_date,
                generated_at=current_time,
                generated_by="audit_system",
                executive_summary=executive_summary,
                compliance_status=compliance_status,
                total_events=len(framework_events),
                violations_count=len(violations),
                anomalies_count=len(anomalies),
                risk_events_count=len(risk_events),
                event_summary=dict(event_summary),
                violation_analysis=violation_analysis,
                risk_assessment=risk_assessment,
                remediation_actions=remediation_actions,
                control_effectiveness=control_effectiveness,
                supporting_evidence=self._collect_supporting_evidence(framework_events),
                audit_trail_references=self._generate_audit_references(
                    framework_events
                ),
                metadata={
                    "framework_config": self.framework_filters.get(framework, {})
                },
            )

            logger.info(
                f"Compliance report generated: {report_id} for {framework.value}"
            )
            return report

        except Exception as e:
            logger.error(f"Compliance report generation failed: {e}")
            raise

    def _determine_compliance_status(
        self,
        framework: ComplianceFramework,
        violations: List[AuditEvent],
        anomalies: List[AuditEvent],
    ) -> str:
        """Determine overall compliance status"""
        critical_violations = [
            v for v in violations if v.audit_level == AuditLevel.CRITICAL
        ]
        high_violations = [v for v in violations if v.audit_level == AuditLevel.HIGH]

        if critical_violations:
            return "NON_COMPLIANT"
        elif len(high_violations) > 5:
            return "REQUIRES_ATTENTION"
        elif len(violations) > 10:
            return "MINOR_ISSUES"
        else:
            return "COMPLIANT"

    def _generate_executive_summary(
        self,
        framework: ComplianceFramework,
        total_events: int,
        violations: int,
        anomalies: int,
        status: str,
    ) -> str:
        """Generate executive summary"""
        return f"""
        Compliance Summary for {framework.value.replace("_", " ").title()}:

        Total audit events processed: {total_events:,}
        Compliance violations identified: {violations}
        Anomalies detected: {anomalies}
        Overall compliance status: {status}

        The system processed {total_events:,} audit events during the reporting period.
        {"No critical compliance issues were identified." if violations == 0 else f"{violations} compliance violations require attention."}
        {"Anomaly detection systems identified {anomalies} potential issues for review." if anomalies > 0 else "No significant anomalies detected."}
        """

    def _analyze_violations(self, violations: List[AuditEvent]) -> List[Dict[str, Any]]:
        """Analyze compliance violations"""
        analysis = []

        # Group by violation type
        violation_types = defaultdict(list)
        for violation in violations:
            violation_data = violation.event_data.get("violations", ["Unknown"])
            for v_type in violation_data:
                violation_types[v_type].append(violation)

        for v_type, events in violation_types.items():
            analysis.append(
                {
                    "violation_type": v_type,
                    "count": len(events),
                    "severity_distribution": {
                        "critical": len(
                            [e for e in events if e.audit_level == AuditLevel.CRITICAL]
                        ),
                        "high": len(
                            [e for e in events if e.audit_level == AuditLevel.HIGH]
                        ),
                        "medium": len(
                            [e for e in events if e.audit_level == AuditLevel.MEDIUM]
                        ),
                    },
                    "first_occurrence": min(e.timestamp for e in events).isoformat(),
                    "last_occurrence": max(e.timestamp for e in events).isoformat(),
                    "affected_users": list(set(e.user_id for e in events)),
                    "recommendations": self._get_violation_recommendations(v_type),
                }
            )

        return analysis

    def _assess_compliance_risks(self, events: List[AuditEvent]) -> Dict[str, Any]:
        """Assess compliance risks from events"""
        risk_scores = {
            "operational_risk": 0.0,
            "regulatory_risk": 0.0,
            "reputational_risk": 0.0,
            "financial_risk": 0.0,
        }

        critical_events = [e for e in events if e.audit_level == AuditLevel.CRITICAL]
        high_events = [e for e in events if e.audit_level == AuditLevel.HIGH]

        # Calculate risk scores based on event severity and frequency
        if events:
            critical_ratio = len(critical_events) / len(events)
            high_ratio = len(high_events) / len(events)

            base_risk = critical_ratio * 0.8 + high_ratio * 0.5

            risk_scores["operational_risk"] = min(1.0, base_risk * 1.2)
            risk_scores["regulatory_risk"] = min(1.0, base_risk * 1.0)
            risk_scores["reputational_risk"] = min(1.0, base_risk * 0.8)
            risk_scores["financial_risk"] = min(1.0, base_risk * 0.6)

        return {
            "risk_scores": risk_scores,
            "overall_risk": sum(risk_scores.values()) / len(risk_scores),
            "risk_level": self._categorize_risk(
                sum(risk_scores.values()) / len(risk_scores)
            ),
            "key_risk_factors": self._identify_key_risk_factors(events),
        }

    def _evaluate_control_effectiveness(
        self, framework: ComplianceFramework, events: List[AuditEvent]
    ) -> Dict[str, float]:
        """Evaluate control effectiveness"""
        effectiveness = {}

        # Access control effectiveness
        access_events = [
            e for e in events if e.event_type == AuditEventType.DATA_ACCESS
        ]
        access_violations = [
            e
            for e in events
            if e.event_type == AuditEventType.RULE_VIOLATION
            and "access" in e.event_description.lower()
        ]

        if access_events:
            effectiveness["access_controls"] = 1.0 - (
                len(access_violations) / len(access_events)
            )
        else:
            effectiveness["access_controls"] = 1.0

        # Monitoring control effectiveness
        total_processed = [
            e for e in events if e.event_type == AuditEventType.DOCUMENT_PROCESSED
        ]
        anomaly_detection = [
            e for e in events if e.event_type == AuditEventType.ANOMALY_DETECTED
        ]

        if total_processed:
            effectiveness["monitoring_controls"] = min(
                1.0, len(anomaly_detection) / max(1, len(total_processed) * 0.1)
            )
        else:
            effectiveness["monitoring_controls"] = 0.5

        # Overall control effectiveness
        effectiveness["overall"] = sum(effectiveness.values()) / len(effectiveness)

        return effectiveness

    def _generate_remediation_actions(
        self, violations: List[AuditEvent], anomalies: List[AuditEvent]
    ) -> List[str]:
        """Generate remediation action recommendations"""
        actions = []

        if violations:
            actions.append("Review and strengthen access controls")
            actions.append(
                "Conduct user access audit and remove unnecessary permissions"
            )

            critical_violations = [
                v for v in violations if v.audit_level == AuditLevel.CRITICAL
            ]
            if critical_violations:
                actions.append("Immediately address critical compliance violations")
                actions.append("Implement enhanced monitoring for critical processes")

        if anomalies:
            actions.append(
                "Investigate identified anomalies for potential security threats"
            )
            actions.append("Tune anomaly detection thresholds based on analysis")

        if not violations and not anomalies:
            actions.append("Continue regular compliance monitoring")
            actions.append("Schedule periodic compliance assessment")

        # General recommendations
        actions.extend(
            [
                "Regular staff training on compliance procedures",
                "Implement automated compliance checking where possible",
                "Establish regular audit schedule",
                "Document all remediation activities",
            ]
        )

        return actions[:10]  # Limit to top 10

    def _collect_supporting_evidence(self, events: List[AuditEvent]) -> List[str]:
        """Collect supporting evidence references"""
        evidence_refs = []

        for event in events:
            if event.evidence:
                for key, value in event.evidence.items():
                    if isinstance(value, str) and (
                        value.startswith("file://") or value.startswith("http")
                    ):
                        evidence_refs.append(f"{event.event_id}:{key}={value}")

        return evidence_refs

    def _generate_audit_references(self, events: List[AuditEvent]) -> List[str]:
        """Generate audit trail references"""
        return [
            f"audit_chain:{event.session_id}:{event.event_id}" for event in events[:100]
        ]

    def _get_violation_recommendations(self, violation_type: str) -> List[str]:
        """Get recommendations for specific violation types"""
        recommendations_map = {
            "excessive_access": [
                "Implement role-based access controls",
                "Regular access reviews",
            ],
            "missing_evidence": [
                "Mandate evidence collection for critical events",
                "Automated evidence capture",
            ],
            "unauthorized_action": [
                "Strengthen authentication",
                "Implement additional authorization checks",
            ],
            "data_retention": [
                "Review data retention policies",
                "Implement automated data lifecycle management",
            ],
        }

        return recommendations_map.get(
            violation_type, ["Review and strengthen relevant controls"]
        )

    def _categorize_risk(self, risk_score: float) -> str:
        """Categorize overall risk level"""
        if risk_score >= 0.8:
            return "CRITICAL"
        elif risk_score >= 0.6:
            return "HIGH"
        elif risk_score >= 0.4:
            return "MEDIUM"
        elif risk_score >= 0.2:
            return "LOW"
        else:
            return "MINIMAL"

    def _identify_key_risk_factors(self, events: List[AuditEvent]) -> List[str]:
        """Identify key risk factors from events"""
        risk_factors = []

        # Check for patterns
        user_events = defaultdict(int)
        for event in events:
            user_events[event.user_id] += 1

        # High-activity users
        high_activity_users = [
            user for user, count in user_events.items() if count > 50
        ]
        if high_activity_users:
            risk_factors.append(
                f"High activity users: {len(high_activity_users)} users with excessive activity"
            )

        # Time-based patterns
        after_hours_events = [
            e for e in events if e.timestamp.hour < 6 or e.timestamp.hour > 22
        ]
        if len(after_hours_events) > len(events) * 0.1:
            risk_factors.append("Significant after-hours activity detected")

        # Critical event frequency
        critical_events = [e for e in events if e.audit_level == AuditLevel.CRITICAL]
        if len(critical_events) > 5:
            risk_factors.append(
                "Frequent critical events requiring immediate attention"
            )

        return risk_factors

    def verify_audit_integrity(self, session_id: str = None) -> Dict[str, Any]:
        """Verify integrity of audit trail"""
        try:
            integrity_report = {
                "overall_status": "VALID",
                "verified_chains": 0,
                "total_chains": 0,
                "verification_errors": [],
                "verified_events": 0,
                "total_events": 0,
            }

            # Check specific session or all sessions
            sessions_to_check = (
                [session_id] if session_id else list(self.audit_chains.keys())
            )

            for session in sessions_to_check:
                chains = self.audit_chains[session]
                integrity_report["total_chains"] += len(chains)

                for i, chain in enumerate(chains):
                    # Verify chain integrity
                    if chain.verify_chain_integrity():
                        integrity_report["verified_chains"] += 1
                        integrity_report["verified_events"] += len(chain.events)
                    else:
                        integrity_report["verification_errors"].append(
                            {
                                "session_id": session,
                                "chain_index": i,
                                "chain_id": chain.chain_id,
                                "error": "Chain integrity verification failed",
                            }
                        )

                    integrity_report["total_events"] += len(chain.events)

                    # Verify previous hash link
                    if i > 0 and chain.previous_hash != chains[i - 1].block_hash:
                        integrity_report["verification_errors"].append(
                            {
                                "session_id": session,
                                "chain_index": i,
                                "chain_id": chain.chain_id,
                                "error": "Previous hash mismatch",
                            }
                        )

            # Overall status
            if integrity_report["verification_errors"]:
                integrity_report["overall_status"] = "COMPROMISED"

            return integrity_report

        except Exception as e:
            logger.error(f"Audit integrity verification failed: {e}")
            return {"overall_status": "ERROR", "error": str(e)}

    def export_audit_data(
        self,
        start_date: datetime,
        end_date: datetime,
        format_type: str = "json",
        include_evidence: bool = False,
    ) -> str:
        """
        Export audit data for external analysis or compliance

        Returns:
            Path to exported file
        """
        try:
            timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"audit_export_{timestamp_str}.{format_type}"
            export_path = os.path.join(self.storage_path, filename)

            # Collect all events in date range
            all_events = []
            for session_chains in self.audit_chains.values():
                for chain in session_chains:
                    for event in chain.events:
                        if start_date <= event.timestamp <= end_date:
                            all_events.append(event)

            # Sort by timestamp
            all_events.sort(key=lambda x: x.timestamp)

            # Export data
            if format_type == "json":
                export_data = {
                    "export_info": {
                        "generated_at": datetime.now().isoformat(),
                        "date_range": {
                            "start": start_date.isoformat(),
                            "end": end_date.isoformat(),
                        },
                        "total_events": len(all_events),
                        "include_evidence": include_evidence,
                    },
                    "events": [],
                }

                for event in all_events:
                    event_data = {
                        "event_id": event.event_id,
                        "event_type": event.event_type.value,
                        "timestamp": event.timestamp.isoformat(),
                        "user_id": event.user_id,
                        "session_id": event.session_id,
                        "document_id": event.document_id,
                        "description": event.event_description,
                        "compliance_frameworks": [
                            f.value for f in event.compliance_frameworks
                        ],
                        "audit_level": event.audit_level.value,
                        "source_component": event.source_component,
                        "outcome": event.outcome,
                        "hash_value": event.hash_value,
                    }

                    if include_evidence:
                        event_data["evidence"] = event.evidence
                        event_data["event_data"] = event.event_data

                    export_data["events"].append(event_data)

                with open(export_path, "w", encoding="utf-8") as f:
                    json.dump(export_data, f, indent=2, ensure_ascii=False)

            logger.info(f"Audit data exported to: {export_path}")
            return export_path

        except Exception as e:
            logger.error(f"Audit data export failed: {e}")
            raise

    def get_audit_statistics(self) -> Dict[str, Any]:
        """Get audit system statistics"""
        try:
            stats = {
                "total_sessions": len(self.audit_chains),
                "total_chains": sum(
                    len(chains) for chains in self.audit_chains.values()
                ),
                "total_events": sum(
                    len(chain.events)
                    for chains in self.audit_chains.values()
                    for chain in chains
                ),
                "active_sessions": len(self.active_sessions),
                "compliance_framework_events": {},
                "event_type_distribution": defaultdict(int),
                "audit_level_distribution": defaultdict(int),
                "daily_event_counts": defaultdict(int),
            }

            # Compliance framework statistics
            for framework, events in self.compliance_events.items():
                stats["compliance_framework_events"][framework.value] = len(events)

            # Event analysis
            all_events = []
            for chains in self.audit_chains.values():
                for chain in chains:
                    all_events.extend(chain.events)

            for event in all_events:
                stats["event_type_distribution"][event.event_type.value] += 1
                stats["audit_level_distribution"][event.audit_level.value] += 1

                date_key = event.timestamp.strftime("%Y-%m-%d")
                stats["daily_event_counts"][date_key] += 1

            # Convert defaultdict to regular dict
            stats["event_type_distribution"] = dict(stats["event_type_distribution"])
            stats["audit_level_distribution"] = dict(stats["audit_level_distribution"])
            stats["daily_event_counts"] = dict(stats["daily_event_counts"])

            return stats

        except Exception as e:
            logger.error(f"Failed to get audit statistics: {e}")
            return {}

    def cleanup_expired_data(self) -> Dict[str, int]:
        """Clean up expired audit data based on retention policies"""
        try:
            cleanup_stats = {"removed_chains": 0, "removed_events": 0}
            current_time = datetime.now()

            # Clean up based on retention policies
            for framework, retention_days in self.retention_policies.items():
                cutoff_date = current_time - timedelta(days=retention_days)

                # Remove old events from compliance events
                original_count = len(self.compliance_events[framework])
                self.compliance_events[framework] = [
                    event
                    for event in self.compliance_events[framework]
                    if event.timestamp > cutoff_date
                ]
                removed_count = original_count - len(self.compliance_events[framework])
                cleanup_stats["removed_events"] += removed_count

            # Clean up old chain blocks (keeping recent data)
            general_retention = 365  # 1 year for general data
            cutoff_date = current_time - timedelta(days=general_retention)

            for session_id, chains in list(self.audit_chains.items()):
                original_chain_count = len(chains)
                self.audit_chains[session_id] = [
                    chain for chain in chains if chain.block_timestamp > cutoff_date
                ]
                removed_chains = original_chain_count - len(
                    self.audit_chains[session_id]
                )
                cleanup_stats["removed_chains"] += removed_chains

                # Remove empty session entries
                if not self.audit_chains[session_id]:
                    del self.audit_chains[session_id]

            logger.info(f"Audit cleanup completed: {cleanup_stats}")
            return cleanup_stats

        except Exception as e:
            logger.error(f"Audit cleanup failed: {e}")
            return {"error": str(e)}


# Convenience function for quick audit logging
def log_audit_event(
    event_type: AuditEventType, user_id: str, description: str, **kwargs
) -> str:
    """Convenience function for quick audit event logging"""
    # This would use a global audit manager instance
    # Implementation depends on application architecture
    pass
