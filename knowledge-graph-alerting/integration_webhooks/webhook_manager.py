"""
Real-Time Document Intelligence & Compliance Engine
Integration Webhooks System
Version: 1.0
Date: 2026-02-03

This module implements external system integration webhooks (Salesforce, DocuSign)
with event-driven notifications and retry mechanisms.
"""

import logging
import asyncio
import json
import uuid
import hmac
import hashlib
import time
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import threading
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
import urllib.parse

try:
    import aiohttp
    import requests

    HAS_HTTP = True
except ImportError:
    HAS_HTTP = False

    # Mock implementations for testing
    class MockResponse:
        def __init__(self, status=200, json_data=None):
            self.status = status
            self._json_data = json_data or {}

        async def json(self):
            return self._json_data

        def json(self):
            return self._json_data


logger = logging.getLogger(__name__)


class WebhookEvent(Enum):
    """Types of webhook events"""

    DOCUMENT_PROCESSED = "document_processed"
    ANALYSIS_COMPLETED = "analysis_completed"
    COMPLIANCE_VIOLATION = "compliance_violation"
    HIGH_RISK_DETECTED = "high_risk_detected"
    ALERT_GENERATED = "alert_generated"
    ENTITY_RESOLVED = "entity_resolved"
    RISK_PROPAGATED = "risk_propagated"
    CLUSTER_IDENTIFIED = "cluster_identified"
    SYSTEM_ERROR = "system_error"
    USER_ACTION = "user_action"


class IntegrationType(Enum):
    """Supported integration types"""

    SALESFORCE = "salesforce"
    DOCUSIGN = "docusign"
    SLACK = "slack"
    TEAMS = "teams"
    JIRA = "jira"
    SERVICENOW = "servicenow"
    GENERIC_WEBHOOK = "generic_webhook"
    REST_API = "rest_api"


class DeliveryStatus(Enum):
    """Webhook delivery status"""

    PENDING = "pending"
    SUCCESS = "success"
    FAILED = "failed"
    RETRYING = "retrying"
    EXPIRED = "expired"
    DISABLED = "disabled"


@dataclass
class WebhookEndpoint:
    """Webhook endpoint configuration"""

    endpoint_id: str
    name: str
    url: str
    integration_type: IntegrationType
    events: List[WebhookEvent]
    headers: Dict[str, str]
    secret_key: Optional[str]  # For signature verification
    authentication: Dict[str, Any]  # Auth configuration
    filters: Dict[str, Any]  # Event filters
    enabled: bool
    retry_config: Dict[str, Any]
    rate_limit: Dict[str, Any]
    timeout_seconds: int
    verify_ssl: bool
    custom_payload_template: Optional[str]
    metadata: Dict[str, Any]
    created_at: datetime
    updated_at: datetime


@dataclass
class WebhookDelivery:
    """Webhook delivery attempt"""

    delivery_id: str
    endpoint_id: str
    event_id: str
    event_type: WebhookEvent
    payload: Dict[str, Any]
    url: str
    headers: Dict[str, str]
    status: DeliveryStatus
    response_status: Optional[int]
    response_body: Optional[str]
    response_headers: Optional[Dict[str, str]]
    attempt_number: int
    scheduled_at: datetime
    delivered_at: Optional[datetime]
    duration_ms: Optional[int]
    error_message: Optional[str]
    signature: Optional[str]


@dataclass
class IntegrationConfig:
    """Integration-specific configuration"""

    integration_type: IntegrationType
    base_url: str
    api_version: str
    authentication: Dict[str, Any]
    default_headers: Dict[str, str]
    rate_limits: Dict[str, Any]
    endpoints: Dict[str, str]  # Operation -> endpoint mappings
    payload_templates: Dict[str, str]
    field_mappings: Dict[str, Dict[str, str]]


class WebhookIntegrationManager:
    """Comprehensive webhook integration management system"""

    def __init__(
        self,
        max_retries: int = 3,
        retry_delay_seconds: int = 60,
        max_concurrent_deliveries: int = 50,
        delivery_timeout: int = 30,
    ):
        """
        Initialize webhook integration manager

        Args:
            max_retries: Maximum retry attempts
            retry_delay_seconds: Delay between retries
            max_concurrent_deliveries: Maximum concurrent deliveries
            delivery_timeout: Delivery timeout in seconds
        """
        self.max_retries = max_retries
        self.retry_delay_seconds = retry_delay_seconds
        self.max_concurrent_deliveries = max_concurrent_deliveries
        self.delivery_timeout = delivery_timeout

        # Thread safety
        self._lock = threading.RLock()
        self._executor = ThreadPoolExecutor(max_workers=max_concurrent_deliveries)

        # Storage
        self.endpoints: Dict[str, WebhookEndpoint] = {}
        self.integration_configs: Dict[IntegrationType, IntegrationConfig] = {}
        self.delivery_queue: deque = deque()
        self.delivery_history: Dict[str, List[WebhookDelivery]] = defaultdict(list)
        self.retry_queue: deque = deque()

        # Rate limiting
        self.rate_limiters: Dict[str, Dict[str, Any]] = defaultdict(
            lambda: {
                "requests": deque(),
                "limit": 100,
                "window": 3600,  # 1 hour
            }
        )

        # Event handlers
        self.event_handlers: Dict[WebhookEvent, List[Callable]] = defaultdict(list)

        # Background tasks
        self._delivery_worker_running = False
        self._retry_worker_running = False

        # Statistics
        self.stats = {
            "total_deliveries": 0,
            "successful_deliveries": 0,
            "failed_deliveries": 0,
            "retries_attempted": 0,
            "rate_limited": 0,
            "endpoints_active": 0,
        }

        # Initialize default integration configs
        self._initialize_default_configs()

        # Start background workers
        self._start_workers()

    def _initialize_default_configs(self):
        """Initialize default integration configurations"""
        try:
            # Salesforce configuration
            salesforce_config = IntegrationConfig(
                integration_type=IntegrationType.SALESFORCE,
                base_url="https://{instance}.salesforce.com",
                api_version="v57.0",
                authentication={
                    "type": "oauth2",
                    "token_url": "/services/oauth2/token",
                    "scope": "api",
                },
                default_headers={
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                },
                rate_limits={"requests_per_hour": 1000, "burst_limit": 50},
                endpoints={
                    "create_case": "/services/data/v57.0/sobjects/Case/",
                    "create_lead": "/services/data/v57.0/sobjects/Lead/",
                    "create_task": "/services/data/v57.0/sobjects/Task/",
                },
                payload_templates={
                    "compliance_case": '{"Subject": "{title}", "Description": "{description}", "Priority": "{severity}", "Type": "Compliance", "Status": "New"}',
                    "risk_case": '{"Subject": "High Risk: {entity_name}", "Description": "{description}", "Priority": "High", "Type": "Risk Management", "Status": "New"}',
                },
                field_mappings={
                    "case": {
                        "alert_id": "External_Alert_ID__c",
                        "risk_score": "Risk_Score__c",
                        "entity_name": "Related_Entity__c",
                    }
                },
            )
            self.integration_configs[IntegrationType.SALESFORCE] = salesforce_config

            # DocuSign configuration
            docusign_config = IntegrationConfig(
                integration_type=IntegrationType.DOCUSIGN,
                base_url="https://{environment}.docusign.net",
                api_version="v2.1",
                authentication={
                    "type": "jwt",
                    "token_url": "/oauth/token",
                    "scope": "signature impersonation",
                },
                default_headers={
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                },
                rate_limits={"requests_per_hour": 500, "burst_limit": 25},
                endpoints={
                    "create_envelope": "/restapi/v2.1/accounts/{account_id}/envelopes",
                    "get_envelope": "/restapi/v2.1/accounts/{account_id}/envelopes/{envelope_id}",
                },
                payload_templates={
                    "compliance_envelope": '{"emailSubject": "Compliance Document: {title}", "status": "created", "documents": [], "recipients": {}}',
                    "signature_request": '{"emailSubject": "Signature Required: {document_name}", "status": "sent"}',
                },
                field_mappings={
                    "envelope": {
                        "document_id": "customFields.textTabs[0].value",
                        "compliance_type": "customFields.textTabs[1].value",
                    }
                },
            )
            self.integration_configs[IntegrationType.DOCUSIGN] = docusign_config

            # Slack configuration
            slack_config = IntegrationConfig(
                integration_type=IntegrationType.SLACK,
                base_url="https://slack.com/api",
                api_version="v1",
                authentication={"type": "bearer", "header": "Authorization"},
                default_headers={"Content-Type": "application/json"},
                rate_limits={"requests_per_minute": 100, "burst_limit": 50},
                endpoints={
                    "post_message": "/chat.postMessage",
                    "create_channel": "/conversations.create",
                },
                payload_templates={
                    "alert_message": '{"channel": "{channel}", "text": "{message}", "attachments": [{"color": "{color}", "fields": []}]}',
                    "compliance_notification": '{"channel": "#compliance", "text": "🚨 Compliance Violation Detected", "blocks": []}',
                },
                field_mappings={
                    "message": {
                        "alert_severity": "color",
                        "entity_name": "blocks[0].fields[0].text",
                    }
                },
            )
            self.integration_configs[IntegrationType.SLACK] = slack_config

            logger.info("Default integration configurations initialized")

        except Exception as e:
            logger.error(f"Failed to initialize integration configs: {e}")

    def register_webhook_endpoint(self, endpoint: WebhookEndpoint) -> bool:
        """Register a new webhook endpoint"""
        try:
            with self._lock:
                self.endpoints[endpoint.endpoint_id] = endpoint
                self.stats["endpoints_active"] = len(
                    [e for e in self.endpoints.values() if e.enabled]
                )

            logger.info(
                f"Registered webhook endpoint: {endpoint.name} ({endpoint.integration_type.value})"
            )
            return True

        except Exception as e:
            logger.error(f"Failed to register webhook endpoint: {e}")
            return False

    def register_event_handler(self, event_type: WebhookEvent, handler: Callable):
        """Register an event handler"""
        try:
            self.event_handlers[event_type].append(handler)
            logger.info(f"Registered event handler for {event_type.value}")

        except Exception as e:
            logger.error(f"Failed to register event handler: {e}")

    def trigger_webhook_event(
        self,
        event_type: WebhookEvent,
        payload: Dict[str, Any],
        target_endpoints: List[str] = None,
    ) -> str:
        """
        Trigger a webhook event

        Args:
            event_type: Type of event to trigger
            payload: Event payload data
            target_endpoints: Specific endpoints to target (None for all matching)

        Returns:
            Event ID
        """
        try:
            event_id = str(uuid.uuid4())

            # Get matching endpoints
            matching_endpoints = self._get_matching_endpoints(
                event_type, target_endpoints
            )

            if not matching_endpoints:
                logger.info(f"No endpoints found for event {event_type.value}")
                return event_id

            # Create deliveries for each endpoint
            for endpoint in matching_endpoints:
                if self._should_deliver_to_endpoint(endpoint, payload):
                    delivery = self._create_webhook_delivery(
                        endpoint, event_id, event_type, payload
                    )

                    with self._lock:
                        self.delivery_queue.append(delivery)

            # Call event handlers
            for handler in self.event_handlers[event_type]:
                try:
                    self._executor.submit(handler, event_type, payload)
                except Exception as e:
                    logger.error(f"Event handler failed: {e}")

            logger.info(
                f"Triggered webhook event: {event_id} ({event_type.value}) to {len(matching_endpoints)} endpoints"
            )
            return event_id

        except Exception as e:
            logger.error(f"Webhook event trigger failed: {e}")
            return ""

    def _get_matching_endpoints(
        self, event_type: WebhookEvent, target_endpoints: List[str] = None
    ) -> List[WebhookEndpoint]:
        """Get endpoints that should receive this event"""
        try:
            matching_endpoints = []

            for endpoint_id, endpoint in self.endpoints.items():
                # Check if endpoint is enabled
                if not endpoint.enabled:
                    continue

                # Check target filter
                if target_endpoints and endpoint_id not in target_endpoints:
                    continue

                # Check event subscription
                if event_type in endpoint.events:
                    matching_endpoints.append(endpoint)

            return matching_endpoints

        except Exception as e:
            logger.error(f"Failed to get matching endpoints: {e}")
            return []

    def _should_deliver_to_endpoint(
        self, endpoint: WebhookEndpoint, payload: Dict[str, Any]
    ) -> bool:
        """Check if payload should be delivered to endpoint based on filters"""
        try:
            # Check endpoint filters
            for filter_key, filter_value in endpoint.filters.items():
                payload_value = payload.get(filter_key)

                if isinstance(filter_value, list):
                    if payload_value not in filter_value:
                        return False
                elif isinstance(filter_value, str):
                    if payload_value != filter_value:
                        return False
                elif isinstance(filter_value, dict):
                    if "min" in filter_value and payload_value < filter_value["min"]:
                        return False
                    if "max" in filter_value and payload_value > filter_value["max"]:
                        return False
                    if "contains" in filter_value and isinstance(payload_value, str):
                        if filter_value["contains"] not in payload_value:
                            return False

            return True

        except Exception as e:
            logger.error(f"Filter evaluation failed: {e}")
            return True  # Default to allowing delivery

    def _create_webhook_delivery(
        self,
        endpoint: WebhookEndpoint,
        event_id: str,
        event_type: WebhookEvent,
        payload: Dict[str, Any],
    ) -> WebhookDelivery:
        """Create a webhook delivery"""
        try:
            delivery_id = str(uuid.uuid4())

            # Transform payload if template is provided
            transformed_payload = self._transform_payload(endpoint, payload)

            # Prepare headers
            headers = endpoint.headers.copy()

            # Add authentication headers
            auth_headers = self._get_auth_headers(endpoint)
            headers.update(auth_headers)

            # Generate signature if secret key provided
            signature = None
            if endpoint.secret_key:
                signature = self._generate_signature(
                    endpoint.secret_key, transformed_payload
                )
                headers["X-Webhook-Signature"] = signature

            delivery = WebhookDelivery(
                delivery_id=delivery_id,
                endpoint_id=endpoint.endpoint_id,
                event_id=event_id,
                event_type=event_type,
                payload=transformed_payload,
                url=endpoint.url,
                headers=headers,
                status=DeliveryStatus.PENDING,
                response_status=None,
                response_body=None,
                response_headers=None,
                attempt_number=1,
                scheduled_at=datetime.now(),
                delivered_at=None,
                duration_ms=None,
                error_message=None,
                signature=signature,
            )

            return delivery

        except Exception as e:
            logger.error(f"Failed to create webhook delivery: {e}")
            raise

    def _transform_payload(
        self, endpoint: WebhookEndpoint, payload: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Transform payload using endpoint template"""
        try:
            if not endpoint.custom_payload_template:
                return payload

            # Simple template substitution
            template = endpoint.custom_payload_template

            # Replace placeholders with payload values
            for key, value in payload.items():
                placeholder = "{" + key + "}"
                if placeholder in template:
                    template = template.replace(placeholder, str(value))

            # Parse as JSON if it's a JSON template
            try:
                return json.loads(template)
            except json.JSONDecodeError:
                # Return original payload if template parsing fails
                logger.warning(
                    f"Invalid payload template for endpoint {endpoint.endpoint_id}"
                )
                return payload

        except Exception as e:
            logger.error(f"Payload transformation failed: {e}")
            return payload

    def _get_auth_headers(self, endpoint: WebhookEndpoint) -> Dict[str, str]:
        """Get authentication headers for endpoint"""
        try:
            auth_headers = {}
            auth_config = endpoint.authentication

            if not auth_config:
                return auth_headers

            auth_type = auth_config.get("type", "").lower()

            if auth_type == "bearer":
                token = auth_config.get("token")
                if token:
                    auth_headers["Authorization"] = f"Bearer {token}"

            elif auth_type == "api_key":
                api_key = auth_config.get("api_key")
                header_name = auth_config.get("header_name", "X-API-Key")
                if api_key:
                    auth_headers[header_name] = api_key

            elif auth_type == "basic":
                username = auth_config.get("username")
                password = auth_config.get("password")
                if username and password:
                    import base64

                    credentials = base64.b64encode(
                        f"{username}:{password}".encode()
                    ).decode()
                    auth_headers["Authorization"] = f"Basic {credentials}"

            return auth_headers

        except Exception as e:
            logger.error(f"Auth header generation failed: {e}")
            return {}

    def _generate_signature(self, secret_key: str, payload: Dict[str, Any]) -> str:
        """Generate webhook signature"""
        try:
            payload_bytes = json.dumps(payload, sort_keys=True).encode("utf-8")
            signature = hmac.new(
                secret_key.encode("utf-8"), payload_bytes, hashlib.sha256
            ).hexdigest()
            return f"sha256={signature}"

        except Exception as e:
            logger.error(f"Signature generation failed: {e}")
            return ""

    def _start_workers(self):
        """Start background worker threads"""
        try:
            if not self._delivery_worker_running:
                self._delivery_worker_running = True
                self._executor.submit(self._delivery_worker)

            if not self._retry_worker_running:
                self._retry_worker_running = True
                self._executor.submit(self._retry_worker)

        except Exception as e:
            logger.error(f"Failed to start workers: {e}")

    def _delivery_worker(self):
        """Background worker for processing webhook deliveries"""
        try:
            while self._delivery_worker_running:
                try:
                    # Check for pending deliveries
                    if not self.delivery_queue:
                        time.sleep(1)
                        continue

                    # Get next delivery
                    with self._lock:
                        if self.delivery_queue:
                            delivery = self.delivery_queue.popleft()
                        else:
                            continue

                    # Check rate limiting
                    if self._is_rate_limited(delivery.endpoint_id):
                        # Re-queue delivery
                        with self._lock:
                            self.delivery_queue.appendleft(delivery)
                            self.stats["rate_limited"] += 1
                        time.sleep(5)
                        continue

                    # Execute delivery
                    self._execute_delivery(delivery)

                except Exception as e:
                    logger.error(f"Delivery worker error: {e}")
                    time.sleep(5)

        except Exception as e:
            logger.error(f"Delivery worker failed: {e}")
        finally:
            self._delivery_worker_running = False

    def _retry_worker(self):
        """Background worker for processing retries"""
        try:
            while self._retry_worker_running:
                try:
                    # Check for pending retries
                    if not self.retry_queue:
                        time.sleep(10)
                        continue

                    current_time = datetime.now()
                    ready_retries = []
                    remaining_retries = []

                    # Check which retries are ready
                    with self._lock:
                        while self.retry_queue:
                            retry_delivery = self.retry_queue.popleft()

                            if current_time >= retry_delivery.scheduled_at:
                                ready_retries.append(retry_delivery)
                            else:
                                remaining_retries.append(retry_delivery)

                        # Put back retries that aren't ready
                        self.retry_queue.extend(remaining_retries)

                    # Execute ready retries
                    for delivery in ready_retries:
                        if delivery.attempt_number <= self.max_retries:
                            self._execute_delivery(delivery)
                        else:
                            # Mark as expired
                            delivery.status = DeliveryStatus.EXPIRED
                            self._store_delivery_result(delivery)

                except Exception as e:
                    logger.error(f"Retry worker error: {e}")
                    time.sleep(10)

        except Exception as e:
            logger.error(f"Retry worker failed: {e}")
        finally:
            self._retry_worker_running = False

    def _is_rate_limited(self, endpoint_id: str) -> bool:
        """Check if endpoint is rate limited"""
        try:
            endpoint = self.endpoints.get(endpoint_id)
            if not endpoint or not endpoint.rate_limit:
                return False

            limiter = self.rate_limiters[endpoint_id]
            current_time = time.time()
            window_size = endpoint.rate_limit.get("window_seconds", limiter["window"])
            max_requests = endpoint.rate_limit.get("max_requests", limiter["limit"])

            # Remove old requests from sliding window
            while (
                limiter["requests"]
                and limiter["requests"][0] < current_time - window_size
            ):
                limiter["requests"].popleft()

            # Check if under limit
            if len(limiter["requests"]) >= max_requests:
                return True

            # Add current request
            limiter["requests"].append(current_time)
            return False

        except Exception as e:
            logger.error(f"Rate limit check failed: {e}")
            return False

    def _execute_delivery(self, delivery: WebhookDelivery):
        """Execute webhook delivery"""
        try:
            start_time = time.time()
            delivery.status = (
                DeliveryStatus.RETRYING
                if delivery.attempt_number > 1
                else DeliveryStatus.PENDING
            )

            # Get endpoint configuration
            endpoint = self.endpoints.get(delivery.endpoint_id)
            if not endpoint or not endpoint.enabled:
                delivery.status = DeliveryStatus.DISABLED
                self._store_delivery_result(delivery)
                return

            if HAS_HTTP:
                # Make HTTP request
                response = self._make_http_request(delivery)

                # Process response
                end_time = time.time()
                delivery.duration_ms = int((end_time - start_time) * 1000)
                delivery.delivered_at = datetime.now()

                if 200 <= response.status_code < 300:
                    delivery.status = DeliveryStatus.SUCCESS
                    delivery.response_status = response.status_code
                    delivery.response_body = response.text[
                        :1000
                    ]  # Limit response body size
                    delivery.response_headers = dict(response.headers)

                    with self._lock:
                        self.stats["successful_deliveries"] += 1
                        self.stats["total_deliveries"] += 1

                else:
                    delivery.status = DeliveryStatus.FAILED
                    delivery.response_status = response.status_code
                    delivery.response_body = response.text[:1000]
                    delivery.error_message = f"HTTP {response.status_code}"

                    # Schedule retry if attempts remaining
                    if delivery.attempt_number < self.max_retries:
                        self._schedule_retry(delivery)
                    else:
                        with self._lock:
                            self.stats["failed_deliveries"] += 1
                            self.stats["total_deliveries"] += 1

            else:
                # Mock successful delivery
                delivery.status = DeliveryStatus.SUCCESS
                delivery.response_status = 200
                delivery.delivered_at = datetime.now()
                delivery.duration_ms = 100

                with self._lock:
                    self.stats["successful_deliveries"] += 1
                    self.stats["total_deliveries"] += 1

            # Store result
            self._store_delivery_result(delivery)

        except Exception as e:
            logger.error(f"Delivery execution failed: {e}")

            delivery.status = DeliveryStatus.FAILED
            delivery.error_message = str(e)
            delivery.delivered_at = datetime.now()

            # Schedule retry if attempts remaining
            if delivery.attempt_number < self.max_retries:
                self._schedule_retry(delivery)
            else:
                with self._lock:
                    self.stats["failed_deliveries"] += 1
                    self.stats["total_deliveries"] += 1

            self._store_delivery_result(delivery)

    def _make_http_request(self, delivery: WebhookDelivery):
        """Make HTTP request for webhook delivery"""
        try:
            endpoint = self.endpoints[delivery.endpoint_id]
            timeout = endpoint.timeout_seconds
            verify_ssl = endpoint.verify_ssl

            response = requests.post(
                delivery.url,
                json=delivery.payload,
                headers=delivery.headers,
                timeout=timeout,
                verify=verify_ssl,
            )

            return response

        except Exception as e:
            logger.error(f"HTTP request failed: {e}")
            raise

    def _schedule_retry(self, delivery: WebhookDelivery):
        """Schedule delivery retry"""
        try:
            # Calculate retry delay (exponential backoff)
            delay_seconds = self.retry_delay_seconds * (
                2 ** (delivery.attempt_number - 1)
            )
            retry_time = datetime.now() + timedelta(seconds=delay_seconds)

            # Create retry delivery
            retry_delivery = WebhookDelivery(
                delivery_id=str(uuid.uuid4()),
                endpoint_id=delivery.endpoint_id,
                event_id=delivery.event_id,
                event_type=delivery.event_type,
                payload=delivery.payload,
                url=delivery.url,
                headers=delivery.headers,
                status=DeliveryStatus.PENDING,
                response_status=None,
                response_body=None,
                response_headers=None,
                attempt_number=delivery.attempt_number + 1,
                scheduled_at=retry_time,
                delivered_at=None,
                duration_ms=None,
                error_message=None,
                signature=delivery.signature,
            )

            with self._lock:
                self.retry_queue.append(retry_delivery)
                self.stats["retries_attempted"] += 1

            logger.info(
                f"Scheduled retry {delivery.attempt_number + 1} for delivery {delivery.delivery_id}"
            )

        except Exception as e:
            logger.error(f"Retry scheduling failed: {e}")

    def _store_delivery_result(self, delivery: WebhookDelivery):
        """Store delivery result in history"""
        try:
            with self._lock:
                self.delivery_history[delivery.event_id].append(delivery)

                # Limit history size per event
                if len(self.delivery_history[delivery.event_id]) > 100:
                    self.delivery_history[delivery.event_id] = self.delivery_history[
                        delivery.event_id
                    ][-50:]  # Keep last 50

        except Exception as e:
            logger.error(f"Delivery result storage failed: {e}")

    def get_delivery_status(self, event_id: str) -> List[Dict[str, Any]]:
        """Get delivery status for an event"""
        try:
            deliveries = self.delivery_history.get(event_id, [])

            status_list = []
            for delivery in deliveries:
                status_list.append(
                    {
                        "delivery_id": delivery.delivery_id,
                        "endpoint_id": delivery.endpoint_id,
                        "status": delivery.status.value,
                        "attempt_number": delivery.attempt_number,
                        "response_status": delivery.response_status,
                        "duration_ms": delivery.duration_ms,
                        "error_message": delivery.error_message,
                        "delivered_at": delivery.delivered_at.isoformat()
                        if delivery.delivered_at
                        else None,
                    }
                )

            return status_list

        except Exception as e:
            logger.error(f"Failed to get delivery status: {e}")
            return []

    def get_endpoint_statistics(self, endpoint_id: str = None) -> Dict[str, Any]:
        """Get delivery statistics for endpoint(s)"""
        try:
            if endpoint_id:
                endpoints_to_analyze = (
                    [endpoint_id] if endpoint_id in self.endpoints else []
                )
            else:
                endpoints_to_analyze = list(self.endpoints.keys())

            endpoint_stats = {}

            for ep_id in endpoints_to_analyze:
                # Count deliveries by status
                total_deliveries = 0
                successful_deliveries = 0
                failed_deliveries = 0

                for deliveries in self.delivery_history.values():
                    for delivery in deliveries:
                        if delivery.endpoint_id == ep_id:
                            total_deliveries += 1
                            if delivery.status == DeliveryStatus.SUCCESS:
                                successful_deliveries += 1
                            elif delivery.status == DeliveryStatus.FAILED:
                                failed_deliveries += 1

                # Calculate success rate
                success_rate = (
                    (successful_deliveries / total_deliveries * 100)
                    if total_deliveries > 0
                    else 0
                )

                endpoint_stats[ep_id] = {
                    "total_deliveries": total_deliveries,
                    "successful_deliveries": successful_deliveries,
                    "failed_deliveries": failed_deliveries,
                    "success_rate": round(success_rate, 2),
                    "endpoint_enabled": self.endpoints[ep_id].enabled
                    if ep_id in self.endpoints
                    else False,
                }

            return endpoint_stats

        except Exception as e:
            logger.error(f"Statistics calculation failed: {e}")
            return {}

    def get_integration_statistics(self) -> Dict[str, Any]:
        """Get comprehensive integration statistics"""
        try:
            stats = self.stats.copy()

            # Integration type breakdown
            integration_counts = defaultdict(int)
            for endpoint in self.endpoints.values():
                if endpoint.enabled:
                    integration_counts[endpoint.integration_type.value] += 1

            # Event type breakdown
            event_counts = defaultdict(int)
            for endpoint in self.endpoints.values():
                for event in endpoint.events:
                    event_counts[event.value] += 1

            # Queue statistics
            queue_sizes = {
                "pending_deliveries": len(self.delivery_queue),
                "pending_retries": len(self.retry_queue),
            }

            # Worker status
            worker_status = {
                "delivery_worker_running": self._delivery_worker_running,
                "retry_worker_running": self._retry_worker_running,
            }

            stats.update(
                {
                    "integration_breakdown": dict(integration_counts),
                    "event_breakdown": dict(event_counts),
                    "queue_sizes": queue_sizes,
                    "worker_status": worker_status,
                    "total_endpoints": len(self.endpoints),
                    "total_events_tracked": len(self.delivery_history),
                }
            )

            return stats

        except Exception as e:
            logger.error(f"Statistics calculation failed: {e}")
            return self.stats.copy()

    def cleanup_old_deliveries(self, retention_days: int = 30):
        """Clean up old delivery history"""
        try:
            cutoff_date = datetime.now() - timedelta(days=retention_days)

            with self._lock:
                events_to_remove = []

                for event_id, deliveries in self.delivery_history.items():
                    # Filter deliveries to keep only recent ones
                    recent_deliveries = [
                        d
                        for d in deliveries
                        if d.delivered_at and d.delivered_at > cutoff_date
                    ]

                    if recent_deliveries:
                        self.delivery_history[event_id] = recent_deliveries
                    else:
                        events_to_remove.append(event_id)

                # Remove empty event histories
                for event_id in events_to_remove:
                    del self.delivery_history[event_id]

            logger.info(
                f"Cleaned up delivery history: removed {len(events_to_remove)} old events"
            )

        except Exception as e:
            logger.error(f"Delivery cleanup failed: {e}")

    def close(self):
        """Close the webhook integration manager and cleanup resources"""
        try:
            # Stop workers
            self._delivery_worker_running = False
            self._retry_worker_running = False

            if hasattr(self, "_executor") and self._executor:
                self._executor.shutdown(wait=True)

            logger.info("Webhook integration manager closed")

        except Exception as e:
            logger.error(f"Error closing webhook integration manager: {e}")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
