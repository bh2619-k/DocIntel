"""
Real-Time Document Intelligence & Compliance Engine
WebSocket Result Push System
Version: 1.0
Date: 2026-02-03

This module implements real-time notification system for analysis results, alerts,
and graph updates using WebSocket connections.
"""

import logging
import asyncio
import json
import uuid
from typing import Dict, Any, List, Optional, Set, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import threading
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
import weakref

try:
    import websockets
    from websockets.server import WebSocketServerProtocol

    HAS_WEBSOCKETS = True
except ImportError:
    HAS_WEBSOCKETS = False

    # Mock implementation for testing
    class WebSocketServerProtocol:
        def __init__(self):
            self.closed = False

        async def send(self, message):
            pass

        async def close(self):
            self.closed = True


logger = logging.getLogger(__name__)


class MessageType(Enum):
    """WebSocket message types"""

    ALERT = "alert"
    ANALYSIS_RESULT = "analysis_result"
    GRAPH_UPDATE = "graph_update"
    RISK_UPDATE = "risk_update"
    ENTITY_RESOLVED = "entity_resolved"
    SYSTEM_STATUS = "system_status"
    USER_NOTIFICATION = "user_notification"
    HEARTBEAT = "heartbeat"
    SUBSCRIPTION_ACK = "subscription_ack"
    ERROR = "error"


class SubscriptionType(Enum):
    """Subscription types for real-time updates"""

    ALL_ALERTS = "all_alerts"
    USER_ALERTS = "user_alerts"
    ANALYSIS_RESULTS = "analysis_results"
    HIGH_RISK_ENTITIES = "high_risk_entities"
    GRAPH_UPDATES = "graph_updates"
    SYSTEM_EVENTS = "system_events"
    COMPLIANCE_VIOLATIONS = "compliance_violations"
    DOCUMENT_PROCESSING = "document_processing"


@dataclass
class WebSocketConnection:
    """WebSocket connection information"""

    connection_id: str
    websocket: Any  # WebSocket connection object
    user_id: str
    session_id: str
    subscriptions: Set[SubscriptionType]
    filters: Dict[str, Any]
    connected_at: datetime
    last_heartbeat: datetime
    ip_address: str
    user_agent: str
    active: bool


@dataclass
class RealtimeMessage:
    """Real-time message structure"""

    message_id: str
    message_type: MessageType
    subscription_type: SubscriptionType
    payload: Dict[str, Any]
    target_users: List[str]  # Empty list means broadcast
    filters: Dict[str, Any]
    priority: int  # Higher number = higher priority
    created_at: datetime
    expires_at: Optional[datetime] = None


@dataclass
class MessageDelivery:
    """Message delivery tracking"""

    delivery_id: str
    message_id: str
    connection_id: str
    status: str  # 'pending', 'sent', 'failed', 'expired'
    attempts: int
    last_attempt: datetime
    error_message: Optional[str]


class WebSocketPushServer:
    """Real-time WebSocket notification server"""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 8765,
        max_connections: int = 1000,
        heartbeat_interval: int = 30,
        message_retention_hours: int = 24,
    ):
        """
        Initialize WebSocket push server

        Args:
            host: Server host address
            port: Server port
            max_connections: Maximum concurrent connections
            heartbeat_interval: Heartbeat interval in seconds
            message_retention_hours: Message retention time
        """
        self.host = host
        self.port = port
        self.max_connections = max_connections
        self.heartbeat_interval = heartbeat_interval
        self.message_retention_hours = message_retention_hours

        # Server state
        self.server = None
        self.running = False
        self.connections: Dict[str, WebSocketConnection] = {}
        self.user_connections: Dict[str, Set[str]] = defaultdict(set)

        # Message handling
        self.message_queue: deque = deque()
        self.message_history: Dict[str, RealtimeMessage] = {}
        self.delivery_tracking: Dict[str, List[MessageDelivery]] = defaultdict(list)

        # Subscription management
        self.subscription_handlers: Dict[SubscriptionType, Callable] = {}
        self.subscription_filters: Dict[SubscriptionType, Callable] = {}

        # Threading
        self._lock = threading.RLock()
        self._executor = ThreadPoolExecutor(max_workers=5)

        # Event loop for async operations
        self.loop = None
        self.loop_thread = None

        # Statistics
        self.stats = {
            "total_connections": 0,
            "active_connections": 0,
            "messages_sent": 0,
            "messages_failed": 0,
            "subscriptions_active": 0,
            "heartbeats_sent": 0,
            "errors": 0,
        }

        # Initialize default handlers
        self._initialize_default_handlers()

    def _initialize_default_handlers(self):
        """Initialize default subscription handlers"""
        try:
            # Alert handler
            self.register_subscription_handler(
                SubscriptionType.ALL_ALERTS, self._handle_alert_subscription
            )

            # Analysis result handler
            self.register_subscription_handler(
                SubscriptionType.ANALYSIS_RESULTS, self._handle_analysis_subscription
            )

            # High-risk entity handler
            self.register_subscription_handler(
                SubscriptionType.HIGH_RISK_ENTITIES, self._handle_risk_subscription
            )

            # System events handler
            self.register_subscription_handler(
                SubscriptionType.SYSTEM_EVENTS, self._handle_system_subscription
            )

            logger.info("Default subscription handlers initialized")

        except Exception as e:
            logger.error(f"Failed to initialize default handlers: {e}")

    def register_subscription_handler(
        self, subscription_type: SubscriptionType, handler: Callable
    ):
        """Register a handler for subscription type"""
        try:
            self.subscription_handlers[subscription_type] = handler
            logger.info(f"Registered handler for {subscription_type.value}")

        except Exception as e:
            logger.error(f"Failed to register subscription handler: {e}")

    async def start_server(self):
        """Start the WebSocket server"""
        try:
            if not HAS_WEBSOCKETS:
                logger.warning(
                    "WebSockets library not available, using mock implementation"
                )
                return

            # Start WebSocket server
            self.server = await websockets.serve(
                self._handle_connection,
                self.host,
                self.port,
                max_size=1024 * 1024,  # 1MB max message size
                ping_interval=self.heartbeat_interval,
                ping_timeout=self.heartbeat_interval * 2,
            )

            self.running = True
            logger.info(f"WebSocket server started on {self.host}:{self.port}")

            # Start background tasks
            asyncio.create_task(self._message_processor())
            asyncio.create_task(self._heartbeat_monitor())
            asyncio.create_task(self._cleanup_task())

        except Exception as e:
            logger.error(f"Failed to start WebSocket server: {e}")
            raise

    def start_in_thread(self):
        """Start server in a separate thread"""
        try:

            def run_server():
                self.loop = asyncio.new_event_loop()
                asyncio.set_event_loop(self.loop)
                self.loop.run_until_complete(self.start_server())
                self.loop.run_forever()

            self.loop_thread = threading.Thread(target=run_server, daemon=True)
            self.loop_thread.start()

            logger.info("WebSocket server started in background thread")

        except Exception as e:
            logger.error(f"Failed to start server in thread: {e}")

    async def _handle_connection(self, websocket, path):
        """Handle new WebSocket connection"""
        try:
            connection_id = str(uuid.uuid4())

            # Get connection info
            remote_address = websocket.remote_address
            ip_address = remote_address[0] if remote_address else "unknown"

            # Wait for authentication message
            auth_message = await asyncio.wait_for(websocket.recv(), timeout=30)
            auth_data = json.loads(auth_message)

            if not self._authenticate_connection(auth_data):
                await websocket.send(
                    json.dumps(
                        {
                            "type": "error",
                            "message": "Authentication failed",
                            "timestamp": datetime.now().isoformat(),
                        }
                    )
                )
                return

            # Create connection object
            connection = WebSocketConnection(
                connection_id=connection_id,
                websocket=websocket,
                user_id=auth_data.get("user_id", "anonymous"),
                session_id=auth_data.get("session_id", str(uuid.uuid4())),
                subscriptions=set(),
                filters=auth_data.get("filters", {}),
                connected_at=datetime.now(),
                last_heartbeat=datetime.now(),
                ip_address=ip_address,
                user_agent=auth_data.get("user_agent", "unknown"),
                active=True,
            )

            # Register connection
            with self._lock:
                self.connections[connection_id] = connection
                self.user_connections[connection.user_id].add(connection_id)
                self.stats["total_connections"] += 1
                self.stats["active_connections"] += 1

            # Send connection acknowledgment
            await self._send_to_connection(
                connection_id,
                {
                    "type": "connection_ack",
                    "connection_id": connection_id,
                    "timestamp": datetime.now().isoformat(),
                },
            )

            logger.info(
                f"WebSocket connection established: {connection_id} (user: {connection.user_id})"
            )

            # Handle messages from client
            await self._handle_client_messages(connection)

        except asyncio.TimeoutError:
            logger.warning("Connection authentication timeout")
        except Exception as e:
            logger.error(f"WebSocket connection handling failed: {e}")
        finally:
            # Clean up connection
            await self._cleanup_connection(connection_id)

    def _authenticate_connection(self, auth_data: Dict[str, Any]) -> bool:
        """Authenticate WebSocket connection"""
        try:
            # Simple authentication - in production, validate JWT token
            user_id = auth_data.get("user_id")
            session_token = auth_data.get("session_token")

            if not user_id or not session_token:
                return False

            # Validate session token (mock implementation)
            # In production: validate JWT, check user permissions, etc.
            return len(session_token) > 10

        except Exception as e:
            logger.error(f"Connection authentication failed: {e}")
            return False

    async def _handle_client_messages(self, connection: WebSocketConnection):
        """Handle messages from WebSocket client"""
        try:
            async for message in connection.websocket:
                try:
                    data = json.loads(message)
                    message_type = data.get("type")

                    if message_type == "subscribe":
                        await self._handle_subscription_request(connection, data)
                    elif message_type == "unsubscribe":
                        await self._handle_unsubscription_request(connection, data)
                    elif message_type == "heartbeat":
                        await self._handle_heartbeat(connection)
                    elif message_type == "get_history":
                        await self._handle_history_request(connection, data)
                    else:
                        logger.warning(f"Unknown message type: {message_type}")

                except json.JSONDecodeError:
                    logger.error("Invalid JSON received from client")
                except Exception as e:
                    logger.error(f"Error handling client message: {e}")

        except Exception as e:
            logger.error(f"Client message handling failed: {e}")

    async def _handle_subscription_request(
        self, connection: WebSocketConnection, data: Dict[str, Any]
    ):
        """Handle subscription request from client"""
        try:
            subscription_types = data.get("subscriptions", [])
            filters = data.get("filters", {})

            for sub_type_str in subscription_types:
                try:
                    sub_type = SubscriptionType(sub_type_str)
                    connection.subscriptions.add(sub_type)

                    # Apply subscription-specific filters
                    if filters:
                        connection.filters.update(filters)

                    # Call subscription handler if available
                    if sub_type in self.subscription_handlers:
                        handler = self.subscription_handlers[sub_type]
                        await self._executor.submit(handler, connection, filters)

                    # Send acknowledgment
                    await self._send_to_connection(
                        connection.connection_id,
                        {
                            "type": "subscription_ack",
                            "subscription_type": sub_type.value,
                            "status": "subscribed",
                            "timestamp": datetime.now().isoformat(),
                        },
                    )

                    logger.info(
                        f"Subscription added: {sub_type.value} for connection {connection.connection_id}"
                    )

                except ValueError:
                    logger.warning(f"Invalid subscription type: {sub_type_str}")

            with self._lock:
                self.stats["subscriptions_active"] = sum(
                    len(conn.subscriptions) for conn in self.connections.values()
                )

        except Exception as e:
            logger.error(f"Subscription request handling failed: {e}")

    async def _handle_unsubscription_request(
        self, connection: WebSocketConnection, data: Dict[str, Any]
    ):
        """Handle unsubscription request from client"""
        try:
            subscription_types = data.get("subscriptions", [])

            for sub_type_str in subscription_types:
                try:
                    sub_type = SubscriptionType(sub_type_str)
                    connection.subscriptions.discard(sub_type)

                    # Send acknowledgment
                    await self._send_to_connection(
                        connection.connection_id,
                        {
                            "type": "subscription_ack",
                            "subscription_type": sub_type.value,
                            "status": "unsubscribed",
                            "timestamp": datetime.now().isoformat(),
                        },
                    )

                    logger.info(
                        f"Subscription removed: {sub_type.value} for connection {connection.connection_id}"
                    )

                except ValueError:
                    logger.warning(f"Invalid subscription type: {sub_type_str}")

            with self._lock:
                self.stats["subscriptions_active"] = sum(
                    len(conn.subscriptions) for conn in self.connections.values()
                )

        except Exception as e:
            logger.error(f"Unsubscription request handling failed: {e}")

    async def _handle_heartbeat(self, connection: WebSocketConnection):
        """Handle heartbeat from client"""
        try:
            connection.last_heartbeat = datetime.now()

            # Send heartbeat response
            await self._send_to_connection(
                connection.connection_id,
                {"type": "heartbeat_ack", "timestamp": datetime.now().isoformat()},
            )

            with self._lock:
                self.stats["heartbeats_sent"] += 1

        except Exception as e:
            logger.error(f"Heartbeat handling failed: {e}")

    async def _handle_history_request(
        self, connection: WebSocketConnection, data: Dict[str, Any]
    ):
        """Handle request for message history"""
        try:
            subscription_type = data.get("subscription_type")
            limit = min(data.get("limit", 50), 100)  # Max 100 messages
            since = data.get("since")  # ISO timestamp

            if since:
                since_dt = datetime.fromisoformat(since.replace("Z", "+00:00"))
            else:
                since_dt = datetime.now() - timedelta(hours=1)  # Last hour by default

            # Get matching messages from history
            matching_messages = []
            for message in self.message_history.values():
                if (
                    message.created_at >= since_dt
                    and (
                        not subscription_type
                        or message.subscription_type.value == subscription_type
                    )
                    and self._message_matches_connection(message, connection)
                ):
                    matching_messages.append(message)

            # Sort by creation time and limit
            matching_messages.sort(key=lambda m: m.created_at, reverse=True)
            matching_messages = matching_messages[:limit]

            # Send history response
            await self._send_to_connection(
                connection.connection_id,
                {
                    "type": "history_response",
                    "messages": [asdict(msg) for msg in matching_messages],
                    "count": len(matching_messages),
                    "timestamp": datetime.now().isoformat(),
                },
            )

        except Exception as e:
            logger.error(f"History request handling failed: {e}")

    def push_message(
        self,
        message_type: MessageType,
        subscription_type: SubscriptionType,
        payload: Dict[str, Any],
        target_users: List[str] = None,
        filters: Dict[str, Any] = None,
        priority: int = 5,
    ) -> str:
        """
        Push a real-time message to subscribed clients

        Args:
            message_type: Type of message
            subscription_type: Subscription type for routing
            payload: Message payload
            target_users: Specific users to target (None for broadcast)
            filters: Additional filters for message routing
            priority: Message priority

        Returns:
            Message ID
        """
        try:
            message_id = str(uuid.uuid4())

            # Create message
            message = RealtimeMessage(
                message_id=message_id,
                message_type=message_type,
                subscription_type=subscription_type,
                payload=payload,
                target_users=target_users or [],
                filters=filters or {},
                priority=priority,
                created_at=datetime.now(),
                expires_at=datetime.now()
                + timedelta(hours=self.message_retention_hours),
            )

            # Add to queue and history
            with self._lock:
                self.message_queue.append(message)
                self.message_history[message_id] = message

            logger.info(f"Queued message: {message_id} ({message_type.value})")
            return message_id

        except Exception as e:
            logger.error(f"Message pushing failed: {e}")
            return ""

    async def _message_processor(self):
        """Process queued messages for delivery"""
        try:
            while self.running:
                try:
                    if not self.message_queue:
                        await asyncio.sleep(0.1)
                        continue

                    # Get next message
                    with self._lock:
                        if self.message_queue:
                            message = self.message_queue.popleft()
                        else:
                            continue

                    # Check if message expired
                    if message.expires_at and datetime.now() > message.expires_at:
                        continue

                    # Find target connections
                    target_connections = self._get_target_connections(message)

                    # Deliver to each target connection
                    delivery_tasks = []
                    for connection_id in target_connections:
                        delivery_tasks.append(
                            self._deliver_message_to_connection(message, connection_id)
                        )

                    if delivery_tasks:
                        await asyncio.gather(*delivery_tasks, return_exceptions=True)

                except Exception as e:
                    logger.error(f"Message processing error: {e}")
                    await asyncio.sleep(1)  # Brief pause on error

        except Exception as e:
            logger.error(f"Message processor failed: {e}")

    def _get_target_connections(self, message: RealtimeMessage) -> Set[str]:
        """Get target connections for a message"""
        try:
            target_connections = set()

            with self._lock:
                for connection_id, connection in self.connections.items():
                    if not connection.active:
                        continue

                    # Check subscription type
                    if message.subscription_type not in connection.subscriptions:
                        continue

                    # Check user targeting
                    if (
                        message.target_users
                        and connection.user_id not in message.target_users
                    ):
                        continue

                    # Check message filters
                    if not self._message_matches_connection(message, connection):
                        continue

                    target_connections.add(connection_id)

            return target_connections

        except Exception as e:
            logger.error(f"Failed to get target connections: {e}")
            return set()

    def _message_matches_connection(
        self, message: RealtimeMessage, connection: WebSocketConnection
    ) -> bool:
        """Check if message matches connection filters"""
        try:
            # Check message filters against connection filters
            for filter_key, filter_value in message.filters.items():
                connection_filter = connection.filters.get(filter_key)

                if connection_filter is not None:
                    if (
                        isinstance(filter_value, list)
                        and connection_filter not in filter_value
                    ):
                        return False
                    elif (
                        isinstance(filter_value, str)
                        and connection_filter != filter_value
                    ):
                        return False
                    elif isinstance(filter_value, dict):
                        if (
                            "min" in filter_value
                            and connection_filter < filter_value["min"]
                        ):
                            return False
                        if (
                            "max" in filter_value
                            and connection_filter > filter_value["max"]
                        ):
                            return False

            return True

        except Exception as e:
            logger.error(f"Message filter matching failed: {e}")
            return False

    async def _deliver_message_to_connection(
        self, message: RealtimeMessage, connection_id: str
    ):
        """Deliver message to specific connection"""
        try:
            connection = self.connections.get(connection_id)
            if not connection or not connection.active:
                return

            # Prepare message payload
            ws_message = {
                "message_id": message.message_id,
                "type": message.message_type.value,
                "subscription_type": message.subscription_type.value,
                "payload": message.payload,
                "priority": message.priority,
                "timestamp": message.created_at.isoformat(),
            }

            # Track delivery
            delivery = MessageDelivery(
                delivery_id=str(uuid.uuid4()),
                message_id=message.message_id,
                connection_id=connection_id,
                status="pending",
                attempts=1,
                last_attempt=datetime.now(),
                error_message=None,
            )

            try:
                # Send message
                await self._send_to_connection(connection_id, ws_message)

                delivery.status = "sent"
                with self._lock:
                    self.stats["messages_sent"] += 1

            except Exception as e:
                delivery.status = "failed"
                delivery.error_message = str(e)
                with self._lock:
                    self.stats["messages_failed"] += 1
                logger.error(f"Message delivery failed to {connection_id}: {e}")

            # Track delivery
            with self._lock:
                self.delivery_tracking[message.message_id].append(delivery)

        except Exception as e:
            logger.error(f"Message delivery to connection failed: {e}")

    async def _send_to_connection(self, connection_id: str, message: Dict[str, Any]):
        """Send message to specific WebSocket connection"""
        try:
            connection = self.connections.get(connection_id)
            if not connection or not connection.active:
                return

            message_json = json.dumps(message)
            await connection.websocket.send(message_json)

        except Exception as e:
            logger.error(f"Failed to send to connection {connection_id}: {e}")
            # Mark connection as inactive
            if connection_id in self.connections:
                self.connections[connection_id].active = False

    async def _heartbeat_monitor(self):
        """Monitor connection heartbeats"""
        try:
            while self.running:
                try:
                    current_time = datetime.now()
                    timeout_threshold = current_time - timedelta(
                        seconds=self.heartbeat_interval * 3
                    )

                    connections_to_close = []

                    with self._lock:
                        for connection_id, connection in self.connections.items():
                            if connection.last_heartbeat < timeout_threshold:
                                connections_to_close.append(connection_id)

                    # Close timed-out connections
                    for connection_id in connections_to_close:
                        await self._cleanup_connection(connection_id)
                        logger.info(
                            f"Closed connection due to heartbeat timeout: {connection_id}"
                        )

                    await asyncio.sleep(self.heartbeat_interval)

                except Exception as e:
                    logger.error(f"Heartbeat monitoring error: {e}")
                    await asyncio.sleep(10)

        except Exception as e:
            logger.error(f"Heartbeat monitor failed: {e}")

    async def _cleanup_task(self):
        """Background cleanup task"""
        try:
            while self.running:
                try:
                    await self._cleanup_expired_messages()
                    await self._cleanup_old_deliveries()

                    # Run cleanup every hour
                    await asyncio.sleep(3600)

                except Exception as e:
                    logger.error(f"Cleanup task error: {e}")
                    await asyncio.sleep(300)  # Retry after 5 minutes

        except Exception as e:
            logger.error(f"Cleanup task failed: {e}")

    async def _cleanup_expired_messages(self):
        """Clean up expired messages from history"""
        try:
            current_time = datetime.now()
            expired_messages = []

            with self._lock:
                for message_id, message in self.message_history.items():
                    if message.expires_at and current_time > message.expires_at:
                        expired_messages.append(message_id)

                for message_id in expired_messages:
                    del self.message_history[message_id]
                    if message_id in self.delivery_tracking:
                        del self.delivery_tracking[message_id]

            if expired_messages:
                logger.info(f"Cleaned up {len(expired_messages)} expired messages")

        except Exception as e:
            logger.error(f"Message cleanup failed: {e}")

    async def _cleanup_old_deliveries(self):
        """Clean up old delivery tracking data"""
        try:
            cutoff_time = datetime.now() - timedelta(days=7)  # Keep for 7 days

            with self._lock:
                deliveries_to_remove = []
                for message_id, deliveries in self.delivery_tracking.items():
                    filtered_deliveries = [
                        d for d in deliveries if d.last_attempt > cutoff_time
                    ]

                    if filtered_deliveries:
                        self.delivery_tracking[message_id] = filtered_deliveries
                    else:
                        deliveries_to_remove.append(message_id)

                for message_id in deliveries_to_remove:
                    del self.delivery_tracking[message_id]

        except Exception as e:
            logger.error(f"Delivery cleanup failed: {e}")

    async def _cleanup_connection(self, connection_id: str):
        """Clean up a WebSocket connection"""
        try:
            connection = self.connections.get(connection_id)
            if connection:
                # Mark as inactive
                connection.active = False

                # Close WebSocket if still open
                if not connection.websocket.closed:
                    await connection.websocket.close()

                # Remove from tracking
                with self._lock:
                    if connection_id in self.connections:
                        user_id = connection.user_id
                        del self.connections[connection_id]

                        if user_id in self.user_connections:
                            self.user_connections[user_id].discard(connection_id)
                            if not self.user_connections[user_id]:
                                del self.user_connections[user_id]

                        self.stats["active_connections"] -= 1
                        self.stats["subscriptions_active"] = sum(
                            len(conn.subscriptions)
                            for conn in self.connections.values()
                        )

        except Exception as e:
            logger.error(f"Connection cleanup failed: {e}")

    # Default subscription handlers
    def _handle_alert_subscription(
        self, connection: WebSocketConnection, filters: Dict[str, Any]
    ):
        """Handle alert subscription"""
        logger.info(
            f"Alert subscription activated for connection {connection.connection_id}"
        )

    def _handle_analysis_subscription(
        self, connection: WebSocketConnection, filters: Dict[str, Any]
    ):
        """Handle analysis result subscription"""
        logger.info(
            f"Analysis subscription activated for connection {connection.connection_id}"
        )

    def _handle_risk_subscription(
        self, connection: WebSocketConnection, filters: Dict[str, Any]
    ):
        """Handle high-risk entity subscription"""
        logger.info(
            f"Risk subscription activated for connection {connection.connection_id}"
        )

    def _handle_system_subscription(
        self, connection: WebSocketConnection, filters: Dict[str, Any]
    ):
        """Handle system events subscription"""
        logger.info(
            f"System subscription activated for connection {connection.connection_id}"
        )

    def get_connection_statistics(self) -> Dict[str, Any]:
        """Get comprehensive connection statistics"""
        try:
            stats = self.stats.copy()

            # Connection breakdown by user
            user_connection_counts = {
                user_id: len(conn_ids)
                for user_id, conn_ids in self.user_connections.items()
            }

            # Subscription breakdown
            subscription_counts = defaultdict(int)
            with self._lock:
                for connection in self.connections.values():
                    for sub_type in connection.subscriptions:
                        subscription_counts[sub_type.value] += 1

            # Queue statistics
            queue_size = len(self.message_queue)
            history_size = len(self.message_history)

            stats.update(
                {
                    "user_connection_counts": user_connection_counts,
                    "subscription_breakdown": dict(subscription_counts),
                    "queue_size": queue_size,
                    "history_size": history_size,
                    "server_running": self.running,
                    "total_deliveries": sum(
                        len(deliveries)
                        for deliveries in self.delivery_tracking.values()
                    ),
                }
            )

            return stats

        except Exception as e:
            logger.error(f"Statistics calculation failed: {e}")
            return self.stats.copy()

    async def stop_server(self):
        """Stop the WebSocket server"""
        try:
            self.running = False

            # Close all connections
            with self._lock:
                connection_ids = list(self.connections.keys())

            for connection_id in connection_ids:
                await self._cleanup_connection(connection_id)

            # Stop server
            if self.server:
                self.server.close()
                await self.server.wait_closed()

            logger.info("WebSocket server stopped")

        except Exception as e:
            logger.error(f"Server stop failed: {e}")

    def close(self):
        """Close the WebSocket push server and cleanup resources"""
        try:
            # Stop the event loop if running in thread
            if self.loop and self.loop.is_running():
                self.loop.call_soon_threadsafe(
                    lambda: asyncio.create_task(self.stop_server())
                )

            if hasattr(self, "_executor") and self._executor:
                self._executor.shutdown(wait=True)

            logger.info("WebSocket push server closed")

        except Exception as e:
            logger.error(f"Error closing WebSocket server: {e}")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
