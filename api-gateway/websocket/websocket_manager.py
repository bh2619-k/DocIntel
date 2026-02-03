"""
Real-Time Document Intelligence & Compliance Engine
WebSocket Connection Management
Version: 1.0
Date: 2026-02-03

This module provides WebSocket connection management for real-time updates
of document processing status, alerts, and compliance notifications.
"""

import json
import asyncio
import time
from typing import Dict, List, Set, Optional, Any
from dataclasses import dataclass, asdict
from enum import Enum
import logging
import uuid

from fastapi import WebSocket
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class MessageType(Enum):
    """WebSocket message types"""

    # Authentication
    AUTH = "auth"
    AUTH_SUCCESS = "auth_success"
    AUTH_FAILED = "auth_failed"

    # Document processing updates
    DOCUMENT_STATUS = "document_status"
    PROCESSING_PROGRESS = "processing_progress"
    PROCESSING_COMPLETE = "processing_complete"
    PROCESSING_ERROR = "processing_error"

    # Alerts and notifications
    ALERT = "alert"
    COMPLIANCE_VIOLATION = "compliance_violation"
    ANOMALY_DETECTED = "anomaly_detected"

    # System events
    HEARTBEAT = "heartbeat"
    ERROR = "error"
    DISCONNECT = "disconnect"


@dataclass
class WebSocketMessage:
    """WebSocket message structure"""

    type: str
    payload: Dict[str, Any]
    timestamp: float = None
    message_id: str = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()
        if self.message_id is None:
            self.message_id = str(uuid.uuid4())


class AuthMessage(BaseModel):
    """Authentication message payload"""

    token: str
    client_type: Optional[str] = "browser"
    subscriptions: Optional[List[str]] = []


class DocumentStatusMessage(BaseModel):
    """Document status update payload"""

    document_id: str
    status: str
    stage: Optional[str] = None
    progress_percentage: Optional[int] = None
    error_message: Optional[str] = None
    estimated_completion: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = {}


class AlertMessage(BaseModel):
    """Alert notification payload"""

    alert_id: str
    document_id: str
    alert_type: str
    severity: str
    title: str
    message: str
    explanation: str
    confidence: float
    created_at: float


@dataclass
class ConnectionInfo:
    """WebSocket connection information"""

    connection_id: str
    websocket: WebSocket
    user_id: str
    tenant_id: str
    user_role: str
    authenticated: bool = False
    connected_at: float = None
    last_heartbeat: float = None
    subscriptions: Set[str] = None
    client_info: Dict[str, Any] = None

    def __post_init__(self):
        if self.connected_at is None:
            self.connected_at = time.time()
        if self.last_heartbeat is None:
            self.last_heartbeat = time.time()
        if self.subscriptions is None:
            self.subscriptions = set()
        if self.client_info is None:
            self.client_info = {}


class WebSocketManager:
    """
    Manages WebSocket connections for real-time updates
    Provides tenant-isolated message broadcasting and connection management
    """

    def __init__(self, auth_manager, redis_manager):
        self.auth_manager = auth_manager
        self.redis = redis_manager

        # Active connections by connection ID
        self.connections: Dict[str, ConnectionInfo] = {}

        # Connections by tenant for efficient broadcasting
        self.tenant_connections: Dict[str, Set[str]] = {}

        # Connections by user for user-specific updates
        self.user_connections: Dict[str, Set[str]] = {}

        # Subscription-based routing
        self.subscriptions: Dict[str, Set[str]] = {}

        # Heartbeat settings
        self.heartbeat_interval = 30  # seconds
        self.heartbeat_timeout = 60  # seconds

        # Start background tasks
        self._heartbeat_task = None
        self._cleanup_task = None

    async def start_background_tasks(self):
        """Start background maintenance tasks"""
        if self._heartbeat_task is None:
            self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())
        if self._cleanup_task is None:
            self._cleanup_task = asyncio.create_task(self._cleanup_loop())

    async def stop_background_tasks(self):
        """Stop background maintenance tasks"""
        if self._heartbeat_task:
            self._heartbeat_task.cancel()
            self._heartbeat_task = None
        if self._cleanup_task:
            self._cleanup_task.cancel()
            self._cleanup_task = None

    async def connect(self, websocket: WebSocket) -> str:
        """
        Accept WebSocket connection and return connection ID
        Authentication happens in separate step
        """
        await websocket.accept()

        connection_id = str(uuid.uuid4())

        # Create connection info (unauthenticated initially)
        connection = ConnectionInfo(
            connection_id=connection_id,
            websocket=websocket,
            user_id="",
            tenant_id="",
            user_role="",
            authenticated=False,
        )

        self.connections[connection_id] = connection

        logger.info(f"WebSocket connection {connection_id} established")
        return connection_id

    async def authenticate(self, connection_id: str, auth_message: AuthMessage) -> bool:
        """Authenticate WebSocket connection"""
        if connection_id not in self.connections:
            return False

        connection = self.connections[connection_id]

        try:
            # Validate JWT token
            user_context = await self.auth_manager.validate_access_token(
                auth_message.token
            )

            # Update connection info
            connection.user_id = user_context.user_id
            connection.tenant_id = user_context.tenant_id
            connection.user_role = user_context.role.value
            connection.authenticated = True
            connection.client_info = {
                "client_type": auth_message.client_type,
                "user_email": user_context.email,
                "user_name": user_context.name,
            }

            # Add to tenant and user mappings
            if user_context.tenant_id not in self.tenant_connections:
                self.tenant_connections[user_context.tenant_id] = set()
            self.tenant_connections[user_context.tenant_id].add(connection_id)

            if user_context.user_id not in self.user_connections:
                self.user_connections[user_context.user_id] = set()
            self.user_connections[user_context.user_id].add(connection_id)

            # Setup subscriptions
            if auth_message.subscriptions:
                await self.subscribe(connection_id, auth_message.subscriptions)

            # Send authentication success
            await self.send_to_connection(
                connection_id,
                WebSocketMessage(
                    type=MessageType.AUTH_SUCCESS.value,
                    payload={
                        "user_id": user_context.user_id,
                        "tenant_id": user_context.tenant_id,
                        "role": user_context.role.value,
                        "permissions": user_context.permissions,
                    },
                ),
            )

            logger.info(
                f"WebSocket connection {connection_id} authenticated for user {user_context.email}"
            )
            return True

        except Exception as e:
            logger.error(f"WebSocket authentication failed for {connection_id}: {e}")

            await self.send_to_connection(
                connection_id,
                WebSocketMessage(
                    type=MessageType.AUTH_FAILED.value,
                    payload={"error": "Authentication failed", "details": str(e)},
                ),
            )
            return False

    async def disconnect(self, connection_id: str):
        """Disconnect and cleanup WebSocket connection"""
        if connection_id not in self.connections:
            return

        connection = self.connections[connection_id]

        # Remove from mappings
        if connection.authenticated:
            if connection.tenant_id in self.tenant_connections:
                self.tenant_connections[connection.tenant_id].discard(connection_id)
                if not self.tenant_connections[connection.tenant_id]:
                    del self.tenant_connections[connection.tenant_id]

            if connection.user_id in self.user_connections:
                self.user_connections[connection.user_id].discard(connection_id)
                if not self.user_connections[connection.user_id]:
                    del self.user_connections[connection.user_id]

        # Remove from subscriptions
        for subscription_type in list(connection.subscriptions):
            await self.unsubscribe(connection_id, subscription_type)

        # Close WebSocket if still open
        try:
            if connection.websocket.client_state.name != "DISCONNECTED":
                await connection.websocket.close()
        except Exception as e:
            logger.debug(f"Error closing WebSocket {connection_id}: {e}")

        # Remove from connections
        del self.connections[connection_id]

        logger.info(f"WebSocket connection {connection_id} disconnected")

    async def subscribe(self, connection_id: str, subscription_types: List[str]):
        """Subscribe connection to specific message types"""
        if connection_id not in self.connections:
            return

        connection = self.connections[connection_id]

        for sub_type in subscription_types:
            if sub_type not in self.subscriptions:
                self.subscriptions[sub_type] = set()

            self.subscriptions[sub_type].add(connection_id)
            connection.subscriptions.add(sub_type)

        logger.debug(f"Connection {connection_id} subscribed to {subscription_types}")

    async def unsubscribe(self, connection_id: str, subscription_type: str):
        """Unsubscribe connection from message type"""
        if subscription_type in self.subscriptions:
            self.subscriptions[subscription_type].discard(connection_id)
            if not self.subscriptions[subscription_type]:
                del self.subscriptions[subscription_type]

        if connection_id in self.connections:
            self.connections[connection_id].subscriptions.discard(subscription_type)

    async def send_to_connection(self, connection_id: str, message: WebSocketMessage):
        """Send message to specific connection"""
        if connection_id not in self.connections:
            return False

        connection = self.connections[connection_id]

        try:
            message_data = {
                "type": message.type,
                "payload": message.payload,
                "timestamp": message.timestamp,
                "message_id": message.message_id,
            }

            await connection.websocket.send_text(json.dumps(message_data))
            return True

        except Exception as e:
            logger.error(f"Failed to send message to connection {connection_id}: {e}")
            # Connection is probably broken, disconnect it
            await self.disconnect(connection_id)
            return False

    async def broadcast_to_tenant(self, tenant_id: str, message: WebSocketMessage):
        """Broadcast message to all connections for a tenant"""
        if tenant_id not in self.tenant_connections:
            return 0

        sent_count = 0
        connection_ids = list(self.tenant_connections[tenant_id])

        for connection_id in connection_ids:
            if await self.send_to_connection(connection_id, message):
                sent_count += 1

        logger.debug(
            f"Broadcasted message to {sent_count} connections for tenant {tenant_id}"
        )
        return sent_count

    async def send_to_user(self, user_id: str, message: WebSocketMessage):
        """Send message to all connections for a user"""
        if user_id not in self.user_connections:
            return 0

        sent_count = 0
        connection_ids = list(self.user_connections[user_id])

        for connection_id in connection_ids:
            if await self.send_to_connection(connection_id, message):
                sent_count += 1

        return sent_count

    async def broadcast_to_subscribers(
        self, subscription_type: str, message: WebSocketMessage
    ):
        """Broadcast message to all subscribers of a type"""
        if subscription_type not in self.subscriptions:
            return 0

        sent_count = 0
        connection_ids = list(self.subscriptions[subscription_type])

        for connection_id in connection_ids:
            connection = self.connections.get(connection_id)
            if connection and connection.authenticated:
                if await self.send_to_connection(connection_id, message):
                    sent_count += 1

        return sent_count

    async def handle_message(self, connection_id: str, message_data: str):
        """Handle incoming WebSocket message"""
        if connection_id not in self.connections:
            return

        connection = self.connections[connection_id]

        try:
            data = json.loads(message_data)
            message_type = data.get("type")
            payload = data.get("payload", {})

            if message_type == MessageType.AUTH.value and not connection.authenticated:
                # Handle authentication
                auth_msg = AuthMessage(**payload)
                await self.authenticate(connection_id, auth_msg)

            elif message_type == MessageType.HEARTBEAT.value:
                # Update heartbeat timestamp
                connection.last_heartbeat = time.time()

                # Send heartbeat response
                await self.send_to_connection(
                    connection_id,
                    WebSocketMessage(
                        type=MessageType.HEARTBEAT.value,
                        payload={"timestamp": time.time()},
                    ),
                )

            elif connection.authenticated:
                # Handle other message types for authenticated connections
                await self._handle_authenticated_message(
                    connection_id, message_type, payload
                )

            else:
                # Unauthenticated connection trying to send non-auth message
                await self.send_to_connection(
                    connection_id,
                    WebSocketMessage(
                        type=MessageType.ERROR.value,
                        payload={"error": "Authentication required"},
                    ),
                )

        except Exception as e:
            logger.error(f"Error handling message from {connection_id}: {e}")
            await self.send_to_connection(
                connection_id,
                WebSocketMessage(
                    type=MessageType.ERROR.value,
                    payload={"error": "Invalid message format"},
                ),
            )

    async def _handle_authenticated_message(
        self, connection_id: str, message_type: str, payload: Dict
    ):
        """Handle messages from authenticated connections"""
        # Currently, clients mainly receive messages, but we can add
        # client-initiated actions here (e.g., subscription management)

        if message_type == "subscribe":
            subscription_types = payload.get("types", [])
            await self.subscribe(connection_id, subscription_types)

        elif message_type == "unsubscribe":
            subscription_type = payload.get("type")
            if subscription_type:
                await self.unsubscribe(connection_id, subscription_type)

    async def _heartbeat_loop(self):
        """Background task to send heartbeats and check connection health"""
        while True:
            try:
                current_time = time.time()
                stale_connections = []

                for connection_id, connection in self.connections.items():
                    if connection.authenticated:
                        # Check if connection is stale
                        if (
                            current_time - connection.last_heartbeat
                            > self.heartbeat_timeout
                        ):
                            stale_connections.append(connection_id)

                        # Send periodic heartbeat
                        elif (
                            current_time - connection.last_heartbeat
                            > self.heartbeat_interval
                        ):
                            await self.send_to_connection(
                                connection_id,
                                WebSocketMessage(
                                    type=MessageType.HEARTBEAT.value,
                                    payload={"server_time": current_time},
                                ),
                            )

                # Disconnect stale connections
                for connection_id in stale_connections:
                    logger.warning(
                        f"Disconnecting stale WebSocket connection {connection_id}"
                    )
                    await self.disconnect(connection_id)

                await asyncio.sleep(self.heartbeat_interval)

            except Exception as e:
                logger.error(f"Error in heartbeat loop: {e}")
                await asyncio.sleep(5)

    async def _cleanup_loop(self):
        """Background task to cleanup disconnected connections"""
        while True:
            try:
                await asyncio.sleep(300)  # Run every 5 minutes

                disconnected_connections = []

                for connection_id, connection in self.connections.items():
                    try:
                        if connection.websocket.client_state.name == "DISCONNECTED":
                            disconnected_connections.append(connection_id)
                    except Exception:
                        # Connection object is in bad state
                        disconnected_connections.append(connection_id)

                # Cleanup disconnected connections
                for connection_id in disconnected_connections:
                    await self.disconnect(connection_id)

                if disconnected_connections:
                    logger.info(
                        f"Cleaned up {len(disconnected_connections)} disconnected WebSocket connections"
                    )

            except Exception as e:
                logger.error(f"Error in cleanup loop: {e}")

    def get_connection_stats(self) -> Dict[str, Any]:
        """Get WebSocket connection statistics"""
        total_connections = len(self.connections)
        authenticated_connections = sum(
            1 for c in self.connections.values() if c.authenticated
        )

        tenant_stats = {}
        for tenant_id, connection_ids in self.tenant_connections.items():
            tenant_stats[tenant_id] = len(connection_ids)

        return {
            "total_connections": total_connections,
            "authenticated_connections": authenticated_connections,
            "unauthenticated_connections": total_connections
            - authenticated_connections,
            "tenant_connections": tenant_stats,
            "total_tenants": len(self.tenant_connections),
            "subscription_types": len(self.subscriptions),
            "uptime": time.time(),
        }


# Event handlers for different message types


class DocumentProcessingNotifier:
    """Handles document processing notifications via WebSocket"""

    def __init__(self, websocket_manager: WebSocketManager):
        self.ws_manager = websocket_manager

    async def notify_document_status_change(
        self,
        tenant_id: str,
        document_id: str,
        status: str,
        stage: str = None,
        progress_percentage: int = None,
        error_message: str = None,
        estimated_completion: float = None,
        metadata: Dict[str, Any] = None,
    ):
        """Notify about document status changes"""

        message = WebSocketMessage(
            type=MessageType.DOCUMENT_STATUS.value,
            payload=asdict(
                DocumentStatusMessage(
                    document_id=document_id,
                    status=status,
                    stage=stage,
                    progress_percentage=progress_percentage,
                    error_message=error_message,
                    estimated_completion=estimated_completion,
                    metadata=metadata or {},
                )
            ),
        )

        # Broadcast to tenant and subscribers
        await self.ws_manager.broadcast_to_tenant(tenant_id, message)
        await self.ws_manager.broadcast_to_subscribers("document_status", message)

    async def notify_processing_complete(
        self,
        tenant_id: str,
        document_id: str,
        processing_time: float,
        entities_found: int,
        clauses_found: int,
        alerts_generated: int,
    ):
        """Notify when document processing completes"""

        message = WebSocketMessage(
            type=MessageType.PROCESSING_COMPLETE.value,
            payload={
                "document_id": document_id,
                "processing_time": processing_time,
                "entities_found": entities_found,
                "clauses_found": clauses_found,
                "alerts_generated": alerts_generated,
                "completed_at": time.time(),
            },
        )

        await self.ws_manager.broadcast_to_tenant(tenant_id, message)


class AlertNotifier:
    """Handles alert notifications via WebSocket"""

    def __init__(self, websocket_manager: WebSocketManager):
        self.ws_manager = websocket_manager

    async def notify_alert(self, tenant_id: str, alert_data: Dict[str, Any]):
        """Notify about new alerts"""

        message = WebSocketMessage(type=MessageType.ALERT.value, payload=alert_data)

        # Broadcast to tenant and alert subscribers
        await self.ws_manager.broadcast_to_tenant(tenant_id, message)
        await self.ws_manager.broadcast_to_subscribers("alerts", message)

    async def notify_compliance_violation(
        self, tenant_id: str, violation_data: Dict[str, Any]
    ):
        """Notify about compliance violations"""

        message = WebSocketMessage(
            type=MessageType.COMPLIANCE_VIOLATION.value, payload=violation_data
        )

        await self.ws_manager.broadcast_to_tenant(tenant_id, message)
        await self.ws_manager.broadcast_to_subscribers("compliance", message)
