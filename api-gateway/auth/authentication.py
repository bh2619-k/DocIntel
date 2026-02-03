"""
Real-Time Document Intelligence & Compliance Engine
Authentication and Authorization Module
Version: 1.0
Date: 2026-02-03

This module provides OAuth2 + JWT authentication with multi-tenant support,
role-based access control, and tenant context propagation.
"""

import jwt
import bcrypt
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Any
from dataclasses import dataclass
from enum import Enum
import logging

from fastapi import HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class UserRole(Enum):
    """User roles for RBAC"""

    LEGAL = "LEGAL"
    FINANCE = "FINANCE"
    ADMIN = "ADMIN"


class TokenType(Enum):
    """JWT token types"""

    ACCESS = "access"
    REFRESH = "refresh"


@dataclass
class UserContext:
    """User context for request processing"""

    user_id: str
    tenant_id: str
    email: str
    name: str
    role: UserRole
    permissions: List[str]
    is_active: bool = True
    last_login: Optional[datetime] = None


class TokenPayload(BaseModel):
    """JWT token payload structure"""

    sub: str  # user_id
    tenant_id: str
    email: str
    name: str
    role: str
    permissions: List[str]
    token_type: str
    exp: int
    iat: int
    jti: str  # JWT ID for revocation


class LoginRequest(BaseModel):
    """Login request schema"""

    email: str
    password: str
    tenant_subdomain: Optional[str] = None


class TokenResponse(BaseModel):
    """Token response schema"""

    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int
    user: Dict[str, Any]


class RefreshTokenRequest(BaseModel):
    """Refresh token request schema"""

    refresh_token: str


class AuthenticationError(Exception):
    """Custom authentication error"""

    def __init__(self, message: str, error_code: str = "AUTH_ERROR"):
        self.message = message
        self.error_code = error_code
        super().__init__(self.message)


class AuthorizationError(Exception):
    """Custom authorization error"""

    def __init__(self, message: str, required_permission: str = None):
        self.message = message
        self.required_permission = required_permission
        super().__init__(self.message)


class JWTManager:
    """JWT token management"""

    def __init__(
        self,
        secret_key: str,
        algorithm: str = "HS256",
        access_token_expire_minutes: int = 60,
        refresh_token_expire_days: int = 30,
    ):
        self.secret_key = secret_key
        self.algorithm = algorithm
        self.access_token_expire_minutes = access_token_expire_minutes
        self.refresh_token_expire_days = refresh_token_expire_days

    def create_access_token(self, user_context: UserContext, jti: str) -> str:
        """Create JWT access token"""
        expires_delta = timedelta(minutes=self.access_token_expire_minutes)
        return self._create_token(
            user_context=user_context,
            token_type=TokenType.ACCESS,
            expires_delta=expires_delta,
            jti=jti,
        )

    def create_refresh_token(self, user_context: UserContext, jti: str) -> str:
        """Create JWT refresh token"""
        expires_delta = timedelta(days=self.refresh_token_expire_days)
        return self._create_token(
            user_context=user_context,
            token_type=TokenType.REFRESH,
            expires_delta=expires_delta,
            jti=jti,
        )

    def _create_token(
        self,
        user_context: UserContext,
        token_type: TokenType,
        expires_delta: timedelta,
        jti: str,
    ) -> str:
        """Create JWT token with specified parameters"""
        now = datetime.utcnow()
        expire = now + expires_delta

        payload = {
            "sub": user_context.user_id,
            "tenant_id": user_context.tenant_id,
            "email": user_context.email,
            "name": user_context.name,
            "role": user_context.role.value,
            "permissions": user_context.permissions,
            "token_type": token_type.value,
            "exp": int(expire.timestamp()),
            "iat": int(now.timestamp()),
            "jti": jti,
        }

        return jwt.encode(payload, self.secret_key, algorithm=self.algorithm)

    def verify_token(
        self, token: str, expected_type: TokenType = TokenType.ACCESS
    ) -> TokenPayload:
        """Verify and decode JWT token"""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])

            # Validate token type
            if payload.get("token_type") != expected_type.value:
                raise AuthenticationError(
                    f"Invalid token type. Expected {expected_type.value}",
                    "INVALID_TOKEN_TYPE",
                )

            return TokenPayload(**payload)

        except jwt.ExpiredSignatureError:
            raise AuthenticationError("Token has expired", "TOKEN_EXPIRED")
        except jwt.InvalidTokenError:
            raise AuthenticationError("Invalid token", "INVALID_TOKEN")

    def decode_token_payload(self, token: str) -> Dict[str, Any]:
        """Decode token payload without verification (for debugging)"""
        try:
            return jwt.decode(token, options={"verify_signature": False})
        except Exception as e:
            logger.error(f"Failed to decode token: {e}")
            return {}


class PasswordManager:
    """Password hashing and verification"""

    @staticmethod
    def hash_password(password: str) -> str:
        """Hash password using bcrypt"""
        salt = bcrypt.gensalt()
        return bcrypt.hashpw(password.encode("utf-8"), salt).decode("utf-8")

    @staticmethod
    def verify_password(password: str, hashed_password: str) -> bool:
        """Verify password against hash"""
        return bcrypt.checkpw(password.encode("utf-8"), hashed_password.encode("utf-8"))


class PermissionManager:
    """Role-based permission management"""

    # Define role-based permissions
    ROLE_PERMISSIONS = {
        UserRole.LEGAL: [
            "documents:read",
            "documents:upload",
            "clauses:read",
            "compliance:read",
            "compliance:review",
            "alerts:read",
            "alerts:acknowledge",
            "reports:legal",
        ],
        UserRole.FINANCE: [
            "documents:read",
            "documents:upload",
            "entities:read",
            "anomalies:read",
            "anomalies:review",
            "alerts:read",
            "alerts:acknowledge",
            "reports:financial",
        ],
        UserRole.ADMIN: [
            "documents:*",
            "entities:*",
            "clauses:*",
            "compliance:*",
            "anomalies:*",
            "alerts:*",
            "users:*",
            "tenants:*",
            "system:*",
            "reports:*",
        ],
    }

    @classmethod
    def get_role_permissions(cls, role: UserRole) -> List[str]:
        """Get permissions for a role"""
        return cls.ROLE_PERMISSIONS.get(role, [])

    @classmethod
    def has_permission(
        cls, user_permissions: List[str], required_permission: str
    ) -> bool:
        """Check if user has required permission"""
        # Check for wildcard permissions
        permission_parts = required_permission.split(":")
        if len(permission_parts) == 2:
            resource, action = permission_parts
            wildcard_permission = f"{resource}:*"
            if wildcard_permission in user_permissions:
                return True

        # Check for exact permission
        return required_permission in user_permissions

    @classmethod
    def validate_permission(cls, user_permissions: List[str], required_permission: str):
        """Validate permission or raise exception"""
        if not cls.has_permission(user_permissions, required_permission):
            raise AuthorizationError(
                f"Missing required permission: {required_permission}",
                required_permission,
            )


class AuthenticationManager:
    """Main authentication manager"""

    def __init__(self, jwt_manager: JWTManager, database_manager, redis_manager):
        self.jwt_manager = jwt_manager
        self.db = database_manager
        self.redis = redis_manager
        self.password_manager = PasswordManager()

    async def authenticate_user(self, login_request: LoginRequest) -> TokenResponse:
        """Authenticate user and return tokens"""
        try:
            # Find user by email and tenant
            user = await self._get_user_by_email(
                login_request.email, login_request.tenant_subdomain
            )

            if not user:
                raise AuthenticationError("Invalid credentials", "INVALID_CREDENTIALS")

            # Verify password
            if not self.password_manager.verify_password(
                login_request.password, user["password_hash"]
            ):
                raise AuthenticationError("Invalid credentials", "INVALID_CREDENTIALS")

            # Check if user is active
            if not user["is_active"]:
                raise AuthenticationError(
                    "Account is deactivated", "ACCOUNT_DEACTIVATED"
                )

            # Create user context
            user_context = UserContext(
                user_id=user["user_id"],
                tenant_id=user["tenant_id"],
                email=user["email"],
                name=user["name"],
                role=UserRole(user["role"]),
                permissions=PermissionManager.get_role_permissions(
                    UserRole(user["role"])
                ),
                is_active=user["is_active"],
                last_login=user.get("last_login_at"),
            )

            # Generate token IDs
            import uuid

            access_jti = str(uuid.uuid4())
            refresh_jti = str(uuid.uuid4())

            # Create tokens
            access_token = self.jwt_manager.create_access_token(
                user_context, access_jti
            )
            refresh_token = self.jwt_manager.create_refresh_token(
                user_context, refresh_jti
            )

            # Store refresh token in Redis
            await self._store_refresh_token(refresh_jti, user_context.user_id)

            # Update last login
            await self._update_last_login(user_context.user_id)

            # Log successful login
            logger.info(f"User {user_context.email} logged in successfully")

            return TokenResponse(
                access_token=access_token,
                refresh_token=refresh_token,
                expires_in=self.jwt_manager.access_token_expire_minutes * 60,
                user={
                    "user_id": user_context.user_id,
                    "tenant_id": user_context.tenant_id,
                    "email": user_context.email,
                    "name": user_context.name,
                    "role": user_context.role.value,
                    "permissions": user_context.permissions,
                },
            )

        except AuthenticationError:
            raise
        except Exception as e:
            logger.error(f"Authentication error: {e}")
            raise AuthenticationError("Authentication failed", "SYSTEM_ERROR")

    async def refresh_access_token(
        self, refresh_request: RefreshTokenRequest
    ) -> TokenResponse:
        """Refresh access token using refresh token"""
        try:
            # Verify refresh token
            token_payload = self.jwt_manager.verify_token(
                refresh_request.refresh_token, TokenType.REFRESH
            )

            # Check if refresh token is stored (not revoked)
            if not await self._is_refresh_token_valid(token_payload.jti):
                raise AuthenticationError("Refresh token revoked", "TOKEN_REVOKED")

            # Get current user data
            user = await self._get_user_by_id(token_payload.sub)
            if not user or not user["is_active"]:
                raise AuthenticationError("User not found or inactive", "USER_INACTIVE")

            # Create new user context
            user_context = UserContext(
                user_id=user["user_id"],
                tenant_id=user["tenant_id"],
                email=user["email"],
                name=user["name"],
                role=UserRole(user["role"]),
                permissions=PermissionManager.get_role_permissions(
                    UserRole(user["role"])
                ),
                is_active=user["is_active"],
            )

            # Generate new access token
            import uuid

            access_jti = str(uuid.uuid4())
            access_token = self.jwt_manager.create_access_token(
                user_context, access_jti
            )

            return TokenResponse(
                access_token=access_token,
                refresh_token=refresh_request.refresh_token,  # Keep same refresh token
                expires_in=self.jwt_manager.access_token_expire_minutes * 60,
                user={
                    "user_id": user_context.user_id,
                    "tenant_id": user_context.tenant_id,
                    "email": user_context.email,
                    "name": user_context.name,
                    "role": user_context.role.value,
                    "permissions": user_context.permissions,
                },
            )

        except AuthenticationError:
            raise
        except Exception as e:
            logger.error(f"Token refresh error: {e}")
            raise AuthenticationError("Token refresh failed", "SYSTEM_ERROR")

    async def validate_access_token(self, token: str) -> UserContext:
        """Validate access token and return user context"""
        token_payload = self.jwt_manager.verify_token(token, TokenType.ACCESS)

        # Create user context from token
        return UserContext(
            user_id=token_payload.sub,
            tenant_id=token_payload.tenant_id,
            email=token_payload.email,
            name=token_payload.name,
            role=UserRole(token_payload.role),
            permissions=token_payload.permissions,
            is_active=True,  # Token is valid, assume user is active
        )

    async def logout_user(self, refresh_token: str):
        """Logout user by revoking refresh token"""
        try:
            token_payload = self.jwt_manager.verify_token(
                refresh_token, TokenType.REFRESH
            )
            await self._revoke_refresh_token(token_payload.jti)
            logger.info(f"User {token_payload.email} logged out successfully")
        except Exception as e:
            logger.error(f"Logout error: {e}")

    # Private helper methods

    async def _get_user_by_email(
        self, email: str, tenant_subdomain: str = None
    ) -> Optional[Dict]:
        """Get user by email and tenant"""
        # This would query the PostgreSQL database
        # Implementation depends on database manager
        pass

    async def _get_user_by_id(self, user_id: str) -> Optional[Dict]:
        """Get user by ID"""
        # This would query the PostgreSQL database
        pass

    async def _store_refresh_token(self, jti: str, user_id: str):
        """Store refresh token in Redis"""
        key = f"refresh_token:{jti}"
        ttl = self.jwt_manager.refresh_token_expire_days * 24 * 3600
        await self.redis.setex(key, ttl, user_id)

    async def _is_refresh_token_valid(self, jti: str) -> bool:
        """Check if refresh token is still valid"""
        key = f"refresh_token:{jti}"
        return await self.redis.exists(key)

    async def _revoke_refresh_token(self, jti: str):
        """Revoke refresh token"""
        key = f"refresh_token:{jti}"
        await self.redis.delete(key)

    async def _update_last_login(self, user_id: str):
        """Update user's last login timestamp"""
        # This would update the PostgreSQL database
        pass


class AuthenticationBearer(HTTPBearer):
    """FastAPI authentication dependency"""

    def __init__(self, auth_manager: AuthenticationManager, auto_error: bool = True):
        super().__init__(auto_error=auto_error)
        self.auth_manager = auth_manager

    async def __call__(
        self, credentials: HTTPAuthorizationCredentials = None
    ) -> UserContext:
        if credentials is None or credentials.scheme.lower() != "bearer":
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authentication credentials",
                headers={"WWW-Authenticate": "Bearer"},
            )

        try:
            return await self.auth_manager.validate_access_token(
                credentials.credentials
            )
        except AuthenticationError as e:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=e.message,
                headers={"WWW-Authenticate": "Bearer"},
            )


def require_permission(permission: str):
    """Decorator for requiring specific permissions"""

    def decorator(func):
        async def wrapper(*args, **kwargs):
            # Extract user context from kwargs
            user_context = None
            for arg in kwargs.values():
                if isinstance(arg, UserContext):
                    user_context = arg
                    break

            if user_context is None:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="User context not found",
                )

            # Check permission
            try:
                PermissionManager.validate_permission(
                    user_context.permissions, permission
                )
            except AuthorizationError as e:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN, detail=e.message
                )

            return await func(*args, **kwargs)

        return wrapper

    return decorator
