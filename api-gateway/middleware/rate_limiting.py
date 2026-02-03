"""
Real-Time Document Intelligence & Compliance Engine
Rate Limiting Middleware
Version: 1.0
Date: 2026-02-03

This module provides comprehensive rate limiting with Redis backend,
supporting per-tenant, per-user, and per-endpoint limits with sliding window.
"""

import time
from typing import Dict, List, Optional
from dataclasses import dataclass
from enum import Enum
import logging

from fastapi import Request, HTTPException, status
from fastapi.responses import JSONResponse
import redis.asyncio as redis

logger = logging.getLogger(__name__)


class RateLimitType(Enum):
    """Rate limit types"""

    PER_MINUTE = "per_minute"
    PER_HOUR = "per_hour"
    PER_DAY = "per_day"


class RateLimitScope(Enum):
    """Rate limit scope"""

    GLOBAL = "global"
    TENANT = "tenant"
    USER = "user"
    IP = "ip"


@dataclass
class RateLimitConfig:
    """Rate limit configuration"""

    limit: int
    window_seconds: int
    scope: RateLimitScope
    endpoint_pattern: Optional[str] = None
    user_role_limits: Optional[Dict[str, int]] = None
    tenant_overrides: Optional[Dict[str, int]] = None


@dataclass
class RateLimitResult:
    """Rate limit check result"""

    allowed: bool
    limit: int
    remaining: int
    reset_time: int
    retry_after: Optional[int] = None


class RateLimitExceeded(Exception):
    """Rate limit exceeded exception"""

    def __init__(self, result: RateLimitResult, message: str = "Rate limit exceeded"):
        self.result = result
        self.message = message
        super().__init__(self.message)


class SlidingWindowRateLimiter:
    """Sliding window rate limiter using Redis"""

    def __init__(self, redis_client: redis.Redis):
        self.redis = redis_client

    async def check_rate_limit(
        self, key: str, limit: int, window_seconds: int
    ) -> RateLimitResult:
        """
        Check rate limit using sliding window algorithm
        Returns RateLimitResult with current status
        """
        now = int(time.time())
        window_start = now - window_seconds

        # Use Redis pipeline for atomicity
        pipe = self.redis.pipeline()

        # Remove expired entries
        pipe.zremrangebyscore(key, 0, window_start)

        # Count current requests
        pipe.zcard(key)

        # Add current request
        pipe.zadd(key, {str(now): now})

        # Set expiration
        pipe.expire(key, window_seconds)

        # Execute pipeline
        results = await pipe.execute()

        current_count = results[1]  # Count after cleanup

        # Calculate remaining and reset time
        remaining = max(0, limit - current_count - 1)  # -1 for current request
        reset_time = now + window_seconds

        allowed = current_count < limit
        retry_after = None if allowed else window_seconds

        return RateLimitResult(
            allowed=allowed,
            limit=limit,
            remaining=remaining,
            reset_time=reset_time,
            retry_after=retry_after,
        )

    async def get_current_usage(self, key: str, window_seconds: int) -> int:
        """Get current usage count for a key"""
        now = int(time.time())
        window_start = now - window_seconds

        # Clean up expired entries and count
        await self.redis.zremrangebyscore(key, 0, window_start)
        return await self.redis.zcard(key)


class RateLimitManager:
    """Comprehensive rate limiting manager"""

    def __init__(self, redis_client: redis.Redis):
        self.redis = redis_client
        self.limiter = SlidingWindowRateLimiter(redis_client)

        # Default rate limits (can be overridden per tenant/user)
        self.default_limits = {
            RateLimitType.PER_MINUTE: {
                RateLimitScope.GLOBAL: RateLimitConfig(1000, 60, RateLimitScope.GLOBAL),
                RateLimitScope.TENANT: RateLimitConfig(500, 60, RateLimitScope.TENANT),
                RateLimitScope.USER: RateLimitConfig(60, 60, RateLimitScope.USER),
                RateLimitScope.IP: RateLimitConfig(100, 60, RateLimitScope.IP),
            },
            RateLimitType.PER_HOUR: {
                RateLimitScope.TENANT: RateLimitConfig(
                    10000, 3600, RateLimitScope.TENANT
                ),
                RateLimitScope.USER: RateLimitConfig(1000, 3600, RateLimitScope.USER),
                RateLimitScope.IP: RateLimitConfig(2000, 3600, RateLimitScope.IP),
            },
            RateLimitType.PER_DAY: {
                RateLimitScope.TENANT: RateLimitConfig(
                    100000, 86400, RateLimitScope.TENANT
                ),
                RateLimitScope.USER: RateLimitConfig(10000, 86400, RateLimitScope.USER),
                RateLimitScope.IP: RateLimitConfig(20000, 86400, RateLimitScope.IP),
            },
        }

        # Endpoint-specific limits
        self.endpoint_limits = {
            "/api/v1/documents/upload": {
                RateLimitScope.USER: RateLimitConfig(
                    10, 60, RateLimitScope.USER
                ),  # 10 uploads per minute
                RateLimitScope.TENANT: RateLimitConfig(
                    100, 60, RateLimitScope.TENANT
                ),  # 100 uploads per minute per tenant
            },
            "/api/v1/documents/query": {
                RateLimitScope.USER: RateLimitConfig(100, 60, RateLimitScope.USER),
                RateLimitScope.TENANT: RateLimitConfig(1000, 60, RateLimitScope.TENANT),
            },
        }

        # Role-based limit multipliers
        self.role_multipliers = {"ADMIN": 2.0, "LEGAL": 1.5, "FINANCE": 1.5}

    async def check_all_limits(
        self,
        request: Request,
        tenant_id: str = None,
        user_id: str = None,
        user_role: str = None,
    ) -> List[RateLimitResult]:
        """Check all applicable rate limits for a request"""
        results = []
        client_ip = self._get_client_ip(request)
        endpoint = request.url.path

        # Check different scopes and time windows
        checks = [
            (RateLimitScope.IP, client_ip, RateLimitType.PER_MINUTE),
            (RateLimitScope.IP, client_ip, RateLimitType.PER_HOUR),
            (RateLimitScope.IP, client_ip, RateLimitType.PER_DAY),
        ]

        if tenant_id:
            checks.extend(
                [
                    (RateLimitScope.TENANT, tenant_id, RateLimitType.PER_MINUTE),
                    (RateLimitScope.TENANT, tenant_id, RateLimitType.PER_HOUR),
                    (RateLimitScope.TENANT, tenant_id, RateLimitType.PER_DAY),
                ]
            )

        if user_id:
            checks.extend(
                [
                    (RateLimitScope.USER, user_id, RateLimitType.PER_MINUTE),
                    (RateLimitScope.USER, user_id, RateLimitType.PER_HOUR),
                    (RateLimitScope.USER, user_id, RateLimitType.PER_DAY),
                ]
            )

        # Global limits
        checks.append((RateLimitScope.GLOBAL, "global", RateLimitType.PER_MINUTE))

        for scope, identifier, limit_type in checks:
            # Get rate limit config
            config = await self._get_rate_limit_config(
                scope, limit_type, endpoint, tenant_id, user_role
            )

            if config:
                # Generate Redis key
                key = self._generate_rate_limit_key(
                    scope, identifier, limit_type, endpoint
                )

                # Check rate limit
                result = await self.limiter.check_rate_limit(
                    key, config.limit, config.window_seconds
                )

                results.append(result)

                # If any limit is exceeded, raise exception
                if not result.allowed:
                    raise RateLimitExceeded(
                        result, f"Rate limit exceeded for {scope.value}"
                    )

        return results

    async def _get_rate_limit_config(
        self,
        scope: RateLimitScope,
        limit_type: RateLimitType,
        endpoint: str,
        tenant_id: str = None,
        user_role: str = None,
    ) -> Optional[RateLimitConfig]:
        """Get rate limit configuration for scope and type"""

        # Check endpoint-specific limits first
        if endpoint in self.endpoint_limits and scope in self.endpoint_limits[endpoint]:
            config = self.endpoint_limits[endpoint][scope]
        else:
            # Use default limits
            if (
                limit_type not in self.default_limits
                or scope not in self.default_limits[limit_type]
            ):
                return None
            config = self.default_limits[limit_type][scope]

        # Apply tenant-specific overrides
        if tenant_id and config.tenant_overrides:
            if tenant_id in config.tenant_overrides:
                config.limit = config.tenant_overrides[tenant_id]

        # Apply role-based multipliers
        if user_role and user_role in self.role_multipliers:
            config.limit = int(config.limit * self.role_multipliers[user_role])

        return config

    def _generate_rate_limit_key(
        self,
        scope: RateLimitScope,
        identifier: str,
        limit_type: RateLimitType,
        endpoint: str = None,
    ) -> str:
        """Generate Redis key for rate limiting"""
        base_key = f"rate_limit:{scope.value}:{identifier}:{limit_type.value}"

        if endpoint and scope in [RateLimitScope.USER, RateLimitScope.TENANT]:
            # Include endpoint for user/tenant specific limits
            endpoint_hash = (
                hash(endpoint) % 10000
            )  # Simple hash to keep key length reasonable
            base_key += f":ep_{endpoint_hash}"

        return base_key

    def _get_client_ip(self, request: Request) -> str:
        """Extract client IP from request"""
        # Check for forwarded IP headers (for reverse proxy setups)
        forwarded_ips = request.headers.get("x-forwarded-for")
        if forwarded_ips:
            return forwarded_ips.split(",")[0].strip()

        real_ip = request.headers.get("x-real-ip")
        if real_ip:
            return real_ip

        # Fallback to direct connection IP
        if request.client:
            return request.client.host

        return "unknown"

    async def get_rate_limit_status(
        self, tenant_id: str = None, user_id: str = None, endpoint: str = None
    ) -> Dict[str, Dict[str, int]]:
        """Get current rate limit status for monitoring"""
        status = {}

        scopes_to_check = [(RateLimitScope.GLOBAL, "global")]

        if tenant_id:
            scopes_to_check.append((RateLimitScope.TENANT, tenant_id))

        if user_id:
            scopes_to_check.append((RateLimitScope.USER, user_id))

        for scope, identifier in scopes_to_check:
            scope_status = {}

            for limit_type in RateLimitType:
                config = await self._get_rate_limit_config(scope, limit_type, endpoint)
                if config:
                    key = self._generate_rate_limit_key(
                        scope, identifier, limit_type, endpoint
                    )
                    current_usage = await self.limiter.get_current_usage(
                        key, config.window_seconds
                    )

                    scope_status[limit_type.value] = {
                        "limit": config.limit,
                        "current": current_usage,
                        "remaining": max(0, config.limit - current_usage),
                    }

            status[f"{scope.value}_{identifier}"] = scope_status

        return status

    async def reset_rate_limits(self, tenant_id: str = None, user_id: str = None):
        """Reset rate limits for tenant or user (admin function)"""
        keys_to_delete = []

        if tenant_id:
            pattern = f"rate_limit:tenant:{tenant_id}:*"
            keys = await self.redis.keys(pattern)
            keys_to_delete.extend(keys)

        if user_id:
            pattern = f"rate_limit:user:{user_id}:*"
            keys = await self.redis.keys(pattern)
            keys_to_delete.extend(keys)

        if keys_to_delete:
            await self.redis.delete(*keys_to_delete)
            logger.info(f"Reset rate limits for {len(keys_to_delete)} keys")


class RateLimitMiddleware:
    """FastAPI middleware for rate limiting"""

    def __init__(self, rate_limit_manager: RateLimitManager):
        self.rate_limit_manager = rate_limit_manager

        # Paths to exclude from rate limiting
        self.excluded_paths = [
            "/health",
            "/metrics",
            "/docs",
            "/redoc",
            "/openapi.json",
        ]

    async def __call__(self, request: Request, call_next):
        """Process request with rate limiting"""

        # Skip rate limiting for excluded paths
        if any(request.url.path.startswith(path) for path in self.excluded_paths):
            return await call_next(request)

        try:
            # Extract context from request
            tenant_id = getattr(request.state, "tenant_id", None)
            user_id = getattr(request.state, "user_id", None)
            user_role = getattr(request.state, "user_role", None)

            # Check rate limits
            results = await self.rate_limit_manager.check_all_limits(
                request, tenant_id, user_id, user_role
            )

            # Store rate limit info in request state for response headers
            if results:
                # Use the most restrictive result for headers
                most_restrictive = min(results, key=lambda r: r.remaining)
                request.state.rate_limit_result = most_restrictive

            # Process request
            response = await call_next(request)

            # Add rate limit headers to response
            if hasattr(request.state, "rate_limit_result"):
                result = request.state.rate_limit_result
                response.headers["X-RateLimit-Limit"] = str(result.limit)
                response.headers["X-RateLimit-Remaining"] = str(result.remaining)
                response.headers["X-RateLimit-Reset"] = str(result.reset_time)

            return response

        except RateLimitExceeded as e:
            # Return rate limit exceeded response
            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={
                    "error": "rate_limit_exceeded",
                    "message": e.message,
                    "limit": e.result.limit,
                    "remaining": e.result.remaining,
                    "reset_time": e.result.reset_time,
                    "retry_after": e.result.retry_after,
                },
                headers={
                    "X-RateLimit-Limit": str(e.result.limit),
                    "X-RateLimit-Remaining": str(e.result.remaining),
                    "X-RateLimit-Reset": str(e.result.reset_time),
                    "Retry-After": str(e.result.retry_after)
                    if e.result.retry_after
                    else "60",
                },
            )

        except Exception as e:
            logger.error(f"Rate limiting error: {e}")
            # Continue processing if rate limiting fails
            return await call_next(request)


# Utility functions for FastAPI dependencies


async def get_rate_limit_status(request: Request) -> Optional[RateLimitResult]:
    """FastAPI dependency to get current rate limit status"""
    return getattr(request.state, "rate_limit_result", None)


def create_rate_limit_dependency(rate_limit_manager: RateLimitManager):
    """Create rate limit dependency for specific endpoints"""

    async def rate_limit_dependency(request: Request):
        tenant_id = getattr(request.state, "tenant_id", None)
        user_id = getattr(request.state, "user_id", None)
        user_role = getattr(request.state, "user_role", None)

        try:
            await rate_limit_manager.check_all_limits(
                request, tenant_id, user_id, user_role
            )
        except RateLimitExceeded as e:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail={
                    "error": "rate_limit_exceeded",
                    "message": e.message,
                    "retry_after": e.result.retry_after,
                },
            )

    return rate_limit_dependency
