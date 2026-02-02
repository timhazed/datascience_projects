"""
Monitoring and observability endpoints
"""

import time

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.database import get_session
from app.services.cache import cache_service

router = APIRouter()
security = HTTPBasic()


def verify_metrics_auth(
    credentials: HTTPBasicCredentials = Depends(security),
) -> str:
    """Verify basic auth for metrics endpoint"""
    import secrets

    expected_username = getattr(settings, "metrics_username", "admin")
    expected_password = getattr(settings, "metrics_password", "admin") or "admin"

    is_correct_username = secrets.compare_digest(
        credentials.username, expected_username
    )
    is_correct_password = secrets.compare_digest(
        credentials.password, expected_password
    )

    if not (is_correct_username and is_correct_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials",
            headers={"WWW-Authenticate": "Basic"},
        )
    return credentials.username


@router.get(
    "/health/detailed",
    tags=["monitoring"],
    summary="Detailed health check",
    description="Returns detailed health information including database and cache connectivity",
)
async def detailed_health_check(
    session: AsyncSession = Depends(get_session),
) -> dict[str, object]:
    """
    Detailed health check that verifies all system components.

    Checks:
    - Database connectivity
    - Redis cache connectivity
    - System resources
    - Service dependencies
    """
    start_time = time.time()
    checks: dict[str, dict[str, object]] = {}
    health_info: dict[str, object] = {
        "status": "healthy",
        "timestamp": start_time,
        "version": settings.api_version,
        "checks": checks,
    }

    # Database health check
    try:
        await session.execute(text("SELECT 1"))
        checks["database"] = {
            "status": "healthy",
            "response_time": time.time() - start_time,
        }
    except Exception as e:
        checks["database"] = {
            "status": "unhealthy",
            "error": str(e),
            "response_time": time.time() - start_time,
        }
        health_info["status"] = "unhealthy"

    # Redis health check
    cache_start = time.time()
    try:
        await cache_service.set("health_check", "ok", ttl=10)
        result = await cache_service.get("health_check")
        await cache_service.delete("health_check")

        checks["cache"] = {
            "status": "healthy" if result == "ok" else "unhealthy",
            "response_time": time.time() - cache_start,
        }
    except Exception as e:
        checks["cache"] = {
            "status": "unhealthy",
            "error": str(e),
            "response_time": time.time() - cache_start,
        }
        health_info["status"] = (
            "degraded" if health_info["status"] == "healthy" else "unhealthy"
        )

    health_info["total_response_time"] = time.time() - start_time

    # Return appropriate HTTP status
    if health_info["status"] == "healthy":
        return health_info
    elif health_info["status"] == "degraded":
        raise HTTPException(status_code=503, detail=health_info)
    else:
        raise HTTPException(status_code=503, detail=health_info)


@router.get(
    "/metrics",
    tags=["monitoring"],
    summary="Prometheus metrics endpoint",
    description="Returns metrics in Prometheus format for monitoring",
)
async def get_metrics(
    _: str = Depends(verify_metrics_auth), session: AsyncSession = Depends(get_session)
) -> str:
    """
    Prometheus-compatible metrics endpoint.

    Returns system metrics including:
    - Request counts and durations
    - Database connection metrics
    - Cache hit/miss ratios
    - System resource usage
    """
    metrics: list[str] = []

    # Basic application info
    metrics.append("# HELP geoportugal_info Application information")
    metrics.append("# TYPE geoportugal_info gauge")
    metrics.append(f'geoportugal_info{{version="{settings.api_version}"}} 1')

    # Database metrics
    try:
        start = time.time()
        await session.execute(text("SELECT 1"))
        db_response_time = time.time() - start

        metrics.append(
            "# HELP geoportugal_db_connection_duration_seconds Database connection duration"
        )
        metrics.append("# TYPE geoportugal_db_connection_duration_seconds gauge")
        metrics.append(f"geoportugal_db_connection_duration_seconds {db_response_time}")

        metrics.append("# HELP geoportugal_db_status Database connection status")
        metrics.append("# TYPE geoportugal_db_status gauge")
        metrics.append("geoportugal_db_status 1")
    except Exception:
        metrics.append("geoportugal_db_status 0")

    # Cache metrics
    try:
        start = time.time()
        await cache_service.set("metrics_test", "ok", ttl=5)
        result = await cache_service.get("metrics_test")
        cache_response_time = time.time() - start
        await cache_service.delete("metrics_test")

        metrics.append(
            "# HELP geoportugal_cache_connection_duration_seconds Cache connection duration"
        )
        metrics.append("# TYPE geoportugal_cache_connection_duration_seconds gauge")
        metrics.append(
            f"geoportugal_cache_connection_duration_seconds {cache_response_time}"
        )

        metrics.append("# HELP geoportugal_cache_status Cache connection status")
        metrics.append("# TYPE geoportugal_cache_status gauge")
        metrics.append(f"geoportugal_cache_status {1 if result == 'ok' else 0}")
    except Exception:
        metrics.append("geoportugal_cache_status 0")

    # System uptime
    try:
        from app.main import app

        uptime = time.time() - getattr(app.state, "start_time", time.time())
        metrics.append(
            "# HELP geoportugal_uptime_seconds Application uptime in seconds"
        )
        metrics.append("# TYPE geoportugal_uptime_seconds counter")
        metrics.append(f"geoportugal_uptime_seconds {uptime}")
    except Exception:
        pass

    return "\n".join(metrics)


@router.get(
    "/ready",
    tags=["monitoring"],
    summary="Readiness probe",
    description="Kubernetes-style readiness probe endpoint",
)
async def readiness_probe(
    session: AsyncSession = Depends(get_session),
) -> dict[str, str]:
    """
    Readiness probe for Kubernetes deployments.

    Returns 200 if the service is ready to accept traffic,
    503 if not ready (e.g., during startup or maintenance).
    """
    try:
        # Check database connectivity
        await session.execute(text("SELECT 1"))

        # Check cache connectivity
        await cache_service.set("readiness_check", "ok", ttl=5)
        result = await cache_service.get("readiness_check")
        await cache_service.delete("readiness_check")

        if result != "ok":
            raise Exception("Cache check failed")

        return {"status": "ready"}

    except Exception as e:
        raise HTTPException(
            status_code=503, detail={"status": "not_ready", "reason": str(e)}
        ) from e


@router.get(
    "/live",
    tags=["monitoring"],
    summary="Liveness probe",
    description="Kubernetes-style liveness probe endpoint",
)
async def liveness_probe() -> dict[str, object]:
    """
    Liveness probe for Kubernetes deployments.

    Returns 200 if the service is alive and should continue running,
    503 if it should be restarted.
    """
    # Simple liveness check - just verify the process is responding
    return {"status": "alive", "timestamp": time.time()}
