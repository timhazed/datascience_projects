from collections.abc import AsyncGenerator, Awaitable, Callable
from contextlib import asynccontextmanager
from typing import Any

import structlog
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from strawberry.fastapi import GraphQLRouter

from app.api.routes.districts import router as districts_router
from app.api.routes.localities import router as localities_router
from app.api.routes.monitoring import router as monitoring_router
from app.api.routes.municipalities import router as municipalities_router
from app.api.routes.search import router as search_router
from app.core import (
    DistrictNotFoundError,
    InvalidSearchParametersError,
    LocalityNotFoundError,
    MunicipalityNotFoundError,
    configure_logging,
    get_logger,
    settings,
)
from app.graphql.schema import schema
from app.services import cache_service


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan manager"""
    import time

    configure_logging()
    logger: Any = get_logger(__name__)

    # Record startup time
    app.state.start_time = time.time()
    logger.info("Starting GeoPortugal API", version=settings.api_version)
    # Connect shared services
    await cache_service.connect()

    yield

    # Disconnect shared services
    await cache_service.disconnect()
    logger.info("Shutting down GeoPortugal API")


app = FastAPI(
    title=settings.api_title,
    description=settings.api_description,
    version=settings.api_version,
    debug=settings.debug,
    lifespan=lifespan,
    docs_url="/docs" if settings.debug else None,
    redoc_url="/redoc" if settings.debug else None,
    openapi_tags=[
        {
            "name": "districts",
            "description": "Operations with Portuguese districts. Districts are the highest level administrative divisions.",
            "externalDocs": {
                "description": "Portugal administrative divisions",
                "url": "https://en.wikipedia.org/wiki/Districts_of_Portugal",
            },
        },
        {
            "name": "municipalities",
            "description": "Operations with Portuguese municipalities. Municipalities are subdivisions of districts.",
        },
        {
            "name": "localities",
            "description": "Operations with Portuguese localities. Localities are populated places within municipalities.",
        },
        {
            "name": "search",
            "description": "Search and geospatial operations for finding locations by name or coordinates.",
        },
        {
            "name": "health",
            "description": "Health check and monitoring endpoints.",
        },
    ],
    contact={
        "name": "GeoPortugal API Support",
        "url": "https://github.com/geoportugal/api",
        "email": "support@geoportugal.com",
    },
    license_info={
        "name": "MIT License",
        "url": "https://opensource.org/licenses/MIT",
    },
)

# CORS middleware - more permissive for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if settings.debug else settings.cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


# Security headers middleware
@app.middleware("http")
async def add_security_headers(
    request: Request, call_next: Callable[[Request], Awaitable[Response]]
) -> Response:
    """Add security headers to all responses"""
    response = await call_next(request)

    # Security headers
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    response.headers["Permissions-Policy"] = "geolocation=(), microphone=(), camera=()"

    # Content Security Policy
    if not settings.debug:
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; "
            "script-src 'self' 'unsafe-inline' 'unsafe-eval'; "
            "style-src 'self' 'unsafe-inline'; "
            "img-src 'self' data: https:; "
            "font-src 'self'; "
            "connect-src 'self'; "
            "frame-ancestors 'none';"
        )

    return response


# Rate limiting middleware
rate_limit_storage: dict[str, list[float]] = (
    {}
)  # In production, use Redis or a proper rate limiter


@app.middleware("http")
async def rate_limit_middleware(
    request: Request, call_next: Callable[[Request], Awaitable[Response]]
) -> Response:
    """Simple rate limiting middleware"""
    import time

    # Skip rate limiting for health checks and metrics
    if request.url.path in ["/health", "/ready", "/live", "/metrics"]:
        return await call_next(request)

    client_ip = request.client.host if request.client else "unknown"
    current_time = time.time()
    window_size = 60  # 1 minute window
    max_requests = getattr(settings, "rate_limit_requests_per_minute", 100)

    # Clean old entries
    cutoff_time = current_time - window_size
    rate_limit_storage[client_ip] = [
        timestamp
        for timestamp in rate_limit_storage.get(client_ip, [])
        if timestamp > cutoff_time
    ]

    # Check rate limit
    if len(rate_limit_storage.get(client_ip, [])) >= max_requests:
        return JSONResponse(
            status_code=429,
            content={
                "detail": "Rate limit exceeded. Please try again later.",
                "retry_after": window_size,
            },
            headers={"Retry-After": str(window_size)},
        )

    # Record this request
    if client_ip not in rate_limit_storage:
        rate_limit_storage[client_ip] = []
    rate_limit_storage[client_ip].append(current_time)

    response = await call_next(request)

    # Add rate limit headers
    remaining = max(0, max_requests - len(rate_limit_storage[client_ip]))
    response.headers["X-RateLimit-Limit"] = str(max_requests)
    response.headers["X-RateLimit-Remaining"] = str(remaining)
    response.headers["X-RateLimit-Reset"] = str(int(current_time + window_size))

    return response


# Exception handlers
@app.exception_handler(DistrictNotFoundError)
async def district_not_found_handler(
    request: Request, exc: DistrictNotFoundError
) -> JSONResponse:
    return JSONResponse(
        status_code=404, content={"detail": f"District not found: {str(exc)}"}
    )


@app.exception_handler(MunicipalityNotFoundError)
async def municipality_not_found_handler(
    request: Request, exc: MunicipalityNotFoundError
) -> JSONResponse:
    return JSONResponse(
        status_code=404, content={"detail": f"Municipality not found: {str(exc)}"}
    )


@app.exception_handler(LocalityNotFoundError)
async def locality_not_found_handler(
    request: Request, exc: LocalityNotFoundError
) -> JSONResponse:
    return JSONResponse(
        status_code=404, content={"detail": f"Locality not found: {str(exc)}"}
    )


@app.exception_handler(InvalidSearchParametersError)
async def invalid_search_parameters_handler(
    request: Request, exc: InvalidSearchParametersError
) -> JSONResponse:
    return JSONResponse(
        status_code=400, content={"detail": f"Invalid search parameters: {str(exc)}"}
    )


# Request logging middleware
@app.middleware("http")
async def log_requests(
    request: Request, call_next: Callable[[Request], Awaitable[Response]]
) -> Response:
    logger: Any = get_logger(__name__)

    # Start request logging
    structlog.contextvars.bind_contextvars(
        request_id=id(request),
        method=request.method,
        url=str(request.url),
        user_agent=request.headers.get("user-agent"),
    )

    logger.info("Request started")

    response = await call_next(request)

    logger.info(
        "Request completed",
        status_code=response.status_code,
    )

    structlog.contextvars.clear_contextvars()
    return response


# Health check
@app.get(
    "/health",
    tags=["health"],
    summary="Health check endpoint",
    description="Returns the current health status of the API and its version.",
    responses={
        200: {
            "description": "API is healthy and operational",
            "content": {
                "application/json": {
                    "example": {
                        "status": "healthy",
                        "version": "1.0.0",
                        "timestamp": "2024-01-01T12:00:00Z",
                        "uptime": 3600,
                    }
                }
            },
        }
    },
)
async def health_check() -> dict[str, Any]:
    """
    Health check endpoint for monitoring and load balancers.

    Returns basic information about the API status including:
    - Service health status
    - API version
    - Current timestamp
    - Uptime in seconds
    """
    import time

    return {
        "status": "healthy",
        "version": settings.api_version,
        "timestamp": time.time(),
        "uptime": (
            time.time() - app.state.start_time
            if hasattr(app.state, "start_time")
            else 0
        ),
    }


# Include routers
app.include_router(districts_router, prefix="/api/v1", tags=["districts"])
app.include_router(municipalities_router, prefix="/api/v1", tags=["municipalities"])
app.include_router(localities_router, prefix="/api/v1", tags=["localities"])
app.include_router(search_router, prefix="/api/v1", tags=["search"])
app.include_router(monitoring_router, prefix="", tags=["monitoring"])

# GraphQL endpoint
graphql_app = GraphQLRouter(schema)
app.include_router(graphql_app, prefix="/graphql")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.debug,
        log_config=None,  # Use our custom logging
    )
