import os
from pathlib import Path

from pydantic import Field, validator
from pydantic_settings import BaseSettings, SettingsConfigDict


def get_env_file_path() -> str:
    """
    Determine the .env file path with fallback logic:
    1. First check root directory: ./.env
    2. If not found, use: ~/src/python/geoportugal/.env
    """
    # Get the project root directory (assuming this config.py is in backend/app/core/)
    current_file = Path(__file__)
    project_root = (
        current_file.parent.parent.parent.parent
    )  # Go up 4 levels to project root
    root_env_file = project_root / ".env"

    # Option 1: Check root directory first
    if root_env_file.exists():
        return str(root_env_file)

    # Option 2: Fallback to the original location
    fallback_env_file = Path.home() / "src" / "python" / "geoportugal" / ".env"
    return str(fallback_env_file)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=get_env_file_path(),
        env_file_encoding="utf-8",
        case_sensitive=False,
        validate_default=True,
        extra="ignore",
    )

    # API Settings
    api_title: str = "GeoPortugal API"
    api_description: str = (
        "Production-ready geospatial API serving hierarchical Portugal location data"
    )
    api_version: str = "1.0.0"
    debug: bool = False

    # Database Settings
    database_url: str = Field(
        default="",
        description="Database URL - REQUIRED in production",
    )
    database_echo: bool = Field(default=False, description="Enable SQL query logging")
    max_connections: int = Field(
        default=20, description="Database connection pool size"
    )
    max_overflow: int = Field(
        default=10, description="Database connection pool overflow"
    )

    # Redis Settings
    redis_url: str = Field(
        default="",
        description="Redis URL for caching - REQUIRED",
    )
    cache_ttl: int = Field(default=300, description="Default cache TTL in seconds")
    cache_ttl_districts: int = Field(default=3600, description="Districts cache TTL")
    cache_ttl_search: int = Field(default=900, description="Search results cache TTL")

    # Logging
    log_level: str = Field(default="INFO", description="Log level")
    log_format: str = Field(default="json", description="Log format")

    # Security
    secret_key: str = Field(
        default="",
        description="Secret key for JWT and security - REQUIRED in production",
    )
    jwt_algorithm: str = Field(default="HS256", description="JWT algorithm")
    jwt_access_token_expire_minutes: int = Field(
        default=30, description="JWT token expiry"
    )

    # CORS
    cors_origins: list[str] = Field(
        default=[
            "http://localhost:3000",
            "http://localhost:3001",
            "http://localhost:8080",
        ],
        description="CORS origins - MUST be configured for production",
    )
    cors_allow_credentials: bool = Field(
        default=True, description="Allow CORS credentials"
    )

    # Rate Limiting
    rate_limit_requests_per_minute: int = Field(
        default=100, description="Rate limit per minute"
    )
    rate_limit_burst: int = Field(default=20, description="Rate limit burst size")

    # Monitoring
    enable_metrics: bool = Field(default=True, description="Enable Prometheus metrics")
    metrics_username: str | None = Field(
        default="admin", description="Metrics endpoint username"
    )
    metrics_password: str | None = Field(
        default="admin", description="Metrics endpoint password"
    )

    @validator("database_url")
    def validate_database_url(cls, v: str) -> str:
        if not v:
            # In development, provide a reasonable default
            if os.getenv("DEBUG", "false").lower() == "true":
                return (
                    "postgresql+asyncpg://postgres:postgres@localhost:5432/geoportugal"
                )
            raise ValueError("DATABASE_URL is required in production")
        return v

    @validator("redis_url")
    def validate_redis_url(cls, v: str) -> str:
        if not v:
            # In development, provide a reasonable default
            if os.getenv("DEBUG", "false").lower() == "true":
                return "redis://localhost:6379/0"
            raise ValueError("REDIS_URL is required")
        return v

    @validator("secret_key")
    def validate_secret_key(cls, v: str) -> str:
        if not v:
            # In development, provide a reasonable default
            if os.getenv("DEBUG", "false").lower() == "true":
                return "dev-secret-key-change-in-production"
            raise ValueError("SECRET_KEY is required in production")
        if len(v) < 32:
            raise ValueError("SECRET_KEY must be at least 32 characters long")
        return v

    @validator("cors_origins")
    def validate_cors_origins(cls, v: list[str]) -> list[str]:
        # In production, warn if using localhost origins
        if not os.getenv("DEBUG", "false").lower() == "true":
            localhost_origins = [origin for origin in v if "localhost" in origin]
            if localhost_origins:
                import warnings

                warnings.warn(
                    f"Production CORS includes localhost origins: {localhost_origins}",
                    stacklevel=2,
                )
        return v

    @validator("log_level")
    def validate_log_level(cls, v: str) -> str:
        valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if v.upper() not in valid_levels:
            raise ValueError(f"Log level must be one of {valid_levels}")
        return v.upper()

    @property
    def is_production(self) -> bool:
        """Check if running in production mode"""
        return (
            not self.debug
            and os.getenv("ENVIRONMENT", "development").lower() == "production"
        )


settings = Settings()
