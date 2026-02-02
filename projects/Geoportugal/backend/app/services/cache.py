import json
from hashlib import md5
from typing import Any, cast

import redis.asyncio as redis
from redis.asyncio import Redis

from app.core.config import settings
from app.core.logging import get_logger

logger: Any = get_logger(__name__)


class CacheService:
    def __init__(self) -> None:
        self.redis: Redis | None = None
        self.ttl: int = settings.cache_ttl

    async def connect(self) -> None:
        """Connect to Redis"""
        try:
            self.redis = cast(
                Redis,
                redis.from_url(  # type: ignore[no-untyped-call]
                    settings.redis_url,
                    encoding="utf-8",
                    decode_responses=True,
                ),
            )
            _ = await cast(Any, self.redis.ping())
            logger.info("Connected to Redis", url=settings.redis_url)
        except Exception as e:
            logger.warning("Failed to connect to Redis, caching disabled", error=str(e))
            self.redis = None

    async def disconnect(self) -> None:
        """Disconnect from Redis"""
        if self.redis:
            await self.redis.close()
            logger.info("Disconnected from Redis")

    def _generate_key(self, prefix: str, data: Any) -> str:
        """Generate cache key from prefix and data"""
        data_str = json.dumps(data, sort_keys=True, default=str)
        data_hash = md5(data_str.encode()).hexdigest()
        return f"{prefix}:{data_hash}"

    async def get(self, key: str) -> Any | None:
        """Get value from cache"""
        result: Any | None = None

        if not self.redis:
            logger.debug("Cache not available", key=key)
        else:
            try:
                value = await self.redis.get(key)
                if value:
                    logger.debug("Cache hit", key=key)
                    result = json.loads(value)
                else:
                    logger.debug("Cache miss", key=key)
            except Exception as e:
                logger.warning("Cache get error", key=key, error=str(e))

        return result

    async def set(self, key: str, value: Any, ttl: int | None = None) -> bool:
        """Set value in cache"""
        if not self.redis:
            logger.debug("Cache not available for set", key=key)
            return False
        try:
            ttl_value = ttl or self.ttl
            serialized_value = json.dumps(value, default=str)
            await self.redis.setex(key, ttl_value, serialized_value)
            logger.debug("Cache set", key=key, ttl=ttl_value)
            return True
        except Exception as e:
            logger.warning("Cache set error", key=key, error=str(e))
            return False

    async def delete(self, key: str) -> bool:
        """Delete key from cache"""
        if not self.redis:
            logger.debug("Cache not available for delete", key=key)
            return False
        try:
            result = await self.redis.delete(key)
            success = bool(result)
            logger.debug("Cache delete", key=key, deleted=success)
            return success
        except Exception as e:
            logger.warning("Cache delete error", key=key, error=str(e))
            return False

    async def clear_pattern(self, pattern: str) -> int:
        """Clear all keys matching pattern"""
        if not self.redis:
            logger.debug("Cache not available for pattern clear", pattern=pattern)
            return 0
        try:
            keys = await self.redis.keys(pattern)
            if keys:
                count = await self.redis.delete(*keys)
                logger.info("Cache pattern cleared", pattern=pattern, count=count)
                return int(count)
        except Exception as e:
            logger.warning("Cache clear pattern error", pattern=pattern, error=str(e))
        return 0


# Global cache instance
cache_service = CacheService()
