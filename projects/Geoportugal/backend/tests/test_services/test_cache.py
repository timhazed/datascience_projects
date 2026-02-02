import json
from unittest.mock import AsyncMock, patch

import pytest

from app.services.cache import CacheService


class TestCacheService:
    """Test cache service functionality."""

    @pytest.fixture
    def mock_redis(self):
        """Create mock Redis client."""
        redis_mock = AsyncMock()
        redis_mock.ping = AsyncMock()
        redis_mock.get = AsyncMock()
        redis_mock.setex = AsyncMock()
        redis_mock.delete = AsyncMock()
        redis_mock.close = AsyncMock()
        return redis_mock

    @pytest.fixture
    def cache_service(self):
        """Create CacheService instance."""
        return CacheService()

    @pytest.mark.asyncio
    async def test_connect_success(self, cache_service, mock_redis):
        """Test successful Redis connection."""
        with patch("app.services.cache.redis.from_url", return_value=mock_redis):
            with patch(
                "app.services.cache.settings.redis_url", "redis://localhost:6379"
            ):
                await cache_service.connect()

                mock_redis.ping.assert_called_once()
                assert cache_service.redis == mock_redis

    @pytest.mark.asyncio
    async def test_connect_failure(self, cache_service):
        """Test Redis connection failure."""
        with patch(
            "app.services.cache.redis.from_url",
            side_effect=Exception("Connection failed"),
        ):
            with patch(
                "app.services.cache.settings.redis_url", "redis://localhost:6379"
            ):
                await cache_service.connect()

                assert cache_service.redis is None

    @pytest.mark.asyncio
    async def test_disconnect_with_redis(self, cache_service, mock_redis):
        """Test disconnection when Redis is connected."""
        cache_service.redis = mock_redis

        await cache_service.disconnect()

        mock_redis.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_disconnect_without_redis(self, cache_service):
        """Test disconnection when Redis is not connected."""
        cache_service.redis = None

        await cache_service.disconnect()  # Should not raise exception

    def test_generate_key(self, cache_service):
        """Test cache key generation."""
        prefix = "test"
        data = {"name": "Lisboa", "id": 1}

        key = cache_service._generate_key(prefix, data)

        assert key.startswith("test:")
        assert len(key) > len(prefix) + 1

    def test_generate_key_consistent(self, cache_service):
        """Test that same data generates same key."""
        prefix = "test"
        data = {"name": "Lisboa", "id": 1}

        key1 = cache_service._generate_key(prefix, data)
        key2 = cache_service._generate_key(prefix, data)

        assert key1 == key2

    @pytest.mark.asyncio
    async def test_get_cache_hit(self, cache_service, mock_redis):
        """Test cache hit scenario."""
        cache_service.redis = mock_redis
        test_data = {"name": "Lisboa"}
        mock_redis.get.return_value = json.dumps(test_data)

        result = await cache_service.get("test:key")

        assert result == test_data
        mock_redis.get.assert_called_once_with("test:key")

    @pytest.mark.asyncio
    async def test_get_cache_miss(self, cache_service, mock_redis):
        """Test cache miss scenario."""
        cache_service.redis = mock_redis
        mock_redis.get.return_value = None

        result = await cache_service.get("test:key")

        assert result is None
        mock_redis.get.assert_called_once_with("test:key")

    @pytest.mark.asyncio
    async def test_get_no_redis(self, cache_service):
        """Test get when Redis is not available."""
        cache_service.redis = None

        result = await cache_service.get("test:key")

        assert result is None

    @pytest.mark.asyncio
    async def test_get_redis_exception(self, cache_service, mock_redis):
        """Test get when Redis raises exception."""
        cache_service.redis = mock_redis
        mock_redis.get.side_effect = Exception("Redis error")

        result = await cache_service.get("test:key")

        assert result is None

    @pytest.mark.asyncio
    async def test_set_success(self, cache_service, mock_redis):
        """Test successful cache set."""
        cache_service.redis = mock_redis
        mock_redis.setex.return_value = True
        test_data = {"name": "Lisboa"}

        result = await cache_service.set("test:key", test_data, ttl=3600)

        assert result is True
        mock_redis.setex.assert_called_once_with(
            "test:key", 3600, json.dumps(test_data, default=str)
        )

    @pytest.mark.asyncio
    async def test_set_no_redis(self, cache_service):
        """Test set when Redis is not available."""
        cache_service.redis = None

        result = await cache_service.set("test:key", {"data": "test"})

        assert result is False

    @pytest.mark.asyncio
    async def test_set_redis_exception(self, cache_service, mock_redis):
        """Test set when Redis raises exception."""
        cache_service.redis = mock_redis
        mock_redis.setex.side_effect = Exception("Redis error")

        result = await cache_service.set("test:key", {"data": "test"})

        assert result is False

    @pytest.mark.asyncio
    async def test_set_default_ttl(self, cache_service, mock_redis):
        """Test set with default TTL."""
        cache_service.redis = mock_redis
        mock_redis.setex.return_value = True

        await cache_service.set("test:key", {"data": "test"})

        # Should use default TTL of 300 seconds
        mock_redis.setex.assert_called_once_with(
            "test:key", 300, json.dumps({"data": "test"}, default=str)
        )

    @pytest.mark.asyncio
    async def test_delete_success(self, cache_service, mock_redis):
        """Test successful cache delete."""
        cache_service.redis = mock_redis
        mock_redis.delete.return_value = 1  # Redis returns count of deleted keys

        result = await cache_service.delete("test:key")

        assert result is True
        mock_redis.delete.assert_called_once_with("test:key")

    @pytest.mark.asyncio
    async def test_delete_not_found(self, cache_service, mock_redis):
        """Test delete when key doesn't exist."""
        cache_service.redis = mock_redis
        mock_redis.delete.return_value = 0  # Redis returns 0 when key doesn't exist

        result = await cache_service.delete("test:key")

        assert result is False
        mock_redis.delete.assert_called_once_with("test:key")

    @pytest.mark.asyncio
    async def test_delete_no_redis(self, cache_service):
        """Test delete when Redis is not available."""
        cache_service.redis = None

        result = await cache_service.delete("test:key")

        assert result is False

    @pytest.mark.asyncio
    async def test_delete_redis_exception(self, cache_service, mock_redis):
        """Test delete when Redis raises exception."""
        cache_service.redis = mock_redis
        mock_redis.delete.side_effect = Exception("Redis error")

        result = await cache_service.delete("test:key")

        assert result is False
