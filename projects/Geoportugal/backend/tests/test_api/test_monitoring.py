import pytest
from httpx import AsyncClient, BasicAuth

from app.services import cache_service
from app.core.config import settings


@pytest.mark.asyncio
async def test_metrics_requires_auth(client: AsyncClient):
    response = await client.get("/metrics")
    assert response.status_code == 401


@pytest.mark.asyncio
async def test_metrics_with_valid_auth(client: AsyncClient):
    auth = BasicAuth(
        settings.metrics_username or "admin",
        (settings.metrics_password or "admin"),
    )
    response = await client.get("/metrics", auth=auth)
    assert response.status_code == 200
    body = response.text
    assert "geoportugal_info" in body


@pytest.mark.asyncio
async def test_detailed_health_ok(client: AsyncClient):
    response = await client.get("/health/detailed")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "checks" in data


@pytest.mark.asyncio
async def test_readiness_probe_cache_failure(client: AsyncClient):
    response = await client.get("/ready")
    assert response.status_code == 503
    payload = response.json()
    assert payload["detail"]["status"] == "not_ready"


@pytest.mark.asyncio
async def test_readiness_probe_success(monkeypatch, client: AsyncClient):
    async def fake_set(key, value, ttl=None):
        return True

    async def fake_get(key):
        return "ok"

    async def fake_delete(key):
        return 1

    monkeypatch.setattr(cache_service, "set", fake_set)
    monkeypatch.setattr(cache_service, "get", fake_get)
    monkeypatch.setattr(cache_service, "delete", fake_delete)

    response = await client.get("/ready")
    assert response.status_code == 200
    assert response.json()["status"] == "ready"


@pytest.mark.asyncio
async def test_liveness_probe(client: AsyncClient):
    response = await client.get("/live")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "alive"
