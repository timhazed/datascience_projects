import pytest
from httpx import AsyncClient


class TestDistrictsAPI:
    """Test cases for districts API endpoints."""

    @pytest.mark.asyncio
    async def test_list_districts(self, client: AsyncClient, sample_data):
        """Test GET /api/v1/districts"""
        response = await client.get("/api/v1/districts")

        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) == 1

        district = data[0]
        assert district["name"] == "Lisboa"
        assert district["code"] == "11"
        assert district["population"] == 2821697
        assert "id" in district
        assert "created_at" in district

    @pytest.mark.asyncio
    async def test_list_districts_with_pagination(
        self, client: AsyncClient, sample_data
    ):
        """Test GET /api/v1/districts with pagination parameters"""
        response = await client.get("/api/v1/districts?limit=10&offset=0")

        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)

    @pytest.mark.asyncio
    async def test_get_district_by_id(self, client: AsyncClient, sample_data):
        """Test GET /api/v1/districts/{id}"""
        district_id = sample_data["district"].id

        response = await client.get(f"/api/v1/districts/{district_id}")

        assert response.status_code == 200
        data = response.json()
        assert data["id"] == district_id
        assert data["name"] == "Lisboa"
        assert data["code"] == "11"

    @pytest.mark.asyncio
    async def test_get_district_not_found(self, client: AsyncClient, sample_data):
        """Test GET /api/v1/districts/{id} with non-existent ID"""
        response = await client.get("/api/v1/districts/999")

        assert response.status_code == 404
        data = response.json()
        assert "detail" in data
        assert "not found" in data["detail"].lower()

    @pytest.mark.asyncio
    async def test_get_district_municipalities(self, client: AsyncClient, sample_data):
        """Test GET /api/v1/districts/{id}/municipalities"""
        district_id = sample_data["district"].id

        response = await client.get(f"/api/v1/districts/{district_id}/municipalities")

        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) == 1

        municipality = data[0]
        assert municipality["name"] == "Lisboa"
        assert municipality["district_id"] == district_id
        assert municipality["population"] == 547631

    @pytest.mark.asyncio
    async def test_get_district_municipalities_with_pagination(
        self, client: AsyncClient, sample_data
    ):
        """Test GET /api/v1/districts/{id}/municipalities with pagination"""
        district_id = sample_data["district"].id

        response = await client.get(
            f"/api/v1/districts/{district_id}/municipalities?limit=5&offset=0"
        )

        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)

    @pytest.mark.asyncio
    async def test_get_district_municipalities_not_found(
        self, client: AsyncClient, sample_data
    ):
        """Test GET /api/v1/districts/{id}/municipalities with non-existent district"""
        response = await client.get("/api/v1/districts/999/municipalities")

        assert response.status_code == 404
