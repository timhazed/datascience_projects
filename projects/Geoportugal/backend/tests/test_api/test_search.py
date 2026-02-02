import pytest
from httpx import AsyncClient


class TestSearchAPI:
    """Test cases for search API endpoints."""

    @pytest.mark.asyncio
    async def test_search_locations(self, client: AsyncClient, sample_data):
        """Test GET /api/v1/search"""
        response = await client.get("/api/v1/search?q=Lisboa")

        assert response.status_code == 200
        data = response.json()

        # Check response structure
        assert "districts" in data
        assert "municipalities" in data
        assert "localities" in data

        # Should find Lisboa district and municipality
        assert len(data["districts"]) == 1
        assert len(data["municipalities"]) == 1
        assert data["districts"][0]["name"] == "Lisboa"
        assert data["municipalities"][0]["name"] == "Lisboa"

    @pytest.mark.asyncio
    async def test_search_partial_match(self, client: AsyncClient, sample_data):
        """Test search with partial term"""
        response = await client.get("/api/v1/search?q=Lis")

        assert response.status_code == 200
        data = response.json()

        # Should still find Lisboa
        assert len(data["districts"]) == 1
        assert len(data["municipalities"]) == 1

    @pytest.mark.asyncio
    async def test_search_localities(self, client: AsyncClient, sample_data):
        """Test search for localities"""
        response = await client.get("/api/v1/search?q=Alfama")

        assert response.status_code == 200
        data = response.json()

        assert len(data["localities"]) == 1
        assert data["localities"][0]["name"] == "Alfama"

    @pytest.mark.asyncio
    async def test_search_case_insensitive(self, client: AsyncClient, sample_data):
        """Test case insensitive search"""
        response = await client.get("/api/v1/search?q=lisboa")

        assert response.status_code == 200
        data = response.json()

        assert len(data["districts"]) == 1
        assert len(data["municipalities"]) == 1

    @pytest.mark.asyncio
    async def test_search_no_results(self, client: AsyncClient, sample_data):
        """Test search with no matching results"""
        response = await client.get("/api/v1/search?q=NonExistent")

        assert response.status_code == 200
        data = response.json()

        assert len(data["districts"]) == 0
        assert len(data["municipalities"]) == 0
        assert len(data["localities"]) == 0

    @pytest.mark.asyncio
    async def test_search_with_pagination(self, client: AsyncClient, sample_data):
        """Test search with pagination parameters"""
        response = await client.get("/api/v1/search?q=Lisboa&limit=10&offset=0")

        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, dict)

    @pytest.mark.asyncio
    async def test_search_minimum_length(self, client: AsyncClient, sample_data):
        """Test search term minimum length validation"""
        response = await client.get("/api/v1/search?q=L")

        assert response.status_code == 422  # Validation error

    @pytest.mark.asyncio
    async def test_nearby_search(self, client: AsyncClient, sample_data):
        """Test GET /api/v1/nearby"""
        # Search near Alfama coordinates
        response = await client.get(
            "/api/v1/nearby?lat=38.7131&lng=-9.1301&radius_km=5"
        )

        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)

        # Should find localities within radius
        assert len(data) >= 1
        # Alfama should be first (closest to itself)
        assert data[0]["name"] == "Alfama"

    @pytest.mark.asyncio
    async def test_nearby_search_with_radius(self, client: AsyncClient, sample_data):
        """Test nearby search with different radius"""
        # Very small radius - should find fewer results
        response = await client.get(
            "/api/v1/nearby?lat=38.7131&lng=-9.1301&radius_km=0.5"
        )

        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)

    @pytest.mark.asyncio
    async def test_nearby_search_invalid_coordinates(
        self, client: AsyncClient, sample_data
    ):
        """Test nearby search with invalid coordinates"""
        # Invalid latitude
        response = await client.get("/api/v1/nearby?lat=91&lng=-9.1301")
        assert response.status_code == 422

        # Invalid longitude
        response = await client.get("/api/v1/nearby?lat=38.7131&lng=181")
        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_nearby_search_no_results(self, client: AsyncClient, sample_data):
        """Test nearby search with coordinates far from any localities"""
        # Coordinates in the ocean
        response = await client.get("/api/v1/nearby?lat=0&lng=0&radius_km=1")

        assert response.status_code == 200
        data = response.json()
        assert len(data) == 0

    @pytest.mark.asyncio
    async def test_nearby_search_with_limit(self, client: AsyncClient, sample_data):
        """Test nearby search with limit parameter"""
        response = await client.get("/api/v1/nearby?lat=38.7131&lng=-9.1301&limit=2")

        assert response.status_code == 200
        data = response.json()
        assert len(data) <= 2
