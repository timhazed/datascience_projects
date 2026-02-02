import pytest
from httpx import AsyncClient


class TestMunicipalitiesAPI:
    @pytest.mark.asyncio
    async def test_get_municipality(self, client: AsyncClient, sample_data):
        municipality_id = sample_data["municipality"].id

        response = await client.get(f"/api/v1/municipalities/{municipality_id}")

        assert response.status_code == 200
        data = response.json()
        assert data["id"] == municipality_id
        assert data["name"] == "Lisboa"
        assert data["district_id"] == sample_data["district"].id
        assert data["population"] == 547631

    @pytest.mark.asyncio
    async def test_get_municipality_not_found(self, client: AsyncClient):
        response = await client.get("/api/v1/municipalities/9999")

        assert response.status_code == 404
        assert "detail" in response.json()

    @pytest.mark.asyncio
    async def test_list_municipality_localities(self, client: AsyncClient, sample_data):
        municipality_id = sample_data["municipality"].id

        response = await client.get(
            f"/api/v1/municipalities/{municipality_id}/localities"
        )

        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) == len(sample_data["localities"])
        names = {loc["name"] for loc in data}
        assert names.issuperset({"Alfama", "Belém", "Chiado"})

    @pytest.mark.asyncio
    async def test_list_municipality_localities_pagination(
        self, client: AsyncClient, sample_data
    ):
        municipality_id = sample_data["municipality"].id
        response = await client.get(
            f"/api/v1/municipalities/{municipality_id}/localities?limit=1&offset=1"
        )

        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) == 1

    @pytest.mark.asyncio
    async def test_list_municipality_localities_not_found(
        self, client: AsyncClient
    ):
        response = await client.get("/api/v1/municipalities/999/localities")
        assert response.status_code == 404
        assert "detail" in response.json()
