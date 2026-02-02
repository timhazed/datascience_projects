"""
Simple working GraphQL resolver tests to improve coverage
"""

from unittest.mock import AsyncMock, patch

import pytest

from app.db.models import District as DistrictModel
from app.graphql.schema import schema


class TestSimpleGraphQLResolvers:
    """Simple GraphQL tests that work to improve coverage"""

    @pytest.mark.asyncio
    async def test_districts_query_basic(self):
        """Test basic districts query execution"""
        query = """
        {
            districts {
                id
                name
            }
        }
        """

        # Mock the location service creation
        with patch(
            "app.graphql.resolvers.create_location_service"
        ) as mock_service_factory:
            mock_service = AsyncMock()
            mock_districts = [
                DistrictModel(id=1, name="Lisboa", code="11", population=2250000),
            ]
            mock_service.get_districts.return_value = mock_districts
            mock_service_factory.return_value = mock_service

            # Execute the query
            result = await schema.execute(query)

            # Verify no errors and basic structure
            assert result.errors is None
            assert "districts" in result.data
            assert len(result.data["districts"]) == 1
            assert result.data["districts"][0]["name"] == "Lisboa"

    @pytest.mark.asyncio
    async def test_district_by_id_basic(self):
        """Test district by ID query"""
        query = """
        query GetDistrict($id: Int!) {
            district(id: $id) {
                id
                name
            }
        }
        """

        with patch(
            "app.graphql.resolvers.create_location_service"
        ) as mock_service_factory:
            mock_service = AsyncMock()
            mock_district = DistrictModel(
                id=1, name="Lisboa", code="11", population=2250000
            )
            mock_service.get_district.return_value = mock_district
            mock_service_factory.return_value = mock_service

            result = await schema.execute(query, variable_values={"id": 1})

            assert result.errors is None
            assert result.data["district"]["name"] == "Lisboa"

    @pytest.mark.asyncio
    async def test_search_locations_basic(self):
        """Test basic search locations query"""
        query = """
        query SearchLocations($query: String!) {
            searchLocations(query: $query) {
                id
                name
            }
        }
        """

        with patch(
            "app.graphql.resolvers.create_location_service"
        ) as mock_service_factory:
            mock_service = AsyncMock()
            # Return empty results structure
            mock_service.search_locations.return_value = {"localities": []}
            mock_service_factory.return_value = mock_service

            result = await schema.execute(query, variable_values={"query": "test"})

            assert result.errors is None
            assert "searchLocations" in result.data
            assert result.data["searchLocations"] == []

    @pytest.mark.asyncio
    async def test_municipalities_query_basic(self):
        """Test basic municipalities query"""
        query = """
        {
            municipalities {
                id
                name
            }
        }
        """

        with patch(
            "app.graphql.resolvers.create_location_service"
        ) as mock_service_factory:
            mock_service = AsyncMock()
            mock_service.search_locations.return_value = {"municipalities": []}
            mock_service_factory.return_value = mock_service

            result = await schema.execute(query)

            assert result.errors is None
            assert "municipalities" in result.data
            assert result.data["municipalities"] == []

    @pytest.mark.asyncio
    async def test_municipalities_with_district_filter(self):
        """Test municipalities query with district filter"""
        query = """
        query GetMunicipalities($districtId: Int) {
            municipalities(districtId: $districtId) {
                id
                name
                districtId
            }
        }
        """

        with patch(
            "app.graphql.resolvers.create_location_service"
        ) as mock_service_factory:
            mock_service = AsyncMock()
            from app.db.models import Municipality as MunicipalityModel

            mock_municipalities = [
                MunicipalityModel(
                    id=1, name="Sintra", district_id=11, population=377835, area=319.23
                ),
            ]
            mock_service.get_district_municipalities.return_value = mock_municipalities
            mock_service_factory.return_value = mock_service

            result = await schema.execute(query, variable_values={"districtId": 11})

            assert result.errors is None
            assert "municipalities" in result.data
            assert len(result.data["municipalities"]) == 1
            assert result.data["municipalities"][0]["name"] == "Sintra"
            assert result.data["municipalities"][0]["districtId"] == 11

    @pytest.mark.asyncio
    async def test_municipality_localities_query(self):
        """Test municipality localities query"""
        query = """
        query GetMunicipalityLocalities($municipalityId: Int!) {
            municipalityLocalities(municipalityId: $municipalityId) {
                id
                name
                municipalityId
            }
        }
        """

        with patch(
            "app.graphql.resolvers.create_location_service"
        ) as mock_service_factory:
            mock_service = AsyncMock()
            from app.db.models import Locality as LocalityModel

            mock_localities = [
                LocalityModel(
                    id=1,
                    name="Agualva",
                    latitude=38.7533,
                    longitude=-9.3122,
                    feature_type="Cidade",
                    population=81845,
                    municipality_id=1,
                ),
            ]
            mock_service.get_municipality_localities.return_value = mock_localities
            mock_service_factory.return_value = mock_service

            result = await schema.execute(query, variable_values={"municipalityId": 1})

            assert result.errors is None
            assert "municipalityLocalities" in result.data
            assert len(result.data["municipalityLocalities"]) == 1
            assert result.data["municipalityLocalities"][0]["name"] == "Agualva"
            assert result.data["municipalityLocalities"][0]["municipalityId"] == 1

    @pytest.mark.asyncio
    async def test_search_locations_with_results(self):
        """Test search locations with actual results"""
        query = """
        query SearchLocations($query: String!) {
            searchLocations(query: $query) {
                id
                name
                featureType
            }
        }
        """

        with patch(
            "app.graphql.resolvers.create_location_service"
        ) as mock_service_factory:
            mock_service = AsyncMock()
            from app.db.models import Locality as LocalityModel

            mock_localities = [
                LocalityModel(
                    id=1,
                    name="Lisboa",
                    latitude=38.7167,
                    longitude=-9.1333,
                    feature_type="Cidade",
                    population=547773,
                    municipality_id=1,
                ),
            ]
            mock_service.search_locations.return_value = {"localities": mock_localities}
            mock_service_factory.return_value = mock_service

            result = await schema.execute(query, variable_values={"query": "lisboa"})

            assert result.errors is None
            assert "searchLocations" in result.data
            assert len(result.data["searchLocations"]) == 1
            assert result.data["searchLocations"][0]["name"] == "Lisboa"
            assert result.data["searchLocations"][0]["featureType"] == "Cidade"

    @pytest.mark.asyncio
    async def test_localities_near_with_custom_radius(self):
        """Test localities near with custom radius"""
        query = """
        query GetNearbyLocalities($lat: Float!, $lng: Float!, $radiusKm: Float) {
            localitiesNear(lat: $lat, lng: $lng, radiusKm: $radiusKm) {
                id
                name
                latitude
                longitude
            }
        }
        """

        with patch(
            "app.graphql.resolvers.create_location_service"
        ) as mock_service_factory:
            mock_service = AsyncMock()
            from app.db.models import Locality as LocalityModel

            mock_localities = [
                LocalityModel(
                    id=1,
                    name="Cascais",
                    latitude=38.6979,
                    longitude=-9.4215,
                    feature_type="Vila",
                    population=35000,
                    municipality_id=2,
                ),
            ]
            mock_service.search_nearby.return_value = mock_localities
            mock_service_factory.return_value = mock_service

            result = await schema.execute(
                query, variable_values={"lat": 38.7, "lng": -9.1, "radiusKm": 25.0}
            )

            assert result.errors is None
            assert "localitiesNear" in result.data
            assert len(result.data["localitiesNear"]) == 1
            assert result.data["localitiesNear"][0]["name"] == "Cascais"
            assert result.data["localitiesNear"][0]["latitude"] == 38.6979

    @pytest.mark.asyncio
    async def test_municipality_by_id_basic(self):
        """Test municipality by ID query"""
        query = """
        query GetMunicipality($id: Int!) {
            municipality(id: $id) {
                id
                name
            }
        }
        """

        with patch(
            "app.graphql.resolvers.create_location_service"
        ) as mock_service_factory:
            mock_service = AsyncMock()
            mock_service.get_municipality.side_effect = Exception("Not found")
            mock_service_factory.return_value = mock_service

            result = await schema.execute(query, variable_values={"id": 999})

            assert result.errors is None
            assert result.data["municipality"] is None

    @pytest.mark.asyncio
    async def test_locality_by_id_basic(self):
        """Test locality by ID query"""
        query = """
        query GetLocality($id: Int!) {
            locality(id: $id) {
                id
                name
            }
        }
        """

        with patch(
            "app.graphql.resolvers.create_location_service"
        ) as mock_service_factory:
            mock_service = AsyncMock()
            mock_service.get_locality.side_effect = Exception("Not found")
            mock_service_factory.return_value = mock_service

            result = await schema.execute(query, variable_values={"id": 999})

            assert result.errors is None
            assert result.data["locality"] is None

    @pytest.mark.asyncio
    async def test_localities_near_basic(self):
        """Test localities near query"""
        query = """
        query GetNearbyLocalities($lat: Float!, $lng: Float!) {
            localitiesNear(lat: $lat, lng: $lng) {
                id
                name
            }
        }
        """

        with patch(
            "app.graphql.resolvers.create_location_service"
        ) as mock_service_factory:
            mock_service = AsyncMock()
            mock_service.search_nearby.return_value = []
            mock_service_factory.return_value = mock_service

            result = await schema.execute(
                query, variable_values={"lat": 38.7, "lng": -9.1}
            )

            assert result.errors is None
            assert "localitiesNear" in result.data
            assert result.data["localitiesNear"] == []
