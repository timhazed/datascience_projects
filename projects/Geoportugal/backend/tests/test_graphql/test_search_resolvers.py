from unittest.mock import AsyncMock, patch

import pytest

from app.db.models import District as DistrictModel
from app.db.models import Locality as LocalityModel
from app.db.models import Municipality as MunicipalityModel
from app.graphql.schema import schema


class TestNewSearchResolvers:
    """Test new search GraphQL resolvers."""

    @pytest.mark.asyncio
    async def test_city_search_basic(self):
        """Test city_search resolver returns only localities."""
        query = """
        query CitySearch($query: String!) {
            citySearch(query: $query) {
                id
                name
                featureType
                latitude
                longitude
            }
        }
        """

        with patch("app.graphql.resolvers.create_location_service") as mock_factory:
            mock_service = AsyncMock()
            mock_localities = [
                LocalityModel(
                    id=1,
                    name="Porto",
                    latitude=41.1579,
                    longitude=-8.6291,
                    feature_type="city",
                    population=237591,
                    municipality_id=1,
                ),
                LocalityModel(
                    id=2,
                    name="Porto de Mós",
                    latitude=39.6027,
                    longitude=-8.8173,
                    feature_type="city",
                    population=4491,
                    municipality_id=2,
                ),
            ]
            mock_service.search_cities_only.return_value = mock_localities
            mock_factory.return_value = mock_service

            result = await schema.execute(query, variable_values={"query": "Porto"})

            assert result.errors is None
            assert "citySearch" in result.data
            assert len(result.data["citySearch"]) == 2
            assert result.data["citySearch"][0]["name"] == "Porto"
            assert result.data["citySearch"][1]["name"] == "Porto de Mós"

    @pytest.mark.asyncio
    async def test_city_search_no_results(self):
        """Test city_search with no matching cities."""
        query = """
        query CitySearch($query: String!) {
            citySearch(query: $query) {
                id
                name
            }
        }
        """

        with patch("app.graphql.resolvers.create_location_service") as mock_factory:
            mock_service = AsyncMock()
            mock_service.search_cities_only.return_value = []
            mock_factory.return_value = mock_service

            result = await schema.execute(
                query, variable_values={"query": "NonexistentCity"}
            )

            assert result.errors is None
            assert result.data["citySearch"] == []

    @pytest.mark.asyncio
    async def test_general_search_city_found(self):
        """Test general_search when exact city match is found."""
        query = """
        query GeneralSearch($query: String!) {
            generalSearch(query: $query) {
                resultType
                message
                localities {
                    id
                    name
                    featureType
                }
            }
        }
        """

        with patch("app.graphql.resolvers.create_location_service") as mock_factory:
            mock_service = AsyncMock()

            # Mock search results with exact locality match
            mock_localities = [
                LocalityModel(
                    id=1,
                    name="Porto",
                    latitude=41.1579,
                    longitude=-8.6291,
                    feature_type="city",
                    population=237591,
                    municipality_id=1,
                )
            ]
            mock_service.search_locations.return_value = {
                "districts": [],
                "municipalities": [],
                "localities": mock_localities,
            }
            mock_factory.return_value = mock_service

            result = await schema.execute(query, variable_values={"query": "Porto"})

            assert result.errors is None
            data = result.data["generalSearch"]
            assert data["resultType"] == "city_found"
            assert data["message"] == "Found city: Porto"
            assert len(data["localities"]) == 1
            assert data["localities"][0]["name"] == "Porto"

    @pytest.mark.asyncio
    async def test_general_search_municipality_found(self):
        """Test general_search when exact municipality match is found."""
        query = """
        query GeneralSearch($query: String!) {
            generalSearch(query: $query) {
                resultType
                message
                localities {
                    id
                    name
                }
            }
        }
        """

        with patch("app.graphql.resolvers.create_location_service") as mock_factory:
            mock_service = AsyncMock()

            # Mock search results with exact municipality match but no locality match
            mock_municipalities = [
                MunicipalityModel(
                    id=1, name="Sintra", district_id=11, population=377835, area=319.23
                )
            ]
            mock_municipality_localities = [
                LocalityModel(
                    id=10,
                    name="Queluz",
                    latitude=38.7558,
                    longitude=-9.2542,
                    feature_type="city",
                    population=26071,
                    municipality_id=1,
                ),
                LocalityModel(
                    id=11,
                    name="Agualva-Cacém",
                    latitude=38.7533,
                    longitude=-9.3122,
                    feature_type="city",
                    population=81845,
                    municipality_id=1,
                ),
            ]

            mock_service.search_locations.return_value = {
                "districts": [],
                "municipalities": mock_municipalities,
                "localities": [],  # No exact locality match
            }
            mock_service.get_municipality_localities.return_value = (
                mock_municipality_localities
            )
            mock_factory.return_value = mock_service

            result = await schema.execute(query, variable_values={"query": "Sintra"})

            assert result.errors is None
            data = result.data["generalSearch"]
            assert data["resultType"] == "municipality_found"
            assert data["message"] == "Showing all locations in Sintra municipality"
            assert len(data["localities"]) == 2

    @pytest.mark.asyncio
    async def test_general_search_district_found(self):
        """Test general_search when exact district match is found."""
        query = """
        query GeneralSearch($query: String!) {
            generalSearch(query: $query) {
                resultType
                message
                localities {
                    id
                    name
                }
            }
        }
        """

        with patch("app.graphql.resolvers.create_location_service") as mock_factory:
            mock_service = AsyncMock()

            # Mock search results with exact district match but no municipality/locality match
            mock_districts = [
                DistrictModel(id=11, name="Lisboa", code="11", population=2821697)
            ]
            mock_district_localities = [
                LocalityModel(
                    id=20,
                    name="Alfama",
                    latitude=38.7131,
                    longitude=-9.1301,
                    feature_type="neighborhood",
                    population=5000,
                    municipality_id=5,
                ),
                LocalityModel(
                    id=21,
                    name="Belém",
                    latitude=38.6979,
                    longitude=-9.2064,
                    feature_type="neighborhood",
                    population=8000,
                    municipality_id=6,
                ),
            ]

            mock_service.search_locations.return_value = {
                "districts": mock_districts,
                "municipalities": [],  # No exact municipality match
                "localities": [],  # No exact locality match
            }
            mock_service.get_district_localities.return_value = mock_district_localities
            mock_factory.return_value = mock_service

            result = await schema.execute(query, variable_values={"query": "Lisboa"})

            assert result.errors is None
            data = result.data["generalSearch"]
            assert data["resultType"] == "district_found"
            assert data["message"] == "Showing all locations in Lisboa district"
            assert len(data["localities"]) == 2

    @pytest.mark.asyncio
    async def test_general_search_partial_matches(self):
        """Test general_search falls back to partial matches."""
        query = """
        query GeneralSearch($query: String!) {
            generalSearch(query: $query) {
                resultType
                message
                localities {
                    id
                    name
                }
            }
        }
        """

        with patch("app.graphql.resolvers.create_location_service") as mock_factory:
            mock_service = AsyncMock()

            # Mock search results with only partial matches
            mock_localities = [
                LocalityModel(
                    id=30,
                    name="Santarém",
                    latitude=39.2369,
                    longitude=-8.6873,
                    feature_type="city",
                    population=32000,
                    municipality_id=10,
                ),
                LocalityModel(
                    id=31,
                    name="Santa Maria",
                    latitude=38.8,
                    longitude=-9.0,
                    feature_type="village",
                    population=1000,
                    municipality_id=11,
                ),
            ]

            mock_service.search_locations.return_value = {
                "districts": [],  # No exact matches
                "municipalities": [],  # No exact matches
                "localities": mock_localities,  # Only partial matches
            }
            mock_factory.return_value = mock_service

            result = await schema.execute(query, variable_values={"query": "Santa"})

            assert result.errors is None
            data = result.data["generalSearch"]
            assert data["resultType"] == "partial_matches"
            assert data["message"] == "Found 2 partial matches"
            assert len(data["localities"]) == 2

    @pytest.mark.asyncio
    async def test_general_search_no_results(self):
        """Test general_search with no results at all."""
        query = """
        query GeneralSearch($query: String!) {
            generalSearch(query: $query) {
                resultType
                message
                localities {
                    id
                    name
                }
            }
        }
        """

        with patch("app.graphql.resolvers.create_location_service") as mock_factory:
            mock_service = AsyncMock()

            mock_service.search_locations.return_value = {
                "districts": [],
                "municipalities": [],
                "localities": [],
            }
            mock_factory.return_value = mock_service

            result = await schema.execute(
                query, variable_values={"query": "NonexistentPlace"}
            )

            assert result.errors is None
            data = result.data["generalSearch"]
            assert data["resultType"] == "partial_matches"
            assert data["message"] == "Found 0 partial matches"
            assert len(data["localities"]) == 0

    @pytest.mark.asyncio
    async def test_create_location_service_function(self):
        """Test create_location_service helper function."""
        with patch("app.graphql.resolvers.async_session_maker") as mock_sessionmaker:
            mock_session = AsyncMock()
            mock_sessionmaker.return_value = mock_session

            from app.graphql.resolvers import create_location_service

            with patch("app.graphql.resolvers.cache_service"):
                service = await create_location_service()

                assert service is not None
                assert getattr(service, "_session", None) is mock_session
