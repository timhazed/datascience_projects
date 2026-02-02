from unittest.mock import AsyncMock, Mock

import pytest

from app.db.models import Locality as LocalityModel
from app.schemas import Locality, SearchFilters
from app.services.location_service import LocationService


class TestSearchServices:
    """Test new search methods in LocationService."""

    @pytest.fixture
    def mock_repos(self):
        """Create mock repositories."""
        district_repo = Mock()
        municipality_repo = Mock()
        locality_repo = Mock()
        search_repo = Mock()
        cache = Mock()

        # Configure cache mock
        cache._generate_key = Mock(return_value="test_cache_key")
        cache.get = AsyncMock(return_value=None)
        cache.set = AsyncMock(return_value=True)

        return {
            "district_repo": district_repo,
            "municipality_repo": municipality_repo,
            "locality_repo": locality_repo,
            "search_repo": search_repo,
            "cache": cache,
        }

    @pytest.fixture
    def location_service(self, mock_repos):
        """Create LocationService with mocked dependencies."""
        return LocationService(
            district_repo=mock_repos["district_repo"],
            municipality_repo=mock_repos["municipality_repo"],
            locality_repo=mock_repos["locality_repo"],
            search_repo=mock_repos["search_repo"],
            cache=mock_repos["cache"],
        )

    @pytest.fixture
    def sample_locality_model(self):
        """Create sample locality model."""
        from datetime import datetime

        model = LocalityModel(
            id=1,
            municipality_id=1,
            name="Porto",
            feature_type="city",
            population=237591,
            latitude=41.1579,
            longitude=-8.6291,
        )
        model.created_at = datetime(2024, 1, 1, 0, 0, 0)
        return model


class TestSearchCitiesOnly(TestSearchServices):
    """Test search_cities_only method."""

    @pytest.mark.asyncio
    async def test_search_cities_only_cache_miss(
        self, location_service, mock_repos, sample_locality_model
    ):
        """Test search_cities_only when cache is empty."""
        mock_repos["cache"].get.return_value = None
        mock_repos["locality_repo"].search_by_name = AsyncMock(
            return_value=[sample_locality_model]
        )

        filters = SearchFilters(q="Porto", limit=50, offset=0)
        result = await location_service.search_cities_only(filters)

        assert len(result) == 1
        assert isinstance(result[0], Locality)
        assert result[0].name == "Porto"

        mock_repos["locality_repo"].search_by_name.assert_called_once_with(filters)
        mock_repos["cache"].set.assert_called_once()
        mock_repos["cache"]._generate_key.assert_called_once_with(
            "city_search", filters.model_dump()
        )

    @pytest.mark.asyncio
    async def test_search_cities_only_cache_hit(self, location_service, mock_repos):
        """Test search_cities_only when cache has data."""
        cached_data = [
            {
                "id": 1,
                "municipality_id": 1,
                "name": "Cached City",
                "feature_type": "city",
                "population": 100000,
                "latitude": 41.0,
                "longitude": -8.0,
                "created_at": "2024-01-01T00:00:00",
            }
        ]
        mock_repos["cache"].get.return_value = cached_data

        filters = SearchFilters(q="Porto", limit=50)
        result = await location_service.search_cities_only(filters)

        assert len(result) == 1
        assert result[0].name == "Cached City"
        mock_repos["locality_repo"].search_by_name.assert_not_called()

    @pytest.mark.asyncio
    async def test_search_cities_only_empty_results(self, location_service, mock_repos):
        """Test search_cities_only with no results."""
        mock_repos["cache"].get.return_value = None
        mock_repos["locality_repo"].search_by_name = AsyncMock(return_value=[])

        filters = SearchFilters(q="NonExistent", limit=50)
        result = await location_service.search_cities_only(filters)

        assert result == []
        mock_repos["locality_repo"].search_by_name.assert_called_once_with(filters)


class TestGetDistrictLocalities(TestSearchServices):
    """Test get_district_localities method."""

    @pytest.mark.asyncio
    async def test_get_district_localities_cache_miss(
        self, location_service, mock_repos, sample_locality_model
    ):
        """Test get_district_localities when cache is empty."""
        mock_repos["cache"].get.return_value = None
        mock_repos["locality_repo"].get_by_district = AsyncMock(
            return_value=[sample_locality_model]
        )

        result = await location_service.get_district_localities(
            district_id=11, limit=100, offset=0
        )

        assert len(result) == 1
        assert isinstance(result[0], Locality)
        assert result[0].name == "Porto"

        mock_repos["locality_repo"].get_by_district.assert_called_once_with(
            11, limit=100, offset=0
        )
        mock_repos["cache"].set.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_district_localities_cache_hit(
        self, location_service, mock_repos
    ):
        """Test get_district_localities when cache has data."""
        cached_data = [
            {
                "id": 10,
                "municipality_id": 5,
                "name": "Cached District Locality",
                "feature_type": "village",
                "population": 2000,
                "latitude": 38.5,
                "longitude": -9.0,
                "created_at": "2024-01-01T00:00:00",
            }
        ]
        mock_repos["cache"].get.return_value = cached_data

        result = await location_service.get_district_localities(district_id=11)

        assert len(result) == 1
        assert result[0].name == "Cached District Locality"
        mock_repos["locality_repo"].get_by_district.assert_not_called()

    @pytest.mark.asyncio
    async def test_get_district_localities_with_custom_params(
        self, location_service, mock_repos, sample_locality_model
    ):
        """Test get_district_localities with custom limit and offset."""
        mock_repos["cache"].get.return_value = None
        mock_repos["locality_repo"].get_by_district = AsyncMock(
            return_value=[sample_locality_model]
        )

        await location_service.get_district_localities(
            district_id=11, limit=50, offset=10
        )

        mock_repos["cache"]._generate_key.assert_called_once_with(
            "district_localities", {"district_id": 11, "limit": 50, "offset": 10}
        )
        mock_repos["locality_repo"].get_by_district.assert_called_once_with(
            11, limit=50, offset=10
        )

    @pytest.mark.asyncio
    async def test_get_district_localities_empty_results(
        self, location_service, mock_repos
    ):
        """Test get_district_localities with no localities in district."""
        mock_repos["cache"].get.return_value = None
        mock_repos["locality_repo"].get_by_district = AsyncMock(return_value=[])

        result = await location_service.get_district_localities(district_id=99)

        assert result == []
        mock_repos["locality_repo"].get_by_district.assert_called_once_with(
            99, limit=100, offset=0
        )
