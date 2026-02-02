from unittest.mock import AsyncMock, Mock

import pytest

from app.db.models import District as DistrictModel
from app.db.models import Locality as LocalityModel
from app.db.models import Municipality as MunicipalityModel
from app.schemas import District, Locality, Municipality, NearbyFilters, SearchFilters
from app.services.location_service import LocationService


class TestLocationService:
    """Test cases for LocationService business logic and caching."""

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
    def sample_district_model(self):
        """Create sample district model."""
        from datetime import datetime

        model = DistrictModel(id=1, name="Lisboa", code="11", population=2821697)
        model.created_at = datetime(2024, 1, 1, 0, 0, 0)
        return model

    @pytest.fixture
    def sample_municipality_model(self):
        """Create sample municipality model."""
        from datetime import datetime

        model = MunicipalityModel(
            id=1, district_id=1, name="Lisboa", population=547631, area=100.05
        )
        model.created_at = datetime(2024, 1, 1, 0, 0, 0)
        return model

    @pytest.fixture
    def sample_locality_model(self):
        """Create sample locality model."""
        from datetime import datetime

        model = LocalityModel(
            id=1,
            municipality_id=1,
            name="Alfama",
            feature_type="neighborhood",
            population=5000,
            latitude=38.7131,
            longitude=-9.1301,
        )
        model.created_at = datetime(2024, 1, 1, 0, 0, 0)
        return model


class TestGetDistricts(TestLocationService):
    """Test get_districts method."""

    @pytest.mark.asyncio
    async def test_get_districts_cache_miss(
        self, location_service, mock_repos, sample_district_model
    ):
        """Test get_districts when cache is empty."""
        # Setup mocks
        mock_repos["cache"].get.return_value = None
        mock_repos["district_repo"].get_all = AsyncMock(
            return_value=[sample_district_model]
        )

        # Execute
        result = await location_service.get_districts(limit=10, offset=0)

        # Verify
        assert len(result) == 1
        assert isinstance(result[0], District)
        assert result[0].name == "Lisboa"
        assert result[0].code == "11"

        # Verify cache interactions
        mock_repos["cache"]._generate_key.assert_called_once_with(
            "districts", {"limit": 10, "offset": 0}
        )
        mock_repos["cache"].get.assert_called_once_with("test_cache_key")
        mock_repos["cache"].set.assert_called_once()

        # Verify repository call
        mock_repos["district_repo"].get_all.assert_called_once_with(limit=10, offset=0)

    @pytest.mark.asyncio
    async def test_get_districts_cache_hit(self, location_service, mock_repos):
        """Test get_districts when cache has data."""
        # Setup cached data
        cached_data = [
            {
                "id": 1,
                "name": "Porto",
                "code": "13",
                "population": 1287282,
                "created_at": "2024-01-01T00:00:00",
            }
        ]
        mock_repos["cache"].get.return_value = cached_data

        # Execute
        result = await location_service.get_districts()

        # Verify
        assert len(result) == 1
        assert isinstance(result[0], District)
        assert result[0].name == "Porto"

        # Verify no database call
        mock_repos["district_repo"].get_all.assert_not_called()

        # Verify no cache set call
        mock_repos["cache"].set.assert_not_called()

    @pytest.mark.asyncio
    async def test_get_districts_with_custom_params(
        self, location_service, mock_repos, sample_district_model
    ):
        """Test get_districts with custom limit and offset."""
        mock_repos["cache"].get.return_value = None
        mock_repos["district_repo"].get_all = AsyncMock(
            return_value=[sample_district_model]
        )

        await location_service.get_districts(limit=5, offset=10)

        mock_repos["cache"]._generate_key.assert_called_once_with(
            "districts", {"limit": 5, "offset": 10}
        )
        mock_repos["district_repo"].get_all.assert_called_once_with(limit=5, offset=10)


class TestGetDistrict(TestLocationService):
    """Test get_district method."""

    @pytest.mark.asyncio
    async def test_get_district_cache_miss(
        self, location_service, mock_repos, sample_district_model
    ):
        """Test get_district when cache is empty."""
        mock_repos["cache"].get.return_value = None
        mock_repos["district_repo"].get_by_id = AsyncMock(
            return_value=sample_district_model
        )

        result = await location_service.get_district(district_id=1)

        assert isinstance(result, District)
        assert result.name == "Lisboa"
        assert result.id == 1

        mock_repos["district_repo"].get_by_id.assert_called_once_with(1)
        mock_repos["cache"].set.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_district_cache_hit(self, location_service, mock_repos):
        """Test get_district when cache has data."""
        cached_data = {
            "id": 1,
            "name": "Cached District",
            "code": "99",
            "population": 1000000,
            "created_at": "2024-01-01T00:00:00",
        }
        mock_repos["cache"].get.return_value = cached_data

        result = await location_service.get_district(district_id=1)

        assert result.name == "Cached District"
        mock_repos["district_repo"].get_by_id.assert_not_called()


class TestGetMunicipality(TestLocationService):
    """Test get_municipality method."""

    @pytest.mark.asyncio
    async def test_get_municipality_cache_miss(
        self, location_service, mock_repos, sample_municipality_model
    ):
        """Test get_municipality when cache is empty."""
        mock_repos["cache"].get.return_value = None
        mock_repos["municipality_repo"].get_by_id = AsyncMock(
            return_value=sample_municipality_model
        )

        result = await location_service.get_municipality(municipality_id=1)

        assert isinstance(result, Municipality)
        assert result.name == "Lisboa"
        assert result.district_id == 1

        mock_repos["municipality_repo"].get_by_id.assert_called_once_with(1)

    @pytest.mark.asyncio
    async def test_get_municipality_cache_hit(self, location_service, mock_repos):
        """Test get_municipality when cache has data."""
        cached_data = {
            "id": 1,
            "district_id": 1,
            "name": "Cached Municipality",
            "population": 100000,
            "area": 50.0,
            "created_at": "2024-01-01T00:00:00",
        }
        mock_repos["cache"].get.return_value = cached_data

        result = await location_service.get_municipality(municipality_id=1)

        assert result.name == "Cached Municipality"
        mock_repos["municipality_repo"].get_by_id.assert_not_called()


class TestGetLocality(TestLocationService):
    """Test get_locality method."""

    @pytest.mark.asyncio
    async def test_get_locality_cache_miss(
        self, location_service, mock_repos, sample_locality_model
    ):
        """Test get_locality when cache is empty."""
        mock_repos["cache"].get.return_value = None
        mock_repos["locality_repo"].get_by_id = AsyncMock(
            return_value=sample_locality_model
        )

        result = await location_service.get_locality(locality_id=1)

        assert isinstance(result, Locality)
        assert result.name == "Alfama"
        assert result.municipality_id == 1

        mock_repos["locality_repo"].get_by_id.assert_called_once_with(1)

    @pytest.mark.asyncio
    async def test_get_locality_cache_hit(self, location_service, mock_repos):
        """Test get_locality when cache has data."""
        cached_data = {
            "id": 1,
            "municipality_id": 1,
            "name": "Cached Locality",
            "feature_type": "city",
            "population": 10000,
            "latitude": 40.0,
            "longitude": -8.0,
            "created_at": "2024-01-01T00:00:00",
        }
        mock_repos["cache"].get.return_value = cached_data

        result = await location_service.get_locality(locality_id=1)

        assert result.name == "Cached Locality"
        mock_repos["locality_repo"].get_by_id.assert_not_called()


class TestSearchLocations(TestLocationService):
    """Test search_locations method."""

    @pytest.mark.asyncio
    async def test_search_locations_cache_miss(
        self,
        location_service,
        mock_repos,
        sample_district_model,
        sample_municipality_model,
        sample_locality_model,
    ):
        """Test search_locations when cache is empty."""
        mock_repos["cache"].get.return_value = None
        search_results = {
            "districts": [sample_district_model],
            "municipalities": [sample_municipality_model],
            "localities": [sample_locality_model],
        }
        mock_repos["search_repo"].search_locations = AsyncMock(
            return_value=search_results
        )

        filters = SearchFilters(q="Lisboa", limit=50, offset=0)
        result = await location_service.search_locations(filters)

        assert "districts" in result
        assert "municipalities" in result
        assert "localities" in result
        assert len(result["districts"]) == 1
        assert len(result["municipalities"]) == 1
        assert len(result["localities"]) == 1

        mock_repos["search_repo"].search_locations.assert_called_once_with(filters)
        mock_repos["cache"].set.assert_called_once()

    @pytest.mark.asyncio
    async def test_search_locations_cache_hit(self, location_service, mock_repos):
        """Test search_locations when cache has data."""
        cached_data = {
            "districts": [
                {
                    "id": 1,
                    "name": "Cached District",
                    "code": "11",
                    "population": 1000,
                    "created_at": "2024-01-01T00:00:00",
                }
            ],
            "municipalities": [],
            "localities": [],
        }
        mock_repos["cache"].get.return_value = cached_data

        filters = SearchFilters(q="test", limit=10, offset=0)
        result = await location_service.search_locations(filters)

        assert len(result["districts"]) == 1
        assert result["districts"][0].name == "Cached District"
        mock_repos["search_repo"].search_locations.assert_not_called()


class TestSearchNearby(TestLocationService):
    """Test search_nearby method."""

    @pytest.mark.asyncio
    async def test_search_nearby_cache_miss(
        self, location_service, mock_repos, sample_locality_model
    ):
        """Test search_nearby when cache is empty."""
        mock_repos["cache"].get.return_value = None
        mock_repos["locality_repo"].search_nearby = AsyncMock(
            return_value=[sample_locality_model]
        )

        filters = NearbyFilters(lat=38.7131, lng=-9.1301, radius_km=5.0, limit=50)
        result = await location_service.search_nearby(filters)

        assert len(result) == 1
        assert isinstance(result[0], Locality)
        assert result[0].name == "Alfama"

        mock_repos["locality_repo"].search_nearby.assert_called_once_with(filters)
        mock_repos["cache"].set.assert_called_once()

    @pytest.mark.asyncio
    async def test_search_nearby_cache_hit(self, location_service, mock_repos):
        """Test search_nearby when cache has data."""
        cached_data = [
            {
                "id": 1,
                "municipality_id": 1,
                "name": "Cached Nearby",
                "feature_type": "place",
                "population": 1000,
                "latitude": 38.7,
                "longitude": -9.1,
                "created_at": "2024-01-01T00:00:00",
            }
        ]
        mock_repos["cache"].get.return_value = cached_data

        filters = NearbyFilters(lat=38.7, lng=-9.1, radius_km=1.0, limit=10)
        result = await location_service.search_nearby(filters)

        assert len(result) == 1
        assert result[0].name == "Cached Nearby"
        mock_repos["locality_repo"].search_nearby.assert_not_called()


class TestDistrictMunicipalities(TestLocationService):
    """Test get_district_municipalities method."""

    @pytest.mark.asyncio
    async def test_get_district_municipalities_cache_miss(
        self, location_service, mock_repos, sample_municipality_model
    ):
        """Test get_district_municipalities when cache is empty."""
        mock_repos["cache"].get.return_value = None
        mock_repos["district_repo"].get_municipalities = AsyncMock(
            return_value=[sample_municipality_model]
        )

        result = await location_service.get_district_municipalities(
            district_id=1, limit=10, offset=0
        )

        assert len(result) == 1
        assert isinstance(result[0], Municipality)
        assert result[0].name == "Lisboa"

        mock_repos["district_repo"].get_municipalities.assert_called_once_with(
            1, limit=10, offset=0
        )

    @pytest.mark.asyncio
    async def test_get_district_municipalities_cache_hit(
        self, location_service, mock_repos
    ):
        """Test get_district_municipalities when cache has data."""
        cached_data = [
            {
                "id": 1,
                "district_id": 1,
                "name": "Cached Municipality",
                "population": 50000,
                "area": 25.5,
                "created_at": "2024-01-01T00:00:00",
            }
        ]
        mock_repos["cache"].get.return_value = cached_data

        result = await location_service.get_district_municipalities(district_id=1)

        assert len(result) == 1
        assert result[0].name == "Cached Municipality"
        mock_repos["district_repo"].get_municipalities.assert_not_called()


class TestMunicipalityLocalities(TestLocationService):
    """Test get_municipality_localities method."""

    @pytest.mark.asyncio
    async def test_get_municipality_localities_cache_miss(
        self, location_service, mock_repos, sample_locality_model
    ):
        """Test get_municipality_localities when cache is empty."""
        mock_repos["cache"].get.return_value = None
        mock_repos["municipality_repo"].get_localities = AsyncMock(
            return_value=[sample_locality_model]
        )

        result = await location_service.get_municipality_localities(
            municipality_id=1, limit=20, offset=5
        )

        assert len(result) == 1
        assert isinstance(result[0], Locality)
        assert result[0].name == "Alfama"

        mock_repos["municipality_repo"].get_localities.assert_called_once_with(
            1, limit=20, offset=5
        )

    @pytest.mark.asyncio
    async def test_get_municipality_localities_cache_hit(
        self, location_service, mock_repos
    ):
        """Test get_municipality_localities when cache has data."""
        cached_data = [
            {
                "id": 1,
                "municipality_id": 1,
                "name": "Cached Locality",
                "feature_type": "village",
                "population": 500,
                "latitude": 39.0,
                "longitude": -8.5,
                "created_at": "2024-01-01T00:00:00",
            }
        ]
        mock_repos["cache"].get.return_value = cached_data

        result = await location_service.get_municipality_localities(municipality_id=1)

        assert len(result) == 1
        assert result[0].name == "Cached Locality"
        mock_repos["municipality_repo"].get_localities.assert_not_called()


class TestCacheKeyGeneration(TestLocationService):
    """Test cache key generation patterns."""

    @pytest.mark.asyncio
    async def test_cache_keys_are_generated_correctly(
        self,
        location_service,
        mock_repos,
        sample_district_model,
        sample_municipality_model,
        sample_locality_model,
    ):
        """Test that different methods generate appropriate cache keys."""
        mock_repos["cache"].get.return_value = None
        mock_repos["district_repo"].get_all = AsyncMock(return_value=[])
        mock_repos["district_repo"].get_by_id = AsyncMock(
            return_value=sample_district_model
        )
        mock_repos["municipality_repo"].get_by_id = AsyncMock(
            return_value=sample_municipality_model
        )
        mock_repos["locality_repo"].get_by_id = AsyncMock(
            return_value=sample_locality_model
        )

        # Test different methods generate different cache key prefixes
        await location_service.get_districts(limit=10, offset=0)
        await location_service.get_district(district_id=1)
        await location_service.get_municipality(municipality_id=1)
        await location_service.get_locality(locality_id=1)

        # Verify cache key generation was called with different prefixes
        expected_calls = [
            (("districts", {"limit": 10, "offset": 0}),),
            (("district", {"id": 1}),),
            (("municipality", {"id": 1}),),
            (("locality", {"id": 1}),),
        ]

        actual_calls = mock_repos["cache"]._generate_key.call_args_list
        assert len(actual_calls) == 4

        for expected, actual in zip(expected_calls, actual_calls, strict=True):
            assert actual.args == expected[0]
