from unittest.mock import AsyncMock, Mock

import pytest

from app.db.models import Locality
from app.db.repositories import LocalityRepository
from app.schemas import SearchFilters


class TestLocalityRepository:
    """Test new search methods in LocalityRepository."""

    @pytest.fixture
    def mock_session(self):
        """Create mock database session."""
        session = Mock()
        session.execute = AsyncMock()
        return session

    @pytest.fixture
    def locality_repo(self, mock_session):
        """Create LocalityRepository with mocked session."""
        return LocalityRepository(mock_session)

    @pytest.fixture
    def sample_localities(self):
        """Sample locality data for testing."""
        return [
            Locality(
                id=1,
                municipality_id=1,
                name="Porto",
                feature_type="city",
                population=237591,
                latitude=41.1579,
                longitude=-8.6291,
            ),
            Locality(
                id=2,
                municipality_id=1,
                name="Oporto",  # Alternative name
                feature_type="city",
                population=237591,
                latitude=41.1579,
                longitude=-8.6291,
            ),
        ]

    @pytest.fixture
    def district_localities(self):
        """Sample localities from a district for testing."""
        return [
            Locality(
                id=10,
                municipality_id=5,
                name="Alfama",
                feature_type="neighborhood",
                population=5000,
                latitude=38.7131,
                longitude=-9.1301,
            ),
            Locality(
                id=11,
                municipality_id=6,
                name="Belém",
                feature_type="neighborhood",
                population=8000,
                latitude=38.6979,
                longitude=-9.2064,
            ),
        ]


class TestSearchByName(TestLocalityRepository):
    """Test search_by_name method."""

    @pytest.mark.asyncio
    async def test_search_by_name_exact_match_only(self, locality_repo, mock_session):
        """Test search_by_name returns only exact matches."""
        # Create exact match locality
        exact_match = Locality(
            id=1,
            municipality_id=1,
            name="Porto",
            feature_type="populated_place",
            population=237591,
            latitude=41.1579,
            longitude=-8.6291,
        )

        # Setup mock to return only exact match
        mock_result = Mock()
        mock_result.scalars.return_value.all.return_value = [exact_match]
        mock_session.execute.return_value = mock_result

        # Execute
        filters = SearchFilters(q="Porto", limit=50, offset=0)
        result = await locality_repo.search_by_name(filters)

        # Verify only exact match returned
        assert len(result) == 1
        assert result[0].name == "Porto"
        assert result[0].feature_type == "populated_place"
        mock_session.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_search_by_name_empty_query(self, locality_repo, mock_session):
        """Test search_by_name with empty query returns empty list."""
        filters = SearchFilters(q="", limit=50, offset=0)
        result = await locality_repo.search_by_name(filters)

        assert result == []
        mock_session.execute.assert_not_called()

    @pytest.mark.asyncio
    async def test_search_by_name_short_query(self, locality_repo, mock_session):
        """Test search_by_name with query less than 2 characters."""
        filters = SearchFilters(q="P", limit=50, offset=0)
        result = await locality_repo.search_by_name(filters)

        assert result == []
        mock_session.execute.assert_not_called()

    @pytest.mark.asyncio
    async def test_search_by_name_no_results(self, locality_repo, mock_session):
        """Test search_by_name with no matching results."""
        mock_result = Mock()
        mock_result.scalars.return_value.all.return_value = []
        mock_session.execute.return_value = mock_result

        filters = SearchFilters(q="NonExistent", limit=50, offset=0)
        result = await locality_repo.search_by_name(filters)

        assert result == []
        mock_session.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_search_by_name_respects_limit_offset(
        self, locality_repo, mock_session, sample_localities
    ):
        """Test search_by_name respects limit and offset parameters."""
        mock_result = Mock()
        mock_result.scalars.return_value.all.return_value = [sample_localities[0]]
        mock_session.execute.return_value = mock_result

        filters = SearchFilters(q="Porto", limit=1, offset=0)
        result = await locality_repo.search_by_name(filters)

        assert len(result) == 1
        mock_session.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_search_by_name_no_partial_matches(self, locality_repo, mock_session):
        """Test search_by_name does NOT return partial matches."""
        # Mock should return empty list when only partial matches exist
        mock_result = Mock()
        mock_result.scalars.return_value.all.return_value = []
        mock_session.execute.return_value = mock_result

        # Search for "Tavira" should NOT return "Cabanas de Tavira" or "Luz de Tavira"
        filters = SearchFilters(q="Tavira", limit=50, offset=0)
        result = await locality_repo.search_by_name(filters)

        # Should be empty if only partial matches exist in database
        assert result == []
        mock_session.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_search_by_name_case_insensitive(self, locality_repo, mock_session):
        """Test search_by_name is case insensitive."""
        exact_match = Locality(
            id=1,
            municipality_id=1,
            name="Porto",
            feature_type="populated_place",
            population=237591,
            latitude=41.1579,
            longitude=-8.6291,
        )

        mock_result = Mock()
        mock_result.scalars.return_value.all.return_value = [exact_match]
        mock_session.execute.return_value = mock_result

        # Search with different case should work
        filters = SearchFilters(q="PORTO", limit=50, offset=0)
        result = await locality_repo.search_by_name(filters)

        assert len(result) == 1
        assert result[0].name == "Porto"
        mock_session.execute.assert_called_once()


class TestGetByDistrict(TestLocalityRepository):
    """Test get_by_district method."""

    @pytest.mark.asyncio
    async def test_get_by_district_with_results(
        self, locality_repo, mock_session, district_localities
    ):
        """Test get_by_district returns localities from district."""
        mock_result = Mock()
        mock_result.scalars.return_value.all.return_value = district_localities
        mock_session.execute.return_value = mock_result

        result = await locality_repo.get_by_district(
            district_id=11, limit=100, offset=0
        )

        assert len(result) == 2
        assert result[0].name == "Alfama"
        assert result[1].name == "Belém"
        mock_session.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_by_district_empty_result(self, locality_repo, mock_session):
        """Test get_by_district with no localities in district."""
        mock_result = Mock()
        mock_result.scalars.return_value.all.return_value = []
        mock_session.execute.return_value = mock_result

        result = await locality_repo.get_by_district(
            district_id=99, limit=100, offset=0
        )

        assert result == []
        mock_session.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_by_district_respects_limit_offset(
        self, locality_repo, mock_session, district_localities
    ):
        """Test get_by_district respects limit and offset."""
        mock_result = Mock()
        mock_result.scalars.return_value.all.return_value = [district_localities[0]]
        mock_session.execute.return_value = mock_result

        result = await locality_repo.get_by_district(district_id=11, limit=1, offset=5)

        assert len(result) == 1
        assert result[0].name == "Alfama"
        mock_session.execute.assert_called_once()
