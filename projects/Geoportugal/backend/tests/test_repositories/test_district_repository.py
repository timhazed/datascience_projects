from unittest.mock import AsyncMock, Mock

import pytest

from app.core.exceptions import DistrictNotFoundError
from app.db.models import District, Municipality
from app.db.repositories import DistrictRepository


class TestDistrictRepository:
    """Test DistrictRepository methods that need coverage."""

    @pytest.fixture
    def mock_session(self):
        """Create mock database session."""
        session = Mock()
        session.execute = AsyncMock()
        return session

    @pytest.fixture
    def district_repo(self, mock_session):
        """Create DistrictRepository with mocked session."""
        return DistrictRepository(mock_session)

    @pytest.fixture
    def sample_district(self):
        """Sample district for testing."""
        return District(id=1, name="Lisboa", code="11", population=2250533)

    @pytest.fixture
    def sample_municipalities(self):
        """Sample municipalities for testing."""
        return [
            Municipality(
                id=1, district_id=1, name="Lisboa", population=547631, area=100.05
            ),
            Municipality(
                id=2, district_id=1, name="Sintra", population=377835, area=319.23
            ),
        ]

    @pytest.mark.asyncio
    async def test_get_by_id_success(
        self, district_repo, mock_session, sample_district
    ):
        """Test successful get_by_id."""
        mock_result = Mock()
        mock_result.scalar_one_or_none.return_value = sample_district
        mock_session.execute.return_value = mock_result

        result = await district_repo.get_by_id(1)

        assert result == sample_district
        mock_session.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_by_id_not_found(self, district_repo, mock_session):
        """Test get_by_id when district not found."""
        mock_result = Mock()
        mock_result.scalar_one_or_none.return_value = None
        mock_session.execute.return_value = mock_result

        with pytest.raises(
            DistrictNotFoundError, match="District with id 999 not found"
        ):
            await district_repo.get_by_id(999)

    @pytest.mark.asyncio
    async def test_get_municipalities_success(
        self, district_repo, mock_session, sample_district, sample_municipalities
    ):
        """Test successful get_municipalities."""
        # Mock get_by_id call
        mock_get_result = Mock()
        mock_get_result.scalar_one_or_none.return_value = sample_district

        # Mock municipalities query
        mock_municipalities_result = Mock()
        mock_municipalities_result.scalars.return_value.all.return_value = (
            sample_municipalities
        )

        # Configure session.execute to return appropriate results
        mock_session.execute.side_effect = [mock_get_result, mock_municipalities_result]

        result = await district_repo.get_municipalities(1, limit=10, offset=0)

        assert result == sample_municipalities
        assert mock_session.execute.call_count == 2

    @pytest.mark.asyncio
    async def test_get_municipalities_district_not_found(
        self, district_repo, mock_session
    ):
        """Test get_municipalities when district not found."""
        mock_result = Mock()
        mock_result.scalar_one_or_none.return_value = None
        mock_session.execute.return_value = mock_result

        with pytest.raises(DistrictNotFoundError):
            await district_repo.get_municipalities(999)

    @pytest.mark.asyncio
    async def test_get_municipalities_empty_result(
        self, district_repo, mock_session, sample_district
    ):
        """Test get_municipalities with no municipalities."""
        # Mock get_by_id call
        mock_get_result = Mock()
        mock_get_result.scalar_one_or_none.return_value = sample_district

        # Mock empty municipalities query
        mock_municipalities_result = Mock()
        mock_municipalities_result.scalars.return_value.all.return_value = []

        mock_session.execute.side_effect = [mock_get_result, mock_municipalities_result]

        result = await district_repo.get_municipalities(1)

        assert result == []
        assert mock_session.execute.call_count == 2
