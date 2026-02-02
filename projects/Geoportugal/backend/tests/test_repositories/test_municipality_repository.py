from unittest.mock import AsyncMock, Mock

import pytest

from app.core.exceptions import MunicipalityNotFoundError
from app.db.models import Locality, Municipality
from app.db.repositories import MunicipalityRepository


class TestMunicipalityRepository:
    """Test MunicipalityRepository methods that need coverage."""

    @pytest.fixture
    def mock_session(self):
        """Create mock database session."""
        session = Mock()
        session.execute = AsyncMock()
        return session

    @pytest.fixture
    def municipality_repo(self, mock_session):
        """Create MunicipalityRepository with mocked session."""
        return MunicipalityRepository(mock_session)

    @pytest.fixture
    def sample_municipality(self):
        """Sample municipality for testing."""
        return Municipality(
            id=1, district_id=1, name="Lisboa", population=547631, area=100.05
        )

    @pytest.fixture
    def sample_localities(self):
        """Sample localities for testing."""
        return [
            Locality(
                id=1,
                municipality_id=1,
                name="Belém",
                latitude=38.6979,
                longitude=-9.2071,
                feature_type="locality",
                population=16528,
            ),
            Locality(
                id=2,
                municipality_id=1,
                name="Estrela",
                latitude=38.7097,
                longitude=-9.1618,
                feature_type="locality",
                population=21200,
            ),
        ]

    @pytest.mark.asyncio
    async def test_get_by_id_success(
        self, municipality_repo, mock_session, sample_municipality
    ):
        """Test successful get_by_id."""
        mock_result = Mock()
        mock_result.scalar_one_or_none.return_value = sample_municipality
        mock_session.execute.return_value = mock_result

        result = await municipality_repo.get_by_id(1)

        assert result == sample_municipality
        mock_session.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_by_id_not_found(self, municipality_repo, mock_session):
        """Test get_by_id when municipality not found."""
        mock_result = Mock()
        mock_result.scalar_one_or_none.return_value = None
        mock_session.execute.return_value = mock_result

        with pytest.raises(
            MunicipalityNotFoundError, match="Municipality with id 999 not found"
        ):
            await municipality_repo.get_by_id(999)

    @pytest.mark.asyncio
    async def test_get_localities_success(
        self, municipality_repo, mock_session, sample_municipality, sample_localities
    ):
        """Test successful get_localities."""
        # Mock get_by_id call
        mock_get_result = Mock()
        mock_get_result.scalar_one_or_none.return_value = sample_municipality

        # Mock localities query
        mock_localities_result = Mock()
        mock_localities_result.scalars.return_value.all.return_value = sample_localities

        # Configure session.execute to return appropriate results
        mock_session.execute.side_effect = [mock_get_result, mock_localities_result]

        result = await municipality_repo.get_localities(1, limit=10, offset=0)

        assert result == sample_localities
        assert mock_session.execute.call_count == 2

    @pytest.mark.asyncio
    async def test_get_localities_municipality_not_found(
        self, municipality_repo, mock_session
    ):
        """Test get_localities when municipality not found."""
        mock_result = Mock()
        mock_result.scalar_one_or_none.return_value = None
        mock_session.execute.return_value = mock_result

        with pytest.raises(MunicipalityNotFoundError):
            await municipality_repo.get_localities(999)

    @pytest.mark.asyncio
    async def test_get_localities_empty_result(
        self, municipality_repo, mock_session, sample_municipality
    ):
        """Test get_localities with no localities."""
        # Mock get_by_id call
        mock_get_result = Mock()
        mock_get_result.scalar_one_or_none.return_value = sample_municipality

        # Mock empty localities query
        mock_localities_result = Mock()
        mock_localities_result.scalars.return_value.all.return_value = []

        mock_session.execute.side_effect = [mock_get_result, mock_localities_result]

        result = await municipality_repo.get_localities(1)

        assert result == []
        assert mock_session.execute.call_count == 2
