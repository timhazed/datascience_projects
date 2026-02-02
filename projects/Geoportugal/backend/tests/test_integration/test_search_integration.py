import pytest
import pytest_asyncio

from app.db.models import District, Locality, Municipality
from app.db.repositories import LocalityRepository, SearchRepository
from app.schemas import SearchFilters


class TestSearchIntegration:
    """Integration tests for search functionality using real database."""

    @pytest_asyncio.fixture
    async def enhanced_sample_data(self, test_session):
        """Create enhanced sample data for search testing."""
        # Create Porto district and municipality
        porto_district = District(name="Porto", code="13", population=1287282)
        test_session.add(porto_district)
        await test_session.flush()

        porto_municipality = Municipality(
            district_id=porto_district.id, name="Porto", population=237591, area=41.42
        )
        test_session.add(porto_municipality)
        await test_session.flush()

        # Create Lisboa district and municipality (from conftest.py sample_data)
        lisboa_district = District(name="Lisboa", code="11", population=2821697)
        test_session.add(lisboa_district)
        await test_session.flush()

        lisboa_municipality = Municipality(
            district_id=lisboa_district.id,
            name="Lisboa",
            population=547631,
            area=100.05,
        )
        test_session.add(lisboa_municipality)
        await test_session.flush()

        # Create localities
        localities = [
            # Porto localities
            Locality(
                municipality_id=porto_municipality.id,
                name="Porto",
                feature_type="populated_place",
                population=237591,
                latitude=41.1579,
                longitude=-8.6291,
            ),
            Locality(
                municipality_id=porto_municipality.id,
                name="Foz do Douro",
                feature_type="neighborhood",
                population=15000,
                latitude=41.1496,
                longitude=-8.6753,
            ),
            # Lisboa localities
            Locality(
                municipality_id=lisboa_municipality.id,
                name="Alfama",
                feature_type="neighborhood",
                population=5000,
                latitude=38.7131,
                longitude=-9.1301,
            ),
            Locality(
                municipality_id=lisboa_municipality.id,
                name="Belém",
                feature_type="populated_place",
                population=8000,
                latitude=38.6979,
                longitude=-9.2064,
            ),
        ]

        for locality in localities:
            test_session.add(locality)

        await test_session.commit()

        return {
            "porto_district": porto_district,
            "lisboa_district": lisboa_district,
            "porto_municipality": porto_municipality,
            "lisboa_municipality": lisboa_municipality,
            "localities": localities,
        }

    @pytest.mark.asyncio
    async def test_city_search_integration(self, test_session, enhanced_sample_data):
        """Test city search integration with real database."""
        locality_repo = LocalityRepository(test_session)

        # Search for Porto city
        filters = SearchFilters(q="Porto", limit=50, offset=0)
        results = await locality_repo.search_by_name(filters)

        assert len(results) == 1
        assert results[0].name == "Porto"
        assert results[0].feature_type == "populated_place"
        assert results[0].population == 237591

    @pytest.mark.asyncio
    async def test_district_localities_integration(
        self, test_session, enhanced_sample_data
    ):
        """Test getting all localities in a district."""
        locality_repo = LocalityRepository(test_session)

        # Get all localities in Lisboa district
        lisboa_district_id = enhanced_sample_data["lisboa_district"].id
        results = await locality_repo.get_by_district(
            lisboa_district_id, limit=100, offset=0
        )

        assert len(results) == 2
        locality_names = [loc.name for loc in results]
        assert "Alfama" in locality_names
        assert "Belém" in locality_names

    @pytest.mark.asyncio
    async def test_general_search_priority_integration(
        self, test_session, enhanced_sample_data
    ):
        """Test general search priority logic with real data."""
        search_repo = SearchRepository(test_session)

        # Search for "Porto" - should find district, municipality, and city
        filters = SearchFilters(q="Porto", limit=50, offset=0)
        results = await search_repo.search_locations(filters)

        # Verify we found results at all levels
        assert len(results["districts"]) == 1
        assert len(results["municipalities"]) == 1
        assert len(results["localities"]) == 1

        # Verify names match
        assert results["districts"][0].name == "Porto"
        assert results["municipalities"][0].name == "Porto"
        assert results["localities"][0].name == "Porto"

    @pytest.mark.asyncio
    async def test_search_case_insensitive_integration(
        self, test_session, enhanced_sample_data
    ):
        """Test search is case insensitive."""
        locality_repo = LocalityRepository(test_session)

        # Search for "PORTO", "porto", "Porto"
        filters_upper = SearchFilters(q="PORTO", limit=50, offset=0)
        filters_lower = SearchFilters(q="porto", limit=50, offset=0)
        filters_mixed = SearchFilters(q="Porto", limit=50, offset=0)

        results_upper = await locality_repo.search_by_name(filters_upper)
        results_lower = await locality_repo.search_by_name(filters_lower)
        results_mixed = await locality_repo.search_by_name(filters_mixed)

        # All should find the same result
        assert len(results_upper) == 1
        assert len(results_lower) == 1
        assert len(results_mixed) == 1
        assert results_upper[0].name == results_lower[0].name == results_mixed[0].name

    @pytest.mark.asyncio
    async def test_search_no_partial_match_integration(
        self, test_session, enhanced_sample_data
    ):
        """Test search does NOT find partial matches (exact only)."""
        locality_repo = LocalityRepository(test_session)

        # Search for "Bel" should NOT find "Belém" (exact match only)
        filters = SearchFilters(q="Bel", limit=50, offset=0)
        results = await locality_repo.search_by_name(filters)

        assert len(results) == 0  # No results for partial match

        # But exact match should work
        filters_exact = SearchFilters(q="Belém", limit=50, offset=0)
        results_exact = await locality_repo.search_by_name(filters_exact)

        assert len(results_exact) == 1
        assert results_exact[0].name == "Belém"

    @pytest.mark.asyncio
    async def test_empty_search_integration(self, test_session, enhanced_sample_data):
        """Test search with empty or short query."""
        locality_repo = LocalityRepository(test_session)

        # Empty query
        filters_empty = SearchFilters(q="", limit=50, offset=0)
        results_empty = await locality_repo.search_by_name(filters_empty)
        assert results_empty == []

        # Short query (1 character)
        filters_short = SearchFilters(q="P", limit=50, offset=0)
        results_short = await locality_repo.search_by_name(filters_short)
        assert results_short == []

    @pytest.mark.asyncio
    async def test_no_results_integration(self, test_session, enhanced_sample_data):
        """Test search with no matching results."""
        locality_repo = LocalityRepository(test_session)

        filters = SearchFilters(q="NonExistentPlace", limit=50, offset=0)
        results = await locality_repo.search_by_name(filters)

        assert results == []
