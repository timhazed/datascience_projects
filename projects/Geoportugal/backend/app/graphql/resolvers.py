from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import strawberry
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import async_session_maker
from app.db.repositories import (
    DistrictRepository,
    LocalityRepository,
    MunicipalityRepository,
    SearchRepository,
)
from app.graphql.types import District, GeneralSearchResult, Locality, Municipality
from app.schemas import Locality as LocalitySchema
from app.schemas import NearbyFilters, SearchFilters
from app.services import LocationService, cache_service


def to_gql_locality(locality_schema: LocalitySchema) -> Locality:
    return Locality(
        id=locality_schema.id,
        name=locality_schema.name,
        latitude=locality_schema.latitude,
        longitude=locality_schema.longitude,
        feature_type=locality_schema.feature_type,
        population=locality_schema.population,
        municipality_id=locality_schema.municipality_id,
    )


@asynccontextmanager
async def service_ctx() -> AsyncIterator[LocationService]:
    """Provide a LocationService with a properly managed session."""
    async with async_session_maker() as session:
        service = LocationService(
            district_repo=DistrictRepository(session),
            municipality_repo=MunicipalityRepository(session),
            locality_repo=LocalityRepository(session),
            search_repo=SearchRepository(session),
            cache=cache_service,
        )
        yield service


async def create_location_service() -> LocationService:
    """Create a LocationService and keep the session open for the caller to manage."""
    session: AsyncSession = async_session_maker()
    service = LocationService(
        district_repo=DistrictRepository(session),
        municipality_repo=MunicipalityRepository(session),
        locality_repo=LocalityRepository(session),
        search_repo=SearchRepository(session),
        cache=cache_service,
    )
    service._session = session
    return service


@strawberry.type
class Query:
    @strawberry.field
    async def districts(self) -> list[District]:
        """Get all districts"""
        service = await create_location_service()
        try:
            districts = await service.get_districts(limit=50)
            return [
                District(id=d.id, name=d.name, code=d.code, population=d.population)
                for d in districts
            ]
        finally:
            session = getattr(service, "_session", None)
            if session:
                await session.close()

    @strawberry.field
    async def district(self, id: int) -> District | None:
        """Get a specific district by ID"""
        service = await create_location_service()
        try:
            district = await service.get_district(id)
            return District(
                id=district.id,
                name=district.name,
                code=district.code,
                population=district.population,
            )
        except Exception:
            return None
        finally:
            session = getattr(service, "_session", None)
            if session:
                await session.close()

    @strawberry.field
    async def municipalities(
        self, district_id: int | None = None
    ) -> list[Municipality]:
        """Get municipalities, optionally filtered by district"""
        service = await create_location_service()
        try:
            if district_id:
                municipalities = await service.get_district_municipalities(
                    district_id=district_id, limit=100
                )
            else:
                # Get all municipalities - using search with empty query as fallback
                result = await service.search_locations(SearchFilters(q="", limit=500))
                municipalities = result.get("municipalities", [])

            return [
                Municipality(
                    id=m.id,
                    name=m.name,
                    district_id=m.district_id,
                    population=m.population,
                    area=m.area,
                )
                for m in municipalities
            ]
        finally:
            session = getattr(service, "_session", None)
            if session:
                await session.close()

    @strawberry.field
    async def municipality(self, id: int) -> Municipality | None:
        """Get a specific municipality by ID"""
        service = await create_location_service()
        try:
            municipality = await service.get_municipality(id)
            return Municipality(
                id=municipality.id,
                name=municipality.name,
                district_id=municipality.district_id,
                population=municipality.population,
                area=municipality.area,
            )
        except Exception:
            return None
        finally:
            session = getattr(service, "_session", None)
            if session:
                await session.close()

    @strawberry.field
    async def municipality_localities(self, municipality_id: int) -> list[Locality]:
        """Get all localities in a specific municipality"""
        service = await create_location_service()
        try:
            localities = await service.get_municipality_localities(
                municipality_id, limit=100
            )
            return [to_gql_locality(locality) for locality in localities]
        finally:
            session = getattr(service, "_session", None)
            if session:
                await session.close()

    @strawberry.field
    async def locality(self, id: int) -> Locality | None:
        """Get a specific locality by ID"""
        service = await create_location_service()
        try:
            locality = await service.get_locality(id)
            return to_gql_locality(locality)
        except Exception:
            return None
        finally:
            session = getattr(service, "_session", None)
            if session:
                await session.close()

    @strawberry.field
    async def search_locations(self, query: str) -> list[Locality]:
        """Search across all location types, returning localities with coordinates"""
        service = await create_location_service()
        try:
            results = await service.search_locations(SearchFilters(q=query, limit=50))

            localities = results.get("localities", [])
            return [to_gql_locality(locality) for locality in localities]
        finally:
            session = getattr(service, "_session", None)
            if session:
                await session.close()

    @strawberry.field
    async def localities_near(
        self, lat: float, lng: float, radius_km: float = 10.0
    ) -> list[Locality]:
        """Find localities near given coordinates"""
        service = await create_location_service()
        try:
            results = await service.search_nearby(
                NearbyFilters(lat=lat, lng=lng, radius_km=radius_km, limit=20)
            )

            return [to_gql_locality(locality) for locality in results]
        finally:
            session = getattr(service, "_session", None)
            if session:
                await session.close()

    @strawberry.field
    async def city_search(self, query: str) -> list[Locality]:
        """Search only cities/localities - specific places with coordinates"""
        service = await create_location_service()
        try:
            localities = await service.search_cities_only(
                SearchFilters(q=query, limit=50)
            )

            return [to_gql_locality(locality) for locality in localities]
        finally:
            session = getattr(service, "_session", None)
            if session:
                await session.close()

    @strawberry.field
    async def general_search(self, query: str) -> GeneralSearchResult:
        """Search all administrative levels and return smart results"""
        service = await create_location_service()
        try:
            results = await service.search_locations(SearchFilters(q=query, limit=50))

            districts = results.get("districts", [])
            municipalities = results.get("municipalities", [])
            localities = results.get("localities", [])

            # Smart logic: prioritize most specific exact matches
            exact_localities = [
                locality
                for locality in localities
                if locality.name.lower().strip() == query.lower().strip()
            ]
            if exact_localities:
                return GeneralSearchResult(
                    result_type="city_found",
                    localities=[
                        to_gql_locality(locality) for locality in exact_localities
                    ],
                    message=f"Found city: {query}",
                )

            exact_municipalities = [
                municipality
                for municipality in municipalities
                if municipality.name.lower().strip() == query.lower().strip()
            ]
            if exact_municipalities:
                municipality = exact_municipalities[0]
                municipality_localities = await service.get_municipality_localities(
                    municipality.id
                )
                return GeneralSearchResult(
                    result_type="municipality_found",
                    localities=[
                        to_gql_locality(loc) for loc in municipality_localities
                    ],
                    message=f"Showing all locations in {municipality.name} municipality",
                )

            exact_districts = [
                d for d in districts if d.name.lower().strip() == query.lower().strip()
            ]
            if exact_districts:
                district = exact_districts[0]
                district_localities = await service.get_district_localities(district.id)
                return GeneralSearchResult(
                    result_type="district_found",
                    localities=[to_gql_locality(loc) for loc in district_localities],
                    message=f"Showing all locations in {district.name} district",
                )

            # Fallback to partial matches
            return GeneralSearchResult(
                result_type="partial_matches",
                localities=[to_gql_locality(loc) for loc in localities[:20]],
                message=f"Found {len(localities)} partial matches",
            )
        finally:
            session = getattr(service, "_session", None)
            if session:
                await session.close()
