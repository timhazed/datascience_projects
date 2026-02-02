from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.logging import get_logger
from app.db.repositories import (
    DistrictRepository,
    LocalityRepository,
    MunicipalityRepository,
    SearchRepository,
)
from app.schemas import District, Locality, Municipality, NearbyFilters, SearchFilters

from .cache import CacheService

logger: Any = get_logger(__name__)


class LocationService:
    def __init__(
        self,
        district_repo: DistrictRepository,
        municipality_repo: MunicipalityRepository,
        locality_repo: LocalityRepository,
        search_repo: SearchRepository,
        cache: CacheService,
    ):
        self.district_repo = district_repo
        self.municipality_repo = municipality_repo
        self.locality_repo = locality_repo
        self.search_repo = search_repo
        self.cache = cache
        self._session: AsyncSession | None = None

    async def get_districts(self, limit: int = 50, offset: int = 0) -> list[District]:
        """Get all districts with caching"""
        cache_key = self.cache._generate_key(
            "districts", {"limit": limit, "offset": offset}
        )

        # Try cache first
        cached_result = await self.cache.get(cache_key)
        if cached_result:
            logger.debug("Returning cached districts")
            return [District.model_validate(d) for d in cached_result]

        # Get from database
        districts = await self.district_repo.get_all(limit=limit, offset=offset)
        result = [District.model_validate(d) for d in districts]

        # Cache the result
        await self.cache.set(cache_key, [d.model_dump() for d in result])

        logger.info("Retrieved districts", count=len(result))
        return result

    async def get_district(self, district_id: int) -> District:
        """Get district by ID with caching"""
        cache_key = self.cache._generate_key("district", {"id": district_id})

        # Try cache first
        cached_result = await self.cache.get(cache_key)
        if cached_result:
            logger.debug("Returning cached district", district_id=district_id)
            return District.model_validate(cached_result)

        # Get from database
        district = await self.district_repo.get_by_id(district_id)
        result = District.model_validate(district)

        # Cache the result
        await self.cache.set(cache_key, result.model_dump())

        logger.info("Retrieved district", district_id=district_id, name=district.name)
        return result

    async def get_district_municipalities(
        self, district_id: int, limit: int = 50, offset: int = 0
    ) -> list[Municipality]:
        """Get municipalities for a district with caching"""
        cache_key = self.cache._generate_key(
            "district_municipalities",
            {"district_id": district_id, "limit": limit, "offset": offset},
        )

        # Try cache first
        cached_result = await self.cache.get(cache_key)
        if cached_result:
            logger.debug(
                "Returning cached district municipalities", district_id=district_id
            )
            return [Municipality.model_validate(m) for m in cached_result]

        # Get from database
        municipalities = await self.district_repo.get_municipalities(
            district_id, limit=limit, offset=offset
        )
        result = [Municipality.model_validate(m) for m in municipalities]

        # Cache the result
        await self.cache.set(cache_key, [m.model_dump() for m in result])

        logger.info(
            "Retrieved district municipalities",
            district_id=district_id,
            count=len(result),
        )
        return result

    async def get_municipality(self, municipality_id: int) -> Municipality:
        """Get municipality by ID with caching"""
        cache_key = self.cache._generate_key("municipality", {"id": municipality_id})

        # Try cache first
        cached_result = await self.cache.get(cache_key)
        if cached_result:
            logger.debug(
                "Returning cached municipality", municipality_id=municipality_id
            )
            return Municipality.model_validate(cached_result)

        # Get from database
        municipality = await self.municipality_repo.get_by_id(municipality_id)
        result = Municipality.model_validate(municipality)

        # Cache the result
        await self.cache.set(cache_key, result.model_dump())

        logger.info(
            "Retrieved municipality",
            municipality_id=municipality_id,
            name=municipality.name,
        )
        return result

    async def get_municipality_localities(
        self, municipality_id: int, limit: int = 50, offset: int = 0
    ) -> list[Locality]:
        """Get localities for a municipality with caching"""
        cache_key = self.cache._generate_key(
            "municipality_localities",
            {"municipality_id": municipality_id, "limit": limit, "offset": offset},
        )

        # Try cache first
        cached_result = await self.cache.get(cache_key)
        if cached_result:
            logger.debug(
                "Returning cached municipality localities",
                municipality_id=municipality_id,
            )
            return [Locality.model_validate(locality) for locality in cached_result]

        # Get from database
        localities = await self.municipality_repo.get_localities(
            municipality_id, limit=limit, offset=offset
        )
        result = [Locality.model_validate(locality) for locality in localities]

        # Cache the result
        await self.cache.set(cache_key, [locality.model_dump() for locality in result])

        logger.info(
            "Retrieved municipality localities",
            municipality_id=municipality_id,
            count=len(result),
        )
        return result

    async def get_locality(self, locality_id: int) -> Locality:
        """Get locality by ID with caching"""
        cache_key = self.cache._generate_key("locality", {"id": locality_id})

        # Try cache first
        cached_result = await self.cache.get(cache_key)
        if cached_result:
            logger.debug("Returning cached locality", locality_id=locality_id)
            return Locality.model_validate(cached_result)

        # Get from database
        locality = await self.locality_repo.get_by_id(locality_id)
        locality_schema = Locality.model_validate(locality)

        # Cache the result
        await self.cache.set(cache_key, locality_schema.model_dump())

        logger.info("Retrieved locality", locality_id=locality_id, name=locality.name)
        return locality_schema

    async def search_locations(self, filters: SearchFilters) -> dict[str, list[Any]]:
        """Search locations with caching"""
        cache_key = self.cache._generate_key("search", filters.model_dump())

        # Try cache first
        cached_result = await self.cache.get(cache_key)
        if cached_result:
            logger.debug("Returning cached search results", query=filters.q)
            return {
                "districts": [
                    District.model_validate(d) for d in cached_result["districts"]
                ],
                "municipalities": [
                    Municipality.model_validate(m)
                    for m in cached_result["municipalities"]
                ],
                "localities": [
                    Locality.model_validate(locality)
                    for locality in cached_result["localities"]
                ],
            }

        # Get from database
        results = await self.search_repo.search_locations(filters)

        # Convert to schemas
        search_results: dict[str, list[Any]] = {
            "districts": [
                District.model_validate(district) for district in results["districts"]
            ],
            "municipalities": [
                Municipality.model_validate(municipality)
                for municipality in results["municipalities"]
            ],
            "localities": [
                Locality.model_validate(locality) for locality in results["localities"]
            ],
        }

        # Cache the result
        await self.cache.set(
            cache_key,
            {
                "districts": [
                    district.model_dump() for district in search_results["districts"]
                ],
                "municipalities": [
                    municipality.model_dump()
                    for municipality in search_results["municipalities"]
                ],
                "localities": [
                    locality.model_dump() for locality in search_results["localities"]
                ],
            },
        )

        total_results = sum(len(v) for v in search_results.values())
        logger.info("Search completed", query=filters.q, total_results=total_results)
        return search_results

    async def search_nearby(self, filters: NearbyFilters) -> list[Locality]:
        """Search nearby localities with caching"""
        cache_key = self.cache._generate_key("nearby", filters.model_dump())

        # Try cache first
        cached_result = await self.cache.get(cache_key)
        if cached_result:
            logger.debug(
                "Returning cached nearby results", lat=filters.lat, lng=filters.lng
            )
            return [Locality.model_validate(locality) for locality in cached_result]

        # Get from database
        localities = await self.locality_repo.search_nearby(filters)
        locality_schemas = [
            Locality.model_validate(locality) for locality in localities
        ]

        # Cache the result
        await self.cache.set(
            cache_key, [locality.model_dump() for locality in locality_schemas]
        )

        logger.info(
            "Nearby search completed",
            lat=filters.lat,
            lng=filters.lng,
            radius_km=filters.radius_km,
            count=len(locality_schemas),
        )
        return locality_schemas

    async def search_cities_only(self, filters: SearchFilters) -> list[Locality]:
        """Search only localities table for cities"""
        cache_key = self.cache._generate_key("city_search", filters.model_dump())

        # Try cache first
        cached_result = await self.cache.get(cache_key)
        if cached_result:
            return [Locality.model_validate(locality) for locality in cached_result]

        # Search only localities
        localities = await self.locality_repo.search_by_name(filters)
        locality_schemas = [
            Locality.model_validate(locality) for locality in localities
        ]

        # Cache the result
        await self.cache.set(
            cache_key, [locality.model_dump() for locality in locality_schemas]
        )

        return locality_schemas

    async def get_district_localities(
        self, district_id: int, limit: int = 100, offset: int = 0
    ) -> list[Locality]:
        """Get all localities within a district"""
        cache_key = self.cache._generate_key(
            "district_localities",
            {"district_id": district_id, "limit": limit, "offset": offset},
        )

        # Try cache first
        cached_result = await self.cache.get(cache_key)
        if cached_result:
            return [Locality.model_validate(locality) for locality in cached_result]

        # Get from database via locality repository
        localities = await self.locality_repo.get_by_district(
            district_id, limit=limit, offset=offset
        )
        locality_schemas = [
            Locality.model_validate(locality) for locality in localities
        ]

        # Cache the result
        await self.cache.set(
            cache_key, [locality.model_dump() for locality in locality_schemas]
        )

        return locality_schemas
