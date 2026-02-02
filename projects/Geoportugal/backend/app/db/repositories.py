import math
import unicodedata
from typing import Any

from sqlalchemy import and_, func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.core.exceptions import (
    DistrictNotFoundError,
    LocalityNotFoundError,
    MunicipalityNotFoundError,
)
from app.schemas import NearbyFilters, SearchFilters

from .models import District, Locality, Municipality


def normalize_search_text(text: str) -> str:
    """Normalize text for accent-insensitive search"""
    if not text:
        return text
    # Remove accents and convert to lowercase
    normalized = unicodedata.normalize("NFD", text)
    ascii_text = "".join(c for c in normalized if unicodedata.category(c) != "Mn")
    return ascii_text.lower()


class DistrictRepository:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def get_all(self, limit: int = 50, offset: int = 0) -> list[District]:
        stmt = select(District).order_by(District.name).limit(limit).offset(offset)
        result = await self.session.execute(stmt)
        return list(result.scalars().all())

    async def get_by_id(self, district_id: int) -> District:
        stmt = select(District).where(District.id == district_id)
        result = await self.session.execute(stmt)
        district = result.scalar_one_or_none()
        if not district:
            raise DistrictNotFoundError(f"District with id {district_id} not found")
        return district

    async def get_municipalities(
        self, district_id: int, limit: int = 50, offset: int = 0
    ) -> list[Municipality]:
        district = await self.get_by_id(district_id)
        stmt = (
            select(Municipality)
            .where(Municipality.district_id == district.id)
            .order_by(Municipality.name)
            .limit(limit)
            .offset(offset)
        )
        result = await self.session.execute(stmt)
        return list(result.scalars().all())


class MunicipalityRepository:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def get_by_id(self, municipality_id: int) -> Municipality:
        stmt = (
            select(Municipality)
            .options(selectinload(Municipality.district))
            .where(Municipality.id == municipality_id)
        )
        result = await self.session.execute(stmt)
        municipality = result.scalar_one_or_none()
        if not municipality:
            raise MunicipalityNotFoundError(
                f"Municipality with id {municipality_id} not found"
            )
        return municipality

    async def get_localities(
        self, municipality_id: int, limit: int = 50, offset: int = 0
    ) -> list[Locality]:
        municipality = await self.get_by_id(municipality_id)
        stmt = (
            select(Locality)
            .where(Locality.municipality_id == municipality.id)
            .order_by(Locality.name)
            .limit(limit)
            .offset(offset)
        )
        result = await self.session.execute(stmt)
        return list(result.scalars().all())


class LocalityRepository:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def get_by_id(self, locality_id: int) -> Locality:
        stmt = (
            select(Locality)
            .options(
                selectinload(Locality.municipality).selectinload(Municipality.district),
                selectinload(Locality.alternate_names),
            )
            .where(Locality.id == locality_id)
        )
        result = await self.session.execute(stmt)
        locality = result.scalar_one_or_none()
        if not locality:
            raise LocalityNotFoundError(f"Locality with id {locality_id} not found")
        return locality

    async def search_nearby(self, filters: NearbyFilters) -> list[Locality]:
        # Simple distance calculation using Haversine formula approximation
        # For production, consider using PostGIS for better performance
        lat_rad = math.radians(filters.lat)
        # lng_rad = math.radians(filters.lng)

        # Approximate degree difference for given radius
        lat_diff = filters.radius_km / 111.0  # 1 degree lat ≈ 111 km
        lng_diff = filters.radius_km / (111.0 * math.cos(lat_rad))

        stmt = (
            select(Locality)
            .options(
                selectinload(Locality.municipality).selectinload(Municipality.district)
            )
            .where(
                and_(
                    Locality.latitude.between(
                        filters.lat - lat_diff, filters.lat + lat_diff
                    ),
                    Locality.longitude.between(
                        filters.lng - lng_diff, filters.lng + lng_diff
                    ),
                )
            )
            .limit(filters.limit)
        )

        result = await self.session.execute(stmt)
        localities = list(result.scalars().all())

        # Calculate actual distances and sort
        def calculate_distance(locality: Locality) -> float:
            lat1, lng1 = math.radians(filters.lat), math.radians(filters.lng)
            lat2, lng2 = (
                math.radians(locality.latitude),
                math.radians(locality.longitude),
            )

            dlat = lat2 - lat1
            dlng = lng2 - lng1

            a = (
                math.sin(dlat / 2) ** 2
                + math.cos(lat1) * math.cos(lat2) * math.sin(dlng / 2) ** 2
            )
            c = 2 * math.asin(math.sqrt(a))
            return 6371 * c  # Earth radius in km

        # Filter by actual distance and sort
        nearby_localities = [
            locality
            for locality in localities
            if calculate_distance(locality) <= filters.radius_km
        ]

        return sorted(nearby_localities, key=calculate_distance)

    async def search_by_name(self, filters: SearchFilters) -> list[Locality]:
        """Search localities by exact name only - for city search, deduplicated and prioritized"""
        localities = []

        if filters.q and len(filters.q.strip()) >= 2:
            search_term = filters.q.strip()
            normalized_search = normalize_search_text(search_term)

            # Only exact matches (case-insensitive)
            stmt = (
                select(Locality)
                .options(
                    selectinload(Locality.municipality).selectinload(
                        Municipality.district
                    )
                )
                .where(
                    or_(
                        func.lower(Locality.name) == search_term.lower(),
                        func.lower(Locality.name) == normalized_search.lower(),
                    )
                )
                .where(
                    Locality.feature_type.in_(["populated_place", "locality"])
                )  # Exclude parishes
                .order_by(Locality.population.desc().nulls_last(), Locality.name)
            )

            result = await self.session.execute(stmt)
            all_localities = list(result.scalars().all())

            # Deduplicate by coordinates (keep highest population entry for each location)
            seen_coords = {}
            for locality in all_localities:
                coord_key = f"{locality.latitude:.5f},{locality.longitude:.5f}"

                if coord_key not in seen_coords:
                    seen_coords[coord_key] = locality
                else:
                    # Keep the one with higher population, or first one if same population
                    existing = seen_coords[coord_key]
                    if (locality.population or 0) > (existing.population or 0):
                        seen_coords[coord_key] = locality

            # Convert back to list and apply limit/offset
            localities = list(seen_coords.values())

            # Sort again by population (highest first) and apply pagination
            localities.sort(key=lambda x: (-(x.population or 0), x.name))

            # Apply offset and limit
            start_idx = filters.offset
            end_idx = start_idx + filters.limit
            localities = localities[start_idx:end_idx]

        return localities

    async def get_by_district(
        self, district_id: int, limit: int = 100, offset: int = 0
    ) -> list[Locality]:
        """Get all localities within a district"""
        stmt = (
            select(Locality)
            .join(Municipality, Locality.municipality_id == Municipality.id)
            .options(
                selectinload(Locality.municipality).selectinload(Municipality.district)
            )
            .where(Municipality.district_id == district_id)
            .order_by(Locality.name)
            .limit(limit)
            .offset(offset)
        )

        result = await self.session.execute(stmt)
        return list(result.scalars().all())


class SearchRepository:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def search_locations(self, filters: SearchFilters) -> dict[str, list[Any]]:
        districts = []
        municipalities = []
        localities = []

        # Validate search query
        if filters.q and len(filters.q.strip()) >= 2:
            original_term = f"%{filters.q.strip()}%"
            normalized_term = f"%{normalize_search_text(filters.q.strip())}%"

            # Search districts (try both original and normalized)
            district_stmt = (
                select(District)
                .where(
                    or_(
                        District.name.ilike(original_term),
                        District.name.ilike(normalized_term),
                    )
                )
                .order_by(District.name)
                .limit(filters.limit // 3)
                .offset(filters.offset)
            )
            district_result = await self.session.execute(district_stmt)
            districts = list(district_result.scalars().all())

            # Search municipalities (try both original and normalized)
            municipality_stmt = (
                select(Municipality)
                .options(selectinload(Municipality.district))
                .where(
                    or_(
                        Municipality.name.ilike(original_term),
                        Municipality.name.ilike(normalized_term),
                    )
                )
                .order_by(Municipality.name)
                .limit(filters.limit // 3)
                .offset(filters.offset)
            )
            municipality_result = await self.session.execute(municipality_stmt)
            municipalities = list(municipality_result.scalars().all())

            # Search localities (try both original and normalized)
            locality_stmt = (
                select(Locality)
                .options(
                    selectinload(Locality.municipality).selectinload(
                        Municipality.district
                    )
                )
                .where(
                    or_(
                        Locality.name.ilike(original_term),
                        Locality.name.ilike(normalized_term),
                    )
                )
                .order_by(Locality.name)
                .limit(filters.limit // 3)
                .offset(filters.offset)
            )
            locality_result = await self.session.execute(locality_stmt)
            localities = list(locality_result.scalars().all())

        return {
            "districts": districts,
            "municipalities": municipalities,
            "localities": localities,
        }
