from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_session
from app.db.repositories import (
    DistrictRepository,
    LocalityRepository,
    MunicipalityRepository,
    SearchRepository,
)
from app.services import LocationService, cache_service


async def get_district_repository(
    session: AsyncSession = Depends(get_session),
) -> DistrictRepository:
    return DistrictRepository(session)


async def get_municipality_repository(
    session: AsyncSession = Depends(get_session),
) -> MunicipalityRepository:
    return MunicipalityRepository(session)


async def get_locality_repository(
    session: AsyncSession = Depends(get_session),
) -> LocalityRepository:
    return LocalityRepository(session)


async def get_search_repository(
    session: AsyncSession = Depends(get_session),
) -> SearchRepository:
    return SearchRepository(session)


async def get_location_service(
    district_repo: DistrictRepository = Depends(get_district_repository),
    municipality_repo: MunicipalityRepository = Depends(get_municipality_repository),
    locality_repo: LocalityRepository = Depends(get_locality_repository),
    search_repo: SearchRepository = Depends(get_search_repository),
) -> LocationService:
    return LocationService(
        district_repo=district_repo,
        municipality_repo=municipality_repo,
        locality_repo=locality_repo,
        search_repo=search_repo,
        cache=cache_service,
    )
