from fastapi import APIRouter, Depends, Query

from app.api.dependencies import get_location_service
from app.schemas import District, Municipality
from app.services import LocationService

router = APIRouter()


@router.get("/districts", response_model=list[District])
async def list_districts(
    limit: int = Query(
        default=50, ge=1, le=100, description="Number of districts to return"
    ),
    offset: int = Query(default=0, ge=0, description="Number of districts to skip"),
    service: LocationService = Depends(get_location_service),
) -> list[District]:
    """
    Get all districts in Portugal.

    Returns a paginated list of districts with their basic information.
    """
    return await service.get_districts(limit=limit, offset=offset)


@router.get("/districts/{district_id}", response_model=District)
async def get_district(
    district_id: int,
    service: LocationService = Depends(get_location_service),
) -> District:
    """
    Get a specific district by ID.

    Returns detailed information about a district including name, code, and population.
    """
    return await service.get_district(district_id)


@router.get(
    "/districts/{district_id}/municipalities", response_model=list[Municipality]
)
async def list_district_municipalities(
    district_id: int,
    limit: int = Query(
        default=50, ge=1, le=100, description="Number of municipalities to return"
    ),
    offset: int = Query(
        default=0, ge=0, description="Number of municipalities to skip"
    ),
    service: LocationService = Depends(get_location_service),
) -> list[Municipality]:
    """
    Get all municipalities in a specific district.

    Returns a paginated list of municipalities belonging to the specified district.
    """
    return await service.get_district_municipalities(
        district_id, limit=limit, offset=offset
    )
