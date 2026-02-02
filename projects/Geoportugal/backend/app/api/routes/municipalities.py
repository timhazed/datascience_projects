from fastapi import APIRouter, Depends, Query

from app.api.dependencies import get_location_service
from app.schemas import Locality, Municipality
from app.services import LocationService

router = APIRouter()


@router.get("/municipalities/{municipality_id}", response_model=Municipality)
async def get_municipality(
    municipality_id: int,
    service: LocationService = Depends(get_location_service),
) -> Municipality:
    """
    Get a specific municipality by ID.

    Returns detailed information about a municipality including name, population,
    area, and its parent district.
    """
    return await service.get_municipality(municipality_id)


@router.get(
    "/municipalities/{municipality_id}/localities", response_model=list[Locality]
)
async def list_municipality_localities(
    municipality_id: int,
    limit: int = Query(
        default=50, ge=1, le=100, description="Number of localities to return"
    ),
    offset: int = Query(default=0, ge=0, description="Number of localities to skip"),
    service: LocationService = Depends(get_location_service),
) -> list[Locality]:
    """
    Get all localities in a specific municipality.

    Returns a paginated list of localities belonging to the specified municipality.
    """
    return await service.get_municipality_localities(
        municipality_id, limit=limit, offset=offset
    )
