from fastapi import APIRouter, Depends

from app.api.dependencies import get_location_service
from app.schemas import Locality
from app.services import LocationService

router = APIRouter()


@router.get("/localities/{locality_id}", response_model=Locality)
async def get_locality(
    locality_id: int,
    service: LocationService = Depends(get_location_service),
) -> Locality:
    """
    Get a specific locality by ID.

    Returns detailed information about a locality including name, coordinates,
    population, feature type, and its parent municipality and district.
    """
    return await service.get_locality(locality_id)
