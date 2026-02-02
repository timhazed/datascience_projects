from typing import Any

from fastapi import APIRouter, Depends, Query

from app.api.dependencies import get_location_service
from app.schemas import Locality, NearbyFilters, SearchFilters
from app.services import LocationService

router = APIRouter()


@router.get("/search", response_model=dict[str, list[Any]])
async def search_locations(
    q: str = Query(..., min_length=2, description="Search term (minimum 2 characters)"),
    limit: int = Query(
        default=50, ge=1, le=100, description="Maximum number of results per category"
    ),
    offset: int = Query(default=0, ge=0, description="Number of results to skip"),
    service: LocationService = Depends(get_location_service),
) -> dict[str, list[Any]]:
    """
    Search for locations across all types (districts, municipalities, localities).

    Performs a full-text search across all location names and returns results
    grouped by location type. The search is case-insensitive and supports
    partial matching.

    Returns:
        - districts: List of matching districts
        - municipalities: List of matching municipalities
        - localities: List of matching localities
    """
    filters = SearchFilters(q=q, limit=limit, offset=offset)
    return await service.search_locations(filters)


@router.get("/nearby", response_model=list[Locality])
async def search_nearby_localities(
    lat: float = Query(..., ge=-90, le=90, description="Latitude coordinate"),
    lng: float = Query(..., ge=-180, le=180, description="Longitude coordinate"),
    radius_km: float = Query(
        default=10.0, ge=0.1, le=100.0, description="Search radius in kilometers"
    ),
    limit: int = Query(
        default=50, ge=1, le=100, description="Maximum number of results to return"
    ),
    service: LocationService = Depends(get_location_service),
) -> list[Locality]:
    """
    Find localities near a specific geographic coordinate.

    Performs a geospatial search to find localities within the specified radius
    of the given coordinates. Results are sorted by distance from the center point.

    Args:
        lat: Latitude of the center point (-90 to 90)
        lng: Longitude of the center point (-180 to 180)
        radius_km: Search radius in kilometers (0.1 to 100)
        limit: Maximum number of results (1 to 100)

    Returns:
        List of localities within the specified radius, sorted by distance.
    """
    filters = NearbyFilters(lat=lat, lng=lng, radius_km=radius_km, limit=limit)
    return await service.search_nearby(filters)
