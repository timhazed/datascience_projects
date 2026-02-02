from datetime import datetime

from pydantic import BaseModel, ConfigDict


class BaseSchema(BaseModel):
    model_config = ConfigDict(from_attributes=True)


class DistrictBase(BaseSchema):
    name: str
    code: str
    population: int | None = None


class DistrictCreate(DistrictBase):
    pass


class DistrictUpdate(BaseSchema):
    name: str | None = None
    code: str | None = None
    population: int | None = None


class District(DistrictBase):
    id: int
    created_at: datetime


class MunicipalityBase(BaseSchema):
    name: str
    population: int | None = None
    area: float | None = None


class MunicipalityCreate(MunicipalityBase):
    district_id: int


class MunicipalityUpdate(BaseSchema):
    name: str | None = None
    population: int | None = None
    area: float | None = None


class Municipality(MunicipalityBase):
    id: int
    district_id: int
    created_at: datetime


class LocalityBase(BaseSchema):
    name: str
    feature_type: str
    population: int | None = None
    latitude: float
    longitude: float


class LocalityCreate(LocalityBase):
    municipality_id: int


class LocalityUpdate(BaseSchema):
    name: str | None = None
    feature_type: str | None = None
    population: int | None = None
    latitude: float | None = None
    longitude: float | None = None


class Locality(LocalityBase):
    id: int
    municipality_id: int
    created_at: datetime


class AlternateNameBase(BaseSchema):
    name: str
    language: str | None = None
    is_preferred: bool = False


class AlternateNameCreate(AlternateNameBase):
    locality_id: int
    location_type: str


class AlternateName(AlternateNameBase):
    id: int
    locality_id: int
    location_type: str


class SearchFilters(BaseSchema):
    q: str | None = None
    limit: int = 50
    offset: int = 0


class NearbyFilters(BaseSchema):
    lat: float
    lng: float
    radius_km: float = 10.0
    limit: int = 50
