import strawberry


@strawberry.type
class District:
    id: int
    name: str
    code: str
    population: int | None = None


@strawberry.type
class Municipality:
    id: int
    name: str
    district_id: int = strawberry.field(name="districtId")
    population: int | None = None
    area: float | None = None


@strawberry.type
class Locality:
    id: int
    name: str
    latitude: float
    longitude: float
    feature_type: str = strawberry.field(name="featureType")
    population: int | None = None
    municipality_id: int = strawberry.field(name="municipalityId")


@strawberry.type
class GeneralSearchResult:
    result_type: (
        str  # "city_found", "municipality_found", "district_found", "partial_matches"
    )
    localities: list[Locality]
    message: str
