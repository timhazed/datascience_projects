class GeoPortugalException(Exception):
    """Base exception for GeoPortugal API"""

    pass


class LocationNotFoundError(GeoPortugalException):
    """Raised when a location is not found"""

    pass


class DistrictNotFoundError(LocationNotFoundError):
    """Raised when a district is not found"""

    pass


class MunicipalityNotFoundError(LocationNotFoundError):
    """Raised when a municipality is not found"""

    pass


class LocalityNotFoundError(LocationNotFoundError):
    """Raised when a locality is not found"""

    pass


class InvalidSearchParametersError(GeoPortugalException):
    """Raised when search parameters are invalid"""

    pass
