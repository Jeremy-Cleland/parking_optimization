"""
Custom exceptions for the parking optimization system.
Provides clear error handling and debugging information.
"""

from typing import Any, Dict, Optional


class ParkingOptimizationError(Exception):
    """Base exception for all parking optimization errors"""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        self.message = message
        self.details = details or {}
        super().__init__(self.message)

    def __str__(self):
        if self.details:
            details_str = ", ".join(f"{k}={v}" for k, v in self.details.items())
            return f"{self.message} (Details: {details_str})"
        return self.message


class ConfigurationError(ParkingOptimizationError):
    """Raised when there are configuration issues"""

    pass


class APIError(ParkingOptimizationError):
    """Base class for API-related errors"""

    def __init__(
        self, message: str, api_name: str, status_code: Optional[int] = None, **kwargs
    ):
        super().__init__(
            message, {"api_name": api_name, "status_code": status_code, **kwargs}
        )
        self.api_name = api_name
        self.status_code = status_code


class TrafficAPIError(APIError):
    """Raised when traffic API calls fail"""

    pass


class RouteNotFoundError(ParkingOptimizationError):
    """Raised when no route can be found between two points"""

    def __init__(self, start: str, end: str, message: Optional[str] = None):
        self.start = start
        self.end = end
        default_message = f"No route found from {start} to {end}"
        super().__init__(message or default_message, {"start": start, "end": end})


class ParkingZoneError(ParkingOptimizationError):
    """Base class for parking zone errors"""

    pass


class ParkingZoneFullError(ParkingZoneError):
    """Raised when trying to park in a full zone"""

    def __init__(self, zone_id: str, capacity: int):
        self.zone_id = zone_id
        self.capacity = capacity
        super().__init__(
            f"Parking zone {zone_id} is full (capacity: {capacity})",
            {"zone_id": zone_id, "capacity": capacity},
        )


class InvalidParkingOperationError(ParkingZoneError):
    """Raised for invalid parking operations"""

    pass


class PricingError(ParkingOptimizationError):
    """Raised when pricing calculation fails"""

    pass


class SimulationError(ParkingOptimizationError):
    """Base class for simulation errors"""

    pass


class DriverError(SimulationError):
    """Raised when driver behavior/operations fail"""

    pass


class NetworkError(ParkingOptimizationError):
    """Raised when network/graph operations fail"""

    pass


class DataValidationError(ParkingOptimizationError):
    """Raised when data validation fails"""

    def __init__(
        self, field: str, value: Any, expected: str, message: Optional[str] = None
    ):
        self.field = field
        self.value = value
        self.expected = expected
        default_message = f"Invalid {field}: got {value}, expected {expected}"
        super().__init__(
            message or default_message,
            {"field": field, "value": value, "expected": expected},
        )


class PerformanceError(ParkingOptimizationError):
    """Raised when performance thresholds are exceeded"""

    def __init__(self, operation: str, duration: float, threshold: float):
        self.operation = operation
        self.duration = duration
        self.threshold = threshold
        super().__init__(
            f"Operation {operation} took {duration:.2f}s, exceeding threshold {threshold:.2f}s",
            {"operation": operation, "duration": duration, "threshold": threshold},
        )


class ResourceExhaustionError(ParkingOptimizationError):
    """Raised when system resources are exhausted"""

    def __init__(self, resource: str, limit: Any, current: Any):
        self.resource = resource
        self.limit = limit
        self.current = current
        super().__init__(
            f"Resource {resource} exhausted: {current} exceeds limit {limit}",
            {"resource": resource, "limit": limit, "current": current},
        )


# Exception context managers
class SafeOperation:
    """Context manager for safe operations with error handling"""

    def __init__(self, operation_name: str, logger=None, reraise: bool = True):
        self.operation_name = operation_name
        self.logger = logger
        self.reraise = reraise
        self.success = False

    def __enter__(self):
        if self.logger:
            self.logger.debug(f"Starting safe operation: {self.operation_name}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            self.success = True
            if self.logger:
                self.logger.debug(f"Safe operation completed: {self.operation_name}")
            return True

        # Handle the exception
        if self.logger:
            self.logger.error(
                f"Safe operation failed: {self.operation_name}", error=str(exc_val)
            )

        # Decide whether to reraise
        if not self.reraise:
            return True  # Suppress the exception

        return False  # Let the exception propagate


def handle_api_errors(func):
    """Decorator to handle common API errors"""

    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            # Convert common exceptions to our custom ones
            if "timeout" in str(e).lower():
                raise APIError(f"API timeout: {e}", "unknown")
            elif "connection" in str(e).lower():
                raise APIError(f"Connection error: {e}", "unknown")
            elif "404" in str(e):
                raise APIError(f"API endpoint not found: {e}", "unknown", 404)
            elif "401" in str(e) or "403" in str(e):
                raise APIError(f"API authentication error: {e}", "unknown", 401)
            elif "429" in str(e):
                raise APIError(f"API rate limit exceeded: {e}", "unknown", 429)
            else:
                raise APIError(f"Unexpected API error: {e}", "unknown")

    return wrapper


def validate_coordinates(lat: float, lon: float) -> None:
    """Validate geographic coordinates"""
    if not isinstance(lat, (int, float)) or not isinstance(lon, (int, float)):
        raise DataValidationError(
            "coordinates", f"({lat}, {lon})", "numeric latitude and longitude"
        )

    if not (-90 <= lat <= 90):
        raise DataValidationError("latitude", lat, "value between -90 and 90")

    if not (-180 <= lon <= 180):
        raise DataValidationError("longitude", lon, "value between -180 and 180")


def validate_positive_number(value: Any, field_name: str) -> None:
    """Validate that a value is a positive number"""
    if not isinstance(value, (int, float)):
        raise DataValidationError(field_name, value, "numeric value")

    if value <= 0:
        raise DataValidationError(field_name, value, "positive number")


def validate_capacity(
    capacity: int, min_capacity: int = 1, max_capacity: int = 10000
) -> None:
    """Validate parking zone capacity"""
    if not isinstance(capacity, int):
        raise DataValidationError("capacity", capacity, "integer")

    if not (min_capacity <= capacity <= max_capacity):
        raise DataValidationError(
            "capacity", capacity, f"integer between {min_capacity} and {max_capacity}"
        )
