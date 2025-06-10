"""
Core algorithms for parking optimization system
"""

from .config import SystemConfig, get_config
from .coordinator import CityCoordinator, District
from .demand_predictor import DemandPredictor, DemandState
from .dynamic_pricing import DynamicPricingEngine
from .exceptions import (
    APIError,
    ConfigurationError,
    DataValidationError,
    DriverError,
    InvalidParkingOperationError,
    NetworkError,
    ParkingOptimizationError,
    ParkingZoneError,
    ParkingZoneFullError,
    PerformanceError,
    PricingError,
    ResourceExhaustionError,
    RouteNotFoundError,
    SafeOperation,
    SimulationError,
    TrafficAPIError,
    handle_api_errors,
    validate_capacity,
    validate_coordinates,
    validate_positive_number,
)
from .logger import get_logger, metrics, time_it
from .map_data_loader import MapDataLoader, get_map_data_loader
from .parking_zone import ParkingSpot, ParkingZone, get_parking_zone_manager
from .route_optimizer import RouteOptimizer, RouteResult
from .run_manager import RunManager, complete_run, get_run_manager, start_run

__all__ = [
    # Core algorithms
    "ParkingZone",
    "ParkingSpot",
    "get_parking_zone_manager",
    "DynamicPricingEngine",
    "RouteOptimizer",
    "RouteResult",
    "DemandPredictor",
    "DemandState",
    "CityCoordinator",
    "District",
    # Map data integration
    "MapDataLoader",
    "get_map_data_loader",
    # Run management
    "RunManager",
    "get_run_manager",
    "start_run",
    "complete_run",
    # Configuration and utilities
    "SystemConfig",
    "get_config",
    "get_logger",
    "time_it",
    "metrics",
    # Exceptions
    "ParkingOptimizationError",
    "ConfigurationError",
    "APIError",
    "TrafficAPIError",
    "RouteNotFoundError",
    "ParkingZoneError",
    "ParkingZoneFullError",
    "InvalidParkingOperationError",
    "PricingError",
    "SimulationError",
    "DriverError",
    "NetworkError",
    "DataValidationError",
    "PerformanceError",
    "ResourceExhaustionError",
    "SafeOperation",
    "handle_api_errors",
    "validate_coordinates",
    "validate_positive_number",
    "validate_capacity",
]
