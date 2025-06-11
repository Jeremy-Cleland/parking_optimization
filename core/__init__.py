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
    "APIError",
    "CityCoordinator",
    "ConfigurationError",
    "DataValidationError",
    "DemandPredictor",
    "DemandState",
    "District",
    "DriverError",
    "DynamicPricingEngine",
    "InvalidParkingOperationError",
    # Map data integration
    "MapDataLoader",
    "NetworkError",
    # Exceptions
    "ParkingOptimizationError",
    "ParkingSpot",
    # Core algorithms
    "ParkingZone",
    "ParkingZoneError",
    "ParkingZoneFullError",
    "PerformanceError",
    "PricingError",
    "ResourceExhaustionError",
    "RouteNotFoundError",
    "RouteOptimizer",
    "RouteResult",
    # Run management
    "RunManager",
    "SafeOperation",
    "SimulationError",
    # Configuration and utilities
    "SystemConfig",
    "TrafficAPIError",
    "complete_run",
    "get_config",
    "get_logger",
    "get_map_data_loader",
    "get_parking_zone_manager",
    "get_run_manager",
    "handle_api_errors",
    "metrics",
    "start_run",
    "time_it",
    "validate_capacity",
    "validate_coordinates",
    "validate_positive_number",
]
