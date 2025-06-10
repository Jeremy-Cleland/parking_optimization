"""
Configuration management for the parking optimization system.
Centralized settings using environment variables and default values.
"""

import os
from pathlib import Path
from typing import Optional

# Load environment variables from .env file
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    # python-dotenv not available, continue with system env vars only
    pass


class APISettings:
    """API configuration settings"""

    def __init__(self):
        # Load from environment variables with defaults
        self.map_provider = os.getenv(
            "MAP_PROVIDER", "tomtom"
        )  # Default to TomTom for generous free tier
        self.mapquest_api_key = os.getenv("MAPQUEST_API_KEY")
        self.tomtom_api_key = os.getenv("TOMTOM_API_KEY")
        self.google_maps_api_key = os.getenv("GOOGLE_MAPS_API_KEY")

        # Rate limiting optimized for free tiers
        self.max_api_calls_per_minute = int(
            os.getenv("MAX_API_CALLS_PER_MINUTE", "30")
        )  # Conservative rate
        self.api_cache_ttl_seconds = int(
            os.getenv("API_CACHE_TTL_SECONDS", "900")
        )  # 15 min cache

        # Free tier optimization settings
        self.enable_request_batching = (
            os.getenv("ENABLE_REQUEST_BATCHING", "true").lower() == "true"
        )
        self.aggressive_caching = (
            os.getenv("AGGRESSIVE_CACHING", "true").lower() == "true"
        )

        # Provider-specific daily limits (for monitoring)
        self.tomtom_daily_tile_limit = int(
            os.getenv("TOMTOM_DAILY_TILE_LIMIT", "50000")
        )
        self.tomtom_daily_api_limit = int(os.getenv("TOMTOM_DAILY_API_LIMIT", "2500"))
        self.mapquest_monthly_limit = int(os.getenv("MAPQUEST_MONTHLY_LIMIT", "15000"))

    def dict(self):
        return {
            "map_provider": self.map_provider,
            "mapquest_api_key": bool(self.mapquest_api_key),
            "tomtom_api_key": bool(self.tomtom_api_key),
            "google_maps_api_key": bool(
                self.google_maps_api_key
            ),  # Don't expose actual key
            "max_api_calls_per_minute": self.max_api_calls_per_minute,
            "api_cache_ttl_seconds": self.api_cache_ttl_seconds,
            "enable_request_batching": self.enable_request_batching,
            "aggressive_caching": self.aggressive_caching,
            "tomtom_daily_tile_limit": self.tomtom_daily_tile_limit,
            "tomtom_daily_api_limit": self.tomtom_daily_api_limit,
            "mapquest_monthly_limit": self.mapquest_monthly_limit,
        }


class SimulationSettings:
    """Simulation configuration settings"""

    def __init__(self):
        self.default_zones = int(os.getenv("DEFAULT_ZONES", "20"))
        self.default_drivers = int(os.getenv("DEFAULT_DRIVERS", "500"))
        self.default_duration_hours = float(os.getenv("DEFAULT_DURATION_HOURS", "8.0"))
        self.default_time_step_minutes = int(
            os.getenv("DEFAULT_TIME_STEP_MINUTES", "5")
        )
        self.default_city_size_km = float(os.getenv("DEFAULT_CITY_SIZE_KM", "10.0"))

        self.max_concurrent_drivers = int(os.getenv("MAX_CONCURRENT_DRIVERS", "1000"))
        self.simulation_seed = (
            int(os.getenv("SIMULATION_SEED", "42"))
            if os.getenv("SIMULATION_SEED")
            else 42
        )
        self.enable_real_time_traffic = (
            os.getenv("ENABLE_REAL_TIME_TRAFFIC", "true").lower() == "true"
        )

        # Validate ranges
        self.default_zones = max(1, min(1000, self.default_zones))
        self.default_drivers = max(1, min(10000, self.default_drivers))
        self.default_duration_hours = max(0.1, min(168.0, self.default_duration_hours))

    def dict(self):
        return {
            "default_zones": self.default_zones,
            "default_drivers": self.default_drivers,
            "default_duration_hours": self.default_duration_hours,
            "default_time_step_minutes": self.default_time_step_minutes,
            "default_city_size_km": self.default_city_size_km,
            "max_concurrent_drivers": self.max_concurrent_drivers,
            "simulation_seed": self.simulation_seed,
            "enable_real_time_traffic": self.enable_real_time_traffic,
        }


class PricingSettings:
    """Dynamic pricing configuration"""

    def __init__(self):
        self.min_price_per_hour = float(os.getenv("MIN_PRICE_PER_HOUR", "1.0"))
        self.max_price_per_hour = float(os.getenv("MAX_PRICE_PER_HOUR", "15.0"))
        self.base_price_per_hour = float(os.getenv("BASE_PRICE_PER_HOUR", "3.0"))

        self.target_occupancy_rate = float(os.getenv("TARGET_OCCUPANCY_RATE", "0.85"))
        self.price_adjustment_rate = float(os.getenv("PRICE_ADJUSTMENT_RATE", "0.1"))
        self.price_elasticity = float(os.getenv("PRICE_ELASTICITY", "-1.5"))
        self.cross_elasticity = float(os.getenv("CROSS_ELASTICITY", "0.3"))

        self.rush_hour_multiplier = float(os.getenv("RUSH_HOUR_MULTIPLIER", "1.3"))
        self.weekend_discount = float(os.getenv("WEEKEND_DISCOUNT", "0.8"))

        # Validate constraints
        if self.max_price_per_hour <= self.min_price_per_hour:
            self.max_price_per_hour = self.min_price_per_hour + 1.0

        self.target_occupancy_rate = max(0.1, min(0.99, self.target_occupancy_rate))
        self.price_adjustment_rate = max(0.01, min(1.0, self.price_adjustment_rate))

    def dict(self):
        return {
            "min_price_per_hour": self.min_price_per_hour,
            "max_price_per_hour": self.max_price_per_hour,
            "base_price_per_hour": self.base_price_per_hour,
            "target_occupancy_rate": self.target_occupancy_rate,
            "price_adjustment_rate": self.price_adjustment_rate,
            "price_elasticity": self.price_elasticity,
            "cross_elasticity": self.cross_elasticity,
            "rush_hour_multiplier": self.rush_hour_multiplier,
            "weekend_discount": self.weekend_discount,
        }


class OutputSettings:
    """Output and logging configuration"""

    def __init__(self):
        self.output_dir = Path(os.getenv("OUTPUT_DIR", "output"))
        self.visualization_dir = Path(
            os.getenv("VISUALIZATION_DIR", "visualization_output")
        )
        self.logs_dir = Path(os.getenv("LOGS_DIR", "logs"))

        self.save_results_json = (
            os.getenv("SAVE_RESULTS_JSON", "true").lower() == "true"
        )
        self.save_visualizations = (
            os.getenv("SAVE_VISUALIZATIONS", "true").lower() == "true"
        )
        self.generate_reports = os.getenv("GENERATE_REPORTS", "true").lower() == "true"

        self.log_level = os.getenv("LOG_LEVEL", "INFO").upper()
        if self.log_level not in ["DEBUG", "INFO", "WARNING", "ERROR"]:
            self.log_level = "INFO"

        self.log_to_file = os.getenv("LOG_TO_FILE", "true").lower() == "true"
        self.max_log_file_size_mb = int(os.getenv("MAX_LOG_FILE_SIZE_MB", "10"))

    def dict(self):
        return {
            "output_dir": str(self.output_dir),
            "visualization_dir": str(self.visualization_dir),
            "logs_dir": str(self.logs_dir),
            "save_results_json": self.save_results_json,
            "save_visualizations": self.save_visualizations,
            "generate_reports": self.generate_reports,
            "log_level": self.log_level,
            "log_to_file": self.log_to_file,
            "max_log_file_size_mb": self.max_log_file_size_mb,
        }


class PerformanceSettings:
    """Performance and optimization settings"""

    def __init__(self):
        self.enable_parallel_processing = (
            os.getenv("ENABLE_PARALLEL_PROCESSING", "true").lower() == "true"
        )
        self.max_worker_threads = int(os.getenv("MAX_WORKER_THREADS", "10"))
        self.enable_caching = os.getenv("ENABLE_CACHING", "true").lower() == "true"
        self.cache_size_limit = int(os.getenv("CACHE_SIZE_LIMIT", "1000"))

        self.max_memory_usage_gb = float(os.getenv("MAX_MEMORY_USAGE_GB", "48.0"))
        self.garbage_collection_frequency = int(
            os.getenv("GARBAGE_COLLECTION_FREQUENCY", "100")
        )

        self.enable_profiling = os.getenv("ENABLE_PROFILING", "false").lower() == "true"
        self.profile_output_dir = Path(os.getenv("PROFILE_OUTPUT_DIR", "profiling"))

        # Validate ranges
        self.max_worker_threads = max(1, min(32, self.max_worker_threads))
        self.max_memory_usage_gb = max(0.5, min(32.0, self.max_memory_usage_gb))

    def dict(self):
        return {
            "enable_parallel_processing": self.enable_parallel_processing,
            "max_worker_threads": self.max_worker_threads,
            "enable_caching": self.enable_caching,
            "cache_size_limit": self.cache_size_limit,
            "max_memory_usage_gb": self.max_memory_usage_gb,
            "garbage_collection_frequency": self.garbage_collection_frequency,
            "enable_profiling": self.enable_profiling,
            "profile_output_dir": str(self.profile_output_dir),
        }


class SystemConfig:
    """Main configuration class that combines all settings"""

    def __init__(self):
        self.api = APISettings()
        self.simulation = SimulationSettings()
        self.pricing = PricingSettings()
        self.output = OutputSettings()
        self.performance = PerformanceSettings()

        # Create output directories
        self._create_directories()

    def _create_directories(self):
        """Create necessary directories"""
        directories = [
            self.output.output_dir,
            self.output.visualization_dir,
            self.output.logs_dir,
        ]

        if self.performance.enable_profiling:
            directories.append(self.performance.profile_output_dir)

        for directory in directories:
            directory.mkdir(exist_ok=True, parents=True)

    @property
    def active_api_key(self) -> Optional[str]:
        """Get the API key for the currently configured map provider."""
        if self.api.map_provider == "mapquest":
            return self.api.mapquest_api_key
        elif self.api.map_provider == "tomtom":
            return self.api.tomtom_api_key
        elif self.api.map_provider == "google":
            return self.api.google_maps_api_key
        return None

    @property
    def has_api_keys(self) -> bool:
        """Check if any API keys are configured"""
        return bool(
            self.api.mapquest_api_key
            or self.api.tomtom_api_key
            or self.api.google_maps_api_key
        )

    @property
    def has_mapquest(self) -> bool:
        """Check if MapQuest is configured"""
        return bool(self.api.mapquest_api_key)

    @property
    def has_tomtom(self) -> bool:
        """Check if TomTom is configured"""
        return bool(self.api.tomtom_api_key)

    @property
    def has_google_maps(self) -> bool:
        """Check if Google Maps is configured"""
        return bool(self.api.google_maps_api_key)

    def to_dict(self) -> dict:
        """Convert configuration to dictionary for serialization"""
        return {
            "api": self.api.dict(),
            "simulation": self.simulation.dict(),
            "pricing": self.pricing.dict(),
            "output": self.output.dict(),
            "performance": self.performance.dict(),
        }

    def validate(self) -> list[str]:
        """Validate configuration and return any warnings"""
        warnings = []

        if not self.has_api_keys:
            warnings.append("No API keys configured - using fallback mode")

        if self.api.map_provider == "mapquest" and not self.has_mapquest:
            warnings.append(
                "Map provider is 'mapquest' but MAPQUEST_API_KEY is not set."
            )
        elif self.api.map_provider == "tomtom" and not self.has_tomtom:
            warnings.append("Map provider is 'tomtom' but TOMTOM_API_KEY is not set.")
        elif self.api.map_provider == "google" and not self.has_google_maps:
            warnings.append(
                "Map provider is 'google' but GOOGLE_MAPS_API_KEY is not set."
            )

        if (
            self.simulation.default_drivers > 1000
            and not self.performance.enable_parallel_processing
        ):
            warnings.append("High driver count without parallel processing may be slow")

        if self.pricing.max_price_per_hour > 20.0:
            warnings.append("Very high maximum pricing may deter all drivers")

        return warnings


# Global configuration instance
config = SystemConfig()


def get_config() -> SystemConfig:
    """Get the global configuration instance"""
    return config


def reload_config():
    """Reload configuration from environment/files"""
    global config
    config = SystemConfig()
    return config
