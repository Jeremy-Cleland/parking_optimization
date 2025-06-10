from typing import Type

from .base import MapService
from .google_maps_service import GoogleMapsService
from .mapquest_service import MapQuestService
from .tomtom_service import TomTomService

# Import other services here as they are added

# A mapping of provider names to service classes
_SERVICE_PROVIDERS = {
    "mapquest": MapQuestService,
    "tomtom": TomTomService,
    "google": GoogleMapsService,
    # Add other providers here
}


def get_map_service(provider: str, api_key: str) -> MapService:
    """
    Factory function to get a map service provider.

    Args:
        provider: The name of the map service provider (e.g., "mapquest").
        api_key: The API key for the service.

    Returns:
        An instance of a MapService subclass.

    Raises:
        ValueError: If the provider is not supported.
    """
    provider_class = _SERVICE_PROVIDERS.get(provider.lower())
    if not provider_class:
        raise ValueError(f"Unsupported map service provider: {provider}")
    return provider_class(api_key=api_key)


__all__ = ["get_map_service", "MapService"]
