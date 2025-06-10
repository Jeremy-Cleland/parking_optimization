import abc
from typing import Any, Dict, List


class MapService(abc.ABC):
    """Abstract base class for map and traffic data services."""

    def __init__(self, api_key: str):
        if not api_key:
            raise ValueError(f"API key for {self.__class__.__name__} is required.")
        self.api_key = api_key
        self.base_url: str = ""

    @abc.abstractmethod
    def get_traffic_incidents(self, bounding_box: List[float]) -> Dict[str, Any]:
        """
        Get traffic incidents within a bounding box.

        Args:
            bounding_box: A list of four coordinates [lat1, lon1, lat2, lon2].

        Returns:
            A dictionary containing traffic incident data.
        """
        pass

    @abc.abstractmethod
    def get_route(
        self, start_coords: List[float], end_coords: List[float]
    ) -> Dict[str, Any]:
        """
        Get a route between two points.

        Args:
            start_coords: The starting coordinates [lat, lon].
            end_coords: The ending coordinates [lat, lon].

        Returns:
            A dictionary containing route data.
        """
        pass
