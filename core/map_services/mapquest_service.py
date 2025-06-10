from typing import Any, Dict, List

import requests

from .base import MapService


class MapQuestService(MapService):
    """MapQuest API service implementation."""

    def __init__(self, api_key: str):
        super().__init__(api_key)
        self.base_url = "https://www.mapquestapi.com"

    def get_traffic_incidents(self, bounding_box: List[float]) -> Dict[str, Any]:
        """
        Get traffic incidents from MapQuest.

        Args:
            bounding_box: A list of four coordinates [lat1, lon1, lat2, lon2].

        Returns:
            A dictionary containing traffic incident data.
        """
        url = f"{self.base_url}/traffic/v2/incidents"
        params = {
            "key": self.api_key,
            "boundingBox": ",".join(map(str, bounding_box)),
            "filters": "congestion,incidents,construction",
            "outFormat": "json",
        }
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            # In a real app, you'd want to log this error.
            print(f"Error fetching traffic data from MapQuest: {e}")
            return {}

    def get_route(
        self, start_coords: List[float], end_coords: List[float]
    ) -> Dict[str, Any]:
        """
        Get a route from MapQuest.

        Args:
            start_coords: The starting coordinates [lat, lon].
            end_coords: The ending coordinates [lat, lon].

        Returns:
            A dictionary containing route data.
        """
        url = f"{self.base_url}/directions/v2/route"
        params = {
            "key": self.api_key,
            "from": ",".join(map(str, start_coords)),
            "to": ",".join(map(str, end_coords)),
            "outFormat": "json",
        }
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error fetching route data from MapQuest: {e}")
            return {}
