from typing import Any, Dict, List

import requests

from .base import MapService


class GoogleMapsService(MapService):
    """Google Maps API service implementation."""

    def __init__(self, api_key: str):
        super().__init__(api_key)
        self.base_url = "https://maps.googleapis.com/maps/api"

    def get_traffic_incidents(self, bounding_box: List[float]) -> Dict[str, Any]:
        """
        Get traffic incidents from Google Maps.
        Note: Google Maps API (standard) doesn't have a direct "traffic incidents in a bounding box" endpoint.
        This is often a premium feature. We will simulate this by checking the traffic on a route.
        For a more advanced implementation, you'd use the Directions API and look for traffic information.
        This is a placeholder to show the structure.
        """
        # A true implementation would likely require the premium plan.
        # As a workaround, we can get a route and check the 'duration_in_traffic'.
        # This is not ideal for a general bounding box but fits the pattern.
        # Let's use the get_route method to get traffic data for a diagonal of the box.
        start_coords = bounding_box[:2]
        end_coords = bounding_box[2:]
        return self.get_route(start_coords, end_coords)

    def get_route(
        self, start_coords: List[float], end_coords: List[float]
    ) -> Dict[str, Any]:
        """
        Get a route from Google Maps.

        Args:
            start_coords: The starting coordinates [lat, lon].
            end_coords: The ending coordinates [lat, lon].

        Returns:
            A dictionary containing route data.
        """
        url = f"{self.base_url}/directions/json"
        params = {
            "origin": ",".join(map(str, start_coords)),
            "destination": ",".join(map(str, end_coords)),
            "departure_time": "now",
            "traffic_model": "best_guess",
            "key": self.api_key,
        }
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error fetching route data from Google Maps: {e}")
            return {}
