from typing import Any, Dict, List

import requests

from .base import MapService


class TomTomService(MapService):
    """TomTom API service implementation."""

    def __init__(self, api_key: str):
        super().__init__(api_key)
        self.base_url = "https://api.tomtom.com"

    def get_traffic_incidents(self, bounding_box: List[float]) -> Dict[str, Any]:
        """
        Get traffic incidents from TomTom using API v5.

        Args:
            bounding_box: A list of four coordinates [lat1, lon1, lat2, lon2].
                          This will be converted to TomTom's expected format.

        Returns:
            A dictionary containing traffic incident data.
        """
        # TomTom v5 API expects minLon,minLat,maxLon,maxLat format for bbox parameter
        min_lat, min_lon, max_lat, max_lon = bounding_box
        tomtom_bbox = f"{min_lon},{min_lat},{max_lon},{max_lat}"

        # Use TomTom Traffic API v5 endpoint
        url = f"{self.base_url}/traffic/services/5/incidentDetails"
        params = {
            "key": self.api_key,
            "bbox": tomtom_bbox,
            "fields": "{incidents{type,geometry{type,coordinates},properties{iconCategory,magnitudeOfDelay}}}",
            "language": "en-GB",
            "timeValidityFilter": "present",
        }

        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error fetching traffic data from TomTom: {e}")
            return {}

    def get_route(
        self, start_coords: List[float], end_coords: List[float]
    ) -> Dict[str, Any]:
        """
        Get a route from TomTom.

        Args:
            start_coords: The starting coordinates [lat, lon].
            end_coords: The ending coordinates [lat, lon].

        Returns:
            A dictionary containing route data.
        """
        start_str = ",".join(map(str, start_coords))
        end_str = ",".join(map(str, end_coords))

        url = f"{self.base_url}/routing/1/calculateRoute/{start_str}:{end_str}/json"
        params = {"key": self.api_key}
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error fetching route data from TomTom: {e}")
            return {}
