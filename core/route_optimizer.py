"""
Route Optimization using Real-World Road Network
Implements real-time path finding with NetworkX graph and OSMnx
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
import osmnx as ox

from core.exceptions import DataValidationError, NetworkError, RouteNotFoundError
from core.logger import get_logger
from core.map_data_loader import get_map_data_loader

logger = get_logger(__name__)


@dataclass
class RouteResult:
    """Result of a route calculation"""

    path_nodes: List[int]
    total_distance: float  # meters
    total_time: float  # seconds
    parking_zone_id: str
    route_coordinates: List[Tuple[float, float]]  # [(lat, lon), ...]


class RouteOptimizer:
    """
    Real-world route optimization using OSMnx and NetworkX.
    Finds optimal routes on actual Grand Rapids road network.
    """

    def __init__(self):
        self.map_loader = get_map_data_loader()
        self.road_graph: Optional[nx.MultiDiGraph] = None
        self.parking_zones: List[Dict] = []
        self._connected_components = None

        # Initialize if data is available
        if self.map_loader.is_data_available():
            self._initialize_from_map_data()
        else:
            logger.warning(
                "Map data not available. RouteOptimizer operating in fallback mode."
            )

    def _initialize_from_map_data(self):
        """Initialize with real-world map data."""
        self.road_graph = self.map_loader.road_network
        self.parking_zones = self.map_loader.get_parking_zones()

        # Analyze network connectivity
        import networkx as nx

        self._connected_components = list(
            nx.strongly_connected_components(self.road_graph)
        )

        logger.info(
            f"RouteOptimizer initialized with {len(self.road_graph.nodes())} road nodes "
            f"and {len(self.parking_zones)} parking zones"
        )
        logger.info(
            f"Road network has {len(self._connected_components)} connected components"
        )

    def find_optimal_route(
        self,
        start_location: Tuple[float, float],
        available_parking_zones: Optional[List[str]] = None,
        preferences: Optional[Dict] = None,
    ) -> Optional[RouteResult]:
        """
        Find optimal route from start location to best parking zone.

        Args:
            start_location: Starting coordinates (lat, lon)
            available_parking_zones: List of zone IDs to consider. If None, considers all.
            preferences: User preferences (e.g., prefer_distance, price_sensitivity)

        Returns:
            RouteResult with optimal path and details, or None if no route found.
        """
        if not self.road_graph:
            raise NetworkError("Road graph not available", {"road_graph_loaded": False})

        try:
            # Find nearest road network node to start location
            start_node = self._find_nearest_node(start_location)
            if start_node is None:
                raise DataValidationError(
                    "start_location",
                    str(start_location),
                    "location with nearby road network node",
                )

            # Filter parking zones
            candidate_zones = self._filter_parking_zones(available_parking_zones)
            if not candidate_zones:
                raise RouteNotFoundError(
                    "No available parking zones found",
                    {"available_zones": len(candidate_zones)},
                )

            # Filter by connectivity to avoid unreachable zones
            candidate_zones = self._filter_zones_by_connectivity(
                start_node, candidate_zones
            )
            if not candidate_zones:
                raise RouteNotFoundError(
                    "No reachable parking zones found - road network fragmentation",
                    {"start_node": start_node, "reachable_zones": 0},
                )

            # Find best route among all candidate zones
            best_result = None
            best_score = float("inf")

            for zone in candidate_zones:
                # Find nearest road node to parking zone
                zone_node = self._find_nearest_node(zone["coordinates"])
                if zone_node is None:
                    continue

                # Calculate route
                route = self._calculate_route(start_node, zone_node, zone, preferences)
                if route is None:
                    continue

                # Score this route
                score = self._score_route(route, preferences)
                if score < best_score:
                    best_score = score
                    best_result = route

            return best_result

        except Exception as e:
            logger.error(f"Error in route optimization: {e}")
            return None

    def _find_nearest_node(self, location: Tuple[float, float]) -> Optional[int]:
        """Find the nearest road network node to a location."""
        try:
            lat, lon = location
            # Use OSMnx to find nearest node
            nearest_node = ox.nearest_nodes(self.road_graph, lon, lat)
            return nearest_node
        except Exception as e:
            logger.error(f"Error finding nearest node: {e}")
            return None

    def _filter_parking_zones(self, zone_ids: Optional[List[str]]) -> List[Dict]:
        """Filter parking zones based on availability."""
        if zone_ids is None:
            return self.parking_zones

        return [zone for zone in self.parking_zones if zone["id"] in zone_ids]

    def _get_component_for_node(self, node_id: int) -> Optional[set]:
        """Find which connected component a node belongs to."""
        if not self._connected_components:
            return None

        for component in self._connected_components:
            if node_id in component:
                return component
        return None

    def _filter_zones_by_connectivity(
        self, start_node: int, zones: List[Dict]
    ) -> List[Dict]:
        """Filter parking zones to only include those reachable from start_node."""
        if not self._connected_components:
            return zones

        start_component = self._get_component_for_node(start_node)
        if not start_component:
            return zones

        reachable_zones = []
        for zone in zones:
            zone_node = self._find_nearest_node(zone["coordinates"])
            if zone_node and zone_node in start_component:
                reachable_zones.append(zone)

        logger.debug(
            f"Filtered {len(zones)} zones to {len(reachable_zones)} reachable zones"
        )
        return reachable_zones

    def _calculate_route(
        self,
        start_node: int,
        end_node: int,
        parking_zone: Dict,
        preferences: Optional[Dict] = None,
    ) -> Optional[RouteResult]:
        """Calculate route between two nodes."""
        try:
            # Use NetworkX shortest path with different weight options
            weight = self._get_routing_weight(preferences)

            # Calculate shortest path
            path_nodes = nx.shortest_path(
                self.road_graph, start_node, end_node, weight=weight
            )

            # Calculate route metrics
            total_distance = 0
            total_time = 0
            route_coordinates = []

            for i in range(len(path_nodes) - 1):
                edge_data = self.road_graph[path_nodes[i]][path_nodes[i + 1]]

                # Handle MultiDiGraph - get first edge data
                if isinstance(edge_data, dict) and len(edge_data) > 0:
                    edge_attrs = next(iter(edge_data.values()))
                else:
                    edge_attrs = edge_data

                # Accumulate distance and time
                distance = edge_attrs.get("length", 0)  # meters
                total_distance += distance

                # Estimate travel time (speed in km/h, convert to m/s)
                speed_kmh = edge_attrs.get("speed_kph", 50)  # Default 50 km/h
                if isinstance(speed_kmh, list):
                    speed_kmh = speed_kmh[0] if speed_kmh else 50

                speed_ms = speed_kmh / 3.6
                travel_time = (
                    distance / speed_ms if speed_ms > 0 else distance / 13.89
                )  # fallback ~50kmh
                total_time += travel_time

            # Get coordinates for each node in path
            for node_id in path_nodes:
                node_data = self.road_graph.nodes[node_id]
                route_coordinates.append((node_data["y"], node_data["x"]))  # (lat, lon)

            return RouteResult(
                path_nodes=path_nodes,
                total_distance=total_distance,
                total_time=total_time,
                parking_zone_id=parking_zone["id"],
                route_coordinates=route_coordinates,
            )

        except nx.NetworkXNoPath:
            logger.warning(f"No path found between nodes {start_node} and {end_node}")
            return None
        except Exception as e:
            logger.error(f"Error calculating route: {e}")
            return None

    def _get_routing_weight(self, preferences: Optional[Dict] = None) -> str:
        """Determine which edge attribute to use for routing weight."""
        if not preferences:
            return "length"  # Default to shortest distance

        if preferences.get("prefer_time", False):
            return "travel_time"
        elif preferences.get("prefer_distance", True):
            return "length"
        else:
            return "length"

    def _score_route(
        self, route: RouteResult, preferences: Optional[Dict] = None
    ) -> float:
        """
        Score a route based on multiple factors.
        Lower score is better.
        """
        if not preferences:
            # Default scoring: distance + time
            return route.total_distance + (route.total_time * 10)

        score = 0

        # Distance component
        distance_weight = preferences.get("distance_weight", 1.0)
        score += route.total_distance * distance_weight

        # Time component
        time_weight = preferences.get("time_weight", 10.0)
        score += route.total_time * time_weight

        # Parking cost component (if available)
        if "price_sensitivity" in preferences:
            parking_zone = next(
                (z for z in self.parking_zones if z["id"] == route.parking_zone_id),
                None,
            )
            if parking_zone:
                hourly_rate = parking_zone.get("hourly_rate", 2.0)
                price_weight = preferences.get("price_sensitivity", 1.0)
                score += hourly_rate * price_weight * 100  # Scale for comparison

        return score

    def find_optimal_parking(
        self,
        start_location: Tuple[float, float],
        preferences: Optional[Dict] = None,
    ) -> Dict[str, RouteResult]:
        """
        Find optimal parking options with multiple choices.

        Returns:
            Dict mapping zone_id to RouteResult for available parking options,
            sorted by preference score.
        """
        if not self.road_graph:
            logger.error("Road graph not available")
            return {}

        try:
            # Find nearest road network node to start location
            start_node = self._find_nearest_node(start_location)
            if start_node is None:
                logger.error("Could not find nearest road node to start location")
                return {}

            # Get all candidate zones and filter by connectivity
            candidate_zones = self._filter_parking_zones(None)  # Get all zones
            candidate_zones = self._filter_zones_by_connectivity(
                start_node, candidate_zones
            )

            if not candidate_zones:
                logger.warning("No reachable parking zones found")
                return {}

            # Calculate routes to multiple zones (up to 5 best options)
            route_options = {}
            route_scores = []

            for zone in candidate_zones:
                # Find nearest road node to parking zone
                zone_node = self._find_nearest_node(zone["coordinates"])
                if zone_node is None:
                    continue

                # Calculate route
                route = self._calculate_route(start_node, zone_node, zone, preferences)
                if route is None:
                    continue

                # Score this route
                score = self._score_route(route, preferences)
                route_scores.append((score, zone["id"], route))

            # Sort by score and return top options
            route_scores.sort(key=lambda x: x[0])  # Lower score is better

            # Return up to 20 best options (increased from 5)
            for _score, zone_id, route in route_scores[:20]:
                route_options[zone_id] = route

            logger.debug(
                f"Found {len(route_options)} parking options for location {start_location}"
            )
            return route_options

        except Exception as e:
            logger.error(f"Error in find_optimal_parking: {e}")
            return {}

    def calculate_route_to_specific_zone(
        self, start_location: Tuple[float, float], parking_zone_id: str
    ) -> Optional[RouteResult]:
        """Calculate route to a specific parking zone."""
        zone = next((z for z in self.parking_zones if z["id"] == parking_zone_id), None)
        if not zone:
            logger.error(f"Parking zone {parking_zone_id} not found")
            return None

        start_node = self._find_nearest_node(start_location)
        end_node = self._find_nearest_node(zone["coordinates"])

        if start_node is None or end_node is None:
            return None

        return self._calculate_route(start_node, end_node, zone)

    def get_available_parking_zones_in_radius(
        self, center_location: Tuple[float, float], radius_km: float = 2.0
    ) -> List[str]:
        """Get parking zone IDs within a certain radius of a location."""
        center_lat, center_lon = center_location
        available_zones = []

        for zone in self.parking_zones:
            zone_lat, zone_lon = zone["coordinates"]

            # Calculate approximate distance using Haversine-like formula
            lat_diff = np.radians(zone_lat - center_lat)
            lon_diff = np.radians(zone_lon - center_lon)

            a = (
                np.sin(lat_diff / 2) ** 2
                + np.cos(np.radians(center_lat))
                * np.cos(np.radians(zone_lat))
                * np.sin(lon_diff / 2) ** 2
            )
            distance_km = 2 * 6371 * np.arcsin(np.sqrt(a))  # Earth radius = 6371 km

            if distance_km <= radius_km:
                available_zones.append(zone["id"])

        return available_zones

    def update_traffic_conditions(self, traffic_data: Dict[str, float]):
        """
        Update edge weights based on current traffic conditions.

        Args:
            traffic_data: Dict mapping edge identifiers to traffic multipliers
        """
        if not self.road_graph:
            return

        # Update edge weights based on traffic data
        for _edge_id, _traffic_factor in traffic_data.items():
            # This would need to be implemented based on how traffic data maps to edges
            # For now, it's a placeholder for the interface
            pass

    def get_route_summary(self, route: RouteResult) -> Dict:
        """Get a summary of route information."""
        if not route:
            return {}

        return {
            "total_distance_km": route.total_distance / 1000,
            "total_time_minutes": route.total_time / 60,
            "parking_zone_id": route.parking_zone_id,
            "num_road_segments": len(route.path_nodes) - 1,
            "start_coordinate": route.route_coordinates[0]
            if route.route_coordinates
            else None,
            "end_coordinate": route.route_coordinates[-1]
            if route.route_coordinates
            else None,
        }

    def is_operational(self) -> bool:
        """Check if the route optimizer is ready for use."""
        return self.road_graph is not None and len(self.parking_zones) > 0
