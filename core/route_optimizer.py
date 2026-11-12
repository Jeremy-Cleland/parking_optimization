"""
Route Optimization using Real-World Road Network
Implements real-time path finding with NetworkX graph and OSMnx
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
import osmnx as ox

from core.exceptions import DataValidationError, NetworkError, RouteNotFoundError
from core.logger import get_logger
from core.map_data_loader import MapDataLoader, get_map_data_loader

logger = get_logger(__name__)


@dataclass
class RouteResult:
    """Result of a route calculation"""

    path_nodes: List[int]
    total_distance: float  # meters
    total_time: float  # seconds
    parking_zone_id: str
    route_coordinates: List[Tuple[float, float]]  # [(lat, lon), ...]
    walking_distance: float = 0.0  # meters, straight-line estimate to destination
    estimated_parking_cost: float = 0.0


class RouteOptimizer:
    """
    Real-world route optimization using OSMnx and NetworkX.
    Finds optimal routes on actual Grand Rapids road network.
    """

    def __init__(self, map_loader: Optional[MapDataLoader] = None):
        self.map_loader = (
            map_loader if map_loader is not None else get_map_data_loader()
        )
        self.road_graph: Optional[nx.MultiDiGraph] = None
        self.parking_zones: List[Dict] = []
        self._connected_components = None
        self._nearest_nodes = {}
        self._reachable_nodes = {}
        self._route_paths = {}

        # Initialize if data is available
        if self.map_loader.road_network is not None:
            self._initialize_from_map_data()
        else:
            logger.warning(
                "Map data not available. RouteOptimizer operating in fallback mode."
            )

    def _initialize_from_map_data(self):
        """Initialize with real-world map data."""
        self.road_graph = self.map_loader.road_network
        self.parking_zones = self.map_loader.get_parking_zones()
        self._prepare_edge_weights()

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

    def _prepare_edge_weights(self):
        """Supply travel times before routing, using the same units as RouteResult."""
        for _u, _v, attrs in self.road_graph.edges(data=True):
            if "travel_time" not in attrs:
                speed = attrs.get("speed_kph", 50)
                if isinstance(speed, list):
                    speed = speed[0] if speed else 50
                try:
                    speed = float(speed)
                except (ValueError, TypeError):
                    speed = 50
                if not np.isfinite(speed) or speed <= 0:
                    speed = 50
                attrs["travel_time"] = attrs.get("length", 0) / (speed / 3.6)
            attrs["base_travel_time"] = attrs["travel_time"]

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
            location = tuple(location)
            if location in self._nearest_nodes:
                return self._nearest_nodes[location]
            lat, lon = location
            # Use OSMnx to find nearest node
            nearest_node = ox.nearest_nodes(self.road_graph, lon, lat)
            self._nearest_nodes[location] = nearest_node
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
        if self.road_graph is None or start_node not in self.road_graph:
            return []

        # A destination need not have a return path to be reachable on one-way roads.
        if start_node not in self._reachable_nodes:
            self._reachable_nodes[start_node] = nx.descendants(
                self.road_graph, start_node
            ) | {start_node}
        reachable_nodes = self._reachable_nodes[start_node]

        reachable_zones = []
        for zone in zones:
            zone_node = self._find_nearest_node(zone["coordinates"])
            if zone_node is not None and zone_node in reachable_nodes:
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
            preferences = preferences or {}
            destination = preferences.get("destination")
            if "walking_distances" in preferences:
                walking_distance = float(
                    preferences["walking_distances"][
                        preferences["zone_indices"][parking_zone["id"]]
                    ]
                )
            else:
                walking_distance = (
                    self._distance_meters(parking_zone["coordinates"], destination)
                    if destination
                    else 0.0
                )
            if (
                walking_distance
                > preferences.get("max_walk_distance", float("inf")) * 1000
            ):
                return None
            if preferences.get("requires_accessible") and not parking_zone.get(
                "accessible", False
            ):
                return None
            rate = preferences.get("zone_prices", {}).get(
                parking_zone["id"], parking_zone.get("hourly_rate", 0)
            )
            cost = rate * preferences.get("parking_duration_minutes", 60) / 60
            if cost > preferences.get("max_parking_cost", float("inf")):
                return None
            # Use NetworkX shortest path with different weight options
            weight = self._get_routing_weight(preferences)

            # Calculate shortest path
            cache_key = (start_node, end_node, weight)
            if cache_key not in self._route_paths:
                self._route_paths[cache_key] = nx.shortest_path(
                    self.road_graph, start_node, end_node, weight=weight
                )
            path_nodes = list(self._route_paths[cache_key])

            # Calculate route metrics
            total_distance = 0
            total_time = 0
            route_coordinates = []

            for i in range(len(path_nodes) - 1):
                edge_data = self.road_graph[path_nodes[i]][path_nodes[i + 1]]

                # Match NetworkX's minimum-weight parallel edge selection.
                if self.road_graph.is_multigraph():
                    edge_attrs = min(
                        edge_data.values(), key=lambda attrs: attrs.get(weight, 1)
                    )
                else:
                    edge_attrs = edge_data

                # Accumulate distance and time
                distance = edge_attrs.get("length", 0)  # meters
                total_distance += distance

                total_time += edge_attrs["travel_time"]

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
                walking_distance=walking_distance,
                estimated_parking_cost=cost,
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

        if "destination" in preferences:
            # Generalized trip cost in dollars, including the stay and both legs.
            value_of_time = preferences.get("time_value", 15)
            return (
                route.estimated_parking_cost * preferences.get("price_weight", 1)
                + (route.total_time / 3600 + route.walking_distance / 5000)
                * value_of_time
            )

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
        available_parking_zones: Optional[List[str]] = None,
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
            candidate_zones = self._filter_parking_zones(available_parking_zones)
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

    @staticmethod
    def _distance_meters(start, end):
        lat1, lon1 = np.radians(start)
        lat2, lon2 = np.radians(end)
        a = (
            np.sin((lat2 - lat1) / 2) ** 2
            + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
        )
        return float(6371000 * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1))))

    def update_traffic_conditions(
        self, traffic_data: Dict, current_time=None, max_age_seconds=300
    ):
        """
        Update edge weights based on current traffic conditions.

        Args:
            traffic_data: Dict mapping edge identifiers to traffic multipliers
        """
        if not self.road_graph:
            return

        # Update edge weights based on traffic data
        now = current_time or datetime.now()
        self._route_paths = {
            key: value
            for key, value in self._route_paths.items()
            if key[2] != "travel_time"
        }
        for (u, v, key), condition in traffic_data.items():
            if not self.road_graph.has_edge(u, v, key):
                continue
            observed_now = datetime.now() if condition.is_real_data else now
            age = (observed_now - condition.last_updated).total_seconds()
            factor = condition.travel_time_factor
            if (
                not 0 <= age <= max_age_seconds
                or not np.isfinite(factor)
                or factor <= 0
            ):
                continue
            attrs = self.road_graph[u][v][key]
            attrs["travel_time"] = attrs["base_travel_time"] * factor
            attrs["traffic_observed_at"] = condition.last_updated
            attrs["traffic_is_real"] = condition.is_real_data
            attrs["traffic_source"] = condition.source

    def expire_traffic(self, current_time, max_age_seconds=300):
        self._route_paths = {
            key: value
            for key, value in self._route_paths.items()
            if key[2] != "travel_time"
        }
        for _u, _v, attrs in self.road_graph.edges(data=True):
            observed = attrs.get("traffic_observed_at")
            now = datetime.now() if attrs.get("traffic_is_real") else current_time
            if (
                observed is not None
                and not 0 <= (now - observed).total_seconds() <= max_age_seconds
            ):
                attrs["travel_time"] = attrs["base_travel_time"]
                attrs.pop("traffic_observed_at", None)
                attrs["traffic_source"] = "free_flow"

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
