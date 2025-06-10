"""
Map Data Loader for Real-World Geographic Data
Handles loading and processing of Grand Rapids map data including road networks,
parking infrastructure, and city boundaries.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import geopandas as gpd
import networkx as nx
import numpy as np
import osmnx as ox
from shapely.geometry import Point

from core.config import get_config
from core.exceptions import ConfigurationError, DataValidationError, NetworkError
from core.logger import get_logger

logger = get_logger(__name__)


class MapDataLoader:
    """Loads and manages real-world map data for Grand Rapids simulation."""

    def __init__(self, data_dir: Optional[str] = None):
        """
        Initialize the map data loader.

        Args:
            data_dir: Directory containing map data files. If None, uses default.
        """
        config = get_config()
        self.data_dir = Path(data_dir) if data_dir else Path("output/map_data")

        # Initialize data containers
        self.road_network: Optional[nx.MultiDiGraph] = None
        self.parking_lots_gdf: Optional[gpd.GeoDataFrame] = None
        self.parking_meters_gdf: Optional[gpd.GeoDataFrame] = None
        self.boundary_gdf: Optional[gpd.GeoDataFrame] = None

        # Cached data
        self._network_nodes: Optional[List[int]] = None
        self._parking_zones: Optional[List[Dict]] = None

        logger.info(f"MapDataLoader initialized with data directory: {self.data_dir}")

    def load_all_data(self) -> bool:
        """
        Load all map data files.

        Returns:
            True if all data loaded successfully, False otherwise.
        """
        try:
            logger.info("Loading all Grand Rapids map data...")

            # Load road network
            self.load_road_network()

            # Load parking data
            self.load_parking_data()

            # Load boundary
            self.load_boundary()

            logger.info("All map data loaded successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to load map data: {e}")
            return False

    def load_road_network(self) -> nx.MultiDiGraph:
        """Load the road network graph from GraphML file."""
        network_file = self.data_dir / "grand_rapids_drive_network.graphml"

        if not network_file.exists():
            raise DataValidationError(
                "network_file", str(network_file), "existing file path"
            )

        logger.info(f"Loading road network from {network_file}")
        self.road_network = ox.load_graphml(network_file)

        # Cache network nodes for random sampling
        self._network_nodes = list(self.road_network.nodes())

        logger.info(
            f"Road network loaded: {len(self.road_network.nodes())} nodes, "
            f"{len(self.road_network.edges())} edges"
        )

        return self.road_network

    def load_parking_data(self) -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
        """Load parking lots and meters data."""
        lots_file = self.data_dir / "parking_lots.geojson"
        meters_file = self.data_dir / "parking_meters.geojson"

        if not lots_file.exists():
            raise DataValidationError(
                "parking_lots_file", str(lots_file), "existing file path"
            )
        if not meters_file.exists():
            raise DataValidationError(
                "parking_meters_file", str(meters_file), "existing file path"
            )

        logger.info("Loading parking infrastructure data...")

        self.parking_lots_gdf = gpd.read_file(lots_file)
        self.parking_meters_gdf = gpd.read_file(meters_file)

        logger.info(
            f"Loaded {len(self.parking_lots_gdf)} parking lots and "
            f"{len(self.parking_meters_gdf)} meter zones"
        )

        return self.parking_lots_gdf, self.parking_meters_gdf

    def load_boundary(self) -> gpd.GeoDataFrame:
        """Load the downtown boundary."""
        boundary_file = self.data_dir / "downtown_boundary.geojson"

        if not boundary_file.exists():
            raise DataValidationError(
                "boundary_file", str(boundary_file), "existing file path"
            )

        logger.info(f"Loading downtown boundary from {boundary_file}")
        self.boundary_gdf = gpd.read_file(boundary_file)

        return self.boundary_gdf

    def get_random_network_node(self) -> int:
        """Get a random node from the road network."""
        if not self._network_nodes:
            raise NetworkError(
                "Road network not loaded", {"operation": "get_random_node"}
            )

        return np.random.choice(self._network_nodes)

    def get_network_nodes_in_boundary(
        self, sample_size: Optional[int] = None
    ) -> List[int]:
        """
        Get network nodes within the downtown boundary.

        Args:
            sample_size: If provided, return a random sample of this size.

        Returns:
            List of node IDs within the boundary.
        """
        if self.road_network is None or self.boundary_gdf is None:
            raise ConfigurationError(
                "Road network and boundary must be loaded first",
                {
                    "road_network_loaded": self.road_network is not None,
                    "boundary_loaded": self.boundary_gdf is not None,
                },
            )

        # Get boundary polygon
        boundary_polygon = self.boundary_gdf.union_all()

        # Find nodes within boundary
        nodes_in_boundary = []
        for node_id in self.road_network.nodes():
            node_data = self.road_network.nodes[node_id]
            point = Point(node_data["x"], node_data["y"])

            if boundary_polygon.contains(point):
                nodes_in_boundary.append(node_id)

        logger.info(
            f"Found {len(nodes_in_boundary)} network nodes within downtown boundary"
        )

        if sample_size and sample_size < len(nodes_in_boundary):
            nodes_in_boundary = np.random.choice(
                nodes_in_boundary, size=sample_size, replace=False
            ).tolist()

        return nodes_in_boundary

    def get_node_coordinates(self, node_id: int) -> Tuple[float, float]:
        """Get the lat/lon coordinates of a network node."""
        if self.road_network is None:
            raise NetworkError(
                "Road network not loaded", {"operation": "get_node_coordinates"}
            )

        node_data = self.road_network.nodes[node_id]
        return (node_data["y"], node_data["x"])  # (lat, lon)

    def get_parking_zones(self) -> List[Dict]:
        """
        Get standardized parking zone data combining lots and meters.

        Returns:
            List of parking zone dictionaries with standardized format.
        """
        if self._parking_zones is not None:
            return self._parking_zones

        if self.parking_lots_gdf is None or self.parking_meters_gdf is None:
            raise ConfigurationError(
                "Parking data not loaded",
                {
                    "lots_loaded": self.parking_lots_gdf is not None,
                    "meters_loaded": self.parking_meters_gdf is not None,
                },
            )

        zones = []

        # Process parking lots
        for idx, lot in self.parking_lots_gdf.iterrows():
            # Get centroid coordinates
            centroid = lot.geometry.centroid

            # Extract capacity (might need adjustment based on actual data structure)
            capacity = self._extract_capacity(lot)

            zone = {
                "id": f"lot_{idx}",
                "type": "lot",
                "coordinates": (centroid.y, centroid.x),  # (lat, lon)
                "capacity": capacity,
                "hourly_rate": self._extract_rate(lot),
                "geometry": lot.geometry,
                "properties": dict(lot.drop("geometry"))
                if hasattr(lot, "drop")
                else {},
            }
            zones.append(zone)

        # Process parking meters
        for idx, meter in self.parking_meters_gdf.iterrows():
            centroid = meter.geometry.centroid

            zone = {
                "id": f"meter_{idx}",
                "type": "meter",
                "coordinates": (centroid.y, centroid.x),  # (lat, lon)
                "capacity": self._extract_meter_capacity(meter),
                "hourly_rate": self._extract_meter_rate(meter),
                "geometry": meter.geometry,
                "properties": dict(meter.drop("geometry"))
                if hasattr(meter, "drop")
                else {},
            }
            zones.append(zone)

        self._parking_zones = zones
        logger.info(
            f"Processed {len(zones)} parking zones ({len(self.parking_lots_gdf)} lots, "
            f"{len(self.parking_meters_gdf)} meter zones)"
        )

        return zones

    def _extract_capacity(self, lot_row) -> int:
        """Extract capacity from parking lot data."""
        # This may need adjustment based on actual data structure
        capacity_fields = ["capacity", "spaces", "total_spaces", "num_spaces"]

        for field in capacity_fields:
            if field in lot_row and lot_row[field] is not None:
                try:
                    return int(lot_row[field])
                except (ValueError, TypeError):
                    continue

        # Default capacity based on lot size if no explicit capacity
        area = lot_row.geometry.area
        # Rough estimate: 25 sq meters per parking space
        return max(10, int(area / 25))

    def _extract_rate(self, lot_row) -> float:
        """Extract hourly rate from parking lot data."""
        rate_fields = ["hourly_rate", "rate", "price", "cost"]

        for field in rate_fields:
            if field in lot_row and lot_row[field] is not None:
                try:
                    return float(lot_row[field])
                except (ValueError, TypeError):
                    continue

        # Default rate for lots
        return 2.0

    def _extract_meter_capacity(self, meter_row) -> int:
        """Extract capacity from meter zone data."""
        # Meters typically have fewer spaces
        capacity_fields = ["spaces", "capacity", "num_spaces"]

        for field in capacity_fields:
            if field in meter_row and meter_row[field] is not None:
                try:
                    return int(meter_row[field])
                except (ValueError, TypeError):
                    continue

        # Default meter capacity
        return 8

    def _extract_meter_rate(self, meter_row) -> float:
        """Extract hourly rate from meter data."""
        rate_fields = ["rate", "hourly_rate", "price", "cost", "Rate"]

        for field in rate_fields:
            if field in meter_row and meter_row[field] is not None:
                try:
                    # Handle various rate formats (per hour, per minute, etc.)
                    rate = float(meter_row[field])
                    # If it looks like per-minute rate, convert to hourly
                    if rate < 1.0:
                        rate *= 60
                    return rate
                except (ValueError, TypeError):
                    continue

        # Default meter rate
        return 1.50

    def is_data_available(self) -> bool:
        """Check if all required data is loaded."""
        return all(
            [
                self.road_network is not None,
                self.parking_lots_gdf is not None,
                self.parking_meters_gdf is not None,
                self.boundary_gdf is not None,
            ]
        )

    def get_simulation_bounds(self) -> Tuple[float, float, float, float]:
        """
        Get the bounding box for the simulation area.

        Returns:
            Tuple of (min_lat, min_lon, max_lat, max_lon)
        """
        if self.boundary_gdf is None:
            raise ConfigurationError(
                "Boundary data not loaded", {"boundary_loaded": False}
            )

        bounds = self.boundary_gdf.total_bounds
        # bounds format: [minx, miny, maxx, maxy] (lon, lat, lon, lat)
        return (
            bounds[1],
            bounds[0],
            bounds[3],
            bounds[2],
        )  # (min_lat, min_lon, max_lat, max_lon)


# Global instance
_map_data_loader: Optional[MapDataLoader] = None


def get_map_data_loader() -> MapDataLoader:
    """Get the global map data loader instance."""
    global _map_data_loader
    if _map_data_loader is None:
        _map_data_loader = MapDataLoader()
        _map_data_loader.load_all_data()
    return _map_data_loader


def reload_map_data(data_dir: Optional[str] = None):
    """Reload map data from files."""
    global _map_data_loader
    _map_data_loader = MapDataLoader(data_dir)
    _map_data_loader.load_all_data()
    return _map_data_loader
