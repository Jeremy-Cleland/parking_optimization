"""
Map Data Loader for Real-World Geographic Data
Handles loading and processing of Grand Rapids map data including road networks,
parking infrastructure, and city boundaries.
"""

import hashlib
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import geopandas as gpd
import networkx as nx
import numpy as np
import osmnx as ox
import pandas as pd
from pyproj import CRS
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
        get_config()
        self.data_dir = Path(data_dir) if data_dir else Path("output/map_data")

        # Initialize data containers
        self.road_network: Optional[nx.MultiDiGraph] = None
        self.parking_lots_gdf: Optional[gpd.GeoDataFrame] = None
        self.parking_meters_gdf: Optional[gpd.GeoDataFrame] = None
        self.boundary_gdf: Optional[gpd.GeoDataFrame] = None

        # Cached data
        self._network_nodes: Optional[List[int]] = None
        self._parking_zones: Optional[List[Dict]] = None
        self.validation_report = {
            "status": "not_loaded",
            "provenance": "Local files; original publisher and retrieval date are unverified unless supplied separately",
            "warnings": [],
            "files": {},
            "zones": [],
        }

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
            self.get_parking_zones()
            for path in sorted(self.data_dir.iterdir()):
                if path.suffix in {".geojson", ".graphml"}:
                    self.validation_report["files"][path.name] = hashlib.sha256(
                        path.read_bytes()
                    ).hexdigest()

            self.validation_report["status"] = "validated_with_assumptions"
            logger.info("All map data loaded successfully")
            return True

        except Exception as e:
            self.validation_report["status"] = "failed"
            self.validation_report["error"] = str(e)
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
        if CRS(self.road_network.graph.get("crs")) != CRS("EPSG:4326"):
            raise ValueError(
                "Road network must use EPSG:4326 latitude/longitude coordinates"
            )
        for _node, attrs in self.road_network.nodes(data=True):
            self._validate_coordinates((attrs.get("y"), attrs.get("x")))
        for _u, _v, attrs in self.road_network.edges(data=True):
            if (
                not math.isfinite(float(attrs.get("length", -1)))
                or float(attrs.get("length", -1)) < 0
            ):
                raise ValueError(
                    "Every road edge requires finite nonnegative length in meters"
                )

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

        self.parking_lots_gdf = self._normalize_geometry(
            gpd.read_file(lots_file), "lots"
        )
        self.parking_meters_gdf = self._normalize_geometry(
            gpd.read_file(meters_file), "meters"
        )

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
        self.boundary_gdf = self._normalize_geometry(
            gpd.read_file(boundary_file), "boundary"
        )

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
            capacity, capacity_source = self._validated_capacity(
                lot, self.parking_lots_gdf.crs, False
            )
            hourly_rate, rate_source = self._validated_rate(lot, 2.0)

            zone = {
                "id": f"lot_{idx}",
                "type": "lot",
                "coordinates": (centroid.y, centroid.x),  # (lat, lon)
                "capacity": capacity,
                "hourly_rate": hourly_rate,
                "accessible": self._accessible(lot),
                "geometry": lot.geometry,
                "properties": dict(lot.drop("geometry"))
                if hasattr(lot, "drop")
                else {},
            }
            zones.append(zone)
            self.validation_report["zones"].append(
                {
                    "id": zone["id"],
                    "capacity_source": capacity_source,
                    "rate_source": rate_source,
                }
            )

        # Process parking meters
        for idx, meter in self.parking_meters_gdf.iterrows():
            centroid = meter.geometry.centroid

            capacity, capacity_source = self._validated_capacity(
                meter, self.parking_meters_gdf.crs, True
            )
            hourly_rate, rate_source = self._validated_rate(meter, 1.5)
            zone = {
                "id": f"meter_{idx}",
                "type": "meter",
                "coordinates": (centroid.y, centroid.x),  # (lat, lon)
                "capacity": capacity,
                "hourly_rate": hourly_rate,
                "accessible": self._accessible(meter),
                "geometry": meter.geometry,
                "properties": dict(meter.drop("geometry"))
                if hasattr(meter, "drop")
                else {},
            }
            zones.append(zone)
            self.validation_report["zones"].append(
                {
                    "id": zone["id"],
                    "capacity_source": capacity_source,
                    "rate_source": rate_source,
                }
            )

        if not zones:
            raise ValueError("Parking dataset has no zones")
        for zone in zones:
            self._validate_coordinates(zone["coordinates"])
        self._parking_zones = zones
        logger.info(
            f"Processed {len(zones)} parking zones ({len(self.parking_lots_gdf)} lots, "
            f"{len(self.parking_meters_gdf)} meter zones)"
        )

        return zones

    @staticmethod
    def _validate_coordinates(coordinates):
        lat, lon = coordinates
        if (
            lat is None
            or lon is None
            or not math.isfinite(float(lat))
            or not math.isfinite(float(lon))
            or not -90 <= float(lat) <= 90
            or not -180 <= float(lon) <= 180
        ):
            raise ValueError("Invalid latitude/longitude coordinates")

    def _normalize_geometry(self, frame, label):
        if frame.crs is None:
            raise ValueError(f"{label} dataset requires an explicit CRS")
        if (
            frame.geometry.isna().any()
            or frame.geometry.is_empty.any()
            or not frame.geometry.is_valid.all()
        ):
            raise ValueError(f"{label} contains missing, empty or invalid geometry")
        if CRS(frame.crs) != CRS("EPSG:4326"):
            self.validation_report["warnings"].append(
                f"{label}: reprojected {frame.crs} to EPSG:4326"
            )
            frame = frame.to_crs("EPSG:4326")
        return frame

    @staticmethod
    def _accessible(row):
        # Unknown accessibility never qualifies as confirmed accessible parking.
        value = row.get("accessible", row.get("wheelchair", False))
        return str(value).lower() in {"yes", "true", "1"}

    def _validated_capacity(self, row, crs, is_meter):
        for field in ["capacity", "spaces", "spots", "total_spaces", "num_spaces"]:
            value = row.get(field)
            if value is not None and not pd.isna(value):
                number = float(value)
                if not math.isfinite(number) or number < 1 or not number.is_integer():
                    raise ValueError(f"Invalid positive integer capacity in {field}")
                return int(number), f"field:{field}"
        geometry = row.geometry
        if geometry.geom_type in {"Polygon", "MultiPolygon"}:
            series = gpd.GeoSeries([geometry], crs=crs)
            metric_crs = series.estimate_utm_crs()
            if metric_crs is None:
                raise ValueError("Cannot select a metric CRS for area-derived capacity")
            capacity = max(1, int(series.to_crs(metric_crs).area.iloc[0] / 25))
            source = "estimate:metric_area_m2/25"
        else:
            capacity = 8 if is_meter else 10
            source = f"assumption:default_{capacity}_spaces"
        self.validation_report["warnings"].append(f"Capacity uses {source}")
        return capacity, source

    def _validated_rate(self, row, default):
        for field in ["hourly_rate", "cost_per_hour", "rate", "Rate", "price", "cost"]:
            value = row.get(field)
            if value is None or pd.isna(value):
                continue
            number = float(value)
            if not math.isfinite(number) or number < 0:
                raise ValueError(f"Invalid nonnegative price in {field}")
            units = row.get("rate_unit", row.get("rate_units"))
            if units is None or pd.isna(units):
                units = "hour"
                if field not in {"hourly_rate", "cost_per_hour"}:
                    self.validation_report["warnings"].append(
                        f"{field}: missing rate unit; assumed USD/hour"
                    )
            units = str(units).lower()
            if units in {"minute", "per_minute", "usd/minute"}:
                number *= 60
            elif units not in {"hour", "hourly", "per_hour", "usd/hour"}:
                raise ValueError(f"Unsupported rate unit: {units}")
            currency = row.get("currency", "USD")
            if str(currency).upper() != "USD":
                raise ValueError("Only explicitly USD rates are supported")
            return number, f"field:{field};USD/{units}"
        self.validation_report["warnings"].append(
            f"Rate missing; assumed USD {default}/hour"
        )
        return default, "assumption:default_USD/hour"

    def _extract_capacity(self, row):
        return self._validated_capacity(row, "EPSG:4326", False)[0]

    def _extract_meter_capacity(self, row):
        return self._validated_capacity(row, "EPSG:4326", True)[0]

    def _extract_rate(self, row):
        return self._validated_rate(row, 2.0)[0]

    def _extract_meter_rate(self, row):
        return self._validated_rate(row, 1.5)[0]

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
