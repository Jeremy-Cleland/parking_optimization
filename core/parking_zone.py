"""
Parking Zone Management Module
Implements parking zones with real-world data integration and state tracking
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np

from core.logger import get_logger
from core.map_data_loader import get_map_data_loader

logger = get_logger(__name__)


@dataclass
class ParkingSpot:
    """Individual parking spot representation"""

    id: str
    zone_id: str
    location: Tuple[float, float]  # (lat, lon)
    is_occupied: bool = False
    vehicle_id: str = None
    occupied_since: datetime = None
    spot_type: str = "standard"  # standard, handicap, ev_charging

    def occupy(self, vehicle_id: str):
        """Mark spot as occupied"""
        self.is_occupied = True
        self.vehicle_id = vehicle_id
        self.occupied_since = datetime.now()

    def release(self):
        """Mark spot as available"""
        self.is_occupied = False
        self.vehicle_id = None
        self.occupied_since = None


@dataclass
class ParkingZone:
    """
    Represents a parking zone in the city
    Integrates with real-world Grand Rapids parking data and tracks occupancy, pricing, and demand patterns
    """

    id: str
    name: str
    location: Tuple[float, float]  # Center point (lat, lon)
    capacity: int
    base_price: float  # Base price per hour
    spots: List[ParkingSpot] = field(default_factory=list)

    # Historical data for DP predictions
    occupancy_history: List[float] = field(default_factory=list)
    demand_history: List[int] = field(default_factory=list)

    # Real-time metrics
    current_price: float = 0.0
    last_update: datetime = None

    # Zone characteristics
    zone_type: str = "standard"  # commercial, residential, event, lot, meter
    nearby_amenities: List[str] = field(default_factory=list)

    # Real-world data attributes
    real_world_data: Optional[Dict] = None
    geometry: Optional[object] = None  # Shapely geometry object

    def __post_init__(self):
        """Initialize parking spots if not provided"""
        if not self.spots:
            for i in range(self.capacity):
                spot = ParkingSpot(
                    id=f"{self.id}_spot_{i}",
                    zone_id=self.id,
                    location=self._generate_spot_location(i),
                )
                self.spots.append(spot)
        self.current_price = self.base_price
        self.last_update = datetime.now()

    def _generate_spot_location(self, index: int) -> Tuple[float, float]:
        """Generate spot location based on zone center or geometry"""
        if self.geometry and hasattr(self.geometry, "bounds"):
            # Use real geometry bounds for more realistic spot distribution
            minx, miny, maxx, maxy = self.geometry.bounds

            # Create a grid within the geometry bounds
            grid_size = int(np.ceil(np.sqrt(self.capacity)))
            row = index // grid_size
            col = index % grid_size

            # Calculate position within bounds
            x_step = (maxx - minx) / grid_size if grid_size > 1 else 0
            y_step = (maxy - miny) / grid_size if grid_size > 1 else 0

            lon = minx + (col + 0.5) * x_step
            lat = miny + (row + 0.5) * y_step

            return (lat, lon)
        else:
            # Fallback to simple grid layout around zone center
            row = index // 10
            col = index % 10
            lat_offset = (row - 5) * 0.00001  # ~1 meter
            lon_offset = (col - 5) * 0.00001
            return (self.location[0] + lat_offset, self.location[1] + lon_offset)

    @property
    def occupancy_rate(self) -> float:
        """Calculate current occupancy rate"""
        occupied = sum(1 for spot in self.spots if spot.is_occupied)
        return occupied / self.capacity if self.capacity > 0 else 0.0

    @property
    def available_spots(self) -> List[ParkingSpot]:
        """Get list of available spots"""
        return [spot for spot in self.spots if not spot.is_occupied]

    def find_nearest_available_spot(self, location: Tuple[float, float]) -> ParkingSpot:
        """
        Find nearest available spot to given location
        Uses simple Euclidean distance (could be improved with actual routing)
        """
        available = self.available_spots
        if not available:
            return None

        def distance(spot: ParkingSpot) -> float:
            return np.sqrt(
                (spot.location[0] - location[0]) ** 2
                + (spot.location[1] - location[1]) ** 2
            )

        return min(available, key=distance)

    def update_history(self):
        """Update occupancy and demand history for predictions"""
        self.occupancy_history.append(self.occupancy_rate)
        # Keep last 168 hours (1 week) of data
        if len(self.occupancy_history) > 168:
            self.occupancy_history = self.occupancy_history[-168:]

    def has_availability(self) -> bool:
        """Check if zone has available spots"""
        return len(self.available_spots) > 0

    def occupy_spot(self, vehicle_id: Optional[str] = None) -> bool:
        """
        Occupy a spot in this zone.

        Args:
            vehicle_id: ID of the vehicle occupying the spot

        Returns:
            True if spot was successfully occupied, False if no spots available
        """
        available = self.available_spots
        if not available:
            return False

        # Occupy the first available spot
        spot = available[0]
        spot.occupy(vehicle_id or f"vehicle_{datetime.now().timestamp()}")
        return True

    def release_spot(self, vehicle_id: Optional[str] = None) -> bool:
        """
        Release a spot in this zone.

        Args:
            vehicle_id: ID of the vehicle to release (if None, releases first occupied spot)

        Returns:
            True if spot was successfully released, False if no occupied spots found
        """
        occupied_spots = [spot for spot in self.spots if spot.is_occupied]

        if not occupied_spots:
            return False

        if vehicle_id:
            # Find specific vehicle
            for spot in occupied_spots:
                if spot.vehicle_id == vehicle_id:
                    spot.release()
                    return True
            return False
        else:
            # Release first occupied spot
            occupied_spots[0].release()
            return True

    @property
    def occupied_spots(self) -> int:
        """Get number of occupied spots"""
        return sum(1 for spot in self.spots if spot.is_occupied)

    @property
    def hourly_rate(self) -> float:
        """Get current hourly rate (alias for current_price)"""
        return self.current_price

    def get_stats(self) -> Dict:
        """Get zone statistics"""
        stats = {
            "id": self.id,
            "name": self.name,
            "occupancy_rate": self.occupancy_rate,
            "available_spots": len(self.available_spots),
            "current_price": self.current_price,
            "zone_type": self.zone_type,
            "last_update": self.last_update.isoformat() if self.last_update else None,
        }

        # Add real-world data if available
        if self.real_world_data:
            stats["real_world_properties"] = self.real_world_data

        return stats

    @classmethod
    def from_real_world_data(cls, zone_data: Dict) -> "ParkingZone":
        """
        Create a ParkingZone from real-world parking data.

        Args:
            zone_data: Dict containing zone information from MapDataLoader

        Returns:
            ParkingZone instance with real-world data
        """
        zone = cls(
            id=zone_data["id"],
            name=zone_data.get("name", zone_data["id"]),
            location=zone_data["coordinates"],
            capacity=zone_data["capacity"],
            base_price=zone_data["hourly_rate"],
            zone_type=zone_data["type"],
            real_world_data=zone_data.get("properties", {}),
            geometry=zone_data.get("geometry"),
        )

        logger.debug(
            f"Created ParkingZone {zone.id} with {zone.capacity} spots at {zone.location}"
        )
        return zone


class ParkingZoneManager:
    """Manages all parking zones in the simulation with real-world data integration."""

    def __init__(self, use_real_data: bool = True):
        """
        Initialize the parking zone manager.

        Args:
            use_real_data: If True, load zones from real-world data. If False, use synthetic data.
        """
        self.zones: Dict[str, ParkingZone] = {}
        self.use_real_data = use_real_data

        if use_real_data:
            self._load_real_world_zones()

        logger.info(
            f"ParkingZoneManager initialized with {len(self.zones)} zones "
            f"({'real-world' if use_real_data else 'synthetic'} data)"
        )

    def _load_real_world_zones(self):
        """Load parking zones from real-world data."""
        try:
            map_loader = get_map_data_loader()
            if not map_loader.is_data_available():
                logger.warning(
                    "Real-world map data not available, falling back to synthetic zones"
                )
                self._create_synthetic_zones()
                return

            # Get parking zones from map data
            zone_data_list = map_loader.get_parking_zones()

            for zone_data in zone_data_list:
                zone = ParkingZone.from_real_world_data(zone_data)
                self.zones[zone.id] = zone

            logger.info(f"Loaded {len(self.zones)} real-world parking zones")

        except Exception as e:
            logger.error(f"Failed to load real-world zones: {e}")
            logger.info("Falling back to synthetic zones")
            self._create_synthetic_zones()

    def _create_synthetic_zones(self):
        """Create synthetic parking zones for fallback."""
        # This creates a basic set of synthetic zones for when real data isn't available
        synthetic_zones = [
            {
                "id": "synthetic_downtown_1",
                "coordinates": (42.9634, -85.6681),
                "capacity": 150,
                "hourly_rate": 2.0,
                "type": "lot",
            },
            {
                "id": "synthetic_downtown_2",
                "coordinates": (42.9640, -85.6675),
                "capacity": 200,
                "hourly_rate": 1.5,
                "type": "lot",
            },
            {
                "id": "synthetic_meters_1",
                "coordinates": (42.9645, -85.6690),
                "capacity": 25,
                "hourly_rate": 1.25,
                "type": "meter",
            },
        ]

        for zone_data in synthetic_zones:
            zone = ParkingZone.from_real_world_data(zone_data)
            self.zones[zone.id] = zone

    def get_zone(self, zone_id: str) -> Optional[ParkingZone]:
        """Get a specific parking zone by ID."""
        return self.zones.get(zone_id)

    def get_all_zones(self) -> List[ParkingZone]:
        """Get all parking zones."""
        return list(self.zones.values())

    def get_zones_by_type(self, zone_type: str) -> List[ParkingZone]:
        """Get zones of a specific type (lot, meter, etc.)."""
        return [zone for zone in self.zones.values() if zone.zone_type == zone_type]

    def get_available_zones(self) -> List[str]:
        """Get IDs of zones with available spots."""
        return [zone.id for zone in self.zones.values() if zone.available_spots]

    def get_zones_in_radius(
        self, center: Tuple[float, float], radius_km: float
    ) -> List[ParkingZone]:
        """Get zones within a radius of a center point."""
        center_lat, center_lon = center
        nearby_zones = []

        for zone in self.zones.values():
            zone_lat, zone_lon = zone.location

            # Calculate distance using Haversine-like formula
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
                nearby_zones.append(zone)

        return nearby_zones

    def update_all_histories(self):
        """Update occupancy history for all zones."""
        for zone in self.zones.values():
            zone.update_history()

    def get_system_stats(self) -> Dict:
        """Get overall system statistics."""
        total_capacity = sum(zone.capacity for zone in self.zones.values())
        total_occupied = sum(
            sum(1 for spot in zone.spots if spot.is_occupied)
            for zone in self.zones.values()
        )

        return {
            "total_zones": len(self.zones),
            "total_capacity": total_capacity,
            "total_occupied": total_occupied,
            "overall_occupancy_rate": total_occupied / total_capacity
            if total_capacity > 0
            else 0,
            "zones_by_type": {
                zone_type: len(self.get_zones_by_type(zone_type))
                for zone_type in ["lot", "meter", "standard"]
            },
            "using_real_data": self.use_real_data,
        }


# Global instance
_parking_zone_manager: Optional[ParkingZoneManager] = None


def get_parking_zone_manager() -> ParkingZoneManager:
    """Get the global parking zone manager instance."""
    global _parking_zone_manager
    if _parking_zone_manager is None:
        _parking_zone_manager = ParkingZoneManager()
    return _parking_zone_manager
