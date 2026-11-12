"""
City Simulation Environment
Simulates a city with parking zones, drivers, and traffic
"""

import hashlib
import json
import math
import os
import random
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import networkx as nx

from core import CityCoordinator, MapDataLoader, ParkingZone, RouteOptimizer
from core.array_backend import walking_distances
from core.exceptions import (
    ConfigurationError,
    DataValidationError,
    DriverError,
    NetworkError,
    RouteNotFoundError,
)
from core.traffic_manager import TrafficManager

from .driver_behavior import (
    DriverBehaviorModel,
    UrgencyLevel,
)


class CitySimulator:
    """
    Simulates a city environment for testing parking optimization algorithms
    Using real-world Grand Rapids data for realistic simulation
    """

    def __init__(
        self,
        data_directory: str = "output/map_data",
        n_drivers: int = 500,
        use_real_data: bool = True,
        n_zones: int = 20,
        seed: Optional[int] = None,
        start_time: Optional[datetime] = None,
        pricing_policy: str = "dynamic",
        routing_policy: str = "time",
        traffic_source: str = "simulated",
        scenario_event: str = "normal",
        demand_multiplier: float = 1.0,
        min_price: float = 1.0,
        max_price: float = 10.0,
        target_occupancy: float = 0.85,
        max_price_change: float = 0.15,
        warmup_hours: float = 0.0,
        array_backend: str = "auto",
    ):
        """
        Initialize city simulation

        Args:
            data_directory: Directory containing real-world map data
            n_drivers: Number of simulated drivers
            use_real_data: Whether to use real Grand Rapids data
        """
        if n_zones < 1 or n_drivers < 0:
            raise ValueError(
                "n_zones must be positive and n_drivers must be nonnegative"
            )
        self.seed = seed
        # Simulation sampling does not require a cryptographic RNG.
        self.rng = random.Random(seed)  # nosec B311
        self.np_rng = np.random.default_rng(seed)
        self.n_zones = n_zones
        if pricing_policy not in {"static", "dynamic"} or routing_policy not in {
            "distance",
            "time",
        }:
            raise ValueError("Unknown pricing or routing policy")
        if traffic_source not in {
            "simulated",
            "live",
            "free-flow",
        } or scenario_event not in {"normal", "event", "steady"}:
            raise ValueError("Unknown traffic source or event")
        if not math.isfinite(demand_multiplier) or demand_multiplier <= 0:
            raise ValueError("Demand multiplier must be finite and positive")
        self.pricing_policy, self.routing_policy = pricing_policy, routing_policy
        self.traffic_source, self.scenario_event = traffic_source, scenario_event
        if not math.isfinite(warmup_hours) or warmup_hours < 0:
            raise ValueError("Warm-up hours must be finite and nonnegative")
        self.warmup_hours = warmup_hours
        self.array_backend = array_backend
        self.compute_metadata = {"backend": "not_run"}
        self._available_counts = None
        self.demand_multiplier = demand_multiplier
        self.n_drivers = round(n_drivers * demand_multiplier)
        # Fixed default date makes seed-only offline replays independent of run day.
        self.start_time = start_time or datetime(2026, 11, 12, 8)
        self.data_directory = data_directory
        self.use_real_data = use_real_data

        # Load real-world data
        self.map_loader = MapDataLoader(data_directory)
        if use_real_data:
            self.use_real_data = self.map_loader.load_all_data()

        # City components - initialize with real or synthetic data
        try:
            if self.use_real_data:
                parking_zone_data = self.map_loader.get_parking_zones()
            else:
                parking_zone_data = self._generate_synthetic_parking_zones()

            self.parking_zones = [
                ParkingZone(
                    id=zone["id"],
                    name=zone["id"].replace("_", " ").title(),
                    location=zone["coordinates"],
                    capacity=zone["capacity"],
                    base_price=zone["hourly_rate"],
                    zone_type=zone["type"],
                )
                for zone in parking_zone_data
            ]
        except ConfigurationError as e:
            print(f"⚠️  Warning: {e.message}. Using synthetic parking zones.")
            self.use_real_data = False
            parking_zone_data = self._generate_synthetic_parking_zones()
            self.parking_zones = [
                ParkingZone(
                    id=zone["id"],
                    name=zone["id"].replace("_", " ").title(),
                    location=zone["coordinates"],
                    capacity=zone["capacity"],
                    base_price=zone["hourly_rate"],
                    zone_type=zone["type"],
                )
                for zone in parking_zone_data
            ]

        if not self.use_real_data:
            self._initialize_synthetic_network(parking_zone_data)
        self.route_optimizer = RouteOptimizer(map_loader=self.map_loader)
        self.coordinator = CityCoordinator(
            n_districts=4, route_optimizer=self.route_optimizer
        )
        from core.dynamic_pricing import DynamicPricingEngine

        self.coordinator.pricing_engine = DynamicPricingEngine(
            min_price,
            max_price,
            target_occupancy,
            max_change_fraction=max_price_change,
        )
        for zone in self.parking_zones:
            zone.base_price = float(np.clip(zone.base_price, min_price, max_price))
            zone.current_price = zone.base_price
        self.traffic_rng = np.random.default_rng(
            np.random.SeedSequence(seed).spawn(1)[0]
        )
        self.traffic_manager = TrafficManager(
            np_rng=self.traffic_rng, enable_live=traffic_source == "live"
        )
        self.behavior_model = DriverBehaviorModel(
            rng=self.rng,
            np_rng=self.np_rng,
            zone_ids=[z.id for z in self.parking_zones],
        )

        # Simulation state
        self.zones_by_id = {zone.id: zone for zone in self.parking_zones}
        self.zone_indices = {zone.id: i for i, zone in enumerate(self.parking_zones)}
        self.current_time = self.start_time
        self.time_step_minutes = 5
        self.duration_hours = 0.0
        self.drivers = []
        self.active_trips = []

        # Metrics tracking
        self.metrics = {
            "avg_search_time": [],
            "total_revenue": [],
            "avg_occupancy": [],
            "rejected_drivers": 0,
            "successful_parks": 0,
            "elapsed_minutes": [],
            "zone_occupancy": [],
            "zone_prices": [],
            "zone_revenue": [],
            "driving_minutes": [],
            "walking_meters": [],
            "estimated_parking_cost": [],
            "parking_observations": [],
        }

        # Initialize coordinator with real data
        self._initialize_coordinator()

    def _initialize_coordinator(self):
        """Initialize city coordinator with real Grand Rapids districts"""
        # Use real map bounds for district generation
        try:
            bounds = (
                self.map_loader.get_simulation_bounds() if self.use_real_data else None
            )
        except ConfigurationError:
            bounds = None

        if bounds:
            lat_min, lon_min, lat_max, lon_max = bounds
            center_lat = (lat_min + lat_max) / 2
            center_lon = (lon_min + lon_max) / 2

            # Create districts around Grand Rapids downtown
            [
                {
                    "id": "downtown_core",
                    "center": (center_lat, center_lon),
                    "radius_km": 0.5,
                    "zone_type": "commercial",
                },
                {
                    "id": "downtown_north",
                    "center": (center_lat + 0.005, center_lon),
                    "radius_km": 0.3,
                    "zone_type": "mixed",
                },
                {
                    "id": "downtown_south",
                    "center": (center_lat - 0.005, center_lon),
                    "radius_km": 0.3,
                    "zone_type": "commercial",
                },
                {
                    "id": "downtown_east",
                    "center": (center_lat, center_lon + 0.005),
                    "radius_km": 0.3,
                    "zone_type": "residential",
                },
            ]
        else:
            # Fallback districts if bounds not available
            pass

        # Initialize coordinator with parking zones
        self.coordinator.divide_city_into_districts(self.parking_zones)

        # Train demand predictors
        self._train_demand_predictors()

    def _generate_synthetic_parking_zones(self) -> List[Dict]:
        """Generate synthetic parking zones for testing when real data unavailable"""
        zones = []

        # Generate 20 test zones around Grand Rapids downtown area
        base_lat, base_lon = 42.963, -85.668

        for i in range(self.n_zones):
            # Scatter zones in a grid around downtown
            row = i // 5
            col = i % 5

            lat = base_lat + (row - 2) * 0.002
            lon = base_lon + (col - 2) * 0.002

            zone = {
                "id": f"synthetic_zone_{i}",
                "type": ["commercial", "residential", "mixed"][i % 3],
                "coordinates": (lat, lon),
                "capacity": self.rng.randint(10, 50),
                "hourly_rate": self.rng.uniform(2.0, 8.0),
                "accessible": i % 3 != 2,
            }
            zones.append(zone)

        return zones

    def _initialize_synthetic_network(self, zones: List[Dict]):
        """Build a connected demonstration grid without network access or map files."""
        graph = nx.MultiDiGraph(crs="EPSG:4326")
        for i, zone in enumerate(zones):
            lat, lon = zone["coordinates"]
            graph.add_node(i, y=lat, x=lon)
        for i in graph:
            neighbors = []
            if i % 5 < 4 and i + 1 in graph:
                neighbors.append(i + 1)
            if i + 5 in graph:
                neighbors.append(i + 5)
            for neighbor in neighbors:
                lat1, lon1 = zones[i]["coordinates"]
                lat2, lon2 = zones[neighbor]["coordinates"]
                distance = math.hypot(
                    (lat2 - lat1) * 111_000,
                    (lon2 - lon1) * 111_000 * math.cos(math.radians(lat1)),
                )
                graph.add_edge(i, neighbor, length=distance, speed_kph=30)
                graph.add_edge(neighbor, i, length=distance, speed_kph=30)
        self.map_loader.road_network = graph
        self.map_loader._parking_zones = zones

    def _train_demand_predictors(self):
        """Train demand predictors with historical data"""
        for district_id, predictor in self.coordinator.demand_predictors.items():
            historical_data = self._generate_historical_data(district_id)
            # Convert to format expected by predictor
            training_data = []
            for entry in historical_data:
                training_data.append(
                    {
                        "timestamp": entry["timestamp"],
                        "occupancy_rate": entry["occupancy_rate"],
                        "weather": 0,  # Default weather
                        "arrivals": entry["occupancy_rate"] * predictor.capacity * 0.2,
                        "capacity": predictor.capacity,
                    }
                )
            if training_data:
                training_data.sort(key=lambda row: row["timestamp"])
                split = int(len(training_data) * 0.8)
                predictor.train_model(training_data[:split])
                if not hasattr(self, "forecast_evaluation"):
                    self.forecast_evaluation = {}
                self.forecast_evaluation[district_id] = {
                    **predictor.evaluate(training_data[split:]),
                    "source": "synthetic_training_history",
                    "method": "chronological_80_20_holdout",
                    "train_end": training_data[split - 1]["timestamp"].isoformat(),
                    "test_start": training_data[split]["timestamp"].isoformat(),
                }
                predictor.train_model(training_data)

    def _generate_historical_data(self, district_id: str) -> List[Dict]:
        """Generate realistic historical parking data"""
        historical_data = []

        # Generate data for past 30 days
        for day in range(30):
            for hour in range(24):
                # Realistic demand patterns
                if 7 <= hour <= 9:  # Morning rush
                    demand = self.rng.randint(70, 95)
                elif 17 <= hour <= 19:  # Evening rush
                    demand = self.rng.randint(65, 90)
                elif 11 <= hour <= 14:  # Lunch
                    demand = self.rng.randint(50, 75)
                elif 9 <= hour <= 17:  # Business hours
                    demand = self.rng.randint(40, 70)
                elif 19 <= hour <= 22:  # Evening
                    demand = self.rng.randint(30, 60)
                else:  # Night/early morning
                    demand = self.rng.randint(5, 25)

                # Add weekend variations
                if (
                    self.start_time - timedelta(days=day + 1)
                ).weekday() >= 5:  # Weekend
                    if 10 <= hour <= 16:  # Weekend peak
                        demand = self.rng.randint(40, 70)
                    else:
                        demand = int(demand * 0.6)  # Lower demand

                historical_data.append(
                    {
                        "timestamp": (
                            self.start_time - timedelta(days=day + 1)
                        ).replace(hour=hour, minute=0, second=0, microsecond=0),
                        "occupancy_rate": demand / 100.0,
                        "avg_duration_minutes": self.rng.randint(30, 180),
                        "peak_demand": demand > 80,
                    }
                )

        return historical_data

    def run_simulation(self, duration_hours: float = 8, time_step_minutes: float = 5):
        """
        Run the parking simulation for specified duration

        Args:
            duration_hours: How long to simulate in hours
            time_step_minutes: Time step between updates in minutes
        """
        if not math.isfinite(duration_hours) or duration_hours <= 0:
            raise ValueError("duration_hours must be finite and positive")
        if not math.isfinite(time_step_minutes) or time_step_minutes <= 0:
            raise ValueError("time_step_minutes must be finite and positive")
        if timedelta(hours=duration_hours) <= timedelta(0) or timedelta(
            minutes=time_step_minutes
        ) <= timedelta(0):
            raise ValueError("duration and time step must be at least one microsecond")
        if self.warmup_hours >= duration_hours:
            raise ValueError(
                "Warm-up must be shorter than the total simulation duration"
            )
        warmup_steps = self.warmup_hours * 60 / time_step_minutes
        if not math.isclose(warmup_steps, round(warmup_steps), abs_tol=1e-8):
            raise ValueError("Warm-up must align with a simulation step boundary")
        self.duration_hours = duration_hours
        self.time_step_minutes = time_step_minutes
        print("🚀 Starting Grand Rapids parking simulation...")
        road_nodes = (
            len(self.map_loader.road_network.nodes())
            if self.map_loader.road_network
            else 0
        )
        print(
            f"📍 Using {'map' if self.use_real_data else 'synthetic'} data: {len(self.parking_zones)} zones, "
            f"{road_nodes} road nodes"
        )
        print(f"⏱️  Duration: {duration_hours}h, Time step: {time_step_minutes}min")
        print(f"🚗 Drivers: {self.n_drivers}")
        print("=" * 60)

        for predictor in self.coordinator.demand_predictors.values():
            predictor.demand_scale = self.demand_multiplier
            if self.scenario_event == "event":
                predictor.event_windows = [
                    (
                        self.start_time,
                        self.start_time + timedelta(hours=duration_hours / 4),
                        3.2,
                    )
                ]
        # Generate initial drivers
        self._generate_drivers(duration_hours)
        destinations = [driver["destination"] for driver in self.drivers]
        coordinates = [zone.location for zone in self.parking_zones]
        self._walking_distances, self.compute_metadata = walking_distances(
            destinations, coordinates, self.array_backend
        )
        for i, driver in enumerate(self.drivers):
            driver["distance_row"] = i
        scenario_inputs = {
            "zones": [
                (z.id, z.capacity, z.base_price, z.location) for z in self.parking_zones
            ],
            "drivers": [
                {
                    key: d.get(key)
                    for key in (
                        "id",
                        "arrival_time",
                        "start_location",
                        "destination",
                        "requires_accessible",
                        "duration_minutes",
                        "search_duration_minutes",
                        "decision_seed",
                        "profile",
                    )
                }
                for d in self.drivers
            ],
        }
        self.scenario_fingerprint = hashlib.sha256(
            json.dumps(scenario_inputs, default=str, sort_keys=True).encode()
        ).hexdigest()

        end_time = self.current_time + timedelta(hours=duration_hours)
        next_status_time = self.current_time

        while self.current_time < end_time:
            step_minutes = min(
                time_step_minutes, (end_time - self.current_time).total_seconds() / 60
            )

            # Remove completed stays before the next pricing snapshot.
            for driver in self.drivers:
                if driver.get("status") == "parked" and self.current_time >= driver[
                    "park_start_time"
                ] + timedelta(minutes=driver["duration_minutes"]):
                    self._driver_departs(driver)
            # Update traffic conditions
            self._update_traffic()

            # Run coordinator optimization
            optimization_results = (
                self.coordinator.optimize_city_parking(self.current_time)
                if self.pricing_policy == "dynamic"
                else {}
            )

            # Process driver behavior
            self._process_drivers(optimization_results, duration_hours)

            # Update metrics
            self._update_metrics(step_minutes)
            self.current_time += timedelta(minutes=step_minutes)

            # Print status every 30 minutes
            if self.current_time >= next_status_time:
                self._print_status()
                next_status_time = self.current_time + timedelta(minutes=30)

        # Final summary
        self._print_summary()

    def _generate_drivers(self, duration_hours: float = 8.0):
        """Generate drivers with realistic traffic patterns and psychological profiles"""
        if not self.map_loader.road_network:
            print("⚠️  No road network available - generating simplified drivers")
            self._generate_simplified_drivers(duration_hours)
            return

        components = list(
            nx.strongly_connected_components(self.map_loader.road_network)
        )

        # Find the largest connected component that has access to parking zones
        largest_component = max(components, key=len)
        accessible_zones = []

        for zone_data in self.map_loader.get_parking_zones():
            zone_node = self.route_optimizer._find_nearest_node(
                zone_data["coordinates"]
            )
            if zone_node is not None and zone_node in largest_component:
                accessible_zones.append(zone_data)

        print(f"🔗 Using largest component with {len(largest_component)} nodes")
        print(
            f"🅿️  Accessible parking zones: {len(accessible_zones)}/{len(self.parking_zones)}"
        )

        if not accessible_zones:
            print("⚠️  No accessible zones - falling back to simplified generation")
            self._generate_simplified_drivers(duration_hours)
            return

        # Get nodes from the largest component for driver spawning
        component_nodes = sorted(largest_component)

        # Filter nodes to reasonable spawn locations (avoid dead ends)
        spawn_nodes = []
        for node in component_nodes:
            # Check if node has reasonable connectivity (at least 2 connections)
            degree = self.map_loader.road_network.degree(node)
            if degree >= 2:
                spawn_nodes.append(node)

        if len(spawn_nodes) < 50:
            spawn_nodes = component_nodes  # Use all nodes if filtered set too small

        print(f"📍 Using {len(spawn_nodes)} spawn nodes from connected component")

        # Generate drivers using connected spawn points
        patterns = ["commuter", "errand", "tourist", "student"]
        pattern_weights = [0.4, 0.3, 0.2, 0.1]

        for i in range(self.n_drivers):
            # Choose spawn node from largest connected component
            spawn_node = self.rng.choice(spawn_nodes)
            node_data = self.map_loader.road_network.nodes[spawn_node]
            start_location = (node_data["y"], node_data["x"])  # (lat, lon)

            # Generate driver with psychological profile using behavior model
            profile = self.behavior_model.create_driver_profile(f"driver_{i}")
            pattern = self.rng.choices(patterns, weights=pattern_weights)[0]

            # Generate arrival time as minutes from start, then convert to datetime
            arrival_minute = self._generate_arrival_time(pattern, duration_hours, i)
            arrival_time = self.current_time + timedelta(minutes=arrival_minute)

            driver = {
                "id": f"driver_{i}",
                "start_location": start_location,
                "spawn_node": spawn_node,  # Track spawn node for debugging
                "pattern": pattern,
                "arrival_time": arrival_time,
                "destination": self.rng.choice(accessible_zones)["coordinates"],
                "requires_accessible": self.rng.random() < 0.08,
                "search_duration_minutes": int(self.np_rng.integers(2, 6)),
                "decision_seed": int(self.np_rng.integers(0, 2**63)),
                "urgency": self.rng.choice(list(UrgencyLevel)),
                "status": "traveling",  # Start as traveling, become searching when arrived
                "assigned_zone": None,
                "park_start_time": None,
                "search_start_time": None,
                "profile": profile,
                "price_sensitivity": profile.price_sensitivity,  # For compatibility
                "max_walk_distance": profile.patience_level * 0.8,  # Based on patience
                "duration_minutes": self._generate_parking_duration(
                    pattern, duration_hours
                ),
            }
            self.drivers.append(driver)

        print(f"Generated {len(self.drivers)} drivers with connected spawn points")

    def _generate_simplified_drivers(self, duration_hours: float = 8.0):
        """Fallback driver generation when road network unavailable"""
        # Use simulation bounds for driver generation
        try:
            bounds = self.map_loader.get_simulation_bounds()
        except ConfigurationError:
            bounds = None
        if not bounds:
            bounds = (
                42.956,
                -85.683,
                42.973,
                -85.668,
            )  # Grand Rapids downtown fallback

        lat_min, lon_min, lat_max, lon_max = bounds
        patterns = ["commuter", "errand", "tourist", "student"]
        pattern_weights = [0.4, 0.3, 0.2, 0.1]

        for i in range(self.n_drivers):
            # Generate random location within bounds
            start_lat = self.rng.uniform(lat_min, lat_max)
            start_lon = self.rng.uniform(lon_min, lon_max)

            # Generate driver with psychological profile using behavior model
            profile = self.behavior_model.create_driver_profile(f"driver_{i}")
            pattern = self.rng.choices(patterns, weights=pattern_weights)[0]

            # Generate arrival time as minutes from start, then convert to datetime
            arrival_minute = self._generate_arrival_time(pattern, duration_hours, i)
            arrival_time = self.current_time + timedelta(minutes=arrival_minute)

            driver = {
                "id": f"driver_{i}",
                "start_location": (start_lat, start_lon),
                "pattern": pattern,
                "arrival_time": arrival_time,
                "destination": self.rng.choice(self.parking_zones).location,
                "requires_accessible": self.rng.random() < 0.08,
                "search_duration_minutes": int(self.np_rng.integers(2, 6)),
                "decision_seed": int(self.np_rng.integers(0, 2**63)),
                "urgency": self.rng.choice(list(UrgencyLevel)),
                "status": "traveling",  # Start as traveling, become searching when arrived
                "assigned_zone": None,
                "park_start_time": None,
                "search_start_time": None,
                "profile": profile,
                "price_sensitivity": profile.price_sensitivity,  # For compatibility
                "max_walk_distance": profile.patience_level * 0.8,  # Based on patience
                "duration_minutes": self._generate_parking_duration(
                    pattern, duration_hours
                ),
            }
            self.drivers.append(driver)

    def _generate_arrival_time(
        self, pattern: str, duration_hours: float, driver_index: int
    ) -> int:
        """Generate realistic arrival times based on pattern"""
        total_minutes = duration_hours * 60
        if self.scenario_event == "steady":
            return int(self.rng.uniform(0, total_minutes))
        if self.scenario_event == "event":
            return int(self.rng.uniform(0, total_minutes * 0.25))

        if pattern == "rush_hour":
            # 60% arrive in first 25% of time (rush hour surge)
            # 30% arrive in middle 50% of time
            # 10% arrive in last 25% of time
            rand = self.rng.random()
            if rand < 0.6:  # Rush hour surge
                return int(self.rng.uniform(0, total_minutes * 0.25))
            elif rand < 0.9:  # Normal flow
                return int(self.rng.uniform(total_minutes * 0.25, total_minutes * 0.75))
            else:  # Late arrivals
                return int(self.rng.uniform(total_minutes * 0.75, total_minutes))

        elif pattern == "peak_hours":
            # 40% arrive in first third
            # 40% arrive in middle third
            # 20% arrive in last third
            rand = self.rng.random()
            if rand < 0.4:
                return int(self.rng.uniform(0, total_minutes * 0.33))
            elif rand < 0.8:
                return int(self.rng.uniform(total_minutes * 0.33, total_minutes * 0.67))
            else:
                return int(self.rng.uniform(total_minutes * 0.67, total_minutes))

        else:  # steady_flow
            # Even distribution with slight early bias
            return int(self.rng.uniform(0, total_minutes * 0.8))

    def _generate_parking_duration(
        self, pattern: str, drivers_per_minute: float
    ) -> int:
        """Generate realistic parking durations"""
        if pattern == "commuter":
            return self.rng.choice([240, 360, 480])
        if pattern == "errand":
            return self.rng.choice([15, 30, 45, 60])
        if pattern == "tourist":
            return self.rng.choice([60, 90, 120, 180])
        if pattern == "student":
            return self.rng.choice([90, 120, 180])
        if pattern == "rush_hour":
            # Shorter stays during rush hour (commuters, quick errands)
            return self.rng.choice(
                [15, 20, 30, 45, 60, 90] + [30, 45] * 3
            )  # Bias toward shorter

        elif pattern == "peak_hours":
            # Mixed durations (shopping, appointments, meals)
            return self.rng.choice([30, 45, 60, 90, 120, 180, 240])

        else:  # steady_flow
            # Longer stays (work, extended shopping, dining)
            return self.rng.choice(
                [60, 90, 120, 180, 240, 300, 360] + [120, 180] * 2
            )  # Bias toward longer

    def _update_traffic(self):
        """Apply traffic by exact directed multigraph edge, with expiring weights."""
        graph = self.route_optimizer.road_graph
        if graph is None or self.traffic_source == "free-flow":
            return
        self.route_optimizer.expire_traffic(self.current_time)
        edges = list(graph.edges(keys=True))
        if self.traffic_source == "live":
            # Limited provider sampling; a route duration is an edge proxy, not a sensor.
            indices = self.traffic_rng.choice(
                len(edges), size=min(3, len(edges)), replace=False
            )
            edges = [edges[int(i)] for i in indices]
        observations = {}
        for u, v, key in edges:
            origin = (graph.nodes[u]["y"], graph.nodes[u]["x"])
            destination = (graph.nodes[v]["y"], graph.nodes[v]["x"])
            observations[u, v, key] = self.traffic_manager.get_traffic_condition(
                origin, destination, current_time=self.current_time
            )
        self.route_optimizer.update_traffic_conditions(observations, self.current_time)

    def _process_drivers(self, optimization_results: Dict, duration_hours: float = 8.0):
        """Process driver behavior and parking decisions"""
        self._available_counts = {
            zone.id: zone.capacity - zone.occupied_spots for zone in self.parking_zones
        }
        for driver in self.drivers:
            if driver["status"] == "traveling":
                # Check if driver has arrived
                if self.current_time >= driver["arrival_time"]:
                    # Driver starts searching for parking
                    driver["status"] = "searching"
                    driver["search_start_time"] = self.current_time
                    driver.setdefault("search_duration_minutes", 3)

            elif driver["status"] == "searching":
                # Check if search duration has elapsed
                if driver.get("search_start_time"):
                    search_elapsed = (
                        self.current_time - driver["search_start_time"]
                    ).total_seconds() / 60
                    if search_elapsed >= driver.get("search_duration_minutes", 5):
                        # Search time is over, try to find parking
                        self._find_parking_for_driver(driver, optimization_results)

            elif driver["status"] == "parked":
                # Check if parking duration is over
                if driver["park_start_time"]:
                    parked_minutes = (
                        self.current_time - driver["park_start_time"]
                    ).total_seconds() / 60
                    if parked_minutes >= driver["duration_minutes"]:
                        # Driver departs
                        self._driver_departs(driver)

    def _has_space(self, zone):
        return (
            self._available_counts.get(zone.id, 0) > 0
            if self._available_counts is not None
            else zone.has_availability()
        )

    def _find_parking_for_driver(self, driver: Dict, optimization_results: Dict):
        """Find parking for a driver using psychological behavior model"""
        driver["decision_time"] = self.current_time
        try:
            # Get parking recommendations from route optimizer
            recommendations = self.route_optimizer.find_optimal_parking(
                start_location=driver["start_location"],
                preferences={
                    "max_walk_distance": driver["max_walk_distance"],
                    "price_weight": driver["price_sensitivity"],
                    "time_weight": 1.0 - driver["price_sensitivity"],
                    "prefer_time": self.routing_policy == "time",
                    "destination": driver.get("destination", driver["start_location"]),
                    "parking_duration_minutes": driver["duration_minutes"],
                    "requires_accessible": driver.get("requires_accessible", False),
                    "time_value": driver["profile"].time_value,
                    "zone_prices": {z.id: z.hourly_rate for z in self.parking_zones},
                    **(
                        {
                            "walking_distances": self._walking_distances[
                                driver["distance_row"]
                            ],
                            "zone_indices": self.zone_indices,
                        }
                        if "distance_row" in driver
                        else {}
                    ),
                },
                available_parking_zones=[
                    zone.id for zone in self.parking_zones if self._has_space(zone)
                ],
            )
        except RouteNotFoundError as e:
            personality = driver["profile"].personality_type.value
            print(f"  ❌ {driver['id']} ({personality}) failed: {e.message}")
            driver["status"] = "departed"
            self.metrics["rejected_drivers"] += 1
            return
        except (NetworkError, DataValidationError) as e:
            personality = driver["profile"].personality_type.value
            print(f"  ⚠️  {driver['id']} ({personality}) error: {e.message}")
            driver["status"] = "departed"
            self.metrics["rejected_drivers"] += 1
            return

        try:
            if recommendations and len(recommendations) > 0:
                # Convert recommendations to format expected by behavior model
                available_options = []
                for zone_id, route_info in recommendations.items():
                    zone = next(
                        (z for z in self.parking_zones if z.id == zone_id), None
                    )
                    if zone and self._has_space(zone):
                        option = {
                            "zone_id": zone_id,
                            "price": zone.hourly_rate,
                            "walk_distance": route_info.walking_distance
                            / 1000,  # Convert to km
                            "total_cost": route_info.estimated_parking_cost,
                            "drive_distance_meters": route_info.total_distance,
                            "walk_distance_meters": route_info.walking_distance,
                            "travel_time": route_info.total_time
                            / 60,  # Convert to minutes
                            "zone_type": zone.zone_type,
                            "occupancy_rate": 1
                            - self._available_counts[zone.id] / zone.capacity
                            if self._available_counts is not None
                            else zone.occupancy_rate,
                        }
                        available_options.append(option)

                if available_options:
                    # Determine urgency level based on search time
                    search_time = 0
                    if driver.get("search_start_time"):
                        search_time = (
                            self.current_time - driver["search_start_time"]
                        ).total_seconds() / 60

                    urgency = self.behavior_model.get_urgency_level(
                        {
                            "search_time_minutes": search_time,
                            "arrival_time": driver["arrival_time"],
                            "current_time": self.current_time,
                        },
                        self.current_time,
                    )

                    # Use behavior model to make decision
                    # Per-driver streams keep later drivers' shocks stable when
                    # earlier policies offer different numbers of viable options.
                    previous_rng, previous_np_rng = (
                        self.behavior_model.rng,
                        self.behavior_model.np_rng,
                    )
                    decision_seed = driver.get("decision_seed", 0)
                    self.behavior_model.rng = random.Random(decision_seed)  # nosec B311
                    self.behavior_model.np_rng = np.random.default_rng(decision_seed)
                    try:
                        chosen_option = self.behavior_model.make_parking_decision(
                            driver["id"], available_options, self.current_time, urgency
                        )
                    finally:
                        self.behavior_model.rng, self.behavior_model.np_rng = (
                            previous_rng,
                            previous_np_rng,
                        )

                    if chosen_option:
                        zone_id = chosen_option["zone_id"]
                        zone = next(
                            (z for z in self.parking_zones if z.id == zone_id), None
                        )

                        # Successfully parked using psychological decision-making
                        if zone is None or not zone.occupy_spot(
                            driver["id"], current_time=self.current_time
                        ):
                            return
                        if self._available_counts is not None:
                            self._available_counts[zone_id] -= 1
                        driver["assigned_zone"] = zone_id
                        driver["status"] = "parked"
                        driver["park_start_time"] = self.current_time
                        driver["accepted_hourly_rate"] = chosen_option["price"]

                        # Record search time
                        if driver.get("search_start_time"):
                            search_time = (
                                self.current_time - driver["search_start_time"]
                            ).total_seconds() / 60
                            self.metrics["avg_search_time"].append(search_time)

                        self.metrics["parking_observations"].append(
                            {
                                "driver_id": driver["id"],
                                "parked_at_minutes": (
                                    self.current_time - self.start_time
                                ).total_seconds()
                                / 60,
                                "search_minutes": search_time,
                                "driving_minutes": chosen_option["travel_time"],
                                "walking_meters": chosen_option["walk_distance_meters"],
                                "mean_stay_cost": chosen_option["total_cost"],
                            }
                        )
                        self.metrics["successful_parks"] += 1
                        self.metrics["driving_minutes"].append(
                            chosen_option["travel_time"]
                        )
                        self.metrics["walking_meters"].append(
                            chosen_option["walk_distance_meters"]
                        )
                        self.metrics["estimated_parking_cost"].append(
                            chosen_option["total_cost"]
                        )

                        # Update behavior model with experience
                        satisfaction = self.rng.uniform(
                            0.6, 1.0
                        )  # Generally satisfied when finding parking
                        self.behavior_model.learn_from_experience(
                            driver["id"], chosen_option, satisfaction
                        )

                        # Get personality type for logging
                        personality = driver["profile"].personality_type.value
                        print(
                            f"  ✅ {driver['id']} ({personality}) parked at {zone_id}"
                            + (
                                f" (search: {search_time:.1f}min)"
                                if search_time
                                else ""
                            )
                            + f" [urgency: {urgency.name.lower()}]"
                        )
                        return

                # If we reach here, recommendations exist but all zones are full
                print(
                    f"  ❌ {driver['id']} ({driver['profile'].personality_type.value}) failed: "
                    f"{len(recommendations)} options found but all zones full"
                )

            # No parking found - enhanced debugging
            start_node = self.route_optimizer._find_nearest_node(
                driver["start_location"]
            )
            component = (
                self.route_optimizer._get_component_for_node(start_node)
                if start_node
                else None
            )
            personality = driver["profile"].personality_type.value
            print(
                f"  ❌ {driver['id']} ({personality}) failed: {len(recommendations)} options, "
                f"start_node={start_node}, reachable={component is not None and len(component) > 10}"
            )

            driver["status"] = "departed"
            self.metrics["rejected_drivers"] += 1

        except DriverError as e:
            personality = driver["profile"].personality_type.value
            print(
                f"  ⚠️  Driver decision error for {driver['id']} ({personality}): {e.message}"
            )
            driver["status"] = "departed"
            self.metrics["rejected_drivers"] += 1
        except Exception as e:
            personality = driver["profile"].personality_type.value
            print(f"  ⚠️  Unexpected error for {driver['id']} ({personality}): {e}")
            driver["status"] = "departed"
            self.metrics["rejected_drivers"] += 1

    def _driver_departs(self, driver: Dict):
        """Handle driver departure"""
        if driver["assigned_zone"]:
            zone = next(
                z for z in self.parking_zones if z.id == driver["assigned_zone"]
            )
            released = zone.release_spot(driver["id"])
            if released and self._available_counts is not None:
                self._available_counts[zone.id] += 1

        driver["status"] = "departed"

    def _update_metrics(self, step_minutes: float = 5):
        """Reuse occupied-space minutes for interval occupancy and revenue."""
        drivers_by_id = {driver["id"]: driver for driver in self.drivers}
        zone_occupancy = {}
        zone_revenue = {}
        total_occupied_minutes = 0.0
        total_revenue = 0.0
        for zone in self.parking_zones:
            occupied_minutes = 0.0
            earned = 0.0
            for spot in zone.spots:
                if not spot.is_occupied:
                    continue
                driver = drivers_by_id.get(spot.vehicle_id)
                minutes = step_minutes
                if driver and driver.get("park_start_time"):
                    departure = driver["park_start_time"] + timedelta(
                        minutes=driver["duration_minutes"]
                    )
                    minutes = max(
                        0,
                        min(
                            step_minutes,
                            (departure - self.current_time).total_seconds() / 60,
                        ),
                    )
                occupied_minutes += minutes
                rate = (
                    driver.get("accepted_hourly_rate", zone.hourly_rate)
                    if driver
                    else zone.hourly_rate
                )
                earned += minutes * rate / 60
            zone_occupancy[zone.id] = occupied_minutes / (zone.capacity * step_minutes)
            total_occupied_minutes += occupied_minutes
            zone_revenue[zone.id] = earned
            total_revenue += zone_revenue[zone.id]
            zone.occupancy_history.append(zone_occupancy[zone.id])
            zone.occupancy_history = zone.occupancy_history[-168:]
        total_spots = sum(zone.capacity for zone in self.parking_zones)
        self.metrics["avg_occupancy"].append(
            total_occupied_minutes / (total_spots * step_minutes)
        )
        self.metrics["zone_occupancy"].append(zone_occupancy)
        elapsed = (self.current_time - self.start_time).total_seconds() / 60
        self.metrics["elapsed_minutes"].append(elapsed + step_minutes)
        self.metrics["zone_prices"].append(
            {zone.id: zone.hourly_rate for zone in self.parking_zones}
        )
        self.metrics["total_revenue"].append(total_revenue)
        self.metrics["zone_revenue"].append(zone_revenue)

    def _print_status(self):
        """Print current simulation status"""
        time_str = self.current_time.strftime("%H:%M")

        # Driver statistics
        traveling = sum(1 for d in self.drivers if d["status"] == "traveling")
        searching = sum(1 for d in self.drivers if d["status"] == "searching")
        parked = sum(1 for d in self.drivers if d["status"] == "parked")
        departed = sum(1 for d in self.drivers if d["status"] == "departed")

        # Parking statistics
        total_spots = sum(zone.capacity for zone in self.parking_zones)
        occupied_spots = sum(zone.occupied_spots for zone in self.parking_zones)
        occupancy_rate = (occupied_spots / total_spots * 100) if total_spots > 0 else 0

        print(
            f"⏰ {time_str} | 🚗 Traveling: {traveling}, Searching: {searching}, "
            f"Parked: {parked}, Departed: {departed}"
        )
        print(
            f"    🅿️  Occupancy: {occupied_spots}/{total_spots} ({occupancy_rate:.1f}%) | "
            f"✅ Successful: {self.metrics['successful_parks']}, "
            f"❌ Rejected: {self.metrics['rejected_drivers']}"
        )

    def _print_summary(self):
        """Print simulation summary with behavior analysis"""
        print("\n" + "=" * 60)
        print("📊 SIMULATION SUMMARY")
        print("=" * 60)

        if self.metrics["avg_occupancy"]:
            avg_occupancy = (
                np.average(
                    self.metrics["avg_occupancy"],
                    weights=np.diff([0, *self.metrics["elapsed_minutes"]]),
                )
                * 100
            )
            max_occupancy = max(self.metrics["avg_occupancy"]) * 100
            print(f"🅿️  Average Occupancy: {avg_occupancy:.1f}%")
            print(f"🅿️  Peak Occupancy: {max_occupancy:.1f}%")

        if self.metrics["total_revenue"]:
            total_revenue = sum(self.metrics["total_revenue"])
            print(f"💰 Total Revenue: ${total_revenue:.2f}")

        if self.metrics["avg_search_time"]:
            avg_search_time = np.mean(self.metrics["avg_search_time"])
            print(f"⏱️  Average Search Time: {avg_search_time:.1f} minutes")

        success_rate = (
            self.metrics["successful_parks"]
            / (self.metrics["successful_parks"] + self.metrics["rejected_drivers"])
            * 100
            if (self.metrics["successful_parks"] + self.metrics["rejected_drivers"]) > 0
            else 0
        )

        print(f"✅ Successful Parking: {self.metrics['successful_parks']}")
        print(f"❌ Rejected Drivers: {self.metrics['rejected_drivers']}")
        print(f"📈 Success Rate: {success_rate:.1f}%")

        # Driver personality analysis
        print("\n🧠 DRIVER PSYCHOLOGY ANALYSIS")
        print("-" * 30)
        personality_counts = {}
        for driver in self.drivers:
            if "profile" in driver:
                personality = driver["profile"].personality_type.value
                personality_counts[personality] = (
                    personality_counts.get(personality, 0) + 1
                )

        for personality, count in personality_counts.items():
            percentage = (count / len(self.drivers)) * 100
            print(f"  {personality.title()}: {count} drivers ({percentage:.1f}%)")

        print("\n📍 REAL-WORLD DATA INTEGRATION")
        print("-" * 30)
        print(
            f"📍 Zone data source: {'map' if self.use_real_data else 'synthetic'} ({len(self.parking_zones)} zones)"
        )
        road_nodes = (
            len(self.map_loader.road_network.nodes())
            if self.map_loader.road_network
            else 0
        )
        print(f"🛣️  Road network nodes: {road_nodes}")
        print(
            f"🧠 Psychological modeling: ✅ Active ({len(personality_counts)} personality types)"
        )
        print("🎯 Behavior-based decisions: ✅ Enabled")

    def save_results(self, filename: str = "output/simulation_results.json"):
        """Save simulation results to file"""
        results = self.results()

        Path(filename).parent.mkdir(parents=True, exist_ok=True)
        with open(filename, "w") as f:
            json.dump(results, f, indent=2, allow_nan=False)
        print(f"💾 Results saved to {filename}")

    def results(self):
        """Return serializable observations and explicit modeling assumptions."""
        return {
            "scenario_fingerprint": getattr(self, "scenario_fingerprint", None),
            "simulation_config": {
                "pricing_policy": self.pricing_policy,
                "billing_policy": "entry_rate",
                "routing_policy": self.routing_policy,
                "traffic_source": self.traffic_source,
                "scenario_event": self.scenario_event,
                "demand_multiplier": self.demand_multiplier,
                "price_constraints": {
                    "min": self.coordinator.pricing_engine.min_price,
                    "max": self.coordinator.pricing_engine.max_price,
                    "max_change_fraction": self.coordinator.pricing_engine.max_change_fraction,
                    "target_occupancy": self.coordinator.pricing_engine.target_occupancy,
                },
                "use_real_data": self.use_real_data,
                "seed": self.seed,
                "start_time": self.start_time.isoformat(),
                "duration_hours": self.duration_hours,
                "warmup_hours": self.warmup_hours,
                "time_step_minutes": self.time_step_minutes,
                "data_source": "map" if self.use_real_data else "synthetic",
                "data_directory": self.data_directory,
                "n_drivers": self.n_drivers,
                "total_zones": len(self.parking_zones),
                "total_road_nodes": len(self.map_loader.road_network.nodes())
                if self.map_loader.road_network
                else 0,
            },
            "compute": self.compute_metadata,
            "driver_records": [
                {
                    "id": driver["id"],
                    "status": driver["status"],
                    "parking_duration_minutes": driver["duration_minutes"],
                    "accepted_hourly_rate": driver.get("accepted_hourly_rate"),
                    "arrival_minutes": (
                        driver["arrival_time"] - self.start_time
                    ).total_seconds()
                    / 60,
                    "parked_at_minutes": (
                        driver["park_start_time"] - self.start_time
                    ).total_seconds()
                    / 60
                    if driver.get("park_start_time")
                    else None,
                    "decision_minutes": (
                        driver["decision_time"] - self.start_time
                    ).total_seconds()
                    / 60
                    if driver.get("decision_time")
                    else None,
                }
                for driver in self.drivers
            ],
            "metrics": self.metrics,
            "driver_outcomes": {
                "total": len(self.drivers),
                "parked": self.metrics["successful_parks"],
                "rejected": self.metrics["rejected_drivers"],
                "unresolved": sum(
                    d["status"] in {"searching", "traveling"} for d in self.drivers
                ),
            },
            "forecast_evaluation": getattr(self, "forecast_evaluation", {}),
            "data_validation": getattr(self.map_loader, "validation_report", {}),
            "traffic_summary": self.traffic_manager.get_traffic_summary(),
            "traffic_edges": [
                {
                    "origin": (
                        float(self.route_optimizer.road_graph.nodes[u]["y"]),
                        float(self.route_optimizer.road_graph.nodes[u]["x"]),
                    ),
                    "destination": (
                        float(self.route_optimizer.road_graph.nodes[v]["y"]),
                        float(self.route_optimizer.road_graph.nodes[v]["x"]),
                    ),
                    "travel_time_seconds": attrs["travel_time"],
                    "base_travel_time_seconds": attrs["base_travel_time"],
                    "source": attrs.get("traffic_source", "free_flow"),
                    "observed_at": attrs["traffic_observed_at"].isoformat()
                    if attrs.get("traffic_observed_at")
                    else None,
                }
                for u, v, attrs in self.route_optimizer.road_graph.edges(data=True)
            ],
            "assumptions": [
                "Simulated outcomes; demand and behavior are not field calibrated.",
                "Walking distance is a straight-line estimate, not an accessible pedestrian route.",
                "Parking decisions occur on time-step boundaries; route time affects choice and is recorded separately from search delay.",
                "Revenue accrues for occupied minutes at the hourly rate accepted on entry; stays are censored at simulation end.",
            ],
            "final_state": {
                "timestamp": self.current_time.isoformat(),
                "zone_occupancy": [
                    {
                        "id": zone.id,
                        "capacity": zone.capacity,
                        "occupied": zone.occupied_spots,
                        "rate": zone.hourly_rate,
                    }
                    for zone in self.parking_zones
                ],
            },
        }


if __name__ == "__main__":
    # Run a sample simulation
    print("Parking Optimization Simulation")
    print("==============================\n")

    sim = CitySimulator(
        data_directory="output/map_data", n_drivers=500, use_real_data=True
    )

    sim.run_simulation(duration_hours=8, time_step_minutes=5)
    sim.save_results()
