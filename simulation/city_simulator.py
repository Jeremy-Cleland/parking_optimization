"""
City Simulation Environment
Simulates a city with parking zones, drivers, and traffic
"""

import json
import os
import random
import sys
from datetime import datetime, timedelta
from typing import Dict, List

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import networkx as nx

from core import CityCoordinator, MapDataLoader, ParkingZone, RouteOptimizer
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
    ):
        """
        Initialize city simulation

        Args:
            data_directory: Directory containing real-world map data
            n_drivers: Number of simulated drivers
            use_real_data: Whether to use real Grand Rapids data
        """
        self.data_directory = data_directory
        self.n_drivers = n_drivers
        self.use_real_data = use_real_data

        # Load real-world data
        self.map_loader = MapDataLoader(data_directory)
        if use_real_data:
            self.map_loader.load_all_data()

        # City components - initialize with real or synthetic data
        try:
            if use_real_data:
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

        self.route_optimizer = RouteOptimizer()
        self.coordinator = CityCoordinator(n_districts=4)
        self.traffic_manager = TrafficManager()  # Real-time traffic integration!
        self.behavior_model = DriverBehaviorModel()  # Psychological driver modeling

        # Simulation state
        self.current_time = datetime.now().replace(hour=8, minute=0, second=0)
        self.drivers = []
        self.active_trips = []

        # Metrics tracking
        self.metrics = {
            "avg_search_time": [],
            "total_revenue": [],
            "avg_occupancy": [],
            "rejected_drivers": 0,
            "successful_parks": 0,
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
            districts = [
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
            districts = [
                {
                    "id": "district_1",
                    "center": (42.963, -85.668),
                    "radius_km": 0.5,
                    "zone_type": "commercial",
                },
                {
                    "id": "district_2",
                    "center": (42.968, -85.673),
                    "radius_km": 0.3,
                    "zone_type": "mixed",
                },
                {
                    "id": "district_3",
                    "center": (42.971, -85.670),
                    "radius_km": 0.3,
                    "zone_type": "commercial",
                },
                {
                    "id": "district_4",
                    "center": (42.965, -85.675),
                    "radius_km": 0.3,
                    "zone_type": "residential",
                },
            ]

        # Initialize coordinator with parking zones
        self.coordinator.divide_city_into_districts(self.parking_zones)

        # Train demand predictors
        self._train_demand_predictors()

    def _generate_synthetic_parking_zones(self) -> List[Dict]:
        """Generate synthetic parking zones for testing when real data unavailable"""
        zones = []

        # Generate 20 test zones around Grand Rapids downtown area
        base_lat, base_lon = 42.963, -85.668

        for i in range(20):
            # Scatter zones in a grid around downtown
            row = i // 5
            col = i % 5

            lat = base_lat + (row - 2) * 0.002
            lon = base_lon + (col - 2) * 0.002

            zone = {
                "id": f"synthetic_zone_{i}",
                "type": ["commercial", "residential", "mixed"][i % 3],
                "coordinates": (lat, lon),
                "capacity": random.randint(10, 50),
                "hourly_rate": random.uniform(2.0, 8.0),
            }
            zones.append(zone)

        return zones

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
                        "arrivals": int(
                            entry["occupancy_rate"] * 20
                        ),  # Estimate arrivals based on occupancy
                    }
                )
            if training_data:
                predictor.train_model(training_data)

    def _generate_historical_data(self, district_id: str) -> List[Dict]:
        """Generate realistic historical parking data"""
        historical_data = []

        # Generate data for past 30 days
        for day in range(30):
            for hour in range(24):
                # Realistic demand patterns
                if 7 <= hour <= 9:  # Morning rush
                    demand = random.randint(70, 95)
                elif 17 <= hour <= 19:  # Evening rush
                    demand = random.randint(65, 90)
                elif 11 <= hour <= 14:  # Lunch
                    demand = random.randint(50, 75)
                elif 9 <= hour <= 17:  # Business hours
                    demand = random.randint(40, 70)
                elif 19 <= hour <= 22:  # Evening
                    demand = random.randint(30, 60)
                else:  # Night/early morning
                    demand = random.randint(5, 25)

                # Add weekend variations
                if day % 7 in [5, 6]:  # Weekend
                    if 10 <= hour <= 16:  # Weekend peak
                        demand = random.randint(40, 70)
                    else:
                        demand = int(demand * 0.6)  # Lower demand

                historical_data.append(
                    {
                        "timestamp": datetime.now() - timedelta(days=day, hours=hour),
                        "occupancy_rate": demand / 100.0,
                        "avg_duration_minutes": random.randint(30, 180),
                        "peak_demand": demand > 80,
                    }
                )

        return historical_data

    def run_simulation(self, duration_hours: int = 8, time_step_minutes: int = 5):
        """
        Run the parking simulation for specified duration

        Args:
            duration_hours: How long to simulate in hours
            time_step_minutes: Time step between updates in minutes
        """
        print("🚀 Starting Grand Rapids parking simulation...")
        road_nodes = (
            len(self.map_loader.road_network.nodes())
            if self.map_loader.road_network
            else 0
        )
        print(
            f"📍 Using real-world data: {len(self.parking_zones)} zones, "
            f"{road_nodes} road nodes"
        )
        print(f"⏱️  Duration: {duration_hours}h, Time step: {time_step_minutes}min")
        print(f"🚗 Drivers: {self.n_drivers}")
        print("=" * 60)

        # Generate initial drivers
        self._generate_drivers(duration_hours)

        total_steps = int((duration_hours * 60) // time_step_minutes)

        for step in range(total_steps):
            # Update simulation time
            self.current_time += timedelta(minutes=time_step_minutes)

            # Update traffic conditions
            self._update_traffic()

            # Run coordinator optimization
            optimization_results = self.coordinator.optimize_city_parking()

            # Process driver behavior
            self._process_drivers(optimization_results, duration_hours)

            # Update metrics
            self._update_metrics()

            # Print status every 30 minutes
            if step % (30 // time_step_minutes) == 0:
                self._print_status()

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
            if zone_node and zone_node in largest_component:
                accessible_zones.append(zone_data)

        print(f"🔗 Using largest component with {len(largest_component)} nodes")
        print(
            f"🅿️  Accessible parking zones: {len(accessible_zones)}/{len(self.parking_zones)}"
        )

        if len(accessible_zones) < 10:
            print("⚠️  Too few accessible zones - falling back to simplified generation")
            self._generate_simplified_drivers(duration_hours)
            return

        # Get nodes from the largest component for driver spawning
        component_nodes = list(largest_component)

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
            spawn_node = random.choice(spawn_nodes)
            node_data = self.map_loader.road_network.nodes[spawn_node]
            start_location = (node_data["y"], node_data["x"])  # (lat, lon)

            # Generate driver with psychological profile using behavior model
            profile = self.behavior_model.create_driver_profile(f"driver_{i}")
            pattern = random.choices(patterns, weights=pattern_weights)[0]

            # Generate arrival time as minutes from start, then convert to datetime
            arrival_minute = self._generate_arrival_time(pattern, duration_hours, i)
            arrival_time = self.current_time + timedelta(minutes=arrival_minute)

            driver = {
                "id": f"driver_{i}",
                "start_location": start_location,
                "spawn_node": spawn_node,  # Track spawn node for debugging
                "pattern": pattern,
                "arrival_time": arrival_time,
                "parking_duration": self._generate_parking_duration(
                    pattern, duration_hours
                ),
                "urgency": random.choice(list(UrgencyLevel)),
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
        bounds = self.map_loader.get_simulation_bounds()
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
            start_lat = random.uniform(lat_min, lat_max)
            start_lon = random.uniform(lon_min, lon_max)

            # Generate driver with psychological profile using behavior model
            profile = self.behavior_model.create_driver_profile(f"driver_{i}")
            pattern = random.choices(patterns, weights=pattern_weights)[0]

            # Generate arrival time as minutes from start, then convert to datetime
            arrival_minute = self._generate_arrival_time(pattern, duration_hours, i)
            arrival_time = self.current_time + timedelta(minutes=arrival_minute)

            driver = {
                "id": f"driver_{i}",
                "start_location": (start_lat, start_lon),
                "pattern": pattern,
                "arrival_time": arrival_time,
                "parking_duration": self._generate_parking_duration(
                    pattern, duration_hours
                ),
                "urgency": random.choice(list(UrgencyLevel)),
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

        if pattern == "rush_hour":
            # 60% arrive in first 25% of time (rush hour surge)
            # 30% arrive in middle 50% of time
            # 10% arrive in last 25% of time
            rand = random.random()
            if rand < 0.6:  # Rush hour surge
                return int(random.uniform(0, total_minutes * 0.25))
            elif rand < 0.9:  # Normal flow
                return int(random.uniform(total_minutes * 0.25, total_minutes * 0.75))
            else:  # Late arrivals
                return int(random.uniform(total_minutes * 0.75, total_minutes))

        elif pattern == "peak_hours":
            # 40% arrive in first third
            # 40% arrive in middle third
            # 20% arrive in last third
            rand = random.random()
            if rand < 0.4:
                return int(random.uniform(0, total_minutes * 0.33))
            elif rand < 0.8:
                return int(random.uniform(total_minutes * 0.33, total_minutes * 0.67))
            else:
                return int(random.uniform(total_minutes * 0.67, total_minutes))

        else:  # steady_flow
            # Even distribution with slight early bias
            return int(random.uniform(0, total_minutes * 0.8))

    def _generate_parking_duration(
        self, pattern: str, drivers_per_minute: float
    ) -> int:
        """Generate realistic parking durations"""
        if pattern == "rush_hour":
            # Shorter stays during rush hour (commuters, quick errands)
            return random.choice(
                [15, 20, 30, 45, 60, 90] + [30, 45] * 3
            )  # Bias toward shorter

        elif pattern == "peak_hours":
            # Mixed durations (shopping, appointments, meals)
            return random.choice([30, 45, 60, 90, 120, 180, 240])

        else:  # steady_flow
            # Longer stays (work, extended shopping, dining)
            return random.choice(
                [60, 90, 120, 180, 240, 300, 360] + [120, 180] * 2
            )  # Bias toward longer

    def _update_traffic(self):
        """Update traffic conditions using real road network"""
        try:
            # Skip traffic updates if no road network available
            if self.map_loader.road_network is None:
                return

            # Sample fewer edges for traffic updates (to avoid API limits)
            sample_edges = random.sample(
                list(self.map_loader.road_network.edges(data=True)),
                min(3, len(self.map_loader.road_network.edges())),
            )

            for u, v, edge_data in sample_edges:
                # Get coordinates from road network
                start_coords = (
                    self.map_loader.road_network.nodes[u]["y"],
                    self.map_loader.road_network.nodes[u]["x"],
                )
                end_coords = (
                    self.map_loader.road_network.nodes[v]["y"],
                    self.map_loader.road_network.nodes[v]["x"],
                )

                # Get traffic condition (falls back gracefully if API unavailable)
                traffic_condition = self.traffic_manager.get_traffic_condition(
                    start_coords, end_coords
                )

                if traffic_condition.is_real_data:
                    traffic_factor = max(traffic_condition.delay_factor, 0.5)
                    print(
                        f"  🚗 Real traffic: {u}→{v} = {traffic_factor:.2f}x "
                        f"(Speed: {traffic_condition.speed_kmh:.1f} km/h)"
                    )
                else:
                    print(f"  ⚠️  API fallback for {u}→{v}, using synthetic data")

        except Exception as e:
            print(f"  ⚠️  Traffic update error: {e}")

    def _process_drivers(self, optimization_results: Dict, duration_hours: float = 8.0):
        """Process driver behavior and parking decisions"""
        current_minute = self.current_time.hour * 60 + self.current_time.minute

        for driver in self.drivers:
            if driver["status"] == "traveling":
                # Check if driver has arrived
                arrival_minute = (
                    driver["arrival_time"].hour * 60 + driver["arrival_time"].minute
                )
                if current_minute >= arrival_minute:
                    # Driver starts searching for parking
                    driver["status"] = "searching"
                    driver["search_start_time"] = self.current_time
                    # Add realistic search duration based on simulation length
                    if duration_hours < 0.5:  # Short simulations: quick search
                        driver["search_duration_minutes"] = np.random.randint(1, 4)
                    elif duration_hours < 2:  # Medium simulations: normal search
                        driver["search_duration_minutes"] = np.random.randint(2, 6)
                    else:  # Long simulations: realistic search
                        driver["search_duration_minutes"] = np.random.randint(3, 9)

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

    def _find_parking_for_driver(self, driver: Dict, optimization_results: Dict):
        """Find parking for a driver using psychological behavior model"""
        try:
            # Get parking recommendations from route optimizer
            recommendations = self.route_optimizer.find_optimal_parking(
                start_location=driver["start_location"],
                preferences={
                    "max_walk_distance": driver["max_walk_distance"],
                    "price_weight": driver["price_sensitivity"],
                    "time_weight": 1.0 - driver["price_sensitivity"],
                },
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
                    if zone and zone.has_availability():
                        option = {
                            "zone_id": zone_id,
                            "price": zone.hourly_rate,
                            "walk_distance": route_info.total_distance
                            / 1000,  # Convert to km
                            "travel_time": route_info.total_time
                            / 60,  # Convert to minutes
                            "zone_type": zone.zone_type,
                            "occupancy_rate": zone.occupied_spots / zone.capacity,
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
                    chosen_option = self.behavior_model.make_parking_decision(
                        driver["id"], available_options, self.current_time, urgency
                    )

                    if chosen_option:
                        zone_id = chosen_option["zone_id"]
                        zone = next(
                            (z for z in self.parking_zones if z.id == zone_id), None
                        )

                        # Successfully parked using psychological decision-making
                        zone.occupy_spot()
                        driver["assigned_zone"] = zone_id
                        driver["status"] = "parked"
                        driver["park_start_time"] = self.current_time

                        # Record search time
                        if driver.get("search_start_time"):
                            search_time = (
                                self.current_time - driver["search_start_time"]
                            ).total_seconds() / 60
                            self.metrics["avg_search_time"].append(search_time)

                        self.metrics["successful_parks"] += 1

                        # Update behavior model with experience
                        satisfaction = random.uniform(
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
            zone.release_spot()

        driver["status"] = "departed"

    def _update_metrics(self):
        """Update simulation metrics"""
        # Calculate occupancy rates
        total_spots = sum(zone.capacity for zone in self.parking_zones)
        occupied_spots = sum(zone.occupied_spots for zone in self.parking_zones)
        occupancy_rate = occupied_spots / total_spots if total_spots > 0 else 0

        self.metrics["avg_occupancy"].append(occupancy_rate)

        # Calculate revenue (simplified)
        total_revenue = 0
        for zone in self.parking_zones:
            revenue = (
                zone.occupied_spots * zone.hourly_rate * (1 / 12)
            )  # 5-minute increment
            total_revenue += revenue
        self.metrics["total_revenue"].append(total_revenue)

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
            avg_occupancy = np.mean(self.metrics["avg_occupancy"]) * 100
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
        print(f"📍 Real-world zones used: {len(self.parking_zones)}")
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
        results = {
            "simulation_config": {
                "use_real_data": self.use_real_data,
                "data_directory": self.data_directory,
                "n_drivers": self.n_drivers,
                "total_zones": len(self.parking_zones),
                "total_road_nodes": len(self.map_loader.road_network.nodes())
                if self.map_loader.road_network
                else 0,
            },
            "metrics": self.metrics,
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

        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, "w") as f:
            json.dump(results, f, indent=2, default=str)

        print(f"💾 Results saved to {filename}")


if __name__ == "__main__":
    # Run a sample simulation
    print("Parking Optimization Simulation")
    print("==============================\n")

    sim = CitySimulator(
        data_directory="output/map_data", n_drivers=500, use_real_data=True
    )

    sim.run_simulation(duration_hours=8, time_step_minutes=5)
    sim.save_results()
