#!/usr/bin/env python3
"""
Comprehensive test suite for the parking optimization system.
Includes unit tests, integration tests, and performance benchmarks.
"""

import os
import sys
import time
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

import networkx as nx
import pytest

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis import ComplexityAnalyzer
from core import (
    CityCoordinator,
    DemandPredictor,
    DynamicPricingEngine,
    MapDataLoader,
    ParkingZone,
    RouteOptimizer,
)
from core.config import SystemConfig
from simulation import CitySimulator


class TestParkingZone:
    """Unit tests for ParkingZone class"""

    @pytest.fixture
    def sample_zone(self):
        """Create a sample parking zone for testing"""
        return ParkingZone(
            id="test_zone_1",
            name="Test Zone 1",
            location=(40.7128, -74.0060),  # NYC coordinates
            capacity=50,
            base_price=3.0,
            zone_type="commercial",
        )

    def test_zone_creation(self, sample_zone):
        """Test parking zone creation"""
        assert sample_zone.id == "test_zone_1"
        assert sample_zone.capacity == 50
        assert sample_zone.occupied_spots == 0
        assert sample_zone.occupancy_rate == 0.0
        assert sample_zone.current_price == 3.0

    def test_parking_operations(self, sample_zone):
        """Test parking and leaving operations"""
        # Test parking
        assert sample_zone.occupy_spot() is True
        assert sample_zone.occupied_spots == 1
        assert sample_zone.occupancy_rate == 0.02

        # Test leaving
        assert sample_zone.release_spot() is True
        assert sample_zone.occupied_spots == 0
        assert sample_zone.occupancy_rate == 0.0

    def test_capacity_limits(self, sample_zone):
        """Test capacity enforcement"""
        # Fill the zone
        for _ in range(sample_zone.capacity):
            assert sample_zone.occupy_spot() is True

        # Try to exceed capacity
        assert sample_zone.occupy_spot() is False
        assert sample_zone.has_availability() is False

    @pytest.mark.parametrize(
        "zone_type,expected_valid",
        [
            ("commercial", True),
            ("residential", True),
            ("event", True),
            ("invalid_type", True),  # Should still work but maybe log warning
        ],
    )
    def test_zone_types(self, zone_type, expected_valid):
        """Test different zone types"""
        zone = ParkingZone(
            id="test",
            name="Test",
            location=(0, 0),
            capacity=10,
            base_price=2.0,
            zone_type=zone_type,
        )
        assert zone.zone_type == zone_type


class TestDynamicPricing:
    """Unit tests for Dynamic Pricing Engine"""

    @pytest.fixture
    def pricing_engine(self):
        """Create a pricing engine for testing"""
        return DynamicPricingEngine(
            min_price=1.0, max_price=10.0, target_occupancy=0.85
        )

    @pytest.fixture
    def sample_zones(self):
        """Create sample zones for pricing tests"""
        zones = []
        for i in range(3):
            zone = ParkingZone(
                id=f"zone_{i}",
                name=f"Zone {i}",
                location=(i * 0.1, i * 0.1),
                capacity=30,
                base_price=3.0,
            )
            # Simulate different occupancy levels
            for _ in range(i * 10):
                zone.occupy_spot()
            zones.append(zone)
        return zones

    def test_price_calculation(self, pricing_engine, sample_zones):
        """Test basic price calculation"""
        zone = sample_zones[0]
        nearby_zones = sample_zones[1:]

        price = pricing_engine.calculate_zone_price(zone, nearby_zones)

        assert isinstance(price, float)
        assert pricing_engine.min_price <= price <= pricing_engine.max_price

    def test_occupancy_impact(self, pricing_engine):
        """Test that higher occupancy leads to higher prices"""
        # Create zones with different occupancy
        low_occupancy_zone = ParkingZone("low", "Low", (0, 0), 100, 3.0)
        high_occupancy_zone = ParkingZone("high", "High", (0, 0), 100, 3.0)

        # Fill high occupancy zone to 90%
        for _ in range(90):
            high_occupancy_zone.occupy_spot()

        low_price = pricing_engine.calculate_zone_price(low_occupancy_zone, [])
        high_price = pricing_engine.calculate_zone_price(high_occupancy_zone, [])

        assert high_price > low_price

    def test_price_bounds(self, pricing_engine, sample_zones):
        """Test that prices stay within bounds"""
        for zone in sample_zones:
            price = pricing_engine.calculate_zone_price(zone, [])
            assert pricing_engine.min_price <= price <= pricing_engine.max_price

    @pytest.mark.benchmark(group="pricing")
    def test_pricing_performance(self, benchmark, pricing_engine, sample_zones):
        """Benchmark pricing algorithm performance"""
        zone = sample_zones[0]
        nearby_zones = sample_zones[1:]

        result = benchmark(pricing_engine.calculate_zone_price, zone, nearby_zones)

        # Should complete in reasonable time
        assert isinstance(result, float)


class TestRouteOptimizer:
    """Unit tests for Route Optimizer"""

    @pytest.fixture
    def route_optimizer(self):
        """Create a route optimizer with sample network"""
        loader = MapDataLoader()
        graph = nx.MultiDiGraph(crs="EPSG:4326")
        for i in range(5):
            for j in range(5):
                graph.add_node(i * 5 + j, y=i * 0.001, x=j * 0.001)
        for i in range(5):
            for j in range(5):
                node = i * 5 + j
                for neighbor in [node + 1] if j < 4 else []:
                    graph.add_edge(node, neighbor, length=100, travel_time=10)
                    graph.add_edge(neighbor, node, length=100, travel_time=10)
                if i < 4:
                    graph.add_edge(node, node + 5, length=100, travel_time=10)
                    graph.add_edge(node + 5, node, length=100, travel_time=10)
        loader.road_network = graph
        loader._parking_zones = [
            {"id": "target", "coordinates": (0.002, 0.002), "hourly_rate": 2.0}
        ]
        return RouteOptimizer(map_loader=loader)

    def test_pathfinding(self, route_optimizer):
        """Test basic pathfinding functionality"""
        route = route_optimizer.find_optimal_route((0, 0))
        assert route is not None
        assert route.path_nodes[0] == 0
        assert route.path_nodes[-1] == 12
        assert route.total_distance == 400
        assert route.total_time == 40

    def test_no_available_zone(self, route_optimizer):
        assert (
            route_optimizer.find_optimal_route((0, 0), available_parking_zones=[])
            is None
        )

    @pytest.mark.benchmark(group="routing")
    def test_routing_performance(self, benchmark, route_optimizer):
        route = benchmark(route_optimizer.find_optimal_route, (0, 0))
        assert route is not None
        assert route.total_distance == 400


class TestDemandPredictor:
    """Unit tests for Demand Predictor"""

    @pytest.fixture
    def sample_data(self):
        """Create sample historical data"""
        data = []
        base_time = datetime.now()

        for i in range(168):  # One week
            data.append(
                {
                    "timestamp": base_time + timedelta(hours=i),
                    "occupancy_rate": 0.3
                    + 0.4 * abs(((i % 24) - 12) / 12),  # Peak at noon
                    "arrivals": 10 + int(5 * abs(((i % 24) - 12) / 12)),
                    "weather": 0 if i % 24 < 20 else 1,  # Rain in evening
                }
            )

        return data

    def test_model_training(self, sample_data):
        """Test demand prediction model training"""
        predictor = DemandPredictor()
        predictor.train_model(sample_data)

        assert predictor.dp_table is not None
        assert predictor.dp_table.shape[0] > 0

    def test_demand_prediction(self, sample_data):
        """Test demand prediction functionality"""
        predictor = DemandPredictor()
        predictor.train_model(sample_data)

        future_time = datetime.now() + timedelta(hours=1)
        prediction = predictor.predict_demand(
            future_time, current_occupancy=0.5, weather=0
        )

        assert len(prediction) == 4
        assert all(value >= 0 for value in prediction)


class TestIntegration:
    """Integration tests for the complete system"""

    def test_small_simulation(self):
        """Test a small complete simulation"""
        sim = CitySimulator(n_zones=3, n_drivers=10, use_real_data=False, seed=42)

        # Run very short simulation
        sim.run_simulation(duration_hours=0.1, time_step_minutes=5)

        # Verify simulation ran
        assert sim.current_time is not None
        assert len(sim.parking_zones) == 3

    @patch("core.traffic_manager.TrafficManager.get_traffic_condition")
    def test_simulation_without_api(self, mock_traffic):
        """Test simulation works without real API calls"""
        # Mock the traffic API to return dummy data
        from core.traffic_manager import TrafficCondition

        mock_traffic.return_value = TrafficCondition(
            "mock", 30, 1, False, 1.5, datetime(2026, 11, 12, 8)
        )

        sim = CitySimulator(n_zones=2, n_drivers=5, use_real_data=False, seed=42)

        sim.run_simulation(duration_hours=0.05, time_step_minutes=5)

        # Verify it completed without errors
        assert len(sim.parking_zones) == 2

    def test_configuration_loading(self):
        """Test that configuration loads correctly"""
        config = SystemConfig()

        assert config.simulation.default_zones > 0
        assert config.pricing.min_price_per_hour > 0
        assert config.pricing.max_price_per_hour > config.pricing.min_price_per_hour

        # Test validation
        warnings = config.validate()
        assert isinstance(warnings, list)

    def test_complexity_analysis(self):
        """Test complexity analysis runs without errors"""
        analyzer = ComplexityAnalyzer()

        # Test individual complexity analysis
        analyzer.analyze_pricing_complexity()

        # Should complete without errors
        assert True  # If we get here, no exceptions were thrown


class TestPerformance:
    """Performance and benchmark tests"""

    @pytest.mark.slow
    @pytest.mark.benchmark(group="simulation")
    def test_simulation_performance(self, benchmark):
        """Benchmark complete simulation performance"""

        def run_simulation():
            sim = CitySimulator(n_zones=5, n_drivers=25, use_real_data=False, seed=42)
            sim.run_simulation(duration_hours=0.1, time_step_minutes=5)
            return sim

        result = benchmark(run_simulation)
        assert len(result.parking_zones) == 5

    @pytest.mark.benchmark(group="algorithms")
    def test_algorithm_scaling(self, benchmark):
        """Test how algorithms scale with problem size"""

        def create_and_price_zones(n_zones):
            engine = DynamicPricingEngine()
            zones = []

            for i in range(n_zones):
                zone = ParkingZone(
                    id=f"zone_{i}",
                    name=f"Zone {i}",
                    location=(i * 0.1, i * 0.1),
                    capacity=30,
                    base_price=3.0,
                )
                zones.append(zone)

            # Calculate prices for all zones
            prices = []
            for zone in zones:
                nearby = [z for z in zones if z != zone][:5]  # Limit to 5 nearby
                price = engine.calculate_zone_price(zone, nearby)
                prices.append(price)

            return prices

        result = benchmark(create_and_price_zones, 20)
        assert len(result) == 20


@pytest.mark.integration
class TestAPIIntegration:
    """Tests for API integration (when available)"""

    def test_api_availability(self):
        """Test API key detection"""
        config = SystemConfig()

        # These tests should work regardless of whether API keys are present
        assert isinstance(config.has_api_keys, bool)
        assert isinstance(config.has_tomtom, bool)
        assert isinstance(config.has_google_maps, bool)


if __name__ == "__main__":
    # Run tests when executed directly
    pytest.main(
        [
            __file__,
            "-v",
            "--tb=short",
            "-x",  # Stop on first failure
            "--disable-warnings",
        ]
    )
