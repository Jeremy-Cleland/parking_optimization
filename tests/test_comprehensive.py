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

import pytest

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis import ComplexityAnalyzer
from core import (
    CityCoordinator,
    DemandPredictor,
    DynamicPricingEngine,
    Edge,
    Node,
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
        assert sample_zone.try_park() is True
        assert sample_zone.occupied_spots == 1
        assert sample_zone.occupancy_rate == 0.02

        # Test leaving
        assert sample_zone.leave() is True
        assert sample_zone.occupied_spots == 0
        assert sample_zone.occupancy_rate == 0.0

    def test_capacity_limits(self, sample_zone):
        """Test capacity enforcement"""
        # Fill the zone
        for _ in range(sample_zone.capacity):
            assert sample_zone.try_park() is True

        # Try to exceed capacity
        assert sample_zone.try_park() is False
        assert sample_zone.is_full() is True

    def test_price_updates(self, sample_zone):
        """Test price updating"""
        new_price = 5.0
        sample_zone.update_price(new_price)
        assert sample_zone.current_price == new_price

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
                zone.try_park()
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
            high_occupancy_zone.try_park()

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
        optimizer = RouteOptimizer()

        # Create a simple grid network
        for i in range(5):
            for j in range(5):
                node = Node(id=f"node_{i}_{j}", location=(i * 0.1, j * 0.1))
                optimizer.add_node(node)

        # Add horizontal edges
        for i in range(5):
            for j in range(4):
                edge = Edge(
                    from_node=f"node_{i}_{j}",
                    to_node=f"node_{i}_{j + 1}",
                    distance=1.0,
                    base_travel_time=1.0,
                )
                optimizer.add_edge(edge)
                # Add reverse edge
                reverse_edge = Edge(
                    from_node=f"node_{i}_{j + 1}",
                    to_node=f"node_{i}_{j}",
                    distance=1.0,
                    base_travel_time=1.0,
                )
                optimizer.add_edge(reverse_edge)

        # Add vertical edges
        for i in range(4):
            for j in range(5):
                edge = Edge(
                    from_node=f"node_{i}_{j}",
                    to_node=f"node_{i + 1}_{j}",
                    distance=1.0,
                    base_travel_time=1.0,
                )
                optimizer.add_edge(edge)
                # Add reverse edge
                reverse_edge = Edge(
                    from_node=f"node_{i + 1}_{j}",
                    to_node=f"node_{i}_{j}",
                    distance=1.0,
                    base_travel_time=1.0,
                )
                optimizer.add_edge(reverse_edge)

        return optimizer

    def test_pathfinding(self, route_optimizer):
        """Test basic pathfinding functionality"""
        path, cost = route_optimizer._modified_a_star("node_0_0", "node_2_2")

        assert path is not None
        assert len(path) > 0
        assert path[0] == "node_0_0"
        assert path[-1] == "node_2_2"
        assert cost > 0

    def test_no_path_exists(self, route_optimizer):
        """Test handling when no path exists"""
        path, cost = route_optimizer._modified_a_star("node_0_0", "nonexistent_node")

        assert path == []
        assert cost == float("inf")

    @pytest.mark.benchmark(group="routing")
    def test_routing_performance(self, benchmark, route_optimizer):
        """Benchmark routing algorithm performance"""
        result = benchmark(route_optimizer._modified_a_star, "node_0_0", "node_4_4")

        path, cost = result
        assert len(path) > 0


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
        prediction = predictor.predict_demand(future_time, weather_condition=0)

        assert isinstance(prediction, (int, float))
        assert prediction >= 0


class TestIntegration:
    """Integration tests for the complete system"""

    def test_small_simulation(self):
        """Test a small complete simulation"""
        sim = CitySimulator(
            n_zones=3, n_intersections=6, n_drivers=10, city_size_km=1.0
        )

        # Run very short simulation
        sim.run_simulation(duration_hours=0.1, time_step_minutes=5)

        # Verify simulation ran
        assert sim.current_time is not None
        assert len(sim.zones) == 3

    @patch("core.traffic_manager.TrafficManager.get_real_time_traffic")
    def test_simulation_without_api(self, mock_traffic):
        """Test simulation works without real API calls"""
        # Mock the traffic API to return dummy data
        mock_traffic.return_value = 1.5  # 1.5x normal travel time

        sim = CitySimulator(n_zones=2, n_intersections=4, n_drivers=5, city_size_km=0.5)

        sim.run_simulation(duration_hours=0.05, time_step_minutes=5)

        # Verify it completed without errors
        assert len(sim.zones) == 2

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
            sim = CitySimulator(
                n_zones=5, n_intersections=15, n_drivers=25, city_size_km=2.0
            )
            sim.run_simulation(duration_hours=0.1, time_step_minutes=5)
            return sim

        result = benchmark(run_simulation)
        assert len(result.zones) == 5

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
        assert isinstance(config.has_mapbox, bool)
        assert isinstance(config.has_google_maps, bool)

    @pytest.mark.skipif(
        not os.environ.get("MAPBOX_ACCESS_TOKEN"), reason="Mapbox API key not available"
    )
    def test_mapbox_integration(self):
        """Test Mapbox API integration (if key available)"""
        from core.traffic_manager import TrafficManager

        traffic_manager = TrafficManager()

        # Test with real coordinates (if API key is available)
        traffic_factor = traffic_manager.get_real_time_traffic(
            (40.7128, -74.0060),  # NYC
            (40.7589, -73.9851),  # Times Square
        )

        assert isinstance(traffic_factor, (int, float))
        assert traffic_factor > 0


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
