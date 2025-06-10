#!/usr/bin/env python3
"""
Test script to verify parking optimization system functionality
"""

import os
import sys
import time

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
from simulation import CitySimulator


def test_core_components():
    """Test individual core components"""
    print("Testing Core Components")
    print("=" * 50)

    # Test 1: Parking Zone
    print("\n1. Testing Parking Zone...")
    zone = ParkingZone(
        id="test_zone",
        name="Test Zone",
        location=(0.0, 0.0),
        capacity=50,
        base_price=3.0,
    )
    print(f"   ✓ Created zone with {zone.capacity} spots")
    print(f"   ✓ Initial occupancy: {zone.occupancy_rate:.1%}")

    # Test 2: Dynamic Pricing
    print("\n2. Testing Dynamic Pricing Engine...")
    pricing_engine = DynamicPricingEngine()
    price = pricing_engine.calculate_zone_price(zone, [], demand_forecast=25)
    print(f"   ✓ Calculated price: ${price:.2f}")

    # Test 3: Route Optimizer
    print("\n3. Testing Route Optimizer...")
    router = RouteOptimizer()

    # Add test nodes
    for i in range(5):
        node = Node(id=f"node_{i}", location=(i * 0.1, i * 0.1))
        router.add_node(node)

    # Add edges
    for i in range(4):
        edge = Edge(
            from_node=f"node_{i}",
            to_node=f"node_{i + 1}",
            distance=1.0,
            base_travel_time=2.0,
        )
        router.add_edge(edge)

    # Test pathfinding
    path, cost = router._modified_a_star("node_0", "node_4")
    print(f"   ✓ Found path: {' -> '.join(path)}")
    print(f"   ✓ Path cost: {cost:.1f} minutes")

    # Test 4: Demand Predictor
    print("\n4. Testing Demand Predictor...")
    predictor = DemandPredictor()

    # Generate sample data - format must match what train_model expects
    sample_data = []
    from datetime import datetime, timedelta

    base_time = datetime.now()

    for i in range(168):  # 168 hours = 1 week
        sample_data.append(
            {
                "timestamp": base_time + timedelta(hours=i),
                "occupancy_rate": 0.7,  # Use 'occupancy_rate' for train_model
                "arrivals": 20,
                "weather": 0,
            }
        )

    predictor.train_model(sample_data)
    print(f"   ✓ Built DP table with shape: {predictor.dp_table.shape}")

    # Test 5: City Coordinator
    print("\n5. Testing City Coordinator...")
    coordinator = CityCoordinator(n_districts=2)

    # Create test zones
    test_zones = []
    for i in range(4):
        z = ParkingZone(
            id=f"coord_zone_{i}",
            name=f"Zone {i}",
            location=(i * 0.1, i * 0.1),
            capacity=30,
            base_price=2.5,
        )
        test_zones.append(z)

    coordinator.divide_city_into_districts(test_zones)
    print(
        f"   ✓ Divided {len(test_zones)} zones into {coordinator.n_districts} districts"
    )

    print("\n✅ All core components tested successfully!")


def test_simulation():
    """Test simulation with minimal parameters"""
    print("\n\nTesting Simulation")
    print("=" * 50)

    # Create small simulation
    sim = CitySimulator(n_zones=4, n_intersections=10, n_drivers=20, city_size_km=2.0)

    print(f"✓ Created city with {sim.n_zones} zones")
    print(f"✓ Road network has {sim.n_intersections} intersections")

    # Run short simulation
    print("\nRunning 30-minute simulation...")
    sim.run_simulation(duration_hours=0.5, time_step_minutes=5)

    print("\n✅ Simulation completed successfully!")


def test_analysis():
    """Test analysis components"""
    print("\n\nTesting Analysis Components")
    print("=" * 50)

    # Test complexity analyzer
    print("\n1. Testing Complexity Analyzer...")
    analyzer = ComplexityAnalyzer()

    # Test just pricing complexity (quick)
    analyzer.analyze_pricing_complexity()
    print("   ✓ Complexity analysis completed")

    print("\n✅ Analysis components tested successfully!")


def main():
    """Run all tests"""
    print("\nParking Optimization System - Component Tests")
    print("=" * 60)

    start_time = time.time()

    try:
        # Test individual components
        test_core_components()

        # Test simulation (simplified)
        test_simulation()

        # Test analysis
        test_analysis()

        elapsed = time.time() - start_time
        print(f"\n\n{'=' * 60}")
        print(f"✅ ALL TESTS PASSED in {elapsed:.2f} seconds!")
        print(f"{'=' * 60}")

    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback

        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
