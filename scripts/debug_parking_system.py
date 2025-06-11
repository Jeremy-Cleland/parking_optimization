#!/usr/bin/env python3
"""
Comprehensive debugging script for parking optimization system
Analyzes why drivers aren't successfully finding parking
"""

import os
import random
import sys

import networkx as nx

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.logger import get_logger
from core.map_data_loader import get_map_data_loader
from core.route_optimizer import RouteOptimizer

logger = get_logger(__name__)


def analyze_road_network_connectivity():
    """Analyze road network connectivity issues"""
    print("\n🔍 ANALYZING ROAD NETWORK CONNECTIVITY")
    print("=" * 50)

    map_loader = get_map_data_loader()
    if not map_loader.is_data_available():
        print("❌ Map data not available!")
        return

    road_network = map_loader.road_network
    parking_zones = map_loader.get_parking_zones()

    print("📊 Network Stats:")
    print(f"   • Nodes: {len(road_network.nodes())}")
    print(f"   • Edges: {len(road_network.edges())}")
    print(f"   • Parking Zones: {len(parking_zones)}")

    # Analyze connected components
    components = list(nx.strongly_connected_components(road_network))
    print(f"\n🔗 Connected Components: {len(components)}")

    for i, component in enumerate(components):
        print(f"   Component {i + 1}: {len(component)} nodes")
        if len(component) < 10:  # Show small components
            print(f"      Nodes: {list(component)}")

    # Find largest component
    largest_component = max(components, key=len)
    print(
        f"\n🏆 Largest Component: {len(largest_component)} nodes ({len(largest_component) / len(road_network.nodes()) * 100:.1f}%)"
    )

    return largest_component, components, parking_zones


def analyze_parking_zone_reachability(largest_component, parking_zones):
    """Analyze which parking zones are reachable"""
    print("\n🅿️ ANALYZING PARKING ZONE REACHABILITY")
    print("=" * 50)

    route_optimizer = RouteOptimizer()
    reachable_zones = []
    unreachable_zones = []

    # Test a node from the largest component
    next(iter(largest_component))

    for zone in parking_zones:
        zone_node = route_optimizer._find_nearest_node(zone["coordinates"])
        if zone_node and zone_node in largest_component:
            reachable_zones.append(zone)
        else:
            unreachable_zones.append(zone)

    print(f"✅ Reachable zones: {len(reachable_zones)}")
    print(f"❌ Unreachable zones: {len(unreachable_zones)}")
    print(
        f"📈 Reachability rate: {len(reachable_zones) / (len(reachable_zones) + len(unreachable_zones)) * 100:.1f}%"
    )

    if unreachable_zones:
        print("\n⚠️ Unreachable zones (first 5):")
        for zone in unreachable_zones[:5]:
            print(f"   • {zone['id']}: {zone['coordinates']}")

    return reachable_zones


def test_driver_scenarios(reachable_zones, largest_component):
    """Test specific driver scenarios"""
    print("\n🚗 TESTING DRIVER SCENARIOS")
    print("=" * 50)

    route_optimizer = RouteOptimizer()
    map_loader = get_map_data_loader()

    # Get simulation bounds
    bounds = map_loader.get_simulation_bounds()
    if not bounds:
        bounds = (42.956, -85.683, 42.973, -85.668)  # Grand Rapids downtown

    lat_min, lon_min, lat_max, lon_max = bounds
    print(
        f"📍 Simulation bounds: ({lat_min:.4f}, {lon_min:.4f}) to ({lat_max:.4f}, {lon_max:.4f})"
    )

    success_count = 0
    total_tests = 20

    for i in range(total_tests):
        # Generate random driver location (same as simulation)
        start_lat = random.uniform(lat_min, lat_max)
        start_lon = random.uniform(lon_min, lon_max)
        start_location = (start_lat, start_lon)

        print(f"\n🧪 Test {i + 1}: Driver at {start_location}")

        # Find nearest node
        start_node = route_optimizer._find_nearest_node(start_location)
        if start_node is None:
            print("   ❌ No nearest node found")
            continue

        print(f"   📍 Nearest node: {start_node}")

        # Check if in largest component
        if start_node not in largest_component:
            print("   ⚠️ Node not in largest component")
            continue

        # Try to find parking
        recommendations = route_optimizer.find_optimal_parking(
            start_location=start_location,
            preferences={
                "max_walk_distance": 0.5,
                "price_weight": 0.5,
                "time_weight": 0.5,
            },
        )

        if recommendations:
            print(f"   ✅ Found {len(recommendations)} parking options")
            for zone_id, route_info in list(recommendations.items())[
                :2
            ]:  # Show first 2
                print(
                    f"      • {zone_id}: {route_info.total_distance:.0f}m, {route_info.total_time:.1f}s"
                )
            success_count += 1
        else:
            print("   ❌ No parking recommendations found")

    print("\n📊 SCENARIO TEST RESULTS:")
    print(
        f"   Success Rate: {success_count}/{total_tests} ({success_count / total_tests * 100:.1f}%)"
    )

    return success_count / total_tests


def analyze_route_optimizer_internals():
    """Deep dive into route optimizer internals"""
    print("\n🔧 ANALYZING ROUTE OPTIMIZER INTERNALS")
    print("=" * 50)

    route_optimizer = RouteOptimizer()

    print("🔍 Route Optimizer Status:")
    print(f"   • Is operational: {route_optimizer.is_operational()}")
    print(f"   • Has road graph: {route_optimizer.road_graph is not None}")
    print(f"   • Parking zones count: {len(route_optimizer.parking_zones)}")
    print(
        f"   • Connected components: {len(route_optimizer._connected_components) if route_optimizer._connected_components else 'None'}"
    )

    # Test a simple parking zone lookup
    if route_optimizer.parking_zones:
        test_zone = route_optimizer.parking_zones[0]
        print("\n🧪 Testing zone lookup:")
        print(f"   • Test zone: {test_zone['id']} at {test_zone['coordinates']}")

        zone_node = route_optimizer._find_nearest_node(test_zone["coordinates"])
        print(f"   • Nearest node: {zone_node}")

        if zone_node:
            component = route_optimizer._get_component_for_node(zone_node)
            print(f"   • Component size: {len(component) if component else 'None'}")


def suggest_fixes():
    """Suggest specific fixes for identified issues"""
    print("\n🛠️ SUGGESTED FIXES")
    print("=" * 50)

    fixes = [
        "1. 🎯 Driver Spawn Strategy:",
        "   • Only spawn drivers in largest connected component",
        "   • Use actual road nodes as spawn points instead of random coordinates",
        "",
        "2. 🔗 Network Connectivity:",
        "   • Filter parking zones to only include reachable ones at startup",
        "   • Add connectivity check in driver generation",
        "",
        "3. 📝 Enhanced Logging:",
        "   • Add debug logging to route optimizer",
        "   • Log driver parking attempts with detailed failure reasons",
        "",
        "4. 🎲 Fallback Strategy:",
        "   • If no reachable zones found, assign to closest zone anyway",
        "   • Add teleport option for unreachable destinations (simulation mode)",
        "",
        "5. 📊 Data Collection:",
        "   • Ensure search times are recorded properly",
        "   • Add complexity metrics collection during simulation",
    ]

    for fix in fixes:
        print(fix)


def main():
    """Run comprehensive debugging analysis"""
    print("🔍 PARKING SYSTEM DEBUGGING ANALYSIS")
    print("=" * 60)

    try:
        # Step 1: Analyze network connectivity
        (
            largest_component,
            components,
            parking_zones,
        ) = analyze_road_network_connectivity()

        # Step 2: Analyze parking zone reachability
        reachable_zones = analyze_parking_zone_reachability(
            largest_component, parking_zones
        )

        # Step 3: Test driver scenarios
        success_rate = test_driver_scenarios(reachable_zones, largest_component)

        # Step 4: Analyze route optimizer internals
        analyze_route_optimizer_internals()

        # Step 5: Suggest fixes
        suggest_fixes()

        print("\n🎯 SUMMARY:")
        print(f"   • Road network has {len(components)} components")
        print(f"   • {len(reachable_zones)}/{len(parking_zones)} zones reachable")
        print(f"   • Driver test success rate: {success_rate * 100:.1f}%")

        if success_rate < 0.5:
            print("   ⚠️ LOW SUCCESS RATE - Major fixes needed!")
        elif success_rate < 0.8:
            print("   ⚠️ MODERATE SUCCESS RATE - Some fixes needed")
        else:
            print("   ✅ Good success rate - Minor optimizations possible")

    except Exception as e:
        logger.error(f"Debugging analysis failed: {e}")
        print(f"❌ Analysis failed: {e}")


if __name__ == "__main__":
    main()
