#!/usr/bin/env python3
"""
Performance profiling script for the parking optimization system.
Analyzes bottlenecks and provides optimization recommendations.
"""

import cProfile
import io
import os
import pstats
import sys
import time
from datetime import datetime
from typing import Dict, List

import memory_profiler

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import DynamicPricingEngine, MapDataLoader, ParkingZone, RouteOptimizer
from core.config import get_config
from core.logger import get_logger
from simulation import CitySimulator

logger = get_logger("profiler")


class PerformanceProfiler:
    """Comprehensive performance profiler for the parking optimization system"""

    def __init__(self):
        self.config = get_config()
        self.results = {}
        self.profiling_dir = self.config.performance.profile_output_dir
        self.profiling_dir.mkdir(exist_ok=True, parents=True)

    def profile_pricing_engine(self, n_zones: int = 50) -> Dict:
        """Profile the dynamic pricing engine performance"""
        logger.info(f"Profiling pricing engine with {n_zones} zones")

        # Create test data
        engine = DynamicPricingEngine()
        zones = []

        for i in range(n_zones):
            zone = ParkingZone(
                id=f"zone_{i}",
                name=f"Zone {i}",
                location=(42.96 + i * 0.001, -85.67 + i * 0.001),  # Grand Rapids area
                capacity=50,
                base_price=3.0,
            )
            # Add some random occupancy
            for _ in range(i % 30):
                if zone.has_availability():
                    zone.occupy_spot()
            zones.append(zone)

        # Profile the pricing calculation
        def price_all_zones():
            for zone in zones:
                nearby = [z for z in zones if z != zone][:10]  # Limit nearby zones
                engine.calculate_zone_price(zone, nearby)

        # CPU profiling
        profiler = cProfile.Profile()
        start_time = time.time()

        profiler.enable()
        price_all_zones()
        profiler.disable()

        end_time = time.time()

        # Memory profiling
        memory_usage = memory_profiler.memory_usage((price_all_zones, ()))

        # Save detailed profile
        profile_file = self.profiling_dir / f"pricing_engine_{n_zones}zones.prof"
        profiler.dump_stats(str(profile_file))

        # Generate stats
        s = io.StringIO()
        ps = pstats.Stats(profiler, stream=s)
        ps.sort_stats("cumulative").print_stats(20)

        results = {
            "n_zones": n_zones,
            "total_time": end_time - start_time,
            "avg_time_per_zone": (end_time - start_time) / n_zones,
            "memory_peak_mb": max(memory_usage),
            "memory_baseline_mb": min(memory_usage),
            "profile_file": str(profile_file),
            "top_functions": s.getvalue(),
        }

        logger.info(
            f"Pricing engine profiling complete: {results['total_time']:.3f}s total"
        )
        return results

    def profile_route_optimizer(self, use_real_data: bool = True) -> Dict:
        """Profile the route optimizer performance using real Grand Rapids data"""
        logger.info(f"Profiling route optimizer with real_data={use_real_data}")

        if use_real_data:
            # Use real Grand Rapids data
            map_loader = MapDataLoader("output/map_data")
            try:
                map_loader.load_all_data()
                optimizer = RouteOptimizer(
                    road_network=map_loader.road_network,
                    parking_zones=map_loader.parking_zones,
                )
                node_count = len(map_loader.road_network.nodes())
                edge_count = len(map_loader.road_network.edges())
            except Exception as e:
                logger.warning(f"Could not load real data: {e}, using fallback")
                use_real_data = False

        if not use_real_data:
            # Fallback: create minimal test network
            import networkx as nx

            road_network = nx.Graph()
            # Add a small test network
            for i in range(20):
                for j in range(20):
                    node_id = i * 20 + j
                    road_network.add_node(
                        node_id, x=-85.67 + j * 0.001, y=42.96 + i * 0.001
                    )
                    if j > 0:  # Connect to left neighbor
                        road_network.add_edge(node_id, node_id - 1, length=100)
                    if i > 0:  # Connect to top neighbor
                        road_network.add_edge(node_id, node_id - 20, length=100)

            optimizer = RouteOptimizer(road_network=road_network, parking_zones=[])
            node_count = len(road_network.nodes())
            edge_count = len(road_network.edges())

        # Profile pathfinding
        def find_multiple_paths():
            paths = []
            nodes = list(optimizer.road_graph.nodes())
            if len(nodes) < 2:
                return []

            for _ in range(
                min(10, len(nodes) // 2)
            ):  # Find paths based on available nodes
                start_node = nodes[0]
                end_node = nodes[-1] if len(nodes) > 1 else nodes[0]

                try:
                    # Test the find_optimal_parking method
                    start_coords = (
                        optimizer.map_loader.get_node_coordinates(start_node)
                        if hasattr(optimizer, "map_loader")
                        else (42.96, -85.67)
                    )
                    recommendations = optimizer.find_optimal_parking(
                        start_location=start_coords,
                        preferences={"max_walk_distance": 0.5},
                    )
                    paths.append(recommendations)
                except Exception as e:
                    logger.debug(f"Path finding error: {e}")

            return paths

        # CPU profiling
        profiler = cProfile.Profile()
        start_time = time.time()

        profiler.enable()
        paths = find_multiple_paths()
        profiler.disable()

        end_time = time.time()

        # Memory profiling
        memory_usage = memory_profiler.memory_usage((find_multiple_paths, ()))

        # Save profile
        profile_file = (
            self.profiling_dir
            / f"route_optimizer_{'real' if use_real_data else 'synthetic'}.prof"
        )
        profiler.dump_stats(str(profile_file))

        results = {
            "use_real_data": use_real_data,
            "node_count": node_count,
            "edge_count": edge_count,
            "total_time": end_time - start_time,
            "avg_time_per_path": (end_time - start_time) / max(1, len(paths)),
            "memory_peak_mb": max(memory_usage) if memory_usage else 0,
            "successful_paths": len(paths),
            "profile_file": str(profile_file),
        }

        logger.info(
            f"Route optimizer profiling complete: {results['total_time']:.3f}s total"
        )
        return results

    def profile_simulation(
        self, n_drivers: int = 50, duration_hours: float = 0.2
    ) -> Dict:
        """Profile a complete simulation using real Grand Rapids data"""
        logger.info(
            f"Profiling simulation with {n_drivers} drivers for {duration_hours} hours"
        )

        def run_simulation():
            sim = CitySimulator(
                data_directory="output/map_data",
                n_drivers=n_drivers,
                use_real_data=True,
            )
            sim.run_simulation(duration_hours=duration_hours, time_step_minutes=5)
            return sim

        # CPU profiling
        profiler = cProfile.Profile()
        start_time = time.time()

        profiler.enable()
        try:
            sim = run_simulation()
        except Exception as e:
            logger.error(f"Simulation failed: {e}")
            sim = None
        profiler.disable()

        end_time = time.time()

        # Memory profiling
        memory_usage = memory_profiler.memory_usage((run_simulation, ()))

        # Save profile
        profile_file = self.profiling_dir / f"simulation_{n_drivers}drivers.prof"
        profiler.dump_stats(str(profile_file))

        results = {
            "n_drivers": n_drivers,
            "duration_hours": duration_hours,
            "total_time": end_time - start_time,
            "memory_peak_mb": max(memory_usage) if memory_usage else 0,
            "simulation_successful": sim is not None,
            "profile_file": str(profile_file),
        }

        if sim:
            results.update(
                {
                    "total_zones": len(sim.parking_zones),
                    "road_nodes": len(sim.map_loader.road_network.nodes()),
                    "successful_parks": sim.metrics.get("successful_parks", 0),
                    "rejected_drivers": sim.metrics.get("rejected_drivers", 0),
                }
            )

        logger.info(
            f"Simulation profiling complete: {results['total_time']:.3f}s total"
        )
        return results

    def benchmark_scaling(self) -> Dict:
        """Benchmark how algorithms scale with problem size"""
        logger.info("Running scaling benchmarks")

        scaling_results = {"pricing": [], "routing": [], "simulation": []}

        # Test different problem sizes
        zone_sizes = [5, 10, 20, 30, 50]
        grid_sizes = [5, 10, 15, 20]
        sim_sizes = [(5, 20), (10, 50), (15, 100)]

        # Pricing scaling
        for n_zones in zone_sizes:
            result = self.profile_pricing_engine(n_zones)
            scaling_results["pricing"].append(result)

        # Routing scaling
        for grid_size in grid_sizes:
            result = self.profile_route_optimizer(grid_size)
            scaling_results["routing"].append(result)

        # Simulation scaling
        for n_zones, n_drivers in sim_sizes:
            result = self.profile_simulation(n_drivers)
            scaling_results["simulation"].append(result)

        return scaling_results

    def analyze_bottlenecks(self, profile_file: str) -> Dict:
        """Analyze a profile file to identify bottlenecks"""
        stats = pstats.Stats(profile_file)

        # Get top time-consuming functions
        s = io.StringIO()
        stats.sort_stats("cumulative").print_stats(10)
        top_functions = s.getvalue()

        # Get function call counts
        s = io.StringIO()
        stats.sort_stats("calls").print_stats(10)
        most_called = s.getvalue()

        # Get per-call time
        s = io.StringIO()
        stats.sort_stats("pcalls").print_stats(10)
        per_call_time = s.getvalue()

        return {
            "top_time_consumers": top_functions,
            "most_called_functions": most_called,
            "per_call_analysis": per_call_time,
        }

    def generate_optimization_recommendations(self, results: Dict) -> List[str]:
        """Generate optimization recommendations based on profiling results"""
        recommendations = []

        # Check pricing performance
        if "pricing" in results:
            for result in results["pricing"]:
                if result["avg_time_per_zone"] > 0.1:  # > 100ms per zone
                    recommendations.append(
                        f"Pricing engine is slow for {result['n_zones']} zones "
                        f"({result['avg_time_per_zone']:.3f}s per zone). "
                        "Consider caching competition factors or reducing nearby zone count."
                    )

                if result["memory_peak_mb"] > 500:
                    recommendations.append(
                        f"High memory usage in pricing engine ({result['memory_peak_mb']:.1f}MB). "
                        "Consider processing zones in batches."
                    )

        # Check routing performance
        if "routing" in results:
            for result in results["routing"]:
                if result["avg_time_per_path"] > 0.5:  # > 500ms per path
                    recommendations.append(
                        f"Route finding is slow for {result['node_count']} nodes "
                        f"({result['avg_time_per_path']:.3f}s per path). "
                        "Consider implementing bidirectional A* or hierarchical pathfinding."
                    )

        # Check simulation performance
        if "simulation" in results:
            for result in results["simulation"]:
                time_per_driver = result["total_time"] / result["n_drivers"]
                if time_per_driver > 0.1:  # > 100ms per driver
                    recommendations.append(
                        f"Simulation is slow ({time_per_driver:.3f}s per driver). "
                        "Consider parallel driver processing or simplified models."
                    )

        return recommendations

    def create_performance_report(self) -> str:
        """Create a comprehensive performance report"""
        logger.info("Generating comprehensive performance report")

        # Run all benchmarks
        scaling_results = self.benchmark_scaling()

        # Generate recommendations
        recommendations = self.generate_optimization_recommendations(scaling_results)

        # Create report
        report_file = (
            self.profiling_dir
            / f"performance_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        )

        with open(report_file, "w") as f:
            f.write("# Parking Optimization System - Performance Report\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            # Executive Summary
            f.write("## Executive Summary\n\n")
            if recommendations:
                f.write("### Key Recommendations\n\n")
                for i, rec in enumerate(recommendations, 1):
                    f.write(f"{i}. {rec}\n\n")
            else:
                f.write("✅ No major performance issues detected.\n\n")

            # Detailed Results
            f.write("## Detailed Results\n\n")

            # Pricing Engine
            f.write("### Dynamic Pricing Engine\n\n")
            f.write(
                "| Zones | Total Time (s) | Time per Zone (ms) | Memory Peak (MB) |\n"
            )
            f.write(
                "|-------|----------------|-------------------|------------------|\n"
            )
            for result in scaling_results["pricing"]:
                f.write(
                    f"| {result['n_zones']} | {result['total_time']:.3f} | "
                    f"{result['avg_time_per_zone'] * 1000:.1f} | {result['memory_peak_mb']:.1f} |\n"
                )
            f.write("\n")

            # Route Optimizer
            f.write("### Route Optimizer\n\n")
            f.write(
                "| Grid Size | Nodes | Edges | Time per Path (ms) | Memory Peak (MB) |\n"
            )
            f.write(
                "|-----------|-------|-------|-------------------|------------------|\n"
            )
            for result in scaling_results["routing"]:
                f.write(
                    f"| {result['grid_size']}x{result['grid_size']} | "
                    f"{result['node_count']} | {result['edge_count']} | "
                    f"{result['avg_time_per_path'] * 1000:.1f} | {result['memory_peak_mb']:.1f} |\n"
                )
            f.write("\n")

            # Simulation
            f.write("### Complete Simulation\n\n")
            f.write(
                "| Zones | Drivers | Total Time (s) | Time per Driver (ms) | Memory Peak (MB) |\n"
            )
            f.write(
                "|-------|---------|----------------|---------------------|------------------|\n"
            )
            for result in scaling_results["simulation"]:
                time_per_driver = result["total_time"] / result["n_drivers"] * 1000
                f.write(
                    f"| {result['n_zones']} | {result['n_drivers']} | "
                    f"{result['total_time']:.3f} | {time_per_driver:.1f} | {result['memory_peak_mb']:.1f} |\n"
                )
            f.write("\n")

            # Profile Files
            f.write("## Profile Files\n\n")
            f.write("Detailed profiling data available in:\n\n")
            for category in scaling_results:
                for result in scaling_results[category]:
                    if "profile_file" in result:
                        f.write(f"- `{result['profile_file']}`\n")
            f.write("\n")

            f.write("## Analysis Commands\n\n")
            f.write("To analyze profile files in detail:\n\n")
            f.write("```bash\n")
            f.write(
                "python -c \"import pstats; pstats.Stats('profile_file.prof').sort_stats('cumulative').print_stats(20)\"\n"
            )
            f.write("```\n\n")

        logger.info(f"Performance report saved to: {report_file}")
        return str(report_file)


def main():
    """Main profiling execution"""
    profiler = PerformanceProfiler()

    print("🔍 Performance Profiling - Parking Optimization System")
    print("=" * 60)

    try:
        # Generate comprehensive report
        report_file = profiler.create_performance_report()

        print("\n✅ Performance analysis complete!")
        print(f"📊 Report saved to: {report_file}")
        print(f"📁 Profile files available in: {profiler.profiling_dir}")

        # Quick summary
        print("\n📈 Quick Performance Summary:")
        print("-" * 30)

        # Test quick operations
        start = time.time()
        engine = DynamicPricingEngine()
        zone = ParkingZone("test", "Test", (0, 0), 50, 3.0)
        price = engine.calculate_zone_price(zone, [])
        pricing_time = time.time() - start

        print(f"Single price calculation: {pricing_time * 1000:.2f}ms")
        print(f"Estimated capacity: ~{int(1.0 / pricing_time)} calculations/second")

    except Exception as e:
        logger.error(f"Profiling failed: {e}")
        print(f"❌ Profiling failed: {e}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
