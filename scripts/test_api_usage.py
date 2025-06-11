#!/usr/bin/env python3
"""
API Usage Calculator and Test Script
Shows exactly how many API calls different simulation configurations will make
"""

import os
import sys
import time

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.config import get_config
from core.traffic_manager import TrafficManager


def calculate_api_usage(duration_hours: float, time_step_minutes: int = 5):
    """Calculate expected API usage for a simulation"""
    config = get_config()

    # Simulation parameters
    total_steps = int((duration_hours * 60) // time_step_minutes)
    cache_ttl_minutes = config.api.api_cache_ttl_seconds // 60
    road_segments_per_update = 3  # From city_simulator.py

    # Cache cycles (how often cache expires and forces new calls)
    cache_cycles = max(1, int((duration_hours * 60) // cache_ttl_minutes))

    # Expected API calls (accounting for caching)
    expected_calls = cache_cycles * road_segments_per_update

    # Rate limiting constraints
    min_call_interval = 60 / config.api.max_api_calls_per_minute
    min_duration_for_all_calls = expected_calls * min_call_interval

    return {
        "duration_hours": duration_hours,
        "total_steps": total_steps,
        "cache_ttl_minutes": cache_ttl_minutes,
        "cache_cycles": cache_cycles,
        "expected_calls": expected_calls,
        "min_call_interval_seconds": min_call_interval,
        "min_duration_minutes": min_duration_for_all_calls / 60,
        "calls_per_hour": expected_calls / duration_hours if duration_hours > 0 else 0,
    }


def test_real_api_calls(num_calls: int = 5):
    """Test making real API calls to verify they work"""
    print(f"🧪 Testing {num_calls} real API calls...")

    tm = TrafficManager()

    # Grand Rapids coordinates for testing
    test_coords = [
        ((42.9634, -85.6681), (42.9698, -85.6553)),  # Downtown to Northeast
        ((42.9584, -85.6625), (42.9645, -85.6712)),  # Southeast to Northwest
        ((42.9612, -85.6634), (42.9687, -85.6598)),  # Central to East
        ((42.9598, -85.6689), (42.9671, -85.6642)),  # West to Central
        ((42.9655, -85.6701), (42.9623, -85.6599)),  # North to South
    ]

    results = []
    for i in range(min(num_calls, len(test_coords))):
        origin, dest = test_coords[i]

        print(f"📍 Call {i + 1}: {origin} -> {dest}")
        start_time = time.time()

        try:
            result = tm.get_traffic_conditions(origin, dest)
            call_duration = time.time() - start_time

            results.append(
                {
                    "call_number": i + 1,
                    "success": True,
                    "real_data": result.is_real_data,
                    "speed_kmh": result.speed_kmh,
                    "congestion_level": result.congestion_level,
                    "call_duration": call_duration,
                }
            )

            status = "✅ Real API" if result.is_real_data else "🔄 Fallback"
            print(
                f"   {status}: {result.speed_kmh:.1f} km/h, congestion {result.congestion_level}/4 ({call_duration:.2f}s)"
            )

        except Exception as e:
            results.append(
                {
                    "call_number": i + 1,
                    "success": False,
                    "error": str(e),
                    "call_duration": time.time() - start_time,
                }
            )
            print(f"   ❌ Failed: {e}")

    return results


def main():
    """Main test function"""
    print("=" * 60)
    print("🌐 PARKING OPTIMIZATION API USAGE CALCULATOR")
    print("=" * 60)

    # Show configuration
    config = get_config()
    print("\n📋 Current Configuration:")
    print(f"   Provider: {config.api.map_provider}")
    print(f"   API Key: {'✅ Available' if config.active_api_key else '❌ Missing'}")
    print(f"   Rate Limit: {config.api.max_api_calls_per_minute} calls/minute")
    print(f"   Cache TTL: {config.api.api_cache_ttl_seconds // 60} minutes")

    # Calculate usage for different simulation durations
    print("\n📊 Expected API Usage by Simulation Duration:")
    print(
        f"{'Duration':<12} {'Steps':<6} {'Cache':<6} {'API Calls':<10} {'Calls/Hour':<10}"
    )
    print("-" * 54)

    durations = [0.5, 1.0, 2.0, 4.0, 8.0]  # Hours
    for duration in durations:
        usage = calculate_api_usage(duration)
        print(
            f"{duration:>8.1f}h   {usage['total_steps']:>4}   {usage['cache_cycles']:>4}   {usage['expected_calls']:>8}     {usage['calls_per_hour']:>8.1f}"
        )

    # Test real API calls
    print("\n🧪 Testing Real API Calls:")
    print("-" * 40)

    api_results = test_real_api_calls(3)

    # Summary
    real_calls = sum(1 for r in api_results if r.get("real_data", False))
    successful_calls = sum(1 for r in api_results if r.get("success", False))

    print("\n📈 Test Results Summary:")
    print(f"   Total Attempts: {len(api_results)}")
    print(f"   Successful: {successful_calls}/{len(api_results)}")
    print(f"   Real API Data: {real_calls}/{len(api_results)}")
    print(f"   Fallback Used: {successful_calls - real_calls}/{len(api_results)}")

    if real_calls > 0:
        avg_speed = (
            sum(r.get("speed_kmh", 0) for r in api_results if r.get("real_data"))
            / real_calls
        )
        print(f"   Average Speed: {avg_speed:.1f} km/h")

    print("\n💡 To see more API calls in TomTom dashboard:")
    print("   • Run longer simulations (2+ hours)")
    print("   • Check dashboard after 5-10 minutes")
    print("   • Monitor during peak traffic hours")
    print("   • Consider reducing cache TTL for testing")


if __name__ == "__main__":
    main()
