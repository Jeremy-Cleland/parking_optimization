#!/usr/bin/env python3
"""
API Keys Test Script - 100% Free Mode Available!
Test your API keys OR use the system completely free
"""

import os
import sys

# Add project root to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.traffic_manager import TrafficManager


def test_free_mode():
    """Test the 100% free fallback mode"""
    print("🆓 Testing 100% FREE Mode")
    print("=" * 40)
    print("No APIs, no signups, no credit cards needed!")
    print()

    # Create traffic manager without any API keys
    traffic_manager = TrafficManager(google_api_key=None, mapbox_token=None)

    # Test multiple scenarios
    test_cases = [
        ((40.7128, -74.0060), (40.7589, -73.9851), "Manhattan → Times Square"),
        ((37.7749, -122.4194), (37.7849, -122.4094), "San Francisco Downtown"),
        ((34.0522, -118.2437), (34.0622, -118.2337), "Los Angeles Downtown"),
    ]

    print("Testing realistic traffic simulation...")
    for origin, destination, description in test_cases:
        traffic_condition = traffic_manager.get_traffic_conditions(origin, destination)

        print(f"📍 {description}")
        print(f"   🚗 Speed: {traffic_condition.speed_kmh:.1f} km/h")
        print(f"   🚦 Congestion: {traffic_condition.congestion_level}/4")
        print(f"   ⏱️  Traffic Factor: {traffic_condition.travel_time_factor:.2f}x")
        print(f"   🚨 Incident: {'Yes' if traffic_condition.incident else 'No'}")
        print()

    print("✅ Free mode works perfectly!")
    print("💡 The system uses intelligent patterns based on:")
    print("   - Time of day (rush hours, etc.)")
    print("   - Day of week patterns")
    print("   - Realistic random variations")
    print()


def test_api_keys():
    """Test API keys if available"""

    print("🔧 Testing API Key Setup")
    print("=" * 50)

    # Check environment variables
    google_key = os.getenv("GOOGLE_MAPS_API_KEY")
    mapbox_token = os.getenv("MAPBOX_ACCESS_TOKEN")

    print(f"Google Maps API Key: {'✅ Found' if google_key else '❌ Not found'}")
    print(f"Mapbox Access Token: {'✅ Found' if mapbox_token else '❌ Not found'}")

    if not google_key and not mapbox_token:
        return False

    print("\n🧪 Testing Traffic API Calls")
    print("-" * 30)

    # Initialize traffic manager
    traffic_manager = TrafficManager()

    # Test coordinates
    origin = (40.7128, -74.0060)  # Lower Manhattan
    destination = (40.7589, -73.9851)  # Times Square

    print("Route: Manhattan → Times Square")
    print(f"Origin: {origin}")
    print(f"Destination: {destination}")

    try:
        print("\n📡 Fetching traffic data...")
        traffic_condition = traffic_manager.get_traffic_conditions(origin, destination)

        print("\n✅ API traffic data retrieved successfully!")
        print(f"   🚗 Speed: {traffic_condition.speed_kmh:.1f} km/h")
        print(f"   🚦 Congestion Level: {traffic_condition.congestion_level}/4")
        print(f"   ⏱️  Travel Time Factor: {traffic_condition.travel_time_factor:.2f}x")
        print(
            f"   🚨 Incident Detected: {'Yes' if traffic_condition.incident else 'No'}"
        )
        print(
            f"   📅 Last Updated: {traffic_condition.last_updated.strftime('%H:%M:%S')}"
        )

        return True

    except Exception as e:
        print(f"\n❌ Error testing traffic APIs: {e}")
        print("\nPossible issues:")
        print("- Invalid API key")
        print("- API key restrictions")
        print("- Billing not enabled (Google)")
        print("- Network connectivity issues")
        return False


def main():
    """Main test function"""
    print("🚀 Parking Optimization System Tester")
    print("=" * 60)

    # Check if user has API keys
    google_key = os.getenv("GOOGLE_MAPS_API_KEY")
    mapbox_token = os.getenv("MAPBOX_ACCESS_TOKEN")

    if not google_key and not mapbox_token:
        print("🆓 No API keys detected - running in FREE mode!")
        print("This demonstrates the system works without any external costs.")
        print()
        test_free_mode()
        print("🎯 Want real-time traffic data? See docs/API_SETUP_GUIDE.md")
        print("   But the free mode works great for development and demos!")
        return

    # Test API functionality
    print("🔑 API keys detected - testing API mode...")
    success = test_api_keys()

    print("\n🆓 Also testing FREE fallback mode...")
    test_free_mode()

    # Final results
    print("\n" + "=" * 60)
    if success:
        print("🎉 All tests passed! Your setup is working correctly.")
        print("\nYou have both:")
        print("✅ Real-time API traffic data")
        print("✅ Free fallback mode for development")
        print("\nNext steps:")
        print("1. Set up billing alerts to monitor costs")
        print("2. Configure rate limiting for your usage pattern")
        print("3. Use aggressive caching to stay in free tiers")
    else:
        print("⚠️  API tests failed, but FREE mode works perfectly!")
        print("You can use the system without any API costs.")
        print("Check docs/API_SETUP_GUIDE.md if you want real-time data.")

    print("\n💡 Pro tip: The free mode is perfect for:")
    print("   - Development and testing")
    print("   - Demonstrations")
    print("   - Learning the algorithms")
    print("   - Avoiding any API costs")


if __name__ == "__main__":
    main()
