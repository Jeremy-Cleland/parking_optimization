"""
Real-Time Traffic Management
Integrates with external traffic APIs for realistic traffic conditions
"""

import time
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np

from core.config import get_config
from core.map_services import MapService, get_map_service


@dataclass
class TrafficCondition:
    """Represents traffic condition for a road segment"""

    segment_id: str
    speed_kmh: float
    congestion_level: int  # 0-4 (0=free flow, 4=heavy congestion)
    incident: bool
    travel_time_factor: float  # Multiplier for base travel time
    last_updated: datetime
    is_real_data: bool = False  # Whether this data came from real API or fallback
    delay_factor: float = 1.0  # Alias for travel_time_factor


class TrafficManager:
    """
    Manages real-time traffic data from multiple sources
    """

    def __init__(self):
        """
        Initialize traffic manager using the application configuration.
        """
        config = get_config()
        self.map_provider_name = config.api.map_provider
        api_key = config.active_api_key

        print("🔧 Initializing TrafficManager:")
        print(f"   Provider: {self.map_provider_name}")
        print(f"   API Key Available: {'✅ Yes' if api_key else '❌ No'}")
        print(f"   Has API Keys: {config.has_api_keys}")

        self.map_service: Optional[MapService] = None
        if api_key:
            try:
                self.map_service = get_map_service(self.map_provider_name, api_key)
                print(
                    f"   Map Service: ✅ {self.map_service.__class__.__name__} initialized"
                )
            except (ValueError, ImportError) as e:
                print(f"   Map Service: ❌ Failed to initialize: {e}")
        else:
            print("   Map Service: ❌ No API key available")

        # Cache to avoid excessive API calls
        self.traffic_cache = {}
        self.cache_duration = config.api.api_cache_ttl_seconds

        # Fallback patterns when APIs are unavailable
        self.fallback_patterns = self._initialize_fallback_patterns()

        # Rate limiting
        self.last_api_call = {}
        self.min_call_interval = 60 / config.api.max_api_calls_per_minute
        print(
            f"   Rate Limit: {config.api.max_api_calls_per_minute} calls/min ({self.min_call_interval:.1f}s between calls)"
        )

    def get_traffic_condition(
        self, origin: Tuple[float, float], destination: Tuple[float, float]
    ) -> TrafficCondition:
        """Alias for get_traffic_conditions (singular form)"""
        return self.get_traffic_conditions(origin, destination)

    def get_traffic_conditions(
        self, origin: Tuple[float, float], destination: Tuple[float, float]
    ) -> TrafficCondition:
        """
        Get current traffic conditions between two points

        Args:
            origin: (lat, lon) starting point
            destination: (lat, lon) ending point

        Returns:
            TrafficCondition object with current traffic data
        """
        segment_id = f"{origin[0]:.4f},{origin[1]:.4f}->{destination[0]:.4f},{destination[1]:.4f}"

        # Check cache first
        if self._is_cached_valid(segment_id):
            return self.traffic_cache[segment_id]

        # Try real APIs with rate limiting
        traffic_data = None

        if self.map_service and self._can_make_api_call(self.map_provider_name):
            try:
                print(
                    f"🌐 Making API call to {self.map_provider_name} for {segment_id}"
                )
                # Use bounding box for incidents, or route for specific traffic
                bbox = [origin[0], origin[1], destination[0], destination[1]]
                service_data = self.map_service.get_traffic_incidents(bounding_box=bbox)
                self.last_api_call[self.map_provider_name] = time.time()

                if service_data:
                    print(f"✅ API response received: {len(service_data)} items")
                    traffic_data = self._adapt_service_output(service_data, segment_id)
                else:
                    print("⚠️  API returned empty response")

            except Exception as e:
                print(f"❌ API call to {self.map_provider_name} failed: {e}")
        else:
            reasons = []
            if not self.map_service:
                reasons.append("no map service")
            if not self._can_make_api_call(self.map_provider_name):
                time_since = time.time() - self.last_api_call.get(
                    self.map_provider_name, 0
                )
                reasons.append(
                    f"rate limited (last call {time_since:.1f}s ago, need {self.min_call_interval:.1f}s)"
                )
            print(f"⚠️  Skipping API call: {', '.join(reasons)}")

        # Fallback to pattern-based simulation
        if traffic_data is None:
            print(f"🔄 Using fallback traffic simulation for {segment_id}")
            traffic_data = self._get_fallback_traffic(origin, destination)

        # Cache the result
        self.traffic_cache[segment_id] = traffic_data

        return traffic_data

    def _adapt_service_output(
        self, service_data: Dict[str, any], segment_id: str
    ) -> Optional[TrafficCondition]:
        """Adapts the output from a map service to a TrafficCondition object."""
        if not service_data:
            return None

        # This is a generic adapter. It needs to be made more specific for each provider's output.
        # For now, let's assume a simple structure.
        # Example for MapQuest-like incidents data
        incidents = service_data.get("incidents", [])
        congestion_level = 0
        has_incident = False
        if incidents:
            has_incident = True
            # A simple heuristic to determine congestion
            congestion_level = len(incidents)  # More incidents = more congestion
            congestion_level = min(4, congestion_level)

        # A more sophisticated adapter would parse routes for travel time, etc.
        # This is a simplified version for demonstration.
        # We will assume a baseline traffic factor and adjust it based on incidents.
        traffic_factor = 1.0 + (congestion_level * 0.2)
        free_flow_speed = 45  # km/h
        actual_speed = free_flow_speed / traffic_factor

        return TrafficCondition(
            segment_id=segment_id,
            speed_kmh=actual_speed,
            congestion_level=congestion_level,
            incident=has_incident,
            travel_time_factor=traffic_factor,
            last_updated=datetime.now(),
            is_real_data=True,
            delay_factor=traffic_factor,
        )

    def _get_fallback_traffic(
        self, origin: Tuple[float, float], destination: Tuple[float, float]
    ) -> TrafficCondition:
        """Generate realistic traffic patterns when APIs are unavailable"""
        current_hour = datetime.now().hour
        weekday = datetime.now().weekday()

        # Base traffic factor from patterns
        base_factor = self.fallback_patterns[weekday][current_hour]

        # Add some randomness
        noise = np.random.normal(0, 0.2)
        traffic_factor = max(0.8, base_factor + noise)

        # Calculate derived metrics
        free_flow_speed = 45  # km/h
        actual_speed = free_flow_speed / traffic_factor
        congestion_level = min(4, int((traffic_factor - 1) * 2))

        return TrafficCondition(
            segment_id=f"{origin}->{destination}",
            speed_kmh=actual_speed,
            congestion_level=congestion_level,
            incident=traffic_factor > 2.5 and np.random.random() < 0.1,
            travel_time_factor=traffic_factor,
            last_updated=datetime.now(),
            is_real_data=False,
            delay_factor=traffic_factor,
        )

    def _initialize_fallback_patterns(self) -> List[List[float]]:
        """Initialize realistic traffic patterns for fallback"""
        # 7 days x 24 hours of traffic factors
        patterns = []

        for day in range(7):  # 0=Monday, 6=Sunday
            daily_pattern = []

            for hour in range(24):
                if day < 5:  # Weekdays
                    if 7 <= hour <= 9:  # Morning rush
                        factor = np.random.uniform(1.5, 2.2)
                    elif 17 <= hour <= 19:  # Evening rush
                        factor = np.random.uniform(1.6, 2.4)
                    elif 10 <= hour <= 16:  # Business hours
                        factor = np.random.uniform(1.1, 1.4)
                    elif 20 <= hour <= 22:  # Evening
                        factor = np.random.uniform(1.0, 1.3)
                    else:  # Late night/early morning
                        factor = np.random.uniform(0.8, 1.0)
                else:  # Weekends
                    if 10 <= hour <= 14:  # Weekend shopping/activity
                        factor = np.random.uniform(1.2, 1.6)
                    elif 19 <= hour <= 22:  # Weekend evening
                        factor = np.random.uniform(1.1, 1.5)
                    else:
                        factor = np.random.uniform(0.9, 1.1)

                daily_pattern.append(factor)

            patterns.append(daily_pattern)

        return patterns

    def _is_cached_valid(self, segment_id: str) -> bool:
        """Check if cached data is still valid"""
        if segment_id not in self.traffic_cache:
            return False

        cached_condition = self.traffic_cache[segment_id]
        age = (datetime.now() - cached_condition.last_updated).total_seconds()

        return age < self.cache_duration

    def _can_make_api_call(self, api_name: str) -> bool:
        """Check rate limiting for API calls"""
        if api_name not in self.last_api_call:
            return True

        time_since_last = time.time() - self.last_api_call[api_name]
        return time_since_last >= self.min_call_interval

    def bulk_update_traffic(
        self, road_segments: List[Tuple[Tuple[float, float], Tuple[float, float]]]
    ) -> Dict[str, TrafficCondition]:
        """
        Efficiently update traffic for multiple road segments

        Args:
            road_segments: List of (origin, destination) tuples

        Returns:
            Dictionary mapping segment_id to TrafficCondition
        """
        results = {}

        # Prioritize segments that need updates
        segments_to_update = []
        for origin, destination in road_segments:
            segment_id = f"{origin[0]:.4f},{origin[1]:.4f}->{destination[0]:.4f},{destination[1]:.4f}"

            if not self._is_cached_valid(segment_id):
                segments_to_update.append((origin, destination, segment_id))
            else:
                results[segment_id] = self.traffic_cache[segment_id]

        # Update segments in batches to respect rate limits
        for origin, destination, segment_id in segments_to_update:
            traffic_condition = self.get_traffic_conditions(origin, destination)
            results[segment_id] = traffic_condition

            # Small delay to respect rate limits
            time.sleep(0.1)

        return results

    def get_traffic_summary(self) -> Dict[str, float]:
        """Get summary statistics of current traffic conditions"""
        if not self.traffic_cache:
            return {}

        traffic_factors = [tc.travel_time_factor for tc in self.traffic_cache.values()]
        congestion_levels = [tc.congestion_level for tc in self.traffic_cache.values()]

        return {
            "avg_traffic_factor": np.mean(traffic_factors),
            "max_traffic_factor": np.max(traffic_factors),
            "avg_congestion": np.mean(congestion_levels),
            "incidents_count": sum(
                1 for tc in self.traffic_cache.values() if tc.incident
            ),
            "cached_segments": len(self.traffic_cache),
        }
