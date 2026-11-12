"""Traffic observations with explicit sources, clocks, caching and offline replay."""

import math
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from core.config import get_config
from core.map_services import MapService, get_map_service


@dataclass
class TrafficCondition:
    segment_id: str
    speed_kmh: float
    congestion_level: int
    incident: bool
    travel_time_factor: float
    last_updated: datetime
    is_real_data: bool = False
    delay_factor: float = 1.0
    source: str = "simulated"


class TrafficManager:
    """Use measured duration ratios when available, otherwise labeled simulation.

    Incident counts alone do not establish edge travel times. Live providers must
    return a traffic duration and a comparable baseline duration to change weights.
    """

    def __init__(
        self, np_rng: Optional[np.random.Generator] = None, enable_live: bool = False
    ) -> None:
        config = get_config()
        self.np_rng = np_rng if np_rng is not None else np.random.default_rng()
        self.map_provider_name = config.api.map_provider
        self.map_service: Optional[MapService] = None
        if enable_live and config.active_api_key:
            self.map_service = get_map_service(
                self.map_provider_name, config.active_api_key
            )
        self.traffic_cache: Dict[str, TrafficCondition] = {}
        self.cache_duration = config.api.api_cache_ttl_seconds
        self.min_call_interval = 60 / config.api.max_api_calls_per_minute
        self.last_api_call: Dict[str, float] = {}

    def get_traffic_condition(
        self,
        origin: Tuple[float, float],
        destination: Tuple[float, float],
        current_time: Optional[datetime] = None,
    ) -> TrafficCondition:
        return self.get_traffic_conditions(origin, destination, current_time)

    def get_traffic_conditions(
        self,
        origin: Tuple[float, float],
        destination: Tuple[float, float],
        current_time: Optional[datetime] = None,
    ) -> TrafficCondition:
        now = current_time or datetime.now()
        segment_id = f"{origin}->{destination}"
        if self._is_cached_valid(segment_id, now):
            return self.traffic_cache[segment_id]
        condition = None
        if self.map_service and self._can_make_api_call(self.map_provider_name):
            # Account for unsuccessful calls as well as successful calls.
            self.last_api_call[self.map_provider_name] = time.monotonic()
            try:
                response = self.map_service.get_route(list(origin), list(destination))
                condition = self._adapt_service_output(response, segment_id)
            except Exception as exc:
                print(f"Traffic provider unavailable ({type(exc).__name__})")
        if condition is None:
            condition = self._get_fallback_traffic(origin, destination, now)
        self.traffic_cache[segment_id] = condition
        return condition

    def _adapt_service_output(
        self, service_data: Dict[str, Any], segment_id: str
    ) -> Optional[TrafficCondition]:
        """Parse comparable durations; never infer measured speed from incident count."""
        try:
            routes = service_data.get("routes", [])
            if self.map_provider_name == "google" and routes:
                legs = routes[0].get("legs", [])
                base = sum(leg["duration"]["value"] for leg in legs)
                observed = sum(leg["duration_in_traffic"]["value"] for leg in legs)
            elif self.map_provider_name == "tomtom" and routes:
                summary = routes[0]["summary"]
                observed = summary["travelTimeInSeconds"]
                base = summary["noTrafficTravelTimeInSeconds"]
            elif self.map_provider_name == "mapquest":
                route = service_data["route"]
                base, observed = route["time"], route["realTime"]
            else:
                return None
            if (
                not math.isfinite(base)
                or not math.isfinite(observed)
                or base <= 0
                or observed <= 0
            ):
                return None
            factor = observed / base
        except (KeyError, TypeError, ValueError):
            return None
        return TrafficCondition(
            segment_id,
            45 / factor,
            max(0, min(4, int((factor - 1) * 2))),
            False,
            factor,
            datetime.now(),
            True,
            factor,
            source=f"{self.map_provider_name}_route_duration_proxy",
        )

    def _get_fallback_traffic(
        self,
        origin: Tuple[float, float],
        destination: Tuple[float, float],
        current_time: Optional[datetime] = None,
    ) -> TrafficCondition:
        now = current_time or datetime.now()
        rush = now.weekday() < 5 and (7 <= now.hour <= 9 or 17 <= now.hour <= 19)
        factor = max(0.8, (1.8 if rush else 1.1) + float(self.np_rng.normal(0, 0.2)))
        return TrafficCondition(
            f"{origin}->{destination}",
            45 / factor,
            max(0, min(4, int((factor - 1) * 2))),
            False,
            factor,
            now,
            False,
            factor,
            "simulated",
        )

    def _is_cached_valid(
        self, segment_id: str, current_time: Optional[datetime] = None
    ) -> bool:
        condition = self.traffic_cache.get(segment_id)
        if condition is None:
            return False
        now = (
            datetime.now() if condition.is_real_data else current_time or datetime.now()
        )
        return 0 <= (now - condition.last_updated).total_seconds() < self.cache_duration

    def _can_make_api_call(self, api_name: str) -> bool:
        return (
            time.monotonic() - self.last_api_call.get(api_name, -math.inf)
            >= self.min_call_interval
        )

    def bulk_update_traffic(
        self, road_segments: List[Tuple]
    ) -> Dict[str, TrafficCondition]:
        conditions = [self.get_traffic_conditions(a, b) for a, b in road_segments]
        return {condition.segment_id: condition for condition in conditions}

    def get_traffic_summary(self) -> Dict:
        conditions = list(self.traffic_cache.values())
        if not conditions:
            return {}
        return {
            "avg_traffic_factor": float(
                np.mean([c.travel_time_factor for c in conditions])
            ),
            "cached_segments": len(conditions),
            "provider_observations": sum(c.is_real_data for c in conditions),
            "simulated_observations": sum(not c.is_real_data for c in conditions),
        }
