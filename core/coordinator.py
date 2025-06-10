"""
City-wide Coordinator using Divide-and-Conquer
Manages distributed parking optimization across city zones
"""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np

from .demand_predictor import DemandPredictor
from .dynamic_pricing import DynamicPricingEngine
from .parking_zone import ParkingZone
from .route_optimizer import RouteOptimizer


@dataclass
class District:
    """Represents a district containing multiple parking zones"""

    id: str
    name: str
    zones: List[ParkingZone]
    center: Tuple[float, float]

    @property
    def total_capacity(self) -> int:
        return sum(zone.capacity for zone in self.zones)

    @property
    def average_occupancy(self) -> float:
        if not self.zones:
            return 0.0
        return np.mean([zone.occupancy_rate for zone in self.zones])


class CityCoordinator:
    """
    Implements divide-and-conquer strategy for city-wide parking optimization
    Divides city into districts and optimizes hierarchically
    """

    def __init__(
        self, n_districts: int = 4, coordination_interval: int = 300
    ):  # 5 minutes
        """
        Initialize city coordinator

        Args:
            n_districts: Number of districts to divide city into
            coordination_interval: Seconds between coordination cycles
        """
        self.n_districts = n_districts
        self.coordination_interval = coordination_interval

        self.districts: Dict[str, District] = {}
        self.pricing_engine = DynamicPricingEngine()
        self.route_optimizer = RouteOptimizer()
        self.demand_predictors: Dict[str, DemandPredictor] = {}

        # Thread pool for parallel district optimization
        self.executor = ThreadPoolExecutor(max_workers=n_districts)

        # Global state tracking
        self.global_metrics = {
            "total_occupancy": 0.0,
            "average_price": 0.0,
            "total_revenue": 0.0,
            "search_time": 0.0,
        }

        # Inter-district coordination
        self.district_transfers = {}  # Track recommended transfers

    def divide_city_into_districts(self, all_zones: List[ParkingZone]):
        """
        Divide city zones into districts using spatial clustering

        Time Complexity: O(z log z) where z = number of zones

        Args:
            all_zones: List of all parking zones in the city
        """
        # Sort zones by location for spatial division
        zones_sorted_lat = sorted(all_zones, key=lambda z: z.location[0])

        # Simple grid-based division (can be improved with k-means)
        zones_per_district = len(all_zones) // self.n_districts

        for i in range(self.n_districts):
            start_idx = i * zones_per_district
            end_idx = (
                start_idx + zones_per_district
                if i < self.n_districts - 1
                else len(all_zones)
            )

            district_zones = zones_sorted_lat[start_idx:end_idx]

            # Calculate district center
            center_lat = np.mean([z.location[0] for z in district_zones])
            center_lon = np.mean([z.location[1] for z in district_zones])

            district = District(
                id=f"district_{i}",
                name=f"District {i + 1}",
                zones=district_zones,
                center=(center_lat, center_lon),
            )

            self.districts[district.id] = district

            # Initialize demand predictor for district
            self.demand_predictors[district.id] = DemandPredictor()

    def optimize_city_parking(self) -> Dict[str, any]:
        """
        Main optimization loop using divide-and-conquer

        Time Complexity: O(d * (z/d)²) = O(z²/d) where:
            d = number of districts
            z = total zones

        Returns:
            Global optimization results
        """
        # Phase 1: Optimize each district independently (DIVIDE)
        district_results = self._optimize_districts_parallel()

        # Phase 2: Coordinate between districts (CONQUER)
        coordination_results = self._coordinate_districts(district_results)

        # Phase 3: Apply global adjustments
        final_results = self._apply_global_optimization(
            district_results, coordination_results
        )

        # Update global metrics
        self._update_global_metrics(final_results)

        return final_results

    def _optimize_districts_parallel(self) -> Dict[str, Dict]:
        """
        Optimize each district in parallel
        """
        futures = {}

        for district_id, district in self.districts.items():
            future = self.executor.submit(self._optimize_single_district, district)
            futures[district_id] = future

        # Collect results
        results = {}
        for district_id, future in futures.items():
            results[district_id] = future.result()

        return results

    def _optimize_single_district(self, district: District) -> Dict:
        """
        Optimize a single district independently

        Steps:
        1. Predict demand for each zone
        2. Optimize pricing within district
        3. Balance capacity utilization
        """
        results = {
            "district_id": district.id,
            "zone_prices": {},
            "predicted_demands": {},
            "recommended_actions": [],
        }

        # Step 1: Predict demand for next hour
        for zone in district.zones:
            if zone.occupancy_history:
                predictor = self.demand_predictors[district.id]
                predictions = predictor.predict_demand(
                    datetime.now(), zone.occupancy_rate, lookahead_hours=1
                )
                results["predicted_demands"][zone.id] = predictions[0]

        # Step 2: Optimize pricing considering predictions
        zone_prices = self.pricing_engine.optimize_city_pricing(district.zones)
        results["zone_prices"] = zone_prices

        # Step 3: Identify imbalanced zones
        avg_occupancy = district.average_occupancy
        for zone in district.zones:
            if zone.occupancy_rate > 0.95:
                results["recommended_actions"].append(
                    {
                        "zone_id": zone.id,
                        "action": "redirect_overflow",
                        "severity": "high",
                    }
                )
            elif zone.occupancy_rate < 0.3 and avg_occupancy > 0.7:
                results["recommended_actions"].append(
                    {
                        "zone_id": zone.id,
                        "action": "attract_drivers",
                        "severity": "medium",
                    }
                )

        return results

    def _coordinate_districts(self, district_results: Dict[str, Dict]) -> Dict:
        """
        Coordinate between districts to balance load

        Uses greedy matching to suggest inter-district transfers
        """
        coordination = {"transfers": [], "border_adjustments": {}}

        # Find districts with overflow and underflow
        overflow_districts = []
        underflow_districts = []

        for district_id, results in district_results.items():
            district = self.districts[district_id]
            if district.average_occupancy > 0.9:
                overflow_districts.append((district_id, district.average_occupancy))
            elif district.average_occupancy < 0.5:
                underflow_districts.append((district_id, district.average_occupancy))

        # Greedy matching of overflow to underflow
        overflow_districts.sort(key=lambda x: x[1], reverse=True)
        underflow_districts.sort(key=lambda x: x[1])

        for overflow_id, overflow_occ in overflow_districts:
            if not underflow_districts:
                break

            # Find best match (closest district)
            best_match = None
            best_distance = float("inf")

            for i, (underflow_id, underflow_occ) in enumerate(underflow_districts):
                distance = self._district_distance(overflow_id, underflow_id)
                if distance < best_distance:
                    best_distance = distance
                    best_match = i

            if best_match is not None:
                underflow_id, underflow_occ = underflow_districts.pop(best_match)

                # Calculate transfer recommendation
                transfer_amount = min(
                    (overflow_occ - 0.85) * self.districts[overflow_id].total_capacity,
                    (0.7 - underflow_occ) * self.districts[underflow_id].total_capacity,
                )

                coordination["transfers"].append(
                    {
                        "from_district": overflow_id,
                        "to_district": underflow_id,
                        "recommended_flow": int(transfer_amount),
                        "distance": best_distance,
                    }
                )

        # Adjust border zone pricing to encourage transfers
        for transfer in coordination["transfers"]:
            # Increase prices in overflow district borders
            # Decrease prices in underflow district borders
            coordination["border_adjustments"][transfer["from_district"]] = 1.1
            coordination["border_adjustments"][transfer["to_district"]] = 0.9

        return coordination

    def _apply_global_optimization(
        self, district_results: Dict, coordination: Dict
    ) -> Dict:
        """
        Apply final global optimizations
        """
        final_results = {
            "timestamp": datetime.now().isoformat(),
            "district_results": district_results,
            "coordination": coordination,
            "updated_prices": {},
            "routing_recommendations": {},
        }

        # Apply coordinated price adjustments
        for district_id, adjustment in coordination["border_adjustments"].items():
            district = self.districts[district_id]
            for zone in district.zones:
                # Adjust border zones more than interior zones
                if self._is_border_zone(zone, district):
                    old_price = district_results[district_id]["zone_prices"][zone.id]
                    new_price = old_price * adjustment
                    new_price = np.clip(
                        new_price,
                        self.pricing_engine.min_price,
                        self.pricing_engine.max_price,
                    )
                    final_results["updated_prices"][zone.id] = new_price

        return final_results

    def _district_distance(self, district1_id: str, district2_id: str) -> float:
        """Calculate distance between district centers"""
        d1 = self.districts[district1_id]
        d2 = self.districts[district2_id]

        lat_diff = d1.center[0] - d2.center[0]
        lon_diff = d1.center[1] - d2.center[1]

        return np.sqrt(lat_diff**2 + lon_diff**2) * 111  # km approximation

    def _is_border_zone(self, zone: ParkingZone, district: District) -> bool:
        """Check if zone is on district border (simplified)"""
        # Check if zone is far from district center
        lat_diff = abs(zone.location[0] - district.center[0])
        lon_diff = abs(zone.location[1] - district.center[1])
        distance = np.sqrt(lat_diff**2 + lon_diff**2)

        # If zone is in top 25% furthest from center, consider it border
        all_distances = []
        for z in district.zones:
            d = np.sqrt(
                (z.location[0] - district.center[0]) ** 2
                + (z.location[1] - district.center[1]) ** 2
            )
            all_distances.append(d)

        threshold = np.percentile(all_distances, 75)
        return distance >= threshold

    def _update_global_metrics(self, results: Dict):
        """Update global performance metrics"""
        total_occupied = 0
        total_capacity = 0
        total_revenue = 0

        for district in self.districts.values():
            for zone in district.zones:
                occupied = sum(1 for spot in zone.spots if spot.is_occupied)
                total_occupied += occupied
                total_capacity += zone.capacity
                total_revenue += occupied * zone.current_price

        self.global_metrics["total_occupancy"] = (
            total_occupied / total_capacity if total_capacity > 0 else 0
        )
        self.global_metrics["total_revenue"] = total_revenue

        # Calculate average price
        all_prices = []
        for district_id, d_results in results["district_results"].items():
            all_prices.extend(d_results["zone_prices"].values())

        self.global_metrics["average_price"] = np.mean(all_prices) if all_prices else 0

    def analyze_coordination_complexity(self) -> Dict[str, str]:
        """
        Return complexity analysis of coordination algorithms
        """
        return {
            "divide_city": "O(z log z) where z = number of zones",
            "optimize_districts_parallel": "O(z²/d) with d parallel threads",
            "coordinate_districts": "O(d²) for inter-district matching",
            "overall_complexity": "O(z²/d + d²) - optimal when d = √z",
            "space_complexity": "O(z) for storing zone states",
            "parallelization": f"Up to {self.n_districts}x speedup with threading",
            "communication_overhead": "O(d²) for district coordination",
        }
