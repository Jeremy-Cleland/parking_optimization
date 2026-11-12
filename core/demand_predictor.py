"""
Demand Prediction using Dynamic Programming
Predicts parking demand patterns for optimal resource allocation
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import numpy as np


@dataclass
class DemandState:
    """Represents a state in the DP formulation"""

    time_slot: int  # 0-167 for hourly slots in a week
    occupancy_level: int  # Discretized occupancy (0-10)
    event_flag: bool  # Special event indicator
    weather_condition: int  # 0=clear, 1=rain, 2=snow


class DemandPredictor:
    """
    Dynamic Programming based demand prediction
    Models parking demand as a Markov Decision Process
    """

    def __init__(
        self,
        time_slots_per_day: int = 24,
        occupancy_levels: int = 11,
        history_weeks: int = 4,
        capacity: int = 1,
    ):
        """
        Initialize demand predictor

        Args:
            time_slots_per_day: Number of hourly slots per day
            occupancy_levels: Number of discretized occupancy levels
            history_weeks: Weeks of historical data to consider
        """
        if time_slots_per_day != 24 or occupancy_levels < 2 or capacity < 1:
            raise ValueError(
                "Use 24 hourly slots, at least two states and positive capacity"
            )
        self.capacity = capacity
        self.time_slots_per_day = time_slots_per_day
        self.time_slots_per_week = time_slots_per_day * 7
        self.occupancy_levels = occupancy_levels
        self.history_weeks = history_weeks

        # DP table: [time_slot][occupancy_level][weather] -> predicted_demand
        self.dp_table = np.zeros(
            (
                self.time_slots_per_week,
                occupancy_levels,
                3,  # weather conditions
            )
        )

        # Transition probabilities learned from data
        self.transition_probs = {}

        # Historical patterns
        self.weekly_patterns = []
        self.special_events = {}
        self.event_windows = []
        self.demand_scale = 1.0

    def train_model(self, historical_data: List[Dict]):
        """
        Train the DP model using historical parking data

        Time Complexity: O(t * s² * w) where:
            t = time slots (168)
            s = occupancy states (11)
            w = weather conditions (3)

        Args:
            historical_data: List of hourly records with occupancy, weather, etc.
        """
        # Organize data by time slots
        slot_data = [[] for _ in range(self.time_slots_per_week)]

        for record in historical_data:
            time_slot = self._get_time_slot(record["timestamp"])
            occupancy = self._discretize_occupancy(record["occupancy_rate"])
            weather = record.get("weather", 0)
            capacity = record.get("capacity", self.capacity)
            if (
                capacity <= 0
                or not np.isfinite(record["arrivals"])
                or record["arrivals"] < 0
            ):
                raise ValueError(
                    "Training arrivals must be nonnegative and capacity positive"
                )
            demand = record["arrivals"] / capacity  # Hourly arrivals per space

            slot_data[time_slot].append(
                {"occupancy": occupancy, "weather": weather, "demand": demand}
            )

        # Build DP table using Bellman equation
        self._build_dp_table(slot_data)

        # Learn transition probabilities
        self._learn_transitions(
            sorted(historical_data, key=lambda row: row["timestamp"])
        )

    def _build_dp_table(self, slot_data: List[List[Dict]]):
        """
        Build DP table for demand prediction
        Uses backward induction from observed demands
        """
        # Initialize with average demands
        for t in range(self.time_slots_per_week):
            for occ in range(self.occupancy_levels):
                for weather in range(3):
                    # Filter relevant data points
                    relevant_data = [
                        d["demand"]
                        for d in slot_data[t]
                        if d["occupancy"] == occ and d["weather"] == weather
                    ]

                    if relevant_data:
                        # Use mean as base prediction
                        self.dp_table[t][occ][weather] = np.mean(relevant_data)
                    else:
                        # Use interpolation from nearby states
                        self.dp_table[t][occ][weather] = self._interpolate_demand(
                            t, occ, weather, slot_data
                        )

    def predict_demand(
        self,
        current_time: datetime,
        current_occupancy: float,
        weather: int = 0,
        lookahead_hours: int = 4,
        capacity: Optional[int] = None,
    ) -> List[float]:
        """
        Predict parking demand for the next few hours

        Time Complexity: O(h * s) where h = lookahead hours, s = states

        Args:
            current_time: Current timestamp
            current_occupancy: Current occupancy rate (0-1)
            weather: Weather condition
            lookahead_hours: Hours to predict ahead

        Returns:
            List of predicted demands for each hour
        """
        capacity = self.capacity if capacity is None else capacity
        if capacity <= 0 or not np.isfinite(capacity):
            raise ValueError("Forecast capacity must be finite and positive")
        if (
            not 0 <= current_occupancy <= 1
            or weather not in (0, 1, 2)
            or lookahead_hours < 1
        ):
            raise ValueError("Invalid occupancy, weather, or forecast horizon")
        predictions = []
        time_slot = self._get_time_slot(current_time)
        occ_level = self._discretize_occupancy(current_occupancy)

        for hour in range(lookahead_hours):
            # Current slot (with wraparound)
            slot = (time_slot + hour) % self.time_slots_per_week

            # Base prediction from DP table
            base_demand = self.dp_table[slot][occ_level][weather]

            # Adjust for special events
            event_factor = self._get_event_factor(current_time + timedelta(hours=hour))

            # Apply day-of-week patterns
            # Weekly slots already encode day-of-week; do not apply it twice.
            dow_factor = 1.0

            # Final prediction
            predicted_demand = float(
                base_demand * event_factor * dow_factor * capacity * self.demand_scale
            )
            predictions.append(predicted_demand)

            # Update occupancy estimate for next prediction
            # Use more realistic capacity-aware transition model
            arrivals = predicted_demand
            departures = self._estimate_departures(occ_level, slot, capacity)

            # Get the actual capacity instead of assuming 100
            net_change = (arrivals - departures) / capacity

            # Update occupancy level with bounds checking
            current_occupancy = occ_level / (self.occupancy_levels - 1)
            new_occupancy = np.clip(current_occupancy + net_change, 0, 1)
            occ_level = int(new_occupancy * (self.occupancy_levels - 1))

        return predictions

    def _learn_transitions(self, historical_data: List[Dict]):
        """
        Learn state transition probabilities for the MDP
        """
        transitions = {}

        for i in range(len(historical_data) - 1):
            current = historical_data[i]
            next_state = historical_data[i + 1]

            # Create state keys
            current_state = (
                self._get_time_slot(current["timestamp"]),
                self._discretize_occupancy(current["occupancy_rate"]),
                current.get("weather", 0),
            )

            next_occ = self._discretize_occupancy(next_state["occupancy_rate"])

            # Count transitions
            if current_state not in transitions:
                transitions[current_state] = {}

            if next_occ not in transitions[current_state]:
                transitions[current_state][next_occ] = 0

            transitions[current_state][next_occ] += 1

        # Normalize to probabilities
        for state, next_states in transitions.items():
            total = sum(next_states.values())
            for next_state in next_states:
                transitions[state][next_state] /= total

        self.transition_probs = transitions

    def _get_time_slot(self, timestamp: datetime) -> int:
        """Convert timestamp to weekly time slot (0-167)"""
        dow = timestamp.weekday()
        hour = timestamp.hour
        return dow * 24 + hour

    def _discretize_occupancy(self, occupancy_rate: float) -> int:
        """Convert continuous occupancy rate to discrete level"""
        return int(occupancy_rate * (self.occupancy_levels - 1))

    def _interpolate_demand(
        self, time_slot: int, occ: int, weather: int, slot_data: List[List[Dict]]
    ) -> float:
        """
        Interpolate demand when no exact historical match exists
        Uses weighted average of nearby states
        """
        total_weight = 0
        weighted_sum = 0

        # Check nearby occupancy levels
        for occ_offset in [-1, 0, 1]:
            check_occ = occ + occ_offset
            if 0 <= check_occ < self.occupancy_levels:
                relevant_data = [
                    d["demand"]
                    for d in slot_data[time_slot]
                    if d["occupancy"] == check_occ and d["weather"] == weather
                ]

                if relevant_data:
                    weight = 1.0 / (abs(occ_offset) + 1)
                    weighted_sum += np.mean(relevant_data) * weight
                    total_weight += weight

        if total_weight > 0:
            return weighted_sum / total_weight
        else:
            # Fallback to time-based average
            all_demands = [d["demand"] for d in slot_data[time_slot]]
            return float(np.mean(all_demands)) if all_demands else 0.1

    def _get_event_factor(self, timestamp: datetime) -> float:
        """Get demand multiplier for special events"""
        for start, end, multiplier in self.event_windows:
            if start <= timestamp < end:
                return multiplier
        date_key = timestamp.date()
        return self.special_events.get(date_key, 1.0)

    def _get_dow_factor(self, time_slot: int) -> float:
        """Get day-of-week adjustment factor"""
        dow = time_slot // 24
        dow_factors = [0.9, 1.0, 1.0, 1.0, 1.1, 1.3, 1.2]  # Mon-Sun
        return dow_factors[dow]

    def _estimate_departures(
        self, occupancy_level: int, time_slot: int, capacity: int
    ) -> float:
        """Estimate departures based on occupancy and time"""
        # Simple model: more departures when fuller and during transition times
        hour = time_slot % 24

        # Higher departure rate during transition hours
        if hour in [8, 12, 17, 19]:  # Rush hours, lunch
            base_rate = 0.3
        else:
            base_rate = 0.1

        # Scale by occupancy
        return base_rate * occupancy_level / (self.occupancy_levels - 1) * capacity

    def evaluate(self, observations: List[Dict]) -> Dict:
        """Score unseen hourly arrivals; the caller supplies a chronological holdout."""
        errors = []
        actuals = []
        for row in observations:
            forecast = self.predict_demand(
                row["timestamp"],
                row["occupancy_rate"],
                weather=row.get("weather", 0),
                lookahead_hours=1,
                capacity=row.get("capacity", self.capacity),
            )[0]
            errors.append(abs(forecast - row["arrivals"]))
            actuals.append(row["arrivals"])
        return {
            "n": len(errors),
            "mae_arrivals": float(np.mean(errors)) if errors else None,
            "wape": float(sum(errors) / sum(actuals)) if sum(actuals) else None,
        }

    def analyze_prediction_complexity(self) -> Dict[str, str]:
        """
        Return complexity analysis of prediction algorithms
        """
        return {
            "train_model": "O(t * s² * w) where t=time slots, s=states, w=weather",
            "predict_demand": "O(h * s) where h=lookahead hours",
            "space_complexity": "O(t * s * w) for DP table",
            "convergence": "Not established; empirical forecast evaluation required",
            "accuracy_notes": "Depends on discretization granularity and historical data quality",
        }
