"""
Psychological Driver Behavior Model
Models realistic driver decision-making with personality traits and behavioral patterns
"""

import os
import random
import sys
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.exceptions import DriverError


class PersonalityType(Enum):
    OPTIMIZER = "optimizer"  # Always seeks best option
    SATISFICER = "satisficer"  # Takes first "good enough" option
    RISK_AVERSE = "risk_averse"  # Prefers certainty over potential savings
    RISK_SEEKING = "risk_seeking"  # Willing to gamble for better deals
    HABITUAL = "habitual"  # Sticks to familiar choices
    SOCIAL = "social"  # Influenced by others' choices


class UrgencyLevel(Enum):
    LOW = 1  # Plenty of time, can be picky
    MEDIUM = 2  # Some time pressure
    HIGH = 3  # Running late, will take anything
    CRITICAL = 4  # Desperate, will pay premium prices


@dataclass
class DriverProfile:
    """Complete psychological profile of a driver"""

    id: str
    personality_type: PersonalityType
    income_level: float  # 0-1 scale
    tech_savviness: float  # 0-1 scale (affects app usage)
    environmental_consciousness: float  # 0-1 scale
    patience_level: float  # 0-1 scale
    price_sensitivity: float  # 0-1 scale
    time_value: float  # $/hour - how much they value their time

    # Habitual patterns
    preferred_zones: List[str]
    usual_arrival_times: List[int]  # Hours of day

    # Social influence
    follows_crowds: bool
    trusts_recommendations: bool

    # Learning and adaptation
    learning_rate: float  # How quickly they adapt to new information
    memory_span: int  # How many past experiences they remember


class DriverBehaviorModel:
    """
    Models complex driver decision-making with psychological realism
    """

    def __init__(self):
        self.driver_profiles = {}
        self.zone_popularity_history = {}  # Track social influence data
        self.pricing_memory = {}  # Drivers remember past prices

        # Behavioral parameters
        self.social_influence_strength = 0.3
        self.habit_strength = 0.4
        self.urgency_multipliers = {
            UrgencyLevel.LOW: 0.8,
            UrgencyLevel.MEDIUM: 1.0,
            UrgencyLevel.HIGH: 1.3,
            UrgencyLevel.CRITICAL: 2.0,
        }

    def create_driver_profile(self, driver_id: str) -> DriverProfile:
        """Create a realistic driver profile with psychological traits"""

        # Randomly assign personality type with realistic distributions
        personality_weights = {
            PersonalityType.OPTIMIZER: 0.15,
            PersonalityType.SATISFICER: 0.35,  # Most common
            PersonalityType.RISK_AVERSE: 0.20,
            PersonalityType.RISK_SEEKING: 0.10,
            PersonalityType.HABITUAL: 0.15,
            PersonalityType.SOCIAL: 0.05,
        }
        personality = np.random.choice(
            list(personality_weights.keys()), p=list(personality_weights.values())
        )

        # Generate correlated traits
        base_income = np.random.beta(2, 5)  # Skewed toward lower incomes
        tech_savvy = np.random.beta(3, 2) if base_income > 0.5 else np.random.beta(2, 3)

        # Price sensitivity inversely correlated with income
        price_sensitivity = 1 - base_income + np.random.normal(0, 0.2)
        price_sensitivity = np.clip(price_sensitivity, 0, 1)

        # Time value roughly correlated with income (but with variation)
        time_value = base_income * 50 + np.random.normal(0, 10)  # $/hour
        time_value = max(5, time_value)  # Minimum $5/hour

        profile = DriverProfile(
            id=driver_id,
            personality_type=personality,
            income_level=base_income,
            tech_savviness=tech_savvy,
            environmental_consciousness=np.random.beta(2, 2),
            patience_level=np.random.beta(3, 2)
            if personality != PersonalityType.RISK_SEEKING
            else np.random.beta(1, 3),
            price_sensitivity=price_sensitivity,
            time_value=time_value,
            preferred_zones=self._generate_preferred_zones(),
            usual_arrival_times=self._generate_arrival_patterns(),
            follows_crowds=personality == PersonalityType.SOCIAL
            or np.random.random() < 0.3,
            trusts_recommendations=tech_savvy > 0.6 and np.random.random() < 0.7,
            learning_rate=np.random.uniform(0.1, 0.5),
            memory_span=random.randint(5, 20),
        )

        self.driver_profiles[driver_id] = profile
        return profile

    def make_parking_decision(
        self,
        driver_id: str,
        available_options: List[Dict],
        current_time: datetime,
        urgency: UrgencyLevel = UrgencyLevel.MEDIUM,
    ) -> Optional[Dict]:
        """
        Make parking decision based on driver's psychological profile

        Args:
            driver_id: Driver identifier
            available_options: List of parking options with price, distance, etc.
            current_time: Current timestamp
            urgency: How urgent the parking need is

        Returns:
            Selected parking option or None if nothing acceptable
        """
        if not available_options:
            raise DriverError(
                f"No parking options available for driver {driver_id}",
                {"driver_id": driver_id, "options_count": 0},
            )

        if driver_id not in self.driver_profiles:
            self.create_driver_profile(driver_id)

        profile = self.driver_profiles[driver_id]

        if not available_options:
            return None

        # Filter options based on hard constraints first
        viable_options = self._filter_viable_options(
            available_options, profile, urgency
        )

        if not viable_options:
            return None

        # Score each option based on driver's preferences
        scored_options = []
        for option in viable_options:
            score = self._calculate_option_score(option, profile, current_time, urgency)
            scored_options.append((option, score))

        # Make decision based on personality type
        return self._make_personality_based_decision(scored_options, profile, urgency)

    def _filter_viable_options(
        self, options: List[Dict], profile: DriverProfile, urgency: UrgencyLevel
    ) -> List[Dict]:
        """Filter options based on hard constraints"""
        viable = []

        for option in options:
            # Price constraint (adjusted by urgency)
            max_acceptable_price = self._calculate_max_price(profile, urgency)
            if option["price"] > max_acceptable_price:
                continue

            # Distance constraint (varies by urgency and patience)
            max_walk_distance = profile.patience_level * 0.8  # km
            if urgency == UrgencyLevel.CRITICAL:
                max_walk_distance *= 1.5  # Will walk further when desperate

            if option.get("walk_distance", 0) > max_walk_distance:
                continue

            viable.append(option)

        return viable

    def _calculate_option_score(
        self,
        option: Dict,
        profile: DriverProfile,
        current_time: datetime,
        urgency: UrgencyLevel,
    ) -> float:
        """Calculate utility score for a parking option"""
        score = 0.0

        # Price component (negative utility)
        price_utility = -option["price"] * profile.price_sensitivity
        price_utility *= self.urgency_multipliers[urgency]

        # Distance/time component
        walk_time = option.get("walk_distance", 0) * 12  # Assume 12 min/km walking
        time_cost = walk_time / 60 * profile.time_value  # Convert to dollar cost
        distance_utility = -time_cost

        # Convenience factors
        convenience_score = 0
        if option.get("covered", False):  # Covered parking
            convenience_score += 0.2
        if option.get("security", False):  # Secure parking
            convenience_score += 0.3
        if (
            option.get("electric_charging", False)
            and profile.environmental_consciousness > 0.6
        ):
            convenience_score += 0.4

        # Habit/familiarity bonus
        zone_id = option.get("zone_id", "")
        if zone_id in profile.preferred_zones:
            habit_bonus = self.habit_strength * (
                1 if profile.personality_type == PersonalityType.HABITUAL else 0.5
            )
            convenience_score += habit_bonus

        # Social influence (if driver follows crowds)
        if profile.follows_crowds:
            popularity = self._get_zone_popularity(zone_id)
            social_bonus = self.social_influence_strength * popularity
            convenience_score += social_bonus

        # Combine all factors
        score = (
            price_utility + distance_utility + convenience_score * 10
        )  # Scale convenience

        # Add personality-specific adjustments
        if profile.personality_type == PersonalityType.RISK_SEEKING:
            # Add randomness for risk-seekers (might choose suboptimal options)
            score += np.random.normal(0, 0.5)
        elif profile.personality_type == PersonalityType.RISK_AVERSE:
            # Penalty for unknown zones
            if zone_id not in profile.preferred_zones:
                score -= 0.3

        return score

    def _make_personality_based_decision(
        self,
        scored_options: List[Tuple[Dict, float]],
        profile: DriverProfile,
        urgency: UrgencyLevel,
    ) -> Dict:
        """Make final decision based on personality type"""

        # Sort by score (descending)
        scored_options.sort(key=lambda x: x[1], reverse=True)

        if profile.personality_type == PersonalityType.OPTIMIZER:
            # Always choose the best option
            return scored_options[0][0]

        elif profile.personality_type == PersonalityType.SATISFICER:
            # Choose first "good enough" option
            threshold = np.percentile([score for _, score in scored_options], 70)
            for option, score in scored_options:
                if score >= threshold:
                    return option
            return scored_options[0][0]  # Fallback to best

        elif profile.personality_type == PersonalityType.RISK_AVERSE:
            # Prefer options with lower variance in outcomes
            # For simplicity, prefer familiar zones
            for option, _score in scored_options:
                if option.get("zone_id", "") in profile.preferred_zones:
                    return option
            return scored_options[0][0]

        elif profile.personality_type == PersonalityType.RISK_SEEKING:
            # Sometimes choose suboptimal options for variety
            if np.random.random() < 0.3:  # 30% chance of non-optimal choice
                choice_idx = min(random.randint(1, 3), len(scored_options) - 1)
                return scored_options[choice_idx][0]
            return scored_options[0][0]

        elif profile.personality_type == PersonalityType.HABITUAL:
            # Strong preference for familiar zones
            for option, _score in scored_options:
                if option.get("zone_id", "") in profile.preferred_zones:
                    return option
            # If no familiar options, reluctantly choose best
            return scored_options[0][0]

        elif profile.personality_type == PersonalityType.SOCIAL:
            # Influenced by what others are doing
            popularity_weighted = []
            for option, score in scored_options:
                popularity = self._get_zone_popularity(option.get("zone_id", ""))
                adjusted_score = score + popularity * 2  # Strong social influence
                popularity_weighted.append((option, adjusted_score))

            popularity_weighted.sort(key=lambda x: x[1], reverse=True)
            return popularity_weighted[0][0]

        # Default to best option
        return scored_options[0][0]

    def _calculate_max_price(
        self, profile: DriverProfile, urgency: UrgencyLevel
    ) -> float:
        """Calculate maximum acceptable price for this driver"""
        base_max = (1 - profile.price_sensitivity) * 20  # Base willingness to pay

        # Income adjustment
        income_adjustment = profile.income_level * 10

        # Urgency multiplier
        urgency_multiplier = self.urgency_multipliers[urgency]

        return (base_max + income_adjustment) * urgency_multiplier

    def _generate_preferred_zones(self) -> List[str]:
        """Generate realistic preferred zones for a driver"""
        # Most people have 2-4 preferred zones
        num_zones = random.randint(2, 4)
        zone_pool = [f"zone_{i}" for i in range(20)]  # Assume 20 zones available
        return random.sample(zone_pool, min(num_zones, len(zone_pool)))

    def _generate_arrival_patterns(self) -> List[int]:
        """Generate typical arrival time patterns"""
        patterns = []

        # Work commute pattern
        if np.random.random() < 0.7:  # 70% have regular work schedule
            patterns.extend([8, 9])  # Morning arrival

        # Lunch pattern
        if np.random.random() < 0.3:
            patterns.append(12)

        # Evening activities
        if np.random.random() < 0.5:
            patterns.extend([18, 19, 20])

        return patterns

    def _get_zone_popularity(self, zone_id: str) -> float:
        """Get current popularity score for a zone (0-1)"""
        if zone_id not in self.zone_popularity_history:
            return 0.5  # Default neutral popularity

        # Calculate recent popularity based on usage
        recent_usage = self.zone_popularity_history[zone_id][-10:]  # Last 10 time steps
        return np.mean(recent_usage) if recent_usage else 0.5

    def update_zone_popularity(self, zone_id: str, usage_ratio: float):
        """Update zone popularity based on usage"""
        if zone_id not in self.zone_popularity_history:
            self.zone_popularity_history[zone_id] = []

        self.zone_popularity_history[zone_id].append(usage_ratio)

        # Keep only recent history
        if len(self.zone_popularity_history[zone_id]) > 50:
            self.zone_popularity_history[zone_id] = self.zone_popularity_history[
                zone_id
            ][-50:]

    def learn_from_experience(
        self, driver_id: str, chosen_option: Dict, satisfaction: float
    ):
        """Update driver's preferences based on experience"""
        if driver_id not in self.driver_profiles:
            return

        profile = self.driver_profiles[driver_id]

        # Update price sensitivity based on satisfaction
        if satisfaction > 0.8:  # Very satisfied
            # Slightly reduce price sensitivity for this type of option
            profile.price_sensitivity *= 1 - profile.learning_rate * 0.1
        elif satisfaction < 0.3:  # Very unsatisfied
            # Increase price sensitivity
            profile.price_sensitivity *= 1 + profile.learning_rate * 0.1

        # Update preferred zones
        zone_id = chosen_option.get("zone_id", "")
        if satisfaction > 0.7 and zone_id not in profile.preferred_zones:
            # Add to preferred zones if very satisfied
            if len(profile.preferred_zones) < 6:  # Limit to 6 preferred zones
                profile.preferred_zones.append(zone_id)
        elif satisfaction < 0.3 and zone_id in profile.preferred_zones:
            # Remove from preferred zones if very unsatisfied
            profile.preferred_zones.remove(zone_id)

        # Clamp price sensitivity to reasonable bounds
        profile.price_sensitivity = np.clip(profile.price_sensitivity, 0.1, 1.0)

    def get_urgency_level(
        self, driver_data: Dict, current_time: datetime
    ) -> UrgencyLevel:
        """Determine urgency level based on driver's situation"""

        # Check if driver has appointment/meeting
        target_time = driver_data.get("target_arrival_time")
        if target_time:
            time_remaining = (
                target_time - current_time
            ).total_seconds() / 60  # minutes

            if time_remaining < 5:
                return UrgencyLevel.CRITICAL
            elif time_remaining < 15:
                return UrgencyLevel.HIGH
            elif time_remaining < 30:
                return UrgencyLevel.MEDIUM
            else:
                return UrgencyLevel.LOW

        # Default based on time of day and driver profile
        hour = current_time.hour
        if 7 <= hour <= 9 or 17 <= hour <= 19:  # Rush hours
            return UrgencyLevel.HIGH
        elif 12 <= hour <= 13:  # Lunch time
            return UrgencyLevel.MEDIUM
        else:
            return UrgencyLevel.LOW
