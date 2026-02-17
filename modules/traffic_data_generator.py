"""
Module 2: Traffic Data Generator Module

Generates synthetic time-series traffic data including ride demand
and congestion metrics. Introduces time-based variability to simulate
peak and off-peak conditions across city zones.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from modules.city_zoning import CityZoningModule, Zone


class TrafficDataGenerator:
    """
    Generates synthetic traffic time-series data for each city zone.

    Features generated per zone per time step:
        1. ride_requests  – Number of ride requests
        2. congestion     – Congestion level [0, 1]
        3. traffic_flow   – Vehicle flow count
    """

    def __init__(self, city: CityZoningModule, num_days: int = None,
                 seed: int = None):
        """
        Args:
            city:     CityZoningModule instance providing zone information.
            num_days: Number of days of data to generate.
            seed:     Random seed.
        """
        self.city = city
        self.num_days = num_days or config.NUM_DAYS
        self.steps_per_day = config.NUM_TIME_STEPS
        self.total_steps = self.steps_per_day * self.num_days
        self.time_step_minutes = config.TIME_STEP_MINUTES
        self.rng = np.random.RandomState(seed or config.RANDOM_SEED)

        # Store generated data: zone_id → (total_steps, 3) array
        self.zone_data: Dict[int, np.ndarray] = {}

    def generate(self) -> Dict[int, np.ndarray]:
        """
        Generate traffic data for all zones.

        Returns:
            Dictionary mapping zone_id to numpy array of shape
            (total_time_steps, 3) with columns:
            [ride_requests, congestion, traffic_flow]
        """
        print(f"[Traffic Data Generator] Generating {self.num_days} days "
              f"of data ({self.total_steps} time steps) for "
              f"{self.city.num_zones} zones...")

        for zone in self.city.zones:
            self.zone_data[zone.zone_id] = self._generate_zone_data(zone)

        # Add inter-zone correlation (adjacent zones influence each other)
        self._add_spatial_correlation()

        print(f"[Traffic Data Generator] Data generation complete.")
        return self.zone_data

    def _generate_zone_data(self, zone: Zone) -> np.ndarray:
        """Generate traffic data for a single zone."""
        data = np.zeros((self.total_steps, 3))

        for step in range(self.total_steps):
            hour_of_day = (step % self.steps_per_day) * \
                          self.time_step_minutes / 60.0
            day_of_week = step // self.steps_per_day

            # Compute time-based multiplier
            time_mult = self._get_time_multiplier(
                hour_of_day, day_of_week, zone.peak_multiplier
            )

            # ── Ride Requests ──
            base_demand = config.BASE_RIDE_REQUESTS * zone.demand_factor
            ride_requests = base_demand * time_mult
            ride_requests += self.rng.normal(0, base_demand * 0.15)
            ride_requests = max(0, ride_requests)

            # ── Congestion Level ──
            base_cong = config.BASE_CONGESTION * zone.population_density
            congestion = base_cong * time_mult
            congestion += self.rng.normal(0, 0.05)
            congestion = np.clip(congestion, 0.0, 1.0)

            # ── Traffic Flow ──
            base_flow = config.BASE_TRAFFIC_FLOW * zone.population_density
            traffic_flow = base_flow * time_mult
            traffic_flow += self.rng.normal(0, base_flow * 0.1)
            traffic_flow = max(0, traffic_flow)

            data[step] = [ride_requests, congestion, traffic_flow]

        return data

    def _get_time_multiplier(self, hour: float, day: int,
                             peak_mult: float) -> float:
        """
        Compute a time-based demand multiplier with realistic patterns.

        Uses sinusoidal base patterns combined with peak-hour Gaussian bumps.
        Weekend days (5, 6) have reduced peak effects.
        """
        # Base daily pattern (sinusoidal – low at night, higher during day)
        base = 0.3 + 0.7 * max(0, np.sin(np.pi * (hour - 5) / 14))

        # Morning peak Gaussian
        morning_center = (config.MORNING_PEAK[0] + config.MORNING_PEAK[1]) / 2
        morning_bump = peak_mult * np.exp(
            -0.5 * ((hour - morning_center) / 0.8) ** 2
        )

        # Evening peak Gaussian
        evening_center = (config.EVENING_PEAK[0] + config.EVENING_PEAK[1]) / 2
        evening_bump = peak_mult * np.exp(
            -0.5 * ((hour - evening_center) / 1.0) ** 2
        )

        # Lunch peak (smaller)
        lunch_center = (config.LUNCH_PEAK[0] + config.LUNCH_PEAK[1]) / 2
        lunch_bump = peak_mult * 0.4 * np.exp(
            -0.5 * ((hour - lunch_center) / 0.5) ** 2
        )

        # Weekend reduction
        weekend_factor = 0.65 if day >= 5 else 1.0

        multiplier = (base + morning_bump + evening_bump + lunch_bump) \
                     * weekend_factor
        return max(0.1, multiplier)

    def _add_spatial_correlation(self):
        """
        Add spatial correlation: adjacent zones influence each other's
        traffic patterns with a small bleed-over effect.
        """
        alpha = 0.1  # Correlation strength

        original_data = {zid: data.copy()
                         for zid, data in self.zone_data.items()}

        for zone in self.city.zones:
            if zone.adjacent_zones:
                neighbor_avg = np.mean(
                    [original_data[nid] for nid in zone.adjacent_zones],
                    axis=0
                )
                self.zone_data[zone.zone_id] = (
                    (1 - alpha) * original_data[zone.zone_id]
                    + alpha * neighbor_avg
                )

    def get_zone_data(self, zone_id: int) -> np.ndarray:
        """Get generated data for a specific zone."""
        return self.zone_data[zone_id]

    def get_train_test_split(self, zone_id: int,
                             train_ratio: float = 0.8) -> Tuple:
        """
        Split zone data into training and testing sets.

        Returns:
            (train_data, test_data) each of shape (N, 3)
        """
        data = self.zone_data[zone_id]
        split_idx = int(len(data) * train_ratio)
        return data[:split_idx], data[split_idx:]

    def summary(self) -> str:
        """Return a summary of generated data."""
        lines = [
            f"Traffic Data Summary",
            f"{'=' * 50}",
            f"Total zones:       {len(self.zone_data)}",
            f"Days generated:    {self.num_days}",
            f"Steps per day:     {self.steps_per_day}",
            f"Total time steps:  {self.total_steps}",
            f"Time step:         {self.time_step_minutes} minutes",
            f"Features:          ride_requests, congestion, traffic_flow",
            "",
            "Per-Zone Statistics (mean ± std):",
        ]
        for zid in sorted(self.zone_data.keys()):
            data = self.zone_data[zid]
            means = data.mean(axis=0)
            stds = data.std(axis=0)
            lines.append(
                f"  Zone {zid:2d}: requests={means[0]:6.1f}±{stds[0]:5.1f}, "
                f"congestion={means[1]:.3f}±{stds[1]:.3f}, "
                f"flow={means[2]:6.1f}±{stds[2]:5.1f}"
            )
        return "\n".join(lines)
