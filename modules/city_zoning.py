"""
Module 1: City Zoning Module

Divides the simulated city into multiple regions and assigns each
region as a federated client. Maintains zone-level traffic statistics.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Tuple
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


@dataclass
class Zone:
    """Represents a single city zone (federated client)."""
    zone_id: int
    zone_type: str
    grid_position: Tuple[int, int]
    center_coords: Tuple[float, float]      # (x_km, y_km)
    area_km2: float
    population_density: float                # Normalized [0, 1]
    demand_factor: float                     # Demand multiplier
    peak_multiplier: float                   # Peak hour demand multiplier
    adjacent_zones: List[int] = field(default_factory=list)
    traffic_stats: Dict = field(default_factory=dict)

    def __repr__(self):
        return (f"Zone(id={self.zone_id}, type={self.zone_type}, "
                f"pos={self.grid_position}, demand={self.demand_factor:.2f})")


class CityZoningModule:
    """
    Divides the simulated city into a grid of zones.
    Each zone acts as an independent federated learning client.
    """

    def __init__(self, grid_size: Tuple[int, int] = None,
                 city_area_km: float = None, seed: int = None):
        """
        Args:
            grid_size:    Tuple (rows, cols) for the city grid.
            city_area_km: Total city dimension in km.
            seed:         Random seed for zone type assignment.
        """
        self.grid_size = grid_size or config.CITY_GRID_SIZE
        self.city_area_km = city_area_km or config.CITY_AREA_KM
        self.seed = seed or config.RANDOM_SEED
        self.num_zones = self.grid_size[0] * self.grid_size[1]
        self.zone_size_km = self.city_area_km / max(self.grid_size)
        self.zones: List[Zone] = []

        self._build_zones()

    def _build_zones(self):
        """Create all zones with type assignments and adjacency."""
        rng = np.random.RandomState(self.seed)
        zone_types = list(config.ZONE_TYPES.keys())

        # Assign zone types with spatial clustering for realism
        # Downtown zones near the center, suburban at edges
        rows, cols = self.grid_size
        center_r, center_c = rows // 2, cols // 2

        for r in range(rows):
            for c in range(cols):
                zone_id = r * cols + c

                # Distance from center determines zone type bias
                dist = np.sqrt((r - center_r) ** 2 + (c - center_c) ** 2)
                max_dist = np.sqrt(center_r ** 2 + center_c ** 2)
                norm_dist = dist / max_dist if max_dist > 0 else 0

                # Zone type assignment based on distance from center
                if norm_dist < 0.2:
                    zone_type = "downtown"
                elif norm_dist < 0.45:
                    zone_type = rng.choice(["commercial", "residential"],
                                           p=[0.6, 0.4])
                elif norm_dist < 0.7:
                    zone_type = rng.choice(["residential", "commercial",
                                            "industrial"], p=[0.5, 0.2, 0.3])
                else:
                    zone_type = rng.choice(["suburban", "residential",
                                            "industrial"], p=[0.5, 0.3, 0.2])

                zt = config.ZONE_TYPES[zone_type]

                # Compute center coordinates in km
                cx = (c + 0.5) * self.zone_size_km
                cy = (r + 0.5) * self.zone_size_km

                zone = Zone(
                    zone_id=zone_id,
                    zone_type=zone_type,
                    grid_position=(r, c),
                    center_coords=(cx, cy),
                    area_km2=self.zone_size_km ** 2,
                    population_density=zt["density"] + rng.normal(0, 0.05),
                    demand_factor=zt["demand_factor"] + rng.normal(0, 0.1),
                    peak_multiplier=zt["peak_multiplier"] + rng.normal(0, 0.1),
                )
                self.zones.append(zone)

        # Compute adjacency (4-connected grid)
        for r in range(rows):
            for c in range(cols):
                zone_id = r * cols + c
                adj = []
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < rows and 0 <= nc < cols:
                        adj.append(nr * cols + nc)
                self.zones[zone_id].adjacent_zones = adj

    def get_zone(self, zone_id: int) -> Zone:
        """Retrieve a zone by its ID."""
        return self.zones[zone_id]

    def get_zones_by_type(self, zone_type: str) -> List[Zone]:
        """Get all zones of a given type."""
        return [z for z in self.zones if z.zone_type == zone_type]

    def get_zone_distances(self) -> np.ndarray:
        """
        Compute distance matrix between all zone centers (in km).

        Returns:
            Array of shape (num_zones, num_zones)
        """
        coords = np.array([z.center_coords for z in self.zones])
        diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
        distances = np.sqrt((diff ** 2).sum(axis=2))
        return distances

    def get_demand_factors(self) -> np.ndarray:
        """Return demand factor for every zone as an array."""
        return np.array([z.demand_factor for z in self.zones])

    def summary(self) -> str:
        """Return a summary string of the city zoning."""
        lines = [
            f"City Zoning Summary",
            f"{'=' * 50}",
            f"Grid Size:       {self.grid_size[0]} × {self.grid_size[1]}",
            f"Total Zones:     {self.num_zones}",
            f"City Area:       {self.city_area_km:.1f} km × {self.city_area_km:.1f} km",
            f"Zone Size:       {self.zone_size_km:.2f} km × {self.zone_size_km:.2f} km",
            "",
            "Zone Type Distribution:",
        ]
        type_counts = {}
        for z in self.zones:
            type_counts[z.zone_type] = type_counts.get(z.zone_type, 0) + 1
        for zt, count in sorted(type_counts.items()):
            lines.append(f"  {zt:15s}: {count:3d} zones")
        return "\n".join(lines)
