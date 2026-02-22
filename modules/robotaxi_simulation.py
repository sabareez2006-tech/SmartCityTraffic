"""
Module 8: Robotaxi Simulation Module

Simulates robotaxi system behavior under dynamic traffic conditions.
Adjusts demand arrival rates and congestion factors based on
traffic predictions from the federated learning model.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


@dataclass
class RideRequest:
    """Represents a single ride request."""
    request_id: int
    origin_zone: int
    destination_zone: int
    request_time: float           # Simulation time in minutes
    pickup_time: Optional[float] = None
    dropoff_time: Optional[float] = None
    wait_time: Optional[float] = None
    travel_time: Optional[float] = None
    status: str = "pending"       # pending, assigned, completed, expired


@dataclass
class Robotaxi:
    """Represents a single robotaxi vehicle."""
    vehicle_id: int
    current_zone: int
    status: str = "idle"          # idle, en_route, serving, relocating
    current_ride: Optional[int] = None
    available_at: float = 0.0     # Time when vehicle becomes available
    total_rides: int = 0
    total_idle_time: float = 0.0


class RobotaxiSimulationModule:
    """
    Simulates a fleet of robotaxis operating under dynamic traffic
    conditions. Traffic predictions drive proactive repositioning.
    """

    def __init__(self, num_zones: int, zone_distances: np.ndarray,
                 seed: int = None):
        """
        Args:
            num_zones:       Number of city zones.
            zone_distances:  Distance matrix between zones (km).
            seed:            Random seed.
        """
        self.num_zones = num_zones
        self.zone_distances = zone_distances
        self.rng = np.random.RandomState(seed or config.RANDOM_SEED)

        # Fleet initialization
        self.fleet: List[Robotaxi] = []
        self._initialize_fleet()

        # Simulation state
        self.current_time = 0.0
        self.ride_requests: List[RideRequest] = []
        self.completed_rides: List[RideRequest] = []
        self.expired_rides: List[RideRequest] = []
        self.request_counter = 0

        # Per-step metrics
        self.step_metrics: List[Dict] = []

    def _initialize_fleet(self):
        """Distribute robotaxis across zones."""
        self.fleet = []
        vehicles_per_zone = config.NUM_ROBOTAXIS // self.num_zones
        extra = config.NUM_ROBOTAXIS % self.num_zones

        vid = 0
        for zone_id in range(self.num_zones):
            count = vehicles_per_zone + (1 if zone_id < extra else 0)
            for _ in range(count):
                self.fleet.append(Robotaxi(
                    vehicle_id=vid, current_zone=zone_id
                ))
                vid += 1

    def simulate_step(self, step_time: float,
                      actual_demand: Dict[int, float],
                      pred_demand: Dict[int, float],
                      actual_congestion: Dict[int, float],
                      use_dynamic: bool = True) -> Dict:
        """
        Simulate one time step of robotaxi operations.

        Args:
            step_time:        Current simulation time (minutes).
            actual_demand:    Actual ride demand per zone.
            pred_demand:      Predicted ride demand per zone.
            actual_congestion: Actual congestion levels per zone.
            use_dynamic:      If True, use proactive dynamic repositioning.
        """
        self.current_time = step_time

        # ── 1. Update vehicle states ──
        self._update_vehicles()

        # ── 2. Generate ride requests based on ACTUAL demand ──
        new_requests = self._generate_requests(actual_demand)

        # ── 3. Dispatch available vehicles to pending requests ──
        dispatched = self._dispatch_vehicles(actual_congestion)

        # ── 4. Proactive Repositioning (Dynamic Only) ──
        if use_dynamic:
            self._reposition_idle_vehicles(pred_demand, actual_congestion)

        # ── 5. Expire old requests ──
        expired = self._expire_requests()

        # ── 6. Compute metrics ──
        idle_vehicles = sum(1 for v in self.fleet if v.status == "idle")
        active_vehicles = len(self.fleet) - idle_vehicles
        utilization = active_vehicles / len(self.fleet) if self.fleet else 0

        pending = sum(1 for r in self.ride_requests if r.status == "pending")

        avg_wait = 0.0
        # Compute wait block
        recent_completed = [r for r in self.completed_rides 
                            if r.dropoff_time is not None 
                            and self.current_time - config.TIME_STEP_MINUTES <= r.dropoff_time <= self.current_time]
        if recent_completed:
            wait_times = [r.wait_time for r in recent_completed if r.wait_time is not None]
            avg_wait = np.mean(wait_times) if wait_times else 0.0

        metrics = {
            "time": step_time,
            "new_requests": len(new_requests),
            "dispatched": dispatched,
            "completed": len(recent_completed),
            "expired": expired,
            "pending": pending,
            "idle_vehicles": idle_vehicles,
            "fleet_utilization": utilization,
            "avg_wait_time": float(avg_wait),
            "avg_congestion": float(np.mean(list(actual_congestion.values()) if actual_congestion else 0)),
        }
        self.step_metrics.append(metrics)
        return metrics

    def _generate_requests(self, actual_demand: Dict[int, float]) -> List[RideRequest]:
        """Generate ride requests based on actual demand rates to ensure identical worlds."""
        new_requests = []
        robotaxi_market_share = 0.05  # Scale demand to realistic robotaxi fleet size
        for zone_id in range(self.num_zones):
            rate = actual_demand.get(zone_id, config.BASE_RIDE_REQUESTS)

            # Poisson arrivals
            num_requests = self.rng.poisson(max(0, rate * robotaxi_market_share))

            for _ in range(num_requests):
                dest = self._choose_destination(zone_id)
                request = RideRequest(
                    request_id=self.request_counter,
                    origin_zone=zone_id,
                    destination_zone=dest,
                    request_time=self.current_time + self.rng.uniform(0, config.TIME_STEP_MINUTES)
                )
                self.ride_requests.append(request)
                new_requests.append(request)
                self.request_counter += 1

        return new_requests

    def _choose_destination(self, origin: int) -> int:
        """Choose a destination zone with distance-based probability."""
        distances = self.zone_distances[origin].copy()
        distances[origin] = np.inf  # No self-trips
        weights = 1.0 / (distances + 0.1)
        weights /= weights.sum()
        return self.rng.choice(self.num_zones, p=weights)

    def _dispatch_vehicles(self, actual_congestion: Dict[int, float]) -> int:
        """Dispatch idle vehicles to pending requests."""
        dispatched = 0
        pending = [r for r in self.ride_requests if r.status == "pending"]
        idle = [v for v in self.fleet if v.status == "idle" and v.available_at <= self.current_time]

        for request in sorted(pending, key=lambda r: r.request_time):
            if not idle:
                break

            # Find nearest idle vehicle
            best_vehicle = min(idle, key=lambda v: self.zone_distances[v.current_zone, request.origin_zone])
            best_dist = self.zone_distances[best_vehicle.current_zone, request.origin_zone]

            # Physical travel limits
            cong = actual_congestion.get(request.origin_zone, 0.3)
            speed = config.ROBOTAXI_SPEED_KMH * (1 - 0.5 * cong)
            travel_minutes = (best_dist / max(speed, 5)) * 60

            # Fix Negative Wait Time: vehicle cannot start moving before the request is made or it is available
            start_time = max(request.request_time, self.current_time, best_vehicle.available_at)
            pickup_time = start_time + travel_minutes + config.PICKUP_TIME_MINUTES
            
            wait_time = pickup_time - request.request_time
            if wait_time > config.MAX_WAIT_TIME_MINUTES:
                # All remaining idle vehicles are equal distance or further, so no one can reach in time
                # Leave request pending. It will eventually expire.
                pass # Wait, we must check if OTHER vehicles are closer? 
                # We used `min(idle, key=distance)`. So NO other idle vehicle is closer!
                # We just leave this request pending for now and skip to the next request!
                continue

            request.status = "assigned"
            request.pickup_time = pickup_time
            request.wait_time = wait_time

            ride_dist = self.zone_distances[request.origin_zone, request.destination_zone]
            dest_cong = actual_congestion.get(request.destination_zone, 0.3)
            avg_cong = (cong + dest_cong) / 2
            ride_speed = config.ROBOTAXI_SPEED_KMH * (1 - 0.5 * avg_cong)
            ride_time = (ride_dist / max(ride_speed, 5)) * 60

            request.travel_time = ride_time
            request.dropoff_time = pickup_time + ride_time + config.DROPOFF_TIME_MINUTES

            best_vehicle.status = "serving"
            best_vehicle.current_ride = request.request_id
            best_vehicle.available_at = request.dropoff_time

            idle.remove(best_vehicle)
            dispatched += 1

        return dispatched

    def _reposition_idle_vehicles(self, pred_demand: Dict[int, float], actual_congestion: Dict[int, float]):
        """Proactively move idle vehicles to high demand zones."""
        idle = [v for v in self.fleet if v.status == "idle" and v.available_at <= self.current_time]
        if not idle:
            return

        # Rank zones by predicted demand, select top 5
        heavy_zones = sorted(pred_demand.keys(), key=lambda z: pred_demand.get(z, 0), reverse=True)[:5]

        for vehicle in idle:
            if vehicle.current_zone not in heavy_zones:
                target = self.rng.choice(heavy_zones)
                dist = self.zone_distances[vehicle.current_zone, target]
                
                if dist > 1.0: # Minimum distance to relocate
                    cong = actual_congestion.get(vehicle.current_zone, 0.3)
                    speed = config.ROBOTAXI_SPEED_KMH * (1 - 0.5 * cong)
                    travel_minutes = (dist / max(speed, 5)) * 60
                    
                    vehicle.status = "relocating"
                    vehicle.available_at = self.current_time + travel_minutes
                    vehicle.current_zone = target

    def _update_vehicles(self):
        """Update vehicle states and complete rides."""
        for vehicle in self.fleet:
            if vehicle.status == "serving" and vehicle.available_at <= self.current_time:
                ride = next((r for r in self.ride_requests if r.request_id == vehicle.current_ride), None)
                if ride:
                    ride.status = "completed"
                    self.completed_rides.append(ride)
                    vehicle.current_zone = ride.destination_zone

                vehicle.status = "idle"
                vehicle.current_ride = None
                vehicle.total_rides += 1

            elif vehicle.status == "relocating" and vehicle.available_at <= self.current_time:
                vehicle.status = "idle"

    def _expire_requests(self) -> int:
        """Expire requests that waited too long."""
        expired_count = 0
        for request in self.ride_requests:
            if request.status == "pending":
                wait = self.current_time - request.request_time
                if wait > config.MAX_WAIT_TIME_MINUTES:
                    request.status = "expired"
                    self.expired_rides.append(request)
                    expired_count += 1
        return expired_count

    def get_overall_metrics(self) -> Dict:
        """Compute overall simulation metrics."""
        total_completed = len(self.completed_rides)
        total_expired = len(self.expired_rides)
        total_requests = self.request_counter

        avg_wait = 0.0
        avg_travel = 0.0
        if self.completed_rides:
            waits = [r.wait_time for r in self.completed_rides if r.wait_time is not None]
            travels = [r.travel_time for r in self.completed_rides if r.travel_time is not None]
            avg_wait = np.mean(waits) if waits else 0.0
            avg_travel = np.mean(travels) if travels else 0.0

        service_rate = total_completed / max(total_requests, 1)

        avg_utilization = 0.0
        if self.step_metrics:
            avg_utilization = np.mean([m["fleet_utilization"] for m in self.step_metrics])

        avg_rides_per_vehicle = total_completed / len(self.fleet) if self.fleet else 0

        return {
            "total_requests": total_requests,
            "total_completed": total_completed,
            "total_expired": total_expired,
            "service_rate": service_rate,
            "avg_wait_time": float(avg_wait),
            "avg_travel_time": float(avg_travel),
            "avg_fleet_utilization": float(avg_utilization),
            "avg_rides_per_vehicle": float(avg_rides_per_vehicle),
        }

    def reset(self):
        """Reset the simulation state."""
        self.current_time = 0.0
        self.ride_requests = []
        self.completed_rides = []
        self.expired_rides = []
        self.request_counter = 0
        self.step_metrics = []
        self._initialize_fleet()

    def summary(self) -> str:
        """Return simulation summary."""
        m = self.get_overall_metrics()
        return (
            f"Robotaxi Simulation Summary\n"
            f"{'=' * 50}\n"
            f"Fleet size:            {len(self.fleet)}\n"
            f"Total requests:        {m['total_requests']:,}\n"
            f"Completed rides:       {m['total_completed']:,}\n"
            f"Expired requests:      {m['total_expired']:,}\n"
            f"Service rate:          {m['service_rate']:.1%}\n"
            f"Avg wait time:         {m['avg_wait_time']:.1f} min\n"
            f"Avg travel time:       {m['avg_travel_time']:.1f} min\n"
            f"Avg fleet utilization: {m['avg_fleet_utilization']:.1%}\n"
            f"Avg rides/vehicle:     {m['avg_rides_per_vehicle']:.1f}"
        )
