"""
Module 7: Traffic Prediction Module

Uses the updated global model to forecast traffic and ride request
rates for upcoming simulation steps. Predictions are used to drive
dynamic traffic conditions in the robotaxi simulation.
"""

import numpy as np
import torch
from typing import Dict, List, Optional, Tuple
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from models.traffic_model import TrafficLSTM
from utils.helpers import normalize_data, denormalize_data


class TrafficPredictionModule:
    """
    Uses the global federated model to predict next-step traffic
    conditions for each city zone.
    """

    def __init__(self, global_model: TrafficLSTM,
                 device: torch.device = None):
        """
        Args:
            global_model: The trained global TrafficLSTM model.
            device:       Torch device.
        """
        self.model = global_model
        self.device = device or torch.device("cpu")
        self.model.to(self.device)

        # Per-zone normalization parameters
        self.norm_params: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}

        # Prediction history
        self.predictions_history: Dict[int, List[np.ndarray]] = {}

    def set_normalization_params(self, zone_id: int,
                                 min_vals: np.ndarray,
                                 max_vals: np.ndarray):
        """Store normalization parameters for a zone."""
        self.norm_params[zone_id] = (min_vals, max_vals)

    def predict_next_step(self, zone_id: int,
                          recent_data: np.ndarray) -> np.ndarray:
        """
        Predict the next time-step traffic conditions for a zone.

        Args:
            zone_id:     Zone identifier.
            recent_data: Recent traffic data of shape
                         (seq_length, 3) in original scale.

        Returns:
            Predicted traffic values [ride_requests, congestion,
            traffic_flow] in original scale.
        """
        assert zone_id in self.norm_params, \
            f"Normalization params not set for zone {zone_id}"

        min_vals, max_vals = self.norm_params[zone_id]

        # Normalize input
        range_vals = max_vals - min_vals
        range_vals[range_vals == 0] = 1
        norm_input = (recent_data - min_vals) / range_vals

        # Convert to tensor
        input_tensor = torch.FloatTensor(norm_input).unsqueeze(0).to(
            self.device
        )

        # Predict
        self.model.eval()
        with torch.no_grad():
            pred_norm = self.model(input_tensor).cpu().numpy()[0]

        # Denormalize
        prediction = denormalize_data(
            pred_norm.reshape(1, -1), min_vals, max_vals
        )[0]

        # Clip to reasonable ranges
        prediction[0] = max(0, prediction[0])       # ride_requests >= 0
        prediction[1] = np.clip(prediction[1], 0, 1)  # congestion [0, 1]
        prediction[2] = max(0, prediction[2])       # traffic_flow >= 0

        # Store in history
        if zone_id not in self.predictions_history:
            self.predictions_history[zone_id] = []
        self.predictions_history[zone_id].append(prediction.copy())

        return prediction

    def predict_all_zones(
            self, zone_recent_data: Dict[int, np.ndarray]
    ) -> Dict[int, np.ndarray]:
        """
        Predict next step for all zones at once.

        Args:
            zone_recent_data: Dict of zone_id → recent data arrays.

        Returns:
            Dict of zone_id → predicted values.
        """
        predictions = {}
        for zone_id, data in zone_recent_data.items():
            predictions[zone_id] = self.predict_next_step(zone_id, data)
        return predictions

    def get_predicted_demand(self, zone_id: int) -> float:
        """Get the latest predicted ride request rate for a zone."""
        if zone_id in self.predictions_history and \
                self.predictions_history[zone_id]:
            return self.predictions_history[zone_id][-1][0]
        return config.BASE_RIDE_REQUESTS

    def get_predicted_congestion(self, zone_id: int) -> float:
        """Get the latest predicted congestion level for a zone."""
        if zone_id in self.predictions_history and \
                self.predictions_history[zone_id]:
            return self.predictions_history[zone_id][-1][1]
        return config.BASE_CONGESTION

    def summary(self) -> str:
        """Return prediction summary."""
        lines = [
            f"Traffic Prediction Summary",
            f"{'=' * 50}",
            f"Zones with predictions: {len(self.predictions_history)}",
        ]
        for zid in sorted(self.predictions_history.keys()):
            preds = np.array(self.predictions_history[zid])
            lines.append(
                f"  Zone {zid:2d}: {len(preds)} predictions, "
                f"avg demand={preds[:, 0].mean():.1f}, "
                f"avg congestion={preds[:, 1].mean():.3f}"
            )
        return "\n".join(lines)
