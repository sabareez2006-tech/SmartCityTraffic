"""
Module 6: Federated Aggregation Module

Implements the Federated Averaging (FedAvg) algorithm to combine
selected client updates and produce a global traffic prediction model.

Reference: McMahan et al., "Communication-Efficient Learning of
Deep Networks from Decentralized Data" (AISTATS 2017)
"""

import copy
import torch
from typing import Dict, List, Optional
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from models.traffic_model import TrafficLSTM


class FederatedAggregationModule:
    """
    Implements Federated Averaging (FedAvg) for aggregating
    client model updates into a global model.
    """

    def __init__(self, device: torch.device = None):
        """
        Args:
            device: Torch device for the global model.
        """
        self.device = device or torch.device("cpu")

        # Initialize global model
        self.global_model = TrafficLSTM(
            input_size=config.INPUT_FEATURES,
            hidden_size=config.HIDDEN_SIZE,
            num_layers=config.NUM_LAYERS,
            output_size=config.OUTPUT_FEATURES,
        ).to(self.device)

        # Aggregation history
        self.round_history: List[Dict] = []
        self.current_round = 0

    def get_global_params(self) -> Dict[str, torch.Tensor]:
        """Get a copy of the global model parameters."""
        return copy.deepcopy(self.global_model.state_dict())

    def aggregate_gradients(
            self, gradients: List[Dict[str, torch.Tensor]],
            sample_counts: List[int],
            apply_to_global: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Aggregate gradient updates using weighted FedAvg.

        Each client's gradient is weighted proportional to the number
        of training samples it used.

        Args:
            gradients:      List of gradient dictionaries from selected
                            clients.
            sample_counts:  Number of samples per client.
            apply_to_global: Whether to apply the aggregated gradient
                             to the global model immediately.

        Returns:
            Aggregated gradient dictionary.
        """
        total_samples = sum(sample_counts)

        # Compute weighted average of gradients
        aggregated = {}
        for key in gradients[0]:
            weighted_sum = torch.zeros_like(gradients[0][key],
                                            dtype=torch.float32)
            for grad, n_samples in zip(gradients, sample_counts):
                weight = n_samples / total_samples
                weighted_sum += weight * grad[key].float()
            aggregated[key] = weighted_sum

        if apply_to_global:
            self._apply_gradient(aggregated)

        # Record round
        self.current_round += 1
        self.round_history.append({
            "round": self.current_round,
            "num_clients": len(gradients),
            "total_samples": total_samples,
            "avg_gradient_norm": self._compute_gradient_norm(aggregated),
        })

        return aggregated

    def aggregate_models(
            self, client_params: List[Dict[str, torch.Tensor]],
            sample_counts: List[int]
    ) -> Dict[str, torch.Tensor]:
        """
        Alternative: Aggregate full model parameters (not gradients).

        Uses weighted average of model parameters directly.

        Args:
            client_params:  List of client model state_dicts.
            sample_counts:  Number of samples per client.

        Returns:
            Aggregated model state dict.
        """
        total_samples = sum(sample_counts)
        aggregated = {}

        for key in client_params[0]:
            weighted_sum = torch.zeros_like(
                client_params[0][key], dtype=torch.float32
            )
            for params, n_samples in zip(client_params, sample_counts):
                weight = n_samples / total_samples
                weighted_sum += weight * params[key].float()
            aggregated[key] = weighted_sum

        self.global_model.load_state_dict(aggregated)

        self.current_round += 1
        self.round_history.append({
            "round": self.current_round,
            "num_clients": len(client_params),
            "total_samples": total_samples,
            "method": "model_averaging",
        })

        return aggregated

    def _apply_gradient(self, aggregated_gradient: Dict[str, torch.Tensor]):
        """Apply aggregated gradient to the global model parameters."""
        global_params = self.global_model.state_dict()
        for key in global_params:
            if key in aggregated_gradient:
                global_params[key] = (
                    global_params[key].float()
                    + aggregated_gradient[key].float()
                )
        self.global_model.load_state_dict(global_params)

    def _compute_gradient_norm(
            self, gradient: Dict[str, torch.Tensor]
    ) -> float:
        """Compute the L2 norm of a gradient dictionary."""
        total = 0.0
        for val in gradient.values():
            total += val.float().norm().item() ** 2
        return total ** 0.5

    def summary(self) -> str:
        """Return aggregation summary."""
        lines = [
            f"Federated Aggregation Summary",
            f"{'=' * 50}",
            f"Total rounds:        {self.current_round}",
            f"Model parameters:    {self.global_model.count_parameters():,}",
        ]
        if self.round_history:
            avg_clients = sum(
                h["num_clients"] for h in self.round_history
            ) / len(self.round_history)
            lines.append(f"Avg clients/round:   {avg_clients:.1f}")
            norms = [h.get("avg_gradient_norm", 0)
                     for h in self.round_history]
            if norms:
                lines.append(f"Avg gradient norm:   {sum(norms)/len(norms):.6f}")
        return "\n".join(lines)
