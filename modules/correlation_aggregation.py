"""
Module 5: Correlation-Based Aggregation Module

Analyzes similarity among client updates and selects non-redundant
gradients for aggregation. Filters out highly correlated (redundant)
updates to improve communication efficiency and model quality.

Reference: Gradient Compression and Correlation-Driven Federated
Learning for Wireless Traffic Prediction (IEEE IoT Journal)
"""

import torch
import numpy as np
from typing import Dict, List, Tuple
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


class CorrelationAggregationModule:
    """
    Correlation-driven update selection for federated learning.

    Computes pairwise cosine similarity between client gradient updates
    and selects a diverse, non-redundant subset for aggregation.
    """

    def __init__(self, correlation_threshold: float = None,
                 min_selected: int = None):
        """
        Args:
            correlation_threshold: Maximum cosine similarity allowed
                                   between two selected updates.
            min_selected:          Minimum number of clients to select.
        """
        self.correlation_threshold = (
            correlation_threshold or config.CORRELATION_THRESHOLD
        )
        self.min_selected = min_selected or config.MIN_SELECTED_CLIENTS

        # Selection history
        self.selection_history: List[Dict] = []

    def _flatten_gradient(self, gradient: Dict[str, torch.Tensor]) \
            -> torch.Tensor:
        """Flatten a gradient dictionary into a single vector."""
        return torch.cat([g.view(-1) for g in gradient.values()])

    def compute_similarity_matrix(
            self, gradients: List[Dict[str, torch.Tensor]]
    ) -> np.ndarray:
        """
        Compute pairwise cosine similarity between client gradients.

        Args:
            gradients: List of gradient dictionaries from clients.

        Returns:
            Similarity matrix of shape (num_clients, num_clients)
        """
        flat_grads = [self._flatten_gradient(g) for g in gradients]
        n = len(flat_grads)
        sim_matrix = np.zeros((n, n))

        for i in range(n):
            for j in range(i, n):
                # Cosine similarity
                cos_sim = torch.nn.functional.cosine_similarity(
                    flat_grads[i].unsqueeze(0),
                    flat_grads[j].unsqueeze(0)
                ).item()
                sim_matrix[i, j] = cos_sim
                sim_matrix[j, i] = cos_sim

        return sim_matrix

    def select_updates(
            self, gradients: List[Dict[str, torch.Tensor]],
            zone_ids: List[int],
            sample_counts: List[int]
    ) -> Tuple[List[int], List[Dict[str, torch.Tensor]], List[int]]:
        """
        Select non-redundant gradient updates based on correlation.

        Algorithm:
        1. Compute pairwise cosine similarity matrix.
        2. Rank clients by gradient magnitude (informativeness).
        3. Greedily select clients whose updates are sufficiently
           different from already-selected ones.

        Args:
            gradients:     List of gradient dicts from all clients.
            zone_ids:       List of zone IDs corresponding to gradients.
            sample_counts: Number of training samples per client.

        Returns:
            Tuple of (selected_zone_ids, selected_gradients,
                      selected_sample_counts)
        """
        n = len(gradients)

        if n <= self.min_selected:
            # If fewer clients than minimum, select all
            record = {
                "total_clients": n,
                "selected_clients": n,
                "selected_zone_ids": zone_ids,
                "reason": "fewer_than_minimum",
            }
            self.selection_history.append(record)
            return zone_ids, gradients, sample_counts

        # Step 1: Compute similarity matrix
        sim_matrix = self.compute_similarity_matrix(gradients)

        # Step 2: Rank clients by gradient magnitude (informativeness)
        flat_grads = [self._flatten_gradient(g) for g in gradients]
        magnitudes = [fg.norm().item() for fg in flat_grads]
        ranked_indices = np.argsort(magnitudes)[::-1]  # Descending

        # Step 3: Greedy selection
        selected_indices = [ranked_indices[0]]  # Start with most informative

        for idx in ranked_indices[1:]:
            # Check correlation with all already-selected clients
            is_redundant = False
            for sel_idx in selected_indices:
                if sim_matrix[idx, sel_idx] > self.correlation_threshold:
                    is_redundant = True
                    break

            if not is_redundant:
                selected_indices.append(idx)

        # Ensure minimum selection
        if len(selected_indices) < self.min_selected:
            for idx in ranked_indices:
                if idx not in selected_indices:
                    selected_indices.append(idx)
                if len(selected_indices) >= self.min_selected:
                    break

        selected_indices = sorted(selected_indices)

        # Gather selected data
        sel_zone_ids = [zone_ids[i] for i in selected_indices]
        sel_gradients = [gradients[i] for i in selected_indices]
        sel_samples = [sample_counts[i] for i in selected_indices]

        # Record selection details
        record = {
            "total_clients": n,
            "selected_clients": len(selected_indices),
            "selected_zone_ids": sel_zone_ids,
            "redundant_removed": n - len(selected_indices),
            "avg_similarity": float(np.mean(
                sim_matrix[np.triu_indices(n, k=1)]
            )),
        }
        self.selection_history.append(record)

        return sel_zone_ids, sel_gradients, sel_samples

    def get_selection_rate(self) -> float:
        """Average fraction of clients selected across all rounds."""
        if not self.selection_history:
            return 1.0
        rates = [h["selected_clients"] / h["total_clients"]
                 for h in self.selection_history]
        return np.mean(rates)

    def summary(self) -> str:
        """Return summary of correlation-based selection."""
        lines = [
            f"Correlation-Based Aggregation Summary",
            f"{'=' * 50}",
            f"Correlation threshold: {self.correlation_threshold:.2f}",
            f"Minimum clients:       {self.min_selected}",
            f"Total rounds:          {len(self.selection_history)}",
            f"Avg selection rate:    {self.get_selection_rate():.1%}",
        ]
        if self.selection_history:
            total_removed = sum(
                h.get("redundant_removed", 0)
                for h in self.selection_history
            )
            lines.append(f"Total redundant removed: {total_removed}")
        return "\n".join(lines)
