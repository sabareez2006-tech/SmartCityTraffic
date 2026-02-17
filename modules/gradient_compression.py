"""
Module 4: Gradient Compression Module

Reduces the size of model updates before transmission to the central
server. Implements Top-K sparsification to ensure communication
efficiency in the federated learning framework.

Reference: Gradient Compression and Correlation-Driven Federated
Learning for Wireless Traffic Prediction (IEEE IoT Journal)
"""

import torch
import numpy as np
from typing import Dict, Tuple
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


class GradientCompressionModule:
    """
    Implements gradient compression using Top-K sparsification.

    Only the top-K% of gradient values (by magnitude) are retained
    and transmitted. A residual memory mechanism accumulates the
    dropped gradients for the next round to avoid information loss.
    """

    def __init__(self, compression_ratio: float = None,
                 threshold: float = None):
        """
        Args:
            compression_ratio: Fraction of gradients to keep (0, 1].
            threshold:         Minimum absolute gradient value to keep.
        """
        self.compression_ratio = compression_ratio or config.COMPRESSION_RATIO
        self.threshold = threshold or config.COMPRESSION_THRESHOLD

        # Residual memory per client (zone_id → residual dict)
        self.residuals: Dict[int, Dict[str, torch.Tensor]] = {}

        # Statistics
        self.compression_stats: Dict = {
            "total_elements": 0,
            "transmitted_elements": 0,
            "rounds": 0,
        }

    def compress(self, gradient: Dict[str, torch.Tensor],
                 zone_id: int) -> Tuple[Dict[str, torch.Tensor],
                                         Dict[str, torch.Tensor]]:
        """
        Compress a gradient update using Top-K sparsification with
        residual memory.

        Args:
            gradient: Dictionary of parameter name → gradient tensor.
            zone_id:  ID of the zone/client.

        Returns:
            Tuple of (compressed_gradient, mask) where mask indicates
            which elements were transmitted.
        """
        # Initialize residual memory for new clients
        if zone_id not in self.residuals:
            self.residuals[zone_id] = {
                key: torch.zeros_like(val) for key, val in gradient.items()
            }

        compressed = {}
        masks = {}

        total_elems = 0
        transmitted_elems = 0

        for key, grad in gradient.items():
            # Add residual from previous round
            accumulated = grad + self.residuals[zone_id][key]

            # Flatten for Top-K selection
            flat = accumulated.view(-1)
            num_elements = flat.numel()
            total_elems += num_elements

            # Determine K (number of elements to keep)
            k = max(1, int(num_elements * self.compression_ratio))

            # Get indices of top-K elements by magnitude
            abs_vals = flat.abs()
            topk_vals, topk_indices = torch.topk(abs_vals, k)

            # Apply threshold: only keep elements above threshold
            valid_mask = topk_vals >= self.threshold
            topk_indices = topk_indices[valid_mask]

            # Create sparse compressed gradient
            compressed_flat = torch.zeros_like(flat)
            compressed_flat[topk_indices] = flat[topk_indices]
            compressed[key] = compressed_flat.view(grad.shape)

            # Create mask
            mask_flat = torch.zeros_like(flat, dtype=torch.bool)
            mask_flat[topk_indices] = True
            masks[key] = mask_flat.view(grad.shape)

            # Update residual (accumulate non-transmitted gradients)
            residual_flat = flat.clone()
            residual_flat[topk_indices] = 0.0
            self.residuals[zone_id][key] = residual_flat.view(grad.shape)

            transmitted_elems += topk_indices.numel()

        # Update statistics
        self.compression_stats["total_elements"] += total_elems
        self.compression_stats["transmitted_elements"] += transmitted_elems
        self.compression_stats["rounds"] += 1

        return compressed, masks

    def get_compression_ratio_actual(self) -> float:
        """Get the actual achieved compression ratio."""
        if self.compression_stats["total_elements"] == 0:
            return 0.0
        return (self.compression_stats["transmitted_elements"]
                / self.compression_stats["total_elements"])

    def get_communication_savings(self) -> float:
        """Get percentage of communication saved."""
        return (1.0 - self.get_compression_ratio_actual()) * 100

    def reset_residuals(self):
        """Reset all residual memories."""
        self.residuals.clear()

    def summary(self) -> str:
        """Return compression statistics summary."""
        actual_ratio = self.get_compression_ratio_actual()
        savings = self.get_communication_savings()
        return (
            f"Gradient Compression Summary\n"
            f"{'=' * 50}\n"
            f"Target compression ratio: {self.compression_ratio:.1%}\n"
            f"Actual compression ratio: {actual_ratio:.1%}\n"
            f"Communication savings:    {savings:.1f}%\n"
            f"Total rounds processed:   {self.compression_stats['rounds']}\n"
            f"Total elements:           {self.compression_stats['total_elements']:,}\n"
            f"Transmitted elements:     {self.compression_stats['transmitted_elements']:,}"
        )
