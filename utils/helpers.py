"""
Utility helper functions for the project.
"""

import os
import random
import numpy as np
import torch
from datetime import datetime


def set_seed(seed: int = 42):
    """Set random seed for reproducibility across all libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def ensure_dir(directory: str):
    """Create directory if it doesn't exist."""
    os.makedirs(directory, exist_ok=True)


def get_timestamp() -> str:
    """Get current timestamp as formatted string."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def normalize_data(data: np.ndarray) -> tuple:
    """
    Min-max normalize data to [0, 1] range.

    Returns:
        Tuple of (normalized_data, min_vals, max_vals)
    """
    min_vals = data.min(axis=0)
    max_vals = data.max(axis=0)
    range_vals = max_vals - min_vals
    range_vals[range_vals == 0] = 1  # Avoid division by zero
    normalized = (data - min_vals) / range_vals
    return normalized, min_vals, max_vals


def denormalize_data(normalized: np.ndarray, min_vals: np.ndarray,
                     max_vals: np.ndarray) -> np.ndarray:
    """Reverse min-max normalization."""
    range_vals = max_vals - min_vals
    range_vals[range_vals == 0] = 1
    return normalized * range_vals + min_vals


def create_sequences(data: np.ndarray, seq_length: int,
                     pred_horizon: int = 1) -> tuple:
    """
    Create input-output sequence pairs for time-series prediction.

    Args:
        data: Array of shape (num_timesteps, num_features)
        seq_length: Number of past time steps to use as input
        pred_horizon: Number of future steps to predict

    Returns:
        Tuple of (X, y) where X has shape (N, seq_length, features)
        and y has shape (N, features)
    """
    X, y = [], []
    for i in range(len(data) - seq_length - pred_horizon + 1):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length + pred_horizon - 1])
    return np.array(X), np.array(y)


def format_metrics(metrics: dict) -> str:
    """Format a metrics dictionary into a readable string."""
    lines = []
    for key, value in metrics.items():
        if isinstance(value, float):
            lines.append(f"  {key}: {value:.4f}")
        else:
            lines.append(f"  {key}: {value}")
    return "\n".join(lines)
