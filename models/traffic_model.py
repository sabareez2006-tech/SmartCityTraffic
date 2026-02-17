"""
Traffic Prediction LSTM Model.

A multi-layer LSTM neural network for time-series traffic prediction.
Each federated client (city zone) trains a copy of this model on its
local traffic data, and model updates are aggregated via FedAvg.
"""

import torch
import torch.nn as nn


class TrafficLSTM(nn.Module):
    """
    LSTM-based traffic prediction model.

    Takes a sequence of traffic observations (ride_requests, congestion,
    traffic_flow) and predicts the next time-step values.

    Architecture:
        Input → LSTM (multi-layer) → Fully Connected → Output
    """

    def __init__(self, input_size: int = 3, hidden_size: int = 64,
                 num_layers: int = 2, output_size: int = 3,
                 dropout: float = 0.2):
        """
        Args:
            input_size:  Number of input features per time step.
            hidden_size: Number of hidden units in each LSTM layer.
            num_layers:  Number of stacked LSTM layers.
            output_size: Number of output features to predict.
            dropout:     Dropout rate between LSTM layers.
        """
        super(TrafficLSTM, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Multi-layer LSTM
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # Output projection layers
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, output_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch, seq_len, input_size)

        Returns:
            Predictions of shape (batch, output_size)
        """
        # LSTM forward pass
        lstm_out, _ = self.lstm(x)

        # Use only the last time step's output
        last_output = lstm_out[:, -1, :]

        # Project to output space
        prediction = self.fc(last_output)
        return prediction

    def get_flat_params(self) -> torch.Tensor:
        """Get all model parameters as a single flat vector."""
        return torch.cat([p.data.view(-1) for p in self.parameters()])

    def set_flat_params(self, flat_params: torch.Tensor):
        """Set model parameters from a single flat vector."""
        offset = 0
        for p in self.parameters():
            numel = p.numel()
            p.data.copy_(flat_params[offset:offset + numel].view(p.shape))
            offset += numel

    def get_flat_gradients(self) -> torch.Tensor:
        """Get all gradients as a single flat vector."""
        grads = []
        for p in self.parameters():
            if p.grad is not None:
                grads.append(p.grad.data.view(-1))
            else:
                grads.append(torch.zeros(p.numel()))
        return torch.cat(grads)

    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
