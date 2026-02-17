"""
Module 3: Local Training Module

Implements local traffic prediction model training within each zone.
Each zone trains its own LSTM model on its local traffic data,
capturing temporal traffic patterns using historical data.
"""

import copy
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict, List, Tuple, Optional
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from models.traffic_model import TrafficLSTM
from utils.helpers import normalize_data, denormalize_data, create_sequences


class LocalTrainingModule:
    """
    Trains a local TrafficLSTM model for a single zone (federated client).
    """

    def __init__(self, zone_id: int, device: torch.device = None):
        """
        Args:
            zone_id: Identifier of the zone this client represents.
            device:  Torch device (cpu or cuda).
        """
        self.zone_id = zone_id
        self.device = device or torch.device("cpu")

        # Initialize model
        self.model = TrafficLSTM(
            input_size=config.INPUT_FEATURES,
            hidden_size=config.HIDDEN_SIZE,
            num_layers=config.NUM_LAYERS,
            output_size=config.OUTPUT_FEATURES,
        ).to(self.device)

        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=config.LEARNING_RATE
        )
        self.criterion = nn.MSELoss()

        # Normalization parameters (learned from training data)
        self.min_vals: Optional[np.ndarray] = None
        self.max_vals: Optional[np.ndarray] = None

        # Training history
        self.train_losses: List[float] = []

    def prepare_data(self, raw_data: np.ndarray,
                     train_ratio: float = 0.8) -> Tuple[DataLoader, DataLoader]:
        """
        Normalize data, create sequences, and build DataLoaders.

        Args:
            raw_data:    Shape (num_timesteps, 3)
            train_ratio: Fraction of data for training.

        Returns:
            (train_loader, test_loader)
        """
        # Normalize
        norm_data, self.min_vals, self.max_vals = normalize_data(raw_data)

        # Train/test split
        split = int(len(norm_data) * train_ratio)
        train_data = norm_data[:split]
        test_data = norm_data[split:]

        # Create sequences
        X_train, y_train = create_sequences(
            train_data, config.SEQUENCE_LENGTH, config.PREDICTION_HORIZON
        )
        X_test, y_test = create_sequences(
            test_data, config.SEQUENCE_LENGTH, config.PREDICTION_HORIZON
        )

        # Convert to tensors
        train_dataset = TensorDataset(
            torch.FloatTensor(X_train), torch.FloatTensor(y_train)
        )
        test_dataset = TensorDataset(
            torch.FloatTensor(X_test), torch.FloatTensor(y_test)
        )

        train_loader = DataLoader(
            train_dataset, batch_size=config.BATCH_SIZE, shuffle=True
        )
        test_loader = DataLoader(
            test_dataset, batch_size=config.BATCH_SIZE, shuffle=False
        )

        return train_loader, test_loader

    def get_model_params(self) -> Dict[str, torch.Tensor]:
        """Get a copy of the current model parameters."""
        return copy.deepcopy(self.model.state_dict())

    def set_model_params(self, state_dict: Dict[str, torch.Tensor]):
        """Set model parameters (e.g., from global model)."""
        self.model.load_state_dict(state_dict)

    def train_local(self, train_loader: DataLoader,
                    epochs: int = None) -> Dict:
        """
        Train the local model for a number of epochs.

        Args:
            train_loader: DataLoader for training data.
            epochs:       Number of local epochs.

        Returns:
            Dictionary with training metrics.
        """
        epochs = epochs or config.LOCAL_EPOCHS
        self.model.train()

        initial_params = self.get_model_params()
        epoch_losses = []

        for epoch in range(epochs):
            running_loss = 0.0
            num_batches = 0

            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                self.optimizer.zero_grad()
                predictions = self.model(X_batch)
                loss = self.criterion(predictions, y_batch)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()
                num_batches += 1

            avg_loss = running_loss / max(num_batches, 1)
            epoch_losses.append(avg_loss)

        self.train_losses.extend(epoch_losses)

        # Compute gradient (parameter delta) for federated aggregation
        updated_params = self.get_model_params()
        gradient = {}
        for key in initial_params:
            gradient[key] = updated_params[key] - initial_params[key]

        return {
            "zone_id": self.zone_id,
            "avg_loss": np.mean(epoch_losses),
            "final_loss": epoch_losses[-1],
            "gradient": gradient,
            "num_samples": len(train_loader.dataset),
        }

    def evaluate(self, test_loader: DataLoader) -> Dict:
        """
        Evaluate the model on test data.

        Returns:
            Dictionary with evaluation metrics.
        """
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                predictions = self.model(X_batch)
                loss = self.criterion(predictions, y_batch)
                total_loss += loss.item()

                all_preds.append(predictions.cpu().numpy())
                all_targets.append(y_batch.cpu().numpy())

        all_preds = np.concatenate(all_preds, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)

        # Denormalize for real-scale metrics
        preds_real = denormalize_data(all_preds, self.min_vals, self.max_vals)
        targets_real = denormalize_data(all_targets, self.min_vals,
                                        self.max_vals)

        mae = np.mean(np.abs(preds_real - targets_real))
        rmse = np.sqrt(np.mean((preds_real - targets_real) ** 2))

        # MAPE (avoid division by zero)
        mask = np.abs(targets_real) > 1e-6
        if mask.any():
            mape = np.mean(
                np.abs((preds_real[mask] - targets_real[mask])
                       / targets_real[mask])
            ) * 100
        else:
            mape = 0.0

        return {
            "zone_id": self.zone_id,
            "test_loss": total_loss / max(len(test_loader), 1),
            "mae": mae,
            "rmse": rmse,
            "mape": mape,
        }
