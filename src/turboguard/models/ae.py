"""Comparable tabular autoencoder baseline."""

import torch
import torch.nn as nn


class AutoEncoder(nn.Module):
    """MLP autoencoder with an exposed continuous latent representation."""

    def __init__(self, input_dim: int, latent_dim: int = 32, hidden_dim: int = 128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.BatchNorm1d(hidden_dim), nn.Linear(hidden_dim, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim), nn.ReLU(),
            nn.BatchNorm1d(hidden_dim), nn.Linear(hidden_dim, input_dim)
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        z = self.encode(x)
        reconstruction = self.decode(z)
        return {
            "latent": z,
            "reconstruction": reconstruction,
            "reconstruction_loss": torch.mean((reconstruction - x) ** 2),
        }
