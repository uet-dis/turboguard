"""Comparable tabular variational autoencoder baseline."""

import torch
import torch.nn as nn


class VariationalAutoEncoder(nn.Module):
    """MLP VAE with configurable beta-weighted KL divergence."""

    def __init__(self, input_dim: int, latent_dim: int = 32, hidden_dim: int = 128, beta: float = 1.0):
        super().__init__()
        self.beta = beta
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.BatchNorm1d(hidden_dim)
        )
        self.mu = nn.Linear(hidden_dim, latent_dim)
        self.logvar = nn.Linear(hidden_dim, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim), nn.ReLU(),
            nn.BatchNorm1d(hidden_dim), nn.Linear(hidden_dim, input_dim)
        )

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        mu, logvar = self.mu(h), self.logvar(h).clamp(-20.0, 20.0)
        std = torch.exp(0.5 * logvar)
        z = mu + std * torch.randn_like(std) if self.training else mu
        return z, mu, logvar

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        z, mu, logvar = self.encode(x)
        reconstruction = self.decode(z)
        reconstruction_loss = torch.mean((reconstruction - x) ** 2)
        kl = -0.5 * torch.mean(1.0 + logvar - mu.pow(2) - logvar.exp())
        return {
            "latent": z,
            "mu": mu,
            "logvar": logvar,
            "reconstruction": reconstruction,
            "reconstruction_loss": reconstruction_loss,
            "kl_loss": kl,
            "total_loss": reconstruction_loss + self.beta * kl,
        }
