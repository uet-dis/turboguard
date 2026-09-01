"""Backward approximations for non-differentiable detector components."""

from __future__ import annotations

import torch


def straight_through_hard_quantize(x: torch.Tensor, codes: torch.Tensor) -> torch.Tensor:
    """Nearest-code forward with identity backward (STE)."""
    distances = ((x.unsqueeze(1) - codes.unsqueeze(0)) ** 2).sum(dim=-1)
    hard = codes[distances.argmin(dim=1)]
    return x + (hard - x).detach()


def soft_code_assignment(x: torch.Tensor, codes: torch.Tensor, temperature: float = 0.1) -> torch.Tensor:
    """Differentiable soft assignment for attack development only."""
    distances = ((x.unsqueeze(1) - codes.unsqueeze(0)) ** 2).sum(dim=-1)
    weights = torch.softmax(-distances / max(temperature, 1e-6), dim=1)
    return weights @ codes


def finite_difference_margin(
    margin_fn,
    x: torch.Tensor,
    sigma: float = 1e-3,
    directions: int = 32,
) -> torch.Tensor:
    """Estimate a scalar margin gradient for non-differentiable scores."""
    estimates = torch.zeros_like(x)
    for _ in range(directions):
        u = torch.randn_like(x)
        u = u / u.norm(dim=1, keepdim=True).clamp_min(1e-8)
        plus = margin_fn(x + sigma * u)
        minus = margin_fn(x - sigma * u)
        estimates += ((plus - minus) / (2.0 * sigma)).unsqueeze(1) * u
    return estimates / max(directions, 1)
