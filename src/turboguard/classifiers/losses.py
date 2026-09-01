"""Class-imbalance losses selected on validation data."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, weight: torch.Tensor | None = None):
        super().__init__()
        self.gamma = gamma
        self.register_buffer("weight", weight if weight is not None else torch.empty(0))

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits, target, weight=self.weight if self.weight.numel() else None, reduction="none")
        return ((1.0 - torch.exp(-ce)) ** self.gamma * ce).mean()


class ClassBalancedLoss(nn.Module):
    def __init__(self, samples_per_class: list[int], beta: float = 0.9999):
        super().__init__()
        effective = [(1.0 - beta ** n) / (1.0 - beta) if n else 1.0 for n in samples_per_class]
        weights = torch.tensor([1.0 / value for value in effective], dtype=torch.float32)
        self.register_buffer("weight", weights / weights.mean())

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(logits, target, weight=self.weight)


def build_loss(name: str, class_counts: tuple[int, int], gamma: float = 2.0) -> nn.Module:
    """Construct an explicitly named loss for reproducible sweeps."""
    if name == "bce":
        return nn.CrossEntropyLoss()
    if name == "weighted_bce":
        weight = torch.tensor([1.0, class_counts[0] / max(class_counts[1], 1)])
        return nn.CrossEntropyLoss(weight=weight)
    if name == "focal":
        return FocalLoss(gamma=gamma)
    if name == "class_balanced":
        return ClassBalancedLoss(list(class_counts))
    raise ValueError(f"Unknown loss: {name}")
