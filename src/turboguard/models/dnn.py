"""DNN binary classifier for TurboGuard.

A three-layer feedforward network with BatchNorm and Dropout, used both as
a standalone baseline classifier and as the downstream grey-zone classifier
inside TurboGuard's two-stage pipeline.
"""

import torch
import torch.nn as nn


class DNNClassifier(nn.Module):
    """Binary classifier: 0 = benign, 1 = attack.

    Architecture::

        Linear(input_dim, 2H) → BN → ReLU → Dropout(0.3)
        Linear(2H, H)         → BN → ReLU → Dropout(0.2)
        Linear(H, 2)

    The first layer is 2x wider than the second to provide a capacity
    gradient that encourages feature compression. Dropout rates decrease
    toward the output to preserve learned representations.

    Attributes:
        net: Sequential stack of layers.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 128) -> None:
        """Initializes the DNN classifier.

        Args:
            input_dim: Number of input features.
            hidden_dim: Base hidden layer width (first layer uses 2H).
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim * 2),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Computes raw logits for binary classification.

        Args:
            x: Input tensor of shape ``(B, input_dim)``.

        Returns:
            Logits tensor of shape ``(B, 2)``.
        """
        return self.net(x)
