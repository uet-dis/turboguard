"""Projected Gradient Descent (PGD) L-infinity attack + Surrogate MLP.

PGD is an iterative extension of FGSM that projects perturbations back
onto the L-inf epsilon-ball after each gradient step. The surrogate MLP
is a lightweight model trained to mimic a non-differentiable classifier
(e.g. XGBoost) for generating transfer adversarial examples.

Reference:
    Madry et al., "Towards Deep Learning Models Resistant to Adversarial
    Attacks", ICLR 2018. https://arxiv.org/abs/1706.06083
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


def pgd_linf(
    model: nn.Module,
    X: torch.Tensor,
    y: torch.Tensor,
    eps: float = 0.3,
    eps_step: float = 0.01,
    max_iter: int = 40,
    targeted: bool = False,
) -> torch.Tensor:
    """Generates PGD L-infinity adversarial examples.

    Iteratively perturbs inputs, projecting the perturbation back onto
    the L-inf ball of radius ``eps`` after each step.

    Args:
        model: Target classifier. Switched to eval mode.
        X: Clean input samples of shape ``(N, D)``.
        y: True labels of shape ``(N,)`` (long).
        eps: Maximum L-inf perturbation radius.
        eps_step: Gradient step size per iteration.
        max_iter: Number of PGD iterations.
        targeted: If True, minimise loss to push toward ``y``.

    Returns:
        Adversarial examples tensor of shape ``(N, D)``, detached.
    """
    model.eval()
    X_orig = X.clone().detach()
    X_adv = X.clone().detach()

    for _ in range(max_iter):
        X_adv.requires_grad_(True)
        logits = model(X_adv)
        loss = F.cross_entropy(logits, y)
        grad = torch.autograd.grad(loss, X_adv)[0]
        grad = torch.nan_to_num(grad, nan=0.0)

        with torch.no_grad():
            if targeted:
                X_adv = X_adv - eps_step * grad.sign()
            else:
                X_adv = X_adv + eps_step * grad.sign()

            # Project perturbation back to L-inf ball around original.
            delta = torch.clamp(X_adv - X_orig, -eps, eps)
            X_adv = (X_orig + delta).detach()

    return X_adv


class SurrogateMLP(nn.Module):
    """Lightweight MLP trained to mimic a non-differentiable classifier.

    Intentionally uses a different architecture from DNNClassifier
    (no BatchNorm, no Dropout) to prove cross-model transferability.

    Attributes:
        net: Sequential MLP stack.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 128) -> None:
        """Initializes the surrogate MLP.

        Args:
            input_dim: Number of input features.
            hidden_dim: Width of each hidden layer.
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Computes raw logits.

        Args:
            x: Input tensor of shape ``(B, input_dim)``.

        Returns:
            Logits tensor of shape ``(B, 2)``.
        """
        return self.net(x)


def train_surrogate(
    X: torch.Tensor,
    y: torch.Tensor,
    device: torch.device,
    seed: int = 42,
    epochs: int = 10,
    batch_size: int = 512,
) -> SurrogateMLP:
    """Trains a surrogate MLP for transfer attacks.

    The surrogate learns to replicate the decision boundary of a
    non-differentiable classifier so that gradient-based attacks
    (FGSM, PGD) can be applied via the surrogate and then transferred.

    Args:
        X: Training features of shape ``(N, D)``.
        y: Binary labels of shape ``(N,)`` (long, 0/1).
        device: Computation device.
        seed: Random seed for reproducibility.
        epochs: Number of training epochs.
        batch_size: Mini-batch size.

    Returns:
        Trained surrogate model in eval mode.
    """
    torch.manual_seed(seed)
    model = SurrogateMLP(X.shape[1]).to(device)
    model.train()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    for _ in range(epochs):
        perm = torch.randperm(len(X))
        for i in range(0, len(X), batch_size):
            batch_x = X[perm[i : i + batch_size]].to(device)
            batch_y = y[perm[i : i + batch_size]].to(device).long()
            optimizer.zero_grad()
            loss = criterion(model(batch_x), batch_y)
            loss.backward()
            optimizer.step()

    model.eval()
    return model
