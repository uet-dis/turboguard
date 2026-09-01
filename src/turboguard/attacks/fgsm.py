"""Fast Gradient Sign Method (FGSM) L-infinity attack.

Single-step gradient attack that perturbs inputs by ``eps * sign(grad)``
to maximise (untargeted) or minimise (targeted) cross-entropy loss.

Reference:
    Goodfellow et al., "Explaining and Harnessing Adversarial Examples",
    ICLR 2015. https://arxiv.org/abs/1412.6572
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def fgsm_linf(
    model: nn.Module,
    X: torch.Tensor,
    y: torch.Tensor,
    eps: float = 0.3,
    targeted: bool = False,
) -> torch.Tensor:
    """Generates FGSM L-infinity adversarial examples.

    Args:
        model: Target classifier (``nn.Module``). Switched to eval mode.
        X: Clean input samples of shape ``(N, D)``.
        y: True labels of shape ``(N,)`` (long).
        eps: Perturbation magnitude (L-inf bound).
        targeted: If True, minimise loss to push predictions toward ``y``.

    Returns:
        Adversarial examples tensor of shape ``(N, D)``, detached.
    """
    model.eval()
    X_adv = X.clone().detach()
    X_adv.requires_grad_(True)

    logits = model(X_adv)
    loss = F.cross_entropy(logits, y)

    grad = torch.autograd.grad(loss, X_adv)[0]
    # Replace NaN gradients with zero (ART pattern) to prevent
    # invalid perturbations from numerical instability.
    grad = torch.nan_to_num(grad, nan=0.0)

    with torch.no_grad():
        if targeted:
            X_adv = X_adv - eps * grad.sign()
        else:
            X_adv = X_adv + eps * grad.sign()

    return X_adv.detach()
