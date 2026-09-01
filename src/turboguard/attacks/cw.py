"""Carlini & Wagner (C&W) L2 attack with L-inf clipping — batched.

Optimisation-based attack that minimises ``L2(delta) + c * f(x + delta)``
where ``f`` is the C&W objective function. L-inf clipping is applied for
sweep consistency with PGD.

Reference:
    Carlini & Wagner, "Towards Evaluating the Robustness of Neural
    Networks", IEEE S&P 2017. https://arxiv.org/abs/1608.04644
"""

import torch
import torch.nn as nn


def cw_l2(
    model: nn.Module,
    X: torch.Tensor,
    y: torch.Tensor,
    eps: float = 0.5,
    c: float = 10.0,
    lr: float = 0.01,
    max_iter: int = 200,
    confidence: float = 0.0,
    targeted: bool = False,
    target_label: int = 0,
    batch_size: int = 512,
) -> torch.Tensor:
    """Generates C&W L2 adversarial examples (batched, GPU-parallel).

    Processes samples in mini-batches for GPU efficiency. Tracks the
    best successful perturbation per sample (smallest L2 that flips
    the prediction).

    Args:
        model: Target classifier. Switched to eval mode.
        X: Clean input samples of shape ``(N, D)``.
        y: True labels of shape ``(N,)`` (long, 0/1).
        eps: L-inf clipping radius for sweep consistency with PGD.
        c: Trade-off constant between L2 distance and misclassification.
        lr: Adam learning rate for delta optimisation.
        max_iter: Maximum optimisation iterations per batch.
        confidence: Extra margin kappa in the C&W objective.
        targeted: If True, push predictions toward ``target_label``.
        target_label: Target class when ``targeted=True``.
        batch_size: Number of samples to optimise simultaneously.

    Returns:
        Adversarial examples tensor of shape ``(N, D)``, detached.
    """
    model.eval()
    device = X.device
    N = X.shape[0]
    X_adv = X.clone().detach()

    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        x_batch = X[start:end].clone().detach()
        y_batch = y[start:end]
        B = x_batch.shape[0]

        # Learnable perturbation for the entire batch.
        delta = torch.zeros_like(x_batch, requires_grad=True, device=device)
        optimizer = torch.optim.Adam([delta], lr=lr)

        best_delta = torch.zeros_like(x_batch)
        best_l2 = torch.full((B,), float("inf"), device=device)
        found = torch.zeros(B, dtype=torch.bool, device=device)

        for _ in range(max_iter):
            x_new = x_batch + delta
            logits = model(x_new)

            # C&W objective: f(x) = max(Z_true - Z_other + kappa, 0)
            # for untargeted, or max(Z_other - Z_target + kappa, 0)
            # for targeted attacks.
            if targeted:
                z_target = logits[:, target_label]
                z_other = logits[:, 1 - target_label]
                f_obj = torch.clamp(z_other - z_target + confidence, min=0.0)
            else:
                z_true = logits.gather(1, y_batch.unsqueeze(1)).squeeze(1)
                y_other = 1 - y_batch
                z_other = logits.gather(1, y_other.unsqueeze(1)).squeeze(1)
                f_obj = torch.clamp(z_true - z_other + confidence, min=0.0)

            l2_dist = delta.pow(2).sum(dim=1)
            loss = (l2_dist + c * f_obj).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                # L-inf clipping keeps perturbation within eps-ball so
                # results are comparable across epsilon sweep levels.
                delta.data.clamp_(-eps, eps)

                # Track best (smallest L2) successful perturbation.
                preds = model(x_batch + delta).argmax(dim=1)
                if targeted:
                    success = preds == target_label
                else:
                    success = preds != y_batch

                curr_l2 = delta.pow(2).sum(dim=1)
                improved = success & (curr_l2 < best_l2)
                best_delta[improved] = delta[improved].detach().clone()
                best_l2[improved] = curr_l2[improved]
                found = found | success

        # Use best adversarial if found, otherwise keep last attempt.
        with torch.no_grad():
            final_delta = torch.where(
                found.unsqueeze(1).expand_as(delta),
                best_delta,
                delta.detach(),
            )
            X_adv[start:end] = x_batch + final_delta

    return X_adv.detach()
