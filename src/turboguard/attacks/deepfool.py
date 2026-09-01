"""DeepFool L2 attack — batched.

Iteratively linearises the classifier's decision boundary and computes the
minimal L2 perturbation needed to cross it. For binary classification
(2 classes), this reduces to a closed-form per-iteration step.

Reference:
    Moosavi-Dezfooli et al., "DeepFool: a simple and accurate method
    to fool deep neural networks", CVPR 2016.
    https://arxiv.org/abs/1511.04599
"""

import torch
import torch.nn as nn


def deepfool(
    model: nn.Module,
    X: torch.Tensor,
    y: torch.Tensor,
    max_iter: int = 50,
    epsilon: float = 1e-6,
    clip_max: float | None = None,
    batch_size: int = 512,
) -> torch.Tensor:
    """Generates DeepFool L2 adversarial examples (batched, binary).

    For each batch, iteratively finds the minimal L2 perturbation that
    crosses the decision boundary. Samples that have already flipped
    are excluded from subsequent iterations for efficiency.

    Args:
        model: Target classifier. Switched to eval mode.
        X: Clean input samples of shape ``(N, D)``.
        y: True labels of shape ``(N,)`` (long, 0/1).
        max_iter: Maximum perturbation iterations per sample.
        epsilon: Overshoot parameter — the final perturbation is scaled
            by ``(1 + epsilon)`` to ensure the sample crosses the boundary
            reliably rather than sitting exactly on it.
        clip_max: Optional upper bound for feature clipping.
        batch_size: Number of samples to process simultaneously.

    Returns:
        Adversarial examples tensor of shape ``(N, D)``, detached.
    """
    model.eval()
    device = X.device
    N, D = X.shape
    X_adv = X.clone().detach()

    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        x_batch = X[start:end].clone().detach()
        B = x_batch.shape[0]

        with torch.no_grad():
            f_orig = model(x_batch)
            k_hat = f_orig.argmax(dim=1)

        x_pert = x_batch.clone()
        still_active = torch.ones(B, dtype=torch.bool, device=device)

        for _ in range(max_iter):
            if not still_active.any():
                break

            active_idx = still_active.nonzero(as_tuple=True)[0]
            x_active = x_pert[active_idx].detach().requires_grad_(True)

            logits = model(x_active)
            preds = logits.argmax(dim=1)

            # Mark samples whose predictions have already flipped.
            flipped = preds != k_hat[active_idx]
            if flipped.any():
                still_active[active_idx[flipped]] = False

            remaining = ~flipped
            if not remaining.any():
                break

            rem_idx = active_idx[remaining]
            logits_rem = logits[remaining]
            k_hat_rem = k_hat[rem_idx]

            # For binary classification: compute gradient of the logit
            # difference (f_other - f_true) w.r.t. the input.
            k_other = 1 - k_hat_rem
            f_diff = logits_rem.gather(1, k_other.unsqueeze(1)).squeeze(1) - logits_rem.gather(
                1, k_hat_rem.unsqueeze(1)
            ).squeeze(1)

            grad_outputs = torch.ones_like(f_diff)
            grads = torch.autograd.grad(
                f_diff,
                x_active,
                grad_outputs=grad_outputs,
                retain_graph=False,
                create_graph=False,
            )[0]
            grads = grads[remaining]

            # Minimal perturbation: r = |f_diff| / ||grad||^2 * grad.
            # This is the closed-form solution for the shortest vector
            # to the linearised decision boundary.
            w_norm_sq = grads.pow(2).sum(dim=1, keepdim=True).clamp(min=1e-10)
            r = (f_diff.abs().unsqueeze(1) / w_norm_sq) * grads

            x_pert[rem_idx] = x_pert[rem_idx].detach() + r.detach()

            if clip_max is not None:
                x_pert[rem_idx] = torch.clamp(x_pert[rem_idx], 0.0, clip_max)

        # Overshoot: scale perturbation slightly beyond the boundary.
        total_pert = x_pert.detach() - x_batch
        x_final = x_batch + (1 + epsilon) * total_pert
        X_adv[start:end] = x_final

    return X_adv.detach()
