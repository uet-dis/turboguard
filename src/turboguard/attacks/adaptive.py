"""Adaptive attacks against the complete two-stage detector."""

from __future__ import annotations

from typing import Callable, Optional
import torch
import torch.nn as nn


class IFScoreSurrogate(nn.Module):
    """Small differentiable regressor for Isolation-Forest normality scores."""

    def __init__(self, n_signals: int = 6, hidden_dim: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_signals, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, signals: torch.Tensor) -> torch.Tensor:
        return self.net(signals).squeeze(1)


class DifferentiableTurboGuard(nn.Module):
    """BPDA/STE approximation of the frozen TurboGuard scoring pipeline.

    The VQ-VAE encoder, decoder, geometric signals, signal scaling, and DNN
    use their frozen model parameters. Hard nearest-code assignment is used
    in the forward pass with a straight-through gradient. The sklearn
    Isolation Forest is represented by ``if_surrogate`` during optimization;
    all claimed successes must subsequently be checked by exact inference.
    """

    def __init__(self, tg, if_surrogate: IFScoreSurrogate) -> None:
        super().__init__()
        self.vqvae = tg.vqvae
        self.dnn = tg._dnn
        self.if_surrogate = if_surrogate
        self.signal_names = tuple(tg._signal_names)

        self.register_buffer(
            "pca_mean", torch.as_tensor(tg._pca_mean, dtype=torch.float32)
        )
        self.register_buffer(
            "pca_components",
            torch.as_tensor(tg._pca_components, dtype=torch.float32),
        )
        self.register_buffer(
            "ctf_mean", torch.as_tensor(tg._ctf_mean, dtype=torch.float32)
        )
        self.register_buffer(
            "ctf_inv_cov",
            torch.as_tensor(tg._ctf_inv_cov, dtype=torch.float32),
        )
        self.register_buffer(
            "signal_mean",
            torch.as_tensor(tg._scaler.mean_, dtype=torch.float32),
        )
        self.register_buffer(
            "signal_scale",
            torch.as_tensor(tg._scaler.scale_, dtype=torch.float32),
        )

        n_codes = int(tg._num_embeddings)
        global_mu = float(tg._global_mu)
        global_std = max(float(tg._global_std), 1e-8)
        cc_mu = torch.full((n_codes,), global_mu, dtype=torch.float32)
        cc_std = torch.full((n_codes,), global_std, dtype=torch.float32)
        for code, (mu, std) in tg._code_stats.items():
            cc_mu[int(code)] = float(mu)
            cc_std[int(code)] = max(float(std), 1e-8)
        self.register_buffer("cc_mu", cc_mu)
        self.register_buffer("cc_std", cc_std)

        latent_dim = int(tg._latent_dim)
        max_centroids = max(
            (len(entries) for entries in tg.semantic_map.values()), default=1
        )
        centroids = torch.zeros(
            (n_codes, max_centroids, latent_dim), dtype=torch.float32
        )
        counts = torch.ones((n_codes, max_centroids), dtype=torch.float32)
        valid = torch.zeros((n_codes, max_centroids), dtype=torch.bool)
        for code, entries in tg.semantic_map.items():
            for index, (centroid, count) in enumerate(entries):
                centroids[int(code), index] = torch.as_tensor(
                    centroid, dtype=torch.float32
                )
                counts[int(code), index] = max(float(count), 0.1)
                valid[int(code), index] = True
        self.register_buffer("semantic_centroids", centroids)
        self.register_buffer("semantic_counts", counts)
        self.register_buffer("semantic_valid", valid)

        self.vqvae.eval()
        self.dnn.eval()
        self.if_surrogate.eval()
        for module in (self.vqvae, self.dnn, self.if_surrogate):
            for parameter in module.parameters():
                parameter.requires_grad_(False)

    def _signals_and_reconstruction(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.vqvae.encode(x)
        codes = self.vqvae.vq._embedding.weight
        distances = (
            z.pow(2).sum(dim=1, keepdim=True)
            + codes.pow(2).sum(dim=1)
            - 2.0 * z @ codes.t()
        )
        indices = distances.argmin(dim=1)
        hard_codes = codes[indices]
        quantized = z + (hard_codes - z).detach()
        reconstruction = self.vqvae.decoder(quantized)

        centered = distances - self.pca_mean
        projected = centered @ self.pca_components.t()
        ctf_delta = projected - self.ctf_mean
        ctf = torch.sqrt(
            torch.clamp(
                torch.sum((ctf_delta @ self.ctf_inv_cov) * ctf_delta, dim=1),
                min=1e-12,
            )
        )

        nearest = torch.topk(distances, k=2, dim=1, largest=False).values
        vmr = nearest[:, 0] / (nearest[:, 1] + 1e-10)
        probabilities = torch.softmax(-distances, dim=1)
        entropy = -(probabilities * torch.log(probabilities + 1e-10)).sum(dim=1)
        reconstruction_error = (reconstruction - x).pow(2).mean(dim=1)

        code_centroids = self.semantic_centroids[indices]
        code_counts = self.semantic_counts[indices]
        code_valid = self.semantic_valid[indices]
        geo_distances = (
            (z.unsqueeze(1) - code_centroids).pow(2).sum(dim=2)
            / (code_counts + 1e-6)
        )
        geo_distances = geo_distances.masked_fill(~code_valid, torch.inf)
        geo = geo_distances.min(dim=1).values
        finite_geo = torch.isfinite(geo)
        if finite_geo.any():
            fallback_geo = geo[finite_geo].max() * 10.0
        else:
            fallback_geo = torch.as_tensor(
                1e6, dtype=geo.dtype, device=geo.device
            )
        geo = torch.where(finite_geo, geo, fallback_geo)

        cc = (reconstruction_error - self.cc_mu[indices]) / self.cc_std[indices]
        all_signals = torch.stack(
            (ctf, vmr, entropy, reconstruction_error, geo, cc), dim=1
        )
        from turboguard.core.turboguard import SIGNAL_NAMES

        selected = [SIGNAL_NAMES.index(name) for name in self.signal_names]
        return all_signals[:, selected], reconstruction

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        signals, reconstruction = self._signals_and_reconstruction(x)
        transformed = torch.sign(signals) * torch.log1p(torch.abs(signals))
        scaled = (transformed - self.signal_mean) / self.signal_scale
        if_score = self.if_surrogate(scaled)
        dnn_input = torch.cat(
            (x, reconstruction, torch.abs(x - reconstruction), scaled), dim=1
        )
        dnn_probability = torch.softmax(self.dnn(dnn_input), dim=1)[:, 1]
        return if_score, dnn_probability, signals

    def margins(
        self, x: torch.Tensor, tau_if: float, tau_dnn: float
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if_score, dnn_probability, _ = self(x)
        return tau_if - if_score, dnn_probability - tau_dnn


def adaptive_margin_loss(
    if_margin: torch.Tensor,
    dnn_margin: torch.Tensor,
    distortion: torch.Tensor,
    weight_if: float = 1.0,
    weight_dnn: float = 1.0,
    distortion_weight: float = 0.01,
) -> torch.Tensor:
    """Objective requiring both IF and DNN margins to cross below zero."""
    detector_loss = torch.relu(if_margin) * weight_if + torch.relu(dnn_margin) * weight_dnn
    return (detector_loss + distortion_weight * distortion).mean()


def adaptive_pgd(
    X: torch.Tensor,
    score_fn: Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor]],
    eps: float,
    step_size: float,
    steps: int = 100,
    restarts: int = 1,
    projection: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
    weight_if: float = 1.0,
    weight_dnn: float = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Run PGD using differentiable IF and DNN margins.

    ``score_fn(x)`` must return ``(if_margin, dnn_margin)`` where positive
    values indicate detection. The caller must verify the returned tensor
    with exact TurboGuard inference after the attack.
    """
    best = X.detach().clone()
    best_loss = torch.full((len(X),), float("inf"), device=X.device)
    best_if = torch.full((len(X),), float("nan"), device=X.device)
    best_dnn = torch.full((len(X),), float("nan"), device=X.device)
    best_restart = torch.full((len(X),), -1, dtype=torch.long, device=X.device)
    for restart in range(restarts):
        delta = torch.empty_like(X).uniform_(-eps, eps)
        if restart == 0:
            delta.zero_()
        delta.requires_grad_(True)
        for _ in range(steps):
            x_adv = X + delta
            if_margin, dnn_margin = score_fn(x_adv)
            distortion = delta.pow(2).flatten(1).sum(dim=1)
            loss = adaptive_margin_loss(
                if_margin, dnn_margin, distortion, weight_if, weight_dnn
            )
            grad = torch.autograd.grad(loss, delta, retain_graph=False)[0]
            with torch.no_grad():
                delta -= step_size * grad.sign()
                delta.clamp_(-eps, eps)
                if projection is not None:
                    projected = projection(X, X + delta)
                    delta.copy_(projected - X)
            delta.requires_grad_(True)

        with torch.no_grad():
            if_margin, dnn_margin = score_fn(X + delta)
            objective = torch.relu(if_margin) + torch.relu(dnn_margin)
            improved = objective < best_loss
            best[improved] = (X + delta)[improved]
            best_loss[improved] = objective[improved]
            best_if[improved] = if_margin[improved]
            best_dnn[improved] = dnn_margin[improved]
            best_restart[improved] = restart
    return best.detach(), {
        "objective": best_loss.detach(),
        "if_margin": best_if.detach(),
        "dnn_margin": best_dnn.detach(),
        "restart": best_restart.detach(),
    }
