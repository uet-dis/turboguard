"""Vector Quantized Variational Autoencoder (VQ-VAE).

The core representation learning module in TurboGuard. Learns a discrete
codebook of benign network traffic patterns by encoding inputs into a
continuous latent space and quantizing them to the nearest codebook entry.

Uses Exponential Moving Average (EMA) updates for codebook learning, which
avoids mode collapse issues seen with gradient-based codebook updates
(van den Oord et al., 2017).

Includes dead code replacement (SoundStream [2]) to prevent codebook
collapse — the key architectural contribution that ensures fine-grained
Voronoi coverage of the benign data manifold.

References:
    [1] van den Oord et al., "Neural Discrete Representation Learning",
        NeurIPS 2017. DOI: 10.48550/arXiv.1711.00937
    [2] Zeghidour et al., "SoundStream: An End-to-End Neural Audio Codec",
        IEEE/ACM TASLP, 30, 495-507, 2021. DOI: 10.1109/TASLP.2021.3129994
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from turboguard.config import COMMITMENT_COST, EMA_DECAY, EMA_EPSILON


class VectorQuantizer(nn.Module):
    """Discretizes continuous latent vectors using a learned codebook.

    Maps each encoder output to its nearest codebook entry using L2 distance.
    Codebook entries are updated via Exponential Moving Average (EMA) of
    the encoder outputs assigned to each code, so no gradient flows through
    the codebook itself. The straight-through estimator copies gradients
    from the decoder input back to the encoder output.

    Attributes:
        _embedding: Codebook of shape ``(K, D)``.
        _ema_cluster_size: EMA-tracked cluster sizes per code.
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        commitment_cost: float = COMMITMENT_COST,
    ) -> None:
        """Initializes the vector quantizer.

        Args:
            num_embeddings: Codebook size K (number of discrete codes).
            embedding_dim: Dimension D of each code vector.
            commitment_cost: Weight beta for the commitment loss that
                penalises the encoder for producing outputs far from
                codebook entries. Default 0.25 per [1].
        """
        super().__init__()
        self._num_embeddings = num_embeddings
        self._embedding_dim = embedding_dim
        self._commitment_cost = commitment_cost

        self._embedding = nn.Embedding(num_embeddings, embedding_dim)
        # Uniform init scaled by 1/K following [1].
        self._embedding.weight.data.uniform_(-1 / num_embeddings, 1 / num_embeddings)

        self.register_buffer("_ema_cluster_size", torch.zeros(num_embeddings))
        self._ema_w = nn.Parameter(torch.randn(num_embeddings, embedding_dim))
        self.register_buffer("_last_usage", torch.zeros(num_embeddings))
        self._dead_code_resets = 0

        self._decay = EMA_DECAY
        self._epsilon = EMA_EPSILON

    def forward(self, inputs: torch.Tensor) -> dict:
        """Quantizes encoder outputs to nearest codebook entries.

        Args:
            inputs: Encoded latent vectors of shape ``(B, D)``.

        Returns:
            Dict with keys:
                - quantized: Quantized vectors ``(B, D)``.
                - vq_loss: Commitment loss (scalar).
                - perplexity: Codebook usage perplexity (scalar).
                - encoding_indices: Assigned code indices ``(B, 1)``.
        """
        # Compute squared L2 distances to all codes using expanded form:
        # ||z - c||^2 = ||z||^2 + ||c||^2 - 2 * z · c
        distances = (
            inputs.pow(2).sum(dim=1, keepdim=True)
            + self._embedding.weight.pow(2).sum(dim=1)
            - 2 * inputs @ self._embedding.weight.t()
        )

        encoding_indices = distances.argmin(dim=1).unsqueeze(1)
        encodings = torch.zeros(
            encoding_indices.shape[0],
            self._num_embeddings,
            device=inputs.device,
        )
        encodings.scatter_(1, encoding_indices, 1)

        quantized = encodings @ self._embedding.weight

        if self.training:
            # EMA update: track cluster sizes and weighted sum of
            # encoder outputs per code. This replaces gradient-based
            # codebook updates and is more stable for tabular data.
            self._ema_cluster_size = self._ema_cluster_size * self._decay + (
                1 - self._decay
            ) * encodings.sum(0)
            # Laplace smoothing prevents division by zero for unused codes.
            n = self._ema_cluster_size.sum()
            self._ema_cluster_size = (
                (self._ema_cluster_size + self._epsilon)
                / (n + self._num_embeddings * self._epsilon)
                * n
            )

            dw = encodings.t() @ inputs
            self._ema_w.data.mul_(self._decay).add_((1 - self._decay) * dw)
            # Codebook entry = weighted average of assigned encoder outputs.
            self._embedding.weight.data.copy_(
                self._ema_w.data / self._ema_cluster_size.unsqueeze(1)
            )

        # Commitment loss: penalises encoder for drifting away from codes.
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        loss = self._commitment_cost * e_latent_loss

        # Straight-through estimator: copies decoder gradients to encoder.
        quantized = inputs + (quantized - inputs).detach()

        # Perplexity: exp(entropy) of code usage. Higher = more codes used.
        avg_probs = encodings.mean(dim=0)
        self._last_usage = encodings.sum(dim=0).detach()
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        return {
            "quantized": quantized,
            "vq_loss": loss,
            "perplexity": perplexity,
            "encoding_indices": encoding_indices,
        }

    def reset_dead_codes(self, encoder_outputs: torch.Tensor, usage_counts: torch.Tensor) -> int:
        """Replaces dead codebook entries with sampled encoder outputs.

        Dead codes (near-zero EMA usage) create oversized Voronoi cells
        that hide low-epsilon adversarial perturbations because the
        adversarial sample still falls closer to the dead code's centroid
        than to any other code. Replacing dead codes with actual encoder
        outputs forces the codebook to cover the data manifold uniformly.

        Adapted from the codebook reset strategy in SoundStream [2].

        Args:
            encoder_outputs: A batch of encoded latent vectors ``(N, D)``
                to sample replacement codes from.
            usage_counts: Per-code EMA usage counts ``(K,)``. Codes with
                count < 1.0 are considered dead.

        Returns:
            Number of codebook entries that were reset.
        """
        dead_mask = usage_counts < 1.0
        n_dead = int(dead_mask.sum().item())
        if n_dead == 0:
            return 0

        # Sample random encoder outputs and add small noise to break ties
        # if multiple dead codes sample the same encoder output.
        pick = torch.randint(0, len(encoder_outputs), (n_dead,))
        new_codes = encoder_outputs[pick] + torch.randn_like(encoder_outputs[pick]) * 0.01

        self._embedding.weight.data[dead_mask] = new_codes
        self._ema_w.data[dead_mask] = new_codes
        # Reset EMA count to 1.0 so freshly-reset codes aren't immediately
        # killed on the next dead-code check.
        self._ema_cluster_size[dead_mask] = 1.0
        self._dead_code_resets += n_dead

        return n_dead

    def diagnostics(self) -> dict[str, object]:
        """Return code occupancy and reset counters for sensitivity reports."""
        usage = self._last_usage.detach().cpu().numpy()
        probability = usage / max(float(usage.sum()), 1.0)
        nonzero = probability > 0
        return {
            "usage": usage.tolist(),
            "active_codes": int(nonzero.sum()),
            "dead_codes": int((~nonzero).sum()),
            "perplexity": float(np.exp(-np.sum(probability[nonzero] * np.log(probability[nonzero])))) if nonzero.any() else 0.0,
            "dead_code_resets": int(self._dead_code_resets),
        }


class VQVAE(nn.Module):
    """VQ-VAE autoencoder for tabular network traffic data.

    Architecture: encoder → vector quantizer → decoder, where the encoder
    and decoder are shallow MLPs suitable for tabular (non-image) data.

    Attributes:
        encoder: MLP that maps input features to latent space.
        vq: Vector quantizer module.
        decoder: MLP that reconstructs input from quantized latents.
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        num_embeddings: int,
        hidden_dim: int = 128,
    ) -> None:
        """Initializes the VQ-VAE.

        Args:
            input_dim: Number of input features (e.g. 42 for UNSW).
            latent_dim: Bottleneck dimension D.
            num_embeddings: Codebook size K.
            hidden_dim: Width of encoder/decoder hidden layers.
        """
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, latent_dim),
            nn.ReLU(),
        )

        self.vq = VectorQuantizer(num_embeddings, latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encodes input features to continuous latent space.

        Args:
            x: Input tensor of shape ``(B, input_dim)``.

        Returns:
            Latent vectors ``(B, latent_dim)`` before quantization.
        """
        return self.encoder(x)

    def forward(self, x: torch.Tensor) -> dict:
        """Full forward pass: encode → quantize → decode.

        Args:
            x: Input tensor of shape ``(B, input_dim)``.

        Returns:
            Dict with keys: reconstruction, recon_loss, vq_loss,
            total_loss, encoding_indices.
        """
        z = self.encode(x)
        vq_out = self.vq(z)
        x_recon = self.decoder(vq_out["quantized"])

        recon_loss = F.mse_loss(x_recon, x)
        total_loss = recon_loss + vq_out["vq_loss"]

        return {
            "reconstruction": x_recon,
            "recon_loss": recon_loss,
            "vq_loss": vq_out["vq_loss"],
            "total_loss": total_loss,
            "encoding_indices": vq_out["encoding_indices"],
        }
