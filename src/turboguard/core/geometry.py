"""Geometric analysis module for TurboGuard.

Implements the Semantic Map and the Geometric Score (GEO signal), which
together measure how far a sample drifts from known-good latent regions.

The semantic map is a per-code clustering of benign latent vectors built
during training. At inference time, each sample's latent vector is compared
against the centroids of its assigned codebook entry, producing a
density-weighted distance score. This score is the strongest individual
signal in TurboGuard, contributing ~65% of DNN feature importance.
"""

from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
import torch
from sklearn.cluster import KMeans

from turboguard.config import SEED
from turboguard.models.vqvae import VQVAE


def build_semantic_map(
    vqvae: VQVAE,
    X_benign: torch.Tensor,
    device: torch.device,
    num_embeddings: int = 1024,
    n_centroids: int = 5,
    batch_size: int = 1024,
) -> Dict[int, List[Tuple[np.ndarray, float]]]:
    """Builds the Semantic Map from benign traffic latent space.

    For each codebook index k, collects all benign encoder outputs z that
    were assigned to code k, then runs KMeans to find ``n_centroids``
    clusters within that code's Voronoi cell. This captures the internal
    structure of each cell — a sample landing in the right cell but far
    from any centroid is suspicious.

    Args:
        vqvae: Trained VQ-VAE in eval mode.
        X_benign: Benign-only samples tensor on device.
        device: Computation device.
        num_embeddings: Codebook size K.
        n_centroids: Number of KMeans clusters per code.
        batch_size: Encoding batch size.

    Returns:
        Dict mapping codebook index to list of ``(centroid, count)`` tuples,
        where centroid is the cluster center and count is the number of
        samples in that cluster (density weight).
    """
    codebook_latents: Dict[int, list] = defaultdict(list)
    vqvae.eval()

    with torch.no_grad():
        for i in range(0, len(X_benign), batch_size):
            batch = X_benign[i : i + batch_size].to(device)
            z = vqvae.encode(batch)
            vq_out = vqvae.vq(z)
            indices = vq_out["encoding_indices"].squeeze().cpu().numpy()
            z_np = z.cpu().numpy()

            if z_np.ndim == 1:
                indices = [indices]
                z_np = [z_np]

            for j, idx in enumerate(indices):
                codebook_latents[int(idx)].append(z_np[j])

    smap: Dict[int, List[Tuple[np.ndarray, float]]] = {}
    for idx, latent_list in codebook_latents.items():
        if len(latent_list) > n_centroids:
            # Cap at 50k to keep KMeans tractable for populous codes.
            if len(latent_list) > 50000:
                sub_arr = np.array(latent_list)[
                    np.random.choice(len(latent_list), 50000, replace=False)
                ]
            else:
                sub_arr = np.array(latent_list)

            km = KMeans(n_clusters=n_centroids, random_state=SEED, n_init=1).fit(sub_arr)

            smap[idx] = [
                (
                    c,
                    # Scale count back to true population when subsampled.
                    # Floor at 0.1 to avoid division-by-zero in scoring.
                    max(
                        float((km.labels_ == i).sum() * (len(latent_list) / len(sub_arr))),
                        0.1,
                    ),
                )
                for i, c in enumerate(km.cluster_centers_)
            ]
        elif latent_list:
            # Too few samples for KMeans — single mean centroid.
            smap[idx] = [(np.mean(latent_list, axis=0), float(len(latent_list)))]
        else:
            smap[idx] = []

    return smap


def compute_geometric_scores(
    vqvae: VQVAE,
    X: torch.Tensor,
    smap: Dict[int, List[Tuple[np.ndarray, float]]],
    device: torch.device,
    num_embeddings: int = 1024,
    batch_size: int = 8192,
) -> np.ndarray:
    """Computes the Geometric Score (GEO) for a batch of samples.

    For each sample, finds its assigned codebook entry, then computes the
    density-weighted minimum distance from the sample's latent vector to
    the centroids of that code's semantic map cluster. Dense clusters
    (high count) produce lower scores — an adversarial sample landing in
    a sparse region is penalised.

    Higher score = further from benign manifold = more suspicious.

    Args:
        vqvae: Trained VQ-VAE in eval mode.
        X: Input samples to score, on device.
        smap: Pre-computed semantic map from ``build_semantic_map()``.
        device: Computation device.
        num_embeddings: Codebook size K.
        batch_size: Encoding batch size.

    Returns:
        Array of geometric scores, shape ``(len(X),)``.
    """
    scores: list = []
    vqvae.eval()

    max_centroids = max((len(v) for v in smap.values()), default=0)
    dim = list(smap.values())[0][0][0].shape[0] if smap and list(smap.values())[0] else 64

    # Packing thousands of immutable centroids into GPU tensors is model
    # setup, not per-request inference. Cache the packed representation on
    # the frozen VQ-VAE instance and reuse it across calls.
    cache_key = (
        id(smap),
        str(device),
        int(num_embeddings),
        int(max_centroids),
        int(dim),
    )
    cache = getattr(vqvae, "_semantic_tensor_cache", None)
    if cache is None or cache.get("key") != cache_key:
        smap_c_t = torch.zeros(
            (num_embeddings, max(1, max_centroids), dim), device=device
        )
        smap_count_t = torch.ones(
            (num_embeddings, max(1, max_centroids)), device=device
        )
        valid_mask = torch.zeros(
            (num_embeddings, max(1, max_centroids)),
            dtype=torch.bool,
            device=device,
        )
        for idx, centroids_for_code in smap.items():
            for i, (centroid, count) in enumerate(centroids_for_code):
                smap_c_t[idx, i] = torch.as_tensor(
                    centroid, dtype=torch.float32, device=device
                )
                smap_count_t[idx, i] = count
                valid_mask[idx, i] = True
        cache = {
            "key": cache_key,
            "centroids": smap_c_t,
            "counts": smap_count_t,
            "valid": valid_mask,
        }
        setattr(vqvae, "_semantic_tensor_cache", cache)
    else:
        smap_c_t = cache["centroids"]
        smap_count_t = cache["counts"]
        valid_mask = cache["valid"]

    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            batch = X[i : i + batch_size].to(device)
            z = vqvae.encode(batch)
            vq_out = vqvae.vq(z)
            indices = vq_out["encoding_indices"].squeeze(-1)

            if indices.ndim == 0:
                indices = indices.unsqueeze(0)

            # Gather the semantic map entries for each sample's assigned code.
            c_batch = smap_c_t[indices]
            count_batch = smap_count_t[indices]
            mask_batch = valid_mask[indices]

            # Density-weighted distance: ||z - centroid||^2 / count.
            # Dense clusters produce lower scores (more "normal").
            diff = z.unsqueeze(1) - c_batch
            sq_dist = diff.pow(2).sum(dim=-1)
            weighted_dist = sq_dist / (count_batch + 1e-6)
            # Mask invalid centroids with inf so they are never selected.
            weighted_dist = torch.where(mask_batch, weighted_dist, torch.inf)

            min_dist, _ = weighted_dist.min(dim=-1)
            scores.extend(min_dist.cpu().tolist())

    scores = np.array(scores)
    # Replace inf (codes with no semantic map entries) with 10x the
    # maximum finite score to ensure they are flagged as anomalous.
    finite_mask = np.isfinite(scores)
    if finite_mask.any() and (~finite_mask).any():
        max_score = scores[finite_mask].max()
        scores[~finite_mask] = max_score * 10.0

    return scores
