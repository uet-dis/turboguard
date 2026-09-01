"""TurboGuard Adversarial Defense Filter.

A model-agnostic, attack-agnostic adversarial filter for network intrusion
detection systems (NIDS). Uses a VQ-VAE trained exclusively on benign traffic
to extract six complementary anomaly signals from the codebook topology, then
applies a two-stage detection pipeline: IsolationForest hard-drop followed by
a DNN grey-zone classifier.

Architecture::

    Input x
      → VQ-VAE (with dead code reset) → Reconstruct x̂
      → Extract 6 signals: [CTF, VMR, ENT, RE, GEO, CC]
      → IsolationForest: hard-drop obvious outliers
      → DNN on [x, x̂, |x−x̂|, signals]: classify grey-zone
      → Binary output: benign (0) / adversarial (1)

Signals:
    1. CTF: Codebook Topology Fingerprint — Mahalanobis distance in the PCA
       subspace of the full codebook distance profile. Captures global
       distributional shift that per-code metrics miss.
    2. VMR: Voronoi Margin Ratio — ratio of nearest-code distance to
       second-nearest. Low values indicate the sample sits deep inside a
       single Voronoi cell (normal); values near 1.0 indicate boundary
       ambiguity (suspicious).
    3. ENT: Codebook Distance Entropy — entropy of the softmax over all
       code distances. Benign inputs cluster tightly (low entropy);
       adversarial inputs spread mass across codes (high entropy).
    4. RE:  Reconstruction Error — per-sample MSE. The classic anomaly signal.
    5. GEO: Geometric Score — density-weighted distance to the nearest
       semantic map centroid. This is the backbone signal, contributing
       ~65% of DNN feature importance.
    6. CC:  Code-Conditional z-score — reconstruction error normalised by
       the code-specific mean and std. Detects anomalies that are subtle
       globally but extreme for their local codebook region.

References:
    [1] van den Oord et al., "Neural Discrete Representation Learning",
        NeurIPS 2017. DOI: 10.48550/arXiv.1711.00937
    [2] Zeghidour et al., "SoundStream: An End-to-End Neural Audio Codec",
        IEEE/ACM TASLP, 30, 495-507, 2021. DOI: 10.1109/TASLP.2021.3129994
"""

import time
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from rich.progress import (
    Progress,
    BarColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from turboguard.console import console
from turboguard.config import (
    BATCH_SIZE,
    DNN_MAX_EPOCHS,
    DNN_PATIENCE,
    ISO_CONTAMINATION,
    ISO_N_ESTIMATORS,
    LATENT_DIM,
    NUM_EMBEDDINGS,
)
from turboguard.core.geometry import build_semantic_map, compute_geometric_scores
from turboguard.core.if_score_contract import apply_if_score_contract
from turboguard.models.dnn import DNNClassifier
from turboguard.models.vqvae import VQVAE


SIGNAL_NAMES = ("CTF", "VMR", "ENT", "RE", "GEO", "CC")


def isolation_forest_reject(
    if_score_raw: np.ndarray, tau_if_raw: float
) -> np.ndarray:
    """Apply the one closed deployed IF score contract."""
    return apply_if_score_contract(if_score_raw, tau_if_raw).raw_if_reject


def isolation_forest_anomaly_score(if_score_raw: np.ndarray) -> np.ndarray:
    """Return the closed contract's dtype-preserving anomaly orientation."""
    raw = np.asarray(if_score_raw)
    zero = np.asarray(0, dtype=raw.dtype)
    return apply_if_score_contract(raw, zero).canonical_anomaly_score


class TurboGuard:
    """VQ-VAE-based adversarial defense filter.

    The filter is trained on benign traffic only (unsupervised for the VQ-VAE)
    and uses six codebook-derived signals to detect adversarial perturbations
    via a two-stage IsolationForest + DNN pipeline.

    Attributes:
        device: Torch device for all computations.
        vqvae: Trained VQ-VAE model.
        semantic_map: Per-code centroid clusters built from benign latent space.
    """

    def __init__(self, device: torch.device) -> None:
        """Initializes TurboGuard with empty model slots.

        Args:
            device: Torch device (cuda or cpu) for all computations.
        """
        self.device = device
        self.vqvae: Optional[VQVAE] = None
        self.semantic_map: Optional[dict] = None

        # CTF signal references: PCA projection matrix and Mahalanobis stats
        # fitted on benign codebook-distance profiles.
        self._pca_mean: Optional[np.ndarray] = None
        self._pca_components: Optional[np.ndarray] = None
        self._ctf_mean: Optional[np.ndarray] = None
        self._ctf_inv_cov: Optional[np.ndarray] = None

        # CC signal references: per-code reconstruction error (mu, sigma).
        # Codes with < 5 samples fall back to global stats to avoid
        # unstable z-scores from tiny sample sizes.
        self._code_stats: Dict[int, Tuple[float, float]] = {}
        self._global_mu: float = 0.0
        self._global_std: float = 1.0

        # Two-stage detection pipeline.
        self._iso_forest: Optional[IsolationForest] = None
        self._iso_threshold: float = 0.0
        self._dnn_threshold: float = 0.5
        self._dnn: Optional[DNNClassifier] = None
        self._scaler: Optional[StandardScaler] = None
        self._signal_names: tuple[str, ...] = SIGNAL_NAMES
        self._signal_tensor_cache: Optional[dict] = None

        # Hyperparameters set during fit().
        self._num_embeddings: int = NUM_EMBEDDINGS
        self._latent_dim: int = LATENT_DIM

    def fit(
        self,
        X_train: torch.Tensor,
        y_train: np.ndarray,
        input_dim: int,
        seed: int = 42,
        latent_dim: int = LATENT_DIM,
        num_embeddings: int = NUM_EMBEDDINGS,
        vqvae_epochs: int = 20,
        reset_interval: int = 3,
        dnn_batch_size: int = 2048,
    ) -> None:
        """Trains the full TurboGuard pipeline end-to-end.

        Pipeline stages:
            1. Train VQ-VAE on benign-only data with dead code replacement.
            2. Build semantic map (KMeans clustering per codebook index).
            3. Compute benign reference statistics for CTF and CC signals.
            4. Train IsolationForest + DNN grey-zone classifier on all data.

        Args:
            X_train: Training features tensor, already on ``self.device``.
            y_train: Label vector (0 = benign, >0 = attack class).
            input_dim: Number of input features (columns in X_train).
            seed: Random seed for reproducibility.
            latent_dim: VQ-VAE bottleneck dimension.
            num_embeddings: Codebook size K.
            vqvae_epochs: Number of VQ-VAE training epochs.
            reset_interval: Dead code replacement frequency (every N epochs).
        """
        torch.manual_seed(seed)
        np.random.seed(seed)
        self._num_embeddings = num_embeddings
        self._latent_dim = latent_dim

        X_benign = X_train[y_train == 0]
        self.vqvae = VQVAE(input_dim, latent_dim, num_embeddings).to(self.device)
        self._train_vqvae(X_benign, vqvae_epochs, reset_interval)

        self.semantic_map = build_semantic_map(
            self.vqvae,
            X_benign,
            self.device,
            num_embeddings=num_embeddings,
        )

        self._fit_signal_references(X_benign)
        self._fit_classifier(X_train, y_train, seed, dnn_batch_size)

    def _train_vqvae(
        self,
        X_benign: torch.Tensor,
        epochs: int = 20,
        reset_interval: int = 3,
    ) -> None:
        """Trains VQ-VAE with periodic dead code replacement.

        Dead codes (near-zero usage) create oversized Voronoi cells that
        hide low-epsilon adversarial perturbations. Replacing them with
        sampled encoder outputs forces uniform manifold coverage [2].

        The last 2 epochs skip resets so the codebook can stabilise before
        the semantic map is built.

        Args:
            X_benign: Benign-only training samples on device.
            epochs: Total training epochs.
            reset_interval: Replace dead codes every N epochs.
        """
        optimizer = optim.Adam(self.vqvae.parameters(), lr=1e-3)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        self.vqvae.train()

        # EMA usage tracker: decays old counts so recently-dead codes
        # are detected even if they were used early in training.
        code_usage = torch.zeros(self._num_embeddings, device=self.device)

        n_batches = len(X_benign) // BATCH_SIZE
        if len(X_benign) % BATCH_SIZE != 0:
            n_batches += 1

        with Progress(
            TextColumn("      ▸ [cyan]{task.fields[name]}[/cyan]"),
            TextColumn("[dim]Ep {task.fields[epoch]}/{task.fields[epochs]}[/dim]"),
            TextColumn("[dim]Batch {task.fields[batch]}/{task.fields[num_batches]}[/dim]"),
            BarColumn(),
            TextColumn("•"),
            TimeElapsedColumn(),
            TextColumn("• ETA:"),
            TimeRemainingColumn(),
            console=console,
        ) as progress:
            task = progress.add_task(
                "", total=epochs * n_batches, name="Training VQ-VAE", epoch=1, epochs=epochs, batch=0, num_batches=n_batches
            )
            for epoch in range(epochs):
                progress.update(task, epoch=epoch + 1)
                perm = torch.randperm(len(X_benign))
                epoch_usage = torch.zeros(self._num_embeddings, device=self.device)

                batch_counter = 1
                for i in range(0, len(X_benign), BATCH_SIZE):
                    batch = X_benign[perm[i : i + BATCH_SIZE]]
                    # BatchNorm requires >= 2 samples.
                    if len(batch) < 2:
                        batch_counter += 1
                        continue
                    optimizer.zero_grad()
                    out = self.vqvae(batch)
                    out["total_loss"].backward()
                    optimizer.step()

                    with torch.no_grad():
                        indices = out["encoding_indices"].squeeze()
                        epoch_usage.scatter_add_(
                            0,
                            indices,
                            torch.ones_like(indices, dtype=torch.float32),
                        )
                    
                    progress.update(task, advance=1, batch=batch_counter)
                    batch_counter += 1

                # Decay factor 0.9: forgets ~65% of usage after 10 epochs,
                # ensuring stale-but-alive codes don't mask dead regions.
                code_usage = code_usage * 0.9 + epoch_usage
                scheduler.step()

                # Skip last 2 epochs so codebook settles before semantic map.
                if (epoch + 1) % reset_interval == 0 and epoch < epochs - 2:
                    with torch.no_grad():
                        n_sample = min(len(X_benign), 5000)
                        z_sample = self.vqvae.encode(
                            X_benign[torch.randperm(len(X_benign))[:n_sample]]
                        )
                        self.vqvae.vq.reset_dead_codes(z_sample, code_usage)
                        # Floor counts at 1.0 so freshly-reset codes aren't
                        # immediately killed on the next check.
                        code_usage.clamp_(min=1.0)

        self.vqvae.eval()

    def _fit_signal_references(self, X_benign: torch.Tensor) -> None:
        """Fits benign reference statistics for CTF and CC signals.

        CTF reference: PCA on the K-dimensional codebook distance profile
        (one distance per code), then Mahalanobis statistics in the top-5
        principal component subspace.

        CC reference: per-code mean/std of reconstruction error, plus a
        global fallback for codes with < 5 samples.

        Args:
            X_benign: Benign-only training samples on device.
        """
        K = self._num_embeddings

        # Pass 1: streaming mean of codebook distance profiles.
        n_samples = 0
        dists_sum = torch.zeros(K, device=self.device, dtype=torch.float32)

        with torch.no_grad():
            for i in range(0, len(X_benign), 8192):
                b = X_benign[i : i + 8192].to(self.device)
                z = self.vqvae.encode(b)
                cb = self.vqvae.vq._embedding.weight
                # Expanded squared euclidean: ||z - c||^2
                d = z.pow(2).sum(1, keepdim=True) + cb.pow(2).sum(1) - 2 * z @ cb.t()
                dists_sum += d.sum(0)
                n_samples += len(d)

        pca_mean_t = dists_sum / max(n_samples, 1)

        # Pass 2: streaming covariance for PCA eigendecomposition.
        cov_t = torch.zeros((K, K), device=self.device, dtype=torch.float32)

        with torch.no_grad():
            for i in range(0, len(X_benign), 8192):
                b = X_benign[i : i + 8192].to(self.device)
                z = self.vqvae.encode(b)
                cb = self.vqvae.vq._embedding.weight
                d = z.pow(2).sum(1, keepdim=True) + cb.pow(2).sum(1) - 2 * z @ cb.t()
                centered_t = d - pca_mean_t
                cov_t += centered_t.T @ centered_t

        cov_t /= max(n_samples - 1, 1)
        # Force symmetry to avoid numerical drift breaking eigh().
        cov_t = (cov_t + cov_t.T) / 2.0

        # eigh returns ascending eigenvalues; take the last n_pc (largest).
        evals, evecs = torch.linalg.eigh(cov_t.cpu())
        evals = evals.to(self.device)
        evecs = evecs.to(self.device)

        n_pc = min(5, len(evals))
        # Flip to descending order, then transpose to row-major projection.
        vt_t = evecs[:, -n_pc:].flip(dims=[1]).T

        self._pca_mean = pca_mean_t.cpu().numpy()
        self._pca_components = vt_t.cpu().numpy()

        # Pass 3: Mahalanobis reference (mean + inverse covariance) in
        # the PCA subspace for the CTF signal.
        ctf_sum = torch.zeros(n_pc, device=self.device, dtype=torch.float32)

        with torch.no_grad():
            for i in range(0, len(X_benign), 8192):
                b = X_benign[i : i + 8192].to(self.device)
                z = self.vqvae.encode(b)
                cb = self.vqvae.vq._embedding.weight
                d = z.pow(2).sum(1, keepdim=True) + cb.pow(2).sum(1) - 2 * z @ cb.t()
                centered_t = d - pca_mean_t
                projected_t = centered_t @ vt_t.T
                ctf_sum += projected_t.sum(0)

        self._ctf_mean = (ctf_sum / max(n_samples, 1)).cpu().numpy()
        ctf_m_t = torch.tensor(self._ctf_mean, device=self.device, dtype=torch.float32)

        ctf_cov_t = torch.zeros((n_pc, n_pc), device=self.device, dtype=torch.float32)
        with torch.no_grad():
            for i in range(0, len(X_benign), 8192):
                b = X_benign[i : i + 8192].to(self.device)
                z = self.vqvae.encode(b)
                cb = self.vqvae.vq._embedding.weight
                d = z.pow(2).sum(1, keepdim=True) + cb.pow(2).sum(1) - 2 * z @ cb.t()
                centered_t = d - pca_mean_t
                projected_t = centered_t @ vt_t.T
                delta_t = projected_t - ctf_m_t
                ctf_cov_t += delta_t.T @ delta_t

        # Tikhonov regularisation (1e-6 * I) prevents singular inverse
        # when principal components have near-zero variance.
        cov = (ctf_cov_t / max(n_samples - 1, 1)).cpu().numpy() + np.eye(n_pc) * 1e-6
        self._ctf_inv_cov = np.linalg.inv(cov)

        # CC: per-code reconstruction error distribution.
        indices, recon_err = self._code_assignments_and_errors(X_benign)
        self._global_mu = float(recon_err.mean())
        self._global_std = max(float(recon_err.std()), 1e-8)
        self._code_stats = {}
        for k in np.unique(indices):
            mask = indices == k
            # Require >= 5 samples per code for stable statistics;
            # otherwise fall back to global mu/std.
            if mask.sum() >= 5:
                self._code_stats[k] = (
                    float(recon_err[mask].mean()),
                    max(float(recon_err[mask].std()), 1e-8),
                )

    def _fit_classifier(
        self,
        X_train: torch.Tensor,
        y_train: np.ndarray,
        seed: int,
        dnn_batch_size: int = 2048,
    ) -> None:
        """Trains the two-stage detection pipeline.

        Stage 1 (IsolationForest): Fitted on benign signals only. Samples
        with anomaly scores below the threshold are hard-dropped as
        adversarial — achieving ~100% detection for epsilon >= 0.05.

        Stage 2 (DNN): Trained on all data using [raw, reconstruction,
        absolute residual, normalised signals] as features. Catches
        subtle attacks that pass the IF filter (the "grey zone").

        Args:
            X_train: Full training set (benign + attack) on device.
            y_train: Label vector (0 = benign, >0 = attack).
            seed: Random seed for train/val split reproducibility.
        """
        signals = self.extract_signals(X_train, self._signal_names)

        # log1p(|s|) * sign(s) stabilises heavy-tailed signal distributions
        # (especially GEO and CTF) before StandardScaler normalisation.
        self._scaler = StandardScaler()
        signals_norm = self._scaler.fit_transform(np.log1p(np.abs(signals)) * np.sign(signals))

        # DNN sees: original features, reconstruction, residual magnitude,
        # and the selected normalized evidence signals.
        raw = X_train.cpu().numpy()
        recon = self._reconstruct(X_train)
        dnn_input = np.column_stack([raw, recon, np.abs(raw - recon), signals_norm])

        # The augmented matrix is large for CIC2018.  Keep it in host RAM and
        # transfer only minibatches to CUDA; otherwise the full matrix alone
        # consumes several GiB of VRAM before ``dnn_batch_size`` has any
        # effect.
        dnn_tensor = torch.as_tensor(dnn_input, dtype=torch.float32)
        y_binary = (y_train > 0).astype(int)
        y_tensor = torch.as_tensor(y_binary, dtype=torch.long)

        self._dnn = DNNClassifier(input_dim=dnn_input.shape[1]).to(self.device)

        # Inverse-frequency weighting to handle benign-heavy class imbalance.
        counts = np.bincount(y_binary)
        weights = torch.tensor(
            [1.0, counts[0] / max(counts[1], 1)],
            dtype=torch.float32,
            device=self.device,
        )
        criterion = nn.CrossEntropyLoss(weight=weights)
        opt = optim.Adam(self._dnn.parameters(), lr=1e-3, weight_decay=1e-5)
        lr_sched = optim.lr_scheduler.ReduceLROnPlateau(opt, "min", patience=5, factor=0.5)

        # 10% stratified validation split for early stopping.
        tr, vl = train_test_split(
            np.arange(len(dnn_tensor)),
            test_size=0.1,
            random_state=seed,
            stratify=y_binary,
        )
        loader = DataLoader(
            TensorDataset(dnn_tensor[tr], y_tensor[tr]),
            batch_size=dnn_batch_size,
            shuffle=True,
            pin_memory=self.device.type == "cuda",
        )

        best_val, best_state, patience = float("inf"), None, 0
        n_batches = len(loader)

        with Progress(
            TextColumn("      ▸ [cyan]{task.fields[name]}[/cyan]"),
            TextColumn("[dim]Ep {task.fields[epoch]}/{task.fields[epochs]}[/dim]"),
            TextColumn("[dim]Batch {task.fields[batch]}/{task.fields[num_batches]}[/dim]"),
            BarColumn(),
            TextColumn("•"),
            TimeElapsedColumn(),
            TextColumn("• ETA:"),
            TimeRemainingColumn(),
            TextColumn("• {task.fields[loss]}"),
            TextColumn("• val={task.fields[val_loss]}"),
            console=console,
        ) as progress:
            task = progress.add_task(
                "",
                total=DNN_MAX_EPOCHS * n_batches,
                name="Training DNN Grey-Zone",
                epoch=1,
                epochs=DNN_MAX_EPOCHS,
                batch=0,
                num_batches=n_batches,
                loss="--",
                val_loss="--",
            )
            for epoch in range(DNN_MAX_EPOCHS):
                progress.update(task, epoch=epoch + 1)
                self._dnn.train()
                epoch_loss = 0.0
                for i, (bx, by) in enumerate(loader, start=1):
                    bx = bx.to(self.device, non_blocking=True)
                    by = by.to(self.device, non_blocking=True)
                    opt.zero_grad()
                    loss = criterion(self._dnn(bx), by)
                    loss.backward()
                    opt.step()
                    epoch_loss += loss.item()
                    progress.update(task, advance=1, batch=i)
                lr_sched.step(epoch_loss)

                # Early stopping on validation loss.
                self._dnn.eval()
                with torch.no_grad():
                    val_total = 0.0
                    val_count = 0
                    for start in range(0, len(vl), dnn_batch_size):
                        batch_idx = vl[start : start + dnn_batch_size]
                        val_x = dnn_tensor[batch_idx].to(
                            self.device, non_blocking=True
                        )
                        val_y = y_tensor[batch_idx].to(
                            self.device, non_blocking=True
                        )
                        batch_loss = criterion(
                            self._dnn(val_x), val_y
                        )
                        n_batch = len(batch_idx)
                        val_total += float(batch_loss.item()) * n_batch
                        val_count += n_batch
                    val_loss = val_total / max(val_count, 1)

                avg_loss = epoch_loss / len(loader)
                progress.update(task, loss=f"{avg_loss:.4f}", val_loss=f"{val_loss:.4f}")

                if val_loss < best_val:
                    best_val = val_loss
                    best_state = {k: v.clone() for k, v in self._dnn.state_dict().items()}
                    patience = 0
                else:
                    patience += 1
                    if patience >= DNN_PATIENCE:
                        progress.update(task, completed=DNN_MAX_EPOCHS * n_batches)
                        break

        if best_state:
            self._dnn.load_state_dict(best_state)
        self._dnn.eval()

        # IsolationForest on benign signals only — learns the "normal"
        # decision boundary so that out-of-distribution signals are flagged.
        benign_mask = y_train == 0
        self._iso_forest = IsolationForest(
            n_estimators=ISO_N_ESTIMATORS,
            contamination=ISO_CONTAMINATION,
            random_state=seed,
            n_jobs=-1,
        )
        self._iso_forest.fit(signals[benign_mask])

    def calibrate(
        self,
        X_cal: torch.Tensor,
        y_cal: np.ndarray,
        fpr_budget: float = 1.5,
        percentiles: list[float] | None = None,
    ) -> dict:
        """Calibrates the IsolationForest threshold on held-out data.

        Sweeps percentile thresholds on benign calibration scores to find
        the most aggressive threshold that keeps FPR within budget. Returns
        the full sweep table so callers can inspect and justify the choice.

        Different percentile values represent different security trade-offs:
            - Lower P (e.g. P0.5): More aggressive — blocks more traffic,
              catches nearly all attacks, but risks higher false positives.
            - Higher P (e.g. P5): More permissive — fewer false alarms but
              may miss subtle adversarial perturbations.

        The sweep selects the lowest P (most aggressive) where the
        false positive rate stays within ``fpr_budget``.

        Args:
            X_cal: Calibration features tensor (Sector B) on device.
            y_cal: Calibration labels (0 = benign, >0 = attack).
            fpr_budget: Maximum acceptable false positive rate (%).
            percentiles: Percentile values to sweep. If ``None``, uses
                ``config.CALIBRATION_PERCENTILES``.

        Returns:
            Dict containing:
                - selected_percentile: The chosen percentile value.
                - selected_threshold: The IF score threshold.
                - fpr_budget: The FPR budget used.
                - sweep: List of dicts with keys {percentile, threshold,
                  FPR, ADR, within_budget, selected}.
        """
        from turboguard.config import CALIBRATION_PERCENTILES

        if percentiles is None:
            percentiles = CALIBRATION_PERCENTILES

        signals = self.extract_signals(X_cal, self._signal_names)
        benign_mask = y_cal == 0

        scores_ben = self._iso_forest.decision_function(signals[benign_mask])
        scores_atk = self._iso_forest.decision_function(signals[y_cal > 0])

        # Sweep from most aggressive (lowest P) to most permissive.
        # Lower IF scores = more anomalous. Threshold below which we block.
        sweep_results = []
        best_t = np.percentile(scores_ben, min(percentiles))
        best_adr = 0.0
        best_p = percentiles[0]

        for p in sorted(percentiles):
            t = np.percentile(scores_ben, p)
            fpr = float(isolation_forest_reject(scores_ben, t).mean()) * 100
            adr = float(isolation_forest_reject(scores_atk, t).mean()) * 100
            within_budget = fpr <= fpr_budget

            if within_budget and adr >= best_adr:
                best_t, best_adr, best_p = t, adr, p

            sweep_results.append(
                {
                    "percentile": p,
                    "threshold": float(t),
                    "FPR": round(fpr, 4),
                    "ADR": round(adr, 2),
                    "within_budget": within_budget,
                    "selected": False,  # marked below
                }
            )

        self._iso_threshold = best_t

        # Mark the selected row.
        for row in sweep_results:
            if row["percentile"] == best_p:
                row["selected"] = True

        return {
            "selected_percentile": best_p,
            "selected_threshold": float(best_t),
            "fpr_budget": fpr_budget,
            "sweep": sweep_results,
        }

    def calibration_sweep(
        self,
        X_cal: torch.Tensor,
        y_cal: np.ndarray,
        percentiles: list[float] | None = None,
    ) -> list[dict]:
        """Returns the full percentile sweep table without setting threshold.

        Use this for explainability — shows researchers exactly how each
        percentile value affects FPR and ADR, so the trade-off is
        transparent and justifiable.

        Args:
            X_cal: Calibration features tensor (Sector B) on device.
            y_cal: Calibration labels (0 = benign, >0 = attack).
            percentiles: Percentile values to sweep. If ``None``, uses
                a dense grid for detailed analysis:
                ``[0.01, 0.05, 0.1, 0.25, 0.5, 1, 1.5, 2, 3, 5, 10]``.

        Returns:
            List of dicts, each with keys:
                - percentile: The percentile value.
                - threshold: The corresponding IF score threshold.
                - FPR: False positive rate (%) at this threshold.
                - ADR: Attack detection rate (%) at this threshold.
                - n_benign_blocked: Number of benign samples blocked.
                - n_attack_blocked: Number of attack samples blocked.
        """
        if percentiles is None:
            percentiles = [
                0.01,
                0.05,
                0.1,
                0.25,
                0.5,
                1.0,
                1.5,
                2.0,
                3.0,
                5.0,
                10.0,
            ]

        signals = self.extract_signals(X_cal, self._signal_names)
        benign_mask = y_cal == 0
        attack_mask = y_cal > 0

        scores_ben = self._iso_forest.decision_function(signals[benign_mask])
        scores_atk = self._iso_forest.decision_function(signals[attack_mask])

        results = []
        for p in sorted(percentiles):
            t = np.percentile(scores_ben, p)
            ben_blocked = int(isolation_forest_reject(scores_ben, t).sum())
            atk_blocked = int(isolation_forest_reject(scores_atk, t).sum())
            fpr = ben_blocked / max(len(scores_ben), 1) * 100
            adr = atk_blocked / max(len(scores_atk), 1) * 100

            results.append(
                {
                    "percentile": p,
                    "threshold": float(t),
                    "FPR": round(fpr, 4),
                    "ADR": round(adr, 2),
                    "n_benign_blocked": ben_blocked,
                    "n_attack_blocked": atk_blocked,
                    "n_benign_total": len(scores_ben),
                    "n_attack_total": len(scores_atk),
                }
            )

        return results

    def predict(self, X: torch.Tensor) -> np.ndarray:
        """Runs the two-stage defense filter on input samples.

        Stage 1: IsolationForest hard-drops obvious outliers (high anomaly
        score → immediate label 1).
        Stage 2: DNN classifies the remaining "grey zone" samples that
        passed the IF filter.

        Args:
            X: Input features tensor on device.

        Returns:
            Binary predictions array: 0 = benign, 1 = adversarial.
        """
        return self.decision_details(X)["pred"]

    def _dnn_probabilities(
        self, X: torch.Tensor, signals: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Return class-1 probabilities for the supplied samples only."""
        if len(X) == 0:
            return np.empty(0, dtype=np.float32)
        if signals is None:
            signals = self.extract_signals(X, self._signal_names)
        self._dnn.eval()
        probabilities: list[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, len(X), BATCH_SIZE):
                stop = min(start + BATCH_SIZE, len(X))
                block = X[start:stop]
                block_signals = signals[start:stop]
                sig_norm = self._scaler.transform(
                    np.log1p(np.abs(block_signals)) * np.sign(block_signals)
                )
                raw = block.detach().cpu().numpy()
                recon = self._reconstruct(block)
                dnn_in = np.column_stack(
                    [raw, recon, np.abs(raw - recon), sig_norm]
                )
                dnn_t = torch.as_tensor(
                    dnn_in, dtype=torch.float32, device=self.device
                )
                probabilities.append(
                    torch.softmax(self._dnn(dnn_t), dim=1)[:, 1].cpu().numpy()
                )
        return np.concatenate(probabilities)

    def decision_details(
        self, X: torch.Tensor, *, score_all_dnn: bool = False
    ) -> Dict[str, np.ndarray]:
        """Return exact two-stage scores and decisions for every sample.

        ``if_score`` is the Isolation Forest normality score: lower values
        are more anomalous. ``if_anomaly_score`` is its sign-inverted form.
        By default the DNN is evaluated only for IF-passing flows, as defined
        by the deployed cascade.  Set ``score_all_dnn`` only for offline
        calibration, ranking, or attack analysis that needs every DNN score.
        """
        signals = self.extract_signals(X, self._signal_names)
        if_scores = self._iso_forest.decision_function(signals)
        if_contract = apply_if_score_contract(if_scores, self._iso_threshold)
        if_blocked = if_contract.raw_if_reject
        dnn_prob = np.full(len(X), np.nan, dtype=np.float32)
        if score_all_dnn:
            dnn_prob = self._dnn_probabilities(X, signals)
        else:
            dnn_index = np.flatnonzero(~if_blocked)
            if len(dnn_index):
                dnn_tensor_index = torch.as_tensor(dnn_index, device=X.device)
                dnn_prob[dnn_index] = self._dnn_probabilities(
                    X[dnn_tensor_index], signals[dnn_index]
                )
        dnn_blocked = (~if_blocked) & (dnn_prob >= self._dnn_threshold)
        pred = (if_blocked | dnn_blocked).astype(int)
        return {
            "signals": signals,
            "if_score": if_scores,
            "if_anomaly_score": if_contract.canonical_anomaly_score,
            "dnn_probability": dnn_prob,
            "if_blocked": if_blocked,
            "dnn_blocked": dnn_blocked,
            "pred": pred,
        }

    def profile_decision(self, X: torch.Tensor) -> tuple[Dict[str, np.ndarray], Dict[str, float]]:
        """Profile major inference stages without changing the decision path."""
        total_start = time.perf_counter()
        signal_start = time.perf_counter()
        signals = self.extract_signals(X, self._signal_names)
        signal_ms = (time.perf_counter() - signal_start) * 1000.0
        if_start = time.perf_counter()
        if_scores = self._iso_forest.decision_function(signals)
        if_contract = apply_if_score_contract(if_scores, self._iso_threshold)
        if_blocked = if_contract.raw_if_reject
        if_ms = (time.perf_counter() - if_start) * 1000.0
        dnn_start = time.perf_counter()
        dnn_prob = np.full(len(X), np.nan, dtype=np.float32)
        dnn_index = np.flatnonzero(~if_blocked)
        if len(dnn_index):
            dnn_tensor_index = torch.as_tensor(dnn_index, device=X.device)
            dnn_prob[dnn_index] = self._dnn_probabilities(
                X[dnn_tensor_index], signals[dnn_index]
            )
        dnn_blocked = (~if_blocked) & (dnn_prob >= self._dnn_threshold)
        dnn_ms = (time.perf_counter() - dnn_start) * 1000.0
        details = {
            "signals": signals,
            "if_score": if_scores,
            "if_anomaly_score": if_contract.canonical_anomaly_score,
            "dnn_probability": dnn_prob,
            "if_blocked": if_blocked,
            "dnn_blocked": dnn_blocked,
            "pred": (if_blocked | dnn_blocked).astype(int),
        }
        return details, {
            "signals_ms": signal_ms,
            "if_ms": if_ms,
            "dnn_ms": dnn_ms,
            "total_ms": (time.perf_counter() - total_start) * 1000.0,
        }

    def calibrate_cascade(
        self,
        X_cal: torch.Tensor,
        y_cal: np.ndarray,
        X_sel: Optional[torch.Tensor] = None,
        y_sel: Optional[np.ndarray] = None,
        target_fprs: tuple[float, ...] = (0.001, 0.005, 0.01),
        if_fpr_share: float = 0.5,
    ) -> Dict[str, object]:
        """Calibrate an active IF-to-DNN cascade under a total FPR budget.

        The IF share is fixed before evaluation and must be strictly between
        zero and one.  This prevents a validation tie-break from disabling
        the first path, while DNN calibration consumes only the residual FPR
        budget among IF-passing benign flows.
        """
        if not 0.0 < if_fpr_share < 1.0:
            raise ValueError("if_fpr_share must be strictly between 0 and 1")

        cal = self.decision_details(X_cal, score_all_dnn=True)
        y_cal = np.asarray(y_cal)
        benign = y_cal == 0
        if_scores = np.asarray(cal["if_score"])
        dnn_probs = np.asarray(cal["dnn_probability"])
        if not benign.any():
            raise ValueError("Calibration requires benign samples")

        def evaluate(t_if: float, t_dnn: float, details: Dict[str, np.ndarray], labels: np.ndarray) -> dict:
            blocked_if = isolation_forest_reject(details["if_score"], t_if)
            blocked_dnn = (~blocked_if) & (details["dnn_probability"] >= t_dnn)
            pred = blocked_if | blocked_dnn
            b = labels == 0
            a = labels > 0
            return {
                "fpr": float(pred[b].mean()) if b.any() else 0.0,
                "adr": float(pred[a].mean()) if a.any() else 0.0,
                "if_rate": float(blocked_if[b].mean()) if b.any() else 0.0,
                "dnn_rate": float(blocked_dnn[b].mean()) if b.any() else 0.0,
                "dnn_forward_rate": float((~blocked_if).mean()),
            }

        sel = (
            self.decision_details(X_sel, score_all_dnn=True)
            if X_sel is not None else None
        )
        selected: Dict[str, dict] = {}
        benign_scores = np.sort(if_scores[benign])
        for alpha in target_fprs:
            if_budget = alpha * if_fpr_share
            allowed_if = min(
                int(np.floor(if_budget * len(benign_scores))),
                len(benign_scores) - 1,
            )
            tau_if = float(benign_scores[allowed_if])
            if_pass_benign = benign & ~isolation_forest_reject(if_scores, tau_if)
            observed_if_fpr = float((~if_pass_benign & benign).mean())
            remaining_budget = alpha - observed_if_fpr
            pass_probs = dnn_probs[if_pass_benign]
            allowed_dnn = int(np.floor(max(remaining_budget, 0.0) * len(benign)))
            if not len(pass_probs) or allowed_dnn <= 0:
                tau_dnn = np.inf
            elif allowed_dnn >= len(pass_probs):
                tau_dnn = -np.inf
            else:
                tau_dnn = float(np.nextafter(np.sort(pass_probs)[::-1][allowed_dnn - 1], np.inf))

            cal_eval = evaluate(tau_if, tau_dnn, cal, y_cal)
            if cal_eval["fpr"] > alpha + 1e-12:
                raise RuntimeError(f"Cascade calibration exceeded target FPR={alpha}")
            if sel is not None and y_sel is not None:
                sel_eval = evaluate(tau_if, tau_dnn, sel, np.asarray(y_sel))
            else:
                sel_eval = cal_eval
            selected[str(alpha)] = {
                "tau_if": tau_if,
                "tau_dnn": tau_dnn,
                "if_fraction": if_fpr_share,
                "calibration": cal_eval,
                "selection": sel_eval,
                "objective_adr": sel_eval["adr"],
            }

        default_key = "0.01" if "0.01" in selected else str(target_fprs[0])
        default = selected[default_key]
        self._iso_threshold = float(default["tau_if"])
        self._dnn_threshold = float(default["tau_dnn"])
        sweep = []
        for alpha_key, point in selected.items():
            sweep.append({
                "target_fpr": float(alpha_key) * 100.0,
                "percentile": point["if_fraction"] * 100.0,
                "threshold": point["tau_if"],
                "FPR": point["calibration"]["fpr"] * 100.0,
                "ADR": point["calibration"]["adr"] * 100.0,
                "selection_FPR": point["selection"]["fpr"] * 100.0,
                "selection_ADR": point["selection"]["adr"] * 100.0,
                "if_FPR": point["calibration"]["if_rate"] * 100.0,
                "dnn_FPR": point["calibration"]["dnn_rate"] * 100.0,
                "dnn_forward_rate": point["calibration"]["dnn_forward_rate"] * 100.0,
                "within_budget": True,
                "selected": alpha_key == default_key,
                "tau_dnn": point["tau_dnn"],
            })
        return {
            "calibration_policy": "fixed_if_budget_cascade",
            "selected_percentile": float(default["if_fraction"] * 100.0),
            "selected_threshold": float(default["tau_if"]),
            "fpr_budget": float(default_key) * 100.0,
            "target_fprs": list(target_fprs),
            "if_fpr_share": if_fpr_share,
            "operating_points": selected,
            "selected_default": default_key,
            "sweep": sweep,
        }

    def calibrate_joint(
        self,
        X_cal: torch.Tensor,
        y_cal: np.ndarray,
        X_sel: Optional[torch.Tensor] = None,
        y_sel: Optional[np.ndarray] = None,
        target_fprs: tuple[float, ...] = (0.001, 0.005, 0.01),
        if_budget_grid: tuple[float, ...] = tuple(np.linspace(0.0, 1.0, 11)),
    ) -> Dict[str, object]:
        """Jointly calibrate IF and DNN thresholds at target FPRs.

        Thresholds are selected on ``X_cal`` and optionally ranked by attack
        detection on ``X_sel``. The returned operating points are deployment
        thresholds and must be frozen before locked-test evaluation.
        """
        cal = self.decision_details(X_cal)
        y_cal = np.asarray(y_cal)
        benign = y_cal == 0
        if_scores = cal["if_score"]
        dnn_probs = cal["dnn_probability"]
        def evaluate(t_if: float, t_dnn: float, details: Dict[str, np.ndarray], labels: np.ndarray):
            blocked_if = isolation_forest_reject(details["if_score"], t_if)
            blocked_dnn = (~blocked_if) & (details["dnn_probability"] >= t_dnn)
            pred = (blocked_if | blocked_dnn).astype(int)
            b = labels == 0
            a = labels > 0
            return {
                "fpr": float(pred[b].mean()) if b.any() else 0.0,
                "adr": float(pred[a].mean()) if a.any() else 0.0,
                "if_rate": float(blocked_if[b].mean()) if b.any() else 0.0,
            }

        selected: Dict[str, dict] = {}
        sel_details = self.decision_details(X_sel) if X_sel is not None else None
        for alpha in target_fprs:
            candidates = []
            for if_fraction in if_budget_grid:
                if_fraction = float(if_fraction)
                if_budget = alpha * if_fraction
                benign_if_scores = if_scores[benign]
                if if_budget <= 0:
                    t_if = -np.inf
                else:
                    # Use a conservative empirical order statistic. ``higher``
                    # quantiles can include a tied block of scores and exceed
                    # the requested FPR at very small budgets.
                    allowed_if = min(int(np.floor(if_budget * len(benign_if_scores))), len(benign_if_scores) - 1)
                    sorted_if = np.sort(benign_if_scores)
                    t_if = float(sorted_if[allowed_if])
                if_pass_benign = benign & ~isolation_forest_reject(if_scores, t_if)
                dnn_budget = alpha - float((~if_pass_benign & benign).mean())
                pass_probs = dnn_probs[if_pass_benign]
                if not len(pass_probs) or dnn_budget <= 0:
                    t_dnn = np.inf
                else:
                    # Bound the total number of DNN blocks directly rather
                    # than converting to a conditional quantile. This remains
                    # valid when IF and DNN scores contain ties.
                    allowed_dnn = int(np.floor(dnn_budget * len(benign)))
                    if allowed_dnn <= 0:
                        t_dnn = np.inf
                    elif allowed_dnn >= len(pass_probs):
                        t_dnn = -np.inf
                    else:
                        sorted_probs = np.sort(pass_probs)[::-1]
                        t_dnn = float(np.nextafter(sorted_probs[allowed_dnn - 1], np.inf))
                cal_eval = evaluate(t_if, t_dnn, cal, y_cal)
                if cal_eval["fpr"] <= alpha + 1e-12:
                    if sel_details is not None and y_sel is not None:
                        sel_eval = evaluate(t_if, t_dnn, sel_details, np.asarray(y_sel))
                        objective = sel_eval["adr"]
                    else:
                        sel_eval = cal_eval
                        objective = cal_eval["adr"]
                    candidates.append({
                        "tau_if": t_if,
                        "tau_dnn": t_dnn,
                        "if_fraction": if_fraction,
                        "calibration": cal_eval,
                        "selection": sel_eval,
                        "objective_adr": objective,
                    })
            if not candidates:
                raise RuntimeError(f"No joint thresholds satisfy target FPR={alpha}")
            best = max(candidates, key=lambda row: (row["objective_adr"], -row["if_fraction"]))
            selected[str(alpha)] = {"selected": best, "candidates": candidates}

        # Keep the 1% operating point as the default deployment point when present.
        default_key = "0.01" if "0.01" in selected else str(target_fprs[0])
        default = selected[default_key]["selected"]
        self._iso_threshold = float(default["tau_if"])
        self._dnn_threshold = float(default["tau_dnn"])
        summary_rows = []
        for alpha_key, result in selected.items():
            for candidate in result["candidates"]:
                summary_rows.append({
                    "target_fpr": float(alpha_key) * 100.0,
                    "percentile": candidate["if_fraction"] * 100.0,
                    "threshold": float(candidate["tau_if"]),
                    "FPR": candidate["calibration"]["fpr"] * 100.0,
                    "ADR": candidate["calibration"]["adr"] * 100.0,
                    "selection_FPR": candidate["selection"]["fpr"] * 100.0,
                    "selection_ADR": candidate["selection"]["adr"] * 100.0,
                    "within_budget": candidate["calibration"]["fpr"] <= float(alpha_key),
                    "selected": candidate is result["selected"],
                    "tau_dnn": float(candidate["tau_dnn"]),
                })
        return {
            "selected_percentile": float(default["if_fraction"] * 100.0),
            "selected_threshold": float(default["tau_if"]),
            "fpr_budget": float(default_key) * 100.0,
            "sweep": summary_rows,
            "target_fprs": list(target_fprs),
            "if_budget_grid": list(if_budget_grid),
            "operating_points": selected,
            "selected_default": default_key,
        }

    def fit_isolation_forest_reference(self, X_benign: torch.Tensor, seed: int = 42) -> None:
        """Fit the IF reference distribution on verified-benign ``B_fit``."""
        signals = self.extract_signals(X_benign, self._signal_names)
        self._iso_forest = IsolationForest(
            n_estimators=ISO_N_ESTIMATORS,
            contamination=ISO_CONTAMINATION,
            random_state=seed,
            n_jobs=-1,
        )
        self._iso_forest.fit(signals)

    def set_signal_subset(self, signal_names: tuple[str, ...] | list[str]) -> None:
        """Set the evidence columns used by subsequent classifier fitting."""
        names = tuple(signal_names)
        unknown = set(names) - set(SIGNAL_NAMES)
        if unknown or not names:
            raise ValueError(f"Invalid signal subset: {sorted(unknown)}")
        self._signal_names = names

    def extract_signal_dict(self, X: torch.Tensor) -> Dict[str, np.ndarray]:
        """Return all six signals keyed by stable signal name."""
        values = self.extract_signals(X, SIGNAL_NAMES)
        return {name: values[:, i] for i, name in enumerate(SIGNAL_NAMES)}

    def extract_signals(
        self,
        X: torch.Tensor,
        signal_names: Optional[tuple[str, ...] | list[str]] = None,
    ) -> np.ndarray:
        """Extracts the 6 anomaly signals from input samples.

        Signals are computed in batch on GPU for efficiency. The codebook
        distance matrix ``d[i, k] = ||z_i - c_k||^2`` is the foundation
        for CTF, VMR, and ENT; it is computed once per batch and reused.

        Args:
            X: Input features tensor on device.

        Returns:
            Signal matrix with columns ordered according to ``signal_names``.
        """
        geo = compute_geometric_scores(
            self.vqvae,
            X,
            self.semantic_map,
            self.device,
            num_embeddings=self._num_embeddings,
        )

        cache_key = (
            id(self._pca_mean),
            id(self._pca_components),
            id(self._ctf_mean),
            id(self._ctf_inv_cov),
            str(self.device),
        )
        if (
            self._signal_tensor_cache is None
            or self._signal_tensor_cache.get("key") != cache_key
        ):
            self._signal_tensor_cache = {
                "key": cache_key,
                "pca_mean": torch.as_tensor(
                    self._pca_mean,
                    device=self.device,
                    dtype=torch.float32,
                ),
                "pca_components": torch.as_tensor(
                    self._pca_components,
                    device=self.device,
                    dtype=torch.float32,
                ),
                "ctf_mean": torch.as_tensor(
                    self._ctf_mean,
                    device=self.device,
                    dtype=torch.float32,
                ),
                "ctf_inv_cov": torch.as_tensor(
                    self._ctf_inv_cov,
                    device=self.device,
                    dtype=torch.float32,
                ),
            }
        pca_m_t = self._signal_tensor_cache["pca_mean"]
        pca_c_t = self._signal_tensor_cache["pca_components"]
        ctf_m_t = self._signal_tensor_cache["ctf_mean"]
        ctf_inv_t = self._signal_tensor_cache["ctf_inv_cov"]

        ctf_list, vmr_list, ent_list = [], [], []

        with torch.no_grad():
            for i in range(0, len(X), 8192):
                b = X[i : i + 8192].to(self.device)
                z = self.vqvae.encode(b)
                cb = self.vqvae.vq._embedding.weight
                # d[i, k] = ||z_i - c_k||^2 via expanded form.
                d = z.pow(2).sum(1, keepdim=True) + cb.pow(2).sum(1) - 2 * z @ cb.t()

                # CTF: project distance profile into PCA subspace, then
                # compute Mahalanobis distance from benign reference.
                centered_t = d - pca_m_t
                projected_t = centered_t @ pca_c_t.T
                delta_t = projected_t - ctf_m_t
                batch_ctf = torch.sqrt(
                    torch.clamp(
                        torch.sum((delta_t @ ctf_inv_t) * delta_t, dim=1),
                        min=0,
                    )
                )
                ctf_list.append(batch_ctf.cpu().numpy())

                # VMR: ratio of nearest to second-nearest code distance.
                # Near 1.0 → decision boundary → suspicious.
                sorted_t, _ = torch.sort(d, dim=1)
                batch_vmr = sorted_t[:, 0] / (sorted_t[:, 1] + 1e-10)
                vmr_list.append(batch_vmr.cpu().numpy())

                # ENT: softmax entropy over code distances. Subtract max
                # for numerical stability (log-sum-exp trick).
                neg_t = -d
                neg_t -= neg_t.max(dim=1, keepdim=True).values
                exp_neg_t = torch.exp(neg_t)
                probs_t = exp_neg_t / exp_neg_t.sum(dim=1, keepdim=True)
                batch_ent = -torch.sum(probs_t * torch.log(probs_t + 1e-10), dim=1)
                ent_list.append(batch_ent.cpu().numpy())

        ctf = np.concatenate(ctf_list)
        vmr = np.concatenate(vmr_list)
        ent = np.concatenate(ent_list)

        # RE (reconstruction error) and CC (code-conditional z-score).
        indices, recon_err = self._code_assignments_and_errors(X)
        cc = self._code_conditional_zscore(indices, recon_err)

        all_signals = np.column_stack([ctf, vmr, ent, recon_err, geo, cc])
        names = tuple(signal_names) if signal_names is not None else SIGNAL_NAMES
        indices = [SIGNAL_NAMES.index(name) for name in names]
        return all_signals[:, indices]

    def _reconstruct(self, X: torch.Tensor) -> np.ndarray:
        """Computes VQ-VAE reconstructions as a numpy array.

        Args:
            X: Input tensor on device.

        Returns:
            Reconstructed features array of shape ``(N, D)``.
        """
        self.vqvae.eval()
        blocks = []
        with torch.no_grad():
            for i in range(0, len(X), BATCH_SIZE):
                b = X[i : i + BATCH_SIZE].to(self.device)
                blocks.append(self.vqvae(b)["reconstruction"].cpu().numpy())
        return np.concatenate(blocks)

    def _code_assignments_and_errors(self, X: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
        """Computes nearest code indices and per-sample reconstruction error.

        Args:
            X: Input tensor on device.

        Returns:
            Tuple of (code_indices, mse_per_sample), both numpy arrays.
        """
        self.vqvae.eval()
        cb = self.vqvae.vq._embedding.weight
        all_idx, all_err = [], []
        with torch.no_grad():
            for i in range(0, len(X), BATCH_SIZE):
                b = X[i : i + BATCH_SIZE].to(self.device)
                z = self.vqvae.encode(b)
                d = z.pow(2).sum(1, keepdim=True) + cb.pow(2).sum(1) - 2 * z @ cb.t()
                all_idx.append(d.argmin(1).cpu().numpy())
                recon = self.vqvae(b)["reconstruction"]
                all_err.append(F.mse_loss(recon, b, reduction="none").mean(1).cpu().numpy())
        return np.concatenate(all_idx), np.concatenate(all_err)

    def _code_conditional_zscore(self, indices: np.ndarray, recon_err: np.ndarray) -> np.ndarray:
        """Computes per-code z-score of reconstruction error.

        For each sample, normalises its reconstruction error by the mean
        and std of the codebook entry it was assigned to. This detects
        samples that are normal globally but extreme for their local region.

        Args:
            indices: Nearest code index per sample.
            recon_err: Reconstruction MSE per sample.

        Returns:
            Array of z-scores, shape ``(N,)``.
        """
        mu = np.full(len(indices), self._global_mu)
        std = np.full(len(indices), self._global_std)
        for k, (k_mu, k_std) in self._code_stats.items():
            mask = indices == k
            mu[mask] = k_mu
            std[mask] = k_std
        return (recon_err - mu) / std
