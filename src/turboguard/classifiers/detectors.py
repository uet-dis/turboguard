"""Detector wrappers shared by AE/VAE/VQ-VAE baseline experiments."""

from __future__ import annotations

import numpy as np
import torch
from sklearn.ensemble import IsolationForest


def reconstruction_errors(model, X: torch.Tensor, batch_size: int = 8192) -> np.ndarray:
    """Return per-sample MSE without retaining autograd graphs."""
    model.eval()
    errors = []
    with torch.no_grad():
        for start in range(0, len(X), batch_size):
            out = model(X[start : start + batch_size])
            errors.append(torch.mean((out["reconstruction"] - X[start : start + batch_size]) ** 2, dim=1).cpu().numpy())
    return np.concatenate(errors) if errors else np.empty(0, dtype=np.float32)


class ReconstructionDetector:
    """Thresholded reconstruction-only detector."""

    def __init__(self, percentile: float = 99.0):
        self.percentile = percentile
        self.threshold = np.inf

    def fit(self, benign_errors: np.ndarray) -> "ReconstructionDetector":
        self.threshold = float(np.percentile(benign_errors, self.percentile))
        return self

    def predict_from_errors(self, errors: np.ndarray) -> np.ndarray:
        return (np.asarray(errors) >= self.threshold).astype(int)


class LatentIsolationDetector:
    """Isolation Forest on continuous latent vectors."""

    def __init__(self, contamination: float = "auto", random_state: int = 42):
        self.model = IsolationForest(contamination=contamination, random_state=random_state, n_jobs=-1)
        self.threshold = 0.0

    def fit(self, benign_latent: np.ndarray) -> "LatentIsolationDetector":
        self.model.fit(benign_latent)
        self.threshold = float(np.percentile(self.model.decision_function(benign_latent), 1.0))
        return self

    def predict(self, latent: np.ndarray) -> np.ndarray:
        return (self.model.decision_function(latent) < self.threshold).astype(int)
