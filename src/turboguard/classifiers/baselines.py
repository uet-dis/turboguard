"""Baseline classifiers: XGBoost and DNN.

These serve as reference models that do NOT use TurboGuard's geometric
filter — purely standard binary classification for comparison.
"""

from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from xgboost import XGBClassifier
from rich.progress import (
    Progress,
    BarColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
    MofNCompleteColumn,
)

from turboguard.config import SEED
from turboguard.console import console
from turboguard.models.dnn import DNNClassifier


class BaselineXGB:
    """Baseline XGBoost binary classifier (benign=0, attack=1).

    Attributes:
        clf: Underlying XGBClassifier instance (set after ``fit()``).
    """

    def __init__(self) -> None:
        """Initializes with no trained model."""
        self.clf: Optional[XGBClassifier] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Trains XGBoost with automatic class weighting.

        The ``scale_pos_weight`` parameter is set to the ratio of benign
        to attack samples to handle class imbalance without resampling.

        Args:
            X: Training features array of shape ``(N, D)``.
            y: Labels (0 = benign, >0 = attack).
        """
        y_bin = (y > 0).astype(int)
        ratio = float(np.sum(y_bin == 0)) / max(np.sum(y_bin == 1), 1)
        self.clf = XGBClassifier(
            random_state=SEED,
            eval_metric="logloss",
            scale_pos_weight=ratio,
            n_estimators=100,
        )
        with console.status("      [cyan]Fitting XGBoost trees...", spinner="dots"):
            self.clf.fit(X, y_bin)

    def predict(self, X: torch.Tensor) -> np.ndarray:
        """Predicts binary labels from a tensor input.

        Accepts tensor for API consistency with DNN-based classifiers.

        Args:
            X: Input features tensor.

        Returns:
            Binary predictions array (0 or 1).
        """
        return self.clf.predict(X.cpu().numpy()).astype(int)


class BaselineDNN:
    """Baseline DNN binary classifier.

    Attributes:
        device: Torch device for computation.
        dnn: Underlying DNNClassifier instance (set after ``fit()``).
    """

    def __init__(self, device: torch.device) -> None:
        """Initializes with device but no trained model.

        Args:
            device: Torch device (cuda or cpu).
        """
        self.device = device
        self.dnn: Optional[DNNClassifier] = None

    def fit(self, X: torch.Tensor, y: np.ndarray, epochs: int = 30) -> None:
        """Trains the DNN with class weighting and LR scheduling.

        Uses inverse-frequency class weights (capped at 10x) and
        ReduceLROnPlateau scheduler to handle class imbalance and
        convergence plateaus.

        Args:
            X: Training features tensor on device.
            y: Labels (0 = benign, >0 = attack).
            epochs: Number of training epochs.
        """
        torch.manual_seed(SEED)
        self.dnn = DNNClassifier(input_dim=X.shape[1]).to(self.device)
        self.dnn.train()

        y_bin = (y > 0).astype(int)
        counts = np.bincount(y_bin)
        # Cap at 10x to prevent extreme gradients from rare-class batches.
        pos_weight = min(counts[0] / max(counts[1], 1), 10.0)

        weight = torch.tensor([1.0, pos_weight], dtype=torch.float32, device=self.device)
        criterion = nn.CrossEntropyLoss(weight=weight)
        y_t = torch.tensor(y_bin, dtype=torch.long, device=self.device)

        optimizer = optim.Adam(self.dnn.parameters(), lr=1e-3, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", patience=3, factor=0.5)
        loader = DataLoader(TensorDataset(X, y_t), batch_size=512, shuffle=True)

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
            console=console,
        ) as progress:
            task = progress.add_task(
                "",
                total=epochs * n_batches,
                name="Training Baseline DNN",
                epoch=1,
                epochs=epochs,
                batch=0,
                num_batches=n_batches,
                loss="loss=--",
            )
            for epoch in range(epochs):
                progress.update(task, epoch=epoch + 1)
                total_loss = 0.0
                for i, (bx, by) in enumerate(loader, start=1):
                    optimizer.zero_grad()
                    loss = criterion(self.dnn(bx), by)
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
                    progress.update(task, advance=1, batch=i)
                scheduler.step(total_loss)
                avg_loss = total_loss / len(loader)
                progress.update(task, loss=f"loss={avg_loss:.4f}")

        self.dnn.eval()

    def predict(self, X: torch.Tensor) -> np.ndarray:
        """Predicts binary labels.

        Args:
            X: Input features tensor on device.

        Returns:
            Binary predictions array (0 or 1).
        """
        self.dnn.eval()
        with torch.no_grad():
            return torch.argmax(self.dnn(X), dim=1).cpu().numpy()
