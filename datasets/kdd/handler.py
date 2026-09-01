"""NSL-KDD dataset handler."""

from typing import Any, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from datasets.registry import DatasetHandler, register_dataset


@register_dataset
class KDDHandler(DatasetHandler):
    """NSL-KDD dataset (122 features, binary label).

    Expects ``train_processed.parquet`` in the data directory.
    """

    def __init__(self) -> None:
        self._load_audit: dict[str, int] = {}

    def name(self) -> str:
        """Returns ``'kdd'``."""
        return "kdd"

    def input_dim(self) -> int:
        """Returns 122 features."""
        return 122

    def load(
        self, data_dir: str, scaler_type: str = "minmax", fit_indices: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray, Any]:
        """Loads NSL-KDD from preprocessed parquet.

        Args:
            data_dir: Path containing ``train_processed.parquet``.
            scaler_type: ``"minmax"`` or ``"standard"``.

        Returns:
            Tuple of (X, y, scaler).
        """
        df = pd.read_parquet(f"{data_dir}/train_processed.parquet")
        n_loaded = len(df)

        X = df.drop(columns=["attack_class"]).values.astype(np.float32)
        y = (df["attack_class"] != "normal").astype(np.int64).values
        finite_rows = np.isfinite(X).all(axis=1)
        self._load_audit = {
            "n_rows_loaded": int(n_loaded),
            "n_rows_nonfinite": int((~finite_rows).sum()),
            "n_rows_after_cleaning": int(n_loaded),
            "n_duplicate_rows_removed": 0,
        }
        if fit_indices is not None and len(fit_indices) == 0:
            return X, y, None

        # Fit scaler on benign samples only to prevent data leakage.
        scaler = MinMaxScaler() if scaler_type == "minmax" else StandardScaler()
        fit_data = X[np.asarray(fit_indices, dtype=np.int64)] if fit_indices is not None else X[y == 0]
        scaler.fit(fit_data if len(fit_data) else X)
        X = scaler.transform(X).astype(np.float32)

        return X, y, scaler

    def load_audit(self) -> dict[str, int]:
        return dict(self._load_audit)
