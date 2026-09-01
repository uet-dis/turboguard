"""UNSW-NB15 dataset handler."""

from typing import Any, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from datasets.registry import DatasetHandler, register_dataset


@register_dataset
class UNSWHandler(DatasetHandler):
    """CIC-UNSW-NB15 dataset (76 features, binary label).

    Expects ``Data.csv`` and ``Label.csv`` in the data directory.
    """

    def __init__(self) -> None:
        self._load_audit: dict[str, int] = {}

    def name(self) -> str:
        """Returns ``'unsw'``."""
        return "unsw"

    def input_dim(self) -> int:
        """Returns 76 features."""
        return 76

    def load(
        self, data_dir: str, scaler_type: str = "minmax", fit_indices: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray, Any]:
        """Loads UNSW-NB15 from ``Data.csv`` + ``Label.csv``.

        Args:
            data_dir: Path containing ``Data.csv`` and ``Label.csv``.
            scaler_type: ``"minmax"`` or ``"standard"``.

        Returns:
            Tuple of (X, y, scaler).
        """
        df_data = pd.read_csv(f"{data_dir}/Data.csv")
        df_label = pd.read_csv(f"{data_dir}/Label.csv")

        df = pd.concat([df_data, df_label], axis=1)
        n_loaded = len(df)
        n_nonfinite = int(
            (~np.isfinite(df.drop(columns=["Label"]).to_numpy(dtype=np.float64)).all(axis=1)).sum()
        )
        df = df.replace([np.inf, -np.inf], np.nan).dropna()
        n_after_cleaning = len(df)
        n_before_duplicates = len(df)
        df = df.drop_duplicates()
        self._load_audit = {
            "n_rows_loaded": int(n_loaded),
            "n_rows_nonfinite": n_nonfinite,
            "n_rows_after_cleaning": int(n_after_cleaning),
            "n_duplicate_rows_removed": int(n_before_duplicates - len(df)),
        }

        X = df.drop(columns=["Label"]).values.astype(np.float32)
        y = df["Label"].values.astype(np.int64)
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
