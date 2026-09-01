"""IEC-104 SCADA dataset handler."""

from typing import Any, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from datasets.registry import DatasetHandler, register_dataset


@register_dataset
class IEC104Handler(DatasetHandler):
    """IEC-104 SCADA dataset (76 features, binary label).

    Expects CICFlowMeter CSV output at
    ``<data_dir>/tests_cic_15/train_15_cicflow.csv``.

    Attributes:
        METADATA_COLS: Columns to drop before feature extraction.
    """

    def __init__(self) -> None:
        self._load_audit: dict[str, int] = {}

    METADATA_COLS = [
        "Flow ID",
        "Src IP",
        "Src Port",
        "Dst IP",
        "Dst Port",
        "Protocol",
        "Timestamp",
    ]

    def name(self) -> str:
        """Returns ``'iec104'``."""
        return "iec104"

    def input_dim(self) -> int:
        """Returns 76 features."""
        return 76

    def load(
        self, data_dir: str, scaler_type: str = "minmax", fit_indices: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray, Any]:
        """Loads IEC-104 from CICFlowMeter CSV.

        Args:
            data_dir: Path containing the CICFlowMeter output.
            scaler_type: ``"minmax"`` or ``"standard"``.

        Returns:
            Tuple of (X, y, scaler).
        """
        train_path = f"{data_dir}/tests_cic_15/train_15_cicflow.csv"
        df = pd.read_csv(train_path, low_memory=False)
        n_loaded = len(df)
        df = df.replace([np.inf, -np.inf], np.nan).dropna()
        n_after_cleaning = len(df)

        feature_cols = [c for c in df.columns if c not in self.METADATA_COLS]
        n_before_duplicates = len(df)
        df = df.drop_duplicates(subset=feature_cols)
        self._load_audit = {
            "n_rows_loaded": int(n_loaded),
            "n_rows_nonfinite": int(n_loaded - n_after_cleaning),
            "n_rows_after_cleaning": int(n_after_cleaning),
            "n_duplicate_rows_removed": int(n_before_duplicates - len(df)),
        }

        df["Label"] = df["Label"].astype(str).str.strip()
        y = (df["Label"] != "NORMAL").astype(np.int64).values
        X = df.drop(columns=self.METADATA_COLS + ["Label"]).values.astype(np.float32)
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
