"""CSE-CIC-IDS2018 dataset handler.

Loads the preprocessed parquet file (output of ``preprocess_cic2018.py``),
selects the 62 CICFlowMeter features, and applies scaling with binary
features (RST/PSH/ACK/ECE Flag Cnt) kept as 0/1.
"""

from typing import Any, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from datasets.registry import DatasetHandler, register_dataset

# 62 CICFlowMeter features used for classification.
FEATURES = [
    "Dst Port",
    "Flow Duration",
    "Tot Fwd Pkts",
    "Tot Bwd Pkts",
    "TotLen Fwd Pkts",
    "TotLen Bwd Pkts",
    "Fwd Pkt Len Max",
    "Fwd Pkt Len Min",
    "Fwd Pkt Len Mean",
    "Fwd Pkt Len Std",
    "Bwd Pkt Len Max",
    "Bwd Pkt Len Min",
    "Bwd Pkt Len Mean",
    "Bwd Pkt Len Std",
    "Flow Byts/s",
    "Flow Pkts/s",
    "Flow IAT Mean",
    "Flow IAT Std",
    "Flow IAT Max",
    "Flow IAT Min",
    "Fwd IAT Tot",
    "Fwd IAT Mean",
    "Fwd IAT Std",
    "Fwd IAT Max",
    "Fwd IAT Min",
    "Bwd IAT Tot",
    "Bwd IAT Mean",
    "Bwd IAT Std",
    "Bwd IAT Max",
    "Bwd IAT Min",
    "Fwd Header Len",
    "Bwd Header Len",
    "Fwd Pkts/s",
    "Bwd Pkts/s",
    "Pkt Len Min",
    "Pkt Len Max",
    "Pkt Len Mean",
    "Pkt Len Std",
    "Pkt Len Var",
    "RST Flag Cnt",
    "PSH Flag Cnt",
    "ACK Flag Cnt",
    "ECE Flag Cnt",
    "Down/Up Ratio",
    "Pkt Size Avg",
    "Fwd Seg Size Avg",
    "Bwd Seg Size Avg",
    "Subflow Fwd Pkts",
    "Subflow Fwd Byts",
    "Subflow Bwd Pkts",
    "Subflow Bwd Byts",
    "Init Fwd Win Byts",
    "Fwd Act Data Pkts",
    "Fwd Seg Size Min",
    "Active Mean",
    "Active Std",
    "Active Max",
    "Active Min",
    "Idle Mean",
    "Idle Std",
    "Idle Max",
    "Idle Min",
]

# These features are inherently binary (0/1) and must not be scaled,
# since MinMaxScaler would destroy their semantics on unseen data.
BINARY_FEATURES = {"RST Flag Cnt", "PSH Flag Cnt", "ACK Flag Cnt", "ECE Flag Cnt"}


@register_dataset
class CIC2018Handler(DatasetHandler):
    """CSE-CIC-IDS2018 dataset (62 features, binary label).

    Expects ``cic2018_clean.parquet`` in the data directory.
    """

    def __init__(self) -> None:
        self._load_audit: dict[str, int] = {}

    def name(self) -> str:
        """Returns ``'cic2018'``."""
        return "cic2018"

    def input_dim(self) -> int:
        """Returns 62 features."""
        return 62

    def load(
        self,
        data_dir: str,
        scaler_type: str = "minmax",
        fit_indices: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray, Any]:
        """Loads CIC-IDS2018 from preprocessed parquet.

        Binary flag features are excluded from scaling to preserve
        their 0/1 semantics. The scaler is fit on benign samples only
        to prevent data leakage.

        Args:
            data_dir: Path containing ``cic2018_clean.parquet``.
            scaler_type: ``"minmax"`` or ``"standard"``.

        Returns:
            Tuple of (X, y, scaler).
        """
        df = pd.read_parquet(f"{data_dir}/cic2018_clean.parquet")
        n_loaded = len(df)
        label_col = df["Label"].astype(str).str.strip()
        y = (label_col != "Benign").astype(np.int64).values
        X = df[FEATURES].values.astype(np.float32)
        finite_rows = np.isfinite(X).all(axis=1)
        self._load_audit = {
            "n_rows_loaded": int(n_loaded),
            "n_rows_nonfinite": int((~finite_rows).sum()),
            "n_rows_after_cleaning": int(n_loaded),
            "n_duplicate_rows_removed": 0,
        }
        if fit_indices is not None and len(fit_indices) == 0:
            return X, y, None

        # Scale only continuous features; leave binary flags untouched.
        binary_idx = [i for i, f in enumerate(FEATURES) if f in BINARY_FEATURES]
        scale_idx = [i for i in range(len(FEATURES)) if i not in binary_idx]

        scaler = MinMaxScaler() if scaler_type == "minmax" else StandardScaler()
        if fit_indices is not None:
            fit_data = X[np.asarray(fit_indices, dtype=np.int64)][:, scale_idx]
        else:
            benign = y == 0
            fit_data = X[benign][:, scale_idx] if benign.any() else X[:, scale_idx]
        scaler.fit(fit_data)
        X[:, scale_idx] = scaler.transform(X[:, scale_idx]).astype(np.float32)

        return X, y, scaler

    def load_audit(self) -> dict[str, int]:
        return dict(self._load_audit)
