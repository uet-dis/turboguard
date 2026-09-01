"""Dataset registry with auto-registration decorator.

Every dataset handler subclasses ``DatasetHandler`` and is decorated with
``@register_dataset`` so the framework can discover it by name at runtime.

The registry pattern decouples the TurboGuard library from any specific
dataset — all dataset-specific logic (file paths, column names, label
mappings, special preprocessing) lives in the handler subclass.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple, Type

import numpy as np

DATASET_REGISTRY: Dict[str, Type["DatasetHandler"]] = {}
"""Global registry mapping dataset names to handler classes."""


def register_dataset(cls: Type["DatasetHandler"]) -> Type["DatasetHandler"]:
    """Class decorator that registers a DatasetHandler.

    Instantiates the handler to read its ``name()`` and stores the class
    (not the instance) in the global registry.

    Args:
        cls: DatasetHandler subclass to register.

    Returns:
        The same class, unmodified.
    """
    instance = cls()
    DATASET_REGISTRY[instance.name()] = cls
    return cls


def get_handler(name: str) -> "DatasetHandler":
    """Retrieves an instantiated handler by its registry key.

    Args:
        name: Dataset identifier (e.g. "unsw", "kdd", "cic2018").

    Returns:
        A fresh instance of the corresponding DatasetHandler subclass.

    Raises:
        KeyError: If ``name`` is not registered.
    """
    if name not in DATASET_REGISTRY:
        available = ", ".join(sorted(DATASET_REGISTRY.keys()))
        raise KeyError(f"Unknown dataset '{name}'. Available: {available}")
    return DATASET_REGISTRY[name]()


class DatasetHandler(ABC):
    """Abstract interface that every dataset handler must implement.

    A handler encapsulates all dataset-specific logic: file discovery,
    column selection, label encoding, and scaler fitting. The TurboGuard
    library never touches dataset-specific details — it only calls these
    methods through the registry.
    """

    @abstractmethod
    def name(self) -> str:
        """Returns the short registry key (e.g. ``'unsw'``).

        Returns:
            Lowercase string identifier for CLI and directory naming.
        """

    @abstractmethod
    def input_dim(self) -> int:
        """Returns the number of features after preprocessing.

        Returns:
            Integer feature count (e.g. 42 for UNSW, 62 for CIC2018).
        """

    @abstractmethod
    def load(
        self,
        data_dir: str,
        scaler_type: str = "minmax",
    ) -> Tuple[np.ndarray, np.ndarray, Any]:
        """Loads raw data from disk and returns processed arrays.

        The scaler must be fit on benign samples only to prevent data
        leakage from attack samples into the feature normalisation.

        Args:
            data_dir: Path to the directory containing raw dataset files.
            scaler_type: Scaler to apply ("minmax" or "standard").

        Returns:
            Tuple of:
                - X: Feature matrix ``(N, D)``, dtype float32, already scaled.
                - y: Label vector ``(N,)``, dtype int64.
                  ``0`` = benign, ``>0`` = attack class index.
                - scaler: The fitted scaler object (for persistence).
        """

    def load_audit(self) -> Dict[str, Any]:
        """Return preprocessing counters collected by the last ``load`` call."""
        return {}
