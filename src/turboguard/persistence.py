"""Run persistence — save and load pipeline artifacts.

Every pipeline step saves its output to a timestamped directory under
``results/<dataset>/<unix_epoch>_<command>/`` so that results are fully
reproducible and traceable.

Each run directory contains:
    - ``config.json``: Task-specific hyperparameters and provenance links.
    - ``metadata.json``: Full runtime snapshot (all config defaults, CLI
      args, system info) so anyone can reproduce the run exactly.
    - ``models/``: Saved model weights, scalers, etc.
"""

import json
import os
import platform
import sys
import time
from argparse import Namespace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np


def _json_safe(obj: Any) -> Any:
    """Recursively converts non-serialisable objects for JSON.

    Handles numpy scalars/arrays, pathlib Paths, Namespace, sets,
    and bytes.

    Args:
        obj: Any Python object.

    Returns:
        A JSON-serialisable equivalent.
    """
    if isinstance(obj, (np.floating, np.integer)):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, Namespace):
        return _json_safe(vars(obj))
    if callable(obj):
        module = getattr(obj, "__module__", "")
        name = getattr(obj, "__qualname__", getattr(obj, "__name__", type(obj).__name__))
        return f"{module}.{name}" if module else name
    if isinstance(obj, set):
        return sorted(obj)
    if isinstance(obj, bytes):
        return obj.decode("utf-8", errors="replace")
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    return obj


def create_run_dir(base: Path, dataset: str, command: str) -> Path:
    """Creates a timestamped run directory.

    Format: ``<base>/<dataset>/<unix_epoch>_<command>/``

    Also creates a ``models/`` subdirectory for saving model artifacts.

    Args:
        base: Root results directory (e.g. ``Path("results")``).
        dataset: Dataset name (e.g. "unsw", "cic2018").
        command: Pipeline step name (e.g. "prepare", "turboguard").

    Returns:
        Path to the created run directory.
    """
    ts = int(time.time())
    run_dir = base / dataset / f"{ts}_{command}"
    (run_dir / "models").mkdir(parents=True, exist_ok=True)
    return run_dir


def save_config(run_dir: Path, config: Dict[str, Any]) -> None:
    """Writes ``config.json`` with numpy-safe serialisation.

    This saves *task-specific* parameters (what was run, with what
    hyperparameters, linking to which source runs). For the full
    runtime snapshot, use ``save_run_metadata()``.

    Args:
        run_dir: Path to the run directory.
        config: Dict of hyperparameters and metadata to save.
    """
    with open(run_dir / "config.json", "w") as f:
        json.dump(_json_safe(config), f, indent=2)


def load_config(run_dir: Path) -> Dict[str, Any]:
    """Reads ``config.json`` from a run directory.

    Args:
        run_dir: Path to the run directory.

    Returns:
        Dict of saved configuration values.
    """
    with open(Path(run_dir) / "config.json") as f:
        return json.load(f)


def save_run_metadata(
    run_dir: Path,
    args: Optional[Namespace] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    """Saves a comprehensive ``metadata.json`` for full reproducibility.

    Captures everything needed to reproduce a run:
        - All CLI arguments (including defaults that weren't overridden).
        - All TurboGuard config defaults at the time of execution.
        - System information (Python, torch, CUDA, OS, hostname).
        - Timestamps (UTC ISO-8601 and Unix epoch).

    Args:
        run_dir: Path to the run directory.
        args: Parsed argparse Namespace with all CLI arguments.
        extra: Optional additional key-value pairs to include.
    """
    ts_epoch = int(time.time())
    ts_iso = datetime.now(timezone.utc).isoformat()

    meta: Dict[str, Any] = {
        "timestamp_utc": ts_iso,
        "timestamp_epoch": ts_epoch,
        "run_dir": str(run_dir),
    }

    # CLI arguments — includes both explicit and default values.
    if args is not None:
        meta["cli_args"] = _json_safe(vars(args))

    # All framework config defaults at runtime.
    try:
        from turboguard import config as tg_config

        config_defaults = {}
        for name in dir(tg_config):
            if name.isupper() and not name.startswith("_"):
                config_defaults[name] = _json_safe(getattr(tg_config, name))
        meta["turboguard_config_defaults"] = config_defaults
    except ImportError:
        pass

    # System information.
    meta["system"] = {
        "python_version": sys.version,
        "platform": platform.platform(),
        "hostname": platform.node(),
        "cpu": platform.processor() or platform.machine(),
        "pid": os.getpid(),
    }

    # PyTorch / CUDA info.
    try:
        import torch

        meta["system"]["torch_version"] = torch.__version__
        meta["system"]["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            meta["system"]["cuda_version"] = torch.version.cuda or "N/A"
            meta["system"]["gpu_name"] = torch.cuda.get_device_name(0)
            meta["system"]["gpu_memory_mb"] = round(
                torch.cuda.get_device_properties(0).total_memory / 1e6
            )
    except ImportError:
        pass

    # Package versions for key dependencies.
    for pkg in ["numpy", "pandas", "scikit-learn", "xgboost", "shap", "rich"]:
        try:
            import importlib.metadata

            meta["system"][f"{pkg}_version"] = importlib.metadata.version(pkg)
        except Exception:
            pass

    # Extra caller-provided metadata.
    if extra:
        meta.update(_json_safe(extra))

    metadata_path = run_dir / "metadata.json"
    temporary_path = run_dir / "metadata.json.tmp"
    with open(temporary_path, "w") as f:
        json.dump(meta, f, indent=2)
    temporary_path.replace(metadata_path)


def load_metadata(run_dir: Path) -> Dict[str, Any]:
    """Reads ``metadata.json`` from a run directory.

    Args:
        run_dir: Path to the run directory.

    Returns:
        Dict of saved metadata, or empty dict if not found.
    """
    path = Path(run_dir) / "metadata.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


def save_sectors(run_dir: Path, **arrays: np.ndarray) -> None:
    """Saves data sector arrays to ``sectors.npz``.

    Args:
        run_dir: Path to the run directory.
        **arrays: Named arrays (e.g. X_A, y_A, X_B, y_B, X_C, y_C).
    """
    np.savez(run_dir / "sectors.npz", **arrays)


def load_sectors(run_dir: Path) -> Dict[str, np.ndarray]:
    """Loads sector arrays from ``sectors.npz``.

    Args:
        run_dir: Path to the run directory.

    Returns:
        Dict mapping array names to loaded numpy arrays.
    """
    with np.load(Path(run_dir) / "sectors.npz") as npz:
        return {k: npz[k] for k in npz.files}


def save_eval_pool(run_dir: Path, **arrays: np.ndarray) -> None:
    """Saves evaluation pool indices to ``eval_pool.npz``.

    Args:
        run_dir: Path to the run directory.
        **arrays: Named index arrays (e.g. atk_idx, ben_idx).
    """
    np.savez(run_dir / "eval_pool.npz", **arrays)


def load_eval_pool(run_dir: Path) -> Dict[str, np.ndarray]:
    """Loads evaluation pool indices from ``eval_pool.npz``.

    Args:
        run_dir: Path to the run directory.

    Returns:
        Dict mapping array names to loaded numpy index arrays.
    """
    with np.load(Path(run_dir) / "eval_pool.npz") as npz:
        return {k: npz[k] for k in npz.files}
