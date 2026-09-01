"""Device selection utility."""

import torch


def get_device() -> torch.device:
    """Returns the best available computation device.

    Prefers CUDA GPU if available, otherwise falls back to CPU.

    Returns:
        A ``torch.device`` instance.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
