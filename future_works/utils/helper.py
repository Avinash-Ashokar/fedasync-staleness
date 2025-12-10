from __future__ import annotations

import random
import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """Seed all RNGs used in this project.

    Setting a global seed helps to ensure reproducible results.  This
    function touches Python's built‑in random module, NumPy, and
    PyTorch's CPU and GPU RNGs.  Deterministic behaviour in cuDNN
    kernels is also enabled.

    Parameters
    ----------
    seed:
        The random seed to use.  Defaults to ``42``.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # When running on GPUs you may have more than one device
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # Deterministic behaviour comes at a performance cost but
    # reproducibility is more important for experimentation.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """Return the first available computation device.

    Tries to use CUDA if available, otherwise falls back to MPS
    (Apple Silicon) and finally the CPU.

    Returns
    -------
    device:
        A PyTorch ``torch.device`` object indicating where tensors
        should be allocated.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    # MPS stands for Metal Performance Shaders.  It is the backend
    # available on Apple Silicon systems for GPU acceleration.
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
