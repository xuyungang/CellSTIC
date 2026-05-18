"""Global RNG: single entrypoint ``set_global_seed``."""

from __future__ import annotations

import os
import random
from typing import Optional

import numpy as np
import torch

GLOBAL_SEED = 42
_active_base_seed: int = GLOBAL_SEED


def active_base_seed() -> int:
    """Integer base used with epoch for CCC masking; matches the last ``set_global_seed`` call (or ``GLOBAL_SEED``)."""
    return int(_active_base_seed)


def set_global_seed(
    seed: Optional[int] = None,
    *,
    deterministic_cudnn: bool = True,
    cudnn_benchmark: bool = False,
    warn_only_deterministic: bool = True,
) -> int:
    """
    Set ``random`` / ``numpy`` / ``torch`` (+ CUDA) and optional cuDNN deterministic mode.

    Uses ``int(seed)`` when ``seed`` is given, otherwise ``GLOBAL_SEED``. Also sets BLAS thread
    env defaults (``setdefault``). Updates :func:`active_base_seed` to the same value. Returns the
    effective integer seed used.
    """
    global _active_base_seed
    effective = int(seed) if seed is not None else int(GLOBAL_SEED)

    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
    os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

    random.seed(effective)
    np.random.seed(effective)
    torch.manual_seed(effective)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(effective)

    if deterministic_cudnn:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = cudnn_benchmark
    if hasattr(torch, "use_deterministic_algorithms"):
        try:
            torch.use_deterministic_algorithms(True, warn_only=warn_only_deterministic)
        except Exception:
            pass

    _active_base_seed = effective
    return effective
