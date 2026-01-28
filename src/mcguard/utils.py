from __future__ import annotations
import numpy as np

def rng(seed: int | None) -> np.random.Generator:
    return np.random.default_rng(seed)

def as_1d(a) -> np.ndarray:
    a = np.asarray(a)
    if a.ndim != 1:
        raise ValueError("Expected 1D array.")
    return a

def check_same_length(**arrays) -> int:
    n = None
    for k, v in arrays.items():
        v = as_1d(v)
        if n is None:
            n = len(v)
        elif len(v) != n:
            raise ValueError(f"Length mismatch: {k} has {len(v)} but expected {n}.")
    return int(n or 0)

def percentile_interval(samples: np.ndarray, alpha: float) -> tuple[float, float]:
    if not (0 < alpha < 1):
        raise ValueError("alpha must be in (0, 1).")
    lo = np.quantile(samples, alpha / 2)
    hi = np.quantile(samples, 1 - alpha / 2)
    return float(lo), float(hi)
