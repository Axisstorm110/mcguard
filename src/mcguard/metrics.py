from __future__ import annotations
import numpy as np
from .utils import as_1d

def mean(y: np.ndarray) -> float:
    y = as_1d(y).astype(float)
    return float(np.mean(y))

def median(y: np.ndarray) -> float:
    y = as_1d(y).astype(float)
    return float(np.median(y))

def mean_diff(y: np.ndarray, group: np.ndarray) -> float:
    y = as_1d(y).astype(float)
    g = as_1d(group).astype(int)
    if not set(np.unique(g)).issubset({0, 1}):
        raise ValueError("group must be binary 0/1.")
    a = y[g == 1]
    b = y[g == 0]
    if len(a) == 0 or len(b) == 0:
        raise ValueError("Both groups must have at least one observation.")
    return float(np.mean(a) - np.mean(b))
