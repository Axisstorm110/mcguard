"""
mcguard.metrics
===============
Built-in statistic functions compatible with ``bootstrap_ci`` and
``permutation_test``.

Every function here follows one of two signatures accepted by the resampling
functions:

- ``f(y: np.ndarray) -> float``          (no group argument)
- ``f(y: np.ndarray, group: np.ndarray) -> float``

You can pass any user-defined function with either signature — these are
provided as convenient defaults for common effect sizes.
"""

from __future__ import annotations

import numpy as np

__all__ = ["mean", "median", "mean_diff", "median_diff", "cohens_d"]


def _as_1d(a, name: str = "array") -> np.ndarray:
    arr = np.asarray(a)
    if arr.ndim != 1:
        raise ValueError(f"'{name}' must be 1-D, got shape {arr.shape}.")
    return arr.astype(float)


def _split_binary(
    y: np.ndarray, group: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Split *y* into two groups defined by a binary 0/1 *group* array."""
    g = np.asarray(group).astype(int)
    unique = set(np.unique(g).tolist())
    if not unique.issubset({0, 1}):
        raise ValueError(
            f"group must contain only 0 and 1; found values: {sorted(unique)}."
        )
    a, b = y[g == 1], y[g == 0]
    if len(a) == 0 or len(b) == 0:
        raise ValueError(
            "Both groups (0 and 1) must contain at least one observation."
        )
    return a, b


# ---------------------------------------------------------------------------
# Single-sample statistics
# ---------------------------------------------------------------------------

def mean(y: np.ndarray) -> float:
    """
    Arithmetic mean of *y*.

    Parameters
    ----------
    y : array-like, shape (n,)

    Returns
    -------
    float
    """
    return float(np.mean(_as_1d(y, "y")))


def median(y: np.ndarray) -> float:
    """
    Median of *y*.

    Parameters
    ----------
    y : array-like, shape (n,)

    Returns
    -------
    float
    """
    return float(np.median(_as_1d(y, "y")))


# ---------------------------------------------------------------------------
# Two-sample statistics  (f(y, group) -> float)
# ---------------------------------------------------------------------------

def mean_diff(y: np.ndarray, group: np.ndarray) -> float:
    """
    Difference in means: ``mean(y[group==1]) - mean(y[group==0])``.

    Parameters
    ----------
    y : array-like, shape (n,)
        Outcome variable.
    group : array-like, shape (n,)
        Binary group indicator (0 or 1).

    Returns
    -------
    float

    Raises
    ------
    ValueError
        If *group* is not binary, or if either group has no observations.

    Examples
    --------
    >>> mean_diff(np.array([1.0, 2.0, 3.0, 4.0]), np.array([0, 0, 1, 1]))
    2.0
    """
    y = _as_1d(y, "y")
    a, b = _split_binary(y, group)
    return float(np.mean(a) - np.mean(b))


def median_diff(y: np.ndarray, group: np.ndarray) -> float:
    """
    Difference in medians: ``median(y[group==1]) - median(y[group==0])``.

    Parameters
    ----------
    y : array-like, shape (n,)
        Outcome variable.
    group : array-like, shape (n,)
        Binary group indicator (0 or 1).

    Returns
    -------
    float
    """
    y = _as_1d(y, "y")
    a, b = _split_binary(y, group)
    return float(np.median(a) - np.median(b))


def cohens_d(y: np.ndarray, group: np.ndarray) -> float:
    """
    Cohen's d effect size using pooled standard deviation.

    ``d = (mean(y[group==1]) - mean(y[group==0])) / pooled_std``

    Parameters
    ----------
    y : array-like, shape (n,)
        Outcome variable.
    group : array-like, shape (n,)
        Binary group indicator (0 or 1).

    Returns
    -------
    float
        Returns 0.0 if the pooled standard deviation is zero.

    Notes
    -----
    Uses the unbiased (ddof=1) standard deviation within each group.
    The pooled standard deviation is computed as:

    .. math::

        s_p = \\sqrt{\\frac{(n_1 - 1)s_1^2 + (n_0 - 1)s_0^2}{n_1 + n_0 - 2}}
    """
    y = _as_1d(y, "y")
    a, b = _split_binary(y, group)
    n1, n0 = len(a), len(b)
    pooled_var = (
        ((n1 - 1) * np.var(a, ddof=1) + (n0 - 1) * np.var(b, ddof=1))
        / (n1 + n0 - 2)
    )
    if pooled_var == 0.0:
        return 0.0
    return float((np.mean(a) - np.mean(b)) / np.sqrt(pooled_var))
