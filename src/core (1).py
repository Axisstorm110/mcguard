"""
mcguard.core
============
Guardrailed resampling for clustered and structured biomedical data.

Public API
----------
ResamplingPlan  – encodes the dependence structure of a dataset
bootstrap_ci    – cluster-aware bootstrap confidence intervals
permutation_test – constrained permutation tests
BootstrapResult  – frozen result dataclass
PermutationResult – frozen result dataclass
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

__all__ = [
    "ResamplingPlan",
    "bootstrap_ci",
    "permutation_test",
    "BootstrapResult",
    "PermutationResult",
]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _as_1d(a: Any, name: str = "array") -> np.ndarray:
    """Convert input to a 1-D numpy array, raising a clear error if it fails."""
    arr = np.asarray(a)
    if arr.ndim != 1:
        raise ValueError(
            f"'{name}' must be 1-D, but got shape {arr.shape}."
        )
    return arr


def _check_lengths(**arrays: np.ndarray) -> int:
    """Assert all arrays have the same length; return that length."""
    n: int | None = None
    for name, arr in arrays.items():
        arr = _as_1d(arr, name)
        if n is None:
            n = len(arr)
        elif len(arr) != n:
            raise ValueError(
                f"Length mismatch: '{name}' has {len(arr)} elements but expected {n}."
            )
    return int(n or 0)


def _make_rng(seed: int | np.random.Generator | None) -> np.random.Generator:
    """Return a Generator from an int seed, an existing Generator, or None."""
    if isinstance(seed, np.random.Generator):
        return seed
    return np.random.default_rng(seed)


def _percentile_ci(samples: np.ndarray, alpha: float) -> tuple[float, float]:
    """Percentile bootstrap confidence interval."""
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"alpha must be in (0, 1), got {alpha}.")
    lo = float(np.quantile(samples, alpha / 2))
    hi = float(np.quantile(samples, 1.0 - alpha / 2))
    return lo, hi


# ---------------------------------------------------------------------------
# ResamplingPlan
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ResamplingPlan:
    """
    Encodes the dependence structure of a dataset for resampling.

    Parameters
    ----------
    unit : array-like, optional
        Cluster / unit identifier (e.g. patient_id). When provided, resampling
        draws *whole units* (with replacement) rather than individual rows.
        This is the primary guard against the iid assumption being violated.
    strata : array-like, optional
        Stratification label (e.g. site_id). Resampling is performed
        independently within each stratum to preserve stratum sizes.
    blocks : array-like, optional
        Block label for a secondary partition within strata (e.g. time_block).
        Units are bucketed by (strata, blocks) before resampling.
    paired : array-like, optional
        Pair identifier for matched designs (e.g. matched_pair_id).
        Used only in permutation tests: labels are permuted within each pair.

    Examples
    --------
    Cluster bootstrap — resample patients, not rows:
    >>> plan = ResamplingPlan(unit=patient_id)

    Stratified cluster bootstrap — resample patients within each site:
    >>> plan = ResamplingPlan(unit=patient_id, strata=site_id)

    Paired permutation test:
    >>> plan = ResamplingPlan(paired=pair_id)

    iid (no constraints):
    >>> plan = ResamplingPlan()
    """

    unit:   np.ndarray | None = field(default=None, repr=False)
    strata: np.ndarray | None = field(default=None, repr=False)
    blocks: np.ndarray | None = field(default=None, repr=False)
    paired: np.ndarray | None = field(default=None, repr=False)

    def validate(
        self,
        *,
        y: np.ndarray,
        group: np.ndarray | None = None,
    ) -> None:
        """
        Check that all plan arrays are 1-D and share the same length as *y*.

        Raises
        ------
        ValueError
            If any array is not 1-D or has a different length from *y*.
        """
        arrays: dict[str, np.ndarray] = {"y": _as_1d(y, "y")}
        if group is not None:
            arrays["group"] = _as_1d(group, "group")
        for name in ("unit", "strata", "blocks", "paired"):
            arr = getattr(self, name)
            if arr is not None:
                arrays[name] = _as_1d(arr, name)
        _check_lengths(**arrays)

    # ------------------------------------------------------------------
    # Internal bucketing used by bootstrap
    # ------------------------------------------------------------------

    def _bucket_keys(self, n: int) -> np.ndarray:
        """
        Build a (strata, blocks) composite key per observation.
        Returns an object array of None when no grouping is specified.
        """
        parts = []
        if self.strata is not None:
            parts.append(_as_1d(self.strata, "strata").astype(object))
        if self.blocks is not None:
            parts.append(_as_1d(self.blocks, "blocks").astype(object))
        if not parts:
            return np.full(n, None, dtype=object)
        cols = np.stack(parts, axis=1)
        return np.array([tuple(row) for row in cols], dtype=object)

    def bucket_units(self, y: np.ndarray) -> dict[Any, list[Any]]:
        """
        Partition unit IDs into buckets defined by (strata × blocks).

        Returns a dict mapping bucket key → ordered list of unique unit IDs
        within that bucket. When *unit* is None, each row index is its own unit.
        """
        y = _as_1d(y, "y")
        n = len(y)
        self.validate(y=y)

        unit_arr = (
            np.arange(n, dtype=object)
            if self.unit is None
            else _as_1d(self.unit, "unit").astype(object)
        )
        keys = self._bucket_keys(n)

        buckets: dict[Any, list[Any]] = {}
        for i in range(n):
            k = keys[i]
            buckets.setdefault(k, [])
            buckets[k].append(unit_arr[i])

        # Deduplicate while preserving insertion order
        return {k: list(dict.fromkeys(v)) for k, v in buckets.items()}


# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BootstrapResult:
    """
    Result of a bootstrap confidence interval computation.

    Attributes
    ----------
    estimate : float
        The statistic computed on the original data.
    ci_lo : float
        Lower bound of the (1 - alpha) confidence interval.
    ci_hi : float
        Upper bound of the (1 - alpha) confidence interval.
    samples : np.ndarray
        Bootstrap replicate values (empty unless ``return_samples=True``).
    meta : dict
        Metadata: B, alpha, seed, and plan configuration flags.
    """
    estimate: float
    ci_lo:    float
    ci_hi:    float
    samples:  np.ndarray
    meta:     dict


def _bootstrap_indices(
    rng: np.random.Generator,
    *,
    y: np.ndarray,
    plan: ResamplingPlan,
) -> np.ndarray:
    """
    Draw one bootstrap replicate of row indices respecting the plan.

    Strategy
    --------
    - Split observations into (strata × blocks) buckets.
    - Within each bucket, resample *units* (or rows if unit is None)
      with replacement, then include all rows belonging to each drawn unit.
    """
    y = _as_1d(y, "y")
    buckets = plan.bucket_units(y)

    # Pre-build unit → row-index map (always; empty dict when unit is None)
    unit_to_rows: dict[Any, np.ndarray] = {}
    if plan.unit is not None:
        unit_arr = _as_1d(plan.unit, "unit").astype(object)
        tmp: dict[Any, list[int]] = {}
        for i, uid in enumerate(unit_arr):
            tmp.setdefault(uid, []).append(i)
        unit_to_rows = {k: np.array(v, dtype=int) for k, v in tmp.items()}

    out: list[np.ndarray] = []
    for _, unit_ids in buckets.items():
        m = len(unit_ids)
        draws = rng.integers(0, m, size=m)
        sampled = [unit_ids[i] for i in draws]

        if plan.unit is None:
            # unit_ids are already row indices
            out.append(np.array(sampled, dtype=int))
        else:
            for uid in sampled:
                out.append(unit_to_rows[uid])

    return np.concatenate(out)


def bootstrap_ci(
    stat_fn: Callable,
    *,
    y: np.ndarray,
    group: np.ndarray | None = None,
    plan: ResamplingPlan | None = None,
    B: int = 2000,
    alpha: float = 0.05,
    seed: int | np.random.Generator | None = None,
    return_samples: bool = False,
) -> BootstrapResult:
    """
    Compute a bootstrap confidence interval with guardrailed resampling.

    Parameters
    ----------
    stat_fn : callable
        Statistic function.  Signature ``f(y) -> float`` when *group* is None,
        or ``f(y, group) -> float`` when *group* is provided.
    y : array-like, shape (n,)
        Outcome or response variable.
    group : array-like, shape (n,), optional
        Group / treatment indicator (passed to *stat_fn* if provided).
    plan : ResamplingPlan, optional
        Resampling structure.  Defaults to iid row resampling.
    B : int, default 2000
        Number of bootstrap replicates.
    alpha : float, default 0.05
        Significance level; the function returns a (1 - alpha) CI.
    seed : int or numpy.random.Generator, optional
        Random seed for reproducibility.
    return_samples : bool, default False
        If True, attach the B replicate values to the result.

    Returns
    -------
    BootstrapResult

    Examples
    --------
    Cluster bootstrap CI for the difference in means:

    >>> from mcguard import bootstrap_ci, ResamplingPlan, metrics
    >>> plan = ResamplingPlan(unit=patient_id)
    >>> result = bootstrap_ci(metrics.mean_diff, y=outcome, group=treatment,
    ...                       plan=plan, B=2000, seed=0)
    >>> print(result.ci_lo, result.ci_hi)

    Raises
    ------
    ValueError
        If array lengths are inconsistent or alpha is out of range.
    """
    if not isinstance(B, int) or B < 1:
        raise ValueError(f"B must be a positive integer, got {B!r}.")

    y = _as_1d(y, "y")
    g = _as_1d(group, "group") if group is not None else None

    if plan is None:
        plan = ResamplingPlan()

    plan.validate(y=y, group=g)

    r = _make_rng(seed)

    def _apply(idx: np.ndarray | None = None) -> float:
        yi = y if idx is None else y[idx]
        if g is None:
            return float(stat_fn(yi))
        gi = g if idx is None else g[idx]
        return float(stat_fn(yi, gi))

    estimate = _apply()
    samples = np.empty(B, dtype=float)
    for b in range(B):
        idx = _bootstrap_indices(r, y=y, plan=plan)
        samples[b] = _apply(idx)

    lo, hi = _percentile_ci(samples, alpha)

    meta: dict[str, Any] = {
        "B": B,
        "alpha": alpha,
        "seed": seed if not isinstance(seed, np.random.Generator) else "Generator",
        "plan": {
            "unit":   plan.unit is not None,
            "strata": plan.strata is not None,
            "blocks": plan.blocks is not None,
        },
    }

    return BootstrapResult(
        estimate=estimate,
        ci_lo=lo,
        ci_hi=hi,
        samples=samples if return_samples else np.array([], dtype=float),
        meta=meta,
    )


# ---------------------------------------------------------------------------
# Permutation test
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PermutationResult:
    """
    Result of a constrained permutation test.

    Attributes
    ----------
    estimate : float
        The statistic computed on the original (unpermuted) data.
    p_value : float
        Two-sided or one-sided p-value (with +1 continuity correction).
    null_samples : np.ndarray
        Null distribution values (empty unless ``return_samples=True``).
    meta : dict
        Metadata: P, seed, alternative, and plan configuration flags.
    """
    estimate:     float
    p_value:      float
    null_samples: np.ndarray
    meta:         dict


def _permute_labels(
    rng: np.random.Generator,
    *,
    group: np.ndarray,
    plan: ResamplingPlan,
) -> np.ndarray:
    """
    Permute group labels while respecting plan constraints.

    Priority
    --------
    1. ``paired``  → permute within each matched pair.
    2. ``strata``  → permute within each stratum.
    3. No constraint → global permutation.
    """
    g = _as_1d(group, "group").astype(int)
    out = g.copy()

    def _permute_within(label_arr: np.ndarray) -> None:
        groups: dict[Any, list[int]] = {}
        for i, lbl in enumerate(label_arr.astype(object)):
            groups.setdefault(lbl, []).append(i)
        for _, idxs in groups.items():
            idx_arr = np.array(idxs, dtype=int)
            out[idx_arr] = out[idx_arr][rng.permutation(len(idx_arr))]

    if plan.paired is not None:
        _permute_within(_as_1d(plan.paired, "paired"))
    elif plan.strata is not None:
        _permute_within(_as_1d(plan.strata, "strata"))
    else:
        out[:] = out[rng.permutation(len(g))]

    return out


def permutation_test(
    stat_fn: Callable,
    *,
    y: np.ndarray,
    group: np.ndarray,
    plan: ResamplingPlan | None = None,
    P: int = 10_000,
    seed: int | np.random.Generator | None = None,
    alternative: str = "two-sided",
    return_samples: bool = False,
) -> PermutationResult:
    """
    Constrained permutation test for a group difference.

    Parameters
    ----------
    stat_fn : callable
        Statistic function with signature ``f(y, group) -> float``.
    y : array-like, shape (n,)
        Outcome variable.
    group : array-like, shape (n,)
        Group / treatment labels (passed to *stat_fn*).
    plan : ResamplingPlan, optional
        Permutation constraints.  Defaults to global permutation.
    P : int, default 10_000
        Number of permutation replicates.
    seed : int or numpy.random.Generator, optional
        Random seed for reproducibility.
    alternative : {'two-sided', 'greater', 'less'}, default 'two-sided'
        Direction of the alternative hypothesis.
    return_samples : bool, default False
        If True, attach the P null-distribution values to the result.

    Returns
    -------
    PermutationResult

    Notes
    -----
    P-values use the +1 continuity correction recommended by
    Phipson & Smyth (2010) to avoid p = 0.

    Examples
    --------
    Within-stratum permutation test:

    >>> from mcguard import permutation_test, ResamplingPlan, metrics
    >>> plan = ResamplingPlan(strata=site_id)
    >>> result = permutation_test(metrics.mean_diff, y=outcome, group=treatment,
    ...                           plan=plan, P=10000, seed=0)
    >>> print(result.p_value)

    Raises
    ------
    ValueError
        If *alternative* is not one of the accepted values, or array lengths
        are inconsistent.
    """
    if alternative not in ("two-sided", "greater", "less"):
        raise ValueError(
            f"alternative must be 'two-sided', 'greater', or 'less'; got '{alternative}'."
        )

    y = _as_1d(y, "y").astype(float)
    g = _as_1d(group, "group")

    if plan is None:
        plan = ResamplingPlan()

    plan.validate(y=y, group=g)

    r = _make_rng(seed)
    estimate = float(stat_fn(y, g))

    null = np.empty(P, dtype=float)
    for i in range(P):
        gp = _permute_labels(r, group=g, plan=plan)
        null[i] = float(stat_fn(y, gp))

    if alternative == "two-sided":
        p_value = (np.sum(np.abs(null) >= abs(estimate)) + 1) / (P + 1)
    elif alternative == "greater":
        p_value = (np.sum(null >= estimate) + 1) / (P + 1)
    else:  # less
        p_value = (np.sum(null <= estimate) + 1) / (P + 1)

    meta: dict[str, Any] = {
        "P": P,
        "seed": seed if not isinstance(seed, np.random.Generator) else "Generator",
        "alternative": alternative,
        "plan": {
            "paired": plan.paired is not None,
            "strata": plan.strata is not None,
        },
    }

    return PermutationResult(
        estimate=estimate,
        p_value=float(p_value),
        null_samples=null if return_samples else np.array([], dtype=float),
        meta=meta,
    )
