from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from .plan import ResamplingPlan
from .utils import as_1d, rng, percentile_interval

@dataclass(frozen=True)
class BootstrapResult:
    estimate: float
    ci_lo: float
    ci_hi: float
    samples: np.ndarray
    meta: dict

def _resample_indices(r: np.random.Generator, *, y: np.ndarray, plan: ResamplingPlan) -> np.ndarray:
    y = as_1d(y)
    buckets = plan.bucket_units(y)

    # If unit is None, buckets contain row IDs already
    if plan.unit is None:
        out = []
        for _, row_ids in buckets.items():
            m = len(row_ids)
            draws = r.integers(0, m, size=m)
            out.append(np.array([row_ids[i] for i in draws], dtype=int))
        return np.concatenate(out)

    # Build unit -> indices map
    unit = as_1d(plan.unit).astype(object)
    unit_to_idx: dict[object, list[int]] = {}
    for i, uid in enumerate(unit):
        unit_to_idx.setdefault(uid, []).append(i)
    unit_to_idx_np = {k: np.array(v, dtype=int) for k, v in unit_to_idx.items()}

    out = []
    for _, unit_ids in buckets.items():
        m = len(unit_ids)
        draws = r.integers(0, m, size=m)
        sampled_units = [unit_ids[i] for i in draws]
        for uid in sampled_units:
            out.append(unit_to_idx_np[uid])

    return np.concatenate(out)

def bootstrap_ci(
    stat_fn,
    *,
    y: np.ndarray,
    group: np.ndarray | None = None,
    plan: ResamplingPlan | None = None,
    B: int = 2000,
    alpha: float = 0.05,
    seed: int | None = None,
    return_samples: bool = False,
) -> BootstrapResult:
    y = as_1d(y)
    g = as_1d(group) if group is not None else None

    if plan is None:
        plan = ResamplingPlan()

    plan.validate(y=y, group=g)
    r = rng(seed)

    est = float(stat_fn(y, g)) if g is not None else float(stat_fn(y))

    samples = np.empty(B, dtype=float)
    for b in range(B):
        idx = _resample_indices(r, y=y, plan=plan)
        if g is None:
            samples[b] = float(stat_fn(y[idx]))
        else:
            samples[b] = float(stat_fn(y[idx], g[idx]))

    lo, hi = percentile_interval(samples, alpha)

    meta = {
        "B": B,
        "alpha": alpha,
        "seed": seed,
        "plan": {
            "unit": plan.unit is not None,
            "strata": plan.strata is not None,
            "blocks": plan.blocks is not None,
        },
    }

    if not return_samples:
        samples = np.array([], dtype=float)

    return BootstrapResult(estimate=est, ci_lo=lo, ci_hi=hi, samples=samples, meta=meta)
