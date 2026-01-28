from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from .plan import ResamplingPlan
from .utils import as_1d, rng

@dataclass(frozen=True)
class PermutationResult:
    estimate: float
    p_value: float
    null_samples: np.ndarray
    meta: dict

def _permute_group(r: np.random.Generator, *, group: np.ndarray, plan: ResamplingPlan) -> np.ndarray:
    g = as_1d(group).astype(int)

    if plan.paired is not None:
        pair = as_1d(plan.paired).astype(object)
        out = g.copy()
        d: dict[object, list[int]] = {}
        for i, pid in enumerate(pair):
            d.setdefault(pid, []).append(i)
        for _, idxs in d.items():
            idxs = np.array(idxs, dtype=int)
            out[idxs] = out[idxs][r.permutation(len(idxs))]
        return out

    if plan.strata is not None:
        strata = as_1d(plan.strata).astype(object)
        out = g.copy()
        d: dict[object, list[int]] = {}
        for i, s in enumerate(strata):
            d.setdefault(s, []).append(i)
        for _, idxs in d.items():
            idxs = np.array(idxs, dtype=int)
            out[idxs] = out[idxs][r.permutation(len(idxs))]
        return out

    return g[r.permutation(len(g))]

def permutation_test(
    stat_fn,
    *,
    y: np.ndarray,
    group: np.ndarray,
    plan: ResamplingPlan | None = None,
    P: int = 10000,
    seed: int | None = None,
    alternative: str = "two-sided",
    return_samples: bool = False,
) -> PermutationResult:
    y = as_1d(y).astype(float)
    g = as_1d(group)

    if plan is None:
        plan = ResamplingPlan()

    plan.validate(y=y, group=g)
    r = rng(seed)

    est = float(stat_fn(y, g))
    null = np.empty(P, dtype=float)

    for i in range(P):
        gp = _permute_group(r, group=g, plan=plan)
        null[i] = float(stat_fn(y, gp))

    if alternative == "two-sided":
        pval = (np.sum(np.abs(null) >= abs(est)) + 1) / (P + 1)
    elif alternative == "greater":
        pval = (np.sum(null >= est) + 1) / (P + 1)
    elif alternative == "less":
        pval = (np.sum(null <= est) + 1) / (P + 1)
    else:
        raise ValueError("alternative must be 'two-sided', 'greater', or 'less'")

    meta = {
        "P": P,
        "seed": seed,
        "alternative": alternative,
        "plan": {"paired": plan.paired is not None, "strata": plan.strata is not None},
    }

    if not return_samples:
        null = np.array([], dtype=float)

    return PermutationResult(estimate=est, p_value=float(pval), null_samples=null, meta=meta)
