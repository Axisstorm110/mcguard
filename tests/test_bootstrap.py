import numpy as np
from mcguard import ResamplingPlan, bootstrap_ci, metrics


def test_bootstrap_reproducible_seed():
    rng = np.random.default_rng(0)
    y = rng.normal(size=200)
    g = rng.integers(0, 2, size=200)

    r1 = bootstrap_ci(metrics.mean_diff, y=y, group=g, plan=ResamplingPlan(), B=200, seed=123)
    r2 = bootstrap_ci(metrics.mean_diff, y=y, group=g, plan=ResamplingPlan(), B=200, seed=123)

    assert (r1.ci_lo, r1.ci_hi) == (r2.ci_lo, r2.ci_hi)


def test_cluster_bootstrap_runs():
    rng = np.random.default_rng(0)
    patient = np.repeat(np.arange(30), 5)
    y = rng.normal(size=len(patient))
    g = rng.integers(0, 2, size=len(patient))

    plan = ResamplingPlan(unit=patient)
    r = bootstrap_ci(metrics.mean_diff, y=y, group=g, plan=plan, B=100, seed=0)

    assert np.isfinite(r.estimate)
