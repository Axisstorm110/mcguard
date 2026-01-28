import numpy as np
from mcguard import ResamplingPlan, permutation_test, metrics


def test_permutation_pvalue_in_range():
    rng = np.random.default_rng(0)
    y = rng.normal(size=200)
    g = rng.integers(0, 2, size=200)

    r = permutation_test(metrics.mean_diff, y=y, group=g, plan=ResamplingPlan(), P=500, seed=0)
    assert 0.0 <= r.p_value <= 1.0
