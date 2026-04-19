"""
Tests for mcguard.

Run with:  pytest tests/ -v
"""

from __future__ import annotations

import numpy as np
import pytest

from mcguard import ResamplingPlan, bootstrap_ci, permutation_test
from mcguard import metrics


# ===========================================================================
# Fixtures
# ===========================================================================

@pytest.fixture()
def clustered_data():
    """30 patients × 5 observations each, binary treatment balanced per patient."""
    rng = np.random.default_rng(0)
    n_patients = 30
    obs_per = 5
    patient = np.repeat(np.arange(n_patients), obs_per)
    y = rng.normal(loc=patient * 0.1, scale=1.0, size=len(patient))
    group = np.repeat(rng.integers(0, 2, size=n_patients), obs_per)
    return y, group, patient


@pytest.fixture()
def stratified_data():
    """3 sites, 20 patients each, 4 obs per patient."""
    rng = np.random.default_rng(1)
    n_sites, pts_per_site, obs_per = 3, 20, 4
    site = np.repeat(np.arange(n_sites), pts_per_site * obs_per)
    patient = np.repeat(np.arange(n_sites * pts_per_site), obs_per)
    y = rng.normal(size=len(patient))
    group = rng.integers(0, 2, size=len(patient))
    return y, group, patient, site


# ===========================================================================
# ResamplingPlan
# ===========================================================================

class TestResamplingPlan:
    def test_default_plan_valid(self):
        y = np.arange(10, dtype=float)
        plan = ResamplingPlan()
        plan.validate(y=y)  # should not raise

    def test_mismatched_unit_raises(self):
        y = np.arange(10, dtype=float)
        unit = np.arange(5)  # wrong length
        plan = ResamplingPlan(unit=unit)
        with pytest.raises(ValueError, match="Length mismatch"):
            plan.validate(y=y)

    def test_2d_y_raises(self):
        with pytest.raises(ValueError, match="1-D"):
            ResamplingPlan().validate(y=np.ones((5, 2)))

    def test_bucket_units_no_strata(self):
        y = np.ones(6)
        unit = np.array([0, 0, 1, 1, 2, 2])
        plan = ResamplingPlan(unit=unit)
        buckets = plan.bucket_units(y)
        assert len(buckets) == 1  # single bucket (key=None)
        assert set(list(buckets.values())[0]) == {0, 1, 2}

    def test_bucket_units_with_strata(self):
        y = np.ones(6)
        unit = np.array([0, 1, 2, 3, 4, 5])
        strata = np.array([0, 0, 0, 1, 1, 1])
        plan = ResamplingPlan(unit=unit, strata=strata)
        buckets = plan.bucket_units(y)
        assert len(buckets) == 2
        for v in buckets.values():
            assert len(v) == 3


# ===========================================================================
# bootstrap_ci
# ===========================================================================

class TestBootstrapCI:
    def test_iid_ci_contains_true_mean(self):
        rng = np.random.default_rng(42)
        y = rng.normal(loc=5.0, scale=1.0, size=500)
        result = bootstrap_ci(metrics.mean, y=y, B=1000, alpha=0.05, seed=0)
        assert result.ci_lo < 5.0 < result.ci_hi

    def test_reproducible_with_seed(self, clustered_data):
        y, g, patient = clustered_data
        plan = ResamplingPlan(unit=patient)
        r1 = bootstrap_ci(metrics.mean_diff, y=y, group=g, plan=plan, B=200, seed=7)
        r2 = bootstrap_ci(metrics.mean_diff, y=y, group=g, plan=plan, B=200, seed=7)
        assert r1.ci_lo == r2.ci_lo
        assert r1.ci_hi == r2.ci_hi

    def test_different_seeds_differ(self, clustered_data):
        y, g, patient = clustered_data
        plan = ResamplingPlan(unit=patient)
        r1 = bootstrap_ci(metrics.mean_diff, y=y, group=g, plan=plan, B=200, seed=1)
        r2 = bootstrap_ci(metrics.mean_diff, y=y, group=g, plan=plan, B=200, seed=2)
        assert r1.ci_lo != r2.ci_lo

    def test_cluster_ci_wider_than_iid(self, clustered_data):
        """Cluster bootstrap should produce wider CIs than iid bootstrap
        when observations within clusters are correlated."""
        y, g, patient = clustered_data
        iid = bootstrap_ci(
            metrics.mean_diff, y=y, group=g,
            plan=ResamplingPlan(), B=500, seed=0
        )
        clustered = bootstrap_ci(
            metrics.mean_diff, y=y, group=g,
            plan=ResamplingPlan(unit=patient), B=500, seed=0
        )
        iid_width = iid.ci_hi - iid.ci_lo
        cluster_width = clustered.ci_hi - clustered.ci_lo
        assert cluster_width >= iid_width * 0.8  # allow some Monte Carlo noise

    def test_return_samples_populates_array(self, clustered_data):
        y, g, patient = clustered_data
        result = bootstrap_ci(
            metrics.mean_diff, y=y, group=g,
            plan=ResamplingPlan(unit=patient), B=100, seed=0,
            return_samples=True,
        )
        assert len(result.samples) == 100

    def test_no_return_samples_is_empty(self, clustered_data):
        y, g, patient = clustered_data
        result = bootstrap_ci(
            metrics.mean_diff, y=y, group=g, B=100, seed=0, return_samples=False
        )
        assert len(result.samples) == 0

    def test_stratified_cluster_runs(self, stratified_data):
        y, g, patient, site = stratified_data
        plan = ResamplingPlan(unit=patient, strata=site)
        result = bootstrap_ci(metrics.mean_diff, y=y, group=g, plan=plan, B=200, seed=0)
        assert np.isfinite(result.estimate)
        assert np.isfinite(result.ci_lo)
        assert np.isfinite(result.ci_hi)

    def test_invalid_alpha_raises(self):
        y = np.ones(10)
        with pytest.raises(ValueError, match="alpha"):
            bootstrap_ci(metrics.mean, y=y, B=10, alpha=1.5)

    def test_meta_records_plan_flags(self, clustered_data):
        y, g, patient = clustered_data
        plan = ResamplingPlan(unit=patient)
        result = bootstrap_ci(metrics.mean_diff, y=y, group=g, plan=plan, B=50, seed=0)
        assert result.meta["plan"]["unit"] is True
        assert result.meta["plan"]["strata"] is False


# ===========================================================================
# permutation_test
# ===========================================================================

class TestPermutationTest:
    def test_pvalue_in_unit_interval(self, clustered_data):
        y, g, _ = clustered_data
        result = permutation_test(metrics.mean_diff, y=y, group=g, P=500, seed=0)
        assert 0.0 <= result.p_value <= 1.0

    def test_null_effect_high_pvalue(self):
        """Under the null (same distribution for both groups), p-value should
        not be systematically small."""
        rng = np.random.default_rng(99)
        y = rng.normal(size=200)
        g = rng.integers(0, 2, size=200)
        result = permutation_test(metrics.mean_diff, y=y, group=g, P=2000, seed=0)
        assert result.p_value > 0.01  # should not reject at any reasonable level

    def test_strong_effect_low_pvalue(self):
        """With a very large effect, p-value should be very small."""
        rng = np.random.default_rng(0)
        y = np.concatenate([rng.normal(0, 1, 100), rng.normal(10, 1, 100)])
        g = np.array([0] * 100 + [1] * 100)
        result = permutation_test(metrics.mean_diff, y=y, group=g, P=999, seed=0)
        assert result.p_value < 0.01

    def test_reproducible(self, clustered_data):
        y, g, _ = clustered_data
        r1 = permutation_test(metrics.mean_diff, y=y, group=g, P=500, seed=5)
        r2 = permutation_test(metrics.mean_diff, y=y, group=g, P=500, seed=5)
        assert r1.p_value == r2.p_value

    def test_within_strata_permutation(self, stratified_data):
        y, g, _, site = stratified_data
        plan = ResamplingPlan(strata=site)
        result = permutation_test(
            metrics.mean_diff, y=y, group=g, plan=plan, P=500, seed=0
        )
        assert 0.0 <= result.p_value <= 1.0

    def test_paired_permutation(self):
        rng = np.random.default_rng(0)
        n_pairs = 50
        pair_id = np.repeat(np.arange(n_pairs), 2)
        y = rng.normal(size=n_pairs * 2)
        g = np.tile([0, 1], n_pairs)
        plan = ResamplingPlan(paired=pair_id)
        result = permutation_test(
            metrics.mean_diff, y=y, group=g, plan=plan, P=500, seed=0
        )
        assert 0.0 <= result.p_value <= 1.0

    def test_invalid_alternative_raises(self, clustered_data):
        y, g, _ = clustered_data
        with pytest.raises(ValueError, match="alternative"):
            permutation_test(metrics.mean_diff, y=y, group=g, alternative="both")

    def test_one_sided_greater(self):
        y = np.array([1.0, 2.0, 3.0, 4.0])
        g = np.array([0, 0, 1, 1])
        result = permutation_test(
            metrics.mean_diff, y=y, group=g, P=500, seed=0, alternative="greater"
        )
        assert 0.0 <= result.p_value <= 1.0

    def test_return_samples(self, clustered_data):
        y, g, _ = clustered_data
        result = permutation_test(
            metrics.mean_diff, y=y, group=g, P=100, seed=0, return_samples=True
        )
        assert len(result.null_samples) == 100


# ===========================================================================
# metrics
# ===========================================================================

class TestMetrics:
    def test_mean(self):
        assert metrics.mean(np.array([1.0, 2.0, 3.0])) == pytest.approx(2.0)

    def test_median(self):
        assert metrics.median(np.array([1.0, 3.0, 2.0])) == pytest.approx(2.0)

    def test_mean_diff(self):
        y = np.array([1.0, 2.0, 3.0, 4.0])
        g = np.array([0, 0, 1, 1])
        assert metrics.mean_diff(y, g) == pytest.approx(2.0)

    def test_mean_diff_non_binary_raises(self):
        y = np.array([1.0, 2.0, 3.0])
        g = np.array([0, 1, 2])
        with pytest.raises(ValueError, match="binary"):
            metrics.mean_diff(y, g)

    def test_mean_diff_empty_group_raises(self):
        y = np.array([1.0, 2.0, 3.0])
        g = np.array([0, 0, 0])  # no group-1 observations
        with pytest.raises(ValueError, match="at least one"):
            metrics.mean_diff(y, g)

    def test_median_diff(self):
        y = np.array([1.0, 2.0, 10.0, 11.0])
        g = np.array([0, 0, 1, 1])
        assert metrics.median_diff(y, g) == pytest.approx(9.0)

    def test_cohens_d_sign(self):
        y = np.array([0.0, 0.0, 1.0, 1.0])
        g = np.array([0, 0, 1, 1])
        d = metrics.cohens_d(y, g)
        assert d > 0  # group 1 has higher mean

    def test_cohens_d_zero_variance(self):
        y = np.array([1.0, 1.0, 1.0, 1.0])
        g = np.array([0, 0, 1, 1])
        assert metrics.cohens_d(y, g) == 0.0
