# mcguard

**mcguard** is a lightweight Python library for statistically valid resampling in datasets with non-independent observations, common in biomedical research and ML evaluation.

If you have clustered data like multiple measurements per patient, repeated trials per subject, or samples grouped by site, the standard iid bootstrap underestimates uncertainty and produces overconfident confidence intervals. `mcguard` fixes this.

[![PyPI](https://img.shields.io/pypi/v/mcguard)](https://pypi.org/project/mcguard/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)

---

## Installation

```bash
pip install mcguard
```

---

## Quick Start

```python
from mcguard import bootstrap_ci, permutation_test, ResamplingPlan, metrics

# Cluster bootstrap: resample whole patients, not individual rows
plan = ResamplingPlan(unit=patient_id)

result = bootstrap_ci(
    metrics.mean_diff,
    y=outcome,
    group=treatment,
    plan=plan,
    B=2000,
    seed=42,
)
print(result.ci_lo, result.ci_hi)

# Constrained permutation test within strata
plan = ResamplingPlan(strata=site_id)

result = permutation_test(
    metrics.mean_diff,
    y=outcome,
    group=treatment,
    plan=plan,
    P=10000,
    seed=42,
)
print(result.p_value)
```

---

## Features

- **Cluster bootstrap CIs** -- resample at the unit level (patient, subject, site) rather than row level
- **Stratified resampling** -- resample independently within each stratum to preserve group balance
- **Constrained permutation tests** -- permute labels within matched pairs or within strata
- **Built-in metrics** -- `mean`, `median`, `mean_diff`, `median_diff`, `cohens_d`
- **Any custom statistic** -- pass any `f(y)` or `f(y, group)` function
- **Reproducible** -- seed support via `numpy.random.default_rng`
- **Minimal dependencies** -- only `numpy` required

---

## ResamplingPlan

The `ResamplingPlan` dataclass encodes the dependence structure of your dataset:

| Parameter | Purpose |
|-----------|---------|
| `unit` | Cluster ID (e.g. patient_id). Resamples whole units, not rows. |
| `strata` | Stratification label (e.g. site_id). Resamples within each stratum. |
| `blocks` | Block label for secondary partition within strata (e.g. time_block). |
| `paired` | Pair ID for matched designs. Permutation test only. |

```python
# iid -- no constraints (default)
ResamplingPlan()

# Cluster bootstrap
ResamplingPlan(unit=patient_id)

# Stratified cluster bootstrap
ResamplingPlan(unit=patient_id, strata=site_id)

# Paired permutation test
ResamplingPlan(paired=pair_id)
```

---

## Why This Matters

Consider a dataset with 30 patients and 5 observations each (150 rows total). An iid bootstrap treats all 150 rows as independent, inflating the effective sample size by 5x and producing confidence intervals that are far too narrow.

`mcguard` resamples at the patient level, drawing 30 patients with replacement and including all their rows, which correctly reflects the true uncertainty in the data.

---

## Running Tests

```bash
pip install pytest
pytest tests/ -v
```

---

## Citation

If you use `mcguard` in your research, please cite:

```
Rizvi, S. H. (2026). mcguard: Guardrailed Resampling for Valid Bootstrap
Confidence Intervals and Constrained Permutation Tests in Clustered
Biomedical Data. Journal of Open Source Software (under review).
```

---

## License

MIT License. See [LICENSE](LICENSE) for details.
