---
title: 'mcguard: Guardrailed Resampling for Valid Bootstrap Confidence Intervals and Constrained Permutation Tests in Clustered Biomedical Data'
tags:
  - Python
  - statistics
  - bootstrap
  - permutation tests
  - biomedical informatics
  - clustered data
  - repeated measures
authors:
  - name: Syed Hamzah Rizvi
    orcid: 0009-0004-9864-6607
    affiliation: 1
affiliations:
  - name: Independent Researcher
    index: 1
date: 2026
bibliography: paper.bib
---

# Summary

`mcguard` is a lightweight Python library for statistically valid resampling in datasets
where observations are not independent — a condition that is common in biomedical
research, clinical informatics, and machine learning evaluation, yet frequently
overlooked. The library provides cluster-aware bootstrap confidence intervals and
constrained permutation tests that respect the dependence structure of any dataset
through a single unified `ResamplingPlan` interface. When observations are clustered —
for example, multiple measurements per patient, repeated trials per subject, or samples
grouped by site — standard resampling methods that treat each row as independent
systematically underestimate uncertainty, producing overconfident intervals and inflated
test statistics [@field2007bootstrapping]. `mcguard` addresses this directly with
minimal dependencies (`numpy` only) and a clean, composable API designed to slot into
existing analysis workflows without friction.

# Statement of Need

A pervasive and well-documented error in biomedical data analysis is applying
independent-and-identically-distributed (iid) resampling to clustered or
repeated-measures datasets [@cameron2008bootstrap]. If a study collects ten longitudinal
measurements per patient, a naive row-level bootstrap treats those ten rows as ten
independent observations, effectively inflating the apparent sample size and producing
confidence intervals that are far too narrow. The same problem applies to multi-site
clinical trials where observations are clustered by site, laboratory assays with
repeated biological replicates, and machine learning evaluation pipelines where the same
patient appears across multiple prediction records.

Existing Python tools do not adequately address this for biomedical practitioners:

- `scipy.stats` provides basic bootstrap and permutation functionality but offers no
  mechanism for cluster-aware or stratified resampling [@virtanen2020scipy].
- `bootstrapped` (Meta Research) implements variance reduction techniques but assumes
  iid observations.
- `arch` includes block bootstrap for time series [@sheppard2023arch] but is oriented
  toward econometrics and does not support arbitrary unit-level clustering with optional
  stratification.
- `pingouin` provides permutation tests for paired designs [@vallat2018pingouin] but
  does not generalise to arbitrary cluster structures or combined stratified-clustered
  sampling.

`mcguard` fills this gap by providing a single composable `ResamplingPlan` object that
encodes the dependence structure of any dataset — clusters, strata, time blocks, or
paired designs — and applies it consistently across both bootstrap and permutation
procedures. The target audience is researchers in biomedical informatics, clinical data
science, epidemiology, and computational biology who work with electronic health records,
longitudinal cohort studies, or multi-site trials and need resampling tools that respect
the structure of their data.

# State of the Field

Resampling methods for dependent data have been studied extensively in the statistical
literature. The cluster bootstrap resamples at the level of independent units rather
than individual observations [@cameron2008bootstrap], and stratified variants
[@davison1997bootstrap] further constrain resampling within pre-specified groups to
preserve covariate balance across resamples. Block bootstraps [@kunsch1989jackknife]
address temporal dependence in time series. Constrained permutation tests — permuting
labels only within matched pairs or within strata — are standard in matched cohort
studies and multi-site trials [@good2005permutation].

Despite this literature, software implementations remain fragmented across the Python
ecosystem. The R language provides substantially better coverage through packages such
as `boot` [@canty2022boot] and `coin` [@hothorn2008coin], which support stratified and
clustered resampling through well-maintained APIs. Python lacks an equivalent unified
library. `mcguard` is designed to fill this gap with a particular focus on biomedical
use cases where clustered data structures arise naturally from study design.

# Software Design

## Core Abstraction: ResamplingPlan

The central abstraction in `mcguard` is the `ResamplingPlan` dataclass, which encodes
the dependence structure of a dataset in a single, reusable, validated object:

```python
from mcguard import ResamplingPlan

# Cluster bootstrap: resample whole patients, not individual rows
plan = ResamplingPlan(unit=patient_id)

# Stratified cluster bootstrap: resample patients independently within each site
plan = ResamplingPlan(unit=patient_id, strata=site_id)

# Paired permutation: permute labels only within each matched pair
plan = ResamplingPlan(paired=pair_id)
```

A `ResamplingPlan()` with no arguments falls back to iid resampling, making `mcguard`
a drop-in replacement for standard procedures. The `validate()` method checks that all
provided arrays have consistent lengths and raises informative errors before any
computation begins.

## Bootstrap Confidence Intervals

`bootstrap_ci` accepts any scalar statistic function and applies it under the resampling
plan. When a `unit` array is provided, entire units are resampled (with replacement) as
atomic blocks rather than individual rows. Optional `strata` and `blocks` arrays
partition units into independent buckets so that resampling happens within each bucket
separately, preserving covariate distributions across resamples [@efron1994introduction]:

```python
from mcguard import bootstrap_ci, ResamplingPlan, metrics

result = bootstrap_ci(
    metrics.mean_diff,
    y=outcome, group=treatment,
    plan=ResamplingPlan(unit=patient_id, strata=site_id),
    B=2000, alpha=0.05, seed=42,
)
print(result.ci_lo, result.ci_hi)  # BootstrapResult dataclass
```

## Constrained Permutation Tests

`permutation_test` permutes group labels while respecting constraints in the plan. With
`paired` set, labels are permuted only within each matched pair; with `strata` set,
permutation is restricted within each stratum. All three alternative hypotheses
(`two-sided`, `greater`, `less`) are supported, with p-values computed using the +1
continuity correction recommended by Phipson & Smyth [@phipson2010permutation] to
avoid p = 0:

```python
from mcguard import permutation_test, ResamplingPlan, metrics

result = permutation_test(
    metrics.mean_diff,
    y=outcome, group=treatment,
    plan=ResamplingPlan(strata=site_id),
    P=10000, seed=42, alternative="two-sided",
)
print(result.p_value)  # PermutationResult dataclass
```

## Built-in Metrics

`mcguard.metrics` provides statistic functions compatible with both procedures:
`mean`, `median`, `mean_diff` (difference in group means), `median_diff`, and
`cohens_d` (pooled standard deviation effect size). Any user-defined function with
the signature `f(y)` or `f(y, group)` works equally well, giving full flexibility
for custom effect sizes.

# Research Impact Statement

`mcguard` was developed to address a concrete reproducibility risk encountered while
building a hospital readmission prediction pipeline on electronic health records, where
multiple prediction records per patient violated the independence assumption of standard
bootstrap confidence intervals. The cluster-aware bootstrap and constrained permutation
tests it provides are directly applicable to benchmarking studies that compare model
performance across patient subgroups in clinical AI research, where standard independence
assumptions are routinely violated by the repeated-measures structure of EHR data.
The library is currently under consideration for citation and methodological
incorporation in a methods paper on EHR data quality and reproducibility in clinical AI,
in collaboration with researchers at the MIT Laboratory of Computational Physiology.

More broadly, `mcguard` addresses a reproducibility risk that affects a substantial
fraction of published biomedical ML evaluations: any study that reports bootstrap
confidence intervals or permutation p-values on datasets with repeated patient records,
without accounting for clustering, may be systematically overconfident in reported
uncertainty — a problem the statistical literature has documented extensively
[@field2007bootstrapping; @cameron2008bootstrap] but that lacks accessible tooling
in Python.

# AI Usage Disclosure

Claude (Anthropic, claude-sonnet-4-6) was used for assistance with paper drafting,
code documentation, refactoring suggestions, and README preparation. All scientific
content, algorithmic design decisions, statistical methodology choices, and validation
of all outputs were made and verified by the author. The author retains full
responsibility for the accuracy and originality of all submitted materials.

# Acknowledgements

The author thanks the open-source scientific Python community, in particular the
maintainers of NumPy [@harris2020numpy], whose foundational work makes lightweight
libraries like `mcguard` possible.

# References
