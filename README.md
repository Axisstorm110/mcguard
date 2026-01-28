![TestPyPI](https://img.shields.io/badge/TestPyPI-mcguard-blue)


\# mcguard



\*\*mcguard\*\* is a lightweight Python library for \*statistically valid resampling\* in datasets with

\*\*non-independent observations\*\*, common in biomedical research and ML evaluation.



It helps you avoid a very common mistake:

> treating repeated measurements (e.g., multiple rows per patient) as independent.



This mistake often produces \*\*overconfident\*\* bootstrap confidence intervals and misleading p-values.



---



\## Why it matters



If you have clustered data like:



\- 10 measurements per patient

\- multiple timepoints per subject

\- repeated trials per unit

\- grouped samples per site/lab



Then the \*\*iid bootstrap\*\* (resampling rows) usually underestimates uncertainty.



`mcguard` lets you \*\*resample correctly\*\* using cluster-aware/guardrailed bootstrapping and constrained permutation tests.



---



\## Features (v0.1.0)



✅ Bootstrap confidence intervals (percentile CI)

\- iid row bootstrap (default)

\- cluster/unit-level bootstrap (e.g., patient-level resampling)



✅ Constrained permutation tests

\- global permutation

\- within-strata permutation (`strata=...`)

\- within-pair permutation (`paired=...`)



✅ Reproducible results with seeds  

✅ Minimal dependencies (`numpy`)  

✅ Unit tests included



---



\## Installation



For development (editable install):



```bash

pip install -e .



