import numpy as np
from mcguard import ResamplingPlan, bootstrap_ci, permutation_test, metrics

rng = np.random.default_rng(0)
n_patients = 60
m = 10
patient_id = np.repeat(np.arange(n_patients), m)

treat_patient = rng.integers(0, 2, size=n_patients)
group = np.repeat(treat_patient, m)

patient_effect = rng.normal(0, 1.0, size=n_patients)
y = 5 + 0.6 * group + patient_effect[patient_id] + rng.normal(0, 0.8, size=n_patients * m)

naive = bootstrap_ci(metrics.mean_diff, y=y, group=group, plan=ResamplingPlan(), B=2000, seed=1)
cluster = bootstrap_ci(metrics.mean_diff, y=y, group=group, plan=ResamplingPlan(unit=patient_id), B=2000, seed=1)

print("Naive iid bootstrap CI:", (naive.ci_lo, naive.ci_hi))
print("Patient-level bootstrap CI:", (cluster.ci_lo, cluster.ci_hi))

y_patient = np.array([y[patient_id == pid].mean() for pid in range(n_patients)])
g_patient = treat_patient.copy()
perm = permutation_test(metrics.mean_diff, y=y_patient, group=g_patient, P=5000, seed=2)
print("Permutation p-value (patient-level):", perm.p_value)
