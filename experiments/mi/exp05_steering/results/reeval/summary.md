# Re-evaluation with scipy and scikit-learn

Existing code re-evaluated with `scipy==1.18.0`, `scikit-learn==1.9.0` added to the sandbox (same traced problem as the original evaluation). `aocc` counts failures as 0; `concept_working` is the rate of the target concept among the working algorithms.

## Elicited algorithms (corpus, 24 per target)

| group | n | failed_before | failed_after | aocc | aocc_working | concept | concept_rate | concept_working |
|---|---|---|---|---|---|---|---|---|
| none | 24 | 0.250 | 0.250 | 0.209 | 0.279 |  | nan | nan |
| archive | 24 | 0.875 | 0.875 | 0.033 | 0.261 | archive | 1.000 | 1.000 |
| quasi_random_init | 24 | 1.000 | 0.292 | 0.184 | 0.260 | quasi_random_init | 0.958 | 1.000 |
| surrogate | 24 | 0.958 | 0.875 | 0.029 | 0.235 | surrogate | 1.000 | 1.000 |

Most frequent remaining errors:

| group | err | n |
|---|---|---|
| archive | ValueError: setting an array element with a sequence. The requested array has an inhomogen | 15 |
| surrogate | Evaluation timed out after 120 seconds. | 11 |
| archive | ValueError: a must be 1-dimensional | 2 |
| archive | On line: return np.clip(x, func.bounds.lb, func.bounds.ub) | 1 |
| archive | On line: x += np.random.normal(0, 0.1, self.dim) | 1 |
| quasi_random_init | On line: from pyDOE import lhs  # For Latin Hypercube Sampling | 1 |
| quasi_random_init | On line: from pySOT import LHS  # Assuming pySOT is used for Latin Hypercube Sampling | 1 |
| quasi_random_init | On line: return self.func(x) | 1 |
| quasi_random_init | On line: samples = self.sobol.sample(self.num_initial_points) | 1 |
| surrogate | AttributeError: module 'numpy' has no attribute 'erf' | 1 |
| surrogate | On line: ei_values[i] = (mean_i - self.f_opt) * np.sqrt(1 / (2 * np.pi)) * np.exp(-0.5 * ( | 1 |
| surrogate | On line: if f < self.f_opt: | 1 |

## Steered children (`contrast-c`, 16 per condition)

`d_*`: paired change against the unsteered child of the same prompt.

| group | n | failed_before | failed_after | aocc | aocc_working | concept | concept_rate | concept_working | d_failed_before | d_failed_after | d_aocc_after |
|---|---|---|---|---|---|---|---|---|---|---|---|
| c_quasi_random_init@L15@+0.3 | 16 | 0.188 | 0.188 | 0.338 | 0.416 | quasi_random_init | 0.000 | 0.000 | 0.062 | 0.062 | -0.010 |
| c_archive@L15@+0.5 | 16 | 0.750 | 0.750 | 0.075 | 0.300 | archive | 0.062 | 0.000 | 0.625 | 0.625 | -0.273 |
| c_quasi_random_init@L15@+0.5 | 16 | 0.188 | 0.188 | 0.273 | 0.336 | quasi_random_init | 0.000 | 0.000 | 0.062 | 0.062 | -0.075 |
| c_surrogate@L15@+0.15 | 16 | 0.312 | 0.312 | 0.222 | 0.324 | surrogate | 0.062 | 0.000 | 0.188 | 0.188 | -0.126 |
| c_surrogate@L15@+0.5 | 16 | 0.688 | 0.688 | 0.161 | 0.515 | surrogate | 0.438 | 0.000 | 0.562 | 0.562 | -0.187 |
| c_archive@L15@+0.3 | 16 | 0.375 | 0.375 | 0.266 | 0.426 | archive | 0.000 | 0.000 | 0.250 | 0.250 | -0.082 |
| c_surrogate@L15@+0.3 | 16 | 0.438 | 0.438 | 0.234 | 0.416 | surrogate | 0.125 | 0.000 | 0.312 | 0.312 | -0.114 |
| c_archive@L15@+0.15 | 16 | 0.250 | 0.250 | 0.290 | 0.386 | archive | 0.000 | 0.000 | 0.125 | 0.125 | -0.058 |
| c_quasi_random_init@L15@+0.15 | 16 | 0.125 | 0.125 | 0.319 | 0.364 | quasi_random_init | 0.000 | 0.000 | 0.000 | 0.000 | -0.030 |
| none | 16 | 0.125 | 0.125 | 0.348 | 0.398 |  | nan | nan | nan | nan | nan |
