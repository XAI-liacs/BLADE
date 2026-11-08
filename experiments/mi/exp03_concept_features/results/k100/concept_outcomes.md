# Concept features vs outcome

Out-of-fold scores (AUROC for `child_failed` / `over_budget`,
Spearman for `child_fitness`), 95% bootstrap CI over runs. Blocks:
M metadata, L concept labels + length, C line-mask-robust concept
features, A all selected concept features, S whole SAE code, T code
TF-IDF.

| target | n | model | score | ci_lo | ci_hi |
|---|---|---|---|---|---|
| child_failed | 8381 | M | 0.701 | 0.680 | 0.724 |
| child_failed | 8381 | M+L | 0.757 | 0.736 | 0.777 |
| child_failed | 8381 | M+L+C | 0.770 | 0.750 | 0.790 |
| child_failed | 8381 | M+L+A | 0.826 | 0.808 | 0.845 |
| child_failed | 8381 | M+L+S | 0.877 | 0.863 | 0.891 |
| child_failed | 8381 | M+L+T | 0.804 | 0.786 | 0.822 |
| child_failed | 8381 | M+L+T+C | 0.805 | 0.787 | 0.824 |
| child_failed | 8381 | M+L+T+S | 0.877 | 0.863 | 0.891 |
| child_failed | 8381 | M+L+C - M+L | 0.013 | 0.005 | 0.022 |
| child_failed | 8381 | M+L+A - M+L | 0.070 | 0.052 | 0.088 |
| child_failed | 8381 | M+L+S - M+L | 0.120 | 0.099 | 0.142 |
| child_failed | 8381 | M+L+T+C - M+L+T | 0.001 | -0.004 | 0.006 |
| child_failed | 8381 | M+L+T+S - M+L+T | 0.074 | 0.058 | 0.089 |
| over_budget | 6968 | M | 0.909 | 0.861 | 0.936 |
| over_budget | 6968 | M+L | 0.921 | 0.879 | 0.946 |
| over_budget | 6968 | M+L+C | 0.925 | 0.884 | 0.949 |
| over_budget | 6968 | M+L+A | 0.936 | 0.904 | 0.955 |
| over_budget | 6968 | M+L+S | 0.938 | 0.903 | 0.959 |
| over_budget | 6968 | M+L+T | 0.935 | 0.902 | 0.955 |
| over_budget | 6968 | M+L+T+C | 0.937 | 0.903 | 0.956 |
| over_budget | 6968 | M+L+T+S | 0.939 | 0.905 | 0.959 |
| over_budget | 6968 | M+L+C - M+L | 0.004 | -0.002 | 0.011 |
| over_budget | 6968 | M+L+A - M+L | 0.015 | 0.006 | 0.028 |
| over_budget | 6968 | M+L+S - M+L | 0.016 | 0.007 | 0.031 |
| over_budget | 6968 | M+L+T+C - M+L+T | 0.002 | -0.005 | 0.008 |
| over_budget | 6968 | M+L+T+S - M+L+T | 0.004 | -0.002 | 0.012 |
| child_fitness | 6968 | M | 0.736 | 0.705 | 0.763 |
| child_fitness | 6968 | M+L | 0.746 | 0.713 | 0.775 |
| child_fitness | 6968 | M+L+C | 0.747 | 0.714 | 0.775 |
| child_fitness | 6968 | M+L+A | 0.752 | 0.719 | 0.780 |
| child_fitness | 6968 | M+L+S | 0.761 | 0.729 | 0.788 |
| child_fitness | 6968 | M+L+T | 0.749 | 0.717 | 0.778 |
| child_fitness | 6968 | M+L+T+C | 0.750 | 0.717 | 0.779 |
| child_fitness | 6968 | M+L+T+S | 0.761 | 0.729 | 0.788 |
| child_fitness | 6968 | M+L+C - M+L | 0.001 | -0.002 | 0.004 |
| child_fitness | 6968 | M+L+A - M+L | 0.006 | 0.002 | 0.009 |
| child_fitness | 6968 | M+L+S - M+L | 0.016 | 0.009 | 0.022 |
| child_fitness | 6968 | M+L+T+C - M+L+T | 0.001 | -0.002 | 0.003 |
| child_fitness | 6968 | M+L+T+S - M+L+T | 0.012 | 0.004 | 0.018 |

## Robust concept features within the solutions that use the concept

| target | concept | layer | feature | n_uses | stat | value | ci_lo | ci_hi |
|---|---|---|---|---|---|---|---|---|
| child_failed | adaptive_parameters | 15 | 14313 | 948 | auroc | 0.522 | 0.458 | 0.588 |
| child_failed | covariance | 15 | 5776 | 1240 | auroc | 0.536 | 0.472 | 0.606 |
| child_failed | gradient | 15 | 6720 | 5317 | auroc | 0.423 | 0.390 | 0.456 |
| child_failed | local_search | 27 | 26058 | 812 | auroc | 0.359 | 0.278 | 0.429 |
| child_failed | particle_swarm | 27 | 38837 | 1364 | auroc | 0.468 | 0.428 | 0.504 |
| child_failed | population | 9 | 22950 | 6036 | auroc | 0.640 | 0.610 | 0.666 |
| child_failed | restart | 27 | 44846 | 496 | auroc | 0.645 | 0.484 | 0.828 |
| child_failed | step_size_adaptation | 9 | 3169 | 3644 | auroc | 0.474 | 0.433 | 0.517 |
| child_failed | surrogate | 27 | 62130 | 231 | auroc | 0.060 | 0.000 | 0.402 |
| over_budget | adaptive_parameters | 15 | 14313 | 852 | auroc | 0.328 | 0.232 | 0.449 |
| over_budget | covariance | 15 | 5776 | 977 | auroc | 0.500 | 0.366 | 0.626 |
| over_budget | gradient | 15 | 6720 | 4483 | auroc | 0.605 | 0.480 | 0.723 |
| over_budget | local_search | 27 | 26058 | 700 | auroc | 0.486 | 0.301 | 0.676 |
| over_budget | particle_swarm | 27 | 38837 | 1183 | auroc | 0.372 | 0.295 | 0.489 |
| over_budget | population | 9 | 22950 | 4954 | auroc | 0.467 | 0.389 | 0.535 |
| over_budget | restart | 27 | 44846 | 249 | auroc | 0.500 | 0.346 | 0.785 |
| over_budget | step_size_adaptation | 9 | 3169 | 3089 | auroc | 0.493 | 0.382 | 0.598 |
| over_budget | surrogate | 27 | 62130 | 16 | auroc | nan | nan | nan |
| child_fitness | adaptive_parameters | 15 | 14313 | 852 | spearman | -0.100 | -0.178 | -0.038 |
| child_fitness | covariance | 15 | 5776 | 977 | spearman | 0.001 | -0.137 | 0.129 |
| child_fitness | gradient | 15 | 6720 | 4483 | spearman | -0.077 | -0.175 | 0.024 |
| child_fitness | local_search | 27 | 26058 | 700 | spearman | -0.002 | -0.151 | 0.172 |
| child_fitness | particle_swarm | 27 | 38837 | 1183 | spearman | -0.220 | -0.315 | -0.130 |
| child_fitness | population | 9 | 22950 | 4954 | spearman | -0.090 | -0.175 | -0.009 |
| child_fitness | restart | 27 | 44846 | 249 | spearman | 0.175 | -0.021 | 0.320 |
| child_fitness | step_size_adaptation | 9 | 3169 | 3089 | spearman | -0.027 | -0.153 | 0.101 |
| child_fitness | surrogate | 27 | 62130 | 16 | spearman | -0.035 | -0.324 | 1.000 |
