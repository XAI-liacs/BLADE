# Transfer screen: transfer-bbob-f20 (bbob-f20)

Dial `b_exploration_pct_resid@L27`, doses [-0.4, -0.2, 0.2, 0.4] x residual norm; random direction at 0.4. Score: AOCC (20-D) of the working algorithms. `probe_expl`: exploration % read out by a ridge probe on the L15 activations, trained on the MA-BBOB corpus (CV Spearman there 0.66); concept rates among working algorithms.

## Conditions

| phase | condition | n | failed | score | exploration_pct | intensification | last_improvement | probe_expl | population | differential_evolution | particle_swarm | restart | simulated_annealing | local_search | gradient | gen_tokens |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| init | none | 32 | 0.156 | 0.057 | 64.968 | 0.213 | 0.485 | 58.039 | 0.333 | 0.037 | 0.222 | 0.000 | 0.000 | 0.037 | 0.556 | 522.875 |
| init | dial@-0.4 | 32 | 0.406 | 0.030 | 45.718 | 0.357 | 0.577 | 39.694 | 0.053 | 0.000 | 0.000 | 0.000 | 0.158 | 0.053 | 0.579 | 715.500 |
| init | dial@-0.2 | 32 | 0.250 | 0.064 | 37.279 | 0.316 | 0.391 | 49.573 | 0.125 | 0.000 | 0.167 | 0.000 | 0.000 | 0.000 | 0.542 | 527.094 |
| init | dial@+0.2 | 32 | 0.062 | 0.032 | 71.205 | 0.133 | 0.566 | 60.672 | 0.333 | 0.067 | 0.133 | 0.100 | 0.000 | 0.033 | 0.567 | 490.219 |
| init | dial@+0.4 | 32 | 0.250 | 0.084 | 55.065 | 0.270 | 0.386 | 55.855 | 0.708 | 0.167 | 0.083 | 0.083 | 0.000 | 0.042 | 0.333 | 515.094 |
| init | random@+0.4 | 32 | 0.188 | 0.060 | 49.331 | 0.225 | 0.455 | 48.322 | 0.269 | 0.077 | 0.231 | 0.000 | 0.115 | 0.000 | 0.385 | 559.125 |
| refine | none | 32 | 0.125 | 0.053 | 62.153 | 0.144 | 0.530 | 54.803 | 0.321 | 0.000 | 0.250 | 0.000 | 0.036 | 0.107 | 0.750 | 663.406 |
| refine | dial@-0.4 | 32 | 0.344 | 0.033 | 58.240 | 0.225 | 0.515 | 40.912 | 0.095 | 0.000 | 0.048 | 0.000 | 0.048 | 0.095 | 0.857 | 865.938 |
| refine | dial@-0.2 | 32 | 0.125 | 0.039 | 66.316 | 0.206 | 0.456 | 53.434 | 0.250 | 0.000 | 0.179 | 0.000 | 0.000 | 0.071 | 0.750 | 709.812 |
| refine | dial@+0.2 | 32 | 0.156 | 0.027 | 69.695 | 0.199 | 0.559 | 55.646 | 0.667 | 0.074 | 0.148 | 0.000 | 0.000 | 0.074 | 0.704 | 696.656 |
| refine | dial@+0.4 | 32 | 0.250 | 0.047 | 63.005 | 0.224 | 0.521 | 54.026 | 0.792 | 0.125 | 0.083 | 0.042 | 0.000 | 0.083 | 0.500 | 715.469 |
| refine | random@+0.4 | 32 | 0.094 | 0.033 | 70.422 | 0.146 | 0.546 | 51.288 | 0.379 | 0.000 | 0.172 | 0.000 | 0.000 | 0.000 | 0.690 | 725.594 |

## Slopes on the signed dial dose

Fixed effects per prompt x replicate; 95% CI from a bootstrap over them. `failed_f`: failure rate (all samples); other metrics: working only.

| phase | metric | n | slope | lo | hi |
|---|---|---|---|---|---|
| pooled | exploration_pct | 250 | 20.453 | -1.524 | 41.269 |
| pooled | intensification | 252 | -0.116 | -0.288 | 0.061 |
| pooled | last_improvement | 252 | 0.009 | -0.147 | 0.167 |
| pooled | probe_expl_w | 252 | 14.232 | 2.798 | 24.898 |
| pooled | score | 252 | 0.017 | -0.016 | 0.047 |
| pooled | failed_f | 320 | -0.164 | -0.336 | 0.008 |
| pooled | population | 320 | 0.836 | 0.688 | 0.977 |
| pooled | differential_evolution | 320 | 0.234 | 0.133 | 0.352 |
| pooled | particle_swarm | 320 | 0.016 | -0.102 | 0.133 |
| pooled | restart | 320 | 0.070 | 0.008 | 0.148 |
| pooled | simulated_annealing | 320 | -0.102 | -0.188 | -0.031 |
| pooled | local_search | 320 | -0.023 | -0.078 | 0.023 |
| pooled | gradient | 320 | -0.352 | -0.531 | -0.172 |
| init | exploration_pct | 123 | 28.900 | -3.444 | 60.873 |
| init | intensification | 124 | -0.193 | -0.480 | 0.094 |
| init | last_improvement | 124 | -0.067 | -0.249 | 0.118 |
| init | probe_expl_w | 124 | 17.500 | -2.740 | 36.086 |
| init | score | 124 | 0.034 | -0.019 | 0.082 |
| init | failed_f | 160 | -0.250 | -0.500 | 0.016 |
| init | population | 160 | 0.797 | 0.562 | 1.016 |
| init | differential_evolution | 160 | 0.250 | 0.109 | 0.422 |
| init | particle_swarm | 160 | 0.078 | -0.047 | 0.235 |
| init | restart | 160 | 0.109 | 0.000 | 0.250 |
| init | simulated_annealing | 160 | -0.125 | -0.250 | -0.031 |
| init | local_search | 160 | -0.047 | -0.156 | 0.031 |
| init | gradient | 160 | -0.312 | -0.594 | -0.031 |
| refine | exploration_pct | 127 | 12.556 | -17.018 | 38.012 |
| refine | intensification | 128 | -0.045 | -0.240 | 0.159 |
| refine | last_improvement | 128 | 0.079 | -0.168 | 0.304 |
| refine | probe_expl_w | 128 | 11.185 | -3.090 | 25.512 |
| refine | score | 128 | 0.001 | -0.035 | 0.038 |
| refine | failed_f | 160 | -0.078 | -0.328 | 0.156 |
| refine | population | 160 | 0.875 | 0.687 | 1.047 |
| refine | differential_evolution | 160 | 0.219 | 0.078 | 0.375 |
| refine | particle_swarm | 160 | -0.047 | -0.234 | 0.141 |
| refine | restart | 160 | 0.031 | 0.000 | 0.094 |
| refine | simulated_annealing | 160 | -0.078 | -0.188 | 0.000 |
| refine | local_search | 160 | -0.000 | -0.047 | 0.047 |
| refine | gradient | 160 | -0.391 | -0.609 | -0.172 |

## Paired change against the unsteered sample

| condition | metric | delta | lo | hi |
|---|---|---|---|---|
| dial@-0.4 | exploration_pct | -14.113 | -32.411 | 5.066 |
| dial@-0.4 | intensification | 0.167 | 0.017 | 0.316 |
| dial@-0.4 | last_improvement | -0.009 | -0.158 | 0.133 |
| dial@-0.4 | probe_expl_w | -11.003 | -18.387 | -3.552 |
| dial@-0.4 | score | -0.029 | -0.057 | -0.000 |
| dial@-0.4 | failed_f | 0.234 | 0.078 | 0.391 |
| dial@-0.2 | exploration_pct | -14.761 | -28.678 | -0.382 |
| dial@-0.2 | intensification | 0.082 | -0.026 | 0.190 |
| dial@-0.2 | last_improvement | -0.131 | -0.244 | -0.020 |
| dial@-0.2 | probe_expl_w | -5.773 | -12.528 | 1.100 |
| dial@-0.2 | score | -0.001 | -0.025 | 0.021 |
| dial@-0.2 | failed_f | 0.047 | -0.062 | 0.156 |
| dial@+0.2 | exploration_pct | 7.606 | -5.943 | 21.195 |
| dial@+0.2 | intensification | -0.031 | -0.139 | 0.082 |
| dial@+0.2 | last_improvement | 0.038 | -0.078 | 0.154 |
| dial@+0.2 | probe_expl_w | 3.746 | -6.488 | 14.049 |
| dial@+0.2 | score | -0.019 | -0.041 | 0.003 |
| dial@+0.2 | failed_f | -0.031 | -0.156 | 0.094 |
| dial@+0.4 | exploration_pct | -1.345 | -16.429 | 12.329 |
| dial@+0.4 | intensification | 0.109 | 0.006 | 0.215 |
| dial@+0.4 | last_improvement | -0.088 | -0.210 | 0.039 |
| dial@+0.4 | probe_expl_w | -0.694 | -10.122 | 8.482 |
| dial@+0.4 | score | 0.013 | -0.014 | 0.041 |
| dial@+0.4 | failed_f | 0.109 | -0.016 | 0.250 |
| random@+0.4 | exploration_pct | -7.032 | -21.776 | 8.256 |
| random@+0.4 | intensification | 0.046 | -0.063 | 0.153 |
| random@+0.4 | last_improvement | -0.062 | -0.199 | 0.074 |
| random@+0.4 | probe_expl_w | -7.220 | -15.505 | 0.031 |
| random@+0.4 | score | 0.002 | -0.023 | 0.027 |
| random@+0.4 | failed_f | 0.000 | -0.109 | 0.109 |

## Probe readout vs measured exploration

Spearman 0.41 over 305 working algorithms.
