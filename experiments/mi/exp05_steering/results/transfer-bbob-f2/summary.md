# Transfer screen: transfer-bbob-f2 (bbob-f2)

Dial `b_exploration_pct_resid@L27`, doses [-0.4, -0.2, 0.2, 0.4] x residual norm; random direction at 0.4. Score: AOCC (20-D) of the working algorithms. `probe_expl`: exploration % read out by a ridge probe on the L15 activations, trained on the MA-BBOB corpus (CV Spearman there 0.66); concept rates among working algorithms.

## Conditions

| phase | condition | n | failed | score | exploration_pct | intensification | last_improvement | probe_expl | population | differential_evolution | particle_swarm | restart | simulated_annealing | local_search | gradient | gen_tokens |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| init | none | 32 | 0.250 | 0.010 | 56.648 | 0.203 | 0.494 | 40.972 | 0.333 | 0.083 | 0.125 | 0.000 | 0.042 | 0.083 | 0.417 | 544.781 |
| init | dial@-0.4 | 32 | 0.375 | 0.000 | 62.578 | 0.101 | 0.590 | 48.237 | 0.000 | 0.000 | 0.000 | 0.000 | 0.250 | 0.000 | 0.550 | 747.375 |
| init | dial@-0.2 | 32 | 0.188 | 0.027 | 60.030 | 0.125 | 0.481 | 42.846 | 0.115 | 0.038 | 0.154 | 0.000 | 0.192 | 0.000 | 0.538 | 543.562 |
| init | dial@+0.2 | 32 | 0.062 | 0.000 | 77.907 | 0.107 | 0.481 | 60.542 | 0.433 | 0.100 | 0.133 | 0.000 | 0.000 | 0.067 | 0.533 | 515.250 |
| init | dial@+0.4 | 32 | 0.312 | 0.013 | 66.605 | 0.094 | 0.360 | 52.060 | 0.636 | 0.136 | 0.091 | 0.000 | 0.000 | 0.000 | 0.364 | 516.250 |
| init | random@+0.4 | 32 | 0.156 | 0.001 | 46.301 | 0.253 | 0.444 | 40.640 | 0.333 | 0.074 | 0.111 | 0.000 | 0.148 | 0.037 | 0.407 | 549.781 |
| refine | none | 32 | 0.219 | 0.027 | 58.950 | 0.208 | 0.468 | 44.944 | 0.560 | 0.320 | 0.080 | 0.040 | 0.040 | 0.080 | 0.520 | 701.781 |
| refine | dial@-0.4 | 32 | 0.312 | 0.001 | 46.445 | 0.197 | 0.525 | 45.703 | 0.455 | 0.045 | 0.000 | 0.000 | 0.182 | 0.045 | 0.591 | 843.188 |
| refine | dial@-0.2 | 32 | 0.250 | 0.007 | 40.047 | 0.365 | 0.549 | 40.772 | 0.500 | 0.208 | 0.125 | 0.000 | 0.042 | 0.083 | 0.458 | 769.156 |
| refine | dial@+0.2 | 32 | 0.219 | 0.035 | 55.789 | 0.239 | 0.332 | 38.276 | 0.720 | 0.480 | 0.120 | 0.000 | 0.000 | 0.080 | 0.440 | 696.438 |
| refine | dial@+0.4 | 32 | 0.188 | 0.028 | 51.427 | 0.163 | 0.433 | 33.004 | 0.846 | 0.500 | 0.115 | 0.077 | 0.038 | 0.077 | 0.154 | 712.406 |
| refine | random@+0.4 | 32 | 0.219 | 0.039 | 50.369 | 0.333 | 0.431 | 37.616 | 0.520 | 0.240 | 0.200 | 0.000 | 0.040 | 0.080 | 0.480 | 728.125 |

## Slopes on the signed dial dose

Fixed effects per prompt x replicate; 95% CI from a bootstrap over them. `failed_f`: failure rate (all samples); other metrics: working only.

| phase | metric | n | slope | lo | hi |
|---|---|---|---|---|---|
| pooled | exploration_pct | 244 | 11.547 | -3.955 | 28.033 |
| pooled | intensification | 244 | -0.079 | -0.191 | 0.029 |
| pooled | last_improvement | 244 | -0.223 | -0.382 | -0.058 |
| pooled | probe_expl_w | 244 | 0.519 | -10.712 | 12.039 |
| pooled | score | 244 | 0.024 | 0.000 | 0.062 |
| pooled | failed_f | 320 | -0.133 | -0.297 | 0.031 |
| pooled | population | 320 | 0.664 | 0.516 | 0.820 |
| pooled | differential_evolution | 320 | 0.383 | 0.250 | 0.531 |
| pooled | particle_swarm | 320 | 0.047 | -0.039 | 0.148 |
| pooled | restart | 320 | 0.016 | -0.031 | 0.063 |
| pooled | simulated_annealing | 320 | -0.227 | -0.352 | -0.117 |
| pooled | local_search | 320 | 0.023 | -0.070 | 0.117 |
| pooled | gradient | 320 | -0.406 | -0.562 | -0.266 |
| init | exploration_pct | 122 | 17.008 | -4.953 | 41.638 |
| init | intensification | 122 | -0.047 | -0.173 | 0.063 |
| init | last_improvement | 122 | -0.210 | -0.410 | 0.017 |
| init | probe_expl_w | 122 | 14.027 | -3.962 | 31.255 |
| init | score | 122 | -0.001 | -0.015 | 0.009 |
| init | failed_f | 160 | -0.125 | -0.359 | 0.109 |
| init | population | 160 | 0.797 | 0.594 | 0.984 |
| init | differential_evolution | 160 | 0.188 | 0.078 | 0.312 |
| init | particle_swarm | 160 | 0.047 | -0.047 | 0.172 |
| init | restart | 160 | 0.000 | 0.000 | 0.000 |
| init | simulated_annealing | 160 | -0.266 | -0.453 | -0.094 |
| init | local_search | 160 | 0.031 | -0.062 | 0.125 |
| init | gradient | 160 | -0.359 | -0.579 | -0.156 |
| refine | exploration_pct | 122 | 6.526 | -16.330 | 28.695 |
| refine | intensification | 122 | -0.109 | -0.297 | 0.066 |
| refine | last_improvement | 122 | -0.234 | -0.475 | 0.010 |
| refine | probe_expl_w | 122 | -11.902 | -24.843 | 0.562 |
| refine | score | 122 | 0.047 | 0.003 | 0.110 |
| refine | failed_f | 160 | -0.141 | -0.360 | 0.078 |
| refine | population | 160 | 0.531 | 0.297 | 0.781 |
| refine | differential_evolution | 160 | 0.578 | 0.359 | 0.812 |
| refine | particle_swarm | 160 | 0.047 | -0.094 | 0.203 |
| refine | restart | 160 | 0.031 | -0.062 | 0.125 |
| refine | simulated_annealing | 160 | -0.188 | -0.344 | -0.062 |
| refine | local_search | 160 | 0.016 | -0.141 | 0.188 |
| refine | gradient | 160 | -0.453 | -0.641 | -0.250 |

## Paired change against the unsteered sample

| condition | metric | delta | lo | hi |
|---|---|---|---|---|
| dial@-0.4 | exploration_pct | -3.358 | -19.818 | 13.734 |
| dial@-0.4 | intensification | -0.052 | -0.182 | 0.083 |
| dial@-0.4 | last_improvement | 0.116 | -0.062 | 0.280 |
| dial@-0.4 | probe_expl_w | 0.392 | -8.647 | 9.389 |
| dial@-0.4 | score | -0.028 | -0.078 | 0.001 |
| dial@-0.4 | failed_f | 0.109 | -0.031 | 0.266 |
| dial@-0.2 | exploration_pct | -8.798 | -26.259 | 8.211 |
| dial@-0.2 | intensification | 0.026 | -0.119 | 0.174 |
| dial@-0.2 | last_improvement | 0.052 | -0.093 | 0.199 |
| dial@-0.2 | probe_expl_w | -1.638 | -10.316 | 7.686 |
| dial@-0.2 | score | -0.014 | -0.054 | 0.012 |
| dial@-0.2 | failed_f | -0.016 | -0.172 | 0.125 |
| dial@+0.2 | exploration_pct | 15.670 | 0.365 | 30.286 |
| dial@+0.2 | intensification | -0.080 | -0.197 | 0.033 |
| dial@+0.2 | last_improvement | -0.078 | -0.229 | 0.066 |
| dial@+0.2 | probe_expl_w | 8.467 | -1.870 | 18.785 |
| dial@+0.2 | score | -0.001 | -0.043 | 0.037 |
| dial@+0.2 | failed_f | -0.094 | -0.219 | 0.031 |
| dial@+0.4 | exploration_pct | -1.863 | -16.045 | 12.065 |
| dial@+0.4 | intensification | -0.069 | -0.205 | 0.064 |
| dial@+0.4 | last_improvement | -0.033 | -0.159 | 0.094 |
| dial@+0.4 | probe_expl_w | -3.284 | -14.833 | 7.851 |
| dial@+0.4 | score | 0.003 | -0.043 | 0.042 |
| dial@+0.4 | failed_f | 0.016 | -0.125 | 0.172 |
| random@+0.4 | exploration_pct | -8.364 | -26.647 | 9.426 |
| random@+0.4 | intensification | 0.093 | -0.066 | 0.260 |
| random@+0.4 | last_improvement | 0.002 | -0.156 | 0.162 |
| random@+0.4 | probe_expl_w | -4.492 | -14.695 | 5.202 |
| random@+0.4 | score | 0.001 | -0.043 | 0.039 |
| random@+0.4 | failed_f | -0.047 | -0.188 | 0.094 |

## Probe readout vs measured exploration

Spearman 0.38 over 296 working algorithms.
