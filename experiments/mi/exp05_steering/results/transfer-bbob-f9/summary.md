# Transfer screen: transfer-bbob-f9 (bbob-f9)

Dial `b_exploration_pct_resid@L27`, doses [-0.4, -0.2, 0.2, 0.4] x residual norm; random direction at 0.4. Score: AOCC (20-D) of the working algorithms. `probe_expl`: exploration % read out by a ridge probe on the L15 activations, trained on the MA-BBOB corpus (CV Spearman there 0.66); concept rates among working algorithms.

## Conditions

| phase | condition | n | failed | score | exploration_pct | intensification | last_improvement | probe_expl | population | differential_evolution | particle_swarm | restart | simulated_annealing | local_search | gradient | gen_tokens |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| init | none | 32 | 0.188 | 0.123 | 74.879 | 0.106 | 0.528 | 52.069 | 0.308 | 0.077 | 0.115 | 0.000 | 0.000 | 0.038 | 0.654 | 504.188 |
| init | dial@-0.4 | 32 | 0.219 | 0.088 | 56.003 | 0.275 | 0.636 | 41.487 | 0.120 | 0.000 | 0.000 | 0.000 | 0.120 | 0.040 | 0.560 | 765.406 |
| init | dial@-0.2 | 32 | 0.281 | 0.109 | 66.880 | 0.144 | 0.600 | 49.238 | 0.217 | 0.043 | 0.043 | 0.000 | 0.174 | 0.000 | 0.609 | 578.406 |
| init | dial@+0.2 | 32 | 0.188 | 0.167 | 63.571 | 0.280 | 0.446 | 63.154 | 0.346 | 0.192 | 0.115 | 0.000 | 0.000 | 0.000 | 0.654 | 483.125 |
| init | dial@+0.4 | 32 | 0.312 | 0.179 | 42.457 | 0.345 | 0.401 | 54.211 | 0.682 | 0.227 | 0.091 | 0.000 | 0.000 | 0.000 | 0.227 | 509.812 |
| init | random@+0.4 | 32 | 0.281 | 0.159 | 58.181 | 0.250 | 0.524 | 48.036 | 0.478 | 0.087 | 0.130 | 0.000 | 0.087 | 0.000 | 0.391 | 561.531 |
| refine | none | 32 | 0.188 | 0.147 | 51.308 | 0.419 | 0.507 | 51.928 | 0.462 | 0.115 | 0.154 | 0.000 | 0.038 | 0.000 | 0.692 | 699.625 |
| refine | dial@-0.4 | 32 | 0.375 | 0.158 | 45.226 | 0.297 | 0.420 | 50.433 | 0.400 | 0.000 | 0.150 | 0.050 | 0.000 | 0.050 | 0.550 | 945.688 |
| refine | dial@-0.2 | 32 | 0.094 | 0.156 | 49.592 | 0.223 | 0.474 | 46.327 | 0.517 | 0.103 | 0.138 | 0.000 | 0.034 | 0.000 | 0.586 | 722.406 |
| refine | dial@+0.2 | 32 | 0.281 | 0.157 | 61.554 | 0.189 | 0.527 | 48.889 | 0.696 | 0.130 | 0.130 | 0.000 | 0.000 | 0.000 | 0.565 | 716.531 |
| refine | dial@+0.4 | 32 | 0.406 | 0.169 | 67.972 | 0.164 | 0.517 | 38.765 | 0.895 | 0.211 | 0.053 | 0.053 | 0.000 | 0.000 | 0.632 | 739.250 |
| refine | random@+0.4 | 32 | 0.219 | 0.155 | 53.880 | 0.293 | 0.387 | 44.383 | 0.680 | 0.200 | 0.200 | 0.000 | 0.040 | 0.000 | 0.600 | 759.625 |

## Slopes on the signed dial dose

Fixed effects per prompt x replicate; 95% CI from a bootstrap over them. `failed_f`: failure rate (all samples); other metrics: working only.

| phase | metric | n | slope | lo | hi |
|---|---|---|---|---|---|
| pooled | exploration_pct | 238 | 4.002 | -17.609 | 24.707 |
| pooled | intensification | 238 | 0.015 | -0.180 | 0.214 |
| pooled | last_improvement | 238 | -0.071 | -0.258 | 0.101 |
| pooled | probe_expl_w | 239 | 8.370 | -4.280 | 21.071 |
| pooled | score | 239 | 0.058 | 0.014 | 0.102 |
| pooled | failed_f | 320 | 0.086 | -0.070 | 0.250 |
| pooled | population | 320 | 0.578 | 0.398 | 0.750 |
| pooled | differential_evolution | 320 | 0.234 | 0.117 | 0.359 |
| pooled | particle_swarm | 320 | 0.031 | -0.047 | 0.117 |
| pooled | restart | 320 | 0.070 | 0.000 | 0.148 |
| pooled | simulated_annealing | 320 | -0.133 | -0.211 | -0.063 |
| pooled | local_search | 320 | -0.031 | -0.094 | 0.031 |
| pooled | gradient | 320 | -0.164 | -0.360 | 0.031 |
| init | exploration_pct | 121 | -14.921 | -48.749 | 17.148 |
| init | intensification | 121 | 0.148 | -0.154 | 0.446 |
| init | last_improvement | 121 | -0.290 | -0.530 | -0.053 |
| init | probe_expl_w | 122 | 21.069 | 3.190 | 38.183 |
| init | score | 122 | 0.115 | 0.062 | 0.173 |
| init | failed_f | 160 | 0.047 | -0.188 | 0.266 |
| init | population | 160 | 0.594 | 0.312 | 0.875 |
| init | differential_evolution | 160 | 0.297 | 0.125 | 0.484 |
| init | particle_swarm | 160 | 0.078 | -0.078 | 0.250 |
| init | restart | 160 | 0.000 | 0.000 | 0.000 |
| init | simulated_annealing | 160 | -0.250 | -0.406 | -0.125 |
| init | local_search | 160 | -0.000 | -0.094 | 0.094 |
| init | gradient | 160 | -0.297 | -0.610 | 0.031 |
| refine | exploration_pct | 117 | 25.058 | 0.916 | 47.578 |
| refine | intensification | 117 | -0.132 | -0.359 | 0.109 |
| refine | last_improvement | 117 | 0.172 | -0.076 | 0.416 |
| refine | probe_expl_w | 117 | -6.035 | -22.033 | 8.674 |
| refine | score | 117 | -0.007 | -0.061 | 0.048 |
| refine | failed_f | 160 | 0.125 | -0.094 | 0.344 |
| refine | population | 160 | 0.562 | 0.344 | 0.797 |
| refine | differential_evolution | 160 | 0.172 | 0.000 | 0.344 |
| refine | particle_swarm | 160 | -0.016 | -0.047 | 0.000 |
| refine | restart | 160 | 0.141 | 0.000 | 0.281 |
| refine | simulated_annealing | 160 | -0.016 | -0.047 | 0.000 |
| refine | local_search | 160 | -0.062 | -0.156 | 0.000 |
| refine | gradient | 160 | -0.031 | -0.219 | 0.156 |

## Paired change against the unsteered sample

| condition | metric | delta | lo | hi |
|---|---|---|---|---|
| dial@-0.4 | exploration_pct | -6.070 | -23.318 | 11.483 |
| dial@-0.4 | intensification | -0.023 | -0.164 | 0.114 |
| dial@-0.4 | last_improvement | -0.035 | -0.164 | 0.105 |
| dial@-0.4 | probe_expl_w | -7.701 | -15.874 | 0.282 |
| dial@-0.4 | score | -0.012 | -0.049 | 0.024 |
| dial@-0.4 | failed_f | 0.109 | -0.031 | 0.250 |
| dial@-0.2 | exploration_pct | -8.224 | -21.459 | 5.325 |
| dial@-0.2 | intensification | -0.041 | -0.156 | 0.074 |
| dial@-0.2 | last_improvement | 0.037 | -0.086 | 0.166 |
| dial@-0.2 | probe_expl_w | -5.251 | -13.038 | 2.741 |
| dial@-0.2 | score | -0.011 | -0.036 | 0.016 |
| dial@-0.2 | failed_f | 0.000 | -0.141 | 0.141 |
| dial@+0.2 | exploration_pct | -7.040 | -21.880 | 8.268 |
| dial@+0.2 | intensification | -0.012 | -0.172 | 0.147 |
| dial@+0.2 | last_improvement | -0.095 | -0.222 | 0.036 |
| dial@+0.2 | probe_expl_w | 0.891 | -8.622 | 10.189 |
| dial@+0.2 | score | 0.048 | 0.011 | 0.082 |
| dial@+0.2 | failed_f | 0.047 | -0.078 | 0.188 |
| dial@+0.4 | exploration_pct | -9.331 | -29.085 | 10.072 |
| dial@+0.4 | intensification | -0.005 | -0.202 | 0.183 |
| dial@+0.4 | last_improvement | -0.187 | -0.352 | -0.029 |
| dial@+0.4 | probe_expl_w | -2.487 | -13.211 | 8.791 |
| dial@+0.4 | score | 0.038 | -0.008 | 0.082 |
| dial@+0.4 | failed_f | 0.172 | 0.000 | 0.344 |
| random@+0.4 | exploration_pct | -14.845 | -30.188 | -0.015 |
| random@+0.4 | intensification | 0.051 | -0.102 | 0.198 |
| random@+0.4 | last_improvement | -0.001 | -0.150 | 0.135 |
| random@+0.4 | probe_expl_w | -8.910 | -18.408 | 0.983 |
| random@+0.4 | score | 0.016 | -0.018 | 0.052 |
| random@+0.4 | failed_f | 0.062 | -0.062 | 0.203 |

## Probe readout vs measured exploration

Spearman 0.33 over 285 working algorithms.
