# Transfer screen: transfer-bragg (bragg)

Dial `b_exploration_pct_resid@L27`, doses [-0.4, -0.2, 0.2, 0.4] x residual norm; random direction at 0.4. Score: AOCC of the working algorithms. `probe_expl`: exploration % read out by a ridge probe on the L15 activations, trained on the MA-BBOB corpus (CV Spearman there 0.66); concept rates among working algorithms.

## Conditions

| phase | condition | n | failed | score | exploration_pct | intensification | last_improvement | probe_expl | population | differential_evolution | particle_swarm | restart | simulated_annealing | local_search | gradient | gen_tokens |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| init | none | 32 | 0.688 | 0.114 | 41.254 | 0.403 | 0.475 | 37.599 | 1.000 | 0.900 | 0.000 | 0.000 | 0.000 | 0.900 | 0.100 | 728.844 |
| init | dial@-0.4 | 32 | 0.781 | 0.083 | 43.957 | 0.367 | 0.754 | 32.199 | 0.857 | 0.286 | 0.000 | 0.000 | 0.286 | 1.000 | 0.000 | 1486.344 |
| init | dial@-0.2 | 32 | 0.781 | 0.146 | 54.787 | 0.270 | 0.489 | 54.898 | 1.000 | 1.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.143 | 980.344 |
| init | dial@+0.2 | 32 | 0.656 | 0.122 | 41.167 | 0.232 | 0.621 | 32.396 | 1.000 | 0.818 | 0.000 | 0.000 | 0.000 | 0.636 | 0.000 | 747.344 |
| init | dial@+0.4 | 32 | 0.812 | 0.143 | 87.518 | 0.004 | 0.502 | 30.336 | 1.000 | 1.000 | 0.000 | 0.000 | 0.000 | 0.333 | 0.000 | 773.312 |
| init | random@+0.4 | 32 | 0.812 | 0.162 | 33.464 | 0.635 | 0.189 | 32.351 | 1.000 | 0.500 | 0.000 | 0.000 | 0.000 | 0.667 | 0.000 | 834.781 |
| refine | none | 20 | 0.550 | 0.067 | 38.191 | 0.249 | 0.672 | 47.708 | 1.000 | 0.889 | 0.111 | 0.000 | 0.000 | 0.889 | 0.222 | 988.600 |
| refine | dial@-0.4 | 20 | 0.750 | 0.085 | 30.604 | 0.448 | 0.562 | 56.716 | 0.800 | 0.600 | 0.000 | 0.000 | 0.000 | 0.800 | 0.200 | 1503.950 |
| refine | dial@-0.2 | 20 | 0.400 | 0.129 | 43.238 | 0.222 | 0.563 | 43.849 | 0.917 | 0.917 | 0.000 | 0.083 | 0.083 | 0.917 | 0.167 | 1093.150 |
| refine | dial@+0.2 | 20 | 0.250 | 0.130 | 42.978 | 0.296 | 0.466 | 44.504 | 1.000 | 0.867 | 0.000 | 0.067 | 0.000 | 0.867 | 0.200 | 939.150 |
| refine | dial@+0.4 | 20 | 0.550 | 0.145 | 46.059 | 0.156 | 0.321 | 29.545 | 1.000 | 1.000 | 0.000 | 0.000 | 0.000 | 0.889 | 0.222 | 884.200 |
| refine | random@+0.4 | 20 | 0.400 | 0.112 | 47.371 | 0.221 | 0.541 | 35.675 | 1.000 | 0.917 | 0.000 | 0.000 | 0.000 | 0.750 | 0.167 | 929.550 |

## Slopes on the signed dial dose

Fixed effects per prompt x replicate; 95% CI from a bootstrap over them. `failed_f`: failure rate (all samples); other metrics: working only.

| phase | metric | n | slope | lo | hi |
|---|---|---|---|---|---|
| pooled | exploration_pct | 87 | 26.260 | 0.039 | 51.117 |
| pooled | intensification | 90 | -0.436 | -0.728 | -0.173 |
| pooled | last_improvement | 90 | -0.208 | -0.478 | 0.089 |
| pooled | probe_expl_w | 91 | -9.195 | -29.866 | 7.605 |
| pooled | score | 91 | 0.056 | 0.012 | 0.096 |
| pooled | failed_f | 260 | -0.125 | -0.327 | 0.087 |
| pooled | population | 260 | 0.356 | 0.221 | 0.510 |
| pooled | differential_evolution | 260 | 0.471 | 0.279 | 0.663 |
| pooled | particle_swarm | 260 | -0.019 | -0.058 | 0.000 |
| pooled | restart | 260 | 0.010 | -0.019 | 0.038 |
| pooled | simulated_annealing | 260 | -0.058 | -0.125 | -0.010 |
| pooled | local_search | 260 | -0.221 | -0.423 | -0.010 |
| pooled | gradient | 260 | -0.000 | -0.125 | 0.115 |
| init | exploration_pct | 39 | 41.633 | -3.809 | 128.767 |
| init | intensification | 40 | -0.613 | -1.577 | -0.219 |
| init | last_improvement | 40 | 0.059 | -0.429 | 0.640 |
| init | probe_expl_w | 41 | -5.543 | -72.595 | 21.374 |
| init | score | 41 | 0.060 | -0.050 | 0.149 |
| init | failed_f | 160 | -0.031 | -0.281 | 0.219 |
| init | population | 160 | 0.281 | 0.125 | 0.437 |
| init | differential_evolution | 160 | 0.469 | 0.188 | 0.734 |
| init | particle_swarm | 160 | -0.031 | -0.094 | 0.000 |
| init | restart | 160 | 0.000 | 0.000 | 0.000 |
| init | simulated_annealing | 160 | -0.078 | -0.172 | 0.000 |
| init | local_search | 160 | -0.391 | -0.672 | -0.109 |
| init | gradient | 160 | 0.078 | -0.047 | 0.203 |
| refine | exploration_pct | 48 | 18.390 | -17.966 | 42.339 |
| refine | intensification | 50 | -0.344 | -0.697 | 0.035 |
| refine | last_improvement | 50 | -0.346 | -0.644 | -0.005 |
| refine | probe_expl_w | 50 | -11.121 | -27.460 | 6.612 |
| refine | score | 50 | 0.053 | 0.003 | 0.097 |
| refine | failed_f | 100 | -0.275 | -0.600 | 0.075 |
| refine | population | 100 | 0.475 | 0.225 | 0.750 |
| refine | differential_evolution | 100 | 0.475 | 0.250 | 0.725 |
| refine | particle_swarm | 100 | 0.000 | 0.000 | 0.000 |
| refine | restart | 100 | 0.025 | -0.050 | 0.100 |
| refine | simulated_annealing | 100 | -0.025 | -0.075 | 0.000 |
| refine | local_search | 100 | 0.050 | -0.200 | 0.325 |
| refine | gradient | 100 | -0.125 | -0.375 | 0.125 |

## Paired change against the unsteered sample

| condition | metric | delta | lo | hi |
|---|---|---|---|---|
| dial@-0.4 | exploration_pct | 4.942 | -1.545 | 11.429 |
| dial@-0.4 | intensification | 0.002 | -0.000 | 0.007 |
| dial@-0.4 | last_improvement | -0.041 | -0.296 | 0.227 |
| dial@-0.4 | probe_expl_w | 13.104 | -11.008 | 37.216 |
| dial@-0.4 | score | 0.017 | -0.010 | 0.056 |
| dial@-0.4 | failed_f | 0.135 | -0.038 | 0.308 |
| dial@-0.2 | exploration_pct | -3.645 | -26.749 | 14.943 |
| dial@-0.2 | intensification | 0.036 | -0.026 | 0.138 |
| dial@-0.2 | last_improvement | 0.004 | -0.060 | 0.062 |
| dial@-0.2 | probe_expl_w | -0.029 | -20.357 | 18.373 |
| dial@-0.2 | score | 0.013 | -0.005 | 0.043 |
| dial@-0.2 | failed_f | 0.000 | -0.173 | 0.192 |
| dial@+0.2 | exploration_pct | 4.049 | -31.168 | 37.124 |
| dial@+0.2 | intensification | -0.125 | -0.468 | 0.256 |
| dial@+0.2 | last_improvement | 0.016 | -0.205 | 0.273 |
| dial@+0.2 | probe_expl_w | 1.476 | -13.032 | 17.572 |
| dial@+0.2 | score | 0.021 | -0.016 | 0.058 |
| dial@+0.2 | failed_f | -0.135 | -0.308 | 0.038 |
| dial@+0.4 | exploration_pct | 33.005 | -32.380 | 83.627 |
| dial@+0.4 | intensification | -0.351 | -0.891 | 0.189 |
| dial@+0.4 | last_improvement | -0.178 | -0.469 | 0.239 |
| dial@+0.4 | probe_expl_w | -6.584 | -40.304 | 24.388 |
| dial@+0.4 | score | 0.091 | 0.053 | 0.128 |
| dial@+0.4 | failed_f | 0.077 | -0.115 | 0.269 |
| random@+0.4 | exploration_pct | -2.325 | -28.314 | 28.901 |
| random@+0.4 | intensification | 0.135 | 0.015 | 0.289 |
| random@+0.4 | last_improvement | -0.238 | -0.465 | -0.020 |
| random@+0.4 | probe_expl_w | -7.191 | -24.470 | 7.598 |
| random@+0.4 | score | 0.043 | -0.035 | 0.134 |
| random@+0.4 | failed_f | 0.019 | -0.154 | 0.192 |

## Probe readout vs measured exploration

Spearman -0.03 over 105 working algorithms.
