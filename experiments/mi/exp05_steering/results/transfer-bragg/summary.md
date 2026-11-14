# Transfer screen: transfer-bragg (bragg)

Dial `b_exploration_pct_resid@L27`, doses [-0.4, -0.2, 0.2, 0.4] x residual norm; random direction at 0.4. Score: AOCC of the working algorithms. `probe_expl`: exploration % read out by a ridge probe on the L15 activations, trained on the MA-BBOB corpus (CV Spearman there 0.66); concept rates among working algorithms.

## Conditions

| phase | condition | n | failed | score | exploration_pct | intensification | last_improvement | probe_expl | population | differential_evolution | particle_swarm | restart | simulated_annealing | local_search | gradient | gen_tokens |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| init | none | 64 | 0.734 | 0.120 | 46.395 | 0.313 | 0.478 | 33.650 | 1.000 | 0.941 | 0.000 | 0.000 | 0.000 | 0.706 | 0.059 | 733.750 |
| init | dial@-0.4 | 64 | 0.797 | 0.076 | 23.007 | 0.596 | 0.757 | 42.264 | 0.923 | 0.308 | 0.000 | 0.000 | 0.231 | 1.000 | 0.000 | 1460.000 |
| init | dial@-0.2 | 32 | 0.781 | 0.146 | 54.787 | 0.270 | 0.489 | 54.898 | 1.000 | 1.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.143 | 980.344 |
| init | dial@+0.2 | 32 | 0.656 | 0.122 | 41.167 | 0.232 | 0.621 | 32.396 | 1.000 | 0.818 | 0.000 | 0.000 | 0.000 | 0.636 | 0.000 | 747.344 |
| init | dial@+0.4 | 64 | 0.875 | 0.132 | 91.084 | 0.003 | 0.491 | 37.949 | 1.000 | 1.000 | 0.000 | 0.000 | 0.000 | 0.250 | 0.000 | 789.875 |
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
| pooled | exploration_pct | 102 | 29.650 | 2.062 | 54.981 |
| pooled | intensification | 105 | -0.455 | -0.736 | -0.192 |
| pooled | last_improvement | 105 | -0.181 | -0.429 | 0.100 |
| pooled | probe_expl_w | 106 | -4.012 | -22.949 | 14.174 |
| pooled | score | 106 | 0.046 | -0.006 | 0.085 |
| pooled | failed_f | 356 | -0.032 | -0.184 | 0.122 |
| pooled | population | 356 | 0.341 | 0.218 | 0.467 |
| pooled | differential_evolution | 356 | 0.457 | 0.296 | 0.609 |
| pooled | particle_swarm | 356 | -0.013 | -0.039 | 0.000 |
| pooled | restart | 356 | 0.006 | -0.013 | 0.032 |
| pooled | simulated_annealing | 356 | -0.064 | -0.122 | -0.019 |
| pooled | local_search | 356 | -0.393 | -0.567 | -0.226 |
| pooled | gradient | 356 | -0.013 | -0.120 | 0.085 |
| init | exploration_pct | 54 | 49.081 | 4.042 | 132.363 |
| init | intensification | 55 | -0.643 | -1.364 | -0.262 |
| init | last_improvement | 55 | 0.100 | -0.338 | 0.610 |
| init | probe_expl_w | 56 | 7.942 | -42.402 | 42.958 |
| init | score | 56 | 0.034 | -0.089 | 0.113 |
| init | failed_f | 256 | 0.052 | -0.112 | 0.224 |
| init | population | 256 | 0.295 | 0.158 | 0.430 |
| init | differential_evolution | 256 | 0.451 | 0.261 | 0.644 |
| init | particle_swarm | 256 | -0.017 | -0.053 | 0.000 |
| init | restart | 256 | 0.000 | 0.000 | 0.000 |
| init | simulated_annealing | 256 | -0.078 | -0.151 | -0.017 |
| init | local_search | 256 | -0.547 | -0.753 | -0.347 |
| init | gradient | 256 | 0.026 | -0.080 | 0.133 |
| refine | exploration_pct | 48 | 18.390 | -18.538 | 42.930 |
| refine | intensification | 50 | -0.344 | -0.681 | 0.024 |
| refine | last_improvement | 50 | -0.346 | -0.624 | 0.004 |
| refine | probe_expl_w | 50 | -11.121 | -27.584 | 6.413 |
| refine | score | 50 | 0.053 | 0.000 | 0.097 |
| refine | failed_f | 100 | -0.275 | -0.576 | 0.075 |
| refine | population | 100 | 0.475 | 0.225 | 0.750 |
| refine | differential_evolution | 100 | 0.475 | 0.250 | 0.725 |
| refine | particle_swarm | 100 | 0.000 | 0.000 | 0.000 |
| refine | restart | 100 | 0.025 | -0.050 | 0.100 |
| refine | simulated_annealing | 100 | -0.025 | -0.075 | 0.000 |
| refine | local_search | 100 | 0.050 | -0.225 | 0.300 |
| refine | gradient | 100 | -0.125 | -0.350 | 0.125 |

## Paired change against the unsteered sample

| condition | metric | delta | lo | hi |
|---|---|---|---|---|
| dial@-0.4 | exploration_pct | 4.942 | -1.545 | 11.429 |
| dial@-0.4 | intensification | 0.002 | -0.000 | 0.007 |
| dial@-0.4 | last_improvement | -0.041 | -0.296 | 0.227 |
| dial@-0.4 | probe_expl_w | 13.104 | -11.008 | 37.216 |
| dial@-0.4 | score | 0.017 | -0.008 | 0.056 |
| dial@-0.4 | failed_f | 0.095 | -0.048 | 0.238 |
| dial@-0.2 | exploration_pct | -3.645 | -25.295 | 14.931 |
| dial@-0.2 | intensification | 0.036 | -0.026 | 0.138 |
| dial@-0.2 | last_improvement | 0.004 | -0.056 | 0.061 |
| dial@-0.2 | probe_expl_w | -0.029 | -19.107 | 18.499 |
| dial@-0.2 | score | 0.013 | -0.005 | 0.043 |
| dial@-0.2 | failed_f | 0.000 | -0.173 | 0.173 |
| dial@+0.2 | exploration_pct | 4.049 | -32.145 | 38.532 |
| dial@+0.2 | intensification | -0.125 | -0.466 | 0.232 |
| dial@+0.2 | last_improvement | 0.016 | -0.203 | 0.271 |
| dial@+0.2 | probe_expl_w | 1.476 | -13.024 | 18.122 |
| dial@+0.2 | score | 0.021 | -0.016 | 0.061 |
| dial@+0.2 | failed_f | -0.135 | -0.308 | 0.058 |
| dial@+0.4 | exploration_pct | 36.079 | -9.790 | 77.384 |
| dial@+0.4 | intensification | -0.350 | -0.747 | 0.063 |
| dial@+0.4 | last_improvement | -0.063 | -0.356 | 0.230 |
| dial@+0.4 | probe_expl_w | 10.754 | -23.179 | 40.832 |
| dial@+0.4 | score | 0.037 | -0.043 | 0.103 |
| dial@+0.4 | failed_f | 0.107 | -0.024 | 0.226 |
| random@+0.4 | exploration_pct | -2.325 | -29.697 | 27.225 |
| random@+0.4 | intensification | 0.135 | 0.011 | 0.288 |
| random@+0.4 | last_improvement | -0.238 | -0.464 | -0.015 |
| random@+0.4 | probe_expl_w | -7.191 | -23.773 | 7.631 |
| random@+0.4 | score | 0.043 | -0.034 | 0.137 |
| random@+0.4 | failed_f | 0.019 | -0.154 | 0.192 |

## Probe readout vs measured exploration

Spearman -0.08 over 120 working algorithms.
