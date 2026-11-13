# Transfer screen: transfer-packing (packing26)

Dial `b_exploration_pct_resid@L27`, doses [-0.4, -0.2, 0.2, 0.4] x residual norm; random direction at 0.4. Score: sum of radii (repaired) of the working algorithms. `probe_expl`: exploration % read out by a ridge probe on the L15 activations, trained on the MA-BBOB corpus (CV Spearman there 0.66); concept rates among working algorithms.

## Conditions

| phase | condition | n | failed | score | feasible_f | probe_expl | population | differential_evolution | particle_swarm | restart | simulated_annealing | local_search | gradient | gen_tokens |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| init | none | 32 | 0.406 | 1.211 | 0.156 | 11.884 | 0.000 | 0.000 | 0.000 | 0.000 | 0.947 | 0.053 | 0.000 | 754.062 |
| init | dial@-0.4 | 32 | 0.750 | 1.272 | 0.062 | 12.427 | 0.000 | 0.000 | 0.000 | 0.000 | 0.375 | 0.875 | 0.125 | 1008.125 |
| init | dial@-0.2 | 32 | 0.594 | 1.304 | 0.125 | 17.050 | 0.000 | 0.000 | 0.000 | 0.000 | 0.615 | 0.385 | 0.077 | 780.625 |
| init | dial@+0.2 | 32 | 0.469 | 1.098 | 0.094 | 14.559 | 0.000 | 0.000 | 0.000 | 0.000 | 0.882 | 0.118 | 0.000 | 769.031 |
| init | dial@+0.4 | 32 | 0.281 | 0.975 | 0.094 | 6.626 | 0.217 | 0.000 | 0.000 | 0.000 | 0.739 | 0.043 | 0.000 | 818.812 |
| init | random@+0.4 | 32 | 0.562 | 0.887 | 0.031 | 22.323 | 0.000 | 0.000 | 0.000 | 0.000 | 0.786 | 0.214 | 0.000 | 780.719 |
| refine | none | 32 | 0.438 | 0.825 | 0.094 | 4.608 | 0.167 | 0.000 | 0.000 | 0.000 | 0.333 | 0.278 | 0.000 | 882.656 |
| refine | dial@-0.4 | 32 | 0.656 | 0.517 | 0.000 | 9.822 | 0.000 | 0.000 | 0.000 | 0.000 | 0.636 | 0.364 | 0.091 | 1106.344 |
| refine | dial@-0.2 | 32 | 0.469 | 1.035 | 0.062 | -0.470 | 0.000 | 0.000 | 0.000 | 0.059 | 0.529 | 0.294 | 0.059 | 997.875 |
| refine | dial@+0.2 | 32 | 0.375 | 0.984 | 0.125 | 9.414 | 0.400 | 0.000 | 0.050 | 0.000 | 0.250 | 0.100 | 0.000 | 977.562 |
| refine | dial@+0.4 | 32 | 0.500 | 0.963 | 0.000 | 5.500 | 0.438 | 0.000 | 0.125 | 0.000 | 0.312 | 0.062 | 0.000 | 1014.656 |
| refine | random@+0.4 | 32 | 0.594 | 0.791 | 0.031 | 7.270 | 0.308 | 0.000 | 0.000 | 0.000 | 0.231 | 0.154 | 0.000 | 878.031 |

## Slopes on the signed dial dose

Fixed effects per prompt x replicate; 95% CI from a bootstrap over them. `failed_f`: failure rate (all samples); other metrics: working only.

| phase | metric | n | slope | lo | hi |
|---|---|---|---|---|---|
| pooled | feasible_f | 320 | 0.023 | -0.047 | 0.094 |
| pooled | probe_expl_w | 162 | 2.190 | -12.713 | 14.606 |
| pooled | score | 162 | 0.030 | -0.400 | 0.457 |
| pooled | failed_f | 320 | -0.367 | -0.547 | -0.188 |
| pooled | population | 320 | 0.617 | 0.469 | 0.773 |
| pooled | differential_evolution | 320 | 0.000 | 0.000 | 0.000 |
| pooled | particle_swarm | 320 | 0.086 | 0.016 | 0.172 |
| pooled | restart | 320 | -0.008 | -0.023 | 0.000 |
| pooled | simulated_annealing | 320 | 0.055 | -0.141 | 0.258 |
| pooled | local_search | 320 | -0.898 | -1.055 | -0.750 |
| pooled | gradient | 320 | -0.125 | -0.211 | -0.047 |
| init | feasible_f | 160 | 0.016 | -0.125 | 0.141 |
| init | probe_expl_w | 80 | -10.781 | -33.738 | 8.845 |
| init | score | 80 | -0.465 | -0.940 | 0.055 |
| init | failed_f | 160 | -0.531 | -0.750 | -0.297 |
| init | population | 160 | 0.375 | 0.219 | 0.547 |
| init | differential_evolution | 160 | 0.000 | 0.000 | 0.000 |
| init | particle_swarm | 160 | 0.000 | 0.000 | 0.000 |
| init | restart | 160 | 0.000 | 0.000 | 0.000 |
| init | simulated_annealing | 160 | 0.531 | 0.265 | 0.797 |
| init | local_search | 160 | -1.031 | -1.203 | -0.859 |
| init | gradient | 160 | -0.141 | -0.281 | -0.031 |
| refine | feasible_f | 160 | 0.031 | -0.047 | 0.109 |
| refine | probe_expl_w | 82 | 15.140 | -1.680 | 31.405 |
| refine | score | 82 | 0.523 | -0.057 | 1.040 |
| refine | failed_f | 160 | -0.203 | -0.484 | 0.094 |
| refine | population | 160 | 0.859 | 0.641 | 1.078 |
| refine | differential_evolution | 160 | 0.000 | 0.000 | 0.000 |
| refine | particle_swarm | 160 | 0.172 | 0.031 | 0.312 |
| refine | restart | 160 | -0.016 | -0.047 | 0.000 |
| refine | simulated_annealing | 160 | -0.422 | -0.609 | -0.234 |
| refine | local_search | 160 | -0.766 | -1.031 | -0.500 |
| refine | gradient | 160 | -0.109 | -0.234 | -0.015 |

## Paired change against the unsteered sample

| condition | metric | delta | lo | hi |
|---|---|---|---|---|
| dial@-0.4 | feasible_f | -0.094 | -0.188 | 0.000 |
| dial@-0.4 | probe_expl_w | 2.433 | -6.705 | 11.734 |
| dial@-0.4 | score | -0.022 | -0.435 | 0.376 |
| dial@-0.4 | failed_f | 0.281 | 0.125 | 0.422 |
| dial@-0.2 | feasible_f | -0.031 | -0.125 | 0.047 |
| dial@-0.2 | probe_expl_w | -4.462 | -15.548 | 6.099 |
| dial@-0.2 | score | 0.120 | -0.131 | 0.370 |
| dial@-0.2 | failed_f | 0.109 | -0.031 | 0.250 |
| dial@+0.2 | feasible_f | -0.016 | -0.125 | 0.094 |
| dial@+0.2 | probe_expl_w | 6.687 | -4.435 | 17.678 |
| dial@+0.2 | score | -0.151 | -0.395 | 0.076 |
| dial@+0.2 | failed_f | 0.000 | -0.125 | 0.141 |
| dial@+0.4 | feasible_f | -0.078 | -0.172 | 0.016 |
| dial@+0.4 | probe_expl_w | -6.983 | -22.562 | 9.168 |
| dial@+0.4 | score | 0.048 | -0.221 | 0.348 |
| dial@+0.4 | failed_f | -0.031 | -0.203 | 0.141 |
| random@+0.4 | feasible_f | -0.094 | -0.188 | -0.016 |
| random@+0.4 | probe_expl_w | 10.622 | -5.168 | 25.336 |
| random@+0.4 | score | -0.345 | -0.771 | 0.176 |
| random@+0.4 | failed_f | 0.156 | -0.016 | 0.328 |
