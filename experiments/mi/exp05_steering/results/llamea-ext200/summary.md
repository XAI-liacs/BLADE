# LLaMEA campaign `llamea-ext200`

2 arms, 16 runs, 3200 algorithms. Means over runs with 95% bootstrap CIs.

## Arms

| arm | runs | best_aocc | failed | div_codebleu | div_ast | div_concepts | div_final_codebleu | family_entropy | n_families | exploration_pct | exploration_pct_final | intensification | intensification_final |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| adaptive_cal | 8 | 0.673 (0.594 to 0.746) | 0.151 (0.119 to 0.186) | 0.506 (0.474 to 0.534) | 0.681 (0.639 to 0.719) | 0.482 (0.408 to 0.584) | 0.202 (0.120 to 0.282) | 3.928 (3.584 to 4.251) | 41.625 (35.625 to 47.375) | 24.723 (20.414 to 30.868) | 8.544 (6.102 to 11.255) | 0.646 (0.597 to 0.693) | 0.851 (0.792 to 0.905) |
| control | 8 | 0.667 (0.556 to 0.756) | 0.129 (0.096 to 0.168) | 0.478 (0.440 to 0.509) | 0.676 (0.645 to 0.706) | 0.452 (0.362 to 0.550) | 0.267 (0.179 to 0.339) | 3.400 (2.850 to 4.018) | 37.375 (27.872 to 48.753) | 18.250 (13.651 to 22.763) | 6.599 (4.661 to 8.294) | 0.716 (0.648 to 0.786) | 0.900 (0.872 to 0.929) |

## Operators (children pooled over runs, CIs over runs)

| arm | operator | n | failed | aocc | improved | novelty_codebleu | novelty_concepts | step_codebleu |
|---|---|---|---|---|---|---|---|---|
| adaptive_cal | new | 515 | 0.262 (0.203 to 0.315) | 0.291 (0.260 to 0.326) | 0.113 (0.089 to 0.154) | 0.286 (0.244 to 0.313) | 0.118 (0.099 to 0.140) | 0.311 (0.270 to 0.338) |
| adaptive_cal | refine | 549 | 0.109 (0.069 to 0.149) | 0.385 (0.329 to 0.456) | 0.160 (0.102 to 0.213) | 0.228 (0.213 to 0.244) | 0.045 (0.028 to 0.064) | 0.258 (0.239 to 0.278) |
| adaptive_cal | simplify | 504 | 0.081 (0.041 to 0.129) | 0.447 (0.385 to 0.499) | 0.214 (0.174 to 0.252) | 0.115 (0.107 to 0.125) | 0.019 (0.013 to 0.027) | 0.139 (0.128 to 0.152) |
| control | new | 515 | 0.216 (0.156 to 0.266) | 0.309 (0.260 to 0.363) | 0.082 (0.053 to 0.116) | 0.270 (0.247 to 0.288) | 0.112 (0.079 to 0.141) | 0.305 (0.282 to 0.326) |
| control | refine | 547 | 0.121 (0.075 to 0.197) | 0.381 (0.316 to 0.450) | 0.112 (0.078 to 0.152) | 0.225 (0.193 to 0.246) | 0.049 (0.029 to 0.073) | 0.258 (0.222 to 0.280) |
| control | simplify | 506 | 0.051 (0.031 to 0.080) | 0.476 (0.369 to 0.572) | 0.140 (0.074 to 0.199) | 0.114 (0.097 to 0.123) | 0.014 (0.007 to 0.019) | 0.144 (0.121 to 0.162) |

## Per lever

| steer | n | failed | aocc | novelty_codebleu | novelty_concepts |
|---|---|---|---|---|---|
| b_exploration_pct_resid@L27 | 1560 | 0.151 | 0.374 | 0.210 | 0.059 |
