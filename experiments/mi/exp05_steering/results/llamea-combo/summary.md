# LLaMEA campaign `llamea-combo`

1 arms, 24 runs, 4800 algorithms. Means over runs with 95% bootstrap CIs.

## Arms

| arm | runs | best_aocc | failed | div_codebleu | div_ast | div_concepts | div_final_codebleu | family_entropy | n_families | exploration_pct | exploration_pct_final | intensification | intensification_final |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| adaptive_temp11 | 24 | 0.723 (0.689 to 0.754) | 0.224 (0.208 to 0.241) | 0.531 (0.514 to 0.546) | 0.695 (0.676 to 0.712) | 0.466 (0.441 to 0.490) | 0.241 (0.198 to 0.285) | 4.181 (3.970 to 4.395) | 52.750 (48.667 to 57.084) | 27.697 (24.140 to 31.705) | 9.721 (6.950 to 13.053) | 0.620 (0.576 to 0.664) | 0.876 (0.849 to 0.898) |

## Operators (children pooled over runs, CIs over runs)

| arm | operator | n | failed | aocc | improved | novelty_codebleu | novelty_concepts | step_codebleu |
|---|---|---|---|---|---|---|---|---|
| adaptive_temp11 | new | 1563 | 0.329 (0.296 to 0.353) | 0.258 (0.242 to 0.275) | 0.098 (0.077 to 0.116) | 0.354 (0.331 to 0.374) | 0.144 (0.132 to 0.156) | 0.388 (0.364 to 0.410) |
| adaptive_temp11 | refine | 1625 | 0.249 (0.225 to 0.275) | 0.347 (0.314 to 0.385) | 0.134 (0.111 to 0.163) | 0.260 (0.246 to 0.273) | 0.055 (0.047 to 0.064) | 0.288 (0.273 to 0.302) |
| adaptive_temp11 | simplify | 1516 | 0.094 (0.069 to 0.118) | 0.473 (0.432 to 0.518) | 0.210 (0.181 to 0.236) | 0.135 (0.125 to 0.145) | 0.023 (0.018 to 0.028) | 0.160 (0.151 to 0.171) |

## Per lever

| steer | n | failed | aocc | novelty_codebleu | novelty_concepts |
|---|---|---|---|---|---|
| b_exploration_pct_resid@L27 | 4680 | 0.226 | 0.359 | 0.251 | 0.073 |
