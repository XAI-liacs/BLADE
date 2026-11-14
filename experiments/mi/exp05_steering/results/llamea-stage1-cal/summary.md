# LLaMEA campaign `llamea-stage1-cal`

1 arms, 8 runs, 800 algorithms. Means over runs with 95% bootstrap CIs.

## Arms

| arm | runs | best_aocc | failed | div_codebleu | div_ast | div_concepts | div_final_codebleu | family_entropy | n_families | exploration_pct | exploration_pct_final | intensification | intensification_final |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| adaptive_cal | 8 | 0.594 (0.508 to 0.674) | 0.172 (0.133 to 0.220) | 0.554 (0.527 to 0.579) | 0.729 (0.703 to 0.757) | 0.547 (0.488 to 0.616) | 0.299 (0.248 to 0.354) | 3.885 (3.708 to 4.080) | 28.875 (25.500 to 32.500) | 29.616 (23.479 to 36.529) | 15.397 (8.076 to 25.701) | 0.576 (0.525 to 0.627) | 0.816 (0.745 to 0.874) |

## Operators (children pooled over runs, CIs over runs)

| arm | operator | n | failed | aocc | improved | novelty_codebleu | novelty_concepts | step_codebleu |
|---|---|---|---|---|---|---|---|---|
| adaptive_cal | new | 248 | 0.282 (0.209 to 0.351) | 0.229 (0.207 to 0.247) | 0.161 (0.121 to 0.202) | 0.341 (0.303 to 0.366) | 0.164 (0.139 to 0.191) | 0.359 (0.325 to 0.380) |
| adaptive_cal | refine | 272 | 0.132 (0.088 to 0.175) | 0.307 (0.263 to 0.373) | 0.217 (0.134 to 0.299) | 0.266 (0.253 to 0.278) | 0.071 (0.046 to 0.092) | 0.288 (0.274 to 0.300) |
| adaptive_cal | simplify | 248 | 0.109 (0.050 to 0.175) | 0.347 (0.300 to 0.393) | 0.282 (0.217 to 0.340) | 0.163 (0.145 to 0.183) | 0.034 (0.022 to 0.048) | 0.191 (0.181 to 0.204) |

## Per lever

| steer | n | failed | aocc | novelty_codebleu | novelty_concepts |
|---|---|---|---|---|---|
| b_exploration_pct_resid@L27 | 760 | 0.175 | 0.295 | 0.256 | 0.086 |
