# LLaMEA campaign `llamea-anneal200`

1 arms, 16 runs, 3200 algorithms. Means over runs with 95% bootstrap CIs.

## Arms

| arm | runs | best_aocc | failed | div_codebleu | div_ast | div_concepts | div_final_codebleu | family_entropy | n_families | exploration_pct | exploration_pct_final | intensification | intensification_final |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| anneal | 16 | 0.703 (0.649 to 0.741) | 0.164 (0.136 to 0.194) | 0.520 (0.499 to 0.541) | 0.687 (0.650 to 0.723) | 0.406 (0.362 to 0.454) | 0.330 (0.297 to 0.366) | 3.559 (3.260 to 3.868) | 38.188 (33.873 to 43.125) | 24.797 (21.430 to 28.925) | 10.373 (6.400 to 17.128) | 0.655 (0.624 to 0.686) | 0.864 (0.814 to 0.903) |

## Operators (children pooled over runs, CIs over runs)

| arm | operator | n | failed | aocc | improved | novelty_codebleu | novelty_concepts | step_codebleu |
|---|---|---|---|---|---|---|---|---|
| anneal | new | 1020 | 0.230 (0.193 to 0.266) | 0.321 (0.295 to 0.346) | 0.141 (0.115 to 0.169) | 0.315 (0.282 to 0.338) | 0.102 (0.085 to 0.118) | 0.350 (0.315 to 0.378) |
| anneal | refine | 1085 | 0.183 (0.140 to 0.232) | 0.385 (0.347 to 0.427) | 0.143 (0.107 to 0.181) | 0.246 (0.233 to 0.259) | 0.035 (0.027 to 0.043) | 0.274 (0.262 to 0.287) |
| anneal | simplify | 1031 | 0.077 (0.053 to 0.111) | 0.449 (0.407 to 0.488) | 0.189 (0.165 to 0.211) | 0.152 (0.136 to 0.167) | 0.016 (0.012 to 0.021) | 0.181 (0.164 to 0.198) |

## Per lever

| steer | n | failed | aocc | novelty_codebleu | novelty_concepts |
|---|---|---|---|---|---|
| b_exploration_pct_resid@L27 | 3200 | 0.164 | 0.382 | 0.243 | 0.059 |
