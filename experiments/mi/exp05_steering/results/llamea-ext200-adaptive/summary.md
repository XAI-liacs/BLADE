# LLaMEA campaign `llamea-ext200-adaptive`

1 arms, 8 runs, 1600 algorithms. Means over runs with 95% bootstrap CIs.

## Arms

| arm | runs | best_aocc | failed | div_codebleu | div_ast | div_concepts | div_final_codebleu | family_entropy | n_families | exploration_pct | exploration_pct_final | intensification | intensification_final |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| adaptive | 8 | 0.726 (0.669 to 0.784) | 0.221 (0.164 to 0.286) | 0.519 (0.493 to 0.545) | 0.646 (0.584 to 0.704) | 0.497 (0.445 to 0.554) | 0.213 (0.134 to 0.286) | 4.185 (3.861 to 4.585) | 49.625 (42.750 to 57.375) | 28.197 (23.995 to 33.151) | 10.316 (6.536 to 14.882) | 0.637 (0.578 to 0.680) | 0.865 (0.824 to 0.906) |

## Operators (children pooled over runs, CIs over runs)

| arm | operator | n | failed | aocc | improved | novelty_codebleu | novelty_concepts | step_codebleu |
|---|---|---|---|---|---|---|---|---|
| adaptive | new | 522 | 0.343 (0.243 to 0.449) | 0.261 (0.201 to 0.322) | 0.090 (0.063 to 0.125) | 0.322 (0.288 to 0.354) | 0.134 (0.104 to 0.174) | 0.369 (0.332 to 0.405) |
| adaptive | refine | 495 | 0.273 (0.215 to 0.342) | 0.323 (0.269 to 0.400) | 0.127 (0.097 to 0.158) | 0.258 (0.236 to 0.282) | 0.064 (0.045 to 0.081) | 0.298 (0.278 to 0.319) |
| adaptive | simplify | 551 | 0.062 (0.042 to 0.079) | 0.494 (0.430 to 0.562) | 0.180 (0.146 to 0.212) | 0.118 (0.104 to 0.136) | 0.020 (0.012 to 0.028) | 0.151 (0.131 to 0.180) |

## Per lever

| steer | n | failed | aocc | novelty_codebleu | novelty_concepts |
|---|---|---|---|---|---|
| b_exploration_pct_resid@L27 | 1560 | 0.221 | 0.364 | 0.229 | 0.071 |
