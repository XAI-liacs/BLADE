# LLaMEA campaign `llamea-randomdir200`

1 arms, 8 runs, 1600 algorithms. Means over runs with 95% bootstrap CIs.

## Arms

| arm | runs | best_aocc | failed | div_codebleu | div_ast | div_concepts | div_final_codebleu | family_entropy | n_families | exploration_pct | exploration_pct_final | intensification | intensification_final |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| randomdir | 8 | 0.702 (0.586 to 0.799) | 0.172 (0.140 to 0.216) | 0.520 (0.485 to 0.556) | 0.714 (0.679 to 0.753) | 0.470 (0.383 to 0.556) | 0.216 (0.170 to 0.269) | 3.698 (3.232 to 4.139) | 42.250 (36.875 to 48.503) | 33.531 (24.948 to 45.806) | 21.350 (8.049 to 41.210) | 0.592 (0.501 to 0.679) | 0.817 (0.676 to 0.907) |

## Operators (children pooled over runs, CIs over runs)

| arm | operator | n | failed | aocc | improved | novelty_codebleu | novelty_concepts | step_codebleu |
|---|---|---|---|---|---|---|---|---|
| randomdir | new | 543 | 0.256 (0.212 to 0.303) | 0.301 (0.247 to 0.354) | 0.110 (0.081 to 0.140) | 0.310 (0.273 to 0.353) | 0.127 (0.107 to 0.148) | 0.339 (0.299 to 0.386) |
| randomdir | refine | 538 | 0.177 (0.134 to 0.238) | 0.366 (0.306 to 0.437) | 0.130 (0.093 to 0.183) | 0.242 (0.219 to 0.260) | 0.049 (0.036 to 0.066) | 0.271 (0.253 to 0.288) |
| randomdir | simplify | 487 | 0.066 (0.043 to 0.091) | 0.467 (0.386 to 0.566) | 0.181 (0.156 to 0.205) | 0.132 (0.111 to 0.157) | 0.017 (0.010 to 0.024) | 0.154 (0.132 to 0.179) |

## Per lever

| steer | n | failed | aocc | novelty_codebleu | novelty_concepts |
|---|---|---|---|---|---|
| random0 | 187 | 0.193 | 0.368 | 0.231 | 0.058 |
| random1 | 200 | 0.155 | 0.359 | 0.243 | 0.083 |
| random2 | 204 | 0.162 | 0.385 | 0.222 | 0.081 |
| random3 | 178 | 0.146 | 0.401 | 0.241 | 0.070 |
| random4 | 183 | 0.197 | 0.346 | 0.232 | 0.067 |
| random5 | 215 | 0.167 | 0.368 | 0.255 | 0.091 |
| random6 | 214 | 0.168 | 0.370 | 0.245 | 0.071 |
| random7 | 219 | 0.187 | 0.369 | 0.230 | 0.071 |
