# Steering campaign `scope`

99 prompts, 1584 samples. Doses in units of
the feature's peak activation; `steer_norm` is the norm of the added
vector (residual norms: L21 127).

| condition | n | steer_norm | nocode | failed | d_failed | d_failed_lo | d_failed_hi | aocc | d_aocc | fitness_working | sim_none | n_gen_tokens | concept | concept_rate | undefined_name | d_undefined_name | d_undefined_name_lo | d_undefined_name_hi | undefined_attr |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| none | 198 | 0.000 | 0.000 | 0.253 | nan | nan | nan | 0.255 | nan | 0.341 | 1.000 | 902.520 |  | nan | 0.045 | nan | nan | nan | 0.030 |
| random21@-4 | 198 | 50.142 | 0.000 | 0.313 | 0.061 | -0.015 | 0.136 | 0.222 | -0.032 | 0.324 | 0.653 | 986.162 |  | nan | 0.071 | 0.025 | -0.020 | 0.076 | 0.010 |
| random21@+4 | 198 | 50.142 | 0.000 | 0.364 | 0.111 | 0.020 | 0.202 | 0.217 | -0.038 | 0.341 | 0.679 | 859.056 |  | nan | 0.136 | 0.091 | 0.035 | 0.146 | 0.020 |
| resample | 198 | 0.000 | 0.000 | 0.268 | 0.015 | -0.071 | 0.096 | 0.241 | -0.014 | 0.329 | 0.681 | 898.742 |  | nan | 0.061 | 0.015 | -0.025 | 0.056 | 0.020 |
| undef_7944@-4 | 198 | 29.570 | 0.000 | 0.343 | 0.091 | 0.010 | 0.177 | 0.218 | -0.037 | 0.332 | 0.722 | 855.798 |  | nan | 0.051 | 0.005 | -0.040 | 0.045 | 0.051 |
| undef_7944@+4 | 198 | 29.570 | 0.000 | 0.470 | 0.217 | 0.116 | 0.313 | 0.173 | -0.082 | 0.326 | 0.662 | 954.126 |  | nan | 0.303 | 0.258 | 0.187 | 0.328 | 0.086 |
| undef_9410@-4 | 198 | 70.714 | 0.000 | 0.460 | 0.207 | 0.116 | 0.298 | 0.190 | -0.064 | 0.353 | 0.520 | 1212.899 |  | nan | 0.066 | 0.020 | -0.025 | 0.071 | 0.025 |
| undef_9410@+4 | 198 | 70.714 | 0.000 | 0.848 | 0.596 | 0.510 | 0.682 | 0.041 | -0.214 | 0.268 | 0.380 | 304.152 |  | nan | 0.056 | 0.010 | -0.035 | 0.056 | 0.051 |

## Concept rates by condition

| condition |
|---|
| none |
| random21@+4 |
| random21@-4 |
| resample |
| undef_7944@+4 |
| undef_7944@-4 |
| undef_9410@+4 |
| undef_9410@-4 |
