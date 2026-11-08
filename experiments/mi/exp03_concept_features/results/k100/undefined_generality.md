# Undefined references beyond `func`

## Label check against the logged errors (all 8400 solutions)

| kind | n_flagged | flagged_fail | flagged_logged_same_kind | n_logged | logged_found |
|---|---|---|---|---|---|
| name | 99 | 0.929 | 0.687 | 92 | 0.728 |
| attr | 92 | 0.913 | 0.717 | 121 | 0.529 |

Solutions sampled: 588 (188 with an undefined reference + 400 clean); reference tokens: 104953.

## Outcome of solutions with each kind of undefined reference

| group | works | NameError | AttributeError | other |
|---|---|---|---|---|
| elsewhere | 4 | 13 | 1 | 3 |
| func | 2 | 40 | 1 | 9 |
| never | 2 | 15 | 0 | 12 |
| self_attr | 8 | 1 | 66 | 17 |

## Activity (fraction > 0) of the step 12 features, L21

| group | n | f7944 | f9410 | f57487 | f16349 |
|---|---|---|---|---|---|
| func | 88 | 0.545 | 0.716 | 0.602 | 0.591 |
| elsewhere | 32 | 0.312 | 0.500 | 0.188 | 0.125 |
| never | 60 | 0.267 | 0.533 | 0.233 | 0.083 |
| self_attr | 205 | 0.054 | 0.166 | 0.141 | 0.010 |
| valid name | 84310 | 0.000 | 0.003 | 0.002 | 0.003 |
| valid self attr | 20258 | 0.001 | 0.003 | 0.004 | 0.002 |
| other code tokens | 360543 | 0.000 | 0.003 | 0.008 | 0.006 |

## AUROC against valid references of the same kind

| group | f7944 | f9410 | f16349 | f57487 |
|---|---|---|---|---|
| elsewhere | 0.656 | 0.749 | 0.561 | 0.593 |
| func | 0.773 | 0.857 | 0.795 | 0.801 |
| never | 0.633 | 0.766 | 0.540 | 0.616 |
| self_attr | 0.527 | 0.582 | 0.504 | 0.569 |

## Best features per group

| group | rank | feature | auroc | p_active_group | p_active_valid |
|---|---|---|---|---|---|
| func | 0 | 2069 | 0.981 | 1.000 | 0.058 |
| func | 1 | 53254 | 0.975 | 0.989 | 0.063 |
| func | 2 | 23486 | 0.971 | 0.977 | 0.058 |
| func | 3 | 29029 | 0.954 | 1.000 | 0.100 |
| func | 4 | 31959 | 0.940 | 0.955 | 0.106 |
| elsewhere | 0 | 49594 | 0.878 | 0.781 | 0.044 |
| elsewhere | 1 | 61227 | 0.760 | 0.531 | 0.013 |
| elsewhere | 2 | 9410 | 0.749 | 0.500 | 0.003 |
| elsewhere | 3 | 51531 | 0.720 | 0.781 | 0.457 |
| elsewhere | 4 | 9967 | 0.710 | 1.000 | 0.939 |
| never | 0 | 35640 | 0.792 | 0.817 | 0.457 |
| never | 1 | 63553 | 0.772 | 0.900 | 0.630 |
| never | 2 | 9410 | 0.766 | 0.533 | 0.003 |
| never | 3 | 46461 | 0.757 | 0.517 | 0.005 |
| never | 4 | 37710 | 0.736 | 0.883 | 0.779 |
| self_attr | 0 | 22749 | 0.608 | 0.288 | 0.080 |
| self_attr | 1 | 29409 | 0.605 | 0.371 | 0.174 |
| self_attr | 2 | 13109 | 0.603 | 0.234 | 0.031 |
| self_attr | 3 | 28574 | 0.603 | 0.898 | 0.755 |
| self_attr | 4 | 9967 | 0.600 | 0.951 | 0.903 |

## Most common undefined names (fraction active)

| group | name | n | f7944 | f9410 |
|---|---|---|---|---|
| func | func | 88 | 0.545 | 0.716 |
| self_attr | pop_size | 31 | 0.065 | 0.129 |
| self_attr | mu | 17 | 0.059 | 0.176 |
| never | norm | 14 | 0.429 | 0.786 |
| self_attr | stage_budgets | 12 | 0.000 | 0.000 |
| never | j | 11 | 0.000 | 0.364 |
| self_attr | momentum | 10 | 0.000 | 0.000 |
| never | scipy | 9 | 0.333 | 0.778 |
| self_attr | population_size | 8 | 0.125 | 0.250 |
| self_attr | pop | 6 | 0.000 | 0.167 |
| never | d | 3 | 0.333 | 1.000 |
| never | grad | 3 | 0.000 | 0.333 |
| elsewhere | F_high | 2 | 0.000 | 0.000 |
| never | x_opt | 2 | 0.000 | 0.000 |
| elsewhere | sigma | 2 | 1.000 | 1.000 |
| elsewhere | F_low | 2 | 0.500 | 1.000 |
| elsewhere | bounds | 2 | 0.500 | 1.000 |
| elsewhere | dim | 2 | 0.000 | 0.500 |
| elsewhere | grad_learning_rate | 2 | 0.000 | 0.000 |
