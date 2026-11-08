# Undefined-name references

8396 solutions reference `func`; 52 have at least
one reference where it is not in scope. Outcome of those:

| outcome | count |
|---|---|
| NameError | 40 |
| ModuleNotFoundError | 8 |
| works | 2 |
| AttributeError | 1 |
| ValueError | 1 |

Tokens: 88 undefined, 7128 valid `func` references (from 752 solutions).

## Step 11's NameError feature L21 f7944

- AUROC undefined vs valid `func` tokens: 0.773 (within solutions with an undefined reference: 0.773)
- active at undefined refs 0.55, at valid refs 0.00, at all other code tokens 0.000
- mean activation: undefined 3.16, valid 0.00, other tokens 0.001

## By reference group (activation > 0 / mean)

| group | n | f7944_active | f7944_mean | f9410_active | f9410_mean |
|---|---|---|---|---|---|
| undefined | 88 | 0.545 | 3.162 | 0.716 | 7.503 |
| valid, in __call__ | 6374 | 0.000 | 0.000 | 0.006 | 0.018 |
| valid, elsewhere | 754 | 0.000 | 0.000 | 0.005 | 0.014 |

## Features that best separate undefined from valid references

| feature | auroc_undefined_vs_valid | mean_undefined | mean_valid | p_active_undefined | p_active_valid |
|---|---|---|---|---|---|
| 9410.000 | 0.857 | 7.503 | 0.018 | 0.716 | 0.006 |
| 57487.000 | 0.800 | 4.327 | 0.010 | 0.602 | 0.003 |
| 16349.000 | 0.794 | 3.669 | 0.023 | 0.591 | 0.008 |
| 44222.000 | 0.211 | 1.734 | 9.401 | 0.284 | 0.718 |
| 36643.000 | 0.778 | 3.011 | 0.244 | 0.602 | 0.070 |
| 53512.000 | 0.777 | 2.888 | 0.075 | 0.568 | 0.023 |
| 63553.000 | 0.775 | 13.587 | 8.045 | 0.977 | 0.935 |
| 7944.000 | 0.773 | 3.162 | 0.000 | 0.545 | 0.000 |
| 28574.000 | 0.771 | 14.746 | 8.724 | 1.000 | 0.846 |
| 18631.000 | 0.767 | 2.866 | 0.187 | 0.568 | 0.057 |
| 20052.000 | 0.751 | 3.688 | 0.410 | 0.568 | 0.126 |
| 29029.000 | 0.736 | 6.414 | 4.958 | 1.000 | 0.995 |
| 37710.000 | 0.723 | 8.879 | 5.856 | 0.886 | 0.854 |
| 52681.000 | 0.722 | 5.815 | 1.256 | 0.602 | 0.232 |
| 22749.000 | 0.721 | 2.304 | 0.163 | 0.477 | 0.048 |
| 9967.000 | 0.714 | 7.369 | 5.383 | 0.977 | 0.881 |
| 9253.000 | 0.712 | 1.790 | 0.086 | 0.443 | 0.027 |
| 58911.000 | 0.706 | 2.192 | 0.100 | 0.432 | 0.027 |
| 23486.000 | 0.700 | 6.436 | 5.411 | 0.977 | 0.993 |
| 7602.000 | 0.699 | 1.988 | 0.000 | 0.398 | 0.000 |
