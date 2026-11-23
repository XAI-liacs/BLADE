# The adaptive dial and temperature 1.1 together (pre-registered)

24 runs (seeds 316-339) x 200 algorithms, paired with the four arms of `llamea-confirm`.

**Primary (H2, adaptive_temp11 > control, best AOCC @200, one-sided paired t): SUPPORTED.** diff +0.059 [95% CI +0.009, +0.110], t = 2.22, p = 0.0182, wins 15/24 seeds.

## Secondary (two-sided paired t, Holm over two)

* adaptive_temp11 vs adaptive: -0.016 [-0.067, +0.040], wins 10/24, p = 0.5555, Holm 1.0000
* adaptive_temp11 vs temp11: -0.002 [-0.053, +0.054], wins 10/24, p = 0.9380, Holm 1.0000

## Arm means

```
                 best100  best200  auc200  failure_rate  timeout_rate  mean_dose
arm                                                                             
control            0.585    0.664   0.568         0.142         0.011        NaN
adaptive           0.679    0.739   0.639         0.198         0.005      0.256
adaptive_prompt    0.647    0.711   0.630         0.148         0.004      0.269
temp11             0.638    0.725   0.620         0.179         0.005        NaN
adaptive_temp11    0.668    0.723   0.630         0.224         0.005      0.265
```

## Descriptive comparisons

```
              a               b              metric  n    diff    lo95   hi95  wins       t      p alternative                      role
adaptive_temp11 adaptive_prompt             best200 24  0.0123 -0.0474 0.0727    13  0.3880 0.7016   two-sided               descriptive
adaptive_temp11         control             best100 24  0.0833  0.0240 0.1419    16  2.7305 0.0119   two-sided               descriptive
adaptive_temp11        adaptive             best100 24 -0.0110 -0.0723 0.0549    10 -0.3319 0.7429   two-sided               descriptive
adaptive_temp11          temp11             best100 24  0.0304 -0.0363 0.0996    12  0.8550 0.4014   two-sided               descriptive
adaptive_temp11 adaptive_prompt             best100 24  0.0216 -0.0668 0.1160    10  0.4515 0.6558   two-sided               descriptive
adaptive_temp11         control              auc200 24  0.0621  0.0136 0.1109    16  2.4783 0.0210   two-sided               descriptive
adaptive_temp11        adaptive              auc200 24 -0.0094 -0.0690 0.0554    10 -0.2911 0.7736   two-sided               descriptive
adaptive_temp11          temp11              auc200 24  0.0096 -0.0483 0.0695    12  0.3084 0.7605   two-sided               descriptive
adaptive_temp11 adaptive_prompt              auc200 24  0.0000 -0.0785 0.0767    11  0.0005 0.9996   two-sided               descriptive
adaptive_temp11         control              auc100 24  0.0573  0.0021 0.1154    14  1.9270 0.0664   two-sided               descriptive
adaptive_temp11        adaptive              auc100 24 -0.0062 -0.0754 0.0671    10 -0.1655 0.8700   two-sided               descriptive
adaptive_temp11          temp11              auc100 24  0.0158 -0.0539 0.0871    11  0.4271 0.6733   two-sided               descriptive
adaptive_temp11 adaptive_prompt              auc100 24 -0.0150 -0.1041 0.0767    11 -0.3205 0.7515   two-sided               descriptive
 combo-adaptive  temp11-control best200 interaction 24 -0.0774 -0.1619 0.0120     6 -1.7229 0.0983   two-sided descriptive (interaction)
```

