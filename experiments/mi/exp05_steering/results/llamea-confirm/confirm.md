# Confirmation of the adaptive dial (pre-registered)

96 runs (4 arms x 24 seeds 316-339) x 200 algorithms, all complete.

**Primary (H1, adaptive > control, best AOCC @200, one-sided paired t): SUPPORTED.** diff +0.075 [95% CI +0.023, +0.127], t = 2.77, p = 0.0054, adaptive wins 17/24 seeds.

## Secondary (two-sided paired t, Holm over four)

* adaptive vs adaptive_prompt: +0.029 [-0.038, +0.094], wins 14/24, p = 0.4135, Holm 0.8270
* adaptive vs temp11: +0.014 [-0.043, +0.071], wins 13/24, p = 0.6440, Holm 0.8270
* adaptive_prompt vs control: +0.047 [-0.011, +0.108], wins 14/24, p = 0.1506, Holm 0.4517
* temp11 vs control: +0.061 [-0.011, +0.133], wins 14/24, p = 0.1126, Holm 0.4506

## Arm means

```
                 best100  best200  auc200  failure_rate  mean_dose  prompted_share
arm                                                                               
control            0.585    0.664   0.568         0.142        NaN           0.000
adaptive           0.679    0.739   0.639         0.198      0.256           0.000
adaptive_prompt    0.647    0.711   0.630         0.148      0.269           0.902
temp11             0.638    0.725   0.620         0.179        NaN           0.000
```

## Descriptive comparisons

```
              a               b                 metric  n    diff    lo95   hi95  wins       t      p alternative                 role
       adaptive         control                best100 24  0.0943  0.0224 0.1627    18  2.5857 0.0165   two-sided          descriptive
       adaptive adaptive_prompt                best100 24  0.0326 -0.0381 0.1064    16  0.8606 0.3983   two-sided          descriptive
       adaptive          temp11                best100 24  0.0414 -0.0346 0.1171    12  1.0516 0.3039   two-sided          descriptive
adaptive_prompt         control                best100 24  0.0617 -0.0280 0.1534    13  1.3088 0.2035   two-sided          descriptive
         temp11         control                best100 24  0.0529 -0.0249 0.1323    12  1.2887 0.2103   two-sided          descriptive
       adaptive         control                 auc200 24  0.0715  0.0098 0.1298    18  2.2482 0.0344   two-sided          descriptive
       adaptive adaptive_prompt                 auc200 24  0.0094 -0.0585 0.0774    12  0.2652 0.7932   two-sided          descriptive
       adaptive          temp11                 auc200 24  0.0190 -0.0447 0.0834    13  0.5667 0.5764   two-sided          descriptive
adaptive_prompt         control                 auc200 24  0.0621 -0.0108 0.1369    12  1.6139 0.1202   two-sided          descriptive
         temp11         control                 auc200 24  0.0525 -0.0155 0.1205    15  1.4588 0.1581   two-sided          descriptive
       adaptive         control                 auc100 24  0.0635 -0.0082 0.1307    17  1.7343 0.0963   two-sided          descriptive
       adaptive adaptive_prompt                 auc100 24 -0.0089 -0.0863 0.0666    11 -0.2247 0.8242   two-sided          descriptive
       adaptive          temp11                 auc100 24  0.0220 -0.0557 0.1012    13  0.5393 0.5948   two-sided          descriptive
adaptive_prompt         control                 auc100 24  0.0724 -0.0131 0.1594    12  1.6283 0.1171   two-sided          descriptive
         temp11         control                 auc100 24  0.0415 -0.0324 0.1167    13  1.0861 0.2887   two-sided          descriptive
       adaptive         control best200 pooled 300-339 40  0.0757  0.0369 0.1146    29  3.7887 0.0005   two-sided descriptive (pooled)
```

