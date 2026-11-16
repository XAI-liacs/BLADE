# llamea-bbob-f20: final best on the test instances

Best algorithm of each run (training AOCC), re-evaluated on 16 test instances. Mean over seeds; a run whose best fails on the test instances counts as missing.

| arm | n | train | test | gap (train − test) | failed on test | test − control (paired) |
|---|---|---|---|---|---|---|
| adaptive | 10 | 0.259 | 0.257 | +0.002 (+0.000 to +0.003) | 0 | +0.022 (+0.001 to +0.057) |
| control | 10 | 0.240 | 0.236 | +0.005 (+0.002 to +0.007) | 0 | – |
| exploit | 10 | 0.255 | 0.247 | +0.008 (+0.005 to +0.010) | 0 | +0.011 (-0.009 to +0.044) |
| explore | 10 | 0.258 | 0.252 | +0.006 (+0.002 to +0.011) | 0 | +0.016 (-0.006 to +0.050) |
| prompt_exploit | 10 | 0.259 | 0.252 | +0.007 (+0.004 to +0.008) | 0 | +0.016 (-0.004 to +0.049) |
| prompt_explore | 10 | 0.254 | 0.251 | +0.004 (+0.002 to +0.005) | 0 | +0.015 (-0.004 to +0.048) |
