# llamea-bbob-f9: final best on the test instances

Best algorithm of each run (training AOCC), re-evaluated on 16 test instances. Mean over seeds; a run whose best fails on the test instances counts as missing.

| arm | n | train | test | gap (train − test) | failed on test | test − control (paired) |
|---|---|---|---|---|---|---|
| adaptive | 10 | 0.331 | 0.327 | +0.004 (+0.000 to +0.008) | 0 | -0.002 (-0.010 to +0.006) |
| control | 10 | 0.333 | 0.328 | +0.005 (+0.002 to +0.008) | 0 | – |
| exploit | 10 | 0.337 | 0.331 | +0.006 (-0.001 to +0.014) | 0 | +0.003 (-0.008 to +0.013) |
| explore | 10 | 0.347 | 0.336 | +0.011 (+0.006 to +0.017) | 0 | +0.008 (-0.000 to +0.018) |
| prompt_exploit | 10 | 0.340 | 0.333 | +0.007 (+0.003 to +0.012) | 0 | +0.005 (-0.006 to +0.013) |
| prompt_explore | 10 | 0.333 | 0.327 | +0.006 (+0.000 to +0.013) | 0 | -0.001 (-0.013 to +0.011) |
