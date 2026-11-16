# llamea-bbob-f2: final best on the test instances

Best algorithm of each run (training AOCC), re-evaluated on 16 test instances. Mean over seeds; a run whose best fails on the test instances counts as missing.

| arm | n | train | test | gap (train − test) | failed on test | test − control (paired) |
|---|---|---|---|---|---|---|
| adaptive | 10 | 0.598 | 0.577 | +0.021 (+0.002 to +0.046) | 0 | -0.000 (-0.191 to +0.192) |
| control | 10 | 0.593 | 0.578 | +0.015 (-0.003 to +0.039) | 0 | – |
| exploit | 10 | 0.381 | 0.381 | -0.000 (-0.005 to +0.004) | 0 | -0.196 (-0.371 to +0.017) |
| explore | 10 | 0.537 | 0.541 | -0.004 (-0.022 to +0.013) | 0 | -0.037 (-0.125 to +0.051) |
| prompt_exploit | 10 | 0.359 | 0.356 | +0.003 (-0.003 to +0.009) | 0 | -0.222 (-0.402 to +0.002) |
| prompt_explore | 10 | 0.352 | 0.344 | +0.008 (+0.005 to +0.013) | 0 | -0.234 (-0.433 to -0.012) |
