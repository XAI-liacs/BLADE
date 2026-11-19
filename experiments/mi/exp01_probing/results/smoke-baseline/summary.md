# Smoke-test baseline for the pre-screen

* children: 3600; logged failure rate 0.165; re-evaluation agrees with the logged label for 0.999
* full evaluation: median 4.7 s; failing children 2.7 s, working 4.9 s; failing children use 0.124 of the total time
* smoke100 false alarms: 1

```
                            screen  flagged  failures_caught  working_lost  precision  evals_skipped  time_screen  time_saved
                              none    0.000            0.000         0.000        NaN          0.000        0.000       0.000
           probe (fixed threshold)    0.254            0.712         0.164      0.462          0.254        0.000       0.211
       smoke test, 100 evaluations    0.140            0.847         0.000      0.998          0.140        0.546      -0.470
            smoke test, 1 instance    0.159            0.966         0.000      0.998          0.159        0.624      -0.511
             probe, then smoke 100    0.287            0.909         0.164      0.522          0.287        0.406      -0.177
oracle (skip exactly the failures)    0.165            1.000         0.000      1.000          0.165        0.000       0.124
```

## smoke100 false alarms, most common errors

* On line: elite_x, elite_f = min(self.pop, key=lambda p: p[1]) (1)
