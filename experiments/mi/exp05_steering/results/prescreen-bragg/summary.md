# Pre-screen in the loop on Bragg

* runs: 12 seeds paired
* best true fitness: shadow 0.305, screened 0.342; diff +0.037 [95% CI -0.100, +0.157], screened wins 8/12, Wilcoxon p = 0.380
* shadow: failure rate 0.376, probe AUROC 0.784, flagged 0.172, precision 0.587, failures caught 0.268
* screened: evaluations skipped 0.165; evaluation time of flagged children 0.216 of the total (time saved if not evaluated)
* median evaluation time: failing 5.8 s, working 7.5 s
