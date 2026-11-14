# Failure pre-screen on new problems (frozen MA-BBOB probe)

Probe `child_failed_gen_mean_L21.npz`, threshold -0.293 (score > threshold is skipped). Activations: teacher-forced code mean at layer 21.

| problem | n | failure rate | frozen AUROC | frozen AUROC, unsteered | skipped at threshold | precision | failures caught | precision, top 21% skipped | refit AUROC (CV) | code length AUROC |
|---|---|---|---|---|---|---|---|---|---|---|
| Bragg | 408 | 0.70 | 0.77 (0.72 to 0.82) | 0.82 | 1.00 | 0.70 | 1.00 | 0.92 | 0.84 | 0.55 |
| packing | 384 | 0.51 | 0.72 (0.67 to 0.77) | 0.76 | 0.95 | 0.52 | 0.97 | 0.79 | 0.73 | 0.57 |
| MA-BBOB corpus (reference) | 4712 | 0.19 | 0.82 (0.81 to 0.84) | - | 0.52 | 0.32 | 0.88 | 0.53 | 0.88 | nan |
