# Pre-registration: the adaptive dial and temperature together (MA-BBOB)

Registered 2026-11-22, before any run of the campaign below and after the
confirmation (`PREREG_confirm.md`) was analysed. The commit that adds
this file is the registration timestamp.

## Motivation

In the confirmation (24 seeds 316–339, budget 200), the adaptive dial
beat control (+0.075, p = 0.005) and temperature 1.1 gained +0.061
(n.s. after Holm). The two act differently: temperature widens sampling
at every token, the dial shifts the residual stream towards exploratory
code when the run stagnates. Both raised the number of algorithm families
per run (44 and 48, control 35). Combined, they may explore more still,
or the extra variation may only add failures (19.8% and 17.9%, control
14.2%). This campaign adds the combination as a fifth arm.

## Design

* Campaign `llamea-combo` (`exp05_steering/exp_config.py`): one arm,
  `adaptive_temp11` = the `adaptive` arm of `llamea-confirm` unchanged
  (dose schedule, direction, layer, clipping, no steering of the initial
  population) with sampling temperature 1.1 instead of 0.8. Everything
  else as `llamea-confirm`: LLaMEA, Qwen3-8B, MA-BBOB 5-D, 4 parents, 12
  offspring, elitist, refine/new/simplify, traced, extended sandbox,
  budget 200, from scratch.
* Same seeds 316–339, paired by seed with the four arms of
  `llamea-confirm` (already run; their results are fixed and known).
* Run after `llamea-confirm`, with the same code, model and sandbox. The
  only known difference is time (load on the machine during evaluation);
  failure rates by timeout are reported to check for it.
* Power (SD of paired differences in the confirmation about 0.13–0.15):
  for combo vs control, 24 seeds give 86% power for +0.075 and 57% for
  +0.05 (one-sided α = 0.05). The comparisons with the components have
  about 32% power for a difference of 0.05 and 67% for 0.075; a null
  result there is not evidence of no additional gain.

## Hypotheses and tests

**Primary.** H2: the best AOCC after 200 evaluations is higher with
`adaptive_temp11` than with `control`. One-sided paired t-test over the
24 seeds, α = 0.05, with the mean difference, the 95% bootstrap CI
(10,000 resamples over seeds) and the number of seeds won.

**Secondary (two-sided, Holm-corrected over these two, α = 0.05).** Does
the combination add to either component?
1. `adaptive_temp11` vs `adaptive`;
2. `adaptive_temp11` vs `temp11`.

**Descriptive (no test decides a claim).** The interaction contrast
(combo − adaptive − temp11 + control) with its bootstrap CI; the same
comparisons at 100 evaluations and on the area under the best-so-far
curve; combo vs `adaptive_prompt`; failure rate (and the share of
timeouts), families written, family entropy, concept and code diversity,
measured exploration; the adaptive dose over the run.

## What the paper will say

* H2 supported and a secondary significant: the combination is reported
  as better than that component.
* H2 supported, secondaries not: "the combination also beats control;
  it cannot be distinguished from its components" (with the power above).
* H2 not supported: reported as such, with the estimate.

## Exclusions

None planned. A run that crashes for infrastructure reasons is restarted
with the same seed and logged in `exp05_steering/README.md`.
