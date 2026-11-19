# Pre-registration: confirmation of the adaptive dial (MA-BBOB)

Registered 2026-11-19, before any run of the campaign below. The commit
that adds this file is the registration timestamp.

## Motivation

On MA-BBOB, 16 seeds (300–315), the stagnation-adaptive exploration dial
beat control at 200 evaluations: best AOCC +0.076 (95% bootstrap CI
+0.022 to +0.135, 12/16 seeds). The budget-200 extension was decided after
the budget-100 analysis, and about seven arms were compared with control,
so this result is exploratory. This campaign tests it on new seeds, and
adds the two baselines a reviewer will ask for: the same dose schedule
driving the prompt instead of the vector, and a higher sampling
temperature.

## Design

* Campaign `llamea-confirm` (`exp05_steering/exp_config.py`): LLaMEA,
  Qwen3-8B, MA-BBOB 5-D as in all earlier campaigns (4 parents, 12
  offspring, elitist; refine/new/simplify; traced; extended sandbox),
  budget 200 evaluated algorithms per run, from scratch.
* New seeds 316–339 (24 seeds), never used before.
* Arms (paired by seed):
  * `control`: unsteered;
  * `adaptive`: unchanged from the exploratory campaigns (no steering of
    the initial population; dose −0.1 after a child that improves the
    run's best, +0.025 after any other, clipped to ±0.3; layer-27
    residualised exploration direction);
  * `adaptive_prompt`: the same dose schedule, but instead of the vector
    the explore sentence (dose > 0) or exploit sentence (dose < 0) of the
    prompt baseline is appended with probability |dose| / 0.3;
  * `temp11`: unsteered, sampling temperature 1.1 instead of 0.8.
* Sample size: power analysis on the 16 exploratory seeds (SD of the
  paired difference 0.120): 24 seeds give 92% power for +0.076 and 63%
  for +0.05 (one-sided α = 0.05).

## Hypotheses and tests

**Primary (confirmatory).** H1: the best AOCC after 200 evaluations is
higher with `adaptive` than with `control`. One-sided paired t-test over
the 24 seeds, α = 0.05. Reported with the mean difference, the 95%
bootstrap CI (10,000 resamples over seeds) and the number of seeds won.
H1 is supported only if p < 0.05.

**Secondary (two-sided, Holm-corrected over these four, α = 0.05).**
1. `adaptive` vs `adaptive_prompt` (does the vector matter, or only the
   schedule?);
2. `adaptive` vs `temp11`;
3. `adaptive_prompt` vs `control`;
4. `temp11` vs `control`.

**Descriptive (no test decides the claim).** The same comparisons at 100
evaluations and on the area under the best-so-far curve; failure rate,
families written, concept diversity and measured exploration; the
adaptive dose and the share of prompted children over the run; the
pooled 40-seed estimate for `adaptive` vs `control` (seeds 300–339),
labelled as pooled exploratory + confirmatory.

## What the paper will say

* H1 supported: "confirmed on 24 new seeds" with the confirmatory numbers.
* H1 not supported: the adaptive-dial result is reported as exploratory
  and not confirmed, with both estimates.
* Whatever the secondary results, they are reported.

## Exclusions

None planned. A run that crashes for infrastructure reasons (not a
generated algorithm's failure) is restarted with the same seed; this is
logged in `exp05_steering/README.md`.
