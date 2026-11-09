# Experiment 5: Activation steering as a variation operator

**Question.**
- Can adding a direction to the residual stream act as a controllable
  mutation operator, as an alternative to prompt engineering or
  temperature?
- Can an outer CMA-ES search the steering space for directions that make
  LLaMEA produce better algorithms?

**Builds on.**
- `HookedLLM.set_steering` (already implemented and tested).
- Candidate directions from experiment 3 (`../REPORT_exp03.md`): SAE
  features that detect a concept even with the keyword lines masked, and
  the L21 "undefined name" features. Experiment 2 already showed that a
  probe direction (parent fitness) changes the size of the rewrite.

## Design

1. **Directions.** SAE decoder columns (unit norm) of experiment 3's
   features (`exp_config.DIRECTIONS`):

   | name | layer | feature | what it detects |
   |---|---|---|---|
   | `covariance` | 15 | 5776 | covariance-matrix construction |
   | `gradient` | 15 | 6720 | finite-difference gradients |
   | `swarm` | 27 | 38837 | particle-swarm velocity updates |
   | `local_search` | 27 | 26058 | local search |
   | `undef_9410` | 21 | 9410 | a name that is not defined (any) |
   | `undef_7944` | 21 | 7944 | `func` used out of scope |

   Controls: a random unit direction per layer (`random<L>`), and
   `resample`, an unsteered sample with another seed (natural
   variability).
2. **Dose.** A dose `c` adds `c × a × d` at every generated token, where
   `a` is the feature's typical peak activation (median of its top
   examples in experiment 3). `c = 1` therefore adds what the feature
   contributes when it fires strongly, at every position. Random
   directions use the median `a` of the features at their layer.
   Mean residual norms of the generated code: L15 67, L21 127, L27 334.
3. **Prompts and sampling.** Recorded LLaMEA queries from experiment 1
   (Qwen3-8B, T = 0.8, top-p = 0.95, thinking off; max 2048 new tokens).
   The sampling seed depends only on the prompt and the replicate, not on
   the condition (common random numbers), so differences between
   conditions are paired.
4. **Measures.** Every child is evaluated on experiment 1's
   budget-enforced MA-BBOB (AOCC, 5-D, 10 instances). Per sample:
   failure, fitness, experiment 3's concept labels, undefined references
   (experiment 3, step 13) and similarity to the unsteered child.

Campaigns (`exp_config.CAMPAIGNS`):

| tag | prompts | directions | doses | samples |
|---|---|---|---|---|
| `pilot` | 4 init + 4 mutation (`main`, parent worked) | gradient, covariance, swarm, random15/27 | 0.25, 0.5, 1, 2 | 168 |
| `dose` | 8 init + 32 mutation | covariance, gradient, swarm, local search, random15/27 | 0.5, 1, 1.5 | 800 |
| `scope` | the 99 recorded queries whose child used an undefined name | undef_9410, undef_7944, random21 | −4, +4 | 1,584 (k = 2) |
| `screen` | 4 init + 12 mutation | rank-0 detector of 10 concepts at L21 and L27, swarm, random21/27 | 0.3, 0.5 × residual norm | 768 |
| `screen2` | same prompts | the 10 L21 detectors, surrogate27, crossover27, random21 | 0.1, 0.2 × residual norm | 448 |
| `llamea` | LLaMEA runs, budget 100 | arms `refine2`, `new2`, `levers`, `random` | levers of `screen` + `screen2` | 8 seeds × 4 arms |

5. **Steering as a diversity operator** (`04_select_levers.py`,
   `05_llamea.py`, `06_llamea_analyze.py`). The dose response showed that
   a lever changes *what* the model writes without making it better, so
   the operator is tested for diversity, with fitness as a
   non-inferiority check.
   - *Levers.* The `screen` campaign steers the rank-0 detector of ten
     concepts at L21 and L27, at 0.3 and 0.5 × the residual norm (a
     common scale: swarm worked at 0.33–0.49, L15/L21 broke at 0.56+).
     A lever adds its concept (paired gain ≥ 0.2, CI above 0) without
     raising failures by more than 0.15; one lever per concept.
   - *Arms.* LLaMEA as in experiment 1 (refine, new, simplify; 4
     parents, 12 offspring, elitism, 100 algorithms) plus one extra
     operator of equal weight: `refine2` (refine again, control),
     `new2` ("generate a new algorithm" again, prompt-based diversity),
     `levers` (refine prompt + a random lever) and `random` (refine
     prompt + a random direction with the lever's layer and norm). 8
     seeds per arm.
   - *Diversity* (`common/diversity.py`). Distance between two
     algorithms as 1 − CodeBLEU (symmetrised; n-gram, keyword, AST and
     data-flow match), LLaMEA's AST distance, and Jaccard distance of
     their concept sets (algorithm family). Per run: mean pairwise
     distance (all, working, final population), entropy and number of
     distinct concept sets. Per child: novelty (distance to the nearest
     earlier algorithm of the run) and step (distance to its parent),
     compared between operators.
6. **CMA-ES in steering space** (not started). Parameterise the steering vector as a
   combination of k directions with a coefficient vector z. CMA-ES
   optimises z; the fitness of z is the mean AOCC of algorithms generated
   under steering z.

## How to run

```bash
source experiments/mi/env.sh
S=experiments/mi/exp05_steering/scripts
# 1. directions, scales, residual norms, prompts (~2 min, one GPU)
CUDA_VISIBLE_DEVICES=2 uv run python $S/01_prepare.py --tag dose --device cuda:0
# 2. generate + evaluate (2 processes per GPU on GPUs 2/3; ~15 samples/min)
for k in 0 1 2 3; do CUDA_VISIBLE_DEVICES=$((2 + k / 2)) \
  uv run python $S/02_generate.py --tag dose --shard $k --n-shards 4 & done
# 3. summary per condition, paired deltas vs unsteered (CPU)
uv run python $S/03_analyze.py --tag dose
```

## Status

Steps 1–5 done (2026-11-09). Step 6 not started.

### Steered LLaMEA operator (`results/llamea/summary.md`, `diversity.png`)

4 arms × 8 seeds × 100 algorithms (3,200). Each arm is LLaMEA with
refine, new, simplify plus one extra operator; the arms differ only in
that operator. Means over runs; p from Mann–Whitney vs `refine2`.

| Arm (extra operator) | Best AOCC | Failed | Pairwise 1 − CodeBLEU | Concept Jaccard | Final pop. 1 − CodeBLEU | Families |
|---|---|---|---|---|---|---|
| `refine2` (control) | 0.722 | 0.14 | 0.497 | 0.414 | 0.231 | 24.1 |
| `new2` (prompt) | 0.686 | 0.22 | **0.540** (p 0.015) | 0.517 (p 0.05) | 0.264 | 30.6 (p 0.07) |
| `levers` (steered) | 0.679 | 0.16 | 0.523 (p 0.33) | 0.490 (p 0.16) | 0.322 (p 0.08) | 24.9 |
| `random` (random dir.) | 0.661 | 0.19 | 0.516 (p 0.28) | 0.483 (p 0.20) | 0.273 | 30.3 |

Per child of the extra operator:

| Extra operator | Lever | PSO | Gradient | Failed | AOCC | Novelty 1 − CodeBLEU | Novelty concepts |
|---|---|---|---|---|---|---|---|
| `x_refine` | — | 0.15 | 0.66 | 0.12 | 0.43 | 0.250 | 0.049 |
| `x_new` | — | 0.26 | 0.66 | 0.28 | 0.28 | 0.328 | 0.156 |
| `x_levers` | swarm | **0.99** | 0.50 | 0.18 | 0.27 | 0.303 | 0.140 |
| `x_levers` | gradient27 | 0.20 | **0.88** | 0.08 | 0.36 | 0.267 | 0.019 |
| `x_random` | random (both norms) | 0.07 | 0.64 | 0.20 | 0.37 | 0.262 | 0.088 |

- **The levers work inside LLaMEA.** Steered along swarm, 99% of
  children contain a particle swarm (refine control 15%), without the
  failure rate of the "new" prompt. Gradient: 88% vs 66%. A random
  direction of equal norm does neither.
- **The swarm lever is as novel as the "new" prompt** (concept
  novelty 0.140 vs 0.156, CodeBLEU novelty 0.303 vs 0.328) and far more
  than refine (0.049, 0.250), with a lower failure rate (0.18 vs 0.28).
  The gradient lever adds no novelty: gradient steps are already in two
  thirds of the code.
- **But population diversity barely moves.** Only the `new2` arm is
  more diverse than the control at p < 0.05. The `levers` arm's
  per-run diversity is between control and `new2` and not significant
  with 8 seeds; only its final population is somewhat more diverse
  (0.322 vs 0.231, p 0.08). One extra operator in four, half of whose
  children go to one lever, is a small push, and selection removes the
  PSO children: they are worse (AOCC 0.27 vs 0.43), and the arm's best
  algorithms contain a PSO no more often than the control's (2/8 each).
- **Fitness**: no arm differs from the control (all p > 0.4); the
  control is numerically best.
- So steering is a *directed* diversity operator: it puts a chosen
  concept into a child reliably and cheaply, which a prompt cannot. As a
  general diversity operator it is limited by the number of levers: two
  usable features (step 4) give the operator two places to go. What
  would make it useful: more levers (other models, SAEs per concept,
  contrastive directions), a lever schedule that steers *towards*
  under-represented families instead of at random, or protecting the
  steered children from early selection (niching).

### Lever screen (`results/screen/levers.md`)

Of the 20 concept detectors (10 concepts × L21, L27), only two are
clean levers (n = 16 children per cell, so CIs are wide):

| Lever | Dose (× residual norm) | Concept rate, unsteered → steered | Failures (unsteered 0.12–0.25) |
|---|---|---|---|
| `swarm` (L27 f38837) | 0.5 | particle swarm 0.12 → 1.00 | 0.06 |
| `gradient27` (L27 f49680) | 0.5 | gradient 0.56 → 0.88 | 0.12 |

- `gradient21` at 0.3 also qualifies; one lever per concept is kept.
- **L21 is a cliff.** At 0.3–0.5 most L21 detectors break the code
  (failures 0.75–1.0); a random L21 direction of the same norm does not
  (0.2). Some still add their concept on the way: particle_swarm21 0.3
  (+0.81, failures 0.81), restart21 0.5 (+0.44, 0.88). At 0.1–0.2
  (`screen2`) the code survives but no concept rises reliably.
- **Most L27 detectors do nothing** at these doses (covariance,
  restart, elitism, adaptive parameters: gain ≤ 0.13); surrogate27 adds
  surrogates only while breaking the code (0.3: +0.25, failures 0.81).
- So a detector feature is rarely a lever: two of 20 steer their
  concept cleanly, and the step 5 operator draws between these two.


### Undefined-name features (`results/scope/summary.md`)

The 99 recorded queries whose child used an undefined name, 2
replicates per condition (198 samples each), steered at L21 by ±4 ×
the feature's peak activation (norms: f9410 71, f7944 30, random 50;
residual norm 127). "Undefined" is experiment 3's step 13 label (an
out-of-scope `func`, another function's local, or a name bound
nowhere).

| condition | undefined name (Δ, CI) | NameError | failures (Δ, CI) | Δ AOCC (CI) | tokens |
|---|---|---|---|---|---|
| `none` | 0.045 | 0.05 | 0.25 | | 903 |
| `resample` | 0.061 (+0.02, −0.03 to 0.06) | 0.05 | 0.27 (+0.02) | −0.01 | 899 |
| random21 −4 | 0.071 (+0.03, −0.02 to 0.08) | 0.06 | 0.31 (+0.06, −0.02 to 0.14) | −0.03 | 986 |
| random21 +4 | 0.136 (+0.09, 0.04 to 0.15) | 0.14 | 0.36 (+0.11, 0.02 to 0.20) | −0.04 | 859 |
| f7944 −4 | 0.051 (+0.01, −0.04 to 0.05) | 0.05 | 0.34 (+0.09, 0.01 to 0.18) | −0.04 | 856 |
| **f7944 +4** | **0.303 (+0.26, 0.19 to 0.33)** | **0.23** | 0.47 (+0.22, 0.12 to 0.31) | −0.08 | 954 |
| f9410 −4 | 0.066 (+0.02, −0.03 to 0.07) | 0.07 | 0.46 (+0.21, 0.12 to 0.30) | −0.06 | 1,213 |
| f9410 +4 | 0.056 (+0.01, −0.04 to 0.06) | 0.05 | 0.85 (+0.60, 0.51 to 0.68) | −0.21 | 304 |

- **The prompts are not prone to undefined names.** Resampled without
  steering, only 4.5% of their children use an undefined name (the
  recorded child did by selection): the original error was mostly a
  sampling accident. Suppression (dose −4) therefore cannot be tested
  here; neither feature lowers the rate, but there is little to lower.
- **f7944 is a lever.** Adding it raises the undefined-name rate
  sevenfold (0.045 → 0.30) and NameErrors from 5% to 23%, with the
  smallest norm of all steers (30, a quarter of the residual norm). It
  produces all three kinds, not only `func` out of scope (0.11) but
  also other functions' locals (0.13) and names bound nowhere (0.12);
  in experiment 3 f7944 *read* mostly `func`. A random direction of
  larger norm (50) has a third of the effect (+0.09), so a general
  perturbation at L21 makes the model a little sloppier with names, and
  f7944 much more.
- **f9410 is a detector, not a lever.** It reads undefined names more
  generally than f7944 (experiment 3), but adding it does not produce
  them: the model ends the code block early (304 tokens instead of 903,
  a third of the code) and 85% of the children fail as stubs
  (TypeError, unclosed brackets). Subtracting it makes the output
  longer (23% hit the token limit) and fail more (0.46), again without
  changing the undefined-name rate. Reading f9410 as "something is
  wrong here, wrap up" would fit both directions, but this campaign does
  not test that.
- No steer at L21 improves fitness; all lower AOCC.

### Dose response (`results/dose/summary.md`, `dose_response.png`)

40 prompts (8 init, 32 mutation), one child per condition; Δ is the
paired difference to the unsteered child (95% bootstrap CI over
prompts). Unsteered: 15% failures, AOCC 0.297 (failures as 0), 0.350
for working code.

| direction | dose | target concept (Δ, CI) | failures (Δ, CI) | Δ AOCC (CI) |
|---|---|---|---|---|
| swarm (L27) | 0.5 | 0.65 (+0.48, 0.33 to 0.63) | 0.18 (+0.03) | −0.01 (−0.07 to 0.06) |
| | 1 | 0.93 (+0.75, 0.63 to 0.88) | 0.20 (+0.05) | −0.00 (−0.07 to 0.06) |
| | 1.5 | 1.00 (+0.83, 0.70 to 0.93) | 0.05 (−0.10) | +0.00 (−0.06 to 0.06) |
| gradient (L15) | 0.5 | 0.73 (+0.10, −0.05 to 0.28) | 0.08 (−0.08) | +0.04 (−0.02 to 0.11) |
| | 1 | 0.70 (+0.08, −0.08 to 0.23) | 0.13 (−0.03) | +0.02 (−0.04 to 0.08) |
| | 1.5 | 0.35 (−0.28, −0.45 to −0.10) | 0.68 (+0.53, 0.33 to 0.73) | −0.18 (−0.28 to −0.09) |
| covariance (L15) | 0.5 / 1 / 1.5 | 0.18 / 0.23 / 0.28 (+0.03 / +0.08 / +0.13, all CIs include 0) | 0.10 / 0.25 / 0.30 | ≈ 0 |
| local search (L27) | 0.5 / 1 / 1.5 | 0.05 / 0.03 / 0.03 (≤ 0) | 0.13–0.18 | ≈ +0.03 |
| random15, random27 | 0.5–1.5 | — | 0.10–0.18 (all CIs include 0) | +0.03 to +0.06 |
| `resample` | — | — | 0.10 | +0.02 |

- **Swarm steering is a clean, dosed operator.** The rate of particle
  swarms in the child rises from 0.18 to 0.65, 0.93 and 1.00 with the
  dose, in init and mutation prompts alike (mutation 0.22 → 1.00; init
  0 → 1.00), without more failures. Other concepts stay at their
  unsteered rates, except gradients, which fall slightly (0.63 → 0.45),
  as the swarm replaces part of the code. The steered algorithms are
  rarely *named* as swarms (≤ 5% of names): the swarm is added as a
  component of the algorithm the prompt asked for. It does not make the
  algorithms better: AOCC is unchanged, working-code fitness 0.31–0.37
  against 0.35.
- **Gradient steering has no usable window.** Up to dose 1 the gradient
  rate rises by at most 0.10 (CI includes 0); at dose 1.5 generation
  degenerates (83% of children run to the 2,048-token limit and fail to
  execute), so the concept rate *falls*. The pilot's "7 of 8" was the
  high unsteered rate (0.63) plus noise.
- **Covariance and local search do not steer.** Covariance creeps up
  (+0.13 at dose 1.5, CI includes 0) together with failures; local
  search does nothing. Both features read their concept (experiment 3)
  but writing along them does not produce it: a detector is not
  necessarily a lever.
- **Controls.** Random directions of the same norm leave failures and
  every concept rate at their unsteered values. All conditions,
  including random and `resample`, have a slightly higher AOCC than
  `none` (+0.02 to +0.06); the unsteered draw happens to be a little low,
  so AOCC changes are compared with the random controls, not with zero.
- **How much of the code change is the steer?** Similarity to the
  unsteered child is 0.73 for `resample` and 0.74–0.80 for random and
  most feature steers: a weak steer changes the code about as much as a
  new seed does. Only swarm (0.68 at 1.5) and gradient at 1.5 (0.43) go
  below it.

### Pilot (`results/pilot/summary.md`)

8 prompts, 1 sample per condition; too small for CIs and only used to
pick the doses.

- **Swarm steering writes PSO.** At L27, `swarm` at dose 1 put a particle
  swarm into 8 of 8 children (unsteered 0 of 8) with no failures; at
  dose 0.5, 5 of 8.
- **Gradient steering writes gradients but breaks code.** At dose 1, 7 of
  8 children have finite-difference gradients (unsteered 4 of 8), and 4
  of 8 fail (unsteered 1).
- **Covariance steering does little:** 1–2 of 8 at any dose.
- **Dose 2 breaks generation at L15** for every direction, random
  included: the output degenerates and runs to the token limit. At L27
  (residual norm 334) random dose 2 is still harmless.
- Even the smallest steer changes the code a lot (similarity to the
  unsteered child 0.61–0.70 at dose 0.25): at T = 0.8, any change in the
  logits sends the sample down another path. The `resample` control in
  the later campaigns measures how much of that is ordinary sampling
  variability.
