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
| `contrast-c` | 4 init + 12 mutation (seed 1) | contrastive concept directions of 10 rare concepts at L15, random15 | 0.15, 0.3, 0.5 × residual norm | 560 |
| `contrast-b` | same prompts | exploration direction, raw and concept-residualised, at L15 and L27; random15/27 | ±0.15, ±0.3, ±0.5 × residual norm | 608 |
| `dial` | 8 init + 24 mutation (seed 2), k = 2 | residualised exploration direction at L15 and L27; random15/27 | ±0.2, ±0.4 × residual norm | 1,152 |

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
5b. **More levers: contrastive directions** (`07_corpus.py`,
   `08_contrast.py`, campaigns `contrast-c`, `contrast-b`, `dial`). SAE
   detectors gave only two clean levers, and no lever for rare concepts
   or for search behaviour. Directions are instead taken from a labelled
   corpus as differences of mean activations.
   - *Corpus* (`results/corpus/directions.md`). 4,712 algorithms: every
     algorithm of experiment 1 `main` and of step 5 `llamea`, plus 312
     elicited ones (24 initial prompts per target with one extra
     sentence, "The algorithm must use …", for 12 targets: 10 rare
     concepts, "strong exploration", "strong exploitation", plus 24
     plain prompts). All are re-evaluated on the traced problem
     (`common.problems.TracedMABBOB`), which records every evaluation
     and computes BLADE's search-behaviour metrics (exploration %,
     distance to the best so far, intensification, stagnation, success
     rate).
   - *Directions.* Every algorithm is teacher-forced under one neutral
     initial prompt; the mean residual over its code tokens at L9, L15,
     L21, L27, L33 is its activation. Concept direction = mean(with) −
     mean(without); behaviour direction = mean(top third) − mean(bottom
     third) of a metric; the `_resid` variant first regresses the
     activations on the 18 concept labels, so the direction is not just
     "another algorithm family". Validated by 5-fold held-out AUROC
     (concepts) and Spearman (behaviour).
   - *Screens.* `contrast-c` steers the ten rare-concept directions at
     L15 (best held-out layer); `contrast-b` steers the exploration
     direction (raw and residualised, L15 and L27) at signed doses;
     `dial` repeats the residualised directions on new prompts with 4×
     the samples. Behaviour of the children comes from the traced
     problem; the dial is tested by the slope of each metric on signed
     dose (prompt fixed effects, bootstrap over prompts), against random
     directions of the same norm.
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

Steps 1–5, 5b, 5c and 5d done (2026-11-13). Stage 1 of
`../paper_gecco2027/PLAN.md` running (MA-BBOB arms done 2026-11-14) (`scripts/run_stage1.sh`, started
2026-11-13): prompt-baseline and adaptive-dial arms on MA-BBOB, Bragg and
packing, more MA-BBOB seeds, more Bragg screen samples. E4 done (below).

### Stage 1 (`paper_gecco2027/PLAN.md`)

**Arms (`exp_config.STAGE1_ARMS`, `05_llamea.py`).**

* `prompt_explore` / `prompt_exploit`: the corpus elicitation sentence
  ("The algorithm must use strong global exploration: ..." /
  "... strong exploitation: ...") is appended to the prompt of every
  generated algorithm, initial population included. This is the obvious
  alternative to the dial.
* `adaptive`: the dial with the dose set by the 1/5th success rule. The
  initial population is not steered (the Bragg campaign showed that
  steering it breaks the start). The dose starts at 0, moves −0.1 after a
  child that improves the run's best and +0.025 after any other child,
  clipped to [−0.3, +0.3], so it is stationary at a success rate of 1/5.
  It explores while the run stagnates and exploits after progress.
  Note: this is the 1/5th rule's arithmetic with the opposite sign
  (Rechenberg: success -> larger steps), i.e. stagnation-triggered
  exploration. On MA-BBOB the success rate is far below 1/5, so the dose
  saturates near +0.3 (see the result below).
* `adaptive_cal` (`llamea-stage1-cal`, MA-BBOB seeds 300–307, started
  2026-11-14): the same stagnation logic, calibrated. The control's rate
  of children that improve the run's best is 0.055 on MA-BBOB, 0.064 on
  Bragg and 0.038 on packing, so the target is 0.05: −0.15 after an
  improving child, +0.0079 after any other (about 38 non-improving
  children, three generations, from 0 to +0.3). A run that progresses at
  the usual pace stays near 0. If it does better than `adaptive`, it
  replaces `adaptive` on later problems (stage 2).

Campaigns: `llamea-stage1` (MA-BBOB, seeds 300–307, pairs with
`llamea-dial`), `llamea-stage1-{bragg,packing}` (seeds 400–405, pair with
`llamea-dial-<p>`), `llamea-stage1-seeds` (control, explore, exploit and
the three new arms on seeds 308–315). `transfer-bragg` gets 32 more
initial-prompt samples for none and the dial at ±0.4.

**MA-BBOB result (`results/llamea-stage1`, finished 2026-11-14; paired
by seed with the `llamea-dial` control, `07_paired.py llamea-stage1 --with
llamea-dial`, output in `paired.txt`).** Differences against control, mean
over the 8 seeds with 95% bootstrap CIs:

| arm | best AOCC | failed | families | concept diversity | exploration % |
|---|---|---|---|---|---|
| adaptive | +0.07 (−0.05 to +0.19) | +0.07 (+0.02 to +0.12) | +7.9 (+0.4 to +14.6) | +0.08 (+0.03 to +0.12) | +8.9 (−1.1 to +17.6) |
| prompt_explore | +0.05 (−0.10 to +0.19) | +0.02 (−0.06 to +0.10) | −8.5 (−16.4 to −0.8) | +0.04 (−0.10 to +0.17) | +15.9 (+6.1 to +26.4) |
| prompt_exploit | −0.01 (−0.12 to +0.11) | −0.02 (−0.09 to +0.04) | −7.9 (−16.0 to +0.6) | −0.03 (−0.09 to +0.04) | −2.3 (−8.5 to +4.0) |
| dial explore (5c) | +0.05 (−0.10 to +0.19) | +0.06 (+0.00 to +0.11) | +3.0 (−5.6 to +11.9) | −0.04 (−0.12 to +0.04) | +13.0 (+8.5 to +17.7) |
| dial exploit (5c) | −0.18 (−0.34 to −0.01) | −0.01 (−0.10 to +0.06) | −9.4 (−18.6 to −1.4) | +0.03 (−0.07 to +0.13) | +9.5 (−7.4 to +28.0) |

* No arm improves the best AOCC significantly; the sign is positive for
  adaptive, both explore arms and anneal (+0.05 to +0.07), negative for
  the exploit dial.
* The prompt moves the behaviour about as much as the dial in the explore
  direction (+15.9 vs +13.0 points exploration), at no significant cost in
  failures, but it narrows the search: about 8 fewer algorithm families,
  and the sentence leaves a strong lexical imprint (prompt_exploit: 83% of
  working algorithms use step-size adaptation and 52% a covariance, vs 21%
  and 8% in control; prompt_explore drops DE/crossover). The exploit
  prompt does not lower the measured exploration.
* The adaptive dial is the only arm that widens the search: +7.9
  families and +0.08 concept diversity, at +7 points failures. Its 1/5th
  rule is mostly saturated: the success rate stays well below 1/5, so 97%
  of the steered children get a positive dose (mean +0.25, 59% at the +0.3
  cap). In practice it is "explore after an unsteered start", with short
  exploit dips after improvements.

**Bragg and packing (`results/llamea-stage1-{bragg,packing}`, finished
2026-11-14; `07_paired.py llamea-stage1-<p> --with llamea-dial-<p>`).**
Differences against control, 6 seeds, 95% bootstrap CIs; the step 5d dial
arms are repeated for reference:

| arm | Bragg best fitness | packing best fitness | packing strictly feasible |
|---|---|---|---|
| adaptive | −0.29 (−0.50 to −0.05) | −0.20 (−0.60 to +0.16) | −0.21 (−0.39 to +0.01) |
| prompt_explore | −0.15 (−0.43 to +0.15) | −0.18 (−0.47 to +0.08) | −0.23 (−0.40 to −0.04) |
| prompt_exploit | −0.08 (−0.31 to +0.15) | −0.30 (−0.53 to −0.05) | −0.16 (−0.35 to +0.00) |
| anneal (5d) | −0.27 (−0.44 to −0.10) | +0.09 (−0.12 to +0.30) | −0.09 (−0.23 to +0.06) |
| explore (5d) | −0.21 (−0.44 to +0.01) | −0.37 (−0.91 to +0.13) | −0.28 (−0.38 to −0.17) |
| exploit (5d) | −0.36 (−0.56 to −0.19) | −0.42 (−0.62 to −0.22) | −0.28 (−0.43 to −0.11) |

* On the two new problems nothing beats the unsteered control. Leaving
  the initial population unsteered (adaptive) does not rescue Bragg: the
  adaptive dial is as harmful as anneal there (it sits near +0.3 most of
  the time, as on MA-BBOB).
* The prompt is not a free lunch either: both prompts cost strict
  feasibility on packing as much as the dial does (−0.16 to −0.23), and
  the exploit prompt lowers the best packing fitness. So the packing cost
  comes from pushing the search behaviour at all, not from activation
  steering as such. On Bragg the prompts are the least harmful arms.
* Anneal remains the only arm that is neutral on packing.

**Anytime performance (`auc_best` in `paired.txt`: area under the
best-so-far curve, mean over the 100 evaluations; failed algorithms count
as the run's worst).** Paired differences against control:

| arm | MA-BBOB | Bragg | packing |
|---|---|---|---|
| adaptive | +0.02 (−0.10 to +0.13) | −0.17 (−0.33 to +0.00) | −0.18 (−0.45 to +0.09) |
| prompt_explore | −0.01 (−0.13 to +0.10) | −0.09 (−0.28 to +0.11) | −0.22 (−0.40 to −0.02) |
| prompt_exploit | −0.05 (−0.16 to +0.09) | −0.08 (−0.23 to +0.10) | −0.24 (−0.37 to −0.07) |
| anneal | −0.01 (−0.15 to +0.12) | −0.23 (−0.38 to −0.07) | +0.13 (−0.10 to +0.34) |
| explore | +0.02 (−0.13 to +0.14) | −0.17 (−0.33 to −0.02) | −0.34 (−0.77 to +0.06) |
| exploit | −0.21 (−0.36 to −0.07) | −0.26 (−0.37 to −0.16) | −0.46 (−0.71 to −0.22) |

The area tells the same story as the final best: the MA-BBOB gains of
adaptive and explore in final AOCC (+0.05 to +0.07) shrink to +0.02, so
they come late in the run (the question behind extending control and
`adaptive_cal` to 200 evaluations).

**Budget 200 (`llamea-ext200`, `scripts/run_ext200.sh`).** Control and
`adaptive_cal` on MA-BBOB seeds 300–307 continue from their 100-evaluation
state to 200 evaluations. BLADE's LLaMEA has no usable warm start (the
LLaMEA pickle is never written because BLADE disables LLaMEA's own
logger, and `get_population_from` takes the last `n_parents` log lines,
which is not the elitist population), so `05_llamea.py` restores the run
itself (`LLaMEACampaign.warm_from`): it reads the source run's
`log.jsonl`, copies it into the new run's log (the budget counts its
lines), replays the (μ + λ) selection generation by generation to recover
the parent population, the best so far and the generation counter, and
replays the adaptive dose. Checked on a smoke run (`llamea-ext-smoke`, 12
→ 16 evaluations). Not restored: the RNG state, so the continuation is a
valid run but not bit-identical to a 200-evaluation run from scratch.
Starts on GPU 2 when `llamea-stage1-cal` finishes; raw data are backed up
in `$MI_CACHE/backups/` first.

**E4: the failure pre-screen on new problems
(`results/prescreen-transfer`, `12_prescreen_transfer.py`).** The frozen
MA-BBOB child-failure probe (`child_failed_gen_mean_L21`, experiment 1) on
the screens' code activations (teacher-forced code mean, layer 21):

| problem | n | failure rate | frozen AUROC | refit AUROC (CV) | precision, top 21% skipped |
|---|---|---|---|---|---|
| Bragg | 408 | 0.70 | 0.77 (0.72 to 0.82) | 0.84 | 0.92 |
| packing | 384 | 0.51 | 0.72 (0.67 to 0.77) | 0.73 | 0.79 |
| MA-BBOB corpus | 4,712 | 0.19 | 0.82 (0.81 to 0.84) | 0.88 | 0.53 |

The probe's ranking transfers: without retraining it is almost as good
as a probe fitted on each problem, and far above code length (AUROC
0.56–0.57). Its threshold does not: on this representation (which
differs from the in-loop one even on MA-BBOB) it flags 95–100% of the
code. A pre-screen on a new problem needs a recalibrated threshold, for
example the score quantile that skips a fixed share of evaluations.

### Transfer of the dial to other problems (step 5d: `results/transfer-bragg`, `results/transfer-packing`)

**Problems and sandbox fixes (`common/transfer_problems.py`).** Two BLADE
benchmarks: mini-Bragg photonics (10 layer thicknesses, minimise
1 − reflectivity; `TracedPhotonics`) and unit-square circle packing with
26 circles (AlphaEvolve B.12, maximise the sum of radii;
`SandboxedUnitSquarePacking`). Probe programs (a module-level helper
function, helper class and import; scipy and sklearn; a random search)
showed that BLADE's versions fail most code for reasons unrelated to the
algorithm. The subclasses fix this and leave the benchmarks unchanged:

* both run code with `exec(code, globals, locals)`, which hides
  module-level helpers and imports inside the class (`NameError`), the
  same bug as MA-BBOB. Fixed with one namespace;
* photonics: `pyGDM2` imports `six`, which nothing declares and the
  sandbox lacks. Every evaluation failed, so `six==1.17.0` is pinned;
* photonics: AOCC is computed on the raw cost in a log window of
  [1e-8, 1e2], while costs lie in [0.165, 1]. Every algorithm scores
  0.20–0.28; the intended bounds are computed but never used. Replaced
  by the AOCC of the log regret to the optimum (cost 0.164856, the
  quarter-wave stack, confirmed by differential evolution). Random
  search: 0.11;
* photonics: the problem description and algorithmic insights are only
  in `get_prompt`, which BLADE's LLaMEA wrapper does not call, so
  LLaMEA never saw them. They are folded into the task and example
  prompts;
* photonics: an algorithm that overspends the budget scores 0. It is now
  truncated at the budget (as `BudgetEnforcedMABBOB`). The 2-D BBOB test
  run is dropped: it rejected code that uses the problem's structure
  (`dim // 2` periods) but works on the problem itself;
* packing: the example code in the prompt mixes tab and space
  indentation. Fixed;
* packing: 96% of the model's packings are infeasible (circles outside
  the square or overlapping, by 0.01–0.9), and BLADE scores those −inf.
  The fitness is now the sum of radii after a repair: radii are clipped
  to the walls, then scaled by the smallest d_ij / (r_i + r_j) over
  overlapping neighbours. The feedback names the violation, and strict
  feasibility is kept in the metadata (`feasible_f` below);
* the sandbox gets the extended packages of step 5c (scipy,
  scikit-learn). numpy versions were checked: main venv 2.3.0, sandboxes
  2.3.0 (photonics) and 2.5.4
  (packing).

**Screen (`10_transfer.py`, `11_transfer_analyze.py`).** LLaMEA's own
prompts for each problem: the initial prompt (32 samples per condition)
and the refine prompt on the first 16 working unsteered initial
algorithms (2 each; Bragg had only 10 such parents). The dial
`b_exploration_pct_resid@L27` is applied at −0.4, −0.2, +0.2 and +0.4,
plus a random direction at +0.4, with seeds shared across conditions.
Bragg is traced (exploration % and intensification measured as on
MA-BBOB). Packing calls no black box, so exploration is read out by a
ridge probe on the L15 code activations, trained on the 3,798 MA-BBOB
corpus algorithms (CV Spearman 0.66 there).

Slopes per unit dose (prompt × replicate fixed effects, 95% bootstrap
CI), init and refine pooled:

| problem | metric | slope (95% CI) | random dir. +0.4 vs none |
|---|---|---|---|
| Bragg | exploration % (measured) | **+30 (2 to 55)** | −2 (−28 to 29) |
| Bragg | intensification (measured) | **−0.46 (−0.74 to −0.19)** | +0.14 (0.02 to 0.29) |
| Bragg | AOCC (working) | +0.046 (−0.006 to 0.085) | +0.04 (−0.04 to 0.13) |
| Bragg | failure rate | −0.03 (−0.18 to 0.12) | +0.02 |
| Bragg | DE / local search rate | **+0.46 / −0.39** | |
| packing | exploration % (probe) | +2 (−13 to 15) | +11 (−5 to 25) |
| packing | sum of radii (repaired) | +0.03 (−0.40 to 0.46) | −0.35 (−0.77 to 0.18) |
| packing | strictly feasible | +0.02 (−0.05 to 0.09) | −0.09 (−0.19 to −0.02) |
| packing | failure rate | **−0.37 (−0.55 to −0.19)** | +0.16 (−0.02 to 0.33) |
| packing | population / local search rate | **+0.62 / −0.90** | |

* **Bragg: the dial transfers.** Measured exploration rises and
  intensification falls with the dose, in the same direction as on
  MA-BBOB and with a similar size (+20 and −0.16 there). The random
  direction moves intensification the other way. The effect rests on
  few working algorithms (65–88% of Bragg code fails at every dose;
  about 105 working samples), and most of it comes from +0.4 on the
  initial prompt (exploration 91% vs 46% unsteered). Updated 2026-11-14
  with 32 more initial-prompt samples for none and ±0.4 (n = 64 each):
  the behaviour effect holds, but the AOCC gain does not. Paired AOCC at
  +0.4 is now +0.04 (−0.04 to +0.10), down from +0.09 (0.05 to 0.13) on
  32 samples, and the dose slope on AOCC includes 0.
* **On new problems the dial also changes the algorithm family.** On
  MA-BBOB the residualised direction was nearly orthogonal to the
  concept directions. On Bragg it trades local search for differential
  evolution, and on packing it trades local search (0.88 of the code at
  −0.4, 0.04 at +0.4 on the initial prompt) for population-based code
  and PSO. On packing, "explore" also cuts failures, from 0.75 at −0.4
  to 0.28 at +0.4.
* **The probe readout does not transfer.** On Bragg, the corpus probe's
  readout and the measured exploration are uncorrelated (Spearman −0.08,
  120 working algorithms), and the probe moves opposite to the measured
  dial effect. A probe trained on MA-BBOB code reads the code's style,
  not the behaviour on a new problem. The packing probe readout (flat)
  is therefore no evidence either way. The family shift is the
  measurable effect there.
* **Packing score does not move.** The repaired sum of radii (≈1.0–1.3,
  best known 2.635) and the strictly feasible rate (0–16%) are flat in
  the dose; the random direction lowers feasibility.

**Dial in LLaMEA on Bragg and packing (`results/llamea-dial-bragg`,
`results/llamea-dial-packing`; `06_llamea_analyze.py`, `07_paired.py`).**
The step 5c campaign design on both problems: arms control, explore
(+0.3), exploit (−0.3), anneal (+0.3 → −0.3 over the run) and randomdir,
6 seeds (400–405) × 100 algorithms per arm, 3,000 algorithms per
problem. Differences are paired by seed, 95% bootstrap CIs.

| | control | explore | exploit | anneal | randomdir |
|---|---|---|---|---|---|
| Bragg best AOCC | 0.58 | 0.37 (−0.21, −0.44 to 0.01) | **0.23 (−0.36, −0.57 to −0.18)** | **0.32 (−0.27, −0.43 to −0.10)** | 0.43 (−0.15, n.s.) |
| Bragg runs with best > 0.6 | 4/6 | 1/6 | 1/6 | 0/6 | 2/6 |
| Bragg failure rate | 0.37 | 0.48 | **0.48** | 0.43 | 0.40 |
| Bragg failure, initial population | 0.71 | 1.00 | 0.92 | 0.96 | 0.83 |
| Bragg exploration % (working) | 36 | 34 | 34 | 40 | 31 |
| Bragg DE / local search share | 0.69 / 0.81 | 0.28 / 0.41 | 0.20 / 0.79 | 0.38 / 0.40 | 0.48 / 0.72 |
| packing best sum of radii (repaired) | 2.19 | 1.82 (−0.37, −0.89 to 0.13) | **1.77 (−0.42, −0.61 to −0.22)** | 2.28 (+0.09, n.s.) | 2.16 (≈) |
| packing strictly feasible (working) | 0.37 | **0.10** | **0.09** | 0.29 | **0.14** |
| packing best strictly feasible score | 2.10 | **1.22** | **1.21** | 2.24 | 2.01 |
| packing local search / population share | 0.17 / 0.14 | 0.46 / 0.27 | 0.82 / 0.05 | 0.53 / 0.14 | 0.35 / 0.15 |

* **The screen effects do not survive the loop.** On Bragg the screen
  found more exploration and better AOCC at +0.4; in LLaMEA no arm
  raises the measured exploration (31–40% everywhere, CIs ±15), and
  every steered arm is worse than control. Bragg runs are bimodal (best
  AOCC about 0.8 when a run finds the quarter-wave stack, 0.2–0.4
  otherwise), and control finds it in 4 of 6 runs.
* **The initial population is the weak point.** Steering at +0.3 on
  LLaMEA's initial prompt breaks almost all initial Bragg algorithms
  (explore 1.00, anneal 0.96 failed vs 0.71 control), so the steered
  runs start from worse or random parents. The screen used 32 samples
  per condition and kept only working ones; the loop has 4 initial
  samples per run.
* **Family shifts persist, but are not specific.** Exploit concentrates
  packing code on local search (0.82) and removes population code
  (0.05); explore keeps more population code (0.27). On Bragg both signs
  cut DE from 0.69 to 0.20–0.28 and crossover drops under exploit
  (0.81 → 0.23). Against control, both signs move in the same direction
  for most concepts, as on MA-BBOB (step 5c).
* **Steering costs constraint handling on packing.** Strict feasibility
  of working packings falls from 0.37 to about 0.1 under explore,
  exploit and the random direction, while the failure rate is flat.
  Any perturbation of this size degrades the model's care with the
  disjointness and boundary constraints, which the repaired score hides.
* **Anneal is the safe setting on packing (and MA-BBOB), not on
  Bragg.** On packing it matches control on every measure; on Bragg it
  loses through the broken initial population.
* **Conclusion for the paper.** A fixed dial is a behaviour lever in
  single-shot generation (screens, three problems), but in the closed
  loop its effect is swamped by selection and by its cost in
  correctness. This motivates the adaptive dial (dose chosen from search
  progress, zero at initialisation) and the prompt baseline of stage 1.

### Dial in LLaMEA and re-evaluation (step 5c: `results/llamea-dial`, `results/reeval`)

**Re-evaluation in an extended sandbox (`09_reeval.py`).** Two thirds
of the "failed" quasi-random code failed for the harness, not for the
code. BLADE's MA-BBOB runs generated code with separate globals and
locals (`exec(code, globals, locals)`), so a module-level import or
helper is invisible inside the class's methods (`NameError` for
`qmc`, `math`, …), and the sandbox had neither scipy nor scikit-learn.
With one namespace (`common.problems`, `shared_namespace`) and
`scipy==1.18.0`, `scikit-learn==1.9.0` added (24 elicited algorithms per
concept):

| elicited concept | failed before | failed after | AOCC of working |
|---|---|---|---|
| quasi-random init | 1.00 | **0.29** | 0.26 |
| surrogate | 0.96 | 0.88 | 0.24 |
| archive | 0.88 | 0.88 | 0.26 |
| (unprompted) | 0.25 | 0.25 | 0.28 |

Quasi-random initialisation is now a viable family. Surrogate code
still fails: 11 of 21 time out (a GP refit per evaluation exceeds 120 s),
the rest call APIs that do not exist (`np.erf`, `pyDOE`, `pySOT`).
Archive failures are real bugs (ragged arrays of archived solutions).
The steered `contrast-c` children do not change: their failures were
not import failures. All later campaigns (`llamea-dial`, step 5d) use
the extended sandbox.

**The dial in LLaMEA (`llamea-dial`, 5 arms × 8 seeds × 100
algorithms).** Every algorithm generated (all operators) is steered
along `b_exploration_pct_resid@L27`: `explore` at +0.3, `exploit` at
−0.3, `anneal` from +0.3 to −0.3 over the run, `randomdir` one of 8
random unit directions at ±0.3 (× residual norm). Means over seeds (95%
bootstrap CI):

| arm | best AOCC | failed | exploration % (working) | final pop. exploration % | families | CodeBLEU div. |
|---|---|---|---|---|---|---|
| control | 0.61 (0.48–0.72) | 0.16 | 22.5 (16.6–30.2) | 11.5 | 25.8 | 0.51 |
| explore | 0.65 (0.53–0.76) | 0.21 | **35.5 (29.9–42.1)** | 17.2 | 28.8 | 0.55 |
| exploit | **0.43 (0.35–0.51)** | 0.14 | 32.0 (21.0–46.3) | 19.6 | **16.4** | 0.54 |
| anneal | 0.67 (0.57–0.74) | 0.14 | 28.8 (25.7–31.9) | 10.9 | 27.8 | 0.55 |
| randomdir | 0.55 (0.48–0.63) | 0.20 | 34.1 (25.9–41.7) | 23.1 | 23.1 | 0.56 |

* The dial moves exploration in the loop, but mostly unspecifically.
  Paired by seed, `explore` raises the working algorithms' exploration
  by 13 points over control (8/8 seeds). But `exploit` (+9.5, 6/8) and
  the random directions (+11.6) raise it too. The direction-specific
  part, `explore − exploit`, is +3.5 points (CI −14 to +19; 5/8 seeds).
  It is +10 to +11 on the `init` and `new` operators (fresh algorithms,
  as in the screens) and about 0 on `refine` and `simplify`, where the
  parent's code fixes the search behaviour.
* Steering at all makes the code more exploratory. Dose 0.3 at every
  token perturbs the code towards simpler, more random sampling
  algorithms, and the random direction does the same.
* `exploit` hurts the search: best AOCC −0.18 against control (2/8
  seeds better), and 16 algorithm families instead of 26 (lower family
  entropy, 2.8 vs 3.4). `explore` (+0.05) and `anneal` (+0.06) are
  within noise of control. `explore` costs failures (0.21 vs 0.16).
* Selection washes the dial out: the final populations (the 10 best)
  are low-exploration (11–23%) in every arm, without a consistent
  order.
* Every steered arm has a higher CodeBLEU diversity of the generated
  code (0.54–0.56 vs 0.51), the random direction included.

So in a closed loop, the dial is a weak control of behaviour. Steering
"exploit" narrows the search and costs performance. "Explore" and
"anneal" are safe, but do not beat the control.

### More levers: contrastive directions (`results/corpus`, `contrast-c`, `contrast-b`, `dial`)

**Corpus and elicitation.** One sentence in the prompt elicits every rare
concept: archive, restart, simulated annealing, surrogate and tournament
selection in 100% of the elicited code, covariance and opposition 79%,
Lévy flight 63%, local search 33% (0–8% unprompted). Some elicited
families do not run in the sandbox: quasi-random initialisation fails
100% (22/24 use `scipy.stats.qmc`, not installed in the sandbox),
surrogate 96% (20/24 import `sklearn`, not installed), archive 88% (real
bugs, mostly ragged numpy arrays of archived solutions). The prompt is also a strong exploration dial: median
exploration 77% under "strong exploration", 37% unprompted, 18% under
"strong exploitation".

**Contrastive directions.** Concept directions separate held-out code
well (AUROC 0.75–0.99, best at L9/L15). Behaviour directions are weaker
(held-out Spearman 0.2–0.42) and the raw ones are mostly concept
directions: the exploration direction has cosine −0.86 with DE,
population and local search and +0.69 with step-size adaptation.
Residualised on the concept labels they keep Spearman 0.23–0.29 with
small concept cosines.

**Concept levers (`contrast-c`, 16 children per condition).** Four of
the ten rare-concept directions add their concept:

| direction | dose | concept rate | gain (95% CI) | failures (Δ) |
|---|---|---|---|---|
| covariance | 0.3 | 0.44 | +0.31 (0.13 to 0.56) | 0.31 (+0.19) |
| local search | 0.3 | 0.31 | +0.31 (0.13 to 0.56) | 0.31 (+0.19) |
| simulated annealing | 0.5 | 0.56 | +0.38 (0.13 to 0.63) | 0.44 (+0.31) |
| surrogate | 0.5 | 0.44 | +0.44 (0.19 to 0.69) | 0.69 (+0.56) |

Archive, Lévy flight, opposition, quasi-random initialisation, restart
(+0.19, CI includes 0) and tournament selection do not appear at any
dose; at 0.5 most of them break the code (failures 0.5–0.75, unsteered
0.13). A direction that *reads* a concept well (high AUROC) is often not
one that *writes* it. With swarm and gradient the lever set grows from 2
to about 5, at a higher failure cost than swarm; for the other rare
concepts the prompt is the better lever.

**Exploration dial (`contrast-b`, `dial`).** Slope per unit dose
(× residual norm), working children, prompt fixed effects:

| direction | screen | exploration % (95% CI) | intensification (95% CI) |
|---|---|---|---|
| raw @L15 | contrast-b | +7.8 (−24.5 to 35.3) | −0.05 (−0.34 to 0.32) |
| raw @L27 | contrast-b | −1.4 (−20.3 to 21.1) | −0.08 (−0.33 to 0.14) |
| resid @L15 | contrast-b | +18.7 (−2.7 to 43.6) | −0.22 (−0.43 to −0.02) |
| resid @L27 | contrast-b | +21.4 (−6.6 to 46.1) | −0.12 (−0.39 to 0.16) |
| random15 | contrast-b | −7.4 (−28.1 to 12.5) | −0.00 (−0.16 to 0.14) |
| random27 | contrast-b | −15.3 (−31.3 to −0.2) | +0.14 (−0.01 to 0.29) |
| resid @L15 | dial | **+13.4 (0.7 to 29.0)** | −0.12 (−0.29 to 0.03) |
| resid @L27 | dial | **+20.3 (6.2 to 35.7)** | **−0.16 (−0.30 to −0.02)** |
| random15 | dial | +7.8 (−4.7 to 20.3) | −0.07 (−0.20 to 0.05) |
| random27 | dial | −3.2 (−14.9 to 7.3) | +0.08 (−0.05 to 0.23) |

The concept-residualised exploration direction at L27 is a dial: on new
prompts its children's mean exploration rises monotonically with the
dose (27.9, 36.7, 40.8, 46.7% at −0.4, −0.2, +0.2, +0.4; unsteered
34.4%) and intensification falls, while the random directions stay
flat. The raw direction is not a dial: it mixes the behaviour with the
algorithm family. The dial is fine-grained but small: ±0.4 spans about
19 points of exploration, where the prompt spans 59 (18 → 77%).
±0.2 costs no extra failures (0.13 and 0.17, unsteered 0.17); ±0.4
does at the positive end (0.34). Unlike the prompt it needs no
instruction and changes behaviour *within* a family (the residualised
direction is nearly orthogonal to the concept directions).

`dist_to_best` is not reported for the screens: about 10% of the
children evaluate points far outside the box, which makes it unbounded.


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
