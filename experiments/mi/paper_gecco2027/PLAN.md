# Plan: experiments to complete the GECCO 2027 story

Deadline: abstract 18 Jan 2027, paper 25 Jan 2027 (L4EC track). Compute:
GPUs 2 and 3 only. Throughput so far: about 500 generated and evaluated
algorithms per hour with 8 LLaMEA processes on the two GPUs (MA-BBOB
5-D). 20-D evaluations are slower, so plan with about 300 per hour.

## Gaps the current results leave

1. Most results are on one problem (MA-BBOB 5-D). The transfer to
   Bragg works, but on few working algorithms.
2. In the closed LLaMEA loop on MA-BBOB, the dial-specific effect is
   not significant with 8 seeds, and selection washes it out.
3. There is no "right setting of the dial depends on the problem"
   result yet. That is the most interesting claim for an EC audience.
4. No comparison with the obvious alternative knob: telling the model
   to explore or exploit in the prompt.
5. The pre-screen (C1) is shown on MA-BBOB only.

## E1. BBOB f2, f9, f20 in 20-D (proposed by the user)

One LLaMEA campaign per function, each function a single target:

| | f2 | f9 | f20 |
|---|---|---|---|
| function | separable ellipsoid | rotated Rosenbrock | Schwefel x·sin(x) |
| landscape | unimodal, ill-conditioned | unimodal, curved valley | multimodal, weak global structure |
| prediction | exploit ≥ control > explore | exploit ≥ control > explore | explore > control > exploit |

* **Instances.** The proposed sets overlap at 15. Suggestion: train on
  instances 10–14 (5 instances, in the loop) and validate the final
  best algorithm of each run on instances 15–30 (16 held-out
  instances). Alternatively use 16–30 for validation.
* **Problem.** BLADE's `BBOB_SBOX` with `problem_type=ioh.ProblemClass.BBOB`,
  `specific_fid`, `dims=[20]`, via a subclass with our fixes: shared
  namespace, extended sandbox, budget truncation instead of AOCC 0, and
  traced evaluations for exploration % and intensification. Budget
  2000 × D = 40,000 evaluations; AOCC on log regret in [1e-8, 1e2].
  Evaluation timeout raised to 600 s.
* **Prompt.** No function-specific description (`specific_fid` adds
  one). The model sees a black box, so any difference between functions
  comes from the feedback and selection, not from the prompt. Optional
  later: a "described" variant.
* **Arms** (one dose, 0.3, from the MA-BBOB campaign): control,
  dial-explore, dial-exploit, random direction, and the two prompt
  baselines of E2. Six arms × 10 seeds × 100 algorithms = 6,000
  algorithms per function, 18,000 in all: about 60 GPU-hours. Start with
  f20 and f2 (strongest contrast); f9 if time allows.
* **Analysis.** Arm × function interaction on the best AOCC (training)
  and on the validation AOCC of the final best algorithm (instances
  15–30). Also the generalisation gap, exploration % of generated and
  final algorithms, families and diversity, paired by seed. Pre-register
  the predicted ordering above before running.
* **Single-generation screen first** (as `10_transfer.py`, about 2 h per
  function): checks that the dial moves measured behaviour on each
  function before spending GPU-days on the loop.

## E2. Prompt baseline (reviewers will ask)

Add two arms everywhere the dial runs: the same instruction sentence that
elicited exploration and exploitation in the corpus ("use strong
exploration" / "strong exploitation"). On MA-BBOB the prompt moves
exploration much more (18 → 77%) than the dial at ±0.4 (19 points). The
claims to test are where steering differs: it needs no prompt change,
it changes behaviour *within* an algorithm family, and it costs fewer
failures at small doses. If the prompt wins everywhere, the paper should
say so.

## E3. An adaptive dial (EC-native answer to "selection washes it out")

Control the dose per run instead of fixing it. Two options:

* a success rule: raise exploration after k generations without
  improvement, lower it after an improvement (as the 1/5th success rule
  for step sizes);
* a bandit over a few doses, rewarded by improvement of the child.

This is one extra arm in E1, and it is the most EC-relevant
contribution of the steering part. It also returns to the original exp05
question of searching the steering space (with a bandit instead of
CMA-ES, because the space is one-dimensional here).

## E4. Pre-screen on the new problems (cheap)

Record the generated-code activation at the failure-probe layer in every
E1 run (capture on, one layer, mean over the code). Then report the
frozen MA-BBOB failure probe's AUROC on BBOB 20-D, Bragg and packing,
without acting on it. This shows whether C1 transfers across problems at
no extra GPU cost. The Bragg and packing screens already have the
activations: compute it there now.

## E5. More statistical power on what we have

* MA-BBOB `llamea-dial`: the SD of the paired explore − exploit
  difference is large. Compute the seeds needed for a 5-point effect.
  If it is under ~20, add seeds (2 extra arms × 12 seeds ≈ 5 GPU-hours).
* Bragg: the screen has about 90 working algorithms. Double `k_init` to
  64 for the dial at ±0.4 and none (about 2 h).

## E6. Lower priority (journal version)

* Dial on a second model (Qwen3-14B). This needs a new corpus and
  directions for that model: about 2 GPU-days.
* An exploration probe trained on several problems, to test whether a
  problem-independent readout of search behaviour exists. The MA-BBOB
  probe does not transfer to Bragg.
* Comparison of the final algorithms with human baselines (CMA-ES via
  modcma, DE) on the validation instances.

## Order and timeline

| weeks | runs | GPU-h |
|---|---|---|
| now–22 Nov | Bragg/packing LLaMEA (running); E4 on existing screens; E1 screens f2, f20 | ~10 |
| 23 Nov–13 Dec | E1 f20 + f2 with E2 and E3 arms | ~45 |
| 14 Dec–3 Jan | E1 f9; E5 | ~25 |
| 4–17 Jan | analysis, figures, writing; abstract 18 Jan | – |
| 18–25 Jan | final paper, supplement, anonymised code | – |

Total: about 80 GPU-hours on 2 GPUs, which fits in about 5 weeks of
wall time with margin.
