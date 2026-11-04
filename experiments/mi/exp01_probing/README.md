# Experiment 1: Probing for fitness awareness

**Question.** While LLaMEA mutates an algorithm, does the LLM internally
represent (a) how good the parent algorithm is, and (b) whether its own child
will crash, overspend the evaluation budget or improve on the parent?

**Why it matters.**
- Mechanistically, it shows whether the model tracks the search state that
  the evolutionary loop gives it, and at which layers.
- Practically, a probe that predicts failure before evaluation is a free
  surrogate: it lets LLaMEA skip evaluations, which are the expensive part.

## Hypotheses

- **H1.** Parent fitness is linearly decodable from the last prompt token in
  middle layers, clearly above the shuffled-label control.
- **H2.** Child failure (`child_failed`) and budget overspending
  (`over_budget`) are decodable from the generated-code mean (`gen_mean`),
  better than from the last prompt token (`last`). They are also better
  than the output-uncertainty baseline, and at least as good as a
  bag-of-words model on the generated code.
- **H3.** Improvement (`improved`) is weakly decodable before generation
  (`last`, AUROC well below 1, since it depends on sampling), and adds
  information beyond the `meta` baseline (parent fitness + operator).

## Design

- **Search.** LLaMEA (4 parents, 12 offspring, elitism) with three
  single-parent operators: `refine`, `new` and `simplify`. The operators are
  identified from the prompt text.
- **Problem.** MA-BBOB, 5-D, 10 training instances, AOCC fitness. Evaluation
  is sandboxed and runs without SMAC.
- **Model.** Qwen3-8B with thinking disabled, T = 0.8, top-p = 0.95.
- **Recorded per query (all 36 layers).** Residual stream at the `last`
  prompt token, `prompt_mean` and `gen_mean`. Also mean entropy and
  log-probability.
- **Labels** (`02_build_dataset.py`):
  - `parent_fitness`
  - `child_fitness`
  - `child_failed` (exception or timeout)
  - `over_budget` (ran, but used too many evaluations: AOCC 0)
  - `improved` (beats the parent; a working child of a failed parent counts
    as improved)
  - Rows with infrastructure errors are dropped.
- **Probes.** Standardised ridge regression with the regularisation strength
  chosen by built-in leave-one-out CV. Binary targets are fitted on ±1
  labels and scored by AUROC; regression targets by Spearman ρ and R².
  5-fold `GroupKFold` by LLaMEA run.
- **Controls and baselines** (same folds):
  - The same probe on shuffled labels.
  - `meta`: operator, generation, parent fitness/failure, prompt length.
  - `uncertainty`: entropy, log-probability, generated length.
  - `code_bow`: TF-IDF on the generated code.

Caveat: the best layer is chosen after seeing the scores, which biases it
upwards. Read the whole curve against the control. For the final numbers,
fix the layer on `pilot` and report on `main`.

## Run

```bash
source experiments/mi/env.sh

# 1. generate (one process per GPU; rerun to resume)
for g in 0 1 2 3; do
  uv run python experiments/mi/exp01_probing/scripts/01_generate.py \
      --config main --device cuda:$g --shard $g --n-shards 4 &
done; wait

# 2. join activations with BLADE logs
uv run python experiments/mi/exp01_probing/scripts/02_build_dataset.py --tag main

# 3. probes, controls, baselines -> results/main/
uv run python experiments/mi/exp01_probing/scripts/03_train_probes.py --tag main

# 4. analysis
uv run jupyter lab experiments/mi/exp01_probing/notebooks/01_probe_analysis.ipynb
```

Presets in `exp_config.py`:

| Tag | Model | Runs × budget | Purpose | Wall clock (4 GPUs) |
|---|---|---|---|---|
| `smoke` | Qwen2.5-0.5B | 2 × 6 | pipeline check | ~5 min |
| `pilot` | Qwen3-8B | 4 × 20 | label balance, layer selection | ~1–1.5 h |
| `main` | Qwen3-8B | 12 × 100 | ~1200 labelled queries | ~1 day |

Most of the wall clock is spent evaluating the algorithms (CPU), not
generating them (~10 s per algorithm at ~47 tok/s).

Storage: ~0.9 MB of activations per query (3 positions × 36 layers × 4096,
float16), so ~1.1 GB for `main`.

## Outputs

- `data/raw/<tag>/shard-*/run-*/`: BLADE logs + `mi_activations/`.
- `data/processed/<tag>/`: `meta.csv`, `activations.npz`, `code.jsonl`.
- `results/<tag>/`:
  - `probe_scores.csv`: target × position × layer, with control columns.
  - `baselines.csv`
  - `oof_*.csv`: out-of-fold predictions at the best layer.
  - Figures from the notebook: layer curves, pre-screening trade-off,
    PCA plot.

## Results

### Pilot (2026-11-04): Qwen3-8B, 4 runs × 20 algorithms, 80 labelled queries

Figures: `results/pilot/layer_curves.png`, `prescreening.png`,
`pca_parent_fitness.png`. With n = 61–80 and few positives, the
shuffled-label control itself reaches 0.7–0.8 AUROC at some layers. Only
effects that stay above the control across many layers count.

| Target | Probe (mean over layers) | Control (mean) | Best activation-free baseline | Read |
|---|---|---|---|---|
| `parent_fitness` | ρ 0.57 (prompt mean); last token ≈ 0 up to layer 15, then 0.6–0.7 from layer 20 | ρ ≈ 0 | code BoW ρ 0.35 | **signal**; it appears in the middle layers at the decision token (H1) |
| `child_failed` | AUROC 0.85 (gen mean); 0.61 (last) | 0.57 / 0.28 | uncertainty 0.85, code BoW 0.78 | **signal**, but not above the output-entropy baseline yet (H2 open) |
| `over_budget` | 0.60–0.64 | 0.37–0.70 | code BoW 0.87 | no clear signal (only 7 negatives) |
| `improved` | 0.29–0.42 | 0.61–0.78 | uncertainty 0.81 | no signal (9 positives) |
| `child_fitness` | ρ 0.30 (prompt mean) | ≈ 0 | meta ρ 0.23 | weak; fitness is mostly 0 (below) |

**Main issue found: the search is stuck on over-budget code.** 59 of the 70
working algorithms (84%) use more evaluations than allowed and score AOCC 0.
Only 7 of 80 children are valid and within budget, the median fitness is 0,
and the fitness labels are nearly degenerate. Cause: MA-BBOB's feedback says
"scored 0.000" without saying that the budget was exceeded (see the top-level
README), so the model cannot learn the fix. This has to be resolved before
`main`, otherwise a day of compute produces mostly zero-fitness labels.
