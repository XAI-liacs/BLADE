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
- **Problem.** MA-BBOB, 5-D, 10 training instances, AOCC fitness, with the
  **budget enforced** (`common.problems.BudgetEnforcedMABBOB`). A run that
  overspends is stopped at the budget and scored on what it reached, not
  zeroed. The feedback text is unchanged. This was decided after the pilot
  (below). Evaluation is sandboxed and runs without SMAC.
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

# 1. generate: 12 detached shards on 4 GPUs (rerun to resume)
experiments/mi/exp01_probing/scripts/launch.sh main 12 4

# 2. join activations with BLADE logs
uv run python experiments/mi/exp01_probing/scripts/02_build_dataset.py --tag main

# 3. probes, controls, baselines -> results/main/
uv run python experiments/mi/exp01_probing/scripts/03_train_probes.py --tag main

# 4. analysis
uv run jupyter lab experiments/mi/exp01_probing/notebooks/01_probe_analysis.ipynb
```

Follow-up (after `main`):

```bash
S=experiments/mi/exp01_probing/scripts

# 4. refit the chosen probes on all of main -> results/main/probes/
uv run python $S/04_export_probes.py --tag main

# 5. held-out pre-screen: seeds 100-111, shadow + screened, probe frozen
SCRIPT=05_prescreen_generate.py $S/launch.sh prescreen 12 4
uv run python $S/02_build_dataset.py --tag prescreen

# 6. added value of the activations + over-budget confound (nested CV);
#    BLAS threads per fold: FOLD_THREADS (default 8)
uv run python $S/06_incremental_value.py --tag main

# 7. pre-screen generalisation and its effect on the search
uv run python $S/07_prescreen_analysis.py --tag prescreen

# figures
uv run jupyter lab experiments/mi/exp01_probing/notebooks/02_followup_analysis.ipynb
```

Presets in `exp_config.py`:

| Tag | Model | Runs × budget | Purpose | Wall clock (4 GPUs) |
|---|---|---|---|---|
| `smoke` | Qwen2.5-0.5B | 2 × 6 | pipeline check | ~5 min |
| `pilot` | Qwen3-8B | 4 × 20 | label balance, layer selection | ~1–1.5 h |
| `main` | Qwen3-8B | 12 × 100 | ~1200 labelled queries | 1.5 h (12 shards) |
| `prescreen` | Qwen3-8B | 12 seeds × 2 conditions × 100 | held-out test of the failure pre-screen | ~3 h (12 shards) |

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
  - `probes/`: the refit probes (`04_export_probes.py`): weights,
    unit residual-stream `direction`, pre-screen `threshold`.
  - `followup/`: nested-CV added-value scores and over-budget strata
    (`06_incremental_value.py`).
- `results/prescreen/`: `screen_metrics.csv`, `runs.csv`,
  `best_so_far.csv` and figures (`07_prescreen_analysis.py`, notebook 02).

## Results

### Main (2026-11-04): Qwen3-8B, 12 runs × 100 algorithms, 1200 labelled queries

Budget enforced. Runtime 1.5 h on 12 shards, no infrastructure errors.
17% of children fail; 87% of working children overspend on at least one
instance (now truncated, not zeroed). Median fitness of working
algorithms 0.31 (pilot: 0). Figures: `results/main/`.

Shuffled-label controls stay at AUROC ≤ 0.53 and |ρ| ≤ 0.07 on every layer,
so with n ≈ 1000 the curves can be read directly.

| Target | Probe: mean over layers (best layer) | Best activation-free baseline | Verdict |
|---|---|---|---|
| `parent_fitness` | ρ 0.70 prompt mean (0.80, L15); last token 0.69 (0.77) | code BoW 0.66, meta 0.58 | **H1 supported.** Decodable at the decision token, above baselines from layer ~10 on |
| `child_failed` | AUROC 0.82 gen mean (0.86, L21); last token 0.67 | uncertainty 0.71, code BoW 0.71 | **H2 supported.** The state while writing predicts the crash clearly better than the code text or the output entropy. Before writing: weak (0.67) |
| `over_budget` | AUROC 0.88 gen mean (0.93, L9); prompt mean 0.76 | meta 0.75, code BoW 0.74 | **Strongest result.** The model's state encodes "this code overspends" far better than the code tokens. The prompt-mean signal probably partly reflects inherited parent code (to check) |
| `improved` | AUROC 0.58 (0.62, L19) | meta 0.65 | **H3 not supported.** Activations do not beat parent fitness + operator |
| `child_fitness` | ρ 0.60 gen mean (0.69, L15) | meta 0.71 | Below the metadata baseline (which contains parent fitness); added value still to be tested with a combined model |

**Pre-screening** (child-failure probe, generated-code mean, layer 21,
out-of-fold):

| Evaluations skipped | Failures caught | Working algorithms lost | Top-decile algorithms lost |
|---|---|---|---|
| 10% | 38% | 4% | 1% |
| 20% | 60% | 11% | 3% |
| 30% | 75% | 21% | 14% |

At 20% of the budget saved, 3% of the best algorithms are lost. The layer was
picked on the same data, but with flat curves and tight controls the bias
is small; confirm on held-out runs.

### Follow-up on main: added value and the over-budget confound

`06_incremental_value.py`. Leave-one-run-out, with the probe layer chosen
inside each training split (no selection bias). The probe and the code
TF-IDF enter a stacked model through inner out-of-fold predictions.
M = metadata, now including the parent's over-budget status; T = text (code
TF-IDF, entropy, log-prob, length); P = probe. 95% bootstrap CIs over runs.
Figures: `results/main/incremental_value.png`, `over_budget_strata.png`.

| Target | M | M+T | P (gen mean) | M+T+P | Gain from the probe over M+T |
|---|---|---|---|---|---|
| `child_failed` | 0.64 | 0.78 | **0.85** | 0.85 | **+0.08 [+0.05, +0.11]** |
| `over_budget` | 0.91 | 0.93 | 0.94 | 0.96 | +0.03 [+0.01, +0.08] |
| `improved` | 0.66 | 0.68 | 0.61 | 0.68 | 0.00 [−0.02, +0.01] |
| `child_fitness` (ρ) | 0.77 | 0.80 | 0.68 | 0.81 | +0.01 [−0.01, +0.03] |

- **Layer-selection bias is negligible.** With nested layer choice, the
  failure probe scores 0.85 (best-layer estimate: 0.86). The inner choice
  lands on layers 21–23 in 11 of 12 folds.
- **Prompt positions add nothing** beyond metadata for any target: all
  gains are ≤ 0.01, with CIs covering 0. Everything the model "knows" about
  the outcome before writing is already in the parent's fitness and status.
- **H3 confirmed negative.** Neither `improved` nor `child_fitness` gains
  from the activations once metadata and text are known. Most of the
  apparent `child_fitness` signal is parent fitness.

**Over-budget confound.** Of the 780 working children whose parent overspent,
773 (99%) overspend too. With the parent's status in M, metadata alone
reaches 0.91, which explains most of the earlier gap to the probe. Within
strata where that status cannot help:

| Stratum | n (overspend) | last token | prompt mean | gen mean | M+T |
|---|---|---|---|---|---|
| parent within budget | 154 (47) | 0.47 [0.22, 0.60] | 0.47 [0.30, 0.58] | **0.78 [0.72, 0.98]** | 0.60 [0.49, 0.75] |
| no working parent | 56 (43) | 0.41 [0.32, 0.49] | 0.48 [0.39, 0.60] | **0.88 [0.77, 0.96]** | 0.68 [0.58, 0.81] |

The prompt-level over-budget signal was entirely inherited from the parent:
it is at chance within strata. The state while writing still predicts new
overspending well above the text baseline, so this part is the model's own.

**Exported probes** (`04_export_probes.py` → `results/main/probes/`). Each
probe is refit on all of main:
- `child_failed_gen_mean_L21`, alpha 1000, pre-screen threshold −0.293 =
  the 80th percentile of the out-of-fold scores;
- `over_budget_gen_mean_L9`;
- `parent_fitness_prompt_mean_L15`.

Each file has a unit `direction` in residual-stream space for experiments 2
and 5.

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
README), so the model cannot learn the fix. **Decision:** `main` uses
`BudgetEnforcedMABBOB`. Re-scored with it, the pilot's over-budget evolution
strategy goes from AOCC 0.000 to 0.411.
