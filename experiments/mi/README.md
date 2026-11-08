# Mechanistic Interpretability of LLM-driven algorithm design

Research folder for the `experiments/mi` line. These experiments open up the
*generator* inside LLaMEA / BLADE, the LLM that writes and mutates
optimisation heuristics, and study its internal activations while it designs
algorithms.

## Experiments

| # | Folder | Question | Status |
|---|--------|----------|--------|
| 0 | `iohblade.llm.HookedLLM` | Infrastructure: capture and steer activations inside any BLADE method | done, unit-tested |
| 1 | [`exp01_probing/`](exp01_probing/) | Does the model know how good its parent is, and whether its child will crash, overspend the budget or improve? | done: H1, H2 supported; H3 rejected; failure pre-screen transfers to 36 new seeds and is non-inferior at 0.10 AOCC (p = 0.041) while skipping 21% of evaluations; [report](REPORT.md) |
| 2 | [`exp02_feedback_attribution/`](exp02_feedback_attribution/) | Which parts of the prompt (feedback, code, error, operator) drive the edit? | steps 1–6 done: error traces are used to fix the code (deleting them: +22 points of failures; attribution: error text gains influence at the edit); the written score is ignored and barely reaches the fitness representation, which is causal for edit size. Report: [`REPORT_exp02.md`](REPORT_exp02.md) |
| 1–2r | `exp01_probing/`, `exp02_feedback_attribution/` (tags `*-qwen3-14b`, `*-gemma-3-12b`) | Replication of experiments 1 and 2 on a bigger model of the same family (Qwen3-14B) and another family (Gemma-3-12B-it); comparison in `cross_model/`, report `REPORT_replication.md` | done (no attribution graphs) |
| 3 | [`exp03_concept_features/`](exp03_concept_features/) | Which SAE features encode metaheuristic concepts, and do they match code structure and search behaviour? | steps 1–4 + keyword masking done: base SAEs transfer; every concept has a detecting feature; with keyword lines masked, covariance, gradient, swarm, surrogate, local-search and adaptive-parameter features still detect their concept beyond co-occurrence, elitism/restart features are keyword detectors; concepts not visibly planned at the last prompt token; concept features add nothing to outcome prediction beyond code text, the whole SAE code adds +0.074 AUROC for failure |
| 5 | [`exp05_steering/`](exp05_steering/) | Can steering vectors act as a variation operator, and can CMA-ES search the steering space? | planned |

The numbering follows the original research plan. Directions 4 (mode
collapse), 6 (recall vs. invention), 7 (model diffing) and 8 (error circuits)
are listed at the end; 8 is partly covered by the `child_failed` and
`over_budget` probes of experiment 1.

## Layout

```
experiments/mi/
├── README.md            # this file: plan, index, conventions, log
├── env.sh               # cache locations + thread limits (source first)
├── common/              # code shared by all experiments
│   ├── models.py        #   model presets (HF id, chat kwargs, SAEs, transcoders)
│   ├── paths.py         #   data/ and results/ locations (MI_DATA_ROOT)
│   ├── runs.py          #   read BLADE run dirs + HookedLLM records
│   ├── problems.py      #   problem variants (BudgetEnforcedMABBOB)
│   └── plotting.py      #   shared matplotlib style and colours
└── expNN_<name>/
    ├── README.md        # question, hypotheses, design, how to run, results
    ├── exp_config.py    # every setting of a campaign, named presets
    ├── scripts/         # numbered pipeline steps: 01_..., 02_..., 03_...
    ├── notebooks/       # analysis notebooks (read results/, never raw data)
    ├── data/            # git-ignored: raw BLADE logs, activations, datasets
    │   ├── raw/<tag>/
    │   └── processed/<tag>/
    └── results/<tag>/   # committed: small CSVs and figures
```

Conventions:

- A **tag** names a data-collection campaign (`smoke`, `pilot`, `main`, ...)
  and is the preset name in `exp_config.py`. `data/raw/<tag>/config.json`
  records the exact settings.
- Scripts are numbered and idempotent. Generation resumes where it stopped,
  because BLADE's `ExperimentLogger` skips finished runs.
- `data/` is never committed. `results/` holds only small, regenerable
  artefacts (CSV, PNG) that the notebooks and the paper use.
- All experiments use the same model (`common/models.py`, default
  **Qwen3-8B**), so probes, attribution graphs, SAE features and steering
  vectors share one representation space.

## Setup

```bash
uv sync --group dev --group mi
export MI_CACHE=/local/$USER/cache   # anywhere with ~50 GB free
source experiments/mi/env.sh
```

`env.sh` points the Hugging Face, uv, pip and virtualenv caches at
`$MI_CACHE`, and limits BLAS to one thread per evaluation subprocess.
Without that limit, parallel runs exhaust the thread limit and the failures
are wrongly logged as crashes of the generated code.

## Infrastructure: `HookedLLM`

`iohblade.llm.HookedLLM` is a drop-in BLADE LLM provider (works with LLaMEA,
RandomSearch, ...) that runs any HuggingFace causal LM with PyTorch forward
hooks:

```python
from iohblade.llm import HookedLLM

llm = HookedLLM(
    "Qwen/Qwen3-8B",
    device_map="cuda:0",
    chat_template_kwargs={"enable_thinking": False},
)
method = LLaMEA(llm, budget=100)
```

- For every query it records, per decoder layer, the residual stream at the
  **last prompt token**, the **prompt mean** and the **generated-token
  mean**. It also records mean token entropy and log-probability as
  uncertainty baselines.
- Records go to `<run_dir>/mi_activations/` (`index.jsonl` + one `.npz` per
  query, about 0.9 MB for Qwen3-8B). Each record is linked to the
  `Solution` via `solution.id` and `solution.metadata["mi_record_id"]`. The
  prompt and response text are stored as well.
- `llm.replay(prompt, response)` recomputes full per-token activations
  offline. It matches the captured states to cosine > 0.999. Use it for
  experiments 2 and 3 instead of storing everything during runs.
- `llm.set_steering({layer: vector}, alpha, positions)` adds vectors to the
  residual stream (experiment 5). The steering configuration is logged with
  every record.
- Deep copies (one per BLADE run) share the weights and a GPU lock. Run one
  process per GPU and shard seeds across processes.

Plain `transformers` + hooks was chosen over TransformerLens/nnsight for the
*generation* path because it is fast (~47 tok/s for Qwen3-8B on an RTX PRO
6000) and works for any architecture. The analysis libraries (SAELens,
circuit-tracer, nnsight) can load the same weights offline.

## Known caveats

- **MA-BBOB over-budget scoring.** In BLADE's `MA_BBOB`, an algorithm that
  uses more evaluations than allowed raises `OverBudgetException` and scores
  AOCC 0. Experiment 1 therefore uses `common.problems.BudgetEnforcedMABBOB`. The feedback
  then says "scored 0.000" without the cause. Experiment 1 labels this case
  (`over_budget`). For experiment 2 it is a confound, because the feedback
  hides why the score is 0. Experiment 2 tested the missing message as a
  counterfactual: it cuts overspending children of overspending parents
  from 99.6% to 93.5%.
- **SAEs and transcoders for Qwen3-8B are trained on the base model.** Check
  reconstruction quality on our (post-trained) activations before
  interpreting features.

## Research directions (full list)

1. **Fitness awareness (probing).** Can activations predict parent fitness,
   child failure and improvement? If so, a probe gives a free surrogate.
   → `exp01_probing`
2. **Is the feedback used? (attribution).** Attribute the edit to feedback,
   code, error and operator tokens. → `exp02_feedback_attribution`
3. **Algorithmic concept features.** Map SAE features to AST features and
   BLADE behaviour metrics. → `exp03_concept_features`
4. **Mode collapse.** Find the directions behind the default DE/CMA-ES-like
   designs and ablate them.
5. **Steering as a variation operator.** Use steering vectors as a mutation
   operator; run CMA-ES in steering space on top of LLaMEA.
   → `exp05_steering`
6. **Recall or invention.** Use logit lens, entropy and feature signatures to
   tell memorised from recombined code.
7. **Model diffing.** Use crosscoders between a base model and one
   fine-tuned on LLaMEA elites.
8. **Error and timeout circuits.** Reject code before execution.

## Libraries

Installed now (`mi` group): `torch`, `transformers`, `accelerate`,
`scikit-learn`, `matplotlib`. Add these per experiment when it starts:

| Experiment | Libraries |
|---|---|
| 2 | `circuit-tracer` (git, safety-research/circuit-tracer; Anthropic's attribution graphs), `inseq` or `captum` for gradient attribution baselines |
| 3 | `common/sae.py` loads `Qwen/SAE-Res-Qwen3-8B-Base-*` directly (no `sae-lens` needed), `delphi` (auto-interpretation), optionally `sparsify` to train SAEs on LLaMEA code |
| 5 | none beyond `HookedLLM`; `steering-vectors` / `repeng` for comparison, `cma` for the outer CMA-ES loop |
| general | `nnsight` (+ NDIF for 70B models), `transformer-lens`, `pyvene`, Neuronpedia |

## Log

- 2026-11-04: `HookedLLM` + unit tests; experiment 1 pipeline validated on a
  smoke run (Qwen2.5-0.5B) and a synthetic planted-signal dataset. Pilot
  with Qwen3-8B (4 seeds × 20 algorithms) started. Found and fixed two
  label-contaminating infrastructure issues (BLAS threads; SMAC build in the
  sandbox). Pilot finished: parent fitness is decodable from the last
  prompt token from layer ~20 on, and failure from the generated-code mean.
  But 84% of working algorithms overspend the budget (AOCC 0, not
  reported in the feedback), so the fitness labels are degenerate.
  Decision: enforce the budget (`BudgetEnforcedMABBOB`, truncate instead
  of zero). `main` (12 × 100) started on 12 shards. Also capped idle thread
  pools in `env.sh`: this host limits a user to 2048 tasks.
- 2026-11-04 (later): `main` finished (1200 queries, 1.5 h). Parent fitness
  (ρ ≈ 0.7–0.8), child failure (AUROC 0.86 from the generated-code state,
  vs 0.71 for the best text/entropy baseline) and over-budget code (AUROC
  0.93 vs 0.75) are decodable. Improvement and child fitness are not
  decodable beyond the metadata baseline. A failure pre-screen skipping 20%
  of evaluations catches 60% of failures and loses 3% of top-decile
  algorithms.
- 2026-11-04 (evening): experiment 1 follow-up.
  - Nested leave-one-run-out with stacked baselines: the generated-code
    state adds +0.08 AUROC [0.05, 0.11] for failure beyond metadata + text.
    Prompt positions add nothing beyond metadata for any target. H3 is
    rejected.
  - The over-budget signal before writing is inherited from the parent
    (chance within strata). The signal while writing is not (0.78–0.88 vs
    0.60–0.68 for metadata + text).
  - Frozen failure probe on 12 new seeds × 2 conditions (2400 algorithms):
    AUROC 0.86. Flags 23.5% of evaluations and catches 69% of failures.
    Acting on it skips 21% of evaluations with no measurable loss of best
    fitness (p = 0.73).
  - Probe directions exported to `exp01_probing/results/main/probes/` for
    experiments 2 and 5.
- 2026-11-04 (night): experiment 1 concluded.
  - Power analysis on 12 seeds (SD of paired difference 0.23): only ≥ 0.20
    AOCC detectable. Pre-registered a non-inferiority test (margin 0.10,
    36 seeds, 82% power) and ran 24 more seeds.
  - Result: mean difference −0.037, 90% CI [−0.096, +0.023], p = 0.041:
    non-inferior, narrowly. Probe AUROC on the new seeds 0.84.
  - Adaptive thresholds (per operator, calibrated risk, running quantile)
    do not beat the fixed one.
  - Report: `REPORT.md`.
- 2026-11-05: experiment 2, steps 3–6.
  - Generation counterfactuals: 192 queries × variants × 4 replicates with
    common random numbers, 4180 evaluated children.
  - Deleting the error trace of a crashed parent: child failures 43% → 66%,
    mostly the same exception type; a generic "crashed" message recovers
    27% of that.
  - Score set to 0.010 / 0.900 in feedback *and* population list (step 1
    was confounded by the population list): fitness readout moves ±0.04
    for ±0.44 written; no effect on the child.
  - Steering ±2·v (top − bottom tercile mean difference, layer 15):
    towards "good parent" → larger rewrite (similarity −0.032), towards
    "bad" → smaller (+0.025), random direction: none, but +5.5 points of
    failures.
  - Attribution graphs (circuit-tracer, Qwen3-8B transcoders, 69 graphs):
    at the edit token, the error message's influence share is 61% higher
    than at a position-matched copied token (14/16 pairs); the parent
    code's is 38% lower. Replacement score ~0.1.
  - Report: `REPORT_exp02.md`.
- 2026-11-05 (afternoon): replication of experiments 1 and 2 started on
  Qwen3-14B and Gemma-3-12B-it (both have circuit-tracer transcoders:
  `mwhanna/qwen3-14b-transcoders`, Gemma Scope 2 via
  `mwhanna/gemma-scope-2-12b-it`).
  - Gemma 3's residual stream exceeds the float16 range (max ~1.4e5), so
    `HookedLLM` gained `store_dtype`; Gemma stores float32.
  - Scripts made model-agnostic: probes are exported at each campaign's
    best layer; pre-screen presets name the probe by target; experiment 2
    reads the source campaign, model, probes and transcoders from
    `cf_meta.json`. Readouts renamed `pf`/`cf`/`ob` (the Qwen3-8B summary
    reproduces exactly).
- 2026-11-06: replication status.
  - Experiment 1 done for both models (probes, incremental value,
    held-out pre-screen); experiment 2 span deletion done for both.
  - Qwen3-14B counterfactuals done (`results/cf-qwen3-14b`): error-trace
    feedback is used as in Qwen3-8B (removing it: +0.24 failures, same
    error +0.29); score edits and fitness steering change the rewrite size
    but not failures or fitness. The budget message cuts over-budget
    children far more than in 8B (−0.35 vs −0.06).
  - Attribution graphs do not fit for the replication models. Their
    transcoders are dense: ~90 active features per token per layer
    (14B: 1.08M on 300 tokens; Gemma 16k set: 1.29M). On 300 tokens a
    14B graph peaks at 75 GB and takes 18 min; the real contexts
    (~1200 tokens) run out of memory on a 96 GB GPU while collecting the
    active features, for any batch size. Not run; an option is to keep
    only the top-k features per position (the rest goes to error nodes).
  - The machine is shared: from now on at most 2 GPUs (2 and 3).
- 2026-11-07: replication done. Gemma counterfactuals finished
  (`exp02_feedback_attribution/results/cf-gemma-3-12b`).
  `cross_model/compare.py` writes the three-model comparison
  (`cross_model/results/`); report: `REPORT_replication.md`.
  - Replicates: the child-failure probe adds to metadata + text in all
    three models; the pre-screen transfers to new seeds; parent code ≫
    format ≫ feedback; the written score is ignored; the error trace is
    read.
  - Model-specific: the parent-fitness representation beyond baselines
    and its causal effect on rewrite size (8B only); error traces lead
    to working fixes only in the Qwen models (Gemma keeps the failing
    line more without the error, but fails as often); the over-budget
    message works far better in 14B (−35 points).
- 2026-11-07 (evening): experiment 3 started on Qwen3-8B (top-k attribution
  graphs for the replication models skipped: they would mostly re-test
  what the counterfactuals already show, with a weaker method). Steps
  1–4 done, see `exp03_concept_features/README.md`.
- 2026-11-07 (night): experiment 3 keyword masking. Features for
  covariance, gradient, swarm, surrogate, local search and adaptive
  parameters survive masking of the keyword lines; elitism and restart
  features are keyword detectors; crossover/DE features read the DE
  family as a whole.
- 2026-11-08: experiment 3 outcomes. Concept features add nothing to
  failure/fitness prediction beyond the code text; the whole SAE code
  adds +0.074 AUROC for failure (as the residual probe in experiment 1),
  spread over many weak features (GP surrogate, finite-difference
  gradient, code tidiness).
