# Mechanistic Interpretability of LLM-driven algorithm design

Research folder for the `experiments/mi` line. These experiments open up the
*generator* inside LLaMEA / BLADE, the LLM that writes and mutates
optimisation heuristics, and study its internal activations while it designs
algorithms.

## Experiments

| # | Folder | Question | Status |
|---|--------|----------|--------|
| 0 | `iohblade.llm.HookedLLM` | Infrastructure: capture and steer activations inside any BLADE method | done, unit-tested |
| 1 | [`exp01_probing/`](exp01_probing/) | Does the model know how good its parent is, and whether its child will crash, overspend the budget or improve? | `main` analysed: H1, H2 supported; H3 not |
| 2 | [`exp02_feedback_attribution/`](exp02_feedback_attribution/) | Which parts of the prompt (feedback, code, error, operator) drive the edit? | planned |
| 3 | [`exp03_concept_features/`](exp03_concept_features/) | Which SAE features encode metaheuristic concepts, and do they match code structure and search behaviour? | planned |
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
  hides why the score is 0.
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
| 3 | `sae-lens` (load `Qwen/SAE-Res-Qwen3-8B-Base-*`), `delphi` (auto-interpretation), optionally `sparsify` to train SAEs on LLaMEA code |
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
