# Data for "Opening the Black-Box Operator: Mechanistic Interpretability for LLM-Driven Algorithm Design"

This record holds the raw and processed data behind every number, table
and figure of the paper (GECCO 2027 submission): the logs of all
generated algorithms (code, fitness, errors, search behaviour, LLM
prompts and responses), the labels and tables derived from them, and the
residual-stream activations and SAE codes the probes are trained on.

The code (experiment scripts, analysis scripts, the instrumented BLADE
fork, the paper sources and the small result files `results/*`) is in
the anonymised repository linked from the paper. This record contains
only what is not in that repository.

## Files

| File | Size (approx.) | Content |
|---|---|---|
| `blade-mi-results.zip` | see `SHA256SUMS` | this README, `FILES.txt`, all run logs, labels and processed tables (no activation arrays) |
| `blade-mi-activations-exp01.zip` | ~20 GB | residual-stream activations of every generated algorithm in the probing campaigns (Sec. 4.1, 4.2) |
| `blade-mi-activations-exp03-05.zip` | ~4.5 GB | SAE codes (Sec. 4.3), activations of the contrastive corpus and transfer screens (Sec. 6), per-query activations of the Bragg pre-screen runs (Sec. 4.2) |
| `SHA256SUMS` | | checksums of the three zips |

`FILES.txt` (inside `blade-mi-results.zip`) lists every file of all
three zips with its size in bytes.

All paths inside the zips are relative to the root of the code
repository. To restore the full state, clone the repository and unzip
the three archives in its root:

```bash
sha256sum -c SHA256SUMS
cd <repository root>
unzip blade-mi-results.zip -x README.md FILES.txt
unzip blade-mi-activations-exp01.zip
unzip blade-mi-activations-exp03-05.zip
```

This fills `experiments/mi/<experiment>/data/{raw,processed}/<campaign>/`,
the layout every script expects (`experiments/mi/common/paths.py`; set
`MI_DATA_ROOT` to keep the data elsewhere). Analyses that only need logs
and tables run with `blade-mi-results.zip` alone.

## Layout

```
experiments/mi/
  exp01_probing/data/              Sec. 4.1-4.2: probes, pre-screen, replication
    raw/<campaign>/shard-*/run-*/  one directory per LLaMEA run
    processed/<campaign>/          meta.csv, code.jsonl, activations.npz
  exp02_feedback_attribution/data/ Sec. 5: counterfactual prompts
    raw/<campaign>/shard-*.jsonl   one line per generated child
    processed/<campaign>/          prompts, attribution graphs, span attributions
  exp03_concept_features/data/     Sec. 4.3: SAE concept features
    processed/labels.csv           concept labels per algorithm
    processed/k100/                SAE codes (shard-*.npz), keyword-masked codes
  exp05_steering/data/             Sec. 6: steering
    raw/<campaign>/                screens (shard-*.jsonl) and LLaMEA runs (shard-*/run-*)
    processed/<campaign>/          directions, scales, prompts, corpus activations
  exp05_steering/results/llamea-{confirm,combo}/solutions.csv
                                   per-algorithm tables too large for git
```

### Campaigns and where they are used

Experiment 1 (`exp01_probing`), LLaMEA on MA-BBOB 5-D, activations
recorded for every query:

| Campaign | Paper | Content |
|---|---|---|
| `pilot` | Sec. 3 | 4 short runs with the original MA-BBOB scoring (84% of working algorithms scored 0 for overspending) |
| `main` | Sec. 4.1 | 12 runs × 100, Qwen3-8B: probe training data |
| `main-qwen3-14b`, `main-gemma-3-12b` | Sec. 4.1, App. A | the same design with Qwen3-14B and Gemma-3-12B-it |
| `prescreen`, `prescreen2` | Sec. 4.2 | pre-registered pre-screen test, 36 seeds × {shadow, screened} |
| `prescreen-qwen3-14b`, `prescreen-gemma-3-12b` | App. A | held-out pre-screen per replication model |

Experiment 2 (`exp02_feedback_attribution`): `cf`, `cf-qwen3-14b`,
`cf-gemma-3-12b` are the counterfactual feedback rewrites (Sec. 5,
4,180 children for Qwen3-8B), one JSON line per child with the variant,
seed, code, fitness and error. `processed/cf*` hold the prompts, the
steering vectors and the attribution graphs (`attribution_graphs*.jsonl`),
`processed/main*` the span attributions.

Experiment 3 (`exp03_concept_features`): SAE codes (Qwen-Scope, k = 100,
layers 9, 15, 21, 27) of every algorithm of the Qwen3-8B campaigns
`main`, `prescreen` and `prescreen2`, re-encoded by teacher forcing, and
the concept labels (Sec. 4.3).

Experiment 5 (`exp05_steering`):

| Campaign | Paper | Content |
|---|---|---|
| `screen`, `screen2`, `dose`, `scope` | Sec. 4.3, 6.1 | single-generation steering screens with SAE features (incl. the undefined-name feature) |
| `corpus` | Sec. 6 | 4,712 labelled algorithms for contrastive directions (`processed/corpus`: activations and directions) |
| `contrast-c`, `contrast-b`, `dial`, `reeval` | Sec. 6.1, 6.2 | contrastive concept levers, exploration dial, re-evaluation in the extended sandbox |
| `dial-qwen3-14b` (+ `processed/corpus-qwen3-14b`) | Sec. 6.2 | the dial rebuilt and screened on Qwen3-14B |
| `transfer-{bragg,packing,bbob-f2,bbob-f9,bbob-f20}` | Sec. 4.2, 6.3 | transfer screens on five new problems |
| `llamea` | Sec. 6.1 | concept levers in LLaMEA (4 arms × 8 seeds) |
| `llamea-dial`, `llamea-stage1*`, `llamea-ext200*`, `llamea-anneal200`, `llamea-randomdir200` | Sec. 6.4, App. (multiplicity) | the dial in LLaMEA on MA-BBOB, 16 seeds, continued to 200 evaluations |
| `llamea-dial-{bragg,packing}`, `llamea-stage1-{bragg,packing}` | Sec. 6.4 | Bragg and packing |
| `llamea-bbob-{f2,f9,f20}` | Sec. 6.4 | pre-registered landscape test, BBOB 20-D |
| `llamea-confirm` | Sec. 6.4 | pre-registered confirmation, 4 arms × 24 seeds × 200 |
| `llamea-combo` | Sec. 6.4 | pre-registered dial + temperature 1.1 arm, 24 seeds × 200 |
| `prescreen-bragg` | Sec. 4.2 | frozen failure probe in the loop on Bragg, 12 seeds × {shadow, screened} |

The registrations of the pre-registered tests are in the repository
(`experiments/mi/paper_gecco2027/PREREG_*.md`), and the full lab notes
per campaign in `experiments/mi/exp05_steering/README.md`.

## File formats

**LLaMEA run directory** (`shard-*/run-<method>-<model>-<problem>-<seed>/`),
written by BLADE's `ExperimentLogger`:

* `log.jsonl`: one line per evaluated algorithm, in order: `id`, `code`,
  `name`, `description`, `fitness` (AOCC; `-inf` if it failed), `error`
  (traceback), `feedback`, `parent_ids`, `operator`, `generation` and
  `metadata` (steering dose and direction, adaptive dose, per-instance
  AOCCs, over-budget instances, traced search behaviour such as
  exploration %, pre-screen score and flag with the true outcome of
  screened children).
* `conversationlog.jsonl`: every prompt and response of the LLM.
* `mi_activations/index.jsonl`: one line per LLM query: token counts,
  mean entropy and log-probability of the output, generation time,
  steering applied, prompt and response. In the probing campaigns,
  `file` names the per-query activation file; those files are not in
  this record because `processed/<campaign>/activations.npz` holds the
  same arrays stacked (see below), except for `prescreen-bragg`, whose
  per-query files are in `blade-mi-activations-exp03-05.zip`.
* `method.json`, `problem.json`, `llm.json`: the configuration;
  `stdout.log`: stdout of the run; `progress.json` (per shard).

**Screens** (`exp05_steering/data/raw/<screen>/shard-*.jsonl`,
`exp02_feedback_attribution/data/raw/cf*/shard-*.jsonl`): one JSON line
per generated child with the prompt id, condition (direction, dose, or
counterfactual variant), seed, code, evaluation outcome, error and,
where measured, search behaviour.

**Probing datasets** (`exp01_probing/data/processed/<campaign>/`):
`meta.csv` (one row per LLM query with labels: parent and child fitness,
`child_failed`, `over_budget`, `improved`, and the uncertainty
baselines), `code.jsonl` (generated code per row, same order), and
`activations.npz` with arrays `last`, `prompt_mean`, `gen_mean` of shape
`[n_rows, n_layers, d_model]` (float16; residual stream at the last
prompt token, mean over the prompt, mean over the generated tokens) and
`layers`.

**SAE codes** (`exp03_concept_features/data/processed/k100/shard-*.npz`):
per algorithm (`record_id`) and layer `L<n>`, the maximum, mean and
count over the code tokens of each of the 65,536 TopK SAE features as
CSR sparse matrices (`L<n>_{max,mean,count}_{indptr,indices,data}`), the
top-100 codes of the last prompt token (`L<n>_last_{idx,val}`) and the
24 largest activations per feature with their row and token
(`L<n>_top_*`); see
`experiments/mi/exp03_concept_features/scripts/02_encode.py`.

**Contrastive corpus** (`exp05_steering/data/processed/corpus*/`):
`rows.csv` (labels per algorithm), `acts.npz` (teacher-forced mean
residual states), `directions.npz` (the contrastive directions; the
exploration dial is `b_exploration_pct_resid@L27`).

## What is not in this record, and how to get it

* **Models** (Hugging Face; Gemma requires accepting its licence):
  `Qwen/Qwen3-8B` (revision `b968826d9c46dd6066d109eabc6255188de91218`),
  `Qwen/Qwen3-14B` (`40c069824f4251a91eefaf281ebe4c544efd3e18`),
  `google/gemma-3-12b-it` (`96b6f1eccf38110c56df3a15bffe176da04bfd80`).
* **Sparse autoencoders** (Qwen-Scope):
  `Qwen/SAE-Res-Qwen3-8B-Base-W64K-L0_100`
  (`f308c1df7c9746544f0af047fd00d6cf374627bd`).
* **Transcoders for attribution graphs** (Sec. 5):
  `mwhanna/qwen3-8b-transcoders` (`41287db9e56b6dc1f46d03097b8acf4501571ab9`),
  `mwhanna/gemma-scope-2-12b-it` (`e485d768031e1597967460f2acab495963885fef`),
  `mwhanna/qwen3-14b-transcoders` (`e6595bb044503fd8344b9fef13c29e17a3eab1a8`);
  the Qwen3-14B set must be converted with
  `experiments/mi/exp02_feedback_attribution/scripts/fix_transcoders.py`
  (see its docstring).
  Download any of them with, e.g.
  `huggingface-cli download Qwen/Qwen3-8B --revision <revision>`.
* **Per-query activation files of experiment 1**
  (`exp01_probing/data/raw/*/run-*/mi_activations/*.npz`, ~17 GB):
  identical to the rows of `processed/<campaign>/activations.npz`;
  `meta.csv` maps each row to its run and `record_id`.
* **Smoke campaigns** (`*smoke*`): pipeline checks, not used in the paper.
* **Python environment**: `uv sync` in the repository (see its
  `CLAUDE.md`/`README`); the evaluation sandbox installs its own
  dependencies.

## Anonymisation

Absolute paths of the machine that ran the experiments, which appear in
tracebacks and logs, are replaced by `<BLADE_ROOT>` (the repository
root) and `<HOME>`. Nothing else in the data was changed. The
`method.json` files cite the public LLaMEA repository as the source of
the method.

## Licence

Data: CC BY 4.0. Generated code in the logs was written by the listed
open-weight models; model outputs are subject to their licences.
