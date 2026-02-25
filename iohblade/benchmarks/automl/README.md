# AutoML Benchmark (OpenML)

This benchmark evaluates **LLM-generated scikit-learn pipelines** on **OpenML tasks**.
Given an OpenML task ID, BLADE downloads the dataset/splits from OpenML and asks the LLM
to produce a single Python class implementing a pipeline.

## What is being generated?

The LLM must output **one Python class** with the following interface:

- `__init__(self, X, y, **hyperparameters)`  
  Fit exactly once (no CV/HPO loops inside the class).
- `__call__(self, X)`  
  Return predictions for `X`.

The benchmark enforces constraints such as:
- No internal CV/HPO loops (no `GridSearchCV`, no KFold loops, no Optuna, etc.).
- Use only `scikit-learn`, `numpy`, `scipy`, `pandas`.
- Use one estimator, or a small sklearn ensemble wrapper (e.g. `VotingClassifier` / `StackingClassifier`).

## Evaluation

- Uses the **official OpenML split definition** for the task (repeat/fold/sample).
- Metrics are taken from the OpenML task’s evaluation measure (e.g. accuracy, AUC, RMSE).
- Optionally supports **in-the-loop HPO** using SMAC if the LLM provides a `# Space:` block
  describing a configuration space.

## CLI options (run-automl.py)

Common flags:

- `--tasks "id1,id2,..."`: Run explicit OpenML task IDs (skips suite loading).
- `--suite <suite-name>`: OpenML suite to load task IDs from (default: `amlb-classification-all`).
- `--limit N`: Only run the first `N` tasks (after filtering).
- `--skip "id1,id2,..."`: Skip specific task IDs.

Parallelization / sharding:

- `--concurrency K`: Number of tasks to run in parallel **processes** (each task runs in its own process).
- `--num-shards S` and `--shard i`: Split the task list into `S` shards and run shard `i` (0-based).
  Useful for launching multiple tmux jobs without overlapping tasks.
- `--list-tasks`: Print which task IDs this shard would run and exit (sanity check).

Reproducible output paths:

- `--stamp <stamp>`: Use a shared run identifier so multiple shards write under the same results folder.
  If omitted, a timestamp is generated automatically.

Model / search settings:

- `--model <model>`: LLM model name (BLADE supports Ollama/OpenAI/DeepSeek/Gemini/Claude).
- `--budget B`: LLaMEA evaluation budget (number of evaluated solutions).
- `--crossover-rate p`: Probability of crossover vs mutation per offspring (`0.0`–`1.0`).

  
## Running experiments

The recommended entrypoint is:

- `examples/run-automl.py`

Run a suite from OpenML:
```bash
python examples/run-automl.py --suite amlb-classification-all --limit 10
```

Run a specific task ID:

```bash
python examples/run-automl.py \
  --tasks "359965" \
  --budget 15 \
  --model qwen2.5-coder:32b
```
