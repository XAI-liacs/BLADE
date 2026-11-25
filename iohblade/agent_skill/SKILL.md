---
name: blade
description: Set up, check and run BLADE (iohblade) experiments that benchmark LLM-driven algorithm design methods such as LLaMEA on optimisation problems. Use when the user wants to start with BLADE, write or fix an experiment YAML file, add their own problem (benchmark), search method or LLM, run or resume an experiment, or understand BLADE errors and results.
---

# BLADE experiments

BLADE (`iohblade` on PyPI) runs LLM-driven *methods* (LLaMEA, EoH, ReEvo,
FunSearch, ShinkaEvolve, random search, ...) that write optimisation
algorithms for *problems* (benchmarks), using one or more *LLMs*. An
experiment is a YAML file; the `iohblade` command validates, smoke-tests
and runs it. Prefer this route over writing Python experiment scripts.

## Commands

Prefix with `uv run` when the project uses uv (a `uv.lock` or
`pyproject.toml` with uv); otherwise call `iohblade` directly.

| Command | What it does |
|---|---|
| `iohblade init DIR [-t builtin\|custom-problem\|custom-method]` | Starter `experiment.yml` (+ stub code, JSON schema) |
| `iohblade list [problems\|suites\|methods\|llms\|loggers]` | Built-in keys with descriptions |
| `iohblade validate FILE` | Checks everything; runs nothing; no LLM calls |
| `iohblade check FILE [--llm]` | Smoke test: builds evaluation envs, evaluates a reference solution, runs each method with budget 2 and an offline LLM. `--llm` also sends one prompt per LLM |
| `iohblade run FILE [--dry-run] [--set key=value]` | Runs; rerunning resumes and skips finished runs |
| `iohblade-webapp` | Browse the experiments in subfolders of `./results` (`output_dir: results/<name>`) |
| `iohblade cache list\|clear` | Cached evaluation environments |

Exit codes: 0 ok, 1 runs/checks failed, 2 invalid configuration, 3 an
evaluation environment could not be built.

## Workflow

Follow these steps in order; do not start `run` before `validate` and
`check` pass, since `run` spends LLM calls.

1. **Install** (Python >= 3.11.5): `uv add iohblade` or
   `pip install iohblade`. Inside a clone of the BLADE repository:
   `uv sync`. Confirm with `iohblade --help`.
2. **Choose the LLM with the user.** Local without a key: Ollama
   (`provider: ollama`, e.g. model `qwen2.5-coder:14b`, after
   `ollama pull`). Hosted: `openai`, `claude`, `gemini`, `deepseek`, with
   `api_key: "${env:OPENAI_API_KEY}"` etc. Never write a key into the YAML
   file; use the environment or a `.env` file next to the YAML file, and
   do not read or print the user's `.env`.
3. **Create the experiment** with `iohblade init`, choosing the template:
   - `builtin`: LLaMEA and random search on a built-in problem;
   - `custom-problem`: the user's own benchmark (`my_problem.py`);
   - `custom-method`: the user's own search method (`my_method.py`).
4. **Edit `experiment.yml`** (and the stub code). Use `iohblade list` for
   valid `type`/`suite`/`provider` keys. See [reference.md](reference.md)
   for every field, the custom-component contracts and LLaMEA operators.
5. **`iohblade validate`** and fix every reported issue. Messages give
   `file:line:col`, the field path and "did you mean" suggestions.
6. **`iohblade check`**. A `[FAIL]` for a problem means the harness or
   environment is broken (fix the problem code or `dependencies`), not the
   LLM. Use `check --llm` once to test keys and model names.
7. **Trial run** before the full one:
   `iohblade run FILE --set experiment.runs=1 --set experiment.budget=4`
   `--set experiment.output_dir=results/trial` (always a separate
   `output_dir`: an existing one only resumes with the same or more runs
   and the same budget).
8. **Full run**: `iohblade run FILE`. Long runs: tell the user the command
   rather than blocking on it, or run it in the background. Rerunning
   resumes.
9. **Inspect**: `iohblade-webapp`, or the files described below.

## Minimal configuration

```yaml
# yaml-language-server: $schema=./experiment.schema.json
experiment: {output_dir: results/my-experiment, runs: 5, budget: 50}   # budget = LLM-generated solutions per run
llms:
  qwen: {provider: ollama, model: "qwen2.5-coder:14b"}
methods:
  - {type: llamea, name: "LLaMEA-{llm}", llm: qwen, params: {n_parents: 4, n_offspring: 12}}
  - {type: random_search, name: "RS-{llm}", llm: qwen}
problems:
  - {type: bbob_sbox, params: {dims: [5], specific_group: 1}}
```

Paths (`output_dir`, `source`, `smoke_test_solution`) are relative to the
YAML file. A method with `llm: [a, b]` is repeated per LLM; `{llm}` in its
name is replaced by the LLM name.

## Results

`<output_dir>/` holds `experiment.log`, `progress.json`, `blade_config/`
(resolved configuration with secrets redacted, copies of custom sources)
and one folder per run with `log.jsonl` (every generated solution with
code, fitness and feedback), the LLM conversation, `run.log` and, if the
run failed, `error.txt`. Fitness is maximised unless the problem sets
`minimisation = True`.

## When something fails

- Invalid configuration (exit 2): apply the `validate` suggestions.
- Environment build failed (exit 3): the installer output is in
  `<output_dir>/env-errors/`; `check -v` shows it in full. Packages built
  from source may need tools (`swig`, `cmake`) listed in `dependencies`.
- Many `h` (harness) errors in the run table, or a run stopping with
  `HarnessError` after 3 in a row: the problem's evaluation is broken, not
  the generated code. Re-run `check` and fix the problem.
- Many `c` (code) or `t` (timeout) errors: the LLM's code fails; improve
  the task/example prompt, raise `eval_timeout`, or use a stronger model.
- Details: [reference.md](reference.md#troubleshooting).
