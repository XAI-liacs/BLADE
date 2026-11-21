# Development plan: YAML-defined experiments

Status: proposal · Scope: `iohblade` ≥ 1.2

## 1. Goal

A user should be able to define, check and run a complete BLADE experiment
from one YAML file:

```bash
iohblade init my-exp --template custom-problem   # scaffold yml + stubs
iohblade validate my-exp/experiment.yml          # static checks, no LLM calls
iohblade check my-exp/experiment.yml             # smoke test: env + 1 eval
iohblade run my-exp/experiment.yml               # full run, resumable
```

Built-in benchmarks, methods, LLM providers and loggers are named by key and
configured with parameters. Custom problems and methods are linked as a
`.py` file or a package folder. BLADE must then load them in the main
process, the spawned evaluation process **and** the per-problem virtualenv.
When something fails, the user must be told *what* failed, *where* (YAML
line, file, run) and *what to do*, both in the terminal and in the logs.

## 2. Target YAML format (draft)

```yaml
# yaml-language-server: $schema=https://xai-liacs.github.io/BLADE/schema/experiment.json
version: 1

experiment:
  name: sbox-vs-custom
  output_dir: results/sbox-vs-custom   # existing dir => resume
  runs: 5                              # or: seeds: [0, 1, 2]
  budget: 100                          # default for every method
  n_jobs: 4
  show_stdout: false
  log_stdout: true
  logger: {type: file}                 # file | mlflow | trackio (+ options)

llms:                                  # named, reusable
  flash:
    provider: gemini
    model: gemini-2.0-flash
    api_key: ${env:GEMINI_API_KEY}     # secrets never inline
  qwen:
    provider: ollama
    model: qwen2.5-coder:14b

methods:
  - type: llamea
    name: "LLaMEA-{llm}"               # template, expanded per LLM
    llm: [flash, qwen]                 # list => one method per LLM
    params: {n_parents: 1, n_offspring: 1, elitism: true}
  - type: custom
    name: MyMethod
    source: ./methods/my_method.py     # file or package folder
    class: MyMethod
    llm: qwen

problems:
  - type: bbob_sbox
    name: SBOX_fid2
    params:
      training_instances: [[2, 1], [2, 2], [2, 3]]
      test_instances: [[2, 5], [2, 6]]
      dims: [5]
      budget_factor: 2000
      specific_fid: 2
  - suite: hexagon_packing               # factory => several problems
    params: {get_best: false}
  - type: custom
    name: MyProblem
    source: ./problems/my_problem/      # package with __init__.py
    class: MyProblem
    dependencies: [scipy>=1.11]         # or requirements.txt in the folder
    eval_timeout: 600
    params: {n: 26}
    smoke_test_solution: ./problems/my_problem/example_solution.py
```

Notes:
- Relative paths are resolved against the YAML file, not the CWD.
- `${env:VAR}` and `${env:VAR:-default}` interpolation; a `.env` next to the
  YAML is loaded if present.
- CLI overrides: `iohblade run exp.yml --set experiment.runs=1`
  (`--only method=MyMethod` is still open).
- YAML lists are passed as lists. No tuple conversion is needed: the
  benchmarks unpack `(fid, iid)` pairs, which works for lists too. Strings are
  converted to enum values where a parameter expects an enum
  (`problem_type: BBOB`).

## 3. What the current code requires (findings)

These constrain the design:

1. **No registry.** Built-ins are exposed through `try/except` imports in
   `iohblade/benchmarks/__init__.py` and `iohblade/methods/__init__.py`
   that set the name to `None` on failure. This hides *why* a benchmark is
   unavailable. Several benchmarks are factory functions that return lists
   (`get_hexagon_packing_problems`, `get_fourier_problems`, ...).
2. **The budget is set twice.** `Experiment(budget=...)` and
   `Method(llm, budget)` are separate, and scripts pass both. The config
   layer (Phase 1) should default the method budget to the experiment
   budget. (Minimisation needs no change: the methods already read
   `problem.minimisation` when they are called; passing it in
   `run_benchmarks/circle_packing.py` is redundant.)
3. **Custom classes must be importable in three places.**
   - Main process: `Experiment` deep-copies methods and problems and runs
     them in threads. This is fine once the module is imported.
   - Spawned process: `Problem.__call__` starts a
     `multiprocessing.Process` (spawn) and passes `self` with the standard
     pickle. The child must be able to import the user's module **by name**,
     so `importlib.util.spec_from_file_location` with a synthetic name will
     break.
   - Evaluation virtualenv: `evaluate_in_subprocess` cloudpickles the
     problem and runs it with the venv's Python, setting
     `PYTHONPATH=<repo_root>` (`Path(__file__).parents[1]`). User code is
     not on that path. Also, with a pip install, `repo_root` is
     `site-packages`, so the whole host site-packages leaks into the
     "isolated" venv. This needs fixing anyway.
4. **Every experiment builds a new venv** (`tempfile.mkdtemp`, then
   `pip install`). That is too slow for a fast `validate`/`check` loop.
5. **Harness bugs look like candidate bugs.** Any exception inside
   `problem.evaluate` becomes `-inf` plus feedback. `simplify_subprocess_error`
   maps the line number onto `solution.code`. A bug in a *custom problem's*
   evaluator therefore gives each candidate `-inf` and misleading LLM
   feedback ("line 42 of your code ..."), and the run burns its whole budget
   without stopping.
6. **Errors are hidden.** When `show_stdout=False`,
   `logging.disable(logging.CRITICAL)` is set and stdout/stderr go to
   `None`. Only run-level exceptions get an `error.txt`. Environment setup
   failures (pip) surface as one `RuntimeError` per run.
7. `pydantic` (2.x) and `PyYAML` are already installed through dependencies
   (openai, mlflow), so declaring them explicitly costs nothing.

## 4. Architecture

New subpackage `iohblade/config/`:

| Module | Responsibility |
|---|---|
| `registry.py` | Lazy registry `key -> "module:attr"` for problems, suites, methods, LLM providers and loggers, with metadata (optional dependency group, description). `@register_problem("key")` decorators plus entry points (`iohblade.problems`, `iohblade.methods`) so third-party packages can plug in. Generates `StrEnum`s (`BuiltinProblem`, `BuiltinMethod`, `LLMProvider`) for the Python API and the JSON schema. |
| `schema.py` | Pydantic v2 models: `ExperimentConfig`, `LLMSpec`, `MethodSpec`, `ProblemSpec` (discriminated on `type`), `LoggerSpec`. Exports the JSON schema for editor autocompletion. |
| `yaml_loader.py` | Safe YAML load that keeps line/column marks per node, plus env interpolation, `.env` loading, `--set` overrides and relative-path resolution. |
| `user_code.py` | Loads custom files/packages (see §5), snapshots them and hashes them. |
| `build.py` | Spec → objects: validates `params` against `inspect.signature` of the target, expands the method × LLM matrix and suites, propagates budget and minimisation, returns an `Experiment`. |
| `errors.py` | `ConfigError` (collects many issues, each with a YAML location and a hint), `UserCodeError`, `EnvironmentSetupError`, `HarnessError`. |
| `cli.py` | `iohblade` entry point: `init`, `validate`, `check`, `run`, `list`, `schema`. Uses `argparse` (no new dependency); `rich` is optional for nicer output. |

Python API (scripts keep working unchanged):

```python
from iohblade.config import load_experiment
exp = load_experiment("experiment.yml", overrides={"experiment.runs": 1})
exp()
# or Experiment.from_yaml("experiment.yml")
```

## 5. Custom code handling (the hard part)

**Loading.**
- *File* `foo.py`: add its directory to `sys.path` and import `foo` by its
  real name.
- *Folder* `pkg/`: add the parent directory to `sys.path` and import `pkg`,
  so relative imports and data files work.
- If the name shadows an installed module or a stdlib module, raise an
  error, unless the user sets `module_name:`.
- Check that `class` exists and is a subclass of `Problem` / `Method` / `LLM`,
  and that its abstract methods are implemented (`to_dict`, `get_config`,
  `test`, ...). Report the missing ones by name.

**Propagation.** The loader records the source roots on the object, as a new
`Problem.extra_pythonpath: list[str]` attribute:
- Spawn child: `sys.path` is passed through spawn preparation data. Add a
  test that confirms this on Linux, macOS and Windows. Also set
  `PYTHONPATH` in `os.environ` as a fallback.
- Eval venv: `evaluate_in_subprocess` builds `PYTHONPATH` from
  `extra_pythonpath` plus a path that exposes **only** the `iohblade`
  package. To do that, symlink or copy `iohblade/` into a small shim dir
  instead of using `parents[1]`. This also fixes the site-packages leak.
- An alternative for single-file problems is
  `cloudpickle.register_pickle_by_value(module)`. Keep it as an option, not
  the default, because it breaks data files and package-relative imports.

**Dependencies.**
- Problem dependencies come from `dependencies:` or from
  `requirements.txt` / the `[project].dependencies` of a `pyproject.toml`
  in the folder. They are installed into the problem's eval venv, as today.
- Custom *methods* (and LLM providers) run in the main BLADE environment.
  `validate` checks that their imports resolve and prints the exact
  `uv add ...` / `uv sync --group methods` command when they don't. Methods
  are not isolated; the docs say so.

**Venv cache.** Replace the per-experiment `mkdtemp` venv with a
content-addressed cache: `~/.cache/iohblade/envs/<sha256(python version,
sorted deps, use_system_site_packages)>`. Use a file lock for parallel
experiments, use `uv pip install` when `uv` is present (fall back to pip),
and add `iohblade cache clear`. `problem.cleanup()` no longer deletes cached
envs.

**Reproducibility.** Copy each custom source into
`<output_dir>/sources/<name>/` and write a source hash into
`method.json` / `problem.json` (the `source` field `get_config` already
expects). Also write `experiment.yml` (original),
`experiment.resolved.yml` (expanded, secrets redacted) and
`envs/<hash>.freeze.txt`.

## 6. Test runners for users

- **`iohblade validate`** (seconds, offline):
  - schema errors and unknown keys, with "did you mean" suggestions
    (`difflib`)
  - registry keys are known and their optional dependency group is installed
  - `params` bind to the constructor signature
  - env vars are set
  - custom code imports, has the right base class and implements its
    abstract methods
  - paths exist
  - names are unique after expansion
  - the output dir is either new or resumable with a compatible config
    (warn when the config hash differs from the stored one)
- **`iohblade check`** (minutes), per problem:
  - build or reuse the venv and show pip errors verbatim
  - evaluate a reference solution in the real subprocess pipeline: the
    `smoke_test_solution`, or the code in `example_prompt` for built-ins
  - verify that a `Solution` comes back with a finite fitness, and report
    the time against `eval_timeout`

  Per method: run 1–2 iterations with `Dummy_LLM` that replays the smoke
  solution, on the cheapest problem. With `--llm`, send one short ping query
  to each configured LLM to check the key, model name and connectivity.
- **`iohblade run --dry-run`**: print the run matrix (method × LLM × problem
  × seed), the runs that resuming would skip and the cost estimate.
- **Scaffolding**: `iohblade init --template {builtin,custom-problem,custom-method}`
  writes a YAML file and stub classes. The stubs pass `check` out of the box
  and contain TODOs.

## 7. Error reporting and logging

1. **Config errors** are reported all at once, one line per issue:
   `experiment.yml:23:7 problems[1].params.dimz: unknown parameter for BBOB_SBOX; did you mean 'dims'?`
   Exit code 2 for config errors, 3 for environment errors, 1 for run
   failures.
2. **Separate harness errors from candidate errors** (done in Phase 0).
   The evaluation script records the first access to `solution.code`. An
   uncaught error raised before that access cannot come from the candidate,
   so it is a harness error. Errors raised after it keep the current
   feedback behaviour, which is conservative. A circuit breaker aborts the
   run with `HarnessError` after *k* consecutive harness errors (default 3).
3. **Structured logs.** Replace `logging.disable(CRITICAL)` with per-run
   `run.log` handlers (the `iohblade` logger, level set from the YAML or
   `-v`) and one `experiment.log`. Third-party noise is still hidden from
   the terminal but kept in the files.
4. **Terminal UX.** Keep the run overview table and add columns for evals
   used/budget, best fitness and last error category. At the end, print a
   summary grouped by error type, with the paths to `error.txt` / `run.log`
   and one suggested next command (e.g. `iohblade run exp.yml` retries only
   the failed runs).
5. **Environment errors** (`EnvironmentSetupError`) are raised once per
   problem before any run starts. Today they repeat in every run. Include
   the tail of the pip output and the full log path.
6. **Webapp.** Show the resolved config, the source snapshot and the error
   categories per run.

## 8. Phased roadmap

Each phase is one or more PRs that can be merged on their own, each with
tests.

| Phase | Content | Depends on | Size |
|---|---|---|---|
| **0. Groundwork** | Registry with lazy imports and clear "install group X" errors. `extra_pythonpath` in `evaluate_in_subprocess`. `HarnessError` classification and circuit breaker. (The iohblade-only `PYTHONPATH` shim moves to Phase 3 behind a flag, since removing the site-packages leak could break existing pip-installed setups.) | – | M |
| **1. Built-in YAML** | `schema.py`, `yaml_loader.py` (marks, env, overrides), `build.py`, `load_experiment`, `iohblade validate/run/list/schema`. Built-ins only. Convert 3 scripts from `run_benchmarks/` to YAML as dogfooding. | 0 | L |
| **2. Custom code** | `user_code.py` (file and folder, shadowing checks, ABC checks), dependency discovery, source snapshot and hash, resolved-config output, custom LLM providers. | 1 | L |
| **3. Test runners and env cache** | Content-addressed venv cache with lock and uv fast path. `iohblade check`, `--dry-run`, `init` templates. | 2 | M |
| **4. Error UX and logging** | Structured logging, end-of-run summary, exit codes, env errors raised before runs, webapp config/error view. | 0–3 | M |
| **5. Docs and release** | User guide "Experiments from YAML", generated schema reference (published to GH Pages for the `$schema` link), tutorials for custom problems and methods, CHANGELOG, migrate the README quickstart. | all | S |

## 9. Testing strategy

- **Unit** (`tests/unit/config/`):
  - schema round-trips and JSON-schema export
  - env interpolation and `--set` overrides
  - every registry key resolves, or reports a missing group cleanly
  - golden-file tests for error messages (YAML line:col, suggestions)
  - signature binding and tuple coercion
  - method × LLM expansion
- **Custom code**: fixtures in `tests/fixtures/custom/` (single file, package
  with relative import and data file, broken import, missing abstract
  method, name shadowing `numpy`). Pickle round-trip through a real spawn
  process. Evaluation in a real venv, marked `@pytest.mark.slow` and run in
  e2e CI.
- **Harness vs candidate errors**: a problem whose evaluator raises must
  trigger the circuit breaker; a candidate that raises must not.
- **CLI**: call `cli.main([...])` in-process with `tmp_path`; check exit
  codes and stderr.
- **E2E**: one YAML file with `Dummy_LLM`, one built-in problem and one
  custom problem, run on Ubuntu and Windows (path separators,
  `Scripts/python`).
- All LLM calls are mocked; coverage target ≥ 80% for `iohblade/config/`.

## 10. Decisions to confirm

1. **Pydantic v2 + PyYAML as explicit dependencies** (recommended; both are
   already installed transitively). Use `ruamel.yaml` only if
   round-tripping comments matters, which it does not for this feature.
2. **LLMs as a named top-level section, with method × LLM expansion**
   (recommended), rather than nesting an LLM inside every method.
3. **Custom methods run unisolated in the main environment** (recommended).
   Isolating them would require proxying LLM and logger objects across
   processes.
4. **Env cache location and the `uv` fast path** (recommended) instead of
   per-experiment temp venvs.
5. **YAML `version: 1` field**, so the format can evolve with migrations.
