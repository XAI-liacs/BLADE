# Changelog

## v1.2.0 (unreleased)

Changes since [v1.1.1](https://github.com/XAI-liacs/BLADE/releases/tag/v1.1.1).
Experiments can now be defined in YAML and run with the new `iohblade`
command, including your own problems and methods, with checks and error
reports that point at the cause. Python scripts written for 1.1 keep working;
see "Breaking changes" for the two exceptions.

### Breaking changes

- Python >= 3.11.5 is required (was 3.11), because `llamea>=1.4.3` requires it.
- A run now stops with `HarnessError` after 3 consecutive evaluations that
  fail before the generated code is used (a broken problem or environment),
  instead of spending its whole budget. `problem.max_harness_errors = None`
  restores the old behaviour.

### Major changes

- **Skill for coding agents:** `iohblade skill [folder]` installs a skill
  (`SKILL.md` and `reference.md`, default `.claude/skills/blade/`) that
  teaches coding agents such as Claude Code to set up, validate, check and
  run BLADE experiments. The repository has a copy in `.claude/skills/blade/`.
- **Documentation** (#194): tutorials for a custom problem and a custom
  method, the YAML guide, and the JSON schema published at
  `https://xai-liacs.github.io/BLADE/experiment.schema.json` for editor
  completion. The README quick start shows the YAML and the Python route.
- **Error reporting and logs** (#193):
  - Experiments no longer call `logging.disable(logging.CRITICAL)`. All log
    messages, including library output and Python warnings, go to
    `<experiment>/experiment.log`, and each run's messages also go to its
    own `run.log`. Terminal handlers are detached only while the experiment
    runs (`show_stdout=False`). The level is set with
    `Experiment(log_level=...)` or `experiment.log_level` in YAML.
  - The run overview shows the LLM, evaluations used out of budget, the best
    fitness and the failed evaluations by kind: in the generated code, in
    the harness, or by timeout.
  - After the runs, a summary groups failed runs by exception type, gives
    the paths to their tracebacks and error totals, and says what to do
    next.
  - Evaluation environments of all problems are built before any run.
    Failures are reported together (`EnvironmentSetupError`, full output in
    `<experiment>/env-errors/`), and `iohblade run` exits with code 3.
  - `progress.json` entries record `best_fitness`, `error_counts` and
    `last_error`. The web app shows them in a run table, together with the
    resolved configuration, the custom code and `experiment.log`.
- **Smoke tests, starter projects and cached evaluation environments**
  (#192):
  - `iohblade check experiment.yml` builds each problem's evaluation
    environment and evaluates a reference solution through the real
    pipeline (the new `smoke_test_solution` of a problem, or the code in
    its example prompt). It then runs every method briefly with an offline
    LLM replaying that solution. With `--llm`, it also sends one prompt to
    each LLM. The exit code is 1 when a check fails.
  - `iohblade init [dir] --template builtin|custom-problem|custom-method`
    writes a starter experiment that passes `check`.
  - `iohblade run --dry-run` reports how many runs an existing output
    directory has finished, and whether resuming is possible.
  - Evaluation environments are cached and shared between problems with
    the same dependencies (`~/.cache/iohblade/envs`, or
    `IOHBLADE_CACHE_DIR`). Packages are installed with `uv` when available,
    falling back to `pip`; `iohblade cache list|clear` manages the cache.
    Opt out with `experiment.env_cache: false`, `IOHBLADE_ENV_CACHE=0` or
    `problem.env_cache = False`; `Problem.cleanup()` only deletes
    non-cached environments.
  - `experiment.isolated_pythonpath` / `Problem.isolated_pythonpath` expose
    only `iohblade` (not the rest of BLADE's environment) to the
    evaluation.
- **Custom code in YAML experiments** (#191): problems, methods and LLMs
  can be loaded from your own `.py` file or package with `type: custom`
  (`provider: custom`), `source` and `class`. The module is imported under
  its real name, its folder is added to the evaluation environment's
  `PYTHONPATH`, and dependencies come from `dependencies`, a
  `requirements.txt` or a package's `pyproject.toml`. `validate` reports
  missing files, name clashes with installed modules, missing third-party
  imports (with the install command), syntax errors, wrong base classes and
  unimplemented abstract methods. Runs record the resolved configuration
  (secrets redacted) and a copy of the custom code in
  `<output_dir>/blade_config/`, and the source hash in each run's
  `problem.json` / `method.json` / `llm.json`.
- **Experiments from YAML** (#190): an experiment can be defined in a YAML
  file and checked or run with the new `iohblade` command
  (`iohblade validate|run|list|schema`) or `Experiment.from_yaml()` /
  `iohblade.config.load_experiment()`. Components are named by registry key,
  `params` are checked against their constructors, LLMs are defined once and
  methods can be expanded per LLM, problem suites expand into their problems,
  `${env:NAME}` reads secrets from the environment or a `.env` file, and
  `--set key=value` overrides any value. All configuration errors are
  reported together with file, line and column and "did you mean"
  suggestions. See `docs/yaml_experiments.rst` and `examples/configs/`.
  Existing Python scripts are unaffected. `pydantic` and `pyyaml` are now
  explicit dependencies (they were already installed through other
  packages). LLaMEA `operators` can be given as mappings with the fields
  of `llamea.Operator` (`prompt`, `name`, `weight`, `number_of_parents`).

### Bug fixes

- **Optimisation direction (#201).** On minimisation problems (TSP, VRP,
  graph colouring, hexagon packing, ...) a failed evaluation could become
  the best solution:
  - A timeout, crash or exhausted budget scored `-inf` regardless of the
    direction; it now scores the worst value (`Problem.failure_fitness()`:
    `inf` when minimising).
  - RandomSearch, LHNS, EoH, ReEvo and MCTS-AHD never select a failed
    (non-finite) solution as their best. They use the new helper
    `iohblade.fitness.is_better`.
  - EoH and ReEvo returned the fitness with the wrong sign (EoH when
    minimising, ReEvo for negative fitness). ReEvo also crashed on a
    response without code.
  - MCTS-AHD backpropagated the maximum child Q, and ran its UCT selection
    backwards, when minimising. It also treated a score of 0.0 as missing.
  - The MCTS-AHD and MoEH prompts always asked for a "lower" objective
    value; they now state which direction is better.
- Every run appeared twice in `progress.json`, and in the run overview and
  web app: the planned entries had no `llm_name`, so the entry a run used
  was added separately. Planned entries now carry `llm_name`, and progress
  files from older versions are cleaned up when an experiment is resumed.
- LLaMEA operators with more than one parent (crossover) crashed with
  llamea 1.4.2 under the default `parent_selection="random"`. BLADE now
  requires `llamea>=1.4.3`, which fixes this, and therefore Python >= 3.11.5
  (llamea 1.4.3's minimum).
- **Built-in example solutions work again** (#192). The example code that
  the built-in problems show the LLM was broken or a skeleton for 17 of 26
  problems: wrong indentation, hyphens in class names, wrong output lengths,
  degenerate points or typos. ToyMultiObjective showed no example at all,
  because the placeholder prompts set by `Problem.__init__` hid its
  class-level prompts. All examples are now simple, valid baselines that
  evaluate to a finite fitness. A test (`test_benchmark_examples.py`) keeps
  it that way, and `iohblade check` can use them as reference solutions.
- `UncertaintyInequality` called `float()` on one-element arrays, which
  NumPy >= 2.5 (installed in the evaluation environment) rejects, so no
  solution could be scored. It now evaluates the polynomial at scalars.
- Evaluation environments of the smac-based benchmarks (BBOB-SBOX, MA-BBOB,
  Kerneltuner, AutoML) could not be built without a system `swig`, needed
  to build `pyrfr`. `swig` is now installed from PyPI first, and the
  environment's scripts folder is on `PATH` while installing.
- The example scripts `run-sbox.py`, `run-mabbob.py`, `run-bbob-logging.py`,
  `run-kerneltuner.py`, `run-kerneltuner-info.py`, `minimal_example.py` and
  `minimal_example_trackio.py` passed `mutation_prompts` to LLaMEA, which
  `llamea>=1.4` no longer accepts, so they failed when a run started. They now
  pass `llamea.Operator` objects as `operators`.

### Minor changes

- **Component registry** (`iohblade.registry`): built-in problems, problem
  suites, methods, LLM providers and loggers can be looked up by a short key
  (`registry.resolve("method", "llamea")`), with `StrEnum`s such as
  `BuiltinProblem` and `LLMProvider`. Targets are imported lazily; a missing
  optional dependency raises `ComponentUnavailableError` with the install
  command, and an unknown key raises `ComponentNotFoundError` with "did you
  mean" suggestions. Custom components can be added with `register_problem`,
  `register_method`, etc. Existing imports are unchanged (#189).
- `Problem.extra_pythonpath`: directories added to `PYTHONPATH` of the
  evaluation subprocess, so problems defined outside the `iohblade` package
  can be unpickled in the evaluation environment (#189).
- **Harness errors** (#189): an evaluation that fails before the problem has
  read `solution.code` (failing `imports` block, a problem class that cannot
  be unpickled, a missing dependency, a bug in `evaluate` before the code is
  used, an environment setup failure inside `Problem.__call__`) is no
  longer reported to the LLM as an
  error in its code. Its feedback starts with "Evaluation harness error", and
  after `Problem.max_harness_errors` (default 3) consecutive harness errors
  the run is stopped with `HarnessError` and recorded as failed, instead of
  spending the remaining budget. Set `problem.max_harness_errors = None` to
  restore the previous behaviour. Errors raised after the code was read are
  handled as before. (`Experiment` still builds each problem's environment
  before starting any run and stops right away if that fails.)

## v1.1.1 (2026-10-22)

Changes since [v1.1.0](https://github.com/XAI-liacs/BLADE/releases/tag/v1.1.0).

### Major changes

- **LLaMEA 1.4.1**. BLADE now depends on `llamea>=1.4.1`.

### Minor changes

- Run names include the LLM model name
  (`{method}-{llm}-{problem}-{seed}`), so runs of the same method and problem
  with different LLMs no longer share a name (#175). `/` and `:` in model
  names are replaced by `-`.
- GPU detection for the LLM hardware info ignores "VGA compatible controller"
  devices, which are often integrated graphics rather than the GPU running the
  LLM (#175).
- `Solution` has a new `operator_id` field, kept through `copy()`,
  `to_dict()`/`from_dict()` and pickling (#174).
- Dependency updates: aiohttp 3.14.4 (#173).

### Bug fixes

- Every LLaMEA (>= 1.4) offspring was scored as an error: the `Operator`
  object stored on the solution could not be unpickled in the evaluation
  subprocess, where LLaMEA is not installed. The operator is now stored as
  text plus `operator_id` (#174).
- Parallel runs (`n_jobs > 1`) that share an evaluation environment could read
  each other's half-written problem pickle or run each other's evaluation
  script, so a solution could be scored with the wrong code. Each evaluation
  now uses unique file names and removes them afterwards (#174).

## v1.1.0 (2026-10-08)

Changes since [v1.0.0](https://github.com/XAI-liacs/BLADE/releases/tag/v1.0.0).

### Major changes

- **Multi-objective optimisation support** (#120). New `iohblade.Fitness` class
  with Pareto-dominance comparisons, dict serialisation and a scalar summary.
  `Solution`, the loggers and the plots handle multi-objective fitness, with a
  new `plot_pareto_front` plot. Adds the `ToyMultiObjective` bi-sphere benchmark
  and a documentation page (`docs/fitness.rst`).
- **New method: MoEH** (Multi-objective Evolution of Heuristics, #157), available
  as `iohblade.methods.MoEH_Method`, with E2/M1/M2 operators and its own
  population management.
- **Database-ready experiment exports** (#146, #166, #167). Every run now writes
  `method.json`, `problem.json` and `llm.json` next to `log.jsonl` and
  `conversationlog.jsonl`. Methods, problems and LLMs expose `get_config()`; LLM
  configs include hardware information. Exports carry SHA-256 hashes so
  configurations can be deduplicated, and new `iohblade.tags` enums (`PrimaryCategories`, `Benchmark`)
  give consistent problem tagging.
- **Automatic optimisation direction** (#160). LLaMEA, EoH, ReEvo, LHNS,
  MCTS-AHD and Random Search read the problem's `minimisation` flag, so
  minimisation problems no longer need method-specific configuration.
- **AutoML benchmark overhaul** (#169).
  - Candidates are scored on an inner validation split of the OpenML train fold
    (`search_eval_mode="inner_val"`, the default). The official test folds are
    used once, for the final score of the returned pipeline. The old behaviour is
    available with `search_eval_mode="official_test"`.
  - Generated code that runs its own CV/HPO search (e.g. `GridSearchCV`) is
    rejected: tuning is done by SMAC and the budget counts evaluations.
  - Optional TabPFN modes (`passive`, `explicit`; off by default).
  - Per-split scores are stored as `search_scores` and `final_scores` (replacing
    `fold_scores`).
  - Error feedback shows the error and the failing line.
- **Core: final evaluation and site packages** (#169). `Experiment` calls
  `problem.final_evaluation(result)` after the search when a problem defines it.
  New `Problem.use_system_site_packages` option (off by default) lets
  evaluations use the packages installed in the host environment.
- **LLaMEA 1.4.0**. BLADE now depends on `llamea>=1.4.0` from PyPI, replacing the
  pinned GitHub commit. This brings the `llamea.operator.Operator` API used by
  the AutoML example runner.

### Minor changes

- Documented the LMStudio and MLX_LM (beta) Apple Silicon backends in the
  README and docs (#152).
- `examples/run-automl.py` uses the LLaMEA operators and adds `--tasks`,
  `--eval-timeout` and `--eval-cpus`, plus DeepSeek, Gemini and OpenAI models
  next to Ollama.
- All benchmark prompts are wrapped in `textwrap.dedent()` (#133).
- `py-cpu-info` moved to the `dev` dependency group; Apple-Silicon-only tests
  are skipped on other platforms.
- Added `CLAUDE.md` with contributor guidance for AI assistants (#124).
- Many dependency updates (Dependabot), including mlflow 3.11, pyarrow 23,
  pillow 12, starlette 1.0, pytest 9 and black 26.

### Bug fixes

- AutoML benchmark prompts were sent to the LLM with 8 spaces of leading
  indentation (#133).
- Evaluation subprocess (#168):
  - The problem pickle was only written once per environment, so later changes
    to the problem were not seen during evaluation.
  - Results larger than the pipe buffer blocked the worker and were reported as
    timeouts; the result is now read before joining the worker.
  - The pip output is shown when installing problem dependencies fails.
- The ConfigSpace was lost after the second pickle, so HPO never ran inside the
  evaluation (#168).
- Exporting the method config failed when LLaMEA `Operator` objects were
  passed (#168).
- Class names were not extracted for classes with several base classes (#168).
- ReEvo seeding failed when the example prompt had text around the code (#168).
- Tags now subclass `str` so they serialise to JSON without conversion.
- Documentation version selector redirected to the wrong path (#121).
