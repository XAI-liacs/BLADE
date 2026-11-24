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
- **FunSearch (#216):** `iohblade.methods.FunSearch` (`type: funsearch`)
  replaces the stub that raised `NotImplementedError`. It reimplements the
  FunSearch programs database (Romera-Paredes et al., Nature 2024), after
  the reference code and the LLM4AD port:
  - islands of programs clustered by score;
  - best-shot prompts with versions ordered worst to best;
  - temperature-scheduled cluster sampling that prefers shorter programs;
  - resets of the weaker half of the islands.

  Resets are counted in evaluations rather than wall-clock time, so runs
  are reproducible. It evaluates exactly `budget` solutions, respects
  `problem.minimisation`, never keeps failed evaluations, and rejects
  multi-objective fitness. Parameters: `num_islands`,
  `programs_per_prompt`, `samples_per_prompt`, `reset_period`,
  `temperature_init`, `temperature_period`. The old class name
  `funsearch` still works.
- **ShinkaEvolve (#217):** `iohblade.methods.ShinkaEvolve`
  (`type: shinka_evolve`), a reimplementation of ShinkaEvolve (Lange et al.,
  2025) after its reference code (version 0.0.8). It includes:
  - islands with a global archive, and weighted, power-law or beam-search
    parent selection;
  - archive and top-k inspirations;
  - diff (SEARCH/REPLACE), full-rewrite (five prompt variants) and
    crossover patches, with a fix mode for islands without a correct
    program;
  - optional migration;
  - optional novelty rejection with a local n-gram embedding and an LLM
    judge;
  - optional meta recommendations;
  - an asymmetric UCB bandit over the LLMs of a `provider: multi` LLM.

  The `shinka` package itself is not used: it conflicts with BLADE's
  `openai` and `google-genai` pins, and its budget counts generations.
  Every generation here ends in exactly one evaluation, so a run evaluates
  exactly `budget` solutions; novelty-rejected candidates use LLM calls but
  no evaluations. Nothing is written to disk.
  Example: `examples/configs/shinka_evolve.yml`.
- FunSearch and ShinkaEvolve prompts include the context a problem adds
  in `get_prompt()` (such as the photonics descriptions and insights), not
  only its `task_prompt` (`iohblade.methods.prompt_utils.task_context`).
  The same applies to MCTS-AHD (all prompts), LHNS (its variation prompts)
  and MoEH, whose mutation and crossover prompts now also start with the
  task description, as in EoH.

- **Default benchmark suites:** new suites in `iohblade.benchmarks.suites`,
  registered for YAML (`- suite: <name>`):
  - `bbob`: one BBOB problem per function group (5 problems). Instances 1-5
    of each function are for training and 6-15 for testing.
  - `ma_bbob`: MA-BBOB, with instances 0-24 for training and 25-74 for
    testing.
  - `continuous_bbo`: `bbob` and `ma_bbob` together.
  - `alphaevolve_math`: the 17 AlphaEvolve mathematical problems.
  - `combinatorial`: Steiner trees, graph colouring, TSP and CVRP.
  - `quick`: a small cross-domain set.
  - `full`: all of the above, plus circle packing and spherical codes;
    optionally also matrix multiplication.

  `dim` (default 10) and `budget_factor` (default 2000) apply to all
  continuous problems of a suite. Example:
  `examples/configs/default_suites.yml`.
- **Evaluation timeouts:** every built-in problem and suite takes
  `eval_timeout`, with a default of at least 120 s. The AlphaEvolve problems
  default to 600 s (previously a fixed 6000 s), and the combinatorial
  problems to 300 s. MA-BBOB now defaults to 120 s instead of 60 s. In the
  default suites, BBOB and MA-BBOB get 1 ms per function evaluation (at
  least 120 s).
- **Quieter suites:** the AlphaEvolve problems no longer print a banner
  when they are created. The message is logged at DEBUG level on the
  `iohblade.benchmarks` logger.
- **Suite flags:** every AlphaEvolve suite takes `use_best`, which is now
  optional and defaults to `false`. The packing suites used `get_best`;
  that name still works as an alias.

### Bug fixes

- `sum_vs_difference`: the problem was named `"Problem"`, because its name
  was passed as the `logger` argument, and its reference value was stored as
  `best_score` instead of `best_known`. It is now named
  `sums_vs_differences`, takes `best_known` (`best_score` still works) and
  sets both attributes.
- Graph colouring problems were named after their bare id (for example
  `"1"`); they are now named `graph_coloring-gcol<id>`.
- `SphericalCode` used the fixed name `spherical-code-n30` for any
  `n_points`; it now uses its `n_points`.
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
- **Hyperparameter tuning on BBOB-SBOX, MA-BBOB and Kerneltuner (#221).**
  With `HPO=True` (LLaMEA), these benchmarks generated configuration spaces
  but never used them.
  - **Tuning before scoring:** they now tune each algorithm's
    hyperparameters with SMAC before scoring it, on a fixed subset of the
    training instances, minimising 1 − AOCC (or 1 − score). The incumbent
    is scored like a normal evaluation, recorded in the solution's metadata
    and reused by `test()`.
  - **New options:** `hpo`, `hpo_trials` (BBOB-SBOX 100, MA-BBOB and
    Kerneltuner 50) and `hpo_instances` (12). Tuning makes evaluations
    longer, so raise `eval_timeout` accordingly.
  - **Lighter environments:** their evaluation environments no longer
    install smac, swig and ConfigSpace by default, so they no longer build
    pyrfr.
  - **Also fixed:** MA-BBOB `test()` crashed (`Analyzer.reset()`), and an
    ioh logger could be garbage-collected while still attached.
- **AutoML benchmark and optional hyperparameter tuning (#220).**
  - **AutoML could not be loaded:** it imported smac at module level, and
    smac 2.4.0 fails to import with scikit-learn 1.9. smac is now imported
    only when tuning runs (`iohblade.hpo`), and tuning uses `smac>=2.4.1`,
    which works with scikit-learn 1.9. smac is no longer a required
    dependency of BLADE (optional group `hpo`, for calling
    `iohblade.hpo.run_smac` directly).
  - **Tuning dependencies are optional:** a problem's evaluation
    environment installs `smac>=2.4.1,<2.5` and `ConfigSpace` only when HPO
    is enabled (`Problem.enable_hpo()`, AutoML `hpo=True`).
  - **Automatic enabling:** HPO is turned on for supported problems when a
    method samples configuration spaces (LLaMEA `HPO: true`).
    `iohblade validate` notes this and warns for problems that cannot tune.
  - **Failures are reported clearly:** tuning without smac is a clear
    harness error (`HPOUnavailableError`). A configuration space evaluated
    without HPO uses default hyperparameters, and its feedback says so.
  - **`openml`** is in the `extraproblems` dependency group.
- **Best-known seeds and reference values (#204).** All values were
  checked against the AlphaEvolve paper (Table 3) and its results
  notebook. Results of earlier versions on these benchmarks are not
  directly comparable.
  - **Auto-correlation:**
    - `use_best=True` passed the best-known score instead of the seed, so
      every seeded candidate failed;
    - **auto-correlation 3** now measures C3 = |max f*f| / (∫f)² as in the
      paper (it measured max|f*f|, the different constant C3′); the seed
      scores 1.455643 (best known 1.4557).
  - **Hexagon packing:** the seeds were in degrees and in the notebook's
    outer-hexagon orientation; they are converted (3.930075 and 3.941896).
    The best-known values are now 3.930092 and 3.941912.
  - **Sums vs differences:**
    - the limits are raised to 2048 elements and values up to 2²¹, so the
      AlphaEvolve seed is valid (1.147989, now the best-known value; 1.1584
      belongs to a construction with 54,265 elements);
    - invalid sets gave "cannot access local variable 'e'" as feedback.
  - **Erdős minimum overlap:** step functions of any length are accepted
    (up to 100,000 values). The seed is the full 95-step construction and
    scores 0.3809230.
  - **Fourier uncertainty:** as in AlphaEvolve, the evaluator adds the H12
    coefficient that makes P(0) = 0. The seed scores 0.3520991, and the
    best-known value is 0.3521 (0.3216 came from a different construction).
  - **Min/max distance ratio:** a seed of the wrong length was not dropped.
  - **Matrix multiplication:** 8 rank-table entries were wrong (e.g.
    (2,4,4) was 46, Table 3 gives 26; four targets were below any known
    rank), and (4,4,6) was missing.
- **Smaller fixes (#209).**
  - **`Solution.set_scores`:** `solution.error` stayed empty for exceptions
    raised outside the candidate code (numpy, the benchmark's own checks),
    and kept a stale value when a solution was scored again; the error
    counts in the run overview missed these. It now always records
    `Type: message`.
  - **Error feedback to the LLM** pointed at the wrong line of the
    candidate's code for errors raised in other files.
  - **`aoc_logger(scale_log=False)`** returned a function instead of a
    number.
  - **`wrap_problem`** passed the problem as the solution to functions such
    as `evaluate(solution, logger=None)`. Its `eval_timeout` defaults are
    now consistent (6000 s).
  - **`Solution`:** the default `parent_ids` list was shared between
    solutions, and `from_dict` dropped `task_prompt`.
  - **`MCTS_Node`:** pickling and copying kept the search-tree fields, and
    `parent_ids` is a list. MCTS-AHD evaluates a node without its tree
    links, so the tree is not sent to the evaluation subprocess.
  - **Kerneltuner:**
    - `test()` always crashed;
    - the cache path was hard-coded to a lab path; it now uses `cache_dir`.
  - **`imports` argument:** ignored by BBOB-SBOX, MA-BBOB, Kerneltuner and
    Photonics; it is now used.
  - **AutoML:** `get_config()` crashed without an OpenML task.
  - **TSP:** the feedback always said the best known distance was `inf`.
  - **Euclidean Steiner tree:** the problem name was hard-coded
    (`EucildianSteinerTree-n20`); it is now `EuclidianSteinerTree-n<id>`.
  - **Graph colouring:** the prompt printed the whole node list instead of
    the number of nodes.
  - **`misc.ast`:** `print_results` and `process_file_paths` always raised
    `KeyError`.
  - **`prepare_namespace`:** it crashed on `from x import *`, on relative
    imports and on version-pinned requirements; its docstring no longer
    claims to isolate code (the subprocess does).
- **YAML configuration and CLI (#208).**
  - **Resuming with other settings:**
    - running an existing `output_dir` with fewer or other runs, e.g.
      `--set experiment.runs=1`, printed a traceback after it had already
      archived and overwritten the recorded configuration. It is now a
      configuration error (exit 2) raised before anything is written, with
      what differs and how to proceed;
    - adding runs (more `runs`, seeds, methods or problems) resumes the
      finished runs and starts only the new ones.
  - **Misspelled `params`** for `random_search`, `eoh` and `reevo` are now
    reported; they were silently accepted.
  - **`${env:NAME}` values:** an unquoted value that is exactly one
    `${env:NAME}` now gets its YAML type (number, boolean, list). Quoted
    values and credentials stay strings. `--set` values get the same
    substitution.
  - **`.env` files:** inline comments and quotes ended up in the values;
    they are now read like python-dotenv.
  - **Custom code** edited and loaded again in the same process (notebooks,
    the Python API) is reloaded; the old class was used before.
  - **Encoding:** all configuration files are read and written as UTF-8;
    a non-UTF-8 file is a configuration error.
  - **`resolved.yml`** no longer redacts settings such as `max_tokens` or
    `tokenizer`.
  - **`sources.json`** stores paths relative to the configuration, so a
    moved experiment folder no longer reports changed code.
- **Methods respect their evaluation budget (#206).** Unequal budgets made
  method comparisons unfair.
  - **LHNS:**
    - evaluated `budget + 1` solutions;
    - made 5 LLM calls at initialisation (keeping only the last);
    - in tabu-search mode it never sent the tabu prompt, re-evaluating the
      unchanged solution instead.
  - **EoH:** the generation count assumed one evaluation per individual,
    but EoH evaluates `2·pop_size` at the start and
    `n_operators·pop_size` per generation, about 4x the budget. It now runs
    as many whole generations as fit; the adapter also stops at the budget.
    With a budget smaller than the initial population, `pop_size` is
    reduced with a warning.
  - **MCTS-AHD:** whole expansion rounds were evaluated without checking
    the remaining budget (budget 10 gave 15 evaluations). It now stops
    generating nodes at the budget. `alpha` is documented as currently
    unused.
  - **MoEH:** two budget counters (budget 10 gave 16 evaluations) are
    replaced by one, and the last generation is capped.
- **Experiment runner and loggers (#207).**
  - **MLflow:** any experiment with more than one run crashed ("Run ... is
    already active"), and per-step logs from worker threads went to
    auto-created runs. Every run now gets its own MLflow run through an
    `MlflowClient` and is marked FINISHED or FAILED. The docstring
    recommends a `sqlite:///` tracking URI, since mlflow 3.17 refuses
    `file:` stores.
  - **Parallel runs (`n_jobs > 1`):**
    - one thread-safe stdout/stderr router replaces the per-run
      `redirect_stdout`, which could mix runs' output and leave
      `sys.stdout` replaced afterwards;
    - a warning says that individual seeds are not exactly reproducible
      with thread-parallel runs. `random` is now seeded too.
  - **Captured output** (`log_stdout`) is saved when a run fails; it was
    lost exactly then.
  - **`progress.json`** is written atomically, so an interrupted write no
    longer prevents resuming; an unreadable file gives a clear error.
  - **Trackio:**
    - runs were all named `""`;
    - logging from a method's helper threads failed; each run logger now
      logs through its own run object.
  - **ReEvo** no longer writes its reflection files into the current
    directory. They go to the run folder (`reevo/`), to `output_path`, or
    to a temporary folder.
  - **Evaluation-environment shim:** creating the `isolated_pythonpath`
    shim from parallel threads could fail; Windows copies of it are
    refreshed after an upgrade.
  - **`log_configspace`** creates its folder.
- **Plots, web app and analysis (#202).**
  - **Best-so-far direction:** curves, "best" selections and "Top
    Solutions" always maximised. On minimisation problems they showed the
    worst solutions. They now follow the problem's logged direction
    (`ExperimentLogger.get_minimisation`), with an optional `minimisation=`
    override.
  - **Failed evaluations** were replaced by 0 before taking the
    best-so-far, which looked like a perfect score on negative or minimised
    fitness. They are now treated as missing.
  - **Run identity:** runs of one method with several LLMs were merged
    into one curve. Runs are now identified by their folder, and labels
    read `method (llm)` when needed.
  - **Mean curves** of runs with unequal length could go down; each run's
    last best value is now carried forward.
  - **`compare_auc`** follows the direction and treats failures the same
    way.
  - **Web app crashes fixed:**
    - the convergence plot on multi-objective problems;
    - Top Solutions, box plots and `fitness_table` when a run's best
      solution failed;
    - while an experiment is still writing its logs (a half-written last
      line of a `.jsonl` file is now skipped).
  - **Web app auto-refresh** now actually refreshes every 60 s (Streamlit
    >= 1.37).
  - **Plot crashes fixed:**
    - `plot_code_evolution_graphs()` with its default features;
    - the t-SNE fallback.
  - **Other plot fixes:** figures are closed after saving, and the x axis
    of `plot_convergence(separate_lines=True)` starts at 1.
  - **Behaviour metrics:**
    - `compute_behavior_metrics` now uses `bounds` for the exploration
      percentage;
    - `average_distance_to_best_so_far` measures against the best before
      each step, as documented;
    - one-row runs no longer divide by zero.
- **LLM layer (#205).**
  - **Code extraction:** a block in another language before the python
    block (e.g. `bash`) made the prose between them the "code". It now
    takes the first python-tagged block (`python`, `py`, `python3`, also
    with CRLF line endings or a trailing space), else the first untagged
    one.
  - **Class name:** the class name was the first `class` in the code,
    often a helper such as `Particle`. It is now the top-level class that
    no other class uses; a comment like `# helper subclass:` no longer
    matches.
  - **Custom patterns:** `code_pattern`, `name_pattern`, `desc_pattern`
    and `cs_pattern` are now used; they were ignored before.
  - **Gemini:** conversations with more than one message failed, because
    the history used a format google-genai rejects. The error was then
    retried for 150 s.
  - **Retries:** OpenAI/DeepSeek and Gemini no longer retry errors that
    cannot succeed (bad request, authentication, unknown model). OpenAI
    retries keep the requested `temperature`, and a fractional
    `Retry-After` header no longer crashes.
  - **Roles:** RandomSearch and LHNS sent the role `"client"`, which OpenAI
    and Anthropic reject; it is now `"user"`.
  - `sample_solution(..., **kwargs)` passes settings such as `temperature`
    to the provider; it raised `TypeError` before.
  - Prompt tokens are counted in the conversation log; they were always 0.
  - `diff_mode=True` (LLaMEA) now warns, because responses are always
    treated as complete code.
  - Pickling an `MLX_LM_LLM` no longer removes the model from the live
    object, and unpickling an `LMStudio_LLM` works without lmstudio
    installed.
- **Benchmark scores that could be gamed or were wrong (#203).**
  - **Steiner tree:** the fitness is now `steiner_mst / normal_mst`
    (minimised). The old ratio and check were inverted, so real
    improvements were rejected and far-away points scored best.
  - **NaN/inf inputs:** they passed the validity checks of Erdős, graph
    colouring, unit square and rectangle packing, auto-correlation 1–3 and
    matrix multiplication (e.g. a NaN colouring used "1 colour"). They are
    now invalid.
  - **Graph colouring:** colours must also be ints >= 1 for every node.
  - **Toy multi-objective:** the score was whatever the candidate reported.
    The harness now provides the objective (`run(objective, budget)`),
    enforces the budget and scores the values it observed.
  - **Hexagon packing:** the outer side was a lower bound (e.g. 2.02
    instead of 2.31 for 3 hexagons). It is now exact, and the prompt
    states the outer hexagon's orientation.
  - **Min/max distance ratio:** distances are computed without
    cancellation error far from the origin.
  - **Auto-correlation 2:** the integral of (f*f)^2 is exact; an indicator
    function scores 2/3, not 1.
  - **Auto-correlation 3:** the length of f is checked.
  - **Heilbronn triangle:** the points are rescaled with the triangle.
  - **Matrix multiplication:**
    - the rank targets looked up the table with swapped dimensions (27 of
      54 problems had wrong targets); targets are now capped at the naive
      rank;
    - the class named by the solution is used, not the first class.
  - **BBOB-SBOX:** the default test instances are iids 6–15; iid 5 was in
    both training and test.
  - **MA-BBOB and photonics:** going over budget no longer skips the
    remaining instances.
  - **Photonics:**
    - the ellipsometry target is seeded (`reference_seed`, default 42); it
      was random on every evaluation, so ellipsometry scores differ from
      earlier versions;
    - the per-problem AOCC bounds are used, and runs are seeded.
  - **AutoML:**
    - a failed fold counts as the metric's worst value instead of being
      dropped;
    - binary AUC with `predict_proba` no longer fails.
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
