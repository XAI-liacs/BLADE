# Changelog

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
