Experiments from YAML
=====================

The recommended way to define a BLADE experiment is a YAML file. It names
the LLMs, methods and problems to use, and BLADE builds and runs the
experiment:

.. code-block:: bash

    uv run iohblade init my-exp -t custom-problem   # optional: a starter project
    uv run iohblade validate my-exp/experiment.yml  # check it, nothing is run
    uv run iohblade check my-exp/experiment.yml     # smoke test, no LLM calls
    uv run iohblade run my-exp/experiment.yml       # run (or resume) it

The YAML file creates the same :class:`~iohblade.experiment.Experiment` as
the Python API, which remains available for scripting
(``Experiment.from_yaml(path)`` loads a file from Python).

Example
-------

.. code-block:: yaml

    # yaml-language-server: $schema=experiment.schema.json
    version: 1

    experiment:
      output_dir: results/hexagons   # relative to this file; reused to resume
      runs: 3                        # or: seeds: [0, 1, 2]
      budget: 10                     # evaluations per run (and method default)
      n_jobs: 2

    llms:                            # named LLMs, referred to by methods
      gemma:
        provider: ollama
        model: "gemma3:12b"
      gemini:
        provider: gemini
        model: gemini-2.0-flash
        api_key: ${env:GOOGLE_API_KEY}

    methods:
      - type: llamea
        name: "LLaMEA-{llm}"         # {llm} is replaced by the LLM name
        llm: [gemma, gemini]         # one method per LLM
        params: {n_parents: 1, n_offspring: 1}
      - type: random_search
        llm: gemma
        budget: 5                    # optional, defaults to experiment.budget

    problems:
      - type: circle_packing
        name: circles
      - suite: hexagon_packing       # a suite expands into several problems
        params: {get_best: false}

More examples are in ``examples/configs/``.

Sections
--------

``experiment``
    ``output_dir`` (required), ``name``, ``runs`` (default 5), ``seeds``,
    ``budget`` (default 100), ``n_jobs`` (default 1), ``show_stdout``,
    ``log_stdout`` and ``logger`` (``type``: ``file`` (default), ``mlflow``
    or ``trackio``, plus ``params``). With ``n_jobs`` above 1 the runs are
    threads that share the global random number generators, so the results
    of individual seeds are not exactly reproducible; use ``n_jobs: 1`` for
    reproducible per-seed results.

``llms``
    A mapping from a name of your choice to ``provider``, ``model``,
    ``api_key`` and extra constructor ``params``. Provider ``multi`` takes
    ``llms``: a list of other LLM names.

``methods``
    A list of ``type``, ``llm`` (a name or a list of names), optional
    ``name`` and ``budget``, and ``params`` for the method's constructor.

``problems``
    A list of either ``type`` (one problem) with optional ``name``, or
    ``suite`` (a factory returning several problems), each with ``params``
    for the constructor or factory.

``iohblade list`` prints every available ``type``, ``suite``, ``provider``
and logger with a short description. ``params`` are checked against the
constructor of the component, so misspelled or missing parameters are
reported before anything runs. Strings are converted to enum values where a
parameter expects an enum (e.g. ``problem_type: BBOB``).

Your own problems, methods and LLMs
-----------------------------------

Use ``type: custom`` (``provider: custom`` for an LLM) with a ``source``
(a ``.py`` file or a package folder, relative to the YAML file) and a
``class``:

.. code-block:: yaml

    problems:
      - type: custom
        source: custom/sphere_benchmark.py   # or a package folder: my_bench/
        class: SphereProblem                 # "submodule:Class" inside a package
        params: {dim: 5}
        dependencies: [scipy>=1.11]          # pip packages for the evaluation
    methods:
      - type: custom
        source: my_method.py
        class: MyMethod                      # a subclass of iohblade.method.Method
        llm: qwen
    llms:
      local:
        provider: custom
        source: my_llm.py
        class: MyLLM                         # a subclass of iohblade.llm.LLM

How it works:

* The file is imported under its own name (``sphere_benchmark``), a package
  under its folder name, with the parent folder added to ``sys.path``. Rename
  the file if that name clashes with an installed module; ``validate`` tells
  you when it does.
* The class must subclass :class:`~iohblade.problem.Problem`,
  :class:`~iohblade.method.Method` or :class:`~iohblade.llm.LLM` and
  implement all their abstract methods; ``params`` are checked against its
  constructor like for built-in components.
* Problems are evaluated in their own virtual environment, like built-in
  ones. The folder of the source is added to that environment's
  ``PYTHONPATH`` (``Problem.extra_pythonpath``), so relative imports and
  data files next to the code work. Packages the problem needs during
  evaluation go in ``dependencies``, or in a ``requirements.txt`` next to
  the file / in the package, or in the package's ``pyproject.toml``.
  The problem object itself is created in BLADE's process, so its module
  must import there: import evaluation-only packages inside ``evaluate()``
  (as the built-in benchmarks do), or install them in BLADE's environment
  too. ``validate`` says which applies.
* Methods and LLMs run in BLADE's own environment; install what they import
  there (``validate`` names the missing package).

Starting a new experiment
-------------------------

``iohblade init <folder> --template <name>`` writes an ``experiment.yml``,
the JSON schema for editor completion and, depending on the template, stub
code with TODOs:

* ``builtin``: LLaMEA and random search on a built-in problem;
* ``custom-problem``: your own problem class (``my_problem.py``) with a
  reference solution;
* ``custom-method``: your own search method (``my_method.py``) next to a
  random-search baseline.

Every template passes ``iohblade check`` as generated.

.. _checking-an-experiment:

Checking an experiment
----------------------

``iohblade check experiment.yml`` tries every part of an experiment before
you spend LLM time on it:

* **problems**: the evaluation environment is built (or taken from the
  cache) and a known-good *reference solution* is evaluated through the real
  pipeline. The reference is the problem's ``smoke_test_solution`` (a ``.py``
  file next to the YAML file) or else the code block in its example prompt.
  Errors in the harness or in the reference fail the check; non-finite
  scores or evaluations taking over half of ``eval_timeout`` give a warning.
* **methods**: each method runs with a budget of 2 and an offline LLM that
  answers with the reference solution.
* **LLMs**, only with ``--llm``: each LLM gets one short prompt, which tests
  API keys, model names and connectivity.

.. code-block:: text

    [OK  ] problem circle_packing_n40: evaluated smoke_test_solution reference_solution.py in 1.8s, fitness 579.05
    [OK  ] method LLaMEA-local: ran on 'circle_packing_n40' with budget 2
    [FAIL] problem bbob: building the evaluation environment failed: Failed to install the dependencies [...]
        error: command 'swig' failed: No such file or directory

The exit code is 1 if a check failed. ``-v`` shows the full output of a
failed package installation. ``iohblade run --dry-run`` lists the planned
runs and, for an existing output directory, how many are already finished
(and whether resuming is possible).

Evaluation environments
-----------------------

Problems evaluate solutions in a virtual environment with their
``dependencies``. These environments are cached and shared between problems
with the same dependencies, across runs and experiments, so only the first
run pays for the installation. Packages are installed with ``uv`` when it
is available, otherwise with ``pip``.

* ``iohblade cache list`` shows the cached environments and their size;
  ``iohblade cache clear`` removes them.
* The cache lives in ``~/.cache/iohblade/envs`` (on Windows
  ``%LOCALAPPDATA%\iohblade\Cache\envs``); set ``IOHBLADE_CACHE_DIR`` to
  move it.
* ``experiment.env_cache: false`` (or ``IOHBLADE_ENV_CACHE=0``, or
  ``problem.env_cache = False`` in Python) gives each problem a temporary
  environment that is deleted afterwards, as before. Dependencies that are
  local paths are never cached.
* ``experiment.isolated_pythonpath: true`` makes only ``iohblade`` itself
  (plus the problem's dependencies and custom code) importable during
  evaluation. By default the folder containing ``iohblade`` is on the path,
  which for a pip installation is the whole ``site-packages`` of BLADE's
  environment, so undeclared packages may be importable by accident.

Logs and error reports
----------------------

While an experiment runs, the console shows an overview of every run:

.. code-block:: text

    Method        LLM    Problem    Seed  Evals  Best    Errors
    ------------------------------------------------------------
    LLaMEA-local  qwen   MyProblem  0     20/20  -3.12   4c      ✅
    LLaMEA-local  qwen   MyProblem  1     12/20  -4.80   2c 1t   🔄

``Errors`` counts the evaluations that failed in the generated code (``c``),
in the evaluation harness (``h``, a problem with the benchmark rather than the
LLM's code) or by timeout (``t``). At the end, a summary lists the failed runs
grouped by exception type with the path to each ``error.txt``, and the error
totals.

Everything that is logged, by BLADE and by libraries (including Python
warnings), goes to ``<output_dir>/experiment.log``. Each run's folder also gets
a ``run.log`` with the messages of that run. ``experiment.log_level`` (DEBUG,
INFO, WARNING, ERROR; default INFO) sets how much is written. With
``show_stdout: false`` (the default), log output is kept out of the terminal.

If an evaluation environment cannot be built, no run starts. Every failing
problem is listed, and the installer output is saved in
``<output_dir>/env-errors/``. ``iohblade run`` exits with code 0 on success,
1 if runs failed, 2 for an invalid configuration and 3 if an environment
could not be built.

The web app (``uv run iohblade-webapp``) shows the same run table, the
resolved configuration, the custom code and the end of ``experiment.log``.

.. _yaml-reproducibility:

Reproducibility
---------------

``iohblade run`` (and :meth:`ExperimentPlan.make_experiment`) records in
``<output_dir>/blade_config/``:

* ``resolved.yml``: the configuration after ``${env:...}`` substitution and
  ``--set`` overrides, with API keys, tokens and values read from the
  environment replaced by ``<redacted>``;
* ``sources/``: a copy of every custom source, and ``sources.json`` with
  their SHA-256 hashes and paths (relative to the configuration file, so the
  experiment folder can be moved).

Each run's ``problem.json`` / ``method.json`` / ``llm.json`` of a custom
component contains a ``custom_source`` entry with the source path, class
and hash. If you resume an experiment after changing its configuration or
custom code, the previous record is kept as ``blade_config-<timestamp>/``
and a warning is printed.

Resuming identifies runs by method name, problem name and seed. An existing
``output_dir`` can be resumed with the same runs, or with more of them
(higher ``runs``, more ``seeds``, another method or problem): finished runs
are skipped and the new ones start. If the configuration leaves out runs
that are recorded there (fewer ``runs`` or ``seeds``, a renamed or removed
method or problem), ``iohblade run`` stops with exit code 2 before writing
anything, and says which runs differ; use a new ``output_dir`` or restore
the previous settings. ``iohblade run --dry-run`` shows what resuming would
do.

LLaMEA operators
----------------

LLaMEA's mutation and crossover prompts are ``llamea.Operator`` objects. In
YAML each operator is a mapping with the fields of ``llamea.Operator``:

.. code-block:: yaml

    methods:
      - type: llamea
        llm: gemma
        params:
          n_parents: 2
          n_offspring: 2
          operators:
            - name: refine                 # identifies the operator in the logs
              prompt: "Refine the strategy of the selected solution to improve it."
              weight: 0.8                  # relative probability of being picked
              number_of_parents: 1
            - name: crossover
              prompt: "Combine the two selected solutions into a better one."
              weight: 0.2
              number_of_parents: 2         # at most n_parents

``prompt`` is required; ``name`` (default ``mutation``), ``weight`` (default
0.5) and ``number_of_parents`` (default 1) are optional. BLADE creates the
``Operator`` objects and checks the fields, the weights and that no operator
needs more parents than ``n_parents``. In Python:

.. code-block:: python

    from llamea import Operator
    from iohblade.methods import LLaMEA

    operators = [
        Operator(prompt="Refine the strategy of the selected solution to improve it.",
                 name="refine", weight=0.8, number_of_parents=1),
        Operator(prompt="Combine the two selected solutions into a better one.",
                 name="crossover", weight=0.2, number_of_parents=2),
    ]
    method = LLaMEA(llm, budget=100, operators=operators, n_parents=2, n_offspring=2)

Hyperparameter tuning (HPO)
---------------------------

LLaMEA can ask the LLM for a configuration space (a ``ConfigSpace``
definition of the solution's hyperparameters) next to the code. A problem
that supports tuning then tunes each solution's hyperparameters with
`SMAC <https://github.com/automl/SMAC3>`_ before scoring it with the best
configuration found:

.. code-block:: yaml

    methods:
      - type: llamea
        llm: gemma
        params:
          HPO: true

* Supported problems: ``automl``, ``bbob_sbox``, ``ma_bbob`` and
  ``kerneltuner``. Each SMAC trial runs the solution, with the configuration
  passed as constructor keyword arguments (for ``kerneltuner``: as the
  strategy's hyperparameters), on one instance of a fixed, seeded subset of
  the training instances, and returns a cost to minimise: ``1 - score``
  (``1 - AOCC`` for ``bbob_sbox`` and ``ma_bbob``). The best configuration
  (the incumbent) is then scored like any solution, stored in the
  solution's metadata (``incumbent``) and named in the feedback; scoring on
  the test instances (``test()``) reuses it and does not tune again (a
  solution without a recorded incumbent is scored with its defaults).

  ============== ============================ =========================
  Problem        Trials per solution          Tuning instances
  ============== ============================ =========================
  automl         100                          up to 12 search splits
  bbob_sbox      ``hpo_trials: 100``          ``hpo_instances: 12``
  ma_bbob        ``hpo_trials: 50``           ``hpo_instances: 12``
  kerneltuner    ``hpo_trials: 50``           ``hpo_instances: 12``
  ============== ============================ =========================

  ``hpo_instances: null`` tunes on all training instances. Tuning makes
  every evaluation more expensive: a trial is one run on one instance, so
  for the default ``bbob_sbox`` (240 runs per evaluation) tuning adds about
  0.4 of an evaluation, for ``ma_bbob`` (40 runs) about 1.25 and for
  ``kerneltuner`` (24 kernel-GPU pairs) about 2, plus SMAC's own overhead.
  Raise ``eval_timeout`` accordingly.
* HPO is enabled automatically on every supported problem of an experiment
  with a method that samples with ``HPO: true`` (also from Python, by
  :class:`~iohblade.experiment.Experiment`). ``iohblade validate`` prints a
  note for each problem with HPO enabled and a warning for each problem that
  cannot tune. The other methods of the experiment then use the same
  evaluation environment for that problem. HPO can also be enabled directly:
  ``params: {hpo: true}`` on the problem, or ``problem.enable_hpo()`` in
  Python.
* With HPO the problem's evaluation environment also installs
  ``smac>=2.4.1,<2.5`` and ``ConfigSpace>=1.2``
  (:data:`iohblade.hpo.HPO_DEPENDENCIES`); problems without HPO do not
  install smac. (smac 2.4.0 does not import with scikit-learn 1.9; 2.4.1
  does.)
  BLADE's own environment does not need smac (tuning runs in the
  evaluation environments); install it there (``uv sync --group hpo``, or
  ``pip install "smac>=2.4.1,<2.5"``) only to call
  :func:`iohblade.hpo.run_smac` yourself.
* If smac cannot be imported during an evaluation with HPO, the evaluation
  fails with ``HPOUnavailableError`` as a harness error (a problem of the
  environment, not of the generated code), and the run stops after
  ``max_harness_errors`` of them. A solution with a configuration space
  evaluated without HPO uses its default hyperparameters; its feedback says
  so and a warning is logged.

Secrets and environment variables
---------------------------------

``${env:NAME}`` is replaced by the environment variable ``NAME``;
``${env:NAME:-default}`` falls back to ``default``. Variables can also be
put in a ``.env`` file next to the YAML file (the environment takes
precedence). Inside ``[...]`` or ``{...}`` such values must be quoted.

A value that is exactly one placeholder, written without quotes, gets the
type of its text, as if it had been written in the file:
``n_parents: ${env:NP:-4}`` is the integer 4, ``elitism: ${env:ELITISM}``
with ``ELITISM=false`` is ``False``. Everything else stays a string: quoted
values (``api_key: "${env:OPENAI_API_KEY}"``), placeholders inside a longer
string, values of credential keys (``api_key``, ``token``, ``*_token``,
``*_secret``, ``*_password``) and configurations passed to
:func:`~iohblade.config.load_plan` as a dict. Quote API keys, so a key that
happens to look like a number stays text.

The ``.env`` file follows the usual (python-dotenv) rules:

.. code-block:: bash

    # comment lines and blank lines are skipped
    export OPENAI_API_KEY=sk-abc     # "export " is optional; " #" starts a comment
    GOOGLE_API_KEY="xyz # not a comment"   # quoted values end at the closing quote
    PROMPT="line one\nline two"     # double quotes support \n and \" escapes
    RAW='no $escapes\n here'         # single quotes are literal

.. _yaml-overriding-values:

Overriding values
-----------------

Any value can be overridden on the command line, which is handy for quick
tests:

.. code-block:: bash

    uv run iohblade run experiment.yml --set experiment.runs=1 --set methods.0.budget=2 \
        --set experiment.output_dir=results/quick-test
    uv run iohblade run experiment.yml --dry-run    # only show the planned runs

Give a quick test its own ``output_dir``: an existing results folder can only
be resumed with the same or more runs (see `Reproducibility`_).
``${env:...}`` works in ``--set`` values too, e.g.
``--set methods.0.params.n_parents=${env:NP}`` (quote the value,
``--set 'llms.x.api_key="${env:KEY}"'``, to keep it a string).

Error messages
--------------

All problems in a file are reported at once, with the line they refer to:

.. code-block:: text

    Invalid experiment configuration (2 issues):
      - experiment.yml:4:3: experiment.runz: Unknown field 'runz'. Did you mean 'runs'?
      - experiment.yml:14:14: problems[0].params.dimz: Unknown parameter 'dimz' for problem 'bbob_sbox'. Did you mean 'dims'?

``iohblade`` exits with code 0 on success, 1 if runs failed and 2 for an
invalid configuration.

.. _yaml-coding-agents:

Coding agents
-------------

``iohblade skill [folder]`` installs a skill (``SKILL.md`` plus a
reference) that teaches coding agents to write, validate, check and run
these files; the default folder ``.claude/skills`` is where Claude Code
looks. The BLADE repository has it in ``.claude/skills/blade/``.
See :doc:`coding_agents`.

.. _editor-support:

Editor support
--------------

Editors with YAML language support (e.g. VS Code with the YAML extension) give
completion, descriptions and checks for the configuration when its first line
points at the JSON schema of the format:

.. code-block:: yaml

    # yaml-language-server: $schema=https://xai-liacs.github.io/BLADE/experiment.schema.json

The schema is published with the documentation (the version of the latest
release; ``.../BLADE/main/experiment.schema.json`` follows the development
version). ``uv run iohblade schema -o experiment.schema.json`` writes it for
your installed version, which ``iohblade init`` also does next to the
generated ``experiment.yml``.

Python API
----------

.. code-block:: python

    from iohblade.experiment import Experiment

    experiment = Experiment.from_yaml(
        "experiment.yml",
        overrides=["experiment.runs=1", "experiment.output_dir=results/quick-test"],
    )
    experiment()

    from iohblade.config import load_plan

    plan = load_plan("experiment.yml")   # build without creating any files
    print(plan.summary())

.. automodule:: iohblade.config
   :members: load_experiment, load_plan, ExperimentPlan, ConfigError, ConfigIssue

.. automodule:: iohblade.config.schema
   :members:
