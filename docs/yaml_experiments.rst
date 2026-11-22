Experiments from YAML
=====================

An experiment can be described in a YAML file instead of a Python script.
The file names the LLMs, methods and problems to use, and BLADE builds and
runs the experiment:

.. code-block:: bash

    uv run iohblade validate experiment.yml   # check it, nothing is run
    uv run iohblade run experiment.yml        # run (or resume) it

Python scripts keep working as before; the YAML file is another way to
create the same :class:`~iohblade.experiment.Experiment`.

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
    or ``trackio``, plus ``params``).

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

Reproducibility
---------------

``iohblade run`` (and :meth:`ExperimentPlan.make_experiment`) records in
``<output_dir>/blade_config/``:

* ``resolved.yml``: the configuration after ``${env:...}`` substitution and
  ``--set`` overrides, with API keys, tokens and values read from the
  environment replaced by ``<redacted>``;
* ``sources/``: a copy of every custom source, and ``sources.json`` with
  their SHA-256 hashes.

Each run's ``problem.json`` / ``method.json`` / ``llm.json`` of a custom
component contains a ``custom_source`` entry with the source path, class
and hash. If you resume an experiment after changing its configuration or
custom code, the previous record is kept as ``blade_config-<timestamp>/``
and a warning is printed.

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

Secrets and environment variables
---------------------------------

``${env:NAME}`` is replaced by the environment variable ``NAME``;
``${env:NAME:-default}`` falls back to ``default``. Variables can also be
put in a ``.env`` file next to the YAML file (the environment takes
precedence). Inside ``[...]`` or ``{...}`` such values must be quoted.

Overriding values
-----------------

Any value can be overridden on the command line, which is handy for quick
tests:

.. code-block:: bash

    uv run iohblade run experiment.yml --set experiment.runs=1 --set methods.0.budget=2
    uv run iohblade run experiment.yml --dry-run    # only show the planned runs

Error messages
--------------

All problems in a file are reported at once, with the line they refer to:

.. code-block:: text

    Invalid experiment configuration (2 issues):
      - experiment.yml:4:3: experiment.runz: Unknown field 'runz'. Did you mean 'runs'?
      - experiment.yml:14:14: problems[0].params.dimz: Unknown parameter 'dimz' for problem 'bbob_sbox'. Did you mean 'dims'?

``iohblade`` exits with code 0 on success, 1 if runs failed and 2 for an
invalid configuration.

Editor support
--------------

``uv run iohblade schema -o experiment.schema.json`` writes a JSON schema
of the format. Editors with YAML language support (e.g. VS Code with the
YAML extension) use it for completion and checking when the YAML file
starts with ``# yaml-language-server: $schema=experiment.schema.json``.

Python API
----------

.. code-block:: python

    from iohblade.experiment import Experiment

    experiment = Experiment.from_yaml("experiment.yml", overrides=["experiment.runs=1"])
    experiment()

    from iohblade.config import load_plan

    plan = load_plan("experiment.yml")   # build without creating any files
    print(plan.summary())

.. automodule:: iohblade.config
   :members: load_experiment, load_plan, ExperimentPlan, ConfigError, ConfigIssue

.. automodule:: iohblade.config.schema
   :members:
