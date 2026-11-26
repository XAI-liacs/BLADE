Command line
============

Installing BLADE adds two commands: ``iohblade`` for experiments and
``iohblade-webapp`` for browsing results. In a uv project, prefix them with
``uv run``.

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - Command
     - What it does
   * - ``iohblade init [DIR]``
     - Create a starter experiment in ``DIR`` (default ``.``).
   * - ``iohblade validate FILE``
     - Check a configuration: every mistake, with its line. Runs nothing.
   * - ``iohblade check FILE``
     - Smoke test: build the evaluation environments, evaluate a reference
       solution per problem, run each method briefly with an offline LLM.
   * - ``iohblade run FILE``
     - Run the experiment, or resume it.
   * - ``iohblade list [KIND]``
     - List the built-in problems, suites, methods, LLM providers and
       loggers.
   * - ``iohblade schema``
     - Print the JSON schema of the YAML format.
   * - ``iohblade skill [DIR]``
     - Install the BLADE skill for coding agents.
   * - ``iohblade cache list|clear``
     - Show or delete the cached evaluation environments.
   * - ``iohblade-webapp``
     - Browse the experiments in ``./results`` (see :doc:`webapp`).

Exit codes
----------

=====  ===============================================================
Code   Meaning
=====  ===============================================================
``0``  Success.
``1``  Runs (or checks) failed; the other runs were recorded.
``2``  Invalid configuration, or an output folder that cannot be resumed.
``3``  An evaluation environment could not be built.
=====  ===============================================================

Options shared by validate, check and run
-----------------------------------------

``--set KEY=VALUE``
   Override a value of the configuration without editing the file, e.g.
   ``--set experiment.runs=1`` or ``--set methods.0.budget=20``. Repeatable.
   ``${env:...}`` works in values. See :doc:`yaml_experiments`.

``-v``, ``--verbose``
   Show output printed while the components are created.

iohblade init
-------------

.. code-block:: text

   iohblade init [--template {builtin,custom-problem,custom-method}] [--force] [DIR]

``--template``, ``-t``
   ``builtin`` (default): built-in methods and problem. ``custom-problem``:
   adds ``my_problem.py`` for your own benchmark. ``custom-method``: adds
   ``my_method.py`` for your own search method.

``--force``
   Overwrite existing files.

Every template also writes ``reference_solution.py`` (used by ``check``) and
``experiment.schema.json`` (editor completion).

iohblade validate
-----------------

.. code-block:: text

   iohblade validate [--set KEY=VALUE] [-v] FILE

Prints the planned methods, problems, seeds and number of runs, or every
issue with file, line and column. Nothing is run and no LLM is contacted.

iohblade check
--------------

.. code-block:: text

   iohblade check [--set KEY=VALUE] [-v] [--llm] [--no-methods] FILE

``--llm``
   Also send one short prompt to every configured LLM, to test keys and
   model names.

``--no-methods``
   Only check the problems.

Each line reports ``OK``, ``WARN``, ``FAIL`` or ``SKIP`` for a problem,
method or LLM. See :ref:`checking-an-experiment` for what is checked.

iohblade run
------------

.. code-block:: text

   iohblade run [--set KEY=VALUE] [-v] [--dry-run] FILE

``--dry-run``
   Only show the planned runs and what resuming would do.

Running the same command again resumes: finished runs are skipped. A
folder that holds runs of a different setup is not resumed (exit code 2).

iohblade list
-------------

.. code-block:: text

   iohblade list [--check] [{llms,loggers,methods,problems,suites}]

``--check``
   Also check whether each component can be imported (optional
   dependencies).

iohblade schema
---------------

.. code-block:: text

   iohblade schema [-o OUTPUT]

The schema is also published at
https://xai-liacs.github.io/BLADE/experiment.schema.json; see
:ref:`editor-support`.

iohblade skill
--------------

.. code-block:: text

   iohblade skill [--force] [DIR]

Writes the skill to ``DIR/blade`` (default ``.claude/skills``, where Claude
Code looks). See :doc:`coding_agents`.

iohblade cache
--------------

.. code-block:: text

   iohblade cache {list,clear}

Evaluation environments are cached and shared between problems with the
same dependencies, in ``$IOHBLADE_CACHE_DIR/envs`` or the user cache folder.

Environment variables
---------------------

``IOHBLADE_CACHE_DIR``
   Where the evaluation environments are cached.

``IOHBLADE_ENV_CACHE``
   ``0`` turns the cache off: every problem gets a temporary environment.

``IOHBLADE_USE_UV``
   ``0`` installs evaluation environments with pip instead of ``uv pip``.

API keys (``OPENAI_API_KEY``, ``ANTHROPIC_API_KEY``, ...) are read where the
configuration refers to them with ``${env:...}``; a ``.env`` file next to
the experiment works too.
