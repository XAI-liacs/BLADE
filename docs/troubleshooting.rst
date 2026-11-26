Troubleshooting
===============

Most problems show up in ``iohblade validate`` or ``iohblade check``, before
any LLM time is spent. Run both after every change to an experiment.

Configuration errors (exit code 2)
----------------------------------

``Unknown field 'runz'. Did you mean 'runs'?``
   A typo in the YAML file; follow the suggestion. Every issue is reported
   with file, line and column.

``Unknown parameter 'x' for problem 'p'`` (or method, suite)
   ``params`` are checked against the constructor. See the class in the
   :doc:`modules`, or run
   ``python -c "import inspect; from iohblade.registry import registry as r; print(inspect.signature(r.resolve('problem', 'p')))"``.

``Unknown problem type`` (or method, suite, provider)
   Use a key from ``iohblade list``.

A package for a method or LLM is missing
   Methods and LLMs run in BLADE's own environment: install the package
   there (``validate`` names it). EoH, ReEvo and MoEH need
   ``uv sync --group methods``.

``Cannot resume: ... holds runs of a different setup``
   The output folder holds runs that this configuration would leave out
   (fewer runs or seeds, a renamed or removed method or problem) or runs with
   another budget. Use a new ``output_dir``
   (``--set experiment.output_dir=results/trial``) or restore the previous
   settings. Adding runs, seeds, methods or problems is fine.

Evaluation environments (exit code 3)
-------------------------------------

Every problem is evaluated in its own virtual environment with its
``dependencies``. When building one fails:

- the installer output is in ``<output_dir>/env-errors/``, and
  ``iohblade check -v`` shows it in full;
- packages built from source may need tools such as ``swig`` or ``cmake``;
  list them in ``dependencies``;
- ``iohblade cache clear`` removes cached environments, in case one is
  broken.

Failing checks
--------------

``[FAIL] problem ...``
   The reference solution could not be evaluated: the problem's
   ``evaluate()`` or its environment is broken, or the reference solution
   is wrong. The message shows the error and the line.

``[FAIL] method ...``
   The method crashes on a problem with a working reference solution: a bug
   in the method.

``check --llm`` fails
   A wrong API key or model name, or a local server that is not running
   (``ollama serve``, ``vllm serve ...``).

During a run
------------

The run overview counts failed evaluations per run: ``c`` in the generated
code, ``h`` in the harness, ``t`` by timeout.

Mostly ``c`` or ``t`` errors
   The LLM's code fails or is too slow. Normal in moderation; if almost
   everything fails, make the task prompt clearer, give a working example,
   raise ``eval_timeout``, or use a stronger model.

Many ``h`` errors, or a run stops with ``HarnessError``
   The evaluation fails before the generated code is used: the problem or
   its environment is broken, not the LLM. After three harness errors in a
   row the run stops, so the budget is not wasted. See the run's
   ``error.txt`` and ``run.log``, fix the problem and run ``check`` again.

``HPOUnavailableError``
   Hyperparameter tuning needs ``smac`` in the evaluation environment. Let
   BLADE enable HPO on the problem (LLaMEA ``HPO: true``) so the environment
   installs it; ``iohblade cache clear`` if that environment is stale.

A resumed run warns that the configuration changed
   The previous record is kept as ``blade_config-<timestamp>/``. For a
   different experiment, use a new ``output_dir``.

Results do not show in the web app
   The web app lists subfolders of ``./results`` that contain a
   ``progress.json``. Start it in the folder that holds ``results/``, and
   use an ``output_dir`` such as ``results/my-experiment``.

Local models
------------

vLLM, SGLang or another OpenAI-compatible server
   ``provider: openai`` with ``params: {base_url: "http://localhost:8000/v1"}``;
   no ``api_key`` needed. With several GPUs, run one server per GPU and
   combine them with ``provider: multi``. See :doc:`llm`.

Where to look
-------------

- ``<output_dir>/experiment.log``: everything that happened, including
  warnings;
- ``<run folder>/run.log`` and ``error.txt``: one run;
- ``<run folder>/conversationlog.jsonl``: every prompt and answer;
- the :doc:`webapp`, which links to all of these.

Still stuck? Open an issue on
`GitHub <https://github.com/XAI-liacs/BLADE/issues>`_ with the command, the
configuration and the relevant log.
