Tutorial: your own problem
==========================

This tutorial turns a new optimisation problem into a BLADE benchmark and
lets LLaMEA design algorithms for it, using a YAML experiment. It takes
about ten minutes; no LLM is needed until the last step.

1. Create a starter project
---------------------------

.. code-block:: bash

    uv run iohblade init my-bench --template custom-problem

This writes:

* ``experiment.yml``: the experiment (LLMs, methods, the problem);
* ``my_problem.py``: a problem class with TODOs, a shifted sphere function
  that the generated optimisers have to minimise;
* ``reference_solution.py``: a working (random search) solution, used to
  test the problem;
* ``experiment.schema.json``: for completion in your editor.

2. Describe the task to the LLM
-------------------------------

Open ``my_problem.py``. A problem has three parts:

* the **prompts** in ``__init__``: ``task_prompt`` (what to write),
  ``example_prompt`` (an example solution, shown to the LLM as a model
  answer) and ``format_prompt`` (how to answer). ``get_prompt()`` joins them;
* ``evaluate(solution)``: runs ``solution.code`` and sets a score with
  ``solution.set_scores(fitness, feedback)``. The feedback text is shown to
  the LLM in the next iteration, so make it informative;
* ``minimisation``: whether lower fitness is better (default: higher).

Replace the objective in ``evaluate()`` by your own. Packages that only
``evaluate()`` needs should be imported *inside* it and listed under
``dependencies`` in ``experiment.yml``; they are installed in the problem's
own evaluation environment, not in BLADE's. Code at the top of the file runs
in BLADE's process, because the problem object is created there.

Keep ``reference_solution.py`` a valid, simple solution of your task; the
example in the prompt is built from it.

3. Validate
-----------

.. code-block:: bash

    uv run iohblade validate my-bench/experiment.yml

This imports ``my_problem.py`` and checks the whole configuration without
running anything. Typical messages:

.. code-block:: text

    experiment.yml:44:5: problems[0].class: No class 'MyProblm' in my_problem. Classes there: MyProblem.
    experiment.yml:45:14: problems[0].params.dimz: Unknown parameter 'dimz' for custom problem 'MyProblem'. Did you mean 'dim'?
    experiment.yml:43:5: problems[0].source: 'my_problem.py' imports 'scipy' when it is loaded, but 'scipy' is only listed as a dependency of the evaluation environment. BLADE creates the problem in its own process, so either move the import into the method that needs it (e.g. evaluate()), or also install it where BLADE runs (`uv add scipy`).

4. Smoke test
-------------

.. code-block:: bash

    uv run iohblade check my-bench/experiment.yml

``check`` builds the evaluation environment (cached afterwards), evaluates
``reference_solution.py`` exactly as a generated solution would be evaluated,
and runs each method briefly with an offline LLM that answers with the
reference solution:

.. code-block:: text

    [OK  ] problem MyProblem: evaluated smoke_test_solution reference_solution.py in 1.5s, fitness -3.84922
    [OK  ] method LLaMEA-local: ran on 'MyProblem' with budget 2

A ``FAIL`` that mentions the *harness* means the problem itself (or its
environment) is broken, not the solution: fix ``my_problem.py`` before
running. ``check --llm`` additionally sends one prompt to each LLM to test
API keys and model names.

5. Run and inspect
------------------

Choose the LLM in ``experiment.yml`` (API keys via ``${env:NAME}`` or a
``.env`` file next to it), then:

.. code-block:: bash

    uv run iohblade run my-bench/experiment.yml
    uv run iohblade-webapp      # browse the results

Results go to ``my-bench/results``: per run the generated solutions
(``log.jsonl``), the conversation with the LLM, ``run.log`` and, for failed
runs, ``error.txt``; for the experiment ``experiment.log``,
``progress.json`` and ``blade_config/`` (the resolved configuration and a
copy of ``my_problem.py``). Running the same command again resumes: finished
runs are skipped.

Next steps
----------

* Turn the folder into a package (``my_bench/__init__.py`` plus modules and
  data files) and point ``source`` at the folder; use ``class:
  "module:MyProblem"`` for a class in a submodule.
* Add built-in problems next to yours to compare (``uv run iohblade list
  problems``).
* See :doc:`yaml_experiments` for every option.
