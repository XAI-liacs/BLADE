Tutorial: your own search method
================================

This tutorial adds a new LLM-driven search method and compares it with
random search on a built-in benchmark, using a YAML experiment.

1. Create a starter project
---------------------------

.. code-block:: bash

    uv run iohblade init my-method --template custom-method

This writes ``experiment.yml``, ``my_method.py`` (a (1+1) strategy that keeps
asking the LLM to improve the best solution so far) and
``reference_solution.py`` (a valid circle-packing solution used by
``iohblade check``).

2. Write the method
-------------------

A method subclasses :class:`iohblade.method.Method`:

.. code-block:: python

    from iohblade.method import Method

    class MyMethod(Method):
        def __init__(self, llm, budget, name="MyMethod", improve_prompt=None):
            super().__init__(llm, budget, name)
            ...

        def __call__(self, problem):
            prompt = [{"role": "user", "content": problem.get_prompt()}]
            best = problem(self.llm.sample_solution(prompt))
            ...
            return best

        def to_dict(self): ...
        def get_config(self): ...

The essentials:

* ``self.llm.sample_solution(messages)`` asks the LLM and returns a
  :class:`~iohblade.solution.Solution` with the extracted code;
* ``problem(solution)`` evaluates it (in an isolated subprocess) and returns
  it with ``fitness``, ``feedback`` and ``error`` filled in. Each call uses
  one evaluation of the budget;
* ``problem.minimisation`` tells whether lower fitness is better;
* return the best solution.

Constructor arguments other than ``llm``, ``budget`` and ``name`` are set
under ``params`` in ``experiment.yml`` and checked against the signature.
The method runs in BLADE's own environment, so packages it imports must be
installed there (``validate`` names any that are missing).

3. Validate, check and run
--------------------------

.. code-block:: bash

    uv run iohblade validate my-method/experiment.yml
    uv run iohblade check my-method/experiment.yml
    uv run iohblade run my-method/experiment.yml

``check`` runs your method with a budget of 2 and an offline LLM, which finds
most mistakes (wrong return value, exceptions, unused budget) in seconds.

To compare with LLM-specific configurations, list several LLMs for the
method and use ``{llm}`` in its name:

.. code-block:: yaml

    methods:
      - type: custom
        source: my_method.py
        class: MyMethod
        name: "MyMethod-{llm}"
        llm: [local, gemini]
      - type: llamea
        name: "LLaMEA-{llm}"
        llm: [local, gemini]

See :doc:`yaml_experiments` for every option, and
``uv run iohblade list methods`` for the built-in methods.
