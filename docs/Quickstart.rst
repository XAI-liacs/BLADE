Quick Start
-----------

The recommended way to use BLADE is a YAML experiment file and the
``iohblade`` command. (The Python API, shown at the end, is available for
scripting.)

1. Choose an LLM

   - **Local, no API key:** install `Ollama <https://ollama.com/>`_ and pull a
     model, e.g. ``ollama pull qwen2.5-coder:14b`` (used below).
   - **Hosted:** get an API key from OpenAI, Anthropic, Google or another
     provider and set it in your environment, e.g.
     ``export OPENAI_API_KEY='your_api_key_here'``. Configuration files read it
     with ``${env:OPENAI_API_KEY}``; keys are never written in them.

2. Create an experiment

   .. code-block:: bash

      uv run iohblade init my-exp     # or: --template custom-problem / custom-method

   This writes ``my-exp/experiment.yml``. An experiment names the LLMs,
   methods and problems; built-in components are referred to by name
   (``uv run iohblade list`` shows them all):

   .. code-block:: yaml

      experiment: {output_dir: results, runs: 5, budget: 50}
      llms:
        qwen: {provider: ollama, model: "qwen2.5-coder:14b"}
        # gpt: {provider: openai, model: gpt-4o, api_key: "${env:OPENAI_API_KEY}"}
      methods:
        - {type: random_search, name: "RS-{llm}", llm: qwen}
        - {type: llamea, name: "LLaMEA-{llm}", llm: qwen, params: {n_parents: 4, n_offspring: 12, elitism: false}}
      problems:
        - type: bbob_sbox
          name: SBOX_COST
          params: {dims: [5], budget_factor: 2000, specific_group: 1}

3. Validate and smoke test

   .. code-block:: bash

      uv run iohblade validate my-exp/experiment.yml   # all mistakes, with line numbers; runs nothing
      uv run iohblade check my-exp/experiment.yml      # builds the environments, runs everything briefly without an LLM

4. Run

   .. code-block:: bash

      uv run iohblade run my-exp/experiment.yml

   The console shows every run's progress, best fitness and failed
   evaluations; results, logs and the resolved configuration are written to
   ``my-exp/results``. Running the command again resumes: finished runs are
   skipped.

5. Monitor and inspect

   .. code-block:: bash

      uv run iohblade-webapp

   The web app lists the experiments in ``./results``, with run status,
   convergence plots, the best solutions and the configuration. Read more
   :doc:`here <webapp>`.

Next: :doc:`yaml_experiments` (every option), :doc:`tutorial_custom_problem`
and :doc:`tutorial_custom_method`.

Using the Python API
--------------------

The same experiment in Python, e.g. for scripting or inside a notebook:

.. code-block:: python

   from iohblade.experiment import Experiment
   from iohblade.llm import Ollama_LLM
   from iohblade.methods import LLaMEA, RandomSearch
   from iohblade.benchmarks import BBOB_SBOX
   from iohblade.loggers import ExperimentLogger

   llm = Ollama_LLM("qwen2.5-coder:14b")
   budget = 50

   methods = [
       RandomSearch(llm, budget=budget),
       LLaMEA(llm, budget=budget, name="LLaMEA", n_parents=4, n_offspring=12, elitism=False),
   ]
   training_instances = [(f, i) for f in range(1, 25) for i in range(1, 6)]
   test_instances = [(f, i) for f in range(1, 25) for i in range(5, 16)]
   problems = [
       BBOB_SBOX(training_instances=training_instances, test_instances=test_instances,
                 dims=[5], budget_factor=2000, name="SBOX_COST")
   ]
   experiment = Experiment(methods=methods, problems=problems, runs=5,
                           exp_logger=ExperimentLogger("results/SBOX"))
   experiment()

A YAML file can also be loaded from Python:
``Experiment.from_yaml("my-exp/experiment.yml")()``.

Examples
--------

YAML experiments are in the ``examples/configs`` folder on GitHub; Python
scripts and notebooks are in ``examples``.
