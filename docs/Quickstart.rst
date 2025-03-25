Quick Start
-----------

1. Set up an OpenAI API key:

   - Obtain an API key from `OpenAI <https://openai.com/>`_.
   - Set the API key in your environment variables:

   .. code-block:: bash

      export OPENAI_API_KEY='your_api_key_here'

2. Running an Experiment

   .. code-block:: python

      from iohblade import Experiment
      from iohblade import Ollama_LLM
      from iohblade.methods import LLaMEA, RandomSearch
      from iohblade.problems import BBOB_SBOX
      from iohblade.loggers import ExperimentLogger
      import os

      llm = Ollama_LLM("qwen2.5-coder:14b") #qwen2.5-coder:14b, deepseek-coder-v2:16b
      budget = 50 #short budget for testing

      RS = RandomSearch(llm, budget=budget) #Random Search baseline
      LLaMEA_method = LLaMEA(llm, budget=budget, name="LLaMEA", n_parents=4, n_offspring=12, elitism=False) #LLamEA with 4,12 strategy
      methods = [RS, LLaMEA_method]
      
      problems = []
      # include all SBOX_COST functions with 5 instances for training and 10 for final validation as the benchmark problem.
      training_instances = [(f, i) for f in range(1,25) for i in range(1, 6)]
      test_instances = [(f, i) for f in range(1,25) for i in range(5, 16)]
      problems.append(BBOB_SBOX(training_instances=training_instances, test_instances=test_instances, dims=[5], budget_factor=2000, name=f"SBOX_COST"))
      # Set up the experiment object with 5 independent runs per method/problem. (in this case 1 problem)
      logger = ExperimentLogger("results/SBOX")
      experiment = Experiment(methods=methods, problems=problems, llm=llm, runs=5, show_stdout=True, exp_logger=logger) #normal run
      experiment() #run the experiment, all data is logged in the folder results/SBOX/

Examples
--------

Additional examples can be founnd in our `examples` folder on Github.
