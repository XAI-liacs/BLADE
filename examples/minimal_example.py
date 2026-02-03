from iohblade.experiment import Experiment
from iohblade.llm import Gemini_LLM, Ollama_LLM, OpenAI_LLM, Dummy_LLM
from iohblade.methods import LLaMEA, RandomSearch
from iohblade.loggers import TrackioExperimentLogger, ExperimentLogger
from iohblade import Problem, Solution, wrap_problem
from iohblade.problems import AutoML
import numpy as np
import os
import logging

# Let's first define our Dummy problem.

# First we define our evaluate function (how to evaluate a solution)
# It can also access the problem object, for example to get training and testing instance information.
def f(problem:Problem, solution: Solution):
    """
    Evaluate a solution (example), this just returns a random fitness instead and waits a bit.
    """
    code = solution.code
    algorithm_name = solution.name

    # # Wait a bit (max 1 sec) (so we can see the progress of the experiment nicely)
    # waittime = np.random.rand()
    # sleep(waittime)

    # Excecute the code. (errors are handled automatically and would set the fitness to -inf)
    exec(code, globals())

    # Instantiate the generated class.
    algorithm = None
    algorithm = globals()[algorithm_name](budget=5, dim=5)
    # Now we can also call the algorithm, but for this example we omit that.
    # res = algorithm()

    score = np.random.rand()
    # we pass the score and a textual feedback to the solution.
    solution.set_scores(
        score,
        f"The algorithm {algorithm_name} scored {score:.3f} (higher is better, 1.0 is the best).",
    )
    # we finally return the updated solution object.
    return solution

task_prompt = "Write the problem description part here."
example_prompt = """
An example code is as follows:
```python
import numpy as np

class RandomSearch:
    def __init__(self, budget=10000, dim=10):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        self.f_opt = np.inf
        self.x_opt = None
        for i in range(self.budget):
            x = np.random.uniform(func.bounds.lb, func.bounds.ub)
            
            f = func(x)
            if f < self.f_opt:
                self.f_opt = f
                self.x_opt = x
            
        return self.f_opt, self.x_opt
```
"""

minimal_problem = wrap_problem(f, 
                            eval_timeout=30, 
                            training_instances=[], #not used
                            test_instances = [], #not used
                            dependencies = None, #Default dependencies
                            imports = "import numpy as np", # Load numpy for evaluation
                            task_prompt = task_prompt,
                            example_prompt = example_prompt,
                            )

if __name__ == "__main__": # Because we call stuff in parallel, make sure the experiment setup is inside this if.

    llm = Dummy_LLM("dummy-model")
    budget = 4 # a test budget for 4 evaluations (normally you should use 100+)

    # Set up the LLaMEA algorithm
    mutation_prompts = [
        "Refine the strategy of the selected solution to improve it.", 
    ]
    LLaMEA_method = LLaMEA(llm, budget=budget, name="LLaMEA", mutation_prompts=mutation_prompts, n_parents=1, n_offspring=1, elitism=True)
    
    # Set up a random search baseline
    RS = RandomSearch(llm, budget=budget, name="RS")
    
    methods = [LLaMEA_method, RS] 
    # make sure the "results" directory exist.
    if not os.path.exists("results"):
        os.mkdir("results")

    logger = ExperimentLogger("results/minimal-test")
    problems = [minimal_problem] # our dummy problem
    experiment = Experiment(methods=methods, problems=problems, runs=2, show_stdout=True, exp_logger=logger, budget=budget, n_jobs=2) #normal run using 2 parallel jobs

    experiment() #run the experiment