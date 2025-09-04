from iohblade.experiment import Experiment
from iohblade.llm import Gemini_LLM, Ollama_LLM, OpenAI_LLM, Dummy_LLM
from iohblade.methods import LLaMEA, RandomSearch
from iohblade.loggers import TrackioExperimentLogger
from iohblade import Problem, Solution
import numpy as np
import os
import logging

# Let's first define our Dummy problem.

class MinimalProblem(Problem):
    """
    A sample Problem class with all the required ingredients to use in BLADE.

    """

    def __init__(
        self, logger=None, name="minimal example", eval_timeout=30
    ):
        super().__init__(
            logger,
            [], # training instances (not used)
            [], # testing instances (not used)
            name,
            eval_timeout,
        )

        # The next four variables are actually the default, we include them here to explain them.
        self.func_name = "__call__" # the name of the generated function that will be evaluated (defaults to __call__)
        self.init_inputs = ["budget", "dim"] # The variables that the generated class will receive upon initialization. (__init__)
        self.func_inputs = ["func"] # The variables the generated function of the class will receive. (__call__)
        self.func_outputs = ["f_opt", "x_opt"] # The expected output variables of the function. (__call__)

        # Now we have to setup the prompt parts of the task. In this case we just use placeholders
        self.task_prompt = "Write the problem description part here."
        # Next define a code example (it should be runnable! ReEVO even uses it to seed the algorithm)
        self.example_prompt = """
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
        # Next we define the output format. Just leave this the default, otherwise you also need to change the regex options to extract code and descriptions.
        self.format_prompt = """
Give the response in the format:
# Description: <short-description>
# Code: 
```python
<code>
```
"""

    # Now the most important part of a problem is the `evaluate` function. 
    # Here you should take the code from the solution and evaluate it on YOUR problem instances.
    # In the end you should give the fitness back to the solution with solution.set_scores(fitness, feedback).
    def evaluate(self, solution: Solution):
        """
        Evaluate a solution (example), this just returns a random fitness instead and waits a bit.
        """
        code = solution.code
        algorithm_name = solution.name

        # Wait a bit (max 1 sec) (so we can see the progress of the experiment nicely)
        waittime = np.random.rand()
        os.sleep(waittime)

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

    def to_dict(self):
        """
        Converts the problem to a dictionary.
        """
        return {
            "name": self.name,
        }

    def test(self): 
        # The test function can be used to run the evaluation on the test instances. This is only used for post-validation.
        pass



if __name__ == "__main__": # Because we call stuff in parallel, make sure the experiment setup is inside this if.
    llm = Dummy_LLM("dummy-model")
    budget = 10 # a test budget for 10 evaluations (normally you should use 100+)

    # Set up the LLaMEA algorithm
    mutation_prompts = [
        "Refine the strategy of the selected solution to improve it.", 
    ]
    LLaMEA_method = LLaMEA(llm, budget=budget, name="LLaMEA", mutation_prompts=mutation_prompts, n_parents=2, n_offspring=4, elitism=True)
    
    # Set up a random search baseline
    RS = RandomSearch(llm, budget=budget, name="RS")
    
    methods = [LLaMEA_method, RS] 
    # make sure the "results" directory exist.
    if not os.path.exists("results"):
        os.mkdir("results")
    logger = TrackioExperimentLogger("results/minimal-test-trackio")
    problems = [MinimalProblem()] # our dummy problem
    experiment = Experiment(methods=methods, problems=problems, runs=5, show_stdout=False, exp_logger=logger, budget=budget, n_jobs=2) #normal run using 2 parallel jobs

    experiment() #run the experiment