import os
import traceback

import ioh
import numpy as np
import pandas as pd
from ioh import get_problem
from ioh import logger as ioh_logger

from ..problem import BASE_DEPENDENCIES, Problem
from ..solution import Solution
from ..utils import OverBudgetException, aoc_logger, correct_aoc
from .photonics_instances import (
    algorithmic_insights,
    get_photonic_instance,
    problem_descriptions,
)


class Photonics(Problem):
    """
    Problem class for evaluating optimization algorithms on different Real-World Photnoics applications.
    """

    def __init__(
        self,
        logger,
        problem_type="bragg",
        test_instances=None,
        name="Bragg",
        eval_timeout=3600,
        budget_factor=500,
        seeds=5,
        dependencies=None,
        imports=None,
    ):
        """
        Initializes the MA-BBOB problem instance.
        Args:
            logger (RunLogger): The logger to use for logging.
            problem_type (str): The name of the problem instance, can be one of: "bragg", "ellipsometry" or "photovoltaic".
            name (str): The name of the problem.
            eval_timeout (int): The evaluation timeout in seconds.
            budget_factor (int): The factor to multiply the dimensionality with to get the budget.
            seeds (int): Number of random runs.
        """
        if dependencies is None:
            dependencies = [
                "ioh==0.3.22",
                "pandas==2.2.3",
                "pymoosh==3.2",
                "pyGDM2==1.1.12",
            ]
        if imports is None:
            imports = "import numpy as np\nimport ioh\n"

        if problem_type not in ["bragg", "ellipsometry", "photovoltaic"]:
            raise Exception(
                "problem_type should be either 'bragg', 'ellipsometry' or 'photovoltaic'."
            )

        self.problem_type = problem_type
        self.problem = get_photonic_instance(self.problem_type)
        self.func_name = "__call__"
        self.init_inputs = ["budget", "dim"]
        self.func_inputs = ["func"]
        self.func_outputs = ["f_opt", "x_opt"]

        super().__init__(
            logger, [self.problem], [self.problem], name, eval_timeout, dependencies
        )
        self.budget_factor = budget_factor  # The factor to multiply the dimensionality with to get the budget
        self.description_prompt = problem_descriptions[self.problem_type]
        self.extra_prompt = algorithmic_insights[self.problem_type]
        self.seeds = list(range(seeds))
        self.task_prompt = """
You are a Python developer and AI and physics researcher.
Your task is to develop a novel heuristic optimization algorithm for photonic optimization problems.
The code should contain an `__init__(self, budget, dim)` function with optional additional arguments and the function `def __call__(self, func)`, which should optimize the black box function `func` using `self.budget` function evaluations.
The func() can only be called as many times as the budget allows, not more. 
"""
        self.example_prompt = """
An example of such code (a simple random search), is as follows:
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
        self.format_prompt = """
Give an excellent and novel heuristic algorithm to solve this task and also give it a one-line description, describing the main idea. Give the response in the format:
# Description: <short-description>
# Code: 
```python
<code>
```
"""

    def get_prompt(self):
        """
        Returns the problem description and answer format.
        """
        return (
            self.task_prompt
            + self.description_prompt
            + self.example_prompt
            + self.extra_prompt
            + self.format_prompt
        )

    def evaluate(self, solution: Solution, test=False, ioh_dir=""):
        """
        Evaluates a solution on the different problems.
        """
        if self.problem_type == "bragg":
            auc_lower = 0.1648
            auc_upper = 1.0
        elif self.problem_type == "ellipsometry":
            auc_lower = 1e-8
            auc_upper = 40.0
        elif self.problem_type == "photovoltaic":
            auc_lower = 0.1
            auc_upper = 1.0
        auc_mean = 0
        auc_std = 0

        dim = self.problem.meta_data.n_variables
        self.dims = [dim]

        budget = dim * self.budget_factor

        code = solution.code
        algorithm_name = solution.name
        safe_globals = {"np": np}
        local_env = {}
        exec(code, safe_globals, local_env)

        algorithm = None

        # Small test run to catch code errors
        try:
            l2_temp = aoc_logger(100, upper=1e2, triggers=[ioh_logger.trigger.ALWAYS])
            problem = get_problem(11, 1, 2)
            problem.attach_logger(l2_temp)
            algorithm = local_env[algorithm_name](budget=100, dim=2)
            algorithm(problem)
        except OverBudgetException:
            pass

        # Final validation

        aucs = []
        l2 = aoc_logger(budget, upper=1e2, triggers=[ioh_logger.trigger.ALWAYS])
        # add also a normal IOH logger if test = True and set the directory accordingly
        if test:
            l1 = ioh.logger.Analyzer(
                root=ioh_dir,
                folder_name=algorithm_name,
                algorithm_name=algorithm_name,
            )
            combined_logger = ioh.logger.Combine([l1, l2])
            self.problem.attach_logger(combined_logger)
        else:
            self.problem.attach_logger(l2)

        for seed in self.seeds:
            budget = self.budget_factor * dim
            try:
                algorithm = local_env[algorithm_name](budget=budget, dim=dim)
                algorithm(self.problem)
            except OverBudgetException:
                aucs.append(0)
                break

            aucs.append(correct_aoc(self.problem, l2, budget))
            l2.reset(self.problem)
            if test:
                l1.reset(self.problem)
            self.problem.reset()

        auc_mean = np.mean(aucs)
        auc_std = np.std(aucs)

        solution.add_metadata("aucs", aucs)
        solution.set_scores(
            auc_mean,
            f"The algorithm {algorithm_name} scored {auc_mean:.3f} on AOCC (higher is better, 1.0 is the best).",
        )
        return solution

    def test(self, solution: Solution, ioh_dir=""):
        """
        Runs the solution on test instances and returns the fitness score.
        """
        return self.evaluate(solution, True, ioh_dir)

    def to_dict(self):
        """
        Converts the problem to a dictionary.
        """
        return {
            "name": self.name,
            "problem_type": self.problem_type,
            "budget_factor": self.budget_factor,
        }

    def _rebuild_problem(self):
        """(Re)create the IOH/photonic instance after unpickling/copying."""
        self.problem = get_photonic_instance(self.problem_type)
        self.training_instances = [self.problem]
        self.test_instances = [self.problem]

    def __getstate__(self):
        """Return the picklable part of the instance."""
        state = self.__dict__.copy()
        state.pop("problem", None)  # the client itself is NOT picklable
        state.pop("training_instances", None)
        state.pop("test_instances", None)
        return state  # everything else is fine

    def __setstate__(self, state):
        """
        Restore and rebuild the IOH/photonic instance.
        """
        self.__dict__.update(state)
        self._rebuild_problem()
        return

    def __deepcopy__(self, memo):
        import copy

        cls = self.__class__
        new = cls.__new__(cls)
        memo[id(self)] = new
        self.problem = None
        self.training_instances = None
        self.test_instances = None
        # deepcopy everything except the C++ problem
        for k, v in self.__dict__.items():
            setattr(new, k, copy.deepcopy(v, memo))
        new._rebuild_problem()
        return new
