import os
import traceback

import ioh
import numpy as np
import pandas as pd
from ConfigSpace import Configuration, ConfigurationSpace
from ioh import get_problem
from ioh import logger as ioh_logger
from smac import AlgorithmConfigurationFacade, Scenario

from ..problem import Problem
from ..solution import Solution
from ..utils import OverBudgetException, aoc_logger, correct_aoc


class MA_BBOB(Problem):
    """
    Problem class for evaluating optimization algorithms on the MA-BBOB benchmark.
    """

    def __init__(
        self,
        logger=None,
        training_instances=None,
        test_instances=None,
        name="MA_BBOB",
        eval_timeout=60,
        dims=[2, 5],
        budget_factor=2000,
    ):
        """
        Initializes the MA-BBOB problem instance.
        Args:
            logger (RunLogger): The logger to use for logging.
            training_instances (list): The indices of training instances to use.
            test_instances (list): The indices of test instances to use.
            name (str): The name of the problem.
            eval_timeout (int): The evaluation timeout in seconds.
            dims (list): The dimensionalities of the problem instances to run on.
            budget_factor (int): The factor to multiply the dimensionality with to get the budget.
        """
        if training_instances is None:
            training_instances = range(0, 20)
        if test_instances is None:
            test_instances = range(20, 120)
        super().__init__(logger, training_instances, test_instances, name, eval_timeout)
        self.dims = dims  # The dimensionalities of the problem instances to run on
        self.budget_factor = budget_factor  # The factor to multiply the dimensionality with to get the budget
        self.func_name = "__call__"
        self.init_inputs = ["budget", "dim"]
        self.func_inputs = ["func"]
        self.func_outputs = ["f_opt", "x_opt"]
        self.task_prompt = """
You are a Python developer working on a new optimization algorithm.
Your task is to develop a novel heuristic optimization algorithm for continuous optimization problems.
The optimization algorithm should handle a wide range of tasks, which is evaluated on the Many Affine BBOB test suite of noiseless functions. Your task is to write the optimization algorithm in Python code. 
Each of the optimization functions has a search space between -5.0 (lower bound) and 5.0 (upper bound). The dimensionality can be varied.
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
        self.f_opt = np.Inf
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

        # Load data files
        base_path = os.path.dirname(__file__)
        self.weights = pd.read_csv(
            os.path.join(base_path, "mabbob", "weights.csv"), index_col=0
        )
        self.iids = pd.read_csv(
            os.path.join(base_path, "mabbob", "iids.csv"), index_col=0
        )
        self.opt_locs = pd.read_csv(
            os.path.join(base_path, "mabbob", "opt_locs.csv"), index_col=0
        )

    def get_prompt(self):
        """
        Returns the problem description and answer format.
        """
        return self.task_prompt + self.example_prompt + self.format_prompt

    def evaluate(self, solution: Solution, test=False, ioh_dir=""):
        """
        Evaluates a solution on the MA-BBOB benchmark using AOCC.
        """
        auc_mean = 0
        auc_std = 0
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
        instances = self.test_instances if test else self.training_instances
        aucs = []
        for dim in self.dims:
            for idx in instances:
                budget = self.budget_factor * dim
                f_new = ioh.problem.ManyAffine(
                    xopt=np.array(self.opt_locs.iloc[idx])[:dim],
                    weights=np.array(self.weights.iloc[idx]),
                    instances=np.array(self.iids.iloc[idx], dtype=int),
                    n_variables=dim,
                )
                f_new.set_id(100)
                f_new.set_instance(idx)

                l2 = aoc_logger(budget, upper=1e2, triggers=[ioh_logger.trigger.ALWAYS])
                # add also a normal IOH logger if test = True and set the directory accordingly
                if test:
                    l1 = ioh.logger.Analyzer(
                        root=ioh_dir,
                        folder_name=algorithm_name,
                        algorithm_name=algorithm_name,
                    )
                    combined_logger = ioh.logger.Combine([l1, l2])
                    f_new.attach_logger(combined_logger)
                else:
                    f_new.attach_logger(l2)

                try:
                    algorithm = local_env[algorithm_name](budget=budget, dim=dim)
                    algorithm(f_new)
                except OverBudgetException:
                    aucs.append(0)
                    break

                aucs.append(correct_aoc(f_new, l2, budget))
                l2.reset(f_new)
                if test:
                    l1.reset(f_new)
                f_new.reset()

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
            "dims": self.dims,
            "training_instances": self.training_instances,
            "test_instances": self.test_instances,
            "budget_factor": self.budget_factor,
        }
