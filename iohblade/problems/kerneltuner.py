import json
import math
import os
import random
import re
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

try:
    from autotuning_methodology.experiments import (
        execute_experiment,
        generate_experiment_file,
    )
    from autotuning_methodology.report_experiments import get_strategy_scores
    from kernel_tuner import tune_kernel_T1, util
    from kernel_tuner.searchspace import Searchspace
    from kernel_tuner.strategies.common import CostFunc
    from kernel_tuner.strategies.wrapper import OptAlg
except Exception:  # pragma: no cover - optional dependency
    tune_kernel_T1 = None
    util = None
    Searchspace = None
    CostFunc = None
    OptAlg = object
    generate_experiment_file = None
    execute_experiment = None
    get_strategy_scores = None


from ..problem import Problem
from ..solution import Solution


class OverBudgetException(Exception):
    """The algorithm tried to do more evaluations than allowed."""

    pass


class Kerneltuner(Problem):
    """
    Problem class for evaluating optimization algorithms on kernel tuner real world benchmark.
    Note that this problem requires additional installation steps.

    """

    def __init__(
        self,
        logger=None,
        gpus=None,
        kernels=None,
        name="kerneltuner",
        eval_timeout=600,
        budget=1000,
        cache_dir="/data/neocortex/repos/benchmark_hub/",
        extra_info=False,
    ):
        """
        Initializes the Kerneltuner problem instance.
        Args:
            logger (RunLogger): The logger to use for logging.
            gpus (list): The gpus to train on.
            kernels (list): The kernels (applications) to train on.
            name (str): The name of the problem.
            eval_timeout (int): The evaluation timeout in seconds.
            budget (int): The budget for the optimization algorithms/
            cache_dir (str): The directory that contains the kernel tuner data files.
            extra_info (bool): If True, additional information about the problem is added to the prompt. Only works for one kernel.
        """

        self.applications = ["gemm", "convolution", "dedispersion", "hotspot"]
        if gpus is None:
            self.gpus = ["A100", "A4000", "A6000", "MI250X", "W6600", "W7800"]
        else:
            self.gpus = gpus
        if kernels is None:
            self.kernels = self.applications
        else:
            self.kernels = kernels

        self.training_instances = []
        self.test_instances = []
        for gpu in self.gpus:
            for kernel in self.kernels:
                # for now we add them all to both training and test instances.
                self.training_instances.append(f"{kernel}-{gpu}")
                self.test_instances.append(f"{kernel}-{gpu}")

        self.cache_dir = cache_dir

        super().__init__(
            logger, self.training_instances, self.test_instances, name, eval_timeout
        )
        self.budget = budget  # The budget for the optimization algorithms
        self.task_prompt = """
You are a highly skilled computer scientist in the field of natural computing and hardware kernel tuning. Your task is to design novel metaheuristic algorithms to solve kernel tuner problems (integer, variable dimension, contraint).
The optimization algorithm should handle a kernel tuning task. Your task is to write the optimization algorithm in Python code. The code should inherit the `OptAlg` class and contain an `__init__(self, budget=5000)` function with optional arguments and the function `def __call__(self, func, searchspace)`, which should optimize the black box function `func` till the `func.budget_spent_fraction` is 1.0.
The `searchspace` object can be used to sample random instances, neighbouring instances using `searchspace.get_neighbors(param_config: tuple, neighbor_method='Hamming')` where neighbor_method can be any of ["strictly-adjacent", "adjacent", "Hamming"] and to check validity of parameter settings using `searchspace.is_param_config_valid(tuple(instance))`, nothing else. The dimensionality can be varied.
In addition, the variable `tune_params` is a dictionary containing the tuning parameters with their ranges and constraints, it can be obtained directly from the searchspace object `searchspace.tune_params`. The algorithm should be able to handle any number of tuning parameters, and the search space can be continuous or discrete. 

"""
        if len(self.kernels) == 1 and extra_info:
            input_filepath = Path(
                f"{self.cache_dir}kernels/{self.kernels[0]}_milo.json"
            )
            # read the specification file for the kernel
            self.task_prompt += (
                "\nThe kernel to tune is "
                + self.kernels[0]
                + ". The search space specification is as follows:\n"
            )
            with open(input_filepath, "r") as f:
                self.task_prompt += f.read()

        else:
            self.task_prompt += "The algorithm should be able to handle any type of kernel tuning problem, including but not limited to vector addition, matrix multiplication, and convolution.\n"

        self.example_prompt = """
An example code structure with helper functions is as follows:
```python
import numpy as np
import random

class AlgorithmName(OptAlg):
    "Template for a kernel-tune algorithm"

    def __init__(self, budget=5000):
        # any parameters used in the search algorithm.
        self.param = None

    def __call__(self, func, searchspace):
        #this is not really the budget, but the size of the search space. The budget is dynamic and we can see how much fraction we used with `func.budget_spent_fraction`.
        self.budget = searchspace.size
        self.searchspace = searchspace
        self.tune_params = searchspace.tune_params.copy()

        self.f_opt = np.Inf
        self.x_opt = None
        # create initial population and run the search till func.budget_spent_fraction is 1.0.
        # evaluate a solution using `func(x)` where `x` is a list of parameter values.
        # then return the best solution found (tuple, x_opt, f_opt) at the end of the search.
        return self.x_opt, self.f_opt

    def generate_population(self, pop_size=10):
        "We can use a constraint-aware random sampling method (optional), get_random_sample always returns valid configurations."
        pop = list(list(p) for p in self.searchspace.get_random_sample(pop_size))
        return pop

    def get_neighbour(self, solution):
        "We can easily get a random neighbour with hamming distance 1 using the searchspace provided method (for example)."
        neighbors = self.searchspace.get_neighbors(tuple(solution), neighbor_method="Hamming")
        if len(neighbors) > 0:
            return list(random.choice(neighbors))
        return solution

    def repair(self, solution):
        "It is possible that at some point a configuration is not valid (due to mutation, crossover etc). "
        if not self.searchspace.is_param_config_valid(tuple(solution)):
            # solution is not valid, try to repair it
            # search for valid configurations neighboring this config
            # start from strictly-adjacent to increasingly allowing more neighbors
            for neighbor_method in ["strictly-adjacent", "adjacent", "Hamming"]:
                neighbors = self.searchspace.get_neighbors_no_cache(tuple(solution), neighbor_method=neighbor_method)
                # if we have found valid neighboring configurations, select one at random
                if len(neighbors) > 0:
                    new_solution = list(random.choice(neighbors))
                    return new_solution
        return solution
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
        return self.task_prompt + self.example_prompt + self.format_prompt

    def evaluate(self, solution: Solution, test=False):
        repeats = 5  # number of times to repeat for stochasticity, just two for now.

        path = Path(os.path.join(self.logger.get_log_dir(), "evaluation", solution.id))
        path.mkdir(parents=True, exist_ok=True)

        code = solution.code
        algorithm_name = solution.name

        exec(code, globals())
        strategy = globals()[algorithm_name]()

        # get applications & GPUs args
        gpus = self.gpus
        folder = f"{self.cache_dir}kernels"
        applications = []
        for app in self.kernels:
            applications.append(
                {
                    "name": f"{app}_milo",
                    "folder": folder,
                    "input_file": f"{app}_milo.json",
                }
            )
        # write the solution to a file
        alg_code = f"""
import os
import numpy as np
import random
import re
import json
import time
import traceback
import math

from kernel_tuner import util
from kernel_tuner.searchspace import Searchspace
from kernel_tuner.strategies.common import CostFunc
from kernel_tuner.strategies.wrapper import OptAlg

{solution.code}

"""
        solution_path = os.path.join(
            self.logger.get_log_dir(), "evaluation", solution.id, "code.py"
        )
        with open(solution_path, "w") as f:
            f.write(alg_code)

        # strategy settings
        strategy: str = solution.name  # the class name of your strategy
        hyperparams = []
        searchspace_strategies = [
            {
                "autotuner": "KernelTuner",
                "name": strategy,
                "display_name": strategy.replace("_", " ").capitalize(),
                "search_method": strategy,  # TODO give a path string to your strategy here (Can we not make this a callable?)
                "search_method_hyperparameters": hyperparams,
                "custom_search_method_path": solution_path,
            }
        ]
        # any additional settings
        override = {
            "experimental_groups_defaults": {
                "parent_folder": str(path),
                "repeats": repeats,
                "samples": 32,
                "minimum_fraction_of_budget_valid": 0.01,
                "pattern_for_full_search_space_filenames": {
                    "regex": "/data/neocortex/repos/benchmark_hub/cachefiles/${applications}/${gpus}_T4.json"
                },
            }
        }

        name = solution.id
        experiments_filepath = generate_experiment_file(
            name,
            path,
            searchspace_strategies,
            applications,
            gpus,
            override=override,
            generate_unique_file=False,
            overwrite_existing_file=True,
        )

        # run the methodology to get a fitness score for this configuration
        scores = get_strategy_scores(str(experiments_filepath))
        score = scores[list(scores.keys())[0]]["score"]

        # solution.add_metadata("all_scores", scores)
        solution.set_scores(
            score,
            f"The algorithm {solution.name} scored {score:.3f} (higher is better).",
        )
        return solution

    def test(self, solution: Solution, ioh_dir=""):
        """
        Runs the solution on test instances and returns the fitness score.

        To evaluate kernel tuner solutions, use `autotuning_visualize <path to test file>`.
        """
        return self.evaluate(solution, True, ioh_dir)

    def to_dict(self):
        """
        Converts the problem to a dictionary.
        """
        return {
            "name": self.name,
            "training_instances": self.training_instances,
            "test_instances": self.test_instances,
            "budget": self.budget,
        }
