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
from kernel_tuner import tune_kernel_T1, util
from kernel_tuner.searchspace import Searchspace
from kernel_tuner.strategies.common import CostFunc

from ..problem import Problem
from ..solution import Solution


class OverBudgetException(Exception):
    """The algorithm tried to do more evaluations than allowed."""

    pass


class problem_wrapper:
    def __init__(self, f, budget, optimum, scale_log=True):
        self.f = f
        self.budget = budget
        self.aoc = 0
        self.lower = 1e-3
        self.upper = 1e2
        self.budget = budget
        self.eval_count = 0
        self.raw_y_best = self.upper
        self.global_best = optimum
        self.transform = lambda x: np.log10(x) if scale_log else (lambda x: x)

    def __call__(self, x):
        if self.eval_count > self.budget:
            raise OverBudgetException("Budget exceeded")
        y = self.f(x) - self.global_best  # so optimum at 0
        if y < self.raw_y_best:
            self.raw_y_best = y
        y_value = np.clip(self.raw_y_best, self.lower, self.upper)
        self.aoc += (self.transform(y_value) - self.transform(self.lower)) / (
            self.transform(self.upper) - self.transform(self.lower)
        )
        self.eval_count += 1
        return y

    def get_aoc(self):
        while self.eval_count < self.budget:
            y_value = np.clip(self.raw_y_best, self.lower, self.upper)
            self.aoc += (self.transform(y_value) - self.transform(self.lower)) / (
                self.transform(self.upper) - self.transform(self.lower)
            )
            self.eval_count += 1
        return 1 - (self.aoc / self.budget)


class OptAlgWrapper:
    """Wrapper class for user-defined optimization algorithms"""

    def __init__(self, optimizer, budget=100, optimum=0, scaling=False):
        self.optimizer = optimizer
        self.scaling = scaling
        self.budget = budget
        self.aoc = 0
        self.optimum = optimum

    def tune(self, searchspace: Searchspace, runner, tuning_options):
        cost_func = CostFunc(searchspace, tuning_options, runner, scaling=self.scaling)

        # l2 = aoc_logger(
        #     self.budget, upper=1e2, lower=1e-1, scale_log=True, triggers=[logger.trigger.ALWAYS]
        # )
        # problem = get_problem(f"{application}-{gpu}", instance=0, problem_class=ioh.ProblemClass.INTEGER, dimension=len(tuning_options))
        # problem.attach_logger(l2)

        problem = problem_wrapper(cost_func, self.budget, self.optimum)

        self.tuning_options = tuning_options
        self.searchspace = searchspace

        if self.scaling:
            # Initialize costfunc for scaling
            cost_func.get_bounds_x0_eps()

        try:
            self.optimizer(problem, searchspace)
        except OverBudgetException:
            pass
        except util.StopCriterionReached as e:
            if tuning_options.verbose:
                print(e)

        self.aoc = problem.get_aoc()  # correct_aoc(problem, l2, self.budget)
        return problem.f.results


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
        eval_timeout=60,
        budget=100,
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

        self.optima = {
            "gemm-A100": 8.01820807158947,
            "convolution-A100": 0.5536000076681376,
            "dedispersion-A100": 68.1165759563446,
            "hotspot-A100": 0.1966720037162304,
            "gemm-A4000": 13.089913964271545,
            "convolution-A4000": 1.021171996369958,
            "dedispersion-A4000": 147.6977825164795,
            "hotspot-A4000": 1.4143739938735962,
            "gemm-A6000": 6.123518988490105,
            "convolution-A6000": 0.6030377962291835,
            "dedispersion-A6000": 84.21807646751404,
            "hotspot-A6000": 0.8206389956176281,
            "gemm-MI250X": 6.940651074051857,
            "convolution-MI250X": 0.6587962452322245,
            "dedispersion-MI250X": 49.5724800825119,
            "hotspot-MI250X": 0.2868224401026964,
            "gemm-W6600": 22.872174671718053,
            "convolution-W6600": 1.7276193872094154,
            "dedispersion-W6600": 135.0808186531067,
            "hotspot-W6600": 1.3389952592551708,
            "gemm-W7800": 6.276454776525497,
            "convolution-W7800": 0.8161421902477741,
            "dedispersion-W7800": 50.36081421375275,
            "hotspot-W7800": 0.805098531767726,
        }
        self.cache_dir = cache_dir

        super().__init__(
            logger, self.training_instances, self.test_instances, name, eval_timeout
        )
        self.budget = budget  # The budget for the optimization algorithms
        self.task_prompt = """
You are a highly skilled computer scientist in the field of natural computing and hardware kernel tuning. Your task is to design novel metaheuristic algorithms to solve kernel tuner problems (integer, variable dimension, contraint).
The optimization algorithm should handle a kernel tuning task with a given evaluation budget. Your task is to write the optimization algorithm in Python code. The code should contain an `__init__(self, budget)` function with optional additional arguments and the function `def __call__(self, func, searchspace)`, which should optimize the black box function `func` using `self.budget` function evaluations.
The func() can only be called as many times as the budget allows, not more. The `searchspace` object can be used to sample random instances, neighbouring instances using `searchspace.get_neighbors(param_config: tuple, neighbor_method='Hamming')` where neighbor_method can be any of ["strictly-adjacent", "adjacent", "Hamming"] and to check validity of parameter settings using `searchspace.is_param_config_valid(tuple(instance))`, nothing else. The dimensionality can be varied.
In addition, the variable `tune_params` is a dictionary containing the tuning parameters with their ranges and constraints, it can be obtained directly from the searchspace object `searchspace.tune_params`. The algorithm should be able to handle any number of tuning parameters, and the search space can be continuous or discrete. The algorithm should be able to handle any type of kernel tuning problem, including but not limited to vector addition, matrix multiplication, and convolution.
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

        self.example_prompt = """
An example code structure with helper functions is as follows:
```python
import numpy as np
import random

class AlgorithmName:
    "Template for a generic search algorithm"

    def __init__(self, searchspace):
        self.pop_size = 20 # any parameters used in the search algorithm.
        self.searchspace = searchspace
        self.tune_params = searchspace.tune_params.copy()

    def __call__(self, func):
        self.f_opt = np.Inf
        self.x_opt = None
        # create initial population and run the search till evaluation budget is exhausted.
        # then retur the best solution found

    def generate_population(self):
        "We can use a constraint-aware sampling method"
        pop = list(list(p) for p in self.searchspace.get_random_sample(self.pop_size))
        return pop

    def get_neighbour(self, dna):
        "We can easily get a random neighbour with hamming distance 1 using the searchspace provided method (for example)."
        neighbors = self.searchspace.get_neighbors(tuple(dna), neighbor_method="Hamming")
        if len(neighbors) > 0:
            return list(random.choice(neighbors))
        return dna

    def repair(self, dna):
        "It is possible that at some point a configuration is not valid (due to mutation, crossover etc). "
        if not self.searchspace.is_param_config_valid(tuple(dna)):
            # dna is not valid, try to repair it
            # search for valid configurations neighboring this config
            # start from strictly-adjacent to increasingly allowing more neighbors
            for neighbor_method in ["strictly-adjacent", "adjacent", "Hamming"]:
                neighbors = self.searchspace.get_neighbors_no_cache(tuple(dna), neighbor_method=neighbor_method)
                # if we have found valid neighboring configurations, select one at random
                if len(neighbors) > 0:
                    new_dna = list(random.choice(neighbors))
                    return new_dna
        return dna
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
        Evaluates a solution on the kernel tuner benchmark using AOCC.
        """
        aocc_mean = 0
        aocc_std = 0
        code = solution.code
        algorithm_name = solution.name
        safe_globals = {
            "np": np,
            "math": math,
            "random": random,
        }
        exec(code, globals())

        algorithm = None

        # Final validation
        instances = self.test_instances if test else self.training_instances
        aoccs = []
        budget = self.budget
        for idx in instances:
            application, gpu = idx.split("-")
            if idx not in self.optima:
                continue  # Skip if no optimum is defined for this instance
            optimum = self.optima[idx]
            strategy_options = {
                "max_fevals": budget,
                "time_limit": 60,
            }
            iterations = 1  # number of kernel runs (1 because we use cached results anyways, by default 7)
            input_filepath = Path(f"{self.cache_dir}kernels/{application}_milo.json")
            cache_filepath = Path(
                f"{self.cache_dir}cachefiles/{application}_milo/{gpu}.json"
            )

            try:
                optimizer = globals()[algorithm_name](budget=budget)
                # Wrap the algorithm class in the OptAlgWrapper
                # for use in Kernel Tuner
                strategy = OptAlgWrapper(optimizer, budget=budget, optimum=optimum)

                results, env = tune_kernel_T1(
                    input_filepath,
                    cache_filepath,
                    objective="time",
                    objective_higher_is_better=False,
                    simulation_mode=True,
                    output_T4=False,
                    iterations=iterations,
                    device=gpu,
                    strategy=strategy,
                    strategy_options=strategy_options,
                )
                aoc = strategy.aoc
                score = util.get_best_config(results, "time", False)["time"]

                aoccs.append(aoc)
            except OverBudgetException:
                aoc = strategy.aoc
                aoccs.append(aoc)
                break

        aocc_mean = np.mean(aoccs)
        aocc_std = np.std(aoccs)

        solution.add_metadata("aoccs", aoccs)
        solution.set_scores(
            aocc_mean,
            f"The algorithm {algorithm_name} scored {aocc_mean:.3f} on AOCC (higher is better, 1.0 is the best).",
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
            "training_instances": self.training_instances,
            "test_instances": self.test_instances,
            "budget": self.budget,
        }
