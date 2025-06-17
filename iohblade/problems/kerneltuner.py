import os
import numpy as np
import pandas as pd
import polars as pl
import random
import re
import json
import time
import traceback
import math
from ..solution import Solution
import pandas as pd
import traceback
from ..problem import Problem

from kernel_tuner import util
from kernel_tuner.searchspace import Searchspace
from kernel_tuner.strategies.common import CostFunc
from kernel_tuner import tune_kernel_T1
from pathlib import Path
from autotuning_methodology.experiments import generate_experiment_file, execute_experiment
from autotuning_methodology.report_experiments import get_strategy_scores


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

    def __init__(self, optimizer, budget=1000, optimum=0, scaling=False):
        self.optimizer = optimizer
        self.scaling = scaling
        self.budget = budget
        self.aoc = 0
        self.optimum = optimum

    def tune(self, searchspace: Searchspace, runner, tuning_options):
        cost_func = CostFunc(searchspace, tuning_options, runner, scaling=self.scaling)
        #problem = problem_wrapper(cost_func, self.budget, self.optimum)
        self.tuning_options = tuning_options
        self.searchspace = searchspace

        if self.scaling:
            # Initialize costfunc for scaling
            cost_func.get_bounds_x0_eps()
        try:
            #self.optimizer(problem, searchspace)
            self.optimizer(cost_func, searchspace)
        # except OverBudgetException:
        #     pass
        except util.StopCriterionReached as e:
            if tuning_options.verbose:
                print(e)

        #self.aoc = problem.get_aoc()  # correct_aoc(problem, l2, self.budget)
        #return problem.f.results
        return cost_func.results


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

    def __init__(self, budget=1000):
        self.pop_size = 20 # any parameters used in the search algorithm.
        self.budget = budget

    def __call__(self, func, searchspace):
        self.searchspace = searchspace
        self.tune_params = searchspace.tune_params.copy()

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


    def evaluate(self, solution: Solution, test=False):
        repeats = 5 # number of times to repeat for stochasticity

        path = Path(os.path.join(self.logger.get_log_dir(), "evaluation", solution.id))
        path.mkdir(parents=True, exist_ok=True)

        code = solution.code
        algorithm_name = solution.name
        
        exec(code, globals())
        budget = self.budget
        optimizer = globals()[algorithm_name](budget=budget)
        strategy = OptAlgWrapper(optimizer, budget=budget)

        # get applications & GPUs args
        gpus = self.gpus
        folder = f"{self.cache_dir}kernels"
        applications = []
        for app in self.kernels:
            applications.append(
                {
                    "name": f"{app}_milo",
                    "folder": folder,
                    "input_file": f"{app}_milo.json"
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
import traceback

from kernel_tuner import util
from kernel_tuner.searchspace import Searchspace
from kernel_tuner.strategies.common import CostFunc

{solution.code}



class problem_wrapper:
    def __init__(self, f):
        self.f = f

    def __call__(self, x):
        y = self.f(x)
        #with open("log.txt", "a") as f:
        #    f.write(f"Evaluating {{x}} -> {{y}}\\n")
        return y


class OptAlgWrapper:

    def __init__(self, optimizer={solution.name}, budget=1000, scaling=False):
        self.optimizer = optimizer(budget=budget)
        self.scaling = scaling
        self.budget = budget
        print("initialized {solution.name}")

    def tune(self, searchspace: Searchspace, runner, tuning_options):
        print("Tuning started!")
        cost_func = CostFunc(searchspace, tuning_options, runner, scaling=self.scaling)
        problem = problem_wrapper(cost_func)
        self.tuning_options = tuning_options
        self.searchspace = searchspace

        if self.scaling:
            # Initialize costfunc for scaling
            cost_func.get_bounds_x0_eps()
        try:
            print("Calling optimizer")
            self.optimizer(problem, searchspace)
        except Exception as e:
            print("Error during optimization:", e)

        return problem.f.results
"""
        solution_path = os.path.join(self.logger.get_log_dir(), "evaluation", solution.id, "code.py")
        with open(solution_path, "w") as f:
            f.write(alg_code)

        # strategy settings
        strategy: str = solution.name # the class name of your strategy
        hyperparams = []
        searchspace_strategies = [{
            "autotuner": "KernelTuner",
            "name": "OptAlgWrapper",
            "display_name": strategy.replace('_', ' ').capitalize(),
            "search_method": solution_path, # TODO give a path string to your strategy here (Can we not make this a callable?)
            'search_method_hyperparameters': hyperparams
        }]
        # any additional settings
        override = {
            "experimental_groups_defaults": {
                "repeats": repeats,
                "samples": 32,
                "minimum_fraction_of_budget_valid": 0.01,
            }
        }
        
        name = solution.id
        experiments_filepath = generate_experiment_file(name, path, searchspace_strategies, applications, gpus, override=override, generate_unique_file=True, overwrite_existing_file=True)

        # run the methodology to get a fitness score for this configuration
        scores = get_strategy_scores(str(experiments_filepath))
        score = scores[list(scores.keys())[0]]['score']

        #solution.add_metadata("all_scores", scores)
        solution.set_scores(
            score,
            f"The algorithm {solution.name} scored {score:.3f} (higher is better).",
        )
        return solution

    def evaluate_old(self, solution: Solution, test=False, ioh_dir=""):
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
