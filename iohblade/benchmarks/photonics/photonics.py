import inspect
import os
import random
import textwrap
import traceback
from typing import Any

import ioh
import numpy as np
import pandas as pd
from ioh import get_problem
from ioh import logger as ioh_logger

from iohblade.benchmarks.photonics.photonics_instances import (
    ELLIPSOMETRY_REFERENCE_SEED,
    algorithmic_insights,
    get_photonic_instance,
    problem_descriptions,
)
from iohblade.problem import BASE_DEPENDENCIES, Problem
from iohblade.solution import Solution
from iohblade.utils import OverBudgetException, aoc_logger, correct_aoc

# (lower, upper) bounds of the AOC logger for each problem type.
AOC_BOUNDS = {
    "bragg": (0.1648, 1.0),
    "ellipsometry": (1e-8, 40.0),
    "photovoltaic": (0.1, 1.0),
}


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
        reference_seed=ELLIPSOMETRY_REFERENCE_SEED,
    ):
        """
        Initializes the MA-BBOB problem instance.
        Args:
            logger (RunLogger): The logger to use for logging.
            problem_type (str): The name of the problem instance, can be one of: "bragg", "ellipsometry" or "photovoltaic".
            name (str): The name of the problem.
            eval_timeout (int): The evaluation timeout in seconds.
            budget_factor (int): The factor to multiply the dimensionality with to get the budget.
            seeds (int): Number of random runs. Run ``i`` seeds numpy's and
                Python's global random generators with ``i``.
            reference_seed (int): Seed of the random reference structure of
                the ellipsometry problem; fixed so all evaluations use the
                same target spectrum.
        """
        if dependencies is None:
            dependencies = [
                "ioh==0.3.22",
                "pandas==2.2.3",
                "pymoosh==3.2",
            ]
        if imports is None:
            imports = "import numpy as np\nimport ioh\n"

        if problem_type not in ["bragg", "ellipsometry", "photovoltaic"]:
            raise Exception(
                "problem_type should be either 'bragg', 'ellipsometry' or 'photovoltaic'."
            )

        self.problem_type = problem_type
        # Fixed seed of the random reference structure (ellipsometry), so that
        # every copy of the problem scores against the same target.
        self.reference_seed = reference_seed
        self.problem = get_photonic_instance(
            self.problem_type, reference_seed=self.reference_seed
        )
        self.func_name = "__call__"
        self.init_inputs = ["budget", "dim"]
        self.func_inputs = ["func"]
        self.func_outputs = ["f_opt", "x_opt"]

        super().__init__(
            logger,
            [self.problem],
            [self.problem],
            name,
            eval_timeout,
            dependencies,
            imports,
        )
        self.budget_factor = budget_factor  # The factor to multiply the dimensionality with to get the budget
        self.description_prompt = problem_descriptions[self.problem_type]
        self.extra_prompt = algorithmic_insights[self.problem_type]
        self.seeds = list(range(seeds))
        self.task_prompt = textwrap.dedent("""
You are a Python developer and AI and physics researcher.
Your task is to develop a novel heuristic optimization algorithm for photonic optimization problems.
The code should contain an `__init__(self, budget, dim)` function with optional additional arguments and the function `def __call__(self, func)`, which should optimize the black box function `func` using `self.budget` function evaluations.
The func() can only be called as many times as the budget allows, not more.
""")
        self.example_prompt = textwrap.dedent("""
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
""")
        self.format_prompt = textwrap.dedent("""
Give an excellent and novel heuristic algorithm to solve this task and also give it a one-line description, describing the main idea. Give the response in the format:
# Description: <short-description>
# Code:
```python
<code>
```
""")

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
        auc_lower, auc_upper = AOC_BOUNDS[self.problem_type]
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
        l2 = aoc_logger(
            budget,
            lower=auc_lower,
            upper=auc_upper,
            triggers=[ioh_logger.trigger.ALWAYS],
        )
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
            # Each run gets its own seed, so the runs differ from each other
            # but an evaluation is reproducible.
            np.random.seed(seed)
            random.seed(seed)
            try:
                algorithm = local_env[algorithm_name](budget=budget, dim=dim)
                algorithm(self.problem)
            except OverBudgetException:
                # Same as BBOB_SBOX: the run is scored on the evaluations
                # within the budget, and the remaining runs still take place
                # so that every run counts in the mean.
                pass

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
            "reference_seed": self.reference_seed,
        }

    def _rebuild_problem(self):
        """(Re)create the IOH/photonic instance after unpickling/copying."""
        self.problem = get_photonic_instance(
            self.problem_type,
            reference_seed=getattr(self, "reference_seed", ELLIPSOMETRY_REFERENCE_SEED),
        )
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
        # deepcopy everything except the C++ problem (the original keeps it)
        skip = {"problem", "training_instances", "test_instances"}
        for k, v in self.__dict__.items():
            if k not in skip:
                setattr(new, k, copy.deepcopy(v, memo))
        new._rebuild_problem()
        return new

    def get_config(self) -> dict[str, Any]:
        from iohblade.tags import (
            Benchmark,
            NoiseType,
            ObjectiveType,
            PrimaryCategories,
            StructureTag,
            VariableType,
        )

        evaluator = inspect.getsource(self.evaluate)
        tags: list[Any] = [
            PrimaryCategories.BBO,
            Benchmark.PHOTONICS,
            NoiseType.NOISELESS,
            ObjectiveType.SINGLE_OBJECTIVE,
            VariableType.CONTINUOUS,
            StructureTag.PHYSICS,
        ]

        config = {
            "tags": tags,
            "name": "Photonics",
            "prompt": self.get_prompt(),
            "minimisation": False,
            "evaluator": evaluator,
            "config": {
                "problem_type": self.problem_type,
                "budget_factor": self.budget_factor,
                "reference_seed": self.reference_seed,
                "depencencies": self.dependencies,
                "imports": self.imports,
            },
        }
        return config


if __name__ == "__main__":
    pn = Photonics(logger=None)
    for key, value in pn.get_config().items():
        print(f"------------------------------{key}------------------------------")
        print(value)
