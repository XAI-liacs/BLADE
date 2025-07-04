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


class BBOB_SBOX(Problem):
    """
    Problem class for evaluating optimization algorithms on the SBOX-COST or BBOB benchmark. See also https://github.com/sbox-cost/Examples

    Black-Box Optimization Benchmarking (BBOB) problem set, which contains 24 noiseless real-valued test functions supported on [-5, 5]^n, where n is the dimensionality.

    These problems were orginally proposed by Hansen et. al. in [FinckHRA10] and was implemented as the core component of the COmparing Continous Optimizer (COCO) platform [HansenARMTB20].
    We took the implementation of those 24 functions in https://github.com/numbbo/coco/tree/master/code-experiments/src (v2.2) and adopted those to our framework.


    [HansenARMTB20] Nikolaus Hansen, Anne Auger, Raymond Ros, Olaf Mersmann, Tea Tusar, and Dimo Brockhoff. “COCO: A platform for comparing continuous optimizers in a black-box setting.” Optimization Methods and Software (2020): 1-31.

    [FinckHRA10] Steffen Finck, Nikolaus Hansen, Raymond Ros, and Anne Auger. “Real-parameter black-box optimization benchmarking 2009: Presentation of the noiseless functions.” Technical Report 2009/20, Research Center PPE, 2009. Updated February, 2010.
    """

    def __init__(
        self,
        logger=None,
        training_instances=None,
        test_instances=None,
        name="SBOX_COST",
        eval_timeout=120,
        dims=[2, 5],
        budget_factor=2000,
        specific_fid=None,
        specific_group=None,
        problem_type=ioh.ProblemClass.SBOX,
        full_ioh_log=False,
        ioh_dir="",
    ):
        """
        Initializes the MA-BBOB problem instance.
        Args:
            logger (RunLogger): The logger to use for logging.
            training_instances (list): A list of tuples with (fid=function id, iid=instance id) for training instances to use.
            test_instances (list): The indices of test instances to use. A list of tuples with (fid=function id, iid=instance id).
            name (str): The name of the problem.
            eval_timeout (int): The evaluation timeout in seconds.
            dims (list): The dimensionalities of the problem instances to run on.
            budget_factor (int): The factor to multiply the dimensionality with to get the budget.
            specific_fid (int): The specific function id to use. If not None, additional information is added to the prompt about the function.
            specific_group (int): The specific function group (1,2,3,4,5) to use. If not None, additional information is added to the prompt about the function group.
            problem_type (ioh.ProblemClass): The type of problem to use. Can be SBOX or BBOB.
            full_ioh_log (bool): If set to True, additional IOH logs are being kept for each run and each algorithm.
        """
        if training_instances is None:
            training_instances = [(f, i) for f in range(1, 25) for i in range(1, 6)]
        if test_instances is None:
            test_instances = [
                (f, i) for f in range(1, 25) for i in range(5, 16)
            ]  # 10 test instances
        super().__init__(logger, training_instances, test_instances, name, eval_timeout)
        self.dims = dims  # The dimensionalities of the problem instances to run on
        self.budget_factor = budget_factor  # The factor to multiply the dimensionality with to get the budget
        self.specific_fid = specific_fid
        self.specific_group = specific_group
        self.full_ioh_log = full_ioh_log
        self.ioh_dir = ioh_dir

        self.func_name = "__call__"
        self.init_inputs = ["budget", "dim"]
        self.func_inputs = ["func"]
        self.func_outputs = ["f_opt", "x_opt"]

        # List containing descriptions of each function group
        function_groups = [
            "Separable Functions",
            "Functions with low or moderate conditioning",
            "Functions with high conditioning and unimodal",
            "Multi-modal functions with adequate global structure",
            "Multi-modal functions with weak global structure",
        ]

        # List containing information for all 24 functions
        functions = [
            "f1: Sphere Function",
            "f2: Separable Ellipsoidal Function",
            "f3: Rastrigin Function",
            "f4: Büche-Rastrigin Function",
            "f5: Linear Slope",
            "f6: Attractive Sector Function",
            "f7: Step Ellipsoidal Function",
            "f8: Rosenbrock Function, original",
            "f9: Rosenbrock Function, rotated",
            "f10: Ellipsoidal Function",
            "f11: Discus Function",
            "f12: Bent Cigar Function",
            "f13: Sharp Ridge Function",
            "f14: Different Powers Function",
            "f15: Rastrigin Function",
            "f16: Weierstrass Function",
            "f17: Schaffer's F7 Function",
            "f18: Schaffer's F7 Function, moderately ill-conditioned",
            "f19: Composite Griewank-Rosenbrock Function F8F2",
            "f20: Schwefel Function",
            "f21: Gallagher's Gaussian 101-me Peaks Function",
            "f22: Gallagher's Gaussian 21-hi Peaks Function",
            "f23: Katsuura Function",
            "f24: Lunacek bi-Rastrigin Function",
        ]
        self.problem_type = problem_type
        self.benchmark_name = (
            "SBOX-COST test suite of noiseless box-constrained functions."
            if problem_type == ioh.ProblemClass.SBOX
            else "BBOB test suite of noiseless functions."
        )
        box_constrained = (
            "box-constrained"
            if problem_type == ioh.ProblemClass.SBOX
            else "unconstrained"
        )
        extra_prompt = f"The optimization algorithm should handle a wide range of tasks, which is evaluated on the {self.benchmark_name}"
        if (
            self.specific_fid is not None
            and self.specific_fid < 25
            and self.specific_fid > 0
        ):
            extra_prompt = f"The optimization algorithm should work on different instances of noiseless {box_constrained} functions. Specifically function: {functions[self.specific_fid-1]}."
        elif (
            self.specific_group is not None
            and self.specific_group < 6
            and self.specific_group > 0
        ):
            extra_prompt = f"The optimization algorithm should work on different instances of noiseless {box_constrained} functions. Specifically it should work well for {function_groups[self.specific_group-1]}."
        else:
            extra_prompt = f"The optimization algorithm should work on different instances of noiseless {box_constrained} functions."

        self.task_prompt = f"""
You are a Python expert working on a new optimization algorithm.
Your task is to develop a novel heuristic optimization algorithm for continuous optimization problems.
{extra_prompt} Your task is to write the optimization algorithm in Python code. 
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

    def get_prompt(self):
        """
        Returns the problem description and answer format.
        """
        return self.task_prompt + self.example_prompt + self.format_prompt

    def evaluate(self, solution: Solution, test=False):
        """
        Evaluates a solution on the SBOX or BBOB benchmark using AOCC.
        """
        auc_mean = 0
        auc_std = 0
        code = solution.code
        algorithm_name = solution.name
        algorithm_id = solution.id
        safe_globals = {"np": np}
        local_env = {}
        exec(code, safe_globals, local_env)

        algorithm = None

        # Small test run to catch code errors
        try:
            l2_temp = aoc_logger(100, upper=1e2, triggers=[ioh_logger.trigger.ALWAYS])
            problem = get_problem(
                1, instance=1, dimension=2, problem_class=self.problem_type
            )
            problem.attach_logger(l2_temp)
            algorithm = local_env[algorithm_name](budget=100, dim=2)
            algorithm(problem)
        except OverBudgetException:
            pass

        # Final validation
        instances = self.test_instances if test else self.training_instances
        aucs = []
        for dim in self.dims:
            for instance in instances:
                fid, iid = instance  # we expact a tuple of (fid, iid)
                budget = self.budget_factor * dim
                f_new = get_problem(
                    fid, instance=iid, dimension=dim, problem_class=self.problem_type
                )
                l2 = aoc_logger(budget, upper=1e2, triggers=[ioh_logger.trigger.ALWAYS])
                if test or self.full_ioh_log:
                    l1 = ioh.logger.Analyzer(
                        root=self.ioh_dir,
                        folder_name=algorithm_id,
                        algorithm_name=algorithm_id,
                        store_positions=True,
                        triggers=[ioh_logger.trigger.ALWAYS],
                    )
                    combined_logger = ioh.logger.Combine([l1, l2])
                    f_new.attach_logger(combined_logger)
                else:
                    f_new.attach_logger(l2)

                try:
                    algorithm = local_env[algorithm_name](budget=budget, dim=dim)
                    algorithm(f_new)
                except OverBudgetException:
                    pass

                aucs.append(correct_aoc(f_new, l2, budget))
                l2.reset(f_new)
                f_new.reset()

        auc_mean = np.mean(aucs)
        auc_std = np.std(aucs)

        solution.add_metadata("aucs", aucs)
        solution.set_scores(
            auc_mean,
            f"The algorithm {algorithm_name} scored {auc_mean:.3f} on AOCC (higher is better, 1.0 is the best).",
        )

        return solution

    def test(self, solution: Solution):
        """
        Runs the solution on test instances and returns the fitness score.
        """
        return self.evaluate(solution, True)

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
            "problem_type": "SBOX"
            if self.problem_type == ioh.ProblemClass.SBOX
            else "BBOB",
            "specific_fid": self.specific_fid,
            "specific_group": self.specific_group,
        }
