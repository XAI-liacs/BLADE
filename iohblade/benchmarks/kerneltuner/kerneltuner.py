import inspect
import json
import math
import os
import random
import re
import textwrap
import time
import traceback
from pathlib import Path, PurePath
from typing import Any

import numpy as np
import pandas as pd

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


from iohblade.hpo import (
    evaluation_config,
    hpo_instance_subset,
    import_smac,
    plain_config,
    run_smac,
    smac_output_dir,
)
from iohblade.problem import BASE_DEPENDENCIES, Problem
from iohblade.solution import Solution


class OverBudgetException(Exception):
    """The algorithm tried to do more evaluations than allowed."""

    pass


class Kerneltuner(Problem):
    """
    Problem class for evaluating optimization algorithms on kernel tuner real world benchmark.
    Note that this problem requires additional installation steps.

    Supports hyperparameter tuning (HPO) of solutions with a configuration
    space, see ``hpo`` and :mod:`iohblade.hpo`.
    """

    supports_hpo = True

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
        dependencies=None,
        imports=None,
        hpo: bool = False,
        hpo_trials: int = 50,
        hpo_instances: int | None = 12,
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
            dependencies (list, optional): a list of pypi packages to install before evaluation.
            imports (string, optional): the python string to manage imports in the evaluation file.
            hpo (bool): Tune the hyperparameters of solutions that come with
                a configuration space (``Solution.configspace``) with SMAC
                before scoring them; also adds
                :data:`iohblade.hpo.HPO_DEPENDENCIES` to the evaluation
                environment. Enabled automatically (see
                :meth:`~iohblade.problem.Problem.enable_hpo`) when a method
                samples with HPO, e.g. LLaMEA with ``HPO=True``.
            hpo_trials (int): SMAC's budget: the number of target runs per
                tuned solution, each one autotuning-methodology run of the
                optimiser on one kernel-GPU pair. A training evaluation runs
                all pairs (24 by default), so the default of 50 adds about
                two training evaluations plus SMAC's own overhead; consider
                raising ``eval_timeout`` (default 600 s) with HPO.
            hpo_instances (int | None): Tune on a fixed, seeded subset of
                this many kernel-GPU pairs of the training instances
                (default 12, as AutoML; ``None``: all of them).
        """

        if dependencies is None:
            dependencies = [
                "kernel-tuner @ git+https://github.com/XAI-liacs/kernel_tuner.git@hyperparametertuning_custom_strategies",
                "autotuning-methodology @ git+https://github.com/AutoTuningAssociation/autotuning_methodology.git@6a9a50a5a49bc104469b3b753fd43a5324241702",
                "pandas==2.2.3",
                "ioh==0.3.19",
                # smac and ConfigSpace are only installed with HPO, see
                # enable_hpo and iohblade.hpo.HPO_DEPENDENCIES.
            ]
        if imports is None:
            imports = "import numpy as np"

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
            logger,
            self.training_instances,
            self.test_instances,
            name,
            eval_timeout,
            dependencies,
            imports,
        )
        self.budget = budget  # The budget for the optimization algorithms
        self.hpo_trials = int(hpo_trials)
        self.hpo_instances = hpo_instances
        if hpo:
            self.enable_hpo(True)
        self.task_prompt = textwrap.dedent("""
You are a highly skilled computer scientist in the field of natural computing and hardware kernel tuning. Your task is to design novel metaheuristic algorithms to solve kernel tuner problems (integer, variable dimension, contraint).
The optimization algorithm should handle a kernel tuning task. Your task is to write the optimization algorithm in Python code. The code should inherit the `OptAlg` class and contain an `__init__(self, budget=5000)` function with optional arguments and the function `def __call__(self, func, searchspace)`, which should optimize the black box function `func` till the `func.budget_spent_fraction` is 1.0.
The `searchspace` object can be used to sample random instances, neighbouring instances using `searchspace.get_neighbors(param_config: tuple, neighbor_method='Hamming')` where neighbor_method can be any of ["strictly-adjacent", "adjacent", "Hamming"] and to check validity of parameter settings using `searchspace.is_param_config_valid(tuple(instance))`, nothing else. The dimensionality can be varied.
In addition, the variable `tune_params` is a dictionary containing the tuning parameters with their ranges and constraints, it can be obtained directly from the searchspace object `searchspace.tune_params`. The algorithm should be able to handle any number of tuning parameters, and the search space can be continuous or discrete.

""")
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

        self.example_prompt = textwrap.dedent("""
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

        self.f_opt = np.inf
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
        return self.task_prompt + self.example_prompt + self.format_prompt

    def _applications(self, kernels):
        """The autotuning-methodology application entries of the kernels."""
        folder = f"{self.cache_dir}kernels"
        return [
            {
                "name": f"{app}_milo",
                "folder": folder,
                "input_file": f"{app}_milo.json",
            }
            for app in kernels
        ]

    def _score(self, solution, solution_path, path, name, kernels, gpus, config):
        """Score the optimiser with the autotuning methodology.

        Args:
            solution (Solution): The solution (its name is the strategy).
            solution_path (str): The file with the strategy's code.
            path (Path): The folder of this experiment.
            name (str): The name of the experiment file.
            kernels (list[str]): The kernels (applications) to run on.
            gpus (list[str]): The GPUs to run on.
            config (dict): Hyperparameters of the strategy, passed as the
                strategy's ``search_method_hyperparameters``
                (``[{"name": ..., "value": ...}]``).

        Returns:
            float: The score (higher is better).
        """
        repeats = 5  # number of times to repeat for stochasticity

        # strategy settings
        strategy: str = solution.name  # the class name of your strategy
        hyperparams = [{"name": k, "value": v} for k, v in config.items()]
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
                    # A regular expression for kernel_tuner: use "/" also
                    # on Windows, where "\\" would act as an escape.
                    "regex": PurePath(self.cache_dir).as_posix().rstrip("/")
                    + "/cachefiles/${applications}/${gpus}_T4.json"
                },
            }
        }

        experiments_filepath = generate_experiment_file(
            name,
            path,
            searchspace_strategies,
            self._applications(kernels),
            gpus,
            override=override,
            generate_unique_file=False,
            overwrite_existing_file=True,
        )

        # run the methodology to get a fitness score for this configuration
        scores = get_strategy_scores(str(experiments_filepath))
        return scores[list(scores.keys())[0]]["score"]

    def _tune(self, solution, solution_path, path) -> dict[str, Any]:
        """Tune the solution's configuration space with SMAC.

        Each SMAC instance is one kernel-GPU pair of the training instances;
        the cost of a configuration is ``1 - score`` of the optimiser on it
        (SMAC minimises; the score is maximised).

        Returns:
            dict: The incumbent configuration.
        """
        instances = hpo_instance_subset(self.training_instances, self.hpo_instances)
        trial = {"n": 0}

        def target(cfg, instance: str, seed: int = 0) -> float:
            kernel, gpu = instance.rsplit("-", 1)
            trial["n"] += 1
            trial_path = Path(os.path.join(path, "hpo", f"trial{trial['n']}"))
            trial_path.mkdir(parents=True, exist_ok=True)
            try:
                score = self._score(
                    solution,
                    solution_path,
                    trial_path,
                    f"{solution.id}-hpo{trial['n']}",
                    [kernel],
                    [gpu],
                    plain_config(cfg),
                )
            except Exception:
                return 1e9  # a failing configuration: worse than any score
            return float(1.0 - score)

        incumbent = run_smac(
            solution.configspace,
            target,
            instances=list(instances),
            n_trials=self.hpo_trials,
            output_dir=smac_output_dir(self),
            deterministic=True,
            name=f"{self.name}-{solution.id}",
        )
        return plain_config(incumbent)

    def evaluate(self, solution: Solution, test=False):
        """Score the optimiser on the kernels and GPUs.

        With HPO enabled (``hpo``) and a solution with a configuration
        space, its hyperparameters are first tuned with SMAC on (a subset
        of) the training kernel-GPU pairs, and the incumbent is scored on
        all of them and stored in the metadata (``incumbent``). On the test
        instances (``test=True``) the incumbent of the search is used.
        """
        cs = getattr(solution, "configspace", None)
        tune = cs is not None and self.hpo and not test
        if tune:
            # Before the generated code is used: a missing smac is a problem
            # of the evaluation environment (a harness error), not of the code.
            import_smac()

        path = Path(os.path.join(self.logger_dir, "evaluation", solution.id))
        path.mkdir(parents=True, exist_ok=True)

        code = solution.code
        algorithm_name = solution.name

        exec(code, globals())
        strategy = globals()[algorithm_name]()

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
            self.logger_dir, "evaluation", solution.id, "code.py"
        )
        with open(solution_path, "w") as f:
            f.write(alg_code)

        incumbent = None
        if tune:
            incumbent = self._tune(solution, solution_path, path)
            solution.add_metadata("incumbent", incumbent)
        config, hpo_note = evaluation_config(self, solution, test, incumbent)

        score = self._score(
            solution,
            solution_path,
            path,
            solution.id,
            self.kernels,
            self.gpus,
            config,
        )

        # solution.add_metadata("all_scores", scores)
        solution.set_scores(
            score,
            f"The algorithm {solution.name} scored {score:.3f} (higher is better)."
            + (f"\n{hpo_note}" if hpo_note else ""),
        )
        return solution

    def test(self, solution: Solution, ioh_dir=""):
        """
        Runs the solution on test instances and returns the fitness score.

        Uses the HPO incumbent of the search (``incumbent`` in the
        solution's metadata) when there is one.

        To evaluate kernel tuner solutions, use `autotuning_visualize <path to test file>`.
        """
        return self.evaluate(solution, test=True)

    def to_dict(self):
        """
        Converts the problem to a dictionary.
        """
        return {
            "name": self.name,
            "training_instances": self.training_instances,
            "test_instances": self.test_instances,
            "budget": self.budget,
            **(
                {
                    "hpo": True,
                    "hpo_trials": self.hpo_trials,
                    "hpo_instances": self.hpo_instances,
                }
                if self.hpo
                else {}
            ),
        }

    def get_config(self) -> dict[str, Any]:
        from iohblade.tags import (
            Benchmark,
            NoiseType,
            ObjectiveType,
            PrimaryCategories,
            StructureTag,
            VariableType,
        )

        tags: list[Any] = [
            Benchmark.KERNEL_TUNER,
            PrimaryCategories.BBO,
            PrimaryCategories.ML,
            PrimaryCategories.PERFORMANCE,
            StructureTag.MATRIX,
            ObjectiveType.SINGLE_OBJECTIVE,
            NoiseType.NOISELESS,
            VariableType.MIXED,
        ]

        extra_config = {
            "gpus": self.gpus,
            "applications": self.applications,
            "kernels": self.kernels,
            "dependencies": self.dependencies,
            "imports": self.imports,
        }

        config = {
            "tags": tags,
            "name": "KernelTuner",
            "prompt": self.get_prompt(),
            "minimisation": False,
            "evaluator": inspect.getsource(self.evaluate),
            "config": extra_config,
        }

        return config


if __name__ == "__main__":
    kt = Kerneltuner()
    for key, value in kt.get_config().items():
        print(f"------------------------------{key}------------------------------")
        print(value)
