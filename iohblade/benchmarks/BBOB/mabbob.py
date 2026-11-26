import inspect
import os
import random
import textwrap
import traceback
from typing import TYPE_CHECKING, Any

import ioh
import numpy as np
import pandas as pd
from ioh import get_problem
from ioh import logger as ioh_logger

from iohblade.benchmarks.blackbox import BlackBoxFunction
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
from iohblade.utils import OverBudgetException, aoc_logger, correct_aoc

if TYPE_CHECKING:  # ConfigSpace is only installed for HPO (iohblade.hpo)
    from ConfigSpace import Configuration


class MA_BBOB(Problem):
    """
    Problem class for evaluating optimization algorithms on the MA-BBOB benchmark.

    Supports hyperparameter tuning (HPO) of solutions with a configuration
    space, see ``hpo`` and :mod:`iohblade.hpo`.
    """

    supports_hpo = True

    def __init__(
        self,
        logger=None,
        training_instances=None,
        test_instances=None,
        name="MA_BBOB",
        eval_timeout=120,
        dims=[2, 5],
        budget_factor=2000,
        dependencies=None,
        imports=None,
        hpo: bool = False,
        hpo_trials: int = 50,
        hpo_instances: int | None = 12,
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
                tuned solution, each one run of the algorithm on one
                (instance, dimension) of the tuning instances with the usual
                budget (``budget_factor * dim``). A training evaluation is
                only 40 runs by default (20 instances in 2 dimensions), so
                the default is 50 (half of AutoML's 100): tuning adds about
                1.25 training evaluations plus SMAC's own overhead
                (typically tens of seconds), so a tuned evaluation costs
                roughly 2.5 untuned ones. Consider raising ``eval_timeout``
                (default 60 s) with HPO.
            hpo_instances (int | None): Tune on a fixed, seeded subset of
                this many (instance, dimension) combinations of the training
                instances (default 12, as AutoML; ``None``: all of them).
        """
        if dependencies is None:
            dependencies = [
                "pandas==2.2.3",
                "ioh==0.3.22",
                # smac and ConfigSpace are only installed with HPO, see
                # enable_hpo and iohblade.hpo.HPO_DEPENDENCIES.
            ]
        if imports is None:
            imports = "import numpy as np\nimport ioh\nimport math\n"

        if training_instances is None:
            training_instances = range(0, 20)
        if test_instances is None:
            test_instances = range(20, 120)
        super().__init__(
            logger,
            training_instances,
            test_instances,
            name,
            eval_timeout,
            dependencies,
            imports,
        )
        self.dims = dims  # The dimensionalities of the problem instances to run on
        self.budget_factor = budget_factor  # The factor to multiply the dimensionality with to get the budget
        self.hpo_trials = int(hpo_trials)
        self.hpo_instances = hpo_instances
        if hpo:
            self.enable_hpo(True)
        self.func_name = "__call__"
        self.init_inputs = ["budget", "dim"]
        self.func_inputs = ["func"]
        self.func_outputs = ["f_opt", "x_opt"]
        self.task_prompt = textwrap.dedent("""
You are a Python developer working on a new optimization algorithm.
Your task is to develop a novel heuristic optimization algorithm for continuous optimization problems.
The optimization algorithm should handle a wide range of tasks, which is evaluated on a test suite of noiseless functions. Your task is to write the optimization algorithm in Python code.
Each of the optimization functions has a search space between -5.0 (lower bound) and 5.0 (upper bound). The dimensionality can be varied.
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

    def _run_once(self, alg_cls, idx, dim, config, analyzer=None):
        """Run the algorithm once on one MA-BBOB instance.

        Args:
            alg_cls: The class of the generated algorithm.
            idx (int): The MA-BBOB instance index.
            dim (int): The dimensionality.
            config (dict): Hyperparameters, passed to the constructor as
                keyword arguments.
            analyzer (tuple[str, str], optional): ``(root, name)`` to also
                log the run with an IOH Analyzer logger.

        Returns:
            float: The normalised AOCC of the run (higher is better).
        """
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
        l1 = None
        if analyzer is not None:
            root, name = analyzer
            l1 = ioh.logger.Analyzer(root=root, folder_name=name, algorithm_name=name)
            # Keep a reference: ioh does not keep the Combine alive, and
            # the run crashes (segfault) once it is garbage collected.
            combined_logger = ioh.logger.Combine([l1, l2])
            f_new.attach_logger(combined_logger)
        else:
            f_new.attach_logger(l2)

        try:
            algorithm = alg_cls(budget=budget, dim=dim, **config)
            algorithm(BlackBoxFunction(f_new))
        except OverBudgetException:
            # Same as BBOB_SBOX: the run is scored on the evaluations
            # within the budget, and the remaining instances are still
            # evaluated so that every instance counts in the mean.
            pass

        aocc = correct_aoc(f_new, l2, budget)
        l2.reset(f_new)
        f_new.reset()
        if l1 is not None:
            # Analyzer.reset() takes no problem (the old l1.reset(f_new)
            # raised a TypeError); closing it writes its files.
            l1.close()
        return aocc

    def _tune(self, alg_cls, solution: Solution) -> dict[str, Any]:
        """Tune the solution's configuration space with SMAC.

        The cost of a configuration on one (instance, dimension) is
        ``1 - AOCC`` of one run (SMAC minimises; the fitness, mean AOCC, is
        maximised).

        Returns:
            dict: The incumbent configuration.
        """
        combos = [(int(i), d) for d in self.dims for i in self.training_instances]
        combos = hpo_instance_subset(combos, self.hpo_instances)
        instance_ids = [f"m{i}_d{d}" for (i, d) in combos]
        features = {
            iid: [float(i), float(d)] for iid, (i, d) in zip(instance_ids, combos)
        }

        def target(cfg: "Configuration", instance: str, seed: int = 0) -> float:
            idx, dim = (int(part[1:]) for part in instance.split("_"))
            np.random.seed(seed)  # deterministic=True: one run per instance
            random.seed(seed)
            try:
                aocc = self._run_once(alg_cls, idx, dim, plain_config(cfg))
            except Exception:
                return 1.0  # the worst cost: a failing configuration
            return float(min(1.0, max(0.0, 1.0 - aocc)))

        incumbent = run_smac(
            solution.configspace,
            target,
            instances=instance_ids,
            instance_features=features,
            n_trials=self.hpo_trials,
            output_dir=smac_output_dir(self),
            deterministic=True,
            name=f"{self.name}-{solution.id}",
        )
        return plain_config(incumbent)

    def evaluate(self, solution: Solution, test=False, ioh_dir=""):
        """
        Evaluates a solution on the MA-BBOB benchmark using AOCC.

        With HPO enabled (``hpo``) and a solution with a configuration
        space, the hyperparameters are first tuned with SMAC on a subset of
        the training instances; the incumbent is then scored like any
        solution (mean AOCC on all training instances) and stored in the
        metadata (``incumbent``). On the test instances (``test=True``) the
        solution is not tuned again: the incumbent recorded during the
        search is used (default hyperparameters if there is none).
        """
        cs = getattr(solution, "configspace", None)
        tune = cs is not None and self.hpo and not test
        if tune:
            # Before the generated code is used: a missing smac is a problem
            # of the evaluation environment (a harness error), not of the code.
            import_smac()

        code = solution.code
        algorithm_name = solution.name
        safe_globals = {"np": np}
        local_env = {}
        exec(code, safe_globals, local_env)
        alg_cls = local_env[algorithm_name]

        # Small test run to catch code errors
        try:
            l2_temp = aoc_logger(100, upper=1e2, triggers=[ioh_logger.trigger.ALWAYS])
            problem = get_problem(11, 1, 2)
            problem.attach_logger(l2_temp)
            algorithm = alg_cls(budget=100, dim=2)
            algorithm(BlackBoxFunction(problem))
        except OverBudgetException:
            pass

        incumbent = None
        if tune:
            incumbent = self._tune(alg_cls, solution)
            solution.add_metadata("incumbent", incumbent)
        config, hpo_note = evaluation_config(self, solution, test, incumbent)

        # Final validation
        instances = self.test_instances if test else self.training_instances
        analyzer = (ioh_dir, algorithm_name) if test else None
        aucs = []
        for dim in self.dims:
            for idx in instances:
                aucs.append(
                    self._run_once(alg_cls, idx, dim, config, analyzer=analyzer)
                )

        auc_mean = np.mean(aucs)

        solution.add_metadata("aucs", aucs)
        solution.set_scores(
            auc_mean,
            f"The algorithm {algorithm_name} scored {auc_mean:.3f} on AOCC (higher is better, 1.0 is the best)."
            + (f"\n{hpo_note}" if hpo_note else ""),
        )

        return solution

    def test(self, solution: Solution, ioh_dir=""):
        """
        Runs the solution on test instances and returns the fitness score.

        Uses the HPO incumbent of the search (``incumbent`` in the
        solution's metadata) when there is one; the solution is not tuned
        on the test instances.
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
            NoiseType,
            ObjectiveType,
            PrimaryCategories,
            VariableType,
        )

        extra_config = self.to_dict()
        extra_config.pop("name")
        extra_config["imports"] = self.imports
        extra_config["dependencies"] = self.dependencies
        tags: list[Any] = [PrimaryCategories.BBO, PrimaryCategories.CO]
        tags.append(NoiseType.NOISELESS)
        tags.append(ObjectiveType.SINGLE_OBJECTIVE)
        tags.append(VariableType.CONTINUOUS)

        config = {
            "tags": tags,
            "name": self.name,
            "prompt": self.get_prompt(),
            "minimisation": False,
            "evaluator": inspect.getsource(self.evaluate),
            "config": extra_config,
        }
        return config


if __name__ == "__main__":
    bbob = MA_BBOB()
    for key, value in bbob.get_config().items():
        print(f"------------------------------{key}------------------------------")
        print(value)
