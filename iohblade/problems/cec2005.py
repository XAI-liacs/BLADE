import json
import math
import random
from pathlib import Path
from typing import Any

import ioh
import numpy as np
import pandas as pd
from ioh import wrap_problem
from ioh import logger as ioh_logger

from ..problem import BASE_DEPENDENCIES, Problem
from ..solution import Solution
from ..utils import OverBudgetException, aoc_logger, correct_aoc
from .hlp import (
    FEATURE_DESCRIPTIONS,
    MOD_CMAES_GUIDELINES,
    MOD_CMAES_RULES_GLOSSARY,
    RULES_BY_HIGHLEVEL_PROPERTIES_30D,
)

SCORES_PATH = (
    Path(__file__).resolve().parent / "generated_problems" / "cec2005_hlp_scores.json"
)

# Per-function (lb, ub) bounds, read once from opfunu.cec_based.cec2005 and hardcoded here
# (bounds are fixed per function id, not runtime data) so that CEC2005._function_bounds
# never needs opfunu in the main process -- see its docstring.
CEC2005_BOUNDS = {
    1: (-100.0, 100.0),
    2: (-100.0, 100.0),
    3: (-100.0, 100.0),
    4: (-100.0, 100.0),
    5: (-100.0, 100.0),
    6: (-100.0, 100.0),
    7: (0.0, 600.0),
    8: (-32.0, 32.0),
    9: (-5.0, 5.0),
    10: (-5.0, 5.0),
    11: (-0.5, 0.5),
    12: (-3.141592653589793, 3.141592653589793),
    13: (-3.0, 1.0),
    14: (-100.0, 100.0),
    15: (-5.0, 5.0),
    16: (-5.0, 5.0),
    17: (-5.0, 5.0),
    18: (-5.0, 5.0),
    19: (-5.0, 5.0),
    20: (-5.0, 5.0),
    21: (-5.0, 5.0),
    22: (-5.0, 5.0),
    23: (-5.0, 5.0),
    24: (-5.0, 5.0),
    25: (2.0, 5.0),
}

# KNOWN ISSUE (inherited from the CEC2015 scoring run -- see iohblade.problems.cec2015):
# in the LLaMEA new_models/ checkpoints used to compute these scores,
# model_Groups_Basins_ela.json and model_Groups_Multimodality_ela.json are byte-identical
# (same md5), so score_Basins is really just a copy of score_Multimodality for every
# function -- there is currently no independently-trained Basins classifier feeding this
# data. Treat score_Basins (and any "Basins" entry in specific_high_level_features) as
# unverified until a distinct Basins model is trained and the scores are recomputed.

# Order in which measured properties are combined into the same kind of key
# used by RULES_BY_HIGHLEVEL_PROPERTIES_30D (e.g. "Separable_GlobalLocal").
HLP_PROPERTY_ORDER = ["Separable", "GlobalLocal", "Multimodality", "Basins", "Homogeneous"]


def _present_features(scores: dict) -> list:
    """Thresholds measured HLP scores (score_<Property> > 0.5, and the
    boolean `separable` flag) into the list of present high-level features,
    in the same order RULES_BY_HIGHLEVEL_PROPERTIES_30D keys use."""
    present = []
    for feature in HLP_PROPERTY_ORDER:
        if feature == "Separable":
            is_present = bool(scores.get("separable"))
        else:
            is_present = scores.get(f"score_{feature}", 0.0) > 0.5
        if is_present:
            present.append(feature)
    return present


def _feature_strength(feature: str, scores: dict) -> float:
    """A 0-1 confidence value per feature, comparable across the boolean
    `separable` test and the continuous score_<Property> classifiers, used
    to rank features when more than RULES_BY_HIGHLEVEL_PROPERTIES_30D's
    2-property keys are present."""
    if feature == "Separable":
        return (100.0 - scores.get("separable_noncompliance", 100.0)) / 100.0
    return scores.get(f"score_{feature}", 0.0)


def _primary_features(scores: dict, max_features: int = 2) -> list:
    """Present features, capped at `max_features` by keeping only the
    strongest ones (RULES_BY_HIGHLEVEL_PROPERTIES_30D only has keys for 1 or
    2 properties, so a function measured with 3+ present features would
    otherwise never match a rule). Returned in HLP_PROPERTY_ORDER, matching
    how the dict's keys are formatted."""
    present = _present_features(scores)
    if len(present) <= max_features:
        return present
    strongest = sorted(present, key=lambda f: _feature_strength(f, scores), reverse=True)
    strongest = set(strongest[:max_features])
    return [feature for feature in HLP_PROPERTY_ORDER if feature in strongest]


class CEC2005(Problem):
    """
    Problem class for evaluating optimization algorithms on the CEC2005
    "Real-Parameter Optimization" benchmark: 25 noiseless functions, fixed at
    30D, implemented via the `opfunu` package.

    Unlike CEC2015, CEC2005 functions do NOT share a common search space --
    bounds vary per function (e.g. [-100, 100] for the Sphere/Schwefel/Elliptic
    functions, [0, 600] for Griewank, [-32, 32] for Ackley, [-5, 5] for
    Rastrigin/Weierstrass/hybrid-composition functions, etc.), so the actual
    bounds for a given run must be read from the wrapped ioh problem's
    `func.bounds.lb` / `func.bounds.ub` at runtime rather than assumed fixed.

    As with iohblade.problems.cec2015.CEC2015, high-level (HLP) property
    scores were measured offline per function using the same ELA-based
    scoring approach as HLP (see iohblade.problems.hlp), and are stored in
    generated_problems/cec2005_hlp_scores.json. Use `specific_fid` to add a
    single function's measured properties (and, via `add_rules_to_prompt`,
    the matching modular CMA-ES rules from hlp.py) to the prompt.
    """

    DIM = 30  # all 25 CEC2005 functions (via opfunu) support ndim=30; this benchmark fixes 30D.

    def __init__(
        self,
        logger=None,
        training_instances=None,
        test_instances=None,
        name="CEC2005",
        eval_timeout=120,
        budget_factor=2000,
        specific_fid=None,
        add_info_to_prompt=False,
        add_rules_to_prompt=False,
        full_ioh_log=False,
        ioh_dir="",
        dependencies=None,
        imports=None,
    ):
        """
        Initializes the CEC2005 problem instance.
        Args:
            logger (RunLogger): The logger to use for logging.
            training_instances (list): CEC2005 function ids (1-25) to use for training.
            test_instances (list): CEC2005 function ids (1-25) to use for testing.
            name (str): The name of the problem.
            eval_timeout (int): The evaluation timeout in seconds.
            budget_factor (int): The factor to multiply the dimensionality (30) with to get the budget.
            specific_fid (int): The specific CEC2005 function id (1-25) to describe in the prompt. If None,
                no per-function property information is added regardless of add_info_to_prompt/add_rules_to_prompt.
            add_info_to_prompt (bool): If True, adds a description of specific_fid's measured high-level properties.
            add_rules_to_prompt (bool): If True, adds modular CMA-ES rules derived from specific_fid's measured
                properties (reuses iohblade.problems.hlp.RULES_BY_HIGHLEVEL_PROPERTIES_30D).
            full_ioh_log (bool): If set to True, additional IOH logs are being kept for each run and each algorithm.
            dependencies (list, optional): a list of pypi packages to install before evaluation.
            imports (string, optional): the python string to manage imports in the evaluation file.
        """
        if dependencies is None:
            dependencies = [
                "pandas==2.2.3",
                "ioh==0.3.22",
                "opfunu==1.0.1",
                "setuptools<81",  # opfunu imports pkg_resources, removed in setuptools>=81
                "modcma==1.2.0",  # the task/example prompt (mirrored from hlp.py) mandates modcma
            ]
        if imports is None:
            imports = (
                "import numpy as np\nimport ioh\nimport pandas as pd\nimport math\nimport random\n"
            )

        if training_instances is None:
            training_instances = list(range(1, 26))  # all 25 CEC2005 functions
        if test_instances is None:
            test_instances = list(range(1, 26))  # CEC2005 has no natural per-function instance split
        super().__init__(
            logger, training_instances, test_instances, name, eval_timeout, dependencies
        )
        self.budget_factor = budget_factor
        self.full_ioh_log = full_ioh_log
        self.ioh_dir = ioh_dir

        if specific_fid is not None and specific_fid not in range(1, 26):
            raise ValueError(f"specific_fid must be between 1 and 25, got {specific_fid}.")
        self.specific_fid = specific_fid
        self.add_info_to_prompt = add_info_to_prompt
        self.add_rules_to_prompt = add_rules_to_prompt

        with open(SCORES_PATH, "r") as f:
            self.hlp_scores = json.load(f)

        self.func_name = "__call__"
        self.init_inputs = ["budget", "dim"]
        self.func_inputs = ["func"]
        self.func_outputs = ["f_opt", "x_opt"]

        # Properties measured for specific_fid (thresholded score_<Property> > 0.5, plus the
        # boolean `separable` flag) -- the CEC2005 analogue of HLP's user-supplied
        # specific_high_level_features, since here each function's properties are measured,
        # not chosen.
        specific_high_level_features = []
        if self.specific_fid is not None:
            specific_high_level_features = _present_features(self.hlp_scores[str(self.specific_fid)])

        bounds_note = "The search space bounds vary per function -- read them from `func.bounds.lb` / `func.bounds.ub` at runtime rather than assuming a fixed range."
        if self.specific_fid is not None:
            fn_lb, fn_ub = self._function_bounds(self.specific_fid)
            bounds_note = (
                f"This function's search space bounds are {fn_lb:.1f} (lower bound) to {fn_ub:.1f} (upper bound)."
            )

        extra_prompt = f"The optimization algorithm should handle a wide range of tasks, which is evaluated on the CEC2005 set of noiseless functions"
        if self.add_info_to_prompt:
            extra_prompt += ", characterized by the following high-level features: "
            for feature in specific_high_level_features:
                description = FEATURE_DESCRIPTIONS.get(feature, "No description available.")
                extra_prompt += f"\n- {feature}: {description}"
        extra_prompt += "."

        extra_prompt_rules = ""
        if self.add_rules_to_prompt:
            extra_prompt_rules += "\n\nWhen writing the optimization algorithm, consider the following rules derived from known relationships between high-level problem properties and a modular CMA-ES optimization strategy:\n"
            # RULES_BY_HIGHLEVEL_PROPERTIES_30D only has keys for 1 or 2 properties. A
            # measured function can have 3+ properties above threshold at once (unlike
            # HLP's hand-picked combinations), so cap the lookup key at the strongest 2.
            rule_features = (
                _primary_features(self.hlp_scores[str(self.specific_fid)])
                if self.specific_fid is not None
                else []
            )
            key = "_".join(rule_features)
            rules = RULES_BY_HIGHLEVEL_PROPERTIES_30D.get(
                key, "No specific rules available for this combination of high-level features."
            )
            extra_prompt_rules = extra_prompt_rules + rules + "\n"

        self.task_prompt = f"""
You are a Python expert working on a new optimization algorithm. You can use numpy v2 and some other standard libraries.
Your task is to develop a novel heuristic optimization algorithm for continuous optimization problems.
Strictly use the Modular CMA-ES library (modcma) for the optimization algorithm.
Your task is to write the optimization algorithm in Python code.
{bounds_note} The dimensionality is fixed at 30.
The code should contain an `__init__(self, budget, dim)` function with optional additional arguments and the function `def __call__(self, func)`, which should optimize the black box function `func` using `self.budget` function evaluations.
The func() can only be called as many times as the budget allows, not more.

Follow these guidelines whenever you configure the Modular CMA-ES library - for every
algorithm you write or refine, not only the first one:

<mod_cmaes_guidelines>
{MOD_CMAES_GUIDELINES}
</mod_cmaes_guidelines>

{MOD_CMAES_RULES_GLOSSARY}
"""
        self.example_prompt = f"""
{extra_prompt}
{extra_prompt_rules}

An example of the required algorithm structure using the Modular CMA-ES
library is shown below:

```python
import numpy as np
from modcma import c_maes


class ModularCMAESOptimizer:
    def __init__(self, budget=10000, dim=10):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        lower = np.asarray(func.bounds.lb, dtype=float)
        upper = np.asarray(func.bounds.ub, dtype=float)

        x0 = (lower + upper) / 2.0
        sigma0 = 0.3 * float(np.mean(upper - lower))

        x_opt, f_opt, evaluations, optimizer = c_maes.fmin(
            func,
            x0,
            sigma0,
            self.budget,
            active=True,
        )

        return f_opt, x_opt
```

The example demonstrates the required class interface and basic Modular
CMA-ES usage. Do not copy its configuration blindly. Choose the Modular
CMA-ES modules according to the task information and rules supplied above.

Return the answer using the output format specified above.
"""
        self.format_prompt = """
Give an excellent and novel heuristic algorithm to solve this task and also give it a one-line description, describing the main idea. Give the response in the format:
# Description: <short-description>
# Code:
```python
<code>
```
"""

    @staticmethod
    def _function_bounds(instance):
        """Returns (lb, ub) for the given CEC2005 function id.

        Hardcoded from CEC2005_BOUNDS rather than constructed via opfunu, so that
        building a CEC2005 problem with a specific_fid (as __init__ does, to embed
        the exact bounds in the prompt) never requires opfunu in the main process --
        opfunu is only needed inside the isolated per-problem eval subprocess (see
        get_generated_problem), matching HLP/CEC2015's design.
        """
        return CEC2005_BOUNDS[instance]

    def get_generated_problem(self, instance):
        """
        Builds and wraps the opfunu CEC2005 function for the given function id (1-25) at 30D.
        """
        from opfunu.cec_based import cec2005 as opfunu_cec2005

        cls = getattr(opfunu_cec2005, f"F{instance}2005")
        obj = cls(ndim=self.DIM)

        p = wrap_problem(
            obj.evaluate,
            f"CEC2005_F{instance}",
            ioh.ProblemClass.REAL,
            dimension=self.DIM,
            instance=instance,
            calculate_objective=lambda _, dim: (obj.x_global, obj.f_global),
            lb=float(obj.lb[0]),
            ub=float(obj.ub[0]),
        )
        return p

    def get_prompt(self):
        """
        Returns the problem description and answer format.
        """
        return self.task_prompt + self.format_prompt + self.example_prompt

    def evaluate(self, solution: Solution, test=False):
        """
        Evaluates a solution on the CEC2005 benchmark using AOCC.
        """
        code = solution.code
        algorithm_name = solution.name
        algorithm_id = solution.id
        safe_globals = {"np": np, "ioh": ioh, "math": math, "random": random}
        local_env = self._exec_code(code, safe_globals)

        # Small test run to catch code errors
        try:
            l2_temp = aoc_logger(100, upper=1e4, triggers=[ioh_logger.trigger.ALWAYS])
            problem = self.get_generated_problem(self.training_instances[0])
            problem.attach_logger(l2_temp)
            algorithm = local_env[algorithm_name](budget=100, dim=self.DIM)
            algorithm(problem)
        except OverBudgetException:
            pass

        instances = self.test_instances if test else self.training_instances
        aucs = []
        performance_data = []
        budget = self.budget_factor * self.DIM
        for instance in instances:
            f_new = self.get_generated_problem(instance)
            l2 = aoc_logger(budget, upper=1e4, triggers=[ioh_logger.trigger.ALWAYS])
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
                algorithm = local_env[algorithm_name](budget=budget, dim=self.DIM)
                algorithm(f_new)
            except OverBudgetException:
                pass

            corrected_aoc = correct_aoc(f_new, l2, budget)
            performance_data.append({"fid": instance, "dim": self.DIM, "auc": corrected_aoc})
            aucs.append(corrected_aoc)
            l2.reset(f_new)
            f_new.reset()

        auc_mean = np.mean(aucs)
        solution.add_metadata("performance_data", performance_data)
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
            "dim": self.DIM,
            "training_instances": self.training_instances,
            "test_instances": self.test_instances,
            "budget_factor": self.budget_factor,
            "specific_fid": self.specific_fid,
        }

    def get_config(self) -> dict[str, Any]:
        """
        * Return a dictionary of properties to log:
            ```
                {
                    `tags`: list[str],
                    `name`: str,
                    `prompt`: str,
                    `minimisation`: bool,
                    `evaluator`: str,
                    `config`: {}    Extra configuration for a problem; like HPO.
                }
            ```
        """
        return {
            "tags": ["optimization", "continuous", "cec2005"],
            "name": self.name,
            "prompt": self.get_prompt(),
            "minimisation": True,
            "evaluator": "CEC2005",
            "config": {
                "dim": self.DIM,
                "budget_factor": self.budget_factor,
                "specific_fid": self.specific_fid,
                "add_info_to_prompt": self.add_info_to_prompt,
                "add_rules_to_prompt": self.add_rules_to_prompt,
                "full_ioh_log": self.full_ioh_log,
                "ioh_dir": self.ioh_dir,
            },
        }
