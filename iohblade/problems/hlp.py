import os
import traceback

import ioh
import numpy as np
import math
import pandas as pd
import itertools
import random
import json
from ioh import get_problem, wrap_problem
from ioh import logger as ioh_logger


from ..problem import BASE_DEPENDENCIES, Problem
from ..solution import Solution
from ..utils import OverBudgetException, aoc_logger, correct_aoc



# Rulebook-derived summaries for high-level property combinations.
# Priority is descending: earlier (higher-priority) rules override later ones when multiple match.

RULES_BY_HIGHLEVEL_PROPERTIES_5D = {
    "Separable_GlobalLocal": (
        "Known: Separable + GlobalLocal contrast.\n"
        "- If also Many Basins (med/high) AND GlobalLocal is high: disable elitist; use BIPOP restart; PSR step-size.\n"
        "- If additionally Highly Multimodal: treat as 'Separable & Highly Multimodal': disable elitist; use PSR or CSA; "
        "standard mirroring; IPOP or BIPOP; default or half-lambda weights.\n"
        "- If instead Separable & Unimodal: enable elitist; CSA or PSR; no or standard mirroring; default/half-lambda; "
        "avoid equal weights and mirrored pairwise sampling.\n"
        "Notes: if multiple apply, prefer the 'many basins + high global-local' recommendation first."
    ),

    "Separable_Multimodality": (
        "Known: Separable + Multimodality (assume high).\n"
        "- Use the general 'Separable & Highly Multimodal' rule: elitist disabled; PSR or CSA step-size; "
        "standard mirroring; IPOP or BIPOP restart; default or half-lambda weights.\n"
        "- Avoid: elitist (enabled).\n"
        "- If global structure is known to be Strong + separable + highly multimodal: prefer BIPOP + PSR + "
        "standard mirroring + default/half-lambda with elitist disabled.\n"
        "- If global structure is Weak/None and highly multimodal: elitist disabled; BIPOP or IPOP; CSA or PSR; "
        "standard mirroring; default/half-lambda; avoid no-restart."
    ),

    "Separable_Basins": (
        "Known: Separable + Basins (assume med/high basin-related difficulty signal).\n"
        "- If also GlobalLocal contrast is high: disable elitist; use BIPOP restart; PSR step-size.\n"
        "- If additionally Highly Multimodal: disable elitist; PSR or CSA; standard mirroring; IPOP or BIPOP; "
        "default/half-lambda.\n"
        "- If instead Unimodal separable: enable elitist; CSA or PSR; no or standard mirroring; default/half-lambda; "
        "avoid equal weights and mirrored pairwise sampling."
    ),

    "Separable_Homogeneous": (
        "Known: Separable + Homogeneous.\n"
        "- If Funnel-like AND Homogeneous: enable elitist; default or half-lambda weights; avoid equal weights.\n"
        "- If also Unimodal separable: enable elitist; CSA or PSR; no or standard mirroring; default/half-lambda; "
        "avoid equal weights and mirrored pairwise sampling.\n"
        "- If instead Highly Multimodal (despite homogeneity): prefer the 'Separable & Highly Multimodal' package "
        "(elitist disabled; restarts; standard mirroring; PSR/CSA; default/half-lambda)."
    ),

    "Separable": (
        "Known: Separable (other properties unknown).\n"
        "- If Unimodal: enable elitist; CSA or PSR; no or standard mirroring; default/half-lambda; "
        "avoid equal weights and mirrored pairwise sampling.\n"
        "- If Highly Multimodal: disable elitist; PSR or CSA; standard mirroring; IPOP or BIPOP restart; "
        "default/half-lambda.\n"
        "- If Ill-conditioned / high scaling is also present: prefer CSA; default/half-lambda; elitist on; "
        "avoid equal weights.\n"
        "Interpretation: pick the unimodal vs multimodal branch based on what you know; otherwise start with the "
        "safer defaults (default/half-lambda; CSA/PSR; avoid equal weights)."
    ),

    "GlobalLocal_Multimodality": (
        "Known: GlobalLocal contrast + Multimodality (assume high).\n"
        "- If Many Basins (med/high) AND GlobalLocal is high: disable elitist; BIPOP restart; PSR step-size.\n"
        "- If global structure is Weak/None (highly multimodal): disable elitist; BIPOP or IPOP; CSA or PSR; "
        "standard mirroring; default/half-lambda; avoid no-restart.\n"
        "- If global structure is Strong AND separability is high: disable elitist; BIPOP; PSR; standard mirroring; "
        "default/half-lambda."
    ),

    "GlobalLocal_Basins": (
        "Known: GlobalLocal contrast + Basins (assume basins med/high).\n"
        "- Direct hit: (Basins med/high AND GlobalLocal high) => disable elitist; use BIPOP restart; PSR step-size.\n"
        "- If additionally Highly Multimodal with weak/irregular structure: also enforce restarts (BIPOP/IPOP), "
        "standard mirroring, default/half-lambda, and keep elitist disabled."
    ),

    "GlobalLocal_Homogeneous": (
        "Known: GlobalLocal contrast + Homogeneous.\n"
        "- If Funnel-like & Homogeneous: enable elitist; default/half-lambda; avoid equal weights.\n"
        "- If also Many Basins and GlobalLocal is high: disable elitist; BIPOP restart; PSR step-size "
        "(this can override the funnel-like preference when both match).\n"
        "- If Highly Multimodal and structure weak/none: disable elitist; BIPOP/IPOP; CSA/PSR; standard mirroring; "
        "default/half-lambda."
    ),

    "GlobalLocal": (
        "Known: GlobalLocal contrast.\n"
        "- If Many Basins (med/high) and GlobalLocal is high: disable elitist; use BIPOP restart; PSR step-size.\n"
        "- If structure is Deceptive: *opposite direction* => enable elitist; CSA; IPOP or BIPOP; avoid elitist-disabled.\n"
        "Rule of thumb: GlobalLocal-high tends to push toward (elitist off + PSR + restarts), unless structure is "
        "explicitly deceptive."
    ),

    "Multimodality_Basins": (
        "Known: Multimodality (assume high) + Basins (assume med/high).\n"
        "- If also GlobalLocal high: disable elitist; BIPOP restart; PSR step-size (highest priority among these).\n"
        "- Otherwise treat as highly multimodal with potentially irregular structure: disable elitist; "
        "BIPOP or IPOP; CSA or PSR; standard mirroring; default/half-lambda; avoid no-restart."
    ),

    "Multimodality_Homogeneous": (
        "Known: Multimodality (assume high) + Homogeneous.\n"
        "- If Funnel-like: enable elitist; default/half-lambda; avoid equal weights.\n"
        "- If structure is Weak/None (highly multimodal): disable elitist; BIPOP or IPOP; CSA or PSR; "
        "standard mirroring; default/half-lambda; avoid no-restart.\n"
        "Practical: if you don’t know funnel-ness/structure, start with the multimodal package (restarts + "
        "standard mirroring + default/half-lambda + PSR/CSA, elitist off)."
    ),

    "Multimodality": (
        "Known: Multimodality (assume high).\n"
        "- If global structure is Weak/None OR irregular: disable elitist; BIPOP or IPOP restart; CSA or PSR; "
        "standard mirroring; default/half-lambda; avoid no-restart.\n"
        "- If global structure is Strong and separable: disable elitist; BIPOP; PSR; standard mirroring; "
        "default/half-lambda.\n"
        "- If separable but structure unknown: apply 'Separable & Highly Multimodal' package (elitist off; restarts; "
        "standard mirroring; PSR/CSA; default/half-lambda)."
    ),

    "Basins_Homogeneous": (
        "Known: Basins (assume med/high) + Homogeneous.\n"
        "- If Funnel-like & Homogeneous: enable elitist; default/half-lambda; avoid equal weights.\n"
        "- If also GlobalLocal high: disable elitist; BIPOP restart; PSR step-size (overrides funnel-like if both match).\n"
        "If you only know basins+homogeneity, the only explicit homogeneity rule is the funnel-like one; otherwise fall back "
        "to conservative defaults (default/half-lambda; avoid equal weights)."
    ),

    "Basins": (
        "Known: Basins (assume med/high).\n"
        "- If also GlobalLocal high: disable elitist; use BIPOP restart; PSR step-size.\n"
        "- If structure is Deceptive: enable elitist; CSA; IPOP or BIPOP.\n"
        "If you only know 'basins' without GlobalLocal/structure, the rulebook doesn't specify more; start with "
        "default/half-lambda weights and prefer PSR/CSA depending on conditioning."
    ),

    "Homogeneous": (
        "Known: Homogeneous.\n"
        "- If Funnel-like & Homogeneous: enable elitist; default or half-lambda weights; avoid equal weights.\n"
        "Otherwise, the rulebook has no additional homogeneous-only prescriptions; keep weights non-equal "
        "(default/half-lambda) as a safe choice."
    ),
}

RULES_BY_HIGHLEVEL_PROPERTIES_30D = {
    "Separable_GlobalLocal": (
        "Known: Separable + GlobalLocal.\n"
        "- Prefer elitist selection enabled.\n"
        "- Use PSR step-size adaptation as default; switch to CSA only if strong scaling issues are known.\n"
        "- Use standard mirroring.\n"
        "- Use BIPOP restart strategy.\n"
        "- Use default or half-lambda weights.\n"
        "- Use Gaussian or Sobol sampling.\n"
        "- Avoid equal weights, no mirroring, Halton sampling, and disabling elitism."
    ),

    "Separable_Multimodality": (
        "Known: Separable + Multimodal.\n"
        "- If the global structure is strong: enable elitism and consider equal weights.\n"
        "- Otherwise: disable elitism.\n"
        "- Use BIPOP restarts.\n"
        "- Use PSR by default; switch to CSA only if scaling is clearly problematic.\n"
        "- Use standard mirroring.\n"
        "- Prefer Gaussian or Sobol sampling.\n"
        "- Avoid no-restart strategies and no mirroring."
    ),

    "Separable_Basins": (
        "Known: Separable + Basins.\n"
        "- Basins are not decisive at 30D; follow separable defaults.\n"
        "- Enable elitism unless clear multimodality is observed.\n"
        "- Use PSR step-size adaptation.\n"
        "- Use standard mirroring and BIPOP restarts.\n"
        "- Prefer default or half-lambda weights.\n"
        "- Use Gaussian or Sobol sampling."
    ),

    "Separable_Homogeneous": (
        "Known: Separable + Homogeneous.\n"
        "- Enable elitist selection.\n"
        "- Use PSR step-size adaptation; CSA only if scaling is clearly high.\n"
        "- Use standard mirroring.\n"
        "- Use BIPOP restarts.\n"
        "- Prefer default or half-lambda weights.\n"
        "- Homogeneity mainly supports keeping elitism enabled."
    ),

    "Separable": (
        "Known: Separable.\n"
        "- Default to elitist selection enabled.\n"
        "- Use PSR step-size adaptation.\n"
        "- Use standard mirroring.\n"
        "- Use BIPOP restarts.\n"
        "- Prefer default or half-lambda weights.\n"
        "- Switch to CSA only if scaling is high.\n"
        "- If strong multimodality is later detected, consider disabling elitism."
    ),

    "GlobalLocal_Multimodality": (
        "Known: GlobalLocal contrast + Multimodal.\n"
        "- Treat primarily as a multimodal problem.\n"
        "- Disable elitist selection unless funnel-like structure is evident.\n"
        "- Use BIPOP restarts.\n"
        "- Use PSR step-size adaptation; CSA only if scaling is high.\n"
        "- Use standard mirroring.\n"
        "- Prefer Gaussian or Sobol sampling."
    ),

    "GlobalLocal_Basins": (
        "Known: GlobalLocal contrast + Basins.\n"
        "- Basins are not decisive at 30D; focus on global-local contrast only if it is low.\n"
        "- If global-local contrast is low and multimodality is low: enable elitism and use PSR.\n"
        "- Otherwise use default settings: PSR, standard mirroring, Gaussian sampling.\n"
        "- Use restarts if stagnation is observed."
    ),

    "GlobalLocal_Homogeneous": (
        "Known: GlobalLocal contrast + Homogeneous.\n"
        "- Homogeneity supports elitism as a safe default.\n"
        "- Use PSR step-size adaptation.\n"
        "- Use standard mirroring.\n"
        "- Prefer Gaussian or Sobol sampling.\n"
        "- Only change strategy if strong multimodality or scaling issues are observed."
    ),

    "GlobalLocal": (
        "Known: GlobalLocal contrast.\n"
        "- If global-local contrast is low and multimodality is low: enable elitism and use PSR.\n"
        "- Otherwise fall back to generic defaults.\n"
        "- Use standard mirroring and Gaussian sampling.\n"
        "- Introduce restarts if convergence stalls."
    ),

    "Multimodality_Basins": (
        "Known: Multimodal + Basins.\n"
        "- Treat as a multimodal problem; basins do not add decisive guidance at 30D.\n"
        "- Disable elitism unless a clear funnel structure is known.\n"
        "- Use BIPOP restarts.\n"
        "- Use PSR step-size adaptation.\n"
        "- Use standard mirroring.\n"
        "- Prefer Gaussian or Sobol sampling."
    ),

    "Multimodality_Homogeneous": (
        "Known: Multimodal + Homogeneous.\n"
        "- Disable elitism by default.\n"
        "- Homogeneity can justify enabling elitism later if convergence is stable.\n"
        "- Use BIPOP restarts.\n"
        "- Use PSR step-size adaptation; CSA only if scaling is high.\n"
        "- Use standard mirroring."
    ),

    "Multimodality": (
        "Known: Multimodal.\n"
        "- Disable elitist selection by default.\n"
        "- Use BIPOP restarts.\n"
        "- Use PSR step-size adaptation.\n"
        "- Use standard mirroring.\n"
        "- Switch to CSA only if scaling is high.\n"
        "- Enable elitism only if strong funnel-like structure is observed."
    ),

    "Basins_Homogeneous": (
        "Known: Basins + Homogeneous.\n"
        "- These features are weakly informative at 30D.\n"
        "- Use default CMA-ES configuration: PSR, standard mirroring, Gaussian sampling.\n"
        "- Enable elitism cautiously if convergence appears stable."
    ),

    "Basins": (
        "Known: Basins.\n"
        "- Basins alone do not drive decisions at 30D.\n"
        "- Use default settings: PSR, standard mirroring, Gaussian sampling.\n"
        "- Introduce restarts if stagnation occurs."
    ),

    "Homogeneous": (
        "Known: Homogeneous.\n"
        "- Homogeneity mainly acts as a stabilizing signal.\n"
        "- Enable elitist selection as a safe default.\n"
        "- Use PSR step-size adaptation.\n"
        "- Use standard mirroring.\n"
        "- Prefer Gaussian sampling."
    ),
}


class HLP(Problem):
    """
    Problem class for evaluating optimization algorithms on the High Level Property - HLP benchmark. 
    """

    def __init__(
        self,
        logger=None,
        training_instances=None,
        test_instances=None,
        name="HLP",
        eval_timeout=120,
        dim=5,
        budget_factor=2000,
        specific_high_level_features=[],
        add_info_to_prompt=False,
        add_rules_to_prompt=False,
        full_ioh_log=False,
        ioh_dir="",
        dependencies=None,
        imports=None,
    ):
        """
        Initializes the MA-BBOB problem instance.
        Args:
            logger (RunLogger): The logger to use for logging.
            training_instances (list): A list of indices for training instances to use.
            test_instances (list): The indices of test instances to use. A list of indices.
            name (str): The name of the problem.
            eval_timeout (int): The evaluation timeout in seconds.
            dim (int): The dimensionality of the problem instance to run on.
            budget_factor (int): The factor to multiply the dimensionality with to get the budget.
            specific_high_level_features (list): The specific high level features ("basins","seperable" etc) to use.
            add_info_to_prompt (bool): If set to True, additional information about the high-level features will be added to the prompt.
            add_rules_to_prompt (bool): If set to True, additional rules for writing the optimization algorithm will be added to the prompt.
            full_ioh_log (bool): If set to True, additional IOH logs are being kept for each run and each algorithm.
            dependencies (list, optional): a list of pypi packages to install before evaluation.
            imports (string, optional): the python string to manage imports in the evaluation file.
        """
        if dependencies is None:
            dependencies = [
                "pandas==2.2.3",
                "ioh==0.3.22",
            ]
        if imports is None:
            imports = (
                "import numpy as np\nimport ioh\nimport pandas as pd\nimport math\nimport random\n"
            )

        self.function_files = [
            "gpt-5-nano-ELA-Basins_Homogeneous.jsonl",
            "gpt-5-nano-ELA-Basins.jsonl",
            "gpt-5-nano-ELA-GlobalLocal_Basins.jsonl",
            "gpt-5-nano-ELA-GlobalLocal_Homogeneous.jsonl",
            "gpt-5-nano-ELA-GlobalLocal_Multimodality.jsonl",
            "gpt-5-nano-ELA-GlobalLocal.jsonl",
            "gpt-5-nano-ELA-Homogeneous.jsonl",
            "gpt-5-nano-ELA-Multimodality_Basins.jsonl",
            "gpt-5-nano-ELA-Multimodality.jsonl",
            "gpt-5-nano-ELA-NOT Basins_GlobalLocal.jsonl",
            "gpt-5-nano-ELA-NOT Basins_Multimodality.jsonl",
            "gpt-5-nano-ELA-NOT Basins_Separable.jsonl",
            "gpt-5-nano-ELA-NOT Basins.jsonl",
            "gpt-5-nano-ELA-NOT Homogeneous_GlobalLocal.jsonl",
            "gpt-5-nano-ELA-NOT Homogeneous_Multimodality.jsonl",
            "gpt-5-nano-ELA-NOT Homogeneous_Separable.jsonl",
            "gpt-5-nano-ELA-NOT Homogeneous.jsonl",
            "gpt-5-nano-ELA-Separable_Basins.jsonl",
            "gpt-5-nano-ELA-Separable_GlobalLocal.jsonl",
            "gpt-5-nano-ELA-Separable_Homogeneous.jsonl",
            "gpt-5-nano-ELA-Separable_Multimodality.jsonl",
            "gpt-5-nano-ELA-Separable.jsonl",
            "gpt-5-nano-ELA-Multimodality_Homogeneous.jsonl"
        ]

        self.feature_descriptions = {
            "Basins": "The functions to optimize have basin size homogeneity, meaning the size relation (largest to smallest) of all basins of attraction should be homogeneous.",
            "Separable": "The functions to optimize are separable, meaning independent functions per dimension. Meaning, a problem may be partitioned into subproblems which are then of lower dimensionality and should be considerably easier to solve.",
            "GlobalLocal": "The functions should have a global local minima contrast, which refers to the difference between global and local peaks in comparison to the average fitness level of a problem. It thus determines if very good peaks are easily recognized as such.",
            "Multimodality": "The functions are multimodal, Multimodality refers to the number of local minima of a problem.",
            "Structure": "The functions have a clear global structure. Global structure is what remains after deleting all non-optimal points.",
            "Homogeneous": "The functions have a homogeneous search space. Which refers to a search space without phase transitions. Its overall appearance is similar in different search space areas.",
            "NOT Homogeneous": "The functions have a non-homogeneous search space. Which refers to a search space with phase transitions. Its overall appearance is different in different search space areas.",
            "NOT Basins": "The functions do not have basin size homogeneity. Which refers to a search space where the size relation (largest to smallest) of all basins of attraction is not homogeneous.",
        }

        self.add_info_to_prompt = add_info_to_prompt
        self.add_rules_to_prompt = add_rules_to_prompt

        if training_instances is None:
            training_instances = list(range(10)) # 10 training instances
        if test_instances is None:
            test_instances = list(range(10,20))  # 10 test instances
        super().__init__(
            logger, training_instances, test_instances, name, eval_timeout, dependencies
        )
        self.dim = dim  # The dimensionality of the problem instance to run on
        self.budget_factor = budget_factor  # The factor to multiply the dimensionality with to get the budget
        allowed_features = "Basins,Separable,Multimodality,Homogeneous,GlobalLocal,NOT Homogeneous,NOT Basins".split(",")
        for specific_high_level_feature in specific_high_level_features:
            if specific_high_level_feature not in allowed_features:
                raise ValueError(f"specific_high_level_features includes a feature ({specific_high_level_feature}) that is not from the allowed list: {allowed_features}.")
        if len(specific_high_level_features) == 1:
            self.function_file = f"gpt-5-nano-ELA-{specific_high_level_features[0]}.jsonl"
        elif len(specific_high_level_features) > 1:
            features_str = "_".join(specific_high_level_features)
            self.function_file = f"gpt-5-nano-ELA-{features_str}.jsonl"
        if self.function_file not in self.function_files:
            raise ValueError(f"The combination of specific_high_level_features {specific_high_level_features} {self.function_file} does not correspond to a valid function file.")
        self.specific_high_level_features = specific_high_level_features
        self.full_ioh_log = full_ioh_log
        self.ioh_dir = ioh_dir

        self.func_name = "__call__"
        self.init_inputs = ["budget", "dim"]
        self.func_inputs = ["func"]
        self.func_outputs = ["f_opt", "x_opt"]

        extra_prompt = f"The optimization algorithm should handle a wide range of tasks, which is evaluated on a set of noiseless functions"
        if self.add_info_to_prompt:
            extra_prompt += ", characterized by the following high-level features: "
            for feature in specific_high_level_features:
                description = self.feature_descriptions.get(feature, "No description available.")
                extra_prompt += f"\n- {feature}: {description}"
        extra_prompt += "."

        extra_prompt_rules = ""
        if self.add_rules_to_prompt:
            extra_prompt_rules += "\n\nWhen writing the optimization algorithm, please consider the following rules derived from known relationships between high-level problem properties and a modular CMA-ES optimization strategy:\n"
            key = "_".join(specific_high_level_features)
            if dim < 10:
                rules = RULES_BY_HIGHLEVEL_PROPERTIES_5D.get(key, "No specific rules available for this combination of high-level features.")
            else:
                rules = RULES_BY_HIGHLEVEL_PROPERTIES_30D.get(key, "No specific rules available for this combination of high-level features.")
            extra_prompt_rules += rules

        self.task_prompt = f"""
You are a Python expert working on a new optimization algorithm. You can use numpy v2 and some other standard libraries.
Your task is to develop a novel heuristic optimization algorithm for continuous optimization problems.
{extra_prompt} Your task is to write the optimization algorithm in Python code. 
Each of the optimization functions has a search space between -5.0 (lower bound) and 5.0 (upper bound). The dimensionality can be varied.
{extra_prompt_rules}
The code should contain an `__init__(self, budget, dim)` function with optional additional arguments and the function `def __call__(self, func)`, which should optimize the black box function `func` using `self.budget` function evaluations.
The func() can only be called as many times as the budget allows, not more. 
"""
        self.example_prompt = """
An example of such code (a simple random search), is as follows:
```python
import numpy as np
import math

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

    def get_generated_problem(self, instance, dim):
        # --- LOAD JSON LINES ---
        current_dir = os.path.dirname(os.path.abspath(__file__))
        with open(current_dir + "/generated_problems/" + self.function_file, "r") as f:
            data = [json.loads(line) for line in f if line.strip()]

        n = len(data)
        if instance >= n:
            raise ValueError(f"Instance index {instance} is out of range for available functions ({n} functions).")
        
        entry = data[instance]
        code = entry["code"]

        safe_globals = {"np": np, "ioh": ioh, "math": math, "itertools": itertools, "random": random}
        local_env = {}
        exec(code, safe_globals, local_env)
        cls = local_env[entry["name"]]
        objective_f = cls(dim=dim).f

        # --- WRAP PROBLEM ---
        p = wrap_problem(
            objective_f,
            self.function_file.replace(".jsonl", ""),
            ioh.ProblemClass.REAL,
            dimension=dim,
            instance=instance,
            calculate_objective=lambda _, dim: (
                entry[f"x_opt_dim_{dim}"],
                entry[f"f_opt_dim_{dim}"]
            ),
            lb=-5,
            ub=5,
        )
        return p


    def get_prompt(self):
        """
        Returns the problem description and answer format.
        """
        return self.task_prompt + self.example_prompt + self.format_prompt

    def evaluate(self, solution: Solution, test=False):
        """
        Evaluates a solution on the randomly generated benchmark function using AOCC.
        """
        auc_mean = 0
        code = solution.code
        algorithm_name = solution.name
        algorithm_id = solution.id
        safe_globals = {"np": np, "ioh": ioh, "math": math, "itertools": itertools, "random": random}
        local_env = {}
        exec(code, safe_globals, local_env)

        algorithm = None

        # Small test run to catch code errors
        try:
            l2_temp = aoc_logger(100, upper=1e2, triggers=[ioh_logger.trigger.ALWAYS])
            problem = get_problem(
                1, instance=1, dimension=2, problem_class=ioh.ProblemClass.BBOB
            )
            problem.attach_logger(l2_temp)
            algorithm = local_env[algorithm_name](budget=100, dim=2)
            algorithm(problem)
        except OverBudgetException:
            pass

        # Final validation
        instances = self.test_instances if test else self.training_instances
        aucs = []
        performance_data = []
        for dim in [self.dim]:
            for instance in instances:
                budget = self.budget_factor * dim
                f_new = self.get_generated_problem(
                    instance=instance, dim=dim
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
                corrected_aoc = correct_aoc(f_new, l2, budget)
                performance_data.append(
                    {"fid": self.function_file.replace(".jsonl", ""), "iid": instance, "dim": dim, "auc": corrected_aoc}
                )
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
            "dim": self.dim,
            "training_instances": self.training_instances,
            "test_instances": self.test_instances,
            "budget_factor": self.budget_factor,
        }
