import os
import traceback

import ioh
import numpy as np
import math
import pandas as pd
import json
from ioh import get_problem, wrap_problem
from ioh import logger as ioh_logger


from ..problem import BASE_DEPENDENCIES, Problem
from ..solution import Solution
from ..utils import OverBudgetException, aoc_logger, correct_aoc






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
        dims=[2, 5],
        budget_factor=2000,
        specific_high_level_features=[],
        add_info_to_prompt=False,
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
            dims (list): The dimensionalities of the problem instances to run on.
            budget_factor (int): The factor to multiply the dimensionality with to get the budget.
            specific_high_level_features (list): The specific high level features ("basins","seperable" etc) to use.
            add_info_to_prompt (bool): If set to True, additional information about the high-level features will be added to the prompt.
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
            "gpt5-nano-ELA-Multimodality_Homogeneous.jsonl"
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

        if training_instances is None:
            training_instances = np.arange(10) # 10 training instances
        if test_instances is None:
            test_instances = np.arange(10,20)  # 10 test instances
        super().__init__(
            logger, training_instances, test_instances, name, eval_timeout, dependencies
        )
        self.dims = dims  # The dimensionalities of the problem instances to run on
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
            raise ValueError(f"The combination of specific_high_level_features {specific_high_level_features} does not correspond to a valid function file.")
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
        self.task_prompt = f"""
You are a Python expert working on a new optimization algorithm. You can use numpy v2 and some other standard libraries.
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
        with open(self.function_file, "r") as f:
            data = [json.loads(line) for line in f if line.strip()]

        n = len(data)
        if instance >= n:
            raise ValueError(f"Instance index {instance} is out of range for available functions ({n} functions).")
        
        entry = data[instance]
        code = entry["code"]

        exec(code, globals())
        cls = globals()[entry["name"]]
        objective_f = cls(dim=dim).f


        def calculate_objective(instance, dim):
            return entry[f"global_optimum_{dim}"]

        # --- WRAP PROBLEM ---
        wrap_problem(
            objective_f,
            self.function_file.replace(".jsonl", ""),
            ioh.ProblemClass.REAL,
            dimension=dim,
            instance=instance,
            calculate_objective=calculate_objective,
            lb=-5,
            ub=5,
        )
        return get_problem(self.function_file.replace(".jsonl", ""), instance=instance, dimension=dim)


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
        auc_std = 0
        code = solution.code
        algorithm_name = solution.name
        algorithm_id = solution.id
        safe_globals = {"np": np, "ioh": ioh, "math": math}
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
        for dim in self.dims:
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
            "dims": self.dims,
            "training_instances": self.training_instances,
            "test_instances": self.test_instances,
            "budget_factor": self.budget_factor,
        }
