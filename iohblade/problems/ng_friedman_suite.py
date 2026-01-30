import tempfile
import warnings

import ioh
import iohinspector
import nevergrad as ng
import numpy as np
import polars as pl
from nevergrad.optimization.optimizerlib import BFGS, CMA, DE, PSO, Cobyla
from scipy.stats import rankdata

from ..problem import Problem
from ..solution import Solution

ALGORITHMS = {
    "CMA": CMA,
    "DE": DE,
    "PSO": PSO,
    "BFGS": BFGS,
    "Cobyla": Cobyla,
}


class NG_Evaluator:
    def __init__(self, optimizer: str, budget: int = 2000):
        self.alg = optimizer
        self.budget = budget

    def __call__(self, func):
        parametrization = ng.p.Array(shape=(func.meta_data.n_variables,)).set_bounds(
            -5, 5
        )
        optimizer_cls = ALGORITHMS[self.alg]
        optimizer = optimizer_cls(
            parametrization=parametrization, budget=int(self.budget)
        )
        optimizer.minimize(func)


def run_benchmark(problems, meta_dims, budget, repeats, log_root):
    for name, problem in problems.items():
        dim = meta_dims[name]
        for alg_name in ALGORITHMS:
            prob_wrapped = ioh.wrap_problem(
                problem,
                name,
                ioh.ProblemClass.REAL,
                dim,
                lb=-5,
                ub=5,
            )
            logger = ioh.logger.Analyzer(
                root=log_root,
                folder_name=f"{name}_{alg_name}_logs",
                algorithm_name=alg_name,
            )
            prob_wrapped.attach_logger(logger)
            optimizer = NG_Evaluator(alg_name, budget)
            for _ in range(repeats):
                optimizer(prob_wrapped)
                prob_wrapped.reset()
            logger.close()


def get_friedman_val(dt_perf):
    friedman_ranks = []

    for func_name in dt_perf["function_name"].unique():
        func_data = dt_perf.filter(pl.col("function_name") == func_name)
        max_runs = (
            func_data.group_by("algorithm_name")
            .agg(pl.col("best_y").count())
            .select(pl.col("best_y").max())[0, 0]
        )
        for run_idx in range(max_runs):
            run_scores = []
            run_algs = []

            for alg_name in func_data["algorithm_name"].unique():
                alg_scores = func_data.filter(pl.col("algorithm_name") == alg_name)[
                    "best_y"
                ].to_numpy()
                if run_idx < len(alg_scores):
                    run_scores.append(alg_scores[run_idx])
                    run_algs.append(alg_name)

            if not run_scores:
                continue

            ranks = rankdata(run_scores, method="average")

            for alg_name, rank in zip(run_algs, ranks):
                friedman_ranks.append(
                    {
                        "function_name": func_name,
                        "algorithm_name": alg_name,
                        "run": run_idx,
                        "rank": rank,
                    }
                )

    if not friedman_ranks:
        return float("nan")

    friedman_ranks_df = pl.DataFrame(friedman_ranks)
    friedman_avg_ranks = friedman_ranks_df.group_by(
        ["function_name", "algorithm_name"]
    ).agg(pl.col("rank").mean().alias("rank"))
    friedman_iqr = (
        friedman_avg_ranks.group_by("algorithm_name")
        .agg(
            pl.col("rank").quantile(0.75).alias("q75"),
            pl.col("rank").quantile(0.25).alias("q25"),
        )
        .with_columns((pl.col("q75") - pl.col("q25")).alias("iqr"))
    )
    return friedman_iqr["iqr"].mean()


def _validate_suite(problems, meta_dims):
    if not isinstance(problems, dict) or not isinstance(meta_dims, dict):
        return "problems and meta_dims must be dictionaries."
    if len(problems) != 25 or len(meta_dims) != 25:
        return "problems and meta_dims must each contain 25 entries."
    problem_keys = set(problems.keys())
    meta_keys = set(meta_dims.keys())
    if problem_keys != meta_keys:
        return "problems and meta_dims must share identical keys."
    for name, dim in meta_dims.items():
        if not isinstance(dim, int) or dim <= 0:
            return f"meta_dims[{name!r}] must be a positive integer."
    return None


class NG_FriedmanSuite(Problem):
    """
    Problem class for designing suites that discriminate between optimizers.
    """

    def __init__(
        self,
        logger=None,
        training_instances=None,
        test_instances=None,
        name="NG_FriedmanSuite",
        eval_timeout=600,
        budget=5000,
        repeats=5,
        dependencies=None,
        imports=None,
    ):
        if dependencies is None:
            dependencies = [
                "nevergrad>=1.0.0,<2",
                "ioh==0.3.22",
                "iohinspector>=0.3.0,<1",
                "polars>=1.0.0,<2",
                "scipy>=1.11.0,<2",
            ]
        if imports is None:
            imports = (
                "import numpy as np\n"
                "import ioh\n"
                "import nevergrad as ng\n"
                "import polars as pl\n"
            )
        super().__init__(
            logger, training_instances, test_instances, name, eval_timeout, dependencies
        )
        self.budget = budget
        self.repeats = repeats
        self.imports = imports

        self.func_name = "problems"
        self.init_inputs = []
        self.func_inputs = []
        self.func_outputs = []

        self.task_prompt = f"""
You are designing a suite of 25 continuous optimization problems. The goal is
for this suite to strongly discriminate between five optimizers: CMA, DE, PSO,
BFGS, and Cobyla (Nevergrad implementations). The evaluation maximizes the mean
interquartile range (IQR) of Friedman ranks across functions, so more
performance spread across algorithms is better.

Each problem must be a Python callable `f(x)` that accepts a 1D numpy array and
returns a float. Each function is wrapped with IOH on the domain [-5, 5]^d. You
must provide a `problems` dictionary with 25 entries and a `meta_dims`
dictionary with matching keys that provides each function's dimensionality.
"""

        self.example_prompt = """
Example structure (you must provide your own problems):
```python
import numpy as np


def sphere(x):
    return float(np.sum(x ** 2))


def ridge(x):
    return float(np.sum(np.abs(x)) + 0.1 * np.sum(x ** 2))


problems = {
    "sphere": sphere,
    "ridge": ridge,
    # ... add 23 more functions
}

meta_dims = {
    "sphere": 5,
    "ridge": 12,
    # ... add 23 more dimensions
}
```
"""

        self.format_prompt = """
Provide your response in the following format:

# Description: <one-line summary of the suite's discriminative strategy>
# Code:
```python
<python code defining `problems` and `meta_dims`>
```
"""

    def get_prompt(self):
        return self.task_prompt + self.example_prompt + self.format_prompt

    def evaluate(self, solution: Solution, test=False):
        warnings.filterwarnings("ignore", category=Warning)

        local_env = {}
        try:
            exec(solution.code, {}, local_env)
        except Exception as exc:
            solution.set_scores(
                -np.inf,
                feedback="Failed to execute suite definition code.",
                error=str(exc),
            )
            return solution

        problems = local_env.get("problems")
        meta_dims = local_env.get("meta_dims")

        error = _validate_suite(problems, meta_dims)
        if error:
            solution.set_scores(
                -np.inf,
                feedback=error,
                error=error,
            )
            return solution

        with tempfile.TemporaryDirectory() as log_root:
            try:
                run_benchmark(
                    problems,
                    meta_dims,
                    self.budget,
                    self.repeats,
                    log_root,
                )
                dm = iohinspector.DataManager()
                dm.add_folder(log_root)
                dt_perf = dm.overview[["algorithm_name", "function_name", "best_y"]]
                friedman_val = get_friedman_val(dt_perf)
            except Exception as exc:
                solution.set_scores(
                    -np.inf,
                    feedback="Evaluation failed during benchmarking.",
                    error=str(exc),
                )
                return solution

        solution.add_metadata("friedman_iqr", friedman_val)
        solution.set_scores(
            friedman_val,
            "Suite scored on mean IQR of Friedman ranks (higher is better).",
        )
        return solution

    def test(self, solution: Solution, ioh_dir=""):
        return self.evaluate(solution, True)
