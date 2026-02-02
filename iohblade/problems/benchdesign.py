import tempfile
import warnings

import ioh
import iohinspector
import nevergrad as ng
import numpy as np
import polars as pl
from nevergrad.optimization.optimizerlib import BFGS, CMA, DE, PSO, Cobyla
from scipy.stats import rankdata
import math

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


def run_benchmark(local_env, problems, meta_dims, budget, repeats, log_root):
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
    # if len(problems) != 25 or len(meta_dims) != 25:
    #     return "problems and meta_dims must each contain 25 entries."
    problem_keys = set(problems.keys())
    meta_keys = set(meta_dims.keys())
    if problem_keys != meta_keys:
        return "problems and meta_dims must share identical keys."
    for name, dim in meta_dims.items():
        if not isinstance(dim, int) or dim <= 0:
            return f"meta_dims[{name!r}] must be a positive integer."
    return None

def avg_ranks_per_function(dt_perf: pl.DataFrame) -> pl.DataFrame:
    """
    Returns a table with per-function average Friedman rank per algorithm.
    Lower rank = better (1 is best).
    """
    rank_rows = []

    for func_name in dt_perf["function_name"].unique():
        func_data = dt_perf.filter(pl.col("function_name") == func_name)

        # assume multiple runs per algorithm; align by run index
        max_runs = (
            func_data.group_by("algorithm_name")
            .agg(pl.col("best_y").count().alias("n"))
            .select(pl.col("n").max())[0, 0]
        )

        for run_idx in range(max_runs):
            run_scores, run_algs = [], []
            for alg_name in func_data["algorithm_name"].unique():
                ys = func_data.filter(pl.col("algorithm_name") == alg_name)["best_y"].to_numpy()
                if run_idx < len(ys):
                    run_scores.append(float(ys[run_idx]))
                    run_algs.append(alg_name)

            if not run_scores:
                continue

            # smaller best_y is better -> smaller rank is better
            ranks = rankdata(run_scores, method="average")
            for alg, r in zip(run_algs, ranks):
                rank_rows.append(
                    {"function_name": func_name, "algorithm_name": alg, "run": run_idx, "rank": float(r)}
                )

    if not rank_rows:
        return pl.DataFrame({"function_name": [], "algorithm_name": [], "avg_rank": []})

    return (
        pl.DataFrame(rank_rows)
        .group_by(["function_name", "algorithm_name"])
        .agg(pl.col("rank").mean().alias("avg_rank"))
        .sort(["function_name", "avg_rank"])
    )

def format_rank_table(df: pl.DataFrame, top_k: int = 5) -> str:
    """
    Compact text: per function show best top_k algorithms by avg rank.
    """
    if df.is_empty():
        return "(no rank data)"
    lines = []
    for func in df["function_name"].unique():
        sub = df.filter(pl.col("function_name") == func).head(top_k)
        parts = [f"{row['algorithm_name']}:{row['avg_rank']:.2f}" for row in sub.to_dicts()]
        lines.append(f"- {func}: " + ", ".join(parts))
    return "\n".join(lines)

class BenchDesign(Problem):
    """
    Problem class for designing suites that discriminate between optimizers.
    """

    def __init__(
        self,
        logger=None,
        training_instances=None,
        test_instances=None,
        name="BenchDesign",
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
                "iohinspector>=0.0.6,<1",
                "polars>=1.37.0,<2",
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
        self.name = name
        self.budget = budget
        self.repeats = repeats
        self.imports = imports

        self.func_name = "problems"
        self.init_inputs = []
        self.func_inputs = []
        self.func_outputs = []

        number_of_problems = 25

        self.task_prompt = f"""
You are designing a suite of {number_of_problems} continuous optimization problems. The goal is
for this suite to strongly discriminate between five optimizers: CMA, DE, PSO,
BFGS, and Cobyla (Nevergrad implementations). The evaluation maximizes the mean
interquartile range (IQR) of Friedman ranks across functions, so more
performance spread across algorithms is better.

Each problem must be a Python callable `f(x)` that accepts a 1D numpy array and
returns a float. Each function is wrapped with IOH on the domain [-5, 5]^d. You
must provide a `problems` dictionary with {number_of_problems} entries and a `meta_dims`
dictionary with matching keys that provides each function's dimensionality.
"""

        self.example_prompt = """
Example structure of 2 problems (you must provide your own problems):
```python
import numpy as np


def f1(x):
    return float(np.sum(x ** 2))


def f2(x):
    return float(np.sum(np.abs(x)) + 0.1 * np.sum(x ** 2))

    
# def f3(x):... etc.

problems = {
    "f1": f1,
    "f2": f2,
    # "f3": f3, etc.
}

meta_dims = {
    "f1": 5,
    "f2": 12,
    # "f3": ... etc.
}
```
"""

        self.format_prompt = """
Make sure that all the functions are not dependent on each other or on global variables as they will be evaluated in isolated environments.
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
        safe_globals = {"np": np, "math": math}
        try:
            exec(solution.code, safe_globals, local_env)
        except Exception as exc:
            print(exc)
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
            run_benchmark(
                local_env,
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
            rank_df = avg_ranks_per_function(dt_perf)

        # store full table (as python-native) in metadata if you like
        solution.add_metadata("avg_friedman_ranks_per_function", rank_df.to_dicts())

        # add a compact human-readable summary to the LLM feedback
        rank_summary = format_rank_table(rank_df, top_k=5)
        solution.add_metadata("friedman_iqr", friedman_val)

        solution.set_scores(
            friedman_val,
            f"Suite scored {friedman_val} on mean IQR of Friedman ranks (higher is better).\n\n"
            "Per-function average Friedman ranks (lower is better):\n"
            f"{rank_summary}",
        )
        return solution

    def test(self, solution: Solution, ioh_dir=""):
        return self.evaluate(solution, True)
    
    def to_dict(self):
        """
        Converts the problem to a dictionary.
        """
        return {
            "name": self.name,
            "budget": self.budget,
        }
