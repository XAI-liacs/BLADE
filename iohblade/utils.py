import ast
import difflib
import json
import math
import re
import textwrap
from typing import Optional, Tuple

import numpy as np


try:
    from ioh import LogInfo, logger
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    LogInfo = None

    class _DummyLogger:
        class AbstractLogger:
            pass

        def __getattr__(self, name):
            def _noop(*args, **kwargs):
                pass

            return _noop

    logger = _DummyLogger()

try:
    import pandas as pd
    from scipy.stats import ttest_rel, wilcoxon
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    pd = None
    ttest_rel = None
    wilcoxon = None


class TimeoutException(Exception):
    """Custom exception for handling timeouts."""

    pass


def first_class_name(code_string: str) -> str | None:
    try:  # 1. do it the robust way
        for node in ast.parse(code_string).body:  #    (won’t be fooled by comments)
            if isinstance(node, ast.ClassDef):
                return node.name
    except SyntaxError:
        pass  # fall back if the snippet is malformed

    m = re.search(
        r"^\s*class\s+([A-Za-z_]\w*)\s*[:(]", code_string, re.M
    )  # 2. quick-n-dirty
    return m.group(1) if m else None


def class_info(code_string: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Return (class_name, docstring_or_None).
    """
    # 1.  Robust path: use the AST – ignores comments and embedded strings
    try:
        for node in ast.parse(code_string).body:
            if isinstance(node, ast.ClassDef):
                return node.name, ast.get_docstring(node)
    except SyntaxError:
        pass  # malformed snippet ⇒ fall back

    # 2.  Fallback: quick regex for the first  class …\n  "docstring"
    m = re.search(
        r"^\s*class\s+([A-Za-z_]\w*)\s*[:(][^\n]*\n"  # class line
        r'\s*(?P<q>["\']){1,3}(?P<doc>.*?)(?P=q){1,3}',  # first quoted block/line
        code_string,
        re.S | re.M,
    )
    return (m.group(1), textwrap.dedent(m.group("doc")).strip()) if m else (None, None)


def code_compare(code1, code2):
    # Parse the Python code into ASTs
    # Use difflib to find differences
    diff = difflib.ndiff(code1.splitlines(), code2.splitlines())
    # Count the number of differing lines
    diffs = sum(1 for x in diff if x.startswith("- ") or x.startswith("+ "))
    # Calculate total lines for the ratio
    total_lines = max(len(code1.splitlines()), len(code2.splitlines()))
    similarity_ratio = (total_lines - diffs) / total_lines if total_lines else 1
    return 1 - similarity_ratio


def is_jsonable(x):
    try:
        json.dumps(x)
        return True
    except (TypeError, OverflowError):
        return False


def sanitize(o):
    """Helper for sanitizing json data."""
    if isinstance(o, float):
        return o if math.isfinite(o) else str(o)
    if isinstance(o, dict):
        return {k: sanitize(v) for k, v in o.items()}
    if isinstance(o, (list, tuple, set)):
        return [sanitize(v) for v in o]
    return o


def convert_to_serializable(data):
    if isinstance(data, dict):
        return {key: convert_to_serializable(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [convert_to_serializable(item) for item in data]
    elif isinstance(data, np.integer):
        return int(data)
    elif isinstance(data, np.floating):
        return sanitize(float(data))
    if isinstance(data, np.ndarray):
        return data.tolist()
    else:
        if is_jsonable(data):
            return sanitize(data)
        else:
            return str(data)


class NoCodeException(Exception):
    """Could not extract generated code."""

    pass


class ThresholdReachedException(Exception):
    """The algorithm reached the lower threshold."""

    pass


class OverBudgetException(Exception):
    """The algorithm tried to do more evaluations than allowed."""

    pass


def cliffs_delta(a, b):
    """Compute Cliff's delta effect size for two samples.

    Args:
        a: Sample A values.
        b: Sample B values.

    Returns:
        float: Cliff's delta in [-1, 1].
    """
    a = np.asarray(a)
    b = np.asarray(b)
    n = len(a) * len(b)
    gt = sum(x > y for x in a for y in b)
    lt = sum(x < y for x in a for y in b)
    return (gt - lt) / n


def paired_cohens_d(a, b):
    """Compute Cohen's d for paired samples.

    Args:
        a: Sample A values.
        b: Sample B values.

    Returns:
        float: Paired-sample Cohen's d.
    """
    diff = np.asarray(a) - np.asarray(b)
    return diff.mean() / diff.std(ddof=1)


def bootstrap_ci(diff, n_boot=10000, alpha=0.05, rng=None):
    """Compute a bootstrap confidence interval for a mean difference.

    Args:
        diff: Array of paired differences.
        n_boot: Number of bootstrap samples.
        alpha: Significance level (two-sided).
        rng: Random seed or numpy Generator.

    Returns:
        tuple[float, float]: Lower and upper confidence bounds.
    """
    rng = np.random.default_rng(rng)
    boot = rng.choice(diff, (n_boot, len(diff)), replace=True).mean(axis=1)
    lo = np.percentile(boot, 100 * alpha / 2)
    hi = np.percentile(boot, 100 * (1 - alpha / 2))
    return lo, hi


def compare_auc(
    logger: any,
    method_a: str,
    method_b: str,
    budget: int = 100,
    metric: str = "fitness",
    test: str = "wilcoxon",  # or "ttest"
    n_boot: int = 10000,
    alpha: float = 0.05,
    rng: int = 0,
):
    """Compare convergence AUCs between two methods.

    Args:
        logger: Experiment logger containing evaluation data.
        method_a: Name of the first method.
        method_b: Name of the second method.
        budget: Maximum evaluation budget for AUC integration.
        metric: Column name for the fitness metric to integrate.
        test: Statistical test to use ("wilcoxon" or "ttest").
        n_boot: Number of bootstrap samples for confidence intervals.
        alpha: Significance level (two-sided).
        rng: Random seed for bootstrapping.

    Returns:
        pd.DataFrame: Per-problem summary of AUC comparisons.
    """

    methods, problems = logger.get_methods_problems()
    if method_a not in methods or method_b not in methods:
        raise ValueError("Both methods must exist in the logger.")

    results = []

    for problem in problems:
        data = logger.get_problem_data(problem_name=problem).drop(columns=["code"])
        data.replace([-np.inf, np.inf], 0, inplace=True)
        data.fillna(0, inplace=True)

        aucs = {}

        for method in [method_a, method_b]:
            df = data[data["method_name"] == method].copy()
            df = df.sort_values(by=["seed", "_id"])
            df["cummax_fitness"] = df.groupby("seed")[metric].cummax()
            df["_id"] += 1
            df = df[df["_id"] <= budget]

            per_seed_auc = []
            for seed in df["seed"].unique():
                seed_df = df[df["seed"] == seed]
                auc = np.trapz(
                    seed_df["cummax_fitness"].values,
                    seed_df["_id"].values,
                )
                per_seed_auc.append(auc)

            aucs[method] = np.asarray(per_seed_auc)

        # Paired samples
        min_len = min(len(aucs[method_a]), len(aucs[method_b]))
        a = aucs[method_a][:min_len]
        b = aucs[method_b][:min_len]

        diff = a - b

        # Statistical test
        if test == "wilcoxon":
            stat, p = wilcoxon(a, b)
            test_name = "Wilcoxon signed-rank"
        elif test == "ttest":
            stat, p = ttest_rel(a, b)
            test_name = "Paired t-test"
        else:
            raise ValueError("test must be 'wilcoxon' or 'ttest'")

        # Effect sizes
        delta = cliffs_delta(a, b)
        cohens_d = paired_cohens_d(a, b)

        # Bootstrap CI for mean difference
        ci_low, ci_high = bootstrap_ci(diff, n_boot=n_boot, alpha=alpha, rng=rng)

        results.append(
            {
                "problem": problem,
                "method_a": method_a,
                "method_b": method_b,
                "n_seeds": min_len,
                "mean_auc_a": a.mean(),
                "mean_auc_b": b.mean(),
                "median_auc_a": np.median(a),
                "median_auc_b": np.median(b),
                "mean_auc_diff": diff.mean(),
                "ci_low": ci_low,
                "ci_high": ci_high,
                "cliffs_delta": delta,
                "cohens_d": cohens_d,
                "test": test_name,
                "statistic": stat,
                "p_value": p,
            }
        )

    return pd.DataFrame(results)


def correct_aoc(ioh_function, logger, budget):
    """Correct aoc values in case a run stopped before the budget was exhausted

    Args:
        ioh_function: The function in its final state (before resetting!)
        logger: The logger in its final state, so we can ensure the settings for aoc calculation match
        budget: The intended maximum budget

    Returns:
        float: The normalized aoc of the run, corrected for stopped runs
    """
    fraction = (
        logger.transform(
            np.clip(
                ioh_function.state.current_best_internal.y, logger.lower, logger.upper
            )
        )
        - logger.transform(logger.lower)
    ) / (logger.transform(logger.upper) - logger.transform(logger.lower))
    aoc = (
        logger.aoc
        + np.clip(budget - ioh_function.state.evaluations, 0, budget) * fraction
    ) / budget

    return 1 - aoc


class aoc_logger(logger.AbstractLogger):
    """aoc_logger class implementing the logging module for ioh."""

    def __init__(
        self,
        budget,
        lower=1e-8,
        upper=1e8,
        scale_log=True,
        stop_on_threshold=False,
        *args,
        **kwargs,
    ):
        """Initialize the logger.

        Args:
            budget (int): Evaluation budget for calculating aoc.
        """
        super().__init__(*args, **kwargs)
        self.aoc = 0
        self.lower = lower
        self.upper = upper
        self.budget = budget
        self.stop_on_threshold = stop_on_threshold
        self.transform = lambda x: np.log10(x) if scale_log else (lambda x: x)

    def __call__(self, log_info: LogInfo):
        """Subscalculate the aoc.

        Args:
            log_info (ioh.LogInfo): info about current values.
        """
        if log_info.evaluations > self.budget:
            raise OverBudgetException
        if log_info.evaluations == self.budget:
            return
        if self.stop_on_threshold and abs(log_info.raw_y_best) < self.lower:
            raise ThresholdReachedException
        y_value = np.clip(log_info.raw_y_best, self.lower, self.upper)
        self.aoc += (self.transform(y_value) - self.transform(self.lower)) / (
            self.transform(self.upper) - self.transform(self.lower)
        )

    def reset(self, func):
        super().reset()
        self.aoc = 0


class budget_logger(logger.AbstractLogger):
    """budget_logger class implementing the logging module for ioh."""

    def __init__(
        self,
        budget,
        *args,
        **kwargs,
    ):
        """Initialize the logger.

        Args:
            budget (int): Evaluation budget for calculating aoc.
        """
        super().__init__(*args, **kwargs)
        self.budget = budget

    def __call__(self, log_info: LogInfo):
        """Subscalculate the aoc.

        Args:
            log_info (ioh.LogInfo): info about current values.
        """
        if log_info.evaluations > self.budget:
            raise OverBudgetException

    def reset(self):
        super().reset()
