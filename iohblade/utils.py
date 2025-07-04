import ast
import difflib
import json
import re
import textwrap
from typing import Optional, Tuple

import numpy as np
from ioh import LogInfo, logger


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


def convert_to_serializable(data):
    if isinstance(data, dict):
        return {key: convert_to_serializable(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [convert_to_serializable(item) for item in data]
    elif isinstance(data, np.integer):
        return int(data)
    elif isinstance(data, np.floating):
        return float(data)
    if isinstance(data, np.ndarray):
        return data.tolist()
    else:
        if is_jsonable(data):
            return data
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
