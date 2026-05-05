import platform
import subprocess
import time
from unittest.mock import MagicMock

import numpy as np

try:
    import pytest  # only used in tests, make sure cloudpickle doesn't care
except ImportError:
    pytest = None

from iohblade import Problem, Solution, TimeoutException
from iohblade.problem import evaluate_in_subprocess


class SlowProblem(Problem):
    def get_prompt(self):
        return "Prompt"

    def evaluate(self, s):
        time.sleep(2)  # intentionally slow
        s.set_scores(5.0)
        return s

    def test(self, s):
        return s

    def to_dict(self):
        return {}


class DummyProblem(Problem):
    def get_prompt(self):
        return "Problem prompt"

    def evaluate(self, s):
        s.set_scores(1.0, "Feedback")
        return s

    def test(self, s):
        return s

    def to_dict(self):
        return {}


class HangingProblem(Problem):
    def get_prompt(self):
        return "Problem prompt"

    def evaluate(self, s):
        time.sleep(5)
        return s

    def test(self, s):
        return s

    def to_dict(self):
        return {}


def test_problem_abstract_methods():
    dp = DummyProblem(name="dummy")
    assert dp.name == "dummy"
    sol = Solution()
    # Just ensure that calling it doesn't blow up
    sol = dp(sol)
    assert sol.feedback == "Feedback", sol.feedback
    assert sol.fitness == 1.0, sol.fitness


def test_problem_timeout():
    sp = SlowProblem(eval_timeout=1)  # 1 second
    sol = Solution()
    sol = sp(sol)
    # We expect a TimeoutException or similar
    assert "timed out" in str(sol.feedback)


def _active_run_eval_processes():
    try:
        if platform.system() == "Windows":
            output = subprocess.check_output(["tasklist"], text=True)
        else:
            output = subprocess.check_output(["ps", "-eo", "pid,cmd"], text=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        return []

    return [line for line in output.splitlines() if "run_eval.py" in line]


def test_timeout_cleans_child_processes():
    before = _active_run_eval_processes()

    problem = HangingProblem(eval_timeout=1)
    sol = Solution()
    sol = problem(sol)

    assert "timed out" in str(sol.feedback)

    time.sleep(0.5)
    after = _active_run_eval_processes()

    assert not after or after == before
