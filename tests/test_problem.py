import pytest
from unittest.mock import MagicMock
import numpy as np
from iohblade.problem import Problem, TimeoutException, TimeoutException, evaluate_in_subprocess
from iohblade.solution import Solution
import time

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

def test_problem_abstract_methods():
    class DummyProblem(Problem):
        def get_prompt(self):
            return "Problem prompt"
        def evaluate(self, s):
            return s
        def test(self, s):
            return s
        def to_dict(self):
            return {}
    
    dp = DummyProblem(name="dummy")
    assert dp.name == "dummy"
    sol = Solution()
    # Just ensure that calling it doesn't blow up
    dp(sol)
    assert sol.fitness == -np.inf  # because evaluate didn't do anything

def test_problem_timeout():
    sp = SlowProblem(eval_timeout=1)  # 1 second
    sol = Solution()
    sp(sol)
    # We expect a TimeoutException or similar
    assert "timed out" in str(sol.feedback)
