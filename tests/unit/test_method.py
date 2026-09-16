from typing import Any
from unittest.mock import MagicMock

import sys
import os
import pytest
import shutil
from iohblade.llm import LLM
from iohblade.method import Method
from iohblade.methods.random_search import RandomSearch
from iohblade.problem import Problem
from iohblade.solution import Solution
from iohblade.experiment import Experiment
from iohblade.loggers.base import ExperimentLogger

@pytest.fixture
def cleanup_tmp_dir():
    # Creates a temporary directory for tests, yields its name, then cleans up
    dirname = "test_results"
    if not os.path.exists(dirname):
        os.mkdir(dirname)
    yield dirname
    # Cleanup
    if os.path.exists(dirname):
        shutil.rmtree(dirname)

def test_random_search_calls_llm():
    class DummyLLM(LLM):
        def _query(self, s):
            return "# Description: MyAlgo\n```python\nclass MyAlgo:\n  pass\n```"

        def get_config(self) -> list[dict[str, Any]]:
            return [{}]

    class DummyProblem:
        def get_prompt(self):
            return "some prompt"

        def __call__(self, sol):
            # Evaluate solution with random fitness
            sol.set_scores(42.0)
            return sol

    llm = DummyLLM(api_key="xxx")
    rs = RandomSearch(llm, budget=3, name="RS")
    dp = DummyProblem()
    best_sol = rs(dp)
    # The random search calls sample_solution a few times. We didn't fully mock it, but let's check:
    assert best_sol.fitness == 42.0
    assert "class MyAlgo" in best_sol.code


def test_experiment_hashes_method_as_id(cleanup_tmp_dir):
    class DummyLLM(LLM):
        def _query(self, s):
            return "# Description: MyAlgo\n```python\nclass MyAlgo:\n  pass\n```"

        def get_config(self) -> list[dict[str, Any]]:
            return [{}]

    class DummyProblem(Problem):
        def get_prompt(self):
            return "some prompt"

        def __call__(self, sol):
            # Evaluate solution with random fitness
            return self.evaluate

        def evaluate(self, solution: Solution):
            solution.set_scores(42.0)
            return solution

        def test(self, solution: Solution):
            return self.evaluate(solution)

        def to_dict(self):
            return super().to_dict()

        def get_config(self) -> dict[str, Any]:
            return {"name": "Dummy Prompt", "evaluator": "def f(x):\n\treturn 42.0", "minimisation": True}

    class DummyMethod(Method):
        def __call__(self, problem):
            print("out")
            print("err", file=sys.stderr)
            return Solution()

        def to_dict(self):
            return {}

        def get_config(self) -> dict[str, Any]:
            return {}


    llm = DummyLLM("", "")
    method = DummyMethod(llm, 1, name="m")
    problem = DummyProblem(name="p")
    exp = Experiment(
        methods=[method],
        problems=[problem],
        log_stdout=True,
        exp_logger=ExperimentLogger(os.path.join(cleanup_tmp_dir, "exp")),
    )
    hash1 = exp._get_hash(problem.get_config())
    hash2 = exp._get_hash(problem.get_config())
    assert hash1 == hash2, "Hashes of same config files must match..."
    config = problem.get_config()
    config['prompt'] = "some prompt"
    hash3 = exp._get_hash(config)
    assert hash1 != hash3, "Hashes of different problem must differ..."
    print(hash1, hash2, hash3)
