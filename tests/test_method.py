from unittest.mock import MagicMock

import pytest

from iohblade import LLM, Method, Solution
from iohblade.methods.random_search import RandomSearch


def test_random_search_calls_llm():
    class DummyLLM(LLM):
        def _query(self, s):
            return "# Description: MyAlgo\n```python\nclass MyAlgo:\n  pass\n```"

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
