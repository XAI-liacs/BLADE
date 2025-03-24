import pytest
import os
import shutil
from unittest.mock import MagicMock
from iohblade.experiment import Experiment, MA_BBOB_Experiment
from iohblade.llm import LLM
from iohblade.problem import Problem
from iohblade.method import Method

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

def test_ma_bbob_experiment_init(cleanup_tmp_dir):
    class DummyMethod(Method):
        def __call__(self, problem):
            pass
        def to_dict(self):
            return {}

    class DummyLLM(LLM):
        def query(self, s):
            return "res"
    methods = [DummyMethod(None, 10, name="m1")]
    llm = DummyLLM(api_key="", model="")
    exp = MA_BBOB_Experiment(methods, llm, runs=2, budget=50, dims=[2, 3], budget_factor=1000, log_dir=os.path.join(cleanup_tmp_dir, "mabbob_experiment"))
    assert len(exp.problems) == 1  # Just one MA_BBOB instance
    assert exp.runs == 2
    assert exp.budget == 50

def test_experiment_run(cleanup_tmp_dir):
    class DummyExp(Experiment):
        def __call__(self):
            self.exp_logger.add_run(self.methods[0], self.problems[0], self.llm, MagicMock(), log_dir="test", seed=0)

    class DummyMethod(Method):
        def __call__(self, problem):
            pass
        def to_dict(self):
            return {}

    class DummyProblem(Problem):
        def get_prompt(self):
            return "Problem prompt"
        def evaluate(self, s):
            return s
        def test(self, s):
            return s
        def to_dict(self):
            return {}

    class DummyLLM(LLM):
        def query(self, session_messages):
            return "response"

    m = DummyMethod(None, 5, name="DMethod")
    p = DummyProblem()
    l = DummyLLM("", "")
    exp = DummyExp(methods=[m], problems=[p], llm=l, log_dir=os.path.join(cleanup_tmp_dir, "mabbob_experiment"))
    exp()  # call
    # Check something about the exp_logger, or just ensure it doesn't crash
