import os
import shutil
from unittest.mock import MagicMock

import pytest

from iohblade import LLM, Method, Problem
from iohblade.experiment import Experiment, MA_BBOB_Experiment
from iohblade.loggers import ExperimentLogger


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
        def _query(self, s):
            return "res"

    llm = DummyLLM(api_key="", model="")
    methods = [DummyMethod(llm, 10, name="m1")]

    exp = MA_BBOB_Experiment(
        methods,
        runs=2,
        budget=50,
        dims=[2, 3],
        budget_factor=1000,
        exp_logger=ExperimentLogger(os.path.join(cleanup_tmp_dir, "mabbob_experiment")),
    )
    assert len(exp.problems) == 1  # Just one MA_BBOB instance
    assert exp.runs == 2
    assert exp.budget == 50


def test_experiment_run(cleanup_tmp_dir):
    class DummyExp(Experiment):
        def __call__(self):
            self.exp_logger.add_run(
                self.methods[0],
                self.problems[0],
                self.methods[0].llm,
                MagicMock(),
                log_dir="test",
                seed=0,
            )

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
        def _query(self, session_messages):
            return "response"

    l = DummyLLM("", "")
    m = DummyMethod(l, 5, name="DMethod")
    p = DummyProblem()

    exp = DummyExp(
        methods=[m],
        problems=[p],
        exp_logger=ExperimentLogger(os.path.join(cleanup_tmp_dir, "mabbob_experiment")),
    )
    exp()  # call
    # Check something about the exp_logger, or just ensure it doesn't crash
