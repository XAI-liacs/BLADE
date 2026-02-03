import os
import shutil
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from iohblade.experiment import Experiment, MA_BBOB_Experiment
from iohblade.llm import LLM
from iohblade.loggers import ExperimentLogger
from iohblade.method import Method
from iohblade.problem import Problem
from iohblade.solution import Solution


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


class EvalProblem(Problem):
    def get_prompt(self):
        return "Problem prompt"

    def evaluate(self, s):
        print("eval out")
        print("eval err", file=sys.stderr)
        return s

    def test(self, s):
        return s

    def to_dict(self):
        return {}

    def _ensure_env(self):
        import tempfile

        if self._env_path is None:
            self._env_path = Path(tempfile.mkdtemp())
            self._python_bin = Path(sys.executable)


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


def test_experiment_log_stdout(cleanup_tmp_dir):
    class DummyMethod(Method):
        def __call__(self, problem):
            print("out")
            print("err", file=sys.stderr)
            return Solution()

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

        def _ensure_env(self):
            return None

    class DummyLLM(LLM):
        def _query(self, s):
            return "res"

    llm = DummyLLM("", "")
    method = DummyMethod(llm, 1, name="m")
    problem = DummyProblem(name="p")
    exp = Experiment(
        methods=[method],
        problems=[problem],
        log_stdout=True,
        exp_logger=ExperimentLogger(os.path.join(cleanup_tmp_dir, "exp")),
    )
    exp()
    run_dir = os.path.join(exp.exp_logger.dirname, "run-m-p-0")
    with open(os.path.join(run_dir, "stdout.log")) as f:
        assert "out" in f.read()
    with open(os.path.join(run_dir, "stderr.log")) as f:
        assert "err" in f.read()


def test_experiment_logs_problem_eval_stdout(cleanup_tmp_dir):
    class EvalMethod(Method):
        def __call__(self, problem):
            sol = Solution()
            return problem(sol)

        def to_dict(self):
            return {}

    class DummyLLM(LLM):
        def _query(self, s):
            return "res"

    llm = DummyLLM("", "")
    method = EvalMethod(llm, 1, name="m")
    problem = EvalProblem(name="p")
    exp = Experiment(
        methods=[method],
        problems=[problem],
        log_stdout=True,
        runs=1,
        exp_logger=ExperimentLogger(os.path.join(cleanup_tmp_dir, "exp_eval")),
    )
    exp()
    run_dir = os.path.join(exp.exp_logger.dirname, "run-m-p-0")
    with open(os.path.join(run_dir, "stdout.log")) as f:
        assert "eval out" in f.read()
    with open(os.path.join(run_dir, "stderr.log")) as f:
        assert "eval err" in f.read()
