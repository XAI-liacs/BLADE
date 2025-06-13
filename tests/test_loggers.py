import pytest
import os
import shutil
from iohblade.loggers import ExperimentLogger, RunLogger
from iohblade import Solution


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


def test_experiment_logger_add_run(cleanup_tmp_dir):
    exp_logger = ExperimentLogger(name=os.path.join(cleanup_tmp_dir, "my_experiment"))
    from iohblade.method import Method
    from iohblade.problem import Problem
    from iohblade.llm import LLM

    class DummyMethod(Method):
        def __call__(self, problem):
            pass

        def to_dict(self):
            return {}

    class DummyProblem(Problem):
        def get_prompt(self):
            return "prompt"

        def evaluate(self, s):
            return s

        def test(self, s):
            return s

        def to_dict(self):
            return {}

    class DummyLLM(LLM):
        def query(self, s):
            return "res"

    method = DummyMethod(None, 100, name="dummy_method")
    problem = DummyProblem()
    llm = DummyLLM("", "")
    sol = Solution()
    exp_logger.add_run(method, problem, llm, sol, log_dir="dummy_dir", seed=42)

    # Check the log file
    log_file = os.path.join(exp_logger.dirname, "experimentlog.jsonl")
    with open(log_file, "r") as f:
        contents = f.read()
    assert "dummy_method" in contents
    expected_rel = os.path.relpath("dummy_dir", exp_logger.dirname)
    assert expected_rel in contents
    assert '"seed": 42' in contents


def test_run_logger_log_individual(cleanup_tmp_dir):
    run_logger = RunLogger(name="test_run", root_dir=cleanup_tmp_dir)
    sol = Solution(name="test_solution")
    run_logger.log_individual(sol)
    # Check existence of log.jsonl
    log_file = os.path.join(run_logger.dirname, "log.jsonl")
    assert os.path.exists(log_file)
    with open(log_file, "r") as f:
        contents = f.read()
    assert "test_solution" in contents


def test_run_logger_budget_exhausted(cleanup_tmp_dir):
    run_logger = RunLogger(name="test_run", root_dir=cleanup_tmp_dir, budget=1)
    sol = Solution(name="test_solution")
    run_logger.log_individual(sol)
    assert run_logger.budget_exhausted() is True


def test_experiment_logger_get_data(cleanup_tmp_dir):
    exp_logger = ExperimentLogger(
        name=os.path.join(cleanup_tmp_dir, "my_experiment_data")
    )
    # Write a dummy JSON line
    log_file = os.path.join(exp_logger.dirname, "experimentlog.jsonl")
    with open(log_file, "w") as f:
        f.write('{"method_name":"methodA","problem_name":"problemX"}\n')
        f.write('{"method_name":"methodB","problem_name":"problemY"}\n')
    df = exp_logger.get_data()
    assert len(df) == 2
    assert "method_name" in df.columns
    assert "problem_name" in df.columns
