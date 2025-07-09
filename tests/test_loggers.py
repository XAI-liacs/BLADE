import os
import shutil

import pytest

from iohblade import Method, Problem, Solution
from iohblade.loggers import ExperimentLogger, RunLogger


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
    from iohblade.llm import LLM
    from iohblade.method import Method
    from iohblade.problem import Problem

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
        def _query(self, s):
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


def test_read_multiple_log_dirs(cleanup_tmp_dir):
    exp_logger1 = ExperimentLogger(name=os.path.join(cleanup_tmp_dir, "multi_dir1"))
    exp_logger2 = ExperimentLogger(name=os.path.join(cleanup_tmp_dir, "multi_dir2"))

    # create a sample log file in each directory
    run_logger = RunLogger(name="test_run1", root_dir=exp_logger1.dirname)
    sol = Solution(name="test_solution").set_scores(1.0, feedback="Test feedback")
    run_logger.log_individual(sol)
    # Check existence of log.jsonl
    log_file1 = os.path.join(run_logger.get_log_dir(), "log.jsonl")
    assert os.path.exists(log_file1)

    run_logger2 = RunLogger(name="test_run2", root_dir=exp_logger2.dirname)
    sol = Solution(name="test_solution").set_scores(1.0, feedback="Test feedback")
    run_logger2.log_individual(sol)
    # Check existence of log.jsonl
    log_file2 = os.path.join(run_logger2.get_log_dir(), "log.jsonl")
    assert os.path.exists(log_file2)

    log_file = os.path.join(exp_logger1.dirname, "experimentlog.jsonl")
    with open(log_file, "w") as f:
        f.write(
            f'{{"method_name":"methodA","problem_name":"problemX", "log_dir":"run-test_run1", "seed":"0"}}\n'
        )
        f.write(
            f'{{"method_name":"methodB","problem_name":"problemY", "log_dir":"run-test_run1", "seed":"0"}}\n'
        )

    log_file = os.path.join(exp_logger2.dirname, "experimentlog.jsonl")
    with open(log_file, "w") as f:
        f.write(
            f'{{"method_name":"methodC","problem_name":"problemX", "log_dir":"run-test_run2", "seed":"1"}}\n'
        )
        f.write(
            f'{{"method_name":"methodD","problem_name":"problemY", "log_dir":"run-test_run2", "seed":"1"}}\n'
        )

    # now create a new ExperimentLogger that reads both directories.
    explogger_read = ExperimentLogger(name=exp_logger1.dirname, read=True)
    explogger_read.add_read_dir(exp_logger2.dirname)
    data = explogger_read.get_data()
    assert len(data) == 4
    problem_data = explogger_read.get_problem_data("problemX")
    assert len(problem_data) == 2, "Expected 2 entries for problemX, got {}".format(
        len(problem_data)
    )


def test_experiment_logger_get_methods_problems(cleanup_tmp_dir):
    """`get_methods_problems` should return the unique sets of methods and problems
    across all read-in experiment directories, ignoring dirs with no log."""
    import os

    from iohblade.loggers import ExperimentLogger

    # Make three experiment dirs: two with logs, one empty
    dir1 = os.path.join(cleanup_tmp_dir, "exp1")
    dir2 = os.path.join(cleanup_tmp_dir, "exp2")

    os.makedirs(dir1, exist_ok=True)
    os.makedirs(dir2, exist_ok=True)

    # Write JSON-lines logs
    with open(os.path.join(dir1, "experimentlog.jsonl"), "w") as f:
        f.write('{"method_name":"methodA","problem_name":"problemX"}\n')
        f.write('{"method_name":"methodB","problem_name":"problemY"}\n')
    with open(os.path.join(dir2, "experimentlog.jsonl"), "w") as f:
        f.write('{"method_name":"methodC","problem_name":"problemX"}\n')
        # Duplicate methodA, new problemZ – should be deduped
        f.write('{"method_name":"methodA","problem_name":"problemZ"}\n')

    # Read the dirs
    exp_logger = ExperimentLogger(name=dir1, read=True)
    exp_logger.add_read_dir(dir2)

    methods, problems = exp_logger.get_methods_problems()

    assert set(methods) == {"methodA", "methodB", "methodC"}
    assert set(problems) == {"problemX", "problemY", "problemZ"}


def test_start_progress_and_restart(tmp_path):
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

    m = DummyMethod(None, 1, name="m")
    p = DummyProblem(name="p")
    logger_dir = tmp_path / "exp"
    logger = ExperimentLogger(name=str(logger_dir))
    logger.start_progress(1, methods=[m], problems=[p], seeds=[0], budget=1)

    # Restart with same config should succeed
    logger2 = ExperimentLogger(name=str(logger_dir))
    logger2.start_progress(1, methods=[m], problems=[p], seeds=[0], budget=1)
    assert logger2.is_run_pending(m, p, 0)


def test_start_progress_mismatch(tmp_path):
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

    m1 = DummyMethod(None, 1, name="m1")
    m2 = DummyMethod(None, 1, name="m2")
    p = DummyProblem(name="p")
    logger_dir = tmp_path / "exp"
    logger = ExperimentLogger(name=str(logger_dir))
    logger.start_progress(1, methods=[m1], problems=[p], seeds=[0], budget=1)

    logger2 = ExperimentLogger(name=str(logger_dir))
    with pytest.raises(ValueError):
        logger2.start_progress(1, methods=[m2], problems=[p], seeds=[0], budget=1)


def test_open_run_create_and_restart(tmp_path):
    """Check that `open_run` correctly creates, restarts, and updates progress."""

    # --- dummy objects -------------------------------------------------
    class DummyMethod(Method):
        def __call__(self, problem):
            pass

        def to_dict(self):
            return {}

    class DummyProblem(Problem):
        def get_prompt(self):
            return "prompt"

        def evaluate(self, s):  # pragma: no cover
            return s

        def test(self, s):  # pragma: no cover
            return s

        def to_dict(self):
            return {}

        # Keep a reference so we can assert the logger was set.
        def set_logger(self, logger):
            self._logger = logger
            super().set_logger(logger)

    # --- setup ---------------------------------------------------------
    exp_dir = tmp_path / "exp"
    logger = ExperimentLogger(name=str(exp_dir))

    m = DummyMethod(None, 1, name="methodX")
    p = DummyProblem(name="problemY")

    # ------------------------------------------------------------------
    # 1. First call should create a run directory and a progress entry
    # ------------------------------------------------------------------
    run1 = logger.open_run(m, p, budget=5, seed=0)
    entry = logger._get_run_entry("methodX", "problemY", 0)

    assert entry is not None, "Progress entry not created"
    assert entry["evaluations"] == 0
    assert entry["end_time"] is None
    assert os.path.isdir(run1.dirname)
    assert entry["log_dir"] == os.path.relpath(run1.dirname, logger.dirname)
    # Logger should be attached to the problem
    assert getattr(p, "_logger", None) is run1

    # The callback should increment `evaluations`
    run1.log_individual(Solution(name="sol"))
    assert entry["evaluations"] == 1

    first_dir = run1.dirname
    assert os.path.exists(first_dir)

    # ------------------------------------------------------------------
    # 2. Second call with same identifiers (unfinished run) should
    #    delete old dir, reset evaluations, and create a new dir
    # ------------------------------------------------------------------
    run2 = logger.open_run(m, p, budget=5, seed=0)
    entry_after = logger._get_run_entry("methodX", "problemY", 0)

    assert entry_after["evaluations"] == 0, "Evaluations were not reset"
    assert entry_after["log_dir"] == os.path.relpath(run2.dirname, logger.dirname)
    assert getattr(p, "_logger", None) is run2

    # ------------------------------------------------------------------
    # 3. Different seed should create an independent progress entry
    # ------------------------------------------------------------------
    run3 = logger.open_run(m, p, budget=5, seed=1)
    new_entry = logger._get_run_entry("methodX", "problemY", 1)

    assert new_entry is not None
    assert new_entry is not entry_after  # distinct entry
    assert os.path.isdir(run3.dirname)
