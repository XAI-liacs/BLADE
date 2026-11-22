"""Phase 4: progress tracking, log files, summaries and error reporting."""

import json
import logging
import os
import sys
import threading
import warnings
from typing import Any

import pytest

from iohblade.envs import EnvironmentSetupError
from iohblade.experiment import Experiment, _format_error_counts
from iohblade.llm import LLM
from iohblade.loggers import ExperimentLogger
from iohblade.loggers.base import RunLogger, _error_summary, error_kind
from iohblade.method import Method
from iohblade.problem import HARNESS_ERROR_PREFIX, Problem
from iohblade.runlog import experiment_logging, log
from iohblade.solution import Solution


class _LLM(LLM):
    def __init__(self, model="m"):
        super().__init__("", model)

    def _query(self, session, **kwargs):
        return "x"

    def get_config(self):
        return [{}]


class _Problem(Problem):
    def __init__(self, name="p", minimisation=False):
        super().__init__(name=name)
        self.minimisation = minimisation

    def _ensure_env(self):
        pass

    def cleanup(self):
        pass

    def evaluate(self, s):
        return s

    def test(self, s):
        return s

    def to_dict(self):
        return {}

    def get_config(self) -> dict[str, Any]:
        return {}


class _Method(Method):
    """Logs three evaluations: two scores and one candidate error."""

    def __call__(self, problem):
        logging.getLogger("thirdparty").warning("noisy library message")
        best = None
        for fitness, error in ((1.0, ""), (None, "ValueError: bad"), (3.0, "")):
            s = Solution(name="s")
            if error:
                s.set_scores(float("-inf"), error, error)
            else:
                s.set_scores(fitness)
            problem.logger.log_individual(s)
            best = s if best is None or (s.fitness > best.fitness) else best
        return best

    def to_dict(self):
        return {}

    def get_config(self):
        return {}


class _Failing(_Method):
    def __call__(self, problem):
        raise KeyError("broken method")


def _solution(fitness=None, error=""):
    s = Solution(name="s")
    if error:
        s.set_scores(float("-inf"), error, error)
    elif fitness is not None:
        s.set_scores(fitness)
    return s


# --------------------------------------------------------------- progress ---


def test_error_kind_and_summary():
    assert error_kind(_solution(1.0)) is None
    assert error_kind(_solution(error=HARNESS_ERROR_PREFIX + "boom")) == "harness"
    assert (
        error_kind(_solution(error="Evaluation timed out after 5 seconds."))
        == "timeout"
    )
    assert error_kind(_solution(error="NameError: x")) == "candidate"
    text = "In the code, line 3:\nZeroDivisionError: division by zero\nOn line: 1/0"
    assert _error_summary(text) == "ZeroDivisionError: division by zero"
    assert _error_summary("just text") == "just text"


def test_planned_runs_have_llm_name_and_no_duplicates(tmp_path):
    logger = ExperimentLogger(str(tmp_path / "exp"))
    method = _Method(_LLM("gpt/x:1"), 3, name="m")
    problem = _Problem("p")
    logger.start_progress(
        "id", 2, methods=[method], problems=[problem], seeds=[0, 1], budget=3
    )
    logger.open_run(method, problem, 3, 0)
    runs = logger.progress["runs"]
    assert len(runs) == 2  # used to be 3: the started run got a second entry
    assert {r["llm_name"] for r in runs} == {"gpt-x-1"}


def test_old_progress_files_are_migrated(tmp_path):
    exp = tmp_path / "exp"
    exp.mkdir()
    old = {
        "runs": [
            {"method_name": "m", "problem_name": "p", "seed": 0, "end_time": None},
            {"method_name": "m", "problem_name": "p", "seed": 1, "end_time": None},
            {
                "method_name": "m",
                "problem_name": "p",
                "seed": 0,
                "llm_name": "x",
                "end_time": "t",
            },
        ]
    }
    (exp / "progress.json").write_text(json.dumps(old))
    logger = ExperimentLogger(str(exp))
    method = _Method(_LLM("x"), 3, name="m")
    logger.start_progress(
        "id", 2, methods=[method], problems=[_Problem("p")], seeds=[0, 1], budget=3
    )
    runs = logger.progress["runs"]
    assert len(runs) == 2
    assert all(r["llm_name"] == "x" for r in runs)
    assert not logger.is_run_pending(method, "x", _Problem("p"), 0)
    assert logger.is_run_pending(method, "x", _Problem("p"), 1)


@pytest.mark.parametrize("minimisation,best", [(False, 3.0), (True, 1.0)])
def test_best_fitness_and_error_counts(tmp_path, minimisation, best):
    logger = ExperimentLogger(str(tmp_path / "exp"))
    method = _Method(_LLM(), 5, name="m")
    problem = _Problem("p", minimisation=minimisation)
    logger.start_progress(
        "id", 1, methods=[method], problems=[problem], seeds=[0], budget=5
    )
    run = logger.open_run(method, problem, 5, 0)
    for s in (
        _solution(1.0),
        _solution(3.0),
        _solution(error="NameError: x"),
        _solution(error=HARNESS_ERROR_PREFIX + "Traceback\nImportError: nope"),
        _solution(error="Evaluation timed out after 1 seconds."),
    ):
        run.log_individual(s)
    entry = logger.progress["runs"][0]
    assert entry["evaluations"] == 5
    assert entry["best_fitness"] == best
    assert entry["error_counts"] == {"candidate": 1, "harness": 1, "timeout": 1}
    assert entry["last_error"].startswith("timeout: Evaluation timed out")
    # A retry starts clean.
    logger.open_run(method, problem, 5, 0)
    entry = logger.progress["runs"][0]
    assert "best_fitness" not in entry and "error_counts" not in entry


def test_old_style_progress_callbacks_still_work(tmp_path):
    calls = []
    run = RunLogger(
        "r", str(tmp_path), budget=3, progress_callback=lambda: calls.append(1)
    )
    run.log_individual(_solution(1.0))
    seen = []
    run = RunLogger("r2", str(tmp_path), budget=3, progress_callback=seen.append)
    solution = _solution(2.0)
    run.log_individual(solution)
    assert calls == [1] and seen == [solution]


# ------------------------------------------------------------------ logs ---


def test_experiment_logging_files_routing_and_restore(tmp_path):
    terminal = logging.StreamHandler(sys.stderr)
    library = logging.getLogger("some.library")
    library.addHandler(terminal)
    logging.disable(logging.CRITICAL)  # what older versions left behind
    try:
        with experiment_logging(str(tmp_path), "INFO", quiet=True) as router:
            assert terminal not in library.handlers  # detached while running
            log.info("experiment message")
            log.debug("hidden at INFO")

            def run():
                router.register(str(tmp_path / "run.log"))
                logging.getLogger("some.library").warning("from the run thread")
                # (pyproject's pytest config ignores UserWarning)
                warnings.warn("a python warning", FutureWarning)
                router.unregister()

            thread = threading.Thread(target=run)
            thread.start()
            thread.join()
        assert terminal in library.handlers  # restored
        assert logging.root.manager.disable == logging.CRITICAL  # restored
    finally:
        library.removeHandler(terminal)
        logging.disable(logging.NOTSET)
    experiment_log = (tmp_path / "experiment.log").read_text()
    run_log = (tmp_path / "run.log").read_text()
    assert (
        "experiment message" in experiment_log
        and "hidden at INFO" not in experiment_log
    )
    assert "from the run thread" in experiment_log and "from the run thread" in run_log
    assert "a python warning" in run_log
    assert "experiment message" not in run_log


def test_experiment_logging_rejects_unknown_level(tmp_path):
    with pytest.raises(ValueError):
        with experiment_logging(str(tmp_path), "LOUD"):
            pass


# ------------------------------------------------------------ experiment ---


def test_experiment_writes_logs_progress_and_summary(tmp_path, capfd):
    exp = Experiment(
        methods=[_Method(_LLM(), 3, name="good"), _Failing(_LLM(), 3, name="bad")],
        problems=[_Problem("p")],
        runs=1,
        budget=3,
        exp_logger=ExperimentLogger(str(tmp_path / "exp")),
        log_level="DEBUG",
    )
    with pytest.raises(RuntimeError, match="1 of 2 runs failed"):
        exp()
    out = capfd.readouterr().out  # written to the real terminal (sys.__stdout__)
    assert "Summary: 1 of 2 runs finished, 1 failed." in out
    assert "KeyError (1 run(s)):" in out
    assert "Evaluations: 3, of which failed: 1 in the generated code." in out
    assert "Rerunning the experiment retries only the failed runs." in out
    rows = [line.split() for line in out.splitlines() if line.startswith("good ")]
    assert rows and rows[-1][:7] == ["good", "m", "p", "0", "3/3", "3", "1c"]

    root = tmp_path / "exp"
    experiment_log = (root / "experiment.log").read_text()
    assert "Run failed: bad / p / seed 0: KeyError" in experiment_log
    assert "noisy library message" in experiment_log
    progress = json.loads((root / "progress.json").read_text())
    assert len(progress["runs"]) == 2
    good = next(r for r in progress["runs"] if r["method_name"] == "good")
    run_log = (root / good["log_dir"] / "run.log").read_text()
    assert "Run started: good" in run_log and "Run finished" in run_log
    assert "noisy library message" in run_log


def test_environment_errors_are_collected_before_runs(tmp_path):
    class _BrokenEnv(_Problem):
        def _ensure_env(self):
            raise EnvironmentSetupError(
                "Failed to install the dependencies ['x']:\nerror: no swig",
                full_output="line 1 of the resolver\n"
                + "...\n" * 50
                + "error: no swig",
            )

    started = []

    class _Recording(_Method):
        def __call__(self, problem):
            started.append(problem.name)
            return super().__call__(problem)

    exp = Experiment(
        methods=[_Recording(_LLM(), 3, name="m")],
        problems=[
            _BrokenEnv("one"),
            _Problem("fine"),
            _BrokenEnv("two"),
            _BrokenEnv("a b"),
            _BrokenEnv("a_b"),
        ],
        runs=1,
        exp_logger=ExperimentLogger(str(tmp_path / "exp")),
    )
    with pytest.raises(EnvironmentSetupError) as exc:
        exp()
    text = str(exc.value)
    assert "of 4 problem(s); no runs were started" in text
    assert "- one: Failed to install the dependencies ['x']:" in text
    assert "- two:" in text and "iohblade check" in text
    assert started == []
    errors = tmp_path / "exp" / "env-errors"
    one = (errors / "one.txt").read_text()
    assert "line 1 of the resolver" in one  # the full output, not the tail
    assert sorted(p.name for p in errors.iterdir()) == [
        "a_b-2.txt",
        "a_b.txt",
        "one.txt",
        "two.txt",
    ]


def test_format_error_counts():
    assert _format_error_counts({}) == "-"
    assert _format_error_counts({"harness": 2, "candidate": 1}) == "1c 2h"


def test_cli_exit_code_for_environment_errors(tmp_path, monkeypatch, capsys):
    import yaml

    from iohblade import cli

    data = {
        "experiment": {
            "output_dir": "out",
            "runs": 1,
            "budget": 2,
            "log_level": "DEBUG",
        },
        "llms": {"d": {"provider": "dummy"}},
        "methods": [{"type": "random_search", "llm": "d"}],
        "problems": [{"type": "circle_packing"}],
    }
    path = tmp_path / "exp.yml"
    path.write_text(yaml.safe_dump(data))

    seen = {}

    def fail(self):
        seen["level"] = self.log_level
        raise EnvironmentSetupError("Could not build the evaluation environment")

    monkeypatch.setattr(Experiment, "__call__", fail)
    assert cli.main(["run", str(path)]) == cli.EXIT_ENVIRONMENT
    assert seen["level"] == "DEBUG"
    assert "Could not build" in capsys.readouterr().err


# ---------------------------------------------------------------- webapp ---


def test_webapp_helpers(tmp_path):
    from iohblade.webapp import read_record, run_status_rows, tail_file

    progress = {
        "runs": [
            {
                "method_name": "m",
                "problem_name": "p",
                "seed": 0,
                "end_time": "t",
                "evaluations": 3,
                "budget": 3,
                "best_fitness": 2.5,
                "error_counts": {"candidate": 1},
                "llm_name": "x",
            },
            {
                "method_name": "m",
                "problem_name": "p",
                "seed": 1,
                "error": "KeyError: k",
            },
            {"method_name": "m", "problem_name": "p", "seed": 2, "start_time": "s"},
            {"method_name": "m", "problem_name": "p", "seed": 3},
        ]
    }
    rows = run_status_rows(progress)
    assert [r["Status"] for r in rows] == ["done", "failed", "running", "pending"]
    assert rows[0]["Best fitness"] == 2.5 and rows[0]["Errors in code"] == 1
    assert rows[1]["Last error"] == "KeyError: k"
    assert run_status_rows(None) == []

    (tmp_path / "blade_config").mkdir()
    (tmp_path / "blade_config" / "resolved.yml").write_text("a: 1\n")
    (tmp_path / "blade_config" / "sources.json").write_text('{"mod": {"sha256": "ab"}}')
    (tmp_path / "blade_config-20260101-000000").mkdir()
    record = read_record(str(tmp_path))
    assert record["resolved"] == "a: 1\n" and record["sources"] == {
        "mod": {"sha256": "ab"}
    }
    assert record["archived"] == ["blade_config-20260101-000000"]
    assert read_record(str(tmp_path / "missing")) == {
        "resolved": None,
        "sources": {},
        "archived": [],
    }

    log = tmp_path / "experiment.log"
    log.write_text("".join(f"line {i}\n" for i in range(300)))
    assert tail_file(str(log), 2) == "line 298\nline 299\n"
    assert tail_file(str(tmp_path / "nope")) is None
