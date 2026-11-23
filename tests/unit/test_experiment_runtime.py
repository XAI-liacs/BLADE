"""Regression tests for the experiment runner and loggers (issue #207).

Covers: MLflow runs with several (parallel) runs, per-thread output routing,
output of failed runs, parallel seeding warning, atomic progress.json,
Trackio run names and run objects, ReEvo's files, the isolated iohblade path
and the configspace folder.
"""

import json
import os
import sys
import threading
import time
import warnings
from types import SimpleNamespace
from typing import Any

import pytest

from iohblade import envs
from iohblade.experiment import PARALLEL_SEED_WARNING, Experiment
from iohblade.llm import Dummy_LLM
from iohblade.loggers import ExperimentLogger, RunLogger
from iohblade.method import Method
from iohblade.problem import Problem
from iohblade.solution import Solution
from iohblade.utils import HarnessError

# --------------------------------------------------------------------------
# helpers


class NoEnvProblem(Problem):
    """A problem that needs no evaluation environment."""

    def _ensure_env(self):
        pass

    def cleanup(self):
        pass

    def get_prompt(self):
        return "prompt"

    def evaluate(self, s):
        return s

    def test(self, s):
        return s

    def to_dict(self):
        return {}

    def get_config(self) -> dict[str, Any]:
        return {}


class PrintingMethod(Method):
    """Prints tagged lines, logs two evaluations (one from a helper thread).

    A method named ``bad`` raises a HarnessError at the end.
    """

    lines = 100

    def __call__(self, problem):
        tag = f"{self.name}"
        for i in range(self.lines):
            print(f"[{tag}] out {i}")
            print(f"[{tag}] err {i}", file=sys.stderr)
            if i % 10 == 0:
                time.sleep(0.001)  # let the threads interleave
        seed_value = float(len(self.name))
        logger = problem.logger
        if logger is not None:
            first = Solution(name="first")
            first.set_scores(seed_value)
            logger.log_individual(first)
            # Some methods evaluate in helper threads.
            second = Solution(name="second")
            second.set_scores(seed_value + 0.5)
            helper = threading.Thread(target=logger.log_individual, args=(second,))
            helper.start()
            helper.join()
        if self.name == "bad":
            raise HarnessError("broken harness")
        sol = Solution(name="best")
        sol.set_scores(seed_value + 0.5)
        return sol

    def to_dict(self):
        return {}

    def get_config(self) -> dict[str, Any]:
        return {}


def _run_dirs(root):
    return sorted(d for d in os.listdir(root) if d.startswith("run-"))


def _tags_in(path):
    if not os.path.exists(path):
        return set(), 0
    with open(path) as f:
        lines = [l for l in f.read().splitlines() if l.startswith("[")]
    return {l[1:].split("]")[0] for l in lines}, len(lines)


# --------------------------------------------------------------------------
# 1. MLflow


@pytest.fixture
def mlflow_uri(tmp_path, monkeypatch):
    mlflow = pytest.importorskip("mlflow")
    monkeypatch.setenv("MLFLOW_DISABLE_AGENT_HINT", "1")
    monkeypatch.chdir(tmp_path)  # default artifact location: ./mlruns
    previous = mlflow.get_tracking_uri()
    uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    yield uri
    while mlflow.active_run() is not None:
        mlflow.end_run()
    mlflow.set_tracking_uri(previous)


def _mlflow_runs(uri, experiment_id):
    from mlflow.tracking import MlflowClient

    client = MlflowClient(tracking_uri=uri)
    return client, client.search_runs([experiment_id], max_results=100)


@pytest.mark.parametrize("n_jobs", [1, 2])
def test_mlflow_experiment_logs_each_run_to_its_own_run(tmp_path, mlflow_uri, n_jobs):
    import mlflow

    from iohblade.loggers import MLFlowExperimentLogger

    logger = MLFlowExperimentLogger(
        name=str(tmp_path / "exp"), mlflow_tracking_uri=mlflow_uri
    )
    llm = Dummy_LLM("dummy")
    PrintingMethod.lines = 2
    try:
        exp = Experiment(
            methods=[
                PrintingMethod(llm, 2, name="m1"),
                PrintingMethod(llm, 2, name="m22"),
            ],
            problems=[NoEnvProblem(name="p")],
            seeds=[0, 1],
            n_jobs=n_jobs,
            exp_logger=logger,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            exp()
    finally:
        PrintingMethod.lines = 100

    client, runs = _mlflow_runs(mlflow_uri, logger.experiment_id)
    # Exactly one MLflow run per BLADE run: no auto-created runs.
    assert len(runs) == 4
    assert mlflow.active_run() is None
    seen = set()
    for run in runs:
        assert run.info.status == "FINISHED"
        params = run.data.params
        name = params["method_name"]
        seen.add((name, params["seed"]))
        assert run.data.tags["method_name"] == name
        assert run.info.run_name == f"{name}-dummy-p-{params['seed']}"
        expected = float(len(name))
        history = client.get_metric_history(run.info.run_id, "fitness")
        # both evaluations (main and helper thread) went to this run
        assert sorted(m.value for m in history) == [expected, expected + 0.5]
        assert run.data.metrics["final_fitness"] == expected + 0.5
        artifacts = {a.path for a in client.list_artifacts(run.info.run_id)}
        assert "final_run_object.json" in artifacts
        assert sum(a.startswith("solution_") for a in artifacts) == 2
    assert seen == {("m1", "0"), ("m1", "1"), ("m22", "0"), ("m22", "1")}


def test_mlflow_logger_open_runs_and_log_from_worker_threads(tmp_path, mlflow_uri):
    """The sequence Experiment uses: open all runs, then log in threads."""
    from iohblade.loggers import MLFlowExperimentLogger

    logger = MLFlowExperimentLogger(
        name=str(tmp_path / "exp"), mlflow_tracking_uri=mlflow_uri
    )
    llm = Dummy_LLM("dummy")
    opened = []
    for seed in (0, 1, 2):
        method = PrintingMethod(llm, 2, name="m")
        problem = NoEnvProblem(name="p")
        opened.append(
            (method, problem, seed, logger.open_run(method, problem, 2, seed))
        )

    errors = []

    def work(run_logger, seed):
        try:
            sol = Solution(name=f"s{seed}")
            sol.set_scores(float(seed))
            run_logger.log_individual(sol)
            run_logger.log_code(sol)
            run_logger.log_conversation("user", f"hello {seed}")
        except BaseException as e:  # surfaced below; threads swallow errors
            errors.append(repr(e))

    threads = [
        threading.Thread(target=work, args=(run_logger, seed))
        for _, _, seed, run_logger in opened
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []

    method, problem, seed, run_logger = opened[0]
    logger.mark_run_failed(method, problem, seed, "ValueError: boom")
    for method, problem, seed, run_logger in opened[1:]:
        sol = Solution(name="final")
        sol.set_scores(10.0 + seed)
        logger.add_run(method, problem, llm, sol, log_dir=run_logger.dirname, seed=seed)

    client, runs = _mlflow_runs(mlflow_uri, logger.experiment_id)
    assert len(runs) == 3
    by_seed = {run.data.tags["seed"]: run for run in runs}
    assert by_seed["0"].info.status == "FAILED"
    assert by_seed["1"].info.status == "FINISHED"
    assert by_seed["2"].info.status == "FINISHED"
    for seed, run in by_seed.items():
        assert run.data.metrics["fitness"] == float(seed)
        artifacts = {a.path for a in client.list_artifacts(run.info.run_id)}
        assert "conversation_0.json" in artifacts
    assert not logger._mlflow_run_active


def test_mlflow_file_store_needs_opt_in(tmp_path, monkeypatch):
    """The docstring no longer suggests a file: URI that MLflow 3 refuses."""
    pytest.importorskip("mlflow")
    from iohblade.loggers import MLFlowExperimentLogger

    doc = MLFlowExperimentLogger.__init__.__doc__
    assert "sqlite:///" in doc and "MLFLOW_ALLOW_FILE_STORE" in doc


# --------------------------------------------------------------------------
# 2a/3. Output of parallel runs


def _experiment(tmp_path, n_jobs=4, **kwargs):
    llm = Dummy_LLM("dummy")
    return Experiment(
        methods=[
            PrintingMethod(llm, 2, name="aa"),
            PrintingMethod(llm, 2, name="bbb"),
            PrintingMethod(llm, 2, name="bad"),
        ],
        problems=[NoEnvProblem(name="p")],
        seeds=[0, 1],
        n_jobs=n_jobs,
        exp_logger=ExperimentLogger(str(tmp_path / "exp")),
        **kwargs,
    )


def test_parallel_output_goes_to_its_own_run_and_streams_are_restored(tmp_path):
    stdout, stderr = sys.stdout, sys.stderr
    exp = _experiment(tmp_path, log_stdout=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(RuntimeError, match="2 of 6 runs failed"):
            exp()
    assert sys.stdout is stdout and sys.stderr is stderr

    root = exp.exp_logger.dirname
    dirs = _run_dirs(root)
    assert len(dirs) == 6
    for d in dirs:
        tag = d.split("-")[1]
        for name in ("stdout.log", "stderr.log"):
            tags, count = _tags_in(os.path.join(root, d, name))
            # Everything printed by the run, nothing from other runs; also
            # for the runs that failed (their output used to be lost).
            assert tags == {tag}, (d, name, tags)
            assert count == PrintingMethod.lines, (d, name, count)


def test_quiet_runs_print_nothing_and_shown_runs_print(tmp_path, capsys):
    exp = _experiment(tmp_path / "quiet", n_jobs=2)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(RuntimeError):
            exp()
    out, err = capsys.readouterr()
    assert "[aa]" not in out and "[aa]" not in err

    exp = _experiment(tmp_path / "shown", n_jobs=2, show_stdout=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(RuntimeError):
            exp()
    out, err = capsys.readouterr()
    assert "[aa] out 0" in out and "[aa] err 0" in err


def test_streams_are_restored_after_keyboard_interrupt(tmp_path, monkeypatch):
    stdout, stderr = sys.stdout, sys.stderr
    exp = _experiment(tmp_path, n_jobs=2, log_stdout=True)

    def interrupt(*args, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(exp, "_print_summary", interrupt)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(KeyboardInterrupt):
            exp()
    assert sys.stdout is stdout and sys.stderr is stderr


def test_failed_run_output_is_kept_sequential(tmp_path):
    """Item 3 without threads: a run that raises keeps its output."""
    llm = Dummy_LLM("dummy")
    exp = Experiment(
        methods=[PrintingMethod(llm, 2, name="bad")],
        problems=[NoEnvProblem(name="p")],
        seeds=[0],
        log_stdout=True,
        exp_logger=ExperimentLogger(str(tmp_path / "exp")),
    )
    with pytest.raises(RuntimeError, match="1 of 1 runs failed"):
        exp()
    root = exp.exp_logger.dirname
    (run,) = _run_dirs(root)
    assert _tags_in(os.path.join(root, run, "stdout.log")) == (
        {"bad"},
        PrintingMethod.lines,
    )


# --------------------------------------------------------------------------
# 2b. Seeding with parallel runs


def test_parallel_runs_warn_that_seeds_are_not_reproducible(tmp_path):
    exp = _experiment(tmp_path, n_jobs=2)
    with pytest.warns(RuntimeWarning, match="not exactly reproducible"):
        exp._warn_parallel_seeds()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        exp._warn_parallel_seeds()  # only once

    exp = _experiment(tmp_path / "run", n_jobs=2)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(RuntimeError):
            exp()
    with open(os.path.join(exp.exp_logger.dirname, "experiment.log")) as f:
        assert PARALLEL_SEED_WARNING in f.read()
    assert "not exactly reproducible" in Experiment.__init__.__doc__


def test_sequential_runs_do_not_warn(tmp_path):
    exp = _experiment(tmp_path, n_jobs=1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(RuntimeError):
            exp()
    with open(os.path.join(exp.exp_logger.dirname, "experiment.log")) as f:
        assert PARALLEL_SEED_WARNING not in f.read()


# --------------------------------------------------------------------------
# 4. progress.json


def test_progress_is_written_atomically(tmp_path, monkeypatch):
    logger = ExperimentLogger(str(tmp_path / "exp"))
    path = os.path.join(logger.dirname, "progress.json")
    with open(path) as f:
        before = f.read()

    import iohblade.loggers.base as base

    real_dump = json.dump

    def killed(obj, f, *args, **kwargs):
        f.write('{"start_time": "2026')  # partially written ...
        raise KeyboardInterrupt  # ... when the process is interrupted

    monkeypatch.setattr(base.json, "dump", killed)
    logger.progress["current"] = 99
    with pytest.raises(KeyboardInterrupt):
        logger._write_progress()
    monkeypatch.setattr(base.json, "dump", real_dump)

    with open(path) as f:
        assert f.read() == before  # the previous complete progress
    assert os.listdir(logger.dirname) == ["progress.json"]  # no temp file left
    ExperimentLogger(logger.dirname)  # resuming works


def test_unreadable_progress_names_the_file(tmp_path):
    exp_dir = tmp_path / "exp"
    exp_dir.mkdir()
    path = exp_dir / "progress.json"
    path.write_text('{"start_time": "2026-')  # truncated
    with pytest.raises(ValueError, match="progress file") as info:
        ExperimentLogger(str(exp_dir))
    assert str(path) in str(info.value)
    assert path.read_text() == '{"start_time": "2026-'  # not discarded


# --------------------------------------------------------------------------
# 5. Trackio (fake module: trackio is an optional dependency)


class _FakeTrackioRun:
    def __init__(self, module, project, name):
        self.module, self.project, self.name = module, project, name
        self.logs = []
        self.finished = 0

    def log(self, metrics, step=None):
        self.logs.append(metrics)

    def finish(self):
        self.finished += 1


class _FakeTrackio:
    """Mimics trackio: the current run is a per-thread context variable."""

    def __init__(self):
        self.runs = []
        self._current = threading.local()

    def init(self, project, name=None, **kwargs):
        run = _FakeTrackioRun(self, project, name)
        self.runs.append(run)
        self._current.run = run
        return run

    def log(self, metrics, step=None):
        run = getattr(self._current, "run", None)
        if run is None:
            raise RuntimeError("Call trackio.init() before trackio.log().")
        run.log(metrics, step)

    def finish(self):
        run = getattr(self._current, "run", None)
        if run is None:
            raise RuntimeError("Call trackio.init() before trackio.finish().")
        run.finish()
        self._current.run = None


@pytest.fixture
def fake_trackio(monkeypatch):
    import iohblade.loggers.trackio as trackio_logger

    fake = _FakeTrackio()
    monkeypatch.setitem(sys.modules, "trackio", fake)
    monkeypatch.setattr(trackio_logger, "trackio", fake)
    return fake


def test_trackio_runs_are_named_and_kept_apart(tmp_path, fake_trackio):
    from iohblade.loggers.trackio import TrackioExperimentLogger

    logger = TrackioExperimentLogger(name=str(tmp_path / "exp"))
    llm = Dummy_LLM("dummy")
    PrintingMethod.lines = 1
    try:
        exp = Experiment(
            methods=[
                PrintingMethod(llm, 2, name="m1"),
                PrintingMethod(llm, 2, name="m22"),
            ],
            problems=[NoEnvProblem(name="p")],
            seeds=[0, 1],
            n_jobs=2,
            exp_logger=logger,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            exp()
    finally:
        PrintingMethod.lines = 100

    names = sorted(run.name for run in fake_trackio.runs)
    assert names == ["m1-dummy-p-0", "m1-dummy-p-1", "m22-dummy-p-0", "m22-dummy-p-1"]
    for run in fake_trackio.runs:
        method = run.name.split("-")[0]
        expected = float(len(method))
        fitness = [m["fitness"] for m in run.logs if "fitness" in m]
        # including the evaluation logged from the method's helper thread
        assert sorted(fitness) == [expected, expected + 0.5]
        assert run.logs[0]["method_name"] == method
        assert run.finished == 1


def test_trackio_run_name_given_by_keyword_or_position(tmp_path, fake_trackio):
    from iohblade.loggers.trackio import TrackioRunLogger

    by_keyword = TrackioRunLogger(name="kw", root_dir=str(tmp_path), project="p")
    by_position = TrackioRunLogger("pos", str(tmp_path), project="p")
    assert by_keyword._run_name == "kw"
    assert by_position._run_name == "pos"


# --------------------------------------------------------------------------
# 6. ReEvo writes its files into a run directory, never the working directory


def _reevo_problem():
    from tests.unit.test_moeh import DummyProblemWorks

    return DummyProblemWorks(minimisation=False)


def _reevo(**kwargs):
    pytest.importorskip("reevo")
    from iohblade.methods.reevo import ReEvo

    return ReEvo(Dummy_LLM("dummy"), 12, pop_size=2, init_pop_size=2, **kwargs)


def _reflections(directory):
    return sorted(f for f in os.listdir(directory) if f.startswith("problem_iter"))


def test_reevo_files_go_to_the_run_directory(tmp_path, monkeypatch):
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    monkeypatch.chdir(cwd)
    problem = _reevo_problem()
    problem.set_logger(RunLogger(name="r", root_dir=str(tmp_path / "exp")))
    _reevo()(problem)
    assert os.listdir(cwd) == []
    assert os.getcwd() == str(cwd)
    files = _reflections(os.path.join(problem.logger_dir, "reevo"))
    assert any(f.endswith("_long_term_reflection.txt") for f in files)


def test_reevo_respects_output_path(tmp_path, monkeypatch):
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    monkeypatch.chdir(cwd)
    _reevo(output_path=str(tmp_path / "out"))(_reevo_problem())
    assert os.listdir(cwd) == []
    assert _reflections(tmp_path / "out")


def test_reevo_without_logger_uses_a_temporary_directory(tmp_path, monkeypatch):
    import tempfile

    cwd = tmp_path / "cwd"
    cwd.mkdir()
    monkeypatch.chdir(cwd)
    used = []
    real = tempfile.TemporaryDirectory

    class Recording(real):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            used.append(self.name)

    monkeypatch.setattr(tempfile, "TemporaryDirectory", Recording)
    _reevo()(_reevo_problem())
    assert os.listdir(cwd) == []
    assert used and not any(os.path.exists(path) for path in used)  # removed


# --------------------------------------------------------------------------
# 7. iohblade_only_path


@pytest.fixture
def cache_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("IOHBLADE_CACHE_DIR", str(tmp_path / "cache"))
    return tmp_path / "cache"


def test_isolated_path_link_created_concurrently(cache_dir, monkeypatch):
    """Another thread/process creates the link first: that is fine."""
    real_symlink = os.symlink

    def lose_the_race(src, dst, target_is_directory=False):
        real_symlink(src, dst, target_is_directory=target_is_directory)
        raise FileExistsError(17, "File exists", str(dst))

    monkeypatch.setattr(envs.os, "symlink", lose_the_race)
    shim = envs.iohblade_only_path()
    assert (shim / "iohblade").is_symlink()
    assert (shim / "iohblade" / "__init__.py").exists()


def test_isolated_path_from_many_threads(cache_dir):
    barrier = threading.Barrier(8)
    results, errors = [], []

    def call():
        barrier.wait()
        try:
            results.append(envs.iohblade_only_path())
        except Exception as e:  # pragma: no cover - the bug
            errors.append(e)

    threads = [threading.Thread(target=call) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors
    assert len(set(results)) == 1
    assert (results[0] / "iohblade" / "__init__.py").exists()


def test_isolated_path_copy_is_refreshed_after_upgrade(cache_dir, monkeypatch):
    def no_symlinks(*args, **kwargs):
        raise OSError("symlinks not allowed")

    monkeypatch.setattr(envs.os, "symlink", no_symlinks)
    shim = envs.iohblade_only_path()
    copy = shim / "iohblade"
    assert copy.is_dir() and not copy.is_symlink()
    assert (copy / "__init__.py").exists()
    assert sorted(p.name for p in shim.iterdir()) == ["iohblade"]

    stale = copy / "stale_marker.py"
    stale.write_text("# left from the previous version")
    assert envs.iohblade_only_path() == shim
    assert stale.exists()  # unchanged package: the copy is reused

    import importlib.metadata

    monkeypatch.setattr(importlib.metadata, "version", lambda name: "999.0")
    assert envs.iohblade_only_path() == shim
    assert not stale.exists()  # upgraded: copied again
    assert (copy / "__init__.py").exists()


# --------------------------------------------------------------------------
# 8. configspace folder


def test_log_configspace_creates_its_folder(tmp_path):
    logger = RunLogger(name="r", root_dir=str(tmp_path))
    sol = Solution(name="alg")
    logger.log_configspace(sol)
    path = os.path.join(logger.dirname, "configspace", f"{sol.id}-alg.py")
    with open(path) as f:
        assert f.read() == "Failed to extract config space"
