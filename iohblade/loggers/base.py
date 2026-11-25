import json
import os
from datetime import datetime
from functools import partial
from threading import Lock

import jsonlines
import pandas as pd
from ConfigSpace.read_and_write import json as cs_json

from ..llm import LLM
from ..method import Method
from ..problem import Problem
from ..solution import Solution
from ..utils import convert_to_serializable

ERROR_KINDS = ("candidate", "harness", "timeout")


def error_kind(solution) -> str | None:
    """Classify an evaluation error: candidate, harness, timeout (or None).

    * ``harness``: the evaluation failed before the generated code was used
      (see :data:`iohblade.problem.HARNESS_ERROR_PREFIX`);
    * ``timeout``: the evaluation timed out;
    * ``candidate``: any other error, i.e. in the generated code.
    """
    from ..problem import HARNESS_ERROR_PREFIX

    error = str(getattr(solution, "error", "") or "")
    if not error:
        return None
    if error.startswith(HARNESS_ERROR_PREFIX):
        return "harness"
    if "timed out" in error.lower():
        return "timeout"
    return "candidate"


def _error_summary(error) -> str:
    """The line of an error message that names the exception (max 200 chars)."""
    import re

    lines = [l.strip() for l in str(error).strip().splitlines() if l.strip()]
    for line in lines:
        if re.match(r"^[A-Za-z_][\w.]*(Error|Exception|Exit|Interrupt)\b", line):
            return line[:200]
    return (lines[-1] if lines else "")[:200]


def _record_outcome(entry: dict, solution) -> None:
    """Update a progress entry with the outcome of one evaluation."""
    kind = error_kind(solution)
    if kind is not None:
        counts = entry.setdefault("error_counts", {})
        counts[kind] = counts.get(kind, 0) + 1
        entry["last_error"] = f"{kind}: {_error_summary(solution.error)}"
        return
    try:
        fitness = float(solution.fitness)
    except (TypeError, ValueError):
        return  # multi-objective
    if fitness != fitness or fitness in (float("inf"), float("-inf")):
        return
    best = entry.get("best_fitness")
    minimise = entry.get("minimisation", False)
    if best is None or (fitness < best if minimise else fitness > best):
        entry["best_fitness"] = fitness


def _llm_name_from_dict(llm) -> str | None:
    """The cleaned model name of a logged LLM record (``None`` if unknown)."""
    if not isinstance(llm, dict) or not llm.get("model"):
        return None
    return str(llm["model"]).replace("/", "-").replace(":", "-")


def _jsonl_lines(path) -> list[str]:
    """The parsable lines of a jsonl file.

    An unparsable *last* line is skipped: it is a record that is still being
    written (e.g. an experiment that is running). An unparsable line before
    the last one means the file is corrupt and raises an error.

    Raises:
        ValueError: A line other than the last one is not valid JSON.
    """
    with open(path, encoding="utf-8") as f:
        lines = [line for line in f.read().splitlines() if line.strip()]
    good = []
    for i, line in enumerate(lines):
        try:
            json.loads(line)
        except ValueError as e:
            if i == len(lines) - 1:
                break  # half-written trailing record
            raise ValueError(f"{path}: line {i + 1} is not valid JSON: {e}") from e
        good.append(line)
    return good


def read_jsonl_records(path) -> list[dict]:
    """The records of a jsonl file, skipping a half-written last line."""
    return [json.loads(line) for line in _jsonl_lines(path)]


def read_jsonl_dataframe(path) -> pd.DataFrame:
    """A jsonl file as a DataFrame, skipping a half-written last line."""
    from io import StringIO

    lines = _jsonl_lines(path)
    if not lines:
        return pd.DataFrame()
    return pd.read_json(StringIO("\n".join(lines)), lines=True)


def _accepts_argument(callback) -> bool:
    import inspect

    try:
        params = inspect.signature(callback).parameters.values()
    except (TypeError, ValueError):
        return False
    return any(
        p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD, p.VAR_POSITIONAL)
        for p in params
    )


class ExperimentLogger:
    """
    Logs an entire experiment of multiple runs.
    """

    def __init__(self, name="", read=False):
        """Create or load an experiment logging directory.

        If ``read`` is ``True`` the logger is opened in read only mode.
        Otherwise a new directory is created unless ``name`` already exists, in
        which case the existing directory is used so runs can be restarted.

        Parameters
        ----------
        name: str
            Path to the experiment directory.
        read: bool
            If ``True`` open the directory for reading only.
        """

        self.dirs = []
        self._lock = Lock()
        if read:
            # Only reading previous results
            self.dirs.append(name)
            self.dirname = name
            self._load_progress()
            return

        if os.path.exists(name):
            # Reuse the directory for restarting
            self.dirname = name
            self.dirs.append(self.dirname)
            self._load_progress()
            if not hasattr(self, "progress") or not self.progress:
                self.progress = {
                    "start_time": datetime.now().isoformat(),
                    "end_time": None,
                    "current": 0,
                    "total": 0,
                }
                self._write_progress()
        else:
            self.dirname = self.create_log_dir(name)
            self.dirs.append(self.dirname)
            self.progress = {
                "start_time": datetime.now().isoformat(),
                "end_time": None,
                "current": 0,
                "total": 0,
            }
            self._write_progress()

    def add_read_dir(self, dir_path: str):
        """
        Register another finished experiment so that *this* logger will read it
        when you call get_data(), get_problem_data(), etc.
        """
        if not os.path.isdir(dir_path):
            raise ValueError(f"{dir_path} is not a directory")
        if not os.path.isfile(os.path.join(dir_path, "experimentlog.jsonl")):
            raise ValueError("No experimentlog.jsonl found in the given directory")
        if dir_path not in self.dirs:
            self.dirs.append(dir_path)

    def create_log_dir(self, name=""):
        """Create a unique directory for a new experiment."""
        dirname = f"{name}"
        tempi = 0
        while os.path.exists(dirname):
            tempi += 1
            dirname = f"{name}-{tempi}"
        os.makedirs(dirname)
        return dirname

    def _before_open_run(self, run_name, method, problem, budget, seed):
        """Hook executed before a run is opened."""
        return None

    def _create_run_logger(self, run_name, budget, progress_cb):
        """Create and return the run logger for a run."""
        from .base import RunLogger

        return RunLogger(
            name=run_name,
            root_dir=self.dirname,
            budget=budget,
            progress_callback=progress_cb,
        )

    @staticmethod
    def run_budget(method, default) -> int:
        """The evaluation budget of one run of ``method``.

        A method's own ``budget`` (``methods[].budget`` in YAML) wins over
        the experiment-wide ``default``. ``progress.json`` records this value
        per run, the run logger stops evaluations at it, and resuming
        compares it with the planned one.
        """
        budget = getattr(method, "budget", None)
        return int(budget if budget is not None else default)

    def _get_clean_llm_model_name(self, llm):
        llm_name = getattr(llm, "model", "llm-unknown")
        llm_name = llm_name.replace("/", "-").replace(":", "-")
        return llm_name

    def open_run(self, method, problem, budget=100, seed=0):
        """
        Opens (starts) a new run for logging.
        Typically call this right before your run, so that the RunLogger can log step data.
        """
        llm = getattr(method, "llm", None)
        llm_name = getattr(llm, "model", "llm-unknown")
        llm_name = llm_name.replace("/", "-").replace(":", "-")

        run_name = f"{method.name}-{llm_name}-{problem.name}-{seed}"

        self._before_open_run(run_name, method, problem, budget, seed)

        progress_cb = partial(
            self.increment_eval, method.name, llm_name, problem.name, seed
        )

        self.run_logger = self._create_run_logger(run_name, budget, progress_cb)
        problem.set_logger(self.run_logger)
        with self._lock:
            entry = self._get_run_entry(method.name, llm_name, problem.name, seed)
            if entry is None:
                entry = {
                    "method_name": method.name,
                    "llm_name": llm_name,
                    "problem_name": problem.name,
                    "seed": int(seed),
                    "budget": int(budget),
                    "evaluations": 0,
                    "start_time": None,
                    "end_time": None,
                    "log_dir": None,
                }
                self.progress.setdefault("runs", []).append(entry)

            # If a previous attempt exists remove it
            if (
                entry.get("start_time")
                and not entry.get("end_time")
                and entry.get("log_dir")
            ):
                prev = os.path.join(self.dirname, entry["log_dir"])
                if os.path.exists(prev):
                    import shutil

                    shutil.rmtree(prev)
                entry["evaluations"] = 0

            entry["start_time"] = datetime.now().isoformat()
            entry["budget"] = int(budget)  # the budget this attempt runs with
            entry["minimisation"] = bool(getattr(problem, "minimisation", False))
            for key in ("best_fitness", "error_counts", "last_error"):
                entry.pop(key, None)  # a retry starts clean
            entry["log_dir"] = os.path.relpath(self.run_logger.dirname, self.dirname)
            entry["end_time"] = None
            entry.pop("error", None)  # a retry starts clean
            entry.pop("failed_time", None)
            self._write_progress()
        return self.run_logger

    def add_run(
        self,
        method: Method,
        problem: Problem,
        llm: LLM,
        solution: Solution,
        log_dir="",
        seed=None,
    ):
        """
        Adds a run to the experiment log.

        Args:
            method (Method): The method used in the run.
            problem (Problem): The problem used in the run.
            llm (LLM): The llm used in the run.
            solution (Solution): The solution found in the run.
            log_dir (str): The directory where the run is logged.
            seed (int): The seed used in the run.
        """

        try:
            rel_log_dir = os.path.relpath(log_dir, self.dirname)
        except ValueError:
            rel_log_dir = log_dir
        llm_name = self._get_clean_llm_model_name(llm)

        run_object = {
            "method_name": method.name,
            "problem_name": problem.name,
            "llm_name": llm_name,
            "method": method.to_dict(),
            "problem": problem.to_dict(),
            "llm": llm.to_dict(),
            "solution": solution.to_dict(),
            "log_dir": rel_log_dir,
            "seed": seed,
        }
        with jsonlines.open(f"{self.dirname}/experimentlog.jsonl", "a") as file:
            file.write(convert_to_serializable(run_object))
        with self._lock:
            entry = self._get_run_entry(method.name, llm_name, problem.name, seed)
            if entry is not None:
                entry["end_time"] = datetime.now().isoformat()
                entry["log_dir"] = rel_log_dir
            self._write_progress()
        self.increment_progress()

    def get_data(self):
        """
        Retrieves the data from the experiment log and returns a pandas dataframe.

        An ``llm_name`` column is added (from the logged LLM) when the log was
        written without one, so runs of one method with several LLMs can be
        told apart. An unparsable last line (a run being written while the
        log is read) is skipped.

        Returns:
            dataframe: Pandas DataFrame of the experimentlog.
        """
        frames = []
        for d in self.dirs:
            path = os.path.join(d, "experimentlog.jsonl")
            if os.path.exists(path):
                df = read_jsonl_dataframe(path)
                if not df.empty:
                    frames.append(df)
        if not frames:
            return pd.DataFrame()
        df = pd.concat(frames, ignore_index=True)
        if "llm" in df.columns:
            from_llm = df["llm"].apply(_llm_name_from_dict)
            if "llm_name" in df.columns:
                df["llm_name"] = df["llm_name"].fillna(from_llm)
            else:
                df["llm_name"] = from_llm
        return df

    def get_minimisation(self, problem_name) -> bool:
        """Whether lower fitness is better for a problem, as logged.

        The direction is read from (in this order) the run entries of
        ``progress.json``, the ``problem`` record in ``experimentlog.jsonl``
        and the ``problem.json`` file in the run folders, of every read
        directory.

        Args:
            problem_name (str): The name of the problem.

        Returns:
            bool: ``True`` when the problem is minimised; ``False`` when it is
            maximised or the direction was not logged.
        """
        for d in self.dirs:
            progress_path = os.path.join(d, "progress.json")
            if os.path.exists(progress_path):
                try:
                    with open(progress_path) as f:
                        progress = json.load(f)
                except (OSError, ValueError):
                    progress = {}
                for r in progress.get("runs", []) or []:
                    if (
                        r.get("problem_name") == problem_name
                        and r.get("minimisation") is not None
                    ):
                        return bool(r["minimisation"])
        for d in self.dirs:
            for line in self._read_experiment_log(d):
                if line.get("problem_name") != problem_name:
                    continue
                problem = line.get("problem")
                if isinstance(problem, dict) and "minimisation" in problem:
                    return bool(problem["minimisation"])
                log_dir = line.get("log_dir")
                if log_dir:
                    path = os.path.join(d, log_dir, "problem.json")
                    if os.path.exists(path):
                        try:
                            with open(path) as f:
                                config = json.load(f)
                        except (OSError, ValueError):
                            config = {}
                        if isinstance(config, dict) and "minimisation" in config:
                            return bool(config["minimisation"])
        return False

    @staticmethod
    def _read_experiment_log(d) -> list[dict]:
        path = os.path.join(d, "experimentlog.jsonl")
        if not os.path.exists(path):
            return []
        return read_jsonl_records(path)

    @staticmethod
    def _fitness_to_scalar(fitness_val) -> float:
        """Convert a fitness value to a scalar float.

        Handles both plain ``float`` values and multi-objective
        :class:`~iohblade.fitness.Fitness` dicts that were serialised to JSON.
        When the value is a dict (serialised ``Fitness``), the scalar is the
        mean of all objectives — matching ``Fitness.__float__``. Objective
        values that are not numbers (``null``, serialised ``"nan"``) count as
        NaN.

        Args:
            fitness_val: A float, int, or dict representing a fitness value.

        Returns:
            float: A scalar fitness value.
        """
        import math

        if isinstance(fitness_val, dict):
            from ..fitness import Fitness

            values = {}
            for key, value in fitness_val.items():
                try:  # serialised failures: "nan", "-inf", null
                    values[key] = float(value)
                except (TypeError, ValueError):
                    values[key] = math.nan
            return float(Fitness.from_dict(values))
        try:
            return float(fitness_val)
        except (TypeError, ValueError):
            return math.nan

    def get_problem_data(self, problem_name, include_running=False):
        """
        Retrieves the data for a specific method and problem from the experiment log.

        For multi-objective problems the ``fitness`` column contains the per-objective
        dict (as stored in ``log.jsonl``).  A companion column ``fitness_scalar`` is
        added that holds a single float for use in plots that require a scalar metric
        (e.g. convergence curves).

        Every row also gets the run it belongs to: ``method_name``,
        ``llm_name``, ``problem_name``, ``seed``, ``log_dir`` (relative to
        its experiment folder), ``run`` (the run folder path, unique across
        read directories) and ``_id`` (the evaluation index in the run,
        starting at 0). An unparsable last line of a ``log.jsonl`` (a run that
        is being written) is skipped.

        Args:
            problem_name (str): The name of the problem.
            include_running (bool): Also include runs that started but are not
                finished yet (from ``progress.json``), e.g. to watch a running
                experiment.

        Returns:
            pd.DataFrame: DataFrame of run data for the specified problem.
        """
        frames = []
        for d in self.dirs:
            runs = [
                line
                for line in self._read_experiment_log(d)
                if line.get("problem_name") == problem_name
            ]
            if include_running:
                runs += self._running_runs(d, problem_name, runs)
            for line in runs:
                if not line.get("log_dir"):
                    continue
                logdir = os.path.join(d, line["log_dir"])  # relative to *that* exp
                run_log = os.path.join(logdir, "log.jsonl")
                if not os.path.exists(run_log):
                    continue
                df = read_jsonl_dataframe(run_log)
                if df.empty:
                    continue
                df["method_name"] = line.get("method_name")
                llm_name = line.get("llm_name") or _llm_name_from_dict(line.get("llm"))
                df["llm_name"] = llm_name
                df["problem_name"] = line.get("problem_name")
                df["seed"] = line.get("seed")
                df["log_dir"] = line["log_dir"]
                df["run"] = os.path.normpath(logdir)
                df["_id"] = df.index
                # Add scalar fitness column for multi-objective support
                if "fitness" in df.columns:
                    df["fitness_scalar"] = df["fitness"].apply(self._fitness_to_scalar)
                frames.append(df)
        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, ignore_index=True)

    @staticmethod
    def _running_runs(d, problem_name, finished) -> list[dict]:
        """Run entries of ``progress.json`` that started but did not finish."""
        path = os.path.join(d, "progress.json")
        if not os.path.exists(path):
            return []
        try:
            with open(path) as f:
                progress = json.load(f)
        except (OSError, ValueError):
            return []
        seen = {os.path.normpath(r.get("log_dir") or "") for r in finished}
        running = []
        for r in progress.get("runs", []) or []:
            if (
                r.get("problem_name") == problem_name
                and r.get("start_time")
                and not r.get("end_time")
                and r.get("log_dir")
                and os.path.normpath(r["log_dir"]) not in seen
            ):
                running.append(r)
        return running

    def get_methods_problems(self):
        """
        Retrieves the list of methods and problems used in the experiment.

        Returns:
            tuple: Tuple of lists containing the method and problem names.
        """
        methods, problems = set(), set()
        for d in self.dirs:
            exp_log = os.path.join(d, "experimentlog.jsonl")
            if not os.path.exists(exp_log):
                continue
            for line in read_jsonl_records(exp_log):
                methods.add(line["method_name"])
                problems.add(line["problem_name"])
        return list(methods), list(problems)

    # Progress helpers -------------------------------------------------

    def _get_run_entry(self, method_name, llm_name, problem_name, seed):
        """Return the run progress entry matching the identifiers."""
        for r in self.progress.get("runs", []):
            if (
                r.get("method_name") == method_name
                and r.get("problem_name") == problem_name
                and r.get("llm_name") == llm_name
                and r.get("seed") == seed
            ):
                return r
        return None

    def _progress_path(self):
        return os.path.join(self.dirname, "progress.json")

    def _write_progress(self):
        """Write ``progress.json`` atomically.

        The data goes to a temporary file in the same directory, which then
        replaces ``progress.json`` (``os.replace``), so a process killed
        while writing never leaves a truncated file behind.
        """
        import threading

        path = self._progress_path()
        # Unique per process and thread; open() keeps the usual permissions.
        tmp = f"{path}.{os.getpid()}-{threading.get_ident()}.tmp"
        try:
            with open(tmp, "w") as f:
                json.dump(self.progress, f)
            for attempt in range(10):
                try:
                    os.replace(tmp, path)
                    break
                except PermissionError:  # Windows: a reader has it open
                    if attempt == 9:
                        raise
                    import time

                    time.sleep(0.05)
        except BaseException:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise

    def _load_progress(self):
        """Read ``progress.json`` (an empty progress if there is none).

        Raises:
            ValueError: ``progress.json`` exists but cannot be read; the
                message names the file. The progress is not discarded
                silently: fix or remove the file to continue.
        """
        path = self._progress_path()
        if os.path.exists(path):
            try:
                with open(path) as f:
                    self.progress = json.load(f)
            except (OSError, ValueError) as e:
                raise ValueError(
                    f"Cannot read the experiment progress file {path}: {e}. "
                    "It may have been damaged when the experiment was "
                    "interrupted. Fix the file, or delete it to restart the "
                    "progress tracking (finished runs in experimentlog.jsonl "
                    "are kept, but all runs are then treated as pending)."
                ) from e
        else:
            self.progress = {}

    def _planned_runs(self, methods, problems, seeds, budget):
        """Progress entries for every (method, problem, seed) combination."""
        runs = []
        for m in methods:
            llm_name = self._get_clean_llm_model_name(getattr(m, "llm", None))
            for p in problems:
                for s in seeds:
                    runs.append(
                        {
                            "method_name": m.name,
                            "llm_name": llm_name,
                            "problem_name": p.name,
                            "seed": int(s),
                            "budget": self.run_budget(m, budget),
                            "evaluations": 0,
                            "start_time": None,
                            "end_time": None,
                            "log_dir": None,
                        }
                    )
        return runs

    def _migrate_runs(self, methods):
        """Fix progress files written by older versions.

        Older versions created the planned entries without ``llm_name`` and
        added a second entry (with ``llm_name``) once a run started, so runs
        appeared twice. Drop such placeholders and fill in ``llm_name`` for
        runs that never started.
        """
        runs = self.progress.get("runs", [])
        with_llm = {
            (r["method_name"], r["problem_name"], r["seed"])
            for r in runs
            if r.get("llm_name") is not None
        }
        llm_of = {}
        for m in methods or []:
            llm_of.setdefault(
                m.name, self._get_clean_llm_model_name(getattr(m, "llm", None))
            )
        kept = []
        for r in runs:
            if r.get("llm_name") is None:
                if (r["method_name"], r["problem_name"], r["seed"]) in with_llm:
                    continue  # placeholder of a run that has its own entry
                if r["method_name"] in llm_of:
                    r["llm_name"] = llm_of[r["method_name"]]
            kept.append(r)
        self.progress["runs"] = kept

    def start_progress(
        self,
        id: str,
        total_runs: int,
        methods=None,
        problems=None,
        seeds=None,
        budget=None,
    ):
        """Initialize progress tracking with experiment configuration."""
        with self._lock:
            if os.path.exists(self._progress_path()):
                self._load_progress()
                self._migrate_runs(methods)
                # Validate that the planned runs match
                existing = {
                    (r["method_name"], r["problem_name"], r["seed"])
                    for r in self.progress.get("runs", [])
                }
                expected = {
                    (m.name, p.name, int(s))
                    for m in methods
                    for p in problems
                    for s in seeds
                }
                if existing and existing != expected:
                    raise ValueError(
                        "Existing progress does not match experiment setup"
                    )
                if not existing:
                    # initialize runs for the first time
                    self.progress = {
                        "id": id,
                        "start_time": datetime.now().isoformat(),
                        "end_time": None,
                        "current": 0,
                        "total": int(total_runs),
                        "runs": self._planned_runs(methods, problems, seeds, budget),
                    }
            else:
                self.progress = {
                    "id": id,
                    "start_time": datetime.now().isoformat(),
                    "end_time": None,
                    "current": 0,
                    "total": int(total_runs),
                    "runs": self._planned_runs(methods, problems, seeds, budget),
                }
            self._write_progress()

    def increment_progress(self):
        """Recalculate and write progress based on run entries."""
        with self._lock:
            finished = sum(
                1 for r in self.progress.get("runs", []) if r.get("end_time")
            )
            self.progress["current"] = finished
            total = self.progress.get("total", 0)
            if total and finished >= total and self.progress.get("end_time") is None:
                self.progress["end_time"] = datetime.now().isoformat()
            self._write_progress()

    def increment_eval(self, method_name, llm_name, problem_name, seed, solution=None):
        """Count one evaluation of a run (and record its outcome).

        Args:
            solution (Solution, optional): The evaluated solution. When given,
                the run's best fitness and its error counts by kind
                (``candidate``, ``harness``, ``timeout``) are updated.
        """
        with self._lock:
            entry = self._get_run_entry(method_name, llm_name, problem_name, seed)
            if entry is not None:
                entry["evaluations"] = entry.get("evaluations", 0) + 1
                if solution is not None:
                    _record_outcome(entry, solution)
                self._write_progress()

    def mark_run_failed(self, method, problem, seed, error):
        """Records that a run raised an exception.

        The run keeps ``end_time = None``, so it stays pending and a rerun of
        the experiment retries it.

        Args:
            method (Method): The method of the failed run.
            problem (Problem): The problem of the failed run.
            seed (int): The seed of the failed run.
            error (str): Short description of the exception.
        """
        llm_name = self._get_clean_llm_model_name(method.llm)
        with self._lock:
            entry = self._get_run_entry(method.name, llm_name, problem.name, seed)
            if entry is not None:
                entry["error"] = str(error)
                entry["failed_time"] = datetime.now().isoformat()
            self._write_progress()

    def is_run_pending(self, method, llm_name, problem, seed):
        entry = self._get_run_entry(method.name, llm_name, problem.name, seed)
        if entry is None:
            return True
        return entry.get("end_time") is None

    def __getstate__(self):
        state = self.__dict__.copy()
        # locks can't be pickled, recreate after unpickling
        state["_lock"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._lock = Lock()


class RunLogger:
    """
    Logs an LLM-driven optimization run.
    """

    def __init__(self, name="", root_dir="", budget=100, progress_callback=None):
        """
        Initializes an instance of the RunLogger.
        Sets up a new logging directory named with the current date and time.

        Args:
            name (str): The name of the experiment.
            root_dir (str): The directory to create the log folder in.
            budget (int): The evaluation budget (how many algorithms can be generated and evaluated per run).
        """
        self.dirname = self.create_log_dir(name, root_dir)
        self.attempt = 0
        self.budget = budget
        self._progress_callback = progress_callback

    def get_log_dir(self):
        """
        Returns the directory where the run is logged.
        """
        return self.dirname

    def create_log_dir(self, name="", root_dir=""):
        """
        Creates a new directory for logging runs based on the current date and time.
        Also creates subdirectories for IOH experimenter data and code files.

        Args:
            name (str): The name of the run.
            root_dir (str): The directory to create the log folder in.

        Returns:
            str: The name of the created directory.
        """
        dirname = f"run-{name}"
        dirname = os.path.join(root_dir, dirname)
        if not os.path.exists(root_dir):
            os.makedirs(root_dir)

        tempi = 0
        while os.path.exists(dirname):
            tempi += 1
            dirname = f"run-{name}-{tempi}"
            dirname = os.path.join(root_dir, dirname)
        os.makedirs(dirname)
        return dirname

    def budget_exhausted(self):
        """
        Get the number of lines in the log file and return True if the number of lines matches or exceeded the budget.
        """
        count = 0
        if not os.path.isfile(f"{self.dirname}/log.jsonl"):
            return False  # there is no log file yet
        with open(f"{self.dirname}/log.jsonl", "r") as f:
            for _ in f:
                count += 1

        return count >= self.budget

    def log_conversation(self, role, content, cost=0.0, tokens=0):
        """
        Logs the given conversation content into a conversation log file.

        Args:
            role (str): Who (the llm or user) said the content.
            content (str): The conversation content to be logged.
            cost (float, optional): The cost of the conversation.
            tokens (int, optional): Number of tokens used.
        """
        conversation_object = {
            "role": role,
            "time": f"{datetime.now()}",
            "content": content,
            "cost": float(cost),
            "tokens": int(tokens),
        }
        with jsonlines.open(f"{self.dirname}/conversationlog.jsonl", "a") as file:
            file.write(conversation_object)

    def log_population(self, population):
        """
        Logs the given population to code, configspace and the general log file.

        Args:
            population (list): List of individual solutions
        """
        for p in population:
            self.log_individual(p)

    def log_individual(self, individual):
        """
        Logs the given individual in a general logfile.

        Args:
            individual (Solution): potential solution to be logged.
        """
        ind_dict = individual.to_dict()
        with jsonlines.open(f"{self.dirname}/log.jsonl", "a") as file:
            file.write(convert_to_serializable(ind_dict))
        if self._progress_callback:
            if _accepts_argument(self._progress_callback):
                self._progress_callback(individual)
            else:  # callbacks written for older versions take no arguments
                self._progress_callback()

    def log_code(self, individual):
        """
        Logs the provided code into a file, uniquely named based on the attempt number and algorithm name.

        Args:
            individual (Solution): potential solution to be logged.
        """
        # Create code directory if it doesn't exist
        if not os.path.exists(f"{self.dirname}/code"):
            os.makedirs(f"{self.dirname}/code")
        with open(
            f"{self.dirname}/code/{individual.id}-{individual.name}.py", "w"
        ) as file:
            file.write(individual.code)

    def log_configspace(self, individual):
        """
        Logs the provided configuration space (str) into a file, uniquely named based on the attempt number and algorithm name.

        Args:
            individual (Solution): potential solution to be logged.
        """
        os.makedirs(os.path.join(self.dirname, "configspace"), exist_ok=True)
        with open(
            f"{self.dirname}/configspace/{individual.id}-{individual.name}.py", "w"
        ) as file:
            if individual.configspace != None:
                file.write(cs_json.write(individual.configspace))
            else:
                file.write("Failed to extract config space")
        self.attempt += 1

    def log_output(self, stdout: str = "", stderr: str = "", append: bool = False):
        """Log captured standard output and error to files.

        Args:
            stdout (str): Captured text from ``stdout``.
            stderr (str): Captured text from ``stderr``.
            append (bool): If ``True``, append to existing log files instead of
                overwriting them.
        """
        mode = "a" if append else "w"
        if stdout:
            with open(os.path.join(self.dirname, "stdout.log"), mode) as file:
                file.write(stdout)
        if stderr:
            with open(os.path.join(self.dirname, "stderr.log"), mode) as file:
                file.write(stderr)
