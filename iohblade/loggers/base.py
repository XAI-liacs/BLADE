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
        os.mkdir(dirname)
        return dirname

    def open_run(self, method, problem, budget=100, seed=0):
        """
        Opens (starts) a new run for logging.
        Typically call this right before your run, so that the RunLogger can log step data.
        """
        run_name = f"{method.name}-{problem.name}-{seed}"

        progress_cb = partial(self.increment_eval, method.name, problem.name, seed)

        self.run_logger = RunLogger(
            name=run_name,
            root_dir=self.dirname,
            budget=budget,
            progress_callback=progress_cb,
        )
        problem.set_logger(self.run_logger)
        with self._lock:
            entry = self._get_run_entry(method.name, problem.name, seed)
            if entry is None:
                entry = {
                    "method_name": method.name,
                    "problem_name": problem.name,
                    "seed": seed,
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
            entry["log_dir"] = os.path.relpath(self.run_logger.dirname, self.dirname)
            entry["end_time"] = None
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
        rel_log_dir = os.path.relpath(log_dir, self.dirname)
        run_object = {
            "method_name": method.name,
            "problem_name": problem.name,
            "llm_name": llm.model,
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
            entry = self._get_run_entry(method.name, problem.name, seed)
            if entry is not None:
                entry["end_time"] = datetime.now().isoformat()
                entry["log_dir"] = rel_log_dir
            self._write_progress()
        self.increment_progress()

    def get_data(self):
        """
        Retrieves the data from the experiment log and returns a pandas dataframe.

        Returns:
            dataframe: Pandas DataFrame of the experimentlog.
        """
        frames = []
        for d in self.dirs:
            path = os.path.join(d, "experimentlog.jsonl")
            if os.path.exists(path):
                frames.append(pd.read_json(path, lines=True))
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    def get_problem_data(self, problem_name):
        """
        Retrieves the data for a specific method and problem from the experiment log.

        Args:
            problem_name (str): The name of the problem.

        Returns:
            list: List of run data for the specified method and problem.
        """
        bigdf = pd.DataFrame()
        for d in self.dirs:
            exp_log = os.path.join(d, "experimentlog.jsonl")
            if not os.path.exists(exp_log):
                continue
            with jsonlines.open(exp_log) as file:
                for line in file:
                    if line["problem_name"] != problem_name:
                        continue
                    logdir = os.path.join(
                        d, line["log_dir"]
                    )  # relative to *that* experiment
                    run_log = os.path.join(logdir, "log.jsonl")
                    if os.path.exists(run_log):
                        df = pd.read_json(run_log, lines=True)
                        df["method_name"] = line["method_name"]
                        df["problem_name"] = line["problem_name"]
                        df["seed"] = line["seed"]
                        df["_id"] = df.index
                        bigdf = pd.concat([bigdf, df], ignore_index=True)
        return bigdf

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
            with jsonlines.open(exp_log) as file:
                for line in file:
                    methods.add(line["method_name"])
                    problems.add(line["problem_name"])
        return list(methods), list(problems)

    # Progress helpers -------------------------------------------------

    def _get_run_entry(self, method_name, problem_name, seed):
        """Return the run progress entry matching the identifiers."""
        for r in self.progress.get("runs", []):
            if (
                r.get("method_name") == method_name
                and r.get("problem_name") == problem_name
                and r.get("seed") == seed
            ):
                return r
        return None

    def _progress_path(self):
        return os.path.join(self.dirname, "progress.json")

    def _write_progress(self):
        with open(self._progress_path(), "w") as f:
            json.dump(self.progress, f)

    def _load_progress(self):
        path = self._progress_path()
        if os.path.exists(path):
            with open(path) as f:
                self.progress = json.load(f)
        else:
            self.progress = {}

    def start_progress(
        self, total_runs: int, methods=None, problems=None, seeds=None, budget=None
    ):
        """Initialize progress tracking with experiment configuration."""
        with self._lock:
            if os.path.exists(self._progress_path()):
                self._load_progress()
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
                        "start_time": datetime.now().isoformat(),
                        "end_time": None,
                        "current": 0,
                        "total": int(total_runs),
                        "runs": [],
                    }
                    for m in methods:
                        for p in problems:
                            for s in seeds:
                                self.progress["runs"].append(
                                    {
                                        "method_name": m.name,
                                        "problem_name": p.name,
                                        "seed": int(s),
                                        "budget": int(budget),
                                        "evaluations": 0,
                                        "start_time": None,
                                        "end_time": None,
                                        "log_dir": None,
                                    }
                                )
            else:
                self.progress = {
                    "start_time": datetime.now().isoformat(),
                    "end_time": None,
                    "current": 0,
                    "total": int(total_runs),
                    "runs": [],
                }
                for m in methods:
                    for p in problems:
                        for s in seeds:
                            self.progress["runs"].append(
                                {
                                    "method_name": m.name,
                                    "problem_name": p.name,
                                    "seed": int(s),
                                    "budget": int(budget),
                                    "evaluations": 0,
                                    "start_time": None,
                                    "end_time": None,
                                    "log_dir": None,
                                }
                            )
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

    def increment_eval(self, method_name, problem_name, seed):
        with self._lock:
            entry = self._get_run_entry(method_name, problem_name, seed)
            if entry is not None:
                entry["evaluations"] = entry.get("evaluations", 0) + 1
                self._write_progress()

    def is_run_pending(self, method, problem, seed):
        entry = self._get_run_entry(method.name, problem.name, seed)
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
        model_name = name.split("/")[-1]
        dirname = f"run-{name}"
        dirname = os.path.join(root_dir, dirname)
        if not os.path.exists(root_dir):
            os.mkdir(root_dir)

        tempi = 0
        while os.path.exists(dirname):
            tempi += 1
            dirname = f"run-{name}-{tempi}"
            dirname = os.path.join(root_dir, dirname)
        os.mkdir(dirname)
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
        with open(
            f"{self.dirname}/configspace/{individual.id}-{individual.name}.py", "w"
        ) as file:
            if individual.configspace != None:
                file.write(cs_json.write(individual.configspace))
            else:
                file.write("Failed to extract config space")
        self.attempt += 1
