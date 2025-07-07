import os
import json
from datetime import datetime

import jsonlines
import numpy as np
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
        """
        Initializes an instance of the ExperimentLogger.
        Sets up a new logging directory named with the current date and time.

        Args:
            name (str): The name of the experiment.
            read (bool): Whether to read the experiment log or create a new one.
        """
        if read:
            self.dirname = name
        else:
            self.dirname = self.create_log_dir(name)
            self.progress = {
                "start_time": datetime.now().isoformat(),
                "end_time": None,
                "current": 0,
                "total": 0,
            }
            self._write_progress()
        if read:
            self._load_progress()

    def create_log_dir(self, name=""):
        """
        Creates a new directory for logging experiments based on the current date and time.

        Returns:
            str: The name of the created directory.
        """
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

        self.run_logger = RunLogger(
            name=run_name,
            root_dir=self.dirname,
            budget=budget,
        )
        problem.set_logger(self.run_logger)
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
        self.increment_progress()

    def get_data(self):
        """
        Retrieves the data from the experiment log and returns a pandas dataframe.

        Returns:
            dataframe: Pandas DataFrame of the experimentlog.
        """
        df = pd.read_json(f"{self.dirname}/experimentlog.jsonl", lines=True)
        return df

    def get_problem_data(self, problem_name):
        """
        Retrieves the data for a specific method and problem from the experiment log.

        Args:
            problem_name (str): The name of the problem.

        Returns:
            list: List of run data for the specified method and problem.
        """
        logdirs = []
        bigdf = pd.DataFrame()
        with jsonlines.open(f"{self.dirname}/experimentlog.jsonl") as file:
            for line in file:
                if line["problem_name"] == problem_name:
                    logdir = os.path.join(self.dirname, line["log_dir"])
                    # now process the logdirs into one combined PandasDataframe
                    if os.path.exists(f"{logdir}/log.jsonl"):
                        df = pd.read_json(f"{logdir}/log.jsonl", lines=True)
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
        methods = []
        problems = []
        with jsonlines.open(f"{self.dirname}/experimentlog.jsonl") as file:
            for line in file:
                if line["method_name"] not in methods:
                    methods.append(line["method_name"])
                if line["problem_name"] not in problems:
                    problems.append(line["problem_name"])
        return methods, problems

    # Progress helpers -------------------------------------------------

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

    def start_progress(self, total_runs: int):
        """Initialize progress tracking with the total number of runs."""
        self.progress = {
            "start_time": datetime.now().isoformat(),
            "end_time": None,
            "current": 0,
            "total": int(total_runs),
        }
        self._write_progress()

    def increment_progress(self):
        """Increment the finished run counter and write to file."""
        if not hasattr(self, "progress"):
            self.start_progress(0)
        self.progress["current"] = self.progress.get("current", 0) + 1
        total = self.progress.get("total", 0)
        if (
            total
            and self.progress["current"] >= total
            and self.progress.get("end_time") is None
        ):
            self.progress["end_time"] = datetime.now().isoformat()
        self._write_progress()


class RunLogger:
    """
    Logs an LLM-driven optimization run.
    """

    def __init__(self, name="", root_dir="", budget=100):
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
        os.mkdir(os.path.join(dirname, "code"))
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

    def log_conversation(self, role, content, cost=0.0):
        """
        Logs the given conversation content into a conversation log file.

        Args:
            role (str): Who (the llm or user) said the content.
            content (str): The conversation content to be logged.
            cost (float, optional): The cost of the conversation.
        """
        conversation_object = {
            "role": role,
            "time": f"{datetime.now()}",
            "content": content,
            "cost": float(cost),
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

    def log_code(self, individual):
        """
        Logs the provided code into a file, uniquely named based on the attempt number and algorithm name.

        Args:
            individual (Solution): potential solution to be logged.
        """
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
        self.attempt = attempt
