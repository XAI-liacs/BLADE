import os
from datetime import datetime

import jsonlines
import numpy as np
from .utils import convert_to_serializable
from .method import Method
from .problem import Problem
from .llm import LLM
from .solution import Solution
from ConfigSpace.read_and_write import json as cs_json
import pandas as pd


class ExperimentLogger:
    """
    Logs an entire experiment of multiple runs.
    """

    def __init__(self, name=""):
        """
        Initializes an instance of the ExperimentLogger.
        Sets up a new logging directory named with the current date and time.

        Args:
            name (str): The name of the experiment.
        """
        self.dirname = self.create_log_dir(name)
        # Todo: add experiment configuration log file (e.g. methods, seeds etc.)

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

    def add_run(self, method: Method, problem: Problem, llm: LLM, solution: Solution, log_dir="", seed=None):
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
        run_object = {
            "method_name": str(method.__class__.__name__) if method != None else "None",
            "problem_name": str(problem.__class__.__name__) if problem != None else "None",
            "llm_name": llm.model,
            "method": method.to_dict(),
            "problem": problem.to_dict(),
            "llm": llm.to_dict(),
            "solution": solution.to_dict(),
            "log_dir": log_dir,
            "seed": seed,
        }
        with jsonlines.open(f"{self.dirname}/experimentlog.jsonl", "a") as file:
            file.write(convert_to_serializable(run_object))

    def get_data(self):
        """
        Retrieves the data from the experiment log and returns a pandas dataframe.

        Returns:
            dataframe: Pandas DataFrame of the experimentlog.
        """
        df = pd.read_json(f"{self.dirname}/experimentlog.jsonl", lines=True)
        return df

    def get_problem_data(self, problem):
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
                    logdir = line["log_dir"]
                    #now process the logdirs into one combined PandasDataframe
                    df = pd.read_json(f"{logdir}/log.jsonl", lines=True)
                    df["method_name"] = method_name
                    df["problem_name"] = problem_name
                    df["seed"] = line["seed"]
                    bigdf = pd.concat([bigdf, df])
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


class RunLogger:
    """
    Logs an LLM-driven optimization run.
    """

    def __init__(self, name="", root_dir=""):
        """
        Initializes an instance of the RunLogger.
        Sets up a new logging directory named with the current date and time.

        Args:
            name (str): The name of the experiment.
            root_dir (str): The directory to create the log folder in.
        """
        self.dirname = self.create_log_dir(name, root_dir)
        self.attempt = 0

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
        os.mkdir(os.path.join(dirname, "configspace"))
        os.mkdir(os.path.join(dirname, "code"))
        return dirname

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

    def set_attempt(self, attempt):
        self.attempt = attempt

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
            individual (Individual): potential solution to be logged.
        """
        self.log_code(self.attempt, individual.name, individual.code)
        if individual.configspace != None:
            self.log_configspace(self.attempt, individual.name, individual.configspace)
        ind_dict = individual.to_dict()
        with jsonlines.open(f"{self.dirname}/log.jsonl", "a") as file:
            file.write(convert_to_serializable(ind_dict))
        self.attempt += 1

    def log_code(self, attempt, algorithm_name, code):
        """
        Logs the provided code into a file, uniquely named based on the attempt number and algorithm name.

        Args:
            attempt (int): The attempt number of the code execution.
            algorithm_name (str): The name of the algorithm used.
            code (str): The source code to be logged.
        """
        with open(
            f"{self.dirname}/code/try-{attempt}-{algorithm_name}.py", "w"
        ) as file:
            file.write(code)
        self.attempt = attempt

    def log_configspace(self, attempt, algorithm_name, config_space):
        """
        Logs the provided configuration space (str) into a file, uniquely named based on the attempt number and algorithm name.

        Args:
            attempt (int): The attempt number of the code execution.
            algorithm_name (str): The name of the algorithm used.
            config_space (ConfigSpace): The Config space to be logged.
        """
        with open(
            f"{self.dirname}/configspace/try-{attempt}-{algorithm_name}.py", "w"
        ) as file:
            if config_space != None:
                file.write(cs_json.write(config_space))
            else:
                file.write("Failed to extract config space")
        self.attempt = attempt
