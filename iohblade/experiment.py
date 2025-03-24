from abc import ABC, abstractmethod
from .problems import MA_BBOB
from .loggers import ExperimentLogger, RunLogger
from .llm import LLM
from .method import Method
import numpy as np
from tqdm import tqdm
import contextlib


class Experiment(ABC):
    """
    Abstract class for an entire experiment, running multiple algorithms on multiple problems.
    """

    def __init__(
        self,
        methods: list,
        problems: list,
        llm: LLM,
        runs=5,
        budget=100,
        seeds=None,
        show_stdout=False,
        log_dir="results/experiment",
    ):
        """
        Initializes an experiment with multiple methods and problems.

        Args:
            methods (list): List of method instances.
            problems (list): List of problem instances.
            llm (LLM): LLM instance to use.
            runs (int): Number of runs for each method.
            budget (int): Number of evaluations per run for each method.
            seeds (list, optional): The exact seeds to use for the runs, len(seeds) overwrites the number of runs if set.
            show_stdout (bool): Whether to show stdout and stderr (standard output) or not.
            log_dir (str): The folder location to store the logs.
        """
        self.methods = methods
        self.problems = problems
        self.runs = runs
        self.budget = budget
        if seeds is None:
            self.seeds = np.arange(runs)
        else:
            self.seeds = seeds
            self.runs = len(seeds)
        self.llm = llm
        self.show_stdout = show_stdout
        self.exp_logger = ExperimentLogger(log_dir)

    def __call__(self):
        """
        Runs the experiment by executing each method on each problem.
        """
        for problem in tqdm(self.problems, desc="Problems"):
            for method in tqdm(self.methods, leave=False, desc="Methods"):
                for i in tqdm(self.seeds, leave=False, desc="Runs"):
                    np.random.seed(i)

                    logger = RunLogger(
                        name=f"{method.name}-{problem.name}-{i}",
                        root_dir=self.exp_logger.dirname,
                        budget=self.budget,
                    )
                    problem.set_logger(logger)
                    self.llm.set_logger(logger)
                    if self.show_stdout:
                        solution = method(problem)
                    else:
                        with contextlib.redirect_stdout(None):
                            with contextlib.redirect_stderr(None):
                                solution = method(problem)
                    self.exp_logger.add_run(
                        method,
                        problem,
                        self.llm,
                        solution,
                        log_dir=logger.dirname,
                        seed=i,
                    )
        return


class MA_BBOB_Experiment(Experiment):
    def __init__(
        self,
        methods: list,
        llm: LLM,
        show_stdout=False,
        runs=5,
        budget=100,
        seeds=None,
        dims=[2, 5],
        budget_factor=2000,
        log_dir="results/MA_BBOB",
        **kwargs,
    ):
        """
        Initializes an experiment on MA-BBOB.

        Args:
            methods (list): List of method instances.
            llm (LLM): LLM instance to use.
            show_stdout (bool): Whether to show stdout and stderr (standard output) or not.
            runs (int): Number of runs for each method.
            budget (int): Number of algorithm evaluations per run per method.
            seeds (list, optional): Seeds for each run.
            dims (list): List of problem dimensions.
            budget_factor (int): Budget factor for the problems.
            **kwargs: Additional keyword arguments for the MA_BBOB problem.
            log_dir (str): The folder location to store the logs.
        """
        super().__init__(
            methods,
            [MA_BBOB(dims=dims, budget_factor=budget_factor, name="MA_BBOB", **kwargs)],
            llm=llm,
            runs=runs,
            budget=budget,
            seeds=seeds,
            show_stdout=show_stdout,
            log_dir=log_dir,
        )
