from abc import ABC, abstractmethod
from .problems import MA_BOB
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

    def __init__(self, methods: list, problems: list, llm: LLM, runs=1):
        """
        Initializes an experiment with multiple methods and problems.

        Args:
            methods (list): List of method instances.
            problems (list): List of problem instances.
        """
        self.methods = methods
        self.problems = problems
        self.runs = runs
        self.llm = llm

    @abstractmethod
    def __call__(self):
        """
        Runs the experiment by executing each method on each problem.
        """
        pass


class MA_BBOB_Experiment(Experiment):
    def __init__(
        self, methods: list, llm: LLM, show_stdout=False, runs=5, dims=[2, 5], budget_factor=2000, **kwargs
    ):
        """
        Initializes an experiment on MA-BBOB.

        Args:
            methods (list): List of method instances.
            llm (LLM): LLM instance to use.
            show_stdout (bool): Whether to show stdout and stderr (standard output) or not.
            runs (int): Number of runs for each method.
            dims (list): List of problem dimensions.
            budget_factor (int): Budget factor for the problems.
            **kwargs: Additional keyword arguments for the MA_BBOB problem.
        """
        super().__init__(
            methods, [MA_BOB(dims=dims, budget_factor=budget_factor, **kwargs)], llm, runs
        )
        self.exp_logger = ExperimentLogger("MA_BBOB")
        self.show_stdout = show_stdout

    def __call__(self):
        """
        Runs the experiment by executing each method on each problem.
        """
        for problem in tqdm(self.problems, desc="Problems"):
            for method in tqdm(self.methods, leave=False, desc="Methods"):
                for i in tqdm(range(self.runs), leave=False, desc="Runs"):
                    np.random.seed(i)
                    
                    logger = RunLogger(
                        name=f"{method.__class__.__name__}-{problem.__class__.__name__}-{i}",
                        root_dir=self.exp_logger.dirname,
                    )
                    problem.set_logger(logger)
                    self.llm.set_logger(logger)
                    if self.show_stdout:
                        solution = method(problem)
                    else:
                        with contextlib.redirect_stdout(None):
                            with contextlib.redirect_stderr(None):
                                solution = method(problem)
                    self.exp_logger.add_run(method, problem, self.llm, solution, log_dir=logger.dirname, seed=i)
        return
