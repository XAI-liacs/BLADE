from .solution import Solution
from abc import ABC, abstractmethod
import numpy as np
import traceback


class Problem(ABC):
    """
    Abstract problem class.
    """

    def __init__(self, logger=None, training_instances=None, test_instances=None):
        """
        Initializes a problem instance with logging and dataset references.

        Args:
            logger (Logger, optional): Logger object for tracking solutions.
            training_instances (list, optional): List of training problem instances.
            test_instances (list, optional): List of test problem instances.
        """
        self.logger = logger
        self.training_instances = training_instances if training_instances else []
        self.test_instances = test_instances if test_instances else []
        self.task_prompt = "Write the problem description part here."
        self.format_prompt = "Write the format description part here."

    def __call__(self, solution: Solution, logger=None):
        """
        Evaluates a solution on training instances and updates its fitness and feedback.

        Args:
            solution (Solution): Solution object to be evaluated.

        Returns:
            Solution: The evaluated solution with updated fitness and scores.
        """
        try:
            solution = self.evaluate(solution)
        except Exception as e:
            solution.set_scores(-np.Inf, feedback=str(e))

        if self.logger is not None:
            self.logger.log_individual(solution)
        return solution

    def set_logger(self, logger):
        """
        Sets the logger for this problem.
        """
        self.logger = logger

    @abstractmethod
    def get_prompt(self):
        """
        Get the prompt describing the problem and how to format the answer.
        """
        return self.task_prompt + self.format_prompt

    @abstractmethod
    def evaluate(self, solution: Solution):
        """
        Evaluates a solution on training instances and updates its fitness and feedback.

        Args:
            solution (Solution): Solution object to be evaluated.
        """
        pass

    @abstractmethod
    def test(self, solution: Solution):
        """
        Performs a complete evaluation on test instances and returns the fitness score.

        Args:
            solution (Solution): Solution object to be tested.
        """
        pass

    @abstractmethod
    def to_dict(self):
        """
        Returns a dictionary representation of the problem including all parameters.

        Returns:
            dict: Dictionary representation of the problem.
        """
        pass
