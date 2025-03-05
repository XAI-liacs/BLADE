from solution import Solution
from abc import ABC, abstractmethod

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
        self.prompt = "Write the problem description part here."

    @abstractmethod
    def get_prompt(self):
        """
        Get the prompt describing the problem.
        """
        return self.prompt

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