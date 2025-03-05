from abc import ABC, abstractmethod


class Experiment(ABC):
    """
    Abstract class for an entire experiment, running multiple algorithms on multiple problems.
    """
    
    def __init__(self, methods: list, problems: list):
        """
        Initializes an experiment with multiple methods and problems.

        Args:
            methods (list): List of method instances.
            problems (list): List of problem instances.
        """
        self.methods = methods
        self.problems = problems
        self.log_dirs = []

    @abstractmethod
    def run(self):
        """
        Runs the experiment by executing each method on each problem.
        """
        pass



class BBOB_Experiment(Experiment):

    def run(self):
        pass