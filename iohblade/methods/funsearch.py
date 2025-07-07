from ..llm import LLM
from ..method import Method
from ..problem import Problem

# import funsearch


class funsearch(Method):
    def __init__(self, llm: LLM, budget: int, name="funsearch", **kwargs):
        """
        Initializes the funsearch algorithm within the benchmarking framework.

        Args:
            problem (Problem): The problem instance to optimize.
            llm (LLM): The LLM instance to use for solution generation.
            budget (int): The maximum number of evaluations.
            name (str): The name of the method.
            kwargs: Additional arguments for configuring funsearch.
        """
        super().__init__(llm, budget, name)
        self.kwargs = kwargs

    def __call__(self, problem: Problem):
        """
        Executes the evolutionary search process via funsearch.

        Returns:
            Solution: The best solution found.
        """
        raise NotImplementedError("funsearch is not implemented yet")

    def to_dict(self):
        """
        Returns a dictionary representation of the method including all parameters.

        Returns:
            dict: Dictionary representation of the method.
        """
        return {
            "method_name": self.name if self.name != None else "funsearch",
            "budget": self.budget,
            "kwargs": self.kwargs,
        }
