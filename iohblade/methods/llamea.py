from ..problem import Problem
from ..llm import LLM
from ..method import Method

from llamea import LLaMEA as LLAMEA_Algorithm

# We import the LLaMEA algorithm directly from the pypi package. This has the advantage that we can easily get the latest version.


class LLaMEA(Method):
    def __init__(self, llm: LLM, budget: int, name="LLaMEA", **kwargs):
        """
        Initializes the LLaMEA algorithm within the benchmarking framework.

        Args:
            problem (Problem): The problem instance to optimize.
            llm (LLM): The LLM instance to use for solution generation.
            budget (int): The maximum number of evaluations.
            name (str): The name of the method.
            kwargs: Additional arguments for configuring LLaMEA.
        """
        super().__init__(llm, budget, name)
        self.kwargs = kwargs

    def __call__(self, problem: Problem):
        """
        Executes the evolutionary search process via LLaMEA.

        Returns:
            Solution: The best solution found.
        """
        self.llamea_instance = LLAMEA_Algorithm(
            f=problem,  # Ensure evaluation integrates with our framework
            llm=self.llm,
            task_prompt=problem.get_prompt(),
            log=None,  # We do not use the LLaMEA native logger, we use the experiment logger instead which is attached on problem level.
            budget=self.budget,
            **self.kwargs,
        )
        return self.llamea_instance.run()

    def to_dict(self):
        """
        Returns a dictionary representation of the method including all parameters.

        Returns:
            dict: Dictionary representation of the method.
        """
        return {
            "method_name": self.name if self.name != None else "LLaMEA",
            "budget": self.budget,
            "kwargs": self.kwargs,
        }
