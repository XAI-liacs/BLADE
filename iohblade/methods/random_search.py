from ..llm import LLM
from ..method import Method
from ..problem import Problem


class RandomSearch(Method):
    def __init__(self, llm: LLM, budget, name="RandomSearch", **kwargs):
        """
        Initializes the LLaMEA algorithm within the benchmarking framework.

        Args:
            llm (LLM): The LLM instance to use for solution generation.
            budget (int): The maximum number of evaluations.
            kwargs: Additional arguments for configuring LLaMEA.
        """
        super().__init__(llm, budget, name)

    def __call__(self, problem: Problem):
        """
        Executes the evolutionary search process via LLaMEA.

        Returns:
            Solution: The best solution found.
        """
        best_solution = None
        for i in range(self.budget):
            solution = self.llm.sample_solution(
                [{"role": "client", "content": problem.get_prompt()}]
            )
            solution = problem(solution)
            if best_solution is None or solution.fitness > best_solution.fitness:
                best_solution = solution
        return best_solution

    def to_dict(self):
        """
        Returns a dictionary representation of the method including all parameters.

        Returns:
            dict: Dictionary representation of the method.
        """
        return {
            "method_name": self.name if self.name != None else "RandomSearch",
            "budget": self.budget,
        }
