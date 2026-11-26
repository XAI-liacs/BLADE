import importlib.util
from typing import Any

from llamea import LLaMEA as LLAMEA_Algorithm

from ..llm import LLM
from ..method import Method
from ..problem import Problem

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
        if kwargs.get("feature_guided_mutation") and (
            importlib.util.find_spec("xgboost") is None
        ):
            # LLaMEA raises this only when a run starts; raising it here
            # lets `iohblade validate` report it before anything runs.
            raise ImportError(
                "feature_guided_mutation needs xgboost, which is optional since "
                "LLaMEA 1.5.1. Install it with `uv sync --group feature-guidance` "
                "or `pip install 'llamea[feature-guidance]'`."
            )
        self.kwargs = kwargs

    @property
    def uses_hpo(self) -> bool:
        """True when LLaMEA samples configuration spaces (``HPO=True``)."""
        return bool(self.kwargs.get("HPO"))

    def __call__(self, problem: Problem):
        """
        Executes the evolutionary search process via LLaMEA.

        Returns:
            Solution: The best solution found.
        """
        # Get optimisation direction/ set default as maximisaiton.
        minimization = getattr(problem, "minimisation", False)

        self.llamea_instance = LLAMEA_Algorithm(
            f=problem,  # Ensure evaluation integrates with our framework
            llm=self.llm,
            role_prompt="You are an excellent Python programmer.",  # not needed, it is part of the task prompt.
            task_prompt=problem.task_prompt,
            example_prompt=problem.example_prompt,
            output_format_prompt=problem.format_prompt,
            log=None,  # We do not use the LLaMEA native logger, we use the experiment logger instead which is attached on problem level.
            budget=self.budget,
            max_workers=1,  # We do not use parallelization, as it is not supported in combination with the BLADE parrallelization.
            minimization=minimization,
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

    def get_config(self) -> dict[str, Any]:
        config = self.kwargs.copy()
        config["budget"] = self.budget
        return {
            "name": self.name,
            "source": "https://github.com/XAI-liacs/LLaMEA",
            "config": config,
        }
