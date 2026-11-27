import importlib.util
from typing import Any

from llamea import LLaMEA as LLAMEA_Algorithm
from llamea import get_variant

from ..llm import LLM
from ..method import Method
from ..problem import Problem

# We import the LLaMEA algorithm directly from the pypi package. This has the advantage that we can easily get the latest version.

#: Variants (``llamea.get_variant`` names) that cannot run in BLADE, with the
#: reason. LLaMEA's multi-objective mode expects ``llamea.Fitness`` objects
#: and replaces BLADE's ``iohblade.fitness.Fitness`` by the worst value.
UNSUPPORTED_VARIANTS: dict[str, str] = {
    "mo": (
        "LLaMEA's multi-objective mode does not work with BLADE problems yet: "
        "it only accepts llamea.Fitness, so BLADE's multi-objective scores "
        "would all be replaced by the worst value."
    ),
}


def resolve_variant(variant: str) -> tuple[str, type]:
    """Return the canonical short name and class of a LLaMEA variant.

    Args:
        variant: A name accepted by ``llamea.get_variant``, e.g. ``"bo"``,
            ``"LLaMEA-HPO"`` or ``"llamea"``.

    Returns:
        tuple[str, type]: The short name (``"bo"``) and the class.

    Raises:
        ValueError: For an unknown variant, or one that BLADE cannot run.
    """
    try:
        cls = get_variant(str(variant))
    except KeyError as e:
        raise ValueError(e.args[0] if e.args else str(e)) from None
    key = getattr(cls, "key", "") or "llamea"
    if key in UNSUPPORTED_VARIANTS:
        raise ValueError(UNSUPPORTED_VARIANTS[key])
    return key, cls


class LLaMEA(Method):
    def __init__(
        self, llm: LLM, budget: int, name="LLaMEA", variant: str = "llamea", **kwargs
    ):
        """
        Initializes the LLaMEA algorithm within the benchmarking framework.

        Args:
            llm (LLM): The LLM instance to use for solution generation.
            budget (int): The maximum number of evaluations.
            name (str): The name of the method. With a ``variant`` and the
                default name, the name becomes e.g. ``"LLaMEA-BO"``.
            variant (str): A ready-made LLaMEA configuration from
                ``llamea.variants``: ``"llamea"`` (default), ``"hpo"``,
                ``"bo"`` or ``"sage"``. ``kwargs`` override the preset;
                BLADE always sets the prompts (from the problem), the budget
                and the optimisation direction.
            kwargs: Additional arguments for configuring LLaMEA.

        Raises:
            ValueError: For an unknown variant, or one BLADE cannot run
                (``"mo"``), or for ``multi_objective=True``.
            ImportError: When feature-guided mutation is requested (directly
                or by ``variant="sage"``) and xgboost is not installed.
        """
        self.variant, self._variant_cls = resolve_variant(variant)
        if self.variant != "llamea" and name == "LLaMEA":
            name = f"LLaMEA-{self.variant.upper()}"
        super().__init__(llm, budget, name)
        if kwargs.get("multi_objective"):
            raise ValueError(UNSUPPORTED_VARIANTS["mo"])
        preset = self._variant_cls.default_config() if self.variant != "llamea" else {}
        if kwargs.get(
            "feature_guided_mutation", preset.get("feature_guided_mutation")
        ) and (importlib.util.find_spec("xgboost") is None):
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
        """True when LLaMEA samples configuration spaces (``HPO=True``).

        The ``hpo`` variant sets ``HPO=True`` unless ``HPO`` is overridden.
        """
        if "HPO" in self.kwargs:
            return bool(self.kwargs["HPO"])
        if self.variant == "llamea":
            return False
        return bool(self._variant_cls.default_config().get("HPO"))

    def __call__(self, problem: Problem):
        """
        Executes the evolutionary search process via LLaMEA.

        Returns:
            Solution: The best solution found.
        """
        # Get optimisation direction/ set default as maximisaiton.
        minimization = getattr(problem, "minimisation", False)
        # The plain algorithm through the module-level name, so tests can patch it.
        algorithm = LLAMEA_Algorithm if self.variant == "llamea" else self._variant_cls

        self.llamea_instance = algorithm(
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
            "variant": self.variant,
            "kwargs": self.kwargs,
        }

    def get_config(self) -> dict[str, Any]:
        config = self.kwargs.copy()
        config["budget"] = self.budget
        config["variant"] = self.variant
        return {
            "name": self.name,
            "source": "https://github.com/XAI-liacs/LLaMEA",
            "config": config,
        }
