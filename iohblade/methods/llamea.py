import importlib.util
import re
import textwrap
from typing import Any, Iterable

from llamea import LLaMEA as LLAMEA_Algorithm
from llamea import get_variant

from ..llm import LLM
from ..method import Method
from ..problem import Problem

# We import the LLaMEA algorithm directly from the pypi package. This has the advantage that we can easily get the latest version.

#: Variants (``llamea.get_variant`` names) that cannot run in BLADE, with the
#: reason. Empty since llamea 1.6.1, whose multi-objective mode accepts
#: BLADE's ``iohblade.fitness.Fitness``.
UNSUPPORTED_VARIANTS: dict[str, str] = {}


#: Packages that LLaMEA-BO's algorithms need (Gaussian processes, optimisers
#: of the acquisition function); added to the evaluation environment of every
#: problem LLaMEA-BO runs on, so all methods compared there can use them.
BO_DEPENDENCIES: list[str] = ["scipy>=1.11", "scikit-learn>=1.4"]

#: Appended to the problem's task prompt for ``variant="bo"``. It carries the
#: part of LLaMEA-BO's own prompt that makes it design BO algorithms; the
#: problem's prompt keeps defining the interface, bounds and budget.
BO_PROMPT = textwrap.dedent("""
    This is LLaMEA-BO: the algorithm must be a Bayesian optimization (BO)
    algorithm for expensive black-box optimization. It should contain the core
    BO components: an initial design, a surrogate model (e.g. a Gaussian
    process) and an acquisition function (with its optimization). Keep the
    interface described above. You may use numpy, scipy and scikit-learn.
    """)


def _tag_values(problem) -> set[str]:
    try:
        tags = problem.get_config().get("tags", [])
    except Exception:
        return set()
    return {str(getattr(tag, "value", tag)) for tag in tags}


def bo_unsupported_reason(problem) -> str | None:
    """Why LLaMEA-BO cannot run on ``problem``, or None if it can.

    LLaMEA-BO designs Bayesian optimisation algorithms, so it needs a
    black-box optimisation problem: one tagged ``PrimaryCategories.BBO`` in
    ``get_config()["tags"]``. The auto-correlation problems (tagged
    ``Benchmark.ANALYSIS``) are excluded: their candidates construct a
    sequence rather than optimise a black-box function.
    """
    from ..tags import Benchmark, PrimaryCategories

    tags = _tag_values(problem)
    name = getattr(problem, "name", type(problem).__name__)
    if PrimaryCategories.BBO.value not in tags:
        return (
            f"variant 'bo' designs Bayesian optimisation algorithms and only "
            f"runs on black-box optimisation problems (tagged "
            f"PrimaryCategories.BBO); problem '{name}' is not one."
        )
    if Benchmark.ANALYSIS.value in tags:
        return (
            f"variant 'bo' designs Bayesian optimisation algorithms; problem "
            f"'{name}' asks for a constructed sequence, not for an optimiser "
            "of a black-box function."
        )
    return None


def objective_keys(problem) -> list[str] | None:
    """The objective names a multi-objective problem declares, or None.

    Read from ``problem.get_config()["config"]["keys"]`` (e.g. ``["f1",
    "f2"]`` for ``toy_multiobjective``).
    """
    try:
        keys = problem.get_config().get("config", {}).get("keys")
    except Exception:
        return None
    if (
        isinstance(keys, (list, tuple))
        and keys
        and all(isinstance(k, str) for k in keys)
    ):
        return list(keys)
    return None


def mo_unsupported_reason(problem, keys: list[str] | None = None) -> str | None:
    """Why LLaMEA's multi-objective mode cannot run on ``problem``, or None.

    It needs a problem tagged ``ObjectiveType.MULTI_OBJECTIVE`` whose
    objective names are known: given as ``keys`` (``multi_objective_keys``)
    or declared by the problem (see :func:`objective_keys`).
    """
    from ..tags import ObjectiveType

    name = getattr(problem, "name", type(problem).__name__)
    if ObjectiveType.MULTI_OBJECTIVE.value not in _tag_values(problem):
        return (
            f"LLaMEA's multi-objective mode (variant 'mo' or multi_objective: "
            f"true) needs a multi-objective problem (tagged "
            f"ObjectiveType.MULTI_OBJECTIVE); problem '{name}' is not one."
        )
    if not keys and not objective_keys(problem):
        return (
            f"Problem '{name}' does not declare its objective names "
            "(get_config()['config']['keys']); give them as "
            "multi_objective_keys."
        )
    return None


def check_multi_objective(methods: Iterable, problems: Iterable) -> None:
    """Check that multi-objective LLaMEA methods run on suitable problems.

    Raises:
        ValueError: If a multi-objective LLaMEA method is combined with a
            problem that is not multi-objective or has unknown objectives.
    """
    problems = list(problems)
    for method in methods:
        if not getattr(method, "is_multi_objective", False):
            continue
        keys = method.kwargs.get("multi_objective_keys")
        reasons = [mo_unsupported_reason(p, keys) for p in problems]
        reasons = [r for r in reasons if r is not None]
        if reasons:
            raise ValueError(f"{method.name}: " + " ".join(reasons))


def configure_llamea(methods: Iterable, problems: Iterable) -> list[str]:
    """Check and prepare the problems of LLaMEA methods (variants).

    Multi-objective methods need multi-objective problems
    (:func:`check_multi_objective`); LLaMEA-BO needs black-box optimisation
    problems and adds packages to them (:func:`configure_bo`).

    Returns:
        list[str]: Notes.

    Raises:
        ValueError: For an unsuitable problem.
    """
    methods, problems = list(methods), list(problems)
    check_multi_objective(methods, problems)
    return configure_bo(methods, problems)


def _package(dep: str) -> str:
    return re.split(r"[<>=!~\[;@ ]", dep, maxsplit=1)[0].strip().lower()


def configure_bo(methods: Iterable, problems: Iterable) -> list[str]:
    """Prepare the problems that LLaMEA-BO methods run on.

    Adds :data:`BO_DEPENDENCIES` to their evaluation environments (like HPO
    adds SMAC). A problem shared by several methods gets one environment,
    so every method compared on it can use these packages. Idempotent.

    Args:
        methods: The experiment's methods.
        problems: The experiment's problems.

    Returns:
        list[str]: Notes (packages added to a problem).

    Raises:
        ValueError: If a LLaMEA-BO method is combined with a problem that is
            not a black-box optimisation problem.
    """
    problems = list(problems)
    bo_methods = [m for m in methods if getattr(m, "variant", None) == "bo"]
    if not bo_methods:
        return []
    names = ", ".join(sorted({str(getattr(m, "name", m)) for m in bo_methods}))
    reasons = [r for r in map(bo_unsupported_reason, problems) if r is not None]
    if reasons:
        raise ValueError(f"{names}: " + " ".join(reasons))
    notes = []
    for problem in problems:
        deps = list(getattr(problem, "dependencies", None) or [])
        present = {_package(d) for d in deps}
        new = [d for d in BO_DEPENDENCIES if _package(d) not in present]
        if new:
            if getattr(problem, "_env_path", None) is not None:
                problem.cleanup()  # a private environment is deleted
                problem._env_path = None
                problem._python_bin = None
            problem.dependencies = deps + new
            added = list(getattr(problem, "_bo_added_dependencies", []))
            problem._bo_added_dependencies = added + new
        if getattr(problem, "_bo_added_dependencies", None):
            notes.append(
                f"{names} (variant 'bo') runs on problem '{problem.name}'; its "
                "evaluation environment also installs "
                + ", ".join(problem._bo_added_dependencies)
                + "."
            )
    return notes


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


def _representative(front: list, minimisation: bool):
    """The Pareto-front member with the best mean objective (None if empty).

    BLADE's harness (loggers, ``iohblade check``, test evaluation) expects
    one solution per run; all members are equally good in the Pareto sense,
    so the mean objective (``float(fitness)``) is used as a tie-breaker.
    """
    import math

    def mean(solution):
        try:
            value = float(solution.fitness)
        except (TypeError, ValueError):
            return math.nan
        return value

    valid = [s for s in front if math.isfinite(mean(s))]
    if not valid:
        return front[0] if front else None
    pick = min if minimisation else max
    return pick(valid, key=mean)


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
                and the optimisation direction. ``"bo"`` only runs on
                black-box optimisation problems (see
                :func:`bo_unsupported_reason`); :data:`BO_PROMPT` is appended
                to their task prompt and :data:`BO_DEPENDENCIES` are added to
                their evaluation environment. ``"mo"`` (like
                ``multi_objective=True``) only runs on multi-objective
                problems (see :func:`mo_unsupported_reason`); the objective
                names come from ``multi_objective_keys`` or the problem.
            kwargs: Additional arguments for configuring LLaMEA.

        Raises:
            ValueError: For an unknown variant.
            ImportError: When feature-guided mutation is requested (directly
                or by ``variant="sage"``) and xgboost is not installed.
        """
        self.variant, self._variant_cls = resolve_variant(variant)
        if self.variant != "llamea" and name == "LLaMEA":
            name = f"LLaMEA-{self.variant.upper()}"
        super().__init__(llm, budget, name)
        #: The Pareto front of the last multi-objective run (else None).
        self.pareto_front = None
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

    @property
    def is_multi_objective(self) -> bool:
        """True for ``variant="mo"`` or ``multi_objective=True``."""
        if "multi_objective" in self.kwargs:
            return bool(self.kwargs["multi_objective"])
        return self.variant == "mo"

    def __call__(self, problem: Problem):
        """
        Executes the evolutionary search process via LLaMEA.

        Returns:
            Solution: The best solution found. In multi-objective mode the
            whole Pareto front is kept in :attr:`pareto_front` and the front
            member with the best mean objective is returned.
        """
        # Get optimisation direction/ set default as maximisaiton.
        minimization = getattr(problem, "minimisation", False)
        task_prompt = problem.task_prompt
        kwargs = dict(self.kwargs)
        if self.is_multi_objective:
            reason = mo_unsupported_reason(problem, kwargs.get("multi_objective_keys"))
            if reason is not None:
                raise ValueError(reason)
            kwargs.setdefault("multi_objective", True)
            kwargs.setdefault("multi_objective_keys", objective_keys(problem))
        if self.variant == "bo":
            reason = bo_unsupported_reason(problem)
            if reason is not None:
                raise ValueError(reason)
            configure_bo([self], [problem])  # no-op if already prepared
            task_prompt = task_prompt.rstrip() + "\n" + BO_PROMPT
        # The plain algorithm through the module-level name, so tests can patch it.
        algorithm = LLAMEA_Algorithm if self.variant == "llamea" else self._variant_cls

        self.llamea_instance = algorithm(
            f=problem,  # Ensure evaluation integrates with our framework
            llm=self.llm,
            role_prompt="You are an excellent Python programmer.",  # not needed, it is part of the task prompt.
            task_prompt=task_prompt,
            example_prompt=problem.example_prompt,
            output_format_prompt=problem.format_prompt,
            log=None,  # We do not use the LLaMEA native logger, we use the experiment logger instead which is attached on problem level.
            budget=self.budget,
            max_workers=1,  # We do not use parallelization, as it is not supported in combination with the BLADE parrallelization.
            minimization=minimization,
            **kwargs,
        )

        result = self.llamea_instance.run()
        if not self.is_multi_objective:
            return result
        self.pareto_front = list(result or [])
        return _representative(self.pareto_front, minimization)

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
            "multi_objective": self.is_multi_objective,
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
