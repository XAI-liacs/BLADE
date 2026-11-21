"""Registry of the components BLADE ships with (and ones users add).

The registry maps a short, stable key (for example ``"bbob_sbox"`` or
``"llamea"``) to the import path of a class or factory function. Targets are
imported lazily, so looking up a key never pulls in heavy or optional
dependencies of other components, and a missing optional dependency is
reported with the exact install command instead of silently yielding
``None``.

Example:
    >>> from iohblade.registry import BuiltinMethod, registry
    >>> LLaMEA = registry.resolve("method", BuiltinMethod.LLAMEA)

Third-party code can register its own components::

    from iohblade.registry import register_problem

    @register_problem("my_problem", description="My custom benchmark.")
    class MyProblem(Problem):
        ...

The existing import paths (``from iohblade.benchmarks import BBOB_SBOX`` etc.)
are unaffected; the registry is an additional way to look components up.
"""

from __future__ import annotations

import difflib
import importlib
import importlib.util
from dataclasses import dataclass
from enum import StrEnum
from threading import RLock
from typing import Any, Callable

from .utils import ComponentNotFoundError, ComponentUnavailableError


class ComponentKind(StrEnum):
    """The kinds of components an experiment is assembled from."""

    PROBLEM = "problem"
    SUITE = "suite"  # factory returning a list of problems
    METHOD = "method"
    LLM = "llm"
    LOGGER = "logger"


class BuiltinProblem(StrEnum):
    """Keys of the problem classes that ship with BLADE."""

    BBOB_SBOX = "bbob_sbox"
    MA_BBOB = "ma_bbob"
    AUTOML = "automl"
    KERNELTUNER = "kerneltuner"
    PHOTONICS = "photonics"
    TOY_MULTIOBJECTIVE = "toy_multiobjective"
    AUTO_CORRELATION_INEQ1 = "auto_correlation_ineq1"
    AUTO_CORRELATION_INEQ2 = "auto_correlation_ineq2"
    AUTO_CORRELATION_INEQ3 = "auto_correlation_ineq3"
    ERDOS_MIN_OVERLAP = "erdos_min_overlap"
    EUCLIDIAN_STEINER_TREE = "euclidian_steiner_tree"
    GRAPH_COLORING = "graph_coloring"
    UNCERTAINTY_INEQUALITY = "uncertainty_inequality"
    CIRCLE_PACKING = "circle_packing"
    HEXAGON_PACKING = "hexagon_packing"
    RECTANGLE_PACKING = "rectangle_packing"
    UNIT_SQUARE_PACKING = "unit_square_packing"
    HEILBRONN_TRIANGLE = "heilbronn_triangle"
    HEILBRONN_CONVEX_REGION = "heilbronn_convex_region"
    KISSING_NUMBER_11D = "kissing_number_11d"
    MIN_MAX_DISTANCE_RATIO = "min_max_distance_ratio"
    SPHERICAL_CODE = "spherical_code"
    SUM_VS_DIFFERENCE = "sum_vs_difference"
    MATRIX_MULTIPLICATION = "matrix_multiplication"
    TSP = "tsp"
    VRP = "vrp"


class BuiltinSuite(StrEnum):
    """Keys of the factory functions that return a list of problems."""

    AUTO_CORRELATION = "auto_correlation"
    ERDOS_MIN_OVERLAP = "erdos_min_overlap"
    UNCERTAINTY_INEQUALITY = "uncertainty_inequality"
    HEXAGON_PACKING = "hexagon_packing"
    RECTANGLE_PACKING = "rectangle_packing"
    UNIT_SQUARE_PACKING = "unit_square_packing"
    HEILBRONN_TRIANGLE = "heilbronn_triangle"
    HEILBRONN_CONVEX_REGION = "heilbronn_convex_region"
    KISSING_NUMBER_11D = "kissing_number_11d"
    MIN_MAX_DISTANCE_RATIO = "min_max_distance_ratio"
    SUM_VS_DIFFERENCE = "sum_vs_difference"
    MATRIX_MULTIPLICATION = "matrix_multiplication"


class BuiltinMethod(StrEnum):
    """Keys of the LLM-driven search methods that ship with BLADE."""

    LLAMEA = "llamea"
    RANDOM_SEARCH = "random_search"
    EOH = "eoh"
    REEVO = "reevo"
    FUNSEARCH = "funsearch"
    LHNS = "lhns"
    MCTS_AHD = "mcts_ahd"
    MOEH = "moeh"


class LLMProvider(StrEnum):
    """Keys of the LLM providers that ship with BLADE."""

    OPENAI = "openai"
    DEEPSEEK = "deepseek"
    GEMINI = "gemini"
    OLLAMA = "ollama"
    CLAUDE = "claude"
    LMSTUDIO = "lmstudio"
    MLX_LM = "mlx_lm"
    DUMMY = "dummy"
    MULTI = "multi"


class LoggerType(StrEnum):
    """Keys of the experiment loggers that ship with BLADE."""

    FILE = "file"
    MLFLOW = "mlflow"
    TRACKIO = "trackio"


@dataclass(frozen=True)
class ComponentInfo:
    """Description of one registered component.

    Args:
        kind: The kind of component.
        key: The registry key, unique per kind.
        target: Import path ``"package.module:attribute"``, or the object
            itself for components registered with a decorator.
        description: One-line human readable description.
        extra: Name of the optional ``uv`` dependency group that provides
            the component's dependencies, if any.
        requires: Extra pip packages the component needs that are not in
            any dependency group.
        modules: Modules that must be importable for the component to work,
            for components whose module imports fine without them (the
            optional import is guarded and fails later, at construction).
    """

    kind: ComponentKind
    key: str
    target: str | Any
    description: str = ""
    extra: str | None = None
    requires: tuple[str, ...] = ()
    modules: tuple[str, ...] = ()

    @property
    def target_path(self) -> str:
        """Import path of the target, also for registered objects."""
        if isinstance(self.target, str):
            return self.target
        return f"{self.target.__module__}:{self.target.__qualname__}"

    def install_hint(self) -> str:
        """Command that installs the optional dependencies, if any."""
        if self.extra is not None:
            return f"uv sync --group {self.extra}"
        if self.requires:
            return "uv pip install " + " ".join(self.requires)
        return ""

    def load(self) -> Any:
        """Import and return the target class or function.

        Raises:
            ComponentUnavailableError: If importing the target fails, for
                example because an optional dependency is not installed.
        """
        for name in self.modules:
            try:
                found = importlib.util.find_spec(name) is not None
            except (ImportError, ValueError):
                found = False
            if not found:
                cause = ModuleNotFoundError(f"No module named '{name}'")
                raise ComponentUnavailableError(
                    self.kind, self.key, self.target_path, cause, self.install_hint()
                )
        if not isinstance(self.target, str):
            return self.target
        module_name, _, attr = self.target.partition(":")
        try:
            module = importlib.import_module(module_name)
        except Exception as e:
            raise ComponentUnavailableError(
                self.kind, self.key, self.target, e, self.install_hint()
            ) from e
        obj = module
        try:
            for part in attr.split("."):
                obj = getattr(obj, part)
        except AttributeError as e:
            raise ComponentUnavailableError(
                self.kind, self.key, self.target, e, ""
            ) from e
        return obj

    def is_available(self) -> bool:
        """Whether the target can be imported in the current environment."""
        try:
            self.load()
        except ComponentUnavailableError:
            return False
        return True


class Registry:
    """Lookup table from ``(kind, key)`` to :class:`ComponentInfo`."""

    def __init__(self):
        self._components: dict[ComponentKind, dict[str, ComponentInfo]] = {
            kind: {} for kind in ComponentKind
        }
        self._lock = RLock()

    def register(
        self,
        kind: ComponentKind | str,
        key: str,
        target: str | Any,
        *,
        description: str = "",
        extra: str | None = None,
        requires: tuple[str, ...] | list[str] = (),
        modules: tuple[str, ...] | list[str] = (),
        overwrite: bool = False,
    ) -> ComponentInfo:
        """Register a component.

        Args:
            kind: The kind of component.
            key: Registry key, unique per kind.
            target: ``"module:attribute"`` import path or the object itself.
            description: One-line description.
            extra: Optional dependency group providing its dependencies.
            requires: Pip packages it needs that are in no dependency group.
            modules: Modules that must be importable, see ComponentInfo.
            overwrite: Replace an existing registration with the same key.

        Returns:
            ComponentInfo: The registered entry.

        Raises:
            ValueError: If the key is already registered and ``overwrite``
                is False, or if a string target has no ``:``.
        """
        kind = ComponentKind(kind)
        key = str(key)
        if isinstance(target, str) and ":" not in target:
            raise ValueError(
                f"Target '{target}' must have the form 'package.module:attribute'."
            )
        info = ComponentInfo(
            kind, key, target, description, extra, tuple(requires), tuple(modules)
        )
        with self._lock:
            existing = self._components[kind].get(key)
            if existing is not None and not overwrite:
                if existing.target_path == info.target_path:
                    return existing  # re-registration, e.g. module reload
                raise ValueError(
                    f"A {kind} with key '{key}' is already registered "
                    f"({existing.target_path}). Pass overwrite=True to replace it."
                )
            self._components[kind][key] = info
        return info

    def unregister(self, kind: ComponentKind | str, key: str) -> None:
        """Remove a registered component (no error if it does not exist)."""
        with self._lock:
            self._components[ComponentKind(kind)].pop(str(key), None)

    def get(self, kind: ComponentKind | str, key: str) -> ComponentInfo:
        """Return the entry for ``key``.

        Raises:
            ComponentNotFoundError: If the key is unknown. The error lists
                close matches.
        """
        kind = ComponentKind(kind)
        key = str(key)
        with self._lock:
            info = self._components[kind].get(key)
            known = sorted(self._components[kind])
        if info is None:
            suggestions = difflib.get_close_matches(key, known, n=3, cutoff=0.6)
            raise ComponentNotFoundError(kind, key, suggestions, known)
        return info

    def resolve(self, kind: ComponentKind | str, key: str) -> Any:
        """Return the imported class or function registered under ``key``."""
        return self.get(kind, key).load()

    def keys(self, kind: ComponentKind | str) -> list[str]:
        """Sorted keys registered for ``kind``."""
        with self._lock:
            return sorted(self._components[ComponentKind(kind)])

    def entries(self, kind: ComponentKind | str) -> list[ComponentInfo]:
        """Registered entries for ``kind``, sorted by key."""
        with self._lock:
            items = self._components[ComponentKind(kind)]
            return [items[k] for k in sorted(items)]

    def __contains__(self, item: tuple[ComponentKind | str, str]) -> bool:
        kind, key = item
        with self._lock:
            return str(key) in self._components[ComponentKind(kind)]


registry = Registry()


def _decorator(kind: ComponentKind) -> Callable:
    def register_component(
        key: str, *, description: str = "", extra: str | None = None
    ) -> Callable:
        def wrap(obj):
            desc = description or (obj.__doc__ or "").strip().split("\n")[0]
            registry.register(kind, key, obj, description=desc, extra=extra)
            return obj

        return wrap

    register_component.__doc__ = (
        f"Class/function decorator that registers a {kind} under ``key``."
    )
    return register_component


register_problem = _decorator(ComponentKind.PROBLEM)
register_suite = _decorator(ComponentKind.SUITE)
register_method = _decorator(ComponentKind.METHOD)
register_llm = _decorator(ComponentKind.LLM)
register_logger = _decorator(ComponentKind.LOGGER)


_B = "iohblade.benchmarks"
_BUILTINS: list[tuple[ComponentKind, StrEnum, str, str, str | None]] = [
    # problems
    (
        ComponentKind.PROBLEM,
        BuiltinProblem.BBOB_SBOX,
        f"{_B}.BBOB.bbob_sboxcost:BBOB_SBOX",
        "BBOB / SBOX-COST continuous black-box optimisation.",
        None,
    ),
    (
        ComponentKind.PROBLEM,
        BuiltinProblem.MA_BBOB,
        f"{_B}.BBOB.mabbob:MA_BBOB",
        "Many-affine BBOB (MA-BBOB) continuous optimisation.",
        None,
    ),
    (
        ComponentKind.PROBLEM,
        BuiltinProblem.AUTOML,
        f"{_B}.automl.automl:AutoML",
        "Design ML pipelines for a tabular dataset.",
        None,
    ),
    (
        ComponentKind.PROBLEM,
        BuiltinProblem.KERNELTUNER,
        f"{_B}.kerneltuner.kerneltuner:Kerneltuner",
        "GPU kernel tuning optimisers (Kernel Tuner).",
        "kerneltuner",
    ),
    (
        ComponentKind.PROBLEM,
        BuiltinProblem.PHOTONICS,
        f"{_B}.photonics.photonics:Photonics",
        "Photonic structure design problems.",
        "extraproblems",
    ),
    (
        ComponentKind.PROBLEM,
        BuiltinProblem.TOY_MULTIOBJECTIVE,
        f"{_B}.toy_multiobjective.toy_multiobjective:ToyMultiObjective",
        "Small multi-objective toy problem.",
        None,
    ),
    (
        ComponentKind.PROBLEM,
        BuiltinProblem.AUTO_CORRELATION_INEQ1,
        f"{_B}.analysis.auto_correlation_ineq1:AutoCorrIneq1",
        "First auto-correlation inequality.",
        None,
    ),
    (
        ComponentKind.PROBLEM,
        BuiltinProblem.AUTO_CORRELATION_INEQ2,
        f"{_B}.analysis.auto_correlation_ineq2:AutoCorrIneq2",
        "Second auto-correlation inequality.",
        None,
    ),
    (
        ComponentKind.PROBLEM,
        BuiltinProblem.AUTO_CORRELATION_INEQ3,
        f"{_B}.analysis.auto_correlation_ineq3:AutoCorrIneq3",
        "Third auto-correlation inequality.",
        None,
    ),
    (
        ComponentKind.PROBLEM,
        BuiltinProblem.ERDOS_MIN_OVERLAP,
        f"{_B}.combinatorics.erdos_min_overlap:ErdosMinOverlap",
        "Erdős minimum overlap problem.",
        None,
    ),
    (
        ComponentKind.PROBLEM,
        BuiltinProblem.EUCLIDIAN_STEINER_TREE,
        f"{_B}.combinatorics.euclidian_steiner_tree:EuclidianSteinerTree",
        "Euclidean Steiner tree problem.",
        None,
    ),
    (
        ComponentKind.PROBLEM,
        BuiltinProblem.GRAPH_COLORING,
        f"{_B}.combinatorics.graph_coloring:GraphColoring",
        "Graph colouring problem.",
        None,
    ),
    (
        ComponentKind.PROBLEM,
        BuiltinProblem.UNCERTAINTY_INEQUALITY,
        f"{_B}.fourier.uncertainty_inequality:UncertaintyInequality",
        "Fourier uncertainty inequality.",
        None,
    ),
    (
        ComponentKind.PROBLEM,
        BuiltinProblem.CIRCLE_PACKING,
        f"{_B}.packing.circles:CirclePacking",
        "Pack circles in a unit square.",
        None,
    ),
    (
        ComponentKind.PROBLEM,
        BuiltinProblem.HEXAGON_PACKING,
        f"{_B}.packing.hexagon_packing:HexagonPacking",
        "Pack unit hexagons in a minimal hexagon.",
        None,
    ),
    (
        ComponentKind.PROBLEM,
        BuiltinProblem.RECTANGLE_PACKING,
        f"{_B}.packing.rectangle_packing:RectanglePacking",
        "Rectangle packing problem.",
        None,
    ),
    (
        ComponentKind.PROBLEM,
        BuiltinProblem.UNIT_SQUARE_PACKING,
        f"{_B}.packing.unit_square_packing:UnitSquarePacking",
        "Unit square packing problem.",
        None,
    ),
    (
        ComponentKind.PROBLEM,
        BuiltinProblem.HEILBRONN_TRIANGLE,
        f"{_B}.geometry.heilbronn_triangle:HeilbronnTriangle",
        "Heilbronn problem in a triangle.",
        None,
    ),
    (
        ComponentKind.PROBLEM,
        BuiltinProblem.HEILBRONN_CONVEX_REGION,
        f"{_B}.geometry.heilbronn_convex_region:HeilbronnConvexRegion",
        "Heilbronn problem in a convex region.",
        None,
    ),
    (
        ComponentKind.PROBLEM,
        BuiltinProblem.KISSING_NUMBER_11D,
        f"{_B}.geometry.kissing_number_11d:KissingNumber11D",
        "Kissing number in 11 dimensions.",
        None,
    ),
    (
        ComponentKind.PROBLEM,
        BuiltinProblem.MIN_MAX_DISTANCE_RATIO,
        f"{_B}.geometry.min_max_distance_ratio:MinMaxMinDistanceRatio",
        "Minimise the max/min distance ratio of a point set.",
        None,
    ),
    (
        ComponentKind.PROBLEM,
        BuiltinProblem.SPHERICAL_CODE,
        f"{_B}.geometry.spherical_code:SphericalCode",
        "Spherical code problem.",
        None,
    ),
    (
        ComponentKind.PROBLEM,
        BuiltinProblem.SUM_VS_DIFFERENCE,
        f"{_B}.number_theory.sums_vs_differences:SumDifference",
        "Sums versus differences sets.",
        None,
    ),
    (
        ComponentKind.PROBLEM,
        BuiltinProblem.MATRIX_MULTIPLICATION,
        f"{_B}.matrix_multiplication.task_specification:MatMulTensorDecomposition",
        "Low-rank tensor decomposition for matrix multiplication.",
        None,
    ),
    (
        ComponentKind.PROBLEM,
        BuiltinProblem.TSP,
        f"{_B}.logistics.tsp:TravelingSalesmanProblem",
        "Travelling salesman problem.",
        None,
    ),
    (
        ComponentKind.PROBLEM,
        BuiltinProblem.VRP,
        f"{_B}.logistics.vrp:VehicleRoutingProblem",
        "Capacitated vehicle routing problem.",
        None,
    ),
    # suites
    (
        ComponentKind.SUITE,
        BuiltinSuite.AUTO_CORRELATION,
        f"{_B}.analysis.get_analysis_problems:get_analysis_problems",
        "All auto-correlation inequality problems.",
        None,
    ),
    (
        ComponentKind.SUITE,
        BuiltinSuite.ERDOS_MIN_OVERLAP,
        f"{_B}.combinatorics.get_erdos_problem:get_combinatorics_problems",
        "Erdős minimum overlap problems.",
        None,
    ),
    (
        ComponentKind.SUITE,
        BuiltinSuite.UNCERTAINTY_INEQUALITY,
        f"{_B}.fourier.get_fourier_problem:get_fourier_problems",
        "Fourier uncertainty inequality problems.",
        None,
    ),
    (
        ComponentKind.SUITE,
        BuiltinSuite.HEXAGON_PACKING,
        f"{_B}.packing.get_packing_problems:get_hexagon_packing_problems",
        "Hexagon packing problems.",
        None,
    ),
    (
        ComponentKind.SUITE,
        BuiltinSuite.RECTANGLE_PACKING,
        f"{_B}.packing.get_packing_problems:get_rectangle_packing_problems",
        "Rectangle packing problems.",
        None,
    ),
    (
        ComponentKind.SUITE,
        BuiltinSuite.UNIT_SQUARE_PACKING,
        f"{_B}.packing.get_packing_problems:get_square_packing_problems",
        "Unit square packing problems.",
        None,
    ),
    (
        ComponentKind.SUITE,
        BuiltinSuite.HEILBRONN_TRIANGLE,
        f"{_B}.geometry.get_geometry_problems:get_heilbronn_triangle_problems",
        "Heilbronn triangle problems.",
        None,
    ),
    (
        ComponentKind.SUITE,
        BuiltinSuite.HEILBRONN_CONVEX_REGION,
        f"{_B}.geometry.get_geometry_problems:get_heilbronn_convex_region_problems",
        "Heilbronn convex region problems.",
        None,
    ),
    (
        ComponentKind.SUITE,
        BuiltinSuite.KISSING_NUMBER_11D,
        f"{_B}.geometry.get_geometry_problems:get_kissing_number_11D_problems",
        "Kissing number 11D problems.",
        None,
    ),
    (
        ComponentKind.SUITE,
        BuiltinSuite.MIN_MAX_DISTANCE_RATIO,
        f"{_B}.geometry.get_geometry_problems:get_min_max_dist_ratio_problem",
        "Min/max distance ratio problems.",
        None,
    ),
    (
        ComponentKind.SUITE,
        BuiltinSuite.SUM_VS_DIFFERENCE,
        f"{_B}.number_theory.get_num_theory_problems:get_sum_vs_difference_problem",
        "Sums versus differences problems.",
        None,
    ),
    (
        ComponentKind.SUITE,
        BuiltinSuite.MATRIX_MULTIPLICATION,
        f"{_B}.matrix_multiplication.problem_sets:default_problems",
        "AlphaEvolve matrix multiplication suite.",
        None,
    ),
    # methods
    (
        ComponentKind.METHOD,
        BuiltinMethod.LLAMEA,
        "iohblade.methods.llamea:LLaMEA",
        "LLaMEA: LLM-driven evolutionary algorithm design.",
        None,
    ),
    (
        ComponentKind.METHOD,
        BuiltinMethod.RANDOM_SEARCH,
        "iohblade.methods.random_search:RandomSearch",
        "Random sampling of LLM-generated algorithms.",
        None,
    ),
    (
        ComponentKind.METHOD,
        BuiltinMethod.EOH,
        "iohblade.methods.eoh:EoH",
        "Evolution of Heuristics (EoH).",
        "methods",
    ),
    (
        ComponentKind.METHOD,
        BuiltinMethod.REEVO,
        "iohblade.methods.reevo:ReEvo",
        "Reflective Evolution (ReEvo).",
        "methods",
    ),
    (
        ComponentKind.METHOD,
        BuiltinMethod.FUNSEARCH,
        "iohblade.methods.funsearch:funsearch",
        "FunSearch-style island evolution.",
        None,
    ),
    (
        ComponentKind.METHOD,
        BuiltinMethod.LHNS,
        "iohblade.methods.lhns:LHNS_Method",
        "LLM-driven heuristic neighbourhood search (LHNS).",
        None,
    ),
    (
        ComponentKind.METHOD,
        BuiltinMethod.MCTS_AHD,
        "iohblade.methods.mcts_ahd:MCTS_Method",
        "Monte Carlo tree search for heuristic design (MCTS-AHD).",
        None,
    ),
    (
        ComponentKind.METHOD,
        BuiltinMethod.MOEH,
        "iohblade.methods.moeh_method.moeh:MoEH_Method",
        "Multi-objective evolution of heuristics (MoEH).",
        "methods",
    ),
    # LLM providers
    (
        ComponentKind.LLM,
        LLMProvider.OPENAI,
        "iohblade.llm:OpenAI_LLM",
        "OpenAI chat models.",
        None,
    ),
    (
        ComponentKind.LLM,
        LLMProvider.DEEPSEEK,
        "iohblade.llm:DeepSeek_LLM",
        "DeepSeek models (OpenAI-compatible API).",
        None,
    ),
    (
        ComponentKind.LLM,
        LLMProvider.GEMINI,
        "iohblade.llm:Gemini_LLM",
        "Google Gemini models.",
        None,
    ),
    (
        ComponentKind.LLM,
        LLMProvider.OLLAMA,
        "iohblade.llm:Ollama_LLM",
        "Local models served by Ollama.",
        None,
    ),
    (
        ComponentKind.LLM,
        LLMProvider.CLAUDE,
        "iohblade.llm:Claude_LLM",
        "Anthropic Claude models.",
        None,
    ),
    (
        ComponentKind.LLM,
        LLMProvider.LMSTUDIO,
        "iohblade.llm:LMStudio_LLM",
        "Local models served by LM Studio.",
        "apple-silicon",
    ),
    (
        ComponentKind.LLM,
        LLMProvider.MLX_LM,
        "iohblade.llm:MLX_LM_LLM",
        "Local MLX models on Apple silicon.",
        "apple-silicon",
    ),
    (
        ComponentKind.LLM,
        LLMProvider.DUMMY,
        "iohblade.llm:Dummy_LLM",
        "Offline dummy LLM for testing.",
        None,
    ),
    (
        ComponentKind.LLM,
        LLMProvider.MULTI,
        "iohblade.llm:Multi_LLM",
        "Round-robin over several LLMs.",
        None,
    ),
    # loggers
    (
        ComponentKind.LOGGER,
        LoggerType.FILE,
        "iohblade.loggers.base:ExperimentLogger",
        "File-based experiment logger (default).",
        None,
    ),
    (
        ComponentKind.LOGGER,
        LoggerType.MLFLOW,
        "iohblade.loggers.mlflow:MLFlowExperimentLogger",
        "MLflow experiment logger.",
        None,
    ),
    (
        ComponentKind.LOGGER,
        LoggerType.TRACKIO,
        "iohblade.loggers.trackio:TrackioExperimentLogger",
        "Trackio experiment logger.",
        "trackio",
    ),
]

# Packages needed by built-ins that are not part of any dependency group.
_REQUIRES: dict[tuple[ComponentKind, str], tuple[str, ...]] = {
    (ComponentKind.PROBLEM, BuiltinProblem.AUTOML): ("openml",),
}

# Optional modules imported behind a guard: the component's module imports
# without them, so check them explicitly.
_MODULES: dict[tuple[ComponentKind, str], tuple[str, ...]] = {
    (ComponentKind.LLM, LLMProvider.LMSTUDIO): ("lmstudio",),
    (ComponentKind.LLM, LLMProvider.MLX_LM): ("mlx_lm",),
    (ComponentKind.LOGGER, LoggerType.TRACKIO): ("trackio",),
}

for _kind, _key, _target, _description, _extra in _BUILTINS:
    registry.register(
        _kind,
        _key,
        _target,
        description=_description,
        extra=_extra,
        requires=_REQUIRES.get((_kind, _key), ()),
        modules=_MODULES.get((_kind, _key), ()),
    )

__all__ = [
    "BuiltinMethod",
    "BuiltinProblem",
    "BuiltinSuite",
    "ComponentInfo",
    "ComponentKind",
    "LLMProvider",
    "LoggerType",
    "Registry",
    "register_llm",
    "register_logger",
    "register_method",
    "register_problem",
    "register_suite",
    "registry",
]
