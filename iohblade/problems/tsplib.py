from __future__ import annotations

import math
import re
import textwrap
from pathlib import Path
from typing import Iterable

import numpy as np

from ..problem import Problem
from ..solution import Solution

DATA_DIR = Path(__file__).resolve().parent / "tsplib_instances"

NUMERIC_RE = re.compile(r"[-+]?[0-9]*\.?[0-9]+")

DEFAULT_OPTIMA = {
    "eil51": 426.0,
    "berlin52": 7542.0,
    "st70": 675.0,
    "ch130": 6110.0,
    "rat195": 2323.0,
    "d198": 15780.0,
    "lin318": 42029.0,
    "pcb442": 50778.0,
    "att532": 27686.0,
}


def _load_tsplib_instance(name: str):
    path = DATA_DIR / f"{name}.tsp"
    if not path.exists():
        raise FileNotFoundError(f"Missing TSPLIB instance '{name}' at {path}")

    coords: list[tuple[float, float]] = []
    optimum: float | None = None
    edge_weight_type = "EUC_2D"
    reading_coords = False

    with path.open("r", encoding="utf8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            upper = line.upper()
            if upper.startswith("COMMENT"):
                lower = line.lower()
                if optimum is None and any(token in lower for token in ("opt", "best")):
                    numbers = [float(match.group()) for match in NUMERIC_RE.finditer(line)]
                    if numbers:
                        optimum = numbers[-1]
                continue
            if upper.startswith("EDGE_WEIGHT_TYPE"):
                parts = line.split(":", 1)
                if len(parts) == 2:
                    edge_weight_type = parts[1].strip().upper()
                continue
            if upper.startswith("NODE_COORD_SECTION"):
                reading_coords = True
                continue
            if upper.startswith("EOF"):
                break
            if reading_coords:
                parts = line.split()
                if len(parts) < 3:
                    continue
                _, x, y = parts[:3]
                coords.append((float(x), float(y)))

    if not coords:
        raise ValueError(f"Instance '{name}' did not provide node coordinates")

    coords_arr = np.array(coords, dtype=float)
    n = len(coords_arr)
    distance_matrix = np.zeros((n, n), dtype=float)

    if edge_weight_type not in {"EUC_2D", "CEIL_2D", "ATT"}:
        raise ValueError(
            f"Unsupported EDGE_WEIGHT_TYPE '{edge_weight_type}' for instance '{name}'"
        )

    for i in range(n):
        distance_matrix[i, i] = 0.0
        for j in range(i + 1, n):
            dx = coords_arr[i, 0] - coords_arr[j, 0]
            dy = coords_arr[i, 1] - coords_arr[j, 1]
            dist = math.sqrt(dx * dx + dy * dy)
            if edge_weight_type == "CEIL_2D":
                value = math.ceil(dist)
            elif edge_weight_type == "ATT":
                rij = math.sqrt((dx * dx + dy * dy) / 10.0)
                tij = int(rij)
                value = tij + 1 if tij < rij else tij
            else:  # EUC_2D
                value = int(dist + 0.5)
            distance_matrix[i, j] = distance_matrix[j, i] = float(value)

    if optimum is None:
        optimum = DEFAULT_OPTIMA.get(name.lower())

    return {
        "name": name,
        "coords": coords_arr,
        "distance_matrix": distance_matrix,
        "optimum": optimum,
    }


class TSPLibProblem(Problem):
    GROUPS: dict[str, dict[str, Iterable[str]]] = {
        "euclidean_small": {
            "train": ("eil51", "berlin52"),
            "test": ("st70",),
        },
        "euclidean_medium": {
            "train": ("ch130", "rat195"),
            "test": ("d198",),
        },
        "euclidean_large": {
            "train": ("lin318", "pcb442"),
            "test": ("att532",),
        },
    }

    def __init__(
        self,
        *,
        group: str = "euclidean_small",
        logger=None,
        training_instances: Iterable[str] | None = None,
        test_instances: Iterable[str] | None = None,
        name: str | None = None,
        eval_timeout: int = 900,
        budget_factor: float = 20.0,
        dependencies: list[str] | None = None,
        imports: str | None = None,
    ) -> None:
        if dependencies is None:
            dependencies = ["numpy>=2"]
        if imports is None:
            imports = "import numpy as np\n"
        if name is None:
            name = f"TSPLib-{group}"
        group = group.lower()
        if group not in self.GROUPS:
            raise ValueError(
                f"Unknown TSPLib group '{group}'. Available groups: {tuple(self.GROUPS)}"
            )
        default_group = self.GROUPS[group]
        if training_instances is None:
            training_instances = tuple(default_group["train"])
        if test_instances is None:
            test_instances = tuple(default_group["test"])

        super().__init__(
            logger=logger,
            training_instances=list(training_instances),
            test_instances=list(test_instances),
            name=name,
            eval_timeout=eval_timeout,
            dependencies=dependencies,
            imports=imports,
        )
        self.group = group
        self.budget_factor = budget_factor
        self._instance_cache: dict[str, dict] = {}

        self.task_prompt = textwrap.dedent(
            """
            You are designing heuristics for symmetric Euclidean Traveling Salesman Problem (TSP)
            instances taken from TSPLIB. Implement a Python class that constructs and improves tours
            for the supplied coordinates. The class must expose the following interface:

            * The class name must match the identifier you provide in the # Description section.
            * `__init__(self, budget: int = 1000, rng: np.random.Generator | None = None)` stores
              the evaluation budget and optional random number generator.
            * `__call__(self, coords: np.ndarray, distance_matrix: np.ndarray) -> tuple[float, list[int]]`
              where `coords` contains the (x, y) coordinates of each city and `distance_matrix`
              contains the pairwise Euclidean distances. The method must return a tuple of the
              length of the best tour discovered and the tour itself as an ordered list of city
              indices using zero-based indexing.

            The algorithm may only use information available through the arguments passed to `__call__`.
            You may rely on NumPy but avoid external optimisation libraries.
            """
        ).strip()
        self.example_prompt = textwrap.dedent(
            """
            A simple example that obeys the required interface is provided below:

            ```python
            import numpy as np

            class GreedyInsertion:
                def __init__(self, budget: int = 1000, rng: np.random.Generator | None = None):
                    self.budget = budget
                    self.rng = np.random.default_rng() if rng is None else rng

                def __call__(self, coords: np.ndarray, distance_matrix: np.ndarray):
                    n = len(coords)
                    remaining = list(range(n))
                    tour = [remaining.pop(0)]
                    while remaining and len(tour) < self.budget:
                        last = tour[-1]
                        next_idx = min(
                            remaining,
                            key=lambda city: distance_matrix[last, city],
                        )
                        tour.append(next_idx)
                        remaining.remove(next_idx)
                    if len(tour) < n:
                        tour.extend(remaining)
                    tour_length = float(
                        np.sum(distance_matrix[tour[i - 1], tour[i]] for i in range(n))
                        + distance_matrix[tour[-1], tour[0]]
                    )
                    return tour_length, tour
            ```
            """
        ).strip()
        self.format_prompt = textwrap.dedent(
            """
            Format the response as follows:
            # Description: <brief summary of the main idea>
            # Code:
            ```python
            <implementation>
            ```
            """
        ).strip()

    def _load_instance(self, name: str) -> dict:
        if name not in self._instance_cache:
            self._instance_cache[name] = _load_tsplib_instance(name)
        return self._instance_cache[name]

    def _instantiate_algorithm(self, cls, budget: int, rng_seed: int):
        for kwargs in (
            {"budget": budget, "rng": np.random.default_rng(rng_seed)},
            {"budget": budget},
            {"rng": np.random.default_rng(rng_seed)},
            {},
        ):
            try:
                return cls(**kwargs)
            except TypeError:
                continue
        return cls()

    def _tour_length(self, distance_matrix: np.ndarray, tour: Iterable[int]) -> float:
        order = list(tour)
        if not order:
            raise ValueError("Tour must contain at least one city")
        n = distance_matrix.shape[0]
        if any(city >= n for city in order):
            order = [city - 1 for city in order]
        if any(city < 0 or city >= n for city in order):
            raise ValueError("Tour indices out of bounds")
        total = 0.0
        for i in range(len(order)):
            total += distance_matrix[order[i - 1], order[i]]
        total += distance_matrix[order[-1], order[0]]
        return float(total)

    def _evaluate_instances(self, solution: Solution, instance_names: Iterable[str]):
        code_globals = {"np": np, "math": math}
        local_env: dict[str, object] = {}
        try:
            exec(solution.code, code_globals, local_env)
        except Exception as exc:  # pragma: no cover - safety net
            solution.set_scores(
                -math.inf,
                feedback=f"Code execution failed: {exc}",
                error=str(exc),
            )
            return solution, []

        algorithm_cls = local_env.get(solution.name)
        if algorithm_cls is None:
            solution.set_scores(
                -math.inf,
                feedback=(
                    "Submitted code must define a class whose name matches the solution name."
                ),
                error="Missing algorithm class",
            )
            return solution, []

        scores = []
        summaries = []
        for idx, name in enumerate(instance_names):
            instance = self._load_instance(name)
            coords = instance["coords"]
            dist = instance["distance_matrix"]
            budget = max(int(self.budget_factor * len(coords)), len(coords))
            algorithm = self._instantiate_algorithm(algorithm_cls, budget, rng_seed=idx)
            try:
                result = algorithm(coords, dist)
            except TypeError:
                result = algorithm(coords)
            except Exception as exc:
                summaries.append(f"{name}: failed ({exc})")
                scores.append(0.0)
                continue

            if isinstance(result, dict):
                length = result.get("best_length") or result.get("length")
                tour = result.get("tour") or result.get("best_tour")
            elif isinstance(result, tuple) and len(result) >= 2:
                length, tour = result[:2]
            else:
                length, tour = result, None

            if tour is not None:
                try:
                    length = self._tour_length(dist, tour)
                except Exception as exc:
                    summaries.append(f"{name}: invalid tour ({exc})")
                    scores.append(0.0)
                    continue

            try:
                length = float(length)
            except Exception:
                summaries.append(f"{name}: non-numeric length")
                scores.append(0.0)
                continue

            optimum = instance.get("optimum")
            if optimum:
                ratio = max(min(optimum / length, 2.0), 0.0) if length > 0 else 0.0
                summaries.append(
                    f"{name}: {length:.1f} (opt={optimum:.1f}, score={ratio:.3f})"
                )
            else:
                ratio = 1.0 / (1.0 + max(length, 0.0))
                summaries.append(f"{name}: {length:.1f} (score={ratio:.3f})")
            scores.append(ratio)

        fitness = float(np.mean(scores)) if scores else -math.inf
        solution.set_scores(fitness, feedback="\n".join(summaries))
        solution.metadata["instance_scores"] = dict(zip(instance_names, scores, strict=False))
        return solution, scores

    def evaluate(self, solution: Solution):
        evaluated, _ = self._evaluate_instances(solution, self.training_instances)
        return evaluated

    def test(self, solution: Solution):
        _, scores = self._evaluate_instances(solution, self.test_instances)
        return float(np.mean(scores)) if scores else float("nan")

    def to_dict(self):
        return {
            "name": self.name,
            "group": self.group,
            "training_instances": self.training_instances,
            "test_instances": self.test_instances,
            "budget_factor": self.budget_factor,
        }
