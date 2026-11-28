"""Multi-objective continuous benchmarks (ZDT, DTLZ, WFG) from pymoo.

The LLM designs a **multi-objective optimiser**. It is a class with the BBOB
interface, ``__init__(self, budget, dim)`` and ``__call__(self, func)``, where
``func(x)`` returns the vector of objective values (all minimised).

BLADE records every evaluation, as COCO's observer does, so a candidate
cannot misreport its front. The score of a run is the area under the
normalised hypervolume curve over the evaluation budget, the multi-objective
counterpart of AOCC:

* objectives are normalised with the ideal and nadir points of the known
  Pareto front, ``f' = (f - ideal) / (nadir - ideal)``;
* the hypervolume reference point is ``1.1`` in every objective;
* at ``checkpoints`` evaluation counts spread over the budget, the
  hypervolume of everything evaluated so far is divided by the hypervolume
  of the reference front (clipped to 1); the score is their mean.
* As in COCO's bi-objective indicator, while no evaluated point lies inside
  the reference box (hypervolume 0), the value is based on the distance
  ``d`` of the closest point to the box (normalised space): ``-d / (1 + d)``,
  which is bounded but never saturates, so weak algorithms are still
  ranked. Scores are in (-1, 1].

Problems and Pareto fronts come from `pymoo <https://pymoo.org>`_; the
indicators (hypervolume, IGD+) from `moocore
<https://multi-objective.github.io/moocore/python/>`_.
"""

from __future__ import annotations

import math
import textwrap
from types import SimpleNamespace
from typing import Any

import numpy as np

from iohblade.problem import Problem
from iohblade.solution import Solution
from iohblade.utils import OverBudgetException, first_class_name

#: Packages the evaluation environment needs (problems and indicators).
MO_DEPENDENCIES = ["pymoo>=0.6.1,<0.7", "moocore>=0.1.7,<1"]

#: Hypervolume reference point in the normalised objective space.
REFERENCE_POINT = 1.1

#: Default evaluations per run: ``BUDGET_FACTOR * n_var`` (as for BBOB).
BUDGET_FACTOR = 2000

#: Seconds allowed per function evaluation for the default ``eval_timeout``
#: (generous: candidates often evaluate point by point in Python, and the
#: hypervolume is computed at every checkpoint).
SECONDS_PER_EVALUATION = 2e-3
MIN_EVAL_TIMEOUT = 300

#: Function families and their default pymoo arguments.
FAMILIES: dict[str, dict[str, Any]] = {
    "zdt": {},
    "dtlz": {"n_obj": 3},
    "wfg": {"n_var": 10, "n_obj": 2},
}

_EXAMPLE_CODE = textwrap.dedent("""\
    import numpy as np


    class RandomSearch:
        def __init__(self, budget=1000, dim=10):
            self.budget = budget
            self.dim = dim

        def __call__(self, func):
            lb, ub = func.bounds.lb, func.bounds.ub
            # Evaluate the whole budget as one batch; func returns one row
            # of objective values (all minimised) per point. BLADE scores
            # everything that was evaluated, so nothing needs to be returned.
            X = np.random.uniform(lb, ub, size=(self.budget, self.dim))
            F = func(X)
            return X, F
    """)


def function_spec(entry: str | dict) -> dict[str, Any]:
    """A pymoo function as ``{"name": ..., <pymoo arguments>}``.

    ``"zdt1"`` and ``{"name": "dtlz2", "n_obj": 3, "n_var": 12}`` are both
    accepted; the family defaults of :data:`FAMILIES` fill in what is
    missing (DTLZ: 3 objectives; WFG: 10 variables and 2 objectives).
    """
    spec = {"name": entry} if isinstance(entry, str) else dict(entry)
    name = str(spec.get("name", "")).lower()
    if not name:
        raise ValueError(f"A function needs a 'name' (got {entry!r}).")
    family = next((f for f in FAMILIES if name.startswith(f)), None)
    if family is None:
        raise ValueError(
            f"Unknown multi-objective function {name!r}; use a ZDT, DTLZ or "
            "WFG function, e.g. 'zdt1', 'dtlz2' or 'wfg4'."
        )
    return {**FAMILIES[family], **spec, "name": name}


def _label(spec: dict) -> str:
    extra = ",".join(f"{k}={v}" for k, v in spec.items() if k != "name")
    return f"{spec['name']}({extra})" if extra else spec["name"]


def _pymoo_problem(spec: dict):
    from pymoo.problems import get_problem

    kwargs = {k: v for k, v in spec.items() if k != "name"}
    try:
        return get_problem(spec["name"], **kwargs)
    except Exception as e:
        raise ValueError(
            f"pymoo cannot create the function {_label(spec)}: "
            f"{type(e).__name__}: {e}"
        ) from None


class MultiObjectiveFunction:
    """The black-box view of a multi-objective problem given to a candidate.

    * ``func(x)``: evaluate one point (shape ``(dim,)``, returns
      ``(n_objectives,)``) or a batch (shape ``(k, dim)``, returns
      ``(k, n_objectives)``). Every point counts towards the budget; points
      are clipped to the bounds. All objectives are minimised.
    * ``func.bounds.lb`` / ``func.bounds.ub``, ``func.meta_data.n_variables``,
      ``func.meta_data.n_objectives``, ``func.state.evaluations``.

    Calls beyond the budget raise
    :class:`~iohblade.utils.OverBudgetException` (the points that still fit
    are evaluated first).

    Every evaluated objective vector is recorded, in order, for scoring. The
    record is not an attribute of this object (a candidate could otherwise
    add made-up points to it): the harness creates the function with
    :meth:`create`, which also returns a :class:`EvaluationRecord`, and only
    the function is given to the candidate. As for
    :class:`~iohblade.benchmarks.blackbox.BlackBoxFunction`, this removes the
    documented ways to tamper with the score; code in the same process can
    still reach closures by introspection.
    """

    __slots__ = ("_evaluate", "_progress", "bounds", "meta_data")

    @classmethod
    def create(cls, problem, budget: int):
        """The function for the candidate and the record for the harness.

        Returns:
            tuple[MultiObjectiveFunction, EvaluationRecord]: Give only the
            function to the candidate; score with the record.
        """
        record = EvaluationRecord()
        return cls(problem, budget, _record=record), record

    def __init__(self, problem, budget: int, _record: "EvaluationRecord | None" = None):
        lb, ub = (np.array(b, dtype=float) for b in problem.bounds())
        n_var, n_obj = int(problem.n_var), int(problem.n_obj)
        # Kept only in closures (and in the harness's EvaluationRecord).
        archive: list[np.ndarray] = []
        count = [0]
        if _record is not None:
            _record._attach(archive, count, n_obj)

        def evaluate(x):
            X = np.asarray(x, dtype=float)
            single = X.ndim == 1
            X = np.atleast_2d(X)
            if X.ndim != 2 or X.shape[1] != n_var:
                raise ValueError(
                    f"func expects points of dimension {n_var} (one point, "
                    f"or a batch of shape (k, {n_var})); got shape "
                    f"{np.shape(x)}."
                )
            if not np.all(np.isfinite(X)):
                raise ValueError("func got a point with NaN or inf values.")
            allowed = budget - count[0]
            over = len(X) > allowed
            X = np.clip(X[: max(allowed, 0)], lb, ub)
            F = np.empty((0, n_obj))
            if len(X):
                F = np.asarray(problem.evaluate(X), dtype=float).reshape(len(X), n_obj)
                archive.extend(F.copy())
                count[0] += len(X)
            if over:
                raise OverBudgetException(
                    f"The budget of {budget} evaluations is exhausted."
                )
            return F[0].copy() if single else F.copy()

        def progress():
            return SimpleNamespace(evaluations=count[0])

        self._evaluate = evaluate
        self._progress = progress
        self.bounds = SimpleNamespace(lb=lb.copy(), ub=ub.copy())
        self.meta_data = SimpleNamespace(n_variables=n_var, n_objectives=n_obj)

    def __call__(self, x):
        return self._evaluate(x)

    @property
    def state(self) -> SimpleNamespace:
        """Number of evaluations used so far."""
        return self._progress()

    def __repr__(self) -> str:
        return (
            f"MultiObjectiveFunction(dim={self.meta_data.n_variables}, "
            f"n_objectives={self.meta_data.n_objectives})"
        )


class EvaluationRecord:
    """The harness's view of a :class:`MultiObjectiveFunction`'s evaluations.

    Created by :meth:`MultiObjectiveFunction.create`; never given to the
    candidate.
    """

    __slots__ = ("_archive", "_count", "_n_obj")

    def __init__(self):
        self._archive: list[np.ndarray] = []
        self._count = [0]
        self._n_obj = 0

    def _attach(self, archive: list, count: list, n_obj: int) -> None:
        self._archive, self._count, self._n_obj = archive, count, n_obj

    @property
    def evaluations(self) -> int:
        """Number of evaluations so far."""
        return self._count[0]

    def objectives(self) -> np.ndarray:
        """A copy of the evaluated objective vectors, in order."""
        if not self._archive:
            return np.empty((0, self._n_obj))
        return np.array(self._archive, dtype=float)


def anytime_hypervolume(
    F: np.ndarray,
    front: np.ndarray,
    budget: int,
    checkpoints: int = 50,
) -> dict[str, float]:
    """Score a run from its evaluated objective vectors (in order).

    Args:
        F: Objective vectors, shape ``(n_evaluations, n_objectives)``,
            all minimised.
        front: The reference Pareto front of the problem.
        budget: The evaluation budget of the run.
        checkpoints: Number of equal parts of the budget; the hypervolume is
            measured at the end of each (``budget / checkpoints``, ...,
            ``budget``).

    Returns:
        dict[str, float]: ``area`` (the score: the mean over the checkpoints
        of the normalised hypervolume, or, while it is 0, of ``-d / (1 + d)``
        for the distance ``d`` to the reference box; in (-1, 1]),
        ``final_hv`` (normalised
        hypervolume at the end), ``distance`` (distance of the closest final
        point to the reference box, 0 once inside), ``igd_plus`` (IGD+ of the
        final set to the reference front, normalised space) and
        ``n_nondominated``.
    """
    import moocore

    front = np.asarray(front, dtype=float)
    ideal, nadir = front.min(axis=0), front.max(axis=0)
    span = np.where(nadir - ideal > 0, nadir - ideal, 1.0)
    ref = np.full(front.shape[1], REFERENCE_POINT)
    front_n = (front - ideal) / span
    hv_front = float(moocore.hypervolume(front_n, ref=ref))
    F = np.asarray(F, dtype=float).reshape(-1, front.shape[1])
    if len(F) == 0:
        return {
            "area": -1.0,
            "final_hv": 0.0,
            "distance": math.inf,
            "igd_plus": math.inf,
            "n_nondominated": 0,
        }
    F_n = (F - ideal) / span
    # Distance of each point to the reference box [ideal, ref] (0 inside),
    # and the running minimum over the evaluations.
    closest = np.minimum.accumulate(np.linalg.norm(np.maximum(F_n - ref, 0.0), axis=1))
    # Ends of ``checkpoints`` equal parts of the budget: budget/C, ..., budget.
    steps = np.unique(
        np.ceil(np.arange(1, checkpoints + 1) * budget / checkpoints).astype(int)
    )
    values, hv = [], 0.0
    for t in steps:
        n = min(t, len(F_n))
        hv = min(1.0, float(moocore.hypervolume(F_n[:n], ref=ref)) / hv_front)
        d = float(closest[n - 1])
        values.append(hv if hv > 0 else -d / (1.0 + d))
    final = moocore.filter_dominated(F_n)
    return {
        "area": float(np.mean(values)),
        "final_hv": hv,
        "distance": float(closest[-1]),
        "igd_plus": float(moocore.igd_plus(final, ref=front_n)),
        "n_nondominated": int(len(final)),
    }


class MultiObjectiveBenchmark(Problem):
    """Design multi-objective optimisers for ZDT, DTLZ and WFG functions.

    The fitness of a candidate is the mean, over the functions and runs, of
    the area under its normalised hypervolume curve (see
    :func:`anytime_hypervolume`); higher is better, 1.0 is the best, and
    negative values mean no evaluated point reached the reference box.

    Args:
        training_instances: Functions to design on, e.g.
            ``["zdt1", {"name": "dtlz2", "n_obj": 3}]`` (see
            :func:`function_spec`). Default: ZDT1 and ZDT2.
        test_instances: Functions for the final test. Default: ZDT3.
        budget_factor: Evaluations per run = ``budget_factor * n_var`` of the
            function (default 2000, as for BBOB): e.g. 60 000 for the
            30-variable ZDT1, 20 000 for a 10-variable DTLZ or WFG function.
        runs: Independent runs per function (seeds ``0..runs-1``).
        checkpoints: Number of equal parts of each run's budget at whose end
            the hypervolume is measured.
        name: Problem name.
        eval_timeout: Seconds per evaluation of a candidate. Default: from
            the number of function evaluations of the larger of the training
            and test sets (2 ms each, at least 300 s).
        logger, dependencies, imports: See :class:`~iohblade.problem.Problem`.
    """

    minimisation = False

    def __init__(
        self,
        training_instances=None,
        test_instances=None,
        budget_factor: int = BUDGET_FACTOR,
        runs: int = 3,
        checkpoints: int = 50,
        name: str = "MO-ZDT",
        eval_timeout: int | None = None,
        logger=None,
        dependencies=None,
        imports=None,
    ):
        if training_instances is None:
            training_instances = ["zdt1", "zdt2"]
        if test_instances is None:
            test_instances = ["zdt3"]
        for value, label in (
            (budget_factor, "budget_factor"),
            (runs, "runs"),
            (checkpoints, "checkpoints"),
        ):
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"'{label}' must be an integer >= 1 (got {value!r}).")
        train = [function_spec(e) for e in training_instances]
        test = [function_spec(e) for e in test_instances]
        n_vars = {}
        for spec in train + test:  # fail early on a wrong name or argument
            n_vars[_label(spec)] = int(_pymoo_problem(spec).n_var)
        if eval_timeout is None:
            evaluations = max(
                sum(runs * budget_factor * n_vars[_label(s)] for s in specs)
                for specs in (train, test)
            )
            eval_timeout = max(
                MIN_EVAL_TIMEOUT, math.ceil(evaluations * SECONDS_PER_EVALUATION)
            )
        super().__init__(
            logger=logger,
            training_instances=train,
            test_instances=test,
            name=name,
            eval_timeout=eval_timeout,
            dependencies=list(dependencies) if dependencies else list(MO_DEPENDENCIES),
            imports=imports or "import numpy as np\n",
        )
        self.budget_factor = budget_factor
        self.runs = runs
        self.checkpoints = checkpoints
        self.task_prompt = textwrap.dedent("""
            You are a Python expert working on a new multi-objective optimization algorithm. You can use numpy v2.
            Your task is to develop a novel heuristic optimization algorithm for continuous multi-objective optimization problems with 2 or 3 conflicting objectives, all to be minimized (functions from the ZDT, DTLZ and WFG suites).
            The code should contain an `__init__(self, budget, dim)` function with optional additional arguments and the function `def __call__(self, func)`, which should optimize the black box function `func` using `self.budget` function evaluations (the budget is proportional to `dim`).
            `func(x)` takes one point (a numpy array of length `dim`) and returns a numpy array with one value per objective; it also takes a batch of points (shape `(k, dim)`) and returns one row per point. Every point counts towards the budget, and func() can only be called as many times as the budget allows, not more.
            The search space is `func.bounds.lb` to `func.bounds.ub` (points are clipped to it), and `func.meta_data.n_objectives` is the number of objectives.
            The goal is a good approximation of the whole Pareto front, found early: the score is the area under the hypervolume curve of all evaluated points over the budget (normalised, 1.0 is best; negative while no point is close enough to the Pareto front to add hypervolume).
            """)
        self.example_prompt = (
            "\nAn example of such code (a simple random search), is as follows:\n"
            "```python\n" + _EXAMPLE_CODE.rstrip("\n") + "\n```\n"
        )
        self.format_prompt = textwrap.dedent("""
            Give an excellent and novel heuristic algorithm to solve this task and also give it a one-line description, describing the main idea. Give the response in the format:
            # Description: <short-description>
            # Code:
            ```python
            <code>
            ```
            """)

    def get_prompt(self) -> str:
        return self.task_prompt + self.example_prompt + self.format_prompt

    def _run(self, alg_cls, spec: dict, seed: int) -> dict[str, Any]:
        problem = _pymoo_problem(spec)
        front = problem.pareto_front()
        budget = self.budget_factor * int(problem.n_var)
        func, record = MultiObjectiveFunction.create(problem, budget)
        np.random.seed(seed)
        algorithm = alg_cls(budget=budget, dim=int(problem.n_var))
        try:
            algorithm(func)
        except OverBudgetException:
            pass
        F = record.objectives()
        result = anytime_hypervolume(F, front, budget, self.checkpoints)
        return {
            "function": _label(spec),
            "seed": seed,
            "budget": budget,
            "evaluations": len(F),
            **result,
        }

    def evaluate(self, solution: Solution, test: bool = False) -> Solution:
        """Run the candidate on the training (or test) functions and score it.

        Raises:
            Exception: Errors of the candidate code propagate; BLADE turns
                them into the failure fitness with the error as feedback.
        """
        namespace: dict = {"np": np, "math": math}
        exec(solution.code, namespace)  # noqa: S102 (isolated subprocess)
        name = solution.name or first_class_name(solution.code)
        alg_cls = namespace.get(name) or namespace.get(
            first_class_name(solution.code) or ""
        )
        if alg_cls is None:
            raise ValueError(f"Class {name!r} not found in the generated code.")
        specs = self.test_instances if test else self.training_instances
        runs = [
            self._run(alg_cls, spec, seed)
            for spec in specs
            for seed in range(self.runs)
        ]
        if all(r["evaluations"] == 0 for r in runs):
            raise ValueError(
                "The algorithm never called func, so nothing was evaluated."
            )
        score = float(np.mean([r["area"] for r in runs]))
        solution.add_metadata("performance_data", runs)
        lines = []
        for spec in specs:
            rows = [r for r in runs if r["function"] == _label(spec)]
            lines.append(
                f"{_label(spec)}: area {np.mean([r['area'] for r in rows]):.3f}, "
                f"final HV {np.mean([r['final_hv'] for r in rows]):.3f}, "
                f"IGD+ {np.mean([r['igd_plus'] for r in rows]):.4f}, "
                + (
                    f"closest distance to the reference box "
                    f"{np.mean([r['distance'] for r in rows]):.3f}, "
                    if any(r["final_hv"] == 0 for r in rows)
                    else ""
                )
                + f"{np.mean([r['n_nondominated'] for r in rows]):.0f} non-dominated "
                f"points, {np.mean([r['evaluations'] for r in rows]):.0f}/"
                f"{rows[0]['budget']} evaluations"
            )
        feedback = (
            f"The algorithm {solution.name} scored {score:.3f} on the area under "
            "the normalised hypervolume curve (higher is better, 1.0 is the "
            "best; negative while no point reaches the reference box around "
            "the Pareto front).\nPer function (mean over runs): "
            + "; ".join(lines)
            + "."
        )
        return solution.set_scores(score, feedback)

    def test(self, solution: Solution) -> Solution:
        """Score the solution on the test functions."""
        return self.evaluate(solution, test=True)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "training_instances": self.training_instances,
            "test_instances": self.test_instances,
            "budget_factor": self.budget_factor,
            "runs": self.runs,
            "checkpoints": self.checkpoints,
        }

    def get_config(self) -> dict[str, Any]:
        import inspect

        from iohblade.tags import (
            NoiseType,
            ObjectiveType,
            PrimaryCategories,
            VariableType,
        )

        return {
            "tags": [
                PrimaryCategories.BBO,
                ObjectiveType.MULTI_OBJECTIVE,
                NoiseType.NOISELESS,
                VariableType.CONTINUOUS,
            ],
            "name": self.name,
            "prompt": self.get_prompt(),
            "minimisation": self.minimisation,
            "evaluator": inspect.getsource(anytime_hypervolume),
            "config": {
                **self.to_dict(),
                "score": "area under the normalised hypervolume curve",
                "reference_point": REFERENCE_POINT,
                "dependencies": self.dependencies,
            },
        }


def _suite(train, test, name, budget_factor, runs, eval_timeout=None):
    return [
        MultiObjectiveBenchmark(
            training_instances=train,
            test_instances=test,
            budget_factor=budget_factor,
            runs=runs,
            name=name,
            eval_timeout=eval_timeout,
        )
    ]


def mo_quick(budget_factor: int = BUDGET_FACTOR, runs: int = 2) -> list[Problem]:
    """A small multi-objective suite: design on ZDT1 and ZDT2, test on ZDT3."""
    return _suite(["zdt1", "zdt2"], ["zdt3"], "MO-quick", budget_factor, runs)


def mo_zdt(budget_factor: int = BUDGET_FACTOR, runs: int = 3) -> list[Problem]:
    """ZDT (2 objectives): design on ZDT1-3, test on ZDT4 and ZDT6."""
    return _suite(
        ["zdt1", "zdt2", "zdt3"], ["zdt4", "zdt6"], "MO-ZDT", budget_factor, runs
    )


def mo_dtlz(
    budget_factor: int = BUDGET_FACTOR, runs: int = 3, n_obj: int = 3
) -> list[Problem]:
    """DTLZ (3 objectives by default): design on DTLZ1-4, test on DTLZ5-7."""
    train = [{"name": f"dtlz{i}", "n_obj": n_obj} for i in range(1, 5)]
    test = [{"name": f"dtlz{i}", "n_obj": n_obj} for i in range(5, 8)]
    return _suite(train, test, "MO-DTLZ", budget_factor, runs)


def mo_wfg(
    budget_factor: int = BUDGET_FACTOR, runs: int = 3, n_obj: int = 2
) -> list[Problem]:
    """WFG (2 objectives, 10 variables): design on WFG1-5, test on WFG6-9."""
    train = [{"name": f"wfg{i}", "n_var": 10, "n_obj": n_obj} for i in range(1, 6)]
    test = [{"name": f"wfg{i}", "n_var": 10, "n_obj": n_obj} for i in range(6, 10)]
    return _suite(train, test, "MO-WFG", budget_factor, runs)
