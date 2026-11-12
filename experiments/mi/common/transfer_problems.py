"""Circle packing and photonics problems for the transfer tests of exp05.

BLADE's own ``UnitSquarePacking`` and ``Photonics`` have sandbox problems
that would make most generated code fail for reasons unrelated to the
algorithm (checked with probe programs, see the exp05 README):

* both execute the code with separate globals and locals
  (``exec(code, globals, locals)``), so a module-level helper function,
  helper class or import is invisible inside the class's methods
  (``NameError``);
* ``Photonics``' sandbox lacks ``six``, which ``pyGDM2`` imports, and the
  photonics package imports ``pyGDM2`` at import time: every evaluation
  fails;
* ``Photonics`` computes AOCC on the raw cost ``1 - R`` in a log window of
  ``[1e-8, 1e2]``; costs lie in ``[0.165, 1]``, so every algorithm scores
  between 0.20 and 0.28. Its intended bounds (``auc_lower``,
  ``auc_upper``) are computed and never used;
* ``Photonics``' description and algorithmic insights are only in
  ``get_prompt``, which BLADE's LLaMEA wrapper does not use;
* ``Photonics`` gives AOCC 0 and stops evaluating when an algorithm
  overspends the budget (as ``MA_BBOB``; ``BudgetEnforcedMABBOB`` fixes
  this for MA-BBOB).

The classes here fix these in subclasses (the benchmarks are unchanged):
code runs in one namespace, as a module does; the sandbox gets ``six`` and
the extended packages of exp05 (``scipy``, ``scikit-learn``); photonics
AOCC is computed on the log *regret* to the known optimum, a run that
overspends is truncated at the budget, and every evaluation is recorded
for the search-behaviour metrics of ``problems.behaviour``.
"""

import sys as _sys
import time

import cloudpickle
import numpy as np
from ioh import get_problem
from ioh import logger as ioh_logger

from iohblade.benchmarks.packing.unit_square_packing import UnitSquarePacking
from iohblade.benchmarks.photonics.photonics import Photonics
from iohblade.solution import Solution
from iohblade.utils import OverBudgetException, aoc_logger

from .problems import behaviour


def _exec_module(code: str, name: str) -> dict:
    """Run ``code`` in one namespace, as a module, and return it."""
    ns = {"np": np, "__name__": "candidate"}
    exec(compile(code, name, "exec"), ns)
    return ns


class SandboxedUnitSquarePacking(UnitSquarePacking):
    """``UnitSquarePacking`` (AlphaEvolve B.12) evaluated in one namespace.

    The checks and the score (sum of radii, ``-inf`` if infeasible) are
    those of ``UnitSquarePacking.evaluate``; only the execution of the
    code differs. ``metadata["runtime_s"]`` holds the time the candidate
    took.
    """

    def __init__(
        self,
        n_circles: int = 26,
        best_known: float = 2.635,
        eval_timeout: int = 120,
        extra_dependencies=(),
    ):
        super().__init__(n_circles=n_circles, best_known=best_known)
        self.eval_timeout = eval_timeout
        # The base adds an unpinned "scipy"; the extended deps pin it.
        if extra_dependencies:
            self.dependencies = [d for d in self.dependencies if d != "scipy"]
        self.dependencies += list(extra_dependencies)
        # The example's body is indented with tabs after a line indented
        # with spaces; code copied from it fails with a TabError.
        self.example_prompt = self.example_prompt.replace("\n\t", "\n        ")

    def evaluate(self, solution: Solution, explogger=None):
        name = solution.name or "UnitSquarePackingSolver"
        t0 = time.time()
        try:
            cls = _exec_module(solution.code, name)[name]
            if self.best_solution is not None:
                circles = cls(self.n_circles, self.best_solution)()
            else:
                circles = cls(self.n_circles)()
        except Exception as e:
            solution.set_scores(float("-inf"), f"exec-error {e}", e)
            return solution
        solution.add_metadata("runtime_s", time.time() - t0)
        try:
            U = np.asarray(circles, dtype=float)
            n = self.n_circles
            if U.shape != (n, 3):
                raise ValueError(f"Format error: expected ({n},3), got {U.shape}")
            if not np.isfinite(U).all():
                raise ValueError("Non-finite value in the circles.")
            if np.any(U[:, 2] <= 0):
                idx = int(np.where(U[:, 2] <= 0)[0][0])
                raise ValueError(f"Negative radius at index {idx}.")
            x, y, r = U[:, 0], U[:, 1], U[:, 2]
            tol = self.tolerance
            if (
                np.any(x - r < -tol)
                or np.any(x + r > 1 + tol)
                or np.any(y - r < -tol)
                or np.any(y + r > 1 + tol)
            ):
                raise ValueError("Circle outside the unit square.")
            for i in range(n):
                for j in range(i + 1, n):
                    dx, dy = U[i, 0] - U[j, 0], U[i, 1] - U[j, 1]
                    if dx * dx + dy * dy < (U[i, 2] + U[j, 2] - tol) ** 2:
                        raise ValueError(f"circles {i} and {j} overlap.")
            score = float(np.sum(r))
            solution.set_scores(
                score,
                f"sum_of_radii={score:.6f}; n={n}, best known={self.best_known}",
            )
        except Exception as e:
            solution.set_scores(float("-inf"), f"calc-error {e}", e)
        return solution


class _BudgetRecorder:
    """Stand-in for the photonic problem: records every evaluation and
    raises ``OverBudgetException`` instead of exceeding ``budget``."""

    def __init__(self, f, budget: int):
        self._f = f
        self.budget = budget
        self.X, self.y = [], []

    def __call__(self, x):
        arr = np.asarray(x, dtype=float)
        rows = arr.reshape(-1, arr.shape[-1]) if arr.ndim > 1 else arr[None]
        if len(self.y) + len(rows) > self.budget:
            raise OverBudgetException
        out = [float(self._f(r)) for r in rows]
        self.X.extend(rows)
        self.y.extend(out)
        return np.array(out) if arr.ndim > 1 else out[0]

    def __getattr__(self, name):
        return getattr(self._f, name)


def regret_aocc(y, budget: int, f_star: float, lower=1e-8, upper=1.0) -> float:
    """AOCC of the best-so-far log regret, in ``[0, 1]`` (1: optimum at
    once). Evaluations not used count with the final best."""
    y = np.asarray(y, dtype=float)[:budget]
    if y.size == 0:
        return 0.0
    y = np.where(np.isfinite(y), y, np.inf)
    r = np.clip(np.minimum.accumulate(y) - f_star, lower, upper)
    c = (np.log10(r) - np.log10(lower)) / (np.log10(upper) - np.log10(lower))
    c = np.concatenate([c, np.full(budget - c.size, c[-1])])
    return float(1.0 - c.mean())


PHOTONICS_DEPENDENCIES = [
    "ioh==0.3.22",
    "pandas==2.2.3",
    "pymoosh==3.2",
    "pyGDM2==1.1.12",
    "six==1.17.0",  # imported by pyGDM2, not declared by it
]
# Best known cost (1 - reflectivity): the quarter-wave stack; differential
# evolution with polishing finds the same value to 1e-13.
F_STAR = {"bragg": 0.164855771904698}


class TracedPhotonics(Photonics):
    """BLADE's photonics problem (default: mini-Bragg, 10 layers) with the
    fixes listed in the module docstring.

    Fitness: mean over ``seeds`` runs of :func:`regret_aocc` (budget
    ``budget_factor * dim``). ``metadata``: ``aucs`` (per run),
    ``final_regret``, ``aocc_blade`` (BLADE's raw-cost AOCC, for
    reference), ``over_budget_runs``, ``behaviour`` (mean over runs, on
    the design normalised to the unit box) and ``behaviour_per_run``.
    ``np.random`` is seeded with the run index.
    """

    def __init__(
        self,
        problem_type: str = "bragg",
        budget_factor: int = 500,
        seeds: int = 5,
        eval_timeout: int = 300,
        extra_dependencies=(),
    ):
        super().__init__(
            None,
            problem_type=problem_type,
            name=f"Photonics_{problem_type}",
            eval_timeout=eval_timeout,
            budget_factor=budget_factor,
            seeds=seeds,
            dependencies=PHOTONICS_DEPENDENCIES + list(extra_dependencies),
        )
        self.f_star = F_STAR[problem_type]
        # BLADE's LLaMEA wrapper passes task, example and format prompts
        # only, which drops the problem description and the algorithmic
        # insights of ``get_prompt``. Fold them in, in the same order.
        self.task_prompt = self.task_prompt + self.description_prompt
        self.example_prompt = self.example_prompt + self.extra_prompt
        self.description_prompt = self.extra_prompt = ""

    def evaluate(self, solution: Solution, test=False, ioh_dir=""):
        name = solution.name
        ns = _exec_module(solution.code, name)

        # Small test run to catch code errors (as in Photonics).
        try:
            l_tmp = aoc_logger(100, upper=1e2, triggers=[ioh_logger.trigger.ALWAYS])
            f_tmp = get_problem(11, 1, 2)
            f_tmp.attach_logger(l_tmp)
            ns[name](budget=100, dim=2)(f_tmp)
        except OverBudgetException:
            pass

        dim = self.problem.meta_data.n_variables
        budget = self.budget_factor * dim
        lb = np.asarray(self.problem.bounds.lb, dtype=float)
        ub = np.asarray(self.problem.bounds.ub, dtype=float)
        aucs, finals, blade, beh, over = [], [], [], [], 0
        for seed in self.seeds:
            np.random.seed(seed)
            self.problem.reset()
            target = _BudgetRecorder(self.problem, budget)
            try:
                ns[name](budget=budget, dim=dim)(target)
            except OverBudgetException:
                over += 1  # truncated at the budget, scored below
            y = np.asarray(target.y, dtype=float)
            aucs.append(regret_aocc(y, budget, self.f_star))
            blade.append(regret_aocc(y, budget, 0.0, 1e-8, 1e2))
            ok = np.isfinite(y)
            finals.append(float(y[ok].min() - self.f_star) if ok.any() else None)
            if ok.sum() >= 3:
                X = (np.asarray(target.X, dtype=float)[ok] - lb) / (ub - lb)
                beh.append(behaviour(X, y[ok], lb=0.0, ub=1.0))
        self.problem.reset()

        auc_mean = float(np.mean(aucs))
        beh = [b for b in beh if b]
        solution.add_metadata("aucs", aucs)
        solution.add_metadata("final_regret", finals)
        solution.add_metadata("aocc_blade", float(np.mean(blade)))
        solution.add_metadata("over_budget_runs", over)
        solution.add_metadata(
            "behaviour",
            {k: float(np.mean([b[k] for b in beh])) for k in (beh[0] if beh else {})},
        )
        solution.add_metadata("behaviour_per_run", beh)
        solution.set_scores(
            auc_mean,
            f"The algorithm {name} scored {auc_mean:.3f} on AOCC "
            "(higher is better, 1.0 is the best).",
        )
        return solution


# The evaluation sandbox only has BLADE's repo root on its path; ship this
# module inside the pickle so the subclasses can be loaded there.
cloudpickle.register_pickle_by_value(_sys.modules[__name__])
