"""Problem variants used by the MI experiments.

Kept here instead of in ``iohblade/benchmarks`` because they change the
experimental condition, not the benchmark itself.
"""

import cloudpickle
import ioh
import numpy as np
from ioh import get_problem
from ioh import logger as ioh_logger

from iohblade.benchmarks import MA_BBOB
from iohblade.solution import Solution
from iohblade.utils import OverBudgetException, aoc_logger, correct_aoc


class BudgetEnforcedMABBOB(MA_BBOB):
    """MA-BBOB where overspending the budget truncates a run instead of
    zeroing it.

    In ``MA_BBOB`` an algorithm that calls ``func`` more than ``budget`` times
    gets AOCC 0 on that instance and the evaluation stops. The feedback says
    only "scored 0.000", so the LLM never learns why. In the pilot of
    experiment 1, this happened to 84% of working algorithms.

    Here the run is stopped at the budget and scored on what it reached
    within it (``correct_aoc``), and evaluation continues on all instances.
    The feedback text is unchanged. ``metadata["over_budget_instances"]``
    counts the instances on which the code would have overspent, so it can
    still be analysed as a label.
    """

    # ``MA_BBOB`` executes the code with separate globals and locals, so a
    # module-level import (``from scipy.stats.qmc import Sobol``) lands in
    # the locals and is not visible inside the class's methods (NameError).
    # With ``shared_namespace`` the code runs in one namespace, as a module
    # does. Off by default: experiments 1-5b evaluated with MA_BBOB's rule.
    shared_namespace = False

    def evaluate(self, solution: Solution, test=False, ioh_dir=""):
        code = solution.code
        algorithm_name = solution.name
        safe_globals = {"np": np}
        local_env = safe_globals if self.shared_namespace else {}
        exec(code, safe_globals, local_env)

        # Small test run to catch code errors (as in MA_BBOB).
        try:
            l2_temp = aoc_logger(100, upper=1e2, triggers=[ioh_logger.trigger.ALWAYS])
            problem = get_problem(11, 1, 2)
            problem.attach_logger(l2_temp)
            algorithm = local_env[algorithm_name](budget=100, dim=2)
            algorithm(problem)
        except OverBudgetException:
            pass

        instances = self.test_instances if test else self.training_instances
        aucs, over = [], 0
        self._begin()
        for dim in self.dims:
            for idx in instances:
                budget = self.budget_factor * dim
                f_new = ioh.problem.ManyAffine(
                    xopt=np.array(self.opt_locs.iloc[idx])[:dim],
                    weights=np.array(self.weights.iloc[idx]),
                    instances=np.array(self.iids.iloc[idx], dtype=int),
                    n_variables=dim,
                )
                f_new.set_id(100)
                f_new.set_instance(idx)
                l2 = aoc_logger(budget, upper=1e2, triggers=[ioh_logger.trigger.ALWAYS])
                if test:
                    l1 = ioh.logger.Analyzer(
                        root=ioh_dir,
                        folder_name=algorithm_name,
                        algorithm_name=algorithm_name,
                    )
                    f_new.attach_logger(ioh.logger.Combine([l1, l2]))
                else:
                    f_new.attach_logger(l2)
                target = self._target(f_new, idx)
                try:
                    algorithm = local_env[algorithm_name](budget=budget, dim=dim)
                    algorithm(target)
                except OverBudgetException:
                    over += 1  # truncated at the budget, scored below
                self._instance_done(target, idx)
                aucs.append(correct_aoc(f_new, l2, budget))
                l2.reset(f_new)
                if test:
                    l1.reset(f_new)
                f_new.reset()

        auc_mean = float(np.mean(aucs))
        solution.add_metadata("aucs", aucs)
        solution.add_metadata("over_budget_instances", over)
        self._finish(solution)
        solution.set_scores(
            auc_mean,
            f"The algorithm {algorithm_name} scored {auc_mean:.3f} on AOCC "
            "(higher is better, 1.0 is the best).",
        )
        return solution

    # Hooks for subclasses (no-ops here).
    def _begin(self):
        pass

    def _target(self, f, idx):
        return f

    def _instance_done(self, target, idx):
        pass

    def _finish(self, solution):
        pass


class _Recorder:
    """Callable stand-in for an IOH problem that records every evaluation.

    Attribute access (``bounds``, ``meta_data``, ...) goes to the problem.
    """

    def __init__(self, f):
        self._f = f
        self.X, self.y = [], []

    def __call__(self, x):
        y = self._f(x)  # raises OverBudgetException before recording
        arr = np.asarray(x, dtype=float)
        if arr.ndim == 2:
            self.X.extend(arr)
            self.y.extend(np.asarray(y, dtype=float).ravel())
        else:
            self.X.append(arr.ravel())
            self.y.append(float(y))
        return y

    def __getattr__(self, name):
        return getattr(self._f, name)


def _norms(D: np.ndarray) -> np.ndarray:
    # Not np.linalg.norm: cloudpickle would ship numpy.linalg.linalg, which
    # numpy >= 2.4 (the sandbox's) no longer has.
    return np.sqrt((D * D).sum(1))


def behaviour(X: np.ndarray, y: np.ndarray, lb=-5.0, ub=5.0, chunk=500) -> dict:
    """Search-behaviour metrics of one run (``iohblade.behaviour_metrics``,
    reimplemented with numpy so they run in the evaluation sandbox).

    * ``exploration_pct``: mean pairwise distance of each block of
      ``chunk`` consecutive evaluations, as % of that of uniform random
      search (BLADE's ``avg_exploration_pct``);
    * ``dist_to_best``: mean distance of an evaluation to the best point
      found before it (``avg_distance_to_best``), / the domain width;
    * ``intensification``: share of evaluations within 10% of the domain
      width of the final best point (``intensification_ratio``);
    * ``last_improvement``: share of the run after the last improvement
      (``last_improvement_fraction``, high = stagnation);
    * ``success_rate``: share of evaluations that improve the best so far.
    """
    n, d = X.shape
    if n < 3:
        return {}
    w = ub - lb
    rand = w * np.sqrt(d / 6)
    divs = []
    for i in range(0, n, chunk):
        c = X[i : i + chunk]
        if len(c) < 2:
            continue
        sq = (c * c).sum(1)
        D = np.sqrt(np.maximum(sq[:, None] + sq[None] - 2 * c @ c.T, 0))
        divs.append(D[np.triu_indices(len(c), 1)].mean())
    best_so_far = np.minimum.accumulate(y)
    best_idx = np.zeros(n, dtype=int)
    b = 0
    for k in range(n):  # index of the best point among 0..k
        if y[k] < y[b]:
            b = k
        best_idx[k] = b
    prev_best = X[best_idx[:-1]]
    dist_best = _norms(X[1:] - prev_best).mean() / w
    final = X[int(np.argmin(y))]
    inten = float(np.mean(_norms(X - final) < 0.1 * w))
    imp = np.flatnonzero(np.diff(best_so_far) < 0)
    last = imp.max() + 1 if imp.size else 0
    return {
        "exploration_pct": float(np.clip(100 * np.mean(divs) / rand, 0, 100)),
        "dist_to_best": float(dist_best),
        "intensification": inten,
        "last_improvement": float((n - 1 - last) / (n - 1)),
        "success_rate": float(imp.size / (n - 1)),
    }


class TracedMABBOB(BudgetEnforcedMABBOB):
    """``BudgetEnforcedMABBOB`` that also measures how the algorithm searches.

    Every evaluation on the training instances is recorded and summarised
    by :func:`behaviour`; ``metadata["behaviour"]`` holds the mean of each
    metric over instances (``metadata["behaviour_per_instance"]`` the
    values). Fitness and feedback are unchanged.
    """

    def _begin(self):
        self._beh = []

    def _target(self, f, idx):
        return _Recorder(f)

    def _instance_done(self, target, idx):
        if len(target.X) >= 3:
            X = np.asarray(target.X, dtype=float)
            y = np.asarray(target.y, dtype=float)
            ok = np.isfinite(y) & np.isfinite(X).all(1)
            self._beh.append(behaviour(X[ok], y[ok]))
        target.X, target.y = [], []

    def _finish(self, solution):
        beh = [b for b in self._beh if b]
        mean = {k: float(np.mean([b[k] for b in beh])) for k in (beh[0] if beh else {})}
        solution.add_metadata("behaviour", mean)
        solution.add_metadata("behaviour_per_instance", beh)


REJECTED = (
    "The algorithm was rejected before evaluation: a learned check predicts "
    "that it will crash or time out."
)


def _finite_or_none(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if np.isfinite(value) else None


class PreScreenedMABBOB(BudgetEnforcedMABBOB):
    """``BudgetEnforcedMABBOB`` with a probe-based pre-screen.

    The solution must carry ``metadata["prescreen_score"]`` (set by
    ``common.prescreen.ScreenedHookedLLM``). Every solution is still
    evaluated, so the true outcome is known for every flagged one:

    * ``act=False`` (shadow): LLaMEA sees the true result; the flag is only
      logged.
    * ``act=True``: a solution with score above ``threshold`` is shown to
      LLaMEA as failed (fitness ``-inf``, error/feedback ``REJECTED``).

    In both cases ``metadata`` gets ``prescreen_flagged`` and the true
    result (``true_fitness``, ``true_error``, ``true_feedback``), so the
    log can be analysed exactly like an unscreened campaign. Evaluation
    still costs CPU time here; the saving is measured, not realised.
    """

    def __init__(self, *args, threshold: float, act: bool, **kwargs):
        super().__init__(*args, **kwargs)
        self.threshold = threshold
        self.act = act

    def __call__(self, solution: Solution, logger=None):
        run_logger, self.logger = self.logger, None  # log once, below
        try:
            if run_logger is not None and run_logger.budget_exhausted():
                self.logger = run_logger
                return super().__call__(solution)
            solution = super().__call__(solution)
        finally:
            self.logger = run_logger
        score = solution.metadata.get("prescreen_score")
        flagged = score is not None and score > self.threshold
        solution.add_metadata("prescreen_flagged", bool(flagged))
        solution.add_metadata("prescreen_act", bool(self.act))
        solution.add_metadata("true_fitness", _finite_or_none(solution.fitness))
        solution.add_metadata("true_error", solution.error or "")
        solution.add_metadata("true_feedback", solution.feedback)
        if flagged and self.act:
            solution.set_scores(-np.inf, feedback=REJECTED, error=REJECTED)
        if self.logger is not None:
            self.logger.log_individual(solution)
        return solution


# The evaluation sandbox only has BLADE's repo root on its path; ship this
# module inside the pickle so the subclass can be loaded there.
import sys as _sys

cloudpickle.register_pickle_by_value(_sys.modules[__name__])
