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

    def evaluate(self, solution: Solution, test=False, ioh_dir=""):
        code = solution.code
        algorithm_name = solution.name
        safe_globals = {"np": np}
        local_env = {}
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
                try:
                    algorithm = local_env[algorithm_name](budget=budget, dim=dim)
                    algorithm(f_new)
                except OverBudgetException:
                    over += 1  # truncated at the budget, scored below
                aucs.append(correct_aoc(f_new, l2, budget))
                l2.reset(f_new)
                if test:
                    l1.reset(f_new)
                f_new.reset()

        auc_mean = float(np.mean(aucs))
        solution.add_metadata("aucs", aucs)
        solution.add_metadata("over_budget_instances", over)
        solution.set_scores(
            auc_mean,
            f"The algorithm {algorithm_name} scored {auc_mean:.3f} on AOCC "
            "(higher is better, 1.0 is the best).",
        )
        return solution


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
