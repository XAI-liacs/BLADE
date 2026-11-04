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


# The evaluation sandbox only has BLADE's repo root on its path; ship this
# module inside the pickle so the subclass can be loaded there.
import sys as _sys

cloudpickle.register_pickle_by_value(_sys.modules[__name__])
