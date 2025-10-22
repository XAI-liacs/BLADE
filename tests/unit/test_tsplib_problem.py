import math

from iohblade import Solution
from iohblade.experiment import TSPLibExperiment
from iohblade.loggers import ExperimentLogger
from iohblade.problems import TSPLibProblem


def _trivial_solver_code():
    return """
import numpy as np

class TrivialTour:
    def __init__(self, budget: int = 0, rng=None):
        self.budget = budget

    def __call__(self, coords, distance_matrix):
        n = len(coords)
        tour = list(range(n))
        length = float(
            sum(distance_matrix[tour[i - 1], tour[i]] for i in range(n))
            + distance_matrix[tour[-1], tour[0]]
        )
        return length, tour
"""


def test_tsplib_problem_evaluate_sets_fitness():
    problem = TSPLibProblem(group="euclidean_small")
    solution = Solution(code=_trivial_solver_code(), name="TrivialTour")
    evaluated = problem.evaluate(solution)

    assert math.isfinite(evaluated.fitness)
    assert "instance_scores" in evaluated.metadata
    assert set(evaluated.metadata["instance_scores"].keys()) == set(
        problem.training_instances
    )


def test_tsplib_experiment_builds_all_groups(tmp_path):
    logger = ExperimentLogger(tmp_path / "exp")
    experiment = TSPLibExperiment(methods=[], runs=1, budget=10, exp_logger=logger)
    assert len(experiment.problems) == 3
    assert {p.group for p in experiment.problems} == {
        "euclidean_small",
        "euclidean_medium",
        "euclidean_large",
    }
