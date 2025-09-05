import ioh
import numpy as np
import pandas as pd
import pytest

import iohblade.behaviour_metrics as bm
import iohblade.llm as llm_mod
import iohblade.utils as utils


@pytest.fixture
def sample_df():
    return pd.DataFrame(
        {
            "evaluations": [1, 2, 3, 4],
            "raw_y": [4.0, 2.0, 3.0, 1.0],
            "x0": [0.0, 1.0, 2.0, 0.0],
            "x1": [0.0, 0.0, 1.0, 0.0],
        }
    )


def test_individual_metrics(sample_df):
    df = sample_df
    assert bm.get_coordinates(df).shape == (4, 2)
    assert bm.get_objective(df).tolist() == [4.0, 2.0, 3.0, 1.0]
    assert bm.average_nearest_neighbor_distance(df, step=1) == pytest.approx(0.80473785)
    bounds = [(0.0, 2.0), (0.0, 1.0)]
    rng = np.random.default_rng(0)
    assert bm.coverage_dispersion(df, bounds=bounds, n_samples=5, rng=rng) >= 0
    assert bm.spatial_entropy(df) > 0
    assert bm.average_distance_to_best_so_far(df) == pytest.approx(0.47140452)
    assert bm.closed_form_random_search_diversity(bounds) == pytest.approx(0.86602540)
    rng = np.random.default_rng(0)
    assert bm.estimate_random_search_diversity(bounds, n=3, samples=5, rng=rng) > 0
    exp, exl = bm.avg_exploration_exploitation_chunked(df, chunk_size=2, bounds=bounds)
    assert exp == 100.0 and exl == 0.0
    assert bm.intensification_ratio(df, radius=1.5) == pytest.approx(0.75)
    assert bm.average_convergence_rate(df, optimum=0) < 1
    avg_imp, succ = bm.improvement_statistics(df)
    assert avg_imp == pytest.approx(1.5)
    assert succ == pytest.approx(2 / 3)
    assert bm.longest_no_improvement_streak(df) == 1
    assert bm.last_improvement_fraction(df) == 0.0


def test_compute_behavior_metrics_deterministic(sample_df, monkeypatch):
    df = sample_df
    original = np.random.default_rng

    def fixed_rng(_seed=None):
        return original(0)

    monkeypatch.setattr(np.random, "default_rng", fixed_rng)

    metrics = bm.compute_behavior_metrics(
        df, bounds=[(0.0, 2.0), (0.0, 1.0)], radius=1.5, disp_samples=5
    )

    expected_keys = {
        "avg_nearest_neighbor_distance",
        "dispersion",
        "avg_exploration_pct",
        "avg_distance_to_best",
        "intensification_ratio",
        "avg_exploitation_pct",
        "average_convergence_rate",
        "avg_improvement",
        "success_rate",
        "longest_no_improvement_streak",
        "last_improvement_fraction",
    }
    assert expected_keys <= metrics.keys()
    assert metrics["avg_improvement"] == pytest.approx(1.5)
    assert metrics["success_rate"] == pytest.approx(2 / 3)


def test_metric_edge_cases(monkeypatch):
    df_single = pd.DataFrame({"evaluations": [1], "raw_y": [1.0], "x0": [0.0]})
    df_two = pd.DataFrame(
        {"evaluations": [1, 2], "raw_y": [1.0, 0.5], "x0": [0.0, 0.1]}
    )

    # _pairwise_distances should return empty array for single point
    X = bm.get_coordinates(df_single)
    assert bm._pairwise_distances(X).size == 0

    # average_nearest_neighbor_distance should early exit
    assert bm.average_nearest_neighbor_distance(df_single) == 0.0

    # coverage_dispersion with default bounds
    disp = bm.coverage_dispersion(
        df_single, bounds=None, n_samples=5, rng=np.random.default_rng(0)
    )
    assert disp >= 0

    # estimate_random_search_diversity with default rng
    est = bm.estimate_random_search_diversity([(0.0, 1.0)], n=2, samples=3)
    assert est >= 0

    # avg_exploration_exploitation_chunked should return defaults when not enough points
    expl, explo = bm.avg_exploration_exploitation_chunked(df_single, chunk_size=2)
    assert (expl, explo) == (0.0, 100.0)

    # compute_behavior_metrics with defaults using two points
    metrics = bm.compute_behavior_metrics(df_two)
    assert "avg_nearest_neighbor_distance" in metrics


class DummyLogger:
    def __init__(self):
        self.logged = []

    def log_conversation(self, who, text, cost=0.0, tokens=0):
        self.logged.append((who, text, cost, tokens))


class DummyLLM(llm_mod.LLM):
    def __init__(self):
        super().__init__(api_key="", model="dummy", logger=DummyLogger())

    def _query(self, session_messages):
        return """```python\nclass Algo:\n    pass\n```\n# Description: test"""


def test_sample_solution_runs():
    llm = DummyLLM()
    solution = llm.sample_solution([{"role": "user", "content": "hello"}])
    assert solution.name == "Algo"
    assert "pass" in solution.code
    assert solution.description == "test"
    assert llm.to_dict()["model"] == "dummy"


def test_aoc_budget_logger_updates():
    logger_instance = utils.aoc_logger(
        budget=2, upper=1e2, triggers=[ioh.logger.trigger.ALWAYS]
    )
    info = ioh.LogInfo(
        evaluations=1,
        raw_y=5.0,
        raw_y_best=5.0,
        transformed_y=0.0,
        transformed_y_best=0.0,
        y=5.0,
        y_best=5.0,
        x=[0.0],
        violations=[],
        penalties=[],
        optimum=ioh.iohcpp.RealSolution([1.0], -1.0),
        has_improved=False,
    )
    logger_instance(info)
    assert logger_instance.aoc > 0
    logger_instance.reset(None)
    assert logger_instance.aoc == 0

    budget = utils.budget_logger(budget=1, triggers=[ioh.logger.trigger.ALWAYS])
    with pytest.raises(utils.OverBudgetException):
        budget(
            ioh.LogInfo(
                evaluations=2,
                raw_y=0.0,
                raw_y_best=0.0,
                transformed_y=0.0,
                transformed_y_best=0.0,
                y=0.0,
                y_best=0.0,
                x=[0.0],
                violations=[],
                penalties=[],
                optimum=ioh.iohcpp.RealSolution([1.0], -1.0),
                has_improved=False,
            )
        )
