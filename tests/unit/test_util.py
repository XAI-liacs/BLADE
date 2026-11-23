import ioh
import numpy as np
import pandas as pd
import pytest

from iohblade.utils import (
    OverBudgetException,
    aoc_logger,
    bootstrap_ci,
    class_info,
    cliffs_delta,
    code_compare,
    compare_auc,
    convert_to_serializable,
    correct_aoc,
    first_class_name,
    is_jsonable,
    paired_cohens_d,
)


def test_code_compare():
    code1 = "print('Hello')"
    code2 = "print('Hello')\nprint('One more line')"
    distance = code_compare(code1, code2)
    assert 0.0 <= distance <= 1.0


def test_is_jsonable():
    assert is_jsonable({"a": 1})
    assert not is_jsonable(set([1, 2, 3]))


def test_first_class_name():
    code_example = """
class my_class:
    \"\"\"A simple class example.\"\"\"
    def __init__(self):
        pass
"""
    assert (
        first_class_name(code_example) == "my_class"
    ), "The class name should be 'my_class'"
    assert (
        class_info(code_example)[1] == "A simple class example."
    ), "The class docstring should match"


def test_convert_to_serializable():
    data = {
        "a": np.int32(5),
        "b": np.float64(3.14),
        "arr": np.array([1, 2, 3]),
        "normal": "string",
    }
    s = convert_to_serializable(data)
    assert s["a"] == 5
    assert s["b"] == 3.14
    assert s["arr"] == [1, 2, 3]
    assert s["normal"] == "string"


def test_correct_aoc():
    class MockLogger:
        def __init__(self):
            self.lower = 1e-8
            self.upper = 1e8
            self.aoc = 10
            self.transform = lambda x: x

    class MockFunction:
        def __init__(self):
            self.state = type("State", (), {})()
            self.state.current_best_internal = type("Best", (), {})()
            self.state.current_best_internal.y = 2.0
            self.state.evaluations = 50

    aoc_val = correct_aoc(MockFunction(), MockLogger(), budget=100)
    # Because we start with logger.aoc=10 and fraction = 2 / range(1e-8 .. 1e8) ~ 2 / 1e8 => basically 0
    # The math ends up just a small difference. We'll just assert it doesn't error out.
    assert 0 <= aoc_val <= 1


def test_aoc_logger():
    logger_instance = aoc_logger(
        budget=5, upper=1e2, triggers=[ioh.logger.trigger.ALWAYS]
    )
    log_info = ioh.LogInfo(
        evaluations=1,
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

    # If evaluations > budget => OverBudgetException
    with pytest.raises(OverBudgetException):
        logger_instance(
            ioh.LogInfo(
                evaluations=6,
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


def test_cliffs_delta_all_less():
    a = [1, 2]
    b = [3, 4]
    assert cliffs_delta(a, b) == -1.0


def test_paired_cohens_d():
    a = [2, 4, 7]
    b = [1, 3, 5]
    assert paired_cohens_d(a, b) == pytest.approx(2.309, rel=1e-3)


def test_bootstrap_ci_contains_mean():
    diff = np.array([1.0, 2.0, 3.0])
    ci_low, ci_high = bootstrap_ci(diff, n_boot=1000, alpha=0.1, rng=123)
    assert ci_low < diff.mean() < ci_high


def test_compare_auc_basic():
    class DummyLogger:
        def __init__(self, frame):
            self.frame = frame

        def get_methods_problems(self):
            return ["method_a", "method_b"], ["problem_1"]

        def get_problem_data(self, problem_name):
            return self.frame.copy()

    rows = []
    for seed in [1, 2]:
        for idx in range(3):
            rows.append(
                {
                    "method_name": "method_a",
                    "seed": seed,
                    "_id": idx,
                    "fitness": idx + seed,
                    "code": "a",
                }
            )
            rows.append(
                {
                    "method_name": "method_b",
                    "seed": seed,
                    "_id": idx,
                    "fitness": idx,
                    "code": "b",
                }
            )

    logger_instance = DummyLogger(pd.DataFrame(rows))
    result = compare_auc(
        logger_instance,
        "method_a",
        "method_b",
        budget=3,
        metric="fitness",
        test="ttest",
        n_boot=500,
        alpha=0.1,
        rng=123,
    )
    assert result.shape[0] == 1
    assert result["n_seeds"].iloc[0] == 2


def _auc_logger(rows, minimisation=None):
    class Logger:
        def get_methods_problems(self):
            return ["a", "b"], ["p"]

        def get_problem_data(self, problem_name):
            return pd.DataFrame(rows)

        if minimisation is not None:

            def get_minimisation(self, problem_name):
                return minimisation

    return Logger()


def test_compare_auc_follows_direction_and_ignores_failures():
    # Minimisation (#202): method a reaches 1.0, method b stays at 5.0; a
    # failed evaluation (inf) must not count as 0, the best value.
    rows = []
    for seed in (1, 2, 3):
        for idx, (fa, fb) in enumerate([(5.0, 5.0), (1.0, np.inf), (1.0, 5.0)]):
            rows.append({"method_name": "a", "seed": seed, "_id": idx, "fitness": fa})
            rows.append({"method_name": "b", "seed": seed, "_id": idx, "fitness": fb})
    result = compare_auc(
        _auc_logger(rows, minimisation=True),
        "a",
        "b",
        budget=3,
        test="ttest",
        n_boot=50,
    )
    row = result.iloc[0]
    assert bool(row["minimisation"])
    # a: 5,1,1 -> trapezoid 4.0 ; b: 5,5,5 -> 10.0 (lower is better)
    assert row["mean_auc_a"] == pytest.approx(4.0)
    assert row["mean_auc_b"] == pytest.approx(10.0)


def test_compare_auc_carries_last_value_to_budget():
    rows = [
        {"method_name": "a", "seed": 1, "_id": 0, "fitness": 2.0},
        {"method_name": "b", "seed": 1, "_id": 0, "fitness": 1.0},
        {"method_name": "b", "seed": 1, "_id": 1, "fitness": 1.0},
        {"method_name": "b", "seed": 1, "_id": 2, "fitness": 1.0},
        {"method_name": "a", "seed": 2, "_id": 0, "fitness": 2.0},
        {"method_name": "b", "seed": 2, "_id": 0, "fitness": 1.0},
    ]
    result = compare_auc(_auc_logger(rows), "a", "b", budget=3, test="ttest", n_boot=50)
    # a stopped after one evaluation: its 2.0 is carried to evaluation 3.
    assert result.iloc[0]["mean_auc_a"] == pytest.approx(4.0)


def test_compare_auc_pairs_by_seed_with_several_llms():
    # Method a ran seed 1 with two LLMs; method b ran seeds 1 and 2. The
    # paired comparison must use seed 1 only, with a's two LLM runs averaged.
    rows = [
        {"method_name": "a", "llm_name": "x", "seed": 1, "_id": 0, "fitness": 2.0},
        {"method_name": "a", "llm_name": "y", "seed": 1, "_id": 0, "fitness": 4.0},
        {"method_name": "b", "llm_name": "x", "seed": 1, "_id": 0, "fitness": 1.0},
        {"method_name": "b", "llm_name": "x", "seed": 2, "_id": 0, "fitness": 9.0},
    ]
    result = compare_auc(_auc_logger(rows), "a", "b", budget=2, test="ttest", n_boot=50)
    row = result.iloc[0]
    assert row["n_seeds"] == 1
    assert row["mean_auc_a"] == pytest.approx(3.0)  # mean of 2.0 and 4.0 over 1 step
    assert row["mean_auc_b"] == pytest.approx(1.0)
