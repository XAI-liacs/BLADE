"""Analysis and visualisation: optimisation direction, failures, run identity.

See GitHub issue #202. Synthetic experiment folders are written in
``tmp_path`` in the format of :class:`iohblade.loggers.ExperimentLogger`.
"""

import json
import math
import os

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

import iohblade.behaviour_metrics as bm
from iohblade.loggers import ExperimentLogger
from iohblade.plots import (
    aggregate_best_so_far,
    best_so_far,
    clean_fitness,
    fitness_table,
    plot_code_evolution_graphs,
    plot_convergence,
)
from iohblade.webapp import (
    convergence_dataframe,
    final_best_dataframe,
    plotly_convergence,
    run_evaluations,
    top_solutions,
)

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _solution(i, fitness, code="def f():\n    return 1\n"):
    return {
        "id": f"s{i}",
        "name": f"Algo{i}",
        "fitness": fitness,
        "code": code,
        "parent_ids": [],
        "feedback": "",
        "error": "",
    }


def write_experiment(
    root,
    runs,
    minimisation=None,
    progress_minimisation=True,
    finished=True,
):
    """Write an experiment folder.

    Args:
        runs: list of (method, llm, problem, seed, [fitness, ...]).
        minimisation: dict problem -> bool, logged in progress.json.
        progress_minimisation: also log the direction in progress.json.
        finished: write the runs to experimentlog.jsonl (else only to
            progress.json as running).
    """
    os.makedirs(root, exist_ok=True)
    minimisation = minimisation or {}
    progress_runs = []
    for method, llm, problem, seed, fitnesses in runs:
        log_dir = f"run-{method}-{llm}-{problem}-{seed}"
        os.makedirs(os.path.join(root, log_dir), exist_ok=True)
        with open(os.path.join(root, log_dir, "log.jsonl"), "w") as f:
            for i, fit in enumerate(fitnesses):
                f.write(json.dumps(_solution(i, fit)) + "\n")
        entry = {
            "method_name": method,
            "llm_name": llm,
            "problem_name": problem,
            "seed": seed,
            "budget": len(fitnesses),
            "evaluations": len(fitnesses),
            "start_time": "2025-01-01T00:00:00",
            "end_time": "2025-01-01T01:00:00" if finished else None,
            "log_dir": log_dir,
        }
        if progress_minimisation and problem in minimisation:
            entry["minimisation"] = minimisation[problem]
        progress_runs.append(entry)
        if finished:
            best = fitnesses[-1]
            with open(os.path.join(root, "experimentlog.jsonl"), "a") as f:
                f.write(
                    json.dumps(
                        {
                            "method_name": method,
                            "problem_name": problem,
                            "llm_name": llm,
                            "method": {},
                            "problem": {"name": problem},
                            "llm": {"model": llm},
                            "solution": _solution(99, best),
                            "log_dir": log_dir,
                            "seed": seed,
                        }
                    )
                    + "\n"
                )
    with open(os.path.join(root, "progress.json"), "w") as f:
        json.dump(
            {
                "start_time": "2025-01-01T00:00:00",
                "end_time": "2025-01-01T01:00:00" if finished else None,
                "current": len(runs) if finished else 0,
                "total": len(runs),
                "runs": progress_runs,
            },
            f,
        )
    return ExperimentLogger(str(root), read=True)


def _curve(df, run_col="run"):
    return [None if math.isnan(v) else v for v in df["best_so_far"].tolist()]


# ---------------------------------------------------------------------------
# best-so-far helper
# ---------------------------------------------------------------------------


def test_minimisation_curve_is_cumulative_min():
    df = pd.DataFrame({"run": "r", "_id": [0, 1, 2], "fitness": [10.0, 8.0, 12.0]})
    out = best_so_far(df, minimisation=True)
    assert out["best_so_far"].tolist() == [10.0, 8.0, 8.0]
    assert out["eval"].tolist() == [1, 2, 3]
    out = best_so_far(df, minimisation=False)
    assert out["best_so_far"].tolist() == [10.0, 10.0, 12.0]


def test_best_so_far_sorts_by_id_within_run():
    df = pd.DataFrame({"run": "r", "_id": [2, 0, 1], "fitness": [1.0, 5.0, 3.0]})
    out = best_so_far(df, minimisation=True)
    assert out["_id"].tolist() == [0, 1, 2]
    assert out["best_so_far"].tolist() == [5.0, 3.0, 1.0]


@pytest.mark.parametrize(
    "minimisation, failure",
    [
        (True, float("inf")),
        (True, "inf"),
        (True, float("nan")),
        (False, float("-inf")),
        (False, "-inf"),
        (False, float("nan")),
        (False, None),
        (False, "not a number"),
    ],
)
def test_failures_never_improve_and_are_not_zero(minimisation, failure):
    # A failure first, between two valid values and at the end.
    df = pd.DataFrame(
        {
            "run": "r",
            "_id": [0, 1, 2, 3, 4],
            "fitness": pd.Series([failure, 5.0, failure, 3.0, failure], dtype=object),
        }
    )
    out = best_so_far(df, minimisation=minimisation)
    curve = out["best_so_far"].tolist()
    assert math.isnan(curve[0])  # leading NaN until the first valid value
    expected = [5.0, 5.0, 3.0, 3.0] if minimisation else [5.0, 5.0, 5.0, 5.0]
    assert curve[1:] == expected
    assert 0.0 not in curve
    assert out["fitness_value"].isna().tolist() == [True, False, True, False, True]


def test_minimisation_failure_inf_is_not_best():
    """Old code replaced inf by 0, which is the *best* value when minimising."""
    df = pd.DataFrame(
        {"run": "r", "_id": [0, 1, 2], "fitness": [4.0, float("inf"), 2.0]}
    )
    out = best_so_far(df, minimisation=True)
    assert out["best_so_far"].tolist() == [4.0, 4.0, 2.0]


def test_clean_fitness_multiobjective_and_strings():
    out = clean_fitness(
        pd.Series([{"f1": 1.0, "f2": 3.0}, "2.5", "-inf", {"f1": float("nan")}, 7])
    )
    assert out.iloc[0] == pytest.approx(2.0)
    assert out.iloc[1] == pytest.approx(2.5)
    assert math.isnan(out.iloc[2]) and math.isnan(out.iloc[3])
    assert out.iloc[4] == 7.0


def test_two_llms_same_method_and_seed_are_separate_runs():
    df = pd.DataFrame(
        {
            "method_name": ["M"] * 4,
            "llm_name": ["a", "a", "b", "b"],
            "problem_name": ["P"] * 4,
            "seed": [0] * 4,
            "_id": [0, 1, 0, 1],
            "fitness": [1.0, 2.0, 10.0, 5.0],
        }
    )
    out = best_so_far(df, minimisation=False)
    a = out[out["llm_name"] == "a"]["best_so_far"].tolist()
    b = out[out["llm_name"] == "b"]["best_so_far"].tolist()
    assert a == [1.0, 2.0]
    assert b == [10.0, 10.0]  # not mixed with run "a"


@pytest.mark.parametrize("minimisation", [True, False])
def test_mean_of_unequal_runs_is_monotone(minimisation):
    sign = -1 if minimisation else 1
    long_run = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    short_run = [10.0, 11.0]  # stops early with a better value
    rows = []
    for name, values in (("long", long_run), ("short", short_run)):
        for i, v in enumerate(values):
            rows.append({"run": name, "_id": i, "fitness": sign * v})
    curves = best_so_far(pd.DataFrame(rows), minimisation=minimisation)
    summary = aggregate_best_so_far(curves)
    assert summary["eval"].tolist() == [1, 2, 3, 4, 5, 6]
    mean = summary["mean"].to_numpy()
    diffs = np.diff(mean)
    if minimisation:
        assert (diffs <= 0).all()
    else:
        assert (diffs >= 0).all()
    # the short run is carried forward with its final best value (11)
    assert mean[-1] == pytest.approx(sign * (6.0 + 11.0) / 2)
    assert summary["n_runs"].tolist() == [2] * 6


# ---------------------------------------------------------------------------
# logger: direction, run identity, tolerant reading
# ---------------------------------------------------------------------------


def test_get_minimisation_from_progress(tmp_path):
    logger = write_experiment(
        tmp_path / "exp",
        [("M", "llm", "P", 0, [1.0]), ("M", "llm", "Q", 0, [1.0])],
        minimisation={"P": True, "Q": False},
    )
    assert logger.get_minimisation("P") is True
    assert logger.get_minimisation("Q") is False
    assert logger.get_minimisation("unknown") is False


def test_get_minimisation_from_problem_json(tmp_path):
    root = tmp_path / "exp"
    logger = write_experiment(
        root,
        [("M", "llm", "P", 0, [1.0])],
        minimisation={"P": True},
        progress_minimisation=False,
    )
    assert logger.get_minimisation("P") is False  # not logged
    with open(root / "run-M-llm-P-0" / "problem.json", "w") as f:
        json.dump({"name": "P", "minimisation": True}, f)
    assert logger.get_minimisation("P") is True


def test_get_problem_data_run_identity(tmp_path):
    logger = write_experiment(
        tmp_path / "exp",
        [("M", "a", "P", 0, [1.0, 2.0]), ("M", "b", "P", 0, [3.0])],
    )
    df = logger.get_problem_data("P")
    assert {"llm_name", "run", "log_dir", "_id"} <= set(df.columns)
    assert sorted(df["llm_name"].unique()) == ["a", "b"]
    assert df["run"].nunique() == 2
    assert df.groupby("run")["_id"].apply(list).tolist() == [[0, 1], [0]]


def test_get_data_adds_llm_name_from_llm_record(tmp_path):
    root = tmp_path / "exp"
    os.makedirs(root)
    with open(root / "experimentlog.jsonl", "w") as f:
        f.write(
            json.dumps(
                {"method_name": "M", "problem_name": "P", "llm": {"model": "x/y:1"}}
            )
            + "\n"
        )
    df = ExperimentLogger(str(root), read=True).get_data()
    assert df["llm_name"].tolist() == ["x-y-1"]


def test_partial_trailing_jsonl_line_is_skipped(tmp_path):
    root = tmp_path / "exp"
    logger = write_experiment(root, [("M", "llm", "P", 0, [1.0, 2.0])])
    with open(root / "run-M-llm-P-0" / "log.jsonl", "a") as f:
        f.write('{"id": "s2", "fitness": 3.0, "code": "def')  # half written
    with open(root / "experimentlog.jsonl", "a") as f:
        f.write('{"method_name": "M", "problem_')
    df = logger.get_problem_data("P")
    assert df["fitness"].tolist() == [1.0, 2.0]
    assert len(logger.get_data()) == 1
    assert logger.get_methods_problems() == (["M"], ["P"])
    conv = convergence_dataframe(logger)
    assert conv["best_so_far"].tolist() == [1.0, 2.0]


def test_corrupt_jsonl_line_in_the_middle_is_reported(tmp_path):
    root = tmp_path / "exp"
    logger = write_experiment(root, [("M", "llm", "P", 0, [1.0])])
    path = root / "run-M-llm-P-0" / "log.jsonl"
    content = path.read_text()
    path.write_text("{broken\n" + content)
    with pytest.raises(ValueError, match="line 1"):
        logger.get_problem_data("P")


def test_include_running_reads_unfinished_runs(tmp_path):
    root = tmp_path / "exp"
    logger = write_experiment(root, [("M", "llm", "P", 0, [1.0, 2.0])], finished=False)
    assert logger.get_problem_data("P").empty
    df = logger.get_problem_data("P", include_running=True)
    assert df["fitness"].tolist() == [1.0, 2.0]
    conv = convergence_dataframe(logger, include_running=True)
    assert conv["best_so_far"].tolist() == [1.0, 2.0]


# ---------------------------------------------------------------------------
# webapp data functions
# ---------------------------------------------------------------------------


def test_convergence_dataframe_minimisation_and_failures(tmp_path):
    logger = write_experiment(
        tmp_path / "exp",
        [("M", "llm", "P", 0, [10.0, "inf", 8.0, 12.0])],
        minimisation={"P": True},
    )
    df = convergence_dataframe(logger)
    assert df["best_so_far"].tolist() == [10.0, 10.0, 8.0, 8.0]
    assert df["eval"].tolist() == [1, 2, 3, 4]
    assert df["minimisation"].all()
    final = final_best_dataframe(df)
    assert final["best_so_far"].tolist() == [8.0]


def test_convergence_dataframe_multiobjective(tmp_path):
    logger = write_experiment(
        tmp_path / "exp",
        [
            (
                "M",
                "llm",
                "P",
                0,
                [
                    {"f1": 1.0, "f2": 3.0},
                    {"f1": float("nan"), "f2": 1.0},
                    {"f1": 4.0, "f2": 4.0},
                ],
            )
        ],
    )
    df = convergence_dataframe(logger)
    assert df["best_so_far"].tolist() == [2.0, 2.0, 4.0]
    fig = plotly_convergence(df, aggregate=True)
    assert len(fig.data) == 3
    fig = plotly_convergence(df, aggregate=False)
    assert len(fig.data) == 1


def test_webapp_all_failed_run(tmp_path):
    logger = write_experiment(
        tmp_path / "exp",
        [
            ("M", "llm", "P", 0, ["-inf", "-inf"]),
            ("M", "llm", "P", 1, [1.0, "-inf"]),
        ],
        minimisation={"P": False},
    )
    df = convergence_dataframe(logger)
    final = final_best_dataframe(df).sort_values("seed")
    assert math.isnan(final["best_so_far"].iloc[0])
    assert final["best_so_far"].iloc[1] == 1.0
    plotly_convergence(df, aggregate=True)
    plotly_convergence(df, aggregate=False)
    top = top_solutions(logger.get_data(), "M", logger.get_minimisation)
    # failures last, never 0
    assert top["fitness_value"].isna().tolist() == [True, True]


def test_top_solutions_direction_and_failures(tmp_path):
    logger = write_experiment(
        tmp_path / "exp",
        [
            ("M", "llm", "P", 0, [5.0]),
            ("M", "llm", "P", 1, ["inf"]),
            ("M", "llm", "P", 2, [2.0]),
            ("M", "llm", "P", 3, [9.0]),
        ],
        minimisation={"P": True},
    )
    runs = logger.get_data()
    top = top_solutions(runs, "M", logger.get_minimisation, n=3)
    assert top["fitness_value"].tolist() == [2.0, 5.0, 9.0]
    top = top_solutions(runs, "M", None, n=4)  # maximisation
    vals = top["fitness_value"].tolist()
    assert vals[:3] == [9.0, 5.0, 2.0] and math.isnan(vals[3])


def test_top_solutions_multiobjective(tmp_path):
    logger = write_experiment(
        tmp_path / "exp",
        [
            ("M", "llm", "P", 0, [{"f1": 1.0, "f2": 1.0}]),
            ("M", "llm", "P", 1, [{"f1": 3.0, "f2": 5.0}]),
        ],
    )
    top = top_solutions(logger.get_data(), "M", logger.get_minimisation)
    assert top["fitness_value"].tolist() == [4.0, 1.0]


def test_webapp_two_llms_same_method_seed(tmp_path):
    logger = write_experiment(
        tmp_path / "exp",
        [("M", "a", "P", 0, [1.0, 2.0]), ("M", "b", "P", 0, [5.0, 3.0])],
    )
    df = convergence_dataframe(logger)
    assert df["run_label"].nunique() == 2
    assert set(df["method_label"]) == {"M (a)", "M (b)"}
    final = final_best_dataframe(df).sort_values("llm_name")
    assert final["best_so_far"].tolist() == [2.0, 5.0]
    fig = plotly_convergence(df, aggregate=True)
    assert len(fig.data) == 6  # two groups of (lower, upper, mean)
    label = df[df["llm_name"] == "b"]["run_label"].iloc[0]
    run_df = run_evaluations(logger, df, "P", label)
    assert run_df["fitness"].tolist() == [5.0, 3.0]


# ---------------------------------------------------------------------------
# plots
# ---------------------------------------------------------------------------


def _line_data(fig):
    ax = fig.axes[0]
    return {line.get_label(): line for line in ax.get_lines()}


def test_plot_convergence_uses_logged_direction_and_llms(tmp_path):
    logger = write_experiment(
        tmp_path / "exp",
        [
            ("M", "a", "P", 0, [10.0, "inf", 8.0, 12.0]),
            ("M", "b", "P", 0, [7.0, 9.0]),
        ],
        minimisation={"P": True},
    )
    fig = plot_convergence(logger, save=False, return_fig=True, show_std=False)
    lines = _line_data(fig)
    assert set(lines) == {"M (a)", "M (b)"}
    assert list(lines["M (a)"].get_xdata()) == [1, 2, 3, 4]
    assert list(lines["M (a)"].get_ydata()) == [10.0, 10.0, 8.0, 8.0]
    # shorter run carried forward
    assert list(lines["M (b)"].get_ydata()) == [7.0, 7.0]
    plt.close(fig)

    # explicit override: maximisation
    fig = plot_convergence(
        logger, save=False, return_fig=True, show_std=False, minimisation=False
    )
    assert list(_line_data(fig)["M (a)"].get_ydata()) == [10.0, 10.0, 10.0, 12.0]
    plt.close(fig)


def test_plot_convergence_separate_lines_is_one_based(tmp_path):
    logger = write_experiment(tmp_path / "exp", [("M", "llm", "P", 0, [1.0, 2.0, 3.0])])
    fig = plot_convergence(logger, save=False, return_fig=True, separate_lines=True)
    (line,) = fig.axes[0].get_lines()
    assert list(line.get_xdata()) == [1, 2, 3]
    plt.close(fig)


def test_fitness_table_logged_direction(tmp_path):
    logger = write_experiment(
        tmp_path / "exp",
        [("A", "llm", "P", s, [1.0 + 0.01 * s]) for s in range(4)]
        + [("B", "llm", "P", s, [5.0 + 0.01 * s]) for s in range(4)]
        + [("B", "llm", "P", 9, ["inf"])],
        minimisation={"P": True},
    )
    table = fitness_table(logger)
    assert table.loc["A", "P"].startswith("\\textbf{")  # lower is better
    assert "1 failed" in table.loc["B", "P"]
    table = fitness_table(logger, smaller_is_better=False)
    assert not table.loc["A", "P"].startswith("\\textbf{")
    assert table.loc["B", "P"].startswith("\\textbf{")


def _ceg_df():
    return pd.DataFrame(
        {
            "id": ["0", "1", "2"],
            "parent_ids": ["[]", '["0"]', '["1"]'],
            "fitness": [1.0, "-inf", float("nan")],
            "code": ["print('a')", "def f(): pass", "def g(): print('C')"],
        }
    )


def test_plot_code_evolution_graphs_default_features_and_close():
    plt.close("all")
    # plot_features=None used to fail with len(None); "-inf" strings crashed
    plot_code_evolution_graphs(_ceg_df(), save=False)
    # every figure it created is closed again
    assert plt.get_fignums() == []


def test_plot_code_evolution_graphs_tsne_fallback_without_pca(monkeypatch):
    import iohblade.plots as plots

    class BrokenTSNE:
        def __init__(self, *a, **k):
            pass

        def fit_transform(self, X):
            raise ValueError("too few samples")

    monkeypatch.setattr(plots, "TSNE", BrokenTSNE)
    plt.close("all")
    plot_code_evolution_graphs(_ceg_df(), plot_features=["tsne"], save=False)
    assert plt.get_fignums() == []


# ---------------------------------------------------------------------------
# behaviour metrics
# ---------------------------------------------------------------------------


def test_distance_to_best_so_far_uses_previous_best():
    df = pd.DataFrame({"raw_y": [1.0, 0.0], "x0": [0.0, 3.0]})
    # distance of evaluation 1 to the best of evaluation 0 is 3, not 0
    assert bm.average_distance_to_best_so_far(df) == pytest.approx(3.0)


def test_metrics_on_single_evaluation():
    df = pd.DataFrame({"raw_y": [1.0], "x0": [0.0]})
    assert bm.average_distance_to_best_so_far(df) == 0.0
    assert bm.last_improvement_fraction(df) == 0.0


def test_compute_behavior_metrics_passes_bounds(monkeypatch):
    seen = {}
    original = bm.avg_exploration_exploitation_chunked

    def spy(df, *args, **kwargs):
        seen["bounds"] = kwargs.get("bounds")
        return original(df, *args, **kwargs)

    monkeypatch.setattr(bm, "avg_exploration_exploitation_chunked", spy)
    df = pd.DataFrame(
        {"raw_y": [3.0, 2.0, 1.0], "x0": [0.0, 50.0, 100.0], "x1": [0.0, 1.0, 2.0]}
    )
    bounds = [(0.0, 1000.0), (0.0, 1000.0)]
    metrics = bm.compute_behavior_metrics(df, bounds=bounds, disp_samples=5)
    assert seen["bounds"] == bounds
    expected, _ = original(df, bounds=bounds)
    assert metrics["avg_exploration_pct"] == pytest.approx(expected)
    # with the default (-5, 5) bounds the spread would be clipped at 100%
    assert metrics["avg_exploration_pct"] < 50
