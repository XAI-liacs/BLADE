"""Analysis of the multi-objective benchmarks (#257, phase 2).

Runs are written with the real ExperimentLogger (``add_run``) from solutions
evaluated in-process, and read back with ``get_data()``.
"""

import re

import matplotlib
import pytest

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from iohblade import mo_analysis as mo  # noqa: E402
from iohblade.benchmarks.multiobjective import MultiObjectiveBenchmark  # noqa: E402
from iohblade.llm import Dummy_LLM  # noqa: E402
from iohblade.loggers import ExperimentLogger  # noqa: E402
from iohblade.solution import Solution  # noqa: E402

# Samples x1 uniformly with all other variables 0: the Pareto set of ZDT.
FRONT_SAMPLER = """
import numpy as np

class FrontSampler:
    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        X = np.zeros((self.budget, self.dim)) + {noise}
        X[:, 0] = np.random.uniform(0, 1, self.budget)
        func(X)
"""

SEEDS = range(4)


class _Method:
    def __init__(self, name):
        self.name = name

    def to_dict(self):
        return {"method_name": self.name}


def _write_runs(logger, problem, method_name, codes):
    llm = Dummy_LLM()
    for seed, (code, name) in zip(SEEDS, codes):
        np.random.seed(seed)
        solution = problem.evaluate(Solution(code=code, name=name))
        logger.add_run(_Method(method_name), problem, llm, solution, seed=seed)


@pytest.fixture(scope="module")
def logger(tmp_path_factory):
    path = tmp_path_factory.mktemp("mo") / "exp"
    log = ExperimentLogger(str(path))
    two = MultiObjectiveBenchmark(
        ["zdt1", "zdt3"], ["zdt2"], budget_factor=20, runs=2, name="MO-2obj"
    )
    random_code = re.search(r"```python\n(.*?)\n```", two.example_prompt, re.S).group(1)
    _write_runs(log, two, "RandomSearch", [(random_code, "RandomSearch")] * len(SEEDS))
    _write_runs(
        log,
        two,
        "FrontSampler",
        [(FRONT_SAMPLER.format(noise=0.001 * s), "FrontSampler") for s in SEEDS],
    )
    three = MultiObjectiveBenchmark(
        [{"name": "dtlz2", "n_obj": 3}],
        ["zdt2"],
        budget_factor=20,
        runs=1,
        name="MO-3obj",
    )
    _write_runs(log, three, "RandomSearch", [(random_code, "RandomSearch")] * 2)
    return ExperimentLogger(str(path), read=True)


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def test_mo_runs_reads_the_logged_metadata(logger):
    runs = mo.mo_runs(logger, "MO-2obj")
    # 2 methods x 4 BLADE runs x 2 functions x 2 evaluation runs
    assert len(runs) == 32
    assert set(runs["function"]) == {"zdt1", "zdt3"}
    assert set(runs["method_label"]) == {"RandomSearch", "FrontSampler"}
    row = runs.iloc[0]
    assert len(row["curve"]["hv"]) == 50 and len(row["curve"]["fraction"]) == 50
    assert row["spec"]["name"] in {"zdt1", "zdt3"}
    assert row["budget"] == 20 * 30
    assert mo.mo_problems(logger) == ["MO-2obj", "MO-3obj"]


def test_summary_finds_the_better_method(logger):
    area = mo.mo_summary(logger, "MO-2obj", "area")
    best = set(area["best"])
    assert best == {"FrontSampler"}
    worse = area[area["method_label"] == "RandomSearch"]
    assert worse["significantly_worse"].all()
    assert (worse["p_value"] < 0.05).all()
    igd = mo.mo_summary(logger, "MO-2obj", "igd_plus")  # lower is better
    assert set(igd["best"]) == {"FrontSampler"}
    table = mo.mo_summary_table(logger, "MO-2obj", "final_hv")
    assert list(table.columns) == ["zdt1", "zdt3"]
    assert table.loc["FrontSampler", "zdt1"].endswith("*")
    assert "±" in table.loc["RandomSearch", "zdt1"]
    with pytest.raises(ValueError, match="metric must be one of"):
        mo.mo_summary(logger, "MO-2obj", "speed")


def test_front_sampler_is_on_the_reference_front(logger):
    runs = mo.mo_runs(logger, "MO-2obj")
    sampler = runs[
        (runs["method_label"] == "FrontSampler") & (runs["function"] == "zdt1")
    ]
    assert sampler["igd_plus"].max() < 0.05
    assert sampler["final_hv"].min() > 0.9


def test_plots_return_figures(logger):
    for figure in (
        mo.plot_mo_anytime(logger, "MO-2obj"),
        mo.plot_mo_anytime(logger, "MO-2obj", "igd_plus", functions=["zdt1"]),
        mo.plot_mo_fronts(logger, "MO-2obj"),
        mo.plot_mo_fronts(logger, "MO-3obj"),
        mo.plot_mo_attainment(logger, "MO-2obj", "zdt1", percentiles=(25, 50, 75)),
        mo.plot_mo_eaf_difference(
            logger, "MO-2obj", "zdt1", "FrontSampler", "RandomSearch"
        ),
    ):
        assert figure.axes


def test_plot_errors(logger):
    with pytest.raises(ValueError, match="indicator must be"):
        mo.plot_mo_anytime(logger, "MO-2obj", "eps")
    with pytest.raises(ValueError, match="No data for function"):
        mo.plot_mo_fronts(logger, "MO-2obj", functions=["zdt9"])
    with pytest.raises(ValueError, match="need 2 objectives"):
        mo.plot_mo_attainment(logger, "MO-3obj", "dtlz2(n_obj=3)")
    with pytest.raises(ValueError, match="No multi-objective data"):
        mo.plot_mo_fronts(logger, "unknown")


def test_no_multi_objective_data(tmp_path):
    empty = ExperimentLogger(str(tmp_path / "empty"))
    assert mo.mo_runs(empty).empty
    assert mo.mo_problems(empty) == []
    assert mo.mo_summary(empty, "x").empty


def test_webapp_section_runs(logger):
    from iohblade.webapp import multi_objective_section

    multi_objective_section(logger)  # streamlit bare mode: renders nothing
