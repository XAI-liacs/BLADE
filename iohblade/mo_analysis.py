"""Analysis of multi-objective optimiser design (``multiobjective`` problems).

The multi-objective benchmarks (:mod:`iohblade.benchmarks.multiobjective`)
store, for every evaluation run of a candidate, its normalised hypervolume
and IGD+ curves over the budget and its final non-dominated front (in the
normalised objective space). This module reads them from the best solution
of every BLADE run (``ExperimentLogger.get_data()``) and offers:

* :func:`mo_runs`: one row per (method, BLADE run, function, evaluation run);
* :func:`mo_summary` / :func:`mo_summary_table`: mean, standard deviation and
  significance per method and function, for the area under the hypervolume
  curve, the final hypervolume or IGD+;
* :func:`plot_mo_anytime`: hypervolume or IGD+ over the budget;
* :func:`plot_mo_fronts`: the methods' fronts against the reference front
  (2D, 3D, or parallel coordinates for more objectives);
* :func:`plot_mo_attainment` and :func:`plot_mo_eaf_difference`: empirical
  attainment surfaces and EAF differences (2 objectives, via moocore).

All plots use the normalised objective space (ideal point 0, nadir point 1
of the reference front; hypervolume reference point 1.1).
"""

from __future__ import annotations

import math
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .plots import add_method_labels

#: Metrics in ``performance_data`` and whether higher is better.
METRICS: dict[str, bool] = {
    "area": True,
    "final_hv": True,
    "igd_plus": False,
    "n_nondominated": True,
}

_LABELS = {
    "area": "area under HV curve",
    "final_hv": "final normalised HV",
    "igd_plus": "IGD+",
    "n_nondominated": "non-dominated points",
    "hv": "normalised hypervolume",
}


# ---------------------------------------------------------------- data


def _solution_metadata(solution: Any) -> dict:
    if isinstance(solution, dict):
        return solution.get("metadata") or {}
    return getattr(solution, "metadata", None) or {}


def _is_mo_run(record: Any) -> bool:
    return (
        isinstance(record, dict)
        and "curve" in record
        and "front" in record
        and "function" in record
    )


def mo_runs(logger, problem_name: str | None = None) -> pd.DataFrame:
    """Per-run results of the multi-objective benchmarks.

    Reads ``performance_data`` from the best solution of every BLADE run.

    Args:
        logger: An :class:`~iohblade.loggers.ExperimentLogger` (or anything
            with ``get_data()``).
        problem_name: Only this problem (default: all problems with
            multi-objective data).

    Returns:
        pd.DataFrame: One row per (method, BLADE run, function, evaluation
        run) with ``problem_name``, ``method_label``, ``seed`` (of the BLADE
        run), ``eval_seed``, ``function``, ``spec``, ``budget``,
        ``evaluations``, the :data:`METRICS`, ``curve`` and ``front``. Empty
        if there is no multi-objective data.
    """
    data = logger.get_data()
    if data is None or data.empty or "solution" not in data.columns:
        return pd.DataFrame()
    data = data.copy()
    if problem_name is not None:
        data = data[data["problem_name"] == problem_name]
    if data.empty:
        return pd.DataFrame()
    add_method_labels(data)
    rows = []
    for _, run in data.iterrows():
        records = _solution_metadata(run["solution"]).get("performance_data") or []
        for record in records:
            if not _is_mo_run(record):
                continue
            rows.append(
                {
                    "problem_name": run["problem_name"],
                    "method_label": run["method_label"],
                    "seed": run.get("seed"),
                    "eval_seed": record.get("seed"),
                    "function": record["function"],
                    "spec": record.get("spec"),
                    "budget": record.get("budget"),
                    "evaluations": record.get("evaluations"),
                    **{m: record.get(m, math.nan) for m in METRICS},
                    "curve": record["curve"],
                    "front": record["front"],
                }
            )
    return pd.DataFrame(rows)


def mo_problems(logger) -> list[str]:
    """Names of the problems with multi-objective data in the log."""
    runs = mo_runs(logger)
    if runs.empty:
        return []
    return sorted(runs["problem_name"].unique())


def _per_blade_run(runs: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Mean of ``metric`` over the evaluation runs of each BLADE run."""
    return (
        runs.groupby(["method_label", "function", "seed"], dropna=False)[metric]
        .mean()
        .reset_index()
    )


# ---------------------------------------------------------------- tables


def mo_summary(
    logger, problem_name: str, metric: str = "area", alpha: float = 0.05
) -> pd.DataFrame:
    """Mean, standard deviation and significance per method and function.

    The metric is first averaged over the evaluation runs of each BLADE run;
    the statistics are over the BLADE runs (seeds). Each method is compared
    with the best method of the function (two-sided Mann-Whitney U test).

    Args:
        logger: The experiment logger.
        problem_name: The problem.
        metric: One of :data:`METRICS`.
        alpha: Significance level for ``significantly_worse``.

    Returns:
        pd.DataFrame: Columns ``method_label``, ``function``, ``mean``,
        ``std``, ``n``, ``best`` (the best method of the function),
        ``p_value`` (against the best; NaN for the best itself) and
        ``significantly_worse``.
    """
    if metric not in METRICS:
        raise ValueError(f"metric must be one of {sorted(METRICS)} (got {metric!r}).")
    from scipy.stats import mannwhitneyu

    runs = mo_runs(logger, problem_name)
    if runs.empty:
        return pd.DataFrame()
    per_run = _per_blade_run(runs, metric)
    higher = METRICS[metric]
    rows = []
    for function, group in per_run.groupby("function", sort=False):
        stats = group.groupby("method_label")[metric].agg(["mean", "std", "count"])
        best = stats["mean"].idxmax() if higher else stats["mean"].idxmin()
        best_values = group.loc[group["method_label"] == best, metric].dropna()
        for method, row in stats.iterrows():
            p_value = math.nan
            if method != best:
                values = group.loc[group["method_label"] == method, metric].dropna()
                if (
                    len(values)
                    and len(best_values)
                    and not (
                        np.all(values.values == best_values.values[0])
                        and np.all(best_values.values == best_values.values[0])
                    )
                ):
                    p_value = float(
                        mannwhitneyu(
                            values, best_values, alternative="two-sided"
                        ).pvalue
                    )
            rows.append(
                {
                    "method_label": method,
                    "function": function,
                    "mean": float(row["mean"]),
                    "std": float(row["std"]) if row["count"] > 1 else math.nan,
                    "n": int(row["count"]),
                    "best": best,
                    "p_value": p_value,
                    "significantly_worse": bool(p_value < alpha),
                }
            )
    return pd.DataFrame(rows)


def mo_summary_table(
    logger, problem_name: str, metric: str = "area", alpha: float = 0.05
) -> pd.DataFrame:
    """:func:`mo_summary` as a table: methods x functions, ``mean ± std``.

    The best method of each function is marked with ``*``; methods that are
    not significantly worse than it (``p >= alpha``) with ``(=)``.
    """
    summary = mo_summary(logger, problem_name, metric, alpha)
    if summary.empty:
        return summary
    cells = {}
    for _, row in summary.iterrows():
        text = f"{row['mean']:.3f}"
        if not math.isnan(row["std"]):
            text += f" ± {row['std']:.3f}"
        if row["method_label"] == row["best"]:
            text += " *"
        elif not row["significantly_worse"]:
            text += " (=)"
        cells[(row["method_label"], row["function"])] = text
    methods = list(dict.fromkeys(summary["method_label"]))
    functions = list(dict.fromkeys(summary["function"]))
    table = pd.DataFrame(
        [[cells.get((m, f), "") for f in functions] for m in methods],
        index=pd.Index(methods, name="method"),
        columns=functions,
    )
    return table


# ---------------------------------------------------------------- plots


def _functions(runs: pd.DataFrame, functions: Iterable[str] | None) -> list[str]:
    available = list(dict.fromkeys(runs["function"]))
    if functions is None:
        return available
    missing = [f for f in functions if f not in available]
    if missing:
        raise ValueError(f"No data for function(s) {missing}; available: {available}.")
    return list(functions)


def _grid(n: int, projection: str | None = None, size: float = 4.0):
    import matplotlib.pyplot as plt

    cols = min(n, 3)
    rows = math.ceil(n / cols)
    fig = plt.figure(figsize=(size * cols, size * rows * 0.85))
    axes = [fig.add_subplot(rows, cols, i + 1, projection=projection) for i in range(n)]
    return fig, axes


def _colours(methods: list[str]) -> dict[str, Any]:
    import matplotlib.pyplot as plt

    cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", ["C0"])
    return {m: cycle[i % len(cycle)] for i, m in enumerate(methods)}


def plot_mo_anytime(
    logger,
    problem_name: str,
    indicator: str = "hv",
    functions: Iterable[str] | None = None,
):
    """Hypervolume (or IGD+) over the budget, per function.

    The median over all runs (BLADE runs x evaluation runs) of each method's
    best solutions, with the interquartile range shaded. The x-axis is the
    fraction of the budget used.

    Args:
        logger: The experiment logger.
        problem_name: The problem.
        indicator: ``"hv"`` (normalised hypervolume; negative while no point
            reached the reference box) or ``"igd_plus"``.
        functions: The functions to show (default: all).

    Returns:
        matplotlib.figure.Figure: The figure.
    """
    if indicator not in ("hv", "igd_plus"):
        raise ValueError(f"indicator must be 'hv' or 'igd_plus' (got {indicator!r}).")
    runs = mo_runs(logger, problem_name)
    if runs.empty:
        raise ValueError(f"No multi-objective data for problem {problem_name!r}.")
    functions = _functions(runs, functions)
    methods = list(dict.fromkeys(runs["method_label"]))
    colours = _colours(methods)
    fig, axes = _grid(len(functions))
    for ax, function in zip(axes, functions):
        subset = runs[runs["function"] == function]
        for method in methods:
            curves = [
                c
                for c in subset.loc[subset["method_label"] == method, "curve"]
                if c and c.get(indicator)
            ]
            if not curves:
                continue
            x = np.asarray(curves[0]["fraction"], dtype=float)
            values = np.array(
                [np.asarray(c[indicator], dtype=float)[: len(x)] for c in curves]
            )
            median = np.median(values, axis=0)
            low, high = np.percentile(values, [25, 75], axis=0)
            ax.plot(x, median, label=method, color=colours[method])
            ax.fill_between(x, low, high, color=colours[method], alpha=0.2)
        ax.set_title(function)
        ax.set_xlabel("fraction of the budget")
        ax.set_ylabel(_LABELS[indicator])
        if indicator == "hv":
            ax.axhline(0.0, color="grey", lw=0.5)
            ax.set_ylim(top=1.02)
        else:
            ax.set_yscale("log")
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize="small")
    fig.suptitle(f"{problem_name}: {_LABELS[indicator]} over the budget")
    fig.tight_layout()
    return fig


def _reference_front(spec: dict | None) -> np.ndarray | None:
    """The normalised reference front of a function, or None."""
    if not spec:
        return None
    try:
        from .benchmarks.multiobjective.mo_benchmark import _pymoo_problem

        front = np.asarray(_pymoo_problem(dict(spec)).pareto_front(), dtype=float)
    except Exception:
        return None
    ideal, nadir = front.min(axis=0), front.max(axis=0)
    span = np.where(nadir - ideal > 0, nadir - ideal, 1.0)
    return (front - ideal) / span


def _method_front(fronts: Iterable) -> np.ndarray:
    """The non-dominated union of several fronts."""
    import moocore

    points = [np.asarray(f, dtype=float) for f in fronts if len(f)]
    if not points:
        return np.empty((0, 0))
    return moocore.filter_dominated(np.vstack(points))


def plot_mo_fronts(
    logger,
    problem_name: str,
    functions: Iterable[str] | None = None,
    limit: float = 1.5,
):
    """The methods' Pareto fronts against the reference front, per function.

    For each method, the non-dominated union of the final fronts of its best
    solutions (all BLADE runs and evaluation runs). Two objectives: a
    scatter plot; three: a 3D scatter plot; more: parallel coordinates.

    Args:
        logger: The experiment logger.
        problem_name: The problem.
        functions: The functions to show (default: all).
        limit: Axis limit in the normalised space (points beyond it are not
            shown); the hypervolume reference point is at 1.1.

    Returns:
        matplotlib.figure.Figure: The figure.
    """
    runs = mo_runs(logger, problem_name)
    if runs.empty:
        raise ValueError(f"No multi-objective data for problem {problem_name!r}.")
    functions = _functions(runs, functions)
    methods = list(dict.fromkeys(runs["method_label"]))
    colours = _colours(methods)
    n_obj = {
        f: len(
            next((p for p in runs.loc[runs["function"] == f, "front"] if p), [[0, 0]])[
                0
            ]
        )
        for f in functions
    }
    if any(n > 3 for n in n_obj.values()):
        return _plot_parallel(runs, functions, methods, colours, problem_name)
    three_d = any(n == 3 for n in n_obj.values())
    fig, axes = _grid(
        len(functions),
        projection="3d" if three_d else None,
        size=6.0 if three_d else 4.0,
    )
    for ax, function in zip(axes, functions):
        subset = runs[runs["function"] == function]
        reference = _reference_front(next(iter(subset["spec"]), None))
        if reference is not None:
            if three_d:
                ax.scatter(
                    *reference.T, s=2, color="lightgrey", label="reference front"
                )
            else:
                order = np.argsort(reference[:, 0])
                ax.plot(
                    *reference[order].T,
                    ".",
                    ms=2,
                    color="grey",
                    label="reference front",
                )
        for method in methods:
            front = _method_front(subset.loc[subset["method_label"] == method, "front"])
            if not front.size:
                continue
            front = front[np.all(front <= limit, axis=1)]
            if three_d:
                ax.scatter(*front.T, s=10, color=colours[method], label=method)
            else:
                order = np.argsort(front[:, 0])
                ax.plot(*front[order].T, "o", ms=4, color=colours[method], label=method)
        ax.set_title(function)
        ax.set_xlabel("f1 (normalised)")
        ax.set_ylabel("f2 (normalised)")
        ax.set_xlim(-0.05, limit)
        ax.set_ylim(-0.05, limit)
        if three_d:
            ax.set_zlabel("f3 (normalised)")
            ax.set_zlim(-0.05, limit)
        else:
            ax.grid(alpha=0.3)
    axes[0].legend(fontsize="small")
    fig.suptitle(f"{problem_name}: Pareto fronts", fontsize="medium")
    fig.tight_layout()
    return fig


def _plot_parallel(runs, functions, methods, colours, problem_name):
    fig, axes = _grid(len(functions))
    for ax, function in zip(axes, functions):
        subset = runs[runs["function"] == function]
        for method in methods:
            front = _method_front(subset.loc[subset["method_label"] == method, "front"])
            for k, point in enumerate(front):
                ax.plot(
                    range(len(point)),
                    point,
                    color=colours[method],
                    alpha=0.4,
                    label=method if k == 0 else None,
                )
        ax.set_title(function)
        ax.set_xlabel("objective")
        ax.set_ylabel("normalised value")
    axes[0].legend(fontsize="small")
    fig.suptitle(f"{problem_name}: Pareto fronts (parallel coordinates)")
    fig.tight_layout()
    return fig


def _eaf_input(subset: pd.DataFrame, method: str) -> np.ndarray:
    """Points of a method's fronts with a set index column (moocore format)."""
    sets = []
    for k, front in enumerate(subset.loc[subset["method_label"] == method, "front"]):
        front = np.asarray(front, dtype=float)
        if front.size:
            sets.append(np.column_stack([front, np.full(len(front), k + 1)]))
    if not sets:
        return np.empty((0, 3))
    return np.vstack(sets)


def _require_two_objectives(subset: pd.DataFrame, function: str) -> None:
    for front in subset["front"]:
        if front and len(front[0]) != 2:
            raise ValueError(
                f"Attainment plots need 2 objectives; {function} has {len(front[0])}."
            )


def plot_mo_attainment(
    logger,
    problem_name: str,
    function: str,
    methods: Iterable[str] | None = None,
    percentiles: Iterable[float] = (50,),
    limit: float = 1.5,
):
    """Empirical attainment surfaces of the methods for one function.

    Each run's final front is one set; the ``percentiles`` attainment
    surfaces (default: the median) of every method are drawn as steps.
    Two objectives only.

    Returns:
        matplotlib.figure.Figure: The figure.
    """
    import matplotlib.pyplot as plt
    import moocore

    runs = mo_runs(logger, problem_name)
    if runs.empty:
        raise ValueError(f"No multi-objective data for problem {problem_name!r}.")
    _functions(runs, [function])
    subset = runs[runs["function"] == function]
    _require_two_objectives(subset, function)
    methods = list(methods or dict.fromkeys(subset["method_label"]))
    colours = _colours(methods)
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    reference = _reference_front(next(iter(subset["spec"]), None))
    if reference is not None:
        order = np.argsort(reference[:, 0])
        ax.plot(*reference[order].T, ".", ms=2, color="grey", label="reference front")
    styles = ["-", "--", ":", "-."]
    for method in methods:
        data = _eaf_input(subset, method)
        if not len(data):
            continue
        eaf = moocore.eaf(data[:, :2], data[:, 2], percentiles=list(percentiles))
        for i, p in enumerate(percentiles):
            surface = eaf[eaf[:, -1] == p][:, :2]
            surface = surface[np.argsort(surface[:, 0])]
            ax.step(
                surface[:, 0],
                surface[:, 1],
                where="post",
                ls=styles[i % len(styles)],
                color=colours[method],
                label=f"{method} ({p:g}%)",
            )
    ax.set_xlim(-0.05, limit)
    ax.set_ylim(-0.05, limit)
    ax.set_xlabel("f1 (normalised)")
    ax.set_ylabel("f2 (normalised)")
    ax.set_title(f"{problem_name}, {function}: attainment surfaces")
    ax.grid(alpha=0.3)
    ax.legend(fontsize="small")
    fig.tight_layout()
    return fig


def plot_mo_eaf_difference(
    logger,
    problem_name: str,
    function: str,
    method_a: str,
    method_b: str,
    intervals: int = 5,
    limit: float = 1.5,
):
    """Where one method attains the objective space more often than another.

    Left: regions attained more often by ``method_a``; right: by
    ``method_b`` (darker = larger difference in attainment probability).
    Two objectives only (moocore ``eafdiff``).

    Returns:
        matplotlib.figure.Figure: The figure.
    """
    import matplotlib.pyplot as plt
    import moocore
    from matplotlib.patches import Rectangle

    runs = mo_runs(logger, problem_name)
    if runs.empty:
        raise ValueError(f"No multi-objective data for problem {problem_name!r}.")
    _functions(runs, [function])
    subset = runs[runs["function"] == function]
    _require_two_objectives(subset, function)
    a, b = _eaf_input(subset, method_a), _eaf_input(subset, method_b)
    if not len(a) or not len(b):
        raise ValueError(f"No fronts for {method_a!r} or {method_b!r} on {function}.")
    rectangles = moocore.eafdiff(a, b, intervals=intervals, rectangles=True)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharey=True)
    for ax, sign, method in ((axes[0], 1, method_a), (axes[1], -1, method_b)):
        for x1, y1, x2, y2, diff in np.asarray(rectangles):
            if sign * diff <= 0:
                continue
            x2, y2 = min(x2, limit), min(y2, limit)
            if x2 <= x1 or y2 <= y1:
                continue
            ax.add_patch(
                Rectangle(
                    (x1, y1),
                    x2 - x1,
                    y2 - y1,
                    color="tab:blue" if sign > 0 else "tab:red",
                    alpha=min(1.0, abs(diff) / intervals),
                    lw=0,
                )
            )
        for m, colour in ((method_a, "black"), (method_b, "dimgrey")):
            data = _eaf_input(subset, m)
            median = moocore.eaf(data[:, :2], data[:, 2], percentiles=[50])[:, :2]
            median = median[np.argsort(median[:, 0])]
            ax.step(
                *median.T,
                where="post",
                color=colour,
                lw=1,
                ls="-" if m == method else ":",
            )
        ax.set_xlim(-0.05, limit)
        ax.set_ylim(-0.05, limit)
        ax.set_xlabel("f1 (normalised)")
        ax.set_title(f"attained more often by {method}")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("f2 (normalised)")
    fig.suptitle(f"{problem_name}, {function}: EAF difference")
    fig.tight_layout()
    return fig
