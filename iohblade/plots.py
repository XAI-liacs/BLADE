import ast
import copy
import difflib
import os
from collections import Counter

import jsonlines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import seaborn as sns
from scipy.stats import ttest_ind
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

from .llm_costs import estimate_tokens
from .loggers import ExperimentLogger
from .misc.ast import analyse_complexity, process_code


def plot_speedup(
    logger: ExperimentLogger,
    method_fast: str,
    method_slow: str,
    aggregation: str = "mean",
    variance_aggregation: str = "std",
    budget: int = 100,
    save: bool = True,
    return_fig: bool = False,
    minimisation: bool | None = None,
):
    """
    Plots speed-up of method_fast over method_slow.

    Speed-up definition:
    If method_fast reaches fitness x at n evaluations
    and method_slow reaches fitness x at m evaluations,
    speed-up = m / n.

    Args:
        logger (ExperimentLogger)
        method_fast (str): numerator algorithm
        method_slow (str): denominator algorithm
        aggregation (str): mean / median
        variance_aggregation (str): std / sem
        budget (int): max evaluations
        save (bool)
        return_fig (bool)
        minimisation (bool, optional): Whether lower fitness is better.
            Defaults to the direction logged for each problem.
    """

    methods, problems = logger.get_methods_problems()
    if method_fast not in methods or method_slow not in methods:
        raise ValueError("Both methods must exist in the logger.")

    fig, axes = plt.subplots(
        figsize=(8, 4 * len(problems)), nrows=len(problems), ncols=1
    )

    if len(problems) == 1:
        axes = [axes]

    for ax, problem in zip(axes, problems):
        data = logger.get_problem_data(problem_name=problem).drop(
            columns=["code"], errors="ignore"
        )
        minimise = _resolve_minimisation(logger, problem, minimisation)
        curves = best_so_far(data, minimisation=minimise)

        def compute_summary(method):
            return aggregate_best_so_far(
                curves[curves["method_name"] == method],
                aggregation,
                variance_aggregation,
            ).rename(columns={"eval": "_id"})

        fast = compute_summary(method_fast)
        slow = compute_summary(method_slow)

        # Restrict to budget
        fast = fast[fast["_id"] <= budget]
        slow = slow[slow["_id"] <= budget]

        speedups = []

        slow_ids = slow["_id"].values
        slow_fit = slow[aggregation].values

        for n, f in zip(fast["_id"], fast[aggregation]):
            # Find earliest m where slow reaches f (at least as good)
            if np.isnan(f):
                speedups.append(np.nan)
                continue
            reached = slow_fit <= f if minimise else slow_fit >= f
            idx = np.where(reached)[0]
            if len(idx) == 0:
                speedups.append(np.nan)
            else:
                m = slow_ids[idx[0]]
                speedups.append(m / n)

        ax.plot(fast["_id"], speedups, label=f"{method_fast} vs {method_slow}")
        ax.axhline(1.0, linestyle="--", color="gray", alpha=0.6)

        ax.set_xlabel("Number of Evaluations")
        ax.set_ylabel("Speed-up")
        ax.set_xlim(1, budget)
        ax.set_title(problem)
        ax.grid(True)
        ax.legend()

    plt.tight_layout()

    if save:
        fig.savefig(f"{logger.dirname}/speedup_{method_fast}_vs_{method_slow}.png")
    elif not return_fig:
        plt.show()

    if return_fig:
        return fig

    plt.close()


RUN_COLUMNS = ("method_name", "llm_name", "problem_name", "seed")


def _get_scalar_fitness_column(data: pd.DataFrame) -> str:
    """Return the column name to use for scalar fitness comparisons.

    For multi-objective problems ``get_problem_data`` adds a ``fitness_scalar``
    column containing the mean of all objectives.  For single-objective problems
    the plain ``fitness`` column is used directly.

    Args:
        data (pd.DataFrame): DataFrame returned by ``ExperimentLogger.get_problem_data``.

    Returns:
        str: ``"fitness_scalar"`` when available, otherwise ``"fitness"``.
    """
    return "fitness_scalar" if "fitness_scalar" in data.columns else "fitness"


def _fitness_value(value) -> float:
    """A fitness value as a float, ``NaN`` for a failed evaluation.

    Failed evaluations score ``inf`` / ``-inf`` (the worst value for the
    problem's direction) or ``NaN``; serialisation can turn these into
    strings such as ``"-inf"``. Multi-objective fitness dicts are reduced to
    their scalar summary (see :meth:`ExperimentLogger._fitness_to_scalar`).

    Args:
        value: A float, int, numeric string or (serialised) ``Fitness``.

    Returns:
        float: The finite fitness value, or ``NaN`` for a failure or a value
        that is not a number.
    """
    if isinstance(value, dict) or hasattr(value, "to_dict"):
        if not isinstance(value, dict):
            value = value.to_dict()
        try:
            value = ExperimentLogger._fitness_to_scalar(value)
        except Exception:
            return np.nan
    elif isinstance(value, bool):
        return np.nan
    try:
        value = float(value)
    except (TypeError, ValueError):
        return np.nan
    return value if np.isfinite(value) else np.nan


def clean_fitness(values) -> pd.Series:
    """Convert fitness values to floats with failures (±inf, NaN, junk) as NaN.

    Args:
        values (pd.Series | list): Fitness values.

    Returns:
        pd.Series: Float series of the same index, ``NaN`` for failures.
    """
    if not isinstance(values, pd.Series):
        values = pd.Series(list(values))
    return values.apply(_fitness_value).astype(float)


def run_columns(data: pd.DataFrame) -> list[str]:
    """The columns that identify a run in a DataFrame of evaluations.

    ``run`` (the run folder, see ``ExperimentLogger.get_problem_data``) when
    present; otherwise the available columns of method, LLM, problem and seed.
    """
    if "run" in data.columns:
        return ["run"]
    return [c for c in RUN_COLUMNS if c in data.columns]


def _resolve_minimisation(logger, problem, minimisation=None) -> bool:
    """The optimisation direction: the override if given, else as logged."""
    if minimisation is not None:
        return bool(minimisation)
    get_min = getattr(logger, "get_minimisation", None)
    if callable(get_min):
        try:
            return bool(get_min(problem))
        except Exception:
            return False
    return False


def add_method_labels(data: pd.DataFrame, column: str = "method_label") -> pd.DataFrame:
    """Add a label per method that names the LLM when a method used several.

    The label is ``method`` when a method name was used with one LLM, and
    ``method (llm)`` when the same method name was run with several LLMs.

    Args:
        data (pd.DataFrame): Rows with ``method_name`` (and ``llm_name``).
        column (str): The name of the column to add.

    Returns:
        pd.DataFrame: ``data`` with the label column (modified in place).
    """
    if "llm_name" not in data.columns or data.empty:
        data[column] = data["method_name"].astype(str)
        return data
    llms = data["llm_name"].fillna("?").astype(str)
    n_llms = llms.groupby(data["method_name"]).transform("nunique")
    data[column] = np.where(
        n_llms > 1,
        data["method_name"].astype(str) + " (" + llms + ")",
        data["method_name"].astype(str),
    )
    return data


def best_so_far(
    data: pd.DataFrame,
    minimisation: bool = False,
    fitness_col: str | None = None,
    run_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Per-run best-so-far curves of the evaluated solutions.

    Failed evaluations (±inf, NaN, strings such as ``"-inf"``) are treated
    as missing: they never improve the curve and are never shown as 0. The
    curve is ``NaN`` until the first valid evaluation of a run.

    Args:
        data (pd.DataFrame): One row per evaluation with a fitness column,
            the run-identity columns and ``_id`` (evaluation order).
        minimisation (bool): ``True`` when lower fitness is better (curve is
            a cumulative minimum), otherwise a cumulative maximum.
        fitness_col (str, optional): The fitness column; by default
            ``fitness_scalar`` if present, else ``fitness``.
        run_cols (list[str], optional): Columns identifying a run; by
            default :func:`run_columns`.

    Returns:
        pd.DataFrame: A copy of ``data`` sorted by run and ``_id``, with
        ``fitness_value`` (cleaned float fitness), ``best_so_far`` and
        ``eval`` (the 1-based evaluation index within the run).
    """
    if fitness_col is None:
        fitness_col = _get_scalar_fitness_column(data)
    if run_cols is None:
        run_cols = run_columns(data)
    df = data.copy()
    if df.empty:
        for col in ("fitness_value", "best_so_far", "eval"):
            df[col] = pd.Series(dtype=float)
        return df
    df["fitness_value"] = clean_fitness(df[fitness_col])
    order = list(run_cols) + (["_id"] if "_id" in df.columns else [])
    df = df.sort_values(by=order, kind="stable").reset_index(drop=True)
    if run_cols:
        groups = df.groupby(run_cols, sort=False, dropna=False)
        cum = (
            groups["fitness_value"].cummin()
            if minimisation
            else (groups["fitness_value"].cummax())
        )
        df["best_so_far"] = cum.groupby(
            [df[c] for c in run_cols], sort=False, dropna=False
        ).ffill()
        df["eval"] = groups.cumcount() + 1
    else:
        vals = df["fitness_value"]
        cum = vals.cummin() if minimisation else vals.cummax()
        df["best_so_far"] = cum.ffill()
        df["eval"] = np.arange(1, len(df) + 1)
    return df


def aggregate_best_so_far(
    curves: pd.DataFrame,
    aggregation: str = "mean",
    variance_aggregation: str = "std",
    run_cols: list[str] | None = None,
    max_eval: int | None = None,
) -> pd.DataFrame:
    """Aggregate best-so-far curves of several runs per evaluation.

    Each run's curve is carried forward (its last best value) to the common
    maximum evaluation count, so runs that stopped early still count and the
    mean of best-so-far curves stays monotone. Before a run's first valid
    evaluation its curve is missing and it is left out of the aggregate.

    Args:
        curves (pd.DataFrame): Output of :func:`best_so_far` (for one group).
        aggregation (str): Aggregation over runs (``mean``, ``median``, ...).
        variance_aggregation (str): Spread over runs (``std``, ``sem``, ...).
        run_cols (list[str], optional): Columns identifying a run.
        max_eval (int, optional): Extend the curves to this evaluation count
            (default: the longest run).

    Returns:
        pd.DataFrame: Columns ``eval``, ``aggregation``,
        ``variance_aggregation`` and ``n_runs`` (runs with a valid value).
    """
    if run_cols is None:
        run_cols = run_columns(curves)
    if curves.empty:
        return pd.DataFrame(columns=["eval", aggregation, variance_aggregation])
    df = curves.copy()
    if run_cols:
        df["_run_key"] = df.groupby(run_cols, sort=False, dropna=False).ngroup()
    else:
        df["_run_key"] = 0
    wide = df.pivot(index="eval", columns="_run_key", values="best_so_far")
    if max_eval is None:
        max_eval = int(df["eval"].max())
    wide = wide.reindex(range(1, max_eval + 1)).ffill()
    summary = wide.agg([aggregation, variance_aggregation], axis=1)
    summary["n_runs"] = wide.notna().sum(axis=1)
    summary.index.name = "eval"
    return summary.reset_index()


def final_best_per_run(
    curves: pd.DataFrame, run_cols: list[str] | None = None
) -> pd.DataFrame:
    """The last row of each run's best-so-far curve (its best fitness).

    Args:
        curves (pd.DataFrame): Output of :func:`best_so_far`.
        run_cols (list[str], optional): Columns identifying a run.

    Returns:
        pd.DataFrame: One row per run; ``best_so_far`` is ``NaN`` for a run
        in which every evaluation failed.
    """
    if run_cols is None:
        run_cols = run_columns(curves)
    if curves.empty:
        return curves.copy()
    ordered = curves.sort_values(by=list(run_cols) + ["eval"], kind="stable")
    if not run_cols:
        return ordered.tail(1).reset_index(drop=True)
    return (
        ordered.groupby(run_cols, sort=False, dropna=False)
        .tail(1)
        .reset_index(drop=True)
    )


def _solution_fitness(df: pd.DataFrame) -> pd.Series:
    """Cleaned final fitness per run of ``ExperimentLogger.get_data()``."""
    if "fitness" in df.columns:
        values = df["fitness"]
    elif "solution" in df.columns:
        values = df["solution"].apply(
            lambda sol: sol.get("fitness", np.nan) if isinstance(sol, dict) else np.nan
        )
    else:
        return pd.Series(np.nan, index=df.index)
    return clean_fitness(values).set_axis(df.index)


def plot_convergence(
    logger: ExperimentLogger,
    metric: str = "Fitness",
    aggregation: str = "mean",
    variance_aggregation: str = "std",
    methods: list = None,
    budget: int = 100,
    save: bool = True,
    return_fig: bool = False,
    separate_lines: bool = False,
    show_std: bool = True,
    minimisation: bool | None = None,
):
    """
    Plots the convergence of all methods for each problem from an experiment log.

    Each run's best-so-far fitness (cumulative maximum, or minimum for a
    minimisation problem) is plotted against the 1-based evaluation index.
    Failed evaluations never improve the curve. Runs are told apart by their
    run folder (or method, LLM and seed); a method used with several LLMs
    gets one line per LLM. For the aggregate, shorter runs are carried
    forward with their final best value.

    For multi-objective problems the scalar summary (mean of all objectives) is
    used as the convergence metric.  Use :func:`plot_pareto_front` to visualise
    the Pareto front directly.

    Args:
        logger (ExperimentLogger): The experiment logger object.
        metric (str, optional): The metric to show as y-axis label.
        aggregation (str, optional): The aggregation method to use ('mean', 'median', etc.).
        variance_aggregation (str, optional): The method to aggregate variance ('std', 'sem', etc.).
        methods (list, optional): List of method names to include. If None, includes all methods.
        budget (int, optional): The maximum number of evaluations to display on the x-axis.
        save (bool, optional): Whether to save or show the plot.
        return_fig (bool, optional): Whether to return the figure object.
        separate_lines (bool, optional): If True, plots each run using separate line.
        show_std (bool, optional): If True, shows standard deviation as shaded area.
        minimisation (bool, optional): Whether lower fitness is better.
            Defaults to the direction logged for each problem.
    """
    methods_to_use = methods
    methods, problems = logger.get_methods_problems()

    fig, axes = plt.subplots(
        figsize=(8, 6 * len(problems)), nrows=len(problems), ncols=1
    )
    problem_i = 0
    for problem in problems:
        data = logger.get_problem_data(problem_name=problem).drop(
            columns=["code"], errors="ignore"
        )  # for efficiency we drop code for now
        minimise = _resolve_minimisation(logger, problem, minimisation)
        curves = best_so_far(data, minimisation=minimise)
        add_method_labels(curves)
        run_cols = run_columns(curves)

        if methods_to_use is not None:
            curves = curves[curves["method_name"].isin(methods_to_use)]
        ax = axes[problem_i] if len(problems) > 1 else axes
        for label in pd.unique(curves["method_label"]):
            method_data = curves[curves["method_label"] == label]

            if separate_lines:
                for _, run_data in method_data.groupby(
                    run_cols, sort=False, dropna=False
                ):
                    seed = run_data["seed"].iloc[0] if "seed" in run_data else ""
                    ax.plot(
                        run_data["eval"],
                        run_data["best_so_far"],
                        label=f"{label} (Run {seed})",
                        alpha=0.5,
                    )
            else:
                summary = aggregate_best_so_far(
                    method_data, aggregation, variance_aggregation, run_cols
                )
                ax.plot(summary["eval"], summary[aggregation], label=label)

                # Plot the shaded error region
                if show_std:
                    ax.fill_between(
                        summary["eval"],
                        summary[aggregation] - summary[variance_aggregation],
                        summary[aggregation] + summary[variance_aggregation],
                        alpha=0.2,
                    )

        # Add labels and legend
        ax.set_xlabel("Number of Evaluations")
        if budget is not None:
            ax.set_xlim(1, budget)
        ax.set_ylabel(f"{aggregation} best {metric}")
        ax.legend(title="Algorithm")
        ax.grid(True)
        ax.set_title(problem)
        problem_i += 1

    plt.tight_layout()
    if save:
        fig.savefig(f"{logger.dirname}/convergence.png")
    elif not return_fig:
        plt.show()
    if return_fig:
        return fig
    plt.close()


def plot_pareto_front(
    logger: ExperimentLogger,
    objective_x: str,
    objective_y: str,
    methods: list = None,
    problems: list = None,
    save: bool = True,
    return_fig: bool = False,
):
    """
    Plot the Pareto front discovered by each method for each multi-objective problem.

    Each point in the scatter plot represents one evaluated solution.
    Non-dominated solutions (i.e. the empirical Pareto front) are highlighted.

    Args:
        logger (ExperimentLogger): The experiment logger object.
        objective_x (str): Name of the objective to plot on the x-axis.
        objective_y (str): Name of the objective to plot on the y-axis.
        methods (list, optional): Methods to include.  ``None`` includes all.
        problems (list, optional): Problems to include.  ``None`` includes all.
        save (bool, optional): If ``True``, saves the figure to the log directory.
        return_fig (bool, optional): If ``True``, returns the ``Figure`` object.

    Returns:
        matplotlib.figure.Figure | None
    """
    all_methods, all_problems = logger.get_methods_problems()
    problems_to_use = problems if problems is not None else all_problems
    methods_to_use = methods if methods is not None else all_methods

    fig, axes = plt.subplots(
        figsize=(7 * len(problems_to_use), 6),
        nrows=1,
        ncols=len(problems_to_use),
        squeeze=False,
    )

    for col_i, problem in enumerate(problems_to_use):
        ax = axes[0, col_i]
        data = logger.get_problem_data(problem_name=problem)

        if "fitness" not in data.columns:
            ax.set_title(f"{problem}\n(no fitness data)")
            continue

        # Expand multi-objective fitness dicts into per-objective columns
        def _extract_obj(fitness_val, key):
            if isinstance(fitness_val, dict):
                return fitness_val.get(key, float("nan"))
            return float("nan")

        data = data.copy()
        data[f"obj_{objective_x}"] = data["fitness"].apply(
            lambda v: _extract_obj(v, objective_x)
        )
        data[f"obj_{objective_y}"] = data["fitness"].apply(
            lambda v: _extract_obj(v, objective_y)
        )

        data = data.dropna(subset=[f"obj_{objective_x}", f"obj_{objective_y}"])

        unique_methods = [
            m for m in data["method_name"].unique() if m in methods_to_use
        ]
        colors = plt.cm.tab10(np.linspace(0, 1, max(len(unique_methods), 1)))

        for method, color in zip(unique_methods, colors):
            mdata = data[data["method_name"] == method]
            xs = mdata[f"obj_{objective_x}"].values
            ys = mdata[f"obj_{objective_y}"].values

            ax.scatter(xs, ys, color=color, alpha=0.35, s=20, label=f"{method} (all)")

            # Identify the non-dominated (Pareto) front for this method.
            # BLADE stores negated objectives (higher = better), so maximisation=True.
            pareto_mask = _pareto_front_mask(xs, ys, maximisation=True)
            if pareto_mask.any():
                px = xs[pareto_mask]
                py = ys[pareto_mask]
                order = np.argsort(px)
                ax.scatter(
                    px,
                    py,
                    color=color,
                    edgecolors="black",
                    linewidths=0.8,
                    s=60,
                    zorder=5,
                    label=f"{method} (Pareto)",
                )
                ax.step(
                    px[order],
                    py[order],
                    where="post",
                    color=color,
                    linewidth=1.5,
                    linestyle="--",
                    alpha=0.7,
                )

        ax.set_xlabel(objective_x)
        ax.set_ylabel(objective_y)
        ax.set_title(problem)
        ax.legend(fontsize="x-small")
        ax.grid(True)

    plt.tight_layout()
    if save:
        fname = f"{logger.dirname}/pareto_front_{objective_x}_vs_{objective_y}.png"
        fig.savefig(fname)
    elif not return_fig:
        plt.show()

    if return_fig:
        return fig
    plt.close()


def _pareto_front_mask(
    xs: np.ndarray, ys: np.ndarray, maximisation: bool = True
) -> np.ndarray:
    """Return a boolean mask selecting non-dominated points.

    Args:
        xs: x-coordinates of the candidate solutions.
        ys: y-coordinates of the candidate solutions.
        maximisation: When ``True`` (default), higher objective values are
            better — matching BLADE's convention of storing negated raw
            objectives so that the framework can always maximise.  Set to
            ``False`` for raw minimisation problems.

    Returns:
        np.ndarray: Boolean array, ``True`` where a point is non-dominated.
    """
    n = len(xs)
    is_pareto = np.ones(n, dtype=bool)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if maximisation:
                # j dominates i when j is better-or-equal on all axes and
                # strictly better on at least one (higher = better)
                if (
                    xs[j] >= xs[i]
                    and ys[j] >= ys[i]
                    and (xs[j] > xs[i] or ys[j] > ys[i])
                ):
                    is_pareto[i] = False
                    break
            else:
                # minimisation: lower = better
                if (
                    xs[j] <= xs[i]
                    and ys[j] <= ys[i]
                    and (xs[j] < xs[i] or ys[j] < ys[i])
                ):
                    is_pareto[i] = False
                    break
    return is_pareto


def plot_experiment_CEG(
    logger: ExperimentLogger,
    metric: str = "total_token_count",
    methods=None,
    markersize=None,
    budget: int = 100,
    save: bool = True,
    max_seeds=5,
    minimisation: bool | None = None,
):
    """
    Plot the Code evolution graphs for each run in an experiment, splitted by problem.

    Args:
        logger (ExperimentLogger): The experiment logger object.
        metric (str, optional): The metric to show as y-axis label (should be a statistic from AST / Complexity).
        methods (list, optional): An optional list of method names to plot, if None, all methods are included.
        markersize (int. optional): The size of the markers, if not set it will be dynamic based on the degree.
        budget (int): The number of evaluations to plot.
        save (bool, optional): Whether to save or show the plot.
        max_seeds (int, optional): The maximum number of runs to plot.
        minimisation (bool, optional): Whether lower fitness is better (used
            for the marker colours). Defaults to the logged direction.
    """
    methods_to_use = methods
    methods, problems = logger.get_methods_problems()

    problem_i = 0
    for problem in problems:
        data = logger.get_problem_data(problem_name=problem)
        minimise = _resolve_minimisation(logger, problem, minimisation)
        add_method_labels(data)

        # Get unique runs (seeds)
        seeds = data["seed"].unique()
        # sort the seeds
        seeds = sorted(seeds)
        num_seeds = min(len(seeds), max_seeds)
        # One row per method (and LLM, when a method used several)
        if methods_to_use is not None:
            data = data[data["method_name"].isin(methods_to_use)]
        methods = list(pd.unique(data["method_label"]))
        fig, axes = plt.subplots(
            figsize=(5 * num_seeds, 5 * len(methods)),
            nrows=len(methods),
            ncols=num_seeds,
            sharey=True,
            squeeze=False,
        )

        method_i = 0
        for method in methods:
            seed_i = 0
            for seed in seeds[:num_seeds]:
                ax = axes[method_i, seed_i]
                run_data = data[
                    (data["method_label"] == method) & (data["seed"] == seed)
                ].copy()
                if len(run_data) == 0:
                    continue
                plot_code_evolution_graphs(
                    run_data,
                    logger.dirname,
                    plot_features=[metric],
                    markersize=markersize,
                    save=False,
                    ax=ax,
                    minimisation=minimise,
                )
                ax.set_xlim([0, budget])
                ax.set_xticks(np.arange(0, budget + 1, 20))
                ax.set_xticklabels(np.arange(0, budget + 1, 20))
                ax.set_title(f"{method} run:{seed}")
                if seed_i > 0:
                    ax.set_ylabel(None)
                if method_i < len(methods) - 1:
                    ax.set_xlabel(None)
                seed_i += 1
            method_i += 1

        if save:
            plt.tight_layout()
            fig.savefig(f"{logger.dirname}/CEG_{problem}.png")
        else:
            plt.show()
        plt.close()


def _fitness_colour_scale(fitness, minimisation: bool = False) -> pd.Series:
    """Fitness scaled to [0, 1] with 1 the best; failures get 0 (worst)."""
    values = clean_fitness(fitness)
    if minimisation:
        values = -values
    if values.notna().any():
        lo, hi = values.min(), values.max()
        values = (values - lo) / (hi - lo) if hi > lo else values * 0 + 1.0
    return values.fillna(0.0)


# Helper function for key normalization to solve the mismatch issue
def normalize_key(key):
    """Converts any feature key to a standardized format (e.g., 'mean_complexity' -> 'mean complexity')."""
    key = str(key).lower()
    key = key.replace("_", " ")
    return " ".join(key.split()).strip()


# Parse parent ids
def safe_parse_parent_ids(x):
    if not isinstance(x, str):
        return x

    s = x.strip()

    # Case 1: Proper Python literal
    if s.startswith(("[", "(")) and s.endswith(("]", ")")):
        try:
            parsed = ast.literal_eval(s)
            return list(parsed) if isinstance(parsed, (list, tuple)) else [parsed]
        except (ValueError, SyntaxError):
            # Case 2: Fallback for things like "[awdawd213r]"
            inner = s[1:-1].strip()
            if inner:
                # split on commas, keep strings
                return [part.strip() for part in inner.split(",")]
            return []

    # Case 3: Single ID string
    return [s]


def plot_code_evolution_graphs(
    run_data,
    expfolder=None,
    plot_features=None,
    markersize=None,
    save=True,
    ax=None,
    minimisation: bool = False,
):
    """
    Plots optimization progress and relationships between successive solutions in an
    evolutionary run based on AST metrics. Can plot multiple features or a single feature on a provided axis.

    Args:
        run_data (pandas.DataFrame): DataFrame containing code and fitness values.
        expfolder (str, optional): Folder path where the plots are saved. If None, plots are shown.
        plot_features (list, optional): The features to plot. If None, plots multiple default features.
        markersize (int. optional): The size of the markers, if not set it will be dynamic based on the degree.
        save (bool): If True, saves the plots otherwise shows them.
        ax (matplotlib.axes.Axes, optional): The axis to plot on. If None, creates new plots.
        minimisation (bool): Whether lower fitness is better; the marker
            colour shows the best fitness brightest either way, failed
            evaluations get the colour of the worst.
    """
    if ax is not None and (plot_features is None or len(plot_features) > 1):
        raise ValueError(
            "If an axis is provided, the length of plot_features must be 1."
        )

    # Define default features if not provided
    if plot_features is None:
        plot_features = [
            "tsne",
            "pca",
            "total_complexity",
            "total_token_count",
            "total_parameter_count",
        ]

    data = run_data.copy().reset_index(drop=True)
    data["eval_index"] = data.index + 1
    # Failures (inf, NaN, "-inf"), multi-objective dicts -> scaled to [0, 1]
    data["fitness"] = _fitness_colour_scale(data["fitness"], minimisation)
    data.fillna(0, inplace=True)

    complexity_features = [
        "mean_complexity",
        "total_complexity",
        "mean_token_count",
        "total_token_count",
        "mean_parameter_count",
        "total_parameter_count",
    ]

    # Compute AST or complexity-based statistics
    if len(plot_features) == 1 and plot_features[0] in complexity_features:
        analyse_complexity
        df_stats = data["code"].apply(analyse_complexity).apply(pd.Series)
    else:
        if "ast_features" in data.columns:
            df_stats = data["ast_features"].apply(pd.Series)
        elif "metadata" in data.columns:
            ast_series = data["metadata"].apply(
                lambda m: (m.get("ast_features") if isinstance(m, dict) else None)
            )

            # Only use metadata-derived features if at least one row has a non-empty dict
            has_any = ast_series.apply(
                lambda d: isinstance(d, dict) and len(d) > 0
            ).any()

            if has_any:
                df_stats = ast_series.apply(
                    lambda d: d if isinstance(d, dict) else {}
                ).apply(pd.Series)
            else:
                df_stats = data["code"].apply(process_code).apply(pd.Series)
        else:
            df_stats = data["code"].apply(process_code).apply(pd.Series)
    stat_features = df_stats.columns

    # Merge statistics into the dataframe
    data = pd.concat([data, df_stats], axis=1)
    data.fillna(0, inplace=True)

    # check that plot_features are all included in stat_features
    missing = [
        f for f in plot_features if f not in stat_features and f not in {"pca", "tsne"}
    ]
    if missing:
        raise ValueError(
            f"Requested plot_features not found in extracted statistics: {missing}. "
            f"Available features: {sorted(stat_features.tolist())}"
        )

    # Standardize features
    features = data[stat_features].copy()
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    # Perform PCA and t-SNE for dimensionality reduction
    pca_projection = None
    if "pca" in plot_features:
        pca = PCA(n_components=1)
        pca_projection = pca.fit_transform(features_scaled)
        data["pca"] = pca_projection[:, 0]

    if "tsne" in plot_features:
        try:
            tsne = TSNE(n_components=1, random_state=42)
            tsne_projection = tsne.fit_transform(features_scaled)
        except Exception:
            # TNSE did not work, probably too small data, just use pca
            if pca_projection is None:
                pca_projection = PCA(n_components=1).fit_transform(features_scaled)
            tsne_projection = pca_projection

        data["tsne"] = tsne_projection[:, 0]

    # Convert parent IDs from string to list
    data["parent_ids"] = data["parent_ids"].apply(safe_parse_parent_ids)

    # Count occurrences of each parent ID
    parent_counts = Counter(
        parent_id for parent_ids in data["parent_ids"] for parent_id in parent_ids
    )

    data["parent_size"] = data["id"].map(lambda x: parent_counts.get(x, 1) * 2)

    no_axis = False
    if ax is None:
        no_axis = True

    # Calculate max fitness for scaling the marker colors
    max_fitness = data["fitness"].max()
    if max_fitness == 0:
        max_fitness = 1.0  # Avoid division by zero

    for x_data in plot_features:
        if no_axis:
            fig, ax = plt.subplots(figsize=(8, 5))

        for _, row in data.iterrows():
            if len(row["parent_ids"]) == 0:
                ax.plot(
                    row["id"],
                    row[x_data],
                    "o",
                    markersize=row["parent_size"] if markersize is None else markersize,
                    color=plt.cm.viridis(row["fitness"] / max_fitness),
                )
            for parent_id in row["parent_ids"]:
                if parent_id in data["id"].values:
                    parent_row = data[data["id"] == parent_id].iloc[0]
                    plot_marker = "-o"
                    ax.plot(
                        [parent_row["id"], row["id"]],
                        [parent_row[x_data], row[x_data]],
                        plot_marker,
                        markersize=(
                            row["parent_size"] if markersize is None else markersize
                        ),
                        color=plt.cm.viridis(row["fitness"] / max_fitness),
                    )
                else:
                    ax.plot(
                        row["id"],
                        row[x_data],
                        "o",
                        markersize=(
                            row["parent_size"] if markersize is None else markersize
                        ),
                        color=plt.cm.viridis(row["fitness"] / max_fitness),
                    )

        ax.set_xlabel("Evaluation")
        ax.set_ylabel(x_data.replace("_", " "))
        if no_axis:
            ax.set_ylim(data[x_data].min() - 1, data[x_data].max() + 1)
        ax.set_xticks([])  # Remove x-ticks
        ax.set_xticklabels([])  # Remove x-tick labels
        if no_axis:
            ax.set_title(f"Evolution of {x_data}")
        ax.grid(True)

        if save and expfolder is not None:
            plt.tight_layout()
            plt.savefig(f"{expfolder}/{x_data}_Evolution.png")
        elif no_axis:
            plt.show()
        if no_axis:
            plt.close(fig)


CEG_FEATURES = [
    "tsne",
    "pca",
    "Nodes",
    "Edges",
    "Max Degree",
    "Min Degree",
    "Mean Degree",
    "Degree Variance",
    "Transitivity",
    "Max Depth",
    "Min Depth",
    "Mean Depth",
    "Max Clustering",
    "Min Clustering",
    "Mean Clustering",
    "Clustering Variance",
    "Degree Entropy",
    "Depth Entropy",
    "Assortativity",
    "Average Eccentricity",
    "Diameter",
    "Radius",
    "Edge Density",
    "Average Shortest Path",
    "mean_complexity",
    "total_complexity",
    "mean_token_count",
    "total_token_count",
    "mean_parameter_count",
    "total_parameter_count",
]

# Display names for select features in the webapp
CEG_FEATURE_LABELS = {
    "mean_complexity": "Mean Complexity",
    "total_complexity": "Total Complexity",
    "mean_token_count": "Mean Token Count",
    "total_token_count": "Total Token Count",
    "mean_parameter_count": "Mean Parameter Count",
    "total_parameter_count": "Total Parameter Count",
}


def plotly_code_evolution(
    run_data: pd.DataFrame,
    feature: str = "total_token_count",
    minimisation: bool = False,
) -> go.Figure:
    """Create an interactive code evolution graph using Plotly.

    Args:
        run_data (pd.DataFrame): The evaluations of one run.
        feature (str): The code feature on the y-axis.
        minimisation (bool): Whether lower fitness is better; the colour
            scale shows the best fitness brightest, failures as the worst.
    """

    data = run_data.copy().reset_index(drop=True)
    if "_id" in data.columns:
        data = data.sort_values("_id", kind="stable").reset_index(drop=True)
    data["eval_index"] = data.index + 1
    data["fitness"] = _fitness_colour_scale(data["fitness"], minimisation)
    data.fillna(0, inplace=True)

    complexity_features = [
        "mean_complexity",
        "total_complexity",
        "mean_token_count",
        "total_token_count",
        "mean_parameter_count",
        "total_parameter_count",
    ]

    if feature in complexity_features:
        df_stats = data["code"].apply(analyse_complexity).apply(pd.Series)
    else:
        df_stats = data["code"].apply(process_code).apply(pd.Series)

    stat_features = df_stats.columns
    data = pd.concat([data, df_stats], axis=1)
    data.fillna(0, inplace=True)

    # replace infinity values with 0
    data.replace([np.inf, -np.inf], 0, inplace=True)

    features = data[stat_features].copy()
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    pca = PCA(n_components=1)
    data["pca"] = pca.fit_transform(features_scaled)[:, 0]
    try:
        tsne = TSNE(n_components=1, random_state=42)
        data["tsne"] = tsne.fit_transform(features_scaled)[:, 0]
    except Exception:
        data["tsne"] = data["pca"]

    data["parent_ids"] = data["parent_ids"].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) else x
    )

    parent_counts = Counter(
        parent_id for parent_ids in data["parent_ids"] for parent_id in parent_ids
    )
    data["parent_size"] = data["id"].map(lambda x: parent_counts.get(x, 1) * 2)

    fig = go.Figure()

    for _, row in data.iterrows():
        for pid in row["parent_ids"]:
            if pid in data["id"].values:
                pr = data[data["id"] == pid].iloc[0]
                fig.add_trace(
                    go.Scatter(
                        x=[pr["eval_index"], row["eval_index"]],
                        y=[pr[feature], row[feature]],
                        mode="lines",
                        line=dict(color="gray", width=1),
                        hoverinfo="skip",
                        showlegend=False,
                    )
                )

    fig.add_trace(
        go.Scatter(
            x=data["eval_index"],
            y=data[feature],
            mode="markers",
            marker=dict(
                size=data["parent_size"],
                color=data["fitness"],
                colorscale="Viridis",
                colorbar=dict(title="Fitness"),
            ),
            showlegend=False,
        )
    )

    fig.update_layout(
        xaxis_title="Evaluation",
        yaxis_title=feature.replace("_", " "),
    )
    return fig


def _no_finished_runs(ax, title: str) -> None:
    """Mark a box plot axis that has no run with a valid final fitness."""
    ax.text(
        0.5,
        0.5,
        "No successful runs",
        ha="center",
        va="center",
        transform=ax.transAxes,
    )
    ax.set_title(title)
    ax.set_xticks([])


def plot_boxplot_fitness(
    logger: ExperimentLogger, y_label="Fitness", x_label="Method", problems=None
):
    """
    Plots boxplots of fitness grouped by problem_name (subplots) and method_name (categories).
    Each problem has its own subplot, grouped by method_name.

    Args:
        logger
        y_label
        x_label
        problems

    Failed runs (a final fitness of ±inf or NaN) are left out; a method
    used with several LLMs gets one box per LLM.
    """
    df = logger.get_data().copy()
    df["fitness"] = _solution_fitness(df)
    add_method_labels(df)

    if problems is None:
        problems = sorted(df["problem_name"].unique())

    # Create subplots, one per problem
    fig, axes = plt.subplots(
        1, len(problems), figsize=(5 * len(problems), 5), sharey=True
    )
    # In case there's only one problem, axes won't be a list
    if len(problems) == 1:
        axes = [axes]

    for i, problem in enumerate(problems):
        subset = df[(df["problem_name"] == problem) & df["fitness"].notna()]
        if subset.empty:
            _no_finished_runs(axes[i], problem)
            continue
        # Plot with Seaborn
        sns.boxplot(x="method_label", y="fitness", data=subset, ax=axes[i])
        axes[i].set_title(problem)
        axes[i].set_xlabel(x_label)
        if i == 0:
            axes[i].set_ylabel(y_label)
        else:
            axes[i].set_ylabel("")
        # Rotate x-axis labels a bit if needed
        axes[i].tick_params(axis="x", rotation=45)

    plt.tight_layout()
    plt.show()


def plot_boxplot_fitness_hue(
    logger: ExperimentLogger,
    y_label="Fitness",
    x_label="Problem",
    hue="method_name",
    x="problem_name",
    problems=None,
):
    """
    Plots boxplots of fitness grouped by `hue` and method_name `x`.
    Produces one plot with grouped boxplots per `x`. Failed runs (a final
    fitness of ±inf or NaN) are left out.

    Args:
        logger
    """
    df = logger.get_data().copy()
    df["fitness"] = _solution_fitness(df)
    if hue == "method_name":
        add_method_labels(df)
        hue = "method_label"

    if problems is None:
        problems = sorted(df["problem_name"].unique())

    df_filtered = df[df["problem_name"].isin(problems)]

    # Depending on the number of problems at hand, plot it within A-4 dimention 8x11
    # If len(problems) > 1; plot in it 2 columns,
    # The method count is aoutmatically scaled, as it is provided a min width of 4 inches.
    prob_len = len(problems)
    fig, axes = plt.subplots(
        ncols=1 if prob_len <= 1 else 2,
        nrows=int(np.ceil(prob_len / 2)),
        figsize=(8, 5 if prob_len <= 2 else 11),
    )

    # Make sure axes is always a flat array for iteration
    if isinstance(axes, np.ndarray):
        axes = axes.flatten()
    else:
        axes = [axes]

    for j in range(prob_len, len(axes)):
        fig.delaxes(axes[j])

    for i, problem in enumerate(problems):
        ax = axes[i]
        df_problem = df_filtered[
            (df_filtered[x] == problem) & df_filtered["fitness"].notna()
        ]
        if df_problem.empty:
            _no_finished_runs(ax, problem)
            continue
        sns.boxplot(x=x, y="fitness", hue=hue, data=df_problem, ax=ax)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_title(problem)

    plt.tight_layout()
    plt.show()


def fitness_table(
    logger: ExperimentLogger,
    alpha=0.05,
    smaller_is_better: bool | None = None,
    minimisation: bool | None = None,
):
    """
    Creates a LaTeX table with rows = methods, columns = problems.
    Each cell shows mean ± std (p=...).
    Cells are bolded if that method is significantly better than all others (p < alpha)
    for the given problem.

    Runs whose final fitness is a failure (±inf or NaN) are left out of the
    statistics and counted in the cell (``k failed``).

    Args:
        logger (ExperimentLogger): the experiment logger to process.
        alpha (float): Significance level for p-value cutoff.
        smaller_is_better (bool, optional): If True, we treat lower fitness
            as “better”, if False higher is “better”. Defaults to the
            direction logged for each problem.
        minimisation (bool, optional): Alias of ``smaller_is_better``.
    """
    if smaller_is_better is None:
        smaller_is_better = minimisation
    df = logger.get_data().copy()
    df["fitness"] = _solution_fitness(df)
    add_method_labels(df)

    # Group data by (problem_name, method)
    # We'll store the runs for each combination so we can compute stats and pairwise tests
    grouped = df.groupby(["problem_name", "method_label"])["fitness"]
    stats_dict = {(p, m): grouped.get_group((p, m)).values for p, m in grouped.groups}

    problems = sorted(df["problem_name"].unique())
    methods = sorted(df["method_label"].unique())
    smaller = {p: _resolve_minimisation(logger, p, smaller_is_better) for p in problems}

    # Prepare a 2D structure for the table: rows=methods, cols=problems
    table_values = []

    for method in methods:
        row_entries = []
        for problem in problems:
            # Retrieve all runs for (problem, method)
            runs = stats_dict.get((problem, method), np.array([]))
            if len(runs) == 0:
                row_entries.append("N/A")
                continue
            arr = runs[~np.isnan(runs)]
            n_failed = len(runs) - len(arr)
            if len(arr) == 0:
                row_entries.append(f"N/A ({n_failed} failed)")
                continue

            mean_val = np.mean(arr)
            std_val = np.std(arr)

            # Compare this method’s distribution to each other method
            # We'll do a t-test and check p-value
            # For "significantly better than all others" we need:
            # 1) The comparison with each other method's distribution has p < alpha
            # 2) The mean is "better" (depending on the direction).
            all_better = True
            pvals = []
            for other_method in methods:
                if other_method == method:
                    continue
                other_arr = stats_dict.get((problem, other_method), np.array([]))
                other_arr = other_arr[~np.isnan(other_arr)]
                if len(other_arr) == 0:
                    # If there’s no data for the other method, skip
                    continue

                # Mean comparison
                other_mean = np.mean(other_arr)
                is_better = (
                    (mean_val < other_mean)
                    if smaller[problem]
                    else (mean_val > other_mean)
                )
                if not is_better:
                    all_better = False

                # Significance test
                # (We could use ttest_ind, Mann-Whitney U, etc. Here we do ttest_ind.)
                _, pval = ttest_ind(arr, other_arr, equal_var=False)
                pvals.append(pval)

            # We'll store the *maximum* p-value among all pairwise comparisons,
            # because for the method to be "significantly better than all others",
            # *every* p-value must be below alpha.
            max_p = max(pvals) if pvals else 1.0

            failed = f", {n_failed} failed" if n_failed else ""
            cell_str = f"{mean_val:.2f} ± {std_val:.2f} (p={max_p:.3f}{failed})"
            if all_better and (max_p < alpha):
                cell_str = "\\textbf{" + cell_str + "}"
            row_entries.append(cell_str)
        table_values.append(row_entries)

    # Create a DataFrame of the final strings so we can export to LaTeX nicely
    table_df = pd.DataFrame(table_values, index=methods, columns=problems)

    return table_df


def plot_token_usage(
    logger: ExperimentLogger,
    save: bool = True,
    return_fig: bool = False,
    return_df: bool = False,
):
    """Plot total tokens used per method/problem from an experiment logger."""

    df = logger.get_data().copy()
    token_records = []
    for _, row in df.iterrows():
        tokens = 0
        if "log_dir" in row:
            convo_path = os.path.join(
                logger.dirname, row["log_dir"], "conversationlog.jsonl"
            )
            if os.path.isfile(convo_path):
                with jsonlines.open(convo_path) as f:
                    for line in f:
                        newtokens = line.get("tokens", 0) or 0
                        if newtokens > 0:
                            tokens += newtokens
                        else:
                            # Older logs without counts: a rough estimate
                            tokens += estimate_tokens(line.get("content", ""))
        token_records.append(
            {
                "method_name": row["method_name"],
                "problem_name": row["problem_name"],
                "tokens": tokens,
            }
        )
    token_df = pd.DataFrame(token_records)
    if return_df:
        return token_df
    summary = (
        token_df.groupby(["problem_name", "method_name"])["tokens"].sum().reset_index()
    )

    fig, ax = plt.subplots(figsize=(7, 5))
    sns.barplot(
        data=summary,
        x="problem_name",
        y="tokens",
        hue="method_name",
        ax=ax,
    )
    ax.set_xlabel("Problem")
    ax.set_ylabel("Total Tokens")
    plt.tight_layout()

    if save:
        fig.savefig(os.path.join(logger.dirname, "token_usage.png"))
    elif not return_fig:
        plt.show()
    if return_fig:
        return fig
    plt.close()
