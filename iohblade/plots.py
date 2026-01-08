import ast
import copy
import difflib
import os
from collections import Counter

import plotly.graph_objects as go

import jsonlines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import ttest_ind
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler, minmax_scale

from .loggers import ExperimentLogger
from .misc.ast import analyse_complexity, process_code


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
):
    """
    Plots the convergence of all methods for each problem from an experiment log.

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
    """
    methods_to_use = methods
    methods, problems = logger.get_methods_problems()

    fig, axes = plt.subplots(
        figsize=(8, 6 * len(problems)), nrows=len(problems), ncols=1
    )
    problem_i = 0
    for problem in problems:
        # Ensure the data is sorted by 'id' and 'fitness'
        data = logger.get_problem_data(problem_name=problem).drop(
            columns=["code"]
        )  # for efficiency we drop code for now
        data.replace([-np.inf, np.inf], 0, inplace=True)
        data.fillna(0, inplace=True)

        # Get unique method names
        methods = data["method_name"].unique()
        if methods_to_use is not None:
            methods = [m for m in methods if m in methods_to_use]
        ax = axes[problem_i] if len(problems) > 1 else axes
        for method in methods:
            method_data = data[data["method_name"] == method].copy()
            method_data = method_data.sort_values(by=["seed", "_id"])

            # Group by 'seed' and calculate the cumulative max fitness
            method_data["cummax_fitness"] = method_data.groupby("seed")[
                "fitness"
            ].cummax()

            # Calculate mean and std deviation of the cumulative max fitness
            if separate_lines:
                for seed in method_data["seed"].unique():
                    seed_data = method_data[method_data["seed"] == seed]
                    ax.plot(
                        seed_data["_id"],
                        seed_data["cummax_fitness"],
                        label=f"{method} (Run {seed})",
                        alpha=0.5,
                    )
            else:
                summary = (
                    method_data.groupby("_id")["cummax_fitness"]
                    .agg([aggregation, variance_aggregation])
                    .reset_index()
                )
                # Shift X-axis so that _id starts at 1
                summary["_id"] += 1  # Ensures _id starts at 1 instead of 0

                # Plot the mean fitness
                method_label = method
                ax.plot(summary["_id"], summary[aggregation], label=method_label)

                # Plot the shaded error region
                if show_std:
                    ax.fill_between(
                        summary["_id"],
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


def plot_experiment_CEG(
    logger: ExperimentLogger,
    metric: str = "total_token_count",
    methods = None,
    markersize = None,
    budget: int = 100,
    save: bool = True,
    max_seeds=5,
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
    """
    methods_to_use = methods
    methods, problems = logger.get_methods_problems()

    problem_i = 0
    for problem in problems:
        # Ensure the data is sorted by 'id' and 'fitness'
        data = logger.get_problem_data(problem_name=problem)
        data.replace([-np.inf], 0, inplace=True)
        data.fillna(0, inplace=True)

        # Get unique runs (seeds)
        seeds = data["seed"].unique()
        #sort the seeds
        seeds = sorted(seeds)
        num_seeds = min(len(seeds), max_seeds)
        # Get unique method names
        methods = data["method_name"].unique()
        if methods_to_use is not None:
            methods = [m for m in methods if m in methods_to_use]
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
                    (data["method_name"] == method) & (data["seed"] == seed)
                ].copy()
                if len(run_data) == 0:
                    continue
                plot_code_evolution_graphs(
                    run_data,
                    logger.dirname,
                    plot_features=[metric],
                    markersize = markersize,
                    save=False,
                    ax=ax,
                )
                ax.set_xlim([0, budget])
                ax.set_xticks(np.arange(0, budget + 1, 20))
                ax.set_xticklabels(np.arange(0, budget + 1, 20))
                method_title = method
                if method == "ES":
                    method_title = "LLaMEA"
                elif method == "ES-guided":
                    method_title = "LLaMEA-SAGE"
                elif method == "ES-guided-new":
                    method_title = "LLaMEA-SAGE"
                ax.set_title(f"{method_title} run:{seed}")
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
    run_data, expfolder=None, plot_features=None, markersize=None, save=True, ax=None
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
    """
    if ax is not None and (plot_features is None or len(plot_features) > 1):
        raise ValueError(
            "If an axis is provided, the length of plot_features must be 1."
        )

    data = run_data.copy().reset_index(drop=True)
    data["eval_index"] = data.index + 1
    data.replace([np.inf, -np.inf], np.nan, inplace=True)
    data["fitness"] = minmax_scale(data["fitness"])
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
            has_any = ast_series.apply(lambda d: isinstance(d, dict) and len(d) > 0).any()

            if has_any:
                df_stats = ast_series.apply(lambda d: d if isinstance(d, dict) else {}).apply(pd.Series)
            else:
                df_stats = data["code"].apply(process_code).apply(pd.Series)
        else:
            df_stats = data["code"].apply(process_code).apply(pd.Series)
    stat_features = df_stats.columns

    # Merge statistics into the dataframe
    data = pd.concat([data, df_stats], axis=1)
    data.fillna(0, inplace=True)

    # Define default features if not provided
    if plot_features is None:
        plot_features = [
            "tsne",
            "pca",
            "total_complexity",
            "total_token_count",
            "total_parameter_count",
        ]
    else:
        plot_features = plot_features

    # check that plot_features are all included in stat_features
    missing = [
        f for f in plot_features
        if f not in stat_features and f not in {"pca", "tsne"}
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

        # 1. Determine the normalized feature name being plotted for comparison
        normalized_x_data = normalize_key(x_data)

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
                    # 2. Determine edge color based on archive feature
                    if "archive_feature" in row:
                        normalized_archive_feature = normalize_key(
                            row["archive_feature"]
                        )
                        edge_color = "gray"  # Default color
                        if normalized_archive_feature == normalized_x_data:
                            if row["archive_direction"] == "increase":
                                edge_color = "green"
                                if row["operator"] == "random":
                                    edge_color = "darkblue"
                            elif row["archive_direction"] == "decrease":
                                edge_color = "red"
                                if row["operator"] == "random":
                                    edge_color = "darkred"
                        else:
                            if row["operator"] == "random":
                                edge_color = "lightblue"
                            elif row["operator"] == "refine":
                                edge_color = "yellow"
                            elif row["operator"] == "crossover":
                                edge_color = "cyan"

                        # Plot the line (edge)
                        ax.plot(
                            [parent_row["id"], row["id"]],
                            [parent_row[x_data], row[x_data]],
                            "-",  # Line only
                            color=edge_color,
                            alpha=0.7,
                            linewidth=1.5,
                            zorder=1,
                        )
                        plot_marker = "o"  # Only plot points separately below

                    ax.plot(
                        [parent_row["id"], row["id"]],
                        [parent_row[x_data], row[x_data]],
                        plot_marker,
                        markersize=row["parent_size"] if markersize is None else markersize,
                        color=plt.cm.viridis(row["fitness"] / max_fitness),
                    )
                else:
                    ax.plot(
                        row["id"],
                        row[x_data],
                        "o",
                        markersize=row["parent_size"] if markersize is None else markersize,
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
        elif ax is None:
            plt.show()
        if ax is None:
            plt.close()


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
    run_data: pd.DataFrame, feature: str = "total_token_count"
) -> go.Figure:
    """Create an interactive code evolution graph using Plotly."""

    data = run_data.copy().reset_index(drop=True)
    data["eval_index"] = data.index + 1
    data.replace([np.inf, -np.inf], np.nan, inplace=True)
    data["fitness"] = minmax_scale(data["fitness"])
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
    """
    df = logger.get_data().copy()
    # If not already present, create a "fitness" column from the "solution" dictionary
    if "fitness" not in df.columns:
        df["fitness"] = df["solution"].apply(
            lambda sol: sol.get("fitness", float("nan"))
        )

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
        subset = df[df["problem_name"] == problem]
        # Plot with Seaborn
        sns.boxplot(x="method_name", y="fitness", data=subset, ax=axes[i])
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
    Produces one plot with grouped boxplots per `x`.
    Args:
        logger
    """
    df = logger.get_data().copy()
    # If not already present, create a "fitness" column from the "solution" dictionary
    if "fitness" not in df.columns:
        df["fitness"] = df["solution"].apply(
            lambda sol: sol.get("fitness", float("nan"))
        )

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
        df_problem = df_filtered[df_filtered[x] == problem]
        sns.boxplot(x=x, y="fitness", hue=hue, data=df_problem, ax=ax)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_title(problem)

    plt.tight_layout()
    plt.show()


def fitness_table(logger: ExperimentLogger, alpha=0.05, smaller_is_better=False):
    """
    Creates a LaTeX table with rows = methods, columns = problems.
    Each cell shows mean ± std (p=...).
    Cells are bolded if that method is significantly better than all others (p < alpha)
    for the given problem.

    Args:
        logger (ExperimentLogger): the experiment logger to process.
        alpha (float): Significance level for p-value cutoff.
        smaller_is_better (bool): If True, we treat lower fitness as “better.” Otherwise, higher is “better.”
    """
    # Ensure there's a 'fitness' column
    df = logger.get_data().copy()
    if "fitness" not in df.columns:
        df["fitness"] = df["solution"].apply(
            lambda sol: sol.get("fitness", float("nan"))
        )

    # Group data by (problem_name, method_name)
    # We'll store the runs for each combination so we can compute stats and pairwise tests
    grouped = df.groupby(["problem_name", "method_name"])["fitness"]
    stats_dict = {(p, m): grouped.get_group((p, m)).values for p, m in grouped.groups}

    problems = sorted(df["problem_name"].unique())
    methods = sorted(df["method_name"].unique())

    # Prepare a 2D structure for the table: rows=methods, cols=problems
    table_values = []

    for method in methods:
        row_entries = []
        for problem in problems:
            # Retrieve all runs for (problem, method)
            arr = stats_dict.get((problem, method), np.array([]))
            if len(arr) == 0:
                row_entries.append("N/A")
                continue

            mean_val = np.mean(arr)
            std_val = np.std(arr)

            # Compare this method’s distribution to each other method
            # We'll do a t-test and check p-value
            # For "significantly better than all others" we need:
            # 1) The comparison with each other method's distribution has p < alpha
            # 2) The mean is "better" (depending on smaller_is_better).
            all_better = True
            pvals = []
            for other_method in methods:
                if other_method == method:
                    continue
                other_arr = stats_dict.get((problem, other_method), np.array([]))
                if len(other_arr) == 0:
                    # If there’s no data for the other method, skip
                    continue

                # Mean comparison
                other_mean = np.mean(other_arr)
                is_better = (
                    (mean_val < other_mean)
                    if smaller_is_better
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

            cell_str = f"{mean_val:.2f} ± {std_val:.2f} (p={max_p:.3f})"
            if all_better and (max_p < alpha):
                cell_str = "\\textbf{" + cell_str + "}"
            row_entries.append(cell_str)
        table_values.append(row_entries)

    # Create a DataFrame of the final strings so we can export to LaTeX nicely
    table_df = pd.DataFrame(table_values, index=methods, columns=problems)

    return table_df


def plot_token_usage(
    logger: ExperimentLogger, save: bool = True, return_fig: bool = False
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
                        tokens += line.get("tokens", 0)
        token_records.append(
            {
                "method_name": row["method_name"],
                "problem_name": row["problem_name"],
                "tokens": tokens,
            }
        )
    token_df = pd.DataFrame(token_records)
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
