import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter
import copy
import ast
import difflib
import jellyfish
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler, minmax_scale
import pandas as pd

from .loggers import ExperimentLogger
from .misc.ast import process_code, analyse_complexity

def plot_convergence(
    logger: ExperimentLogger,
    metric: str = "Fitness",
    budget: int = 100,
    save: bool = True,
):
    """
    Plots the convergence of all methods for each problem from an experiment log.

    Args:
        logger (ExperimentLogger): The experiment logger object.
        metric (str, optional): The metric to show as y-axis label.
        save (bool, optional): Whether to save or show the plot.
    """
    methods, problems = logger.get_methods_problems()

    fig, axes = plt.subplots(figsize=(10, 6*len(problems)), nrows=1, ncols=len(problems))
    problem_i = 0
    for problem in problems:
        # Ensure the data is sorted by 'id' and 'fitness'
        data = logger.get_problem_data(problem_name=problem).drop(columns=['code'])
        data.replace([-np.Inf], 0, inplace=True)
        data.fillna(0, inplace=True)
        
        # Get unique method names
        methods = data['method_name'].unique()
        ax = axes[problem_i] if len(problems) > 1 else axes
        for method in methods:
            method_data = data[data['method_name'] == method].copy()
            method_data = method_data.sort_values(by=['seed', '_id'])

            # Group by 'seed' and calculate the cumulative max fitness
            method_data['cummax_fitness'] = method_data.groupby('seed')['fitness'].cummax()
            
            # Calculate mean and std deviation of the cumulative max fitness
            summary = method_data.groupby('_id')['cummax_fitness'].agg(['mean', 'std']).reset_index()
            
            # Shift X-axis so that _id starts at 1
            summary['_id'] += 1  # Ensures _id starts at 1 instead of 0

            # Plot the mean fitness
            ax.plot(summary['_id'], summary['mean'], label=method)
            
            # Plot the shaded error region
            ax.fill_between(summary['_id'], summary['mean'] - summary['std'], summary['mean'] + summary['std'], alpha=0.2)
        
        # Add labels and legend
        ax.set_xlabel('Number of Evaluations')
        if budget is not None:
            ax.set_xlim(1, budget)
        ax.set_ylabel(f'Mean Best {metric}')
        ax.legend(title='Algorithm')
        ax.grid(True)
        ax.set_title(problem)
        problem_i += 1

    if save:
        fig.savefig(f"{logger.dirname}/convergence.png")
    else:
        plt.show()
    plt.close()

def plot_experiment_CEG(logger: ExperimentLogger, 
    metric: str = "total_token_count",
    budget: int = 100,
    save: bool = True,
    max_seeds = 5):
    """
    Plot the Code evolution graphs for each run in an experiment, splitted by problem.

    Args:
        logger (ExperimentLogger): The experiment logger object.
        metric (str, optional): The metric to show as y-axis label (should be a statistic from AST / Complexity).
        save (bool, optional): Whether to save or show the plot.
        max_seeds (int, optional): The maximum number of runs to plot.
    """
    methods, problems = logger.get_methods_problems()
    
    problem_i = 0
    for problem in problems:
        # Ensure the data is sorted by 'id' and 'fitness'
        data = logger.get_problem_data(problem_name=problem)
        data.replace([-np.Inf], 0, inplace=True)
        data.fillna(0, inplace=True)

        # Get unique runs (seeds)
        seeds = data["seed"].unique()
        num_seeds = min(len(seeds), max_seeds)
        # Get unique method names
        methods = data['method_name'].unique()
        fig, axes = plt.subplots(figsize=(6*len(methods), 6*num_seeds), nrows=len(methods), ncols=num_seeds, sharey=True, squeeze=False)
        
        method_i = 0
        for method in methods:
            seed_i = 0
            for seed in seeds[:num_seeds]:
                ax = axes[method_i, seed_i]
                run_data = data[(data['method_name'] == method) & (data['seed'] == seed)].copy()
                plot_code_evolution_graphs(run_data, logger.dirname, plot_features=["total_token_count"], save=False, ax=ax)
                ax.set_xlim([0, budget])
                ax.set_xticks(np.arange(0, budget+1, 10))
                ax.set_xticklabels(np.arange(0, budget+1, 10))
                seed_i += 1
            method_i += 1
        
        if save:
            plt.tight_layout()
            plt.savefig(f"{logger.dirname}/CEG_{problem}.png")
        else:
            plt.show()
        plt.close()

def plot_code_evolution_graphs(run_data, expfolder=None, plot_features=None, save=True, ax=None):
    """
    Plots optimization progress and relationships between successive solutions in an
    evolutionary run based on AST metrics. Can plot multiple features or a single feature on a provided axis.

    Args:
        run_data (pandas.DataFrame): DataFrame containing code and fitness values.
        expfolder (str, optional): Folder path where the plots are saved. If None, plots are shown.
        plot_features (list, optional): The features to plot. If None, plots multiple default features.
        save (bool): If True, saves the plots otherwise shows them.
        ax (matplotlib.axes.Axes, optional): The axis to plot on. If None, creates new plots.
    """
    if ax is not None and (plot_features is None or len(plot_features) > 1):
        raise ValueError("If an axis is provided, the length of plot_features must be 1.")

    data = run_data.copy()
    data.replace([np.inf, -np.inf], np.nan, inplace=True)
    data["fitness"] = minmax_scale(data["fitness"])
    data.fillna(0, inplace=True)

    complexity_features = ["mean_complexity",
        "total_complexity",
        "mean_token_count",
        "total_token_count",
        "mean_parameter_count",
        "total_parameter_count"
    ]

    # Compute AST or complexity-based statistics
    if len(plot_features) == 1 and plot_features[0] in complexity_features:
        analyse_complexity
        df_stats = data['code'].apply(analyse_complexity).apply(pd.Series)
    else:
        df_stats = data['code'].apply(process_code).apply(pd.Series)
    stat_features = df_stats.columns

    # Merge statistics into the dataframe
    data = pd.concat([data, df_stats], axis=1)

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
    
    # Standardize features
    features = data[stat_features].copy()
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    # Perform PCA and t-SNE for dimensionality reduction
    pca = PCA(n_components=1)
    pca_projection = pca.fit_transform(features_scaled)
    data["pca"] = pca_projection[:, 0]

    tsne = TSNE(n_components=1, random_state=42)
    tsne_projection = tsne.fit_transform(features_scaled)
    data["tsne"] = tsne_projection[:, 0]

    # Convert parent IDs from string to list
    data["parent_ids"] = data["parent_ids"].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) else x
    )

    # Count occurrences of each parent ID
    parent_counts = Counter(
        parent_id for parent_ids in data["parent_ids"] for parent_id in parent_ids
    )
    
    data["parent_size"] = data["id"].map(lambda x: parent_counts.get(x, 1) * 2)

    no_axis = False
    if ax is None:
        no_axis = True
    for x_data in plot_features:
        if no_axis:
            fig, ax = plt.subplots(figsize=(8, 5))

        for _, row in data.iterrows():
            for parent_id in row["parent_ids"]:
                if parent_id in data["id"].values:
                    parent_row = data[data["id"] == parent_id].iloc[0]
                    ax.plot(
                        [parent_row["id"], row["id"]],
                        [parent_row[x_data], row[x_data]],
                        "-o",
                        markersize=row["parent_size"],
                        color=plt.cm.viridis(row["fitness"] / max(data["fitness"]))
                    )
                else:
                    ax.plot(
                        row["id"],
                        row[x_data],
                        "o",
                        markersize=row["parent_size"],
                        color=plt.cm.viridis(row["fitness"] / max(data["fitness"]))
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