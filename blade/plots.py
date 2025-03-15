import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
import copy
import ast
import difflib
import jellyfish

from .loggers import ExperimentLogger
from .misc.ast import process_code

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

    plt.title('Convergence Plot')
    if save:
        plt.savefig(f"{logger.dirname}/convergence.png")
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
        TODO
    """
    methods, problems = logger.get_methods_problems()
    
    problem_i = 0
    for problem in problems:
        # Ensure the data is sorted by 'id' and 'fitness'
        data = logger.get_problem_data(problem_name=problem).drop(columns=['code'])
        data.replace([-np.Inf], 0, inplace=True)
        data.fillna(0, inplace=True)

        # Get unique runs (seeds)
        seeds = data["seed"].unique()
        num_seeds = min(len(seeds), max_seeds)
        # Get unique method names
        methods = data['method_name'].unique()
        fig, axes = plt.subplots(figsize=(6*len(methods), 6*num_seeds), nrows=len(methods), ncols=num_seeds)
        
        method_i = 0
        for method in methods:
            seed_i = 0
            for seed in seeds[:num_seeds]:
                ax = axes[method_i*len(methods)+seed_i] if len(methods) * num_seeds > 1 else axes
                run_data = data[(data['method_name'] == method) & (data['seed'] == seed)].copy()
                plot_code_evolution_graph(run_data, logger.dirname, plot_features="total_token_count", save=False, ax=ax)
                seed_i += 1
            method_i += 1
        
        plt.title('Code Evolution Graph')
        if save:
            plt.savefig(f"{logger.dirname}/CEG_{problem}.png")
        else:
            plt.show()
        plt.close()


def plot_CEGS(run_data, expfolder, plot_features=None, save: bool = True):
    """
    Plots optimization progress and relationships between successive solutions in an
    evolutionary run based on AST metrics. Generates multiple plots showing how solutions
    change over time according to different projected axes (e.g. t-SNE, PCA).

    Args:
        run_data (pandas.DataFrame): DataFrame containing code and fitness
            values for all solutions in the run.
        expfolder (str): Folder path where the generated plots are saved and the run data is retrieved from.
        plot_features (list): List of features to plot (e.g., ["tsne_x", "pca_x"]). If None,
            defaults to ["tsne", "pca", "total_complexity", "total_token_count",
            "total_parameter_count"].
        save (bool): If True, saves the plots otherwise shows them.
        ax: The axis to plot it on, if None creates a new plot.
    """
    data = run_data.copy()
    data.replace([np.inf, -np.inf], np.nan, inplace=True)
    data["fitness"] = minmax_scale(data["fitness"])
    data.fillna(0, inplace=True)

    # calculate AST features for the run
    # Apply function and expand results into new columns
    df_stats = df['code'].apply(process_code).apply(pd.Series)
    stat_features = df_stats.columns

    # Merge statistics back into the original DataFrame
    df = pd.concat([df, df_stats], axis=1)
    
    if plot_features is None:
        plot_features = [
            "tsne",
            "pca",
            "total_complexity",
            "total_token_count",
            "total_parameter_count",
        ]

    # Separate metadata and features
    features = data[stat_features].copy()
    metadata = data.drop(columns=stat_features)

    # Convert string data to lists when needed
    data["parent_ids"] = data["parent_ids"].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) else x
    )

    # Standardize features for PCA/tSNE
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    # Create a 1D projection using PCA
    pca = PCA(n_components=1)
    pca_projection = pca.fit_transform(features_scaled)
    data["pca"] = pca_projection[:, 0]

    # Create a 1D projection using t-SNE
    tsne = TSNE(n_components=1, random_state=42)
    tsne_projection = tsne.fit_transform(features_scaled)
    data["tsne"] = tsne_projection[:, 0]

    # Plot the evolution in t-SNE feature space
    parent_counts = Counter(
        parent_id for parent_ids in data["parent_ids"] for parent_id in parent_ids
    )

    data["parent_size"] = data["id"].map(
        lambda x: (parent_counts[x]) if x in parent_counts else 1
    )

    for x_data in plot_features:

        plt.figure()
        for _, row in data.iterrows():
            for parent_id in row["parent_ids"]:
                if parent_id in data["id"].values:
                    parent_row = data[data["id"] == parent_id].iloc[0]
                    plt.plot(
                        [parent_row["id"], row["id"]],
                        [parent_row[x_data], row[x_data]],
                        "-o",
                        markersize=row["parent_size"],
                        color=plt.cm.viridis(row["fitness"] / max(data["fitness"])),
                    )
                else:
                    plt.plot(
                        row["id"],
                        row[x_data],
                        "o",
                        markersize=row["parent_size"],
                        color=plt.cm.viridis(row["fitness"] / max(data["fitness"])),
                    )
        plt.xlabel("Evaluation")
        plt.ylabel(x_data.replace("_", " "))
        plt.ylim(data[x_data].min() - 1, data[x_data].max() + 1)
        plt.tight_layout()
        if save:
            plt.savefig(f"{expfolder}/{x_data}_Evolution.png")
        else:
            plt.show()
        plt.close()
        
def plot_CEG(run_data, plot_feature, ax, process_code):
    """
    Plots the evolution of a single feature in an evolutionary run on a provided axis.

    Args:
        run_data (pandas.DataFrame): DataFrame containing code and fitness values.
        plot_feature (str): The feature to plot (e.g., "total_token_count").
        ax (matplotlib.axes.Axes): The axis to plot on.
        process_code (function): Function that processes code and returns a dictionary of statistics.
    """
    if ax is None:
        raise ValueError("An axis object must be provided.")

    data = run_data.copy()
    data.replace([np.inf, -np.inf], np.nan, inplace=True)
    data["fitness"] = minmax_scale(data["fitness"])
    data.fillna(0, inplace=True)

    # Compute AST-based statistics
    df_stats = data['code'].apply(process_code).apply(pd.Series)
    stat_features = df_stats.columns

    # Merge statistics into the dataframe
    data = pd.concat([data, df_stats], axis=1)

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
    
    data["parent_size"] = data["id"].map(lambda x: parent_counts.get(x, 1))

    # Plot the evolution of the selected feature
    for _, row in data.iterrows():
        for parent_id in row["parent_ids"]:
            if parent_id in data["id"].values:
                parent_row = data[data["id"] == parent_id].iloc[0]
                ax.plot(
                    [parent_row["id"], row["id"]],
                    [parent_row[plot_feature], row[plot_feature]],
                    "-o",
                    markersize=row["parent_size"],
                    color=plt.cm.viridis(row["fitness"] / max(data["fitness"]))
                )
            else:
                ax.plot(
                    row["id"],
                    row[plot_feature],
                    "o",
                    markersize=row["parent_size"],
                    color=plt.cm.viridis(row["fitness"] / max(data["fitness"]))
                )
    
    ax.set_xlabel("Evaluation")
    ax.set_ylabel(plot_feature.replace("_", " "))
    ax.set_ylim(data[plot_feature].min() - 1, data[plot_feature].max() + 1)
    ax.set_title(f"Evolution of {plot_feature}")
    ax.grid(True)