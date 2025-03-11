import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
import copy
import ast
import difflib
import jellyfish

from .loggers import ExperimentLogger

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
        plt.savefig(f"convergence_{problem}.png")
    else:
        plt.show()
    plt.close()