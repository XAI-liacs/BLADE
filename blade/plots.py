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
    metric: str = "fitness",
    budget: int = 100,
    save: bool = True,
    show: bool = False,
):
    """
    Plots the convergence of all methods for each problem from an experiment log.

    Args:
        logger (ExperimentLogger): The experiment logger object.
        metric (str, optional): The metric to plot (e.g., fitness, error, etc.).
        save (bool, optional): Whether to save the plot.
        show (bool, optional): Whether to show the plot.
    """
    methods, problems = logger.get_methods_problems()
    for problem in problems:
        for method in methods:
            data = logger.get_data(method_name, problem_name)
            if data is None:
                raise ValueError(f"No data found for {method_name} on {problem_name}.")
            for run in data:
                if metric not in run:
                    raise ValueError(f"No metric {metric} found in data.")
                
        
        fig, ax = plt.subplots()
        for run in data:
            ax.plot(run[metric], label=f"Run {run['run_id']}")

        ax.set_xlabel("Generation")
        ax.set_ylabel(metric)
        ax.set_title(f"{method_name} on {problem_name}")
        ax.legend()
        if save:
            plt.savefig(f"{method_name}_{problem_name}_{metric}.png")
        if show:
            plt.show()




for k in range(budget):
        m_aucs = []
        for j in range(len(exp_dir)):
            d = exp_dir[j]
            if os.path.isfile(f"{d}/try-{k}-aucs.txt"):
                aucs = np.loadtxt(f"{d}/try-{k}-aucs.txt")
                m_aucs.append(np.mean(aucs))
                while j >= len(current_best):
                    current_best.append(0)
                if current_best[j] < np.mean(aucs):
                    current_best[j] = np.mean(aucs)
        if len(current_best) > 0:
            mean_aucs.append(np.mean(current_best))
            std_aucs.append(np.std(current_best))
        else:
            mean_aucs.append(np.nan)
            std_aucs.append(np.nan)
            
    mean_aucs = np.array(mean_aucs)
    std_aucs = np.array(std_aucs)
    x = np.arange(budget)
    
    plt.plot(x, mean_aucs, color=color, linestyle=ls, label=label)
    plt.fill_between(x, mean_aucs - std_aucs, mean_aucs + std_aucs, color=color, alpha=0.05)