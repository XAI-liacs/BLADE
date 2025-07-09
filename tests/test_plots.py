import json
import os

# Use a non-interactive backend for CI
import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")
from unittest.mock import MagicMock

import matplotlib.pyplot as plt

# Adjust imports to match your actual package structure
from iohblade import (
    fitness_table,
    plot_boxplot_fitness,
    plot_boxplot_fitness_hue,
    plot_code_evolution_graphs,
    plot_convergence,
    plot_experiment_CEG,
    plot_token_usage,
)

# ------------------------------------------------------------------------
# Mock objects and data
# ------------------------------------------------------------------------


class MockExperimentLogger:
    """A minimal mock that simulates the interface needed by your plots."""

    def __init__(self):
        # Some dummy data with the columns your plot functions expect
        self.mock_data = pd.DataFrame(
            {
                "method_name": ["m1", "m1", "m2", "m2"],
                "problem_name": ["p1", "p1", "p1", "p1"],
                "seed": [0, 0, 1, 1],
                "_id": [0, 1, 2, 3],
                "id": ["0", "1", "2", "3"],
                "fitness": [0.0, 1.5, 0.5, 2.0],
                "code": [
                    "print('hello')",
                    "def f(): pass",
                    "def g(): pass",
                    """
def run(budget=100, dim=5):
    for i in range(budget):
        print(i)
    return i+1
""",
                ],
                "parent_ids": ["[]", '["0"]', '["1","0"]', '["2"]'],
                "log_dir": [f"run{i}" for i in range(4)],
            }
        )
        # Additional problems, if you want to test multi-problem plotting
        self.mock_data2 = pd.DataFrame(
            {
                "method_name": ["m1", "m2", "m1", "m2"],
                "problem_name": ["p2", "p2", "p2", "p2"],
                "seed": [0, 0, 1, 1],
                "_id": [0, 1, 2, 3],
                "id": ["0", "1", "2", "3"],
                "fitness": [1.0, 2.0, 3.0, 4.0],
                "code": ["code e", "code f", "code g", "code h"],
                "parent_ids": ["[]", "[]", "[]", "[]"],
                "log_dir": [f"run{i+4}" for i in range(4)],
            }
        )

    @property
    def dirname(self):
        return "mock_dir"

    def get_methods_problems(self):
        """Return the unique methods and problems in the log."""
        methods = self.mock_data["method_name"].unique().tolist()
        problems = list(
            set(self.mock_data["problem_name"].unique())
            | set(self.mock_data2["problem_name"].unique())
        )
        return (methods, sorted(problems))

    def get_problem_data(self, problem_name):
        """Return a dataframe filtered by problem name."""
        # Combine both mock_data sets so that each problem can be tested
        combined = pd.concat([self.mock_data, self.mock_data2], ignore_index=True)
        return combined[combined["problem_name"] == problem_name].copy()

    def get_data(self):
        """Return a combined dataframe to mimic the real logger's get_data()."""
        return pd.concat([self.mock_data, self.mock_data2], ignore_index=True)


@pytest.fixture
def mock_logger(tmp_path):
    # If the plotting code tries to save a file, it’ll do so in this tmp path
    os.chdir(tmp_path)  # switch working dir to temp
    logger = MockExperimentLogger()
    os.makedirs(logger.dirname, exist_ok=True)
    df = pd.concat([logger.mock_data, logger.mock_data2], ignore_index=True)
    for ld in df["log_dir"]:
        run_dir = os.path.join(logger.dirname, ld)
        os.makedirs(run_dir, exist_ok=True)
        with open(os.path.join(run_dir, "conversationlog.jsonl"), "w") as f:
            f.write(json.dumps({"tokens": 5, "content": "x"}) + "\n")
    return logger


# ------------------------------------------------------------------------
# Tests for each plotting function
# ------------------------------------------------------------------------
def test_plot_convergence(mock_logger):
    # Just ensure it runs without exception and creates a figure
    plot_convergence(logger=mock_logger, metric="Fitness", budget=3, save=False)
    # We can grab the figure with plt.gcf() if needed
    fig = plt.gcf()
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_plot_experiment_CEG(mock_logger):
    # Ensure the function runs. It creates multiple subplots for multiple problems.
    plot_experiment_CEG(
        logger=mock_logger,
        metric="total_token_count",
        budget=3,
        save=False,
        max_seeds=2,
    )
    fig = plt.gcf()
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_plot_code_evolution_graphs_single_feature():
    # Minimal run_data
    df = pd.DataFrame(
        {
            "id": [0, 1, 2],
            "parent_ids": ["[]", "[0]", "[1]"],
            "fitness": [1.0, 2.0, 3.0],
            "code": [
                """
def runa(budget=2, dim=5):
    for i in range(budget):
        print(i)
        print(i+1)
    return i+3
""",
                """
def runb(budget=100, dim=5):
    for i in range(budget):
        print(i)
    return i+1
""",
                """
def runc(budget=100, dim=5):
    for j in range(dim):
        print(j)
    return j
""",
            ],
        }
    )
    # We pass a single feature
    fig, ax = plt.subplots()
    plot_code_evolution_graphs(
        run_data=df,
        expfolder=None,
        plot_features=["total_token_count"],  # single feature
        save=False,
        ax=ax,
    )
    # If the function runs, we have a figure
    assert isinstance(plt.gcf(), plt.Figure)
    plt.close(fig)


def test_plot_code_evolution_graphs_multiple_features():
    # If we pass multiple features, we must not pass an external ax
    df = pd.DataFrame(
        {
            "id": ["0", "1", "2"],
            "parent_ids": ["[]", '["0"]', '["1"]'],
            "fitness": [1.0, 2.0, 3.0],
            "code": ["print('hello')", "def f(): pass", "def g(): print('C')"],
        }
    )
    plot_code_evolution_graphs(
        run_data=df,
        expfolder=None,
        plot_features=["pca", "tsne"],  # multiple features
        save=False,
        ax=None,
    )
    assert isinstance(plt.gcf(), plt.Figure)
    plt.close("all")


def test_plot_boxplot_fitness(mock_logger):
    # The code references the "fitness" column, so we just run it:
    plot_boxplot_fitness(
        logger=mock_logger, y_label="Fitness", x_label="Method", problems=["p1", "p2"]
    )
    assert isinstance(plt.gcf(), plt.Figure)
    plt.close("all")


def test_plot_boxplot_fitness_hue(mock_logger):
    plot_boxplot_fitness_hue(
        logger=mock_logger, x_label="Problem", hue="method_name", x="problem_name"
    )
    assert isinstance(plt.gcf(), plt.Figure)
    plt.close("all")


def test_fitness_table(mock_logger):
    # Check that it returns a DataFrame with methods x problems
    table_df = fitness_table(mock_logger, alpha=0.05, smaller_is_better=False)
    assert isinstance(table_df, pd.DataFrame)
    methods, problems = mock_logger.get_methods_problems()
    # The shape should be (#methods) x (#problems) if everything's consistent
    assert table_df.shape == (len(methods), len(problems))
    # Optionally check for bold text in some cells, etc.
    # e.g., to confirm the string format:
    for val in table_df.values.flatten():
        assert isinstance(val, str)
    # No figure is created for a table, so nothing to close.


def test_plot_token_usage(mock_logger):
    plot_token_usage(mock_logger, save=False)
    assert isinstance(plt.gcf(), plt.Figure)
    plt.close("all")
