import json
import math
import os
import random
import re
import time
import traceback

import numpy as np
import pandas as pd
import polars as pl
import sklearn
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score

from ..problem import Problem
from ..solution import Solution

# import autosklearn.classification


class AutoML(Problem):
    """
    Problem class for evaluating AutoML pipelines (sample).

    """

    def __init__(
        self, logger=None, datasets=None, name="AutoML-breast_cancer", eval_timeout=360
    ):
        X, y = load_breast_cancer(return_X_y=True)
        (
            self.X_train,
            self.X_test,
            self.y_train,
            self.y_test,
        ) = sklearn.model_selection.train_test_split(X, y, random_state=1)

        super().__init__(
            logger,
            [(self.X_train, self.y_train)],
            [(self.X_test, self.y_test)],
            name,
            eval_timeout,
        )
        self.func_name = "__call__"
        self.init_inputs = ["X", "y"]
        self.func_inputs = ["X"]
        self.func_outputs = ["y_pred"]

        self.task_prompt = f"""
You are a highly skilled computer scientist in the field machine learning. Your task is to design novel machine learning pipelines for a given dataset and task.
The pipeline in this case should handle a breast cancer classification task. Your task is to write the Python code. The code should contain an `__init__(self, X, y)` function that trains a machine learning model and the function `def __call__(self, X)`, which should predict the samples in X and return the predictions.
The training data X has shape {self.X_train.shape} and y has shape {self.y_train.shape}.
"""
        self.example_prompt = """
An example code structure is as follows:
```python
import numpy as np
import sklearn

class AlgorithmName:
    "Template for a ML pipeline"

    def __init__(self, X, y):
        self.train(X, y)

    def train(self, X, y):
        # Standardize the feature data
        scaler = sklearn.preprocessing.StandardScaler()
        X_train = scaler.fit_transform(X)

        # Let's create and train a logistic regression model
        lr_model = sklearn.linear_model.LogisticRegression()
        lr_model.fit(X_train, y)
        self.model = lr_model
        
    def __call__(self, X):
        # predict using the trained model
        return self.model.predict(X)
```
"""
        self.format_prompt = """

Give an excellent and novel ML pipeline to solve this task and also give it a one-line description, describing the main idea. Give the response in the format:
# Description: <short-description>
# Code: 
```python
<code>
```
"""

    def get_prompt(self):
        """
        Returns the problem description and answer format.
        """
        return self.task_prompt + self.example_prompt + self.format_prompt

    def evaluate(self, solution: Solution, test=False, ioh_dir=""):
        """
        Evaluates a solution on the kernel tuner benchmark using AOCC.
        """
        code = solution.code
        algorithm_name = solution.name
        safe_globals = {
            "sklearn": sklearn,
            "math": math,
            "random": random,
            "np": np,
            "pd": pd,
        }

        exec(code, globals())

        algorithm = None

        # Final validation
        algorithm = globals()[algorithm_name](self.X_train, self.y_train)
        y_pred = algorithm(self.X_test)
        score = accuracy_score(self.y_test, y_pred)

        solution.set_scores(
            score,
            f"The algorithm {algorithm_name} scored {score:.3f} on accuracy (higher is better, 1.0 is the best).",
        )

        return solution

    def test(self, solution: Solution, ioh_dir=""):
        """
        Runs the solution on test instances and returns the fitness score.
        """
        return self.evaluate(solution, True, ioh_dir)

    def to_dict(self):
        """
        Converts the problem to a dictionary.
        """
        return {
            "name": self.name,
        }
