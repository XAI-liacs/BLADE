import json
import math
import os
import random
import re
import time
import traceback
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
import numpy as np
import pandas as pd
import polars as pl
import sklearn
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import openml
from ConfigSpace import Configuration, ConfigurationSpace
from smac import AlgorithmConfigurationFacade, Scenario
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    root_mean_squared_error,
    mean_absolute_error,
)
from ..problem import Problem
from ..solution import Solution


def _summarize_dataset(X, y):
    df = pd.DataFrame(X)
    n_samples, n_features = df.shape
    rounded_samples = (
        f"~{int(round(n_samples, -3)):,} samples"
        if n_samples >= 1000
        else f"{n_samples} samples"
    )
    bools = sum(pd.api.types.is_bool_dtype(df[col]) for col in df.columns)
    ints = sum(pd.api.types.is_integer_dtype(df[col]) for col in df.columns)
    reals = n_features - bools - ints
    feats_desc = (
        f"~{n_features} features: {bools} boolean, {ints} integer, {reals} real"
    )
    return rounded_samples, feats_desc


def _is_classification_task(task):
    # OpenML: task.task_type contains 'Supervised Classification' / 'Supervised Regression' etc.
    return "classification" in str(task.task_type).lower()


class AutoML(Problem):
    """
    Problem class for evaluating AutoML pipelines (sample).

    """

    def __init__(
        self,
        logger=None,
        datasets=None,
        name=None,
        eval_timeout=3600,
        openml_task_id: int | None = None,
        use_official_split: bool = True,
    ):
        """
        If openml_task_id is provided, this problem loads the OpenML task
        and uses the official train/test split.
        """
        self.openml_task_id = openml_task_id
        self.eval_name = None
        self.split_info = {}
        self.le_ = None
        self.cat_enc_ = None
        self.imputer_ = None

        if openml_task_id is not None:
            self.task = openml.tasks.get_task(openml_task_id)
            self.eval_name = self.task.evaluation_measure
            if self.eval_name is None:
                self.eval_name = (
                    "predictive_accuracy"
                    if _is_classification_task(self.task)
                    else "root_mean_squared_error"
                )

            # data
            X, y = self.task.get_X_and_y(dataset_format="dataframe")
            (
                self.n_repeats,
                self.n_folds,
                self.n_samples,
            ) = self.task.get_split_dimensions()

            # Label-encode globally so label ids are consistent across folds
            self.le_ = LabelEncoder()
            self.y_all_ = self.le_.fit_transform(y)
            self.X_all_ = X

            # Keep summary for logs
            self.split_info = {
                "n_repeats": self.n_repeats,
                "n_folds": self.n_folds,
                "n_samples": self.n_samples,
            }

            samples_desc, feats_desc = _summarize_dataset(X, y)
            task_type = (
                "classification" if _is_classification_task(self.task) else "regression"
            )

        self.task_prompt = f""" 
        You can use the following Python packages: scikit-learn, numpy, scipy, pandas.
        Design an ML pipeline for a {task_type} task with {samples_desc} and {feats_desc}.
        Write a single Python class:
        - __init__(self, X, y, **hyperparameters)  -> fit exactly once (no CV here)
        - __call__(self, X)    
        IMPORTANT CONSTRAINTS (must follow):
        - Do NOT use any internal HPO or CV search (no GridSearchCV, RandomizedSearchCV, BayesSearchCV, Optuna, Hyperopt, skopt, etc.).
        - Do NOT implement your own tuning loops (no KFold/StratifiedKFold loops, no parameter sweeps).
        - Pick ONE primary estimator inside the class (no estimator-switching in a param grid).
        - If you do preprocessing (e.g., StandardScaler, PCA), assign them to self.* and reuse them in __call__.
        - Expose tunable hyperparameters via __init__ kwargs with sensible defaults; external HPO will tune them.
        """

        self.example_prompt = """
        Here is a minimal template (for reference only; you must output your own improved class). An example code structure is as follows:
        ```python
        import numpy as np
        import sklearn
        from sklearn.preprocessing import StandardScaler
        from sklearn.linear_model import LogisticRegression
        class MyPipeline:
            \"Simple, single-estimator pipeline without internal HPO.\"
            def __init__(self, X, y, C=1.0):
                # keep references for use in __call__
                self.scaler = StandardScaler()
                Xs = self.scaler.fit_transform(X)

                # choose ONE estimator; expose its hyperparameters
                self.model = LogisticRegression(C=C, max_iter=1000, n_jobs=1, random_state=42)
                self.model.fit(Xs, y)

            def __call__(self, X):
                Xs = self.scaler.transform(X)
                return self.model.predict(Xs)
        ```
        """

        self.format_prompt = """
        Give an excellent and novel ML pipeline to solve this task and also give it a one-line description, describing the main idea. Give the response in the format:
        # Description: <short-description>
        # Code: 
        ```python
        <code>
        ```
        # Space:
        ```python
        {
            # e.g. "C": (1e-3, 10.0), "max_depth": (3, 15), "alpha": (1e-4, 1.0)
        }
        ```
        """

        super().__init__(
            logger,
            [],
            [],
            name,
            eval_timeout,
        )
        self.func_name = "__call__"
        self.init_inputs = ["X", "y"]
        self.func_inputs = ["X"]
        self.func_outputs = ["y_pred"]

        self.METRIC_MAP = {
            "predictive_accuracy": accuracy_score,
            "f1": f1_score,
            "area_under_roc_curve": roc_auc_score,
            "root_mean_squared_error": root_mean_squared_error,
            "mean_absolute_error": mean_absolute_error,
        }

    def get_prompt(self):
        """
        Returns the problem description and answer format.
        """
        return self.task_prompt + self.example_prompt + self.format_prompt

    def prepare_split(self, repeat: int, fold: int, sample: int):
        # 1) indices
        train_idx, test_idx = self.task.get_train_test_split_indices(
            repeat=repeat, fold=fold, sample=sample
        )

        # 2) slice features/labels (labels already globally encoded -> y_all_)
        X_tr = self.X_all_.iloc[train_idx].copy()
        X_te = self.X_all_.iloc[test_idx].copy()
        y_tr = self.y_all_[train_idx]
        y_te = self.y_all_[test_idx]

        # 3) per-fold categorical handling
        cat_cols = X_tr.select_dtypes(include=["object", "category", "string"]).columns
        if len(cat_cols) > 0:
            enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
            X_tr[cat_cols] = enc.fit_transform(X_tr[cat_cols])
            X_te[cat_cols] = enc.transform(X_te[cat_cols])

        # 4) booleans -> ints
        bool_cols = X_tr.select_dtypes(include=["bool"]).columns
        if len(bool_cols) > 0:
            X_tr[bool_cols] = X_tr[bool_cols].astype(int)
            X_te[bool_cols] = X_te[bool_cols].astype(int)

        # 5) impute (fit on train)
        imp = SimpleImputer(strategy="most_frequent")
        X_tr = pd.DataFrame(
            imp.fit_transform(X_tr), columns=X_tr.columns, index=X_tr.index
        )
        X_te = pd.DataFrame(imp.transform(X_te), columns=X_te.columns, index=X_te.index)

        return X_tr, X_te, y_tr, y_te

    def _compute_score(self, alg, y_pred):
        """
        Compute score following rules:
        - If metric requires labels (accuracy/f1), convert proba/logits to labels when needed
        - If metric is AUC, prefer predict_proba / decision_function
        - For regression errors (RMSE/MAE), lower is better -> invert for fitness
        """
        metric_name = self.eval_name

        # AUC handled first
        if metric_name == "area_under_roc_curve":
            if hasattr(alg, "predict_proba"):
                proba = alg.predict_proba(self.X_test)
                if proba.ndim == 2 and proba.shape[1] > 1:
                    score = roc_auc_score(self.y_test, proba, multi_class="ovr")
                else:
                    score = roc_auc_score(self.y_test, proba.ravel())
            elif hasattr(alg, "decision_function"):
                scores = alg.decision_function(self.X_test)
                if np.ndim(scores) == 2 and scores.shape[1] > 1:
                    score = roc_auc_score(self.y_test, scores, multi_class="ovr")
                else:
                    score = roc_auc_score(self.y_test, scores)
            else:
                # fallback
                if hasattr(y_pred, "shape") and getattr(y_pred, "ndim", 1) == 2:
                    y_pred = np.argmax(y_pred, axis=1)
                score = accuracy_score(self.y_test, y_pred)
                metric_name = "predictive_accuracy(fallback)"
            return score, metric_name, False  # higher is better

        # non-AUC metrics
        # If y_pred are probabilities or floats for binary, convert to labels
        if hasattr(y_pred, "shape") and getattr(y_pred, "ndim", 1) == 2:
            y_pred = np.argmax(y_pred, axis=1)
        elif (
            np.issubdtype(np.asarray(y_pred).dtype, np.floating)
            and len(np.unique(self.y_train)) == 2
        ):
            y_pred = (np.asarray(y_pred) >= 0.5).astype(int)

        scorer = self.METRIC_MAP.get(metric_name, accuracy_score)
        if metric_name == "f1":
            score = scorer(self.y_test, y_pred, average="macro")
            return score, metric_name, False  # higher is better
        elif metric_name in {"root_mean_squared_error", "mean_absolute_error"}:
            score = scorer(self.y_test, y_pred)
            return score, metric_name, True  # lower is better -> invert for fitness
        else:
            score = scorer(self.y_test, y_pred)
            return score, metric_name, False  # higher is better

    def evaluate(self, solution: Solution, test=False, ioh_dir=""):
        """
        Evaluate a generated pipeline on the OpenML task, optionally doing in-the-loop HPO
        if a configuration space is provided by the LLM (# Space block).
        """
        code = solution.code
        name = solution.name

        # Safe exec globals
        safe_globals = {
            "sklearn": sklearn,
            "math": math,
            "random": random,
            "np": np,
            "pd": pd,
        }
        safe_globals = {k: v for k, v in safe_globals.items() if v is not None}

        # Compile user code
        exec(code, safe_globals)
        if name not in safe_globals:
            raise RuntimeError(f"Class '{name}' not found in generated code.")
        alg_cls = safe_globals[name]

        # Metric helpers
        metric_name = self.eval_name
        METRIC_MAP = {
            "predictive_accuracy": accuracy_score,
            "f1": f1_score,
            "area_under_roc_curve": roc_auc_score,
            "root_mean_squared_error": root_mean_squared_error,
            "mean_absolute_error": mean_absolute_error,
        }
        is_error_metric = metric_name in {
            "root_mean_squared_error",
            "mean_absolute_error",
        }

        # fold evaluation helper
        def eval_single_fold(config_dict, r, s, f):
            """Train on (r,s,f) train fold and return the score on its test fold."""
            X_tr, X_te, y_tr, y_te = self.prepare_split(r, f, s)

            try:
                alg = alg_cls(X_tr, y_tr, **(config_dict or {}))
            except TypeError:
                # If the class didn't accept hyperparams, fallback to default
                alg = alg_cls(X_tr, y_tr)

            # AUC first (needs proba/decision scores)
            if metric_name == "area_under_roc_curve":
                if hasattr(alg, "predict_proba"):
                    proba = alg.predict_proba(X_te)
                    if proba.ndim == 2 and proba.shape[1] > 1:
                        return roc_auc_score(y_te, proba, multi_class="ovr")
                    else:
                        return roc_auc_score(y_te, np.ravel(proba))
                elif hasattr(alg, "decision_function"):
                    scores = alg.decision_function(X_te)
                    if np.ndim(scores) == 2 and scores.shape[1] > 1:
                        return roc_auc_score(y_te, scores, multi_class="ovr")
                    else:
                        return roc_auc_score(y_te, scores)
                else:
                    # fallback to accuracy if probabilities/scores not available
                    y_pred = alg(X_te)
                    if hasattr(y_pred, "shape") and y_pred.ndim == 2:
                        y_pred = np.argmax(y_pred, axis=1)
                    elif (
                        np.issubdtype(np.asarray(y_pred).dtype, np.floating)
                        and len(np.unique(y_tr)) == 2
                    ):
                        y_pred = (np.asarray(y_pred) >= 0.5).astype(int)
                    return accuracy_score(y_te, y_pred)

            # non-AUC metrics
            y_pred = alg(X_te)
            if hasattr(y_pred, "shape") and getattr(y_pred, "ndim", 1) == 2:
                y_pred = np.argmax(y_pred, axis=1)
            elif (
                np.issubdtype(np.asarray(y_pred).dtype, np.floating)
                and len(np.unique(y_tr)) == 2
            ):
                y_pred = (np.asarray(y_pred) >= 0.5).astype(int)

            scorer = METRIC_MAP.get(metric_name, accuracy_score)
            if metric_name == "f1":
                return scorer(y_te, y_pred, average="macro")
            else:
                return scorer(y_te, y_pred)

        # Enumerate ALL official splits
        all_splits = [
            (r, s, f)
            for r in range(self.split_info["n_repeats"])
            for s in range(self.split_info["n_samples"])
            for f in range(self.split_info["n_folds"])
        ]

        # HPO with SMAC (if Space present)
        incumbent_dict = {}
        did_hpo = False

        cs: ConfigurationSpace | None = getattr(solution, "configspace", None)
        if cs is not None:
            # Use a subset of instances for quicker HPO
            rng = np.random.RandomState(42)
            max_hpo_instances = min(12, len(all_splits))
            hpo_instances = all_splits.copy()
            rng.shuffle(hpo_instances)
            hpo_instances = hpo_instances[:max_hpo_instances]

            # Map instances to strings so SMAC can pass them back
            instance_ids = [f"r{r},s{s},f{f}" for (r, s, f) in hpo_instances]
            inst_feats = {
                iid: [r, s, f] for iid, (r, s, f) in zip(instance_ids, hpo_instances)
            }

            def target(cfg: Configuration, instance: str, seed: int = 0) -> float:
                """SMAC target function: return a MINIMIZATION loss."""
                r_str, s_str, f_str = instance.split(",")
                r = int(r_str[1:])
                s = int(s_str[1:])
                f = int(f_str[1:])
                try:
                    score = eval_single_fold(dict(cfg), r, s, f)
                except Exception:
                    # On any fold failure, give worst possible loss
                    return 1.0 if not is_error_metric else 1e9

                # Convert to loss for SMAC (minimize)
                if is_error_metric:
                    # already an error -> minimize directly
                    return float(score)
                else:
                    # higher is better -> minimize (1 - score), clamp to [0, 1]
                    return float(max(0.0, min(1.0, 1.0 - score)))

            out_dir = None
            if getattr(self, "logger", None) and getattr(self.logger, "dirname", None):
                out_dir = os.path.join(self.logger.dirname, "smac")
            elif ioh_dir:
                out_dir = os.path.join(ioh_dir, "smac")
            else:
                out_dir = "smac3_output"
            scenario = Scenario(
                cs,
                name=f"automl-{self.openml_task_id}-{int(time.time())}",
                deterministic=True,
                n_trials=100,
                instances=instance_ids,
                instance_features=inst_feats,
                output_directory=out_dir,
            )

            smac = AlgorithmConfigurationFacade(
                scenario, target_function=target, logging_level=30
            )
            incumbent = smac.optimize()
            incumbent_dict = dict(incumbent)
            solution.add_metadata("incumbent", incumbent_dict)
            did_hpo = True

        # Final validation on all splits
        fold_scores = []
        n_failed = 0
        for r, s, f in all_splits:
            try:
                fold_score = eval_single_fold(incumbent_dict, r, s, f)
                fold_scores.append(float(fold_score))
            except Exception:
                n_failed += 1

        mean_score = float(np.mean(fold_scores)) if fold_scores else float("-inf")
        std_score = float(np.std(fold_scores)) if fold_scores else float("nan")

        # Fitness sign (BLADE expects higher is better)
        fitness = -mean_score if is_error_metric else mean_score

        msg = (
            f"{self.eval_name} = {mean_score:.4f} ± {std_score:.4f} | "
            f"splits: repeats={self.split_info['n_repeats']}, "
            f"folds={self.split_info['n_folds']}, samples={self.split_info['n_samples']} | "
            f"failed_splits={n_failed}"
        )
        if did_hpo:
            msg += f" | HPO used; incumbent={incumbent_dict}"

        solution.add_metadata("fold_scores", fold_scores)
        solution.set_scores(fitness, msg)
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
        d = {"name": self.name}
        if self.openml_task_id is not None:
            d["openml_task_id"] = self.openml_task_id
            d["metric"] = self.eval_name
        return d
