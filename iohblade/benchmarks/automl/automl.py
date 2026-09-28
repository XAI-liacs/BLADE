import ast
import json
import math
import os
import textwrap
import random
import re
import time
import traceback
from typing import Any
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder
import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import openml
from ConfigSpace import Configuration, ConfigurationSpace
from smac import AlgorithmConfigurationFacade, Scenario

from iohblade.tags import PrimaryCategories, Benchmark
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    root_mean_squared_error,
    mean_absolute_error,
)
from iohblade.problem import Problem
from iohblade.solution import Solution
from iohblade.tabpfn_utils import VALID_TABPFN_MODES

VALID_TABPFN_EVAL_SOURCES = ("env", "pip")


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


def _format_solution_exception(exc: Exception, solution: Solution | None = None) -> str:
    """Error message for the LLM, with the line of the generated code that failed."""

    error_type = type(exc).__name__
    error_msg = str(exc)
    frames = traceback.extract_tb(exc.__traceback__)
    frame = None

    for candidate in reversed(frames):
        if candidate.filename == "<string>":
            frame = candidate
            break

    if frame is None and frames:
        frame = frames[-1]

    if frame is None:
        return f"{error_type}: {error_msg}"

    msg = (
        f"In the code, line {frame.lineno}, in {frame.name}, "
        "the following error occurred:\n"
        f"{error_type}: {error_msg}"
    )

    if solution is not None and hasattr(solution, "code"):
        code_lines = solution.code.splitlines()
        if 1 <= frame.lineno <= len(code_lines):
            code_line = code_lines[frame.lineno - 1].strip()
            if code_line:
                msg += f"\nOn line: {code_line}"

    return msg


def _format_split_failure(failure: dict) -> str:
    split = (
        f"repeat={failure['repeat']}, "
        f"fold={failure['fold']}, sample={failure['sample']}"
    )
    return f"First failed split ({split}):\n{failure['error']}"


def _make_tabpfn_prompt_text(
    task_type: str,
    tabpfn_signature: str | None = None,
    tabpfn_allowed_kwargs: list[str] | None = None,
) -> str:
    if task_type == "regression":
        estimator = "TabPFNRegressor"
        task_label = "regression"
    else:
        estimator = "TabPFNClassifier"
        task_label = "classification"

    if tabpfn_signature:
        api_hint = (
            f"\n        The detected constructor for this installed TabPFN "
            f"version is `{tabpfn_signature}`."
        )
    elif tabpfn_allowed_kwargs:
        api_hint = (
            "\n        The detected constructor keyword arguments for this "
            f"installed TabPFN version are: {tabpfn_allowed_kwargs}."
        )
    else:
        api_hint = ""

    return textwrap.dedent(f"""
        The execution environment also contains the `tabpfn` package. For
        {task_label} tasks, you may use
        `from tabpfn import {estimator}` as a valid
        scikit-learn-compatible estimator when appropriate. Treat TabPFN as a
        single estimator family, not as an AutoML system. Do not wrap it in
        GridSearchCV, RandomizedSearchCV, Optuna, Hyperopt,
        estimator-switching search loops, nested AutoML systems, or any
        internal model-selection/search loop. Keep the generated solution
        simple and compatible with the required class interface.{api_hint}
        Use only constructor keyword arguments that exist in the installed
        TabPFN version. Do NOT use old TabPFN API arguments such as
        `N_ensemble_configurations`, `n_neighbors`, or `noise_std` unless they
        appear in the detected constructor above. When unsure, instantiate
        `{estimator}()` with no TabPFN-specific hyperparameters.
        """)


def _append_tabpfn_dependency(dependencies: list[str], version: str | None) -> None:
    for dep in dependencies:
        if re.match(r"^tabpfn(\s|$|\[|[<>=!~])", dep.strip(), re.I):
            return

    dependencies.append(f"tabpfn=={version}" if version else "tabpfn")


def _constrain_sklearn_for_tabpfn(dependencies: list[str]) -> None:
    for i, dep in enumerate(dependencies):
        if re.match(r"^scikit-learn(\s|$|\[|[<>=!~])", dep.strip(), re.I):
            dependencies[i] = "scikit-learn>=1.4,<1.8"
            return

    dependencies.append("scikit-learn>=1.4,<1.8")


def _call_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = _call_name(node.value)
        return f"{parent}.{node.attr}" if parent else node.attr
    return ""


def _constant_value(node: ast.AST):
    if isinstance(node, ast.Constant):
        return node.value
    return None


def _validate_generated_code_constraints(
    code: str,
    tabpfn_allowed_kwargs: list[str] | None = None,
) -> None:
    """Raise an error if the generated code runs its own CV or HPO search."""

    forbidden_names = {
        "GridSearchCV",
        "RandomizedSearchCV",
        "BayesSearchCV",
        "OptunaSearchCV",
        "KFold",
        "StratifiedKFold",
        "RepeatedKFold",
        "RepeatedStratifiedKFold",
        "LeaveOneOut",
        "LeavePOut",
    }
    forbidden_modules = {"optuna", "hyperopt", "skopt"}
    tabpfn_estimators = {"TabPFNClassifier", "TabPFNRegressor"}
    violations = []
    allowed_tabpfn_kwargs = (
        set(tabpfn_allowed_kwargs) if tabpfn_allowed_kwargs is not None else None
    )

    try:
        tree = ast.parse(code)
    except SyntaxError:
        return

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".", 1)[0]
                if root in forbidden_modules:
                    violations.append(f"forbidden import `{alias.name}`")

        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            root = module.split(".", 1)[0]
            if root in forbidden_modules:
                violations.append(f"forbidden import from `{module}`")
            for alias in node.names:
                if alias.name in forbidden_names:
                    violations.append(f"forbidden import `{alias.name}`")

        elif isinstance(node, ast.Call):
            name = _call_name(node.func).split(".")[-1]
            if name in forbidden_names:
                violations.append(f"forbidden call `{name}(...)`")
            if name == "StackingClassifier":
                cv_keyword = next(
                    (kw for kw in node.keywords if kw.arg == "cv"),
                    None,
                )
                cv_value = (
                    _constant_value(cv_keyword.value)
                    if cv_keyword is not None
                    else None
                )
                if cv_value != "prefit":
                    violations.append(
                        "StackingClassifier uses internal CV unless "
                        "`cv='prefit'`; prefer VotingClassifier or a single "
                        "estimator."
                    )
            if name in tabpfn_estimators and allowed_tabpfn_kwargs is not None:
                for kw in node.keywords:
                    if kw.arg is not None and kw.arg not in allowed_tabpfn_kwargs:
                        violations.append(
                            f"`{name}` got unsupported keyword `{kw.arg}` "
                            "for the detected installed TabPFN version"
                        )

    if violations:
        unique = list(dict.fromkeys(violations))
        raise RuntimeError(
            "Generated solution violates the no internal CV/HPO/search-loop "
            "constraint: " + "; ".join(unique)
        )


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
        search_eval_mode: str = "inner_val",
        inner_val_size: float = 0.2,
        inner_val_seed: int = 42,
        dependencies=None,
        imports=None,
        tabpfn_mode: str = "off",
        tabpfn_available: bool | None = None,
        tabpfn_version: str | None = None,
        tabpfn_prompt_text: str | None = None,
        tabpfn_eval_source: str = "env",
        tabpfn_signature: str | None = None,
        tabpfn_allowed_kwargs: list[str] | None = None,
    ):
        """
        If openml_task_id is provided, this problem loads the OpenML task
        and uses the official train/test split.
        """

        if tabpfn_mode not in VALID_TABPFN_MODES:
            raise ValueError(
                "tabpfn_mode must be one of: " + ", ".join(VALID_TABPFN_MODES)
            )
        if tabpfn_mode == "explicit" and tabpfn_available is False:
            raise ValueError("tabpfn_mode='explicit' requires tabpfn_available=True")
        if tabpfn_eval_source not in VALID_TABPFN_EVAL_SOURCES:
            raise ValueError(
                "tabpfn_eval_source must be one of: "
                + ", ".join(VALID_TABPFN_EVAL_SOURCES)
            )

        if dependencies is None:
            dependencies = [
                "pandas>=2",
                "scipy>=1.10",
                "scikit-learn>=1.4",
                "openml>=0.14",
                "ConfigSpace>=1.2",
                "smac>=2.1",
            ]
        else:
            dependencies = list(dependencies)

        if tabpfn_mode in {"passive", "explicit"} and tabpfn_available is not False:
            _constrain_sklearn_for_tabpfn(dependencies)

        if (
            tabpfn_mode in {"passive", "explicit"}
            and tabpfn_available is not False
            and tabpfn_eval_source == "pip"
        ):
            _append_tabpfn_dependency(dependencies, tabpfn_version)

        if imports is None:
            imports = "import numpy as np\nimport sklearn\nimport pandas as pd\nimport math\nimport openml\n"

        super().__init__(
            logger=logger,
            training_instances=[],
            test_instances=[],
            name=name,
            eval_timeout=eval_timeout,
            dependencies=dependencies,
            imports=imports,
        )
        self.openml_task_id = openml_task_id
        self.eval_name = None
        self.split_info = {}
        self.le_ = None
        self.cat_enc_ = None
        self.imputer_ = None
        self.tabpfn_mode = tabpfn_mode
        self.tabpfn_available = tabpfn_available
        self.tabpfn_version = tabpfn_version
        self.tabpfn_prompt_text = tabpfn_prompt_text
        self.tabpfn_eval_source = tabpfn_eval_source
        self.tabpfn_signature = tabpfn_signature
        self.tabpfn_allowed_kwargs = tabpfn_allowed_kwargs
        if tabpfn_mode in {"passive", "explicit"} and tabpfn_eval_source == "env":
            self.use_system_site_packages = True
        if search_eval_mode not in {"official_test", "inner_val"}:
            raise ValueError("search_eval_mode must be 'official_test' or 'inner_val'")
        if not (0.0 < inner_val_size < 1.0):
            raise ValueError("inner_val_size must be in (0, 1)")

        self.search_eval_mode = search_eval_mode
        self.inner_val_size = inner_val_size
        self.inner_val_seed = inner_val_seed
        self.evaluation_phase = "search"
        task_type = "classification"
        samples_desc = "an unknown number of samples"
        feats_desc = "an unknown feature schema"

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

            # Keep features
            self.X_all_ = X

            # Label handling depends on task type
            if _is_classification_task(self.task):
                y_codes, y_uniques = pd.factorize(y, sort=True)
                self.y_all_ = y_codes.astype(int)
                self.y_classes_ = list(map(str, y_uniques))
            else:
                # Regression: keep numeric target as-is
                self.y_all_ = np.asarray(y)
                self.y_classes_ = None

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

        self.task_prompt = textwrap.dedent(f"""
        You can use the following Python packages: scikit-learn, numpy, scipy, pandas.
        Design an ML pipeline for a {task_type} task with {samples_desc} and {feats_desc}.
        Write a single Python class:
        - __init__(self, X, y, **hyperparameters)  -> fit exactly once (no CV here)
        - __call__(self, X)
        IMPORTANT CONSTRAINTS (must follow):
        - Do NOT use any internal HPO or CV search (no GridSearchCV, RandomizedSearchCV, BayesSearchCV, Optuna, Hyperopt, skopt, nested AutoML systems, etc.).
        - Do NOT implement your own tuning loops (no KFold/StratifiedKFold loops, no parameter sweeps).
        - Pick ONE primary estimator OBJECT inside the class. This can be either (a) a single sklearn estimator, OR (b) a simple sklearn ensemble wrapper such as VotingClassifier that combines up to 2-3 base estimators.
        - Do NOT do estimator-switching in a param grid.
        - If you use an ensemble wrapper, keep it small (2-3 models max), and keep preprocessing shared (apply preprocessing once, then feed the same transformed X to all base estimators). Do NOT use StackingClassifier with internal CV; use cv='prefit' with already-fitted base estimators or prefer VotingClassifier.
        - If you do preprocessing (e.g., StandardScaler, PCA), assign them to self.* and reuse them in __call__.
        - Expose tunable hyperparameters via __init__ kwargs with sensible defaults; external HPO will tune them.
        """)

        if self.tabpfn_mode == "explicit":
            self.tabpfn_prompt_text = (
                self.tabpfn_prompt_text
                or _make_tabpfn_prompt_text(
                    task_type,
                    self.tabpfn_signature,
                    self.tabpfn_allowed_kwargs,
                )
            )
            self.task_prompt += self.tabpfn_prompt_text

        self.example_prompt = textwrap.dedent("""
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
        """)

        self.format_prompt = textwrap.dedent("""
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
        """)

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

    def _all_official_splits(self):
        return [
            (r, f, s)
            for r in range(self.split_info["n_repeats"])
            for s in range(self.split_info["n_samples"])
            for f in range(self.split_info["n_folds"])
        ]

    def _compile_solution(self, solution: Solution):
        code = solution.code
        name = solution.name

        _validate_generated_code_constraints(code, self.tabpfn_allowed_kwargs)

        safe_globals = {
            "sklearn": sklearn,
            "math": math,
            "random": random,
            "np": np,
            "pd": pd,
        }
        safe_globals = {k: v for k, v in safe_globals.items() if v is not None}

        exec(code, safe_globals)
        if name not in safe_globals:
            raise RuntimeError(f"Class '{name}' not found in generated code.")
        return safe_globals[name]

    def _instantiate_algorithm(self, alg_cls, X_tr, y_tr, config_dict):
        try:
            return alg_cls(X_tr, y_tr, **(config_dict or {}))
        except TypeError:
            return alg_cls(X_tr, y_tr)

    def _score_algorithm(self, alg, X_ev, y_ev, y_tr):
        metric_name = self.eval_name

        if metric_name == "area_under_roc_curve":
            if hasattr(alg, "predict_proba"):
                proba = alg.predict_proba(X_ev)
                if np.ndim(proba) == 2 and proba.shape[1] > 1:
                    return roc_auc_score(y_ev, proba, multi_class="ovr")
                return roc_auc_score(y_ev, np.ravel(proba))

            if hasattr(alg, "decision_function"):
                scores = alg.decision_function(X_ev)
                if np.ndim(scores) == 2 and scores.shape[1] > 1:
                    return roc_auc_score(y_ev, scores, multi_class="ovr")
                return roc_auc_score(y_ev, scores)

            y_pred = alg(X_ev)
            if hasattr(y_pred, "shape") and getattr(y_pred, "ndim", 1) == 2:
                y_pred = np.argmax(y_pred, axis=1)
            elif (
                np.issubdtype(np.asarray(y_pred).dtype, np.floating)
                and len(np.unique(y_tr)) == 2
            ):
                y_pred = (np.asarray(y_pred) >= 0.5).astype(int)

            return accuracy_score(y_ev, y_pred)

        y_pred = alg(X_ev)
        if hasattr(y_pred, "shape") and getattr(y_pred, "ndim", 1) == 2:
            y_pred = np.argmax(y_pred, axis=1)
        elif (
            np.issubdtype(np.asarray(y_pred).dtype, np.floating)
            and len(np.unique(y_tr)) == 2
        ):
            y_pred = (np.asarray(y_pred) >= 0.5).astype(int)

        scorer = self.METRIC_MAP.get(metric_name, accuracy_score)
        if metric_name == "f1":
            return scorer(y_ev, y_pred, average="macro")
        return scorer(y_ev, y_pred)

    def _evaluate_single_split(
        self,
        alg_cls,
        config_dict,
        repeat: int,
        fold: int,
        sample: int,
        split_role: str,
    ):
        if split_role == "search":
            if self.search_eval_mode == "official_test":
                X_tr, X_ev, y_tr, y_ev = self.prepare_split(repeat, fold, sample)
            elif self.search_eval_mode == "inner_val":
                X_tr, X_ev, y_tr, y_ev = self.prepare_inner_validation_split(
                    repeat, fold, sample
                )
            else:
                raise ValueError(f"Unknown search_eval_mode: {self.search_eval_mode}")
        elif split_role == "outer_test":
            X_tr, X_ev, y_tr, y_ev = self.prepare_split(repeat, fold, sample)
        else:
            raise ValueError(f"Unknown split_role: {split_role}")

        alg = self._instantiate_algorithm(alg_cls, X_tr, y_tr, config_dict)
        return self._score_algorithm(alg, X_ev, y_ev, y_tr)

    def _evaluate_many_splits(
        self,
        alg_cls,
        config_dict,
        splits,
        split_role: str,
        solution: Solution | None = None,
        max_failures: int = 3,
    ):
        scores = []
        n_failed = 0
        failures = []

        for repeat, fold, sample in splits:
            try:
                score = self._evaluate_single_split(
                    alg_cls, config_dict, repeat, fold, sample, split_role
                )
                scores.append(float(score))
            except Exception as exc:
                n_failed += 1
                if len(failures) < max_failures:
                    failures.append(
                        {
                            "repeat": repeat,
                            "fold": fold,
                            "sample": sample,
                            "error": _format_solution_exception(exc, solution),
                        }
                    )

        return scores, n_failed, failures

    def _get_outer_split_indices(self, repeat: int, fold: int, sample: int):
        return self.task.get_train_test_split_indices(
            repeat=repeat, fold=fold, sample=sample
        )

    def _preprocess_pair(self, X_tr: pd.DataFrame, X_ev: pd.DataFrame):
        X_tr = X_tr.copy()
        X_ev = X_ev.copy()

        cat_cols = X_tr.select_dtypes(include=["object", "category", "string"]).columns
        if len(cat_cols) > 0:
            enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
            X_tr[cat_cols] = enc.fit_transform(X_tr[cat_cols])
            X_ev[cat_cols] = enc.transform(X_ev[cat_cols])

        bool_cols = X_tr.select_dtypes(include=["bool"]).columns
        if len(bool_cols) > 0:
            X_tr[bool_cols] = X_tr[bool_cols].astype(int)
            X_ev[bool_cols] = X_ev[bool_cols].astype(int)

        imp = SimpleImputer(strategy="most_frequent")
        X_tr = pd.DataFrame(
            imp.fit_transform(X_tr), columns=X_tr.columns, index=X_tr.index
        )
        X_ev = pd.DataFrame(imp.transform(X_ev), columns=X_ev.columns, index=X_ev.index)

        return X_tr, X_ev

    def _prepare_from_indices(self, train_idx, eval_idx):
        X_tr_raw = self.X_all_.iloc[train_idx].copy()
        X_ev_raw = self.X_all_.iloc[eval_idx].copy()
        y_tr = self.y_all_[train_idx]
        y_ev = self.y_all_[eval_idx]

        X_tr, X_ev = self._preprocess_pair(X_tr_raw, X_ev_raw)
        return X_tr, X_ev, y_tr, y_ev

    def prepare_split(self, repeat: int, fold: int, sample: int):
        """
        Official outer train/test split:
        fit preprocessing on the official train fold, evaluate on the official test fold.
        """
        train_idx, test_idx = self._get_outer_split_indices(repeat, fold, sample)
        return self._prepare_from_indices(train_idx, test_idx)

    def prepare_inner_validation_split(self, repeat: int, fold: int, sample: int):
        """
        Split used during the search: the official OpenML train fold is split into
        an inner train part and a validation part. Preprocessing is fitted on the
        inner train part only and the pipeline is scored on the validation part,
        so the official test fold is not used.
        """
        outer_train_idx, _ = self._get_outer_split_indices(repeat, fold, sample)
        outer_train_idx = np.asarray(outer_train_idx)

        stratify = None
        if _is_classification_task(self.task):
            y_outer = self.y_all_[outer_train_idx]
            uniq, counts = np.unique(y_outer, return_counts=True)
            if len(uniq) > 1 and np.all(counts >= 2):
                stratify = y_outer

        split_seed = int(
            self.inner_val_seed + 1000003 * repeat + 10007 * fold + 101 * sample
        )

        try:
            inner_train_idx, inner_val_idx = train_test_split(
                outer_train_idx,
                test_size=self.inner_val_size,
                random_state=split_seed,
                stratify=stratify,
            )
        except ValueError:
            inner_train_idx, inner_val_idx = train_test_split(
                outer_train_idx,
                test_size=self.inner_val_size,
                random_state=split_seed,
                stratify=None,
            )

        return self._prepare_from_indices(inner_train_idx, inner_val_idx)

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

    def _evaluate_search(self, solution: Solution, ioh_dir=""):
        alg_cls = self._compile_solution(solution)

        metric_name = self.eval_name
        is_error_metric = metric_name in {
            "root_mean_squared_error",
            "mean_absolute_error",
        }

        all_splits = self._all_official_splits()

        incumbent_dict = {}
        did_hpo = False

        cs: ConfigurationSpace | None = getattr(solution, "configspace", None)
        if cs is not None:
            rng = np.random.RandomState(42)
            max_hpo_instances = min(12, len(all_splits))
            hpo_instances = all_splits.copy()
            rng.shuffle(hpo_instances)
            hpo_instances = hpo_instances[:max_hpo_instances]

            instance_ids = [f"r{r},f{f},s{s}" for (r, f, s) in hpo_instances]
            inst_feats = {
                iid: [r, f, s] for iid, (r, f, s) in zip(instance_ids, hpo_instances)
            }

            def target(cfg: Configuration, instance: str, seed: int = 0) -> float:
                r_str, f_str, s_str = instance.split(",")
                r = int(r_str[1:])
                f = int(f_str[1:])
                s = int(s_str[1:])
                try:
                    score = self._evaluate_single_split(
                        alg_cls, dict(cfg), r, f, s, split_role="search"
                    )
                except Exception:
                    return 1.0 if not is_error_metric else 1e9

                if is_error_metric:
                    return float(score)
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

        search_scores, n_failed, failures = self._evaluate_many_splits(
            alg_cls,
            incumbent_dict,
            all_splits,
            split_role="search",
            solution=solution,
        )

        mean_score = float(np.mean(search_scores)) if search_scores else float("-inf")
        std_score = float(np.std(search_scores)) if search_scores else float("nan")
        fitness = -mean_score if is_error_metric else mean_score

        solution.add_metadata("search_eval_mode", self.search_eval_mode)
        solution.add_metadata("search_scores", search_scores)
        solution.add_metadata("search_mean", mean_score)
        solution.add_metadata("search_std", std_score)
        solution.add_metadata("search_failed_splits", n_failed)
        solution.add_metadata("search_failures", failures)

        msg = (
            f"SEARCH[{self.search_eval_mode}] {self.eval_name} = "
            f"{mean_score:.4f} ± {std_score:.4f} | "
            f"splits: repeats={self.split_info['n_repeats']}, "
            f"folds={self.split_info['n_folds']}, "
            f"samples={self.split_info['n_samples']} | "
            f"failed_splits={n_failed}"
        )
        if did_hpo:
            msg += f" | HPO used; incumbent={incumbent_dict}"

        error_msg = ""
        if failures:
            error_msg = _format_split_failure(failures[0])
            msg += f"\n{error_msg}"
            if len(failures) > 1:
                msg += f"\nAdditional captured split failures: {len(failures) - 1}"

        solution.set_scores(
            fitness,
            msg,
            error=error_msg if not search_scores else "",
        )
        return solution

    def _evaluate_final(self, solution: Solution, ioh_dir=""):
        alg_cls = self._compile_solution(solution)

        metric_name = self.eval_name
        is_error_metric = metric_name in {
            "root_mean_squared_error",
            "mean_absolute_error",
        }

        incumbent_dict = solution.get_metadata("incumbent") or {}
        all_splits = self._all_official_splits()

        final_scores, n_failed, failures = self._evaluate_many_splits(
            alg_cls,
            incumbent_dict,
            all_splits,
            split_role="outer_test",
            solution=solution,
        )

        mean_score = float(np.mean(final_scores)) if final_scores else float("-inf")
        std_score = float(np.std(final_scores)) if final_scores else float("nan")
        fitness = -mean_score if is_error_metric else mean_score

        solution.add_metadata("selection_fitness", solution.fitness)
        solution.add_metadata("selection_feedback", solution.feedback)
        solution.add_metadata("final_scores", final_scores)
        solution.add_metadata("final_mean", mean_score)
        solution.add_metadata("final_std", std_score)
        solution.add_metadata("final_failed_splits", n_failed)
        solution.add_metadata("final_failures", failures)

        search_mean = solution.get_metadata("search_mean")
        search_std = solution.get_metadata("search_std")
        search_mode = solution.get_metadata("search_eval_mode")

        msg = (
            f"FINAL[official_test] {self.eval_name} = "
            f"{mean_score:.4f} ± {std_score:.4f} | "
            f"splits: repeats={self.split_info['n_repeats']}, "
            f"folds={self.split_info['n_folds']}, "
            f"samples={self.split_info['n_samples']} | "
            f"failed_splits={n_failed}"
        )

        if search_mean is not None:
            msg += (
                f" | selected_by SEARCH[{search_mode}] = "
                f"{search_mean:.4f} ± {search_std:.4f}"
            )

        if incumbent_dict:
            msg += f" | incumbent={incumbent_dict}"

        error_msg = ""
        if failures:
            error_msg = _format_split_failure(failures[0])
            msg += f"\n{error_msg}"
            if len(failures) > 1:
                msg += f"\nAdditional captured split failures: {len(failures) - 1}"

        solution.set_scores(
            fitness,
            msg,
            error=error_msg if not final_scores else "",
        )
        return solution

    def evaluate(self, solution: Solution, test=False, ioh_dir=""):
        if getattr(self, "evaluation_phase", "search") == "final":
            return self._evaluate_final(solution, ioh_dir)
        return self._evaluate_search(solution, ioh_dir)

    def test(self, solution: Solution, ioh_dir=""):
        prev_phase = getattr(self, "evaluation_phase", "search")
        self.evaluation_phase = "final"
        try:
            return self.evaluate(solution, True, ioh_dir)
        finally:
            self.evaluation_phase = prev_phase

    def final_evaluation(self, solution):
        """
        Scores the solution returned by the method on the official test folds.
        The Experiment calls this once after the search. This score does not use
        the evaluation budget and is not written to log.jsonl.
        """
        if isinstance(solution, list):
            return [self.final_evaluation(s) for s in solution]
        metadata = getattr(solution, "metadata", {}) or {}
        feedback = getattr(solution, "feedback", "") or ""
        if "final_mean" in metadata or feedback.startswith("FINAL["):
            return solution

        prev_phase = self.evaluation_phase
        prev_logger = self.logger
        try:
            self.evaluation_phase = "final"
            self.logger = None
            return self(solution, logger=None)
        finally:
            self.evaluation_phase = prev_phase
            self.logger = prev_logger

    def to_dict(self):
        """
        Converts the problem to a dictionary.
        """
        d = {"name": self.name}
        if self.openml_task_id is not None:
            d["openml_task_id"] = self.openml_task_id
            d["metric"] = self.eval_name
            d["search_eval_mode"] = self.search_eval_mode
            d["inner_val_size"] = self.inner_val_size
            d["inner_val_seed"] = self.inner_val_seed
            d["tabpfn_mode"] = self.tabpfn_mode
            d["tabpfn_available"] = self.tabpfn_available
            d["tabpfn_version"] = self.tabpfn_version
            d["tabpfn_eval_source"] = self.tabpfn_eval_source
            d["tabpfn_signature"] = self.tabpfn_signature
            d["tabpfn_allowed_kwargs"] = self.tabpfn_allowed_kwargs
            d["use_system_site_packages"] = self.use_system_site_packages
        return d

    def get_config(self) -> dict[str, Any]:
        task_type = (
            "classification" if _is_classification_task(self.task) else "regression"
        )
        config = {
            "dependencies": self.dependencies,
            "imports": self.imports,
            "task_id": self.openml_task_id,
            "task_type": task_type,
            "search_eval_mode": self.search_eval_mode,
            "inner_val_size": self.inner_val_size,
            "inner_val_seed": self.inner_val_seed,
            "tabpfn_mode": self.tabpfn_mode,
            "tabpfn_version": self.tabpfn_version,
        }
        tags = [Benchmark.AUTOML, PrimaryCategories.PD]
        tags.extend(self.task.class_labels or [])
        return {
            "tags": tags,
            "name": self.name or f"OpenML task {self.openml_task_id}",
            "prompt": self.get_prompt(),
            "minimisation": False,
            "evaluator": "https://github.com/XAI-liacs/BLADE/tree/main/iohblade/benchmarks/automl",
            "config": config,
        }


if __name__ == "__main__":
    aml = AutoML(openml_task_id=13)
    for key, value in aml.get_config().items():
        print(f"------------------------------{key}------------------------------")
        print(value)
