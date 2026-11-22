import contextlib
import copy
import hashlib
import json
import logging
import os
import sys
import time
import traceback
import uuid
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from tqdm import tqdm

from .benchmarks import MA_BBOB
from .envs import EnvironmentSetupError
from .loggers import ExperimentLogger
from .runlog import experiment_logging, log

BLADE_ASCII = r"""
    ____  __    ___    ____  ______
   / __ )/ /   /   |  / __ \/ ____/
  / __  / /   / /| | / / / / __/
 / /_/ / /___/ ___ |/ /_/ / /___
/_____/_____/_/  |_/_____/_____/
"""


def _format_error_counts(counts: dict) -> str:
    """``{"candidate": 3, "harness": 1}`` -> ``"3c 1h"`` (``"-"`` if none)."""
    short = {"candidate": "c", "harness": "h", "timeout": "t"}
    parts = [f"{counts[k]}{short[k]}" for k in short if counts.get(k)]
    return " ".join(parts) or "-"


class Experiment(ABC):
    """
    Abstract class for an entire experiment, running multiple algorithms on multiple problems.
    """

    def __init__(
        self,
        methods: list,
        problems: list,
        runs=5,
        budget=100,
        seeds=None,
        show_stdout=False,
        log_stdout=False,
        exp_logger=None,
        n_jobs=1,
        log_level="INFO",
    ):
        """
        Initializes an experiment with multiple methods and problems.

        Args:
            methods (list): List of method instances.
            problems (list): List of problem instances.
            runs (int): Number of runs for each method.
            budget (int): Number of evaluations per run for each method.
            seeds (list, optional): The exact seeds to use for the runs, len(seeds) overwrites the number of runs if set.
            show_stdout (bool): Whether to show stdout and stderr (standard output) or not.
            log_stdout (bool): If True, capture stdout and stderr to files via the
                logger when output is hidden.
            exp_logger (ExperimentLogger, optiona): The logger object, can be a standard file logger or a WandB or MLFlow logger.
            n_jobs (int): Number of runs to execute in parallel.
            log_level (str): Minimum level of the messages written to
                ``experiment.log`` in the experiment directory and to each
                run's ``run.log`` (DEBUG, INFO, WARNING or ERROR).
        """
        self.id = uuid.uuid4().hex
        self.methods = methods
        self.problems = problems
        self.runs = runs
        self.budget = budget
        if seeds is None:
            self.seeds = np.arange(runs)
        else:
            self.seeds = seeds
            self.runs = len(seeds)
        self.show_stdout = show_stdout
        self.log_stdout = log_stdout
        self.n_jobs = n_jobs
        self.log_level = log_level
        if exp_logger is None:
            exp_logger = ExperimentLogger("results/experiment")
        self.exp_logger = exp_logger

    @classmethod
    def from_yaml(cls, source, *, overrides=None, env=None) -> "Experiment":
        """Create an experiment from a YAML configuration file.

        Args:
            source: Path of the YAML file (or the configuration as a dict).
            overrides: ``key.path=value`` overrides, e.g.
                ``["experiment.runs=1"]``.
            env: Variables for ``${env:...}`` placeholders (default: a
                ``.env`` file next to the YAML file plus ``os.environ``).

        Returns:
            Experiment: The configured experiment, ready to be called.

        Raises:
            iohblade.config.ConfigError: If the configuration is invalid.
        """
        from .config import load_experiment

        return load_experiment(source, overrides=overrides, env=env)

    def _clear_console(self) -> None:
        """Clear the console using ANSI escape codes."""
        print("\033c", end="")

    def _status_tokens(self):
        enc = (getattr(sys.__stdout__, "encoding", "") or "").lower()
        if "utf" in enc or "65001" in enc:  # utf-8 or Windows UTF-8 codepage
            return {"done": "✅", "running": "🔄", "pending": "⏳", "failed": "❌"}
        else:
            return {
                "done": "[DONE]",
                "running": "[RUN]",
                "pending": "[WAIT]",
                "failed": "[FAIL]",
            }

    def _print_run_overview(self) -> None:
        """Pretty print the planned runs and their status."""
        runs = getattr(self.exp_logger, "progress", {}).get("runs", [])
        tokens = self._status_tokens()
        rows = [("Method", "LLM", "Problem", "Seed", "Evals", "Best", "Errors", "")]
        for r in runs:
            if r.get("end_time"):
                status = tokens["done"]
            elif r.get("error"):
                status = tokens["failed"]
            elif r.get("start_time"):
                status = tokens["running"]
            else:
                status = tokens["pending"]
            best = r.get("best_fitness")
            counts = r.get("error_counts") or {}
            rows.append(
                (
                    str(r["method_name"]),
                    str(r.get("llm_name") or "-"),
                    str(r["problem_name"]),
                    str(r["seed"]),
                    f"{r.get('evaluations', 0)}/{r.get('budget', '?')}",
                    "-" if best is None else f"{best:.6g}",
                    _format_error_counts(counts),
                    status,
                )
            )
        widths = [max(len(row[c]) for row in rows) for c in range(len(rows[0]) - 1)]
        lines = ["Run overview:"]
        for k, row in enumerate(rows):
            cells = [cell.ljust(widths[c]) for c, cell in enumerate(row[:-1])]
            lines.append("  ".join(cells + [row[-1]]).rstrip())
            if k == 0:
                lines.append("-" * len(lines[-1]))
        if any((r.get("error_counts") or {}) for r in runs):
            lines.append(
                "Errors: evaluations that failed in the generated code (c), "
                "in the harness (h) or by timeout (t)."
            )
        print("\n".join(lines))

    def _refresh_console(self) -> None:
        """Clear the console and show the banner and run overview."""
        with contextlib.redirect_stdout(sys.__stdout__):
            self._clear_console()
            self._print_welcome_message()
            self._print_run_overview()
            sys.__stdout__.flush()

    def _print_welcome_message(self) -> None:
        """Print a welcome banner with instructions."""
        message = (
            f"\n{BLADE_ASCII}\n"
            "Welcome to BLADE!\n"
            "You can inspect this experiment in your browser by running:\n"
            "    uv run iohblade-webapp\n\n"
            "While BLADE hides most output from experiments by default, "
            "some logs or warnings may still appear.\n"
        )
        print(message)

    def __call__(self):
        """
        Runs the experiment by executing each method on each problem.

        Raises:
            iohblade.envs.EnvironmentSetupError: If the evaluation
                environment of a problem cannot be built (before any run
                starts; all failing problems are reported).
            RuntimeError: If one or more runs failed (after all other runs
                finished and were recorded).
        """
        directory = getattr(self.exp_logger, "dirname", None)
        with experiment_logging(
            directory, self.log_level, quiet=not self.show_stdout
        ) as self._run_logs:
            return self._run_all()

    def _run_all(self):
        total_runs = len(self.problems) * len(self.methods) * len(self.seeds)
        if hasattr(self.exp_logger, "start_progress"):
            self.exp_logger.start_progress(
                self.id,
                total_runs,
                methods=self.methods,
                problems=self.problems,
                seeds=self.seeds,
                budget=self.budget,
            )
        log.info(
            "Experiment started: %d methods x %d problems x %d seeds",
            len(self.methods),
            len(self.problems),
            len(self.seeds),
        )
        self._prepare_environments()
        if not self.show_stdout:
            self._refresh_console()
        else:
            self._print_welcome_message()
            self._print_run_overview()
        tasks = {}  # future -> (method, problem, logger, seed)
        with ThreadPoolExecutor(max_workers=self.n_jobs) as executor:
            for problem in self.problems:
                for method in self.methods:
                    llm_name = self.exp_logger._get_clean_llm_model_name(method.llm)
                    for seed in self.seeds:
                        np.random.seed(seed)
                        if hasattr(
                            self.exp_logger, "is_run_pending"
                        ) and not self.exp_logger.is_run_pending(
                            method, llm_name, problem, seed
                        ):
                            continue

                        m_copy = copy.deepcopy(method)
                        p_copy = copy.deepcopy(problem)
                        logger = self.exp_logger.open_run(
                            m_copy, p_copy, self.budget, seed
                        )

                        future = executor.submit(
                            self._run_single,
                            m_copy,
                            p_copy,
                            logger,
                            seed,
                        )
                        tasks[future] = (m_copy, p_copy, logger, seed)

            failures = []
            for fut in as_completed(tasks):
                method, problem, logger, seed = tasks[fut]
                error = fut.exception()
                if error is not None:
                    # Record the failure and keep going: raising here would
                    # stop every other run from being recorded as finished.
                    failures.append(
                        self._record_failure(method, problem, logger, seed, error)
                    )
                else:
                    self.exp_logger.add_run(
                        method,
                        problem,
                        method.llm,
                        fut.result(),
                        log_dir=logger.dirname,
                        seed=seed,
                    )

                if not self.show_stdout:
                    self._refresh_console()
                else:
                    self._print_run_overview()
        for problem in self.problems:
            problem.cleanup()
        self._print_summary(failures, len(tasks))
        if failures:
            raise RuntimeError(
                f"{len(failures)} of {len(tasks)} runs failed (the others were "
                "recorded; rerunning retries only the failed ones):\n"
                + "\n".join(line for line, _ in failures)
            )
        return

    def _prepare_environments(self) -> None:
        """Build every problem's evaluation environment before any run starts.

        Raises:
            EnvironmentSetupError: Listing every problem whose environment
                could not be built; the full installer output of each is
                written to ``<experiment>/env-errors/<problem>.txt``.
        """
        failed = []
        for problem in self.problems:
            try:
                problem._ensure_env()
            except Exception as e:
                failed.append((problem, e))
        if not failed:
            return
        directory = getattr(self.exp_logger, "dirname", None)
        lines = []
        used: set[str] = set()
        for problem, error in failed:
            text = f"{type(error).__name__}: {error}"
            full = getattr(error, "full_output", None)
            location = ""
            if directory:
                safe = (
                    "".join(
                        c if c.isalnum() or c in "-_." else "_" for c in problem.name
                    )
                    or "problem"
                )
                name, k = safe, 1
                while name in used:  # names that normalise to the same file
                    k += 1
                    name = f"{safe}-{k}"
                used.add(name)
                path = os.path.join(directory, "env-errors", f"{name}.txt")
                try:
                    os.makedirs(os.path.dirname(path), exist_ok=True)
                    with open(path, "w") as f:
                        f.write(text + "\n")
                        if full:
                            f.write("\nFull installer output:\n" + full + "\n")
                    location = f" (full output: {path})"
                except OSError:
                    pass
            first = str(error).strip().splitlines()[0] if str(error).strip() else text
            lines.append(f"  - {problem.name}: {first}{location}")
            log.error("Evaluation environment of %s failed: %s", problem.name, text)
        raise EnvironmentSetupError(
            f"Could not build the evaluation environment of {len(failed)} "
            "problem(s); no runs were started:\n"
            + "\n".join(lines)
            + "\nRun `iohblade check` on the experiment for details."
        )

    def _print_summary(self, failures, n_started: int) -> None:
        """Print what happened, grouped by error type, and what to do next."""
        runs = getattr(self.exp_logger, "progress", {}).get("runs", [])
        totals = {}
        for r in runs:
            for kind, n in (r.get("error_counts") or {}).items():
                totals[kind] = totals.get(kind, 0) + n
        evaluations = sum(r.get("evaluations", 0) for r in runs)
        lines = [
            "",
            f"Summary: {n_started - len(failures)} of {n_started} runs finished"
            + (f", {len(failures)} failed." if failures else "."),
        ]
        if failures:
            groups: dict[str, list] = {}
            for line, error in failures:
                groups.setdefault(type(error).__name__, []).append((line, error))
            lines.append("Failed runs:")
            for name, items in groups.items():
                lines.append(f"  {name} ({len(items)} run(s)):")
                for line, _ in items:
                    lines.append(f"    - {line}")
            lines.append("  Rerunning the experiment retries only the failed runs.")
        if totals and evaluations:
            parts = []
            labels = {
                "candidate": "in the generated code",
                "harness": "in the evaluation harness",
                "timeout": "by timeout",
            }
            for kind in ("candidate", "harness", "timeout"):
                if totals.get(kind):
                    parts.append(f"{totals[kind]} {labels[kind]}")
            lines.append(
                f"Evaluations: {evaluations}, of which failed: "
                + ", ".join(parts)
                + "."
            )
            if totals.get("harness"):
                lines.append(
                    "  Harness errors are not caused by the LLM's code: check the "
                    "problem (see the runs' run.log)."
                )
        directory = getattr(self.exp_logger, "dirname", None)
        if directory:
            lines.append(
                f"Logs: {os.path.join(directory, 'experiment.log')} and run.log "
                "in each run's folder."
            )
        text = "\n".join(lines)
        print(text, file=sys.__stdout__, flush=True)
        log.info(text.strip())

    def _record_failure(self, method, problem, logger, seed, error) -> str:
        """Writes a failed run's traceback to its log dir and marks it failed.

        Returns:
            tuple: One summary line for the end-of-experiment report, and
            the exception.
        """
        trace = "".join(traceback.format_exception(error))
        error_path = os.path.join(logger.dirname, "error.txt")
        try:
            with open(error_path, "w") as f:
                f.write(trace)
        except OSError:
            error_path = "(could not write error.txt)"
        if hasattr(self.exp_logger, "mark_run_failed"):
            self.exp_logger.mark_run_failed(
                method, problem, seed, f"{type(error).__name__}: {error}"
            )
        line = (
            f"{method.name} / {problem.name} / seed {seed}: "
            f"{type(error).__name__}: {error} (traceback: {error_path})"
        )
        log.error("Run failed: %s\n%s", line, trace)
        # sys.__stderr__: worker threads may have redirected sys.stderr.
        print(f"Run failed: {line}", file=sys.__stderr__, flush=True)
        return line, error

    def _run_single(self, method, problem, logger, seed):
        router = getattr(self, "_run_logs", None)
        if router is not None:
            router.register(os.path.join(logger.dirname, "run.log"))
        log.info(
            "Run started: %s / %s / %s / seed %s",
            method.name,
            getattr(method.llm, "model", "?"),
            problem.name,
            seed,
        )
        start = time.monotonic()
        try:
            result = self._run_single_inner(method, problem, logger, seed)
        except BaseException:
            log.exception("Run raised an exception")
            raise
        else:
            log.info(
                "Run finished in %.1fs; best fitness %s",
                time.monotonic() - start,
                getattr(result, "fitness", "?"),
            )
            return result
        finally:
            if router is not None:
                router.unregister()

    def _run_single_inner(self, method, problem, logger, seed):
        np.random.seed(seed)
        method.llm.set_logger(logger)
        logger.log_stdout = self.log_stdout
        if hasattr(logger, "start_run"):
            logger.start_run(method.llm)
        if self.show_stdout:
            problem._ensure_env()
            result = method(problem)
        elif self.log_stdout:
            import io

            stdout_buf = io.StringIO()
            stderr_buf = io.StringIO()
            with contextlib.redirect_stdout(stdout_buf):
                with contextlib.redirect_stderr(stderr_buf):
                    problem._ensure_env()
                    result = method(problem)
            logger.log_output(stdout_buf.getvalue(), stderr_buf.getvalue(), append=True)
            if getattr(problem, "_last_stdout", "") or getattr(
                problem, "_last_stderr", ""
            ):
                logger.log_output(
                    getattr(problem, "_last_stdout", ""),
                    getattr(problem, "_last_stderr", ""),
                    append=True,
                )
        else:
            with contextlib.redirect_stdout(None):
                with contextlib.redirect_stderr(None):
                    problem._ensure_env()
                    result = method(problem)
        # If the problem keeps a separate test set (e.g. AutoML), score the result on it.
        if hasattr(problem, "final_evaluation"):
            result = problem.final_evaluation(result)
        if hasattr(logger, "finish_run"):
            logger.finish_run(result)
        success = self._log_data(method, problem, logger)
        if not success:
            log.warning("Some configuration files could not be written.")
        return result

    def _get_hash(self, config: dict) -> str:
        "Provides a identifier for a problem/method/llm desciptor. Helps declutter BLADE_db."
        json_data = json.dumps(
            config,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
        return hashlib.sha256(json_data).hexdigest()

    @staticmethod
    def _add_custom_source(obj, config):
        """Add where a custom component's code came from (see iohblade.config)."""
        source = getattr(obj, "_blade_source", None)
        if source and isinstance(config, dict):
            config["custom_source"] = source
        return config

    def _log_data(self, method, problem, logger) -> bool:
        root = logger.get_log_dir()
        success = True
        # method
        method_path = os.path.join(root, "method.json")
        method_config = self._add_custom_source(method, method.get_config())
        method_config["hash"] = self._get_hash(method_config)
        method_success = self._export_config_data(method_config, method_path)
        success = method_success and success

        # problem
        problem_path = os.path.join(root, "problem.json")
        problem_config = self._add_custom_source(problem, problem.get_config())
        problem_config["hash"] = self._get_hash(problem_config)
        problem_success = self._export_config_data(problem_config, problem_path)
        success = problem_success and success

        # llm
        llm_path = os.path.join(root, "llm.json")
        llm_config = method.llm.get_config()
        if llm_config and isinstance(llm_config, list):
            self._add_custom_source(method.llm, llm_config[0])
        for llm in llm_config:
            llm["hash"] = self._get_hash(llm)
        llm_success = self._export_config_data(llm_config, llm_path)
        success = llm_success and success

        return success

    def _export_config_data(self, config: dict, location: str) -> bool:
        """
        Exports configuration data to associated data, for each method, problem and LLM object, configuration
        using `get_config` function and exports it to the provided location.
        ## Parameters:
            `config: dict`: the configuration dictionary that needs to be exported,
            `location: str`: Full location of the file where the configuration needs to be written.

        ## Returns:
            `bool`: Truth about whether the data was exported.
        """
        try:
            with open(location, "w") as f:
                data = json.dumps(config, indent=4, default=str)
                f.write(data)
        except Exception as e:
            print(f"Error logging configuration {config} into file {location}: {e}.")
            return False
        return True


class MA_BBOB_Experiment(Experiment):
    def __init__(
        self,
        methods: list,
        show_stdout=False,
        log_stdout=False,
        runs=5,
        budget=100,
        seeds=None,
        dims=[2, 5],
        budget_factor=2000,
        exp_logger=None,
        n_jobs=1,
        **kwargs,
    ):
        """
        Initializes an experiment on MA-BBOB.

        Args:
            methods (list): List of method instances.
            show_stdout (bool): Whether to show stdout and stderr (standard output) or not.
            log_stdout (bool): If True, capture stdout and stderr to files via the
                logger when output is hidden.
            runs (int): Number of runs for each method.
            budget (int): Number of algorithm evaluations per run per method.
            seeds (list, optional): Seeds for each run.
            dims (list): List of problem dimensions.
            budget_factor (int): Budget factor for the problems.
            **kwargs: Additional keyword arguments for the MA_BBOB problem.
            exp_logger (ExperimentLogger): The logger to store the data.
            n_jobs (int): Number of runs to execute in parallel.
        """
        super().__init__(
            methods,
            [
                MA_BBOB(
                    dims=dims,
                    budget_factor=budget_factor,
                    name="MA_BBOB",
                    **kwargs,
                )
            ],
            runs=runs,
            budget=budget,
            seeds=seeds,
            show_stdout=show_stdout,
            log_stdout=log_stdout,
            exp_logger=exp_logger,
            n_jobs=n_jobs,
        )
