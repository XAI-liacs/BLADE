import inspect
import json
import multiprocessing
import os
import shutil
import subprocess
import tempfile
import traceback
import uuid
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Optional

import cloudpickle
import numpy as np

# Standard packages installed in every evaluation environment
BASE_DEPENDENCIES = [
    "numpy>=2",
    "cloudpickle>=3.1.0,<4",
    "joblib>=1.4.2,<2",
]

import copy
import re

from .solution import Solution
from .utils import HarnessError, TimeoutException


def simplify_subprocess_error(stderr: str, solution=None):
    """Parse a Python traceback string and produce a concise error summary.

    The line number and the offending line of ``solution.code`` are only
    reported for the last traceback frame that lies in the candidate code
    (filename ``"<string>"`` or the solution's name). Frames in other files
    (numpy, the benchmark, ...) have line numbers that do not refer to the
    candidate code, so then only the exception type and message are given.

    Args:
        stderr (str): Standard error of the evaluation subprocess.
        solution (Solution, optional): The evaluated solution.

    Returns:
        str: The summary.
    """
    if not stderr:
        return "Unknown error."

    lines = stderr.strip().splitlines()
    exc_match = re.search(r"([A-Za-z_][\w.]*(?:Error|Exception)): (.*)", lines[-1])
    if not exc_match:
        # fallback: just return the whole output
        return stderr.strip()
    exc_type, exc_msg = exc_match.groups()

    names = {"<string>"}
    sol_name = getattr(solution, "name", "") if solution is not None else ""
    if sol_name:
        names.add(sol_name)
    frames = [
        m
        for m in re.finditer(r'File "(.*?)", line (\d+), in (.+)', stderr)
        if m.group(1) in names
    ]
    if not frames:
        return f"{exc_type}: {exc_msg}"

    last = frames[-1]
    line_no = int(last.group(2))
    func = last.group(3).strip()

    code_line = ""
    code = getattr(solution, "code", None) if solution is not None else None
    if isinstance(code, str):
        code_lines = code.splitlines()
        if 1 <= line_no <= len(code_lines):
            code_line = code_lines[line_no - 1].strip()

    msg = (
        f"In the code, line {line_no}, in {func}, the following error "
        f"occurred:\n{exc_type}: {exc_msg}"
    )
    if code_line:
        msg += f"\nOn line: {code_line}"
    return msg


# Prefix of the feedback given for evaluation errors that happen before the
# generated code is touched (see ``_build_eval_script``).
HARNESS_ERROR_PREFIX = (
    "Evaluation harness error (raised before the generated code was used, "
    "so it is a problem with the benchmark or its evaluation environment, "
    "not with the generated code):\n"
)

# Code that marks an evaluation as failing in the harness. It wraps the
# solution so that the first access to ``solution.code`` is recorded; an
# uncaught exception raised before that access cannot have been caused by
# the generated code.
_HARNESS_TRACKER = """
import sys as _blade_sys

_blade_code_used = [False]


def _blade_excepthook(exc_type, exc, tb):
    if not _blade_code_used[0]:
        try:
            with open(_blade_harness_marker, "w") as _f:
                _f.write("harness")
        except OSError:
            pass
    _blade_sys.__excepthook__(exc_type, exc, tb)


_blade_sys.excepthook = _blade_excepthook


def _blade_track(obj):
    base = type(obj)

    class _BladeTracked(base):
        def __getattribute__(self, name):
            if name == "code":
                _blade_code_used[0] = True
            return base.__getattribute__(self, name)

    try:
        obj.__class__ = _BladeTracked
    except TypeError:  # e.g. classes with __slots__: give up tracking
        _blade_code_used[0] = True
        return obj, base, None
    return obj, base, _BladeTracked


def _blade_untrack(obj, base, tracked):
    if tracked is not None and type(obj) is tracked:
        obj.__class__ = base
    return obj
"""


def _build_eval_script(
    imports_block, problem_pickle, solution_pickle, result_pickle, marker_path
) -> str:
    """Python source of the script that evaluates one solution in the venv.

    ``from __future__`` imports in ``imports_block`` are moved to the top,
    the rest of the block runs after the harness error tracker is installed
    so that failing imports (and syntax errors in the block) are reported as
    harness errors.
    """
    future_lines, other_lines = [], []
    for line in (imports_block or "").splitlines():
        if line.strip().startswith("from __future__"):
            future_lines.append(line)
        else:
            other_lines.append(line)
    parts = []
    if future_lines:
        parts.append("\n".join(future_lines))
    parts.append(f"_blade_harness_marker = {json.dumps(str(marker_path))}")
    parts.append(_HARNESS_TRACKER)
    if other_lines:
        # Compiled separately so that a syntax error in the block is raised
        # at run time, after the tracker is installed (a syntax error in the
        # script itself would stop Python before any line runs).
        block = "\n".join(other_lines)
        parts.append(f"exec(compile({block!r}, '<problem.imports>', 'exec'))")
    parts.append(
        "import cloudpickle as cp\n"
        + "import os, json\n"
        + f"problem_path = {json.dumps(str(problem_pickle))}\n"
        + f"solution_path = {json.dumps(str(solution_pickle))}\n"
        + f"result_path  = {json.dumps(str(result_pickle))}\n"
        + "problem=cp.load(open(problem_path,'rb'))\n"
        + "solution=cp.load(open(solution_path,'rb'))\n"
        + "solution, _blade_base, _blade_tracked = _blade_track(solution)\n"
        + "result=problem.evaluate(solution)\n"
        + "_blade_untrack(solution, _blade_base, _blade_tracked)\n"
        + "result = _blade_untrack(result, _blade_base, _blade_tracked)\n"
        + "with open(result_path,'wb') as f:\n"
        + "    cp.dump(result, f)\n"
    )
    return "\n".join(parts)


def _eval_pythonpath(problem, existing: str = "") -> str:
    """PYTHONPATH for the evaluation subprocess.

    The directory containing the ``iohblade`` package comes first, followed
    by the problem's ``extra_pythonpath`` entries (e.g. the folder of a custom
    problem module) and any PYTHONPATH already set. With
    ``problem.isolated_pythonpath`` only iohblade itself is exposed (for a pip
    installation the folder containing it is the whole site-packages) and an
    existing PYTHONPATH is dropped.
    """
    if getattr(problem, "isolated_pythonpath", False):
        from .envs import iohblade_only_path

        entries = [str(iohblade_only_path())]
        existing = ""  # nothing from BLADE's own environment
    else:
        entries = [str(Path(__file__).resolve().parents[1])]
    for path in getattr(problem, "extra_pythonpath", None) or []:
        path = str(Path(path).resolve())
        if path not in entries:
            entries.append(path)
    if existing:
        entries.append(existing)
    return os.pathsep.join(entries)


def _harness_feedback(stderr: str, max_lines: int = 20) -> str:
    """Feedback for a harness error: the tail of the traceback."""
    lines = (stderr or "").strip().splitlines()
    tail = "\n".join(lines[-max_lines:]) if lines else "Unknown error."
    return HARNESS_ERROR_PREFIX + tail


def evaluate_in_subprocess(problem, conn, solution):
    """Evaluate a solution in a dedicated virtual environment."""
    proc = None
    temp_files = []
    started = False  # True once the evaluation script has finished
    try:
        env_path = problem._env_path
        python_bin = problem._python_bin

        # Unique names per evaluation: runs executed in parallel (n_jobs > 1)
        # share env_path, and fixed names let one evaluation read another's
        # half-written problem or run another's script.
        token = uuid.uuid4().hex
        problem_pickle = env_path / f"problem_{token}.pkl"
        solution_pickle = env_path / f"solution_{token}.pkl"
        result_pickle = Path(tempfile.gettempdir()) / f"blade_result_{token}.pkl"
        script_path = env_path / f"{token}_run_eval.py"
        harness_marker = Path(tempfile.gettempdir()) / f"blade_harness_{token}.txt"
        temp_files = [
            problem_pickle,
            solution_pickle,
            result_pickle,
            script_path,
            harness_marker,
        ]
        problem_copy = copy.deepcopy(problem)
        problem_copy.logger = None
        # Always write the problem, its state can change between evaluations.
        with open(problem_pickle, "wb") as f:
            cloudpickle.dump(problem_copy, f)
        with open(solution_pickle, "wb") as f:
            cloudpickle.dump(solution, f)

        imports_block = getattr(problem, "imports", "")
        script_path.write_text(
            _build_eval_script(
                imports_block,
                problem_pickle,
                solution_pickle,
                result_pickle,
                harness_marker,
            )
        )

        env = os.environ.copy()
        env["PYTHONPATH"] = _eval_pythonpath(problem, env.get("PYTHONPATH", ""))

        proc = subprocess.Popen(
            [str(python_bin), str(script_path)],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        try:
            stdout, stderr = proc.communicate(timeout=problem.eval_timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
            stdout, stderr = proc.communicate()
            conn.send(
                {
                    "error": f"Evaluation timed out after {problem.eval_timeout} seconds.",
                    "stdout": stdout,
                    "stderr": stderr,
                }
            )
            return

        started = True
        if proc.returncode != 0:
            if harness_marker.exists():
                conn.send(
                    {
                        "error": _harness_feedback(stderr),
                        "harness_error": True,
                        "stdout": stdout,
                        "stderr": stderr,
                    }
                )
                return
            error_msg = simplify_subprocess_error(stderr, solution)
            conn.send({"error": error_msg, "stdout": stdout, "stderr": stderr})
            return

        with open(result_pickle, "rb") as f:
            result = cloudpickle.load(f)
        conn.send({"result": result, "stdout": stdout, "stderr": stderr})

    except Exception as e:
        if not started:
            # Preparing or launching the evaluation failed (e.g. the problem
            # cannot be pickled, or the environment's Python is missing).
            conn.send(
                {
                    "error": _harness_feedback("".join(traceback.format_exception(e))),
                    "harness_error": True,
                    "stdout": "",
                    "stderr": "",
                }
            )
            return
        # Raised after the candidate script finished (e.g. a missing or
        # unreadable result pickle): the traceback lies in this file, so its
        # line numbers say nothing about the candidate code.
        error = f"{type(e).__name__}: {e}"
        conn.send(
            {
                "error": error,
                "stdout": "",
                "stderr": "",
            }
        )
    finally:
        if proc and proc.poll() is None:
            proc.kill()
            proc.communicate()
        for path in temp_files:
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass
        conn.close()


class Problem(ABC):
    """
    Abstract problem class.

    Attributes:
        minimisation (bool): Whether lower fitness is better. Subclasses set
            it in their constructor; the default is maximisation.
    """

    minimisation: bool = False
    # Whether the problem can tune a solution's hyperparameters (its
    # ``configspace``) with SMAC; see enable_hpo and iohblade.hpo.
    supports_hpo: bool = False
    # Whether tuning is enabled (set with enable_hpo).
    hpo: bool = False

    def enable_hpo(self, enabled: bool = True) -> None:
        """Enable (or disable) hyperparameter tuning of solutions.

        Enabling adds :data:`iohblade.hpo.HPO_DEPENDENCIES` (smac and the
        packages it needs) to ``self.dependencies``; disabling removes the
        ones enabling added. The dependencies are part of the evaluation
        environment's cache key, so the problem then uses an environment
        with (or without) smac. Call it before the environment is built; if
        it was built already, it is rebuilt at the next evaluation.

        Args:
            enabled: True to enable tuning, False to disable it.

        Raises:
            ValueError: If ``enabled`` is True and the problem does not
                support tuning (``supports_hpo`` is False).
        """
        from .hpo import HPO_DEPENDENCIES

        if enabled and not self.supports_hpo:
            raise ValueError(
                f"Problem '{getattr(self, 'name', type(self).__name__)}' "
                f"({type(self).__name__}) does not support hyperparameter "
                "tuning (supports_hpo is False)."
            )
        deps = list(getattr(self, "dependencies", None) or [])
        added = list(getattr(self, "_hpo_added_dependencies", []))
        if enabled:
            for dep in HPO_DEPENDENCIES:
                if dep not in deps:
                    deps.append(dep)
                    added.append(dep)
        else:
            deps = [d for d in deps if d not in added]
            added = []
        if deps != list(getattr(self, "dependencies", None) or []):
            if getattr(self, "_env_path", None) is not None:
                self.cleanup()  # a private environment is deleted
                self._env_path = None
                self._python_bin = None
        self.dependencies = deps
        self._hpo_added_dependencies = added
        self.hpo = bool(enabled)

    def failure_fitness(self) -> float:
        """Fitness given to a failed evaluation: the worst possible value.

        ``inf`` for a minimisation problem, ``-inf`` otherwise, so a timeout
        or crash can never be mistaken for a good solution.
        """
        from .fitness import worst_fitness

        return worst_fitness(bool(getattr(self, "minimisation", False)))

    def __init__(
        self,
        logger=None,
        training_instances=None,
        test_instances=None,
        name="Problem",
        eval_timeout=6000,
        dependencies=None,
        imports=None,
    ):
        """
        Initializes a problem instance with logging and dataset references.

        Args:
            logger (Logger, optional): Logger object for tracking solutions.
            training_instances (list, optional): List of training problem instances.
            test_instances (list, optional): List of test problem instances.
            name (str, optional): Name of the problem.
            eval_timeout (int, optional): Number of seconds before a timeout error is raised.
            budget (int): number of algorithms are allowed to be generated per run.
            dependencies (list, optional): a list of pypi packages to install before evaluation.
            imports (string, optional): the python string to manage imports in the evaluation file.
        """
        self.logger = logger
        self.logger_dir = ""
        self.training_instances = training_instances if training_instances else []
        self.test_instances = test_instances if test_instances else []
        self.task_prompt = "Write the problem description part here."
        self.example_prompt = "Write an example code here."
        self.format_prompt = "Write the format description part here."
        self.name = name
        self.eval_timeout = eval_timeout
        # If True, evaluations can also use the packages installed where BLADE runs.
        self.use_system_site_packages = False
        # Combine the base dependencies with any problem specific ones
        self.dependencies = BASE_DEPENDENCIES.copy()
        if dependencies:
            self.dependencies.extend(dependencies)
        if imports is None:
            self.imports = "import numpy as np\n"
        else:
            self.imports = imports

        # Extra directories added to PYTHONPATH of the evaluation subprocess,
        # e.g. the folder of a problem defined outside the iohblade package.
        self.extra_pythonpath: list[str] = []
        # Abort the run (HarnessError) after this many consecutive evaluations
        # that failed before the generated code was used. None disables it.
        self.max_harness_errors: int | None = 3
        # Share a cached evaluation environment with problems that have the
        # same dependencies (see iohblade.envs). False: a private one.
        self.env_cache: bool = True
        # Only make iohblade (not the rest of BLADE's own environment)
        # importable in the evaluation process.
        self.isolated_pythonpath: bool = False
        self._consecutive_harness_errors = 0

        # Path to the virtual environment used for evaluations
        self._env_path: Path | None = None
        self._python_bin: Path | None = None

        # These settings are required for EoH, adapt them based on your problem.
        # The function name, inputs, and outputs should match the expected format.
        # For example, if your problem requires a function that takes a function, budget, and dimension,
        # and returns the optimal fitness and solution, set them accordingly.
        self.func_name = "__call__"
        self.init_inputs = ["budget", "dim"]
        self.func_inputs = ["func"]
        self.func_outputs = ["f_opt", "x_opt"]

    def __call__(self, solution: Solution, logger=None):
        """
        Evaluates a solution on training instances and updates its fitness and feedback.

        Args:
            solution (Solution): Solution object to be evaluated.
            logger (RunLogger, optional): The RunLogger object attached to the problem to keep track of evaluations.

        Returns:
            Solution: The evaluated solution with updated fitness and scores.
        """
        if logger != None:
            print("LOGGER is NOT NONE (UNEXPECTED)")
            self.logger = logger

        self._warn_unused_configspace(solution)

        if self.logger != None:
            if self.logger.budget_exhausted():
                solution.set_scores(
                    self.failure_fitness(),
                    feedback="Budget is exhausted.",
                    error="Budget is exhausted.",
                )
                return solution  # Return early if budget is exhausted

        # solution = self.evaluate(solution) #old fashioned way
        # Else create a new process for evaluation with timeout
        stdout = ""
        stderr = ""
        self._last_stdout = ""
        self._last_stderr = ""
        process: multiprocessing.Process | None = None
        parent_conn = None
        child_conn = None
        harness_error = False
        try:
            try:
                self._ensure_env()
            except Exception:
                harness_error = True
                raise
            (
                parent_conn,
                child_conn,
            ) = multiprocessing.Pipe()  # Create pipe for communication
            process = multiprocessing.Process(
                target=evaluate_in_subprocess, args=(self, child_conn, solution)
            )
            process.start()
            child_conn.close()  # Only the child writes to the pipe.
            # Read before joining, otherwise the child blocks in send() when the
            # result is larger than the pipe buffer.
            received = False
            if parent_conn.poll(
                timeout=self.eval_timeout + 60
            ):  # We allow 1 minute for setting up the environment.
                try:
                    result = parent_conn.recv()
                    received = True
                except EOFError:
                    pass
            elif process.is_alive():
                raise TimeoutException(
                    f"Evaluation timed out after {self.eval_timeout} seconds."
                )
            process.join(timeout=60)

            if received:
                if isinstance(result, dict):
                    stdout = result.get("stdout", "")
                    stderr = result.get("stderr", "")
                    if "error" in result:
                        err = result["error"]
                        harness_error = bool(result.get("harness_error"))
                        solution.set_scores(
                            self.failure_fitness(),
                            feedback=err,
                            error=err,
                        )
                    else:
                        data = result.get("result")
                        if isinstance(data, Solution):
                            solution = data
                        elif isinstance(data, str):
                            solution.set_scores(
                                self.failure_fitness(),
                                feedback=data,
                                error=data,
                            )
                        else:
                            raise Exception("No Solution object or string returned.")
                elif isinstance(result, Exception):
                    raise result
                elif isinstance(result, Solution):
                    solution = result
                elif isinstance(result, str):
                    solution.set_scores(
                        self.failure_fitness(),
                        feedback=result,
                        error=result,
                    )
                else:
                    raise Exception("No Solution object or string returned.")
            else:
                raise Exception("Evaluation failed without an exception.")
        except Exception as e:
            msg = f"{e}"
            if harness_error:
                msg = HARNESS_ERROR_PREFIX + msg
            solution.set_scores(
                self.failure_fitness(),
                feedback=msg,
                error=msg,
            )
        finally:
            if process is not None:
                if process.is_alive():
                    process.kill()
                process.join()
            if parent_conn is not None:
                parent_conn.close()
            if child_conn is not None:
                child_conn.close()

        self._last_stdout = stdout
        self._last_stderr = stderr
        if self.logger is not None:
            self.logger.log_individual(solution)
        self._track_harness_error(harness_error, solution)
        return solution

    def _warn_unused_configspace(self, solution: Solution) -> None:
        """Warn (once per problem) that a configuration space is not tuned."""
        if getattr(solution, "configspace", None) is None:
            return
        if getattr(self, "hpo", False) or getattr(self, "_hpo_unused_warned", False):
            return
        self._hpo_unused_warned = True
        from .runlog import log

        if getattr(self, "supports_hpo", False):
            reason = "HPO is not enabled for it (see Problem.enable_hpo)"
        else:
            reason = "it does not support hyperparameter tuning"
        log.warning(
            "Solutions for problem '%s' come with a configuration space, but "
            "%s: they are evaluated with their default hyperparameters.",
            self.name,
            reason,
        )

    def _track_harness_error(self, harness_error: bool, solution: Solution):
        """Count consecutive harness errors and abort the run past the limit.

        Raises:
            HarnessError: After ``max_harness_errors`` consecutive evaluations
                failed before the generated code was used.
        """
        if not harness_error:
            self._consecutive_harness_errors = 0
            return
        count = getattr(self, "_consecutive_harness_errors", 0) + 1
        self._consecutive_harness_errors = count
        limit = getattr(self, "max_harness_errors", 3)
        if limit and count >= limit:
            raise HarnessError(
                f"Problem '{self.name}': {count} consecutive evaluations failed "
                "before the generated code was used, so the run is stopped. "
                "Fix the problem's evaluation code or environment (set "
                "problem.max_harness_errors = None to disable this check). "
                f"Last error:\n{solution.error}"
            )

    def _ensure_env(self):
        """Create (or reuse) the virtual environment for evaluations.

        Environments are cached and shared between problems with the same
        dependencies (see :mod:`iohblade.envs`); set ``self.env_cache =
        False`` (or ``IOHBLADE_ENV_CACHE=0``) for a private temporary
        environment that :meth:`cleanup` deletes.

        Raises:
            iohblade.envs.EnvironmentSetupError: If the environment cannot
                be created or a dependency cannot be installed (a
                ``RuntimeError``).
        """
        if self._env_path is not None:
            return
        from .envs import get_env

        info = get_env(
            list(getattr(self, "dependencies", None) or []),
            system_site_packages=getattr(self, "use_system_site_packages", False),
            cache=getattr(self, "env_cache", True),
        )
        self._env_path = info.path
        self._python_bin = info.python
        self._env_cached = info.cached

    def cleanup(self):
        """Delete a private evaluation environment (cached ones are kept)."""
        if getattr(self, "_env_cached", False):
            return
        try:
            if self._env_path and self._env_path.exists():
                shutil.rmtree(self._env_path)
        except Exception:
            pass

    def set_logger(self, logger):
        """
        Sets the logger for this problem.
        """
        self.logger = logger
        if logger != None:
            self.logger_dir = logger.get_log_dir()

    def get_prompt(self):
        """
        Get the full prompt describing the problem and how to format the answer.
        """
        return self.task_prompt + self.example_prompt + self.format_prompt

    @abstractmethod
    def evaluate(self, solution: Solution):
        """
        Evaluates a solution on training instances and updates its fitness and feedback.

        Args:
            solution (Solution): Solution object to be evaluated.
        """
        pass

    @abstractmethod
    def test(self, solution: Solution):
        """
        Performs a complete evaluation on test instances and returns the fitness score.

        Args:
            solution (Solution): Solution object to be tested.
        """
        pass

    @abstractmethod
    def to_dict(self):
        """
        Returns a dictionary representation of the problem including all parameters.

        Returns:
            dict: Dictionary representation of the problem.
        """
        pass

    @abstractmethod
    def get_config(self) -> dict[str, Any]:
        """
        * Return a dictionary of properties to log:
            ```
                {
                    `tags`: list[str],
                    `name`: str,
                    `prompt`: str,
                    `minimisation`: bool,
                    `evaluator`: str,
                    `config`: {}    Extra configuration for a problem; like HPO.
                }
            ```
        """


def _takes_problem_argument(fn) -> bool:
    """Whether a wrapped ``evaluate`` function expects the problem first.

    Only *required* positional parameters (no default value) are counted:
    ``(solution)`` and ``(solution, logger=None)`` get the solution only,
    ``(self, solution)`` or ``(problem, solution, extra=1)`` get the problem
    and the solution. The name of the first parameter is not used, so
    ``(problem, solution)`` works as well as ``(self, solution)``.
    """
    try:
        params = inspect.signature(fn).parameters.values()
    except (TypeError, ValueError):
        return False
    required = [
        p
        for p in params
        if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
        and p.default is p.empty
    ]
    return len(required) >= 2


class WrappedProblem(Problem):
    def __init__(
        self,
        evaluate_fn,
        *,
        name="Problem",
        eval_timeout=6000,
        training_instances=None,
        test_instances=None,
        dependencies=None,
        imports=None,
        task_prompt="",
        example_prompt="",
        logger=None,
    ):
        super().__init__(
            logger=logger,
            training_instances=training_instances,
            test_instances=test_instances,
            name=name,
            eval_timeout=eval_timeout,
            dependencies=dependencies,
            imports=imports,
        )
        if task_prompt:
            self.task_prompt = task_prompt
        if example_prompt:
            self.example_prompt = example_prompt

        self._evaluate_fn = evaluate_fn
        # support both signatures: (solution) and (self, solution)
        self._takes_self = _takes_problem_argument(evaluate_fn)
        # store by value
        self._evaluate_fn_bytes = cloudpickle.dumps(evaluate_fn)
        self._evaluate_fn = None  # reconstructed lazily

    def _get_evaluate_fn(self):
        if self._evaluate_fn is None:
            self._evaluate_fn = cloudpickle.loads(self._evaluate_fn_bytes)
        return self._evaluate_fn

    def evaluate(self, solution: Solution):
        fn = self._get_evaluate_fn()
        if self._takes_self:
            return fn(self, solution)
        return fn(solution)

    def test(self, solution: Solution):
        return self.evaluate(solution)

    def to_dict(self):
        return {
            "name": self.name,
            "eval_timeout": self.eval_timeout,
            "training_instances": self.training_instances,
            "test_instances": self.test_instances,
            "dependencies": self.dependencies,
            "imports": self.imports,
        }

    def get_config(self) -> dict[str, Any]:
        return {
            "tags": [],
            "name": self.name,
            "prompt": self.get_prompt(),
            "minimisation": False,
            "config": self.__dict__.copy(),
        }


def wrap_problem(
    evaluate_fn,
    *,
    name="Problem",
    eval_timeout=6000,
    training_instances=None,
    test_instances=None,
    dependencies=None,
    imports=None,
    task_prompt="",
    example_prompt="",
    logger=None,
):
    return WrappedProblem(
        evaluate_fn,
        name=name,
        eval_timeout=eval_timeout,
        training_instances=training_instances,
        test_instances=test_instances,
        dependencies=dependencies,
        imports=imports,
        task_prompt=task_prompt,
        example_prompt=example_prompt,
        logger=logger,
    )
