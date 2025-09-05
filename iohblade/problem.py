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

import cloudpickle
import numpy as np

# Standard packages installed in every evaluation environment
BASE_DEPENDENCIES = [
    "numpy>=2",
    "cloudpickle>=3.1.0,<4",
    "joblib>=1.4.2,<2",
]

import copy

from .solution import Solution
from .utils import TimeoutException


def evaluate_in_subprocess(problem, conn, solution):
    """Evaluate a solution in a dedicated virtual environment."""
    try:
        env_path = problem._env_path
        python_bin = problem._python_bin

        problem_pickle = env_path / "problem.pkl"
        solution_pickle = env_path / "solution.pkl"
        result_pickle = (
            Path(tempfile.gettempdir()) / f"blade_result_{uuid.uuid4().hex}.pkl"
        )
        problem_copy = copy.deepcopy(problem)
        problem_copy.logger = None
        with open(problem_pickle, "wb") as f:
            cloudpickle.dump(problem_copy, f)
        with open(solution_pickle, "wb") as f:
            cloudpickle.dump(solution, f)

        script_path = env_path / "run_eval.py"
        imports_block = getattr(problem, "imports", "")
        script_path.write_text(
            (f"{imports_block}\n" if imports_block else "")
            + "import cloudpickle as cp\n"
            + "import os, json\n"
            + f"problem_path = {json.dumps(str(problem_pickle))}\n"
            + f"solution_path = {json.dumps(str(solution_pickle))}\n"
            + f"result_path  = {json.dumps(str(result_pickle))}\n"
            + "problem=cp.load(open(problem_path,'rb'))\n"
            + "solution=cp.load(open(solution_path,'rb'))\n"
            + "result=problem.evaluate(solution)\n"
            + "with open(result_path,'wb') as f:\n"
            + "    cp.dump(result, f)\n"
        )

        env = os.environ.copy()
        repo_root = Path(__file__).resolve().parents[1]
        env["PYTHONPATH"] = f"{repo_root}{os.pathsep}" + env.get("PYTHONPATH", "")

        try:
            res = subprocess.run(
                [str(python_bin), str(script_path)],
                check=True,
                env=env,
                capture_output=True,
                text=True,
            )
            with open(result_pickle, "rb") as f:
                result = cloudpickle.load(f)
            conn.send(result)
        except subprocess.CalledProcessError as e:
            # Process returned non-zero exit code
            conn.send(e.stderr)

    except Exception as e:
        conn.send(f"{e} stracktrace: {traceback.format_exc()}")
    finally:
        conn.close()


class Problem(ABC):
    """
    Abstract problem class.
    """

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
        # Combine the base dependencies with any problem specific ones
        self.dependencies = BASE_DEPENDENCIES.copy()
        if dependencies:
            self.dependencies.extend(dependencies)
        if imports is None:
            self.imports = "import numpy as np\n"
        else:
            self.imports = imports

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

        if self.logger != None:
            if self.logger.budget_exhausted():
                solution.set_scores(
                    -np.inf,
                    feedback="Budget is exhausted.",
                    error="Budget is exhausted.",
                )
                return solution  # Return early if budget is exhausted

        # solution = self.evaluate(solution) #old fashioned way
        # Else create a new process for evaluation with timeout
        try:
            self._ensure_env()
            (
                parent_conn,
                child_conn,
            ) = multiprocessing.Pipe()  # Create pipe for communication
            process = multiprocessing.Process(
                target=evaluate_in_subprocess, args=(self, child_conn, solution)
            )
            process.start()
            process.join(timeout=self.eval_timeout)

            if process.is_alive():
                raise TimeoutException(
                    f"Evaluation timed out after {self.eval_timeout} seconds."
                )
            if parent_conn.poll():
                result = parent_conn.recv()
                if isinstance(result, Exception):
                    raise result
                elif isinstance(result, Solution):
                    solution = result
                elif isinstance(result, str):
                    # If a string is returned, it is likely an error message
                    solution.set_scores(
                        -np.inf, feedback=f"An error occurred: {result}.", error=result
                    )
                else:
                    raise Exception("No Solution object or string returned.")
            else:
                raise Exception("Evaluation failed without an exception.")
        except Exception as e:
            solution.set_scores(
                -np.inf,
                feedback=f"An exception occurred: {e}.",
                error=f"An exception occurred: {e}.",
            )
        finally:
            try:
                process.terminate()
                process.join()
            except Exception:
                pass
        if self.logger is not None:
            self.logger.log_individual(solution)
        return solution

    def _ensure_env(self):
        """Create the virtual environment for evaluations if it does not exist."""
        if self._env_path is not None:
            return
        import virtualenv

        env_dir = tempfile.mkdtemp(prefix="blade_env_")
        self._env_path = Path(env_dir)
        virtualenv.cli_run([env_dir])
        self._python_bin = (
            self._env_path / ("Scripts" if os.name == "nt" else "bin") / "python"
        )

        deps = getattr(self, "dependencies", [])
        if deps:
            subprocess.run(
                [str(self._python_bin), "-m", "pip", "install", *deps],
                check=True,
                capture_output=True,
                text=True,
            )

    def cleanup(self):
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


class WrappedProblem(Problem):
    def __init__(
        self,
        evaluate_fn,
        *,
        name="Problem",
        eval_timeout=600,
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
        self._takes_self = len(inspect.signature(evaluate_fn).parameters) > 1
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
