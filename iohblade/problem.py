import multiprocessing
import traceback
from abc import ABC, abstractmethod

import numpy as np
from joblib.externals.loky import get_reusable_executor

from .solution import Solution
from .utils import TimeoutException


def evaluate_in_subprocess(problem, conn, solution):
    """
    Runs the evaluation and stores the result in a queue.
    Args:
        queue (multiprocessing.Queue): Queue for storing the evaluation result.
        solution (Solution): Solution object to be evaluated.
    """
    try:
        result = problem.evaluate(solution)
        conn.send(result)  # Send result through the pipe
    except Exception as e:
        # print(f"stracktrace: {traceback.format_exc()}")
        conn.send(
            f"{e} stracktrace: {traceback.format_exc()}"
        )  # Send exception for handling in the parent
    finally:
        conn.close()  # Ensure pipe is closed after sending data


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
        """
        self.logger = logger
        self.training_instances = training_instances if training_instances else []
        self.test_instances = test_instances if test_instances else []
        self.task_prompt = "Write the problem description part here."
        self.example_prompt = "Write an example code here."
        self.format_prompt = "Write the format description part here."
        self.name = name
        self.eval_timeout = eval_timeout

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
                    -np.Inf,
                    feedback="Budget is exhausted.",
                    error="Budget is exhausted.",
                )
                return solution  # Return early if budget is exhausted

        # solution = self.evaluate(solution) #old fashioned way
        # Else create a new process for evaluation with timeout
        try:
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
                        -np.Inf, feedback=f"An error occurred: {result}.", error=result
                    )
                else:
                    raise Exception("No Solution object or string returned.")
            else:
                raise Exception("Evaluation failed without an exception.")
        except Exception as e:
            solution.set_scores(
                -np.Inf,
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

    def set_logger(self, logger):
        """
        Sets the logger for this problem.
        """
        self.logger = logger

    @abstractmethod
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
