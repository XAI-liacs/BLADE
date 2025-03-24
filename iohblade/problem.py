from .solution import Solution
from abc import ABC, abstractmethod
import numpy as np
import traceback
import multiprocessing
from joblib.externals.loky import get_reusable_executor


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
        conn.send(e)  # Send exception for handling in the parent
    finally:
        conn.close()  # Ensure pipe is closed after sending data


class TimeoutException(Exception):
    """Custom exception for handling timeouts."""

    pass


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
        eval_timeout=60,
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
        self.format_prompt = "Write the format description part here."
        self.name = name
        self.eval_timeout = eval_timeout

    def __call__(self, solution: Solution, logger=None):
        """
        Evaluates a solution on training instances and updates its fitness and feedback.

        Args:
            solution (Solution): Solution object to be evaluated.
            logger (RunLogger, optional): The RunLogger object attached to the problem to keep track of evaluations.

        Returns:
            Solution: The evaluated solution with updated fitness and scores.
        """

        if self.logger != None:
            if self.logger.budget_exhausted():
                raise Exception("Evaluation failed because budget is exhausted.")

        # Ensure multiprocessing is using spawn mode
        if multiprocessing.get_start_method(allow_none=True) != "spawn":
            multiprocessing.set_start_method("spawn", force=True)

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
                else:
                    solution = result
            else:
                raise Exception("Evaluation failed without an exception.")
        except Exception as e:
            solution.set_scores(-np.Inf, feedback=f"An exception occurred: {e}.")
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
        Get the prompt describing the problem and how to format the answer.
        """
        return self.task_prompt + self.format_prompt

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
