from datetime import datetime

from ConfigSpace.read_and_write import json as cs_json

from ..llm import LLM
from ..method import Method
from ..problem import Problem
from ..solution import Solution
from ..utils import convert_to_serializable
from .base import ExperimentLogger, RunLogger

try:  # pragma: no cover - import guard
    import trackio
except Exception as e:  # pragma: no cover - handled in __init__
    trackio = None
    _import_error = e
else:
    _import_error = None


class TrackioExperimentLogger(ExperimentLogger):
    """Experiment logger that also logs runs to Trackio."""

    def __init__(self, name: str = "", read: bool = False):
        if trackio is None:
            raise ImportError(
                "Trackio is not installed. Install with `pip install trackio`."
            ) from _import_error
        super().__init__(name=name, read=read)
        self.project = name
        self._current_run = {}

    def _before_open_run(self, run_name, method, problem, budget, seed):
        # Store metadata for the upcoming run so the run logger can
        # initialise Trackio in the executing thread.
        self._current_run = {
            "method_name": method.name,
            "problem_name": problem.name,
            "seed": int(seed),
        }

    def _create_run_logger(self, run_name, budget, progress_cb):
        return TrackioRunLogger(
            name=run_name,
            root_dir=self.dirname,
            budget=budget,
            progress_callback=progress_cb,
            project=self.project,
            method_name=self._current_run.get("method_name", ""),
            problem_name=self._current_run.get("problem_name", ""),
            seed=self._current_run.get("seed"),
        )

    def add_run(
        self,
        method: Method,
        problem: Problem,
        llm: LLM,
        solution: Solution,
        log_dir: str = "",
        seed: int | None = None,
    ):
        super().add_run(
            method=method,
            problem=problem,
            llm=llm,
            solution=solution,
            log_dir=log_dir,
            seed=seed,
        )


class TrackioRunLogger(RunLogger):
    """Run logger that mirrors data to Trackio."""

    def __init__(
        self,
        *args,
        project: str = "",
        method_name: str = "",
        problem_name: str = "",
        seed: int | None = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.project = project
        self.method_name = method_name
        self.problem_name = problem_name
        self.seed = seed
        self._run_name = args[0] if args else ""
        self._initialized = False
        self.llm_name = ""

    def _ensure_init(self):
        if not self._initialized:
            trackio.init(project=self.project, name=self._run_name)
            self._initialized = True

    def start_run(self, llm: LLM):
        """Initialise Trackio and log run metadata."""
        self.llm_name = llm.model
        self._ensure_init()
        trackio.log(
            {
                "method_name": self.method_name,
                "problem_name": self.problem_name,
                "llm_name": self.llm_name,
                "seed": int(self.seed),
            }
        )

    def finish_run(self, solution: Solution):
        """Log final fitness and finish the Trackio run."""
        self._ensure_init()
        trackio.log(
            {
                "final_fitness": (
                    convert_to_serializable(solution.fitness)
                    if solution.fitness is not None
                    else float("nan")
                )
            }
        )
        trackio.finish()
        self._initialized = False

    def log_conversation(self, role, content, cost: float = 0.0, tokens: int = 0):
        self._ensure_init()
        trackio.log(
            {
                "role": role,
                "reply_time": str(datetime.now()),
                "content": content,
                "cost": convert_to_serializable(cost),
                "tokens": int(tokens),
            }
        )
        super().log_conversation(role, content, cost, tokens)

    def log_individual(self, individual):
        self._ensure_init()
        ind_dict = individual.to_dict()
        if "fitness" in ind_dict:
            # print(convert_to_serializable(ind_dict["fitness"]))
            trackio.log({"fitness": convert_to_serializable(ind_dict["fitness"])})
        trackio.log({"solution": convert_to_serializable(ind_dict)})
        super().log_individual(individual)

    def log_code(self, individual):
        super().log_code(individual)

    def log_configspace(self, individual):
        super().log_configspace(individual)

    def budget_exhausted(self):  # pragma: no cover - same as parent
        return super().budget_exhausted()
