import threading
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
    """Run logger that mirrors data to Trackio.

    Each run logs to its own Trackio run object (the one returned by
    ``trackio.init``). Trackio keeps its *current* run in a context variable,
    i.e. per thread, so the module-level ``trackio.log`` would fail (or log
    to another run) when a method logs from a helper thread; logging through
    the run object keeps parallel runs (``n_jobs > 1``) apart.
    """

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
        # RunLogger(name, root_dir, ...): the name may be given either way.
        self._run_name = kwargs.get("name", args[0] if args else "")
        self._initialized = False
        self._run = None
        self._init_thread = None
        self._init_lock = threading.Lock()
        self.llm_name = ""

    def _ensure_init(self):
        with self._init_lock:
            if not self._initialized:
                self._run = trackio.init(project=self.project, name=self._run_name)
                self._init_thread = threading.get_ident()
                self._initialized = True
            return self._run

    def _log(self, metrics: dict):
        run = self._ensure_init()
        if run is not None and callable(getattr(run, "log", None)):
            run.log(metrics)
        else:  # an older trackio that does not return the run
            trackio.log(metrics)

    def start_run(self, llm: LLM):
        """Initialise Trackio and log run metadata."""
        self.llm_name = llm.model
        self._log(
            {
                "method_name": self.method_name,
                "problem_name": self.problem_name,
                "llm_name": self.llm_name,
                "seed": int(self.seed) if self.seed is not None else None,
            }
        )

    def finish_run(self, solution: Solution):
        """Log final fitness and finish the Trackio run."""
        self._log(
            {
                "final_fitness": (
                    convert_to_serializable(solution.fitness)
                    if solution.fitness is not None
                    else float("nan")
                )
            }
        )
        with self._init_lock:
            run = self._run
            if (
                run is not None
                and callable(getattr(run, "finish", None))
                and threading.get_ident() != self._init_thread
            ):
                # Initialised in another thread: that thread's current run
                # is ours, so finish the run object itself.
                run.finish()
            else:
                # Finishes this thread's current run (ours) and clears it.
                trackio.finish()
            self._run = None
            self._initialized = False

    def log_conversation(self, role, content, cost: float = 0.0, tokens: int = 0):
        self._log(
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
        ind_dict = individual.to_dict()
        if "fitness" in ind_dict:
            self._log({"fitness": convert_to_serializable(ind_dict["fitness"])})
        self._log({"solution": convert_to_serializable(ind_dict)})
        super().log_individual(individual)

    def log_code(self, individual):
        super().log_code(individual)

    def log_configspace(self, individual):
        super().log_configspace(individual)

    def budget_exhausted(self):  # pragma: no cover - same as parent
        return super().budget_exhausted()

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_init_lock"] = None
        state["_run"] = None  # live Trackio runs cannot be pickled
        state["_initialized"] = False
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._init_lock = threading.Lock()
