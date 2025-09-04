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
                "Trackio is not installed. Install with `uv sync --group trackio`."
            ) from _import_error
        super().__init__(name=name, read=read)
        self.project = name
        self._run_active = False

    def _before_open_run(self, run_name, method, problem, budget, seed):
        trackio.init(project=self.project, name=run_name)
        self._run_active = True

    def _create_run_logger(self, run_name, budget, progress_cb):
        return TrackioRunLogger(
            name=run_name,
            root_dir=self.dirname,
            budget=budget,
            progress_callback=progress_cb,
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
        if not self._run_active:
            super().open_run(method, problem, budget=method.budget, seed=seed)

        trackio.log(
            {
                "method_name": method.name,
                "problem_name": problem.name,
                "llm_name": llm.model,
                "seed": seed,
                "final_fitness": (
                    solution.fitness if solution.fitness is not None else float("nan")
                ),
            }
        )
        trackio.finish()
        self._run_active = False

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

    def log_conversation(self, role, content, cost: float = 0.0, tokens: int = 0):
        trackio.log(
            {
                "role": role,
                "time": str(datetime.now()),
                "content": content,
                "cost": float(cost),
                "tokens": int(tokens),
            }
        )
        super().log_conversation(role, content, cost, tokens)

    def log_individual(self, individual):
        ind_dict = individual.to_dict()
        if "fitness" in ind_dict:
            trackio.log({"fitness": ind_dict["fitness"]})
        trackio.log({"solution": convert_to_serializable(ind_dict)})
        super().log_individual(individual)

    def log_code(self, individual):
        trackio.log({"code": individual.code})
        super().log_code(individual)

    def log_configspace(self, individual):
        if individual.configspace is not None:
            text = cs_json.write(individual.configspace)
        else:
            text = "Failed to extract config space"
        trackio.log({"configspace": text})
        super().log_configspace(individual)

    def budget_exhausted(self):  # pragma: no cover - same as parent
        return super().budget_exhausted()
