import multiprocessing

from .llm import LLM, Gemini_LLM, Ollama_LLM, OpenAI_LLM
from .method import Method
from .plots import (
    fitness_table,
    plot_boxplot_fitness,
    plot_boxplot_fitness_hue,
    plot_code_evolution_graphs,
    plot_convergence,
    plot_experiment_CEG,
    plot_token_usage,
)
from .problem import Problem
from .solution import Solution
from .utils import (
    NoCodeException,
    OverBudgetException,
    ThresholdReachedException,
    TimeoutException,
    aoc_logger,
    budget_logger,
    convert_to_serializable,
    correct_aoc,
)


def ensure_spawn_start_method():
    try:
        if multiprocessing.get_start_method(allow_none=True) != "spawn":
            multiprocessing.set_start_method("spawn", force=True)
    except RuntimeError:
        # Already set
        if multiprocessing.get_start_method(allow_none=True) != "spawn":
            raise RuntimeError(
                "Multiprocessing start method is not 'spawn'. "
                "Set it at the top of your main script:\n"
                "import multiprocessing\n"
                "multiprocessing.set_start_method('spawn', force=True)"
            )


ensure_spawn_start_method()
