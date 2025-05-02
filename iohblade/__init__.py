from .solution import Solution
from .method import Method
from .problem import Problem
from .llm import (
    Ollama_LLM,
    OpenAI_LLM,
    Gemini_LLM,
    LLM,
)
from .plots import (
    plot_convergence,
    plot_experiment_CEG,
    plot_code_evolution_graphs,
    plot_boxplot_fitness,
    plot_boxplot_fitness_hue,
    fitness_table,
)
from .utils import (
    convert_to_serializable,
    aoc_logger,
    correct_aoc,
    OverBudgetException,
    ThresholdReachedException,
    NoCodeException,
    TimeoutException,
    budget_logger,
)

import multiprocessing

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