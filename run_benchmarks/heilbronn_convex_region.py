from os import environ

from iohblade.experiment import Experiment
from iohblade.llm import Gemini_LLM, Ollama_LLM
from iohblade.methods import LLaMEA
from iohblade.loggers import ExperimentLogger

from iohblade.benchmarks.geometry import get_heilbronn_convex_region_problems


if __name__ == "__main__":
    budget = 10

    api_key = environ.get("GOOGLE_API_KEY")

    ollama_llm = Ollama_LLM()
    gemini_llm = Gemini_LLM(api_key=api_key)

    heilbronn_convex_region = get_heilbronn_convex_region_problems(True)
    #Pick a Heilbronn problem, with known best solution.
    # heilbronn_convex_region[0] is 13 points problem and a[1] 14.
    heilbronn_convex_region = heilbronn_convex_region[1]

    methods = []
    for llm in [gemini_llm]:
        method = LLaMEA(
            llm,
            n_parents=1,
            n_offspring=1,
            budget=budget,
            minimization=heilbronn_convex_region.minimisation,
        )
        methods.append(method)
    logger = ExperimentLogger(f"results/{heilbronn_convex_region.task_name}")
    experiment = Experiment(
        methods,
        [heilbronn_convex_region],
        runs=1,
        budget=budget,
        show_stdout=True,
        exp_logger=logger,
    )

    experiment()
