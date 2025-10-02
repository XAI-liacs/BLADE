from os import environ

from iohblade.experiment import Experiment
from iohblade.llm import Gemini_LLM, Ollama_LLM
from iohblade.methods import LLaMEA
from iohblade.loggers import ExperimentLogger

from iohblade.benchmarks.geometry import get_min_max_dist_ratio_problem

if __name__ == "__main__":
    budget = 10

    api_key = environ.get("GOOGLE_API_KEY")

    ollama_llm = Ollama_LLM()
    gemini_llm = Gemini_LLM(api_key=api_key)

    ## Min max Distance ratio problem;
    # a[0] = 2-D min max distance ration problem.
    # a[1] = 3-D min max distance ration problem.

    min_max_min_distance = get_min_max_dist_ratio_problem(True)[0]

    methods = []
    for llm in [gemini_llm]:
        method = LLaMEA(
            llm,
            n_parents=1,
            n_offspring=1,
            budget=budget,
            minimization=min_max_min_distance.minimisation,
        )
        methods.append(method)
    logger = ExperimentLogger(f"results/{min_max_min_distance.name}")
    experiment = Experiment(
        methods,
        [min_max_min_distance],
        runs=1,
        budget=budget,
        show_stdout=True,
        exp_logger=logger,
    )

    experiment()
