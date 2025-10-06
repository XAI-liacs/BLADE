from os import environ

from iohblade.experiment import Experiment
from iohblade.llm import Gemini_LLM, Ollama_LLM
from iohblade.methods import LLaMEA
from iohblade.loggers import ExperimentLogger

from iohblade.benchmarks.number_theory import get_sum_vs_difference_problem


if __name__ == "__main__":
    budget = 10

    api_key = environ.get("GOOGLE_API_KEY")

    ollama_llm = Ollama_LLM()
    gemini_llm = Gemini_LLM(api_key=api_key)

    # Get sums vs differences benchmark, this one only has one instance.
    sum_vs_difference = get_sum_vs_difference_problem(True)[0]

    methods = []
    for llm in [gemini_llm]:
        method = LLaMEA(
            llm,
            n_parents=1,
            n_offspring=1,
            budget=budget,
            minimization=sum_vs_difference.minimisation,
        )
        methods.append(method)
    logger = ExperimentLogger(f"results/{sum_vs_difference.task_name}")
    experiment = Experiment(
        methods,
        [sum_vs_difference],
        runs=1,
        budget=budget,
        show_stdout=True,
        exp_logger=logger,
    )

    experiment()
