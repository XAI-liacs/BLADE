from os import environ

from iohblade.experiment import Experiment
from iohblade.llm import Gemini_LLM, Ollama_LLM
from iohblade.methods import LLaMEA
from iohblade.loggers import ExperimentLogger

from iohblade.benchmarks.fourier import get_fourier_problems


if __name__ == "__main__":
    budget = 10

    # api_key = environ.get("GOOGLE_API_KEY")

    ollama_llm = Ollama_LLM('gemma3:12b')
    # gemini_llm = Gemini_LLM(api_key=api_key)

    # Helibronn n11 benchmark.
    uncertain_ineq = get_fourier_problems(use_best=False)[0]

    methods = []
    for llm in [ollama_llm]:
        method = LLaMEA(
            llm,
            n_parents=1,
            n_offspring=1,
            budget=budget,
            minimization=uncertain_ineq.minimisation,
            elitism=True,
        )
        methods.append(method)
    logger = ExperimentLogger(f"results/Fourier_Unequality")
    experiment = Experiment(
        methods,
        [uncertain_ineq],
        runs=1,
        budget=budget,
        show_stdout=True,
        exp_logger=logger,
    )

    experiment()
