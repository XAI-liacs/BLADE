from os import environ

from iohblade.experiment import Experiment
from iohblade.llm import Gemini_LLM, Ollama_LLM
from iohblade.methods import LLaMEA
from iohblade.loggers import ExperimentLogger

from iohblade.benchmarks.geometry import KissingNumber11D


if __name__ == "__main__":
    budget = 10

    api_key = environ.get("GOOGLE_API_KEY")

    ollama_llm = Ollama_LLM()
    gemini_llm = Gemini_LLM(api_key=api_key)

    # Kissing 11d
    kissing_11d = KissingNumber11D()

    methods = []
    for llm in [gemini_llm]:
        method = LLaMEA(
            llm,
            n_parents=1,
            n_offspring=1,
            budget=budget,
            minimization=kissing_11d.minimisation
        )
        methods.append(method)
    logger=ExperimentLogger("results/Kissing-11D")
    experiment = Experiment(
        methods,
        [kissing_11d],
        runs=1,
        budget=budget,
        show_stdout=True,
        exp_logger=logger
    )

    experiment()
