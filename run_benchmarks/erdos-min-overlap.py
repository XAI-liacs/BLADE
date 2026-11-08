from os import environ

from llamea.llm import LMStudio_LLM

from iohblade.experiment import Experiment
from iohblade.llm import Gemini_LLM, Ollama_LLM
from iohblade.methods import LLaMEA
from iohblade.loggers import ExperimentLogger

from iohblade.benchmarks.combinatorics import get_combinatorics_problems


if __name__ == "__main__":
    budget = 10

    # api_key = environ.get("GOOGLE_API_KEY")


    ollama_llm = Ollama_LLM('gemma4:latest')
    ollama_llm2 = Ollama_LLM('gpt-oss:20b')

    erdos_min_overlap = get_combinatorics_problems(False)

    methods = []

    for llm in [ollama_llm, ollama_llm2]:
        method = LLaMEA(
            llm,
            n_parents=4,
            n_offspring=4,
            budget=budget,
        )

        methods.append(method)

    logger = ExperimentLogger(f"results/ErdösMinOverlap")

    experiment = Experiment(
        methods,
        erdos_min_overlap,
        runs=1,
        budget=budget,
        show_stdout=True,
        exp_logger=logger,
    )

    experiment()
