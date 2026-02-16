from os import environ

from iohblade.experiment import Experiment
from iohblade.llm import Gemini_LLM, Ollama_LLM, LMStudio_LLM
from iohblade.methods import LLaMEA
from iohblade.loggers import ExperimentLogger

from iohblade.benchmarks.combinatorics import EuclidianSteinerTree


if __name__ == "__main__":
    budget = 10

    # api_key = environ.get("GOOGLE_API_KEY")


    ollama_llm = Ollama_LLM('gemma3:12b')
    # gemini_llm = Gemini_LLM(api_key=api_key)
    # mlx_llm = LMStudio_LLM('google/gemma-3-12b')

    est = EuclidianSteinerTree(20)

    methods = []
    for llm in [ollama_llm]:
        method = LLaMEA(
            llm,
            n_parents=1,
            n_offspring=1,
            budget=budget,
            minimization=est.minimisation,
        )
        methods.append(method)
    logger = ExperimentLogger(f"results/EuclidianSteinerProblem")
    experiment = Experiment(
        methods,
        [est],
        runs=1,
        budget=budget,
        show_stdout=True,
        exp_logger=logger,
    )

    experiment()
