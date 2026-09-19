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


    llm1 = Ollama_LLM('llama3.2:latest')
    llm2 = LMStudio_LLM('google/gemma-3-12b')
    # gemini_llm = Gemini_LLM(api_key=api_key)

    erdos_min_overlap = get_combinatorics_problems(False)[0]

    methods = []
    for llm in [llm1, llm2]:
        method = LLaMEA(
            llm,
            n_parents=4,
            n_offspring=4,
            budget=budget,
        )
        method.name += f"-{method.llm.model}"
        methods.append(method)
    logger = ExperimentLogger(f"results/Erdös_Min_Overlap")

    experiment = Experiment(
        methods,
        [erdos_min_overlap],
        runs=1,
        budget=budget,
        show_stdout=True,
        exp_logger=logger,
    )

    experiment()
