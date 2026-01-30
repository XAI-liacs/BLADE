from iohblade.experiment import Experiment
from iohblade.llm import Ollama_LLM
from iohblade.methods import LLaMEA
from iohblade.loggers import ExperimentLogger
from os import environ

from iohblade.benchmarks.analysis import get_analysis_problems

# The prompts, and the evaluation function are provided in the autocorrIneq1 class
# as does all other benchmarks..
#   Provide it as an instance of Problem in Experiment

if __name__ == "__main__":
    budget = 10

    api_key = environ.get("GOOGLE_API_KEY")

    ollama_llm = Ollama_LLM("gemma3:12b")
    # gemini_llm = Gemini_LLM(api_key=api_key)

    # Select the instances of Auto-Correlation 1-3.
    # ===============================================
    autocorrineq = get_analysis_problems(use_best=False)[2]
    # ================================================


    methods = []
    for llm in [ollama_llm]:
        method = LLaMEA(
            llm,
            n_parents=1,
            n_offspring=1,
            budget=budget,
            minimization=autocorrineq.minimisation,  
        )
        methods.append(method)
    logger = ExperimentLogger(f"results/Autocorrelation_Inequality")
    experiment = Experiment(
        methods,
        [autocorrineq],
        runs=1,
        budget=budget,
        show_stdout=True,
        exp_logger=logger,
    )

    experiment()
