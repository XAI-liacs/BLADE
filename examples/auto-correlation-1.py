from iohblade.experiment import Experiment
from iohblade.llm import Gemini_LLM, Ollama_LLM
from iohblade.methods import LLaMEA
from iohblade.loggers import ExperimentLogger
from iohblade.benchmarks.analysis.auto_correlation_ineq1 import AutoCorrIneq1
from os import environ

# The prompts, and the evaluation function are provided in the autocorrIneq1 class
# as does all other benchmarks..
#   Provide it as an instance of Problem in Experiment

if __name__ == "__main__":
    budget=10

    api_key = environ.get("GOOGLE_API_KEY")

    ollama_llm = Ollama_LLM()
    gemini_llm = Gemini_LLM(api_key=api_key)

    autocor1 = AutoCorrIneq1()

    methods = []
    for llm in [gemini_llm]:
        method = LLaMEA(
            llm,
            n_parents=1,
            n_offspring=1,
            budget=budget,
            minimization=True
        )
        methods.append(method)
    logger=ExperimentLogger("results/Auto-Correlation-Inequality-1")
    experiment = Experiment(
        methods,
        [autocor1],
        runs=1,
        budget=budget,
        show_stdout=True,
        exp_logger=logger
    )

    experiment()
