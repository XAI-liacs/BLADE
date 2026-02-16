from os import environ

from iohblade.experiment import Experiment
from iohblade.llm import Gemini_LLM, Ollama_LLM, LMStudio_LLM
from iohblade.methods import LLaMEA
from iohblade.loggers import ExperimentLogger

from iohblade.benchmarks.logistics import VehicleRoutingProblem


if __name__ == "__main__":
    budget = 10

    # api_key = environ.get("GOOGLE_API_KEY")


    ollama_llm = Ollama_LLM('gpt-oss:20b')
    # gemini_llm = Gemini_LLM(api_key=api_key)
    # mlx_llm = LMStudio_LLM('google/gemma-3-12b')

    cvrp = VehicleRoutingProblem('A-n53-k7')

    methods = []
    for llm in [ollama_llm]:
        method = LLaMEA(
            llm,
            n_parents=1,
            n_offspring=1,
            budget=budget,
            minimization=cvrp.minimisation,
        )
        methods.append(method)
    logger = ExperimentLogger(f"results/CapacitatedVehicleRoutingProblem")
    experiment = Experiment(
        methods,
        [cvrp],
        runs=1,
        budget=budget,
        show_stdout=True,
        exp_logger=logger,
    )

    experiment()
