from os import environ

from iohblade.experiment import Experiment
from iohblade.llm import Gemini_LLM, Ollama_LLM
from iohblade.methods import LLaMEA
from iohblade.loggers import ExperimentLogger

from iohblade.benchmarks.packing import get_square_packing_problems


if __name__ == "__main__":
    budget = 10

    api_key = environ.get("GOOGLE_API_KEY")

    ollama_llm = Ollama_LLM()
    gemini_llm = Gemini_LLM(api_key=api_key)

    #----------------------------------------------------------------------------------
    # Gets benchmarks for unit square packing problems:
    #   arr[0]:  Unit Square Packing benchmark for Unit Square, packing 26 circles.
    #   arr[1] = Unit Square Packing benchmark for Unit Square, packing 32 circles.
    #----------------------------------------------------------------------------------
    unit_square_packing = get_square_packing_problems(True)[0]

    methods = []
    for llm in [gemini_llm]:
        method = LLaMEA(
            llm,
            n_parents=1,
            n_offspring=1,
            budget=budget,
            minimization=unit_square_packing.minimisation,
        )
        methods.append(method)
    logger = ExperimentLogger(f"results/{unit_square_packing.task_name}")
    experiment = Experiment(
        methods,
        [unit_square_packing],
        runs=1,
        budget=budget,
        show_stdout=True,
        exp_logger=logger,
    )

    experiment()
