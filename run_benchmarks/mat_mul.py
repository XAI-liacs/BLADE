from os import environ

from virtualenv.discovery.cached_py_info import random

from iohblade.experiment import Experiment
from iohblade.llm import Gemini_LLM, Ollama_LLM
from iohblade.methods import LLaMEA
from iohblade.loggers import ExperimentLogger

from iohblade.benchmarks.matrix_multiplication import default_mat_mul_problems


if __name__ == "__main__":
    budget = 10

    api_key = environ.get("GOOGLE_API_KEY")

    ollama_llm = Ollama_LLM()
    gemini_llm = Gemini_LLM(api_key=api_key)

    # Pick a random matrix multiplication problem.
    random_mat_mul_problem = random.choice(default_mat_mul_problems())

    methods = []
    for llm in [gemini_llm]:
        method = LLaMEA(
            llm,
            n_parents=1,
            n_offspring=1,
            budget=budget,
            minimization=random_mat_mul_problem.minimisation,
        )
        methods.append(method)
    logger = ExperimentLogger("results/MatrixMultiplication")
    experiment = Experiment(
        methods,
        [random_mat_mul_problem],
        runs=1,
        budget=budget,
        show_stdout=True,
        exp_logger=logger,
    )

    experiment()
