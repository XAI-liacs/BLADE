from llamea import Operator

from iohblade.experiment import Experiment
from iohblade.llm import Gemini_LLM, Ollama_LLM, OpenAI_LLM
from iohblade.methods import LLaMEA, RandomSearch
from iohblade.loggers import ExperimentLogger
from iohblade.benchmarks import Kerneltuner
import numpy as np
import os

if __name__ == "__main__": # prevents weird restarting behaviour
    api_key = os.getenv("OPENAI_API_KEY")
    api_key_gemini = os.getenv("GEMINI_API_KEY")

    llm1 = Gemini_LLM(api_key_gemini, "gemini-2.0-flash")
    llm2 = OpenAI_LLM(api_key,"o3-2025-04-16", temperature=1.0)
    llm5 = OpenAI_LLM(api_key,"o4-mini-2025-04-16", temperature=1.0) #Done
    budget = 100

    # Mutation operators (llamea >= 1.4); equal weights pick each one equally often.
    operators = [
        Operator(
            prompt="Refine the strategy of the selected solution to improve it.",
            name="refine",
            weight=1.0,
            number_of_parents=1,
        ),
        Operator(
            prompt="Generate a new algorithm that is different from the algorithms you have tried before.",
            name="new_random",
            weight=1.0,
            number_of_parents=1,
        ),
        Operator(
            prompt="Refine and simplify the selected algorithm to improve it.",
            name="simplify",
            weight=1.0,
            number_of_parents=1,
        ),
    ]

    for llm in [llm5]: #, llm1
        LLaMEA_method = LLaMEA(llm, budget=budget, name="o4 (no info)", operators=operators, n_parents=4, n_offspring=12, elitism=True)

        methods = [LLaMEA_method]
        logger = ExperimentLogger("results/kerneltuner-o4-no-info")
        problems = [
            Kerneltuner(
                gpus=["A100", "A4000", "MI250X"], #, "A4000", "MI250X"
                kernels=["gemm"],
                name="kerneltuner-gemm",
                eval_timeout=600, #5 minutes
                budget=1000,
                cache_dir="/data/neocortex/repos/benchmark_hub/",
                extra_info=False),
            Kerneltuner(
                gpus=["A100", "A4000", "MI250X"],
                kernels=["convolution"],
                name="kerneltuner-convolution",
                eval_timeout=600,
                budget=1000,
                cache_dir="/data/neocortex/repos/benchmark_hub/",
                extra_info=False),
            Kerneltuner(
                gpus=["A100", "A4000", "MI250X"],
                kernels=["dedispersion"],
                name="kerneltuner-dedispersion",
                eval_timeout=600,
                budget=1000,
                cache_dir="/data/neocortex/repos/benchmark_hub/",
                extra_info=False),
            Kerneltuner(
                gpus=["A100", "A4000", "MI250X"],
                kernels=["hotspot"],
                name="kerneltuner-hotspot",
                eval_timeout=600,
                budget=1000,
                cache_dir="/data/neocortex/repos/benchmark_hub/",
                extra_info=False),

            ]
        experiment = Experiment(methods=methods, problems=problems, runs=3, budget = budget, show_stdout=True, exp_logger=logger) #normal run
        experiment() #run the experiment
