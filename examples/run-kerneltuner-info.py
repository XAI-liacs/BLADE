from iohblade.experiment import Experiment
from iohblade.llm import Gemini_LLM, Ollama_LLM, OpenAI_LLM
from iohblade.methods import LLaMEA, RandomSearch
from iohblade.loggers import ExperimentLogger
from iohblade.problems import Kerneltuner
import numpy as np
import os

if __name__ == "__main__": # prevents weird restarting behaviour
    api_key = os.getenv("OPENAI_API_KEY")
    api_key_gemini = os.getenv("GEMINI_API_KEY")

    llm1 = Gemini_LLM(api_key_gemini, "gemini-2.0-flash")
    llm5 = OpenAI_LLM(api_key,"o4-mini-2025-04-16", temperature=1.0)
    budget = 200

    mutation_prompts = [
        "Refine the strategy of the selected solution to improve it.",  # small mutation
        "Generate a new algorithm that is different from the algorithms you have tried before.", #new random solution
        "Refine and simplify the selected algorithm to improve it.", #simplify
    ]

    for llm in [llm5]: #, llm1
        LLaMEA_method = LLaMEA(llm, budget=budget, name="LLaMEA", mutation_prompts=mutation_prompts, n_parents=4, n_offspring=12, elitism=True)

        methods = [LLaMEA_method]
        logger = ExperimentLogger("results/kerneltuner-kernel")
        problems = [
            # Kerneltuner(
            #     gpus=["A100", "A4000", "MI250X"],
            #     kernels=["gemm"],
            #     name="kerneltuner-gemm",
            #     eval_timeout=120,
            #     budget=100,
            #     cache_dir="/data/neocortex/repos/benchmark_hub/",
            #     extra_info=True),
            # Kerneltuner(
            #     gpus=["A100", "A4000", "MI250X"],
            #     kernels=["convolution"],
            #     name="kerneltuner-convolution",
            #     eval_timeout=160,
            #     budget=100,
            #     cache_dir="/data/neocortex/repos/benchmark_hub/",
            #     extra_info=True),
            # Kerneltuner(
            #     gpus=["A100", "A4000", "MI250X"],
            #     kernels=["dedispersion"],
            #     name="kerneltuner-dedispersion",
            #     eval_timeout=360,
            #     budget=100,
            #     cache_dir="/data/neocortex/repos/benchmark_hub/",
            #     extra_info=True),
            Kerneltuner(
                gpus=["A100", "A4000", "MI250X"],
                kernels=["hotspot"],
                name="kerneltuner-hotspot",
                eval_timeout=120,
                budget=100,
                cache_dir="/data/neocortex/repos/benchmark_hub/",
                extra_info=True),
            
            ]
        experiment = Experiment(methods=methods, problems=problems, llm=llm, runs=3, budget = budget, show_stdout=True, exp_logger=logger) #normal run
        experiment() #run the experiment