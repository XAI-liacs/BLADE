import os

import ioh

from iohblade.experiment import Experiment
from iohblade.llm import Gemini_LLM, Ollama_LLM, OpenAI_LLM
from iohblade.loggers import ExperimentLogger
from iohblade.methods import LLaMEA, RandomSearch
from iohblade.benchmarks import BBOB_SBOX

if __name__ == "__main__":  # prevents weird restarting behaviour
    api_key = os.getenv("OPENAI_API_KEY")

    llm1 = OpenAI_LLM(api_key, "gpt-5.6-luna", temperature=1.0)  # Done
    llm3 = Ollama_LLM("qwen3.8:27b")  # Failed
    llm4 = Ollama_LLM("gemma3:27b")  # Done
    llm5 = OpenAI_LLM(api_key, "gpt-5.4-nano-2026-03-17", temperature=1.0)
    budget = 100  # long budgets

    mutation_prompts = [
        "Refine and simplify the selected solution to improve it.",  # simplify mutation
        "Generate a new algorithm that is different from the algorithms you have tried before.",  # new random solution
    ]

    methods = []
    for llm in [llm3,llm4,llm5,llm1]:  # , llm2, llm3, llm4, llm5, llm2
        # RS = RandomSearch(llm, budget=budget) #LLaMEA(llm)
        LLaMEA_method = LLaMEA(
            llm,
            budget=budget,
            name=f"LLaMEA-"+llm.model.replace(":", "-"),
            mutation_prompts=mutation_prompts,
            n_parents=4,
            n_offspring=12,
            elitism=False,
        )
        
        methods.append(LLaMEA_method)

    # List containing function IDs we consider
    fids = range(1, 25)  # BBOB functions 1-24

    training_instances = [(f, i) for f in fids for i in range(1, 3)]
    test_instances = [(f, i) for f in fids for i in range(5, 7)]

    logger = ExperimentLogger("results/BBOB-data")

    problems = []
    problems.append(
        BBOB_SBOX(
            training_instances=training_instances,
            test_instances=test_instances,
            dims=[5],
            budget_factor=2000,
            eval_timeout=600,
            name=f"BBOB",
            problem_type=ioh.ProblemClass.BBOB,
            full_ioh_log=False,
            ioh_dir=f"",
        )
    )

    experiment = Experiment(
        methods=methods,
        problems=problems,
        runs=3,
        show_stdout=False,
        exp_logger=logger,
        budget=budget,
        n_jobs=4,
    )  # normal run
    experiment()  # run the experiment
