import os

import ioh

from llamea import Operator

from iohblade.experiment import Experiment
from iohblade.llm import Gemini_LLM, Ollama_LLM, OpenAI_LLM
from iohblade.loggers import ExperimentLogger
from iohblade.methods import LLaMEA, RandomSearch
from iohblade.benchmarks import BBOB_SBOX

if __name__ == "__main__":  # prevents weird restarting behaviour
    api_key = os.getenv("OPENAI_API_KEY")
    api_key_gemini = os.getenv("GEMINI_API_KEY")

    llm1 = OpenAI_LLM(api_key, "gpt-4.1-2025-04-14")  # Done
    llm2 = Gemini_LLM(
        api_key_gemini, "gemini-2.0-flash"
    )  # Failed partly #running 3/4 in BBOB-4, 5/6 in BBOB-5, rest is in 1/2 BBOB-1 folder.
    llm3 = Ollama_LLM("qwen2.5-coder:32b")  # Failed
    llm4 = Ollama_LLM("gemma3:27b")  # Done
    llm5 = OpenAI_LLM(api_key, "o4-mini-2025-04-16", temperature=1.0)
    # llm2 = Ollama_LLM("codestral")
    # llm3 = Ollama_LLM("qwen2.5-coder:14b") #qwen2.5-coder:14b, deepseek-coder-v2:16b
    # llm4 = Ollama_LLM("deepseek-coder-v2:16b")
    # llm5 = Gemini_LLM(api_key, "gemini-1.5-flash")
    budget = 100  # long budgets

    # Mutation operators (llamea >= 1.4); equal weights pick each one equally often.
    operators1 = [
        Operator(
            prompt="Refine and simplify the selected algorithm to improve it.",
            name="simplify",
            weight=1.0,
            number_of_parents=1,
        ),
    ]
    operators2 = [
        Operator(
            prompt="Generate a new algorithm that is different from the algorithms you have tried before.",
            name="new_random",
            weight=1.0,
            number_of_parents=1,
        ),
    ]
    operators3 = [
        Operator(
            prompt="Refine and simplify the selected solution to improve it.",
            name="simplify",
            weight=1.0,
            number_of_parents=1,
        ),
        Operator(
            prompt="Generate a new algorithm that is different from the algorithms you have tried before.",
            name="new_random",
            weight=1.0,
            number_of_parents=1,
        ),
    ]

    for llm in [llm2]:  # , llm2, llm3, llm4, llm5, llm2
        # RS = RandomSearch(llm, budget=budget) #LLaMEA(llm)
        LLaMEA_method1 = LLaMEA(
            llm,
            budget=budget,
            name=f"LLaMEA-1",
            operators=operators1,
            n_parents=4,
            n_offspring=12,
            elitism=False,
        )
        LLaMEA_method2 = LLaMEA(
            llm,
            budget=budget,
            name=f"LLaMEA-2",
            operators=operators2,
            n_parents=4,
            n_offspring=12,
            elitism=False,
        )
        LLaMEA_method3 = LLaMEA(
            llm,
            budget=budget,
            name=f"LLaMEA-3",
            operators=operators3,
            n_parents=4,
            n_offspring=12,
            elitism=False,
        )
        LLaMEA_method4 = LLaMEA(
            llm,
            budget=budget,
            name=f"LLaMEA-4",
            operators=operators3,
            n_parents=1,
            n_offspring=1,
            elitism=True,
        )
        LLaMEA_method5 = LLaMEA(
            llm,
            budget=budget,
            name=f"LLaMEA-5",
            adaptive_mutation=True,
            n_parents=4,
            n_offspring=12,
            elitism=False,
        )
        LLaMEA_method6 = LLaMEA(
            llm,
            budget=budget,
            name=f"LLaMEA-6",
            adaptive_mutation=True,
            n_parents=1,
            n_offspring=1,
            elitism=True,
        )

        methods = [
            LLaMEA_method1,
            LLaMEA_method2,
            LLaMEA_method3,
            LLaMEA_method4,
            LLaMEA_method5,
            LLaMEA_method6,
        ]

        # List containing function IDs we consider
        fids = [1, 3, 6, 8, 10, 13, 15, 17, 21, 23]

        training_instances = [(f, i) for f in fids for i in range(1, 6)]
        test_instances = [(f, i) for f in fids for i in range(6, 16)]

        logger = ExperimentLogger("results/BBOB")

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
                full_ioh_log=True,
                ioh_dir=f"{logger.dirname}/ioh",
            )
        )

        experiment = Experiment(
            methods=methods,
            problems=problems,
            runs=5,
            show_stdout=False,
            exp_logger=logger,
            budget=budget,
        )  # normal run
        experiment()  # run the experiment
