from iohblade.experiment import MA_BBOB_Experiment, Experiment
from iohblade.problems import BBOB_SBOX
from iohblade.llm import Gemini_LLM, Ollama_LLM, OpenAI_LLM, Claude_LLM, Multi_LLM
from iohblade.methods import LLaMEA, RandomSearch
from iohblade.loggers import ExperimentLogger
import numpy as np
import ioh
import os

from iohblade.utils import code_compare
import lizard


if __name__ == "__main__": # prevents weird restarting behaviour
    api_key_google = os.getenv("GEMINI_API_KEY")
    api_key_openai = os.getenv("OPENAI_API_KEY")
    api_key_claude = os.getenv("CLAUDE_API_KEY")


    #. lets first experiment with local models.
    # qwen3-coder:30b, gemma3:27b, llama3.2:3b

    # first experiments with 3x 1 llm, 2x2 llms, 3 llms

    #llm_qwen = Ollama_LLM("qwen3-coder:30b")
    #llm_gemma3 = Ollama_LLM("gemma3:27b")
    #llm_llama = Ollama_LLM("llama3.2:3b")

    #ai_model = "gemini-2.0-flash"
    #llm2 = Gemini_LLM(api_key_google, "gemini-2.5-flash")
    llm = OpenAI_LLM(api_key_openai, "gpt-5-mini-2025-08-07", temperature=1.0)
    #llm3 = Claude_LLM(api_key_claude, "claude-sonnet-4-5-20250929", temperature=1.0)

    budget = 200 # test run (25 iterations of 8 algs)

    DEBUG = False
    if DEBUG:
        budget = 24

    mutation_prompts = [
        "Refine the strategy of the selected solution to improve it.",  # small mutation
        "Generate a new algorithm that is different from the algorithms you have tried before.", #new random solution
    ]

    #for llm in [llm1]:#, llm2]:
    #RS = RandomSearch(llm, budget=budget) 
    LLaMEA_1 = LLaMEA(llm, budget=budget, name="ES", mutation_prompts=mutation_prompts, n_parents=8, n_offspring=8, elitism=True)
    LLaMEA_2 = LLaMEA(llm, budget=budget, name="ES-guided-new", mutation_prompts=mutation_prompts, n_parents=8, n_offspring=8, elitism=True, feature_guided_mutation=True)
    
    methods = [LLaMEA_2] #LLaMEA_1, 

    # List containing function IDs we consider
    training_fids = [1,2,3,4,5]
    testing_fids = [1,2,3,4,5]

    training_instances = [(f, i) for f in training_fids for i in range(1, 4)]
    test_instances = [(f, i) for f in testing_fids for i in range(5, 10)]

    if DEBUG:
        logger = ExperimentLogger("results/BBOB_guided_debug")
    else:
        logger = ExperimentLogger("results/BBOB_guided3")

    problems = []
    if DEBUG:
        problems.append(
            BBOB_SBOX(
                training_instances=[(f, i) for f in [1,2] for i in range(1, 2)],
                test_instances=[(f, i) for f in [1,2] for i in range(3, 4)],
                dims=[2],
                budget_factor=200,
                eval_timeout=100,
                name=f"SBOX",
                problem_type=ioh.ProblemClass.SBOX,
                full_ioh_log=False,
                ioh_dir=f"{logger.dirname}/ioh",
            )
        )
    else:
        problems.append(
            BBOB_SBOX(
                training_instances=training_instances,
                test_instances=test_instances,
                dims=[10],
                budget_factor=2000,
                eval_timeout=1800,
                name=f"BBOB",
                problem_type=ioh.ProblemClass.BBOB,
                full_ioh_log=False,
                ioh_dir=f"{logger.dirname}/ioh",
            )
        )

    experiment = Experiment(
        methods=methods,
        problems=problems,
        runs=5,
        seeds=[1,2,3,4,5],
        show_stdout=False,
        log_stdout=True,
        exp_logger=logger,
        budget=budget,
        n_jobs=5
    )  # normal run

    if DEBUG:
        experiment = Experiment(
            methods=methods,
            problems=problems,
            runs=1,
            seeds=[1],
            show_stdout=True,
            exp_logger=logger,
            budget=budget,
            n_jobs=5
        )  # test run

    #experiment = MA_BBOB_Experiment(methods=methods, runs=5, seeds=[1,2,3,4,5], dims=[10], budget_factor=2000, budget=budget, eval_timeout=270, show_stdout=True, exp_logger=logger, n_jobs=5) #normal run
    experiment() #run the experiment



    #MA_BBOB_Experiment(methods=methods, llm=llm2, runs=5, dims=[2], budget_factor=1000) #quick run


