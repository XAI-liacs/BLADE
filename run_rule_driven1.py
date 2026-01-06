from iohblade.experiment import MA_BBOB_Experiment, Experiment
from iohblade.problems import HLP
from iohblade.llm import Gemini_LLM, Ollama_LLM, OpenAI_LLM, Claude_LLM, Multi_LLM
from iohblade.methods import LLaMEA, RandomSearch, EoH, ReEvo
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

    llm = OpenAI_LLM(api_key_openai, "gpt-5-mini-2025-08-07", temperature=1.0)

    budget = 200 # test run (25 iterations of 8 algs)

    DEBUG = False
    if DEBUG:
        budget = 24

    mutation_prompts = [
        "Refine and simplify the strategy of the selected solution to improve it.",  # small mutation
        "Generate a new algorithm that is different from the algorithms you have tried before.", #new random solution
    ]

    #for llm in [llm1]:#, llm2]:
    #RS = RandomSearch(llm, budget=budget) 
    LLaMEA_1 = LLaMEA(llm, budget=budget, name="LLaMEA", mutation_prompts=mutation_prompts, n_parents=4, n_offspring=16, elitism=False)

    methods = [LLaMEA_1] 

    training_instances = None
    test_instances = None

    if DEBUG:
        logger = ExperimentLogger("results/rule-driven-DEBUG")
    else:
        logger = ExperimentLogger("results/rule-driven1")

    problems = []
    if DEBUG:
        problems.append(
            HLP(
                dims=[2],
                budget_factor=200,
                eval_timeout=100,
                name=f"HLP-DEBUG",
                add_info_to_prompt=True,
                full_ioh_log=True,
                specific_high_level_features=["Separable", "Multimodality"],
                ioh_dir=f"{logger.dirname}/ioh",
            )
        )
    else:
        problems.append(
            HLP(
                dims=[2],
                budget_factor=200,
                eval_timeout=100,
                name=f"HLP-DEBUG",
                add_info_to_prompt=False,
                full_ioh_log=False,
                specific_high_level_features=["Separable", "Multimodality"],
                ioh_dir=f"{logger.dirname}/ioh",
            ),
            HLP(
                dims=[2],
                budget_factor=200,
                eval_timeout=100,
                name=f"HLP-DEBUG",
                add_info_to_prompt=True,
                full_ioh_log=False,
                specific_high_level_features=["Separable", "Multimodality"],
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
            n_jobs=1
        )  # test run

    #experiment = MA_BBOB_Experiment(methods=methods, runs=5, seeds=[1,2,3,4,5], dims=[10], budget_factor=2000, budget=budget, eval_timeout=270, show_stdout=True, exp_logger=logger, n_jobs=5) #normal run
    experiment() #run the experiment



    #MA_BBOB_Experiment(methods=methods, llm=llm2, runs=5, dims=[2], budget_factor=1000) #quick run


