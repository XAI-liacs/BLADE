from iohblade.experiment import MA_BBOB_Experiment, Experiment
from iohblade.problems import HLP
from iohblade.llm import Gemini_LLM, Ollama_LLM, OpenAI_LLM, Claude_LLM, Multi_LLM
from iohblade.methods import LLaMEA, RandomSearch, EoH, ReEvo
from iohblade.loggers import ExperimentLogger
import numpy as np
import ioh
import os

from iohblade.solution import Solution
from iohblade.utils import code_compare
import lizard


if __name__ == "__main__": # prevents weird restarting behaviour
    api_key_google = os.getenv("GEMINI_API_KEY")
    api_key_openai = os.getenv("OPENAI_API_KEY")
    api_key_claude = os.getenv("CLAUDE_API_KEY")

    #llm = Ollama_LLM("qwen3-coder:30b")
    llm = OpenAI_LLM(api_key_openai, "gpt-5-nano-2025-08-07", temperature=1.0)

    budget = 100 # test run (25 iterations of 8 algs)

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
        logger = ExperimentLogger("results/rule-driven2-DEBUG")
    else:
        logger = ExperimentLogger("results/rule-driven-2")

    all_features = ["Separable", "GlobalLocal", "Multimodality", "Basins", "Homogeneous"] 
    feature_combinations = [["Separable", "Multimodality"],["GlobalLocal", "Multimodality"], ["Separable", "GlobalLocal"]] 
    # for i in range(len(all_features)):
    #     for j in range(i+1, len(all_features)):
    #         feature_combinations.append([all_features[i], all_features[j]])
    #     feature_combinations.append([all_features[i]])

    # not_features = ["NOT Basins", "NOT Homogeneous"] 
    # rest_features = ["Separable", "GlobalLocal", "Multimodality"] 
    # for i in range(len(not_features)):
    #     for j in range(len(rest_features)):
    #         feature_combinations.append([not_features[i], rest_features[j]])
    #     feature_combinations.append([not_features[i]])


    problems = []
    if DEBUG:
        problems.append(
            HLP(
                dim=2,
                budget_factor=200,
                eval_timeout=100,
                name=f"HLP-DEBUG",
                add_info_to_prompt=False,
                add_rules_to_prompt=False,
                full_ioh_log=False,
                specific_high_level_features=["Basins", "Homogeneous"],
                ioh_dir=f"{logger.dirname}/ioh",
            )
        )
        problems.append(
            HLP(
                dim=2,
                budget_factor=200,
                eval_timeout=100,
                name=f"HLP-DEBUG-info",
                add_info_to_prompt=True,
                add_rules_to_prompt=False,
                full_ioh_log=False,
                specific_high_level_features=["Basins", "Homogeneous"],
                ioh_dir=f"{logger.dirname}/ioh",
            )
        )
        problems.append(
            HLP(
                dim=2,
                budget_factor=200,
                eval_timeout=100,
                name=f"HLP-DEBUG-rules",
                add_info_to_prompt=True,
                add_rules_to_prompt=True,
                full_ioh_log=False,
                specific_high_level_features=["Basins", "Homogeneous"],
                ioh_dir=f"{logger.dirname}/ioh",
            )
        )
    else:
        for dim in [5]: #, 30]:
            for feature_set in feature_combinations:
                problems.append(
                    HLP(
                        dim=dim,
                        budget_factor=2000,
                        eval_timeout=360,
                        name=f"HLP-" + "-".join(feature_set),
                        add_info_to_prompt=False,
                        add_rules_to_prompt=False,
                        full_ioh_log=False,
                        specific_high_level_features=feature_set,
                        ioh_dir=f"{logger.dirname}/ioh",
                    )
                )
                problems.append(
                    HLP(
                        dim=dim,
                        budget_factor=2000,
                        eval_timeout=360,
                        name=f"HLP-info-" + "-".join(feature_set),
                        add_info_to_prompt=True,
                        add_rules_to_prompt=False,
                        full_ioh_log=False,
                        specific_high_level_features=feature_set,
                        ioh_dir=f"{logger.dirname}/ioh"
                    )
                )
                problems.append(
                    HLP(
                        dim=dim,
                        budget_factor=2000,
                        eval_timeout=360,
                        name=f"HLP-rules-" + "-".join(feature_set),
                        add_info_to_prompt=True,
                        add_rules_to_prompt=True,
                        full_ioh_log=False,
                        specific_high_level_features=feature_set,
                        ioh_dir=f"{logger.dirname}/ioh",
                    )
                )

    experiment = Experiment(
        methods=methods,
        problems=problems,
        runs=2,
        seeds=[1,2],
        show_stdout=False,
        log_stdout=True,
        exp_logger=logger,
        budget=budget,
        n_jobs=4
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
            n_jobs=2
        )  # test run

    if DEBUG:
        #do some first test
        test_problem = HLP(
                        dim=2,
                        budget_factor=200,
                        eval_timeout=360,
                        name=f"HLP-rules",
                        add_info_to_prompt=True,
                        add_rules_to_prompt=True,
                        full_ioh_log=True,
                        specific_high_level_features=["Basins", "Homogeneous"],
                        ioh_dir=f"{logger.dirname}/ioh",
                    )
        code = """
import numpy as np
import math

class RandomSearch:
    def __init__(self, budget=10000, dim=10):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        self.f_opt = np.inf
        self.x_opt = None
        for i in range(self.budget):
            x = np.random.uniform(func.bounds.lb, func.bounds.ub)
            
            f = func(x)
            if f < self.f_opt:
                self.f_opt = f
                self.x_opt = x
            
        return self.f_opt, self.x_opt"""
        name = "RandomSearch"
        test_solution = Solution(
            name=name,
            code=code
        )
        result = test_problem.evaluate(test_solution)
        print(test_solution.feedback)

    #experiment = MA_BBOB_Experiment(methods=methods, runs=5, seeds=[1,2,3,4,5], dims=[10], budget_factor=2000, budget=budget, eval_timeout=270, show_stdout=True, exp_logger=logger, n_jobs=5) #normal run
    experiment() #run the experiment



    #MA_BBOB_Experiment(methods=methods, llm=llm2, runs=5, dims=[2], budget_factor=1000) #quick run


