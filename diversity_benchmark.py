# Setup a diversity experiment using real world benchmark.
# Will be using some very easy, Hexagonal Packing (hard), in hopes
# of desciminating between the various algorithms.


from iohblade.benchmarks.packing import get_hexagon_packing_problems

from iohblade.llm import Gemini_LLM, OpenAI_LLM
from iohblade.methods import LLaMEA
from iohblade.loggers import ExperimentLogger
from iohblade.experiment import Experiment

from iohblade.utils import code_compare
from os import getenv

def ast_distance(solution1, solution2):
    # AST distance calculation, returns a ratio between 0 and 1
    return code_compare(solution1.code, solution2.code)


if __name__ == '__main__':
    DEBUG = True

    budget = 200
    if DEBUG:
        budget = 10
    
    api_key = getenv('GOOGLE_API_KEY')
    llm = Gemini_LLM(api_key, "gemini-2.0-flash")

    mutation_prompts = [
        "Refine the strategy of the selected solution to improve it.",  # small mutation
        "Generate a new algorithm that is different from the algorithms you have tried before.", #new random solution
    ]



    if DEBUG:
        logger = ExperimentLogger("results/Hexagon_Packing_Problem")
    else:
        logger = ExperimentLogger("results/Hexagon_Packing_Problem")
    
    problem = get_hexagon_packing_problems(False)[0]
    
    LLaMEA_adaptive_prompting = LLaMEA(llm, budget=budget, name="Adaptive-Prompt", mutation_prompts=mutation_prompts, n_parents=8, n_offspring=8, elitism=True, adaptive_prompt=True, minimization=problem.minimisation)
    
    LLaMEA_novelty = LLaMEA(llm, budget=budget, name="Novelty-Search", mutation_prompts=mutation_prompts, n_parents=8, n_offspring=8, elitism=True, niching="novelty", distance_metric=ast_distance, novelty_archive_size=100, novelty_k=5, minimization=problem.minimisation)
    
    LLaMEA_fitness_sharing = LLaMEA(llm, budget=budget, name="Fitness-Sharing", mutation_prompts=mutation_prompts, n_parents=8, n_offspring=8, elitism=True, niching="sharing", distance_metric=ast_distance, adaptive_niche_radius=True, minimization=problem.minimisation)
    LLaMEA_fitness_clearing = LLaMEA(llm, budget=budget, name="Fitness-Cleaning", mutation_prompts=mutation_prompts, n_parents=8, n_offspring=8, elitism=True, niching="clearing", distance_metric=ast_distance, adaptive_niche_radius=True, minimization=problem.minimisation)

    LLaMEA_MAP_elites = LLaMEA(llm, budget=budget, name="MAP-Elites", mutation_prompts=mutation_prompts, n_parents=8, n_offspring=8, elitism=True, niching="map_elites", map_elites_bins=5, minimization=problem.minimisation)

    methods = [LLaMEA_adaptive_prompting, LLaMEA_novelty, LLaMEA_fitness_clearing, LLaMEA_fitness_sharing, LLaMEA_MAP_elites]
    
    experiment = Experiment(
        methods=methods,
        problems=[problem],
        runs=1 if DEBUG else 5,
        budget=budget,
        show_stdout=DEBUG,
        n_jobs=5,
        exp_logger=logger

    )

    experiment()
    



