from iohblade.experiment import MA_BBOB_Experiment, Experiment
from iohblade.problems import HLP
from iohblade.llm import Gemini_LLM, Ollama_LLM, OpenAI_LLM, Claude_LLM, Multi_LLM
from iohblade.methods import LLaMEA, RandomSearch, EoH, ReEvo
from iohblade.loggers import ExperimentLogger
import numpy as np
import ioh
import os
import json
from ioh import get_problem, wrap_problem
from ioh import logger as ioh_logger
from iohblade.baselines.modcma import ModularCMAES

from iohblade.utils import code_compare
import lizard

function_files = [
            "gpt-5-nano-ELA-Basins_Homogeneous.jsonl",
            "gpt-5-nano-ELA-Basins.jsonl",
            "gpt-5-nano-ELA-GlobalLocal_Basins.jsonl",
            "gpt-5-nano-ELA-GlobalLocal_Homogeneous.jsonl",
            "gpt-5-nano-ELA-GlobalLocal_Multimodality.jsonl",
            "gpt-5-nano-ELA-GlobalLocal.jsonl",
            "gpt-5-nano-ELA-Homogeneous.jsonl",
            "gpt-5-nano-ELA-Multimodality_Basins.jsonl",
            "gpt-5-nano-ELA-Multimodality.jsonl",
            "gpt-5-nano-ELA-NOT Basins_GlobalLocal.jsonl",
            "gpt-5-nano-ELA-NOT Basins_Multimodality.jsonl",
            "gpt-5-nano-ELA-NOT Basins_Separable.jsonl",
            "gpt-5-nano-ELA-NOT Basins.jsonl",
            "gpt-5-nano-ELA-NOT Homogeneous_GlobalLocal.jsonl",
            "gpt-5-nano-ELA-NOT Homogeneous_Multimodality.jsonl",
            "gpt-5-nano-ELA-NOT Homogeneous_Separable.jsonl",
            "gpt-5-nano-ELA-NOT Homogeneous.jsonl",
            "gpt-5-nano-ELA-Separable_Basins.jsonl",
            "gpt-5-nano-ELA-Separable_GlobalLocal.jsonl",
            "gpt-5-nano-ELA-Separable_Homogeneous.jsonl",
            "gpt-5-nano-ELA-Separable_Multimodality.jsonl",
            "gpt-5-nano-ELA-Separable.jsonl",
            "gpt-5-nano-ELA-Multimodality_Homogeneous.jsonl"
        ]

def get_generated_problem(function_file, entry, instance, dim):
        

        n = len(data)
        if instance >= n:
            raise ValueError(f"Instance index {instance} is out of range for available functions ({n} functions).")
        
        entry = data[instance]
        code = entry["code"]

        exec(code, globals())
        cls = globals()[entry["name"]]
        objective_f = cls(dim=dim).f


        # --- WRAP PROBLEM ---
        wrap_problem(
            objective_f,
            function_file.replace(".jsonl", ""),
            ioh.ProblemClass.REAL,
            dimension=dim,
            instance=instance,
            lb=-5,
            ub=5,
        )
        return get_problem(function_file.replace(".jsonl", ""), instance=instance, dimension=dim)

if __name__ == "__main__": # prevents weird restarting behaviour
    # process files and retrieve the optimum of each function for 2, 5 and 10 dimensions
    for dim in [30]: #TODO (5,10)
        for function_file in function_files:
            print("Processing", function_file, "for dim", dim)
            # --- LOAD JSON LINES ---
            current_dir = os.path.dirname(os.path.abspath(__file__))
            with open("/home/neocortex/repos/BLADE/iohblade/problems/generated_problems/" + function_file, "r") as f:
                data = [json.loads(line) for line in f if line.strip()]
            
            n = len(data)
            for instance in range(n):
                entry = data[instance]
                problem = get_generated_problem(function_file, entry, instance, dim)

                f_opt = np.inf
                x_opt = None

                #run a large cma-es with restarts to find optimum
                optimizer = ModularCMAES(budget=10000*dim, dim=dim)
                for _ in range(5): # 5 restarts
                    optimizer(problem)
                    if problem.state.current_best.y < f_opt:
                        f_opt = problem.state.current_best.y
                        x_opt = problem.state.current_best.x
                data[instance]["f_opt_dim_" + str(dim)] = f_opt
                data[instance]["x_opt_dim_" + str(dim)] = x_opt.tolist()
                print(f"Function file: {function_file}, Instance: {instance}, Dim: {dim}, f_opt: {f_opt}, x_opt: {x_opt}")
            # now write the updated file back:
            with open("/home/neocortex/repos/BLADE/iohblade/problems/generated_problems/" + function_file, "w") as f:
                for entry in data:
                    f.write(json.dumps(entry) + "\n")
            
        


