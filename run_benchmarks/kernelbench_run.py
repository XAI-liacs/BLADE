from iohblade.methods import LLaMEA
from iohblade.llm import Ollama_LLM
from iohblade.experiment import Experiment
from iohblade.benchmarks import KernelBench
from iohblade.loggers import ExperimentLogger


if __name__ == '__main__':
    budget = 10

    llm = Ollama_LLM('gemma4')
     
    kb = KernelBench(
        problem_id=1,
        sample_id=1,
        gpu_name="T4",
        gpu_type="cuda:5" # Set "cuda" if only 1 gpu in computer; else set 'cuda:{index}' where index is gpu index to test on.
    )

    methods = []
    for llm in [llm]:
        method = LLaMEA(
            llm,
            n_parents=1,
            n_offspring=1,
            budget=budget,
            minimization=kb.minimization,
        )
        methods.append(method)
    logger = ExperimentLogger(f"results/{kb.name}")
    experiment = Experiment(
        methods,
        [kb],
        runs=1,
        budget=budget,
        show_stdout=True,
        exp_logger=logger,
    )

    experiment()