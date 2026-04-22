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
        gpu_name="RTX 5060Ti",
        gpu_type="cuda"
    )

    methods = []
    for llm in [llm]:
        method = LLaMEA(
            llm,
            n_parents=1,
            n_offspring=1,
            budget=budget,
            minimization=kb.minimisation,
        )
        methods.append(method)
    logger = ExperimentLogger(f"results/{kb.task_name}")
    experiment = Experiment(
        methods,
        [kb],
        runs=1,
        budget=budget,
        show_stdout=True,
        exp_logger=logger,
    )

    experiment()