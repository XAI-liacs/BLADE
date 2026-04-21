import torch

from iohblade.problem import Problem, BASE_DEPENDENCIES

from kernelbench.compile import WorkArgs
from kernelbench.dataset import construct_kernelbench_dataset
from kernelbench.prompt_constructor_toml import get_prompt_for_backend
from kernelbench.eval import eval_kernel_against_ref, KernelExecResult

from iohblade.solution import Solution


class KernelBench(Problem):
    def __init__(
        self,
        problem_id: int,
        sample_id: int,
        gpu_name: str,
        precision: str = "fp32",
        gpu_type: str = "cuda",
        backend: str = "triton",
        logger=None,
        training_instances=None,
        test_instances=None,
        name="Problem",
        eval_timeout=6000,
        dependencies=BASE_DEPENDENCIES,
        imports=None,
    ):
        dependencies = list(dependencies)
        dependencies.append("torch")

        self.workargs = WorkArgs(
            problem_id, sample_id=sample_id, device=torch.device(gpu_type)
        )
        self.minimization = True
        self.gpu = gpu_name
        dataset = construct_kernelbench_dataset(
            1, "huggingface", problem_ids=[problem_id]
        )
        self.problem = dataset.get_problem_by_id(problem_id)
        self.backend = backend
        self.precision = precision

        super().__init__(
            logger,
            training_instances,
            test_instances,
            name,
            eval_timeout,
            dependencies,
            imports,
        )

    def get_prompt(self):
        return get_prompt_for_backend(
            self.problem.code, self.backend, gpu_name=self.gpu, precision=self.precision
        )

    def _stringify_metadata(self, metadata: dict) -> str:
        string = "{\n"
        for key, value in metadata.items():
            string += f"\t{key} : {value}\n"
        string += "\n}"
        return string

    def evaluate(self, solution: Solution) -> Solution:
        try:
            evaluation_result = eval_kernel_against_ref(
                self.problem.code,
                solution.code,
                device=self.workargs.device,
                backend=self.backend,
            )
            feedback = self._stringify_metadata(evaluation_result.metadata)
            match ((evaluation_result.compiled, evaluation_result.correctness)):
                case (False, False):
                    raise ValueError(
                        f"Unable to compile the kernel or incorrect kernel behaviour..\nAdditional Feedback:\n{feedback}"
                    )
                case (False, True):
                    raise ValueError(
                        f"Unable to compile the kernel..\nAdditional Feedback:\n{feedback}"
                    )
                case (True, False):
                    raise ValueError(
                        f"Incorrect Compiler Behaviour..\nAdditional Feedback:\n{feedback}"
                    )
                case (True, True):
                    if evaluation_result.ref_runtime == -1.0:
                        raise ValueError(
                            f"Reference Runtime was not set...\nAdditional Feedback:\n{feedback}"
                        )
                    if evaluation_result.runtime == -1.0:
                        raise ValueError(
                            f"LLM Code Runtime was not set...\nAdditional Feedback:\n{feedback}"
                        )
                    score = evaluation_result.ref_runtime / evaluation_result.runtime
                    solution = solution.set_scores(
                        score, f"Got Score {score}...\nAdditional Feedback:\n{feedback}"
                    )
        except Exception as e:
            solution = solution.set_scores(float("inf"), str(e), e)
        return solution

    def to_dict(self):
        return super().to_dict()

    def test(self, solution: Solution):
        return self.evaluate(solution)


if __name__ == "__main__":
    import pickle

    kb = KernelBench(2, 1, "T4")
    print(kb.get_prompt())
    try:
        _ = pickle.dumps(kb)
        print("Pickle was successful.")
    except Exception as e:
        print(e)
