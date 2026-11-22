# isort: skip_file  (import order resolves circular imports; keep it)
from .task_specification import MatMulTensorDecomposition
from .problem_sets import default_problems as default_mat_mul_problems

__all__ = ["MatMulTensorDecomposition", "default_mat_mul_problems"]
