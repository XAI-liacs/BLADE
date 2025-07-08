from .eoh import EoH
from .funsearch import funsearch
from .random_search import RandomSearch
from .reevo import ReEvo

try:
    from .llamea import LLaMEA
except Exception:  # pragma: no cover - optional dependency
    LLaMEA = None

__all__ = [
    "LLaMEA",
    "RandomSearch",
    "EoH",
    "funsearch",
    "ReEvo",
]
