from .funsearch import funsearch
from .llamea import LLaMEA
from .random_search import RandomSearch
from .mcts_ahd import MCTS_Method

try:
    from .eoh import EoH
    from .reevo import ReEvo
except Exception:  # pragma: no cover - optional dependency
    ReEvo = None
    EoH = None

__all__ = ["LLaMEA", "RandomSearch", "EoH", "funsearch", "ReEvo", "MCTS_Method"]
