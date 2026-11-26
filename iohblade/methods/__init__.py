# isort: skip_file  (import order resolves circular imports; keep it)
from .funsearch import FunSearch, funsearch
from .llamea import LLaMEA
from .random_search import RandomSearch
from .shinka_evolve import ShinkaEvolve
from .lhns import LHNS_Method
from .mcts_ahd import MCTS_Method

# Optional methods: they need the `methods` dependency group
# (`uv sync --group methods`). Without it the other methods must still
# import; resolving one of these then reports the missing group.
try:
    from .moeh_method.mutationtype import MutationType
    from .moeh_method.moeh import MoEH_Method
except ImportError:  # pragma: no cover - optional dependency (codebleu)
    MutationType = None
    MoEH_Method = None

try:
    from .eoh import EoH
    from .reevo import ReEvo
except Exception:  # pragma: no cover - optional dependency
    ReEvo = None
    EoH = None


__all__ = [
    "LLaMEA",
    "RandomSearch",
    "EoH",
    "FunSearch",
    "funsearch",
    "ShinkaEvolve",
    "ReEvo",
    "LHNS_Method",
    "MCTS_Method",
    "MoEH_Method",
    "MutationType",
]
