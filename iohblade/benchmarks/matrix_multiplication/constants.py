"""Reference ranks for small matrix multiplication tensors.

Each key is a tuple (m, n, p) and maps to a dict with:
  - best_known: best-known rank from literature
  - alphaevolve_best: best rank reported by AlphaEvolve (when known)

When AlphaEvolve value wasn't explicitly provided, it defaults to best_known.
"""

from __future__ import annotations

from typing import Dict, Tuple


MATMUL_RANKS: Dict[Tuple[int, int, int], Dict[str, int]] = {
    # 2x2xp family
    (2, 2, 2): {"best_known": 7, "alphaevolve_best": 7},
    (2, 2, 3): {"best_known": 11, "alphaevolve_best": 11},
    (2, 2, 4): {"best_known": 14, "alphaevolve_best": 14},
    (2, 2, 5): {"best_known": 18, "alphaevolve_best": 18},
    (2, 2, 6): {"best_known": 21, "alphaevolve_best": 21},
    (2, 2, 7): {"best_known": 25, "alphaevolve_best": 25},
    (2, 2, 8): {"best_known": 28, "alphaevolve_best": 28},
    (2, 2, 9): {"best_known": 32, "alphaevolve_best": 32},
    (2, 2, 10): {"best_known": 35, "alphaevolve_best": 35},
    (2, 2, 11): {"best_known": 39, "alphaevolve_best": 39},
    (2, 2, 12): {"best_known": 42, "alphaevolve_best": 42},
    (2, 2, 13): {"best_known": 46, "alphaevolve_best": 46},
    (2, 2, 14): {"best_known": 49, "alphaevolve_best": 49},
    (2, 2, 15): {"best_known": 53, "alphaevolve_best": 53},
    (2, 2, 16): {"best_known": 56, "alphaevolve_best": 56},
    # 2x3xp family
    (2, 3, 3): {"best_known": 15, "alphaevolve_best": 15},
    (2, 3, 4): {"best_known": 20, "alphaevolve_best": 20},
    (2, 3, 5): {"best_known": 25, "alphaevolve_best": 25},
    (2, 3, 6): {"best_known": 30, "alphaevolve_best": 30},
    (2, 3, 7): {"best_known": 35, "alphaevolve_best": 35},
    (2, 3, 8): {"best_known": 40, "alphaevolve_best": 40},
    (2, 3, 9): {"best_known": 45, "alphaevolve_best": 45},
    (2, 3, 10): {"best_known": 50, "alphaevolve_best": 50},
    # 2x4xp family
    (2, 4, 4): {"best_known": 46, "alphaevolve_best": 46},
    (2, 4, 5): {"best_known": 33, "alphaevolve_best": 32},
    (2, 4, 6): {"best_known": 39, "alphaevolve_best": 39},
    (2, 4, 7): {"best_known": 45, "alphaevolve_best": 45},
    (2, 4, 8): {"best_known": 52, "alphaevolve_best": 51},
    # 2x5xp family
    (2, 5, 5): {"best_known": 40, "alphaevolve_best": 40},
    (2, 5, 6): {"best_known": 36, "alphaevolve_best": 36},
    # 3x3xp family
    (3, 3, 3): {"best_known": 23, "alphaevolve_best": 23},
    (3, 3, 4): {"best_known": 29, "alphaevolve_best": 29},
    (3, 3, 5): {"best_known": 47, "alphaevolve_best": 47},
    (3, 3, 6): {"best_known": 40, "alphaevolve_best": 40},
    (3, 3, 7): {"best_known": 49, "alphaevolve_best": 49},
    (3, 3, 8): {"best_known": 55, "alphaevolve_best": 55},
    # 3x4xp family
    (3, 4, 4): {"best_known": 38, "alphaevolve_best": 38},
    (3, 4, 5): {"best_known": 47, "alphaevolve_best": 47},
    (3, 4, 6): {"best_known": 56, "alphaevolve_best": 54},
    (3, 4, 7): {"best_known": 66, "alphaevolve_best": 63},
    (3, 4, 8): {"best_known": 75, "alphaevolve_best": 74},
    # 3x5xp family
    (3, 5, 5): {"best_known": 49, "alphaevolve_best": 49},
    (3, 5, 6): {"best_known": 70, "alphaevolve_best": 68},
    (3, 5, 7): {"best_known": 82, "alphaevolve_best": 80},
    # 4x4xp family
    (4, 4, 4): {"best_known": 58, "alphaevolve_best": 58},
    (4, 4, 5): {"best_known": 62, "alphaevolve_best": 61},
    (4, 4, 7): {"best_known": 76, "alphaevolve_best": 85},
    (4, 4, 8): {"best_known": 98, "alphaevolve_best": 96},
    (4, 4, 9): {"best_known": 104, "alphaevolve_best": 108},
    # 4x5xp family
    (4, 5, 5): {"best_known": 73, "alphaevolve_best": 73},
    (4, 5, 6): {"best_known": 93, "alphaevolve_best": 90},
    # 5x5xp family
    (5, 5, 5): {"best_known": 93, "alphaevolve_best": 93},
    # 6x6xp family
    (6, 6, 6): {"best_known": 153, "alphaevolve_best": 156},
}
