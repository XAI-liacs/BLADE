"""Code-diversity measures for sets of generated algorithms.

Three views of how different two algorithms are:

* ``codebleu``: 1 - CodeBLEU (Ren et al., 2020), symmetrised (CodeBLEU is
  computed both ways and averaged). CodeBLEU mixes token n-gram overlap,
  keyword-weighted n-grams, AST subtree match and data-flow match, so it
  is sensitive to both surface form and structure.
* ``ast``: LLaMEA's own ``code_distance`` (1 - ``difflib`` ratio of the
  dumped ASTs), the distance LLaMEA's niching uses. Ignores comments and
  formatting.
* ``concepts``: Jaccard distance between the sets of algorithmic concepts
  in the code (experiment 3's regex labels: DE, PSO, CMA-style covariance,
  restarts, surrogates, ...). Coarse but semantic: two DE variants are at
  distance 0.

Set-level measures: mean pairwise distance, and for a sequence (a run in
generation order) the *novelty* of each item, its distance to the nearest
earlier item (``k = 1``). The family view also has the Shannon entropy of
the distribution of concept sets and the number of distinct sets.
"""

import ast
import difflib
import math
import warnings
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from itertools import combinations

import numpy as np

METRICS = ("codebleu", "ast", "concepts")


def codebleu_similarity(a: str, b: str) -> float:
    """Symmetrised CodeBLEU in [0, 1]; nan if either side is empty."""
    from codebleu import calc_codebleu

    if not a.strip() or not b.strip():
        return math.nan
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ab = calc_codebleu([a], [b], lang="python")["codebleu"]
        ba = calc_codebleu([b], [a], lang="python")["codebleu"]
    return 0.5 * (ab + ba)


def _ast_dump(code: str) -> str | None:
    try:
        return ast.dump(ast.parse(code))
    except Exception:  # SyntaxError, ValueError (null bytes), RecursionError
        return None


def ast_similarity(a: str, b: str) -> float:
    """1 - LLaMEA's ``code_distance``; nan if either side does not parse."""
    da, db = _ast_dump(a), _ast_dump(b)
    if da is None or db is None:
        return math.nan
    return difflib.SequenceMatcher(None, da, db).ratio()  # as LLaMEA


def jaccard_similarity(a: frozenset, b: frozenset) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def _pair(args):
    metric, a, b = args
    if metric == "codebleu":
        return codebleu_similarity(a, b)
    return ast_similarity(a, b)


def similarity_matrix(
    codes: list[str], metric: str, pool: ProcessPoolExecutor | None = None
) -> np.ndarray:
    """Symmetric pairwise similarity (1 on the diagonal, nan if undefined).

    ``metric`` is ``"codebleu"`` or ``"ast"``; pass a process pool to
    spread the pairs over cores (CodeBLEU costs ~10 ms per pair).
    """
    n = len(codes)
    S = np.eye(n)
    pairs = list(combinations(range(n), 2))
    jobs = [(metric, codes[i], codes[j]) for i, j in pairs]
    if pool is None:
        vals = [_pair(j) for j in jobs]
    else:
        vals = list(pool.map(_pair, jobs, chunksize=64))
    for (i, j), v in zip(pairs, vals):
        S[i, j] = S[j, i] = v
    return S


def concept_similarity_matrix(sets: list[frozenset]) -> np.ndarray:
    n = len(sets)
    S = np.eye(n)
    for i, j in combinations(range(n), 2):
        S[i, j] = S[j, i] = jaccard_similarity(sets[i], sets[j])
    return S


def mean_pairwise_distance(S: np.ndarray, idx=None) -> float:
    """Mean of 1 - S over the distinct pairs of ``idx`` (all if None)."""
    if idx is not None:
        S = S[np.ix_(idx, idx)]
    n = len(S)
    if n < 2:
        return math.nan
    iu = np.triu_indices(n, 1)
    return float(np.nanmean(1 - S[iu]))


def novelty(S: np.ndarray) -> np.ndarray:
    """Distance of item i to its nearest predecessor (nan for the first)."""
    n = len(S)
    out = np.full(n, np.nan)
    for i in range(1, n):
        prev = S[i, :i]
        if np.isfinite(prev).any():
            out[i] = 1 - np.nanmax(prev)
    return out


def family_entropy(sets: list[frozenset]) -> tuple[float, int]:
    """Shannon entropy (bits) of the concept-set distribution, and the
    number of distinct concept sets."""
    counts = Counter(sets)
    p = np.array(list(counts.values()), dtype=float)
    p /= p.sum()
    return float(-(p * np.log2(p)).sum()), len(counts)
