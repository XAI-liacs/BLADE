"""Code embeddings and similarity for ShinkaEvolve's novelty rejection.

ShinkaEvolve embeds programs with an embedding API (by default OpenAI's
``text-embedding-3-small``). BLADE's default, ``"ngram"``, needs no API:
a hashed bag of character n-grams of the code with whitespace normalised.
Any function from code to a vector can be used instead.
"""

from __future__ import annotations

import re
import zlib
from collections.abc import Callable

import numpy as np

Embedding = Callable[[str], "np.ndarray | list[float]"]


def ngram_embedding(code: str, n: int = 3, dim: int = 2048) -> np.ndarray:
    """A normalised vector of hashed character ``n``-gram counts."""
    text = re.sub(r"\s+", " ", code).strip()
    vector = np.zeros(dim)
    for i in range(max(len(text) - n + 1, 0)):
        vector[zlib.crc32(text[i : i + n].encode()) % dim] += 1.0
    norm = np.linalg.norm(vector)
    return vector / norm if norm > 0 else vector


def resolve_embedding(embedding: str | Embedding) -> Embedding:
    if callable(embedding):
        return embedding
    if embedding == "ngram":
        return ngram_embedding
    raise ValueError(
        f"embedding must be 'ngram' or a function from code to a vector "
        f"(got {embedding!r})."
    )


def cosine_similarity(a, b) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / denom) if denom > 0 else 0.0
