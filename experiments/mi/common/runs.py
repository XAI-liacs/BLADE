"""Load BLADE run directories written with ``HookedLLM``.

A run directory contains BLADE's ``log.jsonl`` (one evaluated solution per
line) and ``mi_activations/`` (``index.jsonl`` + one ``.npz`` per LLM query).
"""

import json
import math
from pathlib import Path

import numpy as np


def iter_run_dirs(root: str | Path):
    """Yields every run directory below ``root`` that has MI records."""
    for index in sorted(Path(root).rglob("mi_activations/index.jsonl")):
        yield index.parent.parent


def read_jsonl(path: str | Path) -> list[dict]:
    path = Path(path)
    if not path.exists():
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def fitness_scalar(value) -> float:
    """Maps BLADE fitness values (float, None, inf, dict) to a float or nan."""
    if isinstance(value, dict):  # multi-objective: mean, as Fitness.__float__
        vals = [fitness_scalar(v) for v in value.values()]
        return float(np.mean(vals)) if vals else math.nan
    try:
        value = float(value)
    except (TypeError, ValueError):
        return math.nan
    return value if math.isfinite(value) else math.nan


def load_run(run_dir: str | Path) -> tuple[dict[str, dict], list[dict]]:
    """Returns ``(solutions by id, MI records)`` for one run directory.

    If a solution was logged more than once, the last entry wins.
    """
    run_dir = Path(run_dir)
    solutions = {s["id"]: s for s in read_jsonl(run_dir / "log.jsonl")}
    records = read_jsonl(run_dir / "mi_activations" / "index.jsonl")
    return solutions, records


def load_arrays(run_dir: str | Path, record: dict) -> dict[str, np.ndarray]:
    """Loads the activation arrays (``last``, ``prompt_mean``, ``gen_mean``)."""
    if not record.get("file"):
        return {}
    with np.load(Path(run_dir) / "mi_activations" / record["file"]) as z:
        return {k: z[k] for k in z.files}


def prompt_text(record: dict) -> str:
    return "\n".join(m["content"] for m in record.get("prompt", []))
