"""Filesystem layout of the MI research folder.

Large data (activations, BLADE logs) lives under ``<experiment>/data`` and is
git-ignored. Set ``MI_DATA_ROOT`` to put it on another disk; the experiment
name is then appended (``$MI_DATA_ROOT/exp01_probing/...``).
"""

import os
from pathlib import Path

MI_ROOT = Path(__file__).resolve().parents[1]


def data_dir(experiment: str, *parts: str) -> Path:
    root = os.environ.get("MI_DATA_ROOT")
    base = Path(root) / experiment if root else MI_ROOT / experiment / "data"
    path = base.joinpath(*parts)
    path.mkdir(parents=True, exist_ok=True)
    return path


def results_dir(experiment: str, *parts: str) -> Path:
    path = (MI_ROOT / experiment / "results").joinpath(*parts)
    path.mkdir(parents=True, exist_ok=True)
    return path
