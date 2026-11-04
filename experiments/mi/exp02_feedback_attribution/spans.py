"""Segment a LLaMEA mutation prompt into named spans.

LLaMEA builds a mutation prompt from fixed pieces (see
``llamea.LLaMEA.construct_prompt``). The markers below split it into:

* ``task``: the problem description (fixed across the run);
* ``population``: names, descriptions and scores of the population;
* ``parent_description``: the one-line description of the selected parent;
* ``parent_code``: the parent's code;
* ``feedback``: the parent's evaluation feedback (score, or the error);
* ``operator``: the mutation instruction (``refine`` / ``new`` / ...);
* ``format``: the output-format instruction.

Spans are character ranges in the user message. They tile the message
from the start of ``task`` to the end of ``format``.
"""

import re

SPAN_NAMES = (
    "task",
    "population",
    "parent_description",
    "parent_code",
    "feedback",
    "operator",
    "format",
)

_MARKERS = [
    ("population", "The current population of algorithms already evaluated"),
    ("parent_description", "The selected solutions to update are:"),
    ("parent_code", "## With Code"),
    ("feedback", "## Feedback:"),
]
_FORMAT_MARKER = "Give an excellent and novel heuristic algorithm"


def segment(text: str, operator_text: str) -> dict[str, tuple[int, int]] | None:
    """Returns ``{span: (start, end)}`` or ``None`` if a marker is missing.

    Only single-parent mutation prompts are supported.
    """
    starts = {"task": 0}
    for name, marker in _MARKERS:
        if text.count(marker) != 1:
            return None
        starts[name] = text.index(marker)
    op = text.rfind(operator_text)
    fmt = text.rfind(_FORMAT_MARKER)
    if op < starts["feedback"] or fmt < op:
        return None
    starts["operator"] = op
    starts["format"] = fmt
    order = sorted(starts.items(), key=lambda kv: kv[1])
    if [k for k, _ in order] != list(SPAN_NAMES):
        return None
    bounds = [s for _, s in order] + [len(text)]
    return {name: (bounds[i], bounds[i + 1]) for i, name in enumerate(SPAN_NAMES)}


def ablate(text: str, spans: dict, name: str) -> str:
    """Deletes one span; the rest of the prompt is unchanged."""
    start, end = spans[name]
    return text[:start] + text[end:]


def feedback_score(text: str, spans: dict) -> float | None:
    """The parent's AOCC as written in its feedback, if present."""
    s, e = spans["feedback"]
    m = re.search(r"scored ([0-9.]+) on AOCC", text[s:e])
    return float(m.group(1)) if m else None
