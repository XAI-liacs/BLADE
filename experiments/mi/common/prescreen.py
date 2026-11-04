"""In-loop pre-screening of generated algorithms with an activation probe.

``ScreenedHookedLLM`` scores every generated algorithm with a saved linear
probe (``exp01_probing/scripts/04_export_probes.py``) on the activations it
already captured, and stores the score in
``solution.metadata["prescreen_score"]``. It adds no forward pass: the probe
is a dot product on the ``[n_layers, d_model]`` summary of the query.

The matching problem, ``common.problems.PreScreenedMABBOB``, decides what
LLaMEA sees for a flagged algorithm.
"""

import numpy as np

from iohblade.llm import HookedLLM


def load_probe(path) -> dict:
    """Loads a probe saved by ``04_export_probes.py`` as float32 arrays."""
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


def probe_score(probe: dict, x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float32)
    z = (x - probe["mean"]) / probe["scale"]
    return float(z @ probe["coef"] + probe["intercept"])


class ScreenedHookedLLM(HookedLLM):
    """``HookedLLM`` that scores each generated solution with a probe.

    Args:
        model: As for ``HookedLLM``.
        probe: Probe arrays from ``load_probe``.
        position: Captured position the probe was trained on (``"last"``,
            ``"prompt_mean"`` or ``"gen_mean"``).
        layer: Decoder layer the probe was trained on.
        **kwargs: Passed to ``HookedLLM``.
    """

    def __init__(self, model: str, probe: dict, position: str, layer: int, **kwargs):
        super().__init__(model, **kwargs)
        self.probe = probe
        self.probe_position = position
        self.probe_layer = layer

    def sample_solution(self, session_messages: list, parent_ids=[], **kwargs):
        solution = super().sample_solution(session_messages, parent_ids, **kwargs)
        record = self.last_record
        if record is None or record.get("solution_id") != solution.id:
            return solution  # no capture for this solution
        arrays = record.get("arrays") or {}
        if self.probe_position in arrays and self.probe_layer in record["layers"]:
            x = arrays[self.probe_position][record["layers"].index(self.probe_layer)]
            solution.add_metadata("prescreen_score", probe_score(self.probe, x))
        return solution
