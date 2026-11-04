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


# ----------------------------------------------------------------------
# Decision rules: when is a probe score high enough to flag?
# ----------------------------------------------------------------------
MUTATION_MARKER = "The selected solutions to update are"


def detect_operator(prompt: str, operators: dict[str, str]) -> str:
    """Recovers the LLaMEA operator from the prompt text (``init`` if none)."""
    if MUTATION_MARKER not in prompt:
        return "init"
    for name, text in operators.items():
        if text in prompt:
            return name
    return "other"


class FlagRule:
    """Base class. Rules are fitted on out-of-fold scores of a past campaign
    and then applied online, one solution at a time, within one run.

    ``flag`` sees only the current score, its operator and the scores seen
    earlier in the same run, so every rule can run inside LLaMEA.
    """

    name = "rule"

    def fit(self, scores: np.ndarray, operators: np.ndarray, rate: float):
        return self

    def flag(self, score: float, operator: str, history: list[float]) -> bool:
        raise NotImplementedError

    def to_dict(self) -> dict:
        return {"rule": self.name, **{k: v for k, v in vars(self).items()}}


class FixedRule(FlagRule):
    """One global threshold: the ``1 - rate`` quantile of the past scores."""

    name = "fixed"

    def fit(self, scores, operators, rate):
        self.threshold = float(np.quantile(scores, 1 - rate))
        self.rate = rate
        return self

    def flag(self, score, operator, history):
        return score > self.threshold


class OperatorRule(FlagRule):
    """One threshold per operator, each flagging ``rate`` of that operator."""

    name = "per_operator"

    def fit(self, scores, operators, rate):
        self.rate = rate
        self.fallback = float(np.quantile(scores, 1 - rate))
        self.thresholds = {
            op: float(np.quantile(scores[operators == op], 1 - rate))
            for op in np.unique(operators)
        }
        return self

    def flag(self, score, operator, history):
        return score > self.thresholds.get(operator, self.fallback)


class RiskRule(FlagRule):
    """Calibrated failure probability from (score, operator); flag above the
    probability that flags ``rate`` of the past campaign. Flags more where
    failures are more common, instead of a fixed share per operator."""

    name = "risk"

    def fit(self, scores, operators, rate, labels=None):
        from sklearn.linear_model import LogisticRegression

        self.rate = rate
        self.operators = sorted(np.unique(operators).tolist())
        X = self._features(scores, operators)
        model = LogisticRegression(max_iter=1000).fit(X, labels)
        self.coef = model.coef_[0].tolist()
        self.intercept = float(model.intercept_[0])
        p = self._prob(X)
        self.p_threshold = float(np.quantile(p, 1 - rate))
        return self

    def _features(self, scores, operators):
        ops = np.asarray(operators)
        dummies = [(ops == op).astype(float) for op in self.operators[1:]]
        return np.column_stack([np.asarray(scores, dtype=float), *dummies])

    def _prob(self, X):
        z = X @ np.asarray(self.coef) + self.intercept
        return 1 / (1 + np.exp(-z))

    def probability(self, score, operator) -> float:
        return float(self._prob(self._features([score], [operator]))[0])

    def flag(self, score, operator, history):
        return self.probability(score, operator) > self.p_threshold


class RunningRule(FlagRule):
    """Flag scores above the ``1 - rate`` quantile of the last ``window``
    scores of the same run (the fixed threshold until ``window`` scores are
    seen). Keeps the flag rate at ``rate`` in every phase of the search."""

    name = "running"

    def __init__(self, window: int = 24):
        self.window = window

    def fit(self, scores, operators, rate):
        self.rate = rate
        self.threshold = float(np.quantile(scores, 1 - rate))
        return self

    def flag(self, score, operator, history):
        if len(history) < self.window:
            return score > self.threshold
        return score > np.quantile(history[-self.window :], 1 - self.rate)


RULES = {r.name: r for r in (FixedRule, OperatorRule, RiskRule, RunningRule)}


def rule_from_dict(d: dict) -> FlagRule:
    """Rebuilds a fitted rule from ``FlagRule.to_dict()``."""
    d = dict(d)
    rule = object.__new__(RULES[d.pop("rule")])
    vars(rule).update(d)
    return rule
