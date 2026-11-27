"""LLaMEA variants (``variant`` parameter of the llamea method), issue #255."""

import copy
from types import SimpleNamespace

import pytest

from iohblade.config import ConfigError, load_plan
from iohblade.methods import LLaMEA
from iohblade.methods.llamea import resolve_variant

BASE = {
    "experiment": {"output_dir": "results/x", "runs": 1, "budget": 7},
    "llms": {"a": {"provider": "dummy", "model": "m-a"}},
    "problems": [{"type": "circle_packing", "name": "circles"}],
}


def _plan_data(params, name=None):
    data = copy.deepcopy(BASE)
    method = {"type": "llamea", "llm": "a", "params": params}
    if name is not None:
        method["name"] = name
    data["methods"] = [method]
    return data


def _issues(params):
    with pytest.raises(ConfigError) as exc:
        load_plan(_plan_data(params), env={})
    return "\n".join(str(i) for i in exc.value.issues)


@pytest.fixture
def llm():
    return SimpleNamespace(model="m")


def test_resolve_variant_names():
    assert resolve_variant("bo")[0] == "bo"
    assert resolve_variant("LLaMEA-HPO")[0] == "hpo"
    assert resolve_variant("llamea")[0] == "llamea"
    with pytest.raises(ValueError, match="Unknown LLaMEA variant"):
        resolve_variant("nope")
    with pytest.raises(ValueError, match="multi-objective"):
        resolve_variant("mo")


def test_variant_sets_name_and_config(llm):
    method = LLaMEA(llm, budget=10, variant="bo")
    assert method.name == "LLaMEA-BO"
    assert method.get_config()["config"]["variant"] == "bo"
    assert method.to_dict()["variant"] == "bo"
    assert LLaMEA(llm, budget=10, name="Mine", variant="bo").name == "Mine"
    assert LLaMEA(llm, budget=10).name == "LLaMEA"


def test_hpo_variant_uses_hpo_unless_overridden(llm):
    assert LLaMEA(llm, budget=10, variant="hpo").uses_hpo
    assert not LLaMEA(llm, budget=10, variant="hpo", HPO=False).uses_hpo
    assert not LLaMEA(llm, budget=10, variant="bo").uses_hpo
    assert LLaMEA(llm, budget=10, HPO=True).uses_hpo


def test_multi_objective_is_rejected(llm):
    with pytest.raises(ValueError, match="multi-objective"):
        LLaMEA(llm, budget=10, variant="mo")
    with pytest.raises(ValueError, match="multi-objective"):
        LLaMEA(llm, budget=10, multi_objective=True)


def test_sage_needs_xgboost(llm, monkeypatch):
    monkeypatch.setattr(
        "iohblade.methods.llamea.importlib.util.find_spec", lambda name: None
    )
    with pytest.raises(ImportError, match="xgboost"):
        LLaMEA(llm, budget=10, variant="sage")
    # turning the preset off needs no xgboost
    LLaMEA(llm, budget=10, variant="sage", feature_guided_mutation=False)


def test_call_builds_the_variant_with_blade_prompts_and_budget(llm, monkeypatch):
    seen = {}

    def fake_run(self):
        seen.update(
            cls=type(self).__name__,
            budget=self.budget,
            n_parents=self.n_parents,
            n_offspring=self.n_offspring,
            task_prompt=self.task_prompt,
            minimization=self.minimization,
            ops={o.name: o.weight for o in self.operators},
        )
        return "best"

    monkeypatch.setattr("llamea.llamea.LLaMEA.run", fake_run)
    problem = SimpleNamespace(
        task_prompt="PROBLEM TASK",
        example_prompt="EXAMPLE",
        format_prompt="FORMAT",
        minimisation=True,
    )
    method = LLaMEA(llm, budget=12, variant="bo", n_offspring=8)
    assert method(problem) == "best"
    assert seen["cls"] == "LLaMEA_BO"
    assert seen["budget"] == 12  # BLADE's budget, not the preset's 100
    assert (seen["n_parents"], seen["n_offspring"]) == (4, 8)  # preset + override
    assert seen["task_prompt"] == "PROBLEM TASK"  # the problem's prompt
    assert seen["minimization"] is True
    assert seen["ops"] == {"mutation": 0.4, "crossover": 0.6}


def test_yaml_variant_builds_named_methods():
    plan = load_plan(_plan_data({"variant": "hpo"}, name="L-{llm}"), env={})
    (method,) = plan.methods
    assert method.variant == "hpo" and method.uses_hpo
    assert method.name == "L-a"
    plan = load_plan(_plan_data({"variant": "LLaMEA-BO"}), env={})
    assert plan.methods[0].variant == "bo"
    assert plan.methods[0].name == "LLaMEA-BO"


def test_yaml_variant_errors_point_at_the_parameter():
    issues = _issues({"variant": "bayes"})
    assert "methods[0].params.variant: Unknown LLaMEA variant 'bayes'" in issues
    issues = _issues({"variant": "mo"})
    assert "methods[0].params.variant: LLaMEA's multi-objective mode" in issues
    issues = _issues({"multi_objective": True})
    assert "methods[0].params.multi_objective: LLaMEA's multi-objective" in issues
    issues = _issues({"variant": 3})
    assert "methods[0].params.variant: 'variant' must be a name" in issues


def test_yaml_operators_use_the_variant_n_parents():
    # The bo preset has 4 parents, so an operator with 5 parents is too many.
    params = {"variant": "bo", "operators": [{"prompt": "x", "number_of_parents": 5}]}
    assert "than n_parents (4)" in _issues(params)
