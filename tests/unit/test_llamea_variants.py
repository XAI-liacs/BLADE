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


def _problem(tags, name="p", deps=None):
    from iohblade.tags import Benchmark, PrimaryCategories  # noqa: F401

    return SimpleNamespace(
        name=name,
        task_prompt="PROBLEM TASK",
        example_prompt="EXAMPLE",
        format_prompt="FORMAT",
        minimisation=True,
        dependencies=list(deps or ["numpy>=2"]),
        get_config=lambda: {"tags": tags},
    )


def _bbo():
    from iohblade.tags import PrimaryCategories

    return _problem([PrimaryCategories.BBO, "noiseless"], name="bbo")


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
    problem = _bbo()
    method = LLaMEA(llm, budget=12, variant="bo", n_offspring=8)
    assert method(problem) == "best"
    assert seen["cls"] == "LLaMEA_BO"
    assert seen["budget"] == 12  # BLADE's budget, not the preset's 100
    assert (seen["n_parents"], seen["n_offspring"]) == (4, 8)  # preset + override
    # The problem's prompt (interface, bounds) followed by LLaMEA-BO's part.
    assert seen["task_prompt"].startswith("PROBLEM TASK\n")
    assert "Bayesian optimization (BO)" in seen["task_prompt"]
    assert seen["minimization"] is True
    assert seen["ops"] == {"mutation": 0.4, "crossover": 0.6}
    assert any(d.startswith("scikit-learn") for d in problem.dependencies)


def test_other_variants_keep_the_problem_prompt(llm, monkeypatch):
    seen = {}
    monkeypatch.setattr(
        "llamea.llamea.LLaMEA.run",
        lambda self: seen.setdefault("prompt", self.task_prompt),
    )
    LLaMEA(llm, budget=5, variant="hpo")(_bbo())
    assert seen["prompt"] == "PROBLEM TASK"


def test_bo_is_refused_for_non_bbo_problems(llm):
    from iohblade.methods.llamea import bo_unsupported_reason
    from iohblade.tags import Benchmark, PrimaryCategories

    assert bo_unsupported_reason(_bbo()) is None
    assert "only runs on black-box optimisation" in bo_unsupported_reason(
        _problem(["combinatorial optimisation"])
    )
    construction = _problem([PrimaryCategories.BBO, Benchmark.ANALYSIS])
    assert "constructed sequence" in bo_unsupported_reason(construction)
    with pytest.raises(ValueError, match="black-box optimisation"):
        LLaMEA(llm, budget=5, variant="bo")(_problem([]))


def test_configure_bo_adds_packages_once(llm):
    from iohblade.methods.llamea import BO_DEPENDENCIES, configure_bo

    bo = LLaMEA(llm, budget=5, variant="bo")
    problem = _bbo()
    notes = configure_bo([bo], [problem])
    assert all(d in problem.dependencies for d in BO_DEPENDENCIES)
    assert "scikit-learn" in notes[0]
    configure_bo([bo], [problem])  # idempotent
    assert len(problem.dependencies) == 1 + len(BO_DEPENDENCIES)
    # An existing pin of a package is kept, not duplicated.
    pinned = _problem([], deps=["numpy>=2", "scikit-learn==1.5.2"])
    pinned.get_config = _bbo().get_config
    configure_bo([bo], [pinned])
    assert [d for d in pinned.dependencies if "scikit" in d] == ["scikit-learn==1.5.2"]
    # Without a bo method nothing changes.
    other = _bbo()
    assert configure_bo([LLaMEA(llm, budget=5)], [other]) == []
    assert other.dependencies == ["numpy>=2"]
    with pytest.raises(ValueError, match="LLaMEA-BO"):
        configure_bo([bo], [_problem(["combinatorial optimisation"])])


def test_yaml_variant_builds_named_methods():
    plan = load_plan(_plan_data({"variant": "hpo"}, name="L-{llm}"), env={})
    (method,) = plan.methods
    assert method.variant == "hpo" and method.uses_hpo
    assert method.name == "L-a"
    data = _plan_data({"variant": "LLaMEA-BO"})
    data["problems"] = [{"type": "bbob_sbox", "name": "sbox"}]  # bo needs BBO
    plan = load_plan(data, env={})
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


def test_yaml_bo_is_refused_for_non_bbo_problems():
    # BASE uses circle_packing, which is not a black-box optimisation problem.
    issues = _issues({"variant": "bo"})
    assert "methods[0].params.variant: variant 'bo' designs Bayesian" in issues
    assert "'circles'" in issues


def test_yaml_bo_on_bbob_adds_packages_and_a_note():
    data = _plan_data({"variant": "bo"})
    data["problems"] = [{"type": "bbob_sbox", "name": "sbox"}]
    plan = load_plan(data, env={})
    (problem,) = plan.problems
    assert any(d.startswith("scikit-learn") for d in problem.dependencies)
    assert any(d.startswith("scipy") for d in problem.dependencies)
    assert any("variant 'bo'" in n and "scikit-learn" in n for n in plan.notes)
