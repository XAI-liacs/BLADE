import copy
import textwrap
from pathlib import Path

import pytest

from iohblade.config import (
    ConfigError,
    ExperimentConfig,
    json_schema,
    load_experiment,
    load_plan,
)
from iohblade.experiment import Experiment

BASE = {
    "experiment": {"output_dir": "results/x", "runs": 2, "budget": 7},
    "llms": {"a": {"provider": "dummy", "model": "m-a"}, "b": {"provider": "dummy"}},
    "methods": [{"type": "random_search", "llm": "a"}],
    "problems": [{"type": "circle_packing", "name": "circles"}],
}


def config(**changes):
    data = copy.deepcopy(BASE)
    for key, value in changes.items():
        data[key] = value
    return data


def issues_of(data, **kwargs):
    with pytest.raises(ConfigError) as exc:
        load_plan(data, env=kwargs.pop("env", {}), **kwargs)
    return [str(i) for i in exc.value.issues]


def test_minimal_config_builds():
    plan = load_plan(config(), env={})
    assert len(plan.methods) == 1 and plan.methods[0].budget == 7
    assert plan.methods[0].llm.model == "m-a"
    assert [p.name for p in plan.problems] == ["circles"]
    assert plan.seeds == [0, 1] and plan.n_runs == 2
    assert plan.output_dir == (Path.cwd() / "results/x").resolve()
    assert "Runs: 1 methods x 1 problems x 2 seeds = 2" in plan.summary()


def test_method_is_expanded_per_llm_with_name_template():
    methods = [
        {
            "type": "llamea",
            "name": "L-{llm}",
            "llm": ["a", "b"],
            "budget": 3,
            "params": {"n_parents": 1},
        },
    ]
    plan = load_plan(config(methods=methods), env={})
    assert [m.name for m in plan.methods] == ["L-a", "L-b"]
    assert plan.method_llm_names == ["a", "b"]
    assert all(m.budget == 3 for m in plan.methods)
    assert plan.methods[0].kwargs == {"n_parents": 1}


def test_name_is_set_on_methods_without_name_parameter():
    methods = [{"type": "mcts_ahd", "name": "MCTS", "llm": "a"}]
    plan = load_plan(config(methods=methods), env={})
    assert plan.methods[0].name == "MCTS"


def test_seeds_override_runs():
    data = config()
    data["experiment"]["seeds"] = [4, 2]
    assert load_plan(data, env={}).seeds == [4, 2]


def test_output_dir_is_relative_to_the_yaml_file(tmp_path):
    path = tmp_path / "sub" / "exp.yml"
    path.parent.mkdir()
    import yaml

    path.write_text(yaml.safe_dump(config()))
    plan = load_plan(path, env={})
    assert plan.output_dir == (tmp_path / "sub" / "results" / "x").resolve()
    assert not plan.output_dir.exists()  # building has no side effects


def test_enum_parameters_accept_strings():
    from ioh import ProblemClass

    problems = [{"type": "bbob_sbox", "params": {"problem_type": "BBOB", "dims": [2]}}]
    plan = load_plan(config(problems=problems), env={})
    assert plan.problems[0].problem_type == ProblemClass.BBOB


def test_suite_expands_into_problems():
    problems = [{"suite": "hexagon_packing", "params": {"get_best": False}}]
    plan = load_plan(config(problems=problems), env={})
    assert len(plan.problems) == 2
    assert len({p.name for p in plan.problems}) == 2


def test_multi_llm():
    llms = copy.deepcopy(BASE["llms"])
    llms["both"] = {"provider": "multi", "llms": ["a", "b"]}
    methods = [{"type": "random_search", "llm": "both"}]
    plan = load_plan(config(llms=llms, methods=methods), env={})
    assert len(plan.llms["both"].llms) == 2


def test_api_key_from_environment():
    llms = {"g": {"provider": "openai", "model": "gpt", "api_key": "${env:KEY}"}}
    methods = [{"type": "random_search", "llm": "g"}]
    plan = load_plan(config(llms=llms, methods=methods), env={"KEY": "sk-1"})
    assert plan.llms["g"].api_key == "sk-1"


def test_make_experiment(tmp_path):
    data = config()
    data["experiment"].update(
        output_dir=str(tmp_path / "out"), n_jobs=3, show_stdout=True
    )
    plan = load_plan(data, env={})
    exp = plan.make_experiment()
    assert isinstance(exp, Experiment)
    assert exp.methods is plan.methods and exp.problems is plan.problems
    assert list(exp.seeds) == [0, 1] and exp.budget == 7 and exp.n_jobs == 3
    assert exp.show_stdout is True
    assert Path(exp.exp_logger.dirname) == tmp_path / "out"


def test_experiment_from_yaml(tmp_path):
    import yaml

    data = config()
    data["experiment"]["output_dir"] = "out"
    path = tmp_path / "exp.yml"
    path.write_text(yaml.safe_dump(data))
    exp = Experiment.from_yaml(path, overrides=["experiment.runs=1"], env={})
    assert len(exp.seeds) == 1
    exp2 = load_experiment(path, env={})
    assert len(exp2.seeds) == 2


# ---------------------------------------------------------------- errors ---


def test_schema_errors_have_locations_and_suggestions(tmp_path):
    path = tmp_path / "exp.yml"
    path.write_text(textwrap.dedent("""
            experiment:
              output_dir: x
              runz: 3
            llms:
              a: {provider: dummy, modle: x}
            methods: []
            problems:
              - {type: circle_packing, suite: hexagon_packing}
            """))
    with pytest.raises(ConfigError) as exc:
        load_plan(path, env={})
    text = str(exc.value)
    assert (
        f"{path}:4:3: experiment.runz: Unknown field 'runz'. Did you mean 'runs'?"
        in text
    )
    assert "llms.a.modle: Unknown field 'modle'. Did you mean 'model'?" in text
    assert "methods" in text  # min_length=1
    assert "Give exactly one of 'type' or 'suite'" in text


def test_missing_sections():
    issues = issues_of({"experiment": {"output_dir": "x"}})
    joined = "\n".join(issues)
    for section in ("llms", "methods", "problems"):
        assert f"Missing required field '{section}'" in joined


def test_not_a_mapping():
    assert "must be a mapping" in issues_of(["a"])[0]


def test_component_errors_are_collected():
    data = config(
        llms={
            "a": {"provider": "dummy"},
            "g": {"provider": "gemini"},
            "x": {"provider": "olama"},
        },
        methods=[
            {
                "type": "llamea",
                "llm": ["a", "nope"],
                "params": {"n_parent": 1, "budget": 3, "minimization": True},
            },
            {"type": "llamae", "llm": "a"},
        ],
        problems=[
            {"type": "circle_packing", "params": {"tolerence": 1}},
            {"type": "tsp"},
            {"type": "bbob_sbox", "params": {"problem_type": "NOPE"}},
            {"suite": "hexagon_packing", "params": {}},
        ],
    )
    issues = "\n".join(issues_of(data))
    expected = [
        "llms.g: Missing required parameter 'api_key'",
        "llms.x.provider: Unknown llm 'olama'. Did you mean 'ollama'?",
        "methods[0].llm: Unknown LLM 'nope'. Defined in 'llms': a, g, x.",
        "methods[0].params.n_parent: Unknown parameter 'n_parent' for method "
        "'llamea'. Did you mean 'n_parents'?",
        "methods[0].params.budget: 'budget' is set by BLADE",
        "methods[0].params.minimization: 'minimization' is set by BLADE",
        "methods[1].type: Unknown method 'llamae'. Did you mean 'llamea'?",
        "problems[0].params.tolerence: Unknown parameter 'tolerence' for problem "
        "'circle_packing'. Did you mean 'tolerance'?",
        "problems[1]: Missing required parameter 'benchmark' for problem 'tsp'.",
        "problems[2].params.problem_type: 'NOPE' is not a valid ProblemClass",
        "problems[3]: Missing required parameter 'get_best' for suite",
    ]
    for line in expected:
        assert line in issues, f"{line!r} not in:\n{issues}"


def test_duplicate_names():
    data = config(
        methods=[
            {"type": "random_search", "name": "same", "llm": "a"},
            {"type": "llamea", "name": "same", "llm": "a"},
        ],
        problems=[
            {"type": "circle_packing", "name": "p"},
            {"type": "circle_packing", "name": "p"},
        ],
    )
    issues = "\n".join(issues_of(data))
    assert "Two methods are named 'same' with LLM 'a'" in issues
    assert "2 problems are named 'p'" in issues


def test_same_method_name_with_llm_aliases_of_one_model():
    llms = {
        "a": {"provider": "dummy", "model": "m"},
        "b": {"provider": "dummy", "model": "m"},
    }
    data = config(
        llms=llms, methods=[{"type": "random_search", "name": "rs", "llm": ["a", "b"]}]
    )
    issues = "\n".join(issues_of(data))
    assert "LLMs 'a' and 'b' (both model 'm')" in issues


def test_repeated_seeds_are_rejected():
    data = config()
    data["experiment"]["seeds"] = [1, 2, 1]
    assert "Seeds must be unique; repeated: [1]" in "\n".join(issues_of(data))
    data["experiment"]["seeds"] = []
    assert "'seeds' must not be empty" in "\n".join(issues_of(data))


def test_unavailable_logger_is_reported(monkeypatch):
    import importlib.util

    real = importlib.util.find_spec
    monkeypatch.setattr(
        importlib.util,
        "find_spec",
        lambda name, *a: None if name == "trackio" else real(name, *a),
    )
    data = config()
    data["experiment"]["logger"] = {"type": "trackio"}
    issues = "\n".join(issues_of(data))
    assert "experiment.logger.type: The logger 'trackio'" in issues
    assert "uv sync --group trackio" in issues


def test_same_method_name_with_different_llms_is_fine():
    data = config(methods=[{"type": "random_search", "name": "rs", "llm": ["a", "b"]}])
    assert len(load_plan(data, env={}).methods) == 2


def test_multi_llm_errors():
    llms = {
        "a": {"provider": "dummy"},
        "m": {"provider": "multi", "llms": ["a", "zzz"]},
        "m2": {"provider": "multi", "llms": ["m"]},
        "m3": {"provider": "multi"},
        "d": {"provider": "dummy", "llms": ["a"]},
    }
    issues = "\n".join(issues_of(config(llms=llms)))
    assert "llms.m.llms[1]: Unknown LLM 'zzz'" in issues
    assert "llms.m2.llms[0]: A 'multi' LLM cannot contain another one" in issues
    assert "llms.m3: Provider 'multi' needs 'llms'" in issues
    assert "llms.d.llms: 'llms' is only used with provider 'multi'" in issues


def test_unsupported_provided_argument():
    llms = {
        "a": {"provider": "multi", "llms": ["b"], "model": "x"},
        "b": {"provider": "dummy"},
    }
    issues = "\n".join(issues_of(config(llms=llms)))
    assert "llms.a.model: LLM provider 'multi' does not take 'model'" in issues


def test_constructor_failure_is_reported(monkeypatch):
    from iohblade.benchmarks.packing import circles

    def boom(self, *args, **kwargs):
        raise RuntimeError("cannot build")

    monkeypatch.setattr(circles.CirclePacking, "__init__", boom)
    issues = "\n".join(issues_of(config()))
    assert (
        "problems[0]: Creating problem 'circle_packing' failed: RuntimeError: cannot build"
        in issues
    )


def test_unknown_logger_and_logger_params():
    data = config()
    data["experiment"]["logger"] = {"type": "mlflow", "params": {"uri": "x"}}
    issues = "\n".join(issues_of(data))
    assert "Unknown parameter 'uri' for logger 'mlflow'" in issues
    data["experiment"]["logger"] = {"type": "wandb"}
    assert "Unknown logger 'wandb'" in "\n".join(issues_of(data))


def test_override_errors_point_at_set():
    issues = issues_of(config(), overrides=["experiment.runs=0"])
    assert issues[0].startswith("<--set>: experiment.runs:")


def test_json_schema():
    schema = json_schema()
    assert schema["title"] == "BLADE experiment configuration"
    props = schema["$defs"]["MethodSpec"]["properties"]
    assert "llamea" in props["type"]["examples"]
    assert ExperimentConfig.model_json_schema()["required"] == [
        "experiment",
        "llms",
        "methods",
        "problems",
    ]


@pytest.mark.parametrize(
    "path", sorted((Path(__file__).parents[2] / "examples" / "configs").glob("*.yml"))
)
def test_example_configs_are_valid(path):
    env = {"GOOGLE_API_KEY": "x", "GEMINI_API_KEY": "x", "OPENAI_API_KEY": "x"}
    plan = load_plan(path, env=env)
    assert plan.n_runs > 0


@pytest.mark.parametrize(
    "relative",
    ["examples/configs/experiment.schema.json", "docs/_extra/experiment.schema.json"],
)
def test_checked_in_schema_is_up_to_date(relative):
    import json

    path = Path(__file__).parents[2] / relative
    assert (
        json.loads(path.read_text()) == json_schema()
    ), f"Regenerate with: uv run iohblade schema -o {relative}"


def _llamea(params, llm="a"):
    return config(
        methods=[{"type": "llamea", "name": "L-{llm}", "llm": llm, "params": params}]
    )


def test_llamea_operators_become_operator_objects():
    from llamea import Operator

    params = {
        "n_parents": 2,
        "operators": [
            {"name": "refine", "prompt": "Refine it.", "weight": 0.7},
            {
                "name": "cx",
                "prompt": "Combine them.",
                "weight": 0.3,
                "number_of_parents": 2,
            },
            "A plain mutation prompt.",
        ],
    }
    plan = load_plan(_llamea(params, llm=["a", "b"]), env={})
    ops_a = plan.methods[0].kwargs["operators"]
    ops_b = plan.methods[1].kwargs["operators"]
    assert isinstance(ops_a[0], Operator)
    assert (
        ops_a[0].name,
        ops_a[0].prompt,
        ops_a[0].weight,
        ops_a[0].number_of_parents,
    ) == ("refine", "Refine it.", 0.7, 1)
    assert ops_a[1].number_of_parents == 2 and ops_a[1].weight == 0.3
    assert ops_a[2] == "A plain mutation prompt."  # llamea turns it into a mutation
    # Each method gets its own Operator objects (they keep per-run state).
    assert ops_a[0].id != ops_b[0].id


def test_llamea_operators_are_accepted_by_llamea():
    from llamea import LLaMEA as LLAMEA_Algorithm

    from iohblade.llm import Dummy_LLM

    params = {
        "n_parents": 2,
        "operators": [{"prompt": "Refine it.", "number_of_parents": 2}],
    }
    plan = load_plan(_llamea(params), env={})
    algo = LLAMEA_Algorithm(
        f=lambda *a: None, llm=Dummy_LLM(), log=False, **plan.methods[0].kwargs
    )
    assert [o.prompt for o in algo.operators] == ["Refine it."]


def test_llamea_operator_errors():
    params = {
        "n_parents": 1,
        "operators": [
            {"name": "a", "promt": "typo"},
            {"prompt": "x", "weight": -1},
            {"prompt": "x", "number_of_parents": 2},
            {"prompt": "x", "number_of_parents": 0},
            42,
        ],
    }
    issues = "\n".join(issues_of(_llamea(params)))
    p = "methods[0].params.operators"
    assert (
        f"{p}[0].promt: Unknown operator field 'promt'. Did you mean 'prompt'?"
        in issues
    )
    assert f"{p}[0]: An operator needs a 'prompt'." in issues
    assert f"{p}[1].weight: 'weight' must be a number >= 0" in issues
    assert (
        f"{p}[2].number_of_parents: An operator cannot use more parents (2) than "
        "n_parents (1)." in issues
    )
    assert (
        f"{p}[3].number_of_parents: 'number_of_parents' must be an integer >= 1"
        in issues
    )
    assert f"{p}[4]: An operator is a mapping" in issues


def test_llamea_operator_parents_default_n_parents():
    # llamea's default n_parents (5) applies when n_parents is not given.
    params = {"operators": [{"prompt": "x", "number_of_parents": 6}]}
    assert "than n_parents (5)" in "\n".join(issues_of(_llamea(params)))


def test_llamea_empty_operators():
    issues = "\n".join(issues_of(_llamea({"operators": []})))
    assert "'operators' is empty" in issues
