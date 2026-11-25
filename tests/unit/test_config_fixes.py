"""Regression tests for the YAML-configuration and CLI fixes of issue #208."""

import ast
import json
import shutil
import sys
import textwrap
import uuid
from pathlib import Path

import pytest
import yaml

import iohblade
from iohblade import cli
from iohblade.config import ConfigError, load_plan
from iohblade.config.record import is_secret_key, redact, resume_status
from iohblade.config.user_code import get_class, load_source
from iohblade.config.yaml_loader import load_yaml, read_dotenv
from iohblade.loggers.base import ExperimentLogger
from iohblade.problem import Problem

PACKAGE = Path(iohblade.__file__).parent


def unique(prefix):
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


@pytest.fixture(autouse=True)
def restore_sys_path():
    before = list(sys.path)
    modules = set(sys.modules)
    yield
    sys.path[:] = before
    for name in set(sys.modules) - modules:
        if name.startswith("fix208_"):
            del sys.modules[name]


def write_yaml(directory, data, name="exp.yml"):
    path = directory / name
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return path


def base_config(output="out", runs=2, methods=None):
    return {
        "experiment": {"output_dir": output, "runs": runs, "budget": 1},
        "llms": {"d": {"provider": "dummy", "model": "m"}},
        "methods": methods or [{"type": "random_search", "name": "rs", "llm": "d"}],
        "problems": [{"type": "circle_packing", "name": "c"}],
    }


def issues_of(source, **kwargs):
    with pytest.raises(ConfigError) as exc:
        load_plan(source, env={}, **kwargs)
    return "\n".join(str(i) for i in exc.value.issues)


# ---------------------------------------------- 1. resume compatibility ---


def _record_runs(out: Path, seeds, finished=(), budget=1):
    """Write a progress.json as the logger would after a run."""
    runs = [
        {
            "method_name": "rs",
            "llm_name": "m",
            "problem_name": "c",
            "seed": s,
            "budget": budget,
            "evaluations": 1 if s in finished else 0,
            "start_time": "t" if s in finished else None,
            "end_time": "t" if s in finished else None,
            "log_dir": None,
        }
        for s in seeds
    ]
    progress = {"id": "x", "start_time": "t", "end_time": None, "runs": runs}
    progress["current"] = len(finished)
    progress["total"] = len(runs)
    (out / "progress.json").write_text(json.dumps(progress), encoding="utf-8")


def test_run_with_fewer_runs_is_a_config_error_and_keeps_record(tmp_path, capsys):
    path = write_yaml(tmp_path, base_config(runs=2))
    load_plan(path, env={}).make_experiment()  # writes blade_config/
    out = tmp_path / "out"
    _record_runs(out, [0, 1], finished=[0, 1])
    record = (out / "blade_config" / "resolved.yml").read_text(encoding="utf-8")
    progress = (out / "progress.json").read_text(encoding="utf-8")

    code = cli.main(["run", str(path), "--set", "experiment.runs=1"])

    assert code == cli.EXIT_CONFIG
    err = capsys.readouterr().err
    assert "experiment.output_dir" in err
    assert "already holds 2 run(s)" in err and "plans 1" in err
    assert "rs (m) on c, seed 1" in err
    assert "Seeds recorded: [0, 1]; planned: [0]" in err
    assert "new output directory" in err and "restore the previous" in err
    # Nothing was archived or overwritten.
    assert not list(out.glob("blade_config-*"))
    assert (out / "blade_config" / "resolved.yml").read_text(encoding="utf-8") == record
    assert (out / "progress.json").read_text(encoding="utf-8") == progress


def test_make_experiment_rejects_renamed_method(tmp_path):
    path = write_yaml(tmp_path, base_config(runs=1))
    _out = tmp_path / "out"
    _out.mkdir()
    _record_runs(_out, [0])
    with pytest.raises(ConfigError) as exc:
        load_plan(path, env={}, overrides=["methods.0.name=other"]).make_experiment()
    text = str(exc.value)
    assert "Methods recorded: ['rs']; planned: ['other']" in text
    assert not (_out / "blade_config").exists()


def test_adding_runs_resumes_and_extends_progress(tmp_path):
    path = write_yaml(tmp_path, base_config(runs=2))
    out = tmp_path / "out"
    out.mkdir()
    _record_runs(out, [0, 1], finished=[0])

    plan = load_plan(path, env={}, overrides=["experiment.runs=3"])
    status = resume_status(plan)
    assert status.compatible and status.finished == 1
    assert status.added == {("rs", "m", "c", 2)}

    experiment = plan.make_experiment()
    assert any("1 run(s) new" in w for w in plan.warnings)
    data = json.loads((out / "progress.json").read_text(encoding="utf-8"))
    assert [r["seed"] for r in data["runs"]] == [0, 1, 2]
    assert data["runs"][0]["end_time"] == "t"  # finished run kept
    assert data["total"] == 3

    # The logger now accepts the plan (it raised ValueError before).
    logger = experiment.exp_logger
    logger.start_progress(
        experiment.id,
        3,
        methods=experiment.methods,
        problems=experiment.problems,
        seeds=experiment.seeds,
        budget=1,
    )
    method = experiment.methods[0]
    problem = experiment.problems[0]
    assert not logger.is_run_pending(method, "m", problem, 0)
    assert logger.is_run_pending(method, "m", problem, 2)


def test_changed_budget_is_a_config_error(tmp_path, capsys):
    """Resuming used to mix budgets: finished runs kept the old one."""
    path = write_yaml(tmp_path, base_config(runs=2))
    out = tmp_path / "out"
    out.mkdir()
    _record_runs(out, [0, 1], finished=[0])  # recorded with budget 1
    progress = (out / "progress.json").read_text(encoding="utf-8")

    plan = load_plan(path, env={}, overrides=["experiment.budget=5"])
    status = resume_status(plan)
    assert not status.compatible
    # Only the started run counts; seed 1 never ran with budget 1.
    assert status.budget_changed == {("rs", "m", "c", 0): (1, 5)}
    assert "Cannot resume" in status.summary()[0]

    code = cli.main(["run", str(path), "--set", "experiment.budget=5"])
    assert code == cli.EXIT_CONFIG
    err = capsys.readouterr().err
    assert "Budget changed for 1 started run(s)" in err
    assert "rs (m) on c, seed 0: 1 recorded, 5 planned" in err
    assert "restore the previous budget" in err
    assert (out / "progress.json").read_text(encoding="utf-8") == progress


def test_same_budget_still_resumes(tmp_path):
    path = write_yaml(tmp_path, base_config(runs=2))
    out = tmp_path / "out"
    out.mkdir()
    _record_runs(out, [0, 1], finished=[0])
    status = resume_status(load_plan(path, env={}))
    assert status.compatible and not status.budget_changed


def test_budget_change_is_listed_with_removed_runs(tmp_path):
    path = write_yaml(tmp_path, base_config(runs=2))
    out = tmp_path / "out"
    out.mkdir()
    _record_runs(out, [0, 1], finished=[0, 1])
    plan = load_plan(
        path, env={}, overrides=["experiment.runs=1", "experiment.budget=5"]
    )
    text = resume_status(plan).mismatch_message()
    assert "Recorded but no longer planned" in text
    assert "Budget changed for 1 started run(s)" in text


def test_method_budget_is_the_run_budget(tmp_path):
    """A method's own budget is what runs, is recorded and is compared."""
    methods = [{"type": "random_search", "name": "rs", "llm": "d", "budget": 3}]
    path = write_yaml(tmp_path, base_config(runs=2, methods=methods))
    out = tmp_path / "out"
    out.mkdir()
    _record_runs(out, [0, 1], finished=[0], budget=3)
    status = resume_status(load_plan(path, env={}))
    assert status.compatible and not status.budget_changed

    _record_runs(out, [0, 1], finished=[0], budget=1)  # the experiment budget
    status = resume_status(load_plan(path, env={}))
    assert status.budget_changed == {("rs", "m", "c", 0): (1, 3)}


def test_logger_records_the_budget_a_run_starts_with(tmp_path):
    from types import SimpleNamespace

    logger = ExperimentLogger(str(tmp_path / "exp"))
    method = SimpleNamespace(name="rs", llm=SimpleNamespace(model="m"), budget=3)
    problem = SimpleNamespace(name="c", set_logger=lambda _logger: None)
    logger.start_progress(
        "x", 1, methods=[method], problems=[problem], seeds=[0], budget=1
    )
    assert logger.progress["runs"][0]["budget"] == 3  # the method's budget

    method.budget = 5  # the configuration changed before this run started
    run = logger.open_run(method, problem, budget=5, seed=0)
    assert logger.progress["runs"][0]["budget"] == 5
    assert run.budget == 5


def test_unreadable_progress_is_a_config_error(tmp_path, capsys):
    path = write_yaml(tmp_path, base_config(runs=1))
    (tmp_path / "out").mkdir()
    (tmp_path / "out" / "progress.json").write_text("{broken", encoding="utf-8")
    assert cli.main(["run", str(path)]) == cli.EXIT_CONFIG
    assert "Could not read the experiment progress file" in capsys.readouterr().err


# --------------------------------------------- 3. **kwargs of methods ---


@pytest.mark.parametrize(
    "method,params,expected",
    [
        ("random_search", {"pop_size": 3}, "Unknown parameter 'pop_size'"),
        ("eoh", {"pop_sise": 4}, "Did you mean 'pop_size'?"),
        ("eoh", {"timeot": 4}, "Did you mean 'timeout'?"),
        ("reevo", {"mutation_rte": 0.3}, "Did you mean 'mutation_rate'?"),
        ("reevo", {"init_popsize": 3}, "Did you mean 'init_pop_size'"),
    ],
)
def test_unknown_kwargs_of_methods_are_reported(tmp_path, method, params, expected):
    data = base_config(methods=[{"type": method, "llm": "d", "params": params}])
    issues = issues_of(write_yaml(tmp_path, data))
    assert expected in issues


@pytest.mark.parametrize(
    "method,params",
    [
        ("eoh", {"pop_size": 4, "timeout": 30, "output_path": "x"}),
        (
            "reevo",
            {
                "pop_size": 4,
                "init_pop_size": 6,
                "mutation_rate": 0.3,
                "timeout": 5,
                "output_path": "x",
            },
        ),
    ],
)
def test_known_kwargs_of_methods_are_accepted(tmp_path, method, params):
    data = base_config(methods=[{"type": method, "llm": "d", "params": params}])
    plan = load_plan(write_yaml(tmp_path, data), env={})
    assert plan.methods[0].kwargs == params


# ----------------------------------------------- 4. typed ${env:...} ---


def test_env_placeholder_gets_a_type_unless_quoted(tmp_path):
    (tmp_path / "c.yml").write_text(
        textwrap.dedent("""
            n: ${env:NP:-1}
            quoted: "${env:NP:-1}"
            single: '${env:NP:-1}'
            inside: abc-${env:NP:-1}
            flag: ${env:FLAG}
            items: ${env:ITEMS}
            ratio: ${env:RATIO}
            text: ${env:TEXT}
            empty: ${env:EMPTY:-}
            api_key: ${env:KEY}
            auth_token: ${env:KEY}
            list: ["${env:NP:-1}", 2]
            """),
        encoding="utf-8",
    )
    env = {
        "FLAG": "false",
        "ITEMS": "[1, 2]",
        "RATIO": "0.5",
        "TEXT": "hello world",
        "KEY": "12345",
    }
    data = load_yaml(tmp_path / "c.yml", env=env).data
    assert data["n"] == 1
    assert data["quoted"] == "1" and data["single"] == "1"
    assert data["inside"] == "abc-1"
    assert data["flag"] is False
    assert data["items"] == [1, 2]
    assert data["ratio"] == 0.5
    assert data["text"] == "hello world"
    assert data["empty"] == ""
    assert data["api_key"] == "12345" and data["auth_token"] == "12345"
    assert data["list"] == ["1", 2]


def test_typed_env_value_reaches_method_params(tmp_path):
    data = base_config(runs=1)
    data["methods"] = [
        {"type": "llamea", "llm": "d", "params": {"n_parents": "__NP__"}}
    ]
    text = yaml.safe_dump(data, sort_keys=False)
    assert "n_parents: __NP__" in text
    text = text.replace("__NP__", "${env:NP:-2}")
    (tmp_path / "exp.yml").write_text(text, encoding="utf-8")
    plan = load_plan(tmp_path / "exp.yml", env={})
    assert plan.methods[0].kwargs["n_parents"] == 2


def test_dict_sources_are_not_typed():
    cfg = load_yaml({"n": "${env:NP}"}, env={"NP": "3"})
    assert cfg.data["n"] == "3"


# ------------------------------------------------------------ 5. .env ---


def test_dotenv_follows_python_dotenv_rules(tmp_path):
    env = tmp_path / ".env"
    env.write_text(
        textwrap.dedent(r"""
            # a comment line

            A=abc # my key
            B="xyz" # c
            C='single # kept'  # comment
            D="two\nlines \"quoted\""
            E='raw\n'
            export F=exported
            G =  spaced value
            H=no#comment
            I=
            J="has # hash"
            """),
        encoding="utf-8",
    )
    values = read_dotenv(env)
    assert values["A"] == "abc"
    assert values["B"] == "xyz"
    assert values["C"] == "single # kept"
    assert values["D"] == 'two\nlines "quoted"'
    assert values["E"] == "raw\\n"
    assert values["F"] == "exported"
    assert values["G"] == "spaced value"
    assert values["H"] == "no#comment"
    assert values["I"] == ""
    assert values["J"] == "has # hash"
    assert len(values) == 10


def test_dotenv_values_are_used_by_load_yaml(tmp_path, monkeypatch):
    monkeypatch.delenv("FIX208_KEY", raising=False)
    (tmp_path / ".env").write_text("FIX208_KEY=abc # my key\n", encoding="utf-8")
    (tmp_path / "c.yml").write_text('k: "${env:FIX208_KEY}"\n', encoding="utf-8")
    assert load_yaml(tmp_path / "c.yml").data["k"] == "abc"


# -------------------------------------------- 6. reload of custom code ---

PROBLEM = """
from iohblade.problem import Problem


class P(Problem):
    VALUE = {value}

    def __init__(self, name="P"):
        super().__init__(name=name)

    def evaluate(self, solution):
        return solution

    def test(self, solution):
        return solution

    def to_dict(self):
        return {{}}

    def get_config(self):
        return {{}}
"""


def test_edited_custom_file_is_reloaded(tmp_path):
    mod = unique("fix208_reload")
    source = tmp_path / f"{mod}.py"
    source.write_text(PROBLEM.format(value=1), encoding="utf-8")
    data = base_config(runs=1)
    data["problems"] = [{"type": "custom", "source": f"{mod}.py", "class": "P"}]
    path = write_yaml(tmp_path, data)

    first = load_plan(path, env={})
    assert type(first.problems[0]).VALUE == 1

    # Same size, so stale bytecode would pass Python's own check.
    source.write_text(PROBLEM.format(value=2), encoding="utf-8")
    second = load_plan(path, env={})
    assert type(second.problems[0]).VALUE == 2
    assert second.sources[0].sha256 != first.sources[0].sha256

    # Unchanged: the module is reused, not imported again.
    third = load_plan(path, env={})
    assert third.sources[0].module is second.sources[0].module


def test_edited_package_submodule_is_reloaded(tmp_path):
    pkg = unique("fix208_pkg")
    folder = tmp_path / pkg
    folder.mkdir()
    (folder / "__init__.py").write_text("from . import sub\n", encoding="utf-8")
    (folder / "sub.py").write_text(PROBLEM.format(value=1), encoding="utf-8")

    loaded = load_source(pkg, tmp_path)
    assert get_class(loaded, "sub:P", Problem, "problem").VALUE == 1

    (folder / "sub.py").write_text(PROBLEM.format(value=3), encoding="utf-8")
    loaded = load_source(pkg, tmp_path)
    assert get_class(loaded, "sub:P", Problem, "problem").VALUE == 3
    assert loaded.module.sub.P.VALUE == 3


# ------------------------------------------------------ 7. encodings ---

_TEXT_CALLS = {"read_text", "write_text", "open"}


def _calls_without_encoding(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    missing = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
        if name not in _TEXT_CALLS:
            continue
        if name == "open" and isinstance(func, ast.Attribute):
            continue  # e.g. jsonlines.open / Path.open are not used here
        mode = node.args[1] if len(node.args) > 1 else None
        if (
            isinstance(mode, ast.Constant)
            and isinstance(mode.value, str)
            and "b" in mode.value
        ):
            continue
        if not any(k.arg == "encoding" for k in node.keywords):
            missing.append(f"{path.name}:{node.lineno} {name}()")
    return missing


def test_text_files_are_read_and_written_as_utf8():
    files = sorted((PACKAGE / "config").glob("*.py")) + [PACKAGE / "cli.py"]
    missing = [m for f in files for m in _calls_without_encoding(f)]
    assert missing == []


def test_non_ascii_configuration_is_read_as_utf8(tmp_path):
    data = base_config(runs=1)
    data["experiment"]["name"] = "Größe – δοκιμή"
    path = write_yaml(tmp_path, data)
    # Also when Python's own default would be another encoding.
    plan = load_plan(path, env={})
    assert plan.config.experiment.name == "Größe – δοκιμή"
    plan.make_experiment()
    text = (tmp_path / "out" / "blade_config" / "resolved.yml").read_text(
        encoding="utf-8"
    )
    assert "Größe – δοκιμή" in text


def test_non_utf8_configuration_is_a_config_error(tmp_path, capsys):
    path = tmp_path / "exp.yml"
    path.write_bytes("experiment:\n  name: Gr\xf6\xdfe\n".encode("latin-1"))
    with pytest.raises(ConfigError) as exc:
        load_yaml(path, env={})
    assert "not valid UTF-8" in str(exc.value)
    assert cli.main(["validate", str(path)]) == cli.EXIT_CONFIG
    assert "not valid UTF-8" in capsys.readouterr().err


# ------------------------------------------------------ 8. redaction ---


@pytest.mark.parametrize(
    "key",
    [
        "api_key",
        "apikey",
        "API_KEY",
        "apiKey",
        "openai_api_key",
        "token",
        "access_token",
        "authToken",
        "secret",
        "client_secret",
        "password",
        "db_password",
        "secret_key",
    ],
)
def test_secret_keys_are_redacted(key):
    assert is_secret_key(key)
    assert redact({key: "v"}, set()) == {key: "<redacted>"}


@pytest.mark.parametrize(
    "key",
    [
        "max_tokens",
        "tokenizer",
        "token_limit",
        "num_tokens",
        "maxTokens",
        "secretary",
        "passwords_allowed",
        "keyword",
        "budget",
    ],
)
def test_harmless_keys_are_kept(key):
    assert not is_secret_key(key)
    assert redact({key: 5}, set()) == {key: 5}


def test_resolved_yml_keeps_max_tokens(tmp_path):
    data = base_config(runs=1)
    data["llms"]["d"]["params"] = {}
    path = write_yaml(tmp_path, data)
    plan = load_plan(path, env={})
    plan.source.data["llms"]["d"]["max_tokens"] = 100
    plan.make_experiment()
    text = (tmp_path / "out" / "blade_config" / "resolved.yml").read_text(
        encoding="utf-8"
    )
    assert "max_tokens: 100" in text


# ---------------------------------------------------- 9a. --set + env ---


def test_overrides_are_interpolated(tmp_path):
    path = write_yaml(tmp_path, base_config(runs=1))
    env = {"R": "3", "OUT": "elsewhere"}
    cfg = load_yaml(
        path,
        env=env,
        overrides=[
            "experiment.runs=${env:R}",
            "experiment.output_dir=results/${env:OUT}",
            'experiment.name="${env:R}"',
        ],
    )
    assert cfg.data["experiment"]["runs"] == 3
    assert cfg.data["experiment"]["output_dir"] == "results/elsewhere"
    assert cfg.data["experiment"]["name"] == "3"
    assert ("experiment", "runs") in cfg.secrets

    with pytest.raises(ConfigError) as exc:
        load_yaml(path, env={}, overrides=["experiment.runs=${env:MISSING_R}"])
    assert "MISSING_R" in str(exc.value) and "<--set>" in str(exc.value)


def test_cli_set_with_env(tmp_path, capsys, monkeypatch):
    monkeypatch.setenv("FIX208_RUNS", "3")
    path = write_yaml(tmp_path, base_config(runs=1))
    code = cli.main(
        ["validate", str(path), "--set", "experiment.runs=${env:FIX208_RUNS}"]
    )
    assert code == cli.EXIT_OK
    assert "x 3 seeds = 3" in capsys.readouterr().out


# ------------------------------------------------- 9b/9c. documentation ---


def test_load_yaml_docstring_does_not_promise_yaml_text():
    assert "YAML text or" not in load_yaml.__doc__


def test_skill_reference_says_module_clash_is_an_error():
    text = (PACKAGE / "agent_skill" / "reference.md").read_text(encoding="utf-8")
    assert "(`validate` warns)" not in text
    assert "reports that as an error" in text


# ------------------------------------------- 9d. relative sources.json ---


def test_sources_json_is_relative_and_survives_moving(tmp_path):
    mod = unique("fix208_moved")
    first = tmp_path / "a"
    first.mkdir()
    (first / f"{mod}.py").write_text(PROBLEM.format(value=1), encoding="utf-8")
    data = base_config(runs=1)
    data["problems"] = [{"type": "custom", "source": f"{mod}.py", "class": "P"}]
    write_yaml(first, data)
    load_plan(first / "exp.yml", env={}).make_experiment()
    hashes = json.loads(
        (first / "out" / "blade_config" / "sources.json").read_text(encoding="utf-8")
    )
    assert hashes[mod]["source"] == f"{mod}.py"

    moved = tmp_path / "b"
    shutil.move(str(first), str(moved))
    sys.modules.pop(mod, None)  # another folder now holds the module
    plan = load_plan(moved / "exp.yml", env={})
    plan.make_experiment()
    assert plan.warnings == []

    # A record written by an older version (absolute path) is still read,
    # without a false "custom code changed" warning.
    sources_json = moved / "out" / "blade_config" / "sources.json"
    old = json.loads(sources_json.read_text(encoding="utf-8"))
    old[mod]["source"] = str(tmp_path / "somewhere" / f"{mod}.py")
    sources_json.write_text(json.dumps(old), encoding="utf-8")
    plan = load_plan(moved / "exp.yml", env={})
    plan.make_experiment()
    assert plan.warnings == []

    # A real change is still reported.
    old[mod]["sha256"] = "0" * 64
    sources_json.write_text(json.dumps(old), encoding="utf-8")
    plan = load_plan(moved / "exp.yml", env={})
    plan.make_experiment()
    assert any("custom code changed" in w for w in plan.warnings)


# ------------------------------------------- review feedback on #218 ---


def test_non_utf8_dotenv_is_a_config_error(tmp_path, capsys):
    path = write_yaml(tmp_path, base_config())
    (tmp_path / ".env").write_bytes("KEY=Gr\xf6\xdfe\n".encode("latin-1"))
    with pytest.raises(ConfigError) as exc:
        load_yaml(path)
    assert ".env file" in str(exc.value) and "UTF-8" in str(exc.value)
    assert cli.main(["validate", str(path)]) == cli.EXIT_CONFIG


def test_typed_env_values_are_redacted():
    data = {"experiment": {"runs": 3}, "methods": [{"params": {"seed": [1, 2]}}]}
    secrets = {("experiment", "runs"), ("methods", 0, "params", "seed")}
    out = redact(data, secrets)
    assert out["experiment"]["runs"] == "<redacted>"
    assert out["methods"][0]["params"]["seed"] == "<redacted>"
    assert data["experiment"]["runs"] == 3  # the input is not changed


def test_typed_env_value_is_redacted_in_resolved_yml(tmp_path):
    config = base_config(runs="${env:RUNS}")
    path = tmp_path / "exp.yml"
    text = yaml.safe_dump(config, sort_keys=False).replace(
        "'${env:RUNS}'", "${env:RUNS}"
    )
    path.write_text(text, encoding="utf-8")
    plan = load_plan(path, env={"RUNS": "2"})
    assert plan.config.experiment.runs == 2
    plan.make_experiment()
    resolved = (tmp_path / "out" / "blade_config" / "resolved.yml").read_text(
        encoding="utf-8"
    )
    assert "runs: <redacted>" in resolved


def test_same_method_name_with_a_new_llm_is_added(tmp_path):
    # One method name used with two LLMs is allowed; adding the second LLM
    # to an experiment recorded with the first must add its runs.
    config = base_config(
        runs=2, methods=[{"type": "random_search", "name": "rs", "llm": ["d", "e"]}]
    )
    config["llms"]["e"] = {"provider": "dummy", "model": "m2"}
    path = write_yaml(tmp_path, config)
    out = tmp_path / "out"
    out.mkdir()
    _record_runs(out, [0, 1], finished=[0, 1])  # recorded with LLM "m" only

    plan = load_plan(path, env={})
    status = resume_status(plan)
    assert status.compatible
    assert status.finished == 2
    assert status.added == {("rs", "m2", "c", 0), ("rs", "m2", "c", 1)}

    plan.make_experiment()
    data = json.loads((out / "progress.json").read_text(encoding="utf-8"))
    assert sorted((r["llm_name"], r["seed"]) for r in data["runs"]) == [
        ("m", 0),
        ("m", 1),
        ("m2", 0),
        ("m2", 1),
    ]
    assert data["total"] == 4
