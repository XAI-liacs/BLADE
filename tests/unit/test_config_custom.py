"""Custom problems, methods and LLMs loaded from user files (Phase 2)."""

import json
import sys
import textwrap
import uuid
from pathlib import Path

import pytest
import yaml

from iohblade.config import ConfigError, load_plan
from iohblade.config.user_code import (
    UserCodeError,
    get_class,
    hash_source,
    load_source,
    read_requirements,
)
from iohblade.experiment import Experiment
from iohblade.llm import LLM
from iohblade.method import Method
from iohblade.problem import Problem

PROBLEM_CODE = """
from typing import Any

from iohblade.problem import Problem


class {cls}(Problem):
    def __init__(self, size=3, name="{cls}"):
        super().__init__(name=name)
        self.size = size

    def evaluate(self, solution):
        solution.set_scores(float(self.size))
        return solution

    def test(self, solution):
        return solution

    def to_dict(self):
        return {{"size": self.size}}

    def get_config(self) -> dict[str, Any]:
        return {{"name": self.name, "size": self.size}}
"""

METHOD_CODE = """
from iohblade.method import Method
from iohblade.solution import Solution


class {cls}(Method):
    def __init__(self, llm, budget, name="{cls}", step=1):
        super().__init__(llm, budget, name)
        self.step = step

    def __call__(self, problem):
        return problem(Solution(code="x = 1", name="s"))

    def to_dict(self):
        return {{"step": self.step}}

    def get_config(self):
        return {{"name": self.name, "step": self.step}}
"""

LLM_CODE = """
from iohblade.llm import LLM


class {cls}(LLM):
    def __init__(self, model="echo", reply="ok"):
        super().__init__("", model)
        self.reply = reply

    def _query(self, session, **kwargs):
        return self.reply

    def get_config(self):
        return [{{"model": self.model}}]
"""


def unique(prefix):
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


@pytest.fixture(autouse=True)
def restore_sys_path():
    before = list(sys.path)
    yield
    sys.path[:] = before


def write_file(directory, module, code, cls):
    path = directory / f"{module}.py"
    path.write_text(textwrap.dedent(code.format(cls=cls)))
    return path


def base_config(problems, methods=None, llms=None, output="out"):
    return {
        "experiment": {"output_dir": output, "runs": 1, "budget": 2},
        "llms": llms or {"d": {"provider": "dummy"}},
        "methods": methods or [{"type": "random_search", "llm": "d"}],
        "problems": problems,
    }


def write_config(directory, data):
    path = directory / "exp.yml"
    path.write_text(yaml.safe_dump(data, sort_keys=False))
    return path


def issues_of(path):
    with pytest.raises(ConfigError) as exc:
        load_plan(path, env={})
    return "\n".join(str(i) for i in exc.value.issues)


# ------------------------------------------------------------- loading ---


def test_custom_problem_method_and_llm_from_files(tmp_path):
    pmod, mmod, lmod = unique("prob"), unique("meth"), unique("llm")
    write_file(tmp_path, pmod, PROBLEM_CODE, "MyProblem")
    write_file(tmp_path, mmod, METHOD_CODE, "MyMethod")
    write_file(tmp_path, lmod, LLM_CODE, "EchoLLM")
    (tmp_path / "requirements.txt").write_text(
        "# deps\nscipy>=1.10  # comment\n\n-r other.txt\n"
    )
    data = base_config(
        problems=[
            {
                "type": "custom",
                "source": f"{pmod}.py",
                "class": "MyProblem",
                "name": "mine",
                "params": {"size": 7},
                "dependencies": ["networkx"],
            }
        ],
        methods=[
            {
                "type": "custom",
                "source": f"./{mmod}.py",
                "class": "MyMethod",
                "llm": "echo",
                "params": {"step": 2},
            }
        ],
        llms={
            "echo": {
                "provider": "custom",
                "source": f"{lmod}.py",
                "class": "EchoLLM",
                "params": {"reply": "hi"},
            }
        },
    )
    plan = load_plan(write_config(tmp_path, data), env={})

    problem, method, llm = plan.problems[0], plan.methods[0], plan.llms["echo"]
    assert type(problem).__module__ == pmod  # imported by its real name
    assert problem.size == 7 and problem.name == "mine"
    assert str(tmp_path.resolve()) in problem.extra_pythonpath
    assert "scipy>=1.10" in problem.dependencies
    assert "networkx" in problem.dependencies
    assert method.step == 2 and method.llm is llm and method.budget == 2
    assert llm.reply == "hi"
    assert problem._blade_source["class"] == "MyProblem"
    assert problem._blade_source["sha256"] == hash_source(tmp_path / f"{pmod}.py")
    assert {s.module_name for s in plan.sources} == {pmod, mmod, lmod}
    assert "Custom code (3)" in plan.summary()


def test_package_with_relative_import_data_file_and_submodule(tmp_path):
    pkg = unique("pkg")
    root = tmp_path / pkg
    (root / "data").mkdir(parents=True)
    (root / "data" / "size.txt").write_text("11")
    (root / "__init__.py").write_text("")
    (root / "helpers.py").write_text(
        "from pathlib import Path\n\n"
        "def read_size():\n"
        "    return int((Path(__file__).parent / 'data' / 'size.txt').read_text())\n"
    )
    (root / "problems.py").write_text(
        textwrap.dedent(PROBLEM_CODE.format(cls="PkgProblem")).replace(
            "self.size = size", "self.size = size if size else read_size()"
        )
        + "\nfrom .helpers import read_size\n"
    )
    (root / "pyproject.toml").write_text(
        '[project]\nname = "x"\ndependencies = ["sympy>=1.12"]\n'
    )
    data = base_config(
        problems=[
            {
                "type": "custom",
                "source": pkg,
                "class": "problems:PkgProblem",
                "params": {"size": 0},
            }
        ]
    )
    plan = load_plan(write_config(tmp_path, data), env={})
    problem = plan.problems[0]
    assert problem.size == 11
    assert "sympy>=1.12" in problem.dependencies
    assert plan.sources[0].module_name == pkg


def test_loading_twice_reuses_the_module(tmp_path):
    mod = unique("prob")
    write_file(tmp_path, mod, PROBLEM_CODE, "P")
    first = load_source(f"{mod}.py", tmp_path)
    second = load_source(tmp_path / f"{mod}.py", tmp_path)
    assert first.module is second.module
    assert sys.path.count(str(tmp_path.resolve())) == 1


# -------------------------------------------------------------- errors ---


@pytest.mark.parametrize(
    "setup,source,cls,expected",
    [
        (None, "missing.py", "X", "does not exist"),
        ("notes.txt", "notes.txt", "X", "is not a .py file"),
        ("folder/", "folder", "X", "folder without __init__.py"),
        ("json.py", "json.py", "X", "module name 'json'"),
        ("bad-name.py", "bad-name.py", "X", "not a valid Python module name"),
    ],
)
def test_source_errors(tmp_path, setup, source, cls, expected):
    if setup and setup.endswith("/"):
        (tmp_path / setup).mkdir()
    elif setup:
        (tmp_path / setup).write_text("x = 1\n")
    data = base_config(problems=[{"type": "custom", "source": source, "class": cls}])
    issues = issues_of(write_config(tmp_path, data))
    assert "problems[0].source:" in issues and expected in issues


def test_missing_third_party_import_has_install_hint(tmp_path):
    mod = unique("prob")
    (tmp_path / f"{mod}.py").write_text("import not_installed_pkg_xyz.sub\n")
    data = base_config(
        problems=[{"type": "custom", "source": f"{mod}.py", "class": "P"}]
    )
    issues = issues_of(write_config(tmp_path, data))
    assert "imports 'not_installed_pkg_xyz', which is not installed" in issues
    assert "uv add not_installed_pkg_xyz" in issues


@pytest.mark.parametrize("where", ["yaml", "requirements"])
def test_evaluation_dependency_imported_at_load_time(tmp_path, where):
    mod = unique("prob")
    (tmp_path / f"{mod}.py").write_text("import not_installed_eval_dep\n")
    problem = {"type": "custom", "source": f"{mod}.py", "class": "P"}
    if where == "yaml":
        problem["dependencies"] = ["Not_Installed.Eval-Dep>=1.0"]
    else:
        (tmp_path / "requirements.txt").write_text("not-installed-eval-dep\n")
    issues = issues_of(write_config(tmp_path, base_config(problems=[problem])))
    assert "only listed as a dependency of the evaluation environment" in issues
    assert "move the import into the method that needs it" in issues


def test_single_file_requirements_are_hashed_and_copied(tmp_path):
    mod = unique("prob")
    write_file(tmp_path, mod, PROBLEM_CODE, "P")
    loaded = load_source(f"{mod}.py", tmp_path)
    (tmp_path / "requirements.txt").write_text("scipy\n")
    with_req = hash_source(tmp_path / f"{mod}.py")
    assert with_req != loaded.sha256
    (tmp_path / "requirements.txt").write_text("scipy>=1.12\n")
    assert hash_source(tmp_path / f"{mod}.py") != with_req
    copy = load_source(f"{mod}.py", tmp_path).snapshot(tmp_path / "snap")
    assert sorted(p.name for p in copy.iterdir()) == [f"{mod}.py", "requirements.txt"]


def test_syntax_error_in_source(tmp_path):
    mod = unique("prob")
    (tmp_path / f"{mod}.py").write_text("def broken(:\n")
    data = base_config(
        problems=[{"type": "custom", "source": f"{mod}.py", "class": "P"}]
    )
    issues = issues_of(write_config(tmp_path, data))
    assert "Syntax error in" in issues and "line 1" in issues


def test_class_errors(tmp_path):
    mod = unique("prob")
    write_file(tmp_path, mod, PROBLEM_CODE, "RealProblem")
    with open(tmp_path / f"{mod}.py", "a") as f:
        f.write(
            "\n\nclass NotAProblem:\n    pass\n\n"
            "class Incomplete(Problem):\n"
            "    def evaluate(self, solution):\n        return solution\n"
        )
    data = base_config(
        problems=[
            {"type": "custom", "source": f"{mod}.py", "class": "Missing"},
            {"type": "custom", "source": f"{mod}.py", "class": "NotAProblem"},
            {"type": "custom", "source": f"{mod}.py", "class": "Incomplete"},
            {"type": "custom", "source": f"{mod}.py", "class": "nosuch:X"},
        ]
    )
    issues = issues_of(write_config(tmp_path, data))
    assert "problems[0].class: No class 'Missing'" in issues
    assert "Classes there: Incomplete, RealProblem." in issues
    assert (
        "problems[1].class: 'NotAProblem' must be a subclass of iohblade.problem.Problem"
        in issues
    )
    assert (
        "problems[2].class: 'Incomplete' does not implement the abstract method(s) "
        "get_config, test, to_dict of Problem." in issues
    )
    assert "problems[3].class: Submodule 'nosuch' not found" in issues


def test_custom_method_must_be_a_method(tmp_path):
    mod = unique("prob")
    write_file(tmp_path, mod, PROBLEM_CODE, "P")
    data = base_config(
        problems=[{"type": "circle_packing"}],
        methods=[{"type": "custom", "source": f"{mod}.py", "class": "P", "llm": "d"}],
    )
    issues = issues_of(write_config(tmp_path, data))
    assert (
        "methods[0].class: 'P' must be a subclass of iohblade.method.Method" in issues
    )


def test_schema_rules_for_custom(tmp_path):
    data = base_config(
        problems=[
            {"type": "custom", "source": "x.py"},
            {"type": "circle_packing", "class": "X"},
        ],
        llms={"d": {"provider": "custom", "clas": "x"}},
    )
    issues = issues_of(write_config(tmp_path, data))
    assert "problems[0]: type 'custom' needs 'class'." in issues
    assert (
        "problems[1]: 'source' and 'class' are only used with type: custom." in issues
    )
    assert "llms.d.clas: Unknown field 'clas'. Did you mean 'class'?" in issues


def test_custom_params_are_checked(tmp_path):
    mod = unique("prob")
    write_file(tmp_path, mod, PROBLEM_CODE, "P")
    data = base_config(
        problems=[
            {
                "type": "custom",
                "source": f"{mod}.py",
                "class": "P",
                "params": {"sise": 3},
            }
        ]
    )
    issues = issues_of(write_config(tmp_path, data))
    assert (
        "Unknown parameter 'sise' for custom problem 'P'. Did you mean 'size'?"
        in issues
    )


def test_get_class_and_requirements_helpers(tmp_path):
    mod = unique("llm")
    write_file(tmp_path, mod, LLM_CODE, "EchoLLM")
    loaded = load_source(f"{mod}.py", tmp_path)
    assert get_class(loaded, "EchoLLM", LLM, "llm").__name__ == "EchoLLM"
    with pytest.raises(UserCodeError):
        get_class(loaded, "EchoLLM", Method, "method")
    assert read_requirements(tmp_path / f"{mod}.py") == []
    (tmp_path / "pyproject.toml").write_text("not = [valid")
    pkg = tmp_path / "p"
    pkg.mkdir()
    (pkg / "pyproject.toml").write_text("not = [valid")
    with pytest.raises(UserCodeError, match="Cannot parse"):
        read_requirements(pkg)


def test_hash_ignores_caches_and_changes_with_content(tmp_path):
    pkg = tmp_path / "pkg"
    (pkg / "__pycache__").mkdir(parents=True)
    (pkg / "__init__.py").write_text("x = 1\n")
    first = hash_source(pkg)
    (pkg / "__pycache__" / "a.pyc").write_bytes(b"junk")
    assert hash_source(pkg) == first
    (pkg / "__init__.py").write_text("x = 2\n")
    assert hash_source(pkg) != first


# ------------------------------------------------------ reproducibility ---


def test_record_is_written_and_secrets_redacted(tmp_path, capsys):
    mod = unique("prob")
    write_file(tmp_path, mod, PROBLEM_CODE, "P")
    data = base_config(
        problems=[{"type": "custom", "source": f"{mod}.py", "class": "P"}],
        llms={
            "d": {"provider": "dummy"},
            "o": {"provider": "openai", "model": "m", "api_key": "${env:KEY}"},
            "inline": {"provider": "openai", "model": "m", "api_key": "sk-inline"},
        },
        methods=[
            {"type": "random_search", "llm": ["d", "o", "inline"], "name": "rs-{llm}"}
        ],
    )
    config_path = write_config(tmp_path, data)
    plan = load_plan(config_path, env={"KEY": "sk-from-env"})
    plan.make_experiment()

    record = tmp_path / "out" / "blade_config"
    text = (record / "resolved.yml").read_text()
    assert "sk-from-env" not in text and "sk-inline" not in text
    assert text.count("<redacted>") == 2
    assert f"# from: {config_path}" in text
    hashes = json.loads((record / "sources.json").read_text())
    assert hashes[mod]["sha256"] == plan.sources[0].sha256
    assert (record / "sources" / mod / f"{mod}.py").read_text() == (
        tmp_path / f"{mod}.py"
    ).read_text()
    assert plan.warnings == []

    # Resuming with the same configuration and code: nothing archived.
    plan = load_plan(config_path, env={"KEY": "sk-from-env"})
    plan.make_experiment()
    assert plan.warnings == []
    assert not list((tmp_path / "out").glob("blade_config-*"))

    # Changed code: the old record is archived with a warning.
    with open(tmp_path / f"{mod}.py", "a") as f:
        f.write("\n# changed\n")
    plan = load_plan(config_path, env={"KEY": "sk-from-env"})
    plan.make_experiment()
    assert "custom code changed" in plan.warnings[0]
    assert "Warning: The custom code changed" in capsys.readouterr().err
    assert len(list((tmp_path / "out").glob("blade_config-*"))) == 1


def test_record_for_changed_configuration(tmp_path):
    data = base_config(problems=[{"type": "circle_packing"}])
    path = write_config(tmp_path, data)
    load_plan(path, env={}).make_experiment()
    plan = load_plan(path, overrides=["experiment.budget=3"], env={})
    plan.make_experiment()
    assert plan.warnings and "configuration changed" in plan.warnings[0]


def test_custom_source_is_added_to_logged_config():
    class Obj:
        _blade_source = {"source": "x.py", "sha256": "abc"}

    config = Experiment._add_custom_source(Obj(), {"name": "x"})
    assert config["custom_source"]["sha256"] == "abc"
    assert Experiment._add_custom_source(object(), {"a": 1}) == {"a": 1}


# ---------------------------------------------------------- end to end ---


def test_cli_run_with_custom_problem_and_method(tmp_path):
    """A real run: the custom problem is evaluated in its own virtualenv.

    This checks that the module can be imported in the spawned evaluation
    process and in the virtualenv (via Problem.extra_pythonpath).
    """
    from iohblade import cli

    pkg = unique("pkg")
    root = tmp_path / "code" / pkg
    root.mkdir(parents=True)
    (root / "__init__.py").write_text("from .problem import EvalProblem\n")
    (root / "helpers.py").write_text("SCORE = 42.0\n")
    (root / "problem.py").write_text(
        textwrap.dedent(PROBLEM_CODE.format(cls="EvalProblem")).replace(
            "solution.set_scores(float(self.size))",
            "from .helpers import SCORE\n        solution.set_scores(SCORE)",
        )
    )
    mmod = unique("meth")
    write_file(tmp_path / "code", mmod, METHOD_CODE, "OneShot")
    data = base_config(
        problems=[{"type": "custom", "source": f"code/{pkg}", "class": "EvalProblem"}],
        methods=[
            {
                "type": "custom",
                "source": f"code/{mmod}.py",
                "class": "OneShot",
                "llm": "d",
            }
        ],
    )
    path = write_config(tmp_path, data)
    assert cli.main(["run", str(path)]) == cli.EXIT_OK

    out = tmp_path / "out"
    (run_dir,) = [d for d in out.iterdir() if d.name.startswith("run-")]
    logged = [
        json.loads(line) for line in (run_dir / "log.jsonl").read_text().splitlines()
    ]
    assert logged[0]["fitness"] == 42.0, logged[0].get("error")
    problem_json = json.loads((run_dir / "problem.json").read_text())
    assert problem_json["custom_source"]["module"] == pkg
    method_json = json.loads((run_dir / "method.json").read_text())
    assert method_json["custom_source"]["class"] == "OneShot"
    assert (out / "blade_config" / "sources" / pkg / "helpers.py").exists()
