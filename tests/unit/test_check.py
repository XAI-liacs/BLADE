import json
import textwrap

import pytest

from iohblade import cli
from iohblade.config import load_plan
from iohblade.config.check import (
    FAIL,
    OK,
    SKIP,
    WARN,
    ReplayLLM,
    check_method,
    check_problem,
    ping_llm,
    reference_solution,
    run_checks,
)
from iohblade.config.templates import TEMPLATES, write_template
from iohblade.llm import LLM
from iohblade.solution import Solution

from .harness_problems import CandidateBugProblem, GoodProblem, HarnessBugProblem


@pytest.mark.parametrize("template", sorted(TEMPLATES))
def test_templates_pass_validate_and_check(template, tmp_path, capsys):
    files = write_template(template, tmp_path)
    config = tmp_path / "experiment.yml"
    assert config in files and (tmp_path / "experiment.schema.json").exists()
    assert cli.main(["validate", str(config)]) == cli.EXIT_OK
    capsys.readouterr()
    assert cli.main(["check", str(config)]) == cli.EXIT_OK
    out = capsys.readouterr().out
    assert "[FAIL]" not in out and "[WARN]" not in out, out
    assert "ready to run" in out and "add --llm" in out


@pytest.mark.parametrize("template", sorted(TEMPLATES))
def test_template_results_are_listed_by_the_webapp(template, tmp_path):
    """The web app lists subfolders of results/; the templates used to write
    straight to results/, so their experiments never showed up."""
    import yaml

    from iohblade.webapp import discover_experiments

    write_template(template, tmp_path)
    config = yaml.safe_load((tmp_path / "experiment.yml").read_text())
    output_dir = tmp_path / config["experiment"]["output_dir"]
    assert output_dir.parent == tmp_path / "results"
    output_dir.mkdir(parents=True)
    (output_dir / "progress.json").write_text("{}")
    assert discover_experiments(str(tmp_path / "results")) == [output_dir.name]


def test_init_refuses_to_overwrite(tmp_path, capsys):
    assert cli.main(["init", str(tmp_path), "-t", "custom-problem"]) == cli.EXIT_OK
    assert "Created a 'custom-problem' experiment" in capsys.readouterr().out
    assert cli.main(["init", str(tmp_path), "-t", "custom-problem"]) == cli.EXIT_CONFIG
    assert "use --force" in capsys.readouterr().err
    assert cli.main(["init", str(tmp_path), "-t", "builtin", "--force"]) == cli.EXIT_OK


def _write(tmp_path, name, code):
    path = tmp_path / name
    path.write_text(textwrap.dedent(code))
    return path


def test_check_problem_results(tmp_path):
    good = _write(tmp_path, "good.py", "x = 1\n")
    result, code = check_problem(GoodProblem(), good)
    assert result.status == OK and "fitness 6" in result.message and code == "x = 1\n"

    bad = _write(tmp_path, "bad.py", "def run():\n    raise ValueError('nope')\n")
    result, _ = check_problem(CandidateBugProblem(), bad)
    assert (
        result.status == FAIL
        and "evaluating smoke_test_solution bad.py failed" in result.message
    )
    assert "nope" in result.message

    result, _ = check_problem(HarnessBugProblem(), good)
    assert (
        result.status == FAIL and "fails before the solution is used" in result.message
    )

    result, code = check_problem(GoodProblem(), None)
    assert (
        result.status == WARN
        and "smoke_test_solution" in result.message
        and code is None
    )


def test_check_problem_environment_failure(monkeypatch):
    problem = GoodProblem()

    def broken():
        raise RuntimeError(
            "Failed to install the dependencies ['x']:\n"
            + "\n".join(f"line {i}" for i in range(30))
            + "\nerror: command 'swig' failed"
        )

    monkeypatch.setattr(problem, "_ensure_env", broken)
    result, _ = check_problem(problem, None)
    assert result.status == FAIL
    assert "Failed to install the dependencies ['x']" in result.message
    assert "swig" in result.message and "line 3\n" not in result.message
    assert "iohblade check -v" in result.message
    full, _ = check_problem(problem, None, verbose=True)
    assert "line 3" in full.message


def test_reference_solution_from_example_prompt():
    problem = GoodProblem()
    problem.example_prompt = "Example:\n```python\nclass A:\n    pass\n```\n"
    code, origin = reference_solution(problem, None)
    assert code == "class A:\n    pass" and "example" in origin


class _BrokenMethod:
    name = "broken"
    budget = 5

    def __init__(self):
        self.llm = None

    def __call__(self, problem):
        raise KeyError("missing config")


class _ReturnsBad(_BrokenMethod):
    name = "bad"

    def __call__(self, problem):
        solution = Solution()
        solution.set_scores(float("-inf"), "it failed", "it failed")
        return solution


def test_check_method_results():
    assert check_method(_BrokenMethod(), GoodProblem(), "x = 1").status == FAIL
    result = check_method(_ReturnsBad(), GoodProblem(), "x = 1")
    assert result.status == WARN and "it failed" in result.message


class _LLM(LLM):
    def __init__(self, answer=None, error=None):
        super().__init__("", "test")
        self.answer, self.error = answer, error

    def _query(self, session, **kwargs):
        if self.error:
            raise self.error
        return self.answer

    def get_config(self):
        return []


def test_ping_llm():
    assert ping_llm("a", _LLM("OK")).status == OK
    assert ping_llm("b", _LLM("")).status == WARN
    failed = ping_llm("c", _LLM(error=ConnectionError("no route")))
    assert failed.status == FAIL and "ConnectionError: no route" in failed.message


def test_replay_llm_returns_code():
    llm = ReplayLLM("class Opt:\n    pass\n")
    solution = llm.sample_solution([{"role": "user", "content": "hi"}])
    assert "class Opt" in solution.code and solution.name == "Opt"


def test_run_checks_skips_methods_without_reference_and_pings(tmp_path):
    from types import SimpleNamespace

    from iohblade.methods import RandomSearch

    problem = GoodProblem()  # its example prompt has no code block
    plan = SimpleNamespace(
        problems=[problem],
        smoke_tests=[None],
        methods=[RandomSearch(_LLM("x"), budget=1)],
        llms={"d": _LLM("OK")},
    )
    results = run_checks(plan, llms=True)
    statuses = [(r.kind, r.status) for r in results]
    assert statuses == [("problem", WARN), ("method", SKIP), ("llm", OK)]


def test_check_cli_exit_code_on_failure(tmp_path, capsys):
    (tmp_path / "bad.py").write_text("raise SystemError('broken reference')\n")
    data = {
        "experiment": {"output_dir": "out", "runs": 1, "budget": 2},
        "llms": {"d": {"provider": "dummy"}},
        "methods": [{"type": "random_search", "llm": "d"}],
        "problems": [
            {"type": "circle_packing", "smoke_test_solution": "bad.py"},
        ],
    }
    import yaml

    path = tmp_path / "exp.yml"
    path.write_text(yaml.safe_dump(data))
    assert cli.main(["check", str(path), "--no-methods"]) == cli.EXIT_RUN_FAILED
    out = capsys.readouterr().out
    assert "[FAIL] problem" in out and "fix the failures" in out


def test_missing_smoke_test_file_is_a_config_error(tmp_path):
    from iohblade.config import ConfigError

    data = {
        "experiment": {"output_dir": "out"},
        "llms": {"d": {"provider": "dummy"}},
        "methods": [{"type": "random_search", "llm": "d"}],
        "problems": [{"type": "circle_packing", "smoke_test_solution": "nope.py"}],
    }
    with pytest.raises(ConfigError, match="problems\\[0\\].smoke_test_solution"):
        load_plan(data, env={})


def test_env_settings_are_applied_to_problems():
    data = {
        "experiment": {
            "output_dir": "out",
            "env_cache": False,
            "isolated_pythonpath": True,
        },
        "llms": {"d": {"provider": "dummy"}},
        "methods": [{"type": "random_search", "llm": "d"}],
        "problems": [
            {"type": "circle_packing"},
            {"suite": "hexagon_packing", "params": {"get_best": False}},
        ],
    }
    plan = load_plan(data, env={})
    assert all(p.env_cache is False and p.isolated_pythonpath for p in plan.problems)


def test_dry_run_reports_resume_state(tmp_path, capsys):
    import yaml

    data = {
        "experiment": {"output_dir": "out", "runs": 2, "budget": 2},
        "llms": {"d": {"provider": "dummy", "model": "m"}},
        "methods": [{"type": "random_search", "name": "rs", "llm": "d"}],
        "problems": [{"type": "circle_packing", "name": "c"}],
    }
    path = tmp_path / "exp.yml"
    path.write_text(yaml.safe_dump(data))
    assert cli.main(["run", str(path), "--dry-run"]) == cli.EXIT_OK
    assert "Output directory is new" in capsys.readouterr().out

    out = tmp_path / "out"
    out.mkdir()
    runs = [
        {
            "method_name": "rs",
            "problem_name": "c",
            "seed": 0,
            "llm_name": "m",
            "end_time": "x",
        },
        {
            "method_name": "rs",
            "problem_name": "c",
            "seed": 1,
            "llm_name": "m",
            "end_time": None,
        },
    ]
    (out / "progress.json").write_text(json.dumps({"runs": runs}))
    cli.main(["run", str(path), "--dry-run"])
    text = capsys.readouterr().out
    assert "1 of 2 runs already finished" in text and "Warning" not in text

    # Adding runs to an existing output directory is supported (#208).
    cli.main(["run", str(path), "--dry-run", "--set", "experiment.runs=3"])
    text = capsys.readouterr().out
    assert "1 of 3 runs already finished" in text
    assert "1 are new in this configuration" in text and "seed 2" in text

    # Leaving out recorded runs is not.
    cli.main(["run", str(path), "--dry-run", "--set", "experiment.runs=1"])
    text = capsys.readouterr().out
    assert "Cannot resume" in text and "seed 1" in text

    (out / "progress.json").write_text("{broken")
    cli.main(["run", str(path), "--dry-run"])
    assert "Could not read" in capsys.readouterr().out


def test_cache_cli(capsys):
    from iohblade import envs

    envs.get_env([], cache=True)
    assert cli.main(["cache", "list"]) == cli.EXIT_OK
    out = capsys.readouterr().out
    assert "Cached evaluation environments" in out and "MB" in out
    assert cli.main(["cache", "clear"]) == cli.EXIT_OK
    assert "Removed" in capsys.readouterr().out
    cli.main(["cache", "list"])
    assert "(none)" in capsys.readouterr().out
