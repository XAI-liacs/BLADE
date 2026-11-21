import json

import pytest
import yaml

from iohblade import cli

CONFIG = {
    "experiment": {"output_dir": "out", "runs": 1, "budget": 2},
    "llms": {"d": {"provider": "dummy"}},
    "methods": [{"type": "random_search", "name": "RS-{llm}", "llm": "d"}],
    "problems": [{"type": "circle_packing", "name": "circles"}],
}


@pytest.fixture
def config_file(tmp_path):
    path = tmp_path / "exp.yml"
    path.write_text(yaml.safe_dump(CONFIG))
    return path


def test_validate_ok(config_file, capsys):
    assert cli.main(["validate", str(config_file)]) == cli.EXIT_OK
    out = capsys.readouterr().out
    assert "RS-d (LLM: d, budget: 2)" in out
    assert "Configuration is valid." in out
    assert not (config_file.parent / "out").exists()


def test_validate_with_override(config_file, capsys):
    cli.main(["validate", str(config_file), "--set", "experiment.runs=3", "-v"])
    assert "x 3 seeds = 3" in capsys.readouterr().out


def test_validate_invalid(tmp_path, capsys):
    path = tmp_path / "bad.yml"
    path.write_text("experiment:\n  output_dir: x\n  runz: 1\n")
    assert cli.main(["validate", str(path)]) == cli.EXIT_CONFIG
    err = capsys.readouterr().err
    assert f"{path}:3:3: experiment.runz: Unknown field 'runz'" in err


def test_run_dry_run(config_file, capsys):
    assert cli.main(["run", str(config_file), "--dry-run"]) == cli.EXIT_OK
    assert "Runs: 1 methods x 1 problems x 1 seeds = 1" in capsys.readouterr().out
    assert not (config_file.parent / "out").exists()


def test_run_calls_experiment(config_file, monkeypatch, capsys):
    from iohblade.experiment import Experiment

    calls = []
    monkeypatch.setattr(Experiment, "__call__", lambda self: calls.append(self))
    assert cli.main(["run", str(config_file)]) == cli.EXIT_OK
    assert len(calls) == 1 and calls[0].methods[0].name == "RS-d"
    assert "Results:" in capsys.readouterr().out


def test_run_failure_exit_code(config_file, monkeypatch, capsys):
    from iohblade.experiment import Experiment

    def fail(self):
        raise RuntimeError("1 of 1 runs failed")

    monkeypatch.setattr(Experiment, "__call__", fail)
    assert cli.main(["run", str(config_file)]) == cli.EXIT_RUN_FAILED
    assert "1 of 1 runs failed" in capsys.readouterr().err


def test_run_interrupted(config_file, monkeypatch):
    from iohblade.experiment import Experiment

    def interrupt(self):
        raise KeyboardInterrupt

    monkeypatch.setattr(Experiment, "__call__", interrupt)
    assert cli.main(["run", str(config_file)]) == 130


def test_list(capsys):
    assert cli.main(["list"]) == cli.EXIT_OK
    out = capsys.readouterr().out
    for kind in ("problems:", "suites:", "methods:", "llms:", "loggers:"):
        assert kind in out
    assert "llamea" in out and "uv sync --group methods" in out


def test_list_kind_with_check(capsys):
    assert cli.main(["list", "loggers", "--check"]) == cli.EXIT_OK
    out = capsys.readouterr().out
    assert "methods:" not in out
    assert "file" in out and "[ok]" in out


def test_schema(tmp_path, capsys):
    assert cli.main(["schema"]) == cli.EXIT_OK
    assert json.loads(capsys.readouterr().out)["title"]
    target = tmp_path / "schema.json"
    assert cli.main(["schema", "-o", str(target)]) == cli.EXIT_OK
    assert "experiment" in json.loads(target.read_text())["properties"]


def test_usage_errors_exit_2():
    with pytest.raises(SystemExit) as exc:
        cli.main(["frobnicate"])
    assert exc.value.code == 2
