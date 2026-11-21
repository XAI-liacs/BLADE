import textwrap

import pytest

from iohblade.config.errors import ConfigError, ConfigIssue, format_path
from iohblade.config.yaml_loader import (
    apply_overrides,
    load_yaml,
    parse_yaml,
    read_dotenv,
)


def _write(tmp_path, text, name="exp.yml"):
    path = tmp_path / name
    path.write_text(textwrap.dedent(text))
    return path


def test_marks_point_at_keys_and_items():
    cfg = parse_yaml("a:\n  b: 1\n  c:\n    - x\n    - y\n", "f.yml")
    assert cfg.data == {"a": {"b": 1, "c": ["x", "y"]}}
    assert cfg.marks[("a", "b")] == (2, 3)
    assert cfg.marks[("a", "c", 1)] == (5, 7)
    issue = cfg.issue(("a", "c", 1, "deeper"), "bad")
    assert (issue.file, issue.line, issue.column) == ("f.yml", 5, 7)
    assert str(issue) == "f.yml:5:7: a.c[1].deeper: bad"


def test_duplicate_keys_are_reported():
    with pytest.raises(ConfigError) as exc:
        parse_yaml("a: 1\nb: 2\na: 3\n", "f.yml")
    (issue,) = exc.value.issues
    assert issue.line == 3 and "Duplicate key" in issue.message


def test_merged_keys_can_be_overridden():
    cfg = parse_yaml(
        "defaults: &d {runs: 5, budget: 10}\n"
        "exp:\n  <<: *d\n  runs: 2\n"
        "two:\n  <<: [*d, {runs: 7}]\n"
    )
    assert cfg.data["exp"] == {"runs": 2, "budget": 10}
    assert cfg.data["two"]["runs"] == 5  # first merged mapping wins
    with pytest.raises(ConfigError, match="Duplicate key"):
        parse_yaml("exp:\n  <<: {runs: 5}\n  runs: 2\n  runs: 3\n")


def test_syntax_error_has_location():
    with pytest.raises(ConfigError) as exc:
        parse_yaml("a: [1, 2\nb: 3\n", "f.yml")
    (issue,) = exc.value.issues
    assert issue.file == "f.yml" and issue.line is not None
    assert "YAML syntax error" in issue.message


def test_unquoted_placeholder_in_flow_sequence_gets_a_hint():
    with pytest.raises(ConfigError, match="in quotes"):
        parse_yaml("b: [${env:X}]\n")


def test_merge_keys_and_empty_document():
    cfg = parse_yaml("base: &b {x: 1}\nother:\n  <<: *b\n  y: 2\n")
    assert cfg.data["other"] == {"x": 1, "y": 2}
    assert parse_yaml("").data is None


def test_env_interpolation(tmp_path):
    path = _write(
        tmp_path,
        """
        key: ${env:MY_KEY}
        url: http://${env:HOST:-localhost}:1
        plain: no placeholders
        """,
    )
    cfg = load_yaml(path, env={"MY_KEY": "secret"})
    assert cfg.data["key"] == "secret"
    assert cfg.data["url"] == "http://localhost:1"
    assert cfg.secrets == {("key",), ("url",)}


def test_missing_env_variables_are_all_reported(tmp_path):
    path = _write(tmp_path, "a: ${env:NOPE_1}\nb:\n  - ${env:NOPE_2}\n")
    with pytest.raises(ConfigError) as exc:
        load_yaml(path, env={})
    lines = sorted(i.line for i in exc.value.issues)
    assert lines == [1, 3]
    assert "NOPE_1" in str(exc.value) and "NOPE_2" in str(exc.value)


def test_dotenv_next_to_config(tmp_path, monkeypatch):
    (tmp_path / ".env").write_text(
        "# comment\nexport FROM_FILE='file value'\nBOTH=file\n\nnot a line\n"
    )
    monkeypatch.setenv("BOTH", "environment")
    path = _write(tmp_path, "a: ${env:FROM_FILE}\nb: ${env:BOTH}\n")
    cfg = load_yaml(path)
    assert cfg.data == {"a": "file value", "b": "environment"}


def test_read_dotenv_missing_file(tmp_path):
    assert read_dotenv(tmp_path / "nope") == {}


def test_overrides_from_strings(tmp_path):
    path = _write(
        tmp_path,
        """
        experiment: {runs: 5}
        methods:
          - {type: a}
          - {type: b}
        """,
    )
    cfg = load_yaml(
        path,
        overrides=[
            "experiment.runs=2",
            "methods.1.type=c",
            "experiment.new.nested=[1, 2]",
            "experiment.name=",
        ],
        env={},
    )
    assert cfg.data["experiment"]["runs"] == 2
    assert cfg.data["methods"][1]["type"] == "c"
    assert cfg.data["experiment"]["new"] == {"nested": [1, 2]}
    assert cfg.data["experiment"]["name"] == ""
    issue = cfg.issue(("experiment", "runs"), "x")
    assert issue.file == "<--set>"


def test_overrides_from_dict():
    cfg = load_yaml({"a": {"b": 1}}, overrides={"a.b": 3}, env={})
    assert cfg.data == {"a": {"b": 3}}


@pytest.mark.parametrize(
    "override",
    ["no_equals_sign", "a..b=1", "methods.5.type=x", "a.b.c=1"],
)
def test_bad_overrides(override):
    cfg = load_yaml({"a": {"b": 1}, "methods": [{"type": "x"}]}, env={})
    with pytest.raises(ConfigError):
        apply_overrides(cfg, [override])


def test_unreadable_file(tmp_path):
    with pytest.raises(ConfigError, match="Cannot read configuration"):
        load_yaml(tmp_path / "missing.yml")


def test_dict_source_is_copied():
    data = {"a": {"b": 1}}
    cfg = load_yaml(data, overrides=["a.b=2"], env={})
    assert data["a"]["b"] == 1 and cfg.data["a"]["b"] == 2


def test_error_formatting():
    assert format_path(("methods", 0, "params", "x")) == "methods[0].params.x"
    assert format_path(()) == "<root>"
    err = ConfigError([ConfigIssue(("a",), "bad"), ConfigIssue(("b",), "worse")])
    assert "(2 issues)" in str(err)
    assert "  - a: bad" in str(err)
