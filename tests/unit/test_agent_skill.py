import re
from pathlib import Path

import yaml

from iohblade import cli
from iohblade.agent_skill import SKILL_DIR, SKILL_FILES, install_skill

REPO_COPY = Path(__file__).resolve().parents[2] / ".claude" / "skills" / "blade"


def _frontmatter(text):
    match = re.match(r"---\n(.*?)\n---\n", text, re.S)
    assert match, "SKILL.md must start with YAML frontmatter"
    return yaml.safe_load(match.group(1))


def test_frontmatter():
    meta = _frontmatter((SKILL_DIR / "SKILL.md").read_text())
    assert meta["name"] == "blade"
    assert re.fullmatch(r"[a-z0-9-]{1,64}", meta["name"])
    assert 0 < len(meta["description"]) <= 1024


def test_repository_copy_is_up_to_date():
    # Regenerate with: uv run iohblade skill
    for name in SKILL_FILES:
        assert (REPO_COPY / name).read_text() == (
            SKILL_DIR / name
        ).read_text(), (
            f".claude/skills/blade/{name} is outdated; run `uv run iohblade skill`."
        )


def test_minimal_configuration_validates(tmp_path):
    from iohblade.config import load_plan

    text = (SKILL_DIR / "SKILL.md").read_text()
    block = text.split("## Minimal configuration", 1)[1]
    config = re.search(r"```yaml\n(.*?)```", block, re.S).group(1)
    path = tmp_path / "experiment.yml"
    path.write_text(config)
    plan = load_plan(path)
    assert len(plan.methods) == 2


def test_reference_fields_validate(tmp_path):
    from iohblade.config import load_plan

    text = (SKILL_DIR / "reference.md").read_text()
    block = text.split("## Fields", 1)[1]
    config = yaml.safe_load(re.search(r"```yaml\n(.*?)```", block, re.S).group(1))
    # Make the illustrative values concrete.
    config["llms"] = {"qwen": {"provider": "dummy"}}
    config["experiment"]["logger"] = {"type": "file"}
    config["problems"][0].pop("smoke_test_solution")
    path = tmp_path / "experiment.yml"
    path.write_text(yaml.safe_dump(config))
    load_plan(path)


def test_install_and_refuse_overwrite(tmp_path):
    files = install_skill(tmp_path)
    assert [p.name for p in files] == list(SKILL_FILES)
    assert (tmp_path / "blade" / "SKILL.md").read_text() == (
        SKILL_DIR / "SKILL.md"
    ).read_text()
    install_skill(tmp_path)  # unchanged files: fine
    (tmp_path / "blade" / "SKILL.md").write_text("edited")
    try:
        install_skill(tmp_path)
    except FileExistsError as e:
        assert "--force" in str(e)
    else:
        raise AssertionError("expected FileExistsError")
    install_skill(tmp_path, force=True)
    assert (tmp_path / "blade" / "SKILL.md").read_text() != "edited"


def test_cli_skill(tmp_path, capsys):
    assert cli.main(["skill", str(tmp_path)]) == cli.EXIT_OK
    assert "blade/SKILL.md" in capsys.readouterr().out.replace("\\", "/")
    (tmp_path / "blade" / "reference.md").write_text("edited")
    assert cli.main(["skill", str(tmp_path)]) == cli.EXIT_CONFIG
