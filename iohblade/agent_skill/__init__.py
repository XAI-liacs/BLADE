"""A skill that teaches coding agents to set up and run BLADE experiments.

The skill (``SKILL.md`` and ``reference.md`` in this folder) follows the
Agent Skills format used by Claude Code and other coding agents.
``iohblade skill [directory]`` copies it to ``<directory>/blade``
(default ``.claude/skills/blade``). The BLADE repository keeps an identical
copy in ``.claude/skills/blade``.
"""

from __future__ import annotations

from pathlib import Path

SKILL_NAME = "blade"
SKILL_DIR = Path(__file__).resolve().parent
SKILL_FILES = ("SKILL.md", "reference.md")


def install_skill(directory: str | Path = ".claude/skills", force=False) -> list[Path]:
    """Copy the skill to ``<directory>/blade``.

    Args:
        directory: The agent's skills folder.
        force: Overwrite existing files.

    Returns:
        list[Path]: The files written.

    Raises:
        FileExistsError: If a file exists (and differs) and ``force`` is False.
    """
    target = Path(directory) / SKILL_NAME
    contents = {
        name: (SKILL_DIR / name).read_text(encoding="utf-8") for name in SKILL_FILES
    }
    existing = [
        target / name
        for name, text in contents.items()
        if (target / name).exists()
        and (target / name).read_text(encoding="utf-8") != text
    ]
    if existing and not force:
        raise FileExistsError(
            "Not overwriting changed files: "
            + ", ".join(str(p) for p in existing)
            + " (use --force)."
        )
    target.mkdir(parents=True, exist_ok=True)
    written = []
    for name, text in contents.items():
        path = target / name
        path.write_text(text, encoding="utf-8")
        written.append(path)
    return written
