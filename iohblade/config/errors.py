"""Errors raised while loading an experiment configuration."""

from __future__ import annotations

from dataclasses import dataclass, field

Path_ = tuple[str | int, ...]


def format_path(path: Path_) -> str:
    """Render ``("methods", 0, "params", "x")`` as ``methods[0].params.x``."""
    out = ""
    for part in path:
        if isinstance(part, int):
            out += f"[{part}]"
        else:
            out += f".{part}" if out else str(part)
    return out or "<root>"


@dataclass
class ConfigIssue:
    """One problem found in a configuration.

    Args:
        path: Location in the configuration, e.g. ``("methods", 0, "type")``.
        message: What is wrong, ideally with a hint how to fix it.
        file: The configuration file, if known.
        line: 1-based line number, if known.
        column: 1-based column number, if known.
    """

    path: Path_
    message: str
    file: str | None = None
    line: int | None = None
    column: int | None = None

    def location(self) -> str:
        """``file:line:col`` (as far as known)."""
        parts = [p for p in (self.file, self.line, self.column) if p is not None]
        return ":".join(str(p) for p in parts)

    def __str__(self) -> str:
        loc = self.location()
        prefix = f"{loc}: " if loc else ""
        return f"{prefix}{format_path(self.path)}: {self.message}"


@dataclass
class ConfigError(Exception):
    """A configuration is invalid. Holds every issue that was found.

    Attributes:
        issues: All issues, in the order they were found.
    """

    issues: list[ConfigIssue] = field(default_factory=list)

    def __post_init__(self):
        super().__init__(str(self))

    def __str__(self) -> str:
        n = len(self.issues)
        header = f"Invalid experiment configuration ({n} issue{'s' * (n != 1)}):"
        return "\n".join([header] + [f"  - {issue}" for issue in self.issues])
