"""Load YAML experiment files, keeping track of where every value came from.

Besides parsing, this module

* records the line and column of every mapping key and sequence item, so
  errors can point at the offending line;
* substitutes ``${env:NAME}`` and ``${env:NAME:-default}`` in strings, using
  the process environment and an optional ``.env`` file next to the YAML
  file (the process environment wins);
* applies ``key.path=value`` overrides (as given to ``iohblade run --set``).
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .errors import ConfigError, ConfigIssue, Path_

_ENV_PATTERN = re.compile(r"\$\{env:([A-Za-z_][A-Za-z0-9_]*)(?::-(.*?))?\}")


@dataclass
class LoadedConfig:
    """Raw configuration data plus provenance.

    Attributes:
        data: The parsed (and interpolated) configuration.
        file: Path of the YAML file, or None for in-memory configurations.
        marks: ``path -> (line, column)`` (1-based) for every key and item.
        secrets: Paths of values that were filled from the environment;
            they are redacted when the configuration is printed or saved.
        overridden: Paths set by overrides (reported as ``<--set>``).
    """

    data: Any
    file: str | None = None
    marks: dict[Path_, tuple[int, int]] = field(default_factory=dict)
    secrets: set[Path_] = field(default_factory=set)
    overridden: set[Path_] = field(default_factory=set)

    @property
    def base_dir(self) -> Path:
        """Directory that relative paths in the configuration refer to."""
        return Path(self.file).resolve().parent if self.file else Path.cwd()

    def issue(self, path: Path_, message: str) -> ConfigIssue:
        """Build a :class:`ConfigIssue` located at (the closest mark of) path."""
        path = tuple(path)
        for end in range(len(path), 0, -1):
            if path[:end] in self.overridden:
                return ConfigIssue(path, message, "<--set>")
        for end in range(len(path), -1, -1):
            mark = self.marks.get(path[:end])
            if mark is not None:
                return ConfigIssue(path, message, self.file, mark[0], mark[1])
        return ConfigIssue(path, message, self.file)


class _Loader(yaml.SafeLoader):
    pass


def _to_python(loader, node, path, marks, issues, file):
    """Convert a composed YAML node into Python objects, recording marks."""
    marks[path] = (node.start_mark.line + 1, node.start_mark.column + 1)
    if isinstance(node, yaml.MappingNode):
        # Keys written in this mapping itself (not inherited through '<<').
        explicit = {id(k) for k, _ in node.value if k.tag != "tag:yaml.org,2002:merge"}
        loader.flatten_mapping(node)  # resolve '<<' merge keys
        result = {}
        explicit_keys: set[str] = set()
        for key_node, value_node in node.value:
            key = loader.construct_object(key_node, deep=True)
            if not isinstance(key, (str, int, float, bool)) or key is None:
                issues.append(
                    ConfigIssue(
                        path,
                        f"Mapping keys must be scalars, got {key!r}.",
                        file,
                        key_node.start_mark.line + 1,
                        key_node.start_mark.column + 1,
                    )
                )
                continue
            key = str(key) if not isinstance(key, str) else key
            child = path + (key,)
            is_explicit = id(key_node) in explicit
            # flatten_mapping puts inherited pairs first, so a key written
            # here overrides a merged one (YAML merge semantics); only a key
            # written twice in the same mapping is a duplicate.
            if is_explicit and key in explicit_keys:
                issues.append(
                    ConfigIssue(
                        child,
                        "Duplicate key; the earlier value would be ignored.",
                        file,
                        key_node.start_mark.line + 1,
                        key_node.start_mark.column + 1,
                    )
                )
            if is_explicit:
                explicit_keys.add(key)
            result[key] = _to_python(loader, value_node, child, marks, issues, file)
            # Point at the key rather than the value.
            marks[child] = (
                key_node.start_mark.line + 1,
                key_node.start_mark.column + 1,
            )
        return result
    if isinstance(node, yaml.SequenceNode):
        return [
            _to_python(loader, item, path + (i,), marks, issues, file)
            for i, item in enumerate(node.value)
        ]
    return loader.construct_object(node, deep=True)


def parse_yaml(text: str, file: str | None = None) -> LoadedConfig:
    """Parse YAML text into a :class:`LoadedConfig` (no interpolation).

    Raises:
        ConfigError: On YAML syntax errors or duplicate keys.
    """
    loader = _Loader(text)
    try:
        node = loader.get_single_node()
    except yaml.YAMLError as e:
        mark = getattr(e, "problem_mark", None)
        problem = getattr(e, "problem", None) or str(e)
        lines = text.splitlines()
        if mark is not None and mark.line < len(lines) and "${" in lines[mark.line]:
            problem += (
                ". Put values with ${env:...} in quotes, e.g. "
                '"${env:NAME}", when they are inside [...] or {...}'
            )
        raise ConfigError(
            [
                ConfigIssue(
                    (),
                    f"YAML syntax error: {problem}",
                    file,
                    mark.line + 1 if mark else None,
                    mark.column + 1 if mark else None,
                )
            ]
        ) from e
    finally:
        loader.dispose()
    marks: dict[Path_, tuple[int, int]] = {}
    issues: list[ConfigIssue] = []
    data = (
        _to_python(loader, node, (), marks, issues, file) if node is not None else None
    )
    if issues:
        raise ConfigError(issues)
    return LoadedConfig(data, file, marks)


def read_dotenv(path: str | os.PathLike) -> dict[str, str]:
    """Read ``KEY=VALUE`` lines from a ``.env`` file (missing file: empty).

    Supports comments, blank lines, an optional ``export`` prefix and single
    or double quotes around the value.
    """
    path = Path(path)
    if not path.is_file():
        return {}
    values = {}
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export ") :]
        key, _, value = line.partition("=")
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        values[key.strip()] = value
    return values


def interpolate_env(cfg: LoadedConfig, env: dict[str, str]) -> None:
    """Replace ``${env:NAME}`` placeholders in all strings, in place.

    Raises:
        ConfigError: Listing every variable that is not set and has no
            default.
    """
    issues: list[ConfigIssue] = []

    def substitute(value, path):
        if isinstance(value, dict):
            return {k: substitute(v, path + (k,)) for k, v in value.items()}
        if isinstance(value, list):
            return [substitute(v, path + (i,)) for i, v in enumerate(value)]
        if not isinstance(value, str) or "${env:" not in value:
            return value

        def repl(match):
            name, default = match.group(1), match.group(2)
            if name in env:
                return env[name]
            if default is not None:
                return default
            issues.append(
                cfg.issue(
                    path,
                    f"Environment variable '{name}' is not set (set it, add it "
                    "to a .env file next to the configuration, or use "
                    f"${{env:{name}:-default}}).",
                )
            )
            return match.group(0)

        cfg.secrets.add(path)
        return _ENV_PATTERN.sub(repl, value)

    cfg.data = substitute(cfg.data, ())
    if issues:
        raise ConfigError(issues)


def _parse_override(item: str) -> tuple[list[str | int], Any]:
    if "=" not in item:
        raise ValueError(f"Override '{item}' must have the form key.path=value.")
    key, _, raw = item.partition("=")
    parts: list[str | int] = []
    for part in key.strip().split("."):
        if not part:
            raise ValueError(f"Override '{item}' has an empty key segment.")
        parts.append(int(part) if part.isdigit() else part)
    value = yaml.safe_load(raw) if raw.strip() else ""
    return parts, value


def apply_overrides(cfg: LoadedConfig, overrides: list[str] | dict) -> None:
    """Apply ``key.path=value`` overrides in place.

    ``overrides`` is a list of ``"a.b=value"`` strings (values are parsed as
    YAML, so ``runs=3`` gives an int) or a dict ``{"a.b": value}``. List
    items are addressed by index (``methods.0.budget=10``); missing mapping
    keys are created.

    Raises:
        ConfigError: If an override is malformed or its path does not exist.
    """
    if isinstance(overrides, dict):
        items = []
        for key, value in overrides.items():
            parts = [int(p) if p.isdigit() else p for p in str(key).split(".")]
            items.append((str(key), parts, value))
    else:
        items = []
        issues = []
        for item in overrides or []:
            try:
                parts, value = _parse_override(item)
            except (ValueError, yaml.YAMLError) as e:
                issues.append(ConfigIssue((), str(e), "<--set>"))
                continue
            items.append((item, parts, value))
        if issues:
            raise ConfigError(issues)

    issues = []
    if cfg.data is None:
        cfg.data = {}
    for text, parts, value in items:
        node = cfg.data
        ok = True
        for i, part in enumerate(parts[:-1]):
            if isinstance(node, dict):
                node = node.setdefault(part if isinstance(part, str) else str(part), {})
            elif isinstance(node, list) and isinstance(part, int) and part < len(node):
                node = node[part]
            else:
                issues.append(
                    ConfigIssue(
                        tuple(parts[: i + 1]),
                        f"Override '{text}': this path does not exist.",
                        "<--set>",
                    )
                )
                ok = False
                break
        if not ok:
            continue
        last = parts[-1]
        if isinstance(node, dict):
            node[last if isinstance(last, str) else str(last)] = value
        elif isinstance(node, list) and isinstance(last, int) and last < len(node):
            node[last] = value
        else:
            issues.append(
                ConfigIssue(
                    tuple(parts),
                    f"Override '{text}': this path does not exist.",
                    "<--set>",
                )
            )
            continue
        cfg.overridden.add(tuple(parts))
    if issues:
        raise ConfigError(issues)


def load_yaml(
    source: str | os.PathLike | dict,
    *,
    overrides: list[str] | dict | None = None,
    env: dict[str, str] | None = None,
) -> LoadedConfig:
    """Load a configuration from a YAML file, YAML text or a dict.

    Args:
        source: Path to a ``.yml``/``.yaml`` file, or already parsed data
            (relative paths then refer to the current directory).
        overrides: ``key.path=value`` overrides applied after interpolation.
        env: Variables for ``${env:...}``. Defaults to the ``.env`` file next
            to the configuration updated with ``os.environ``.

    Returns:
        LoadedConfig: The data with provenance information.

    Raises:
        ConfigError: If the file cannot be read or parsed, a variable is
            missing, or an override is invalid.
    """
    if not isinstance(source, (str, os.PathLike)):
        import copy

        cfg = LoadedConfig(copy.deepcopy(source))
    else:
        path = Path(source)
        try:
            text = path.read_text()
        except OSError as e:
            raise ConfigError(
                [ConfigIssue((), f"Cannot read configuration: {e}", str(path))]
            ) from e
        cfg = parse_yaml(text, str(path))
    if env is None:
        env = {**read_dotenv(cfg.base_dir / ".env"), **os.environ}
    interpolate_env(cfg, env)
    if overrides:
        apply_overrides(cfg, overrides)
    return cfg
