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
from .record import is_secret_key

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
        plain: Paths of unquoted (plain) YAML scalars; only these get a
            type from a ``${env:...}`` value (see :func:`interpolate_env`).
    """

    data: Any
    file: str | None = None
    marks: dict[Path_, tuple[int, int]] = field(default_factory=dict)
    secrets: set[Path_] = field(default_factory=set)
    overridden: set[Path_] = field(default_factory=set)
    plain: set[Path_] = field(default_factory=set)

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


def _to_python(loader, node, path, marks, issues, file, plain):
    """Convert a composed YAML node into Python objects, recording marks.

    Paths of plain (unquoted, non-block) scalars are added to ``plain``.
    """
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
            result[key] = _to_python(
                loader, value_node, child, marks, issues, file, plain
            )
            # Point at the key rather than the value.
            marks[child] = (
                key_node.start_mark.line + 1,
                key_node.start_mark.column + 1,
            )
        return result
    if isinstance(node, yaml.SequenceNode):
        return [
            _to_python(loader, item, path + (i,), marks, issues, file, plain)
            for i, item in enumerate(node.value)
        ]
    if isinstance(node, yaml.ScalarNode) and node.style is None:
        plain.add(path)
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
    plain: set[Path_] = set()
    data = (
        _to_python(loader, node, (), marks, issues, file, plain)
        if node is not None
        else None
    )
    if issues:
        raise ConfigError(issues)
    return LoadedConfig(data, file, marks, plain=plain)


_DOTENV_ESCAPES = {"n": "\n", "r": "\r", "t": "\t", '"': '"', "\\": "\\"}


def _dotenv_value(text: str) -> str:
    """The value of a ``.env`` line (the text after ``=``).

    Like python-dotenv: a quoted value ends at its closing quote (in double
    quotes ``\\n``, ``\\t``, ``\\"`` and ``\\\\`` are escapes; single quotes
    are literal); an unquoted value ends at `` #`` (an inline comment).
    Surrounding whitespace is removed.
    """
    text = text.strip()
    if text[:1] in ("'", '"'):
        quote = text[0]
        out = []
        i = 1
        while i < len(text):
            char = text[i]
            if char == quote:
                return "".join(out)
            if quote == '"' and char == "\\" and i + 1 < len(text):
                nxt = text[i + 1]
                out.append(_DOTENV_ESCAPES.get(nxt, "\\" + nxt))
                i += 2
                continue
            out.append(char)
            i += 1
        # No closing quote: keep the text as written (minus a comment).
    match = re.search(r"\s#", text)
    if match:
        text = text[: match.start()]
    return text.strip()


def read_dotenv(path: str | os.PathLike) -> dict[str, str]:
    """Read ``KEY=VALUE`` lines from a ``.env`` file (missing file: empty).

    Follows python-dotenv: blank lines and ``#`` comment lines are skipped,
    an ``export`` prefix is allowed, unquoted values end at an inline
    `` # comment``, and single- or double-quoted values end at the closing
    quote (double quotes support ``\\n`` and ``\\"`` escapes).
    """
    path = Path(path)
    if not path.is_file():
        return {}
    values = {}
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError as e:
        raise ConfigError(
            [
                ConfigIssue(
                    (),
                    "Cannot read the .env file: it is not valid UTF-8 (byte "
                    f"{e.start}: {e.reason}). Save it with UTF-8 encoding.",
                    str(path),
                )
            ]
        ) from e
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export ") or line.startswith("export\t"):
            line = line[len("export ") :].lstrip()
        key, _, value = line.partition("=")
        key = key.strip()
        if not key:
            continue
        values[key] = _dotenv_value(value)
    return values


def _typed(text: str) -> Any:
    """Parse the text of a lone ``${env:...}`` placeholder as a YAML value.

    ``"3"`` gives 3, ``"true"`` gives True, ``"[1, 2]"`` a list; text that
    is not valid YAML, and the empty string, stay strings.
    """
    if not text.strip():
        return text
    try:
        return yaml.safe_load(text)
    except yaml.YAMLError:
        return text


def _substitute(value, path, env, issue, secrets, plain) -> Any:
    """Replace ``${env:...}`` in ``value`` (recursively).

    Args:
        value: The value to substitute in.
        path: Its configuration path.
        env: The variables.
        issue: ``issue(path, message)`` reports a missing variable.
        secrets: Paths of substituted values are added to it.
        plain: Whether ``path`` may be typed: a callable ``path -> bool``.
    """
    if isinstance(value, dict):
        return {
            k: _substitute(v, path + (k,), env, issue, secrets, plain)
            for k, v in value.items()
        }
    if isinstance(value, list):
        return [
            _substitute(v, path + (i,), env, issue, secrets, plain)
            for i, v in enumerate(value)
        ]
    if not isinstance(value, str) or "${env:" not in value:
        return value
    missing = False

    def repl(match):
        nonlocal missing
        name, default = match.group(1), match.group(2)
        if name in env:
            return env[name]
        if default is not None:
            return default
        missing = True
        issue(
            path,
            f"Environment variable '{name}' is not set (set it, add it "
            "to a .env file next to the configuration, or use "
            f"${{env:{name}:-default}}).",
        )
        return match.group(0)

    secrets.add(path)
    result = _ENV_PATTERN.sub(repl, value)
    key = path[-1] if path else None
    if (
        not missing
        and _ENV_PATTERN.fullmatch(value)
        and plain(path)
        and not is_secret_key(key)
    ):
        return _typed(result)
    return result


def interpolate_env(cfg: LoadedConfig, env: dict[str, str]) -> None:
    """Replace ``${env:NAME}`` placeholders in all strings, in place.

    A value that is exactly one placeholder written without quotes in the
    YAML file gets the type of its text, like a YAML value or a ``--set``
    value: ``n_parents: ${env:NP:-1}`` gives the integer 1. Quoted values
    (``"${env:NAME}"``), placeholders inside a longer string, values of
    credential keys (``api_key``, ``*_token``, ...) and configurations
    passed as a dict always stay strings.

    Raises:
        ConfigError: Listing every variable that is not set and has no
            default.
    """
    issues: list[ConfigIssue] = []

    def issue(path, message):
        issues.append(cfg.issue(path, message))

    cfg.data = _substitute(
        cfg.data, (), env, issue, cfg.secrets, lambda p: p in cfg.plain
    )
    if issues:
        raise ConfigError(issues)


def _parse_override(item: str) -> tuple[list[str | int], Any, bool]:
    """``"a.b=value"`` -> (path parts, parsed value, value was quoted)."""
    if "=" not in item:
        raise ValueError(f"Override '{item}' must have the form key.path=value.")
    key, _, raw = item.partition("=")
    parts: list[str | int] = []
    for part in key.strip().split("."):
        if not part:
            raise ValueError(f"Override '{item}' has an empty key segment.")
        parts.append(int(part) if part.isdigit() else part)
    value = yaml.safe_load(raw) if raw.strip() else ""
    return parts, value, raw.strip()[:1] in ("'", '"')


def apply_overrides(
    cfg: LoadedConfig,
    overrides: list[str] | dict,
    env: dict[str, str] | None = None,
) -> None:
    """Apply ``key.path=value`` overrides in place.

    ``overrides`` is a list of ``"a.b=value"`` strings (values are parsed as
    YAML, so ``runs=3`` gives an int) or a dict ``{"a.b": value}``. List
    items are addressed by index (``methods.0.budget=10``); missing mapping
    keys are created. ``${env:...}`` in override values is substituted like
    in the file: an unquoted lone placeholder (``--set
    methods.0.params.n_parents=${env:NP}``) gets the type of its text, a
    quoted one (``--set 'a="${env:X}"'``) and dict overrides stay strings.

    Args:
        cfg: The configuration to change.
        overrides: The overrides.
        env: Variables for ``${env:...}`` (default: the ``.env`` file next
            to the configuration updated with ``os.environ``).

    Raises:
        ConfigError: If an override is malformed or its path does not exist.
    """
    if isinstance(overrides, dict):
        items = []
        for key, value in overrides.items():
            parts = [int(p) if p.isdigit() else p for p in str(key).split(".")]
            items.append((str(key), parts, value, False))
    else:
        items = []
        issues = []
        for item in overrides or []:
            try:
                parts, value, quoted = _parse_override(item)
            except (ValueError, yaml.YAMLError) as e:
                issues.append(ConfigIssue((), str(e), "<--set>"))
                continue
            items.append((item, parts, value, not quoted))
        if issues:
            raise ConfigError(issues)

    if env is None and any(
        "${env:" in repr(value) for _, _, value, _ in items
    ):  # only read .env when an override needs it
        env = {**read_dotenv(cfg.base_dir / ".env"), **os.environ}
    issues = []
    substituted = []
    for text, parts, value, typed in items:

        def issue(path, message, text=text):
            issues.append(ConfigIssue(path, f"Override '{text}': {message}", "<--set>"))

        top = tuple(parts)
        value = _substitute(
            value,
            top,
            env or {},
            issue,
            cfg.secrets,
            lambda p, t=typed, top=top: t and p == top,
        )
        substituted.append((text, parts, value))
    if issues:
        raise ConfigError(issues)
    items = substituted

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
    """Load a configuration from a YAML file or a dict.

    Args:
        source: Path to a ``.yml``/``.yaml`` file (a ``str`` is always taken
            as a path, never as YAML text; use :func:`parse_yaml` for text),
            or already parsed data (relative paths then refer to the current
            directory).
        overrides: ``key.path=value`` overrides applied after interpolation.
        env: Variables for ``${env:...}``. Defaults to the ``.env`` file next
            to the configuration updated with ``os.environ``.

    Returns:
        LoadedConfig: The data with provenance information.

    Raises:
        ConfigError: If the file cannot be read, is not UTF-8 or cannot be
            parsed, a variable is missing, or an override is invalid.
    """
    if not isinstance(source, (str, os.PathLike)):
        import copy

        cfg = LoadedConfig(copy.deepcopy(source))
    else:
        path = Path(source)
        try:
            text = path.read_text(encoding="utf-8")
        except OSError as e:
            raise ConfigError(
                [ConfigIssue((), f"Cannot read configuration: {e}", str(path))]
            ) from e
        except UnicodeDecodeError as e:
            raise ConfigError(
                [
                    ConfigIssue(
                        (),
                        "Cannot read configuration: the file is not valid "
                        f"UTF-8 (byte {e.start}: {e.reason}). Save it with "
                        "UTF-8 encoding.",
                        str(path),
                    )
                ]
            ) from e
        cfg = parse_yaml(text, str(path))
    if env is None:
        env = {**read_dotenv(cfg.base_dir / ".env"), **os.environ}
    interpolate_env(cfg, env)
    if overrides:
        apply_overrides(cfg, overrides, env)
    return cfg
