"""Record what an experiment was run with, for reproducibility.

``ExperimentPlan.make_experiment`` writes into ``<output_dir>/blade_config/``:

* ``resolved.yml`` - the configuration after ``${env:...}`` substitution and
  ``--set`` overrides, with secrets (API keys, tokens, values read from the
  environment) redacted;
* ``sources/`` - a copy of every custom source file or package;
* ``sources.json`` - the SHA-256 of every custom source.

When an experiment is resumed with a different configuration or changed
custom code, the previous record is moved to ``blade_config-<timestamp>/``
and a warning is returned, so it stays clear which runs used what.
"""

from __future__ import annotations

import copy
import json
import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

if TYPE_CHECKING:  # pragma: no cover
    from .build import ExperimentPlan

RECORD_DIR = "blade_config"
REDACTED = "<redacted>"
# A key is a credential if it is (or ends in) one of these words; camelCase is
# split first, so ``apiKey`` and ``authToken`` match but ``max_tokens``,
# ``tokenizer`` and ``token_limit`` do not.
_SECRET_KEY = re.compile(
    r"^(?:.*[_-])?(?:api[_-]?key|apikey|token|secret|password|passwd"
    r"|secret[_-]?key|access[_-]?key|private[_-]?key)$",
    re.IGNORECASE,
)


def is_secret_key(key: Any) -> bool:
    """Whether a mapping key names a credential (``api_key``, ``*_token``, ...).

    Matches ``api_key``/``apikey``, ``token``, ``secret``, ``password`` and
    keys ending in ``_token``, ``_secret``, ``_password``, ``_api_key`` (also
    with ``-`` or in camelCase), but not keys that merely contain such a word
    (``max_tokens``, ``tokenizer``, ``token_limit``).
    """
    if not isinstance(key, str):
        return False
    snake = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", key)
    return bool(_SECRET_KEY.match(snake))


def redact(data: Any, secret_paths: set[tuple]) -> Any:
    """Copy of ``data`` with secrets replaced by ``<redacted>``.

    A value is a secret if it was read from the environment (its path is in
    ``secret_paths``) or its key names a credential (see
    :func:`is_secret_key`).
    """

    def walk(value, path):
        if path in secret_paths and isinstance(value, str):
            return REDACTED
        if isinstance(value, dict):
            out = {}
            for k, v in value.items():
                if is_secret_key(k) and v is not None:
                    out[k] = REDACTED
                else:
                    out[k] = walk(v, path + (k,))
            return out
        if isinstance(value, list):
            return [walk(v, path + (i,)) for i, v in enumerate(value)]
        return value

    return walk(copy.deepcopy(data), ())


def _to_yaml(data: Any) -> str:
    try:
        return yaml.safe_dump(data, sort_keys=False, allow_unicode=True)
    except yaml.YAMLError:
        # Configurations built in Python may hold arbitrary objects.
        plain = json.loads(json.dumps(data, default=str))
        return yaml.safe_dump(plain, sort_keys=False, allow_unicode=True)


def _version() -> str:
    try:
        from importlib.metadata import version

        return version("iohblade")
    except Exception:  # pragma: no cover - not installed as a package
        return "unknown"


def write_record(plan: "ExperimentPlan", output_dir: str | Path) -> list[str]:
    """Write the configuration and custom sources of ``plan``.

    Args:
        plan: The plan being run.
        output_dir: The experiment's results directory (must exist).

    Returns:
        list[str]: Warnings, e.g. that a resumed experiment changed.
    """
    output_dir = Path(output_dir)
    record = output_dir / RECORD_DIR
    resolved = _to_yaml(redact(plan.source.data, plan.source.secrets))
    hashes = {
        s.module_name: {"source": str(s.path), "sha256": s.sha256} for s in plan.sources
    }

    warnings: list[str] = []
    old_resolved = record / "resolved.yml"
    old_hashes = record / "sources.json"
    if old_resolved.exists():
        previous = _strip_header(old_resolved.read_text())
        prev_hashes = json.loads(old_hashes.read_text()) if old_hashes.exists() else {}
        changed = []
        if previous != resolved:
            changed.append("configuration")
        if prev_hashes != hashes:
            changed.append("custom code")
        if changed:
            stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            archive = output_dir / f"{RECORD_DIR}-{stamp}"
            shutil.move(str(record), str(archive))
            warnings.append(
                f"The {' and '.join(changed)} changed since this experiment was "
                f"last started; runs already finished used the old version "
                f"(kept in {archive})."
            )

    record.mkdir(parents=True, exist_ok=True)
    header = (
        f"# Resolved BLADE experiment configuration (secrets redacted)\n"
        f"# written: {datetime.now().isoformat(timespec='seconds')}\n"
        f"# from: {plan.source.file or '<in-memory configuration>'}\n"
        f"# iohblade: {_version()}\n"
    )
    old_resolved.write_text(header + resolved)
    old_hashes.write_text(json.dumps(hashes, indent=2) + "\n")
    sources_dir = record / "sources"
    for source in plan.sources:
        source.snapshot(sources_dir)
    return warnings


def _strip_header(text: str) -> str:
    lines = text.splitlines(keepends=True)
    while lines and lines[0].startswith("#"):
        lines.pop(0)
    return "".join(lines)
