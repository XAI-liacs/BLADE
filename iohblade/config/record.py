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
import os
import re
import shutil
from dataclasses import dataclass, field
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
        # Values read from the environment are redacted whatever their type
        # (an unquoted ${env:...} value can be a number or a list).
        if path in secret_paths and value is not None:
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


def _relative_source(path: Path, base_dir: Path) -> str:
    """``path`` relative to the configuration folder (POSIX separators).

    Relative paths keep the record valid when the experiment folder is moved;
    a path that cannot be made relative (another drive) stays absolute.
    """
    try:
        return Path(os.path.relpath(path, base_dir)).as_posix()
    except ValueError:  # pragma: no cover - different drive on Windows
        return Path(path).as_posix()


def _comparable_hashes(hashes: Any, base_dir: Path) -> dict[str, tuple]:
    """``module -> (relative source or None, sha256)`` of a ``sources.json``.

    Records written by older versions hold absolute paths; those are made
    relative to ``base_dir`` when they lie inside it, and otherwise only
    their hash is compared (the folder may have been moved since).
    """
    out: dict[str, tuple] = {}
    if not isinstance(hashes, dict):
        return out
    for module, entry in hashes.items():
        if not isinstance(entry, dict):
            continue
        source = entry.get("source")
        if isinstance(source, str) and Path(source).is_absolute():
            try:
                source = Path(source).relative_to(base_dir).as_posix()
            except ValueError:
                source = None
        out[module] = (source, entry.get("sha256"))
    return out


def _same_sources(previous: Any, current: dict, base_dir: Path) -> bool:
    old = _comparable_hashes(previous, base_dir)
    new = _comparable_hashes(current, base_dir)
    if old.keys() != new.keys():
        return False
    for module, (source, sha) in new.items():
        old_source, old_sha = old[module]
        if old_sha != sha or (old_source is not None and old_source != source):
            return False
    return True


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
    base_dir = plan.source.base_dir
    resolved = _to_yaml(redact(plan.source.data, plan.source.secrets))
    hashes = {
        s.module_name: {
            "source": _relative_source(s.path, base_dir),
            "sha256": s.sha256,
        }
        for s in plan.sources
    }

    warnings: list[str] = []
    old_resolved = record / "resolved.yml"
    old_hashes = record / "sources.json"
    if old_resolved.exists():
        previous = _strip_header(old_resolved.read_text(encoding="utf-8"))
        try:
            prev_hashes = (
                json.loads(old_hashes.read_text(encoding="utf-8"))
                if old_hashes.exists()
                else {}
            )
        except ValueError:
            prev_hashes = None  # damaged: treat as changed
        changed = []
        if previous != resolved:
            changed.append("configuration")
        if not _same_sources(prev_hashes, hashes, base_dir):
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
    old_resolved.write_text(header + resolved, encoding="utf-8")
    old_hashes.write_text(json.dumps(hashes, indent=2) + "\n", encoding="utf-8")
    sources_dir = record / "sources"
    for source in plan.sources:
        source.snapshot(sources_dir)
    return warnings


# ---------------------------------------------------------------- resume ---

# (method, LLM or None, problem, seed)
RunKey = tuple[str, Any, str, int]


def _describe(key: RunKey) -> str:
    method, llm, problem, seed = key
    with_llm = f" ({llm})" if llm else ""
    return f"{method}{with_llm} on {problem}, seed {seed}"


def _listing(keys: set[RunKey], limit: int = 5) -> str:
    items = sorted(keys, key=lambda k: tuple(str(part) for part in k))
    text = "; ".join(_describe(k) for k in items[:limit])
    if len(items) > limit:
        text += f"; ... ({len(items) - limit} more)"
    return text


@dataclass
class ResumeStatus:
    """What running a plan in an existing output directory would do.

    A run is identified as in ``progress.json``: by (method name, problem
    name, seed). Resuming skips the finished runs. Runs that are planned
    but not recorded yet (more ``runs``/``seeds``, another method or
    problem) are added. Runs that are recorded but no longer planned make
    the directory incompatible: BLADE never drops recorded runs.

    Attributes:
        output_dir: The results directory.
        planned: The planned runs.
        recorded: The runs recorded in ``progress.json`` (empty if new).
        finished: How many planned runs already finished.
        error: Why ``progress.json`` could not be read, if it could not.
    """

    output_dir: Path
    planned: set[RunKey] = field(default_factory=set)
    recorded: set[RunKey] = field(default_factory=set)
    finished: int = 0
    error: str | None = None

    @property
    def is_new(self) -> bool:
        """No runs are recorded in the directory yet."""
        return self.error is None and not self.recorded

    @property
    def added(self) -> set[RunKey]:
        """Planned runs that are not recorded yet."""
        return self.planned - self.recorded if self.recorded else set()

    @property
    def removed(self) -> set[RunKey]:
        """Recorded runs that are no longer planned."""
        return self.recorded - self.planned

    @property
    def compatible(self) -> bool:
        """Whether the plan can run in (resume) this directory."""
        return self.error is None and not self.removed

    def summary(self) -> list[str]:
        """Lines for ``iohblade run --dry-run``."""
        if self.error is not None:
            return [f"Cannot resume: {self.error}"]
        if self.is_new:
            return ["Output directory is new: all runs will start."]
        if not self.compatible:
            return [f"Cannot resume: {self.mismatch_message()}"]
        n = len(self.planned)
        lines = [
            f"Resuming: {self.finished} of {n} runs already finished and "
            f"will be skipped; {n - self.finished} will run."
        ]
        if self.added:
            lines.append(
                f"Of these, {len(self.added)} are new in this configuration "
                f"and will be added to the experiment: {_listing(self.added)}."
            )
        return lines

    def mismatch_message(self) -> str:
        """Why the plan cannot resume, and what to do instead."""
        if self.error is not None:
            return self.error
        lines = [
            f"The output directory {self.output_dir} already holds "
            f"{len(self.recorded)} run(s) of a different setup, and this "
            f"configuration plans {len(self.planned)}. Resuming can skip "
            "finished runs and add new ones, but not leave recorded runs out.",
            f"Recorded but no longer planned: {_listing(self.removed)}.",
        ]
        rec_seeds = sorted({k[3] for k in self.recorded})
        plan_seeds = sorted({k[3] for k in self.planned})
        if rec_seeds != plan_seeds:
            lines.append(f"Seeds recorded: {rec_seeds}; planned: {plan_seeds}.")
        for what, index in (("Methods", 0), ("LLMs", 1), ("Problems", 2)):
            rec = sorted({str(k[index]) for k in self.recorded})
            new = sorted({str(k[index]) for k in self.planned})
            if rec != new:
                lines.append(f"{what} recorded: {rec}; planned: {new}.")
        lines.append(
            "Either use a new output directory (change experiment.output_dir, "
            "e.g. --set experiment.output_dir=results/trial), or restore the "
            "previous settings (runs/seeds, methods and problems) to resume."
        )
        return "\n    ".join(lines)


def _progress_runs(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("not a JSON object")
    runs = data.get("runs", [])
    if not isinstance(runs, list):
        raise ValueError("'runs' is not a list")
    return [r for r in runs if isinstance(r, dict)]


def resume_status(plan: "ExperimentPlan", output_dir=None) -> ResumeStatus:
    """Compare the plan with the progress recorded in its output directory.

    Args:
        plan: The plan to run.
        output_dir: The results directory (default: ``plan.output_dir``).

    Returns:
        ResumeStatus: What resuming would do; nothing is written.
    """
    from ..loggers.base import ExperimentLogger

    output_dir = Path(output_dir if output_dir is not None else plan.output_dir)
    planned_runs = [
        (m.name, ExperimentLogger._get_clean_llm_model_name(None, m.llm), p.name, s)
        for m in plan.methods
        for p in plan.problems
        for s in plan.seeds
    ]
    status = ResumeStatus(output_dir, {_identity(*run) for run in planned_runs})
    path = output_dir / "progress.json"
    if not path.is_file():
        return status
    try:
        runs = _progress_runs(path)
        with_llm = all(r.get("llm_name") for r in runs)
        status.recorded = {_identity_of(r, with_llm) for r in runs}
        if not with_llm:  # older progress files have no llm_name
            status.planned = {_identity(m, None, p, s) for m, _, p, s in planned_runs}
    except (OSError, ValueError, KeyError, TypeError) as e:
        status.error = (
            f"Could not read the experiment progress file {path}: {e}. Fix or "
            "delete it, or use a new output directory."
        )
        return status
    for method in plan.methods:
        llm = ExperimentLogger._get_clean_llm_model_name(None, method.llm)
        for problem in plan.problems:
            for seed in plan.seeds:
                status.finished += any(
                    r.get("method_name") == method.name
                    and r.get("problem_name") == problem.name
                    and r.get("llm_name") == llm
                    and r.get("seed") == int(seed)
                    and r.get("end_time")
                    for r in runs
                )
    return status


def _identity(method, llm, problem, seed) -> tuple:
    """A run's identity: method, LLM (None if unknown), problem and seed.

    One method name may be used with several LLMs, so the LLM is part of it.
    """
    return (method, llm, problem, int(seed))


def _identity_of(run: dict, with_llm: bool = True) -> tuple:
    """The identity of a progress entry (without the LLM for old files)."""
    llm = run.get("llm_name") if with_llm else None
    return _identity(run["method_name"], llm, run["problem_name"], run["seed"])


def add_planned_runs(exp_logger, plan: "ExperimentPlan") -> int:
    """Add runs that are planned but not recorded to ``progress.json``.

    The experiment logger only resumes when the recorded runs equal the
    planned ones; adding the new runs (as not yet started entries in the
    logger's own format) first lets an experiment grow, e.g. with more
    ``runs``. Recorded runs are never changed or removed.

    Returns:
        int: The number of runs added.
    """
    progress = getattr(exp_logger, "progress", None)
    if not isinstance(progress, dict) or not hasattr(exp_logger, "_planned_runs"):
        return 0
    runs = progress.get("runs") or []
    if not runs:
        return 0  # a new experiment; the logger plans all runs itself
    with_llm = all(r.get("llm_name") for r in runs)
    recorded = {_identity_of(r, with_llm) for r in runs}
    new = [
        r
        for r in exp_logger._planned_runs(
            plan.methods, plan.problems, plan.seeds, plan.config.experiment.budget
        )
        if _identity_of(r, with_llm) not in recorded
    ]
    if not new:
        return 0
    with exp_logger._lock:
        progress["runs"] = runs + new
        progress["total"] = len(progress["runs"])
        progress["end_time"] = None
        exp_logger._write_progress()
    return len(new)


def _strip_header(text: str) -> str:
    lines = text.splitlines(keepends=True)
    while lines and lines[0].startswith("#"):
        lines.pop(0)
    return "".join(lines)
