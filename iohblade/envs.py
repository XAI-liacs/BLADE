"""Cached virtual environments for evaluating solutions.

Every problem evaluates solutions in a virtual environment with its own
dependencies (see :meth:`iohblade.problem.Problem._ensure_env`). Creating
such an environment and installing packages takes seconds to minutes, so
environments are cached and shared: problems with the same Python, the same
dependencies and the same ``use_system_site_packages`` setting reuse one
environment, across runs and across experiments.

* Location: ``$IOHBLADE_CACHE_DIR/envs`` if set, otherwise the user cache
  directory (``~/.cache/iohblade/envs``, ``%LOCALAPPDATA%\\iohblade\\Cache\\envs``
  on Windows).
* Turn caching off with ``IOHBLADE_ENV_CACHE=0``, ``problem.env_cache = False``
  or ``experiment.env_cache: false`` in YAML; problems then get a temporary
  environment that :meth:`~iohblade.problem.Problem.cleanup` deletes, as
  before.
* Dependencies that are local paths are never cached (their content can
  change without the requirement string changing).
* Packages are installed with ``uv pip`` when ``uv`` is available (much
  faster), falling back to ``pip``.
* ``iohblade cache list`` / ``iohblade cache clear`` show and remove cached
  environments.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

ENV_FORMAT = 1
MARKER = ".blade-env.json"


class EnvironmentSetupError(RuntimeError):
    """Creating an evaluation environment or installing its packages failed."""


@dataclass
class EnvInfo:
    """An evaluation environment.

    Attributes:
        path: Root of the virtual environment.
        python: Its Python executable.
        cached: Whether it lives in the shared cache (and must not be
            deleted by a single problem).
        reused: Whether an existing cached environment was reused.
        seconds: Time spent creating it (0 when reused).
    """

    path: Path
    python: Path
    cached: bool
    reused: bool = False
    seconds: float = 0.0


def cache_root() -> Path:
    """Directory holding the cached environments."""
    base = os.environ.get("IOHBLADE_CACHE_DIR")
    if base:
        return Path(base).expanduser() / "envs"
    if os.name == "nt":
        local = os.environ.get("LOCALAPPDATA") or str(Path.home() / "AppData/Local")
        return Path(local) / "iohblade" / "Cache" / "envs"
    xdg = os.environ.get("XDG_CACHE_HOME")
    return Path(xdg or Path.home() / ".cache") / "iohblade" / "envs"


def cache_enabled() -> bool:
    """False when ``IOHBLADE_ENV_CACHE`` is set to 0/false/no/off."""
    value = os.environ.get("IOHBLADE_ENV_CACHE", "1").strip().lower()
    return value not in ("0", "false", "no", "off")


def _is_local_path(requirement: str) -> bool:
    req = requirement.strip()
    if "file:" in req or req.startswith((".", "/", "~", "\\")):
        return True
    return os.path.exists(req)


def cacheable(dependencies: list[str]) -> bool:
    """Whether an environment with these dependencies can be cached."""
    return not any(_is_local_path(d) for d in dependencies)


def env_key(dependencies: list[str], system_site_packages: bool) -> str:
    """Cache key: Python interpreter and version, dependencies, settings."""
    data = {
        "format": ENV_FORMAT,
        "python": sys.version,
        "executable": os.path.realpath(sys.executable),
        "dependencies": sorted(d.strip() for d in dependencies),
        "system_site_packages": bool(system_site_packages),
    }
    blob = json.dumps(data, sort_keys=True).encode()
    return hashlib.sha256(blob).hexdigest()[:20]


def _python_in(env: Path) -> Path:
    return (
        env
        / ("Scripts" if os.name == "nt" else "bin")
        / ("python.exe" if os.name == "nt" else "python")
    )


def _tail(text: str, lines: int = 40) -> str:
    parts = (text or "").strip().splitlines()
    return "\n".join(parts[-lines:])


def _install(python: Path, dependencies: list[str]) -> None:
    """Install packages into an environment (uv if available, else pip)."""
    if not dependencies:
        return
    errors = []
    uv = shutil.which("uv")
    if uv and os.environ.get("IOHBLADE_USE_UV", "1") != "0":
        result = subprocess.run(
            [uv, "pip", "install", "--python", str(python), *dependencies],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            return
        errors.append(
            f"uv pip install failed:\n{_tail(result.stderr or result.stdout)}"
        )
    result = subprocess.run(
        [str(python), "-m", "pip", "install", *dependencies],
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        return
    errors.append(f"pip install failed:\n{_tail(result.stderr or result.stdout)}")
    raise EnvironmentSetupError(
        f"Failed to install the dependencies {dependencies}:\n" + "\n\n".join(errors)
    )


def _create(env: Path, dependencies: list[str], system_site_packages: bool) -> Path:
    import virtualenv

    args = ["--system-site-packages"] if system_site_packages else []
    try:
        virtualenv.cli_run([*args, str(env)])
    except BaseException as e:  # virtualenv raises SystemExit on bad input
        raise EnvironmentSetupError(
            f"Creating a virtual environment in {env} failed: {e}"
        ) from e
    python = _python_in(env)
    _install(python, dependencies)
    return python


def get_env(
    dependencies: list[str] | None = None,
    system_site_packages: bool = False,
    cache: bool = True,
) -> EnvInfo:
    """Return an evaluation environment with ``dependencies`` installed.

    Args:
        dependencies: pip requirements.
        system_site_packages: Give the environment access to the packages
            of the interpreter BLADE runs with.
        cache: Use (and fill) the shared cache. Ignored when caching is
            disabled with ``IOHBLADE_ENV_CACHE=0`` or the dependencies
            contain local paths.

    Raises:
        EnvironmentSetupError: If creating the environment or installing a
            package fails; the message contains the installer's output.
    """
    deps = list(dependencies or [])
    start = time.monotonic()
    if not (cache and cache_enabled() and cacheable(deps)):
        env = Path(tempfile.mkdtemp(prefix="blade_env_"))
        try:
            python = _create(env, deps, system_site_packages)
        except BaseException:
            shutil.rmtree(env, ignore_errors=True)
            raise
        return EnvInfo(env, python, cached=False, seconds=time.monotonic() - start)

    from filelock import FileLock

    root = cache_root()
    root.mkdir(parents=True, exist_ok=True)
    key = env_key(deps, system_site_packages)
    env = root / key
    with FileLock(str(root / f"{key}.lock")):
        marker = env / MARKER
        python = _python_in(env)
        if marker.is_file() and python.exists():
            _touch(marker)
            return EnvInfo(env, python, cached=True, reused=True)
        if env.exists():  # an earlier build was interrupted
            shutil.rmtree(env, ignore_errors=True)
        try:
            python = _create(env, deps, system_site_packages)
        except BaseException:
            shutil.rmtree(env, ignore_errors=True)
            raise
        marker.write_text(
            json.dumps(
                {
                    "dependencies": deps,
                    "system_site_packages": system_site_packages,
                    "python": sys.version.split()[0],
                    "executable": os.path.realpath(sys.executable),
                    "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    "last_used": time.strftime("%Y-%m-%dT%H:%M:%S"),
                },
                indent=2,
            )
        )
    return EnvInfo(env, python, cached=True, seconds=time.monotonic() - start)


def _touch(marker: Path) -> None:
    try:
        data = json.loads(marker.read_text())
        data["last_used"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        marker.write_text(json.dumps(data, indent=2))
    except (OSError, ValueError):
        pass


def _size(path: Path) -> int:
    total = 0
    for root, _, files in os.walk(path):
        for name in files:
            try:
                total += os.lstat(os.path.join(root, name)).st_size
            except OSError:
                pass
    return total


def list_envs() -> list[dict]:
    """Cached environments with their metadata and size in bytes."""
    root = cache_root()
    if not root.is_dir():
        return []
    envs = []
    for env in sorted(p for p in root.iterdir() if p.is_dir()):
        marker = env / MARKER
        try:
            meta = json.loads(marker.read_text()) if marker.is_file() else {}
        except ValueError:
            meta = {}
        envs.append(
            {
                "key": env.name,
                "path": str(env),
                "complete": marker.is_file(),
                "size": _size(env),
                **meta,
            }
        )
    return envs


def clear_cache() -> int:
    """Delete all cached environments. Returns the number removed."""
    root = cache_root()
    if not root.is_dir():
        return 0
    count = 0
    for entry in root.iterdir():
        if entry.is_dir():
            count += 1
            shutil.rmtree(entry, ignore_errors=True)
        else:
            try:
                entry.unlink()
            except OSError:
                pass
    return count


def iohblade_only_path() -> Path:
    """A directory containing only the ``iohblade`` package (for PYTHONPATH).

    Used when ``Problem.isolated_pythonpath`` is set: the evaluation then
    sees iohblade but no other package of the environment BLADE runs in.
    (Without it, the folder containing iohblade is used, which for a pip
    installation is the whole site-packages directory.)
    """
    package = Path(__file__).resolve().parent
    digest = hashlib.sha256(str(package).encode()).hexdigest()[:16]
    shim = cache_root().parent / "shims" / digest
    link = shim / "iohblade"
    if link.exists():
        return shim
    shim.mkdir(parents=True, exist_ok=True)
    try:
        os.symlink(package, link, target_is_directory=True)
    except (OSError, NotImplementedError):  # e.g. Windows without privileges
        shutil.copytree(
            package, link, ignore=shutil.ignore_patterns("__pycache__", "*.pyc")
        )
    return shim
