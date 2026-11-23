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
* Dependencies that are local paths, or pip options such as ``-e ./pkg``,
  are never cached (their content can change without the requirement
  string changing).
* Packages are installed with ``uv pip`` when ``uv`` is available (much
  faster), falling back to ``pip``.
* ``iohblade cache list`` / ``iohblade cache clear`` show and remove cached
  environments.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
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
    """Creating an evaluation environment or installing its packages failed.

    Attributes:
        full_output: The complete installer output (the message only holds
            its last lines), or None.
    """

    def __init__(self, message: str, full_output: str | None = None):
        super().__init__(message)
        self.full_output = full_output


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
    # pip options (-e/--editable, -r, --find-links, ...) can refer to local
    # paths, relative to wherever the experiment runs: never cache them.
    if req.startswith("-"):
        return True
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


# Packages that provide build tools other packages need while they are
# built from source (e.g. pyrfr, a dependency of smac, needs swig). They are
# installed first, and the environment's scripts folder is put on PATH for
# the rest of the installation so the build backends find them.
BUILD_TOOLS = ("swig", "cmake", "ninja")


def _requirement_name(requirement: str) -> str:
    name = re.split(r"[\s<>=!~;\[@]", requirement.strip(), maxsplit=1)[0]
    return name.lower().replace("_", "-")


def _install(python: Path, dependencies: list[str]) -> None:
    """Install packages into an environment (uv if available, else pip).

    Build tools in :data:`BUILD_TOOLS` are installed first; the
    environment's scripts folder is on ``PATH`` while installing.
    """
    if not dependencies:
        return
    tools = [d for d in dependencies if _requirement_name(d) in BUILD_TOOLS]
    rest = [d for d in dependencies if d not in tools]
    env = os.environ.copy()
    env["PATH"] = str(python.parent) + os.pathsep + env.get("PATH", "")
    for group in (tools, rest):
        if group:
            _install_group(python, group, env, dependencies)


def _install_group(python: Path, group: list[str], env: dict, everything) -> None:
    errors = []
    full = []
    uv = shutil.which("uv")
    if uv and os.environ.get("IOHBLADE_USE_UV", "1") != "0":
        result = subprocess.run(
            [uv, "pip", "install", "--python", str(python), *group],
            capture_output=True,
            text=True,
            env=env,
        )
        if result.returncode == 0:
            return
        output = (result.stdout or "") + (result.stderr or "")
        errors.append(
            f"uv pip install failed:\n{_tail(result.stderr or result.stdout)}"
        )
        full.append(f"$ uv pip install {' '.join(group)}\n{output}")
    result = subprocess.run(
        [str(python), "-m", "pip", "install", *group],
        capture_output=True,
        text=True,
        env=env,
    )
    if result.returncode == 0:
        return
    output = (result.stdout or "") + (result.stderr or "")
    errors.append(f"pip install failed:\n{_tail(result.stderr or result.stdout)}")
    full.append(f"$ pip install {' '.join(group)}\n{output}")
    raise EnvironmentSetupError(
        f"Failed to install the dependencies {everything}:\n" + "\n\n".join(errors),
        full_output="\n\n".join(full),
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


def _points_to(link: Path, target: Path) -> bool:
    """Whether ``link`` is a symlink that resolves to ``target``."""
    try:
        return link.is_symlink() and link.resolve() == target.resolve()
    except OSError:
        return False


def _package_stamp(package: Path) -> str:
    """Identifies the installed iohblade: its version and its files' mtimes.

    Used to refresh the copy of the package (see :func:`iohblade_only_path`)
    after iohblade was upgraded or edited at the same path.
    """
    try:
        from importlib.metadata import version

        installed = version("iohblade")
    except Exception:
        installed = "unknown"
    digest = hashlib.sha256(installed.encode())
    for path in sorted(package.rglob("*")):
        if "__pycache__" in path.parts or path.suffix == ".pyc":
            continue
        try:
            info = path.stat()
        except OSError:
            continue
        rel = path.relative_to(package).as_posix()
        digest.update(f"{rel}:{info.st_mtime_ns}:{info.st_size}\n".encode())
    return digest.hexdigest()


def _read_text(path: Path) -> str | None:
    try:
        return path.read_text()
    except OSError:
        return None


def iohblade_only_path() -> Path:
    """A directory containing only the ``iohblade`` package (for PYTHONPATH).

    Used when ``Problem.isolated_pythonpath`` is set: the evaluation then
    sees iohblade but no other package of the environment BLADE runs in.
    (Without it, the folder containing iohblade is used, which for a pip
    installation is the whole site-packages directory.)

    The directory holds a symlink to the package, or a copy of it where
    symlinks cannot be created (e.g. Windows without privileges). It is
    created under a file lock, as evaluations call this from parallel
    threads and processes. A copy is refreshed when iohblade changes (a
    stamp of its version and file modification times is kept next to it).
    """
    package = Path(__file__).resolve().parent
    digest = hashlib.sha256(str(package).encode()).hexdigest()[:16]
    shims = cache_root().parent / "shims"
    shim = shims / digest
    link = shim / "iohblade"
    if _points_to(link, package):
        return shim  # fast path: no lock, no file walk
    stamp_file = shims / f"{digest}.stamp"

    from filelock import FileLock

    shim.mkdir(parents=True, exist_ok=True)
    with FileLock(str(shims / f"{digest}.lock")):
        if _points_to(link, package):
            return shim
        stamp = None
        if link.is_dir() and not link.is_symlink():
            stamp = _package_stamp(package)
            if _read_text(stamp_file) == stamp:
                return shim  # an up-to-date copy
        # Missing, stale copy or a link to something else: (re)create.
        if link.is_symlink() or link.is_file():
            link.unlink()
        elif link.exists():
            shutil.rmtree(link)
        try:
            os.symlink(package, link, target_is_directory=True)
            return shim
        except FileExistsError:
            # Created concurrently by a process that does not take the
            # lock (an older iohblade): fine if it is the right link.
            if _points_to(link, package):
                return shim
            if link.is_symlink():
                link.unlink()
            else:
                shutil.rmtree(link, ignore_errors=True)
        except (OSError, NotImplementedError):  # e.g. Windows without privileges
            pass
        if stamp is None:
            stamp = _package_stamp(package)
        tmp = shims / f"{digest}.tmp-{os.getpid()}"
        shutil.rmtree(tmp, ignore_errors=True)
        shutil.copytree(
            package, tmp, ignore=shutil.ignore_patterns("__pycache__", "*.pyc")
        )
        os.replace(tmp, link)
        stamp_file.write_text(stamp)
    return shim
