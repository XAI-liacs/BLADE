"""Load custom problems, methods and LLMs from a Python file or package.

A custom component is referenced in YAML as::

    - type: custom
      source: ./my_problem.py        # a .py file, or a package folder
      class: MyProblem               # or "submodule:MyProblem" in a package

The module is imported under its real name (the file name without ``.py``,
or the package folder name) with its parent directory appended to
``sys.path``. Importing by name, rather than from a file location, is what
lets the evaluation subprocesses unpickle the objects: they import the same
module, with the same directory on their path (see
``Problem.extra_pythonpath``).
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.util
import inspect
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType

# Files and folders that are not part of a package's source.
_IGNORED = {"__pycache__", ".git", ".venv", ".mypy_cache", ".pytest_cache"}


class UserCodeError(Exception):
    """A custom source cannot be used; the message says why and how to fix it."""


@dataclass
class LoadedSource:
    """A custom module that was imported.

    Attributes:
        path: The file or package folder given as ``source``.
        root: The directory that was added to ``sys.path``.
        module_name: The name the module is imported under.
        module: The imported module.
        sha256: Hash of the source file(s), for reproducibility.
        dependencies: Requirements found next to the source
            (``requirements.txt`` or ``pyproject.toml``), if any.
    """

    path: Path
    root: Path
    module_name: str
    module: ModuleType
    sha256: str
    dependencies: list[str] = field(default_factory=list)

    def info(self, class_name: str) -> dict[str, str]:
        """Provenance recorded with the component's logged configuration."""
        return {
            "source": str(self.path),
            "module": self.module_name,
            "class": class_name,
            "sha256": self.sha256,
        }

    def files(self) -> list[Path]:
        """The source files (one file, or all files of the package)."""
        return _source_files(self.path)

    def snapshot(self, target_dir: Path) -> Path:
        """Copy the source into ``target_dir`` (returns the copy's path)."""
        target_dir.mkdir(parents=True, exist_ok=True)
        if self.path.is_file():
            dest = target_dir / self.path.name
            shutil.copy2(self.path, dest)
            return dest
        dest = target_dir / self.path.name
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(
            self.path, dest, ignore=shutil.ignore_patterns(*_IGNORED, "*.pyc")
        )
        return dest


def _source_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    return sorted(
        p
        for p in path.rglob("*")
        if p.is_file()
        and not any(part in _IGNORED for part in p.relative_to(path).parts)
        and p.suffix != ".pyc"
    )


def hash_source(path: Path) -> str:
    """SHA-256 over the relative names and contents of the source files."""
    digest = hashlib.sha256()
    base = path.parent if path.is_file() else path
    for file in _source_files(path):
        digest.update(str(file.relative_to(base)).replace("\\", "/").encode())
        digest.update(b"\0")
        digest.update(file.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def read_requirements(path: Path) -> list[str]:
    """Requirements declared next to a source.

    For a package folder: its ``requirements.txt``, or else the
    ``[project] dependencies`` of its ``pyproject.toml``. For a single file:
    a ``requirements.txt`` in the same directory.
    """
    folder = path if path.is_dir() else path.parent
    req = folder / "requirements.txt"
    if req.is_file():
        lines = []
        for raw in req.read_text().splitlines():
            line = raw.split("#", 1)[0].strip()
            if line and not line.startswith("-"):
                lines.append(line)
        return lines
    pyproject = folder / "pyproject.toml"
    if path.is_dir() and pyproject.is_file():
        import tomllib

        try:
            data = tomllib.loads(pyproject.read_text())
        except tomllib.TOMLDecodeError as e:
            raise UserCodeError(f"Cannot parse {pyproject}: {e}") from e
        return list(data.get("project", {}).get("dependencies", []))
    return []


def _module_origin(name: str) -> Path | None:
    """Where ``name`` would be imported from now (None if not importable)."""
    module = sys.modules.get(name)
    if module is not None:
        origin = getattr(module, "__file__", None)
        return Path(origin).resolve() if origin else Path("<built-in>")
    try:
        spec = importlib.util.find_spec(name)
    except (ImportError, ValueError):
        return None
    if spec is None:
        return None
    if spec.origin in (None, "built-in", "frozen"):
        return Path(f"<{spec.origin or 'namespace'}>")
    return Path(spec.origin).resolve()


def load_source(source: str | Path, base_dir: Path) -> LoadedSource:
    """Import a custom ``.py`` file or package folder.

    Args:
        source: Path to the file or folder, relative to ``base_dir``.
        base_dir: Directory of the configuration file.

    Returns:
        LoadedSource: The imported module with its provenance.

    Raises:
        UserCodeError: If the path does not exist, is not a Python file or
            package, clashes with an installed module, or fails to import.
    """
    path = Path(source)
    if not path.is_absolute():
        path = base_dir / path
    path = path.resolve()
    if not path.exists():
        raise UserCodeError(f"'{source}' does not exist (looked for {path}).")
    if path.is_file():
        if path.suffix != ".py":
            raise UserCodeError(f"'{source}' is not a .py file or a package folder.")
        module_name = path.stem
        expected = path
    else:
        if not (path / "__init__.py").is_file():
            raise UserCodeError(
                f"'{source}' is a folder without __init__.py. Add an (empty) "
                "__init__.py to make it a package, or point 'source' at a .py file."
            )
        module_name = path.name
        expected = path / "__init__.py"
    if not module_name.isidentifier():
        raise UserCodeError(
            f"'{module_name}' (from '{source}') is not a valid Python module name; "
            "rename it to use only letters, digits and underscores."
        )
    root = path.parent

    origin = _module_origin(module_name)
    if origin is not None and origin != expected:
        raise UserCodeError(
            f"The module name '{module_name}' (from '{source}') is already taken "
            f"by {origin}. Rename the {'file' if path.is_file() else 'folder'} "
            "so it does not clash."
        )
    if str(root) not in sys.path:
        # Appended, so the folder cannot shadow installed packages.
        sys.path.append(str(root))
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as e:
        missing = e.name or str(e)
        if missing.split(".")[0] == module_name:
            raise UserCodeError(f"Importing '{source}' failed: {e}") from e
        raise UserCodeError(
            f"'{source}' imports '{missing}', which is not installed where BLADE "
            f"runs. Install it (e.g. `uv add {missing.split('.')[0]}` or "
            f"`uv pip install {missing.split('.')[0]}`)."
        ) from e
    except SyntaxError as e:
        raise UserCodeError(
            f"Syntax error in {e.filename}, line {e.lineno}: {e.msg}"
        ) from e
    except Exception as e:
        raise UserCodeError(
            f"Importing '{source}' failed: {type(e).__name__}: {e}"
        ) from e
    return LoadedSource(
        path=path,
        root=root,
        module_name=module_name,
        module=module,
        sha256=hash_source(path),
        dependencies=read_requirements(path),
    )


def get_class(loaded: LoadedSource, class_name: str, base: type, what: str) -> type:
    """Look up ``class_name`` in a loaded source and check it.

    Args:
        loaded: The imported source.
        class_name: ``"Name"``, or ``"submodule:Name"`` for a class in a
            submodule of a package.
        base: Required base class (Problem, Method or LLM).
        what: Description for messages, e.g. ``"problem"``.

    Raises:
        UserCodeError: If the class is missing, has the wrong base class or
            leaves abstract methods unimplemented.
    """
    module = loaded.module
    attr = class_name
    if ":" in class_name:
        sub, _, attr = class_name.partition(":")
        full = f"{loaded.module_name}.{sub}"
        try:
            module = importlib.import_module(full)
        except ModuleNotFoundError as e:
            raise UserCodeError(
                f"Submodule '{sub}' not found in '{loaded.path}': {e}"
            ) from e
    obj = getattr(module, attr, None)
    if obj is None:
        candidates = [
            name
            for name, value in vars(module).items()
            if inspect.isclass(value)
            and issubclass(value, base)
            and value is not base
            and value.__module__ == module.__name__
        ]
        hint = (
            " Classes there: " + ", ".join(sorted(candidates)) + "."
            if candidates
            else ""
        )
        raise UserCodeError(f"No class '{attr}' in {module.__name__}.{hint}")
    if not inspect.isclass(obj):
        raise UserCodeError(f"'{class_name}' is not a class.")
    if not issubclass(obj, base):
        raise UserCodeError(
            f"'{class_name}' must be a subclass of "
            f"{base.__module__}.{base.__name__} to be used as a {what}."
        )
    abstract = sorted(getattr(obj, "__abstractmethods__", ()))
    if abstract:
        raise UserCodeError(
            f"'{class_name}' does not implement the abstract method(s) "
            + ", ".join(abstract)
            + f" of {base.__name__}."
        )
    return obj
