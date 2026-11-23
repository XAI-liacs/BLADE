import ast
import importlib
import re
from typing import Any


def _collect_imports(code: str):
    """Collect import info from code using AST."""
    tree = ast.parse(code)
    imports = []

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append(
                    {"type": "import", "module": alias.name, "alias": alias.asname}
                )
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                imports.append(
                    {
                        "type": "from",
                        "module": node.module,
                        "level": node.level,
                        "name": alias.name,
                        "alias": alias.asname,
                    }
                )
    return imports


def _add_builtins_into(allowed_list):
    allowed_list += [
        "copy",
        "math",
        "heapq",
        "cmath",
        "random",
        "bisect",
        "typing",
        "pprint",
        "decimal",
        "operator",
        "functools",
        "fractions",
        "itertools",
        "statistics",
        "collections",
        "dataclasses",
    ]


def _requirement_name(requirement: str) -> str:
    """Distribution name of a requirement such as ``"numpy>=2"``.

    Version specifiers, extras, markers and direct references are removed:
    ``"pkg==1.0"``, ``"pkg<2"``, ``"pkg[extra]"``, ``"pkg; python_version>'3'"``
    and ``"pkg @ git+https://..."`` all give ``"pkg"``.
    """
    return re.split(r"[<>=!~\[; @]", requirement.strip(), maxsplit=1)[0]


def _is_allowed(module: str, allowed: list[str]) -> bool:
    return any(module == a or module.startswith(a + ".") for a in allowed)


def prepare_namespace(code: str, allowed: list[str]) -> dict[str, Any]:
    """Prepare the globals for ``exec`` with the modules that `code` imports.

    Modules that are allowed (in `allowed` or a few standard library
    modules) are imported and put in the namespace; for any other module a
    generic ``object`` placeholder is used. This only saves the generated
    code from failing on the import statements themselves: it is **not** a
    sandbox, the code still runs with full Python access. Isolation comes
    from evaluating solutions in a separate subprocess (see
    ``Problem.__call__``).

    Star imports (``from math import *``) and relative imports
    (``from . import x``) are not resolved here; they are left to the
    ``exec`` of the code itself.

    Args:
        `code: str`: Code parameter that is to be passed to `exec` function.
        `allowed: list[str]`: Requirements (e.g. ``"numpy>=2"``) whose
            modules may be imported; version specifiers are ignored.

    Returns:
        Returns a prepared global_namespace dictionary for exec, of type `dict[str, Any]`.
    """
    ns = {}
    imports = _collect_imports(code)

    allowed = [_requirement_name(a) for a in (allowed or [])]
    allowed = [a for a in allowed if a]
    _add_builtins_into(allowed)
    for imp in imports:
        if imp["type"] == "import":
            module = imp["module"]

            if allowed and not _is_allowed(module, allowed):
                ns[imp["alias"] or module.split(".")[0]] = object
            else:
                mod = importlib.import_module(module)
                ns[imp["alias"] or module.split(".")[0]] = mod

        elif imp["type"] == "from":
            module = imp["module"]
            if module is None or imp.get("level", 0) or imp["name"] == "*":
                # relative or star import: left to the exec of the code
                continue

            if allowed and not _is_allowed(module, allowed):
                ns[imp["alias"] or imp["name"]] = object
            else:
                mod = importlib.import_module(module)
                try:
                    obj = getattr(mod, imp["name"])
                except AttributeError:
                    # a submodule that the package does not import itself
                    obj = importlib.import_module(f"{module}.{imp['name']}")
                ns[imp["alias"] or imp["name"]] = obj

    return ns


def clean_local_namespace(
    local_namespace: dict[str, Any], global_namespace: dict[str, Any]
):
    """The exec command upon execution, adds global_namespace parameters to local_namespace parameters.
    This function returns local_ns - gobal_ns, so that sweeping for object type never returns a library imported objects.

    Args:
        `local_namespace : dict[str, Any]`: Dictionary that was passed as local_namespace to `exec` block.
        `global_namespace : dict[str, Any]`: Dictionary/Mapping passed as global_namespace to `exec` block.

    Returns:
        Original `local_namespace`, that is `local_namespace` - `global_namespace`.
    """
    for key in global_namespace:
        if key in local_namespace:
            local_namespace.pop(key)
    return local_namespace
