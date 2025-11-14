import ast, importlib

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
                        "name": alias.name,
                        "alias": alias.asname,
                    }
                )
    return imports


def _add_builtins_into(allowed_list):
    allowed_list += ["math", "random", "statistics", "itertools", "operator", "heapq"]


def prepare_namespace(code: str, allowed: list[str]) -> dict[str, Any]:
    """Prepare exec global_namespace, with the libraries imported in the text, `code` parameter accepts.
        If the imports are not allowed in the environment, a generic object is provided.

    Args:
        `code: str`: Code parameter that is to be passed to `exec` function.
        `allowed: list[str]`: A list of allowed pip installable libraries, that are acceptable to be imported.

    Returns:
        Returns a prepared global_namespace dictionary for exec, of type `dict[str, Any]`.
    """
    ns = {}
    imports = _collect_imports(code)

    allowed = allowed.copy()
    allowed = list(map(lambda x: x.split(">")[0], allowed))
    _add_builtins_into(allowed)
    for imp in imports:
        if imp["type"] == "import":
            module = imp["module"]

            if allowed and not any(
                module == a or module.startswith(a + ".") for a in allowed
            ):
                ns[imp["alias"] or module.split(".")[0]] = object
            else:
                mod = importlib.import_module(module)
                ns[imp["alias"] or module.split(".")[0]] = mod

        elif imp["type"] == "from":
            module = imp["module"]

            if allowed and not any(
                module == a or module.startswith(a + ".") for a in allowed
            ):
                ns[imp["alias"] or imp["name"]] = object
            else:
                mod = importlib.import_module(module)
                obj = getattr(mod, imp["name"])
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
