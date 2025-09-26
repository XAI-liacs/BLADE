import ast, importlib

from typing import Any


def collect_imports(code: str):
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


def prepare_namespace(code: str, allowed: list[str]):
    """Prepare exec namespace with only whitelisted imports preloaded."""
    ns = {}
    imports = collect_imports(code)

    allowed = allowed.copy()
    allowed = list(map(lambda x: x.split(">")[0], allowed))
    _add_builtins_into(allowed)
    print(allowed)
    for imp in imports:
        if imp["type"] == "import":
            module = imp["module"]

            if allowed and not any(
                module == a or module.startswith(a + ".") for a in allowed
            ):
                raise ImportError(f"Import of {module} not allowed")

            mod = importlib.import_module(module)
            ns[imp["alias"] or module.split(".")[0]] = mod

        elif imp["type"] == "from":
            module = imp["module"]

            if allowed and not any(
                module == a or module.startswith(a + ".") for a in allowed
            ):
                raise ImportError(f"Import of {module} not allowed")

            mod = importlib.import_module(module)
            obj = getattr(mod, imp["name"])
            ns[imp["alias"] or imp["name"]] = obj

    return ns


def clean_local_namespace(
    local_namespace: dict["str", Any], global_namespace: dict["str", Any]
):
    for key in global_namespace:
        if key in local_namespace:
            local_namespace.pop(key)
    return local_namespace
