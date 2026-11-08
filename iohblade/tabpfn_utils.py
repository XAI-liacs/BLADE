"""Checks if TabPFN is installed and which arguments it accepts."""

from __future__ import annotations

import inspect
from importlib import metadata
from typing import Any

VALID_TABPFN_MODES = ("off", "passive", "explicit")


def _format_import_error(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {exc}"


def _tabpfn_version(tabpfn_module: Any) -> str | None:
    version = getattr(tabpfn_module, "__version__", None)
    if version:
        return str(version)

    try:
        return metadata.version("tabpfn")
    except Exception:
        return None


def _required_estimators(task_type: str) -> list[str]:
    normalized = (task_type or "classification").lower()
    if normalized in {"both", "all"}:
        return ["TabPFNClassifier", "TabPFNRegressor"]
    if "regress" in normalized:
        return ["TabPFNRegressor"]
    return ["TabPFNClassifier"]


def _constructor_info(cls: Any) -> dict[str, Any]:
    try:
        signature = inspect.signature(cls.__init__)
    except Exception as exc:
        return {
            "signature": None,
            "kwargs": None,
            "error": _format_import_error(exc),
        }

    kwargs = []
    for name, param in signature.parameters.items():
        if name == "self":
            continue
        if param.kind in {
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        }:
            kwargs.append(name)

    return {
        "signature": f"{cls.__name__}{signature}",
        "kwargs": kwargs,
        "error": None,
    }


def detect_tabpfn(task_type: str = "classification") -> dict[str, Any]:
    """Detect TabPFN and the estimator class needed for a task type.

    Classification checks require ``TabPFNClassifier``. Regression checks also
    try ``TabPFNRegressor``. The returned ``available`` flag means all required
    imports for the requested task type succeeded.
    """

    required = _required_estimators(task_type)
    info: dict[str, Any] = {
        "available": False,
        "version": None,
        "error": None,
        "package_available": False,
        "classifier_available": None,
        "classifier_error": None,
        "classifier_signature": None,
        "classifier_kwargs": None,
        "regressor_available": None,
        "regressor_error": None,
        "regressor_signature": None,
        "regressor_kwargs": None,
        "required_estimators": required,
    }

    try:
        import tabpfn
    except Exception as exc:
        info["error"] = _format_import_error(exc)
        return info

    info["package_available"] = True
    info["version"] = _tabpfn_version(tabpfn)

    try:
        from tabpfn import TabPFNClassifier  # noqa: F401

        info["classifier_available"] = True
        constructor = _constructor_info(TabPFNClassifier)
        info["classifier_signature"] = constructor["signature"]
        info["classifier_kwargs"] = constructor["kwargs"]
    except Exception as exc:
        info["classifier_available"] = False
        info["classifier_error"] = _format_import_error(exc)

    if "TabPFNRegressor" in required:
        try:
            from tabpfn import TabPFNRegressor  # noqa: F401

            info["regressor_available"] = True
            constructor = _constructor_info(TabPFNRegressor)
            info["regressor_signature"] = constructor["signature"]
            info["regressor_kwargs"] = constructor["kwargs"]
        except Exception as exc:
            info["regressor_available"] = False
            info["regressor_error"] = _format_import_error(exc)

    missing_errors = []
    if "TabPFNClassifier" in required and info["classifier_available"] is not True:
        missing_errors.append(
            "TabPFNClassifier import failed: "
            f"{info['classifier_error'] or 'unknown error'}"
        )
    if "TabPFNRegressor" in required and info["regressor_available"] is not True:
        missing_errors.append(
            "TabPFNRegressor import failed: "
            f"{info['regressor_error'] or 'unknown error'}"
        )

    info["available"] = not missing_errors
    if missing_errors:
        info["error"] = "; ".join(missing_errors)
    return info
