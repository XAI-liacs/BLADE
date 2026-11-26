import inspect

import pytest

from iohblade.method import Method
from iohblade.problem import Problem
from iohblade.registry import (
    BuiltinMethod,
    BuiltinProblem,
    BuiltinSuite,
    ComponentKind,
    LLMProvider,
    LoggerType,
    Registry,
    register_problem,
    registry,
)
from iohblade.utils import ComponentNotFoundError, ComponentUnavailableError

ENUMS = {
    ComponentKind.PROBLEM: BuiltinProblem,
    ComponentKind.SUITE: BuiltinSuite,
    ComponentKind.METHOD: BuiltinMethod,
    ComponentKind.LLM: LLMProvider,
    ComponentKind.LOGGER: LoggerType,
}


@pytest.mark.parametrize("kind", list(ENUMS))
def test_enum_and_registry_agree(kind):
    assert {m.value for m in ENUMS[kind]} <= set(registry.keys(kind))


def _builtin_entries():
    for kind, enum in ENUMS.items():
        for member in enum:
            yield kind, member


@pytest.mark.parametrize("kind,key", list(_builtin_entries()), ids=lambda v: str(v))
def test_builtin_resolves_or_reports_missing_dependency(kind, key):
    info = registry.get(kind, key)
    try:
        target = info.load()
    except ComponentUnavailableError as e:
        # Only components behind an optional dependency group may be missing,
        # and the error must say how to install them.
        assert info.install_hint(), f"{key} has no install hint: {e}"
        assert info.install_hint() in str(e)
        return
    assert target is not None
    if kind == ComponentKind.PROBLEM:
        assert inspect.isclass(target) and issubclass(target, Problem)
    elif kind == ComponentKind.METHOD and inspect.isclass(target):
        assert issubclass(target, Method)
    elif kind == ComponentKind.SUITE:
        assert callable(target)


def test_resolve_matches_existing_import_paths():
    from iohblade.benchmarks import BBOB_SBOX
    from iohblade.llm import Ollama_LLM
    from iohblade.loggers import ExperimentLogger
    from iohblade.methods import LLaMEA

    assert registry.resolve("problem", "bbob_sbox") is BBOB_SBOX
    assert registry.resolve(ComponentKind.METHOD, BuiltinMethod.LLAMEA) is LLaMEA
    assert registry.resolve("llm", LLMProvider.OLLAMA) is Ollama_LLM
    assert registry.resolve("logger", "file") is ExperimentLogger


def test_unknown_key_suggests_close_match():
    with pytest.raises(ComponentNotFoundError) as exc:
        registry.get("problem", "bbob_sbx")
    assert "bbob_sbox" in exc.value.suggestions
    assert "Did you mean 'bbob_sbox'" in str(exc.value)
    assert isinstance(exc.value, KeyError)  # dict-like lookups keep working


def test_hyphens_are_accepted_for_underscores():
    assert registry.get("problem", "bbob-sbox") is registry.get("problem", "bbob_sbox")
    assert ("suite", "continuous-bbo") in registry


def test_unknown_key_without_match_lists_known():
    reg = Registry()
    reg.register("method", "alpha", "math:sqrt")
    with pytest.raises(ComponentNotFoundError, match="Known: alpha"):
        reg.get("method", "zzzzzz")


def test_unknown_kind_is_rejected():
    with pytest.raises(ValueError):
        registry.get("benchmark", "bbob_sbox")


def test_unavailable_component_mentions_install_hint():
    reg = Registry()
    reg.register("method", "ghost", "not_a_real_module_xyz:Thing", extra="methods")
    assert not reg.get("method", "ghost").is_available()
    with pytest.raises(ComponentUnavailableError) as exc:
        reg.resolve("method", "ghost")
    assert "uv sync --group methods" in str(exc.value)
    assert isinstance(exc.value, ImportError)


def test_requires_install_hint():
    reg = Registry()
    reg.register("problem", "ghost", "not_a_real_module_xyz:P", requires=["a", "b"])
    with pytest.raises(ComponentUnavailableError, match="uv pip install a b"):
        reg.resolve("problem", "ghost")


def test_modules_are_checked():
    reg = Registry()
    reg.register(
        "logger", "ghost", "math:sqrt", modules=["not_a_module_xyz"], extra="g"
    )
    with pytest.raises(ComponentUnavailableError, match="not_a_module_xyz"):
        reg.resolve("logger", "ghost")
    assert not reg.get("logger", "ghost").is_available()
    reg.register("logger", "fine", "math:sqrt", modules=["json"])
    assert reg.resolve("logger", "fine").__name__ == "sqrt"


def test_missing_attribute_is_unavailable():
    reg = Registry()
    reg.register("method", "ghost", "math:not_there")
    with pytest.raises(ComponentUnavailableError, match="not_there"):
        reg.resolve("method", "ghost")


def test_register_validation_and_overwrite():
    reg = Registry()
    with pytest.raises(ValueError, match="module:attribute"):
        reg.register("problem", "x", "no_colon_here")
    reg.register("problem", "x", "math:sqrt")
    reg.register("problem", "x", "math:sqrt")  # same target: no error
    with pytest.raises(ValueError, match="already registered"):
        reg.register("problem", "x", "math:floor")
    reg.register("problem", "x", "math:floor", overwrite=True)
    assert reg.resolve("problem", "x").__name__ == "floor"
    assert ("problem", "x") in reg
    reg.unregister("problem", "x")
    assert ("problem", "x") not in reg


def test_decorator_registers_object():
    @register_problem("test_decorated_problem")
    class Decorated:
        """A decorated test problem."""

    try:
        info = registry.get("problem", "test_decorated_problem")
        assert info.load() is Decorated
        assert info.description == "A decorated test problem."
        assert info.target_path.endswith(
            ":test_decorator_registers_object.<locals>.Decorated"
        )
    finally:
        registry.unregister("problem", "test_decorated_problem")


def test_entries_are_sorted_and_described():
    entries = registry.entries("llm")
    assert [e.key for e in entries] == sorted(e.key for e in entries)
    assert all(e.description for e in entries)


def test_enum_members_are_strings():
    assert BuiltinProblem.BBOB_SBOX == "bbob_sbox"
    assert f"{LLMProvider.GEMINI}" == "gemini"
