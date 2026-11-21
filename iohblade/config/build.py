"""Turn a YAML configuration into LLM, method, problem and experiment objects.

All problems found while building are collected and raised together as one
:class:`~iohblade.config.errors.ConfigError`, each with the line in the YAML
file it refers to.
"""

from __future__ import annotations

import difflib
import enum
import inspect
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Union, get_args, get_origin

from pydantic import BaseModel, ValidationError

from ..registry import ComponentKind, registry
from ..utils import ComponentNotFoundError, ComponentUnavailableError
from .errors import ConfigError, ConfigIssue, Path_
from .schema import ExperimentConfig, LLMSpec
from .user_code import LoadedSource, UserCodeError, get_class, load_source
from .yaml_loader import LoadedConfig, load_yaml

# Methods that keep **kwargs and pass them on later, when the run starts.
# Their keyword arguments are checked against the real target up front, so
# typos are reported before an experiment starts rather than mid-run.
# key -> (import path of the target, parameters BLADE sets itself)
_KWARGS_TARGETS: dict[tuple[str, str], tuple[str, set[str]]] = {
    ("method", "llamea"): (
        "llamea:LLaMEA",
        {
            "f",
            "llm",
            "role_prompt",
            "task_prompt",
            "example_prompt",
            "output_format_prompt",
            "log",
            "budget",
            "max_workers",
            "minimization",
        },
    ),
}


def _llamea_operators(value, path, params, issue):
    """Turn LLaMEA ``operators`` from YAML into ``llamea.Operator`` objects.

    Each item is a mapping with ``prompt`` and optionally ``name``,
    ``weight`` and ``number_of_parents`` (the fields of ``llamea.Operator``);
    a plain string is accepted too and becomes a mutation operator.
    LLaMEA's own checks (``number_of_parents <= n_parents``, ``weight >= 0``)
    are done here, so they are reported before the run starts.
    """
    import dataclasses

    from llamea import LLaMEA as LLAMEA_Algorithm
    from llamea import Operator

    fields = {f.name: f for f in dataclasses.fields(Operator)}
    n_parents = params.get(
        "n_parents",
        inspect.signature(LLAMEA_Algorithm).parameters["n_parents"].default,
    )
    items = value if isinstance(value, list) else [value]
    if not items:
        issue(path, "'operators' is empty; leave it out to use the default mutation.")
        return None
    result, ok = [], True
    for i, item in enumerate(items):
        p = path + (i,) if isinstance(value, list) else path
        if isinstance(item, str):
            result.append(item)
            continue
        if not isinstance(item, dict):
            issue(
                p,
                "An operator is a mapping with 'prompt' and optionally 'name', "
                f"'weight' and 'number_of_parents' (got {item!r}).",
            )
            ok = False
            continue
        unknown = [k for k in item if k not in fields]
        for key in unknown:
            issue(
                p + (key,),
                f"Unknown operator field '{key}'."
                + (_did_you_mean(key, fields) or " Fields: " + ", ".join(fields) + "."),
            )
        if "prompt" not in item:
            issue(p, "An operator needs a 'prompt'.")
        weight = item.get("weight", fields["weight"].default)
        if (
            not isinstance(weight, (int, float))
            or isinstance(weight, bool)
            or weight < 0
        ):
            issue(p + ("weight",), f"'weight' must be a number >= 0 (got {weight!r}).")
            weight = None
        parents = item.get("number_of_parents", fields["number_of_parents"].default)
        if not isinstance(parents, int) or isinstance(parents, bool) or parents < 1:
            issue(
                p + ("number_of_parents",),
                f"'number_of_parents' must be an integer >= 1 (got {parents!r}).",
            )
            parents = None
        elif isinstance(n_parents, int) and parents > n_parents:
            issue(
                p + ("number_of_parents",),
                f"An operator cannot use more parents ({parents}) than "
                f"n_parents ({n_parents}).",
            )
            parents = None
        if unknown or "prompt" not in item or weight is None or parents is None:
            ok = False
            continue
        result.append(Operator(**item))
    return result if ok else None


# Converters for parameters that need Python objects instead of plain YAML
# values: (kind, key) -> {param: converter(value, path, params, issue)}.
_PARAM_CONVERTERS: dict[tuple[str, str], dict[str, Any]] = {
    ("method", "llamea"): {"operators": _llamea_operators},
}


@dataclass
class ExperimentPlan:
    """Everything built from a configuration, before anything runs.

    Building a plan has no side effects on disk; :meth:`make_experiment`
    creates the logger (and with it the output directory).

    Attributes:
        config: The validated configuration.
        source: The loaded YAML data with line information.
        output_dir: Absolute results directory.
        llms: LLM objects by configuration name.
        methods: One method object per (method entry, LLM).
        method_llm_names: For each method, the configuration name of its LLM.
        problems: Problem objects (suites expanded).
        seeds: Seeds of the runs.
        sources: Custom code loaded for this experiment.
    """

    config: ExperimentConfig
    source: LoadedConfig
    output_dir: Path
    llms: dict[str, Any] = field(default_factory=dict)
    methods: list[Any] = field(default_factory=list)
    method_llm_names: list[str] = field(default_factory=list)
    problems: list[Any] = field(default_factory=list)
    seeds: list[int] = field(default_factory=list)
    sources: list[LoadedSource] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def n_runs(self) -> int:
        """Total number of runs (methods x problems x seeds)."""
        return len(self.methods) * len(self.problems) * len(self.seeds)

    def make_logger(self):
        """Create the experiment logger (creates/reuses ``output_dir``)."""
        spec = self.config.experiment.logger
        cls = registry.resolve(ComponentKind.LOGGER, spec.type)
        return cls(name=str(self.output_dir), **spec.params)

    def make_experiment(self, exp_logger=None):
        """Create the :class:`~iohblade.experiment.Experiment` to run.

        Also records the resolved configuration and a copy of the custom
        code in ``<output_dir>/blade_config/`` (see :mod:`.record`).
        """
        from ..experiment import Experiment
        from .record import write_record

        section = self.config.experiment
        if exp_logger is None:
            exp_logger = self.make_logger()
        self.warnings = write_record(self, exp_logger.dirname)
        for warning in self.warnings:
            print(f"Warning: {warning}", file=sys.stderr)
        return Experiment(
            methods=self.methods,
            problems=self.problems,
            runs=len(self.seeds),
            budget=section.budget,
            seeds=list(self.seeds),
            show_stdout=section.show_stdout,
            log_stdout=section.log_stdout,
            exp_logger=exp_logger,
            n_jobs=section.n_jobs,
        )

    def summary(self) -> str:
        """Human readable overview of the planned runs."""
        lines = [
            f"Output directory: {self.output_dir}",
            f"Methods ({len(self.methods)}):",
        ]
        for method, llm_name in zip(self.methods, self.method_llm_names):
            lines.append(
                f"  - {method.name} (LLM: {llm_name}, budget: {method.budget})"
            )
        lines.append(f"Problems ({len(self.problems)}):")
        lines += [f"  - {p.name}" for p in self.problems]
        if self.sources:
            lines.append(f"Custom code ({len(self.sources)}):")
            lines += [
                f"  - {s.module_name} ({s.path}, sha256 {s.sha256[:12]})"
                for s in self.sources
            ]
        lines.append(f"Seeds: {list(self.seeds)}")
        lines.append(
            f"Runs: {len(self.methods)} methods x {len(self.problems)} problems"
            f" x {len(self.seeds)} seeds = {self.n_runs}"
        )
        return "\n".join(lines)


# ---------------------------------------------------------------- schema ---


def _unwrap(tp):
    """Strip Optional/Union down to a model, dict or list type if present."""
    origin = get_origin(tp)
    if origin is Union or (origin is not None and origin.__name__ == "UnionType"):
        for arg in get_args(tp):
            inner = _unwrap(arg)
            if (isinstance(inner, type) and issubclass(inner, BaseModel)) or get_origin(
                inner
            ) in (dict, list):
                return inner
        return None
    return tp


def _fields_at(loc: tuple) -> list[str]:
    """Field names allowed at ``loc`` in :class:`ExperimentConfig`."""
    tp = ExperimentConfig
    for part in loc:
        if isinstance(tp, type) and issubclass(tp, BaseModel):
            info = None
            if isinstance(part, str):
                info = tp.model_fields.get(part) or next(
                    (f for f in tp.model_fields.values() if f.alias == part), None
                )
            tp = _unwrap(info.annotation) if info else None
        elif get_origin(tp) is dict:
            tp = _unwrap(get_args(tp)[1])
        elif get_origin(tp) is list:
            tp = _unwrap(get_args(tp)[0])
        else:
            return []
        if tp is None:
            return []
    if isinstance(tp, type) and issubclass(tp, BaseModel):
        return [info.alias or name for name, info in tp.model_fields.items()]
    return []


def _did_you_mean(value: str, options) -> str:
    matches = difflib.get_close_matches(str(value), list(options), n=3, cutoff=0.6)
    if not matches:
        return ""
    return " Did you mean " + ", ".join(f"'{m}'" for m in matches) + "?"


def validate_config(source: LoadedConfig) -> ExperimentConfig:
    """Validate raw configuration data against the schema.

    Raises:
        ConfigError: With one issue per schema violation.
    """
    if not isinstance(source.data, dict):
        raise ConfigError(
            [source.issue((), "The configuration must be a mapping (key: value).")]
        )
    try:
        return ExperimentConfig.model_validate(source.data)
    except ValidationError as e:
        issues, seen = [], set()
        for err in e.errors():
            loc = tuple(err["loc"])
            if err["type"] == "extra_forbidden":
                allowed = _fields_at(loc[:-1])
                msg = f"Unknown field '{loc[-1]}'." + _did_you_mean(loc[-1], allowed)
                if allowed and not _did_you_mean(loc[-1], allowed):
                    msg += " Allowed: " + ", ".join(allowed) + "."
            elif err["type"] == "missing":
                msg = f"Missing required field '{loc[-1]}'."
            else:
                msg = err["msg"].removeprefix("Value error, ")
                if err["type"] not in ("value_error",) and "input" in err:
                    msg += f" (got {err['input']!r})"
            # Union errors report the same location once per member type.
            clean = tuple(p for p in loc if not (isinstance(p, str) and "[" in p))
            clean = tuple(p for p in clean if p not in ("str", "int"))
            key = (clean, msg)
            if key in seen:
                continue
            seen.add(key)
            issues.append(source.issue(clean, msg))
        raise ConfigError(issues) from None


# ------------------------------------------------------------- building ---


def _coerce_enum(param: inspect.Parameter, value):
    """Turn a string into the Enum a parameter expects, if it expects one."""
    if not isinstance(value, str):
        return value
    enum_cls = None
    if isinstance(param.default, enum.Enum):
        enum_cls = type(param.default)
    elif isinstance(param.annotation, type) and issubclass(param.annotation, enum.Enum):
        enum_cls = param.annotation
    if enum_cls is None:
        return value
    try:
        return enum_cls(value)
    except ValueError:
        pass
    try:
        return enum_cls[value]
    except KeyError:
        options = [str(m.value) for m in enum_cls]
        raise ValueError(
            f"'{value}' is not a valid {enum_cls.__name__}; choose from "
            + ", ".join(options)
        ) from None


class _Builder:
    def __init__(self, source: LoadedConfig, config: ExperimentConfig):
        self.source = source
        self.config = config
        self.issues: list[ConfigIssue] = []
        self.sources: dict[Path, LoadedSource] = {}

    def issue(self, path: Path_, message: str) -> None:
        self.issues.append(self.source.issue(path, message))

    def resolve(self, kind: str, key: str, path: Path_):
        try:
            return registry.resolve(kind, key)
        except ComponentNotFoundError as e:
            self.issue(path, str(e))
        except ComponentUnavailableError as e:
            self.issue(path, str(e))
        return None

    def resolve_spec(self, kind: str, key: str, spec, path: Path_, field: str):
        """Resolve a registry key, or load the class of a 'custom' component.

        Returns:
            tuple: ``(class or None, LoadedSource or None)``.
        """
        if key != "custom":
            return self.resolve(kind, key, path + (field,)), None
        from ..llm import LLM
        from ..method import Method
        from ..problem import Problem

        base = {"problem": Problem, "method": Method, "llm": LLM}[kind]
        try:
            loaded = self._load(spec.source)
        except UserCodeError as e:
            self.issue(path + ("source",), str(e))
            return None, None
        try:
            cls = get_class(loaded, spec.class_, base, kind)
        except UserCodeError as e:
            self.issue(path + ("class",), str(e))
            return None, None
        return cls, loaded

    def _load(self, source: str) -> LoadedSource:
        base_dir = self.source.base_dir
        key = Path(source) if Path(source).is_absolute() else base_dir / source
        key = key.resolve()
        if key not in self.sources:
            self.sources[key] = load_source(source, base_dir)
        return self.sources[key]

    def check_kwargs(
        self,
        target,
        kwargs: dict,
        path: Path_,
        *,
        provided: dict | None = None,
        kwargs_target: tuple[str, set[str]] | None = None,
        what: str = "",
    ) -> dict | None:
        """Check ``kwargs`` against ``target``'s signature.

        Args:
            target: Class or function to call.
            kwargs: Arguments from ``params``.
            path: Configuration path of ``params``.
            provided: Arguments BLADE passes itself (llm, budget, ...).
            kwargs_target: Where ``**kwargs`` end up, see _KWARGS_TARGETS.
            what: Component description for messages.

        Returns:
            dict | None: The (enum coerced) keyword arguments, or None if
            there were issues.
        """
        provided = provided or {}
        try:
            sig = inspect.signature(target)
        except (TypeError, ValueError):
            return {**kwargs, **provided}
        params = {
            n: p
            for n, p in sig.parameters.items()
            if p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD)
        }
        var_kw = any(p.kind == p.VAR_KEYWORD for p in sig.parameters.values())
        extra_names: set[str] | None = None
        owned: set[str] = set()
        if kwargs_target is not None:
            try:
                module_name, _, attr = kwargs_target[0].partition(":")
                obj = __import__(module_name, fromlist=[attr])
                for part in attr.split("."):
                    obj = getattr(obj, part)
                extra_names = set(inspect.signature(obj).parameters) - {"self"}
                owned = kwargs_target[1]
            except Exception:  # target missing: fall back to no checking
                extra_names = None
        ok = True
        result = {}
        for key in provided:
            if key not in params and not var_kw:
                self.issue(
                    (
                        path[:-1] + (key,)
                        if key in ("model", "api_key", "llms")
                        else path[:-1]
                    ),
                    f"{what} does not take '{key}'."
                    + (" Accepted: " + ", ".join(params) + "." if params else ""),
                )
                ok = False
        for key, value in kwargs.items():
            p = path + (key,)
            if key in provided:
                self.issue(
                    p,
                    f"'{key}' is set by BLADE for {what}; do not set it in "
                    "'params'"
                    + (
                        " (use the field next to 'params')." if key == "budget" else "."
                    ),
                )
                ok = False
                continue
            if key in params:
                try:
                    result[key] = _coerce_enum(params[key], value)
                except ValueError as e:
                    self.issue(p, str(e))
                    ok = False
                continue
            if not var_kw:
                self.issue(
                    p,
                    f"Unknown parameter '{key}' for {what}."
                    + (
                        _did_you_mean(key, params)
                        or " Accepted: " + ", ".join(params) + "."
                    ),
                )
                ok = False
                continue
            if extra_names is not None:
                if key in owned:
                    self.issue(p, f"'{key}' is set by BLADE for {what}.")
                    ok = False
                    continue
                if key not in extra_names:
                    self.issue(
                        p,
                        f"Unknown parameter '{key}' for {what}."
                        + _did_you_mean(key, (set(params) | extra_names) - owned),
                    )
                    ok = False
                    continue
            result[key] = value
        for name, p in params.items():
            if (
                p.default is p.empty
                and name not in result
                and name not in provided
                and name not in kwargs
            ):
                hint = ""
                if name == "api_key":
                    hint = " Use api_key: ${env:YOUR_KEY_VARIABLE}."
                self.issue(
                    path[:-1] if path and path[-1] == "params" else path,
                    f"Missing required parameter '{name}' for {what}.{hint}",
                )
                ok = False
        if not ok:
            return None
        result.update(provided)
        return result

    def construct(self, target, kwargs: dict, path: Path_, what: str):
        try:
            return target(**kwargs)
        except Exception as e:
            self.issue(path, f"Creating {what} failed: {type(e).__name__}: {e}")
            return None

    # -- components --

    def build_llms(self) -> dict[str, Any]:
        llms: dict[str, Any] = {}
        specs = self.config.llms
        # Plain providers first, then 'multi' that combine them.
        order = sorted(specs.items(), key=lambda kv: kv[1].provider == "multi")
        for name, spec in order:
            path = ("llms", name)
            llm = self._build_llm(name, spec, path, llms)
            if llm is not None:
                llms[name] = llm
        return llms

    def _build_llm(self, name: str, spec: LLMSpec, path: Path_, built: dict):
        cls, loaded = self.resolve_spec("llm", spec.provider, spec, path, "provider")
        if cls is None:
            return None
        what = (
            f"custom LLM '{spec.class_}'"
            if loaded
            else f"LLM provider '{spec.provider}'"
        )
        provided = {}
        if spec.provider == "multi":
            if not spec.llms:
                self.issue(path, "Provider 'multi' needs 'llms': a list of LLM names.")
                return None
            members = []
            for i, member in enumerate(spec.llms):
                if member not in self.config.llms:
                    self.issue(
                        path + ("llms", i),
                        f"Unknown LLM '{member}'."
                        + _did_you_mean(member, self.config.llms),
                    )
                elif self.config.llms[member].provider == "multi":
                    self.issue(
                        path + ("llms", i), "A 'multi' LLM cannot contain another one."
                    )
                elif member in built:
                    members.append(built[member])
            if len(members) != len(spec.llms):
                return None
            provided["llms"] = members
        elif spec.llms is not None:
            self.issue(path + ("llms",), "'llms' is only used with provider 'multi'.")
            return None
        if spec.model is not None:
            provided["model"] = spec.model
        if spec.api_key is not None:
            provided["api_key"] = spec.api_key
        kwargs = self.check_kwargs(
            cls, spec.params, path + ("params",), provided=provided, what=what
        )
        if kwargs is None:
            return None
        llm = self.construct(cls, kwargs, path, what)
        if llm is not None and loaded is not None:
            _tag(llm, loaded, spec.class_)
        return llm

    def build_methods(self, llms: dict[str, Any]):
        methods, llm_names = [], []
        default_budget = self.config.experiment.budget
        for i, spec in enumerate(self.config.methods):
            path = ("methods", i)
            cls, loaded = self.resolve_spec("method", spec.type, spec, path, "type")
            unknown = [n for n in spec.llm_names if n not in self.config.llms]
            for n in unknown:
                self.issue(
                    path + ("llm",),
                    f"Unknown LLM '{n}'."
                    + (
                        _did_you_mean(n, self.config.llms)
                        or " Defined in 'llms': " + ", ".join(self.config.llms) + "."
                    ),
                )
            if cls is None:
                continue
            what = (
                f"custom method '{spec.class_}'" if loaded else f"method '{spec.type}'"
            )
            sig_params = _param_names(cls)
            budget = spec.budget or default_budget
            # Check the parameters once (with a stand-in LLM), so parameter
            # errors are reported even when an LLM is unknown or broken.
            placeholder = {"llm": None, "budget": budget}
            if spec.name is not None and "name" in sig_params:
                placeholder["name"] = spec.name
            kwargs = self.check_kwargs(
                cls,
                spec.params,
                path + ("params",),
                provided=placeholder,
                kwargs_target=_KWARGS_TARGETS.get(("method", spec.type)),
                what=what,
            )
            converters = _PARAM_CONVERTERS.get(("method", spec.type), {})
            if kwargs is not None:
                for param, convert in converters.items():
                    if param in kwargs:
                        converted = convert(
                            kwargs[param],
                            path + ("params", param),
                            kwargs,
                            self.issue,
                        )
                        if converted is None:
                            kwargs = None
                            break
            if kwargs is None or unknown:
                continue
            for llm_name in spec.llm_names:
                llm = llms.get(llm_name)
                if llm is None:  # its own error was reported under 'llms'
                    continue
                call = {**kwargs, "llm": llm}
                for param, convert in converters.items():
                    if param in spec.params:  # fresh objects per method
                        call[param] = convert(
                            spec.params[param], (), kwargs, lambda *a: None
                        )
                name = None
                if spec.name is not None:
                    name = spec.name.replace("{llm}", llm_name)
                    if "name" in call:
                        call["name"] = name
                method = self.construct(cls, call, path, what)
                if method is None:
                    break  # same failure for every LLM; report once
                if name is not None:
                    method.name = name
                if loaded is not None:
                    _tag(method, loaded, spec.class_)
                methods.append(method)
                llm_names.append(llm_name)
        # The logger identifies a run by method name and the LLM's model
        # name (not the YAML alias), so check uniqueness on those.
        from ..loggers.base import ExperimentLogger

        seen: dict[tuple[str, str], str] = {}
        for method, llm_name in zip(methods, llm_names):
            model = ExperimentLogger._get_clean_llm_model_name(None, method.llm)
            key = (method.name, model)
            if key in seen:
                other = seen[key]
                via = (
                    f"LLMs '{other}' and '{llm_name}' (both model '{model}')"
                    if other != llm_name
                    else f"LLM '{llm_name}'"
                )
                self.issue(
                    ("methods",),
                    f"Two methods are named '{method.name}' with {via}; give "
                    "them different 'name's (e.g. with '{llm}'), or the results "
                    "of one would overwrite the other.",
                )
            seen[key] = llm_name
        return methods, llm_names

    def build_problems(self):
        from ..problem import Problem

        problems = []
        for i, spec in enumerate(self.config.problems):
            path = ("problems", i)
            if spec.suite is not None:
                fn = self.resolve("suite", spec.suite, path + ("suite",))
                if fn is None:
                    continue
                what = f"suite '{spec.suite}'"
                kwargs = self.check_kwargs(
                    fn, spec.params, path + ("params",), what=what
                )
                if kwargs is None:
                    continue
                result = self.construct(fn, kwargs, path, what)
                if result is None:
                    continue
                items = list(result) if not isinstance(result, Problem) else [result]
                bad = [p for p in items if not isinstance(p, Problem)]
                if bad:
                    self.issue(path, f"{what} returned non-Problem objects: {bad!r}")
                    continue
                for item in items:
                    _add_unique(item, "dependencies", spec.dependencies)
                problems.extend(items)
                continue
            cls, loaded = self.resolve_spec("problem", spec.type, spec, path, "type")
            if cls is None:
                continue
            what = (
                f"custom problem '{spec.class_}'"
                if loaded
                else f"problem '{spec.type}'"
            )
            provided = {}
            if spec.name is not None and "name" in _param_names(cls):
                provided["name"] = spec.name
            kwargs = self.check_kwargs(
                cls, spec.params, path + ("params",), provided=provided, what=what
            )
            if kwargs is None:
                continue
            problem = self.construct(cls, kwargs, path, what)
            if problem is None:
                continue
            if spec.name is not None:
                problem.name = spec.name
            if loaded is not None:
                _tag(problem, loaded, spec.class_)
                _add_unique(problem, "extra_pythonpath", [str(loaded.root)])
                _add_unique(problem, "dependencies", loaded.dependencies)
            _add_unique(problem, "dependencies", spec.dependencies)
            problems.append(problem)
        names: dict[str, int] = {}
        for problem in problems:
            names[problem.name] = names.get(problem.name, 0) + 1
        for name, count in names.items():
            if count > 1:
                self.issue(
                    ("problems",),
                    f"{count} problems are named '{name}'; give them different "
                    "'name's (their results would be mixed up).",
                )
        return problems

    def check_logger(self) -> None:
        spec = self.config.experiment.logger
        path = ("experiment", "logger")
        cls = self.resolve("logger", spec.type, path + ("type",))
        if cls is not None:
            self.check_kwargs(
                cls,
                spec.params,
                path + ("params",),
                provided={"name": "<output_dir>"},
                what=f"logger '{spec.type}'",
            )


def _tag(obj, loaded: LoadedSource, class_name: str) -> None:
    """Record where a custom component's code came from (logged with it)."""
    try:
        obj._blade_source = loaded.info(class_name)
    except Exception:  # e.g. __slots__; provenance is optional
        pass


def _add_unique(obj, attr: str, values) -> None:
    """Append values to a list attribute, skipping ones already present."""
    if not values:
        return
    current = list(getattr(obj, attr, None) or [])
    for value in values:
        if value not in current:
            current.append(value)
    setattr(obj, attr, current)


def _param_names(target) -> set[str]:
    try:
        return set(inspect.signature(target).parameters)
    except (TypeError, ValueError):
        return set()


def build_plan(source: LoadedConfig) -> ExperimentPlan:
    """Validate a loaded configuration and build all its components.

    Raises:
        ConfigError: With every issue found (schema, unknown components,
            bad parameters, failing constructors, duplicate names).
    """
    config = validate_config(source)
    builder = _Builder(source, config)
    llms = builder.build_llms()
    methods, llm_names = builder.build_methods(llms)
    problems = builder.build_problems()
    builder.check_logger()
    if builder.issues:
        raise ConfigError(builder.issues)
    section = config.experiment
    seeds = list(section.seeds) if section.seeds else list(range(section.runs))
    output_dir = Path(section.output_dir)
    if not output_dir.is_absolute():
        output_dir = source.base_dir / output_dir
    return ExperimentPlan(
        config=config,
        source=source,
        output_dir=output_dir.resolve(),
        llms=llms,
        methods=methods,
        method_llm_names=llm_names,
        problems=problems,
        seeds=seeds,
        sources=list(builder.sources.values()),
    )


def load_plan(
    source: str | os.PathLike | dict,
    *,
    overrides: list[str] | dict | None = None,
    env: dict[str, str] | None = None,
) -> ExperimentPlan:
    """Load a YAML file (or dict) and build its :class:`ExperimentPlan`.

    Args:
        source: Path of the YAML file, or the configuration as a dict.
        overrides: ``key.path=value`` overrides, see
            :func:`~iohblade.config.yaml_loader.apply_overrides`.
        env: Variables for ``${env:...}`` (default: ``.env`` + environment).

    Raises:
        ConfigError: If anything about the configuration is wrong.
    """
    return build_plan(load_yaml(source, overrides=overrides, env=env))


def load_experiment(
    source: str | os.PathLike | dict,
    *,
    overrides: list[str] | dict | None = None,
    env: dict[str, str] | None = None,
):
    """Load a YAML experiment and return a ready-to-run Experiment.

    Example:
        >>> from iohblade.config import load_experiment
        >>> experiment = load_experiment("experiment.yml")  # doctest: +SKIP
        >>> experiment()  # doctest: +SKIP

    Raises:
        ConfigError: If anything about the configuration is wrong.
    """
    return load_plan(source, overrides=overrides, env=env).make_experiment()
