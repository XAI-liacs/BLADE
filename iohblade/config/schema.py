"""Schema of a YAML experiment configuration (Pydantic v2 models).

Example::

    version: 1
    experiment:
      output_dir: results/sbox
      runs: 5
      budget: 100
    llms:
      flash: {provider: gemini, model: gemini-2.0-flash, api_key: "${env:GEMINI_API_KEY}"}
    methods:
      - {type: llamea, llm: flash, params: {n_parents: 1, n_offspring: 1}}
    problems:
      - {type: bbob_sbox, params: {dims: [5], specific_fid: 2}}

Component ``type`` / ``provider`` / ``suite`` values are keys of
:data:`iohblade.registry.registry`; ``params`` are passed as keyword
arguments to the component's constructor (or the suite's factory).
"""

from __future__ import annotations

from typing import Any, ClassVar, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PositiveInt,
    field_validator,
    model_validator,
)

from ..registry import (
    BuiltinMethod,
    BuiltinProblem,
    BuiltinSuite,
    LLMProvider,
    LoggerType,
)

SCHEMA_VERSION = 1


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class LoggerSpec(_Strict):
    """Where and how results are logged."""

    type: str = Field(
        "file",
        description="Logger key: " + ", ".join(m.value for m in LoggerType),
        examples=[m.value for m in LoggerType],
    )
    params: dict[str, Any] = Field(
        default_factory=dict,
        description="Extra constructor arguments, e.g. mlflow_tracking_uri.",
    )


class ExperimentSection(_Strict):
    """Settings of the experiment as a whole."""

    name: str | None = Field(None, description="Name of the experiment.")
    output_dir: str = Field(
        ...,
        description=(
            "Results directory, relative to the configuration file. An "
            "existing directory is resumed."
        ),
    )
    runs: PositiveInt = Field(5, description="Independent runs per method/problem.")
    seeds: list[int] | None = Field(
        None, description="Explicit seeds; overrides 'runs' when given."
    )
    budget: PositiveInt = Field(
        100,
        description=("Evaluations per run. Also the default budget of every method."),
    )
    n_jobs: PositiveInt = Field(1, description="Runs executed in parallel.")
    show_stdout: bool = Field(False, description="Show output of the runs.")
    log_stdout: bool = Field(
        False, description="Capture output of the runs to log files."
    )
    logger: LoggerSpec = Field(default_factory=LoggerSpec)
    env_cache: bool = Field(
        True,
        description=(
            "Reuse cached evaluation environments between problems with the "
            "same dependencies (also across experiments). False: a fresh "
            "temporary environment per problem."
        ),
    )
    isolated_pythonpath: bool = Field(
        False,
        description=(
            "Evaluate with only iohblade importable from BLADE's own "
            "environment (plus the problem's dependencies and custom code)."
        ),
    )

    @field_validator("seeds")
    @classmethod
    def _unique_seeds(cls, seeds):
        if seeds is not None:
            if not seeds:
                raise ValueError("'seeds' must not be empty (or leave it out).")
            repeated = sorted({s for s in seeds if seeds.count(s) > 1})
            if repeated:
                raise ValueError(
                    f"Seeds must be unique; repeated: {repeated}. A run is "
                    "identified by method, LLM, problem and seed, so repeated "
                    "seeds would share (and overwrite) their results."
                )
        return seeds


class _CustomSource(_Strict):
    """Fields for a component loaded from the user's own Python code."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)
    _kind_field: ClassVar[str] = "type"

    source: str | None = Field(
        None,
        description=(
            "With type/provider 'custom': a .py file or package folder, "
            "relative to the configuration file."
        ),
    )
    class_: str | None = Field(
        None,
        alias="class",
        description=(
            "With type/provider 'custom': the class to use, 'Name' or "
            "'submodule:Name' for a class in a package's submodule."
        ),
    )

    @model_validator(mode="after")
    def _custom_fields(self):
        kind = getattr(self, self._kind_field)
        if kind == "custom":
            missing = [
                name
                for name, value in (("source", self.source), ("class", self.class_))
                if not value
            ]
            if missing:
                raise ValueError(
                    f"{self._kind_field} 'custom' needs "
                    + " and ".join(f"'{m}'" for m in missing)
                    + "."
                )
        elif self.source is not None or self.class_ is not None:
            raise ValueError(
                f"'source' and 'class' are only used with {self._kind_field}: custom."
            )
        return self


class LLMSpec(_CustomSource):
    """A named LLM that methods refer to."""

    _kind_field: ClassVar[str] = "provider"

    provider: str = Field(
        ...,
        description=(
            "LLM provider key ('custom' for your own class): "
            + ", ".join(m.value for m in LLMProvider)
        ),
        examples=[m.value for m in LLMProvider] + ["custom"],
    )
    model: str | None = Field(None, description="Model name of the provider.")
    api_key: str | None = Field(
        None,
        description="API key. Use ${env:NAME} instead of writing the key here.",
    )
    llms: list[str] | None = Field(
        None, description="For provider 'multi': names of the LLMs to combine."
    )
    params: dict[str, Any] = Field(
        default_factory=dict, description="Extra constructor arguments."
    )


class MethodSpec(_CustomSource):
    """A search method; one method instance is created per LLM."""

    type: str = Field(
        ...,
        description=(
            "Method key ('custom' for your own class): "
            + ", ".join(m.value for m in BuiltinMethod)
        ),
        examples=[m.value for m in BuiltinMethod] + ["custom"],
    )
    name: str | None = Field(
        None,
        description=(
            "Display name. '{llm}' is replaced by the LLM name, e.g. " "'LLaMEA-{llm}'."
        ),
    )
    llm: str | list[str] = Field(
        ..., description="Name (or list of names) of LLMs from 'llms'."
    )
    budget: PositiveInt | None = Field(
        None, description="Method budget; defaults to experiment.budget."
    )
    params: dict[str, Any] = Field(
        default_factory=dict, description="Extra constructor arguments."
    )

    @property
    def llm_names(self) -> list[str]:
        """The LLM names as a list."""
        return [self.llm] if isinstance(self.llm, str) else list(self.llm)


class ProblemSpec(_CustomSource):
    """A problem (``type``) or a suite of problems (``suite``)."""

    type: str | None = Field(
        None,
        description=(
            "Problem key ('custom' for your own class): "
            + ", ".join(m.value for m in BuiltinProblem)
        ),
        examples=[m.value for m in BuiltinProblem] + ["custom"],
    )
    suite: str | None = Field(
        None,
        description=(
            "Suite key (expands into several problems): "
            + ", ".join(m.value for m in BuiltinSuite)
        ),
        examples=[m.value for m in BuiltinSuite],
    )
    name: str | None = Field(None, description="Problem name (not for suites).")
    params: dict[str, Any] = Field(
        default_factory=dict,
        description="Constructor (or suite factory) arguments.",
    )

    smoke_test_solution: str | None = Field(
        None,
        description=(
            "A .py file with a known-good solution, evaluated by "
            "`iohblade check` (default: the code in the problem's example "
            "prompt)."
        ),
    )
    dependencies: list[str] = Field(
        default_factory=list,
        description=(
            "Extra pip requirements installed in the problem's evaluation "
            "environment (added to requirements.txt / pyproject.toml found "
            "next to a custom source)."
        ),
    )

    @model_validator(mode="after")
    def _type_xor_suite(self):
        if (self.type is None) == (self.suite is None):
            raise ValueError("Give exactly one of 'type' or 'suite'.")
        if self.suite is not None and self.name is not None:
            raise ValueError(
                "'name' cannot be used with 'suite'; the suite names its problems."
            )
        return self


class ExperimentConfig(_Strict):
    """A complete experiment: settings, LLMs, methods and problems."""

    version: Literal[1] = Field(
        SCHEMA_VERSION, description="Version of the configuration format."
    )
    experiment: ExperimentSection
    llms: dict[str, LLMSpec] = Field(..., min_length=1)
    methods: list[MethodSpec] = Field(..., min_length=1)
    problems: list[ProblemSpec] = Field(..., min_length=1)


def json_schema() -> dict[str, Any]:
    """JSON schema of the configuration format (for editor completion)."""
    schema = ExperimentConfig.model_json_schema()
    schema["$schema"] = "https://json-schema.org/draft/2020-12/schema"
    schema["title"] = "BLADE experiment configuration"
    return schema
