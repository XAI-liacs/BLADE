"""Define experiments in YAML.

Example:
    >>> from iohblade.config import load_experiment
    >>> experiment = load_experiment("experiment.yml")  # doctest: +SKIP
    >>> experiment()  # doctest: +SKIP

The same is available on the command line as ``iohblade run experiment.yml``.
"""

from .build import ExperimentPlan, build_plan, load_experiment, load_plan
from .errors import ConfigError, ConfigIssue
from .schema import (
    ExperimentConfig,
    ExperimentSection,
    LLMSpec,
    LoggerSpec,
    MethodSpec,
    ProblemSpec,
    json_schema,
)
from .user_code import LoadedSource, UserCodeError
from .yaml_loader import LoadedConfig, load_yaml

__all__ = [
    "ConfigError",
    "ConfigIssue",
    "ExperimentConfig",
    "ExperimentPlan",
    "ExperimentSection",
    "LLMSpec",
    "LoadedConfig",
    "LoadedSource",
    "LoggerSpec",
    "MethodSpec",
    "ProblemSpec",
    "UserCodeError",
    "build_plan",
    "json_schema",
    "load_experiment",
    "load_plan",
    "load_yaml",
]
