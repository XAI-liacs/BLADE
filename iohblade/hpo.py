"""Hyperparameter tuning (HPO) of generated solutions with SMAC.

Some methods (e.g. LLaMEA with ``HPO=True``) ask the LLM for a
configuration space next to the code (``Solution.configspace``). A problem
that supports HPO (``Problem.supports_hpo``) then tunes the solution's
hyperparameters with SMAC inside its evaluation environment before scoring
it. This module holds what those problems share:

* :data:`HPO_DEPENDENCIES`, the packages tuning needs in the evaluation
  environment (added by :meth:`iohblade.problem.Problem.enable_hpo`);
* :func:`import_smac`, a lazy import of smac with a clear error;
* :func:`run_smac`, a small wrapper around SMAC's algorithm configuration
  facade;
* :func:`hpo_instance_subset` and :func:`plain_config`, small helpers for
  the problems that tune (a fixed subset of tuning instances, and an
  incumbent that can be logged as JSON);
* :func:`configure_hpo`, which enables HPO on the problems of an experiment
  whose methods sample configuration spaces.

smac is an optional dependency: nothing here imports it until tuning runs.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Iterable

log = logging.getLogger("iohblade")

# Requirements added to a problem's evaluation environment when HPO is
# enabled (see Problem.enable_hpo); runs without HPO do not install them.
HPO_DEPENDENCIES: list[str] = [
    # smac 2.4.0 imports sklearn.tree._tree.DTYPE, which scikit-learn 1.9
    # removed, so it fails to import with scikit-learn 1.9 (issue #220);
    # 2.4.1 works with it and no longer needs pyrfr (so no swig either).
    # Capped below 2.5 so that a new SMAC release (API changes to Scenario /
    # AlgorithmConfigurationFacade, used by run_smac) is adopted on purpose.
    "smac>=2.4.1,<2.5",
    "ConfigSpace>=1.2",
]


# Feedback line of a solution whose configuration space was not tuned.
UNUSED_CONFIGSPACE_FEEDBACK = (
    "HPO is not enabled for this problem: the configuration space was "
    "ignored and the default hyperparameters were used."
)


class HPOUnavailableError(ImportError):
    """smac, needed for hyperparameter tuning, cannot be imported.

    A subclass of ``ImportError`` (code that treats a missing optional
    dependency as an ``ImportError`` keeps working). Raised by
    :func:`import_smac`; it is a problem of the evaluation environment, not
    of the generated code.
    """


def import_smac():
    """Import and return the ``smac`` module.

    Returns:
        module: The imported ``smac`` package.

    Raises:
        HPOUnavailableError: If smac is not installed or fails to import,
            with the cause and how to fix it.
    """
    try:
        import smac
    except Exception as e:  # ImportError, or errors raised while importing
        cause = f"{type(e).__name__}: {e}"
        hint = ""
        if "sklearn" in str(e) or "DTYPE" in str(e):
            hint = (
                " This smac version is not compatible with the installed "
                "scikit-learn (smac 2.4.0 does not import with "
                "scikit-learn 1.9; smac >= 2.4.1 does)."
            )
        raise HPOUnavailableError(
            "Hyperparameter tuning (HPO) needs smac, which cannot be imported "
            f"({cause}).{hint} Enable HPO on the problem "
            "(problem.enable_hpo(), or a method that samples with HPO) so its "
            "evaluation environment installs "
            f"{', '.join(HPO_DEPENDENCIES)}; for an environment you manage "
            "yourself, install these packages."
        ) from e
    return smac


def run_smac(
    configspace,
    target: Callable[..., float],
    instances: list[str],
    instance_features: dict[str, list[float]] | None = None,
    n_trials: int = 100,
    output_dir: str | None = None,
    seed: int = 0,
    deterministic: bool = True,
    name: str | None = None,
) -> dict[str, Any]:
    """Tune a configuration space with SMAC's algorithm configuration facade.

    Args:
        configspace: The ``ConfigSpace.ConfigurationSpace`` to search (or
            its serialised dict).
        target: ``target(config, instance, seed=0) -> float``, the cost to
            minimise for one configuration on one instance.
        instances: Instance names passed to ``target``.
        instance_features: Feature vector per instance name (SMAC needs
            features for its instance-aware model). Default: the index of
            each instance.
        n_trials: Number of target evaluations.
        output_dir: Directory for SMAC's output (default ``smac3_output``).
        seed: Seed of the SMAC run.
        deterministic: Whether ``target`` is deterministic (one seed per
            configuration and instance).
        name: Name of the SMAC run (its sub-directory of ``output_dir``).

    Returns:
        dict: The best configuration found (the incumbent) as a plain dict.

    Raises:
        HPOUnavailableError: If smac cannot be imported.
    """
    smac = import_smac()
    if isinstance(configspace, dict):  # serialised (Solution.to_dict)
        from ConfigSpace import ConfigurationSpace

        configspace = ConfigurationSpace.from_serialized_dict(configspace)
    if instance_features is None:
        instance_features = {iid: [float(i)] for i, iid in enumerate(instances)}
    scenario = smac.Scenario(
        configspace,
        name=name,
        deterministic=deterministic,
        n_trials=n_trials,
        instances=instances,
        instance_features=instance_features,
        output_directory=output_dir or "smac3_output",
        seed=seed,
    )
    facade = smac.AlgorithmConfigurationFacade(
        scenario, target_function=target, logging_level=30
    )
    incumbent = facade.optimize()
    if isinstance(incumbent, list):  # multi-objective: several incumbents
        incumbent = incumbent[0]
    return dict(incumbent)


def hpo_instance_subset(
    instances: Iterable, max_instances: int | None, seed: int = 42
) -> list:
    """A fixed, seeded subset of the instances to tune on.

    Args:
        instances: The candidate instances (e.g. a problem's training
            instances).
        max_instances: The maximum number of instances to keep; ``None``
            (or a value of at least ``len(instances)``) keeps all of them.
        seed: Seed of the shuffle, so every solution of a problem is tuned
            on the same instances.

    Returns:
        list: The selected instances, in a seeded random order.
    """
    import numpy as np

    instances = list(instances)
    order = np.random.RandomState(seed).permutation(len(instances))
    selected = [instances[i] for i in order]
    if max_instances is not None:
        selected = selected[: max(1, int(max_instances))]
    return selected


def smac_output_dir(problem) -> str:
    """Directory for SMAC's output of a problem's tuning runs.

    ``<log dir>/smac`` when the problem has a logger directory
    (``Problem.logger_dir``), otherwise ``iohblade-smac`` in the system's
    temporary directory (not the working directory of the evaluation).

    Args:
        problem: The problem that tunes.

    Returns:
        str: The directory (SMAC creates it).
    """
    import os
    import tempfile

    logger_dir = getattr(problem, "logger_dir", "") or ""
    if logger_dir:
        return os.path.join(logger_dir, "smac")
    return os.path.join(tempfile.gettempdir(), "iohblade-smac")


def plain_config(config) -> dict[str, Any]:
    """A configuration (or dict) as a dict of plain Python values.

    NumPy scalars are converted with ``.item()``, so the incumbent can be
    stored in a solution's metadata (logged as JSON) and passed as keyword
    arguments to the generated code.

    Args:
        config: A ``ConfigSpace.Configuration``, a dict or ``None``.

    Returns:
        dict: The hyperparameters by name (empty for ``None``).
    """
    if not config:
        return {}
    out = {}
    for key, value in dict(config).items():
        if hasattr(value, "item") and callable(value.item):
            value = value.item()
        out[str(key)] = value
    return out


def evaluation_config(
    problem, solution, test: bool = False, incumbent: dict | None = None
) -> tuple[dict[str, Any], str]:
    """The hyperparameters to score a solution with, and a feedback line.

    Shared by the benchmarks that tune (BBOB-SBOX, MA-BBOB, Kerneltuner):

    * ``incumbent`` given (tuned in this evaluation): the incumbent;
    * on the test instances (``test=True``): the incumbent recorded during
      the search (``solution.metadata["incumbent"]``), if any;
    * otherwise the generated code's defaults (an empty dict), with a note
      when the solution's configuration space was not used.

    Args:
        problem: The problem (``hpo`` and ``hpo_trials`` are read).
        solution: The solution being scored.
        test: Whether the solution is scored on the test instances.
        incumbent: The incumbent of a tuning run in this evaluation.

    Returns:
        tuple[dict, str]: The keyword arguments for the algorithm's
        constructor, and a feedback line ("" when there is nothing to say).
    """
    if incumbent is not None:
        config = plain_config(incumbent)
        trials = getattr(problem, "hpo_trials", None)
        budget = f", {trials} trials" if trials else ""
        return config, f"HPO used (SMAC{budget}); scored with incumbent={config}."
    configspace = getattr(solution, "configspace", None)
    recorded = solution.get_metadata("incumbent") if test else None
    if recorded:
        config = plain_config(recorded)
        return config, f"Scored with the HPO incumbent={config}."
    if configspace is None:
        return {}, ""
    if not getattr(problem, "hpo", False):
        return {}, UNUSED_CONFIGSPACE_FEEDBACK
    return {}, (
        "No HPO incumbent was recorded for this solution: the default "
        "hyperparameters were used."
    )


def configure_hpo(methods: Iterable, problems: Iterable) -> tuple[list, list]:
    """Enable HPO on the problems when a method samples with HPO.

    Called before evaluation environments are built, by
    :class:`~iohblade.experiment.Experiment` and when a YAML experiment is
    built (so ``iohblade validate`` and ``iohblade check`` see it). It is
    idempotent. A problem shared by several methods gets one evaluation
    environment, so all methods compared on it use the same packages.

    Args:
        methods: The experiment's methods (``Method.uses_hpo``).
        problems: The experiment's problems (``Problem.supports_hpo``).

    Returns:
        tuple[list[str], list[str]]: Notes (HPO enabled on a problem) and
        warnings (HPO requested for a problem that does not support it).
    """
    hpo_methods = [m for m in methods if getattr(m, "uses_hpo", False)]
    names = ", ".join(sorted({str(getattr(m, "name", m)) for m in hpo_methods}))
    notes, warnings = [], []
    for problem in problems:
        if hpo_methods and not getattr(problem, "hpo", False):
            if getattr(problem, "supports_hpo", False):
                problem.enable_hpo(True)
            else:
                warnings.append(
                    f"Method(s) {names} sample configuration spaces (HPO), "
                    f"but problem '{problem.name}' does not support "
                    "hyperparameter tuning: its solutions are evaluated with "
                    "their default hyperparameters."
                )
                continue
        if getattr(problem, "hpo", False):
            notes.append(
                f"HPO (SMAC) is enabled for problem '{problem.name}'; its "
                "evaluation environment also installs "
                + ", ".join(HPO_DEPENDENCIES)
                + "."
            )
    return notes, warnings
