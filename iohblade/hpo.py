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
    # 2.3.1 is the version pinned by the other benchmarks and the lower bound
    # of BLADE's own dependency; 2.3.1 - 2.4.1 share the Scenario /
    # AlgorithmConfigurationFacade API used by run_smac. Capped below 2.5 so
    # that a new SMAC release (API or scikit-learn changes) is adopted on
    # purpose, together with the scikit-learn cap below.
    "smac>=2.3.1,<2.5",
    "swig",  # builds pyrfr, the random forest of smac < 2.4
    "ConfigSpace>=1.2",
    # smac 2.4.0 imports sklearn.tree._tree.DTYPE, which scikit-learn 1.9
    # removed, so `import smac` fails with scikit-learn 1.9 (issue #220).
    # Only the HPO environment is capped; evaluations without HPO keep the
    # latest scikit-learn.
    "scikit-learn<1.9",
]


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
                "scikit-learn 1.9)."
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
