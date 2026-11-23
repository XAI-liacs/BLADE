"""Optional hyperparameter tuning with SMAC (issue #220).

smac is only imported on the tuning path; these tests replace it with small
fake modules (or make its import fail) and never install anything.
"""

import builtins
import importlib
import os
import subprocess
import sys
import types
import warnings
from pathlib import Path
from unittest.mock import MagicMock

import cloudpickle
import pytest
import yaml
from ConfigSpace import ConfigurationSpace

from iohblade import cli, hpo
from iohblade.envs import env_key
from iohblade.hpo import HPO_DEPENDENCIES, HPOUnavailableError
from iohblade.method import Method
from iohblade.problem import (
    HARNESS_ERROR_PREFIX,
    Problem,
    _build_eval_script,
    _harness_feedback,
)
from iohblade.solution import Solution

# Stubs openml (not installed in BLADE's own environment) and imports the
# AutoML module; reused from the AutoML scoring tests.
from .test_benchmark_scores_bbob import (  # noqa: F401
    _AUTOML_PREFIX,
    GOOD_PIPELINE,
    _automl_problem,
    _binary_data,
    _drop_modules,
    automl_module,
)

TUNABLE_PIPELINE = """
from sklearn.linear_model import LogisticRegression

class TunablePipeline:
    def __init__(self, X, y, C=1.0):
        self.C = C
        self.model = LogisticRegression(C=C, max_iter=1000).fit(X, y)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def __call__(self, X):
        return self.model.predict(X)
"""


def _configspace():
    return ConfigurationSpace({"C": (0.01, 10.0)})


def _block_smac(monkeypatch, calls=None):
    """Make every ``import smac`` fail (as with scikit-learn 1.9)."""
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "smac" or name.startswith("smac."):
            if calls is not None:
                calls.append(name)
            raise ImportError(
                "cannot import name 'DTYPE' from 'sklearn.tree._tree' (stub)"
            )
        return real_import(name, *args, **kwargs)

    for name in list(sys.modules):
        if name == "smac" or name.startswith("smac."):
            monkeypatch.delitem(sys.modules, name)
    monkeypatch.setattr(builtins, "__import__", fake_import)


def _fake_smac(monkeypatch, incumbent):
    """A fake smac module whose optimiser evaluates the target once."""
    fake = types.ModuleType("smac")
    record = {}

    class Scenario:
        def __init__(self, configspace, **kwargs):
            record["scenario"] = dict(kwargs, configspace=configspace)
            self.kwargs = kwargs

    class AlgorithmConfigurationFacade:
        def __init__(self, scenario, target_function, logging_level=20):
            record["logging_level"] = logging_level
            self.scenario = scenario
            self.target = target_function

        def optimize(self):
            instance = self.scenario.kwargs["instances"][0]
            record["cost"] = self.target(incumbent, instance, seed=0)
            return incumbent

    fake.Scenario = Scenario
    fake.AlgorithmConfigurationFacade = AlgorithmConfigurationFacade
    monkeypatch.setitem(sys.modules, "smac", fake)
    return record


class _BaseProblem(Problem):
    def evaluate(self, solution):
        return solution

    def test(self, solution):
        return solution

    def to_dict(self):
        return {}

    def get_config(self):
        return {}


class _TunableProblem(_BaseProblem):
    supports_hpo = True


class _Method(Method):
    def __init__(self, hpo=False, name="M"):
        super().__init__(llm=None, budget=1, name=name)
        self._hpo = hpo

    @property
    def uses_hpo(self):
        return self._hpo

    def __call__(self, problem):
        return None

    def to_dict(self):
        return {}

    def get_config(self):
        return {}


# ------------------------------------------------------------ hpo module ---


def test_hpo_dependencies():
    names = [d.split(">")[0].split("<")[0].split("=")[0] for d in HPO_DEPENDENCIES]
    assert names == ["smac", "ConfigSpace"]
    # smac >= 2.4.1 imports with scikit-learn 1.9 and needs no pyrfr/swig.
    assert "smac>=2.4.1,<2.5" in HPO_DEPENDENCIES
    assert not any(d.startswith(("scikit-learn", "swig")) for d in HPO_DEPENDENCIES)
    assert issubclass(HPOUnavailableError, ImportError)


def test_import_smac_unavailable(monkeypatch):
    _block_smac(monkeypatch)
    with pytest.raises(HPOUnavailableError) as exc:
        hpo.import_smac()
    text = str(exc.value)
    assert "DTYPE" in text and "scikit-learn" in text
    assert "smac>=2.4.1" in text and "enable_hpo" in text


def test_run_smac_returns_incumbent(monkeypatch):
    record = _fake_smac(monkeypatch, {"C": 2.0})
    cs = _configspace()
    target = MagicMock(return_value=0.25)
    result = hpo.run_smac(cs, target, ["a", "b"], n_trials=7, output_dir="out")
    assert result == {"C": 2.0}
    scenario = record["scenario"]
    assert scenario["configspace"] is cs
    assert scenario["n_trials"] == 7
    assert scenario["instances"] == ["a", "b"]
    assert scenario["instance_features"] == {"a": [0.0], "b": [1.0]}
    assert scenario["output_directory"] == "out"
    assert scenario["deterministic"] is True and scenario["seed"] == 0
    target.assert_called_once_with({"C": 2.0}, "a", seed=0)


def test_run_smac_accepts_serialised_configspace(monkeypatch):
    record = _fake_smac(monkeypatch, {"C": 1.0})
    serialised = Solution(configspace=_configspace()).to_dict()["configspace"]
    assert isinstance(serialised, dict)
    hpo.run_smac(serialised, lambda *a, **k: 0.0, ["a"])
    assert isinstance(record["scenario"]["configspace"], ConfigurationSpace)


# ---------------------------------------------------------- enable_hpo ---


def test_enable_hpo_adds_and_removes_dependencies():
    problem = _TunableProblem(dependencies=["pandas"])
    before = list(problem.dependencies)
    key_before = env_key(problem.dependencies, False)

    problem.enable_hpo(True)
    assert problem.hpo is True
    assert problem.dependencies == before + HPO_DEPENDENCIES
    assert env_key(problem.dependencies, False) != key_before
    problem.enable_hpo(True)  # idempotent
    assert problem.dependencies == before + HPO_DEPENDENCIES

    problem.enable_hpo(False)
    assert problem.hpo is False
    assert problem.dependencies == before
    assert env_key(problem.dependencies, False) == key_before


def test_enable_hpo_keeps_dependencies_it_did_not_add():
    problem = _TunableProblem(dependencies=["ConfigSpace>=1.2"])
    problem.enable_hpo(True)
    problem.enable_hpo(False)
    assert "ConfigSpace>=1.2" in problem.dependencies


def test_enable_hpo_resets_a_built_environment(tmp_path):
    problem = _TunableProblem()
    problem._env_path, problem._python_bin = tmp_path, tmp_path / "python"
    problem._env_cached = True
    problem.enable_hpo(True)
    assert problem._env_path is None and problem._python_bin is None


def test_enable_hpo_unsupported_problem():
    problem = _BaseProblem(name="plain")
    with pytest.raises(ValueError, match="'plain'.*does not support"):
        problem.enable_hpo(True)
    problem.enable_hpo(False)  # disabling is always fine
    assert problem.hpo is False


def test_configspace_without_hpo_warns_once(caplog):
    problem = _TunableProblem(name="p")
    problem._ensure_env = MagicMock(side_effect=RuntimeError("no env"))
    with caplog.at_level("WARNING", logger="iohblade"):
        problem(Solution(configspace=_configspace()))
        problem(Solution(configspace=_configspace()))
    messages = [r.getMessage() for r in caplog.records if "configuration" in r.msg]
    assert len(messages) == 1
    assert "'p'" in messages[0] and "default hyperparameters" in messages[0]


# --------------------------------------------------------------- AutoML ---


def test_automl_imports_and_constructs_without_smac(monkeypatch, automl_module):
    _block_smac(monkeypatch)
    _drop_modules(_AUTOML_PREFIX)  # import it again, now without smac
    module = importlib.import_module(f"{_AUTOML_PREFIX}.automl")
    problem = module.AutoML()
    assert module.AutoML.supports_hpo is True
    assert problem.hpo is False
    deps = " ".join(problem.dependencies)
    assert "smac" not in deps and "swig" not in deps
    assert "ConfigSpace" not in deps and "<1.9" not in deps
    assert "scikit-learn>=1.4" in problem.dependencies
    assert "openml>=0.14" in problem.dependencies

    tuned = module.AutoML(hpo=True)
    assert tuned.hpo is True
    for dep in HPO_DEPENDENCIES:
        assert dep in tuned.dependencies
    assert tuned.get_config()["config"]["hpo"] is True


def test_automl_without_hpo_never_imports_smac(monkeypatch, automl_module):
    calls = []
    _block_smac(monkeypatch, calls)
    X, y = _binary_data()
    problem = _automl_problem(automl_module, "predictive_accuracy", X, y)
    solution = Solution(
        code=TUNABLE_PIPELINE, name="TunablePipeline", configspace=_configspace()
    )
    solution = problem._evaluate_search(solution)
    assert calls == []
    assert solution.get_metadata("incumbent") is None
    assert solution.fitness > 0.5
    assert "HPO is not enabled" in solution.feedback
    assert "default hyperparameters" in solution.feedback

    plain = problem._evaluate_search(Solution(code=GOOD_PIPELINE, name="GoodPipeline"))
    assert "HPO" not in plain.feedback
    assert calls == []


def test_automl_uses_run_smac_incumbent(monkeypatch, automl_module):
    record = _fake_smac(monkeypatch, {"C": 0.5})
    X, y = _binary_data()
    problem = _automl_problem(automl_module, "predictive_accuracy", X, y)
    problem.enable_hpo(True)
    solution = Solution(
        code=TUNABLE_PIPELINE, name="TunablePipeline", configspace=_configspace()
    )
    solution = problem._evaluate_search(solution, ioh_dir="somewhere")
    assert solution.get_metadata("incumbent") == {"C": 0.5}
    assert "HPO used; incumbent={'C': 0.5}" in solution.feedback
    scenario = record["scenario"]
    assert scenario["n_trials"] == 100
    assert scenario["deterministic"] is True
    assert scenario["output_directory"].endswith("smac")
    assert scenario["name"].startswith("automl-None-")
    assert sorted(scenario["instances"]) == [f"r0,f{i},s0" for i in range(4)]
    assert scenario["instance_features"]["r0,f1,s0"] == [0, 1, 0]
    assert 0.0 <= record["cost"] <= 1.0  # 1 - accuracy of the incumbent
    assert record["logging_level"] == 30


def test_automl_hpo_without_smac_fails_before_code_is_read(monkeypatch, automl_module):
    _block_smac(monkeypatch)
    X, y = _binary_data()
    problem = _automl_problem(automl_module, "predictive_accuracy", X, y)
    problem.enable_hpo(True)

    class Watched(Solution):
        code_read = False

        def __getattribute__(self, name):
            if name == "code":
                type(self).code_read = True
            return super().__getattribute__(name)

    solution = Watched(
        code=TUNABLE_PIPELINE, name="TunablePipeline", configspace=_configspace()
    )
    Watched.code_read = False
    with pytest.raises(HPOUnavailableError):
        problem._evaluate_search(solution)
    assert Watched.code_read is False


STUB_OPENML = "tasks = None\n"
STUB_SMAC = (
    "raise ImportError(\"cannot import name 'DTYPE' from "
    "'sklearn.tree._tree' (stub)\")\n"
)


def test_automl_hpo_without_smac_is_a_harness_error(tmp_path, automl_module):
    """The evaluation script marks the failure as a harness error."""
    stubs = tmp_path / "stubs"
    (stubs / "smac").mkdir(parents=True)
    (stubs / "openml.py").write_text(STUB_OPENML)
    (stubs / "smac" / "__init__.py").write_text(STUB_SMAC)

    problem = automl_module.AutoML(hpo=True)
    problem.split_info = {"n_repeats": 1, "n_folds": 2, "n_samples": 1}
    solution = Solution(
        code=TUNABLE_PIPELINE, name="TunablePipeline", configspace=_configspace()
    )
    problem_pkl = tmp_path / "problem.pkl"
    solution_pkl = tmp_path / "solution.pkl"
    marker = tmp_path / "marker.txt"
    problem_pkl.write_bytes(cloudpickle.dumps(problem))
    solution_pkl.write_bytes(cloudpickle.dumps(solution))
    script = tmp_path / "run.py"
    script.write_text(
        _build_eval_script(
            problem.imports, problem_pkl, solution_pkl, tmp_path / "out", marker
        )
    )
    root = str(Path(automl_module.__file__).resolve().parents[3])
    proc = subprocess.run(
        [sys.executable, str(script)],
        # Extend (not replace) the environment: on Windows the interpreter
        # needs SYSTEMROOT etc. to start (WinError 10106 otherwise).
        env={**os.environ, "PYTHONPATH": os.pathsep.join([str(stubs), root])},
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert proc.returncode != 0
    assert "HPOUnavailableError" in proc.stderr
    assert "smac>=2.4.1" in proc.stderr
    assert marker.exists(), proc.stderr  # harness error, not a candidate one

    feedback = _harness_feedback(proc.stderr)
    assert feedback.startswith(HARNESS_ERROR_PREFIX)
    assert "HPOUnavailableError" in feedback


# ----------------------------------------------- experiments and config ---


def test_llamea_uses_hpo():
    from iohblade.methods import LLaMEA

    assert LLaMEA(None, budget=1, HPO=True).uses_hpo is True
    assert LLaMEA(None, budget=1).uses_hpo is False
    assert LLaMEA(None, budget=1, HPO=False).uses_hpo is False


def test_configure_hpo():
    tunable, plain = _TunableProblem(name="t"), _BaseProblem(name="plain")
    notes, warns = hpo.configure_hpo([_Method()], [tunable, plain])
    assert (notes, warns) == ([], [])
    assert tunable.hpo is False

    notes, warns = hpo.configure_hpo([_Method(hpo=True, name="L")], [tunable, plain])
    assert tunable.hpo is True
    assert tunable.dependencies[-len(HPO_DEPENDENCIES) :] == HPO_DEPENDENCIES
    assert len(notes) == 1 and "'t'" in notes[0] and "smac>=2.4.1" in notes[0]
    assert len(warns) == 1 and "'plain'" in warns[0] and "L" in warns[0]
    assert plain.hpo is False

    deps = list(tunable.dependencies)
    hpo.configure_hpo([_Method(hpo=True)], [tunable])  # idempotent
    assert tunable.dependencies == deps


def test_experiment_enables_hpo(tmp_path):
    from iohblade.experiment import Experiment

    tunable, plain = _TunableProblem(name="t"), _BaseProblem(name="plain")
    with pytest.warns(UserWarning, match="'plain' does not support"):
        Experiment(
            methods=[_Method(hpo=True)],
            problems=[tunable, plain],
            runs=1,
            exp_logger=MagicMock(),
        )
    assert tunable.hpo is True
    assert all(dep in tunable.dependencies for dep in HPO_DEPENDENCIES)
    assert plain.hpo is False


def test_experiment_without_hpo_method_changes_nothing():
    from iohblade.experiment import Experiment

    tunable = _TunableProblem(name="t")
    before = list(tunable.dependencies)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        Experiment(methods=[_Method()], problems=[tunable], exp_logger=MagicMock())
    assert tunable.hpo is False and tunable.dependencies == before


def test_experiment_enables_hpo_before_environments():
    """Methods changed after __init__ still enable HPO before the envs."""
    from iohblade.experiment import Experiment

    tunable = _TunableProblem(name="t")
    exp = Experiment(methods=[], problems=[tunable], runs=1, exp_logger=MagicMock())
    exp.methods = [_Method(hpo=True)]
    seen = []

    def prepare():
        seen.append((tunable.hpo, list(tunable.dependencies)))
        raise KeyboardInterrupt  # stop _run_all here

    exp._prepare_environments = prepare
    with pytest.raises(KeyboardInterrupt):
        exp._run_all()
    ((hpo_enabled, deps),) = seen
    assert hpo_enabled is True
    assert deps[-len(HPO_DEPENDENCIES) :] == HPO_DEPENDENCIES


def _write_config(tmp_path, problems):
    config = {
        "experiment": {"output_dir": "out", "runs": 1, "budget": 2},
        "llms": {"d": {"provider": "dummy"}},
        "methods": [
            {"type": "llamea", "name": "L-{llm}", "llm": "d", "params": {"HPO": True}}
        ],
        "problems": problems,
    }
    path = tmp_path / "exp.yml"
    path.write_text(yaml.safe_dump(config))
    return path


def test_validate_reports_hpo(tmp_path, capsys, automl_module):
    path = _write_config(
        tmp_path,
        [{"type": "automl", "name": "aml"}, {"type": "circle_packing", "name": "c"}],
    )
    assert cli.main(["validate", str(path)]) == cli.EXIT_OK
    out = capsys.readouterr().out
    assert "Note: HPO (SMAC) is enabled for problem 'aml'" in out
    assert "Warning: Method(s) L-d sample configuration spaces" in out
    assert "problem 'c' does not support" in out
    assert "Configuration is valid." in out


def test_plan_enables_hpo_for_check(tmp_path, automl_module):
    from iohblade.config import load_plan

    plan = load_plan(_write_config(tmp_path, [{"type": "automl", "name": "aml"}]))
    (problem,) = plan.problems
    assert problem.hpo is True
    assert all(dep in problem.dependencies for dep in HPO_DEPENDENCIES)
    assert len(plan.notes) == 1 and plan.warnings == []
