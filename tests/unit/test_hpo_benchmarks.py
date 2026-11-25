"""SMAC tuning on BBOB-SBOX, MA-BBOB and Kerneltuner (issue #221).

smac is replaced by a small fake module (or its import is made to fail);
no real SMAC run happens. BBOB-SBOX and MA-BBOB use the real ioh with tiny
budgets and an algorithm whose result depends on a hyperparameter, so the
incumbent visibly changes the score. kernel_tuner is not needed: the
autotuning-methodology calls are replaced with fakes.
"""

import ast
import builtins
import importlib
import os
import subprocess
import sys
import types
import warnings
from pathlib import Path

import cloudpickle
import pytest
from ConfigSpace import ConfigurationSpace

from iohblade import hpo
from iohblade.benchmarks.BBOB.bbob_sboxcost import BBOB_SBOX
from iohblade.benchmarks.BBOB.mabbob import MA_BBOB
from iohblade.hpo import HPO_DEPENDENCIES, HPOUnavailableError
from iohblade.problem import HARNESS_ERROR_PREFIX, _build_eval_script, _harness_feedback
from iohblade.solution import Solution

# Runs a deterministic compass search from the centre (w > 0.5, a good
# AOCC) or evaluates a bad corner of the domain once (w <= 0.5, the
# default): the incumbent decides the score. (Generated code cannot read
# func.optimum; see iohblade.benchmarks.blackbox.)
AIMED = """
import numpy as np

class Aimed:
    def __init__(self, budget=100, dim=2, w=0.0):
        self.budget = budget
        self.dim = dim
        self.w = w

    def __call__(self, func):
        lb = np.asarray(func.bounds.lb, dtype=float)
        ub = np.asarray(func.bounds.ub, dtype=float)
        if self.w <= 0.5:
            return func(ub), ub
        x = (lb + ub) / 2
        f = func(x)
        evals = 1
        step = (ub - lb) / 4
        while evals < self.budget:
            improved = False
            for i in range(self.dim):
                for sign in (1.0, -1.0):
                    if evals >= self.budget:
                        break
                    y = x.copy()
                    y[i] = np.clip(y[i] + sign * step[i], lb[i], ub[i])
                    fy = func(y)
                    evals += 1
                    if fy < f:
                        x, f, improved = y, fy, True
            if not improved:
                step = step / 2
        return f, x
"""


def _configspace():
    return ConfigurationSpace({"w": (0.0, 1.0)})


def _solution(**kwargs):
    return Solution(code=AIMED, name="Aimed", configspace=_configspace(), **kwargs)


def _block(monkeypatch, module, calls=None):
    """Make every import of ``module`` fail."""
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == module or name.startswith(module + "."):
            if calls is not None:
                calls.append(name)
            raise ImportError(f"No module named '{module}' (blocked)")
        return real_import(name, *args, **kwargs)

    for name in list(sys.modules):
        if name == module or name.startswith(module + "."):
            monkeypatch.delitem(sys.modules, name)
    monkeypatch.setattr(builtins, "__import__", fake_import)


def _fake_smac(monkeypatch, incumbent, others=({"w": 0.0},)):
    """A fake smac: evaluates the incumbent and ``others`` on every instance."""
    fake = types.ModuleType("smac")
    record = {"costs": []}

    class Scenario:
        def __init__(self, configspace, **kwargs):
            record["scenario"] = dict(kwargs, configspace=configspace)
            self.kwargs = kwargs

    class AlgorithmConfigurationFacade:
        def __init__(self, scenario, target_function, logging_level=20):
            self.scenario = scenario
            self.target = target_function

        def optimize(self):
            for config in [incumbent, *others]:
                for instance in self.scenario.kwargs["instances"]:
                    cost = self.target(config, instance, seed=0)
                    record["costs"].append((dict(config), instance, cost))
            return incumbent

    fake.Scenario = Scenario
    fake.AlgorithmConfigurationFacade = AlgorithmConfigurationFacade
    monkeypatch.setitem(sys.modules, "smac", fake)
    return record


def _sbox(**kwargs):
    defaults = dict(
        training_instances=[(1, 1), (2, 1)],
        test_instances=[(3, 1)],
        dims=[2],
        budget_factor=10,
    )
    defaults.update(kwargs)
    return BBOB_SBOX(**defaults)


def _mabbob(**kwargs):
    defaults = dict(
        training_instances=[0, 1], test_instances=[2], dims=[2], budget_factor=10
    )
    defaults.update(kwargs)
    return MA_BBOB(**defaults)


def _kerneltuner(monkeypatch, tmp_path, **kwargs):
    from iohblade.benchmarks.kerneltuner import kerneltuner as kt_module

    record = {"runs": []}

    def fake_generate(name, path, strategies, applications, gpus, **kw):
        record["runs"].append(
            {
                "name": name,
                "path": Path(path),
                "applications": [a["name"] for a in applications],
                "gpus": list(gpus),
                "hyperparams": strategies[0]["search_method_hyperparameters"],
            }
        )
        return Path(path) / "experiment.json"

    def fake_scores(path):
        # The score is the hyperparameter "w" (0.25 without it).
        params = {p["name"]: p["value"] for p in record["runs"][-1]["hyperparams"]}
        return {"s": {"score": float(params.get("w", 0.25))}}

    monkeypatch.setattr(kt_module, "generate_experiment_file", fake_generate)
    monkeypatch.setattr(kt_module, "get_strategy_scores", fake_scores)
    problem = kt_module.Kerneltuner(
        gpus=["A100", "MI250X"],
        kernels=["gemm", "hotspot"],
        cache_dir=str(tmp_path / "cache") + "/",
        **kwargs,
    )
    problem.logger_dir = str(tmp_path / "log")
    return problem, record


KT_CODE = "class MyTuner:\n    def __init__(self, w=0.25):\n        self.w = w\n"


# ------------------------------------------------- support/dependencies ---


@pytest.mark.parametrize("factory", [_sbox, _mabbob])
def test_bbob_supports_hpo_and_default_dependencies(factory):
    problem = factory()
    assert type(problem).supports_hpo is True and problem.hpo is False
    deps = " ".join(problem.dependencies).lower()
    assert "smac" not in deps and "swig" not in deps and "configspace" not in deps

    tuned = factory(hpo=True, hpo_trials=7)
    assert tuned.hpo is True and tuned.hpo_trials == 7
    assert tuned.dependencies[-len(HPO_DEPENDENCIES) :] == HPO_DEPENDENCIES
    assert tuned.to_dict()["hpo_trials"] == 7
    assert "hpo" not in problem.to_dict()

    problem.enable_hpo(True)
    assert all(dep in problem.dependencies for dep in HPO_DEPENDENCIES)
    problem.enable_hpo(False)
    assert "smac" not in " ".join(problem.dependencies)


def test_default_trial_counts():
    assert BBOB_SBOX().hpo_trials == 100
    assert MA_BBOB().hpo_trials == 50
    assert BBOB_SBOX().hpo_instances == 12 and MA_BBOB().hpo_instances == 12


def test_kerneltuner_supports_hpo(monkeypatch, tmp_path):
    problem, _ = _kerneltuner(monkeypatch, tmp_path)
    assert type(problem).supports_hpo is True and problem.hpo is False
    deps = " ".join(problem.dependencies).lower()
    assert "smac" not in deps and "swig" not in deps and "configspace" not in deps
    assert problem.hpo_trials == 50

    tuned, _ = _kerneltuner(monkeypatch, tmp_path, hpo=True)
    assert all(dep in tuned.dependencies for dep in HPO_DEPENDENCIES)


def test_configure_hpo_enables_the_benchmarks_without_warning(monkeypatch, tmp_path):
    from .test_hpo import _Method

    kt, _ = _kerneltuner(monkeypatch, tmp_path)
    problems = [_sbox(), _mabbob(), kt]
    notes, warns = hpo.configure_hpo([_Method(hpo=True)], problems)
    assert warns == []
    assert len(notes) == 3
    assert all(p.hpo for p in problems)


def test_bbob_modules_import_without_smac_and_configspace(monkeypatch):
    """The evaluation environment without HPO has neither package."""
    calls = []
    _block(monkeypatch, "smac", calls)
    real_import = builtins.__import__

    def no_configspace(name, *args, **kwargs):
        if name == "ConfigSpace" or name.startswith("ConfigSpace."):
            calls.append(name)
            raise ImportError("No module named 'ConfigSpace' (blocked)")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_configspace)
    for prefix in ("iohblade.benchmarks.BBOB", "iohblade.benchmarks.kerneltuner"):
        for name in list(sys.modules):
            if name.startswith(prefix):
                monkeypatch.delitem(sys.modules, name)
    sbox = importlib.import_module("iohblade.benchmarks.BBOB.bbob_sboxcost")
    importlib.import_module("iohblade.benchmarks.BBOB.mabbob")
    importlib.import_module("iohblade.benchmarks.kerneltuner.kerneltuner")
    problem = sbox.BBOB_SBOX(training_instances=[(1, 1)], dims=[2], budget_factor=10)
    solution = problem.evaluate(Solution(code=AIMED, name="Aimed"))
    assert solution.fitness > 0
    assert calls == []


def test_no_module_imports_smac_outside_hpo():
    root = Path(hpo.__file__).resolve().parent
    offenders = []
    for path in root.rglob("*.py"):
        if path.name == "hpo.py" and path.parent == root:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module]
            if any(n == "smac" or n.startswith("smac.") for n in names):
                offenders.append(f"{path.relative_to(root)}:{node.lineno}")
    assert offenders == []


# ----------------------------------------------------------- BBOB-SBOX ---


def test_sbox_tunes_with_training_instances_and_uses_incumbent(monkeypatch, tmp_path):
    record = _fake_smac(monkeypatch, {"w": 1.0})
    problem = _sbox(hpo=True, hpo_trials=9)
    problem.logger_dir = str(tmp_path)
    solution = problem.evaluate(_solution())

    scenario = record["scenario"]
    assert sorted(scenario["instances"]) == ["f1_i1_d2", "f2_i1_d2"]
    assert scenario["instance_features"]["f2_i1_d2"] == [2.0, 1.0, 2.0]
    assert scenario["n_trials"] == 9
    assert scenario["deterministic"] is True
    assert scenario["output_directory"] == os.path.join(str(tmp_path), "smac")
    assert scenario["name"] == f"{problem.name}-{solution.id}"

    # The target's cost is 1 - AOCC of one run: low for the incumbent.
    costs = {(c["w"], inst): cost for c, inst, cost in record["costs"]}
    alg_cls = {}
    exec(AIMED, alg_cls)
    for (w, inst), cost in costs.items():
        fid, iid, dim = (int(p[1:]) for p in inst.split("_"))
        aocc = problem._run_once(alg_cls["Aimed"], fid, iid, dim, {"w": w})
        assert cost == pytest.approx(1.0 - aocc)
        assert 0.0 <= cost <= 1.0
    assert costs[(1.0, "f1_i1_d2")] < costs[(0.0, "f1_i1_d2")]

    # The incumbent is scored on all training instances and recorded.
    assert solution.get_metadata("incumbent") == {"w": 1.0}
    assert "HPO used (SMAC, 9 trials); scored with incumbent={'w': 1.0}" in (
        solution.feedback
    )
    default = problem.evaluate(Solution(code=AIMED, name="Aimed"))
    assert solution.fitness > default.fitness
    assert solution.fitness == pytest.approx(
        1.0 - sum(costs[(1.0, i)] for i in ("f1_i1_d2", "f2_i1_d2")) / 2
    )


def test_sbox_tuning_subset_is_fixed(monkeypatch):
    record = _fake_smac(monkeypatch, {"w": 1.0}, others=())
    problem = _sbox(
        hpo=True,
        hpo_instances=3,
        training_instances=[(f, 1) for f in range(1, 6)],
    )
    problem.evaluate(_solution())
    first = record["scenario"]["instances"]
    problem.evaluate(_solution())
    assert record["scenario"]["instances"] == first and len(first) == 3


def test_sbox_test_uses_the_search_incumbent(monkeypatch, tmp_path):
    calls = []
    _block(monkeypatch, "smac", calls)  # no tuning on the test instances
    problem = _sbox(hpo=True, ioh_dir=str(tmp_path))
    tuned = _solution()
    tuned.add_metadata("incumbent", {"w": 1.0})
    tuned = problem.test(tuned)
    assert "Scored with the HPO incumbent={'w': 1.0}" in tuned.feedback
    assert calls == []

    untuned = problem.test(_solution())
    assert untuned.fitness < tuned.fitness
    assert "No HPO incumbent" in untuned.feedback


def test_sbox_configspace_without_hpo_uses_defaults(monkeypatch):
    calls = []
    _block(monkeypatch, "smac", calls)
    problem = _sbox()
    solution = problem.evaluate(_solution())
    plain = problem.evaluate(Solution(code=AIMED, name="Aimed"))
    assert calls == []
    assert solution.get_metadata("incumbent") is None
    assert solution.fitness == pytest.approx(plain.fitness)
    assert hpo.UNUSED_CONFIGSPACE_FEEDBACK in solution.feedback
    assert "HPO" not in plain.feedback


@pytest.mark.parametrize("factory", [_sbox, _mabbob])
def test_hpo_without_smac_fails_before_code_is_read(monkeypatch, factory):
    _block(monkeypatch, "smac")
    problem = factory(hpo=True)

    class Watched(Solution):
        code_read = False

        def __getattribute__(self, name):
            if name == "code":
                type(self).code_read = True
            return super().__getattribute__(name)

    solution = Watched(code=AIMED, name="Aimed", configspace=_configspace())
    Watched.code_read = False
    with pytest.raises(HPOUnavailableError):
        problem.evaluate(solution)
    assert Watched.code_read is False


STUB_SMAC = "raise ImportError('smac is not installed (stub)')\n"


def test_sbox_hpo_without_smac_is_a_harness_error(tmp_path):
    """The evaluation script marks the failure as a harness error."""
    stubs = tmp_path / "stubs"
    (stubs / "smac").mkdir(parents=True)
    (stubs / "smac" / "__init__.py").write_text(STUB_SMAC)

    problem = _sbox(hpo=True)
    problem_pkl = tmp_path / "problem.pkl"
    solution_pkl = tmp_path / "solution.pkl"
    marker = tmp_path / "marker.txt"
    problem_pkl.write_bytes(cloudpickle.dumps(problem))
    solution_pkl.write_bytes(cloudpickle.dumps(_solution()))
    script = tmp_path / "run.py"
    script.write_text(
        _build_eval_script(
            problem.imports, problem_pkl, solution_pkl, tmp_path / "out", marker
        )
    )
    root = str(Path(hpo.__file__).resolve().parents[1])
    proc = subprocess.run(
        [sys.executable, str(script)],
        env={**os.environ, "PYTHONPATH": os.pathsep.join([str(stubs), root])},
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert proc.returncode != 0
    assert "HPOUnavailableError" in proc.stderr
    assert marker.exists(), proc.stderr  # harness error, not a candidate one
    feedback = _harness_feedback(proc.stderr)
    assert feedback.startswith(HARNESS_ERROR_PREFIX)


# ------------------------------------------------------------- MA-BBOB ---


def test_mabbob_tunes_and_uses_incumbent(monkeypatch, tmp_path):
    record = _fake_smac(monkeypatch, {"w": 1.0})
    problem = _mabbob(hpo=True)
    solution = problem.evaluate(_solution())

    scenario = record["scenario"]
    assert sorted(scenario["instances"]) == ["m0_d2", "m1_d2"]
    assert scenario["n_trials"] == 50
    costs = {(c["w"], inst): cost for c, inst, cost in record["costs"]}
    alg_cls = {}
    exec(AIMED, alg_cls)
    for (w, inst), cost in costs.items():
        idx, dim = (int(p[1:]) for p in inst.split("_"))
        aocc = problem._run_once(alg_cls["Aimed"], idx, dim, {"w": w})
        assert cost == pytest.approx(1.0 - aocc)
    assert costs[(1.0, "m0_d2")] < costs[(0.0, "m0_d2")]

    assert solution.get_metadata("incumbent") == {"w": 1.0}
    assert "incumbent={'w': 1.0}" in solution.feedback
    default = problem.evaluate(Solution(code=AIMED, name="Aimed"))
    assert solution.fitness > default.fitness

    tested = problem.test(solution, ioh_dir=str(tmp_path / "ioh"))
    assert "Scored with the HPO incumbent" in tested.feedback


def test_mabbob_configspace_without_hpo_uses_defaults(monkeypatch):
    calls = []
    _block(monkeypatch, "smac", calls)
    solution = _mabbob().evaluate(_solution())
    assert calls == []
    assert hpo.UNUSED_CONFIGSPACE_FEEDBACK in solution.feedback


# --------------------------------------------------------- Kerneltuner ---


def test_kerneltuner_tunes_and_uses_incumbent(monkeypatch, tmp_path):
    smac_record = _fake_smac(monkeypatch, {"w": 0.875}, others=({"w": 0.5},))
    problem, record = _kerneltuner(monkeypatch, tmp_path, hpo=True)
    solution = Solution(code=KT_CODE, name="MyTuner", configspace=_configspace())
    solution = problem.evaluate(solution)

    scenario = smac_record["scenario"]
    assert sorted(scenario["instances"]) == sorted(problem.training_instances)
    assert scenario["n_trials"] == 50
    # One kernel-GPU pair per trial, cost = 1 - score.
    tuning_runs = record["runs"][:-1]
    assert len(tuning_runs) == 2 * len(problem.training_instances)
    for run, (config, instance, cost) in zip(tuning_runs, smac_record["costs"]):
        kernel, gpu = instance.rsplit("-", 1)
        assert run["applications"] == [f"{kernel}_milo"] and run["gpus"] == [gpu]
        assert run["hyperparams"] == [{"name": "w", "value": config["w"]}]
        assert cost == pytest.approx(1.0 - config["w"])

    final = record["runs"][-1]
    assert final["applications"] == ["gemm_milo", "hotspot_milo"]
    assert final["hyperparams"] == [{"name": "w", "value": 0.875}]
    assert solution.fitness == pytest.approx(0.875)
    assert solution.get_metadata("incumbent") == {"w": 0.875}
    assert "incumbent={'w': 0.875}" in solution.feedback

    tested = problem.test(solution)
    assert record["runs"][-1]["hyperparams"] == [{"name": "w", "value": 0.875}]
    assert "Scored with the HPO incumbent" in tested.feedback


def test_kerneltuner_without_hpo_uses_defaults(monkeypatch, tmp_path):
    calls = []
    _block(monkeypatch, "smac", calls)
    problem, record = _kerneltuner(monkeypatch, tmp_path)
    solution = Solution(code=KT_CODE, name="MyTuner", configspace=_configspace())
    solution = problem.evaluate(solution)
    assert calls == []
    assert len(record["runs"]) == 1 and record["runs"][0]["hyperparams"] == []
    assert solution.fitness == pytest.approx(0.25)
    assert hpo.UNUSED_CONFIGSPACE_FEEDBACK in solution.feedback


# ----------------------------------------------------------- helpers ---


def test_hpo_instance_subset_and_plain_config():
    import numpy as np

    items = list(range(30))
    subset = hpo.hpo_instance_subset(items, 12)
    assert len(subset) == 12 and subset == hpo.hpo_instance_subset(items, 12)
    assert sorted(hpo.hpo_instance_subset(items, None)) == items
    assert hpo.plain_config({"a": np.float64(0.5), "b": np.int64(2)}) == {
        "a": 0.5,
        "b": 2,
    }
    assert type(hpo.plain_config({"a": np.float64(0.5)})["a"]) is float
    assert hpo.plain_config(None) == {}


def test_evaluation_config_without_problem_hpo():
    problem = types.SimpleNamespace(hpo=False)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert hpo.evaluation_config(problem, Solution()) == ({}, "")
