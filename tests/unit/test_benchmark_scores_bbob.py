"""Regression tests for scoring bugs in the BBOB, MA-BBOB, photonics and
AutoML benchmarks (issue #203).

PyMoosh and openml are optional; when they are missing, small stub
modules stand in for them so the scoring logic can still be tested.
"""

import copy
import importlib
import math
import pickle
import sys
import types

import numpy as np
import pytest

from iohblade.solution import Solution

# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

_PHOTONICS_PREFIX = "iohblade.benchmarks.photonics"
_AUTOML_PREFIX = "iohblade.benchmarks.automl"


def _drop_modules(prefix):
    for name in list(sys.modules):
        if name == prefix or name.startswith(prefix + "."):
            sys.modules.pop(name)


def _installed(name):
    try:
        importlib.import_module(name)
        return True
    except ImportError:
        return False


def _fake_pymoosh():
    """Minimal PyMoosh: a reflection coefficient that depends on the stack."""
    mod = types.ModuleType("PyMoosh")

    class Structure:
        def __init__(self, materials, stack, thicknesses, verbose=False):
            self.materials = materials
            self.stack = stack
            self.thicknesses = np.asarray(thicknesses, dtype=float)

    def coefficient(struct, wavelength, angle, polarization):
        eps = [m for m in struct.materials if isinstance(m, (int, float))]
        value = complex(
            1.0 + polarization * sum(eps),
            polarization * float(np.sum(struct.thicknesses)) / wavelength,
        )
        return value, None, None, None

    def coefficient_I(struct, wavelength, angle, polarization):
        reflectivity = 0.5 + 0.4 * np.sin(float(np.sum(struct.thicknesses)) / 100)
        return None, reflectivity

    mod.Structure = Structure
    mod.coefficient = coefficient
    mod.coefficient_I = coefficient_I
    return mod


@pytest.fixture
def photonics_module(monkeypatch):
    """Import the photonics benchmark, stubbing PyMoosh if it is missing.

    pyGDM2 is not needed: only the plasmonic problem imports it, lazily.
    """
    if not _installed("PyMoosh"):
        monkeypatch.setitem(sys.modules, "PyMoosh", _fake_pymoosh())
    _drop_modules(_PHOTONICS_PREFIX)
    try:
        yield importlib.import_module(f"{_PHOTONICS_PREFIX}.photonics")
    finally:
        _drop_modules(_PHOTONICS_PREFIX)


@pytest.fixture
def automl_module(monkeypatch):
    """Import the AutoML benchmark, stubbing openml if it is missing."""
    if not _installed("openml"):
        fake = types.ModuleType("openml")
        fake.tasks = types.SimpleNamespace(get_task=None)
        monkeypatch.setitem(sys.modules, "openml", fake)
    if not _installed("smac"):
        # smac (HPO only) is missing, or fails to import with this sklearn
        fake = types.ModuleType("smac")
        fake.AlgorithmConfigurationFacade = None
        fake.Scenario = None
        monkeypatch.setitem(sys.modules, "smac", fake)
    _drop_modules(_AUTOML_PREFIX)
    try:
        yield importlib.import_module(f"{_AUTOML_PREFIX}.automl")
    finally:
        _drop_modules(_AUTOML_PREFIX)


# A candidate that overspends its budget on exactly one run: the first run
# after the small 100-evaluation smoke test.
OVERSPEND_ONCE = """
import numpy as np

class OverspendOnce:
    runs = 0

    def __init__(self, budget=100, dim=2):
        self.budget = budget
        self.dim = dim
        type(self).runs += 1
        self.overspend = type(self).runs == 2

    def __call__(self, func):
        n = self.budget + 10 if self.overspend else self.budget
        f_opt, x_opt = np.inf, None
        for _ in range(n):
            x = np.random.uniform(func.bounds.lb, func.bounds.ub)
            f = func(x)
            if f < f_opt:
                f_opt, x_opt = f, x
        return f_opt, x_opt
"""

RANDOM_SEARCH = """
import numpy as np

class RandomSearch:
    def __init__(self, budget=100, dim=2):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        f_opt, x_opt = np.inf, None
        for _ in range(self.budget):
            x = np.random.uniform(func.bounds.lb, func.bounds.ub)
            f = func(x)
            if f < f_opt:
                f_opt, x_opt = f, x
        return f_opt, x_opt
"""


# --------------------------------------------------------------------------
# 1. BBOB-SBOX default instances
# --------------------------------------------------------------------------


def test_sbox_default_training_and_test_instances_are_disjoint():
    from iohblade.benchmarks.BBOB.bbob_sboxcost import BBOB_SBOX

    problem = BBOB_SBOX()
    train = set(problem.training_instances)
    test = set(problem.test_instances)
    assert train.isdisjoint(test)
    # 24 functions x 5 training iids and x 10 test iids
    assert len(train) == 24 * 5
    assert len(test) == 24 * 10


# --------------------------------------------------------------------------
# 2. Over-budget runs must not skip the remaining instances
# --------------------------------------------------------------------------


def test_mabbob_over_budget_scores_every_instance():
    from iohblade.benchmarks.BBOB.mabbob import MA_BBOB

    problem = MA_BBOB(training_instances=[0, 1, 2], dims=[2], budget_factor=20)
    solution = Solution(code=OVERSPEND_ONCE, name="OverspendOnce")
    problem.evaluate(solution)

    aucs = solution.get_metadata("aucs")
    assert len(aucs) == 3
    assert all(0.0 <= a <= 1.0 for a in aucs)
    assert solution.fitness == pytest.approx(np.mean(aucs))


def test_photonics_over_budget_scores_every_run(photonics_module):
    problem = photonics_module.Photonics(
        logger=None, problem_type="ellipsometry", budget_factor=5, seeds=3
    )
    solution = Solution(code=OVERSPEND_ONCE, name="OverspendOnce")
    problem.evaluate(solution)

    aucs = solution.get_metadata("aucs")
    assert len(aucs) == 3
    assert solution.fitness == pytest.approx(np.mean(aucs))


def test_photonics_does_not_need_pygdm2(photonics_module, monkeypatch):
    # Only the plasmonic problem uses pyGDM2; it imports it lazily.
    assert "pyGDM2" not in sys.modules
    problem = photonics_module.Photonics(logger=None, problem_type="bragg")
    assert not any("pygdm2" in d.lower() for d in problem.dependencies)

    plasmonic = importlib.import_module(
        f"{_PHOTONICS_PREFIX}.photonics_instances.problems.plasmonic_nanostructure"
    )
    monkeypatch.setitem(sys.modules, "pyGDM2", None)  # import raises
    with pytest.raises(ImportError, match=r"pip install pyGDM2==1\.1\.12"):
        plasmonic.plasmonic_nanostructure(element_sim=None, method="lu")


# --------------------------------------------------------------------------
# 3. Photonics: deterministic reference and per-problem AOC bounds
# --------------------------------------------------------------------------


def test_ellipsometry_reference_is_identical_across_rebuilds(photonics_module):
    problem = photonics_module.Photonics(logger=None, problem_type="ellipsometry")
    x = [2.0, 100.0]

    np.random.seed(1)
    first = copy.deepcopy(problem)
    np.random.seed(2)
    second = copy.deepcopy(problem)
    third = pickle.loads(pickle.dumps(problem))

    costs = [p.problem(x) for p in (first, second, third)]
    assert costs[0] == pytest.approx(costs[1])
    assert costs[0] == pytest.approx(costs[2])
    # the original keeps its problem after being deep-copied
    assert problem.problem is not None
    assert problem.problem(x) == pytest.approx(costs[0])
    assert problem.training_instances == [problem.problem]


def test_ellipsometry_reference_structure_is_seeded(photonics_module):
    from iohblade.benchmarks.photonics.photonics_instances.problems.ellipsometry_inverse import (
        ellipsometry,
    )

    args = (1.0, "Gold", 1, 50, 150, 1.1, 3, np.linspace(400, 800, 5), 0.7)
    np.random.seed(1)
    a = ellipsometry(*args)
    np.random.seed(2)
    b = ellipsometry(*args)
    np.testing.assert_allclose(a.ref_ellipso, b.ref_ellipso)
    np.testing.assert_allclose(a.x_ref, b.x_ref)


def test_photonics_config_records_reference_seed(photonics_module):
    problem = photonics_module.Photonics(
        logger=None, problem_type="ellipsometry", reference_seed=7
    )
    assert problem.to_dict()["reference_seed"] == 7
    assert problem.get_config()["config"]["reference_seed"] == 7


@pytest.mark.parametrize(
    "problem_type, lower, upper",
    [("ellipsometry", 1e-8, 40.0), ("bragg", 0.1648, 1.0)],
)
def test_photonics_aoc_logger_gets_problem_bounds(
    photonics_module, monkeypatch, problem_type, lower, upper
):
    created = []
    real_logger = photonics_module.aoc_logger

    def recording_logger(budget, *args, **kwargs):
        created.append((budget, kwargs.get("lower"), kwargs.get("upper")))
        return real_logger(budget, *args, **kwargs)

    monkeypatch.setattr(photonics_module, "aoc_logger", recording_logger)
    problem = photonics_module.Photonics(
        logger=None, problem_type=problem_type, budget_factor=2, seeds=1
    )
    budget = problem.problem.meta_data.n_variables * 2
    problem.evaluate(Solution(code=RANDOM_SEARCH, name="RandomSearch"))

    final = [c for c in created if c[0] == budget]
    assert final == [(budget, lower, upper)]


def test_photonics_runs_are_seeded(photonics_module):
    problem = photonics_module.Photonics(
        logger=None, problem_type="ellipsometry", budget_factor=5, seeds=2
    )
    a = problem.evaluate(Solution(code=RANDOM_SEARCH, name="RandomSearch"))
    b = problem.evaluate(Solution(code=RANDOM_SEARCH, name="RandomSearch"))
    assert a.get_metadata("aucs") == b.get_metadata("aucs")


# --------------------------------------------------------------------------
# 4. AutoML: binary AUC and failed splits
# --------------------------------------------------------------------------

GOOD_PIPELINE = """
from sklearn.linear_model import LogisticRegression

class GoodPipeline:
    def __init__(self, X, y):
        self.model = LogisticRegression(max_iter=1000).fit(X, y)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def __call__(self, X):
        return self.model.predict(X)
"""

# Works on the first split only, then raises on every other split.
FLAKY_PIPELINE = """
from sklearn.linear_model import LogisticRegression

class FlakyPipeline:
    fits = 0

    def __init__(self, X, y):
        type(self).fits += 1
        if type(self).fits > 1:
            raise RuntimeError("fails on this split")
        self.model = LogisticRegression(max_iter=1000).fit(X, y)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def __call__(self, X):
        return self.model.predict(X)
"""

BROKEN_PIPELINE = """
class BrokenPipeline:
    def __init__(self, X, y):
        raise RuntimeError("always fails")

    def __call__(self, X):
        return X
"""

NAN_REGRESSOR = """
import numpy as np

class NanRegressor:
    def __init__(self, X, y):
        self.mean = float(np.mean(y))

    def __call__(self, X):
        out = np.full(len(X), self.mean)
        out[0] = np.nan
        return out
"""

MEAN_REGRESSOR = """
import numpy as np

class MeanRegressor:
    def __init__(self, X, y):
        self.mean = float(np.mean(y))

    def __call__(self, X):
        return np.full(len(X), self.mean)
"""


def _binary_data():
    from sklearn.datasets import make_classification

    return make_classification(
        n_samples=200, n_features=5, n_informative=3, random_state=0
    )


def _automl_problem(module, metric, X, y, n_splits=4):
    """An AutoML problem on synthetic data with ``n_splits`` search splits."""
    from sklearn.model_selection import KFold

    problem = module.AutoML(openml_task_id=None)
    problem.eval_name = metric
    problem.split_info = {"n_repeats": 1, "n_folds": n_splits, "n_samples": 1}
    folds = list(KFold(n_splits=n_splits, shuffle=True, random_state=0).split(X))

    def split(repeat, fold, sample):
        tr, ev = folds[fold]
        return X[tr], X[ev], y[tr], y[ev]

    problem.prepare_inner_validation_split = split
    problem.prepare_split = split
    return problem


def test_automl_roc_auc_binary_predict_proba(automl_module):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score

    X, y = _binary_data()
    model = LogisticRegression(max_iter=1000).fit(X, y)
    proba = model.predict_proba(X)
    assert proba.shape == (len(y), 2)

    expected = roc_auc_score(y, proba[:, 1])
    assert automl_module._roc_auc(y, proba) == pytest.approx(expected)

    problem = _automl_problem(automl_module, "area_under_roc_curve", X, y)
    assert problem._score_algorithm(model, X, y, y) == pytest.approx(expected)


def test_automl_roc_auc_multiclass_uses_ovr(automl_module):
    from sklearn.datasets import make_classification
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score

    X, y = make_classification(
        n_samples=300, n_features=6, n_informative=4, n_classes=3, random_state=0
    )
    proba = LogisticRegression(max_iter=1000).fit(X, y).predict_proba(X)
    expected = roc_auc_score(y, proba, multi_class="ovr")
    assert automl_module._roc_auc(y, proba) == pytest.approx(expected)


def test_automl_binary_auc_search_has_no_failed_splits(automl_module):
    X, y = _binary_data()
    problem = _automl_problem(automl_module, "area_under_roc_curve", X, y)
    solution = problem._evaluate_search(
        Solution(code=GOOD_PIPELINE, name="GoodPipeline")
    )
    assert solution.get_metadata("search_failed_splits") == 0
    assert 0.5 < solution.fitness <= 1.0


@pytest.mark.parametrize("metric", ["predictive_accuracy", "area_under_roc_curve"])
def test_automl_failed_splits_count_as_worst_score(automl_module, metric):
    X, y = _binary_data()
    problem = _automl_problem(automl_module, metric, X, y, n_splits=4)

    good = problem._evaluate_search(Solution(code=GOOD_PIPELINE, name="GoodPipeline"))
    flaky = problem._evaluate_search(
        Solution(code=FLAKY_PIPELINE, name="FlakyPipeline")
    )

    scores = flaky.get_metadata("search_scores")
    assert len(scores) == 4
    assert flaky.get_metadata("search_failed_splits") == 3
    assert scores[1:] == [0.0, 0.0, 0.0]
    assert flaky.fitness == pytest.approx(np.mean(scores))
    assert flaky.fitness < good.fitness
    # some splits worked, so this is a (bad) score, not an error
    assert flaky.error == ""


def test_automl_all_splits_failed_gives_failure_fitness(automl_module):
    X, y = _binary_data()
    problem = _automl_problem(automl_module, "predictive_accuracy", X, y)
    solution = problem._evaluate_search(
        Solution(code=BROKEN_PIPELINE, name="BrokenPipeline")
    )
    assert solution.fitness == problem.failure_fitness() == -math.inf
    assert solution.error


def test_automl_nan_predictions_fail_error_metric(automl_module):
    from sklearn.datasets import make_regression

    X, y = make_regression(n_samples=120, n_features=4, random_state=0)
    problem = _automl_problem(automl_module, "root_mean_squared_error", X, y)

    ok = problem._evaluate_search(Solution(code=MEAN_REGRESSOR, name="MeanRegressor"))
    nan = problem._evaluate_search(Solution(code=NAN_REGRESSOR, name="NanRegressor"))

    assert math.isfinite(ok.fitness) and ok.fitness < 0
    assert nan.get_metadata("search_failed_splits") == 4
    assert nan.fitness == -math.inf
    assert nan.fitness < ok.fitness
