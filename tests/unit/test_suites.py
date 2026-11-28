"""The default benchmark suites (iohblade.benchmarks.suites)."""

import inspect

import ioh
import pytest

from iohblade.benchmarks import suites
from iohblade.config import load_plan
from iohblade.registry import BuiltinSuite, registry

DEFAULT_SUITES = [
    "bbob_groups",
    "bbob",
    "ma_bbob",
    "continuous_bbo",
    "alphaevolve_math",
    "combinatorial",
    "default",
    "quick",
    "full",
]


# region BBOB and MA-BBOB.
def test_bbob_groups_has_one_problem_per_function_group():
    problems = suites.bbob_groups()
    assert [p.name for p in problems] == [f"BBOB-g{g}-10d" for g in range(1, 6)]
    fids = []
    for group, problem in zip(range(1, 6), problems):
        group_fids = suites.BBOB_GROUPS[group]
        fids += group_fids
        assert problem.problem_type == ioh.ProblemClass.BBOB
        assert problem.specific_group == group
        assert problem.dims == [10] and problem.budget_factor == 2000
        assert problem.training_instances == [
            (f, i) for f in group_fids for i in range(1, 6)
        ]
        assert problem.test_instances == [
            (f, i) for f in group_fids for i in range(6, 16)
        ]
        # 1 ms per function evaluation of the training runs, >= 120 s.
        assert problem.eval_timeout == len(group_fids) * 5 * 10 * 2000 // 1000
    assert sorted(fids) == list(range(1, 25))


def test_bbob_groups_dimension_budget_and_instances_are_parameters():
    problems = suites.bbob_groups(
        dim=5, budget_factor=1000, groups=[2], n_train=2, n_test=3
    )
    (problem,) = problems
    assert problem.name == "BBOB-g2-5d"
    assert problem.dims == [5] and problem.budget_factor == 1000
    assert problem.training_instances == [(f, i) for f in (6, 7, 8, 9) for i in (1, 2)]
    assert problem.test_instances == [(f, i) for f in (6, 7, 8, 9) for i in (3, 4, 5)]
    assert problem.eval_timeout == 120  # the minimum
    assert suites.bbob_groups(eval_timeout=999)[0].eval_timeout == 999


def test_bbob_has_one_problem_per_function():
    problems = suites.bbob()
    assert [p.name for p in problems] == [f"BBOB-f{f}-10d" for f in range(1, 25)]
    for fid, problem in zip(range(1, 25), problems):
        assert problem.problem_type == ioh.ProblemClass.BBOB
        assert problem.specific_fid == fid
        assert problem.dims == [10] and problem.budget_factor == 2000
        assert problem.training_instances == [(fid, i) for i in range(1, 11)]
        assert problem.test_instances == [(fid, i) for i in range(11, 31)]
        assert problem.eval_timeout == 10 * 10 * 2000 // 1000


def test_bbob_functions_dimension_and_instances_are_parameters():
    (problem,) = suites.bbob(
        dim=5, budget_factor=1000, functions=[15], n_train=2, n_test=3
    )
    assert problem.name == "BBOB-f15-5d"
    assert problem.dims == [5] and problem.budget_factor == 1000
    assert problem.training_instances == [(15, 1), (15, 2)]
    assert problem.test_instances == [(15, 3), (15, 4), (15, 5)]


def test_ma_bbob_instances():
    (problem,) = suites.ma_bbob()
    assert problem.name == "MA_BBOB-10d"
    assert problem.dims == [10] and problem.budget_factor == 2000
    assert list(problem.training_instances) == list(range(0, 25))
    assert list(problem.test_instances) == list(range(25, 75))
    assert problem.eval_timeout == 25 * 10 * 2000 // 1000
    (small,) = suites.ma_bbob(dim=2, budget_factor=100, n_train=3, n_test=4)
    assert small.dims == [2] and list(small.test_instances) == [3, 4, 5, 6]


def test_continuous_bbo_shares_dimension_and_budget():
    problems = suites.continuous_bbo(dim=20, budget_factor=500)
    assert len(problems) == 5 + 24 + 1
    assert all(p.dims == [20] and p.budget_factor == 500 for p in problems)
    names = [p.name for p in problems]
    assert names[0] == "BBOB-g1-20d" and names[5] == "BBOB-f1-20d"
    assert names[-1] == "MA_BBOB-20d"


@pytest.mark.parametrize(
    "call",
    [
        lambda: suites.bbob_groups(dim=0),
        lambda: suites.bbob_groups(budget_factor=1.5),
        lambda: suites.bbob_groups(groups=[6]),
        lambda: suites.bbob_groups(groups=[]),
        lambda: suites.bbob_groups(eval_timeout=-1),
        lambda: suites.bbob(functions=[25]),
        lambda: suites.bbob(functions=[1.0]),
        lambda: suites.bbob(functions=[True]),
        lambda: suites.bbob_groups(groups=[2.0]),
        lambda: suites.bbob(functions=[]),
        lambda: suites.bbob(n_test=0),
        lambda: suites.ma_bbob(n_train=600, n_test=500),
        lambda: suites.ma_bbob(n_test=0),
    ],
)
def test_invalid_continuous_settings_are_rejected(call):
    with pytest.raises(ValueError):
        call()


# endregion


# region Other suites.
def test_alphaevolve_math_problems():
    problems = suites.alphaevolve_math()
    assert len(problems) == 17
    assert len({p.name for p in problems}) == 17
    assert all(p.eval_timeout == 600 for p in problems)
    assert "sums_vs_differences" in {p.name for p in problems}
    with_best = suites.alphaevolve_math(use_best=True, eval_timeout=900)
    assert all(p.eval_timeout == 900 for p in with_best)
    # The best known solutions change the examples of some problems.
    assert [p.example_prompt for p in with_best] != [p.example_prompt for p in problems]


def test_combinatorial_problems():
    names = [p.name for p in suites.combinatorial()]
    assert names == [
        "EuclidianSteinerTree-n50",
        "EuclidianSteinerTree-n100",
        "EuclidianSteinerTree-n250",
        "graph_coloring-gcol1",
        "TSP-A-n53-k7",
        "CVRP-A-n53-k7",
    ]


def test_quick_and_full():
    quick = suites.quick()
    assert [p.name for p in quick][:2] == ["BBOB-g1-5d", "BBOB-g4-5d"]
    assert len(quick) == 5
    full = suites.full()
    assert len(full) == 55 and len({p.name for p in full}) == 55
    assert {f"BBOB-f{f}-10d" for f in range(1, 25)} <= {p.name for p in full}
    with_mm = suites.full(include_matrix_multiplication=True)
    assert len(with_mm) == 55 + 54


def test_default_collection():
    names = [p.name for p in suites.default()]
    groups = [p.name for p in suites.bbob_groups()]
    maths = [p.name for p in suites.alphaevolve_math()]
    combinatorial = [p.name for p in suites.combinatorial()]
    assert names == groups + maths + combinatorial
    assert len(names) == 28


def test_hyphenated_suite_keys_resolve():
    assert ("suite", "bbob-groups") in registry
    assert registry.resolve("suite", "bbob-groups") is suites.bbob_groups


@pytest.mark.parametrize("key", DEFAULT_SUITES)
def test_default_suites_are_registered_with_timeouts(key):
    assert key in {s.value for s in BuiltinSuite}
    problems = registry.resolve("suite", key)()
    assert problems
    assert all(p.eval_timeout >= 120 for p in problems)


# endregion


# region Flags, timeouts and names.
@pytest.mark.parametrize("key", [s.value for s in BuiltinSuite])
def test_suite_flags_are_consistent(key):
    params = inspect.signature(registry.resolve("suite", key)).parameters
    if "use_best" in params:
        assert params["use_best"].default is False
    assert "get_best" not in params or params["get_best"].kind.name == "KEYWORD_ONLY"
    assert all(p.default is not inspect.Parameter.empty for p in params.values())


def test_get_best_is_an_alias_of_use_best():
    from iohblade.benchmarks.packing.get_packing_problems import (
        get_hexagon_packing_problems,
    )

    alias = get_hexagon_packing_problems(get_best=True)
    new = get_hexagon_packing_problems(use_best=True)
    assert [p.example_prompt for p in alias] == [p.example_prompt for p in new]


@pytest.mark.parametrize("key", sorted(registry.keys("problem")))
def test_problem_default_timeout_is_at_least_120_seconds(key):
    try:
        cls = registry.resolve("problem", key)
    except Exception:
        pytest.skip(f"{key} needs an optional dependency group")
    params = inspect.signature(cls).parameters
    assert "eval_timeout" in params, f"{key} has no eval_timeout parameter"
    default = params["eval_timeout"].default
    if default is None:  # computed from the budget: check a default instance
        default = cls().eval_timeout
    assert default >= 120


def test_sum_vs_difference_name_and_reference():
    from iohblade.benchmarks.number_theory.sums_vs_differences import SumDifference

    problem = SumDifference(eval_timeout=200)
    assert problem.name == "sums_vs_differences"
    assert problem.best_known == pytest.approx(1.147989)
    assert problem.eval_timeout == 200
    assert SumDifference(best_score=1.2).best_known == 1.2  # old name


def test_suite_params_from_yaml(tmp_path):
    config = tmp_path / "experiment.yml"
    config.write_text("""
version: 1
experiment: {output_dir: out, runs: 1, budget: 2}
llms: {d: {provider: dummy}}
methods: [{type: random_search, llm: d}]
problems:
  - suite: bbob-groups
    params: {dim: 5, budget_factor: 100, groups: [1, 5]}
  - suite: bbob
    params: {dim: 5, functions: [3]}
  - suite: alphaevolve_math
    params: {eval_timeout: 300}
""")
    plan = load_plan(str(config))
    names = [p.name for p in plan.problems]
    assert names[:3] == ["BBOB-g1-5d", "BBOB-g5-5d", "BBOB-f3-5d"]
    assert all(p.eval_timeout == 300 for p in plan.problems[3:])


# endregion


# region use_best examples.
def _evaluate_example(problem):
    import re

    from iohblade.solution import Solution
    from iohblade.utils import first_class_name

    code = re.search(r"```python\n(.*?)\n```", problem.example_prompt, re.S).group(1)
    return problem.evaluate(Solution(code=code, name=first_class_name(code)))


@pytest.mark.parametrize(
    "index", range(17), ids=lambda i: f"math-{i}"  # 17 AlphaEvolve problems
)
def test_use_best_example_reaches_the_best_known_score(index):
    """With use_best, the example returns the best known configuration
    unchanged, so it scores the best known value (which is rounded to a few
    digits for some problems)."""
    problem = suites.alphaevolve_math(use_best=True)[index]
    result = _evaluate_example(problem)
    assert not result.error, f"{problem.name}: {result.error}"
    assert float(result.fitness) == pytest.approx(
        float(problem.best_known), rel=2e-3
    ), f"{problem.name}: {result.fitness} vs best known {problem.best_known}"


# endregion
