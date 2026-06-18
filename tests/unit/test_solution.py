import math
import pickle

import numpy as np
import pytest

from iohblade import Fitness, Solution


def test_solution_initialization():
    s = Solution(code="print('Hello')", name="MyAlgo", description="A test algo")
    assert s.code == "print('Hello')"
    assert s.name == "MyAlgo"
    assert s.description == "A test algo"
    assert math.isnan(s.fitness)


def test_solution_set_scores():
    s = Solution()
    s.set_scores(42.0, feedback="OK", error="None")
    assert s.fitness == 42.0
    assert s.feedback == "OK"
    assert s.error == "None"


def test_solution_copy():
    s = Solution(name="Original")
    s2 = s.copy()
    assert s2.name == s.name
    assert s2.id != s.id
    assert s2.parent_ids == [s.id]


def test_solution_to_dict():
    s = Solution(
        code="some code",
        name="TestName",
        description="TestDesc",
    )
    d = s.to_dict()
    assert d["code"] == "some code"
    assert d["name"] == "TestName"
    assert d["description"] == "TestDesc"
    assert "fitness" in d


def test_solution_from_dict():
    data = {
        "id": "some-id",
        "fitness": 123.0,
        "name": "Algo",
        "description": "Desc",
        "code": "Code()",
        "generation": 2,
        "feedback": "Good",
        "error": "None",
        "parent_ids": [],
        "operator": "MockOp",
        "metadata": {"key": "value"},
    }
    s = Solution()
    s.from_dict(data)
    assert s.id == "some-id"
    assert s.fitness == 123.0
    assert s.name == "Algo"
    assert s.description == "Desc"
    assert s.code == "Code()"
    assert s.generation == 2
    assert s.feedback == "Good"
    assert s.error == "None"
    assert s.operator == "MockOp"
    assert s.metadata["key"] == "value"


# ---------------------------------------------------------------------------
# Multi-objective Fitness support
# ---------------------------------------------------------------------------


def test_solution_set_scores_multiobjective():
    """set_scores() should accept a Fitness object."""
    s = Solution()
    f = Fitness({"f1": 0.5, "f2": 1.5})
    s.set_scores(f, feedback="ok")
    assert isinstance(s.fitness, Fitness)
    assert s.fitness["f1"] == 0.5
    assert s.fitness["f2"] == 1.5


def test_solution_to_dict_multiobjective():
    """to_dict() must serialise Fitness as a plain dict."""
    s = Solution()
    s.set_scores(Fitness({"f1": 1.0, "f2": 2.0}))
    d = s.to_dict()
    assert d["fitness"] == {"f1": 1.0, "f2": 2.0}


def test_solution_from_dict_multiobjective():
    """from_dict() must reconstruct Fitness from a dict fitness value."""
    data = {
        "id": "abc",
        "fitness": {"f1": 3.0, "f2": 4.0},
        "name": "MO",
        "description": "",
        "code": "",
        "generation": 0,
        "feedback": "",
        "error": "",
        "parent_ids": [],
        "operator": None,
        "metadata": {},
    }
    s = Solution()
    s.from_dict(data)
    assert isinstance(s.fitness, Fitness)
    assert s.fitness["f1"] == 3.0
    assert s.fitness["f2"] == 4.0


def test_solution_to_dict_from_dict_roundtrip_multiobjective():
    """to_dict → from_dict round-trip preserves Fitness values."""
    s = Solution(name="algo")
    s.set_scores(Fitness({"speed": 0.8, "quality": 0.95}))
    d = s.to_dict()

    s2 = Solution()
    s2.from_dict(d)
    assert isinstance(s2.fitness, Fitness)
    assert s2.fitness["speed"] == pytest.approx(0.8)
    assert s2.fitness["quality"] == pytest.approx(0.95)


def test_solution_pickle_roundtrip_multiobjective():
    """Fitness must survive pickle/unpickle as a Fitness instance, not a plain dict.

    Problem.__call__ pickles solutions via cloudpickle/multiprocessing; without
    __setstate__ reconstructing Fitness, comparisons like
    ``solution.fitness > best.fitness`` would receive a dict and raise TypeError.
    """
    s = Solution(name="pickled")
    s.set_scores(Fitness({"f1": -0.5, "f2": -1.5}))

    s2 = pickle.loads(pickle.dumps(s))

    assert isinstance(s2.fitness, Fitness), (
        "fitness must be a Fitness instance after unpickling, not "
        f"{type(s2.fitness).__name__}"
    )
    assert s2.fitness["f1"] == pytest.approx(-0.5)
    assert s2.fitness["f2"] == pytest.approx(-1.5)
    # Pareto comparison must work without TypeError
    assert not (s2.fitness < s2.fitness)  # equal → not strictly dominated


def test_empty_solution_fitness_validity():
    s = Solution()

    assert not s.fitness_is_valid(), "Empty solution should not have valid solution."

def test_scalar_solution_fitness_validity():
    s = Solution()
    s.set_scores(
        0.5, "Scored 0.5, best known score is 0.991"
    )

    assert s.fitness_is_valid(), "Scalar solution has valid fitness, not invalid."

    s.set_scores(
        float('inf'),
        'Import failure nonepy is not a library.',
        Exception('Cannot find nonepy.')
    )

    assert not s.fitness_is_valid(), "±inf is invalid solution."
    
    s.set_scores(
        float('-inf'),
        'Import failure nonepy is not a library.',
        Exception('Cannot find nonepy.')
    )

    assert not s.fitness_is_valid(), "±inf is invalid solution."

def test_fitness_type_invalidity():
    s = Solution()
    s.set_scores(
        Fitness({
            'distance': 788,
            'fuel': float('inf')
        }),
        "Got Distance = 788 km, and Fuel calculation failed."
    )
    assert not s.fitness_is_valid(), 'Fitness type with invalid values, expects invalidity.'

    s.set_scores(
        Fitness({
            'distance': 788,
            'fuel': 917
        }),
        "Got Distance = 788 km, and Fuel = 917 mL."
    )
    assert s.fitness_is_valid(), 'Fitness type with valid values, expects validity.'
    
    s.set_scores(
        Fitness({
            'distance': 788,
            'fuel': float('nan')
        }),
        "Got Distance = 788 km, and Fuel capture failed."
    )
    assert not s.fitness_is_valid(), 'Fitness type with invalid values, expects invalidity.'
