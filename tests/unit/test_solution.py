import math

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
