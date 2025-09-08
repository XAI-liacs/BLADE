import numpy as np
import pytest

from iohblade import Solution


def test_solution_initialization():
    s = Solution(code="print('Hello')", name="MyAlgo", description="A test algo")
    assert s.code == "print('Hello')"
    assert s.name == "MyAlgo"
    assert s.description == "A test algo"
    assert s.fitness == -np.inf


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
