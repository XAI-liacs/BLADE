"""Tests for iohblade.fitness.Fitness."""

import math

import pytest

from iohblade.fitness import Fitness


# ---------------------------------------------------------------------------
# Construction and basic attribute access
# ---------------------------------------------------------------------------


def test_empty_fitness():
    f = Fitness()
    assert list(f.keys()) == []
    assert math.isnan(float(f))


def test_fitness_from_dict():
    f = Fitness({"f1": 1.0, "f2": 2.0})
    assert f["f1"] == 1.0
    assert f["f2"] == 2.0


def test_fitness_missing_key_returns_nan():
    f = Fitness({"f1": 1.0})
    assert math.isnan(f["missing"])


def test_fitness_setitem():
    f = Fitness()
    f["f1"] = 3.0
    assert f["f1"] == 3.0


def test_to_dict_roundtrip():
    data = {"alpha": 0.5, "beta": 1.5}
    f = Fitness(data)
    assert f.to_dict() == data


def test_from_dict_classmethod():
    data = {"obj1": 10.0, "obj2": 20.0}
    f = Fitness.from_dict(data)
    assert f["obj1"] == 10.0
    assert f["obj2"] == 20.0


def test_to_vector_sorted():
    f = Fitness({"b": 2.0, "a": 1.0})
    # to_vector sorts by objective name
    assert f.to_vector() == [1.0, 2.0]


# ---------------------------------------------------------------------------
# __float__
# ---------------------------------------------------------------------------


def test_float_mean():
    f = Fitness({"f1": 2.0, "f2": 4.0})
    assert float(f) == pytest.approx(3.0)


def test_float_nan_if_any_nan():
    f = Fitness({"f1": 1.0, "f2": float("nan")})
    assert math.isnan(float(f))


def test_float_empty_is_nan():
    f = Fitness()
    assert math.isnan(float(f))


# ---------------------------------------------------------------------------
# Equality
# ---------------------------------------------------------------------------


def test_eq_same():
    assert Fitness({"f1": 1.0, "f2": 2.0}) == Fitness({"f1": 1.0, "f2": 2.0})


def test_eq_different_values():
    assert Fitness({"f1": 1.0}) != Fitness({"f1": 2.0})


def test_eq_non_fitness():
    assert Fitness({"f1": 1.0}) != 1.0


# ---------------------------------------------------------------------------
# Strict dominance  (a < b  means a strictly dominates b, minimisation)
# ---------------------------------------------------------------------------


def test_strict_dominance_lt():
    a = Fitness({"f1": 1.0, "f2": 1.0})
    b = Fitness({"f1": 2.0, "f2": 2.0})
    assert a < b
    assert not b < a


def test_no_dominance_incomparable():
    # a is better on f1, b is better on f2 → neither dominates
    a = Fitness({"f1": 1.0, "f2": 3.0})
    b = Fitness({"f1": 3.0, "f2": 1.0})
    assert not (a < b)
    assert not (b < a)


def test_strict_dominance_requires_at_least_one_strictly_better():
    # a == b on all objectives → neither strictly dominates
    a = Fitness({"f1": 1.0, "f2": 2.0})
    b = Fitness({"f1": 1.0, "f2": 2.0})
    assert not (a < b)
    assert not (b < a)


# ---------------------------------------------------------------------------
# Weak dominance  (a <= b means a dominates or is on same Pareto front)
# ---------------------------------------------------------------------------


def test_weak_dominance_le_equal():
    a = Fitness({"f1": 1.0, "f2": 2.0})
    b = Fitness({"f1": 1.0, "f2": 2.0})
    assert a <= b
    assert b <= a


def test_weak_dominance_le_dominates():
    a = Fitness({"f1": 1.0, "f2": 1.0})
    b = Fitness({"f1": 2.0, "f2": 2.0})
    assert a <= b
    assert not (b <= a)


def test_weak_dominance_le_incomparable():
    a = Fitness({"f1": 1.0, "f2": 3.0})
    b = Fitness({"f1": 3.0, "f2": 1.0})
    # a and b are incomparable (each is better on one objective).
    # __le__ returns True when a is better-or-equal on ALL objectives (be)
    # OR strictly better on SOME objective (sb).
    # For incomparable solutions sb=True for both directions, so both hold.
    assert a <= b  # a is better on f1
    assert b <= a  # b is better on f2


# ---------------------------------------------------------------------------
# Maximisation direction  (a > b  means a strictly dominates b in max)
# ---------------------------------------------------------------------------


def test_gt_maximisation():
    a = Fitness({"f1": 3.0, "f2": 3.0})
    b = Fitness({"f1": 1.0, "f2": 1.0})
    assert a > b
    assert not b > a


def test_ge_maximisation_equal():
    a = Fitness({"f1": 2.0})
    assert a >= a


# ---------------------------------------------------------------------------
# __repr__
# ---------------------------------------------------------------------------


def test_repr_non_empty():
    f = Fitness({"x": 1.5, "y": 2.5})
    r = repr(f)
    assert "x" in r and "1.5" in r


def test_repr_empty():
    f = Fitness()
    assert repr(f) == "Fitness()"


# ---------------------------------------------------------------------------
# Immutability of the input dict (value is copied)
# ---------------------------------------------------------------------------


def test_init_copies_dict():
    d = {"f1": 1.0}
    f = Fitness(d)
    d["f1"] = 99.0
    assert f["f1"] == 1.0
