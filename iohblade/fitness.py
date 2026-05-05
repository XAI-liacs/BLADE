import math


class Fitness:
    """
    A class for multi-objective fitness management.
    Meant for easy comparison between fitness values.

    ``Note``: Do NOT use ``sort`` on this value — sorting makes certain
    assumptions that cannot be guaranteed by multi-objective fitnesses.

    ``Usage``: Initialise with a ``dict[str, float]`` multi-objective fitness.
    Comparisons are defined in terms of Pareto dominance (minimisation by
    default, i.e. lower is better):

    * ``a < b``  : ``a`` strictly dominates ``b`` in minimisation; vice-versa in maximisation.
    * ``a > b``  : ``a`` strictly dominates ``b`` in maximisation; vice-versa in minimisation.
    * ``a <= b`` : ``a`` dominates or is on the same Pareto-front as ``b`` (minimisation).
    * ``a >= b`` : ``a`` dominates or is on the same Pareto-front as ``b`` (maximisation).
    * ``a == b`` : ``a`` and ``b`` have exactly the same fitness on every objective.

    :param value: Mapping from objective name to objective value.  Pass
        ``None`` (or omit) to create an empty Fitness object.
    :type value: dict[str, float] | None
    """

    def __init__(self, value: dict[str, float] | None = None):
        if value is None:
            self._fitness: dict[str, float] = {}
        else:
            self._fitness = value.copy()

    def keys(self):
        """Return the objective names stored in this Fitness object."""
        return self._fitness.keys()

    def __getitem__(self, key):
        return self._fitness.get(key, float("nan"))

    def __setitem__(self, key: str, value: float):
        self._fitness[key] = value

    def _dominates(self, other: "Fitness") -> tuple[bool, bool]:
        better_or_equal = all(self[k] <= other[k] for k in self.keys())
        strictly_better = any(self[k] < other[k] for k in self.keys())
        return better_or_equal, strictly_better

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Fitness):
            return False
        return self._fitness == other._fitness

    def __lt__(self, other: "Fitness") -> bool:
        if not isinstance(other, Fitness):
            return NotImplemented
        be, sb = self._dominates(other)
        return be and sb

    def __gt__(self, other: "Fitness") -> bool:
        if not isinstance(other, Fitness):
            return NotImplemented
        be, sb = other._dominates(self)
        return be and sb

    def __le__(self, other: "Fitness") -> bool:
        if not isinstance(other, Fitness):
            return NotImplemented
        be, sb = self._dominates(other)
        return be or sb

    def __ge__(self, other: "Fitness") -> bool:
        if not isinstance(other, Fitness):
            return NotImplemented
        be, sb = other._dominates(self)
        return be or sb

    def to_vector(self) -> list[float]:
        """Return objective values as a sorted list (sorted by objective name)."""
        vector = []
        for objective in sorted(self.keys()):
            vector.append(self[objective])
        return vector

    def to_dict(self) -> dict[str, float]:
        """Return a plain dict copy of the objective values."""
        return self._fitness.copy()

    @classmethod
    def from_dict(cls, data: dict[str, float]) -> "Fitness":
        """Construct a :class:`Fitness` from a plain dict (e.g. after ``json.loads``)."""
        return cls(value=data)

    def __float__(self) -> float:
        """Scalar summary: mean of all objective values, or NaN if any objective is NaN."""
        if not self._fitness:
            return math.nan
        return (
            math.nan
            if any(math.isnan(value) for value in self._fitness.values())
            else sum(self.to_vector()) / len(self._fitness)
        )

    def __repr__(self) -> str:
        parts = ", ".join(f"{k}: {v}" for k, v in self._fitness.items())
        return parts if parts else "Fitness()"
