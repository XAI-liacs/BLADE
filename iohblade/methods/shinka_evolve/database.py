# Adapted for BLADE (iohblade) from ShinkaEvolve 0.0.8
# (https://github.com/SakanaAI/ShinkaEvolve, shinka/database), Copyright
# Sakana AI, licensed under the Apache License, Version 2.0
# (http://www.apache.org/licenses/LICENSE-2.0). Changes: an in-memory
# database instead of SQLite, scores are maximised internally (BLADE
# converts its fitness with the problem's direction), and randomness comes
# from numpy's global generator, which the BLADE experiment seeds per run.
"""The ShinkaEvolve programs database: islands, archive and sampling."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ...solution import Solution

PARENT_SELECTIONS = ("weighted", "power_law", "beam_search")


@dataclass(eq=False)
class Program:
    """A program on an island.

    ``score`` is maximised; ``None`` marks an incorrect (failed) program.
    The initial program is copied to every island: copies share the
    solution and are not evaluated again.
    """

    solution: Solution
    score: float | None
    island: int
    generation: int
    children: int = 0
    embedding: np.ndarray | None = field(default=None, repr=False)

    @property
    def correct(self) -> bool:
        return self.score is not None

    @property
    def id(self) -> str:
        return self.solution.id


@dataclass
class Sample:
    """What a generation starts from."""

    island: int
    parent: Program
    inspirations: list[Program]
    fix: bool = False


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


class ProgramDatabase:
    """Programs organised on islands, with one global archive.

    Args:
        num_islands: Number of islands.
        archive_size: Maximum number of correct programs in the archive.
        num_archive_inspirations: Archive programs shown in a prompt.
        num_top_k_inspirations: Best programs shown in a prompt.
        elite_selection_ratio: Share of the archive inspirations taken from
            the island's elites.
        parent_selection: ``weighted``, ``power_law`` or ``beam_search``.
        parent_selection_lambda: Sharpness of the weighted selection.
        exploitation_alpha: Exponent of the power-law selection.
        num_beams: Children of a beam-search parent before moving on.
        migration_interval: Generations between migrations.
        migration_rate: Share of an island's programs that migrate (0: off).
        island_elitism: Keep an island's best program from migrating.
    """

    def __init__(
        self,
        num_islands: int = 2,
        archive_size: int = 40,
        num_archive_inspirations: int = 1,
        num_top_k_inspirations: int = 1,
        elite_selection_ratio: float = 0.3,
        parent_selection: str = "weighted",
        parent_selection_lambda: float = 10.0,
        exploitation_alpha: float = 1.0,
        num_beams: int = 5,
        migration_interval: int = 10,
        migration_rate: float = 0.0,
        island_elitism: bool = True,
    ):
        if parent_selection not in PARENT_SELECTIONS:
            raise ValueError(
                f"parent_selection must be one of {PARENT_SELECTIONS} "
                f"(got {parent_selection!r})."
            )
        self.num_islands = num_islands
        self.archive_size = archive_size
        self.num_archive_inspirations = num_archive_inspirations
        self.num_top_k_inspirations = num_top_k_inspirations
        self.elite_selection_ratio = elite_selection_ratio
        self.parent_selection = parent_selection
        self.parent_selection_lambda = parent_selection_lambda
        self.exploitation_alpha = exploitation_alpha
        self.num_beams = num_beams
        self.migration_interval = migration_interval
        self.migration_rate = migration_rate
        self.island_elitism = island_elitism
        self.programs: list[Program] = []
        self.archive: list[Program] = []

    # region Adding programs.
    def add_initial(self, solution: Solution, score: float | None, embedding=None):
        """Add the initial program to island 0 and a copy to every island."""
        programs = []
        for island in range(self.num_islands):
            program = Program(solution, score, island, 0, embedding=embedding)
            self._store(program)
            programs.append(program)
        return programs[0]

    def add(
        self,
        solution: Solution,
        score: float | None,
        parent: Program,
        generation: int,
        embedding=None,
    ) -> Program:
        """Add a child; it goes to its parent's island."""
        parent.children += 1
        program = Program(
            solution, score, parent.island, generation, embedding=embedding
        )
        self._store(program)
        # A correct program also seeds islands that have no correct program.
        if program.correct:
            for island in range(self.num_islands):
                if island != program.island and not self._correct(island):
                    copy = Program(
                        solution, score, island, generation, embedding=embedding
                    )
                    self._store(copy)
        return program

    def _store(self, program: Program) -> None:
        self.programs.append(program)
        if program.correct:
            self._update_archive(program)

    def _update_archive(self, program: Program) -> None:
        if len(self.archive) < self.archive_size:
            self.archive.append(program)
            return
        worst = min(self.archive, key=lambda p: p.score)
        if program.score > worst.score:
            self.archive[self.archive.index(worst)] = program

    # endregion

    # region Queries.
    def _correct(self, island: int) -> list[Program]:
        return [p for p in self.programs if p.island == island and p.correct]

    def island_programs(self, island: int) -> list[Program]:
        return [p for p in self.programs if p.island == island]

    def best(self, island: int | None = None) -> Program | None:
        """The best correct program (newest first on ties)."""
        candidates = [
            p
            for p in self.programs
            if p.correct and (island is None or p.island == island)
        ]
        if not candidates:
            return None
        return max(reversed(candidates), key=lambda p: p.score)

    @property
    def initialised(self) -> bool:
        """Whether every island has a correct program."""
        return all(self._correct(i) for i in range(self.num_islands))

    # endregion

    # region Sampling.
    def sample(self) -> Sample:
        """Choose an island, a parent and inspirations.

        On an island without a correct program, a random incorrect program
        is returned with ``fix=True`` (Shinka's fix mode).
        """
        if not self.programs:
            raise ValueError("The programs database is empty.")
        island = int(np.random.randint(self.num_islands))
        if not self._correct(island):
            failed = self.island_programs(island) or self.programs
            parent = failed[np.random.randint(len(failed))]
            return Sample(island, parent, [], fix=True)
        parent = self._select_parent(island)
        return Sample(island, parent, self._inspirations(parent, island))

    def _select_parent(self, island: int) -> Program:
        candidates = [p for p in self.archive if p.island == island]
        if not candidates:  # fall back to the island's correct programs
            candidates = self._correct(island)
        if self.parent_selection == "weighted":
            scores = np.array([p.score for p in candidates], dtype=float)
            median = np.median(scores)
            mad = max(float(np.median(np.abs(scores - median))), 1e-6)
            s = _sigmoid(self.parent_selection_lambda * (scores - median) / mad)
            h = 1.0 / (1.0 + np.array([p.children for p in candidates], dtype=float))
            weights = s * h
            if not np.isfinite(weights).all() or weights.sum() <= 0:
                weights = np.ones(len(candidates))
            return candidates[
                np.random.choice(len(candidates), p=weights / weights.sum())
            ]
        ranked = sorted(candidates, key=lambda p: p.score, reverse=True)
        if self.parent_selection == "power_law":
            ranks = np.arange(1, len(ranked) + 1, dtype=float)
            weights = ranks ** (-self.exploitation_alpha)
            return ranked[np.random.choice(len(ranked), p=weights / weights.sum())]
        # beam_search: the best program until it has num_beams children.
        for program in ranked:
            if program.children < self.num_beams:
                return program
        return ranked[0]

    def _inspirations(self, parent: Program, island: int) -> list[Program]:
        """Archive and top-k inspirations from the parent's island, sorted
        from the lowest to the highest score."""
        pool = [p for p in self.archive if p.island == island and p.id != parent.id]
        # One program per solution (copies share a solution).
        unique: dict[str, Program] = {}
        for p in pool:
            unique.setdefault(p.id, p)
        pool = list(unique.values())
        chosen: list[Program] = []

        def take(program: Program) -> None:
            if program not in chosen:
                chosen.append(program)

        n = self.num_archive_inspirations
        if n > 0 and pool:
            best = self.best()
            if best is not None and best.id != parent.id:
                for p in pool:
                    if p.id == best.id:
                        take(p)
            elites = sorted(pool, key=lambda p: p.score, reverse=True)
            for p in elites[: int(n * self.elite_selection_ratio)]:
                if len(chosen) < n:
                    take(p)
            rest = [p for p in pool if p not in chosen]
            order = np.random.permutation(len(rest))
            for i in order:
                if len(chosen) >= n:
                    break
                take(rest[i])
            chosen = chosen[:n]
        remaining = sorted(
            (p for p in pool if p not in chosen), key=lambda p: p.score, reverse=True
        )
        chosen += remaining[: self.num_top_k_inspirations]
        return sorted(chosen, key=lambda p: p.score)

    # endregion

    def migrate(self, generation: int) -> list[Program]:
        """Move programs between islands every ``migration_interval``
        generations (Shinka's elitist migration).

        Returns:
            list[Program]: The programs that moved.
        """
        if (
            self.migration_rate <= 0
            or self.num_islands < 2
            or generation <= 0
            or generation % self.migration_interval
        ):
            return []
        moves = []  # decided first, so a program moves at most once
        for island in range(self.num_islands):
            candidates = [p for p in self._correct(island) if p.generation > 0]
            if self.island_elitism:
                best = self.best(island)
                candidates = [p for p in candidates if p is not best]
            if not candidates:
                continue
            count = min(
                len(candidates),
                max(1, int(len(self.island_programs(island)) * self.migration_rate)),
            )
            others = [j for j in range(self.num_islands) if j != island]
            for i in np.random.choice(len(candidates), size=count, replace=False):
                moves.append((candidates[i], others[np.random.randint(len(others))]))
        for program, island in moves:
            program.island = int(island)
        return [program for program, _ in moves]
