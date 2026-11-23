# Copyright 2023 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Modified for BLADE (iohblade): adapted from the FunSearch programs database
# (https://github.com/google-deepmind/funsearch) and its LLM4AD port
# (https://github.com/Optima-CityU/llm4ad, llm4ad/method/funsearch,
# BSD-2-Clause, Copyright (c) 2025 Optima Group). Changes: programs are
# complete BLADE solutions instead of single functions, islands are reset
# after a number of evaluations instead of a wall-clock period (so runs are
# reproducible), and randomness comes from numpy's global generator, which
# the BLADE experiment seeds per run.
"""The FunSearch programs database: islands of programs clustered by score."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ...solution import Solution


def _softmax(logits: np.ndarray, temperature: float) -> np.ndarray:
    """The tempered softmax of finite 1D ``logits``."""
    logits = np.asarray(logits, dtype=float)
    if not np.all(np.isfinite(logits)):
        raise ValueError(f"logits contain non-finite values: {logits}")
    z = logits / max(temperature, 1e-12)
    z = z - z.max()
    weights = np.exp(z)
    return weights / weights.sum()


class Cluster:
    """Programs of one island that have the same score."""

    def __init__(self, score: float, program: Solution):
        self.score = score
        self.programs: list[Solution] = [program]
        self.lengths: list[int] = [len(program.code)]

    def register(self, program: Solution) -> None:
        self.programs.append(program)
        self.lengths.append(len(program.code))

    def sample_program(self) -> Solution:
        """Sample a program, giving shorter programs a higher probability."""
        lengths = np.asarray(self.lengths, dtype=float)
        normalised = (lengths - lengths.min()) / (lengths.max() + 1e-6)
        probabilities = _softmax(-normalised, temperature=1.0)
        return self.programs[np.random.choice(len(self.programs), p=probabilities)]


class Island:
    """A sub-population of the programs database."""

    def __init__(
        self,
        programs_per_prompt: int,
        temperature_init: float,
        temperature_period: int,
    ):
        self.programs_per_prompt = programs_per_prompt
        self.temperature_init = temperature_init
        self.temperature_period = temperature_period
        self.clusters: dict[float, Cluster] = {}
        self.num_programs = 0

    def register(self, program: Solution, score: float) -> None:
        """Store ``program`` in the cluster of its score."""
        if score in self.clusters:
            self.clusters[score].register(program)
        else:
            self.clusters[score] = Cluster(score, program)
        self.num_programs += 1

    def temperature(self) -> float:
        """The cluster sampling temperature (linear decay, restarted each
        ``temperature_period`` programs)."""
        period = self.temperature_period
        return self.temperature_init * (1 - (self.num_programs % period) / period)

    def sample_programs(self) -> list[Solution]:
        """Programs for a prompt, ordered from the lowest to the highest score.

        Clusters are drawn (with replacement) by a softmax over their scores,
        then one program is drawn from each chosen cluster.
        """
        scores = np.array(list(self.clusters), dtype=float)
        # Scale scores to [-1, 1] when they are larger, so the temperature
        # has the same meaning for every problem (as in LLM4AD).
        max_abs = float(np.abs(scores).max())
        logits = scores / max_abs if max_abs > 1 else scores
        probabilities = _softmax(logits, self.temperature())
        k = min(len(scores), self.programs_per_prompt)
        chosen = np.random.choice(len(scores), size=k, p=probabilities)
        picked = [self.clusters[scores[i]] for i in chosen]
        picked.sort(key=lambda cluster: cluster.score)
        return [cluster.sample_program() for cluster in picked]


@dataclass(frozen=True)
class Prompt:
    """Programs sampled from one island, ordered from worst to best."""

    programs: list[Solution]
    island_id: int


class ProgramsDatabase:
    """Programs organised as islands; scores are maximised.

    Args:
        num_islands: Number of islands (independent sub-populations).
        programs_per_prompt: Number of programs shown in a prompt.
        temperature_init: Initial temperature of the cluster sampling.
        temperature_period: Period (in programs per island) of the linear
            temperature decay.
    """

    def __init__(
        self,
        num_islands: int = 10,
        programs_per_prompt: int = 2,
        temperature_init: float = 0.1,
        temperature_period: int = 30_000,
    ):
        self.num_islands = num_islands
        self.programs_per_prompt = programs_per_prompt
        self.temperature_init = temperature_init
        self.temperature_period = temperature_period
        self.islands = [self._new_island() for _ in range(num_islands)]
        self.best_score = [-np.inf] * num_islands
        self.best_program: list[Solution | None] = [None] * num_islands

    def _new_island(self) -> Island:
        return Island(
            self.programs_per_prompt, self.temperature_init, self.temperature_period
        )

    @property
    def empty(self) -> bool:
        return all(program is None for program in self.best_program)

    def register(self, program: Solution, island_id: int | None, score: float):
        """Register ``program`` with ``score`` on an island, or on all islands
        when ``island_id`` is None (the initial program)."""
        if not np.isfinite(score):
            raise ValueError("Only programs with a finite score are registered.")
        ids = range(self.num_islands) if island_id is None else [island_id]
        for i in ids:
            self.islands[i].register(program, score)
            if score > self.best_score[i]:
                self.best_score[i] = score
                self.best_program[i] = program

    def get_prompt(self) -> Prompt:
        """Sample programs from a random non-empty island."""
        candidates = [i for i, p in enumerate(self.best_program) if p is not None]
        if not candidates:
            raise ValueError("The programs database is empty.")
        island_id = int(np.random.randint(self.num_islands))
        if island_id not in candidates:
            island_id = int(candidates[np.random.randint(len(candidates))])
        return Prompt(self.islands[island_id].sample_programs(), island_id)

    def reset_islands(self) -> list[int]:
        """Reset the weaker half of the islands.

        Each reset island is restarted with the best program of a random
        island from the stronger half.

        Returns:
            list[int]: The ids of the islands that were reset.
        """
        # Small noise breaks ties between equally good islands.
        noisy = np.asarray(self.best_score) + np.random.randn(self.num_islands) * 1e-6
        order = np.argsort(noisy)
        num_reset = self.num_islands // 2
        reset_ids = [int(i) for i in order[:num_reset]]
        keep_ids = [
            int(i) for i in order[num_reset:] if self.best_program[i] is not None
        ]
        if not keep_ids:
            return []
        for island_id in reset_ids:
            founder_id = keep_ids[np.random.randint(len(keep_ids))]
            self.islands[island_id] = self._new_island()
            self.best_score[island_id] = -np.inf
            self.best_program[island_id] = None
            self.register(
                self.best_program[founder_id],
                island_id,
                self.best_score[founder_id],
            )
        return reset_ids
