# Adapted for BLADE (iohblade) from ShinkaEvolve 0.0.8
# (https://github.com/SakanaAI/ShinkaEvolve, shinka/llm/prioritization.py,
# AsymmetricUCB), Copyright Sakana AI, licensed under the Apache License,
# Version 2.0 (http://www.apache.org/licenses/LICENSE-2.0). Changes: only the
# linear reward variant (``exponential_base=None``), without API-cost
# awareness and delivery-reliability weighting (BLADE has no per-call costs);
# randomness from numpy's global generator.
"""Asymmetric UCB over LLMs."""

from __future__ import annotations

import math

import numpy as np


class AsymmetricUCB:
    """Upper confidence bound selection with asymmetric (improvement-only)
    rewards.

    The reward of a child is its improvement over the better of its parent
    and the initial program, clipped at 0; failed children get 0. Rewards
    and counts decay by ``decay`` after every update, so recent results
    weigh more. Arms are scored by their normalised mean reward plus
    ``c * sqrt(2 ln t / n)``; the best arm gets ``1 - epsilon`` of the
    probability and the others share ``epsilon``. Unseen arms are tried
    first.

    Args:
        n_arms: Number of arms (LLMs).
        c: Exploration coefficient.
        epsilon: Probability shared by the non-best arms.
        decay: Factor applied to rewards and counts after each update.
    """

    def __init__(
        self, n_arms: int, c: float = 1.0, epsilon: float = 0.2, decay: float = 0.95
    ):
        if n_arms < 1:
            raise ValueError("n_arms must be >= 1")
        if not 0.0 <= epsilon <= 1.0:
            raise ValueError("epsilon must be in [0, 1]")
        if not 0.0 < decay <= 1.0:
            raise ValueError("decay must be in (0, 1]")
        self.n_arms = n_arms
        self.c = c
        self.epsilon = epsilon
        self.decay = decay
        self.n_submitted = np.zeros(n_arms)
        self.n_completed = np.zeros(n_arms)
        self.s = np.zeros(n_arms)
        self.divs = np.zeros(n_arms)
        self.obs_max = 0.0
        self.baseline = 0.0

    @property
    def n(self) -> np.ndarray:
        return np.maximum(self.n_submitted, self.n_completed)

    def set_baseline(self, score: float | None) -> None:
        """Set the score of the initial program (rewards are measured from
        it, or from the parent when that is better)."""
        if score is not None and math.isfinite(score):
            self.baseline = float(score)

    def probabilities(self) -> np.ndarray:
        n = self.n
        if self.n_arms == 1:
            return np.ones(1)
        unseen = n <= 0
        if unseen.any():
            return unseen / unseen.sum()
        mean = self.s / np.maximum(self.divs, 1e-7)
        if self.obs_max > 0:
            mean = mean / self.obs_max
        t = float(n.sum())
        scores = mean + self.c * np.sqrt(2.0 * np.log(max(t, 2.0)) / n)
        winners = scores == scores.max()
        if winners.all():
            return np.full(self.n_arms, 1.0 / self.n_arms)
        p = np.where(
            winners,
            (1.0 - self.epsilon) / winners.sum(),
            self.epsilon / (~winners).sum(),
        )
        return p / p.sum()

    def select(self) -> int:
        """Choose an arm and count it as submitted."""
        arm = int(np.random.choice(self.n_arms, p=self.probabilities()))
        self.n_submitted[arm] += 1
        return arm

    def update(self, arm: int, score: float | None, parent_score=None) -> float:
        """Record the result of a child of ``arm``.

        Args:
            arm: The arm that produced the child.
            score: The child's (maximised) score; None when it failed.
            parent_score: The parent's score, if it had one.

        Returns:
            float: The reward.
        """
        baseline = self.baseline
        if parent_score is not None and math.isfinite(parent_score):
            baseline = max(baseline, parent_score)
        real = score is not None and math.isfinite(score)
        reward = max(score - baseline, 0.0) if real else 0.0
        self.divs[arm] += 1
        self.n_completed[arm] += 1
        self.s[arm] += reward
        if real:
            self.obs_max = max(self.obs_max, reward)
        self.s *= self.decay
        self.divs *= self.decay
        return reward
