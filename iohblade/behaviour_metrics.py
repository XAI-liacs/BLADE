"""
behavior_metrics.py

Utility functions for analyzing single-run black box optimization traces.

Each function expects a pandas DataFrame that contains at least:
    * a column "evaluations"   integer evaluation index (starting at 1 or 0)
    * a column "raw_y"         objective function value (the lower the better)
    * coordinate columns       named "x0", "x1", … "x{d-1}"

The helper `get_coordinates(df)` extracts the X matrix.

Many metrics also accept optional keyword arguments such as `bounds`
(the search space lower/upper bounds) or `radius`.

All functions return a scalar float unless stated otherwise.

Author: Niki van Stein (May 2025)
"""

from __future__ import annotations

import math
from typing import Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull, distance_matrix
from scipy.spatial.distance import pdist, squareform
from scipy.stats import gaussian_kde
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import NearestNeighbors

# ---------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------


def get_coordinates(df: pd.DataFrame) -> np.ndarray:
    """Return (N, d) array with decision variables extracted from x-columns."""
    x_cols = [c for c in df.columns if c.startswith("x")]
    return df[x_cols].to_numpy()


def get_objective(df: pd.DataFrame) -> np.ndarray:
    """Return 1‑D array with objective values (raw_y)."""
    return df["raw_y"].to_numpy()


def _pairwise_distances(X: np.ndarray) -> np.ndarray:
    """Return condensed distance vector (upper triangle, no diag)."""
    return distance_matrix(X, X)[np.triu_indices(X.shape[0], k=1)]


# ---------------------------------------------------------------------
# Exploration metrics
# ---------------------------------------------------------------------


def average_nearest_neighbor_distance(df: pd.DataFrame, step=10, history=1000) -> float:
    """
    Average Euclidean distance from each point (except the first) to its
    nearest previous point -- proxy for sequential novelty / exploration.
    """
    X = get_coordinates(df)
    if len(X) < 2:
        return 0.0
    dists = []
    for k in range(1, len(X), step):
        prev = X[max(0, k - history) : k]
        dmin = np.min(np.linalg.norm(X[k] - prev, axis=1))
        dists.append(dmin)
    return float(np.mean(dists))


# 1 4608790451.0    5e+09     58.5          "dispersion": coverage_dispersion(df, bounds, disp_samples),
#   285         1 3133208393.0    3e+09     39.8          "spatial_entropy": spatial_entropy(df),


def coverage_dispersion(
    df: pd.DataFrame,
    bounds: Sequence[Tuple[float, float]] = None,
    n_samples: int = 10_000,
    rng: Optional[np.random.Generator] = None,
) -> float:
    """
    Approximate dispersion: max distance from a random point in the domain
    to its nearest evaluated sample. Lower is better (more coverage).
    """
    if rng is None:
        rng = np.random.default_rng()
    X = get_coordinates(df)
    d = X.shape[1]
    if bounds is None:
        bounds = [(-5.0, 5.0)] * d
    bounds = np.asarray(bounds, dtype=float)
    assert bounds.shape == (d, 2), "bounds should be shape (d, 2)"

    rand_points = rng.uniform(bounds[:, 0], bounds[:, 1], size=(n_samples, d))

    # Use k-d tree for fast nearest neighbor search
    nn = NearestNeighbors(n_neighbors=1, algorithm="kd_tree")
    nn.fit(X)
    distances, _ = nn.kneighbors(rand_points)

    return float(distances.max())


def spatial_entropy(df: pd.DataFrame, bandwidth: str | float = "scott") -> float:
    """
    Differential entropy estimate of the empirical sample distribution.
    Uses Gaussian KDE from scipy. High entropy -> diverse sampling.
    """
    X = get_coordinates(df).T  # gaussian_kde expects shape (d, N)
    kde = gaussian_kde(X, bw_method=bandwidth)
    log_probs = kde.logpdf(X)
    return float(-log_probs.mean())


# ---------------------------------------------------------------------
# Exploitation metrics
# ---------------------------------------------------------------------


def average_distance_to_best_so_far(df: pd.DataFrame) -> float:
    """
    For each evaluation k>0, compute distance to best point found up to k-1.
    Average these distances   lower = more exploitative.
    """
    X = get_coordinates(df)
    y = get_objective(df)
    best_idx = 0

    dists = []
    for k in range(1, len(X)):
        if y[k] < y[best_idx]:
            best_idx = k

        dists.append(np.linalg.norm(X[k] - X[best_idx]))
    return float(np.mean(dists))


def closed_form_random_search_diversity(bounds: Sequence[Tuple[float, float]]) -> float:
    bounds = np.asarray(bounds, dtype=float)
    widths = bounds[:, 1] - bounds[:, 0]
    d = len(bounds)
    w = np.mean(widths)  # assume roughly cubic
    return w * np.sqrt(d / 6)


def estimate_random_search_diversity(
    bounds: Sequence[Tuple[float, float]],
    n: int = 500,
    samples: int = 10_000,
    rng: Optional[np.random.Generator] = None,
) -> float:
    if rng is None:
        rng = np.random.default_rng()
    d = len(bounds)
    bounds = np.asarray(bounds)
    rand_points = rng.uniform(bounds[:, 0], bounds[:, 1], size=(samples, d))
    return pdist(rand_points[:n]).mean()


def avg_exploration_exploitation_chunked(
    df: pd.DataFrame,
    chunk_size: int = 500,
    bounds: Optional[Sequence[Tuple[float, float]]] = None,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[float, float]:
    X = get_coordinates(df)
    n, d = X.shape

    if bounds is None:
        bounds = [(-5.0, 5.0)] * d

    if rng is None:
        rng = np.random.default_rng()

    # Estimate diversity baseline from random sampling
    max_div = closed_form_random_search_diversity(bounds=bounds)

    chunk_diversities = []
    for i in range(0, n, chunk_size):
        chunk = X[i : i + chunk_size]
        if len(chunk) < 2:
            continue
        chunk_div = pdist(chunk).mean()
        chunk_diversities.append(chunk_div)

    if not chunk_diversities:
        return 0.0, 100.0  # no meaningful data

    normalized_explorations = 100 * np.array(chunk_diversities) / max_div
    normalized_explorations = np.clip(normalized_explorations, 0, 100)
    exploration = normalized_explorations.mean()
    exploitation = 100 - exploration

    return float(exploration), float(exploitation)


def intensification_ratio(df: pd.DataFrame, radius: float) -> float:
    """
    Fraction of evaluations lying within a given radius of the final best point.
    Higher -> stronger intensification / exploitation near the best.
    """
    X = get_coordinates(df)
    y = get_objective(df)
    best_idx = y.argmin()
    dists = np.linalg.norm(X - X[best_idx], axis=1)
    return float(np.mean(dists < radius))


# ---------------------------------------------------------------------
# Convergence metrics
# ---------------------------------------------------------------------


def average_convergence_rate(df: pd.DataFrame, optimum: float | None = None) -> float:
    """
    Geometric mean of successive error ratios (Chen & He, 2020).
    ACR < 1 implies convergence; smaller = faster.
    """
    y = get_objective(df)
    best_so_far = np.minimum.accumulate(y)
    if optimum is None:
        optimum = best_so_far[-1]
    errors = best_so_far - optimum
    # avoid zeros by adding tiny eps
    eps = np.finfo(float).eps
    ratios = (errors[1:] + eps) / (errors[:-1] + eps)
    return float(np.exp(np.log(ratios).mean()))


def improvement_statistics(df: pd.DataFrame) -> Tuple[float, float]:
    """
    Returns (avg_improvement, success_rate) where success_rate is the fraction
    of iterations that improved the best, and avg_improvement is the mean
    improvement magnitude on improving iterations.
    """
    y = get_objective(df)
    best = y[0]
    improvements = []
    successes = 0
    for val in y[1:]:
        if val < best:
            improvements.append(best - val)
            successes += 1
            best = val
    avg_imp = float(np.mean(improvements)) if improvements else 0.0
    success_rate = successes / (len(y) - 1) if len(y) > 1 else 0.0
    return avg_imp, success_rate


# ---------------------------------------------------------------------
# Stagnation metrics
# ---------------------------------------------------------------------


def longest_no_improvement_streak(df: pd.DataFrame) -> int:
    """Return the length of the longest consecutive streak with no improvement."""
    y = get_objective(df)
    best = y[0]
    curr = longest = 0
    for val in y[1:]:
        if val < best:
            best = val
            curr = 0
        else:
            curr += 1
            longest = max(longest, curr)
    return int(longest)


def last_improvement_fraction(df: pd.DataFrame) -> float:
    """Fraction of evaluations since the last improvement (0-1)."""
    best_so_far = np.minimum.accumulate(get_objective(df))
    last_imp_idx = np.where(np.diff(best_so_far) < 0)[0]
    last_imp_idx = last_imp_idx.max() + 1 if last_imp_idx.size else 0
    return (len(best_so_far) - 1 - last_imp_idx) / (len(best_so_far) - 1)


# ---------------------------------------------------------------------
# ---------------------------------------------------------------------
# Unified summary function
# ---------------------------------------------------------------------


def compute_behavior_metrics(
    df: pd.DataFrame,
    *,
    bounds: Optional[Sequence[Tuple[float, float]]] = None,
    radius: Optional[float] = None,
    disp_samples: int = 10_000,
) -> Dict[str, float]:
    """Compute all scalar behavior metrics and return them in a dictionary."""

    # default bounds and radius
    if bounds is None:
        d = len([c for c in df.columns if c.startswith("x")])
        bounds = [(-5.0, 5.0)] * d
    if radius is None:
        radius = 0.1 * (bounds[0][1] - bounds[0][0])

    # time‑series based exploration/exploitation percentages (averaged)
    avg_exploration_pct, avg_exploitation_pct = avg_exploration_exploitation_chunked(df)

    # improvement stats
    avg_imp, success_rate = improvement_statistics(df)

    metrics = {
        # Exploration & diversity
        "avg_nearest_neighbor_distance": average_nearest_neighbor_distance(df),
        "dispersion": coverage_dispersion(df, bounds, disp_samples),
        # "spatial_entropy": spatial_entropy(df),
        "avg_exploration_pct": avg_exploration_pct,
        # Exploitation
        "avg_distance_to_best": average_distance_to_best_so_far(df),
        "intensification_ratio": intensification_ratio(df, radius),
        "avg_exploitation_pct": avg_exploitation_pct,
        # Convergence
        "average_convergence_rate": average_convergence_rate(df),
        "avg_improvement": avg_imp,
        "success_rate": success_rate,
        # Stagnation
        "longest_no_improvement_streak": longest_no_improvement_streak(df),
        "last_improvement_fraction": last_improvement_fraction(df),
    }
    return metrics
