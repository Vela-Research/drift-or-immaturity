"""Adaptive training windows over cohorts with delayed labels (T4 note: Theorems A and B, Corollary B').

Cohorts are ordered from the youngest in use (column 0) to the oldest. Window k = the k youngest cohorts, and the
target is the eventual rate of the youngest cohort. Every function works on a batch: arrays of shape
(replications, cohorts); `n` (units per cohort) and `f` (share of the eventual events already seen) broadcast to it.

Naive rules take the labels at face value (window mean = events / units). Corrected rules use the pooled (Cape Cod)
estimate, events / sum(n * F), whose effective sample size is sum(n * F).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "KAPPA_GRID",
    "Origin",
    "adwin",
    "choose_kappa",
    "corrected_estimates",
    "corrected_lepski",
    "corrected_lepski_proven",
    "gaussian_multiplier",
    "lepski",
    "naive_estimates",
    "naive_lepski",
    "pick",
]

Array = NDArray[np.float64]
Windows = NDArray[np.int64]
KAPPA_GRID: Final = (
    3.0,
    2.0,
    1.5,
    1.0,
    0.5,
)  # descending: ties go to the larger multiplier (more pooling)


@dataclass(frozen=True, slots=True)
class Origin:
    """Data seen at an earlier pseudo-snapshot, and what the then-youngest cohort did afterwards."""

    events: Array  # (replications, cohorts), youngest first, as seen at the pseudo-snapshot
    n: Array
    f: Array  # maturity of each cohort at the pseudo-snapshot
    later_events: (
        Array  # (replications,) events of the then-youngest cohort from the pseudo-snapshot to now
    )
    later_exposure: (
        Array  # n * (F now - F then) of that cohort: expected later events = rate * later_exposure
    )


def gaussian_multiplier(cohorts: int, alpha: float = 0.05) -> float:
    return float(np.sqrt(2 * np.log(cohorts / alpha)))


def naive_estimates(events: Array, n: Array) -> tuple[Array, Array]:
    """Window means that take the labels at face value, and their binomial standard errors."""
    units = np.cumsum(np.broadcast_to(n, events.shape), axis=1)
    mean = np.cumsum(events, axis=1) / units
    return mean, np.sqrt(np.maximum(mean * (1 - mean), 1e-12) / units)


def corrected_estimates(events: Array, n: Array, f: Array) -> tuple[Array, Array]:
    """Pooled estimates per window, and the effective sample sizes sum(n * F)."""
    size = np.cumsum(np.broadcast_to(n * f, events.shape), axis=1)
    return np.cumsum(events, axis=1) / size, size


def lepski(est: Array, radius: Array, multiplier: float | Array = 1.0) -> Windows:
    """Largest window k (1-based) with |est_k - est_j| <= multiplier * (radius_k + radius_j) for every j < k."""
    mult = np.asarray(multiplier, dtype=float).reshape(-1, 1, 1)
    gap = np.abs(est[:, :, None] - est[:, None, :])
    ok = gap <= mult * (radius[:, :, None] + radius[:, None, :])
    ok |= np.triu(np.ones(est.shape[1:] * 2, dtype=bool))  # only compare with smaller windows
    passes = ok.all(axis=2)
    return np.asarray(est.shape[1] - np.argmax(passes[:, ::-1], axis=1), dtype=np.int64)


def pick(est: Array, window: Windows) -> Array:
    return np.asarray(est[np.arange(len(est)), window - 1], dtype=float)


def naive_lepski(events: Array, n: Array, *, alpha: float = 0.05) -> tuple[Windows, Array]:
    est, radius = naive_estimates(events, n)
    window = lepski(est, radius, gaussian_multiplier(events.shape[1], alpha))
    return window, pick(est, window)


def adwin(events: Array, n: Array, *, delta: float = 0.05) -> tuple[Windows, Array]:
    """ADWIN (Bifet & Gavalda 2007) with one bucket per cohort and labels at face value. The window drops its
    oldest cohort while some split into an older and a newer part differs by more than the variance-based cut
    sqrt(2 v log(2 / d') / m) + 2 log(2 / d') / (3 m), m = harmonic size of the parts, d' = delta / cohorts."""
    reps, cohorts = events.shape
    zero = np.zeros((reps, 1))
    done = np.concatenate([zero, np.cumsum(events, axis=1)], axis=1)
    units = np.concatenate([zero, np.cumsum(np.broadcast_to(n, events.shape), axis=1)], axis=1)
    log_term = float(np.log(2 * cohorts / delta))
    window = np.ones(reps, dtype=np.int64)
    for k in range(2, cohorts + 1):
        new, old = units[:, 1:k], units[:, [k]] - units[:, 1:k]
        gap = np.abs(done[:, 1:k] / new - (done[:, [k]] - done[:, 1:k]) / old)
        mean = done[:, [k]] / units[:, [k]]
        m = 1 / (1 / new + 1 / old)
        cut = np.sqrt(2 * mean * (1 - mean) * log_term / m) + 2 * log_term / (3 * m)
        window[~(gap > cut).any(axis=1)] = k  # the largest window with no detected change survives
    est, _ = naive_estimates(events, n)
    return window, pick(est, window)


def corrected_lepski(
    events: Array, n: Array, f: Array, *, kappa: float | Array | None = None, alpha: float = 0.05
) -> tuple[Windows, Array]:
    """Corollary B': Gaussian radii sqrt(p_k / effective size) and threshold multiplier kappa (default z)."""
    est, size = corrected_estimates(events, n, f)
    mult = gaussian_multiplier(events.shape[1], alpha) if kappa is None else kappa
    window = lepski(est, np.sqrt(np.maximum(est, 1e-12) / size), mult)
    return window, pick(est, window)


def corrected_lepski_proven(
    events: Array, n: Array, f: Array, *, alpha: float = 0.05, f_error: float | Array = 0.0
) -> tuple[Windows, Array]:
    """Theorem B with its proven constants: data-driven Bernstein radii, made nonincreasing, threshold 2 * (sum).
    f_error = sup-norm error bound of an estimated F (T4 Update, part b), a number or one value per
    replication with shape (replications, 1); it widens the radii."""
    est, size = corrected_estimates(events, n, f)
    log_term = float(np.log(2 * events.shape[1] / alpha))
    upper = (np.sqrt(log_term / 2) + np.sqrt(np.cumsum(events, axis=1) + 5 * log_term / 6)) ** 2
    radius = (np.sqrt(2 * upper * log_term) + log_term / 3) / size
    radius = radius + 2 * est * f_error * np.cumsum(np.broadcast_to(n, events.shape), axis=1) / size
    radius = np.maximum.accumulate(radius[:, ::-1], axis=1)[:, ::-1]
    window = lepski(est, radius, 2.0)
    return window, pick(est, window)


def choose_kappa(origins: Sequence[Origin], grid: Sequence[float] = KAPPA_GRID) -> Array:
    """Rolling-origin choice of the multiplier, per replication: run the corrected rule at each pseudo-snapshot,
    predict the later events of the then-youngest cohort, and keep the multiplier with the smallest Poisson
    deviance summed over the pseudo-snapshots."""
    total = np.zeros((len(origins[0].events), len(grid)))
    for origin in origins:
        for i, kappa in enumerate(grid):
            _, rate = corrected_lepski(origin.events, origin.n, origin.f, kappa=kappa)
            expected = np.maximum(rate * origin.later_exposure, 1e-12)
            seen = origin.later_events
            ratio = np.where(seen > 0, seen / expected, 1.0)
            total[:, i] += 2 * (seen * np.log(ratio) - (seen - expected))
    return np.asarray(grid, dtype=float)[np.argmin(total, axis=1)]
