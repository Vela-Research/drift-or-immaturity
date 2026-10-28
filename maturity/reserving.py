"""Classical reserving baselines on cohort-level development: Bornhuetter-Ferguson, Cape Cod, Generalized Cape Cod.

A cohort has n units, an incidence so far g at its age, and the timing curve gives the share f_age of the horizon
level that is reached at that age and the share f_target at the cohort's target horizon (1 at the full horizon).
  chain ladder             g x f_target / f_age
  Bornhuetter-Ferguson     g + prior x (f_target - f_age), the prior is an a priori horizon rate
  Cape Cod                 the same with prior = sum(n g) / sum(n f_age) over all cohorts (Buehlmann-Straub)
  Generalized Cape Cod     the same with a prior per cohort: the sums are weighted by decay^|distance| (Gluck 1997)
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

__all__ = ["bornhuetter_ferguson", "cape_cod_prior", "chain_ladder"]

Array = NDArray[np.float64]


def chain_ladder(g: Array, f_age: Array, f_target: Array) -> Array:
    return np.asarray(np.minimum(1.0, g * f_target / np.maximum(f_age, 1e-9)), dtype=np.float64)


def bornhuetter_ferguson(g: Array, f_age: Array, f_target: Array, prior: Array | float) -> Array:
    """Incidence so far plus the prior's share that is still to come."""
    return np.asarray(
        np.minimum(1.0, g + prior * np.maximum(f_target - f_age, 0.0)), dtype=np.float64
    )


def cape_cod_prior(
    n: Array, g: Array, f_age: Array, keys: Array | None = None, decay: float | None = None
) -> Array:
    """Horizon rate implied by all cohorts: sum(n g) / sum(n f_age). With `decay` in (0, 1) and cohort positions
    `keys`, cohort c gets its own rate, with weights decay^|key_c - key_k| (decay = 1 is Cape Cod, decay -> 0 is the
    chain ladder)."""
    if decay is None or keys is None:
        return np.full(len(g), float(np.sum(n * g) / max(float(np.sum(n * f_age)), 1e-300)))
    weight = decay ** np.abs(keys[:, None] - keys[None, :])
    return np.asarray(
        (weight @ (n * g)) / np.maximum(weight @ (n * f_age), 1e-300), dtype=np.float64
    )
