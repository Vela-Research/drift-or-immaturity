"""Empirical-Bayes cohort speeds: a random walk on log s(c) observed through noisy per-cohort fits.

Model, over cohorts ordered by entry time: x_c = log s(c) = x_{c-1} + eta_c, eta_c ~ N(0, omega2), x_0 = 0 (the
reference speed); each fitted log speed y_c = x_c + e_c with e_c ~ N(0, kappa / events_c). Cohorts without a usable
fit (too young) have no observation and take the smoothed value from their neighbours. omega2 = 0 forces every speed
to the reference speed 1, i.e. it gives chain ladder exactly. (omega2, kappa) are chosen by marginal likelihood.
"""

from __future__ import annotations

from typing import Final

import numpy as np
from numpy.typing import NDArray

__all__ = ["EbFit", "eb_log_speeds"]

OMEGA2_GRID: Final = np.array([0.0, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 0.1, 0.3, 1.0])
KAPPA_GRID: Final = np.array([0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0])

Array = NDArray[np.float64]


class EbFit:
    """Smoothed log speeds `x` (one per cohort) and the chosen hyperparameters."""

    def __init__(self, x: Array, omega2: float, kappa: float, loglik: float) -> None:
        self.x, self.omega2, self.kappa, self.loglik = x, omega2, kappa, loglik


def eb_log_speeds(y: Array, events: Array) -> EbFit:
    """`y[c]` = fitted log speed (NaN if unavailable), `events[c]` = events behind the fit. Cohorts ordered by entry."""
    best: EbFit | None = None
    for omega2 in OMEGA2_GRID:
        for kappa in KAPPA_GRID:
            fit = _smooth(y, events, float(omega2), float(kappa))
            if best is None or fit.loglik > best.loglik + 1e-9:
                best = fit
    assert best is not None
    return best


def _smooth(y: Array, events: Array, omega2: float, kappa: float) -> EbFit:
    n = len(y)
    m_f, p_f, m_p, p_p = (np.zeros(n) for _ in range(4))
    m, p, ll = 0.0, 0.0, 0.0
    for c in range(n):
        m_p[c], p_p[c] = m, p + omega2
        m, p = m_p[c], p_p[c]
        if not np.isnan(y[c]):
            v = kappa / max(events[c], 1.0)
            s = p + v
            ll += -0.5 * (np.log(2 * np.pi * s) + (y[c] - m) ** 2 / s)
            gain = p / s
            m, p = m + gain * (y[c] - m), (1 - gain) * p
        m_f[c], p_f[c] = m, p
    x = m_f.copy()
    for c in range(n - 2, -1, -1):  # Rauch-Tung-Striebel
        gain = p_f[c] / p_p[c + 1] if p_p[c + 1] > 0 else 0.0
        x[c] = m_f[c] + gain * (x[c + 1] - m_p[c + 1])
    return EbFit(x, omega2, kappa, ll)
