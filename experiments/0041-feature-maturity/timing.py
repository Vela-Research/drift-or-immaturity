"""Accelerated-failure-time model of the event age given an event within the horizon (FEATURE_MATURITY_SPEC, `pu_feat`).

log age = mu + xi'x + sigma * eps, right-truncated at H. eps is the log of a generalized gamma variate (shape a,
power c = 1 / sigma); a = 1 is the Weibull version. Fitted by maximum likelihood on events with a complete age
within H (mature cohorts). `maturity` gives m = F(A | x) / F(H | x).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import optimize, stats


@dataclass(frozen=True, slots=True)
class Timing:
    a: float
    c: float
    mu: float
    xi: np.ndarray

    def cdf(self, age: np.ndarray, x: np.ndarray) -> np.ndarray:
        scale = np.exp(self.mu + x @ self.xi)
        return np.asarray(
            stats.gengamma.cdf(np.maximum(age, 0.0) / scale, self.a, self.c), dtype=float
        )

    def maturity(self, age: np.ndarray, x: np.ndarray, horizon: float) -> np.ndarray:
        """Share of the within-horizon events of units like x that happen by `age`."""
        top = np.maximum(self.cdf(np.full(len(x), horizon), x), 1e-300)
        return np.clip(self.cdf(np.minimum(age, horizon), x) / top, 0.0, 1.0)


def fit(age: np.ndarray, x: np.ndarray, horizon: float, *, weibull: bool = False) -> Timing:
    """Right-truncated maximum likelihood. `age` > 0 are complete event ages <= horizon; x has no constant."""
    k = x.shape[1]
    center = x.mean(axis=0)
    z = x - center

    def unpack(theta: np.ndarray) -> tuple[float, float, float, np.ndarray]:
        a = 1.0 if weibull else float(np.exp(np.clip(theta[0], -4, 7)))
        return a, float(np.exp(np.clip(theta[1], -4, 4))), float(theta[2]), theta[3:]

    def nll(theta: np.ndarray) -> float:
        a, c, mu, xi = unpack(theta)
        scale = np.exp(mu + z @ xi)
        logpdf = stats.gengamma.logpdf(age / scale, a, c) - np.log(scale)
        logcdf = stats.gengamma.logcdf(horizon / scale, a, c)
        value = -float(np.sum(logpdf - logcdf))
        return value if np.isfinite(value) else 1e300

    start = np.concatenate([[0.0, np.log(1.5), np.log(np.median(age)) + 0.3], np.zeros(k)])
    best = optimize.minimize(nll, start, method="L-BFGS-B")
    a, c, mu, xi = unpack(best.x)
    return Timing(a, c, mu - float(center @ xi), np.asarray(xi, dtype=float))


def demo() -> None:
    """Recovers a known feature effect on the timing from right-truncated ages."""
    rng = np.random.default_rng(1)
    x = rng.integers(0, 2, size=(40_000, 1)).astype(float)
    age = 40.0 * np.exp(0.5 * x[:, 0]) * rng.weibull(1.5, size=len(x))
    keep = (age <= 84) & (age > 0)
    model = fit(age[keep], x[keep], 84.0, weibull=True)
    assert abs(model.xi[0] - 0.5) < 0.06 and abs(model.c - 1.5) < 0.08, (model.xi, model.c)


if __name__ == "__main__":
    demo()
