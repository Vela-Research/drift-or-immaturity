"""tab:precision, re-implemented: the standard error of one cohort's log eventual rate with its speed known and
unknown (thm:precision (i)), and simulated.

DATASET: none (simulated; fixed seed). The script that made the paper's table was not kept; this one follows the
table's caption and the theorem.
Model: one cohort of n units, eventual rate p = 3%, Weibull F0 (shape 1.5, median 5 years, known), true speed s = 1,
monthly ages up to the follow-up T; the event count of month j is Poisson with mean n p [F0(s a_j) - F0(s a_{j-1})].
  theory  m = F0(T); D = n p m; speed known: SE = 1/sqrt(D); speed unknown: SE = sqrt((1 + e^2 / V) / D), where
          eta_j = Delta[u f0(u)] / Delta F0(u) over month j, and e, V are its event-weighted mean and variance.
  simulated  REPS replications per row, each row from SEED; (log p, log s) by maximum likelihood with the shape known
          (L-BFGS-B, bounds below); robust SE = 1.4826 x the median absolute deviation of log p_hat - log p.
  spread  the robust SE of SPREAD_REPS replications over the seeds SEEDS_FOR_RANGE (the paper's earlier column used
          300 replications from a script that was not kept).
Usage: uv run python run.py -> results.json
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Final

import numpy as np
from numpy.typing import NDArray
from scipy import optimize

log = logging.getLogger(__name__)
HERE: Final = Path(__file__).resolve().parent
P: Final = 0.03
SHAPE: Final = 1.5
SCALE: Final = 5.0 / math.log(2.0) ** (1.0 / SHAPE)  # median 5 years
REPS: Final = 10_000  # the paper's simulated column
SEED: Final = 20261027
SPREAD_REPS: Final = 300
SEEDS_FOR_RANGE: Final = range(
    30
)  # heavy tails at low maturity: a 300-replication value moves with the seed
BOUNDS: Final = [(math.log(1e-6), 0.0), (-3.0, 3.0)]  # log p (p <= 1), log s (speed 0.05 to 20)
ROWS: Final = [(20_000, 2), (20_000, 4), (20_000, 10), (200_000, 2), (200_000, 4)]
PAPER: Final = {  # (n, years): (m, known, unknown, simulated robust with REPS replications)
    (20_000, 2): (0.16, 0.10, 1.85, 1.52),
    (20_000, 4): (0.39, 0.07, 0.36, 0.36),
    (20_000, 10): (0.86, 0.04, 0.05, 0.05),
    (200_000, 2): (0.16, 0.03, 0.58, 0.56),
    (200_000, 4): (0.39, 0.02, 0.11, 0.11),
}
Array = NDArray[np.float64]


def cdf(t: Array) -> Array:
    return 1.0 - np.exp(-((t / SCALE) ** SHAPE))


def u_density(t: Array) -> Array:
    """u f0(u) of the Weibull: d F0(s a) / d log s at u = s a."""
    z = (t / SCALE) ** SHAPE
    return SHAPE * z * np.exp(-z)


def theory(years: int, n: int) -> dict[str, float]:
    a = np.arange(12 * years + 1) / 12.0
    d_f, d_uf = np.diff(cdf(a)), np.diff(u_density(a))
    m = float(cdf(a)[-1])
    w = d_f / d_f.sum()
    eta = d_uf / d_f
    e = float(w @ eta)
    v = float(w @ (eta - e) ** 2)
    d = n * P * m
    return {
        "maturity": m,
        "events_expected": d,
        "e": e,
        "V": v,
        "inflation": 1.0 + e * e / v,
        "se_known": 1.0 / math.sqrt(d),
        "se_unknown": math.sqrt((1.0 + e * e / v) / d),
    }


def fit(y: Array, a: Array, n: int) -> tuple[float, float]:
    """(log p, log s) by maximum likelihood, F0 known."""

    def negloglik(theta: Array) -> tuple[float, Array]:
        lp, ls = theta
        s = math.exp(ls)
        mu = np.maximum(n * math.exp(lp) * np.diff(cdf(s * a)), 1e-300)
        dmu_ls = n * math.exp(lp) * np.diff(u_density(s * a))
        r = y / mu - 1.0
        return float(-(y @ np.log(mu) - mu.sum())), -np.array([float(r @ mu), float(r @ dmu_ls)])

    start = np.array([math.log(max(y.sum(), 1.0) / (n * float(cdf(a)[-1]))), 0.0])
    start[0] = min(max(start[0], BOUNDS[0][0]), BOUNDS[0][1])
    res = optimize.minimize(negloglik, start, jac=True, method="L-BFGS-B", bounds=BOUNDS)
    return float(res.x[0]), float(res.x[1])


def simulate(years: int, n: int, rng: np.random.Generator, reps: int = REPS) -> dict[str, float]:
    a = np.arange(12 * years + 1) / 12.0
    mean = n * P * np.diff(cdf(a))
    err = []
    at_bound = 0
    for _ in range(reps):
        lp, ls = fit(rng.poisson(mean).astype(float), a, n)
        err.append(lp - math.log(P))
        at_bound += any(
            abs(x - b) < 1e-6 for x, bounds in zip((lp, ls), BOUNDS, strict=True) for b in bounds
        )
    e = np.array(err)
    return {
        "se_robust": float(1.4826 * np.median(np.abs(e - np.median(e)))),
        "sd_plain": float(e.std(ddof=1)),
        "median_error": float(np.median(e)),
        "fits_at_a_bound": at_bound,
    }


def self_check() -> None:
    """eta = d log(Delta F0(s a)) / d log s: the analytic increment equals a numerical derivative."""
    a = np.arange(25) / 12.0
    h = 1e-6
    num = (np.log(np.diff(cdf(math.exp(h) * a))) - np.log(np.diff(cdf(math.exp(-h) * a)))) / (2 * h)
    assert np.allclose(num, np.diff(u_density(a)) / np.diff(cdf(a)), rtol=1e-5)


def main() -> None:
    self_check()
    rows = []
    for n, years in ROWS:
        t, s = theory(years, n), simulate(years, n, np.random.default_rng(SEED))
        spread = [
            simulate(years, n, np.random.default_rng(seed), SPREAD_REPS)["se_robust"]
            for seed in SEEDS_FOR_RANGE
        ]
        m, known, unknown, simulated = PAPER[(n, years)]
        assert all(
            abs(x - y) <= 0.005
            for x, y in zip(
                (t["maturity"], t["se_known"], t["se_unknown"], s["se_robust"]),
                PAPER[(n, years)],
                strict=True,
            )
        ), (n, years)
        rows.append(
            {
                "n": n,
                "follow_up_years": years,
                **t,
                **s,
                "se_robust_300_replications_over_30_seeds": {
                    "min": min(spread),
                    "q10": float(np.quantile(spread, 0.1)),
                    "median": float(np.median(spread)),
                    "q90": float(np.quantile(spread, 0.9)),
                    "max": max(spread),
                },
                "paper": {
                    "maturity": m,
                    "se_known": known,
                    "se_unknown": unknown,
                    "simulated_robust": simulated,
                },
            }
        )
        log.info(
            "n %d, %d years: m %.2f, known %.2f, unknown %.2f, simulated %.2f (300 replications, 30 seeds: %.2f-%.2f);"
            " paper %.2f, %.2f, %.2f, %.2f",
            n,
            years,
            t["maturity"],
            t["se_known"],
            t["se_unknown"],
            s["se_robust"],
            min(spread),
            max(spread),
            m,
            known,
            unknown,
            simulated,
        )
    out = {
        "seed": SEED,
        "replications": REPS,
        "bounds_log_p_log_s": BOUNDS,
        "spread_replications": SPREAD_REPS,
        "seeds_for_range": list(SEEDS_FOR_RANGE),
        "rows": rows,
    }
    (HERE / "results.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    main()
