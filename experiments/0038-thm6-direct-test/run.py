"""Direct test of the precision theorem (`thm:precision`) on single cohorts (POST HOC, round 4, P0.2).

DATASET: none; simulated data only (seed 20261006).
One cohort with a known timing distribution F0 (Weibull shape 1.5, or log-normal sigma 0.8; median 60 months), horizon
H = 84 months, true speed 1. Event counts in monthly age intervals up to age A are Poisson with mean
D * [F0(a_k) - F0(a_k-1)] / F0(A), so D is the expected number of events so far. A is the month whose maturity
m = F0(A) / F0(H) is nearest to the target.
Estimators of log p (p = level x F0(s H)): maximum likelihood over (level, speed) with F0 known (the level is profiled
out; the speed is searched on a grid of 4,001 points in [1/4, 4]); and the same with the speed known.
Theory: D * Var(log p-hat) -> I = 1 + (e(u_c) - e(u_H))^2 / V, with e the elasticity of F0 and V the variance of eta
over the events so far, on the same monthly intervals; with the speed known the limit is 1.
Usage: uv run python run.py -> results.json, research/drift-or-maturity/figdata/thm6_check.csv
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd
from scipy import stats

HERE: Final = Path(__file__).resolve().parent
OUT: Final = (
    HERE.parents[1] / "research" / "drift-or-maturity" / "figdata" / "thm6_check.csv"
)
log = logging.getLogger(__name__)
H: Final = 84
MEDIAN: Final = 60.0
TIMINGS: Final = {
    "weibull_1.5": stats.weibull_min(1.5, scale=MEDIAN / np.log(2) ** (1 / 1.5)),
    "lognormal_0.8": stats.lognorm(0.8, scale=MEDIAN),
}
MATURITIES: Final = (0.05, 0.1, 0.2, 0.4, 0.7)
EVENTS: Final = (30, 100, 300, 1000, 3000)
REPLICATIONS: Final = 4000
LOG_S: Final = np.linspace(-np.log(4.0), np.log(4.0), 4001)


def inflation(f0: stats.rv_continuous, age: int) -> float:
    """I = 1 + (e(u_c) - e(u_H))^2 / V on the monthly intervals up to `age` (speed 1)."""
    u = np.arange(age + 1.0)
    mass = np.diff(f0.cdf(u))
    eta = np.diff(u * f0.pdf(u)) / mass
    w = mass / mass.sum()
    e_c = float(np.sum(w * eta))  # = u f0(u) / F0(u) at u = age
    v = float(np.sum(w * (eta - e_c) ** 2))
    e_h = float(H * f0.pdf(H) / f0.cdf(H))
    return 1 + (e_c - e_h) ** 2 / v


def cell(
    f0: stats.rv_continuous, age: int, events: int, rng: np.random.Generator
) -> dict[str, float]:
    u = np.arange(age + 1.0)
    share = np.diff(f0.cdf(u)) / f0.cdf(age)
    counts = rng.poisson(events * share, size=(REPLICATIONS, age))
    counts = counts[counts.sum(axis=1) > 0]
    total = counts.sum(axis=1)
    edges = np.exp(LOG_S)[:, None] * u[None, :]
    cdf = f0.cdf(edges)
    log_share = np.log(np.maximum(np.diff(cdf, axis=1), 1e-300)) - np.log(cdf[:, -1:])
    best = np.argmax(counts @ log_share.T, axis=1)
    # log p-hat - log p; the level is total / (n F0(s A)), and n cancels against the truth
    correction = np.log(f0.cdf(np.exp(LOG_S) * H) / cdf[:, -1]) - np.log(f0.cdf(H) / f0.cdf(age))
    unknown = np.log(total / events) + correction[best]
    known = np.log(total / events)
    return {
        "maturity": float(f0.cdf(age) / f0.cdf(H)),
        "age_months": age,
        "expected_events": events,
        "replications": len(counts),
        "theory_inflation": inflation(f0, age),
        "d_var_speed_unknown": float(events * unknown.var()),
        "d_var_speed_known": float(events * known.var()),
        "median_abs_error_unknown": float(np.median(np.abs(unknown))),
        "p90_abs_error_unknown": float(np.quantile(np.abs(unknown), 0.9)),
        "median_abs_error_known": float(np.median(np.abs(known))),
        "p90_abs_error_known": float(np.quantile(np.abs(known), 0.9)),
        "share_at_boundary": float(np.mean((best == 0) | (best == len(LOG_S) - 1))),
        "bias_unknown": float(unknown.mean()),
    }


def main() -> None:
    rng = np.random.default_rng(20261006)
    rows = []
    for name, f0 in TIMINGS.items():
        months = np.arange(3, H)
        for m in MATURITIES:
            age = int(months[np.argmin(np.abs(f0.cdf(months) / f0.cdf(H) - m))])
            for events in EVENTS:
                rows.append({"timing": name, "target_maturity": m} | cell(f0, age, events, rng))
        log.info("%s done", name)
    table = pd.DataFrame(rows)
    table["ratio_to_theory"] = table.d_var_speed_unknown / table.theory_inflation
    OUT.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUT, index=False, float_format="%.5g")
    (HERE / "results.json").write_text(
        json.dumps(json.loads(table.round(5).to_json(orient="records")), indent=1)
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    main()
