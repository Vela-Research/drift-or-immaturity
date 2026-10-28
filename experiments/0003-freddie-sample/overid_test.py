"""Over-identification test as a quasi-likelihood ratio (T2 Update 8, Proposition 2h).

DATASET: Freddie Mac 50k Sample, development only. Sub-distribution risk sets, calendar-adjusted (T2c section 8).
For vintage c with adjusted cumulative hazard on ages 6..hi (hi = 72): fit (i) one speed and (ii) separate early and
late speeds (clock s1 * a up to a* = 36, then s1 * a* + s2 * (a - a*)), with the reference shape F0 pooled from the
OTHER reference vintages (1999-2003, leave-one-out). LR = 2 (l_ii - l_i), multinomial shape likelihood with the
vintage's default count. T = LR / phi ~ chi2(1), phi = Pearson X^2 / df of the free-speed fit on vintage x age-quarter
cells (maturity.onestage.dispersion). Effect size = log(s2 / s1).
Usage: overid_test.py [mtm|hinge]  (reads artifacts/hazard_cells[_mode]_subdist.parquet)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd
from scipy import optimize, stats

from maturity import onestage

HERE: Final = Path(__file__).resolve().parent
AMAX: Final = 120
HI: Final = 72
LO: Final = 6
SPLIT: Final = 36
REF: Final = range(1999, 2004)
YEARS: Final = range(1999, 2020)


def main(mode: str) -> None:
    suffix = (f"_{mode}" if mode else "") + "_subdist"
    cells = pd.read_parquet(HERE / "artifacts" / f"hazard_cells{suffix}.parquet")
    g = (
        cells.groupby(["vintage", "age"])
        .agg(ev=("events", "sum"), w=("w_adj", "sum"))
        .reset_index()
    )
    curves, events = {}, {}
    for v, t in g.groupby("vintage"):
        inc = np.zeros(AMAX + 1)
        inc[t.age.to_numpy()] = t.ev.to_numpy() / np.maximum(t.w.to_numpy(), 1e-300)
        curves[int(v)] = np.cumsum(inc)
        events[int(v)] = float(t.ev[t.age <= HI].sum())
    ages = np.arange(AMAX + 1, dtype=float)
    cohorts = [
        onestage.Cohort(float(v), ages[: HI + 1], curves[v][: HI + 1], events[v], float(HI), True)
        for v in YEARS
    ]
    phi = onestage.dispersion(cohorts)
    rows = []
    for v in YEARS:
        ref = _reference(curves, events, [r for r in REF if r != v])
        d = np.diff(curves[v][LO - 1 : HI + 1])
        n = events[v] * d / d.sum()  # pseudo-counts of events by age LO..HI
        ll1, s = _fit_one(n, ref)
        ll2, s1, s2 = _fit_two(n, ref, s)
        lr = max(0.0, 2 * (ll2 - ll1))
        rows.append(
            {
                "vintage": v,
                "events": int(events[v]),
                "s_one": s,
                "s_early": s1,
                "s_late": s2,
                "log_ratio": float(np.log(s2 / s1)),
                "LR": lr,
                "T": lr / phi,
                "p_value": float(stats.chi2.sf(lr / phi, 1)),
                "p_value_untempered": float(stats.chi2.sf(lr, 1)),
            }
        )
    t = pd.DataFrame(rows).set_index("vintage")
    t["rejects"] = t.p_value < 0.05
    t.round(4).to_csv(HERE / "artifacts" / f"overid_test{suffix}.csv")
    summary = {
        "phi": round(phi, 2),
        "reject_share_1999_2012": round(float(t.loc[1999:2012].rejects.mean()), 2),
        "reject_share_2013_2019": round(float(t.loc[2013:2019].rejects.mean()), 2),
        "reject_share_untempered_1999_2012": round(
            float((t.loc[1999:2012].p_value_untempered < 0.05).mean()), 2
        ),
        "reject_share_untempered_2013_2019": round(
            float((t.loc[2013:2019].p_value_untempered < 0.05).mean()), 2
        ),
    }
    (HERE / "artifacts" / f"overid_test{suffix}.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary))
    print(
        t[["events", "s_early", "s_late", "log_ratio", "T", "p_value", "rejects"]]
        .round(3)
        .to_string()
    )


def _reference(
    curves: dict[int, np.ndarray], events: dict[int, float], years: list[int]
) -> np.ndarray:
    """Event-weighted pooled shape on ages 0..AMAX from the given vintages (not normalised)."""
    return sum(events[v] * curves[v] / curves[v][HI] for v in years) / sum(events[v] for v in years)  # type: ignore[return-value]


def _loglik(n: np.ndarray, ref: np.ndarray, clock: np.ndarray) -> float:
    f = np.interp(clock, np.arange(AMAX + 1), ref)
    p = np.diff(f)
    if f[-1] - f[0] <= 0 or (p <= 0).any():
        return -1e12
    return float(np.sum(n * np.log(p / (f[-1] - f[0]))))


def _fit_one(n: np.ndarray, ref: np.ndarray) -> tuple[float, float]:
    a = np.arange(LO - 1, HI + 1, dtype=float)
    res = optimize.minimize_scalar(
        lambda x: -_loglik(n, ref, np.exp(x) * a),
        bounds=(np.log(0.3), np.log(1.6)),
        method="bounded",
    )
    return -float(res.fun), float(np.exp(res.x))


def _fit_two(n: np.ndarray, ref: np.ndarray, start: float) -> tuple[float, float, float]:
    a = np.arange(LO - 1, HI + 1, dtype=float)

    def clock(x: np.ndarray) -> np.ndarray:
        s1, s2 = np.exp(x)
        return np.where(a <= SPLIT, s1 * a, s1 * SPLIT + s2 * (a - SPLIT))

    def neg(x: np.ndarray) -> float:
        c = clock(x)
        return 1e12 if c[-1] > AMAX else -_loglik(n, ref, c)

    best = None
    for d0 in (0.0, 0.3, -0.3):
        res = optimize.minimize(
            neg,
            np.array([np.log(start) - d0 / 2, np.log(start) + d0 / 2]),
            method="Nelder-Mead",
            options={"xatol": 1e-4, "fatol": 1e-6},
        )
        if best is None or res.fun < best.fun:
            best = res
    assert best is not None
    return -float(best.fun), float(np.exp(best.x[0])), float(np.exp(best.x[1]))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "")
