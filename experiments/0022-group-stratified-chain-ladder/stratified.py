"""Post hoc decomposition of the calendar step's gain (2026-10-04): covariate or group structure?

"Group-stratified chain ladder" = calendar + chain ladder with the covariates taken out: the same fixed-effects
Poisson (cohort x age cells and group effects, sub-distribution risk sets) but NO covariates, then chain ladder on
the group-adjusted cumulative hazards and the same unit-level plug-in. It keeps the group structure (group effects,
unit-level ages, Aalen-Johansen observed part) and drops the calendar covariates.
"Unit-level chain ladder" also drops the group effects: it shows what the unit-level plug-in alone does.
DATASETS: (1) USPTO PatEx sample of experiments/0020 with the backtest design of 0021 (11 snapshots, 43 pairs);
(2) Freddie Mac 200k, borrower-followed version of 0018, sets E3a + E3b of the frozen spec (13 snapshots, 78 pairs).
The other methods' estimates for the same (cohort, snapshot) pairs are read from the artifacts of 0021 and 0018.
The group effects come from iterative proportional fitting, which is the Poisson maximum likelihood for a model with
two sets of fixed effects and no covariate.
Usage: PYTHONHASHSEED=0 uv run python stratified.py -> results.json, artifacts/pairs_<dataset>.csv
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd
from scipy import stats

from maturity import data, onestage

HERE: Final = Path(__file__).resolve().parent
EXP: Final = HERE.parent
sys.path[:0] = [
    str(EXP / "0021-patex-backtest"),
    str(EXP / "0020-patex-feasibility"),
    str(EXP / "0012-freddie-200k-harp-correction"),
]
import backtest as patents  # the PatEx data object of the pre-registered backtest
import check
import pipeline as pl

log = logging.getLogger(__name__)
METHODS: Final = ["chain", "unit_chain", "stratified", "cal_chain", "ours"]
CUT: Final = int(pl.month_idx(202002))


def group_effects(d: pl.Data, s0: int) -> pd.Series:
    """Group effects of events ~ Poisson(exposure x exp(cell effect + group effect)) on the cells seen at S, centred
    as in the frozen pipeline (exposure-weighted mean zero). Groups without events get no effect (treated as 0)."""
    c = d.cells[(d.cells.mi <= s0) & (d.cells.vintage <= s0 // 12)]
    cell = pd.factorize(c.vintage.to_numpy() * 1000 + c.age.to_numpy())[0]
    group, names = pd.factorize(c.state.to_numpy())
    events, exposure = c.events.to_numpy(dtype=float), c.exposure.to_numpy(dtype=float)
    by_cell, by_group = np.bincount(cell, events), np.bincount(group, events)
    k = np.zeros(len(names))
    for _ in range(500):
        alpha = by_cell / np.maximum(np.bincount(cell, exposure * np.exp(k[group])), 1e-300)
        new = np.log(np.maximum(by_group, 1e-300) / np.bincount(group, exposure * alpha[cell]))
        done = float(np.max(np.abs(new - k)[by_group > 0])) < 1e-10
        k = new
        if done:
            break
    k = k - np.average(k[by_group > 0], weights=np.bincount(group, exposure)[by_group > 0])
    return pd.Series(k[by_group > 0], index=names[by_group > 0])


def stratified(
    d: pl.Data, year: int, horizon: int, cutoff: int | None = None, *, groups: bool = True
) -> dict[float, float]:
    """The calendar + chain ladder path of pipeline.estimate, with group effects only. groups=False also drops the
    group effects: chain ladder on the pooled sub-distribution hazard with the unit-level plug-in."""
    s = int(pl.month_idx(year * 100 + 12))
    info, _ = pl.observed(d, s, "year")
    t = info.copy()
    t["h"] = horizon if cutoff is None else np.minimum(horizon, t.A + (cutoff - s))
    t["mature"] = t.A >= horizon
    t["a_used"] = np.minimum(t.A, t.h)
    young = (t.A < t.h).to_numpy()
    zero = np.zeros(len(d.zcols))
    fit = (zero, group_effects(d, s - 1) if groups else pd.Series(dtype=float), zero)
    adjusted = pl.adjusted_curves(d, s - 1, fit, "year")
    ages = np.arange(pl.AMAX + 1, dtype=float)
    cohorts = [
        onestage.Cohort(
            r.cohort,
            ages[: int(r.a_used) + 1].copy(),
            adjusted[r.cohort][: int(r.a_used) + 1].copy(),
            float(r.events),
            float(r.h),
            bool(r.mature),
        )
        for r in t.itertuples()
    ]
    chain = onestage.fit(cohorts, 0.0)
    cells = {float(k): g for k, g in d.cells[d.cells["year"].isin(t.cohort[young])].groupby("year")}
    out = {}
    for i, r in enumerate(t.itertuples()):
        if young[i]:
            inc = pl.increments(
                adjusted[r.cohort],
                int(r.a_used),
                float(chain.speeds[i]),
                chain.ugrid,
                chain.f0,
                int(r.h),
            )
            out[float(r.cohort)] = pl.plug_in(
                d, cells[r.cohort], s - 1, fit, inc, int(r.h), conditional=True
            )[0]
    return out


def compare(v: pd.DataFrame, a: str, b: str) -> dict[str, object]:
    """Method a against b: sign test over snapshots and the cohort-level error ratio with a cluster bootstrap."""
    err = pd.DataFrame(
        {m: (np.abs(v[m] - v.truth) / v.truth).groupby(v.snapshot).mean() for m in (a, b)}
    )
    diff = (err[a] - err[b]).to_numpy()
    wins, n = int((diff < 0).sum()), int((diff != 0).sum())
    return {
        "wins": wins,
        "n": n,
        "sign_test_p": round(float(stats.binomtest(wins, n, 0.5).pvalue), 4),
    } | patents.vintage_level(v, a, b)


def summarise(v: pd.DataFrame) -> dict[str, object]:
    err = pd.DataFrame(
        {m: (np.abs(v[m] - v.truth) / v.truth).groupby(v.snapshot).mean() for m in METHODS}
    )
    signed = pd.DataFrame(
        {m: ((v[m] - v.truth) / v.truth).groupby(v.snapshot).mean() for m in METHODS}
    )
    return {
        "pairs": len(v),
        "median_error": err.median().round(4).to_dict(),
        "error_by_snapshot": err.round(4).to_dict("index"),
        "signed_error_by_snapshot": signed.round(4).to_dict("index"),
        "comparisons": {
            f"{a} vs {b}": compare(v, a, b)
            for a, b in (
                ("unit_chain", "chain"),
                ("stratified", "unit_chain"),
                ("stratified", "chain"),
                ("cal_chain", "stratified"),
                ("cal_chain", "chain"),
                ("ours", "stratified"),
            )
        },
    }


def run_patents() -> pd.DataFrame:
    applications = check.load()
    pool = applications[applications.year.isin(check.YEARS)]
    sample = pool.groupby("year").sample(n=check.N_PER_YEAR, random_state=check.SEED)
    d = patents.build(sample, check.inventory(applications))
    v = pd.read_csv(EXP / "0021-patex-backtest" / "artifacts" / "pairs_main.csv")
    for column, groups in (("stratified", True), ("unit_chain", False)):
        est = {year: stratified(d, year, patents.H, groups=groups) for year in patents.SNAPSHOTS}
        v[column] = [est[s][c] for s, c in zip(v.snapshot, v.cohort, strict=True)]
    return v


def run_freddie() -> pd.DataFrame:
    linked = EXP / "0018-freddie-200k-follow-borrowers" / "artifacts"
    pl.LOANS_PATH = data.DATA_DIR / "processed" / "freddie_200k_loans_linked.parquet"  # type: ignore[misc]
    pl.CELLS_DIR = linked  # type: ignore[misc]
    d = pl.load("base", "linked")
    parts = []
    for name, cutoff in (("E3a", None), ("E3b", CUT)):
        v = pd.read_csv(linked / "linked" / f"vintages_{name}.csv")
        for column, groups in (("stratified", True), ("unit_chain", False)):
            est = {
                int(y): stratified(d, int(y), 84, cutoff, groups=groups)
                for y in v.snapshot.unique()
            }
            v[column] = [est[s][c] for s, c in zip(v.snapshot, v.cohort, strict=True)]
        parts.append(v)
    return pd.concat(parts, ignore_index=True)


def main() -> None:
    (HERE / "artifacts").mkdir(exist_ok=True)
    out = {}
    for name, runner in (("patents", run_patents), ("freddie_borrower_followed", run_freddie)):
        v = runner()
        v.round(6).to_csv(HERE / "artifacts" / f"pairs_{name}.csv", index=False)
        out[name] = summarise(v)
        log.info("%s: median error %s", name, out[name]["median_error"])
        for k, c in out[name]["comparisons"].items():  # type: ignore[union-attr]
            log.info(
                "  %s: %s/%s (p = %s), ratio %s, CI %s",
                k,
                c["wins"],
                c["n"],
                c["sign_test_p"],
                c["error_ratio"],
                c["ci95"],
            )
    (HERE / "results.json").write_text(json.dumps(out, indent=1, default=str))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    main()
