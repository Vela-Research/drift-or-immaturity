"""Inputs of the round 4 follow-up that need the loan table (POST HOC; review round 4, item P0.4).

The band curves were computed before the specification of 2026-10-06 for the band test (experiment 0041 runs that
test). No classifier here uses them; they stay as a diagnostic only.

DATASET: 200k loans per vintage, Freddie Mac sample of experiments/0010, version `corrected` (pipeline of 0012).
  artifacts/bands.npz         per December snapshot S and FICO band (below 660, 660-719, 720 and up): the timing curve
                              of the chain ladder at ages 0..84 months, from the yearly cohorts of the band's loans as
                              S shows them (the construction of `share_cl` in prepare.py of 0033, on a subset)
  artifacts/band_speeds.csv   per snapshot and band: completeness at 12 to 60 months and the speed s that brings the
                              pooled reference closest to the band's curve, F(s a) / F(s H) (least squares over ages
                              1..84; the reference continues beyond H with `onestage.extended_reference`)
  artifacts/months_2006.npz   default incidence by origination month as the snapshots January to November 2006 show
                              it (the drift snapshots of 0033 start in December 2006)
Check in the code: the same construction on all loans gives the stored `share_cl` of 0033.
Usage: PYTHONHASHSEED=0 uv run python prepare_extra.py
"""

from __future__ import annotations

import logging
import os
import sys
from dataclasses import replace
from pathlib import Path
from types import ModuleType
from typing import Final

import numpy as np
import pandas as pd

from maturity import onestage

HERE: Final = Path(__file__).resolve().parent
ART: Final = HERE / "artifacts"
OLD: Final = HERE.parent / "0033-downstream-classifier" / "artifacts"
sys.path[:0] = [str(HERE.parent / "0031-round3-backtests"), str(OLD.parent)]
import extras
import freddie
import run

log = logging.getLogger(__name__)
H: Final = run.H
SPEEDS: Final = np.exp(np.linspace(np.log(0.4), np.log(2.5), 367))


def timing_curve(pl: ModuleType, d: object, year: int) -> onestage.Fit:
    """Chain ladder (equal speeds) on the yearly cohorts of `d.loans` at December `year`."""
    info, curves = pl.observed(d, year * 12 + 12, "year")
    t = info.assign(a_used=np.minimum(info.A, H), h=H, mature=info.A >= H)
    return onestage.fit(extras.cohorts_of(t, curves, pl.AMAX), 0.0)


def share(fit: onestage.Fit) -> np.ndarray:
    return np.asarray(np.interp(np.arange(H + 1.0), fit.ugrid, fit.f0, right=1.0))


def fitted_speed(band: np.ndarray, pooled: onestage.Fit) -> float:
    reference = onestage.extended_reference(pooled.ugrid, pooled.f0)
    ages = np.arange(1.0, H + 1)
    target = band[1:] / band[H]
    loss = [
        float(np.sum((reference(s * ages) / reference(np.array(s * H)) - target) ** 2))
        for s in SPEEDS
    ]
    return float(SPEEDS[int(np.argmin(loss))])


def band_curves(pl: ModuleType, d: object) -> None:
    stored = pd.read_parquet(OLD / "cohorts_corrected.parquet").groupby("snapshot").share_cl.first()
    fico = d.loans.fico.to_numpy()  # type: ignore[attr-defined]
    curves, rows = {}, []
    for year in run.SNAPSHOTS:
        pooled = timing_curve(pl, d, year)
        assert np.allclose(share(pooled), np.asarray(stored[year], dtype=float)), (
            "not share_cl of 0033"
        )
        for band, (lo, hi) in run.FICO_BANDS.items():
            subset = replace(d, loans=d.loans[(fico >= lo) & (fico < hi)])  # type: ignore[type-var,attr-defined]
            fit = timing_curve(pl, subset, year)
            curve = share(fit)
            curves[f"{year}_{band}"] = curve
            rows.append(
                {"snapshot": year, "band": band, "speed": round(fitted_speed(curve, pooled), 3)}
                | {f"e_{a}": round(float(curve[a] / curve[H]), 4) for a in (12, 24, 36, 48, 60)}
                | {f"pooled_e_{a}": round(float(share(pooled)[a]), 4) for a in (12, 24, 36, 48, 60)}
            )
        log.info("band curves %d", year)
    np.savez_compressed(ART / "bands.npz", **curves)
    pd.DataFrame(rows).to_csv(ART / "band_speeds.csv", index=False)


def months_2006(pl: ModuleType, d: object) -> None:
    """The views of `prepare.months` of 0033 for the eleven months before its first snapshot."""
    om = np.load(OLD / "months_corrected.npz")["om"]
    views = {}
    for s in range(2006 * 12 + 1, 2006 * 12 + 12):
        info, curves = pl.observed(d, s, "om")
        view = np.full((len(om), H + 1), np.nan, dtype=np.float32)
        for m in info.cohort.to_numpy().astype(int):
            a = min(s - m, H)
            view[np.searchsorted(om, m), : a + 1] = curves[float(m)][: a + 1]
        views[f"seen_{s}"] = view
    np.savez_compressed(ART / "months_2006.npz", **views)
    log.info("monthly curves at %d snapshots", len(views))


if __name__ == "__main__":
    if os.environ.get("PYTHONHASHSEED") != "0":
        raise SystemExit("run with PYTHONHASHSEED=0 (as prepare.py of 0033)")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    ART.mkdir(exist_ok=True)
    pipeline, data = freddie.load_pipeline("corrected")
    months_2006(pipeline, data)
    band_curves(pipeline, data)
