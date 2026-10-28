"""CONFIRM2 §7.2 windows and §7.3 drift detectors on A and B (PRE-REGISTERED, secondary).

DATASET: the units of data.py; for the detectors also the units file of backtest.py (full-method completeness).
  windows    the E2 code of experiments/0014 (`seen_at`, `origins`, `snapshot`, imported): cohorts = entry years
             up to S - 1, youngest first; target = the H-month incidence of cohort S - 1 (truth: its Aalen-Johansen
             incidence in all units). Naive Lepski on events so far (`naive_lepski`) against the maturity-weighted
             Lepski rule with kappa = z_{alpha/N} (`corrected_z`); the other rules of 0014 come along. Random
             subsamples of 10% and 30% of the units of each entry year (10 each, seed 20261025) and all units.
             Snapshots: A 2000-2014 (decision A2), B 1994-2016.
  drift      the detectors of experiments/0033 `drift.py` (imported): ADWIN (delta 0.002) and Page-Hinkley
             (0.005, 50) on the stream of entry months with at least 24 months of age and 2,000 units, at the
             December snapshots (A 2000-2013, decision A2; B 1994-2016). Streams: truth (H-month incidence in all
             units), naive (seen at S), cl (seen / completeness of a chain ladder on the monthly cohorts), full
             (seen / the full method's completeness of the month, forecast path; 0033 `streams`). "Rate fell" =
             an alarm with direction -1 at an immature month (age < H). Counts: the rule of the paper (0039: an
             alarm is false when the truth stream has none of the same direction at most 6 months away; a truth
             alarm is missed when the stream has none) and the one-to-one matching of 0067.
Usage: uv run python secondary.py <sba|civil> <windows|drift> -> artifacts/<name>/{windows,alarms,streams}.parquet
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Final

import numpy as np
import pandas as pd

import data

HERE: Final = Path(__file__).resolve().parent
EXP: Final = HERE.parent
sys.path[:0] = [
    str(EXP / "0014-e2-freddie-adaptive-windows"),
    str(EXP / "0033-downstream-classifier"),
]
import drift  # detectors of 0033 (its `import run` gets 0014's run.py, which drift uses only in `streams`)
import run as e2  # E2 windows of 0014 (its `pipeline` is the frozen one already loaded by data.py)

from maturity import reserving

log = logging.getLogger(__name__)
pl = data.pl
SEED: Final = 20261025
FRACTIONS: Final = {0.1: 10, 0.3: 10, 1.0: 1}
WINDOW_YEARS: Final = {"sba": range(2000, 2015), "civil": range(1994, 2017)}
DRIFT_YEARS: Final = {"sba": range(2000, 2014), "civil": range(1994, 2017)}
DETECTORS: Final = ("adwin(0.002)", "ph(0.005,50)")
STREAMS: Final = ("truth", "naive", "cl", "full")


def loans(name: str) -> SimpleNamespace:
    return SimpleNamespace(loans=data.units(name))


def windows(name: str) -> pd.DataFrame:
    """0014 `main` with fractions of the units in place of loan counts."""
    e2.H = h = data.H[name]
    full = loans(name)
    truth = pl.truth(full, "year", h)
    years = WINDOW_YEARS[name]
    rng = np.random.default_rng(SEED)
    rows = []
    for fraction, draws in FRACTIONS.items():
        for draw in range(draws):
            d = full if fraction == 1.0 else _subsample(full, fraction, rng)
            seen = {y: e2.seen_at(d, y) for y in range(years.start - max(e2.ORIGINS), years.stop)}
            for year in years:
                rows += [
                    r | {"fraction": fraction, "draw": draw}
                    for r in e2.snapshot(seen, year, truth[float(year - 1)][0])
                ]
            log.info("%s windows: fraction %.1f, draw %d", name, fraction, draw)
    return pd.DataFrame(rows)


def streams(
    name: str, d: SimpleNamespace, s: int, units: pd.DataFrame, truth: pd.Series
) -> pd.DataFrame:
    """0033 `drift.streams` at a December snapshot, with this run's completeness of the full method."""
    h = data.H[name]
    info, curves = pl.observed(d, s, "om")
    om = info.cohort.to_numpy().astype(int)
    n = info.n.to_numpy(dtype=float)
    age = np.clip(s - om, 0, h)
    seen = np.full((len(om), h + 1), np.nan)
    for i, (m, a) in enumerate(zip(om, age, strict=True)):
        seen[i, : a + 1] = curves[float(m)][: a + 1]
    rows = (s - om >= drift.MIN_AGE) & (n >= drift.MIN_LOANS)
    cl = _link_ratio_completeness(seen, n, age, h)[age]
    t = pd.DataFrame(
        {
            "om": om,
            "n": n,
            "age": age,
            "truth": truth.reindex(om).to_numpy(),
            "naive": seen[np.arange(len(om)), age],
            "e_cl": cl,
        }
    )[rows].reset_index(drop=True)
    months = _month_completeness(units).reindex(t.om.to_numpy()).to_numpy()
    t["e_full"] = np.where(t.age >= h, 1.0, np.where(np.isnan(months), t.e_cl, months))
    for stream in ("cl", "full"):
        t[stream] = reserving.chain_ladder(
            t.naive.to_numpy(), t[f"e_{stream}"].to_numpy(), np.ones(len(t))
        )
    return t


def alarms(name: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    h = data.H[name]
    d = loans(name)
    truth = pd.Series({int(k): v[0] for k, v in pl.truth(d, "om", h).items()})
    units = pd.read_parquet(data.ART / name / "units.parquet")
    rows, values = [], []
    for year in DRIFT_YEARS[name]:
        s = year * 12 + 12
        drift.H = h
        t = streams(name, d, s, units[units.snapshot == year], truth)
        values.append(t.assign(snapshot=year))
        for stream in STREAMS:
            found, _ = drift.detect(t, stream)
            rows += [
                {
                    "snapshot": year,
                    "stream": stream,
                    "detector": k,
                    "om": int(t.om[i]),
                    "direction": sign,
                    "immature": bool(t.age[i] < h),
                }
                for k in DETECTORS
                for i, sign in found[k]
            ]
        log.info("%s drift %d: %d months in the stream", name, year, len(t))
    return pd.DataFrame(rows), pd.concat(values, ignore_index=True)


def main(name: str, step: str) -> None:
    out = data.ART / name
    if step == "windows":
        windows(name).to_parquet(out / "windows.parquet")
        return
    drift.self_check()
    a, values = alarms(name)
    a.to_parquet(out / "alarms.parquet")
    values.to_parquet(out / "streams.parquet")


def _subsample(d: SimpleNamespace, fraction: float, rng: np.random.Generator) -> SimpleNamespace:
    keep = np.concatenate(
        [
            rng.choice(idx, size=round(fraction * len(idx)), replace=False)
            for _, idx in sorted(d.loans.groupby("orig_year").indices.items())
        ]
    )
    return SimpleNamespace(loans=d.loans.iloc[np.sort(keep)])


def _link_ratio_completeness(
    seen: np.ndarray, n: np.ndarray, age: np.ndarray, h: int
) -> np.ndarray:
    """0033 `drift.link_ratio_completeness` with the horizon of the data set."""
    drift.H = h
    return np.asarray(drift.link_ratio_completeness(seen, n, age))


def _month_completeness(units: pd.DataFrame) -> pd.Series:
    seen = (units.n * units.aj).groupby(units.fp_mi).sum()
    final = (units.n * (1 - (1 - units.aj) * np.exp(-units.future_full))).groupby(units.fp_mi).sum()
    return seen / final.where(final > 0)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    main(sys.argv[1], sys.argv[2])
