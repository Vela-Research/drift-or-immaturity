"""All backtests with the corrected plug-in ("corrected code", POST HOC, round 4), and the calibration errors of
both codes for the intervals.

DATASETS: Freddie Mac 200k (versions as_run, corrected, noharp, linked of experiments/0031) and the USPTO PatEx
sample of 0020/0021. Same pairs, truth, seeds and scoring as experiments/0031. No new data.
  pairs        the scored (cohort, snapshot) pairs: the frozen estimates of 0031 and, from this run, the full method
               with the corrected plug-in on the realised path (`ours`) and on the forecast path (`ours_uncond`)
  calibration  out-of-sample errors of quarterly cohorts at the December pseudo-snapshots (the rule of 0011/0021),
               for `ours`, `ours_uncond`, `cal_chain`, `chain`; code = frozen or corrected
Checks: every method that has no speed is equal to the stored pairs of 0031; a pair whose rescaled horizon s * h is
within the reference horizon keeps its frozen estimate; with code = frozen the run is the stored run.
Usage: PYTHONHASHSEED=0 uv run python backtests.py <as_run|corrected|noharp|linked|patents> <frozen|corrected> [pairs] [calibration]
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from types import ModuleType
from typing import Final

import numpy as np
import pandas as pd

HERE: Final = Path(__file__).resolve().parent
EXP: Final = HERE.parent
sys.path[:0] = [str(EXP / "0031-round3-backtests"), str(HERE)]
import corrected

log = logging.getLogger(__name__)
STORED: Final = EXP / "0031-round3-backtests" / "artifacts"
SPEED_FREE: Final = ["truth", "naive", "drop_recent", "fixed_horizon", "chain", "cal_chain"]
ERRORS: Final = ["ours", "ours_uncond", "cal_chain", "chain"]


def freddie(version: str) -> tuple[ModuleType, object, list[tuple[int, int | None, int, str]], int]:
    """Pipeline, data, (snapshot, truth cutoff, least months ahead, set) per snapshot, horizon."""
    import freddie as round3

    pl, d = round3.load_pipeline(version)
    cut = int(pl.month_idx(202002))
    plan = [
        (year, cut if truncated else None, min_ahead, name)
        for name, (years, truncated, min_ahead) in round3.SETS.items()
        for year in years
    ]
    return pl, d, plan, round3.H


def freddie_pairs(
    pl: ModuleType, d: object, plan: list[tuple[int, int | None, int, str]], horizon: int
) -> pd.DataFrame:
    rows = []
    for year, cutoff, min_ahead, name in plan:
        truth = pl.truth(d, "year", horizon, cutoff)
        t, _ = pl.estimate(d, year, horizon=horizon, cutoff=cutoff)
        t["truth"] = [
            truth[k][0]
            if k in truth and (cutoff is not None or k + horizon / 12 <= 2026.25)
            else np.nan
            for k in t.cohort
        ]
        y = t[t.young & t.truth.notna() & (t.truth > 0) & (t.h - t.A >= min_ahead)]
        rows.append(y.sort_values("cohort").tail(6).assign(snapshot=year, set=name))
        log.info("pairs %d", year)
    return pd.concat(rows, ignore_index=True)


def freddie_calibration(pl: ModuleType, d: object, horizon: int) -> pd.DataFrame:
    """The rule of 0011/0012 `calibration_errors` (that script cannot be imported: it reads sys.argv)."""
    truth = pl.truth(d, "quarter", horizon)
    rows = []
    for year in range(2006, 2019):
        t, _ = pl.estimate(d, year, cohort="quarter")
        y = t[t.young].copy()
        y["truth"] = [
            truth[k][0] if k in truth and k + horizon / 12 <= 2026.25 else np.nan for k in y.cohort
        ]
        rows.append(errors(y, year))
        log.info("calibration %d: %d quarterly cohorts", year, len(rows[-1]))
    return pd.concat(rows, ignore_index=True)


def errors(y: pd.DataFrame, year: int) -> pd.DataFrame:
    y = y[y.truth.notna() & (y.truth > 0) & (y.A >= 3)].copy()
    for m in ERRORS:
        y[f"r_{m}"] = np.log(np.maximum(y[m], 1e-9)) - np.log(y.truth)
    return y.assign(snapshot=year)[["snapshot", "cohort", "A", *(f"r_{m}" for m in ERRORS)]]


def patents() -> tuple[ModuleType, ModuleType, object]:
    sys.path[:0] = [
        str(EXP / "0021-patex-backtest"),
        str(EXP / "0020-patex-feasibility"),
        str(EXP / "0012-freddie-200k-harp-correction"),
    ]
    import backtest as frozen
    import check
    import pipeline as pl

    pl._persistence = frozen.persistence
    applications = check.load()
    pool = applications[applications.year.isin(check.YEARS)]
    sample = pool.groupby("year").sample(n=check.N_PER_YEAR, random_state=check.SEED)
    return pl, frozen, frozen.build(sample, check.inventory(applications))


def patent_pairs(pl: ModuleType, frozen: ModuleType, d: object) -> pd.DataFrame:
    truth = pl.truth(d, "year", frozen.H)
    rows = []
    for year in frozen.SNAPSHOTS:
        t, _ = pl.estimate(frozen.seen_by(d, year), year, horizon=frozen.H)
        t["truth"] = [truth[k][0] for k in t.cohort]
        keep = t.cohort.between(max(year - 4, frozen.SCORED[0]), min(year - 1, frozen.SCORED[1]))
        rows.append(t[keep].assign(snapshot=year))
        log.info("pairs %d", year)
    return pd.concat(rows, ignore_index=True)


def patent_calibration(pl: ModuleType, frozen: ModuleType, d: object) -> pd.DataFrame:
    """The rule of 0021 `calibration_errors`, with the forecast-path estimate added."""
    truth = pl.truth(d, "quarter", frozen.H)
    rows = []
    for year in range(frozen.SNAPSHOTS.start, frozen.SNAPSHOTS.stop - 1):
        t, _ = pl.estimate(frozen.seen_by(d, year), year, cohort="quarter", horizon=frozen.H)
        y = t[t.young].copy()
        y["truth"] = [
            truth[k][0] if k + 0.25 + frozen.H / 12 <= frozen.DATA_END else np.nan for k in y.cohort
        ]
        rows.append(errors(y, year))
        log.info("calibration %d: %d quarterly cohorts", year, len(rows[-1]))
    return pd.concat(rows, ignore_index=True)


def check_pairs(pairs: pd.DataFrame, stored: pd.DataFrame, code: str) -> None:
    """The run against the stored frozen pairs of 0031."""
    assert len(pairs) == len(stored), "not the scored pairs of 0031"
    same = SPEED_FREE if code == "corrected" else [*SPEED_FREE, "ours", "ours_uncond", "speed"]
    for m in same:
        gap = float(np.max(np.abs(pairs[m].to_numpy() - stored[m].to_numpy())))
        assert gap < 1e-9, f"{m}: not the stored run (largest difference {gap})"
    if code == "corrected":
        assert np.allclose(pairs.speed.to_numpy(), stored.speed.to_numpy(), rtol=1e-9), (
            "speeds changed"
        )
        slow = (pairs.speed <= 1.0).to_numpy()
        for m in ("ours", "ours_uncond"):
            assert np.allclose(pairs[m].to_numpy()[slow], stored[m].to_numpy()[slow], rtol=1e-9), m
            assert np.all(pairs[m].to_numpy()[~slow] >= stored[m].to_numpy()[~slow] - 1e-12), m


def main(version: str, code: str, steps: list[str]) -> None:
    out = HERE / "artifacts"
    out.mkdir(exist_ok=True)
    if version == "patents":
        pl, frozen, d = patents()
    else:
        pl, d, plan, horizon = freddie(version)
    if code == "corrected":
        corrected.patch(pl)
    if "pairs" in steps:
        pairs = (
            patent_pairs(pl, frozen, d)
            if version == "patents"
            else freddie_pairs(pl, d, plan, horizon)
        )
        name = "pairs_patents_main" if version == "patents" else f"pairs_freddie_{version}"
        check_pairs(pairs, pd.read_parquet(STORED / f"{name}.parquet"), code)
        pairs.to_parquet(out / f"pairs_{version}_{code}.parquet")
        log.info("%s %s: %d pairs; checks passed", version, code, len(pairs))
    if "calibration" in steps:
        cal = (
            patent_calibration(pl, frozen, d)
            if version == "patents"
            else freddie_calibration(pl, d, horizon)
        )
        cal.to_parquet(out / f"calibration_{version}_{code}.parquet")
        log.info("%s %s: %d calibration errors", version, code, len(cal))


if __name__ == "__main__":
    if os.environ.get("PYTHONHASHSEED") != "0":
        raise SystemExit("run with PYTHONHASHSEED=0 (the fixed-effects fit depends on set order)")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    main(sys.argv[1], sys.argv[2], sys.argv[3:] or ["pairs", "calibration"])
