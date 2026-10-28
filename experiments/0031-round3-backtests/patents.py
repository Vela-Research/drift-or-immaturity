"""Patent backtest again, at full precision, with the round 3 additions (POST HOC; review round 3, P0 item 1).

DATASET: USPTO PatEx, release folder "2022" (data of 2023-06); the sample, snapshots, estimand and scoring of the
pre-specified backtest (experiments/0021): 100,000 applications per filing year 2001-2019, 8 technology centres,
snapshots December 2007-2017, 72-month grant share, the filing years S-4 .. S-1 within 2004-2016.
  main    frozen methods (they must equal the stored pair file of 0021) + the additions of extras.py. The forecast
          path holds the log pending inventory of each centre at its mean of the 12 months up to S and the Alice
          indicator at its value at S: it uses nothing after S.
  no_own  conditional (realised) path with an inventory that leaves out the scored cohorts' own applications: at
          snapshot S the pending inventory counts only the applications that were NOT filed in the years S-4 .. S-1.
          The calendar fit at S and the realised path both use this inventory.
Usage: PYTHONHASHSEED=0 uv run python patents.py -> artifacts/pairs_patents_{main,no_own}.parquet,
artifacts/meta_patents_{main,no_own}.json
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd

HERE: Final = Path(__file__).resolve().parent
EXP: Final = HERE.parent
sys.path[:0] = [
    str(EXP / "0021-patex-backtest"),
    str(EXP / "0020-patex-feasibility"),
    str(EXP / "0012-freddie-200k-harp-correction"),
]
import backtest as frozen  # the pre-specified patent backtest
import check
import extras
import pipeline as pl

log = logging.getLogger(__name__)


def evaluate(build_data: object, name: str) -> pd.DataFrame:
    """build_data(year) -> the data object that snapshot `year` uses."""
    rows, metas = [], {}
    for year in frozen.SNAPSHOTS:
        d = build_data(year)  # type: ignore[operator]
        truth = pl.truth(d, "year", frozen.H)
        t, meta = extras.snapshot(pl, frozen.seen_by(d, year), year, horizon=frozen.H)
        t["truth"] = [truth[k][0] for k in t.cohort]
        t["ref"] = float(t[t.mature].truth.iloc[-1])
        keep = t.cohort.between(max(year - 4, frozen.SCORED[0]), min(year - 1, frozen.SCORED[1]))
        rows.append(t[keep].assign(snapshot=year))
        metas[str(year)] = meta
        log.info("%s %d model=%s calendar=%s", name, year, meta["model"], meta["calendar_on"])
    pairs = pd.concat(rows, ignore_index=True)
    pairs.to_parquet(HERE / "artifacts" / f"pairs_patents_{name}.parquet")
    (HERE / "artifacts" / f"meta_patents_{name}.json").write_text(
        json.dumps(metas, indent=1, default=str)
    )
    return pairs


def main() -> None:
    (HERE / "artifacts").mkdir(exist_ok=True)
    pl._persistence = frozen.persistence
    applications = check.load()
    pool = applications[applications.year.isin(check.YEARS)]
    sample = pool.groupby("year").sample(n=check.N_PER_YEAR, random_state=check.SEED)
    full = frozen.build(sample, check.inventory(applications))
    pairs = evaluate(lambda year: full, "main")
    stored = pd.read_csv(EXP / "0021-patex-backtest" / "artifacts" / "pairs_main.csv")
    for m in ("truth", *frozen.METHODS):
        gap = float(np.max(np.abs(pairs[m].to_numpy() - stored[m].to_numpy())))
        assert gap < 1e-6, f"{m}: not the stored run (largest difference {gap})"
    log.info("main: %d pairs; equal to the stored run to 6 decimals", len(pairs))

    def without_scored(year: int) -> object:
        others = applications[~applications.year.isin(range(year - 4, year))]
        return frozen.build(sample, check.inventory(others))

    evaluate(without_scored, "no_own")


if __name__ == "__main__":
    if os.environ.get("PYTHONHASHSEED") != "0":
        raise SystemExit(
            "run with PYTHONHASHSEED=0 (the fixed-effects fit depends on set order, see 0011)"
        )
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    main()
