"""Mortgage backtests again, at full precision, with the round 3 baselines (POST HOC additions; review round 3).

DATASET: 200k loans per vintage, simple random sample of the full Freddie Mac Standard files (experiments/0010).
Versions (same evaluation sets, scoring and truth as the runs they repeat):
  as_run     the pre-registered run (pipeline of 0011): frozen methods only, to have full-precision pairs
  corrected  corrected censoring weights (pipeline of 0012, variant ipcw): frozen methods + round 3 additions
  noharp     HARP terminations are prepayments (0012, variant noharp): frozen methods + additions
  linked     borrower-followed estimand (0012 code on the linked table of 0018): frozen methods + additions
The frozen estimates are compared with the stored pair files of those runs (they must agree to 6 decimals).
Usage: PYTHONHASHSEED=0 uv run python freddie.py <version> -> artifacts/pairs_freddie_<version>.parquet,
artifacts/meta_freddie_<version>.json
"""

from __future__ import annotations

import importlib.util
import json
import logging
import os
import sys
from pathlib import Path
from types import ModuleType
from typing import Final

import numpy as np
import pandas as pd

from maturity import data

HERE: Final = Path(__file__).resolve().parent
EXP: Final = HERE.parent
H: Final = 84
N_YOUNG: Final = 6
SETS: Final = {"E3a": (range(2006, 2013), False, 1), "E3b": (range(2013, 2019), True, 12)}
STORED: Final = {
    "as_run": EXP / "0011-freddie-200k-confirmatory" / "artifacts",
    "corrected": EXP / "0012-freddie-200k-harp-correction" / "artifacts" / "ipcw",
    "noharp": EXP / "0012-freddie-200k-harp-correction" / "artifacts" / "noharp",
    "linked": EXP / "0018-freddie-200k-follow-borrowers" / "artifacts" / "linked",
}
log = logging.getLogger(__name__)


def load_pipeline(version: str) -> tuple[ModuleType, object]:
    folder = EXP / (
        "0011-freddie-200k-confirmatory"
        if version == "as_run"
        else "0012-freddie-200k-harp-correction"
    )
    sys.path.insert(0, str(folder))
    spec = importlib.util.spec_from_file_location("pipeline", folder / "pipeline.py")
    assert spec is not None and spec.loader is not None
    pl = importlib.util.module_from_spec(spec)
    sys.modules["pipeline"] = pl
    spec.loader.exec_module(pl)
    if version == "as_run":
        return pl, pl.load("base")
    if version == "linked":
        pl.LOANS_PATH = data.DATA_DIR / "processed" / "freddie_200k_loans_linked.parquet"
        pl.CELLS_DIR = EXP / "0018-freddie-200k-follow-borrowers" / "artifacts"
        return pl, pl.load("base", "linked")
    return pl, pl.load("base", {"corrected": "ipcw", "noharp": "noharp"}[version])


def main(version: str) -> None:
    pl, d = load_pipeline(version)
    import extras  # after the pipeline, which must be the module named "pipeline"

    cut = int(pl.month_idx(202002))
    rows, metas = [], {}
    for name, (years, truncated, min_ahead) in SETS.items():
        cutoff = cut if truncated else None
        truth = pl.truth(d, "year", H, cutoff)
        for year in years:
            if version == "as_run":
                t, meta = pl.estimate(d, year, horizon=H, cutoff=cutoff)
                meta = {k: v for k, v in meta.items() if k != "f0"}
            else:
                t, meta = extras.snapshot(pl, d, year, horizon=H, cutoff=cutoff)
            t["truth"] = [
                truth[k][0] if k in truth and (truncated or k + H / 12 <= 2026.25) else np.nan
                for k in t.cohort
            ]
            y = t[t.young & t.truth.notna() & (t.truth > 0) & (t.h - t.A >= min_ahead)]
            rows.append(y.sort_values("cohort").tail(N_YOUNG).assign(snapshot=year, set=name))
            metas[str(year)] = meta
            log.info("%s %s %d model=%s", version, name, year, meta["model"])
    pairs = pd.concat(rows, ignore_index=True)
    stored = pd.concat(
        [pd.read_csv(STORED[version] / f"vintages_{s}.csv") for s in SETS], ignore_index=True
    )
    for m in (
        "truth",
        "naive",
        "drop_recent",
        "fixed_horizon",
        "chain",
        "cal_chain",
        "ours",
        "ours_uncond",
    ):
        gap = float(np.max(np.abs(pairs[m].to_numpy() - stored[m].to_numpy())))
        assert gap < 1e-6, f"{m}: not the stored run (largest difference {gap})"
    (HERE / "artifacts").mkdir(exist_ok=True)
    pairs.to_parquet(HERE / "artifacts" / f"pairs_freddie_{version}.parquet")
    (HERE / "artifacts" / f"meta_freddie_{version}.json").write_text(
        json.dumps(metas, indent=1, default=str)
    )
    log.info("%s: %d pairs; equal to the stored run to 6 decimals", version, len(pairs))


if __name__ == "__main__":
    if os.environ.get("PYTHONHASHSEED") != "0":
        raise SystemExit(
            "run with PYTHONHASHSEED=0 (the fixed-effects fit depends on set order, see 0011)"
        )
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    main(sys.argv[1])
