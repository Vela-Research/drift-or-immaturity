"""CONFIRM2, A: how much do the placeholder paid-in-full dates move the frozen methods? (POST HOC check, deviation P1)

DATASET: the SBA units of data.py. Every loan whose paid-in-full date is a placeholder (2005-05-31 or 2005-07-31, and
approved before 2005-05) gets a payoff month drawn at random between its first month and the placeholder month
(seed 20261025). The frozen methods (backtest.py `pairs_at`) are run on these data at three snapshots, 2002 and 2006 (72-month
truth) and 2016 (truth cut at 2020-02), and compared with the stored pairs. The sub-distribution cells and each
cell's incidence do not depend on payoff dates under administrative censoring; the cohort incidence curves change
only through loans that are censored at S within their cohort.
Usage: PYTHONHASHSEED=0 uv run python pif_check.py -> artifacts/sba/pif_check.json
"""

from __future__ import annotations

import json
import logging
import os
from typing import Final

import backtest
import numpy as np
import pandas as pd

import data

log = logging.getLogger(__name__)
SEED: Final = 20261025
PLACEHOLDERS: Final = (2005 * 12 + 5, 2005 * 12 + 7)  # 1-based month indices of 2005-05 and 2005-07
METHODS: Final = [
    "truth",
    "naive",
    "chain",
    "unit_chain",
    "stratified",
    "cal_chain",
    "cal_chain_forecast",
    "ours",
    "ours_uncond",
]


def redated(u: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    rng = np.random.default_rng(SEED)
    hit = (u.event_type_clean == "prepay").to_numpy() & np.isin(u.ev_mi.to_numpy(), PLACEHOLDERS)
    hit &= (u.om < 2005 * 12 + 5).to_numpy()
    dur = u.event_age_clean.to_numpy().copy()
    dur[hit] = rng.integers(1, dur[hit] + 1)
    return u.assign(event_age_clean=dur, ev_mi=u.om.to_numpy() + dur), int(hit.sum())


def main() -> None:
    backtest.corrected.patch(data.pl)
    u, moved = redated(data.units("sba"))
    d = data.build("sba", u)
    stored = pd.read_parquet(data.ART / "sba" / "pairs.parquet")
    out: dict[str, object] = {"loans_redated": moved}
    for year, cutoff, ahead in [(2002, None, 1), (2006, None, 1), (2016, data.CUT_A, 12)]:
        y, _, _ = backtest.pairs_at(d, "sba", year, cutoff, ahead)
        s = stored[stored.snapshot == year].set_index("cohort")
        y = y.set_index("cohort")
        out[str(year)] = {m: round(float((y[m] / s[m] - 1).abs().max()), 6) for m in METHODS}
        log.info("%d: %s", year, out[str(year)])
    (data.ART / "sba" / "pif_check.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    if os.environ.get("PYTHONHASHSEED") != "0":
        raise SystemExit("run with PYTHONHASHSEED=0")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    main()
