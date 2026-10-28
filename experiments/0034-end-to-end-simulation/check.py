"""Check that `sim.estimate` is the frozen pipeline: one simulated world in the table layout of experiments/0012
(loan table and hazard cells), estimated by the frozen `pipeline.estimate`, must give the same numbers.

DATASET: none (simulated). The two covariates take the frozen names `hpi_yoy` and `unemp`.
Usage: PYTHONHASHSEED=0 uv run python check.py   (about 1 minute; raises AssertionError on a difference)
"""

from __future__ import annotations

import logging
from typing import Final

import numpy as np
import pandas as pd
import sim

log = logging.getLogger(__name__)
BASE_YEAR: Final = 1999
BASE: Final = 12 * BASE_YEAR  # month index of entry month 0 in the frozen layout
NEVER: Final = 2 * 10**9
COLUMNS: Final = ("A", "events", "naive", "chain", "cal_chain", "ours", "ours_uncond", "speed")


def frozen_tables(w: sim.World) -> object:
    """The world as `pipeline.Data`: one row per unit in `loans`, one row per (entry month, group, age) in `cells`."""
    entry, group, age = np.meshgrid(
        np.arange(sim.MONTHS), np.arange(sim.G), sim.AGES, indexing="ij"
    )
    counts = w.events.astype(int)
    never = (w.n - w.cum[:, :, -1]).astype(int)
    e = np.concatenate(
        [np.repeat(entry.ravel(), counts.ravel()), np.repeat(entry[:, :, 0].ravel(), never.ravel())]
    )
    a = np.concatenate([np.repeat(age.ravel(), counts.ravel()), np.full(never.sum(), 999)])
    event = np.arange(len(e)) < counts.sum()
    loans = pd.DataFrame(
        {
            "om": BASE + e,
            "orig_year": BASE_YEAR + e // 12,
            "ev_mi": np.where(event, BASE + e + a, NEVER),
            "event_age_clean": a.astype(float),
            "event_type_clean": np.where(event, "default", "censored"),
            "year": (BASE_YEAR + e // 12).astype(float),
            "quarter": ((BASE + e - 1) // 3) / 4.0,
        }
    )
    month = entry + age - 1
    cells = pd.DataFrame(
        {
            "vintage": BASE_YEAR + entry.ravel() // 12,
            "fp_mi": BASE + entry.ravel(),
            "state": sim.LABELS[group.ravel()],
            "age": age.ravel(),
            "mi": BASE + month.ravel(),
            "exposure": (w.n - w.cum[:, :, :-1]).ravel(),
            "events": w.events.ravel(),
            "aj": (w.cum[:, :, 1:] / w.n).ravel(),
            "hpi_yoy": w.z[0, group, month].ravel(),
            "unemp": w.z[1, group, month].ravel(),
        }
    )
    raw = ["hpi_yoy", "unemp"]
    for r in raw:
        cells[f"{r}_z"] = (cells[r] - cells[r].mean()) / cells[r].std()
    cells["unit"] = cells.groupby(["vintage", "fp_mi", "state"]).ngroup()
    cells["year"] = cells.vintage.astype(float)
    cells["quarter"] = ((cells.fp_mi - 1) // 3) / 4.0
    stats = {r: (float(cells[r].mean()), float(cells[r].std())) for r in raw}
    return sim.pl.Data(loans, cells, stats, raw, [f"{r}_z" for r in raw])


def compare(w: sim.World, data: object, year: int, per: int) -> float:
    """Largest relative difference between the two pipelines over the columns both have."""
    ours, meta = sim.estimate(w, year, per)
    frozen, frozen_meta = sim.pl.estimate(
        data, BASE_YEAR + year, cohort="year" if per == 12 else "quarter", horizon=sim.H
    )
    assert np.allclose(frozen.cohort.to_numpy() - BASE_YEAR, ours.cohort.to_numpy())
    assert meta["model"] == frozen_meta["model"], (meta["model"], frozen_meta["model"])
    assert meta["calendar_on"] == frozen_meta["calendar_on"]
    worst = 0.0
    for column in COLUMNS:
        a, b = ours[column].to_numpy(dtype=float), frozen[column].to_numpy(dtype=float)
        assert np.allclose(a, b, rtol=1e-6, atol=1e-9), (year, per, column, np.abs(a - b).max())
        worst = max(worst, float(np.max(np.abs(a - b) / np.maximum(np.abs(b), 1e-12))))
    if per == 12:
        truth = sim.pl.truth(data, "year", sim.H)
        mine = sim.truth(w, 12)
        assert all(abs(truth[BASE_YEAR + k][0] - v) < 1e-12 for k, v in mine.items())
    return worst


def main() -> None:
    for index in (4, 9, 14):  # drift, calendar, 20k; step, no calendar, 200k; log-normal, step, 20k
        setting = sim.SETTINGS[index]
        w = sim.simulate(setting, np.random.default_rng([sim.SEED, index, 0]))
        data = frozen_tables(w)
        for year, per in ((sim.TARGET, 12), (sim.TARGET - 2, 3), (sim.CALIBRATION[0], 3)):
            log.info(
                "%s, year %d, %s cohorts: equal to the frozen pipeline (largest relative difference %.1e)",
                setting.name,
                year,
                "annual" if per == 12 else "quarterly",
                compare(w, data, year, per),
            )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    main()
