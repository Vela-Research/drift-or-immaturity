"""Drift-detector alarms against the truth stream, and the delay of the 2005-2008 rise (POST HOC; round 4, P0.4).

DATASET: the monthly incidence files of experiments/0033 (`corrected` version) and artifacts/months_2006.npz of
prepare_extra.py (snapshots January to November 2006). Streams, detectors and settings are the ones of drift.py of
0033 (imported, not changed).
  counts  December snapshots 2006 to 2018, alarms at immature months (less than 84 months old at the snapshot).
          The truth stream = realised 84-month default rate of each origination month. An alarm of another stream is
          CONFIRMED when the truth stream (same snapshot, detector, direction) has an alarm at most 6 months away,
          else FALSE. A truth alarm is MISSED by a stream when that stream has no alarm of the same direction at
          most 6 months away.
  rise    first snapshot at which a stream has an up alarm at an origination month from January 2004 on. Monthly
          snapshots January 2006 (the first one whose stream holds January 2004) to December 2009 for truth, naive
          and cl; December snapshots for all five streams. Delay = months after the truth stream's first snapshot.
Usage: uv run python alarms.py -> results_alarms.json, research/drift-or-maturity/figdata/drift_alarms_truth.csv, drift_rise_delay.csv
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd

HERE: Final = Path(__file__).resolve().parent
OLD: Final = HERE.parent / "0033-downstream-classifier"
FIG: Final = HERE.parents[1] / "research" / "drift-or-maturity" / "figdata"
sys.path.insert(0, str(OLD))
import drift  # detectors and streams of 0033

KEYS: Final = ["snapshot", "detector", "direction"]
FIRST_MONTHLY: Final = 2006 * 12 + 1
LAST_MONTHLY: Final = 2009 * 12 + 12


def all_alarms() -> pd.DataFrame:
    """One row per alarm: snapshot, stream, detector, origination month, direction, immature."""
    z = dict(np.load(OLD / "artifacts" / "months_corrected.npz"))
    z |= dict(np.load(HERE / "artifacts" / "months_2006.npz"))
    snapshots = sorted({*z["snapshots"].tolist(), *range(FIRST_MONTHLY, 2006 * 12 + 12)})
    rows = []
    for s in snapshots:
        t = drift.streams(z, s, "corrected")
        for stream in (st for st in drift.STREAMS if st in t):
            for detector, found in drift.detect(t, stream)[0].items():
                rows += [
                    {
                        "snapshot": s,
                        "stream": stream,
                        "detector": detector,
                        "om": int(t.om[i]),
                        "direction": d,
                        "immature": bool(t.age[i] < drift.H),
                    }
                    for i, d in found
                ]
    return pd.DataFrame(rows)


def near(a: pd.DataFrame, other: pd.DataFrame) -> pd.Series:
    """Per alarm of `a`: has `other` an alarm at the same snapshot, detector and direction at most MATCH months
    away?"""
    there = other.groupby(KEYS).om.apply(np.array)
    hit = [
        k in there.index and bool((np.abs(there[k] - om) <= drift.MATCH).any())
        for k, om in zip(a[KEYS].itertuples(index=False, name=None), a.om, strict=True)
    ]
    return pd.Series(hit, index=a.index, dtype=bool)


def counts(a: pd.DataFrame) -> pd.DataFrame:
    """Per stream, detector and direction: alarms, confirmed by the truth stream, false, and truth alarms missed."""
    truth = a[a.stream == "truth"]
    rows = []
    for (stream, detector, direction), g in a.groupby(["stream", "detector", "direction"]):
        ok = near(g, truth)
        t = truth[(truth.detector == detector) & (truth.direction == direction)]
        rows.append(
            {
                "stream": stream,
                "detector": detector,
                "direction": "up" if direction == 1 else "down",
                "alarms": len(g),
                "confirmed_by_truth": int(ok.sum()),
                "false": int((~ok).sum()),
                "snapshots_with_alarm": int(g.snapshot.nunique()),
                "snapshots_with_false_alarm": int(g[~ok].snapshot.nunique()),
                "truth_alarms": len(t),
                "truth_alarms_missed": int((~near(t, g)).sum()),
            }
        )
    order = {s: i for i, s in enumerate(drift.STREAMS)}
    return pd.DataFrame(rows).sort_values(
        ["detector", "direction", "stream"], key=lambda c: c.map(order) if c.name == "stream" else c
    )


def rise(a: pd.DataFrame) -> pd.DataFrame:
    """First snapshot with an up alarm at an origination month from January 2004 on, per stream and detector, on
    the monthly grid and on the December grid; delay in months after the truth stream."""
    up = a[(a.direction == 1) & (a.om >= drift.RISE) & a.immature]
    grids = {
        "monthly 2006-01 to 2009-12": up[up.snapshot.between(FIRST_MONTHLY, LAST_MONTHLY)],
        "December 2006 to 2018": up[up.snapshot % 12 == 0],
    }
    rows = []
    for grid, g in grids.items():
        first = g.groupby(["detector", "stream"]).snapshot.min()
        for (detector, stream), s in first.items():
            if grid.startswith("monthly") and stream in ("cal", "full"):
                continue  # these two streams exist at December snapshots only
            rows.append(
                {
                    "grid": grid,
                    "detector": detector,
                    "stream": stream,
                    "first_snapshot": drift.month(int(s)),
                    "delay_months_after_truth": int(s - first[(detector, "truth")]),
                    "origination_month_of_the_alarm": drift.month(
                        int(
                            g[
                                (g.detector == detector) & (g.stream == stream) & (g.snapshot == s)
                            ].om.min()
                        )
                    ),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    drift.self_check()
    a = all_alarms()
    december = a[a.immature & (a.snapshot % 12 == 0)]
    calm = december[december.om.between(*drift.CALM)]
    tables = {
        "immature months, December snapshots": counts(december),
        "originations 2010 to 2015, December snapshots": counts(calm),
    }
    delay = rise(a)
    out = {
        "definitions": __doc__,
        "december_snapshots": int(december.snapshot.nunique()),
        **{k: v.to_dict("records") for k, v in tables.items()},
        "rise_of_the_2005_2008_vintages": delay.to_dict("records"),
    }
    (HERE / "results_alarms.json").write_text(json.dumps(out, indent=1))
    pd.concat([t.assign(scope=k) for k, t in tables.items()]).to_csv(
        FIG / "drift_alarms_truth.csv", index=False
    )
    delay.to_csv(FIG / "drift_rise_delay.csv", index=False)


if __name__ == "__main__":
    main()
