"""Drift detectors on the stream of default labels by origination month (POST HOC; round 3, item P1.9).

DATASET: 200k loans per vintage, simple random sample of the full Freddie Mac Standard files (experiments/0010);
input: the files of prepare.py (incidence by origination month as each snapshot shows it; completeness per unit).

At a snapshot S the stream is the origination months from 1999 to S - MIN_AGE months, in time order. Month m has
n_m loans and a label rate:
  truth  84-month default incidence in today's records (NOT available at S; the reference)
  naive  incidence seen at S (events so far)
  cl     seen / completeness, completeness from a chain ladder on the monthly cohorts that S shows (volume-weighted
         link ratios; at most 1)
  cal    seen / completeness of the calendar-adjusted chain ladder (December snapshots, forecast path): the
         method's estimate for the month, units summed (run.month_completeness)
  full   the same for the full method
Detectors. Both read the stream as loans in origination order and are evaluated at month ends:
  ADWIN         Bifet & Gavalda (2007), variance bound, delta' = delta / ln(n): the window drops its oldest month
                while two sub-windows differ by more than the bound. A corrected label is 0 or 1 / completeness, so
                its variance and range are larger; the bound uses both.
  Page-Hinkley  two-sided cumulative sum of (label - running mean -/+ delta) over loans; alarm when it leaves its
                extreme by more than the threshold; restart after an alarm.
An alarm has a month and a direction (up = the newer rate is higher). Alarms count when their month is immature at
S (less than 84 months old). An alarm is FALSE when the truth stream (same snapshot, detector and direction) has no
alarm at most 6 months away. For ADWIN the window that is left at the end of the stream is the detector's view of
"now": its length and its mean label rate over the true rate of the same months are reported.
Usage: uv run python drift.py <corrected|linked> -> results_drift_<version>.json, research/drift-or-maturity/figdata/drift_*.csv
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd
import run
from numpy.typing import NDArray

from maturity import reserving

log = logging.getLogger(__name__)
HERE: Final = Path(__file__).resolve().parent
ART: Final = HERE / "artifacts"
FIG: Final = HERE.parents[1] / "research" / "drift-or-maturity" / "figdata"
Array = NDArray[np.float64]
Alarms = list[tuple[int, int]]  # (position in the stream, +1 up / -1 down)
H: Final = 84
MIN_AGE: Final = 24  # months; younger origination months are not in the stream
MIN_LOANS: Final = 2000
STREAMS: Final = ("truth", "naive", "cl", "cal", "full")
# detector settings: the first of each is the usual default, the second is stricter
ADWIN_DELTA: Final = (0.002, 1e-5)
PH_SETTINGS: Final = ((0.005, 50.0), (0.01, 100.0))  # (delta, threshold)
CALM: Final = (2010 * 12 + 1, 2015 * 12 + 12)  # originations 2010 to 2015
RISE: Final = 2004 * 12 + 1  # an up alarm at an origination month from 2004 on: the rise is seen
MATCH: Final = 6  # months between an alarm and the truth alarm that confirms it


def adwin(n: Array, mean: Array, variance: Array, top: Array, delta: float) -> tuple[Alarms, int]:
    """Month t has n[t] loans with labels in [0, top[t]], mean mean[t] and variance variance[t]. Returns the alarms
    and the first month of the window that is left at the end of the stream."""
    alarms, lo = [], 0
    for t in range(len(n)):
        direction = 0
        while t > lo:
            w = slice(lo, t + 1)
            total, s = n[w].sum(), np.cumsum(n[w] * mean[w])
            mu = s[-1] / total
            sigma2 = float((n[w] * (variance[w] + (mean[w] - mu) ** 2)).sum() / total)
            n0 = np.cumsum(n[w])[:-1]
            gap = (s[-1] - s[:-1]) / (total - n0) - s[:-1] / n0
            m, log_term = 1 / (1 / n0 + 1 / (total - n0)), np.log(2 * np.log(total) / delta)
            bound = np.sqrt(2 * sigma2 * log_term / m) + 2 * top[w].max() * log_term / (3 * m)
            excess = np.abs(gap) - bound
            if excess.max() <= 0:
                break
            direction = int(np.sign(gap[np.argmax(excess)]))
            lo += 1
        if direction:
            alarms.append((t, direction))
    return alarms, lo


def page_hinkley(n: Array, mean: Array, delta: float, threshold: float) -> Alarms:
    alarms: Alarms = []
    count = total = up = down = low = high = 0.0
    for t in range(len(n)):
        count, total = count + n[t], total + n[t] * mean[t]
        up += n[t] * (mean[t] - total / count - delta)
        down += n[t] * (mean[t] - total / count + delta)
        low, high = min(low, up), max(high, down)
        if up - low > threshold or high - down > threshold:
            alarms.append((t, 1 if up - low > threshold else -1))
            count = total = up = down = low = high = 0.0
    return alarms


def self_check() -> None:
    """A constant rate gives no alarm; a step gives an alarm in the right direction soon after the step."""
    rng = np.random.default_rng(0)
    n = np.full(120, 15_000.0)
    for step, sign in ((1.0, 0), (3.0, 1), (0.3, -1)):
        rate = np.where(np.arange(120) < 60, 0.02, 0.02 * step)
        mean = rng.binomial(15_000, rate) / 15_000
        for alarms in (
            adwin(n, mean, mean * (1 - mean), np.ones(120), ADWIN_DELTA[0])[0],
            page_hinkley(n, mean, *PH_SETTINGS[0]),
        ):
            assert (not alarms) if sign == 0 else alarms[0] == (60, sign), (step, alarms)


def link_ratio_completeness(seen: Array, n: Array, age: NDArray[np.int64]) -> Array:
    """Chain ladder on monthly cohorts: completeness at ages 0..H = product of the inverse link ratios from that age
    to H. seen[m, a] = incidence of month m at age a (nan after the month's age)."""
    ratio = np.ones(H)
    for a in range(H):
        rows = age >= a + 1
        low = float((n[rows] * seen[rows, a]).sum())
        ratio[a] = float((n[rows] * seen[rows, a + 1]).sum()) / low if low > 0 else np.inf
    return np.concatenate([1 / np.cumprod(ratio[::-1])[::-1], [1.0]])


def streams(z: dict[str, Array], s: int, version: str) -> pd.DataFrame:
    """One row per origination month in the stream at snapshot s: loans, age, and the label rate per stream with
    the completeness behind it."""
    om, n = z["om"], z["n"].astype(float)
    born = s - om >= 0
    age = np.clip(s - om, 0, H)
    seen = z[f"seen_{s}"].astype(float)
    rows = (s - om >= MIN_AGE) & (n >= MIN_LOANS) & ~np.isnan(seen[np.arange(len(om)), age])
    cl = link_ratio_completeness(seen[born], n[born], age[born])[age]
    t = pd.DataFrame(
        {
            "om": om,
            "n": n,
            "age": age,
            "truth": z["truth"],
            "naive": seen[np.arange(len(om)), age],
            "e_cl": cl,
        }
    )[rows].reset_index(drop=True)
    if s % 12 == 0:
        months = run.month_completeness(version, (s - 12) // 12)[1].reindex(t.om.to_numpy())
        for name in ("cal", "full"):
            e = months[name].to_numpy()
            t[f"e_{name}"] = np.where(t.age >= H, 1.0, np.where(np.isnan(e), t.e_cl, e))
    for name in ("cl", "cal", "full"):
        if f"e_{name}" in t:
            t[name] = reserving.chain_ladder(
                t.naive.to_numpy(), t[f"e_{name}"].to_numpy(), np.ones(len(t))
            )
    return t


def detect(t: pd.DataFrame, stream: str) -> tuple[dict[str, Alarms], dict[str, int]]:
    """Alarms of each detector setting on one stream, and the start of the final ADWIN window."""
    n, mean = t.n.to_numpy(), t[stream].to_numpy()
    e = t[f"e_{stream}"].to_numpy() if f"e_{stream}" in t else np.ones(len(t))
    e = np.maximum(e, 1e-6)
    seen = mean * e  # a corrected label is 0 or 1 / e
    windows = {f"adwin({d})": adwin(n, mean, seen / e**2 - mean**2, 1 / e, d) for d in ADWIN_DELTA}
    alarms = {k: v[0] for k, v in windows.items()}
    alarms |= {f"ph({d},{th:.0f})": page_hinkley(n, mean, d, th) for d, th in PH_SETTINGS}
    return alarms, {k: v[1] for k, v in windows.items()}


def month(mi: int) -> str:
    return f"{(mi - 1) // 12}-{(mi - 1) % 12 + 1:02d}"


def confirmed(a: pd.DataFrame) -> pd.Series:
    """An alarm is confirmed when the truth stream (same snapshot, detector and direction) has an alarm at most
    MATCH months away."""
    keys = ["snapshot", "detector", "direction"]
    truth = a[a.stream == "truth"].groupby(keys).om.apply(np.array)
    near = [
        k in truth.index and bool((np.abs(truth[k] - om) <= MATCH).any())
        for k, om in zip(a[keys].itertuples(index=False, name=None), a.om, strict=True)
    ]
    return pd.Series(near, index=a.index)


def main(version: str) -> None:
    self_check()
    z = dict(np.load(ART / f"months_{version}.npz"))
    FIG.mkdir(exist_ok=True)
    rows, windows, values = [], [], []
    for s in z["snapshots"].tolist():
        t = streams(z, s, version)
        values.append(t.assign(snapshot=month(s), month=t.om.map(month)))
        for stream in (st for st in STREAMS if st in t):
            alarms, starts = detect(t, stream)
            for detector, found in alarms.items():
                rows += [
                    {
                        "snapshot": s,
                        "stream": stream,
                        "detector": detector,
                        "om": int(t.om[i]),
                        "direction": d,
                        "immature": bool(t.age[i] < H),
                    }
                    for i, d in found
                ]
            for detector, lo in starts.items():
                w = t.iloc[lo:]
                windows.append(
                    {
                        "snapshot": s,
                        "stream": stream,
                        "detector": detector,
                        "months": len(w),
                        "estimate_over_truth": float(
                            (w.n * w[stream]).sum() / (w.n * w.truth).sum()
                        ),
                    }
                )
    a = pd.DataFrame(rows)
    a["confirmed"] = confirmed(a)
    young = a[a.immature]
    december = young[young.snapshot % 12 == 0]
    n_december = int((z["snapshots"] % 12 == 0).sum())

    def counts(g: pd.DataFrame) -> dict[str, int]:
        up, down = g[g.direction == 1], g[g.direction == -1]
        return {
            "up": len(up),
            "down": len(down),
            "false_up": int((~up.confirmed).sum()),
            "false_down": int((~down.confirmed).sum()),
            "snapshots_with_false_up": int(up[~up.confirmed].snapshot.nunique()),
            "snapshots_with_false_down": int(down[~down.confirmed].snapshot.nunique()),
        }

    calm = december[december.om.between(*CALM)]
    rise = young[(young.direction == 1) & (young.om >= RISE)]
    first = rise.groupby(["stream", "detector"]).snapshot.min()
    first_december = rise[rise.snapshot % 12 == 0].groupby(["stream", "detector"]).snapshot.min()
    w = pd.DataFrame(windows)
    w = w[w.snapshot % 12 == 0]
    out = {
        "streams": f"origination months with at least {MIN_AGE} months of age and {MIN_LOANS} loans",
        "december_snapshots": n_december,
        "alarms_at_immature_months_december_snapshots": {
            f"{st} {d}": counts(g) for (st, d), g in december.groupby(["stream", "detector"])
        },
        "alarms_at_originations_2010_to_2015_december_snapshots": {
            f"{st} {d}": counts(g) for (st, d), g in calm.groupby(["stream", "detector"])
        },
        "adwin_final_window_december_snapshots": {
            f"{st} {d}": {
                "median_months": float(g.months.median()),
                "median_estimate_over_truth": round(float(g.estimate_over_truth.median()), 3),
                "range_estimate_over_truth": [
                    round(float(g.estimate_over_truth.min()), 3),
                    round(float(g.estimate_over_truth.max()), 3),
                ],
            }
            for (st, d), g in w.groupby(["stream", "detector"])
        },
        "first_snapshot_with_up_alarm_at_originations_from_2004": {
            "monthly_snapshots_2006-12_to_2009-12": {
                f"{st} {d}": month(v)
                for (st, d), v in first.items()
                if st in ("truth", "naive", "cl")
            },
            "december_snapshots": {f"{st} {d}": month(v) for (st, d), v in first_december.items()},
        },
    }
    (HERE / f"results_drift_{version}.json").write_text(json.dumps(out, indent=1))
    if version == "corrected":
        a.assign(snapshot=a.snapshot.map(month), month=a.om.map(month)).drop(columns="om").to_csv(
            FIG / "drift_detectors.csv", index=False
        )
        cols = ["snapshot", "month", "n", "age", *STREAMS, "e_cl", "e_cal", "e_full"]
        pd.concat(values, ignore_index=True)[cols].round(6).to_csv(
            FIG / "drift_streams.csv", index=False
        )
    log.info("%s: %d alarms", version, len(a))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    main(sys.argv[1])
