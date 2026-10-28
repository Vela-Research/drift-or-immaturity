"""Numbers for the v10 answers (POST HOC; the v10 review requests).

DATASET: no new data; stored outputs only, plus the hazard classifier of hazard_classifier.py.
  1  patents: the pair files of experiments/0031 (`main`, and `no_own` = the pending inventory without the filing years
     S-4 .. S-1) and of 0066 (hazard model M1 with that inventory); scoring as Table 2 (maturity.inference.compare)
  2  mortgages, classifiers: artifacts/parts of hazard_classifier.py and the stored outputs of 0033 (`corrected`,
     design `vintage`); scoring as 0033 report.py
  3  drift alarms: the alarms of experiments/0039 alarms.py (recomputed by its code), with a one-to-one matching
  4  simulation: results.json of 0034 (frozen code) and 0037 (corrected code)
Usage: uv run python report.py -> results.json
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from maturity import inference

HERE: Final = Path(__file__).resolve().parent
EXP: Final = HERE.parent
sys.path[:0] = [str(EXP / "0039-classifier-followup"), str(EXP / "0033-downstream-classifier")]
import alarms  # the alarm counts of 0039

SETS: Final = {"crisis 2006-2012": range(2006, 2013), "pre-COVID 2013-2018": range(2013, 2019)}
PATENT_ROWS: Final = ["chain", "cal_chain", "cal_chain_forecast", "ours", "ours_uncond"]


def compare(p: pd.DataFrame, a: str, b: str, truth: str = "truth") -> dict[str, object]:
    c = inference.compare(
        ((p[a] - p[truth]) / p[truth]).to_numpy(),
        ((p[b] - p[truth]) / p[truth]).to_numpy(),
        p.cohort.to_numpy().astype(np.int64),
        p.snapshot.to_numpy().astype(np.int64),
    )
    low, high = (math.exp(x) for x in c.ci95_cluster_bootstrap)
    return {
        "ratio": round(c.error_ratio, 3),
        "ci95": [round(low, 3), round(high, 3)],
        "sign_flip_p": round(c.sign_flip_p, 4),
        "mae_ratio": round(c.mae_ratio, 3),
        "snapshots_a_better": c.snapshots_where_a_is_better,
        "snapshots": c.snapshots,
    }


def median_error(p: pd.DataFrame, m: str, truth: str = "truth") -> float:
    error = ((p[m] - p[truth]) / p[truth]).abs()
    return round(float(error.groupby(p.snapshot).mean().median()), 4)


def patents() -> dict[str, object]:
    folder = EXP / "0031-round3-backtests" / "artifacts"
    main, own = (pd.read_parquet(folder / f"pairs_patents_{k}.parquet") for k in ("main", "no_own"))
    assert np.allclose(main.truth, own.truth) and np.allclose(main.chain, own.chain)
    hazard = pd.read_parquet(
        EXP / "0066-patent-hazard-baseline" / "artifacts" / "pairs_hazard.parquet"
    )
    own = own.merge(
        hazard[["snapshot", "cohort", "m1own_real", "m1own_fcst"]],
        on=["snapshot", "cohort"],
        validate="one_to_one",
    )
    big = (own.truth - own.ref).abs() >= 0.02
    sign = np.sign(own.truth - own.ref)[big]
    rows = {}
    for m in [*PATENT_ROWS, "m1own_real", "m1own_fcst"]:
        rows[m] = {
            "median_error": median_error(own, m),
            "vs_chain": compare(own, m, "chain"),
            "wrong_direction": int((np.sign(own[m] - own.ref)[big] != sign).sum()),
        }
    steps = {
        "calendar step, realised (cal_chain vs stratified)": compare(
            own, "cal_chain", "stratified"
        ),
        "calendar step, forecast (cal_chain_forecast vs stratified)": compare(
            own, "cal_chain_forecast", "stratified"
        ),
        "speed step, realised (ours vs cal_chain)": compare(own, "ours", "cal_chain"),
        "speed step, forecast (ours_uncond vs cal_chain_forecast)": compare(
            own, "ours_uncond", "cal_chain_forecast"
        ),
    }
    return {"pairs": len(own), "pairs_for_direction": int(big.sum()), "rows": rows, "steps": steps}


def classifier() -> dict[str, object]:
    old = EXP / "0033-downstream-classifier"
    parts = HERE / "artifacts" / "parts"
    years = sorted(int(f.stem) for f in parts.glob("*.json"))
    stored = pd.read_parquet(old / "artifacts" / "pairs_corrected_vintage.parquet")
    new = pd.concat(pd.read_parquet(parts / f"{y}.parquet") for y in years)
    p = stored.merge(new, on=["snapshot", "cohort"], suffixes=("", "_new"), validate="one_to_one")
    assert np.allclose(p.realised, p.realised_new, atol=1e-12), "not the same test loans"
    results = json.loads((old / "results_corrected_vintage.json").read_text())
    reps = dict(np.load(old / "artifacts" / "auc_bootstrap_corrected_vintage.npz"))
    metrics = {y: json.loads((parts / f"{y}.json").read_text()) for y in years}
    methods = [
        "hazard_logit",
        "final_logit",
        "naive_logit",
        "dfm_logit",
        "pu_cl_logit",
        "pu_full_logit",
    ]

    def metric(m: str, y: int, key: str) -> float:
        return float(
            metrics[y]["metrics"][key]
            if m == "hazard_logit"
            else results[str(y)]["methods"][m][key]
        )

    error = (
        p[methods].sub(p.realised, axis=0).div(p.realised, axis=0).abs().groupby(p.snapshot).mean()
    )
    summary = {
        name: {
            m: {
                "level_error": round(float(error.loc[error.index.isin(ys), m].median()), 4),
                **{
                    k: round(float(np.median([metric(m, y, k) for y in ys if y in metrics])), 4)
                    for k in ("level_ratio", "auc", "auc_within_vintage", "slow_over_fast", "ece")
                },
            }
            for m in methods
        }
        for name, ys in SETS.items()
    }
    tests = {f"hazard_logit vs {b}": compare(p, "hazard_logit", b, "realised") for b in methods[1:]}
    auc = {}
    for b in methods[1:]:
        diff = [np.load(parts / f"{y}.npy") - reps[f"{y}_{b}"] for y in years]
        point = [metric("hazard_logit", y, "auc") - metric(b, y, "auc") for y in years]
        low, high = np.array([np.percentile(d, [2.5, 97.5]) for d in diff]).T
        auc[f"hazard_logit vs {b}"] = {
            "median_difference": round(float(np.median(point)), 4),
            "range": [round(float(min(point)), 4), round(float(max(point)), 4)],
            "snapshots_interval_above_0": int((low > 0).sum()),
            "snapshots_interval_below_0": int((high < 0).sum()),
        }
    fit = {str(y): {k: v for k, v in m.items() if k != "metrics"} for y, m in metrics.items()}
    return {
        "snapshots": years,
        "pairs": len(p),
        "summary": summary,
        "level_tests": tests,
        "auc": auc,
        "fit": fit,
    }


def one_to_one(a: NDArray[np.int64], b: NDArray[np.int64]) -> int:
    """Largest number of pairs (one alarm of a, one of b, each used once) at most MATCH months apart: greedy on the
    sorted months is optimal for this interval rule."""
    a, b = np.sort(a), np.sort(b)
    i = j = n = 0
    while i < len(a) and j < len(b):
        if abs(a[i] - b[j]) <= alarms.drift.MATCH:
            i, j, n = i + 1, j + 1, n + 1
        elif a[i] < b[j]:
            i += 1
        else:
            j += 1
    return n


def alarm_matching() -> dict[str, object]:
    a = alarms.all_alarms()
    d = a[a.immature & (a.snapshot % 12 == 0) & (a.direction == -1)]
    out = {}
    for detector in ("adwin(0.002)", "ph(0.005,50)"):
        g = d[d.detector == detector]
        truth = g[g.stream == "truth"]
        for stream in ("naive", "cl", "cal", "full"):
            s = g[g.stream == stream]
            confirmed = alarms.near(s, truth)
            hit = alarms.near(truth, s)
            paired = sum(
                one_to_one(
                    s[s.snapshot == y].om.to_numpy(), truth[truth.snapshot == y].om.to_numpy()
                )
                for y in sorted(g.snapshot.unique())
            )
            out[f"{detector} {stream}"] = {
                "alarms": len(s),
                "truth_alarms": len(truth),
                "rule_of_the_paper": {
                    "confirmed": int(confirmed.sum()),
                    "false": int((~confirmed).sum()),
                    "truth_alarms_missed": int((~hit).sum()),
                    "truth_alarms_hit": int(hit.sum()),
                },
                "one_to_one": {
                    "matched": paired,
                    "false": len(s) - paired,
                    "truth_alarms_missed": len(truth) - paired,
                },
            }
    return out


def simulation() -> dict[str, object]:
    keys = ("naive", "chain", "cal_chain", "ours", "cal_chain_forecast", "ours_uncond")
    out = {}
    for code, file, prefix in (
        ("frozen", EXP / "0034-end-to-end-simulation" / "results.json", ()),
        ("corrected", EXP / "0037-corrected-plugin-simulation" / "results.json", ("corrected",)),
    ):
        s = json.loads(file.read_text())["settings"]
        for k in prefix:
            s = s[k]
        for timing in ("weibull", "lognormal"):
            for calendar in ("calendar", "no calendar"):
                rows = {
                    key: v["sign_recovery"]
                    for key, v in s.items()
                    if key.split("/")[0] == timing and key.split("/")[2] == calendar
                }
                if rows:
                    out[f"{code} {timing} {calendar}"] = {
                        "range": {
                            m: [min(r[m] for r in rows.values()), max(r[m] for r in rows.values())]
                            for m in keys
                        },
                        "settings": rows,
                    }
    return out


def main() -> None:
    out = {"patents_without_own": patents(), "alarms": alarm_matching(), "simulation": simulation()}
    if (HERE / "artifacts" / "parts").exists():
        out["hazard_classifier"] = classifier()
    (HERE / "results.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
