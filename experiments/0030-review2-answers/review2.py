"""Answers to the second review (2026-10-05) for mortgages and patents. No new backtest: stored pair files and
calibration files are scored again; only the calibration errors that were not stored are computed again with the
code of the runs (patents: the persistence version of ours).

DATASETS: Freddie Mac 200k (stored files of 0011, 0012, 0018); USPTO PatEx sample of 0020/0021; aggregates only.
Interval rule of all runs (T2d section 6). For a scored pair (annual cohort c, snapshot S, median age A months):
  calibration error = log(estimate) - log(truth at the horizon H) of a QUARTERLY cohort at an earlier December
  snapshot S' < S; it is used if |A' - A| <= 6 months (A' = median age of the quarterly cohort at S') and
  A' + 12 (S - S') >= H (the quarterly cohort is H months old at S); J = number of such errors;
  q = the ceil((J + 1) 0.9)-th smallest |error|; interval = [max(share so far, estimate exp(-M q)),
  min(1, estimate exp(M q))].
"Strict" below asks for A' + 12 (S - S') >= H + 2: then every unit of the quarterly cohort is H months old at S
(the units of the last month of a quarter are 1 month younger than the median unit).
Usage: PYTHONHASHSEED=0 uv run python review2.py -> results.json
"""

from __future__ import annotations

import importlib.util
import json
import logging
import math
import os
import sys
from pathlib import Path
from types import ModuleType
from typing import Final

import numpy as np
import pandas as pd

HERE: Final = Path(__file__).resolve().parent
EXP: Final = HERE.parent


def load(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


sys.path[:0] = [str(EXP / "0021-patex-backtest"), str(EXP / "0012-freddie-200k-harp-correction")]
import backtest as patents  # the patent backtest
import pipeline as pl  # the frozen method

review = load("review", EXP / "0029-review-answers" / "review.py")

log = logging.getLogger(__name__)
ALPHA: Final = 0.10
MS: Final = (1.0, 1.5, 2.0)
FREDDIE: Final = review.FREDDIE


def limits(
    scored: pd.DataFrame, cal: pd.DataFrame, method: str, horizon: int, margin: int = 0
) -> pd.DataFrame:
    """One row per scored pair with at least one calibration error: J, the anatomy of the calibration set and the
    limits for each M. `scored` needs snapshot, cohort, A, g_a, truth and the method's estimate; `cal` needs
    snapshot, cohort, A and r_<method>. A calibration row may carry `months_to_target`."""
    rows = []
    for r in scored.itertuples():
        gap = (
            cal.months_to_target if "months_to_target" in cal else 12 * (r.snapshot - cal.snapshot)
        )
        age_at_target = cal.A + gap
        c = cal[
            (cal.snapshot < r.snapshot)
            & ((cal.A - r.A).abs() <= 6)
            & (age_at_target >= horizon + margin)
        ]
        errors = np.sort(np.abs(c[f"r_{method}"].to_numpy()))
        j, est = len(errors), float(getattr(r, method))
        if j == 0 or est <= 0:
            continue
        q = float(errors[min(j, math.ceil((j + 1) * (1 - ALPHA))) - 1])
        row: dict[str, object] = {
            "snapshot": r.snapshot,
            "cohort": r.cohort,
            "J": j,
            "calibration_snapshots": c.snapshot.nunique(),
            "calibration_cohorts": c.cohort.nunique(),
            "errors_with_margin_below_2_months": int((age_at_target[c.index] < horizon + 2).sum()),
        }
        for m in MS:
            lo, hi = max(float(r.g_a), est * math.exp(-m * q)), min(1.0, est * math.exp(m * q))
            row |= {f"lo_{m}": lo, f"hi_{m}": hi, f"covered_{m}": lo <= float(r.truth) <= hi}
        rows.append(row)
    return pd.DataFrame(rows)


def summary(
    scored: pd.DataFrame, cal: pd.DataFrame, horizon: int, methods: tuple[str, str], j_min: int
) -> dict[str, object]:
    """Coverage, anatomy and width ratios (a against b) on the pairs with J >= j_min; and the strict version."""
    a, b = methods
    out: dict[str, object] = {
        "scored_pairs": len(scored),
        "calibration_errors_in_all": len(cal),
        "quarterly_cohorts_in_all": int(cal.cohort.nunique()),
        "pseudo_snapshots": [int(cal.snapshot.min()), int(cal.snapshot.max())],
        "errors_per_quarterly_cohort_max": int(cal.groupby("cohort").size().max()),
    }
    for label, margin in (("as_run", 0), ("strict", 2)):
        la, lb = (limits(scored, cal, m, horizon, margin) for m in (a, b))
        if la.empty:
            out[label] = {"pairs_with_J_ge_1": 0}
            continue
        keep = la[la.J >= j_min]
        both = keep.merge(lb, on=["snapshot", "cohort"], suffixes=("", "_b"))
        block: dict[str, object] = {
            "pairs_with_J_ge_1": len(la),
            "pairs_with_J_ge_9": int((la.J >= 9).sum()),
            f"pairs_used_J_ge_{j_min}": len(keep),
            "J_min_median_max": [int(keep.J.min()), float(keep.J.median()), int(keep.J.max())],
            "calibration_snapshots_per_interval_min_median_max": [
                int(keep.calibration_snapshots.min()),
                float(keep.calibration_snapshots.median()),
                int(keep.calibration_snapshots.max()),
            ],
            "quarterly_cohorts_per_interval_min_median_max": [
                int(keep.calibration_cohorts.min()),
                float(keep.calibration_cohorts.median()),
                int(keep.calibration_cohorts.max()),
            ],
            "share_of_errors_used_twice_for_one_interval": round(
                float(1 - keep.calibration_cohorts.sum() / keep.J.sum()), 3
            ),
            "share_of_errors_with_margin_below_2_months": round(
                float(keep.errors_with_margin_below_2_months.sum() / keep.J.sum()), 3
            ),
            f"pairs_where_{b}_has_an_interval": len(both),
        }
        for m in MS:
            wa, wb = both[f"hi_{m}"] / both[f"lo_{m}"], both[f"hi_{m}_b"] / both[f"lo_{m}_b"]
            block[f"M{m}"] = {
                f"coverage_{a}": f"{int(keep[f'covered_{m}'].sum())}/{len(keep)}",
                f"coverage_{b}": f"{int(both[f'covered_{m}_b'].sum())}/{len(both)}",
                "median_upper_over_lower": [
                    round(float((keep[f"hi_{m}"] / keep[f"lo_{m}"]).median()), 2),
                    round(float(wb.median()), 2),
                ],
                "ratio_upper_over_lower": round(float((wa / wb).median()), 3),
                "ratio_of_log_widths": round(float((np.log(wa) / np.log(wb)).median()), 3),
            }
        out[label] = block
    return out


def patents_persistence() -> dict[str, object]:
    """Question 5: the calibration errors of the patent backtest again, now also for ours with the persistence path."""
    pl._persistence = patents.persistence
    check = patents.check
    applications = check.load()
    pool = applications[applications.year.isin(check.YEARS)]
    sample = pool.groupby("year").sample(n=check.N_PER_YEAR, random_state=check.SEED)
    d = patents.build(sample, check.inventory(applications))
    truth = pl.truth(d, "quarter", patents.H)
    rows = []
    for year in range(patents.SNAPSHOTS.start, patents.SNAPSHOTS.stop - 1):
        t, _ = pl.estimate(patents.seen_by(d, year), year, cohort="quarter", horizon=patents.H)
        y = t[t.young].copy()
        y["truth"] = [
            truth[k][0] if k + 0.25 + patents.H / 12 <= patents.DATA_END else np.nan
            for k in y.cohort
        ]
        y = y[y.truth.notna() & (y.truth > 0) & (y.A >= 3)]
        for m in ("ours", "ours_uncond", "chain"):
            y[f"r_{m}"] = np.log(np.maximum(y[m], 1e-9)) - np.log(y.truth)
        rows.append(
            y.assign(snapshot=year)[
                ["snapshot", "cohort", "A", "r_ours", "r_ours_uncond", "r_chain"]
            ]
        )
        log.info("patents calibration %d: %d quarterly cohorts", year, len(y))
    cal = pd.concat(rows, ignore_index=True)
    stored = pd.read_csv(EXP / "0021-patex-backtest" / "artifacts" / "calibration_main.csv")
    assert len(stored) == len(cal) and np.allclose(stored.r_ours, cal.r_ours, atol=2e-5), (
        "not the calibration of 0021"
    )
    pairs = pd.read_csv(
        EXP / "0022-group-stratified-chain-ladder" / "artifacts" / "pairs_patents.csv"
    )
    big = pairs[(pairs.truth - pairs.ref).abs() >= 0.02]
    wrong = {
        m: round(float((np.sign(big[m] - big.ref) != np.sign(big.truth - big.ref)).mean()), 3)
        for m in ("chain", "unit_chain", "cal_chain", "ours", "ours_uncond")
    }
    (HERE / "artifacts").mkdir(exist_ok=True)
    cal.round(6).to_csv(HERE / "artifacts" / "calibration_patents.csv", index=False)
    return {
        "P4_ours_realised_path": summary(pairs, cal, patents.H, ("ours", "chain"), 1),
        "P4_ours_persistence_path": summary(pairs, cal, patents.H, ("ours_uncond", "chain"), 1),
        "P4_ours_realised_path_J_ge_9": summary(pairs, cal, patents.H, ("ours", "chain"), 9),
        "P4_ours_persistence_path_J_ge_9": summary(
            pairs, cal, patents.H, ("ours_uncond", "chain"), 9
        ),
        "P5_wrong_direction": {"pairs": len(big), "share": wrong},
        "ours_persistence vs exact-age chain ladder": review.compare(
            pairs, "ours_uncond", "unit_chain"
        ),
        "ours_realised vs exact-age chain ladder": review.compare(pairs, "ours", "unit_chain"),
        "ours_persistence vs chain ladder": review.compare(pairs, "ours_uncond", "chain"),
        "ours_persistence vs calendar + chain (realised)": review.compare(
            pairs, "ours_uncond", "cal_chain"
        ),
        "median_error_over_snapshots": {
            m: round(
                float(
                    (np.abs(pairs[m] - pairs.truth) / pairs.truth)
                    .groupby(pairs.snapshot)
                    .mean()
                    .median()
                ),
                4,
            )
            for m in ("chain", "unit_chain", "cal_chain", "ours", "ours_uncond")
        },
    }


def freddie() -> dict[str, object]:
    out = {}
    for name, folder in FREDDIE.items():
        scored = review.freddie_pairs(folder)
        cal = pd.read_csv(folder / "calibration_errors.csv")
        out[name] = summary(scored, cal, 84, ("ours", "chain"), 9)
    return out


def main() -> None:
    out = {"freddie": freddie(), "patents": patents_persistence()}
    (HERE / "results.json").write_text(json.dumps(out, indent=1, default=str))


if __name__ == "__main__":
    if os.environ.get("PYTHONHASHSEED") != "0":
        raise SystemExit(
            "run with PYTHONHASHSEED=0 (the fixed-effects fit depends on set order, see 0011)"
        )
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    main()
