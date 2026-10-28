"""Answers to the review questions 1 to 5 (2026-10-05). No new estimate: everything is computed from the stored
pair files and results of the runs.

DATASETS (through stored artifacts only): Freddie Mac 200k sample of the full Standard files (0011 as run, 0012
corrected and no-HARP, 0018 borrower-followed); USPTO PatEx (0021). Aggregates only.
1. Cluster sign-flip test. d = paired difference of log absolute relative error of a scored (cohort, snapshot) pair
   (the definition of the pre-specified vintage-level statistic). D_c = sum of d over the pairs of cohort c. Test
   statistic = sum of D_c. Under the null the sign of each D_c is exchangeable: the two-sided p value is the share of
   all 2^C sign vectors with |sum of s_c D_c| >= the observed value. All 2^C vectors are used (C <= 18), so it is exact.
2. The persistence (forecast) version of P1: "ours, persistence" against the chain ladder.
3. The speed model that the evidence chose at each Freddie snapshot.
4. Freddie intervals at M = 1, 1.5, 2 and the breakdown value M*.
5. Interval width of ours against the chain ladder's interval on three scales.
Usage: uv run python review.py -> results.json
"""

from __future__ import annotations

import json
import math
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
import backtest as patents  # vintage_level: the cluster bootstrap of the pre-specified statistic

ALPHA: Final = 0.10
FREDDIE: Final = {
    "as run": EXP / "0011-freddie-200k-confirmatory" / "artifacts",
    "corrected": EXP / "0012-freddie-200k-harp-correction" / "artifacts" / "ipcw",
    "no HARP rule": EXP / "0012-freddie-200k-harp-correction" / "artifacts" / "noharp",
    "follows borrowers": EXP / "0018-freddie-200k-follow-borrowers" / "artifacts" / "linked",
}


def freddie_pairs(folder: Path) -> pd.DataFrame:
    return pd.concat(
        [pd.read_csv(folder / f"vintages_{s}.csv") for s in ("E3a", "E3b")], ignore_index=True
    )


def log_difference(v: pd.DataFrame, a: str, b: str) -> pd.Series:
    eps = 1e-6
    return np.log(np.abs(v[a] - v.truth) / v.truth + eps) - np.log(
        np.abs(v[b] - v.truth) / v.truth + eps
    )


def sign_flip(d: pd.Series, cohort: pd.Series) -> dict[str, object]:
    by_cluster = d.groupby(cohort).sum().to_numpy()
    c = len(by_cluster)
    signs = 1 - 2 * ((np.arange(2**c)[:, None] >> np.arange(c)) & 1)
    flipped = np.abs(signs @ by_cluster)
    observed = abs(float(by_cluster.sum()))
    return {
        "clusters": c,
        "sign_vectors": 2**c,
        "clusters_where_a_is_better": int((by_cluster < 0).sum()),
        "sign_flip_p": round(float((flipped >= observed - 1e-12).mean()), 5),
    }


def compare(
    v: pd.DataFrame, a: str, b: str, stored: dict[str, object] | None = None
) -> dict[str, object]:
    """stored = the vintage-level statistic of the run (computed at full precision). The pair files have 6
    decimals, so a pair with an almost exact estimate can get a different log error. If the mean differs from the
    stored mean by more than 0.01, the pair with the smallest relative error gets the difference back, and the
    sign-flip p value is given for both versions."""
    boot = patents.vintage_level(v, a, b)
    d = log_difference(v, a, b)
    err = pd.DataFrame(
        {m: (np.abs(v[m] - v.truth) / v.truth).groupby(v.snapshot).mean() for m in (a, b)}
    )
    keys = ("error_ratio", "mean_paired_log_difference", "ci95")
    out: dict[str, object] = {
        "pairs": boot["pairs"],
        "from_pair_file": {k: boot[k] for k in keys},
        "stored_in_the_run": {k: stored[k] for k in keys} if stored else None,
        "snapshot_sign_test": f"{int((err[a] < err[b]).sum())}/{int((err[a] != err[b]).sum())}",
    } | sign_flip(d, v.cohort)
    if stored and abs(float(d.mean()) - float(stored["mean_paired_log_difference"])) > 0.01:  # type: ignore[arg-type]
        rel = np.minimum(np.abs(v[a] - v.truth), np.abs(v[b] - v.truth)) / v.truth
        i = int(rel.idxmin())
        fixed = d.copy()
        fixed[i] += len(d) * (float(stored["mean_paired_log_difference"]) - float(d.mean()))  # type: ignore[arg-type]
        out["near_exact_pair"] = {
            "snapshot": int(v.snapshot[i]),
            "cohort": int(v.cohort[i]),
            "relative_error_in_file": float(rel[i]),
        }
        out["sign_flip_p_with_that_pair_at_full_precision"] = sign_flip(fixed, v.cohort)[
            "sign_flip_p"
        ]
    return out


def limits(scored: pd.DataFrame, cal: pd.DataFrame, method: str, horizon: int) -> pd.DataFrame:
    """Lower and upper limit (M = 1) of each scored pair: the rule of the runs (T2d section 6)."""
    rows = []
    for r in scored.itertuples():
        c = cal[
            (cal.snapshot < r.snapshot)
            & ((cal.A - r.A).abs() <= 6)
            & (cal.A + 12 * (r.snapshot - cal.snapshot) >= horizon)
        ]
        res = np.sort(np.abs(c[f"r_{method}"].to_numpy()))
        j, est = len(res), float(getattr(r, method))
        if j == 0 or est <= 0:
            continue
        q = float(res[min(j, math.ceil((j + 1) * (1 - ALPHA))) - 1])
        rows.append(
            {
                "snapshot": r.snapshot,
                "cohort": r.cohort,
                "J": j,
                "lo": max(float(r.g_a), est * math.exp(-q)),
                "hi": min(1.0, est * math.exp(q)),
            }
        )
    return pd.DataFrame(rows)


def widths(scored: pd.DataFrame, cal: pd.DataFrame, horizon: int, j_min: int) -> dict[str, object]:
    w = limits(scored, cal, "ours", horizon).merge(
        limits(scored, cal, "chain", horizon),
        on=["snapshot", "cohort", "J"],
        suffixes=("", "_chain"),
    )
    w = w[w.J >= j_min]
    ratio, ratio_chain = w.hi / w.lo, w.hi_chain / w.lo_chain
    return {
        "pairs": len(w),
        "median_upper_over_lower": {
            "ours": round(float(ratio.median()), 2),
            "chain": round(float(ratio_chain.median()), 2),
        },
        "median_width_in_points": {
            "ours": round(float(100 * (w.hi - w.lo).median()), 2),
            "chain": round(float(100 * (w.hi_chain - w.lo_chain).median()), 2),
        },
        "ratio_of_upper_over_lower__as_reported": round(float((ratio / ratio_chain).median()), 3),
        "ratio_of_log_widths": round(float((np.log(ratio) / np.log(ratio_chain)).median()), 3),
        "ratio_of_absolute_widths": round(
            float(((w.hi - w.lo) / (w.hi_chain - w.lo_chain)).median()), 3
        ),
    }


def freddie_intervals(folder: Path) -> dict[str, object]:
    out: dict[str, object] = {}
    for method in ("ours", "chain"):
        c = pd.read_csv(folder / f"conformal_{method}.csv")
        ok, full = c[c.J >= 1], c[c.J >= 9]
        out[method] = {
            "vintages_with_J_ge_1": len(ok),
            "vintages_with_J_ge_9": len(full),
            "vintages_with_J_ge_9_and_an_interval": int(full["width_M1.0"].notna().sum()),
            "covered_M1_of_all_J_ge_9": f"{int(full['covered_M1.0'].fillna(0).astype(float).sum())}/{len(full)}",
            **{
                f"coverage_M{m}_J_ge_9": round(float(full[f"covered_M{m}"].astype(float).mean()), 3)
                for m in (1.0, 1.5, 2.0)
            },
            **{
                f"coverage_M{m}_J_ge_1": round(float(ok[f"covered_M{m}"].astype(float).mean()), 3)
                for m in (1.0, 1.5, 2.0)
            },
            **{
                f"median_upper_over_lower_M{m}_J_ge_9": round(
                    float(full[f"width_M{m}"].median()), 2
                )
                for m in (1.0, 1.5, 2.0)
            },
            "M_star_J_ge_9": {
                "median": round(float(full.m_star.median()), 2),
                "quartiles": np.round(full.m_star.quantile([0.25, 0.75]), 2).tolist(),
                "share_above_1": round(float((full.m_star > 1).mean()), 3),
                "share_above_1.5": round(float((full.m_star > 1.5).mean()), 3),
                "share_above_2": round(float((full.m_star > 2).mean()), 3),
            },
        }
    return out


def main() -> None:
    out: dict[str, object] = {}
    freddie: dict[str, object] = {}
    for name, folder in FREDDIE.items():
        v = freddie_pairs(folder)
        results = json.loads((folder / "summary.json").read_text())
        pooled = {
            k: s["vintage_level"] for k, s in results["secondary"].items() if "vintage_level" in s
        }
        p1 = results["primary"]["P1 (ii) ours vs chain, E3a+E3b pooled, vintage-level"]
        freddie[name] = {
            "P1 ours vs chain": compare(v, "ours", "chain", p1),
            "P3 ours vs naive": compare(v, "ours", "naive"),
            "P3 ours vs drop_recent": compare(v, "ours", "drop_recent"),
            "persistence: ours_uncond vs chain": compare(
                v, "ours_uncond", "chain", pooled["ours_uncond vs chain, E3a+E3b pooled"]
            ),
            "calendar + chain vs chain": compare(
                v, "cal_chain", "chain", pooled["cal_chain vs chain, E3a+E3b pooled"]
            ),
            "speed_model_by_snapshot": {
                year: s["model"]
                for part in ("E3a", "E3b")
                for year, s in results[part]["snapshots"].items()
            },
            "calendar_step_on_by_snapshot": {
                year: s["calendar_on"]
                for part in ("E3a", "E3b")
                for year, s in results[part]["snapshots"].items()
            },
            "intervals": freddie_intervals(folder),
            "widths_M1_J_ge_9": widths(v, pd.read_csv(folder / "calibration_errors.csv"), 84, 9),
        }
    out["freddie"] = freddie
    p = pd.read_csv(EXP / "0021-patex-backtest" / "artifacts" / "pairs_main.csv")
    main_run = json.loads((EXP / "0021-patex-backtest" / "results.json").read_text())["main"]
    out["patents"] = {
        "P1 ours vs chain": compare(
            p, "ours", "chain", main_run["primary"]["P1 ours vs chain"]["vintage_level"]
        ),
        "P2 ours vs cal_chain": compare(
            p, "ours", "cal_chain", main_run["primary"]["P2 ours vs cal_chain"]["vintage_level"]
        ),
        "P3 ours vs naive": compare(p, "ours", "naive"),
        "P3 ours vs drop_recent": compare(p, "ours", "drop_recent"),
        "P3 ours vs fixed_horizon": compare(p, "ours", "fixed_horizon"),
        "persistence: ours_uncond vs chain": compare(
            p,
            "ours_uncond",
            "chain",
            main_run["secondary"]["ours_uncond vs chain"]["vintage_level"],
        ),
        "widths_M1_J_ge_1": widths(
            p,
            pd.read_csv(EXP / "0021-patex-backtest" / "artifacts" / "calibration_main.csv"),
            72,
            1,
        ),
    }
    (HERE / "results.json").write_text(json.dumps(out, indent=1, default=str))


def _self_check() -> None:
    """Three clusters with sums -1, -2, -3: only 2 of the 8 sign vectors reach |6|."""
    assert sign_flip(pd.Series([-1.0, -2.0, -3.0]), pd.Series([1, 2, 3]))["sign_flip_p"] == 0.25


if __name__ == "__main__":
    _self_check()
    main()
