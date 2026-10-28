"""Frozen-spec evaluation (the frozen mortgage specification sections 1-10) on the 200k sample. CONFIRMATORY.

E3a  snapshots 2006-12..2012-12, truth = 7-year default incidence
E3b  snapshots 2013-12..2018-12, truth = incidence up to min(84 months, 2020-02); vintage kept if >= 12 months ahead
E3c  snapshots 2013-12..2019-12, truth = 7-year incidence (COVID-affected; reported, not used for claims)
P1 ours vs chain ladder, E3a + E3b pooled: (i) sign test over snapshots, (ii) mean paired log error difference with a
   cluster bootstrap over vintages; P3 ours vs naive and drop-recent (E3a + E3b);
P4 coverage of the 90% conformal interval (M = 1), calibrated on the same method's out-of-sample backtest errors of
   quarterly cohorts at the same age (T2d section 6).
Secondary: calendar + chain ladder, unconditional (persistence), rolling-origin omega2, H = 10 years, MTM LTV, E3c,
   T3 diagnostic columns; then the over-identification test of experiments/0003 on the 200k cells.
Usage: cells.py, then `PYTHONHASHSEED=0 uv run python run.py` -> artifacts/*.csv, artifacts/summary.json,
artifacts/overid_test*.{csv,json}. The hash seed is fixed because the fixed-effects fit (pyfixest) depends on set order
at the 1e-13 level, and the estimate of a cohort with about 5 months of follow-up amplifies this to about 0.1%.
"""

from __future__ import annotations

import json
import logging
import math
import os
import sys
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd
import pipeline as pl
from scipy import stats

HERE: Final = Path(__file__).resolve().parent
OUT_DIR: Final = HERE / "artifacts"
sys.path.insert(0, str(HERE.parent / "0003-freddie-sample"))
import overid_test

log = logging.getLogger(__name__)
H: Final = 84
CUT: Final = int(pl.month_idx(202002))
N_YOUNG: Final = 6
ALPHA: Final = 0.10
MS: Final = (1.0, 1.5, 2.0)
SETS: Final = {"E3a": range(2006, 2013), "E3b": range(2013, 2019), "E3c": range(2013, 2020)}
M: Final = pl.METHODS


def scored(t: pd.DataFrame, min_ahead: int = 1) -> pd.DataFrame:
    """The N_YOUNG most recent young cohorts with a truth and at least `min_ahead` months of future window."""
    y = t[t.young & t.truth.notna() & (t.truth > 0) & (t.h - t.A >= min_ahead)]
    return y.sort_values("cohort").tail(N_YOUNG)


def evaluate(
    d: pl.Data,
    years: range,
    horizon: int,
    cutoff: int | None,
    min_ahead: int,
    tag: str,
    *,
    rolling: bool = False,
) -> tuple[pd.DataFrame, dict[str, dict[str, object]]]:
    tr = pl.truth(d, "year", horizon, cutoff)
    rows, metas = [], {}
    for year in years:
        t, meta = pl.estimate(d, year, horizon=horizon, cutoff=cutoff, rolling=rolling)
        t["truth"] = [
            tr[k][0] if k in tr and (cutoff is not None or k + horizon / 12 <= 2026.25) else np.nan
            for k in t.cohort
        ]
        y = scored(t, min_ahead).assign(snapshot=year)
        rows.append(y)
        metas[str(year)] = {k: v for k, v in meta.items() if k != "f0"}
        log.info(
            "%s %d model=%s phi=%s on=%s err %s",
            tag,
            year,
            meta["model"],
            meta["phi"],
            meta["calendar_on"],
            {
                m: round(float((np.abs(y[m] - y.truth) / y.truth).mean()), 3)
                for m in ("chain", "cal_chain", "ours", "ours_uncond")
            },
        )
    return pd.concat(rows, ignore_index=True), metas


def snapshot_errors(v: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            m: (np.abs(v[m] - v.truth) / v.truth).groupby(v.snapshot).mean()
            for m in [*M, "ours_rolling"]
            if m in v
        }
    )


def paired(err: pd.DataFrame, a: str, b: str) -> dict[str, float]:
    diff = (err[a] - err[b]).to_numpy()
    wins, n = int((diff < 0).sum()), int((diff != 0).sum())
    return {
        "median_paired_difference": round(float(np.median(diff)), 4),
        "wins": wins,
        "n": n,
        "sign_test_p": round(float(stats.binomtest(wins, n, 0.5).pvalue), 3) if n else 1.0,
    }


def pooled_p1(
    v: pd.DataFrame, a: str = "ours", b: str = "chain", n_boot: int = 2000
) -> dict[str, object]:
    """Frozen spec section 9, P1 (ii): mean paired difference in log absolute relative error over all scored
    (vintage, snapshot) pairs; 95% CI by cluster bootstrap over vintages (the same vintage is scored at several
    snapshots). Negative = method `a` has the smaller error."""
    eps = 1e-6
    d = (
        np.log(np.abs(v[a] - v.truth) / v.truth + eps)
        - np.log(np.abs(v[b] - v.truth) / v.truth + eps)
    ).to_numpy()
    cl = v.cohort.to_numpy()
    keys = np.unique(cl)
    idx = {k: np.where(cl == k)[0] for k in keys}
    rng = np.random.default_rng(0)
    boot = np.empty(n_boot)
    for i in range(n_boot):
        pick = np.concatenate([idx[k] for k in rng.choice(keys, size=len(keys), replace=True)])
        boot[i] = d[pick].mean()
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return {
        "pairs": len(d),
        "vintage_clusters": len(keys),
        "mean_paired_log_difference": round(float(d.mean()), 3),
        "ci95": [round(float(lo), 3), round(float(hi), 3)],
        "error_ratio": round(float(np.exp(d.mean())), 3),
    }


def calibration_errors(d: pl.Data, years: range) -> pd.DataFrame:
    """Out-of-sample backtest errors of quarterly cohorts: one row per (pseudo-snapshot, young cohort)."""
    tr = pl.truth(d, "quarter", H)
    rows = []
    for year in years:
        t, _ = pl.estimate(d, year, cohort="quarter")
        y = t[t.young].copy()
        y["truth"] = [tr[k][0] if k in tr and k + H / 12 <= 2026.25 else np.nan for k in y.cohort]
        y = y[y.truth.notna() & (y.truth > 0) & (y.A >= 3)]
        for m in ("ours", "chain"):
            y[f"r_{m}"] = np.log(np.maximum(y[m], 1e-9)) - np.log(y.truth)
        rows.append(y.assign(snapshot=year)[["snapshot", "cohort", "A", "r_ours", "r_chain"]])
        log.info("calibration %d: %d quarterly cohorts", year, len(y))
    return pd.concat(rows, ignore_index=True)


def conformal(v: pd.DataFrame, cal: pd.DataFrame, method: str) -> pd.DataFrame:
    """Intervals for each scored vintage from calibration errors at the same age (within 6 months), using only
    pseudo-snapshots before S and quarterly cohorts that are mature (84 months) at S."""
    out = []
    for r in v.itertuples():
        c = cal[
            (cal.snapshot < r.snapshot)
            & ((cal.A - r.A).abs() <= 6)
            & (cal.A + 12 * (r.snapshot - cal.snapshot) >= H)
        ]
        res = np.sort(np.abs(c[f"r_{method}"].to_numpy()))
        j = len(res)
        row: dict[str, object] = {
            "snapshot": r.snapshot,
            "cohort": r.cohort,
            "A": r.A,
            "J": j,
            "set": r.set,
        }
        est, tru = float(getattr(r, method)), float(r.truth)
        if j == 0 or est <= 0:
            out.append(row | {"level": 0.0})
            continue
        k = min(j, math.ceil((j + 1) * (1 - ALPHA)))
        q = float(res[k - 1])
        row["level"] = round(min(1 - ALPHA, j / (j + 1)), 3)
        row["q"] = q
        for m in MS:
            lo, hi = max(float(r.g_a), est * math.exp(-m * q)), min(1.0, est * math.exp(m * q))
            row[f"covered_M{m}"] = lo <= tru <= hi
            row[f"width_M{m}"] = hi / lo if lo > 0 else np.inf
        signed = np.sort(c[f"r_{method}"].to_numpy())
        q_lo, q_hi = np.quantile(signed, [ALPHA / 2, 1 - ALPHA / 2])
        lo, hi = max(float(r.g_a), est * math.exp(-q_hi)), min(1.0, est * math.exp(-q_lo))
        row["covered_signed"] = lo <= tru <= hi
        if not math.isnan(r.ref):
            row["m_star"] = (
                abs(math.log(est / r.ref)) / q if q > 0 else np.inf
            )  # smallest M at which "no drift" enters
        out.append(row)
    return pd.DataFrame(out)


def main() -> None:
    d = pl.load("base")
    out: dict[str, object] = {}
    vint, errs = {}, {}
    for name, years in SETS.items():
        v, metas = evaluate(
            d,
            years,
            H,
            CUT if name == "E3b" else None,
            12 if name == "E3b" else 1,
            name,
            rolling=name != "E3c",  # rolling-origin omega2: sensitivity (spec section 4)
        )
        v["set"] = name
        v.round(6).to_csv(OUT_DIR / f"vintages_{name}.csv", index=False)
        vint[name], errs[name] = v, snapshot_errors(v)
        out[name] = {
            "snapshots": {
                str(y): {
                    "n": int((v.snapshot == y).sum()),
                    "error": errs[name].loc[y].round(4).to_dict(),
                    **metas[str(y)],
                }
                for y in years
            },
            "median_error": errs[name].median().round(4).to_dict(),
            "win_rate_vs_chain": {
                m: round(float((errs[name][m] < errs[name]["chain"]).mean()), 2)
                for m in M
                if m != "chain"
            },
        }
    ab = pd.concat([errs["E3a"], errs["E3b"]])
    pooled = pd.concat([vint["E3a"], vint["E3b"]], ignore_index=True)
    out["primary"] = {
        "P1 (i) ours vs chain, E3a+E3b pooled, sign test over snapshots": paired(
            ab, "ours", "chain"
        ),
        "P1 (ii) ours vs chain, E3a+E3b pooled, vintage-level": pooled_p1(pooled),
        "descriptive: ours vs chain, E3a": paired(errs["E3a"], "ours", "chain")
        | {"vintage_level": pooled_p1(vint["E3a"])},
        "descriptive: ours vs chain, E3b": paired(errs["E3b"], "ours", "chain")
        | {"vintage_level": pooled_p1(vint["E3b"])},
        "P3 ours vs naive, E3a+E3b": paired(ab, "ours", "naive"),
        "P3 ours vs drop_recent, E3a+E3b": paired(ab, "ours", "drop_recent"),
    }
    out["secondary"] = {
        "ours vs cal_chain, E3a": paired(errs["E3a"], "ours", "cal_chain"),
        "ours vs cal_chain, E3b": paired(errs["E3b"], "ours", "cal_chain"),
        "cal_chain vs chain, E3a": paired(errs["E3a"], "cal_chain", "chain"),
        "cal_chain vs chain, E3b": paired(errs["E3b"], "cal_chain", "chain"),
        "ours_uncond vs chain, E3a": paired(errs["E3a"], "ours_uncond", "chain"),
        "ours_uncond vs chain, E3b": paired(errs["E3b"], "ours_uncond", "chain"),
        "ours vs chain, E3c": paired(errs["E3c"], "ours", "chain"),
        "ours_free vs chain, E3a+E3b": paired(ab, "ours_free", "chain"),
    } | {
        f"{a} vs {b}, E3a+E3b pooled": paired(ab, a, b) | {"vintage_level": pooled_p1(pooled, a, b)}
        for a, b in (
            ("ours", "cal_chain"),
            ("cal_chain", "chain"),
            ("ours_uncond", "chain"),
            ("ours_free", "chain"),
            ("ours_rolling", "chain"),
            ("ours_rolling", "ours"),
        )
    }
    # P4: conformal intervals
    cal = calibration_errors(d, range(2006, 2019))
    cal.round(5).to_csv(OUT_DIR / "calibration_errors.csv", index=False)
    full7 = pl.truth(d, "year", H)
    both = pd.concat([vint["E3a"], vint["E3b"]], ignore_index=True)
    both["ref"] = [
        max((full7[k][0] for k in full7 if k + H / 12 <= s + 1), default=np.nan)
        if False
        else np.nan
        for s in both.snapshot
    ]
    for i, r in enumerate(both.itertuples()):  # most recent vintage that is mature at the snapshot
        mat = [k for k in full7 if k <= r.snapshot - 7]
        both.loc[i, "ref"] = full7[max(mat)][0] if mat else np.nan
    conf = {}
    for method in ("ours", "chain"):
        c = conformal(both, cal, method)
        c.round(5).to_csv(OUT_DIR / f"conformal_{method}.csv", index=False)
        ok = c[c.J >= 1]
        full = c[c.J >= 9]
        conf[method] = {
            "vintages": len(c),
            "with_J_ge_1": len(ok),
            "with_J_ge_9": len(full),
            "J_median": float(c.J.median()),
            "J_by_set": c.groupby("set").J.median().to_dict(),
            **{
                f"coverage_M{m}_all_with_J": round(float(ok[f"covered_M{m}"].mean()), 3) for m in MS
            },
            **{
                f"coverage_M{m}_J_ge_9": round(float(full[f"covered_M{m}"].mean()), 3)
                if len(full)
                else None
                for m in MS
            },
            "coverage_signed_all_with_J": round(float(ok.covered_signed.mean()), 3),
            **{f"median_width_ratio_M{m}": round(float(ok[f"width_M{m}"].median()), 2) for m in MS},
            "coverage_M1_by_set": ok.groupby("set")["covered_M1.0"].mean().round(3).to_dict(),
            "coverage_M1_J_ge_9_by_set": full.groupby("set")["covered_M1.0"]
            .mean()
            .round(3)
            .to_dict(),
            "coverage_signed_J_ge_9": round(float(full.covered_signed.mean()), 3)
            if len(full)
            else None,
            "m_star_median": round(float(ok.m_star.median()), 2),
        }
    w = pd.read_csv(OUT_DIR / "conformal_ours.csv").merge(
        pd.read_csv(OUT_DIR / "conformal_chain.csv"),
        on=["snapshot", "cohort"],
        suffixes=("", "_chain"),
    )
    w = w[w.J >= 9]
    conf[
        "width_ratio_ours_over_chain_J_ge_9"
    ] = {  # median over vintages of (upper/lower)_ours / (upper/lower)_chain
        f"M{m}": round(float((w[f"width_M{m}"] / w[f"width_M{m}_chain"]).median()), 2) for m in MS
    }
    out["P4_conformal"] = conf
    # sensitivities
    v10, _ = evaluate(d, range(2009, 2017), 120, None, 1, "H10")
    v10.round(6).to_csv(OUT_DIR / "vintages_H10.csv", index=False)
    e10 = snapshot_errors(v10)
    out["H10"] = {
        "median_error": e10.median().round(4).to_dict(),
        "ours vs chain": paired(e10, "ours", "chain"),
        "errors": e10.round(4).to_dict(),
    }
    dm = pl.load("mtm")
    mt = {}
    for name in ("E3a", "E3b"):
        v, _ = evaluate(
            dm,
            SETS[name],
            H,
            CUT if name == "E3b" else None,
            12 if name == "E3b" else 1,
            f"MTM-{name}",
        )
        e = snapshot_errors(v)
        mt[name] = {
            "median_error": e.median().round(4).to_dict(),
            "ours vs chain": paired(e, "ours", "chain"),
        }
    out["MTM"] = mt
    for path in (
        OUT_DIR / "summary.json",
        HERE / "results.json",
    ):  # results.json is the tracked copy
        path.write_text(json.dumps(out, indent=2, default=str))
    log.info("primary %s", out["primary"])
    log.info("conformal %s", conf)


if __name__ == "__main__":
    if os.environ.get("PYTHONHASHSEED") != "0":
        raise SystemExit("run with PYTHONHASHSEED=0 (reproducibility, see the module docstring)")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    main()
    overid_test.HERE = HERE  # type: ignore[misc]  # the 0003 test reads and writes under HERE / "artifacts"
    overid_test.main("")
    overid_test.main("mtm")
