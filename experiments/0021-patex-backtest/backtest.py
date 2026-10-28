"""Pre-registered forecasting backtest on US patent applications (the pre-registered patent specification).

DATASET: USPTO PatEx, release folder "2022" (data of 2023-06), application_data only; the sample of experiment 0020
(100,000 nonprovisional utility applications per filing year 2001-2019, 8 technology centres, same seed).
Cohort = filing year; group = technology centre; event = grant; competing exit = abandonment; censoring = data date.
Estimand: share of a filing year granted within 72 months. Snapshots: December 2007 to 2017. Scored at S: the filing
years S-4 .. S-1 within 2004-2016.
"Ours" and all baselines are the code of experiments/0012 (pipeline.py: FROZEN_SPEC_200k section 3, with the
censoring-correct cells and plug-in, which equal the as-run recipe when censoring is administrative only). Only the
data are new: covariates = log pending inventory of the centre and the Alice indicator (centres 36 and 21 from
2014-07; snapshots before 2014-12 see only zeros of it, so they use the inventory alone). Unconditional scenario:
inventory at its last-12-month mean per centre, Alice at its value at S.
Usage: PYTHONHASHSEED=0 uv run python backtest.py -> results.json, artifacts/*.csv
"""

from __future__ import annotations

import dataclasses
import json
import logging
import math
import os
import sys
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd
from scipy import stats

HERE: Final = Path(__file__).resolve().parent
sys.path[:0] = [
    str(HERE.parent / "0020-patex-feasibility"),
    str(HERE.parent / "0012-freddie-200k-harp-correction"),
]
import check  # the PatEx sample and the inventory covariate of the feasibility check
import pipeline as pl  # the frozen method

log = logging.getLogger(__name__)
H: Final = 72
SNAPSHOTS: Final = range(2007, 2018)
SCORED: Final = (2004, 2016)  # filing years that can be scored
ALPHA: Final = 0.10
MS: Final = (1.0, 1.5, 2.0)
METHODS: Final = [
    "naive",
    "drop_recent",
    "fixed_horizon",
    "chain",
    "ours_free",
    "cal_chain",
    "ours",
    "ours_uncond",
]
RAW: Final = ["log_inventory", "alice"]
DATA_END: Final = 2023.5  # data date 2023-06, in years


def build(sample: pd.DataFrame, inv: pd.DataFrame) -> pl.Data:
    """The frozen pipeline's data object. Month indices as in the Freddie code: `om` = filing month (1-based index),
    age a = months since filing, calendar month of a cell (0-based index) = om + a - 1."""
    loans = pd.DataFrame(
        {
            "orig_year": sample.year.to_numpy(),
            "om": sample.fm.to_numpy() + 1,
            "ev_mi": sample.end.to_numpy().astype(int) + 1,
            "event_age_clean": sample.dur.to_numpy(),
            "event_type_clean": sample.kind.to_numpy(),
        }
    )
    loans["year"] = loans.orig_year.astype(float)
    loans["quarter"] = ((loans.om - 1) // 3) / 4.0
    ages = np.arange(1, check.AMAX + 1)
    frames = []
    for (year, fm, tc), g in sample.groupby(["year", "fm", "tc"]):
        exposure, events, aj = check.ipcw.unit_arrays(
            g.dur.to_numpy(), g.kind.to_numpy(), check.CUTOFF - fm
        )
        frames.append(
            pd.DataFrame(
                {
                    "vintage": year,
                    "fp_mi": fm + 1,
                    "state": tc,
                    "age": ages,
                    "mi": fm + ages,
                    "exposure": exposure,
                    "events": events,
                    "aj": aj,
                }
            )
        )
    c = pd.concat(frames, ignore_index=True)
    c = c[c.exposure > 0].merge(
        inv.rename(columns={"tc": "state"}), on=["state", "mi"], how="inner"
    )
    c["alice"] = (c.state.isin(["36", "21"]) & (c.mi >= check.ALICE)).astype(float)
    for r in RAW:
        c[f"{r}_z"] = (c[r] - c[r].mean()) / c[r].std()
    c["unit"] = c.groupby(["vintage", "fp_mi", "state"]).ngroup()
    c["year"] = c.vintage.astype(float)
    c["quarter"] = ((c.fp_mi - 1) // 3) / 4.0
    return pl.Data(
        loans,
        c,
        {r: (float(c[r].mean()), float(c[r].std())) for r in RAW},
        list(RAW),
        [f"{r}_z" for r in RAW],
    )


def seen_by(d: pl.Data, year: int) -> pl.Data:
    """Before 2014-07 the Alice indicator is zero in everything a snapshot shows, so it cannot be a covariate yet:
    snapshots before 2014-12 use the inventory covariate only."""
    return d if year >= 2014 else dataclasses.replace(d, raw=RAW[:1], zcols=[f"{RAW[0]}_z"])


def persistence(
    data: pl.Data,
    z: np.ndarray,
    a_u: np.ndarray,
    sub: pd.DataFrame,
    s0: int,
    uid: np.ndarray,
    n_u: int,
) -> np.ndarray:
    """Unconditional path after each unit's age at S: log inventory at the centre's last-12-month mean; the Alice
    indicator at its value at S."""
    last = sub[(sub.mi > s0 - 12) & (sub.mi <= s0)]
    if last.empty:
        last = sub[sub.mi <= s0] if (sub.mi <= s0).any() else sub
    by_centre = last.groupby("state").log_inventory.mean()
    centre = np.empty(n_u, dtype=object)
    centre[uid] = sub.state.to_numpy()
    level = np.array([by_centre.get(s, by_centre.mean()) for s in centre])
    alice = np.array([float(s in ("36", "21") and s0 >= check.ALICE) for s in centre])
    future = np.arange(z.shape[1])[None, :] > a_u[:, None]
    out = z.copy()
    out[:, :, 0] = np.where(future, level[:, None], z[:, :, 0])
    if len(data.raw) > 1:
        out[:, :, 1] = np.where(future, alice[:, None], z[:, :, 1])
    return out


def evaluate(
    d: pl.Data, years: range, horizon: int, last_target: int, first_target: int = SCORED[0]
) -> tuple[pd.DataFrame, dict[str, dict[str, object]]]:
    truth = pl.truth(d, "year", horizon)
    rows, metas = [], {}
    for year in years:
        t, meta = pl.estimate(seen_by(d, year), year, horizon=horizon)
        t["truth"] = [truth[k][0] for k in t.cohort]
        mature = t[t.mature]
        t["ref"] = float(
            mature.truth.iloc[-1]
        )  # the latest cohort with `horizon` months at S (its rate is known)
        keep = t.cohort.between(max(year - 4, first_target), min(year - 1, last_target))
        rows.append(t[keep].assign(snapshot=year))
        metas[str(year)] = {k: v for k, v in meta.items() if k != "f0"}
        err = {
            m: round(float((np.abs(rows[-1][m] - rows[-1].truth) / rows[-1].truth).mean()), 3)
            for m in ("chain", "cal_chain", "ours")
        }
        log.info(
            "%d model=%s phi=%s calendar=%s err %s",
            year,
            meta["model"],
            meta["phi"],
            meta["calendar_on"],
            err,
        )
    return pd.concat(rows, ignore_index=True), metas


def snapshot_errors(v: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {m: (np.abs(v[m] - v.truth) / v.truth).groupby(v.snapshot).mean() for m in METHODS}
    )


def sign_test(err: pd.DataFrame, a: str, b: str) -> dict[str, float]:
    diff = (err[a] - err[b]).to_numpy()
    wins, n = int((diff < 0).sum()), int((diff != 0).sum())
    return {
        "median_paired_difference": round(float(np.median(diff)), 4),
        "wins": wins,
        "n": n,
        "sign_test_p": round(float(stats.binomtest(wins, n, 0.5).pvalue), 4) if n else 1.0,
    }


def vintage_level(v: pd.DataFrame, a: str, b: str, n_boot: int = 2000) -> dict[str, object]:
    """Mean paired difference of log absolute relative error over the scored (cohort, snapshot) pairs; 95% CI and a
    two-sided p value from a cluster bootstrap over filing years."""
    eps = 1e-6
    d = (
        np.log(np.abs(v[a] - v.truth) / v.truth + eps)
        - np.log(np.abs(v[b] - v.truth) / v.truth + eps)
    ).to_numpy()
    cl = v.cohort.to_numpy()
    keys = np.unique(cl)
    idx = {k: np.where(cl == k)[0] for k in keys}
    rng = np.random.default_rng(0)
    boot = np.array(
        [
            d[
                np.concatenate([idx[k] for k in rng.choice(keys, size=len(keys), replace=True)])
            ].mean()
            for _ in range(n_boot)
        ]
    )
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return {
        "pairs": len(d),
        "filing_year_clusters": len(keys),
        "mean_paired_log_difference": round(float(d.mean()), 3),
        "ci95": [round(float(lo), 3), round(float(hi), 3)],
        "bootstrap_p": round(float(min(1.0, 2 * min((boot >= 0).mean(), (boot <= 0).mean()))), 4),
        "error_ratio": round(float(np.exp(d.mean())), 3),
    }


def holm(p: dict[str, float]) -> dict[str, float]:
    order = sorted(p, key=lambda k: p[k])
    out, running = {}, 0.0
    for i, k in enumerate(order):
        running = max(running, min(1.0, (len(order) - i) * p[k]))
        out[k] = round(running, 4)
    return out


def calibration_errors(d: pl.Data, years: range, horizon: int) -> pd.DataFrame:
    """Out-of-sample errors of quarterly cohorts at the December pseudo-snapshots (T2d section 6)."""
    truth = pl.truth(d, "quarter", horizon)
    rows = []
    for year in years:
        t, _ = pl.estimate(seen_by(d, year), year, cohort="quarter", horizon=horizon)
        y = t[t.young].copy()
        y["truth"] = [
            truth[k][0] if k + 0.25 + horizon / 12 <= DATA_END else np.nan for k in y.cohort
        ]
        y = y[y.truth.notna() & (y.truth > 0) & (y.A >= 3)]
        for m in ("ours", "chain"):
            y[f"r_{m}"] = np.log(np.maximum(y[m], 1e-9)) - np.log(y.truth)
        rows.append(y.assign(snapshot=year)[["snapshot", "cohort", "A", "r_ours", "r_chain"]])
    return pd.concat(rows, ignore_index=True)


def conformal(v: pd.DataFrame, cal: pd.DataFrame, method: str, horizon: int) -> pd.DataFrame:
    """Interval of each scored pair from calibration errors at the same age (within 6 months), from pseudo-snapshots
    before S and quarterly cohorts whose truth is known at S."""
    out = []
    for r in v.itertuples():
        c = cal[
            (cal.snapshot < r.snapshot)
            & ((cal.A - r.A).abs() <= 6)
            & (cal.A + 12 * (r.snapshot - cal.snapshot) >= horizon)
        ]
        res = np.sort(np.abs(c[f"r_{method}"].to_numpy()))
        j = len(res)
        row: dict[str, object] = {"snapshot": r.snapshot, "cohort": r.cohort, "A": r.A, "J": j}
        est, tru = float(getattr(r, method)), float(r.truth)
        if j == 0 or est <= 0:
            out.append(row)
            continue
        q = float(res[min(j, math.ceil((j + 1) * (1 - ALPHA))) - 1])
        for m in MS:
            lo, hi = max(float(r.g_a), est * math.exp(-m * q)), min(1.0, est * math.exp(m * q))
            row[f"covered_M{m}"] = lo <= tru <= hi
            row[f"width_M{m}"] = hi / lo if lo > 0 else np.inf
        out.append(row)
    return pd.DataFrame(out)


def wrong_direction(v: pd.DataFrame) -> dict[str, object]:
    """P5: among scored pairs whose true rate differs from the reference cohort's by at least 2 points, the share
    where a method puts the cohort on the wrong side of the reference rate."""
    big = v[(v.truth - v.ref).abs() >= 0.02]
    sign = np.sign(big.truth - big.ref)
    methods = [
        m for m in METHODS if m != "fixed_horizon"
    ]  # the fixed horizon predicts the reference rate itself
    return {
        "pairs": len(big),
        "wrong_share": {
            m: round(float((np.sign(big[m] - big.ref) != sign).mean()), 3) for m in methods
        },
    }


def run(
    d: pl.Data, years: range, horizon: int, last_target: int, name: str, *, intervals: bool = True
) -> dict[str, object]:
    out_dir = HERE / "artifacts"
    v, metas = evaluate(d, years, horizon, last_target)
    v.round(6).to_csv(out_dir / f"pairs_{name}.csv", index=False)
    err = snapshot_errors(v)
    primary = {
        "P1 ours vs chain": sign_test(err, "ours", "chain")
        | {"vintage_level": vintage_level(v, "ours", "chain")},
        "P2 ours vs cal_chain": sign_test(err, "ours", "cal_chain")
        | {"vintage_level": vintage_level(v, "ours", "cal_chain")},
    }
    out: dict[str, object] = {
        "snapshots": {
            str(y): {
                "n": int((v.snapshot == y).sum()),
                "error": err.loc[y].round(4).to_dict(),
                **metas[str(y)],
            }
            for y in years
        },
        "scored_pairs": len(v),
        "median_error": err.median().round(4).to_dict(),
        "primary": primary,
        "holm_sign_test_p": holm({k: p["sign_test_p"] for k, p in primary.items()}),
        "holm_bootstrap_p": holm(
            {k: p["vintage_level"]["bootstrap_p"] for k, p in primary.items()}
        ),  # type: ignore[index]
        "P3": {
            f"ours vs {b}": sign_test(err, "ours", b)
            for b in ("naive", "drop_recent", "fixed_horizon")
        },
        "secondary": {
            f"{a} vs {b}": sign_test(err, a, b) | {"vintage_level": vintage_level(v, a, b)}
            for a, b in (
                ("cal_chain", "chain"),
                ("ours_free", "chain"),
                ("ours_uncond", "chain"),
                ("ours_uncond", "cal_chain"),
            )
        },
        "P5_wrong_direction": wrong_direction(v),
    }
    if intervals:
        cal = calibration_errors(d, range(years.start, years.stop - 1), horizon)
        cal.round(5).to_csv(out_dir / f"calibration_{name}.csv", index=False)
        conf = {m: conformal(v, cal, m, horizon) for m in ("ours", "chain")}
        p4: dict[str, object] = {}
        for m, c in conf.items():
            ok, full = c[c.J >= 1], c[c.J >= 9]
            p4[m] = {
                "pairs": len(c),
                "with_J_ge_1": len(ok),
                "with_J_ge_9": len(full),
                **{
                    f"coverage_M{k}_J_ge_1": round(float(ok[f"covered_M{k}"].mean()), 3)
                    if len(ok)
                    else None
                    for k in MS
                },
                **{
                    f"coverage_M{k}_J_ge_9": round(float(full[f"covered_M{k}"].mean()), 3)
                    if len(full)
                    else None
                    for k in MS
                },
                "median_width_M1": round(float(ok["width_M1.0"].median()), 3) if len(ok) else None,
            }
        w = conf["ours"].merge(conf["chain"], on=["snapshot", "cohort"], suffixes=("", "_chain"))
        w = w[w.J >= 1]
        p4["width_ratio_ours_over_chain_J_ge_1"] = (
            round(float((w["width_M1.0"] / w["width_M1.0_chain"]).median()), 3) if len(w) else None
        )
        out["P4_conformal"] = p4
    log.info(
        "%s: P1 %s | P2 %s", name, primary["P1 ours vs chain"], primary["P2 ours vs cal_chain"]
    )
    return out


def main() -> None:
    (HERE / "artifacts").mkdir(exist_ok=True)
    pl._persistence = persistence  # type: ignore[assignment]  # the patent covariates need their own persistence path
    applications = check.load()
    inv = check.inventory(applications)
    pool = applications[applications.year.isin(check.YEARS)]
    sample = pool.groupby("year").sample(
        n=check.N_PER_YEAR, random_state=check.SEED
    )  # the sample of 0020
    d = build(sample, inv)
    out = {"main": run(d, SNAPSHOTS, H, SCORED[1], "main")}
    published = (
        pool[pool.published].groupby("year").sample(n=check.N_PER_YEAR, random_state=check.SEED)
    )
    out["sensitivity_published_only"] = run(
        build(published, inv), SNAPSHOTS, H, SCORED[1], "published_only", intervals=False
    )
    late = build(sample[sample.year >= 2004], inv)
    out["sensitivity_without_2001_2003"] = run(
        late, range(2010, 2018), H, SCORED[1], "without_2001_2003", intervals=False
    )
    out["sensitivity_H96"] = run(d, range(2009, 2018), 96, 2014, "H96", intervals=False)
    (HERE / "results.json").write_text(json.dumps(out, indent=1, default=str))


if __name__ == "__main__":
    if os.environ.get("PYTHONHASHSEED") != "0":
        raise SystemExit(
            "run with PYTHONHASHSEED=0 (the fixed-effects fit depends on set order, see 0011)"
        )
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    main()
