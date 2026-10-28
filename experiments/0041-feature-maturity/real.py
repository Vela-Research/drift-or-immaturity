"""The pre-specified test of FEATURE_MATURITY_SPEC v1 on the Freddie Mac data (commit c5482de of the spec).

DATASET: Freddie Mac 200k, corrected weights; the loan sample, folds, snapshot views, classifiers and PU algorithm
of experiments/0033 (imported, not copied): 13 December snapshots, 50,000 training loans per vintage as S shows
them, 50,000 other loans of each of the 6 youngest vintages as the test, truth = default within 84 months.
Strategies trained here (logistic = primary, boosting = secondary; EM of 0033 with a propensity e per loan):
  pu_cl        e = chain-ladder timing curve at the loan's months of follow-up (0033; trained again as the reference)
  pu_feat      e = F(follow-up | x) / F(H | x) from the timing model of timing.py (generalized gamma), fitted on the
               loans of the mature vintages that S shows with a default within H; x = FICO band, LTV band, DTI, rate
               spread, purpose, occupancy
  pu_feat_weibull   the same with the Weibull version (robustness)
  pu_strat     e = empirical distribution of the within-H default ages of the mature vintages in the loan's stratum
               (FICO band x LTV band; a stratum with fewer than 200 defaults is merged with its FICO neighbour)
  pu_feat_cal  e of pu_feat x (calendar-adjusted completeness / chain-ladder completeness of the origination month)
`naive` (events so far) and `final` (today's labels, the oracle) are trained again as in 0033, to have them per
vintage; the other baselines are in the result files of 0033.
Usage: PYTHONHASHSEED=0 uv run python real.py [workers] -> artifacts/real/<year>.parquet (pairs), <year>.json
"""

from __future__ import annotations

import json
import logging
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd

HERE: Final = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent / "0033-downstream-classifier"), str(HERE)]
import run as base  # experiment 0033
import timing

log = logging.getLogger(__name__)
H: Final = base.H
OUT: Final = HERE / "artifacts" / "real"
LTV_EDGES: Final = (0.0, 80.0, 90.0, 1e9)  # <= 80, 80 to 90, > 90
MIN_STRATUM: Final = 200
STRATEGIES: Final = ("pu_cl", "pu_feat", "pu_feat_weibull", "pu_strat", "pu_feat_cal")
SPLITS: Final = {
    "fico": ("fico_720_up", "fico_below_660"),  # slow over fast, as in 0033
    "ltv": ("ltv_le_80", "ltv_gt_90"),
    "purpose": ("purchase", "refinance"),
}


def bands(x: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """FICO band 0..2 (-1 = no score) and LTV band 0..2 (a loan without LTV goes to the first band)."""
    fico = np.full(len(x), -1)
    for i, (low, high) in enumerate(base.FICO_BANDS.values()):
        fico[((x.fico >= low) & (x.fico < high)).to_numpy()] = i
    ltv = np.searchsorted(LTV_EDGES[1:], x.ltv.fillna(0.0).to_numpy(), side="left")
    return fico, np.minimum(ltv, 2)


def timing_features(
    train: pd.DataFrame, other: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Origination features of the timing model, coded as for the classifier, without state and vintage."""
    median, spread = train.dti.median(), train.rate_spread
    scale = {"dti": train.dti.fillna(median).std(), "rate_spread": spread.std()}

    def build(x: pd.DataFrame) -> np.ndarray:
        fico, ltv = bands(x)
        columns = [
            fico == 1,
            fico == 2,
            fico == -1,
            ltv == 1,
            ltv == 2,
            (x.dti.fillna(median) - median) / scale["dti"],
            x.dti.isna(),
            (x.rate_spread - spread.mean()) / scale["rate_spread"],
            x.purpose == "N",
            x.purpose == "C",
            x.occupancy == "I",
            x.occupancy == "S",
        ]
        return np.column_stack([np.asarray(c, dtype=float) for c in columns])

    names = [
        "fico_660_719",
        "fico_720_up",
        "fico_missing",
        "ltv_80_90",
        "ltv_gt_90",
        "dti_sd",
        "dti_missing",
        "rate_spread_sd",
        "refinance_no_cash_out",
        "refinance_cash_out",
        "investor",
        "second_home",
    ]
    return build(train), build(other), names


def strata_completeness(train: pd.DataFrame, done: np.ndarray, pooled: np.ndarray) -> np.ndarray:
    """pu_strat: per FICO x LTV stratum, the empirical distribution of the within-H default ages of the mature
    vintages, at the loan's follow-up. A stratum with fewer than MIN_STRATUM defaults joins its FICO neighbour in
    the same LTV band (the middle band joins the neighbour with more defaults); a loan without a score gets the
    pooled value."""
    fico, ltv = bands(train)
    followed = np.minimum(train.followed.to_numpy(), H)
    age = train.dur.to_numpy()
    out = pooled.copy()
    for band in range(3):
        groups = [[0], [1], [2]]
        defaults = np.array([int((done & (ltv == band) & (fico == k)).sum()) for k in range(3)])
        while len(groups) > 1 and min(defaults[g].sum() for g in groups) < MIN_STRATUM:
            i = min(range(len(groups)), key=lambda j: defaults[groups[j]].sum())
            near = [j for j in (i - 1, i + 1) if 0 <= j < len(groups)]
            j = max(near, key=lambda n: defaults[groups[n]].sum())
            groups[min(i, j)] = groups[min(i, j)] + groups[max(i, j)]
            del groups[max(i, j)]
        for g in groups:
            member = (ltv == band) & np.isin(fico, g)
            ages = np.sort(age[done & member])
            if len(ages) >= MIN_STRATUM:
                out[member] = np.searchsorted(ages, followed[member], side="right") / len(ages)
    return np.where(followed >= H, 1.0, np.clip(out, 0.0, 1.0))


def propensities(train: pd.DataFrame, year: int) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    e = base.completeness(train, "corrected", year)
    median_age = train.groupby("orig_year").age.median()
    mature = train.orig_year.isin(median_age[median_age >= H].index).to_numpy()
    done = mature & train.positive.to_numpy()  # default within H, as S shows it
    x, _, names = timing_features(train, train.iloc[:1])
    age = np.maximum(train.dur.to_numpy()[done] - 0.5, 0.5)
    followed = train.followed.to_numpy().astype(float)
    out = {"pu_cl": e["cl"]}
    info: dict[str, object] = {
        "mature_vintages": sorted(int(v) for v in median_age[median_age >= H].index),
        "defaults_for_the_timing_model": int(done.sum()),
    }
    for name, weibull in (("pu_feat", False), ("pu_feat_weibull", True)):
        model = timing.fit(age, x[done], float(H), weibull=weibull)
        m = model.maturity(followed, x, float(H))
        out[name] = np.where(followed >= H, 1.0, m)
        info[f"xi_{name}"] = dict(zip(names, np.round(model.xi, 4).tolist(), strict=True))
        info[f"shape_{name}"] = {
            "a": round(model.a, 3),
            "c": round(model.c, 3),
            "mu": round(model.mu, 3),
        }
        reference = np.zeros((3, x.shape[1]))
        reference[1, 0] = reference[2, 1] = (
            1.0  # FICO bands; other features at their reference values
        )
        info[f"maturity_by_fico_{name}"] = {
            str(a): np.round(model.maturity(np.full(3, float(a)), reference, float(H)), 4).tolist()
            for a in (12, 24, 48)
        }
        if not weibull:
            vintages = np.array(info["mature_vintages"])
            if len(vintages) >= 2:  # stability: early against late mature vintages
                year_of = train.orig_year.to_numpy()[done]
                early = year_of <= vintages[(len(vintages) - 1) // 2]
                info["xi_fico_720_up_early_late"] = [
                    round(float(timing.fit(age[k], x[done][k], float(H)).xi[1]), 4)
                    for k in (early, ~early)
                ]
    out["pu_strat"] = strata_completeness(train, done, e["cl"])
    with np.errstate(divide="ignore", invalid="ignore"):
        factor = np.where(e["cl"] > 0, e["cal"] / e["cl"], 1.0)
    out["pu_feat_cal"] = np.where(followed >= H, 1.0, np.clip(out["pu_feat"] * factor, 0.0, 1.0))
    young = train.age.to_numpy() < H
    info["mean_completeness_of_young_loans"] = {
        k: round(float(v[young].mean()), 4) for k, v in out.items()
    }
    return out, info


def predictions(
    train: pd.DataFrame, test: pd.DataFrame, e: dict[str, np.ndarray]
) -> dict[str, np.ndarray]:
    """The PU algorithm of 0033 `predictions` (start from the naive model, EM_ITERATIONS steps), per propensity."""
    design, design_test = base.designs(train, test)
    vintage, vintage_test = train.orig_year.to_numpy(), test.orig_year.to_numpy()
    positive, ones = train.positive.to_numpy(dtype=float), np.ones(len(train))
    code, code_test = base.vintage_codes("all", ones > 0, vintage, vintage_test)
    levels = np.unique(code)
    x, x_test = (
        base.with_vintage(design, code, levels),
        base.with_vintage(design_test, code_test, levels),
    )
    out = {}
    final = ((train.event_type_clean == "default") & (train.event_age_clean <= H)).to_numpy(float)
    today = base.ipcw(
        vintage,
        train.event_age_clean.to_numpy().astype(int),
        train.event_type_clean.to_numpy().astype(str),
    )
    for head in base.HEADS:
        out[f"final_{head}"] = base.fit_predict(head, x, final, today, x_test)[
            0
        ]  # oracle, as in 0033
        out[f"naive_{head}"], naive = base.fit_predict(head, x, positive, ones, x_test)
        for name, propensity in e.items():
            g = pd.DataFrame({"v": vintage, "s": positive, "e": propensity, "p": naive}).groupby(
                "v"
            )
            level = g.s.sum() / np.maximum(g.e.sum(), 1e-9) / np.maximum(g.p.mean(), 1e-12)
            p = np.minimum(level.reindex(vintage).to_numpy() * naive, 0.9)
            for _ in range(base.EM_ITERATIONS):
                w = p * (1 - propensity) / np.maximum(1 - p * propensity, 1e-9)
                out[f"{name}_{head}"], p = base.fit_predict(
                    head, x, np.where(positive > 0, 1.0, w), ones, x_test
                )
    return out


def groups(test: pd.DataFrame) -> dict[str, np.ndarray]:
    fico, ltv = bands(test)
    return {
        "fico_720_up": fico == 2,
        "fico_below_660": fico == 0,
        "ltv_le_80": ltv == 0,
        "ltv_gt_90": ltv == 2,
        "purchase": (test.purpose == "P").to_numpy(),
        "refinance": test.purpose.isin(["N", "C"]).to_numpy(),
    }


def score(p: np.ndarray, test: pd.DataFrame, y: np.ndarray, w: np.ndarray) -> pd.DataFrame:
    """One row per test vintage: level ratio, level ratios of the groups, ECE over deciles, AUC, all inside the
    vintage and with the inverse-probability-of-censoring weights of 0033."""
    member = groups(test)
    rows = []
    for vintage, idx in test.groupby("orig_year").indices.items():
        ok = idx[w[idx] > 0]
        pv, yv, wv = np.clip(p[ok], 1e-9, 1 - 1e-9), y[ok], w[ok]
        row: dict[str, float] = {
            "cohort": int(vintage),
            "realised": float((wv * yv).sum() / wv.sum()),
        }
        row["level_ratio"] = float((wv * pv).sum() / max((wv * yv).sum(), 1e-12))
        for name, mask in member.items():
            m = mask[ok]
            events = float((wv[m] * yv[m]).sum())
            row[f"events_{name}"] = float(yv[m].sum())
            row[f"level_ratio_{name}"] = (
                float((wv[m] * pv[m]).sum() / events) if events > 0 else np.nan
            )
        order = np.argsort(pv, kind="stable")
        decile = np.minimum((np.cumsum(wv[order]) / wv.sum() * 10).astype(int), 9)
        pred, real = (
            np.bincount(decile, weights=v[order], minlength=10) for v in (wv * pv, wv * yv)
        )
        row["ece"] = float(np.abs(pred - real).sum() / wv.sum())
        row["auc"] = (
            float(base.roc_auc_score(yv, pv, sample_weight=wv))
            if 0 < yv.sum() < len(yv)
            else np.nan
        )
        rows.append(row)
    return pd.DataFrame(rows)


def snapshot(year: int) -> int:
    if (OUT / f"{year}.json").exists():
        return year
    loans = pd.read_parquet(base.ART / "loans_corrected.parquet")
    x = base.view(loans, year)
    train = x[x.fold < base.N_TRAIN].reset_index(drop=True)
    test = x[(x.fold >= base.N_TRAIN) & (x.orig_year > year - base.N_SCORED)].reset_index(drop=True)
    e, info = propensities(train, year)
    preds = predictions(train, test, e)
    y = ((test.event_type_clean == "default") & (test.event_age_clean <= H)).to_numpy(float)
    w = base.ipcw(
        test.orig_year.to_numpy(),
        test.event_age_clean.to_numpy().astype(int),
        test.event_type_clean.to_numpy().astype(str),
    )
    pairs = pd.concat(
        [score(p, test, y, w).assign(method=name, snapshot=year) for name, p in preds.items()],
        ignore_index=True,
    )
    pairs.to_parquet(OUT / f"{year}.parquet")
    (OUT / f"{year}.json").write_text(json.dumps(info, indent=1))
    return year


def main(workers: int) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[key] = "2"
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for year in pool.map(snapshot, base.SNAPSHOTS):
            log.info("snapshot %d done", year)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 3)
