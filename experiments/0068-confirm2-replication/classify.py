"""CONFIRM2 §4 classifiers (H1) and §7.5 (classifier c) on A and B (PRE-REGISTERED).

DATASET: the units of data.py; the scored pairs, units and `share_cl` of backtest.py. Snapshots: A 2000-2013
(decision A1: the 72-month truth that the classifiers predict), B 1994-2016.
At each snapshot S:
  1. the units of the scored cohorts are split 50/50 at random (seed 20261025): one random permutation per entry
     year, drawn once; the first half is the test part, at every snapshot;
  2. training set: the units that entered in the 10 years up to S (entry years S-9 .. S), minus the test part;
     features of §2/§3 and an entry-year term (0033 design `vintage`);
  3. logistic regression (L2, C = 10): 0033 `fit_predict`, head `logit`;
  4. labels (0033 `view` and `predictions`): naive = an event within H months seen at S; pu_cl = positive-unlabelled
     EM (Bekker-Davis) with e = the chain ladder's timing curve at the unit's months of follow-up; pu_full = the
     same with e = the full method's completeness of the unit's entry month (forecast path; 0033 `completeness`);
     start from the naive model, EM_ITERATIONS = 2 iterations;
  5. per scored cohort: the mean predicted probability on the test part and the truth = the Aalen-Johansen
     incidence of the test units at H.
Design (0033 `designs`, with the features of this data set): numerics standardised, a missing value gets the
training median and a missing flag; categories one-hot with "missing" as a level, first level left out.
Usage: uv run python classify.py <sba|civil> -> artifacts/<name>/classifier.parquet (checkpoints in parts/)
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd

import data

HERE: Final = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "0033-downstream-classifier"))
import run as clf  # view, with_vintage, fit_predict, EM_ITERATIONS of the frozen classifiers

log = logging.getLogger(__name__)
SEED: Final = 20261025
YEARS: Final = {"sba": range(2000, 2014), "civil": range(1994, 2017)}
WINDOW: Final = 10  # entry years in the training set
NUMERIC: Final = {
    "sba": [
        "log_gross",
        "guaranteed_share",
        "term_months",
        "log1p_jobs",
        "interest_rate",
        "franchise",
    ],
    "civil": ["log1p_demanded", "classact"],
}
CATEGORICAL: Final = {
    "sba": ["naics2", "business_type", "business_age", "revolver", "subprogram", "state"],
    "civil": ["nos", "juris", "origin", "jury", "prose", "district"],
}
Design = tuple[np.ndarray, np.ndarray]


def split(u: pd.DataFrame) -> np.ndarray:
    """True for the test part: half of each entry year, by one random permutation per year (fixed seed)."""
    rng = np.random.default_rng(SEED)
    test = np.zeros(len(u), dtype=bool)
    for _, idx in sorted(u.groupby("orig_year").indices.items()):
        test[idx[rng.permutation(len(idx)) < len(idx) // 2]] = True
    return test


def designs(train: pd.DataFrame, test: pd.DataFrame, name: str) -> tuple[Design, Design]:
    numeric, categorical = NUMERIC[name], CATEGORICAL[name]
    median = (
        train[numeric].median().fillna(0.0)
    )  # a feature missing for every training unit: its flag only
    flagged = [c for c in numeric if train[c].isna().any()]
    filled = train[numeric].fillna(median)
    mean, sd = filled.mean(), filled.std().replace(0.0, 1.0)

    def labels(x: pd.DataFrame, c: str) -> pd.Series:
        return x[c].astype(object).where(x[c].notna(), "missing").astype(str)

    levels = {c: sorted(labels(train, c).unique()) for c in categorical}

    def build(x: pd.DataFrame) -> Design:
        codes = [pd.Categorical(labels(x, c), categories=levels[c]).codes for c in categorical]
        hot = [
            np.eye(len(levels[c]) + 1)[k][:, 1:-1] for c, k in zip(categorical, codes, strict=True)
        ]
        z = ((x[numeric].fillna(median) - mean) / sd).to_numpy()
        flags = x[flagged].isna().to_numpy(dtype=float)
        matrix = np.hstack([z, flags, *hot])
        return matrix, matrix[:, :0]

    return build(train), build(test)


def completeness(
    x: pd.DataFrame, share: np.ndarray, months: pd.Series, h: int
) -> dict[str, np.ndarray]:
    """0033 `completeness` for `cl` and `full`: e at the unit's months of follow-up; `full` = the value of the
    unit's entry month (outside the young cohorts: the chain-ladder value)."""
    f, a = (np.minimum(x[c].to_numpy(), h).astype(int) for c in ("followed", "age"))
    out = {"cl": share[f] / share[h]}
    left = np.where(f < a, share[f] / np.maximum(share[a], 1e-12), 1.0)
    e = months.reindex(x.om.to_numpy()).to_numpy() * left
    out["full"] = np.where(np.isnan(e), out["cl"], e)
    return {k: np.where(f >= h, 1.0, np.clip(v, 0.0, 1.0)) for k, v in out.items()}


def month_completeness(units: pd.DataFrame) -> pd.Series:
    """0033 `month_completeness`: per entry month, incidence seen at S over the full method's estimate."""
    seen = (units.n * units.aj).groupby(units.fp_mi).sum()
    final = (units.n * (1 - (1 - units.aj) * np.exp(-units.future_full))).groupby(units.fp_mi).sum()
    return seen / final.where(final > 0)


def pu_em(
    x: Design,
    x_test: Design,
    positive: np.ndarray,
    e: np.ndarray,
    naive: np.ndarray,
    vintage: np.ndarray,
) -> tuple[np.ndarray, float]:
    """The EM loop of 0033 `predictions` (head logit). Returns the test predictions and the last mean change."""
    g = pd.DataFrame({"v": vintage, "s": positive, "e": e, "p": naive}).groupby("v")
    level = g.s.sum() / np.maximum(g.e.sum(), 1e-9) / np.maximum(g.p.mean(), 1e-12)
    p = np.minimum(level.reindex(vintage).to_numpy() * naive, 0.9)
    ones = np.ones(len(positive))
    for _ in range(clf.EM_ITERATIONS):
        w = p * (1 - e) / np.maximum(1 - p * e, 1e-9)
        last = p
        pred, p = clf.fit_predict("logit", x, np.where(positive > 0, 1.0, w), ones, x_test)
    return pred, float(np.abs(p - last).mean())


def snapshot(
    u: pd.DataFrame,
    is_test: np.ndarray,
    name: str,
    year: int,
    cohorts: list[float],
    share: np.ndarray,
    units: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, object]]:
    h = data.H[name]
    clf.H = h
    x = clf.view(u.assign(test=is_test), year)
    x = x[x.orig_year > year - WINDOW]
    test = x[x.test & x.year.isin(cohorts)].reset_index(drop=True)
    train = x[~x.test].reset_index(drop=True)
    base, base_test = designs(train, test, name)
    vintage, vintage_test = train.orig_year.to_numpy(), test.orig_year.to_numpy()
    levels = np.unique(vintage)
    xd, xd_test = (
        clf.with_vintage(base, vintage, levels),
        clf.with_vintage(base_test, vintage_test, levels),
    )
    positive, ones = train.positive.to_numpy(dtype=float), np.ones(len(train))
    preds, info = (
        {},
        {"train_units": len(train), "test_units": len(test), "columns": int(xd[0].shape[1])},
    )
    preds["naive"], naive_fitted = clf.fit_predict("logit", xd, positive, ones, xd_test)
    e = completeness(train, share, month_completeness(units), h)
    for label, source in (("pu_cl", "cl"), ("pu_full", "full")):
        preds[label], info[f"em_last_change_{label}"] = pu_em(
            xd, xd_test, positive, e[source], naive_fitted, vintage
        )
        info[f"mean_completeness_{source}"] = round(
            float(e[source][train.age.to_numpy() < h].mean()), 4
        )
    rows = []
    for cohort, idx in test.groupby("year").indices.items():
        g = test.iloc[idx]
        curve = data.pl.aj_default_cif(g.event_age_clean.to_numpy(), g.event_type_clean.to_numpy())
        rows.append(
            {
                "snapshot": year,
                "cohort": float(cohort),
                "test_units": len(idx),
                "truth": float(curve[h]),
                **{k: float(v[idx].mean()) for k, v in preds.items()},
            }
        )
    return pd.DataFrame(rows), info


def main(name: str) -> None:
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ.setdefault(key, "8")
    out = data.ART / name
    parts = out / "parts"
    u = data.units(name)
    is_test = split(u)
    pairs = pd.read_parquet(out / "pairs.parquet")
    units = pd.read_parquet(out / "units.parquet")
    meta = json.loads((out / "meta.json").read_text())
    for year in YEARS[name]:
        if (parts / f"classifier_{year}.parquet").exists():
            continue
        cohorts = sorted(pairs[pairs.snapshot == year].cohort.tolist())
        share = np.asarray(meta[str(year)]["share_cl"], dtype=float)
        rows, info = snapshot(u, is_test, name, year, cohorts, share, units[units.snapshot == year])
        rows.to_parquet(parts / f"classifier_{year}.parquet")
        (parts / f"classifier_{year}.json").write_text(json.dumps(info, indent=1))
        log.info("%s %d: %s", name, year, info)
    files = [parts / f"classifier_{y}.parquet" for y in YEARS[name]]
    pd.concat(pd.read_parquet(f) for f in files).to_parquet(out / "classifier.parquet")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    main(sys.argv[1])
