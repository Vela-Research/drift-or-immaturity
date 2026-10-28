"""Discrete-time (loan-month) hazard classifier of 84-month default for Table 1 (POST HOC; the v10 review requests
item 2).

DATASET: the `corrected` loan file of experiments/0033 (200k loans per vintage, Freddie Mac sample of experiments/0010;
HARP parents censored). Same snapshots, training loans (50,000 per vintage, as the snapshot shows them), test loans
(50,000 other loans of each of the 6 youngest vintages), truth, weights and scoring as 0033 `run.py` (imported, not
changed).
Model: multinomial logistic regression on the loan-months that S shows, outcome per month = default / prepayment /
none, with monthly age dummies (age since first payment, 1..84), vintage effects and the logistic design of 0033
(origination features, missing flags, one-hot categories; C = 10 as the other logistic rows). A loan leaves the risk
set at its first event or at the end of its follow-up at S (a HARP parent at its censoring age). Every event month is
kept; months with no event are sampled at rate r (about NONEVENT_TARGET of them) and weighted 1 / r.
Prediction for a test loan: P(default within 84 months) = sum over ages a of S(a - 1) h_default(a), with
S(a) = prod (1 - h_default - h_prepay): prepayment is a competing exit, as in the truth (a prepaid loan is a
non-default).
Usage: uv run python hazard_classifier.py [snapshot years; default all] -> artifacts/parts/<year>.json|.parquet|.npy
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.linear_model import LogisticRegression

HERE: Final = Path(__file__).resolve().parent
OLD: Final = HERE.parent / "0033-downstream-classifier"
sys.path.insert(0, str(OLD))
import run as clf  # the classifier experiment of 0033

log = logging.getLogger(__name__)
Array = NDArray[np.float64]
H: Final = clf.H
NONEVENT_TARGET: Final = 1_500_000
PARTS: Final = HERE / "artifacts" / "parts"


def loan_months(
    train: pd.DataFrame,
) -> tuple[NDArray[np.int64], NDArray[np.int64], NDArray[np.int64]]:
    """(loan row, age 1..H, outcome 0 none / 1 default / 2 prepayment) of every loan-month at risk that S shows."""
    dur = train.dur.to_numpy().astype(int)
    kind = train.kind.to_numpy().astype(str)
    last = np.clip(dur, 0, H)
    loan = np.repeat(np.arange(len(train)), last)
    start = np.repeat(np.cumsum(last) - last, last)
    age = np.arange(len(loan)) - start + 1
    outcome = np.zeros(len(loan), dtype=np.int64)
    end = np.cumsum(last)[last > 0] - 1
    ends = np.where(last > 0)[0]
    event = np.select([kind[ends] == "default", kind[ends] == "prepay"], [1, 2], 0)
    outcome[end] = np.where(dur[ends] <= H, event, 0)
    return loan, age, outcome


def cif(h_default: Array, h_prepay: Array) -> Array:
    """[loan, age 1..H] monthly hazards -> P(default by H) with prepayment as a competing exit."""
    survive = np.cumprod(1 - h_default - h_prepay, axis=1)
    before = np.concatenate([np.ones((len(survive), 1)), survive[:, :-1]], axis=1)
    return np.asarray((before * h_default).sum(axis=1))


def fit_predict(
    train: pd.DataFrame, test: pd.DataFrame, year: int
) -> tuple[Array, dict[str, float]]:
    base, base_test = clf.designs(train, test)
    vintage, vintage_test = train.orig_year.to_numpy(), test.orig_year.to_numpy()
    levels = np.unique(vintage)
    x, x_test = (
        clf.with_vintage(base, vintage, levels),
        clf.with_vintage(base_test, vintage_test, levels),
    )
    loan, age, outcome = loan_months(train)
    rng = np.random.default_rng(clf.SEED + year)
    none = outcome == 0
    rate = min(1.0, NONEVENT_TARGET / max(int(none.sum()), 1))
    keep = ~none | (rng.random(len(outcome)) < rate)
    weight = np.where(none, 1 / rate, 1.0)[keep]
    ages = np.eye(H)[age[keep] - 1][:, 1:]  # age 1 is the reference
    design = np.hstack([x[0][loan[keep]], ages])
    model = LogisticRegression(C=10.0, solver="newton-cholesky", max_iter=100)
    model.fit(design, outcome[keep], sample_weight=weight)
    assert list(model.classes_) == [0, 1, 2]
    k = x[0].shape[1]
    eta = (x_test[0] @ model.coef_[:, :k].T + model.intercept_)[:, None, :] + np.vstack(
        [np.zeros((1, 3)), model.coef_[:, k:].T]
    )[None, :, :]
    eta -= eta.max(axis=2, keepdims=True)
    prob = np.exp(eta) / np.exp(eta).sum(axis=2, keepdims=True)
    info = {
        "loan_months": len(outcome),
        "rows_fitted": int(keep.sum()),
        "nonevent_sampling_rate": round(rate, 5),
        "defaults": int((outcome == 1).sum()),
        "prepayments": int((outcome == 2).sum()),
        "iterations": int(model.n_iter_.max()),
    }
    return cif(prob[:, :, 1], prob[:, :, 2]), info


def snapshot(year: int, loans: pd.DataFrame) -> None:
    """The split, truth, weights and scores of 0033 `run.snapshot`, for one more method."""
    started = time.time()
    x = clf.view(loans, year)
    train = x[x.fold < clf.N_TRAIN].reset_index(drop=True)
    test = x[(x.fold >= clf.N_TRAIN) & (x.orig_year > year - clf.N_SCORED)].reset_index(drop=True)
    p, info = fit_predict(train, test, year)
    y = ((test.event_type_clean == "default") & (test.event_age_clean <= H)).to_numpy(float)
    w = clf.ipcw(
        test.orig_year.to_numpy(),
        test.event_age_clean.to_numpy().astype(int),
        test.event_type_clean.to_numpy().astype(str),
    )
    metrics, replicates = clf.score(p, test, y, w)
    pairs = (
        pd.DataFrame({"w": w, "wy": w * y, "hazard_logit": w * p})
        .groupby(test.orig_year.to_numpy())
        .sum()
    )
    pairs = pairs.div(pairs.w, axis=0).drop(columns="w").rename(columns={"wy": "realised"})
    pairs.rename_axis("cohort").reset_index().assign(snapshot=year).to_parquet(
        PARTS / f"{year}.parquet"
    )
    np.save(PARTS / f"{year}.npy", replicates)
    out = {**info, "seconds": round(time.time() - started, 1), "metrics": metrics}
    (PARTS / f"{year}.json").write_text(json.dumps(out, indent=1))
    log.info(
        "%d: %s, level ratio %.3f, AUC %.4f", year, info, metrics["level_ratio"], metrics["auc"]
    )


def demo() -> None:
    """cif() against a direct loop."""
    rng = np.random.default_rng(0)
    d, q = rng.uniform(0, 0.01, (3, H)), rng.uniform(0, 0.03, (3, H))
    for i in range(3):
        alive, total = 1.0, 0.0
        for a in range(H):
            total += alive * d[i, a]
            alive *= 1 - d[i, a] - q[i, a]
        assert abs(cif(d, q)[i] - total) < 1e-12


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    demo()
    PARTS.mkdir(parents=True, exist_ok=True)
    data = pd.read_parquet(OLD / "artifacts" / "loans_corrected.parquet")
    for snapshot_year in [int(y) for y in sys.argv[1:]] or list(clf.SNAPSHOTS):
        if not (PARTS / f"{snapshot_year}.json").exists():
            snapshot(snapshot_year, data)
