"""Stage 2 of 0033: classifiers of 84-month default under each label strategy (POST HOC; round 3, item P1.9).

DATASET: 200k loans per vintage, simple random sample of the full Freddie Mac Standard files (experiments/0010);
version `corrected` (HARP parents censored) or `linked` (borrower-followed, 0018). Input: the files of prepare.py.

At each December snapshot S (2006 to 2018) every model is trained on what S shows for 50,000 loans per vintage and
scored on 50,000 OTHER loans of each of the 6 youngest vintages (the vintages the backtest scores at S) against the
default within 84 months in today's records (prepayment = no default).
A loan is `positive` when a default within 84 months of age is seen at S. Label strategies:
  final        default within 84 months from today's records, all loans (NOT available at S; the reference)
  naive        events so far: label = positive, all loans
  mature       only loans with 84 months of age at S, final label; the vintage term of a younger loan is the one
               of the last trained vintage
  drop_recent  the repo's rule for loans: only the 36 most recent origination months with 84 months of age (the 3
               most recent mature cohorts), final label, no vintage term
  dfm          Chapelle (2014): logistic rate (with the vintage term) x exponential delay with rate exp(b'x)
               (no vintage term: the delay distribution is stationary); logistic only
  esdfm        importance weights in the spirit of ES-DFM (Yang et al. 2021): a loan that is not positive and has
               less than 84 months of follow-up is a positive with weight q and a negative with weight 1 - q;
               q = P(default later, within 84 months | none at this age, x) from a gradient-boosting model
               trained on the loans with 84 months of age, replayed at the ages of the younger loans
  pu_cl        positive-unlabelled correction (SAR, Bekker et al. 2019), EM form: a loan that is not positive is a
               positive with weight w = p (1 - e) / (1 - p e) and a negative with weight 1 - w; e = the chain
               ladder's timing curve at the loan's age. Start: level per vintage = positives / sum of e, differences
               between loans from the naive model; then EM_ITERATIONS iterations
  pu_cal       the same with e = (incidence seen at S) / (estimate of the calendar-adjusted chain ladder, forecast
               path) of the loan's origination month
  pu_full      the same with the estimate of the full method (calendar step + selected speed model, forecast path)
Classifiers: logistic regression and gradient boosting (scikit-learn), sample weights. Features at origination:
FICO, LTV, DTI, note rate, rate spread, units, purpose, occupancy, first-time buyer, state; and a vintage term
(design `vintage`; design `features` has none).
Sub-distribution view, as in the paper: a prepaid loan stays in the data to S as "no default". A loan censored
before S (a HARP parent in `corrected`) is followed to its censoring age; where a final label is needed (mature,
drop_recent, final, scoring) loans with an unknown label get weight 0 and the others the inverse probability of
not being censored (by vintage).
Usage: PYTHONHASHSEED=0 uv run python run.py <corrected|linked> <vintage|features> [snapshot years; default all]
  -> results_<version>_<design>.json, artifacts/pairs_<version>_<design>.parquet
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy import optimize
from scipy.special import expit
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

log = logging.getLogger(__name__)
HERE: Final = Path(__file__).resolve().parent
ART: Final = HERE / "artifacts"
Array = NDArray[np.float64]
Mask = NDArray[np.bool_]
H: Final = 84
SNAPSHOTS: Final = range(2006, 2019)
N_TRAIN: Final = 50_000
N_SCORED: Final = 6
EM_ITERATIONS: Final = 2
WORKERS: Final = 4
N_BOOT: Final = 300
AUC_BINS: Final = 500
SEED: Final = 20261006
NUMERIC: Final = ["fico", "ltv", "dti", "rate", "rate_spread", "units"]
CATEGORICAL: Final = ["purpose", "occupancy", "first_time_buyer", "state"]
FICO_BANDS: Final = {
    "fico_below_660": (0, 660),
    "fico_660_719": (660, 720),
    "fico_720_up": (720, 1000),
}
HEADS: Final = ("logit", "gbm")
PU: Final = {"pu_cl": "cl", "pu_cal": "cal", "pu_full": "full"}
Design = tuple[Array, Array]  # (logistic matrix, tree matrix)


def view(loans: pd.DataFrame, year: int) -> pd.DataFrame:
    """Loans observable at December `year`, as the snapshot shows them (the rule of pipeline.observed)."""
    s = year * 12 + 12
    x = loans[(loans.orig_year <= year) & (loans.om <= s)].copy()
    seen = (x.ev_mi <= s).to_numpy()
    x["age"] = s - x.om
    x["dur"] = np.where(seen, x.event_age_clean, x.age).astype(int)
    x["kind"] = np.where(seen, x.event_type_clean, "censored")
    x["followed"] = np.where(x.kind == "censored", x.dur, x.age)
    x["positive"] = (x.kind == "default") & (x.dur <= H)
    return x


def ipcw(vintage: NDArray[np.int64], dur: NDArray[np.int64], kind: NDArray[np.str_]) -> Array:
    """0 if the status at H months is unknown (censored before H); else 1 / P(not censored before the status is
    known), with the Kaplan-Meier estimate of the censoring ages of the vintage (as cells.py of 0012)."""
    out = np.zeros(len(dur))
    censored = kind == "censored"
    for v in np.unique(vintage):
        i = np.where(vintage == v)[0]
        d = np.clip(dur[i], 0, H + 1)
        active = len(i) - np.concatenate([[0], np.cumsum(np.bincount(d, minlength=H + 2))[:-1]])
        g = np.cumprod(1 - np.bincount(d[censored[i]], minlength=H + 2) / np.maximum(active, 1))
        known = ~censored[i] | (dur[i] >= H)
        out[i] = known / np.maximum(g[np.maximum(np.minimum(d, H) - 1, 0)], 1e-6)
    return out


def designs(train: pd.DataFrame, test: pd.DataFrame) -> tuple[Design, Design]:
    """Encoders are fitted on the training loans. Logistic: standardised numerics (median for a missing value, with
    missing flags for FICO and DTI) and one-hot categories. Trees: raw numerics (missing stays missing) and codes."""
    median = train[NUMERIC].median()
    filled = train[NUMERIC].fillna(median)
    mean, sd = filled.mean(), filled.std()
    levels = {c: sorted(train[c].dropna().unique()) for c in CATEGORICAL}

    def build(x: pd.DataFrame) -> Design:
        codes = [pd.Categorical(x[c], categories=levels[c]).codes for c in CATEGORICAL]
        hot = [
            np.eye(len(levels[c]) + 1)[k][:, 1:-1] for c, k in zip(CATEGORICAL, codes, strict=True)
        ]
        z = ((x[NUMERIC].fillna(median) - mean) / sd).to_numpy()
        flags = x[["fico", "dti"]].isna().to_numpy(dtype=float)
        trees = [np.where(k < 0, np.nan, k) for k in codes]
        return np.hstack([z, flags, *hot]), np.column_stack([x[NUMERIC].to_numpy(float), *trees])

    return build(train), build(test)


def with_vintage(x: Design, code: NDArray[np.int64], levels: NDArray[np.int64]) -> Design:
    hot = (code[:, None] == levels[None, 1:]).astype(float)
    return np.hstack([x[0], hot]), np.column_stack([x[1], code])


def vintage_codes(
    mode: str, used: Mask, year: NDArray[np.int64], year_test: NDArray[np.int64]
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    """all = the loan's vintage; carry = a vintage after the last trained one gets the last trained one;
    pooled = no vintage term."""
    if mode == "pooled":
        return np.zeros_like(year), np.zeros_like(year_test)
    top = int(year[used].max())
    return np.minimum(year, top), np.minimum(year_test, top)


def boosting() -> HistGradientBoostingClassifier:
    first = len(NUMERIC)
    return HistGradientBoostingClassifier(
        max_iter=60,
        learning_rate=0.15,
        max_leaf_nodes=31,
        min_samples_leaf=200,
        l2_regularization=1.0,
        categorical_features=list(range(first, first + len(CATEGORICAL))),
        early_stopping=False,
        random_state=0,
    )


def fit_predict(
    head: str, x: Design, target: Array, weight: Array, x_test: Design
) -> tuple[Array, Array]:
    """A row with target t in (0, 1) enters as a positive with weight t and as a negative with weight 1 - t.
    Returns the predictions for the test rows and for the training rows."""
    a, b = (x[0], x_test[0]) if head == "logit" else (x[1], x_test[1])
    pos, neg = np.where(weight * target > 0)[0], np.where(weight * (1 - target) > 0)[0]
    rows = np.concatenate([pos, neg])
    y = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
    w = np.concatenate([weight[pos] * target[pos], weight[neg] * (1 - target[neg])])
    model = (
        LogisticRegression(C=10.0, solver="newton-cholesky", max_iter=100)
        if head == "logit"
        else boosting()
    )
    model.fit(a[rows], y, sample_weight=w)
    return model.predict_proba(b)[:, 1], model.predict_proba(a)[:, 1]


def delayed_feedback(
    rate: Array, delay: Array, seen: Array, t: Array, rate_test: Array, delay_test: Array
) -> Array:
    """Chapelle (2014): P(default) = sigmoid(a'x), delay ~ Exponential(exp(b'x)); maximum likelihood (the objective
    of experiments/0019, with separate designs for the two parts). Prediction: P(default within H months)."""
    one, one_test = np.ones((len(seen), 1)), np.ones((len(rate_test), 1))
    xa, xb = np.hstack([one, rate]), np.hstack([one, delay])
    k, n = xa.shape[1], len(seen)

    def objective(par: Array) -> tuple[float, Array]:
        p, lam = expit(xa @ par[:k]), np.exp(np.minimum(xb @ par[k:], 20.0))
        decay = np.exp(-lam * t)
        rest = np.maximum(1 - p + p * decay, 1e-12)
        loglik = seen * (np.log(np.maximum(p, 1e-12)) + np.log(lam) - lam * t)
        loglik = loglik + (1 - seen) * np.log(rest)
        d_a = seen * (1 - p) - (1 - seen) * p * (1 - p) * (1 - decay) / rest
        d_b = seen * (1 - lam * t) - (1 - seen) * p * decay * lam * t / rest
        grad = np.concatenate([xa.T @ d_a, xb.T @ d_b]) - 1e-4 * n * par
        return -(float(loglik.sum()) - 0.5e-4 * n * float(par @ par)) / n, -grad / n

    start = np.zeros(k + xb.shape[1])
    start[0], start[k] = np.log(0.05 / 0.95), np.log(1 / 60)
    par = optimize.minimize(
        objective, start, jac=True, method="L-BFGS-B", options={"maxiter": 300}
    ).x
    p = expit(np.hstack([one_test, rate_test]) @ par[:k])
    return np.asarray(p * (1 - np.exp(-np.exp(np.hstack([one_test, delay_test]) @ par[k:]) * H)))


def later_default_probability(
    train: pd.DataFrame, trees: Array, known: Array, rng: np.random.Generator
) -> Array:
    """esdfm: q = P(default after this age, within H months | no default seen at this age, x) for the loans that
    are not positive and have less than H months of follow-up; 0 for the others. The model is trained on the loans
    with H months of age, each replayed at one age drawn from the follow-up of the younger loans."""
    old = known > 0
    young = ~train.positive.to_numpy() & (train.followed.to_numpy() < H)
    replay = rng.choice(train.followed.to_numpy()[young], size=int(old.sum()))
    default_age = np.where(train.kind == "default", train.dur, 10**6)[old]
    keep = default_age > replay
    model = boosting().fit(
        np.column_stack([trees[old][keep], replay[keep]]),
        default_age[keep] <= H,
        sample_weight=known[old][keep],
    )
    q = np.zeros(len(train))
    q[young] = model.predict_proba(
        np.column_stack([trees[young], train.followed.to_numpy()[young]])
    )[:, 1]
    return q


def month_completeness(version: str, year: int) -> tuple[Array, pd.DataFrame]:
    """What snapshot `year` knows (prepare.py): the chain ladder's timing curve at ages 0..H, and per origination
    month of the young cohorts the share of the method's 84-month estimate that S shows (units of the month summed;
    `cal` = calendar-adjusted chain ladder, `full` = full method, both on the forecast path)."""
    cohorts = pd.read_parquet(ART / f"cohorts_{version}.parquet")
    u = pd.read_parquet(ART / f"units_{version}.parquet")
    u = u[u.snapshot == year]
    seen = (u.n * u.aj).groupby(u.fp_mi).sum()
    final = {
        k: (u.n * (1 - (1 - u.aj) * np.exp(-u[f"future_{k}"]))).groupby(u.fp_mi).sum()
        for k in ("cal", "full")
    }
    share = np.asarray(cohorts[cohorts.snapshot == year].share_cl.iloc[0], dtype=float)
    return share, pd.DataFrame({k: seen / v.where(v > 0) for k, v in final.items()})


def completeness(x: pd.DataFrame, version: str, year: int) -> dict[str, Array]:
    """Labelling propensity of each loan: the share of its 84-month default incidence that is visible after its
    months of follow-up. cl = timing curve of the chain ladder; cal, full = the value of the loan's origination
    month (a month outside the young cohorts gets the chain-ladder value; a loan that left before S gets the
    month's value times the chain-ladder ratio between its age and the month's age)."""
    share, months = month_completeness(version, year)
    f, a = (np.minimum(x[c].to_numpy(), H).astype(int) for c in ("followed", "age"))
    out = {"cl": share[f] / share[H]}
    left = np.where(f < a, share[f] / np.maximum(share[a], 1e-12), 1.0)
    for name in ("cal", "full"):
        e = months[name].reindex(x.om.to_numpy()).to_numpy() * left
        out[name] = np.where(np.isnan(e), out["cl"], e)
    return {k: np.where(f >= H, 1.0, np.clip(v, 0.0, 1.0)) for k, v in out.items()}


def bootstrap_auc(
    p: Array, y: Array, w: Array, cluster: NDArray[np.int64], vintage_of_cluster: NDArray[np.int64]
) -> tuple[Array, Array]:
    """Cluster bootstrap of the pooled AUC and of the mean within-vintage AUC, from weighted histograms of the
    prediction (AUC_BINS quantile bins) per cluster. The seed is fixed: every method gets the same resamples."""
    edges = np.quantile(p, np.linspace(0, 1, AUC_BINS + 1)[1:-1])
    flat = cluster * AUC_BINS + np.searchsorted(edges, p)
    shape = (int(cluster.max()) + 1, AUC_BINS)
    pos = np.bincount(flat, weights=w * y, minlength=shape[0] * AUC_BINS).reshape(shape)
    neg = np.bincount(flat, weights=w * (1 - y), minlength=shape[0] * AUC_BINS).reshape(shape)
    counts = np.random.default_rng(SEED).multinomial(
        shape[0], np.full(shape[0], 1 / shape[0]), size=N_BOOT
    )

    def auc(c: Array, a: Array, b: Array) -> Array:
        ps, ns = c @ a, c @ b
        below = np.cumsum(ns, axis=1) - ns
        return np.asarray(
            (ps * (below + 0.5 * ns)).sum(axis=1)
            / np.maximum(ps.sum(axis=1) * ns.sum(axis=1), 1e-300)
        )

    within = [
        auc(counts[:, m], pos[m], neg[m])
        for m in (vintage_of_cluster == v for v in np.unique(vintage_of_cluster))
    ]
    return auc(counts, pos, neg), np.mean(within, axis=0)


def score(p: Array, test: pd.DataFrame, y: Array, w: Array) -> tuple[dict[str, object], Array]:
    """Metrics on the test loans with a known label (weights = inverse probability of censoring). Returns the
    metrics and the bootstrap replicates of the pooled AUC."""
    ok = w > 0
    p, y, w = np.clip(p[ok], 1e-9, 1 - 1e-9), y[ok], w[ok]
    year, fico = test.orig_year.to_numpy()[ok], test.fico.to_numpy()[ok]
    cluster, keys = pd.factorize(pd.Series(year).astype(str) + test.state.to_numpy()[ok])
    pooled, within = bootstrap_auc(p, y, w, cluster, np.array([int(k[:4]) for k in keys]))
    order = np.argsort(p, kind="stable")
    bins = np.minimum((np.cumsum(w[order]) / w.sum() * 10).astype(int), 9)
    mass, pred, real = (
        np.bincount(bins, weights=v[order], minlength=10) for v in (w, w * p, w * y)
    )

    def ratio(m: Mask) -> float:
        return float((w[m] * p[m]).sum() / max(float((w[m] * y[m]).sum()), 1e-12))

    bands = {k: ratio((fico >= lo) & (fico < hi)) for k, (lo, hi) in FICO_BANDS.items()}
    by_vintage = {int(v): ratio(year == v) for v in np.unique(year)}
    out: dict[str, object] = {
        "level_ratio": round(ratio(np.ones(len(p), dtype=bool)), 4),
        "level_ratio_by_fico": {k: round(v, 4) for k, v in bands.items()},
        "slow_over_fast": round(bands["fico_720_up"] / bands["fico_below_660"], 4),
        "level_ratio_by_vintage": {k: round(v, 4) for k, v in by_vintage.items()},
        "ece": round(float(np.abs(pred - real).sum() / w.sum()), 5),
        "log_loss": round(
            float(-(w * (y * np.log(p) + (1 - y) * np.log(1 - p))).sum() / w.sum()), 5
        ),
        "auc": round(float(roc_auc_score(y, p, sample_weight=w)), 4),
        "auc_ci95": [round(float(v), 4) for v in np.percentile(pooled, [2.5, 97.5])],
        "auc_within_vintage": round(
            float(
                np.mean(
                    [
                        roc_auc_score(y[year == v], p[year == v], sample_weight=w[year == v])
                        for v in np.unique(year)
                    ]
                )
            ),
            4,
        ),
        "auc_within_vintage_ci95": [round(float(v), 4) for v in np.percentile(within, [2.5, 97.5])],
        "deciles_predicted": (pred / mass).round(5).tolist(),
        "deciles_realised": (real / mass).round(5).tolist(),
    }
    return out, pooled


def predictions(
    train: pd.DataFrame, test: pd.DataFrame, year: int, version: str, design: str
) -> tuple[dict[str, Array], dict[str, float]]:
    """Predicted probability of default within H months for the test loans, per strategy and head."""
    base, base_test = designs(train, test)
    vintage, vintage_test = train.orig_year.to_numpy(), test.orig_year.to_numpy()
    positive, age = train.positive.to_numpy(dtype=float), train.age.to_numpy()
    ones = np.ones(len(train))
    old = age >= H
    at_s = np.zeros(len(train))  # weights of the loans with H months of age, from what S shows
    at_s[old] = ipcw(
        vintage[old], train.dur.to_numpy()[old], train.kind.to_numpy()[old].astype(str)
    )
    today = ipcw(
        vintage,
        train.event_age_clean.to_numpy().astype(int),
        train.event_type_clean.to_numpy().astype(str),
    )
    final = ((train.event_type_clean == "default") & (train.event_age_clean <= H)).to_numpy(float)
    e = completeness(train, version, year)
    rng = np.random.default_rng(SEED + year)
    q = later_default_probability(train, base[1], at_s, rng)
    info = {f"mean_completeness_{k}": round(float(v[age < H].mean()), 4) for k, v in e.items()}
    # strategy: (target, weight, vintage term)
    plain = {
        "final": (final, today, "all"),
        "naive": (positive, ones, "all"),
        "mature": (positive, at_s, "carry"),
        "drop_recent": (positive, at_s * (age < H + 36), "pooled"),
        "esdfm": (np.where(positive > 0, 1.0, q), ones, "all"),
    }
    out: dict[str, Array] = {}

    def matrices(mode: str, used: Mask) -> tuple[Design, Design]:
        code, code_test = vintage_codes(
            mode if design == "vintage" else "pooled", used, vintage, vintage_test
        )
        levels = np.unique(code[used])
        return with_vintage(base, code, levels), with_vintage(base_test, code_test, levels)

    naive: dict[str, Array] = {}  # predictions of the naive models for the training loans
    for name, (target, weight, mode) in plain.items():
        x, x_test = matrices(mode, weight > 0)
        for head in HEADS:
            out[f"{name}_{head}"], fitted = fit_predict(head, x, target, weight, x_test)
            if name == "naive":
                naive[head] = fitted
    x, x_test = matrices("all", ones > 0)
    seen = (train.kind == "default").to_numpy(float)
    t = np.maximum(np.where(seen > 0, train.dur - 0.5, train.followed), 0.5).astype(float)
    out["dfm_logit"] = delayed_feedback(x[0], base[0], seen, t, x_test[0], base_test[0])
    for name, source in PU.items():
        for head in HEADS:
            # start: level per vintage = positives / sum of e; differences between loans from the naive model
            g = pd.DataFrame(
                {"v": vintage, "s": positive, "e": e[source], "p": naive[head]}
            ).groupby("v")
            level = g.s.sum() / np.maximum(g.e.sum(), 1e-9) / np.maximum(g.p.mean(), 1e-12)
            p = np.minimum(level.reindex(vintage).to_numpy() * naive[head], 0.9)
            for _ in range(EM_ITERATIONS):
                w = p * (1 - e[source]) / np.maximum(1 - p * e[source], 1e-9)
                last = p
                out[f"{name}_{head}"], p = fit_predict(
                    head, x, np.where(positive > 0, 1.0, w), ones, x_test
                )
            info[f"em_last_change_{name}_{head}"] = round(float(np.abs(p - last).mean()), 6)
    return out, info


def snapshot(
    args: tuple[int, str, str],
) -> tuple[int, dict[str, object], pd.DataFrame, dict[str, Array]]:
    year, version, design = args
    started = time.time()
    loans = pd.read_parquet(ART / f"loans_{version}.parquet")
    x = view(loans, year)
    train = x[x.fold < N_TRAIN].reset_index(drop=True)
    test = x[(x.fold >= N_TRAIN) & (x.orig_year > year - N_SCORED)].reset_index(drop=True)
    preds, info = predictions(train, test, year, version, design)
    y = ((test.event_type_clean == "default") & (test.event_age_clean <= H)).to_numpy(float)
    w = ipcw(
        test.orig_year.to_numpy(),
        test.event_age_clean.to_numpy().astype(int),
        test.event_type_clean.to_numpy().astype(str),
    )
    methods, replicates = {}, {}
    for name, p in preds.items():
        methods[name], replicates[name] = score(p, test, y, w)
    pairs = (
        pd.DataFrame({"w": w, "wy": w * y, **{k: w * p for k, p in preds.items()}})
        .groupby(test.orig_year.to_numpy())
        .sum()
    )
    pairs = pairs.div(pairs.w, axis=0).drop(columns="w").rename(columns={"wy": "realised"})
    pairs = pairs.rename_axis("cohort").reset_index().assign(snapshot=year)
    out: dict[str, object] = {
        "train_loans": len(train),
        "test_loans": len(test),
        "test_loans_with_known_label": int((w > 0).sum()),
        "realised_rate": round(float((w * y).sum() / w.sum()), 5),
        "rate_seen_at_snapshot": round(float(test.positive.mean()), 5),
        "seconds": round(time.time() - started, 1),
        **info,
        "methods": methods,
    }
    return year, out, pairs, replicates


def main(version: str, design: str, years: list[int]) -> None:
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[key] = "2"  # 4 workers x 2 threads
    with ProcessPoolExecutor(max_workers=min(WORKERS, len(years))) as pool:
        done = list(pool.map(snapshot, [(year, version, design) for year in years]))
    pd.concat([d[2] for d in done], ignore_index=True).to_parquet(
        ART / f"pairs_{version}_{design}.parquet"
    )
    np.savez_compressed(
        ART / f"auc_bootstrap_{version}_{design}.npz",
        **{f"{year}_{name}": r for year, _, _, reps in done for name, r in reps.items()},
    )
    (HERE / f"results_{version}_{design}.json").write_text(
        json.dumps({str(year): out for year, out, _, _ in done}, indent=1)
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    main(sys.argv[1], sys.argv[2], [int(y) for y in sys.argv[3:]] or list(SNAPSHOTS))
