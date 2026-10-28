"""E5: prediction with loan features when labels are delayed (design notes T5, sections 6 and 8).

DATASET: 200k loans per vintage, simple random sample of the full Freddie Mac Standard files (experiments/0010),
borrower-followed version (experiments/0018: a HARP parent continues as its relief loan).
Snapshots: December 2008, 2012, 2016. Every model is trained on what the snapshot shows for 50,000 loans per vintage
(vintages 1999 to the snapshot year). It is scored on 50,000 OTHER loans of each of the 3 youngest vintages against
the realised default within 84 months of origination (prepayment = no default).
Models (features: FICO, LTV, DTI, rate spread, units, purpose, occupancy, channel, first-time buyer, property type,
state). Only "ours" has cohort terms: the origination year in its rate head and one speed per vintage.
  naive          label = defaulted by the snapshot, all loans
  fixed_horizon  label = defaulted within 84 months, only loans with 84 months of follow-up at the snapshot
  drop_recent    naive label, only loans with at least 36 months of follow-up
  dfm            Chapelle (2014) delayed-feedback model: logistic rate, exponential delay, no cohort terms
  ours           T5 section 6: eventual rate p(x, c); latency Weibull AFT, F(a) = 1 - exp(-(s_c e^{g'x} a)^k), one
                 speed per vintage with a random-walk penalty; no calendar term (v1). Prediction: p(x, c) F(84 | x, c).
                 ours_logit = joint maximum likelihood with a logistic rate head; ours_hgb = EM started at that fit,
                 with a gradient-boosting rate head (plain EM from a flat start moved too slowly).
The naive, fixed-horizon, drop-recent and "ours" heads come in two versions: gradient boosting (hgb) and logistic.
Slow-against-fast splits (fixed before the run; "slow" and "fast" are set from training data at the snapshot):
FICO band and loan purpose (slow = the category with the latest median default age among loans with 84 months of
follow-up), and terciles of the fitted AFT speed e^{g'x}.
Usage: PYTHONHASHSEED=0 uv run python run.py -> results.json
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
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

from maturity import data

log = logging.getLogger(__name__)
HERE: Final = Path(__file__).resolve().parent
Array = NDArray[np.float64]
LINKED: Final = data.DATA_DIR / "processed" / "freddie_200k_loans_linked.parquet"
SNAPSHOTS: Final = (2008, 2012, 2016)
H: Final = 84
N_TRAIN: Final = 50_000
N_TEST: Final = 50_000
N_YOUNG: Final = 3
SEED: Final = 20261004
NUMERIC: Final = ["fico", "ltv", "dti", "rate_spread", "units"]
CATEGORICAL: Final = [
    "purpose",
    "occupancy",
    "channel",
    "first_time_buyer",
    "property_type",
    "state",
]
LATENCY_CATEGORICAL: Final = ["purpose", "occupancy", "first_time_buyer"]
EM_ITERATIONS: Final = 8
OMEGA2: Final = 0.03  # random-walk variance of log speed across vintages
FICO_BANDS: Final = [0, 660, 720, 1000]


@dataclass(frozen=True, slots=True)
class Loans:
    """Loans as a snapshot shows them, plus the later truth (used only for scoring)."""

    frame: pd.DataFrame  # features, orig_year
    trees: pd.DataFrame  # design for gradient boosting
    linear: Array  # standardised numerics + missing flags + one-hot categoricals + origination year
    latency: Array  # compact design for the AFT term
    cohort: NDArray[np.int64]
    seen: Array  # defaulted by the snapshot
    time: Array  # months: default age if seen, else follow-up at the snapshot
    truth: Array  # defaulted within 84 months (nan if unknown)


def month_idx(yyyymm: pd.Series | int) -> pd.Series | int:
    return (yyyymm // 100) * 12 + yyyymm % 100


def load() -> pd.DataFrame:
    cols = [*NUMERIC, *CATEGORICAL, "orig_year", "first_payment"]
    cols += ["event_type_clean", "event_age_clean", "event_period_clean"]
    d = pd.read_parquet(LINKED, columns=cols)
    d = d[d.event_age_clean.notna() & d.orig_year.between(1999, max(SNAPSHOTS))].copy()
    d["om"] = month_idx(d.first_payment.astype(int)) - 1
    d["ev_mi"] = month_idx(d.event_period_clean.astype(int))
    d["default"] = d.event_type_clean == "default"
    known = d.default | (d.event_type_clean == "prepay") | (d.event_age_clean >= H)
    d["truth"] = np.where(known, (d.default & (d.event_age_clean <= H)).astype(float), np.nan)
    rng = np.random.default_rng(SEED)
    d["fold"] = d.groupby("orig_year").orig_year.transform(lambda x: rng.permutation(len(x)))
    return d


def designs(train: pd.DataFrame, test: pd.DataFrame, s: int) -> tuple[Loans, Loans]:
    """Encoders are fitted on the training loans only."""
    med = train[NUMERIC].median()
    mean, sd = train[NUMERIC].fillna(med).mean(), train[NUMERIC].fillna(med).std()
    levels = {c: sorted(train[c].dropna().unique()) for c in CATEGORICAL}
    year0, year_sd = train.orig_year.mean(), train.orig_year.std()

    def build(x: pd.DataFrame) -> Loans:
        z = (x[NUMERIC].fillna(med) - mean) / sd
        flags = x[["fico", "dti"]].isna().astype(float).add_suffix("_missing")
        cats = {c: pd.Categorical(x[c], categories=levels[c]) for c in CATEGORICAL}
        hot = pd.concat(
            [pd.get_dummies(cats[c], prefix=c, dtype=float).iloc[:, 1:] for c in CATEGORICAL],
            axis=1,
        )
        year = ((x.orig_year - year0) / year_sd).to_numpy()[:, None]
        trees = x[NUMERIC].assign(
            **{c: cats[c].codes for c in CATEGORICAL}, orig_year=x.orig_year.to_numpy()
        )
        lat_hot = [c for c in hot.columns if c.rsplit("_", 1)[0] in LATENCY_CATEGORICAL]
        seen = (x.default & (x.ev_mi <= s)).to_numpy()
        # sub-distribution view: a prepaid loan stays at risk to the snapshot; a censored loan leaves
        censored = (x.event_type_clean == "censored") & (x.ev_mi <= s)
        follow = np.where(censored, x.event_age_clean, s - x.om).astype(float)
        return Loans(
            x,
            trees,
            np.hstack([z.to_numpy(), flags.to_numpy(), hot.to_numpy(), year]),
            np.hstack(
                [z[["fico", "ltv", "dti", "rate_spread"]].to_numpy(), hot[lat_hot].to_numpy()]
            ),
            (x.orig_year - 1999).to_numpy(dtype=np.int64),
            seen.astype(float),
            np.maximum(np.where(seen, x.event_age_clean - 0.5, follow), 0.5).astype(float),
            x.truth.to_numpy(),
        )

    return build(train), build(test)


def boosting(
    x: pd.DataFrame, y: Array, weight: Array | None = None
) -> HistGradientBoostingClassifier:
    model = HistGradientBoostingClassifier(
        max_iter=120,
        learning_rate=0.1,
        max_leaf_nodes=31,
        min_samples_leaf=200,
        l2_regularization=1.0,
        categorical_features=CATEGORICAL,
        early_stopping=False,
        random_state=0,
    )
    return model.fit(x, y, sample_weight=weight)


def logistic(x: Array, y: Array, weight: Array | None = None) -> LogisticRegression:
    return LogisticRegression(C=10.0, max_iter=300).fit(x, y, sample_weight=weight)


def weibull_cdf(theta: Array, t: Array | float, shape: float) -> Array:
    return np.asarray(1 - np.exp(-((theta * t) ** shape)))


def fit_latency(z: Loans, weight: Array, start: Array) -> Array:
    """Weighted Weibull AFT: parameters = [log shape, gamma (latency columns), log speed per vintage]. Events weigh 1,
    other loans weigh P(eventual default | none yet); random-walk penalty on the log speeds."""
    n_gamma, n_cohort = z.latency.shape[1], int(z.cohort.max()) + 1
    log_t, n = np.log(z.time), len(z.time)

    def objective(par: Array) -> tuple[float, Array]:
        shape, gamma, speed = float(np.exp(par[0])), par[1 : 1 + n_gamma], par[1 + n_gamma :]
        eta = speed[z.cohort] + z.latency @ gamma + log_t
        u = np.exp(np.minimum(shape * eta, 30.0))
        loglik = weight * (z.seen * (np.log(shape) + shape * eta - log_t) - u)
        d_eta = weight * shape * (z.seen - u)
        steps = np.diff(speed)
        penalty = float(np.sum(steps**2) / (2 * OMEGA2))
        d_speed = np.bincount(z.cohort, weights=d_eta, minlength=n_cohort)
        d_speed[1:] -= steps / OMEGA2
        d_speed[:-1] += steps / OMEGA2
        d_shape = shape * float(np.sum(weight * (z.seen * (1 / shape + eta) - u * eta)))
        grad = np.concatenate([[d_shape], z.latency.T @ d_eta, d_speed])
        return -(float(loglik.sum()) - penalty) / n, -grad / n

    res = optimize.minimize(objective, start, jac=True, method="L-BFGS-B", options={"maxiter": 200})
    return np.asarray(res.x)


def latency_theta(par: Array, z: Loans) -> tuple[Array, float]:
    n_gamma = z.latency.shape[1]
    return np.exp(par[1 + n_gamma :][z.cohort] + z.latency @ par[1 : 1 + n_gamma]), float(
        np.exp(par[0])
    )


def joint_logistic(train: Loans) -> tuple[Array, Array]:
    """Maximum likelihood of the whole model with a logistic rate head: p = sigmoid(a'x) (x includes the origination
    year), Weibull AFT latency with one speed per vintage. A loan with a default seen at t contributes p f(t); any
    other loan contributes 1 - p F(t). Returns (a, latency parameters)."""
    x = np.hstack([np.ones((len(train.seen), 1)), train.linear])
    k, n_gamma, n_cohort = x.shape[1], train.latency.shape[1], int(train.cohort.max()) + 1
    log_t, n, seen = np.log(train.time), len(train.seen), train.seen

    def objective(par: Array) -> tuple[float, Array]:
        a, lat = par[:k], par[k:]
        shape, gamma, speed = float(np.exp(lat[0])), lat[1 : 1 + n_gamma], lat[1 + n_gamma :]
        p = expit(x @ a)
        eta = speed[train.cohort] + train.latency @ gamma + log_t
        u = np.exp(np.minimum(shape * eta, 30.0))
        surv = np.exp(-u)
        rest = np.maximum(1 - p + p * surv, 1e-12)
        loglik = seen * (np.log(np.maximum(p, 1e-12)) + np.log(shape) + shape * eta - log_t - u)
        loglik = loglik + (1 - seen) * np.log(rest)
        d_a = seen * (1 - p) - (1 - seen) * p * (1 - p) * (1 - surv) / rest
        d_eta = seen * shape * (1 - u) - (1 - seen) * p * surv * u * shape / rest
        d_shape = float(
            np.sum(
                seen * (1 + shape * eta * (1 - u)) - (1 - seen) * p * surv * u * shape * eta / rest
            )
        )
        steps = np.diff(speed)
        d_speed = np.bincount(train.cohort, weights=d_eta, minlength=n_cohort)
        d_speed[1:] -= steps / OMEGA2
        d_speed[:-1] += steps / OMEGA2
        ridge = 1e-4 * n
        grad = np.concatenate([x.T @ d_a - ridge * a, [d_shape], train.latency.T @ d_eta, d_speed])
        value = (
            float(loglik.sum())
            - float(np.sum(steps**2)) / (2 * OMEGA2)
            - 0.5 * ridge * float(a @ a)
        )
        return -value / n, -grad / n

    start = np.concatenate(
        [np.zeros(k), [np.log(1.5)], np.zeros(n_gamma), np.full(n_cohort, np.log(1 / 70))]
    )
    start[0] = np.log(0.05 / 0.95)
    res = optimize.minimize(
        objective, start, jac=True, method="L-BFGS-B", options={"maxiter": 1500, "maxfun": 3000}
    )
    log.info(
        "  joint logistic fit: %d iterations, shape %.2f, converged %s",
        res.nit,
        np.exp(res.x[k]),
        res.success,
    )
    return np.asarray(res.x[:k]), np.asarray(res.x[k:])


def ours(train: Loans, test: Loans) -> tuple[dict[str, Array], Array]:
    """T5 section 6. "ours_logit" = the joint maximum-likelihood fit. "ours_hgb" = EM started at that fit, with a
    gradient-boosting rate head: E-step w = P(eventual default | none yet); M-step = weighted classifier (each loan
    without a default enters with label 1, weight w, and label 0, weight 1 - w) and weighted latency fit.
    Returns the predicted 84-month default probabilities of the test loans and their AFT speed factor e^{g'x}."""
    a, par = joint_logistic(train)
    n_gamma = train.latency.shape[1]
    ones = np.ones((len(test.seen), 1))
    theta, shape = latency_theta(par, test)
    preds = {
        "ours_logit": expit(np.hstack([ones, test.linear]) @ a)
        * weibull_cdf(theta, float(H), shape)
    }
    p = expit(np.hstack([np.ones((len(train.seen), 1)), train.linear]) @ a)
    idle = train.seen == 0
    x_trees = pd.concat([train.trees, train.trees[idle]], ignore_index=True)
    y = np.concatenate([train.seen, np.ones(int(idle.sum()))])
    model = None
    for it in range(EM_ITERATIONS):
        theta, shape = latency_theta(par, train)
        f = weibull_cdf(theta, train.time, shape)
        w = np.where(idle, p * (1 - f) / np.maximum(1 - p * f, 1e-12), 1.0)
        if it:
            par = fit_latency(train, w, par)
        model = boosting(x_trees, y, np.concatenate([np.where(idle, 1 - w, 1.0), w[idle]]))
        p = model.predict_proba(train.trees)[:, 1]
        log.info("  em hgb iteration %d: mean p %.4f, shape %.2f", it + 1, p.mean(), np.exp(par[0]))
    assert model is not None
    theta, shape = latency_theta(par, test)
    preds["ours_hgb"] = model.predict_proba(test.trees)[:, 1] * weibull_cdf(theta, float(H), shape)
    return preds, np.exp(test.latency @ par[1 : 1 + n_gamma])


def delayed_feedback(train: Loans, test: Loans) -> Array:
    """Chapelle (2014): p(x) = sigmoid(a'x), delay ~ Exponential(exp(b'x)); no cohort terms (the year column is
    dropped)."""
    x = np.hstack([np.ones((len(train.seen), 1)), train.linear[:, :-1]])
    k, n, t, seen = x.shape[1], len(train.seen), train.time, train.seen

    def objective(par: Array) -> tuple[float, Array]:
        p, rate = expit(x @ par[:k]), np.exp(np.minimum(x @ par[k:], 20.0))
        decay = np.exp(-rate * t)
        rest = np.maximum(1 - p + p * decay, 1e-12)
        loglik = seen * (np.log(np.maximum(p, 1e-12)) + np.log(rate) - rate * t) + (
            1 - seen
        ) * np.log(rest)
        d_a = seen * (1 - p) - (1 - seen) * p * (1 - p) * (1 - decay) / rest
        d_b = seen * (1 - rate * t) - (1 - seen) * p * decay * rate * t / rest
        grad = np.concatenate([x.T @ d_a, x.T @ d_b]) - 1e-4 * n * par
        return -(float(loglik.sum()) - 0.5e-4 * n * float(par @ par)) / n, -grad / n

    start = np.zeros(2 * k)
    start[0], start[k] = np.log(0.05 / 0.95), np.log(1 / 60)
    par = optimize.minimize(
        objective, start, jac=True, method="L-BFGS-B", options={"maxiter": 500}
    ).x
    xt = np.hstack([np.ones((len(test.seen), 1)), test.linear[:, :-1]])
    return np.asarray(expit(xt @ par[:k]) * (1 - np.exp(-np.exp(xt @ par[k:]) * H)))


def baselines(train: Loans, test: Loans, head: str) -> dict[str, Array]:
    # features only: no origination year (the usual practice, and the target q(x) of Proposition 5c)
    if head == "hgb":
        x, xt, fit = (
            train.trees.drop(columns="orig_year"),
            test.trees.drop(columns="orig_year"),
            boosting,
        )
    else:
        x, xt, fit = train.linear[:, :-1], test.linear[:, :-1], logistic
    follow = train.frame.s_age.to_numpy()
    mature, older = follow >= H, follow >= 36
    within = (train.seen == 1) & (train.time <= H)
    rows = {
        "naive": (np.ones(len(follow), dtype=bool), train.seen),
        "drop_recent": (older, train.seen),
    }
    rows["fixed_horizon"] = (mature, within.astype(float))
    return {
        f"{name}_{head}": fit(x[keep], y[keep]).predict_proba(xt)[:, 1]
        for name, (keep, y) in rows.items()
    }


def slow_fast(
    train: Loans, test: Loans, speed: Array
) -> dict[str, tuple[NDArray[np.bool_], NDArray[np.bool_]]]:
    """Per split: masks of the slow and of the fast test loans."""
    f = train.frame
    done = (f.s_age >= H).to_numpy() & (train.seen == 1) & (train.time <= H)
    groups = {
        "fico_band": (
            pd.cut(f.fico, FICO_BANDS, labels=False),
            pd.cut(test.frame.fico, FICO_BANDS, labels=False),
        ),
        "purpose": (f.purpose, test.frame.purpose),
    }
    out = {}
    for name, (g_train, g_test) in groups.items():
        median = pd.Series(train.time[done]).groupby(g_train.to_numpy()[done]).median()
        median = median[[k for k in median.index if k != "9"]]
        out[name] = ((g_test == median.idxmax()).to_numpy(), (g_test == median.idxmin()).to_numpy())
        log.info(
            "  split %s: slow = %s, fast = %s (median default age %s)",
            name,
            median.idxmax(),
            median.idxmin(),
            median.round(1).to_dict(),
        )
    low, high = np.quantile(speed, [1 / 3, 2 / 3])
    out["aft_speed_tercile"] = (speed <= low, speed >= high)
    return out


def score(
    pred: Array, test: Loans, splits: dict[str, tuple[NDArray[np.bool_], NDArray[np.bool_]]]
) -> dict[str, object]:
    ok = ~np.isnan(test.truth)
    y, p, year = test.truth[ok], pred[ok], test.frame.orig_year.to_numpy()[ok]
    decile = pd.qcut(pd.Series(p).rank(method="first"), 10, labels=False)
    table = pd.DataFrame({"p": p, "y": y}).groupby(decile.to_numpy()).mean()
    out: dict[str, object] = {
        "auc_pooled": round(float(roc_auc_score(y, p)), 4),
        "auc_within_vintage": round(
            float(np.mean([roc_auc_score(y[year == v], p[year == v]) for v in np.unique(year)])), 4
        ),
        "predicted_over_realised": round(float(p.mean() / y.mean()), 4),
        "deciles_predicted": table.p.round(5).tolist(),
        "deciles_realised": table.y.round(5).tolist(),
    }
    for name, (slow, fast) in splits.items():
        ratio = {
            k: float(p[m[ok]].mean() / y[m[ok]].mean()) for k, m in (("slow", slow), ("fast", fast))
        }
        out[f"under_ranking_{name}"] = {
            "slow": round(ratio["slow"], 4),
            "fast": round(ratio["fast"], 4),
            "slow_minus_fast": round(ratio["slow"] - ratio["fast"], 4),
            "relative": round(ratio["slow"] / ratio["fast"], 4),
        }
    return out


def snapshot(d: pd.DataFrame, year: int) -> dict[str, object]:
    s = int(month_idx(year * 100 + 12))
    x = d[(d.orig_year <= year) & (d.om <= s)].assign(s_age=lambda t: s - t.om)
    train = x[x.fold < N_TRAIN]
    young = x.orig_year > year - N_YOUNG
    test = x[young & (x.fold >= N_TRAIN) & (x.fold < N_TRAIN + N_TEST)]
    a, b = designs(train, test, s)
    log.info(
        "snapshot %d: train %d loans (%d defaults seen), test %d",
        year,
        len(train),
        int(a.seen.sum()),
        len(test),
    )
    preds: dict[str, Array] = {}
    for head in ("hgb", "logit"):
        preds |= baselines(a, b, head)
    preds["dfm"] = delayed_feedback(a, b)
    fitted, speed = ours(a, b)
    preds |= fitted
    splits = slow_fast(a, b, speed)
    truth = b.truth[~np.isnan(b.truth)]
    return {
        "train_loans": len(train),
        "test_loans": len(truth),
        "test_default_rate": round(float(truth.mean()), 5),
        "test_default_rate_seen_at_snapshot": round(float(b.seen.mean()), 5),
        "methods": {m: score(p, b, splits) for m, p in preds.items()},
    }


def main() -> None:
    d = load()
    out = {str(year): snapshot(d, year) for year in SNAPSHOTS}
    (HERE / "results.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    main()
