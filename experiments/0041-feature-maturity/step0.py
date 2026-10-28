"""Step 0 of FEATURE_MATURITY_SPEC: simulation of PU corrections with cohort-level and feature-dependent maturity.

DATASET: none; simulated data only (seed 20261006 + replication).
World: K annual cohorts with 12 entry months each, snapshot at the end of year K, horizon H = 84 months. The 6
youngest cohorts (median age 6 to 66 months) are scored. K = 12 is the spec's run (5 mature cohorts). K = 8 to 12
is the extra setting (1) added on 2026-10-06: the training mix of 0033 at the snapshots 2006 to 2010
(equal loans per vintage from 1999; 1 to 5 mature vintages, so most training vintages are immature). Two loan types with equal
horizon rates: fast (speed 1) and slow (speed 0.6) on a Weibull (shape 1.5, median 60 months) default-age clock.
A risk score r (standard normal, independent of the type) sets the horizon rate: logit p = -3.2 + 0.8 r. Cohort
speed drift: the 5 youngest cohorts are faster, log speed rising linearly to log 1.3 at the youngest. No calendar
effect. A training loan is positive when its default age is at most its age at the snapshot.
Label strategies (logistic regression on type, r and cohort dummies, the coding of 0033; a second classifier adds
type x cohort terms, as a flexible learner would; test = fresh loans of the 6 young cohorts,
truth = default within H):
  naive   events so far
  pu_cl   PU by EM with one completeness per origination month: the pooled distribution of the default ages
          within H in the mature cohorts, at the month's age
  pu_feat the same EM with a completeness per loan from the timing model of `timing.py` (Weibull and generalized
          gamma versions) fitted on the mature cohorts' defaults, feature = type
  final   the true within-H labels (oracle)
Reported per replication and cohort: level ratio by type, D = level ratio(slow) / level ratio(fast), level error,
AUC within the cohort.
Usage: uv run python step0.py [replications] [workers] -> results_step0.json, artifacts/step0.parquet,
research/drift-or-maturity/figdata/feature_maturity_step0.csv
"""

from __future__ import annotations

import json
import logging
import sys
from multiprocessing import Pool
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd
import timing
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

HERE: Final = Path(__file__).resolve().parent
FIGDATA: Final = HERE.parents[1] / "research" / "drift-or-maturity" / "figdata"
log = logging.getLogger(__name__)
H: Final = 84
SPEC_COHORTS: Final = (
    12  # the spec's run; 8 to 11 give the training mix of 0033 at the snapshots 2006 to 2009
)
YOUNG: Final = 6
PER_MONTH: Final = 1000  # training loans per entry month; the same number of test loans
SLOW: Final = 0.6
DRIFT: Final = float(np.log(1.3))
CLOCK: Final = stats.weibull_min(1.5, scale=60.0 / np.log(2) ** (1 / 1.5))
EM_STEPS: Final = 5
METHODS: Final = ("naive", "pu_cl", "pu_feat", "pu_feat_gengamma", "final")


def world(rng: np.random.Generator, k: int) -> pd.DataFrame:
    """One row per loan: cohort, age at the snapshot, type, score, default within H, default age."""
    month = np.repeat(np.arange(k * 12), PER_MONTH)
    cohort = month // 12
    n = len(month)
    slow = rng.integers(0, 2, n)
    score = rng.normal(size=n)
    rate = 1 / (1 + np.exp(3.2 - 0.8 * score))
    steps = np.clip(cohort - (k - 6), 0, 5)  # 0 for the old cohorts, 5 for the youngest
    speed = np.where(slow == 1, SLOW, 1.0) * np.exp(DRIFT * steps / 5)
    # default age given default within H: inverse of F(s a) / F(s H)
    u = rng.uniform(size=n) * CLOCK.cdf(speed * H)
    default_age = CLOCK.ppf(u) / speed
    event = rng.uniform(size=n) < rate
    return pd.DataFrame(
        {
            "month": month,
            "cohort": cohort,
            "age": k * 12 - month - 0.5,
            "slow": slow.astype(float),
            "score": score,
            "event": event,
            "default_age": np.where(event, default_age, np.inf),
        }
    )


def design(d: pd.DataFrame, k: int, *, interaction: bool) -> np.ndarray:
    """Type, score and cohort dummies; with `interaction` also type x cohort (a flexible learner's view)."""
    dummies = np.eye(k)[d.cohort.to_numpy()][:, 1:]
    extra = [dummies * d.slow.to_numpy()[:, None]] if interaction else []
    return np.column_stack([d.slow.to_numpy(), d.score.to_numpy(), dummies, *extra])


def logistic(x: np.ndarray, y: np.ndarray, w: np.ndarray) -> LogisticRegression:
    return LogisticRegression(C=1e4, max_iter=300).fit(x, y, sample_weight=w)


def pu(x: np.ndarray, positive: np.ndarray, e: np.ndarray) -> LogisticRegression:
    """EM for labels seen with probability e: an unlabelled loan is a positive with weight p (1 - e) / (1 - p e)."""
    e = np.clip(e, 1e-3, 1.0)
    p = np.clip(np.full(len(e), positive.sum() / e.sum()), 1e-4, 0.5)
    stacked = np.vstack([x, x])
    y = np.concatenate([np.ones(len(x)), np.zeros(len(x))])
    for _ in range(EM_STEPS):
        w = np.where(positive, 1.0, p * (1 - e) / (1 - p * e))
        model = logistic(stacked, y, np.concatenate([w, 1 - w]))
        p = model.predict_proba(x)[:, 1]
    return model


def replication(job: tuple[int, int]) -> pd.DataFrame:
    seed, k = job
    rng = np.random.default_rng(20261006 + 1000 * k + seed)
    train, test = world(rng, k), world(rng, k)
    out = [scores(train, test, seed, k, interaction=flag) for flag in (False, True)]
    return pd.concat(out, ignore_index=True).assign(cohorts=k, mature_cohorts=k - YOUNG - 1)


def scores(
    train: pd.DataFrame, test: pd.DataFrame, seed: int, k: int, *, interaction: bool
) -> pd.DataFrame:
    x = design(train, k, interaction=interaction)
    positive = (train.default_age <= train.age).to_numpy()
    mature = (train.cohort < k - YOUNG - 1).to_numpy()  # median loan at least H months old
    done = mature & train.event.to_numpy()
    ages = np.sort(train.default_age.to_numpy()[done])
    pooled = np.searchsorted(ages, np.minimum(train.age.to_numpy(), H), side="right") / len(ages)
    kind = train[["slow"]].to_numpy()
    models = {
        "naive": logistic(x, positive.astype(float), np.ones(len(x))),
        "pu_cl": pu(x, positive, pooled),
        "final": logistic(x, train.event.to_numpy().astype(float), np.ones(len(x))),
    }
    fitted = {}
    for name, weibull in (("pu_feat", True), ("pu_feat_gengamma", False)):
        fitted[name] = timing.fit(
            train.default_age.to_numpy()[done], kind[done], float(H), weibull=weibull
        )
        models[name] = pu(x, positive, fitted[name].maturity(train.age.to_numpy(), kind, float(H)))
    rows = []
    young = test[test.cohort >= k - YOUNG]
    xt, truth = design(young, k, interaction=interaction), young.event.to_numpy()
    for name, model in models.items():
        p = model.predict_proba(xt)[:, 1]
        for cohort, idx in young.groupby("cohort").indices.items():
            slow = young.slow.to_numpy()[idx] == 1
            level = {
                k: p[idx][m].mean() / truth[idx][m].mean()
                for k, m in (("slow", slow), ("fast", ~slow))
            }
            rows.append(
                {
                    "replication": seed,
                    "classifier": "type x cohort" if interaction else "additive",
                    "method": name,
                    "cohort": int(cohort) - k,  # -1 = the youngest
                    "age": float(young.age.to_numpy()[idx].mean()),
                    "level_ratio": p[idx].mean() / truth[idx].mean(),
                    "level_ratio_slow": level["slow"],
                    "level_ratio_fast": level["fast"],
                    "D": level["slow"] / level["fast"],
                    "auc_within_cohort": roc_auc_score(truth[idx], p[idx]),
                    "xi_slow": float(fitted[name].xi[0]) if name in fitted else np.nan,
                }
            )
    return pd.DataFrame(rows)


def main(replications: int, workers: int) -> None:
    parts = HERE / "artifacts" / "step0"
    parts.mkdir(parents=True, exist_ok=True)
    with Pool(workers) as pool:
        for k in range(8, SPEC_COHORTS + 1):  # one checkpoint per setting: a stopped run continues
            if (parts / f"{k}.parquet").exists():
                continue
            jobs = [(seed, k) for seed in range(replications)]
            done = pd.concat(pool.imap_unordered(replication, jobs, chunksize=4), ignore_index=True)
            done.to_parquet(parts / f"{k}.parquet")
            log.info("setting with %d cohorts done", k)
    t = pd.concat([pd.read_parquet(f) for f in sorted(parts.glob("*.parquet"))], ignore_index=True)
    t["abs_log_D"] = np.log(t.D).abs()
    t["level_error"] = (t.level_ratio - 1).abs()
    columns = [
        "D",
        "abs_log_D",
        "level_ratio",
        "level_error",
        "level_ratio_slow",
        "level_ratio_fast",
        "auc_within_cohort",
    ]
    keys = ["cohorts", "classifier", "method"]
    by_age = t.groupby([*keys, "cohort"])[[*columns, "age"]].mean().reset_index()
    overall = t.groupby(keys)[columns].mean().reset_index()
    # setting (1) of the request of 2026-10-06: the training mixes of 0033 at 2006 to 2010, pooled
    pooled = t.groupby(["classifier", "method"])[columns].mean().reset_index()
    per_rep = t.groupby([*keys, "replication"]).abs_log_D.mean().unstack("method")
    closer = (per_rep.pu_feat < per_rep.pu_cl).groupby(level=["cohorts", "classifier"]).mean()
    xi = t[t.xi_slow.notna()].groupby(["cohorts", "method"]).xi_slow.agg(["mean", "std"]).round(4)
    summary = {
        "replications_per_setting": replications,
        "true_xi_of_the_slow_type": float(-np.log(SLOW)),
        "fitted_xi_slow": json.loads(xi.reset_index().to_json(orient="records")),
        "by_setting": json.loads(overall.round(4).to_json(orient="records")),
        "pooled_over_the_mixes_2006_to_2010": json.loads(pooled.round(4).to_json(orient="records")),
        "share_of_replications_where_pu_feat_is_closer_to_1": json.loads(
            closer.round(3).reset_index(name="share").to_json(orient="records")
        ),
        "by_cohort_age": json.loads(by_age.round(4).to_json(orient="records")),
    }
    (HERE / "results_step0.json").write_text(json.dumps(summary, indent=1))
    FIGDATA.mkdir(parents=True, exist_ok=True)
    by_age.round(5).to_csv(FIGDATA / "feature_maturity_step0.csv", index=False)
    log.info("%s", overall.round(3).to_string())


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    main(
        int(sys.argv[1]) if len(sys.argv) > 1 else 200, int(sys.argv[2]) if len(sys.argv) > 2 else 2
    )
