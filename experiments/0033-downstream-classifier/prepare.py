"""Stage 1 of 0033: what each December snapshot knows about label completeness, from the frozen pipeline (POST HOC).

DATASET: 200k loans per vintage, simple random sample of the full Freddie Mac Standard files (experiments/0010);
version `corrected` = corrected censoring weights (pipeline of 0012, variant ipcw), `linked` = borrower-followed
(table of 0018). No frozen estimate changes: every snapshot calls `pipeline.estimate`.

Per snapshot S (December 2006 to 2018):
  artifacts/cohorts_<version>.parquet  the frozen cohort table, and `share_cl`: the timing curve of the chain ladder
                                       (yearly cohorts, equal speeds) at ages 0..84 months
  artifacts/units_<version>.parquet    per unit (vintage x first-payment month x state) of the young cohorts: loans,
                                       incidence seen at S, age at S, and the model's cumulative default hazard from
                                       that age to 84 months on the forecast path (covariates at their mean of the
                                       last 12 months): `future_cal` = calendar-adjusted chain ladder (equal speeds),
                                       `future_full` = full method (selected speed model)
The method's estimate for a unit is 1 - (1 - seen) exp(-future). Completeness = seen / estimate (run.py, drift.py).
Check in the code: the units give the frozen forecast-path estimate of each young cohort (`ours_uncond` of the
frozen table, and `pipeline.plug_in` for the calendar-adjusted chain ladder).
Also: artifacts/loans_<version>.parquet (training and test samples, the columns the classifiers need) and
artifacts/months_<version>.npz (default incidence by origination month as each drift snapshot shows it, and the
84-month incidence in today's records).
Usage: PYTHONHASHSEED=0 uv run python prepare.py <corrected|linked>
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from types import ModuleType
from typing import Final

import numpy as np
import pandas as pd

from maturity import onestage

HERE: Final = Path(__file__).resolve().parent
ART: Final = HERE / "artifacts"
sys.path.insert(0, str(HERE.parent / "0031-round3-backtests"))
import extras
import freddie

log = logging.getLogger(__name__)
H: Final = 84
SNAPSHOTS: Final = range(2006, 2019)
# drift detectors: every month from December 2006 to December 2009, then each December (month index = 12 y + m)
DRIFT_SNAPSHOTS: Final = [
    *range(2006 * 12 + 12, 2009 * 12 + 12),
    *(12 * y + 12 for y in SNAPSHOTS if y >= 2009),
]
N_TRAIN: Final = 50_000  # loans per vintage
N_TEST: Final = 50_000  # other loans per vintage
SEED: Final = 20261006
NUMERIC: Final = ["fico", "ltv", "dti", "rate", "rate_spread", "units"]
CATEGORICAL: Final = ["purpose", "occupancy", "first_time_buyer", "state"]
Step1 = tuple[np.ndarray, pd.Series, np.ndarray]


def unit_futures(
    pl: ModuleType, d: object, sub: pd.DataFrame, s0: int, fit1: Step1, inc: np.ndarray
) -> tuple[pd.DataFrame, float]:
    """Units of one cohort with their loans, incidence seen at S, age at S and the model's cumulative default hazard
    from that age to H on the forecast path; and the cohort estimate they imply. This is
    `pipeline.plug_in(conditional=False)` kept per unit; the caller compares the estimate with the frozen one."""
    beta, kappa, _ = fit1
    raw, stats = d.raw, d.stats  # type: ignore[attr-defined]
    sub = sub[sub.age <= H]
    units, first, uid = np.unique(sub.unit.to_numpy(), return_index=True, return_inverse=True)
    age, seen = sub.age.to_numpy(), (sub.mi <= s0).to_numpy()
    head = sub.iloc[first][["vintage", "fp_mi", "state"]].reset_index(drop=True)
    n, aj = np.zeros(len(units)), np.zeros(len(units))
    n[uid[age == 1]] = sub.exposure.to_numpy()[age == 1]
    np.maximum.at(aj, uid[seen], sub.aj.to_numpy()[seen])
    a_u = np.clip(s0 - head.fp_mi.to_numpy() + 1, 0, H).astype(int)
    last = sub[(sub.mi > s0 - 12) & seen]
    if last.empty:  # the rule of pipeline._persistence
        last = sub[seen] if seen.any() else sub
    by_state = last.groupby("state")[raw].mean()
    z = np.column_stack(
        [
            (np.array([by_state[r].get(s, by_state[r].mean()) for s in head.state]) - stats[r][0])
            / stats[r][1]
            for r in raw
        ]
    )
    rate = np.exp(head.state.map(kappa).fillna(0.0).to_numpy() + z @ beta)
    future = rate * (inc[: H + 1].sum() - np.cumsum(inc[: H + 1])[a_u])
    estimate = float((n * (1 - (1 - aj) * np.exp(-future))).sum() / max(n.sum(), 1e-12))
    return head.assign(n=n, aj=aj, a_u=a_u, future=future), estimate


def snapshot(pl: ModuleType, d: object, year: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The frozen cohort table of December `year` and, per unit of its young cohorts, the forecast-path plug-in."""
    s0 = int(pl.month_idx(year * 100 + 12)) - 1
    t, meta = pl.estimate(d, year, horizon=H)
    assert meta["calendar_on"], "the calendar step is off: no calendar completeness"
    ugrid, f0 = meta["f0"]  # type: ignore[misc]
    _, curves = pl.observed(d, s0 + 1, "year")
    chain = onestage.fit(extras.cohorts_of(t, curves, pl.AMAX), 0.0)
    assert np.allclose(chain.level, t.chain.to_numpy()), "not the frozen chain ladder"
    fit1 = pl.step1(d, s0)
    adjusted = pl.adjusted_curves(d, s0, fit1, "year")
    chain_cal = onestage.fit(extras.cohorts_of(t, adjusted, pl.AMAX), 0.0)
    cells = d.cells  # type: ignore[attr-defined]
    heads = []
    for i, r in enumerate(t.itertuples()):
        if not r.young:
            continue
        sub = cells[cells.year == r.cohort]
        a = int(r.a_used)
        incs = {
            "cal": pl.increments(
                adjusted[r.cohort], a, float(chain_cal.speeds[i]), chain_cal.ugrid, chain_cal.f0, H
            ),
            "full": pl.increments(adjusted[r.cohort], a, float(r.speed), ugrid, f0, H),
        }
        frozen = {
            "cal": pl.plug_in(d, sub, s0, fit1, incs["cal"], H, conditional=False)[0],
            "full": float(r.ours_cal_uncond),
        }
        parts = {name: unit_futures(pl, d, sub, s0, fit1, inc) for name, inc in incs.items()}
        for name, (_, estimate) in parts.items():
            assert abs(estimate - frozen[name]) < 1e-9, (year, r.cohort, name, estimate, frozen)
        heads.append(
            parts["cal"][0]
            .rename(columns={"future": "future_cal"})
            .assign(future_full=parts["full"][0].future)
        )
    units = pd.concat(heads, ignore_index=True).assign(snapshot=year)
    share = np.interp(np.arange(H + 1.0), chain.ugrid, chain.f0, right=1.0)
    log.info("%s %d: %d units, model %s", VERSION, year, len(units), meta["model"])
    keep = ["cohort", "A", "n", "events", "young", "naive", "drop_recent", "fixed_horizon"]
    keep += ["chain", "cal_chain", "ours", "ours_uncond", "speed"]
    cohorts = t[keep].assign(snapshot=year, model=meta["model"], share_cl=[share] * len(t))
    return cohorts, units


def months(pl: ModuleType, d: object) -> None:
    """Default incidence by origination month: what each drift snapshot shows (incidence at ages 0..H, nan after the
    month's age at the snapshot) and the 84-month incidence in today's records."""
    truth = pl.truth(d, "om", H)
    om = np.array(sorted(int(m) for m in truth))
    views = {}
    for s in DRIFT_SNAPSHOTS:
        info, curves = pl.observed(d, s, "om")
        view = np.full((len(om), H + 1), np.nan, dtype=np.float32)
        for m in info.cohort.to_numpy().astype(int):
            a = min(s - m, H)
            view[np.searchsorted(om, m), : a + 1] = curves[float(m)][: a + 1]
        views[f"seen_{s}"] = view
    counts = d.loans.groupby("om").size()  # type: ignore[attr-defined]
    np.savez_compressed(
        ART / f"months_{VERSION}.npz",
        om=om,
        n=counts.reindex(om).to_numpy(),
        truth=np.array([truth[float(m)][0] for m in om]),
        snapshots=np.array(DRIFT_SNAPSHOTS),
        **views,
    )
    log.info("%s: monthly curves at %d snapshots", VERSION, len(DRIFT_SNAPSHOTS))


def sample(d: object) -> None:
    """Training and test loans: N_TRAIN and N_TEST other loans per vintage, fixed seed."""
    x = d.loans  # type: ignore[attr-defined]
    rng = np.random.default_rng(SEED)
    fold = x.groupby("orig_year").orig_year.transform(lambda v: rng.permutation(len(v)))
    cols = [*NUMERIC, *CATEGORICAL, "orig_year", "om", "ev_mi", "event_type_clean"]
    out = x.loc[fold < N_TRAIN + N_TEST, [*cols, "event_age_clean"]].assign(fold=fold)
    out.to_parquet(ART / f"loans_{VERSION}.parquet")
    log.info("%s: %d sampled loans", VERSION, len(out))


def main() -> None:
    ART.mkdir(exist_ok=True)
    pl, d = freddie.load_pipeline(VERSION)
    sample(d)
    months(pl, d)
    done = [snapshot(pl, d, year) for year in SNAPSHOTS]
    cohorts = pd.concat([c for c, _ in done], ignore_index=True)
    truth = pl.truth(d, "year", H)
    cohorts["truth"] = [truth[c][0] for c in cohorts.cohort]
    cohorts.to_parquet(ART / f"cohorts_{VERSION}.parquet")
    pd.concat([u for _, u in done], ignore_index=True).to_parquet(ART / f"units_{VERSION}.parquet")


if __name__ == "__main__":
    if os.environ.get("PYTHONHASHSEED") != "0":
        raise SystemExit(
            "run with PYTHONHASHSEED=0 (the fixed-effects fit depends on set order, see 0011)"
        )
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    VERSION: Final = sys.argv[1]
    main()
