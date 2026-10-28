"""The condition of the shrinkage proposition per scored cohort, with the paper's V (POST HOC; replaces the criterion
numbers of experiments/0031).

DATASETS: Freddie Mac 200k (corrected weights; borrower-followed table of 0018) and the USPTO PatEx sample of
0020/0021, exactly as experiments/0031 uses them. No new data.

0031 took the information about log speed from second differences of the shape log-likelihood on the pooled,
piecewise-linear reference. The simulation of 0034 showed that this number is not stable (zero for half of the
cohorts, no rank correlation with the true information). Here V is the quantity of the paper (section "limits"):
the variance of eta(u) = 1 + dlog f0 / dlog u over the cohort's events so far, from the generalized gamma fitted to
the reference of the selected speed model, on the cohort's monthly intervals. It is the code of the frozen T3
diagnostic (`calendar_methods.precision`); the inflation factor that it implies must equal the frozen column.
Per cohort: D events so far, V, own variance phi / (D V) (and 1 / (D V)), omega2 = variance of the random-walk
candidate with the largest evidence, tau2 = posterior variance of the older neighbour's log speed under that random
walk with the information D V / phi. Condition of the proposition: omega2 + tau2 > own variance.
Usage: PYTHONHASHSEED=0 uv run python criterion.py -> artifacts/criterion_<name>.parquet; then report.py
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
from scipy import stats

from maturity import onestage

HERE: Final = Path(__file__).resolve().parent
EXP: Final = HERE.parent
sys.path[:0] = [str(EXP / "0031-round3-backtests"), str(EXP / "0005-freddie-backtests")]
log = logging.getLogger(__name__)


def information_per_event(
    par: tuple[float, float, float, float], u: np.ndarray
) -> tuple[float, float]:
    """(V, e): variance and mean of eta over the events in the intervals with end-points u (u[0] = 0), for the
    generalized gamma `par`. e is the elasticity of F0 at the last point. Same steps as the frozen diagnostic."""
    a, c, scale, _ = par
    cdf = stats.gengamma.cdf(u, a, c, scale=scale)
    g = u * stats.gengamma.pdf(u, a, c, scale=scale)
    d_f = np.diff(cdf)
    ok = d_f > 1e-300
    if ok.sum() < 3:
        return float("nan"), float("nan")
    eta = np.diff(g)[ok] / d_f[ok]
    w = d_f[ok] / d_f[ok].sum()
    e = float(np.sum(w * eta))
    return float(np.sum(w * (eta - e) ** 2)), e


def snapshot(
    pl: ModuleType, cal: ModuleType, d: object, year: int, *, horizon: int, cutoff: int | None
) -> pd.DataFrame:
    """All cohorts of the fit at one snapshot with the quantities of the proposition."""
    import extras  # of 0031; needs the pipeline module loaded under the name "pipeline"

    t, meta = pl.estimate(d, year, horizon=horizon, cutoff=cutoff)
    s0 = int(pl.month_idx(year * 100 + 12)) - 1
    cohorts = extras.cohorts_of(t, pl.adjusted_curves(d, s0, pl.step1(d, s0), "year"), pl.AMAX)
    selection = onestage.select_model(cohorts, onestage.dispersion(cohorts))
    assert selection.model == meta["model"], "not the frozen speed model"
    assert np.allclose(selection.fit.speeds, t.speed.to_numpy(), rtol=1e-9), "not the frozen speeds"
    par = cal.fit_gengamma(selection.fit.ugrid, selection.fit.f0)
    pieces = [
        information_per_event(par, float(s) * np.arange(int(a) + 1.0))
        for s, a in zip(selection.fit.speeds, t.a_used, strict=True)
    ]
    v, e = (np.array([p[i] for p in pieces]) for i in range(2))
    with np.errstate(divide="ignore", invalid="ignore"):
        inflation = 1 + e**2 / v
    stored = t.inflation_gg.to_numpy(dtype=float)
    both = np.isfinite(stored) & np.isfinite(inflation)
    assert np.allclose(inflation[both], stored[both], rtol=1e-6), "not the frozen inflation factor"
    events, phi = t.events.to_numpy(dtype=float), float(selection.phi)
    rw = {k: x for k, x in selection.log_evidence.items() if k.startswith("rw(")}
    omega2 = float(max(rw, key=lambda k: rw[k])[3:-1])
    info = np.nan_to_num(events * v)
    k = len(t)
    difference = np.eye(k) - np.eye(k, k=-1)  # first row anchors the first cohort at speed 1
    posterior = np.linalg.inv(np.diag(info / phi) + difference.T @ difference / omega2)
    tau2 = np.concatenate([[np.nan], np.diag(posterior)[:-1]])
    with np.errstate(divide="ignore"):
        own = np.where(info > 0, 1 / np.maximum(info, 1e-300), np.inf)
    out = t[["cohort", "A", "a_used", "h", "mature", "young", "speed", "maturity_m"]].copy()
    return out.assign(
        snapshot=year,
        D=events,
        V=v,
        elasticity=e,
        inflation=inflation,
        phi=phi,
        own_var=own,
        own_var_phi=phi * own,
        omega2=omega2,
        tau2=tau2,
        holds=omega2 + tau2 > phi * own,
        holds_without_phi=omega2 + tau2 > own,
        model=selection.model,
        gengamma_a=par[0],
        gengamma_c=par[1],
        gengamma_scale=par[2],
    )


def freddie(version: str) -> pd.DataFrame:
    import freddie as round3  # of 0031

    pl, d = round3.load_pipeline(version)
    import calendar_methods as cal

    cut = int(pl.month_idx(202002))
    rows = []
    for name, (years, truncated, _) in round3.SETS.items():
        for year in years:
            rows.append(
                snapshot(
                    pl, cal, d, year, horizon=round3.H, cutoff=cut if truncated else None
                ).assign(set=name)
            )
            log.info("freddie %s %d", version, year)
    return pd.concat(rows, ignore_index=True)


def patents() -> pd.DataFrame:
    sys.path[:0] = [
        str(EXP / "0021-patex-backtest"),
        str(EXP / "0020-patex-feasibility"),
        str(EXP / "0012-freddie-200k-harp-correction"),
    ]
    import backtest as frozen
    import calendar_methods as cal
    import check
    import pipeline as pl

    pl._persistence = frozen.persistence
    applications = check.load()
    pool = applications[applications.year.isin(check.YEARS)]
    sample = pool.groupby("year").sample(n=check.N_PER_YEAR, random_state=check.SEED)
    d = frozen.build(sample, check.inventory(applications))
    rows = []
    for year in frozen.SNAPSHOTS:
        rows.append(snapshot(pl, cal, frozen.seen_by(d, year), year, horizon=frozen.H, cutoff=None))
        log.info("patents %d", year)
    return pd.concat(rows, ignore_index=True)


if __name__ == "__main__":
    if os.environ.get("PYTHONHASHSEED") != "0":
        raise SystemExit("run with PYTHONHASHSEED=0 (the fixed-effects fit depends on set order)")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    (HERE / "artifacts").mkdir(exist_ok=True)
    name = sys.argv[1]
    table = patents() if name == "patents" else freddie(name)
    table.to_parquet(HERE / "artifacts" / f"criterion_{name}.parquet")
