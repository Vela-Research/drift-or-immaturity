"""Cause-specific hazard model for default and prepayment: the domain-standard credit baseline (POST HOC, round 3).

DATASET: 200k loans per vintage, simple random sample of the full Freddie Mac Standard files (experiments/0010);
outcome definitions of experiments/0012, variant ipcw (HARP parents censored). Covariates: FHFA state house prices,
state unemployment, Freddie Mac PMMS 30-year rate (data/raw/covariates).

Model, for each cause k in {default, prepayment}, fitted on the loan-months observable at the snapshot S:
    events_k ~ Poisson(loans at risk x exp(age effect + state effect [+ vintage effect] + x'beta_k))
a proportional cause-specific hazard with a free monthly age baseline (piecewise exponential), on aggregated cells
(first-payment month x state x FICO band x LTV band x age). A loan leaves the risk set at default, prepayment or
censoring. x = loan features at origination (FICO, LTV, DTI, rate spread at origination, purpose, occupancy) and
time-varying covariates (state house-price growth, state unemployment, log mark-to-market LTV, refinance incentive =
note rate - market rate, and its positive part).
Estimate for a cohort: per cell, Aalen-Johansen default incidence seen at S + survival at S x the product integral
of the fitted hazards to the horizon; loan-weighted mean. Same estimand, cohorts, horizons and truth as the backtest.
Paths of the time-varying covariates after S: `real` = realised (as `ours`); `fcst` = last-12-month means, the
mark-to-market LTV follows the implied price path (as `ours_uncond`).
Usage: PYTHONHASHSEED=0 uv run python hazard.py -> artifacts/pairs_hazard.parquet, artifacts/coefficients.json
"""

from __future__ import annotations

import json
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd
import pyfixest as pf

from maturity import data

HERE: Final = Path(__file__).resolve().parent
EXP: Final = HERE.parent
sys.path.insert(0, str(EXP / "0003-freddie-sample"))
import shock_overlay as so

log = logging.getLogger(__name__)
LOANS: Final = data.DATA_DIR / "processed" / "freddie_200k_loans.parquet"
PAIRS: Final = EXP / "0031-round3-backtests" / "artifacts" / "pairs_freddie_corrected.parquet"
H: Final = 84
MI0: Final = 1998 * 12
N_MI: Final = so.CUTOFF_MI - MI0 + 1
FICO_EDGES: Final = [0, 680, 740, 1000]
STATIC: Final = [
    "fico",
    "fico_missing",
    "ltv",
    "dti",
    "dti_missing",
    "sato",
    "purchase",
    "cash_out",
    "investor",
]
DYNAMIC: Final = ["hpi_yoy", "unemp", "log_mtm", "incentive", "incentive_pos"]
X: Final = STATIC + DYNAMIC
CAUSES: Final = ("default", "prepay")


@dataclass(frozen=True, slots=True)
class Cells:
    units: pd.DataFrame  # one row per cell: keys, n, features
    active: np.ndarray  # [unit, age] loans at risk at the start of the age
    events: np.ndarray  # [unit, age, cause]
    aj: np.ndarray  # [unit, age] Aalen-Johansen default incidence up to the age
    surv: np.ndarray  # [unit, age] all-cause survival up to the age
    macro: dict[str, np.ndarray]  # hpi_yoy, unemp, log_hpi: [state, month]; pmms: [month]


def build() -> Cells:
    cols = ["orig_year", "first_payment", "state", "fico", "ltv", "dti", "rate", "rate_spread"]
    d = pd.read_parquet(
        LOANS, columns=[*cols, "purpose", "occupancy", "event_age_clean", "event_type_clean"]
    )
    d = d[
        d.event_age_clean.notna() & d.first_payment.notna() & d.orig_year.between(1999, 2019)
    ].copy()
    fp = d.first_payment.astype(int)
    d["fp_mi"] = (fp // 100) * 12 + fp % 100 - 1
    d["fico_band"] = pd.cut(d.fico, FICO_EDGES, labels=False, right=False).fillna(3).astype(int)
    d["ltv"] = d.ltv.fillna(d.ltv.median()).clip(1, 150)  # 200 loans have no LTV
    d["ltv_band"] = pd.cut(d.ltv, so.LTV_EDGES, labels=False).astype(int)
    keys = ["orig_year", "fp_mi", "state", "fico_band", "ltv_band"]
    flags = {
        "dti_missing": d.dti.isna(),
        "purchase": d.purpose == "P",
        "cash_out": d.purpose == "C",
        "investor": d.occupancy == "I",
    }
    d = d.assign(**{k: v.astype(float) for k, v in flags.items()}, sato=d.rate_spread)
    grouped = d.groupby(keys, observed=True)
    uid = grouped.ngroup().to_numpy()
    units = grouped[["fico", "ltv", "dti", "rate", "sato", *flags]].mean().reset_index()
    units["n"] = np.bincount(uid, minlength=len(units))
    units["state_code"] = pd.factorize(units.state, sort=True)[0]
    units["fico_missing"] = (units.fico_band == 3).astype(float)
    # null is not zero: a cell without a value gets the overall mean and keeps its missing indicator
    units["fico"] = units.fico.fillna(float(d.fico.mean())) / 100
    units["dti"] = units.dti.fillna(float(d.dti.mean())) / 10
    units["ltv"] = units.ltv / 10
    n_units = len(units)
    dur = d.event_age_clean.clip(lower=1).astype(int).to_numpy()
    kind = d.event_type_clean.map({"default": 0, "prepay": 1, "censored": 2}).to_numpy()
    ok = (dur <= H) & ~np.isnan(kind)
    flat = (uid[ok] * (H + 1) + dur[ok]) * 3 + kind[ok].astype(int)
    counts = np.bincount(flat, minlength=n_units * (H + 1) * 3).reshape(n_units, H + 1, 3)
    exits = counts.sum(axis=2)
    active = units.n.to_numpy()[:, None] - np.concatenate(
        [np.zeros((n_units, 1)), np.cumsum(exits, axis=1)[:, :-1]], axis=1
    )
    zero = np.zeros(active.shape)
    gone = np.divide(counts[:, :, :2].sum(axis=2), active, out=zero.copy(), where=active > 0)
    surv = np.cumprod(1 - gone, axis=1)
    before = np.concatenate([np.ones((n_units, 1)), surv[:, :-1]], axis=1)
    aj = np.cumsum(
        before * np.divide(counts[:, :, 0], active, out=zero.copy(), where=active > 0), axis=1
    )
    return Cells(units, active, counts[:, :, :2].astype(float), aj, surv, macro(units))


def macro(units: pd.DataFrame) -> dict[str, np.ndarray]:
    """Covariates by state code and 0-based month index minus MI0."""
    hpi, unemp = so._covariates()
    states = dict(zip(units.state, units.state_code, strict=True))
    out = {k: np.full((len(states), N_MI), np.nan) for k in ("hpi_yoy", "unemp", "log_hpi")}
    months = np.arange(MI0, so.CUTOFF_MI + 1)
    for state, g in hpi[hpi.state.isin(states)].groupby("state"):
        by_quarter = g.set_index("qi")
        out["hpi_yoy"][states[state]] = by_quarter.hpi_yoy.reindex(months // 3).to_numpy()
        out["log_hpi"][states[state]] = np.log(by_quarter.hpi.reindex(months // 3).to_numpy())
    for state, g in unemp[unemp.state.isin(states)].groupby("state"):
        out["unemp"][states[state]] = g.set_index("mi").unemp.reindex(months).to_numpy()
    p = pd.read_csv(data.DATA_DIR / "raw" / "covariates" / "pmms.csv", usecols=["date", "pmms30"])
    date = pd.to_datetime(p.date, format="%m/%d/%Y")
    monthly = p.pmms30.groupby(date.dt.year * 12 + date.dt.month - 1).mean()
    out["pmms"] = monthly.reindex(months).to_numpy()
    for k in ("hpi_yoy", "unemp", "log_hpi"):  # short gaps: carry the last value
        out[k] = pd.DataFrame(out[k]).ffill(axis=1).to_numpy()
    return out


def dynamic(
    c: Cells, unit: np.ndarray, age: np.ndarray, s0: int | None = None
) -> dict[str, np.ndarray]:
    """Time-varying covariates of (unit, age) cells. With s0, months after s0 use the forecast path."""
    u = c.units
    state, fp_mi = u.state_code.to_numpy()[unit], u.fp_mi.to_numpy()[unit]
    mi = fp_mi + age - 1
    m = c.macro
    at = np.clip(mi, MI0, so.CUTOFF_MI) - MI0
    origin = np.clip(fp_mi - 1, MI0, so.CUTOFF_MI) - MI0
    hpi_yoy, unemp, pmms, log_hpi = (
        m["hpi_yoy"][state, at],
        m["unemp"][state, at],
        m["pmms"][at],
        m["log_hpi"][state, at],
    )
    if s0 is not None:
        last = slice(s0 - 11 - MI0, s0 + 1 - MI0)
        growth, jobless = (
            np.nanmean(m["hpi_yoy"][:, last], axis=1),
            np.nanmean(m["unemp"][:, last], axis=1),
        )
        future = mi > s0
        hpi_yoy = np.where(future, growth[state], hpi_yoy)
        unemp = np.where(future, jobless[state], unemp)
        pmms = np.where(future, float(np.nanmean(m["pmms"][last])), pmms)
        log_hpi = np.where(
            future, m["log_hpi"][state, s0 - MI0] + growth[state] * (mi - s0) / 12, log_hpi
        )
    incentive = u.rate.to_numpy()[unit] - pmms
    return {
        "hpi_yoy": hpi_yoy,
        "unemp": unemp,
        "log_mtm": np.log(u.ltv.to_numpy()[unit] / 10) + m["log_hpi"][state, origin] - log_hpi,
        "incentive": incentive,
        "incentive_pos": np.maximum(incentive, 0.0),
    }


def fit(
    c: Cells, s0: int, cause: int, *, vintage: bool
) -> tuple[pd.Series, dict[str, dict[str, float]]]:
    """Poisson fit on the cells observable at S. Returns the coefficients and the fixed effects by factor."""
    u = c.units
    unit, age = np.nonzero(c.active[:, 1:] > 0)
    age = age + 1
    seen = (u.fp_mi.to_numpy()[unit] + age - 1 <= s0) & (u.orig_year.to_numpy()[unit] <= s0 // 12)
    unit, age = unit[seen], age[seen]
    frame = pd.DataFrame(
        {
            "ev": c.events[unit, age, cause],
            "log_active": np.log(c.active[unit, age]),
            "age": age,
            "state": u.state_code.to_numpy()[unit],
            "vintage": u.orig_year.to_numpy()[unit].astype(int),
            **{k: u[k].to_numpy()[unit] for k in STATIC},
            **dynamic(c, unit, age),
        }
    ).dropna()
    effects = "age + state" + (" + vintage" if vintage else "")
    model = pf.fepois(f"ev ~ {' + '.join(X)} | {effects}", data=frame, offset="log_active")
    fixed = {
        name.removeprefix("C(").removesuffix(")"): {
            str(int(float(k))): float(v) for k, v in levels.items()
        }
        for name, levels in model.fixef().items()
    }
    return model.coef().reindex(X), fixed


def hazards(
    c: Cells,
    unit: np.ndarray,
    h: int,
    coef: pd.Series,
    fixed: dict[str, dict[str, float]],
    s0: int | None,
) -> np.ndarray:
    """Fitted hazard [unit, age 0..h] of one cause. A level with no events at S has hazard 0 (its maximum-likelihood
    value), except a vintage: it takes the effect of the latest older vintage that has one."""
    u = c.units
    grid_u, grid_a = np.repeat(unit, h + 1), np.tile(np.arange(h + 1), len(unit))
    x = {k: u[k].to_numpy()[grid_u] for k in STATIC} | dynamic(c, grid_u, grid_a, s0)
    eta = sum(float(coef[k]) * x[k] for k in X)
    eta = eta + np.array([fixed["age"].get(str(a), -np.inf) for a in range(h + 1)])[grid_a]
    by_state = np.array(
        [fixed["state"].get(str(s), -np.inf) for s in range(int(u.state_code.max()) + 1)]
    )
    eta = eta + by_state[u.state_code.to_numpy()[grid_u]]
    if "vintage" in fixed:
        known = sorted(int(k) for k in fixed["vintage"])
        effect = {
            y: fixed["vintage"][str(max((k for k in known if k <= y), default=known[0]))]
            for y in range(1999, 2020)
        }
        eta = eta + np.array([effect[int(y)] for y in u.orig_year.to_numpy()[grid_u]])
    return np.asarray(np.exp(eta), dtype=float).reshape(len(unit), h + 1)


def cohort_estimate(
    c: Cells, young: np.ndarray, year: int, h: int, s0: int, lam: tuple[np.ndarray, ...]
) -> float:
    """`lam` = hazards of default and prepayment for the cells `young`, ages 0..h_max."""
    u = c.units
    known = ~np.isnan(c.macro["hpi_yoy"]).all(axis=1) & ~np.isnan(c.macro["unemp"]).all(axis=1)
    pick = (
        (u.orig_year.to_numpy()[young] == year)
        & (u.fp_mi.to_numpy()[young] <= s0)
        & known[u.state_code.to_numpy()[young]]  # as the frozen plug-in: states with covariates
    )
    unit = young[pick]
    a_u = np.clip(s0 - u.fp_mi.to_numpy()[unit] + 1, 0, h)
    default, prepay = (np.nan_to_num(x[pick][:, : h + 1]) for x in lam)
    future = np.arange(h + 1)[None, :] > a_u[:, None]
    total = np.where(future, default + prepay, 0.0)
    before = np.exp(-(np.cumsum(total, axis=1) - total))
    with np.errstate(divide="ignore", invalid="ignore"):
        share = np.where(total > 0, default / total, 0.0)
    ahead = (before * (1 - np.exp(-total)) * share).sum(axis=1)
    cif = c.aj[unit, a_u] + c.surv[unit, a_u] * ahead
    n = u.n.to_numpy()[unit]
    return float((n * cif).sum() / n.sum())


def main() -> None:
    """One checkpoint per snapshot in artifacts/parts, so a stopped run continues where it was."""
    parts = HERE / "artifacts" / "parts"
    parts.mkdir(parents=True, exist_ok=True)
    c = build()
    self_check(c)
    log.info("cells: %d units, %d loans", len(c.units), int(c.units.n.sum()))
    pairs = pd.read_parquet(PAIRS)
    for year, g in pairs.groupby("snapshot"):
        if (parts / f"{year}.json").exists():
            continue
        s0 = int(year) * 12 + 11
        h_max = int(g.h.max())
        young = np.nonzero(c.units.orig_year.isin(g.cohort).to_numpy())[0]
        rows, coefficients = [], {}
        for vintage in (False, True):
            name = "hazard_vintage" if vintage else "hazard"
            fits = [fit(c, s0, k, vintage=vintage) for k in range(2)]
            coefficients[name] = {
                cause: f[0].round(4).to_dict() for cause, f in zip(CAUSES, fits, strict=True)
            }
            for path, forecast in (("real", None), ("fcst", s0)):
                lam = tuple(hazards(c, young, h_max, f[0], f[1], forecast) for f in fits)
                rows += [
                    {
                        "snapshot": int(year),
                        "cohort": r.cohort,
                        "method": f"{name}_{path}",
                        "estimate": cohort_estimate(c, young, int(r.cohort), int(r.h), s0, lam),
                    }
                    for r in g.itertuples()
                ]
            log.info("%d %s done", year, name)
        pd.DataFrame(rows).to_parquet(parts / f"{year}.parquet")
        (parts / f"{year}.json").write_text(json.dumps(coefficients, indent=1))
    years = sorted(int(y) for y in pairs.snapshot.unique())
    wide = pd.concat(pd.read_parquet(parts / f"{y}.parquet") for y in years).pivot(
        index=["snapshot", "cohort"], columns="method", values="estimate"
    )
    out = pairs.merge(wide.reset_index(), on=["snapshot", "cohort"], validate="one_to_one")
    out.to_parquet(HERE / "artifacts" / "pairs_hazard.parquet")
    coefficients = {
        f"{y}_{name}": value
        for y in years
        for name, value in json.loads((parts / f"{y}.json").read_text()).items()
    }
    (HERE / "artifacts" / "coefficients.json").write_text(json.dumps(coefficients, indent=1))


def self_check(c: Cells) -> None:
    """With each cell's own empirical hazards in place of the fitted ones, the estimate at an early snapshot must be
    the cell-weighted Aalen-Johansen incidence at the horizon."""
    year, s0 = 2005, 2008 * 12 + 11
    u = c.units
    young = np.nonzero((u.orig_year == year).to_numpy())[0]
    exits = c.events[young].sum(axis=2)
    gone = np.divide(exits, c.active[young], out=np.zeros(exits.shape), where=c.active[young] > 0)
    total = -np.log(np.maximum(1 - gone, 1e-12))
    lam = tuple(
        total * np.divide(c.events[young, :, k], exits, out=np.zeros(exits.shape), where=exits > 0)
        for k in range(2)
    )
    known = ~np.isnan(c.macro["hpi_yoy"]).all(axis=1) & ~np.isnan(c.macro["unemp"]).all(axis=1)
    ok = known[u.state_code.to_numpy()[young]] & (u.fp_mi.to_numpy()[young] <= s0)
    n = u.n.to_numpy()[young][ok]
    direct = float((n * c.aj[young][ok][:, H]).sum() / n.sum())
    assert abs(cohort_estimate(c, young, year, H, s0, lam) - direct) < 1e-9, "product integral"


if __name__ == "__main__":
    if os.environ.get("PYTHONHASHSEED") != "0":
        raise SystemExit("run with PYTHONHASHSEED=0 (the fixed-effects fit depends on set order)")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    main()
