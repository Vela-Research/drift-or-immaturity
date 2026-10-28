"""Cause-specific hazard model for grant and abandonment of patent applications (POST HOC; the patent version of
experiment 0032; spec: the patent hazard specification).

DATASET: USPTO PatEx, release folder "2022" (data of 2023-06), application_data only; the sample of experiment 0020
(100,000 nonprovisional utility applications per filing year 2001-2019, 8 technology centres, seed 20261004). The
pairs, truth (72-month grant share) and scoring are those of experiment 0031 (`pairs_patents_main.parquet`).

Model, for each cause k in {grant, abandonment}, fitted at the December snapshot S on the application-months that S
shows (filing month + age <= S):
    events_k ~ Poisson(pending applications x exp(age effect + centre effect [+ filing-year effect] + x'beta_k))
a free monthly age baseline (ages above 96 months pooled into one level), technology-centre effects, and
x = log pending inventory of the centre in that month (z-scored) and the post-Alice indicator, defined as in the
calendar step of 0021 (the indicator enters only from the snapshot 2014, as there).
    M1   the terms above with filing-year effects
    M2   M1 without filing-year effects
    M3   M1 + USPC class group effects + the AIA first-inventor-to-file indicator (fields fixed at filing)
    M1own  M1 with the pending inventory that leaves out the filing years S-4 .. S-1 (as 0031 `no_own`)
Cells are filing month x centre (x class group x AIA) x age: the spec says filing quarter; the month cells make the cut
at S and the inventory of each cell exact, with the same parameters.
Estimate for a scored filing year: per cell, the share granted by S + the share pending at S x the probability of a
grant between the age at S and 72 months, from the discrete product integral of the two fitted monthly hazards (the
formula of 0032). Paths after S: `real` = actual inventory and Alice; `fcst` = inventory of each centre at its mean of
the 12 months up to S, Alice at its value at S.
Usage: PYTHONHASHSEED=0 uv run python hazard.py -> artifacts/pairs_hazard.parquet, artifacts/coefficients.json
(one checkpoint per snapshot in artifacts/parts)
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd
import pyfixest as pf

HERE: Final = Path(__file__).resolve().parent
EXP: Final = HERE.parent
sys.path[:0] = [str(EXP / "0020-patex-feasibility"), str(EXP / "0012-freddie-200k-harp-correction")]
import check  # the PatEx sample, the outcomes and the inventory covariate of 0020

log = logging.getLogger(__name__)
PAIRS: Final = EXP / "0031-round3-backtests" / "artifacts" / "pairs_patents_main.parquet"
FEATURES: Final = HERE / "artifacts" / "features.parquet"
H: Final = 72
POOL: Final = 96  # ages above this share one level of the age baseline
MIN_CLASS: Final = 5_000  # a USPC class with fewer sample applications goes into "other"
MI0: Final = 2001 * 12
AMAX: Final = check.CUTOFF - MI0
CAUSES: Final = ("grant", "abandon")
MODELS: Final = {  # name -> (filing-year effects, features, inventory without the scored years)
    "m1": (True, False, False),
    "m2": (False, False, False),
    "m3": (True, True, False),
    "m1own": (True, False, True),
}


@dataclass(frozen=True, slots=True)
class Cells:
    units: pd.DataFrame  # one row per cell: year, fm, tc_code, cls_code, aia, n
    active: np.ndarray  # [unit, age] pending at the start of the age
    events: np.ndarray  # [unit, age, cause]
    aj: np.ndarray  # [unit, age] share granted up to the age (Aalen-Johansen)
    surv: np.ndarray  # [unit, age] share pending after the age
    alice_tc: np.ndarray  # [tc_code] True for the centres 21 and 36


def sample() -> tuple[pd.DataFrame, pd.DataFrame]:
    """All utility applications (for the inventory) and the sample of 0020 with the M3 fields."""
    applications = check.load()
    pool = applications[applications.year.isin(check.YEARS)]
    s = pool.groupby("year").sample(n=check.N_PER_YEAR, random_state=check.SEED)
    number = pd.read_parquet(check.SOURCE, columns=["application_number"]).application_number
    f = pd.read_parquet(FEATURES).drop_duplicates("application_number")
    s = s.assign(application_number=number.loc[s.index].to_numpy()).merge(
        f, on="application_number", how="left", validate="many_to_one"
    )
    assert s.uspc_class.notna().all(), "every sampled application has a USPC class"
    size = s.uspc_class.value_counts()
    s["cls"] = s.uspc_class.where(s.uspc_class.isin(size[size >= MIN_CLASS].index), "other")
    # "Other" (1,369 applications, nearly all abandoned) is not an AIA status at filing: it counts as not AIA
    s["aia"] = (s.aia_first_to_file == "true").astype(float)
    return applications, s


def build(s: pd.DataFrame) -> Cells:
    s = s.assign(
        tc_code=pd.factorize(s.tc, sort=True)[0], cls_code=pd.factorize(s.cls, sort=True)[0]
    )
    keys = ["year", "fm", "tc_code", "cls_code", "aia"]
    grouped = s.groupby(keys)
    uid = grouped.ngroup().to_numpy()
    units = grouped.size().rename("n").reset_index()
    n_units = len(units)
    kind = s.kind.map({"default": 0, "prepay": 1, "censored": 2}).to_numpy()
    dur = s.dur.to_numpy()
    assert dur.max() <= AMAX
    counts = np.bincount(
        (uid * (AMAX + 1) + dur) * 3 + kind, minlength=n_units * (AMAX + 1) * 3
    ).reshape(n_units, AMAX + 1, 3)
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
    centres = sorted(s.tc.unique())
    alice_tc = np.array([tc in ("21", "36") for tc in centres])
    return Cells(units, active, counts[:, :, :2].astype(float), aj, surv, alice_tc)


def inventory_array(inv: pd.DataFrame) -> np.ndarray:
    """[tc_code, month - MI0] log pending inventory (centres in sorted order, as tc_code)."""
    wide = inv.pivot(index="tc", columns="mi", values="log_inventory").sort_index()
    assert int(wide.columns[0]) == MI0
    return wide.to_numpy()


def table(
    c: Cells, s0: int, inv: np.ndarray, stats: tuple[float, float], *, features: bool
) -> pd.DataFrame:
    """Cells observable at S. Without features, the cells are summed over class group and AIA."""
    u = c.units
    unit, age = np.nonzero(c.active[:, 1:] > 0)
    age = age + 1
    fm = u.fm.to_numpy()[unit]
    seen = (fm + age <= s0) & (u.year.to_numpy()[unit] <= s0 // 12)
    unit, age, fm = unit[seen], age[seen], fm[seen]
    frame = pd.DataFrame(
        {
            "grant": c.events[unit, age, 0],
            "abandon": c.events[unit, age, 1],
            "active": c.active[unit, age],
            "age": age,
            "fm": fm,
            "year": u.year.to_numpy()[unit],
            "tc_code": u.tc_code.to_numpy()[unit],
            "cls_code": u.cls_code.to_numpy()[unit],
            "aia": u.aia.to_numpy()[unit],
        }
    )
    if not features:
        frame = frame.groupby(["year", "fm", "tc_code", "age"], as_index=False)[
            ["grant", "abandon", "active"]
        ].sum()
    mi = frame.fm.to_numpy() + frame.age.to_numpy()
    tc = frame.tc_code.to_numpy()
    frame["inv_z"] = (inv[tc, mi - MI0] - stats[0]) / stats[1]
    frame["alice"] = (c.alice_tc[tc] & (mi >= check.ALICE)).astype(float)
    frame["age_level"] = np.minimum(frame.age, POOL + 1)
    frame["log_active"] = np.log(frame.active)
    return frame


@dataclass(frozen=True, slots=True)
class Fit:
    coef: dict[str, float]
    se_iid: dict[str, float]
    se_centre: dict[str, float]  # clustered by technology centre (8 clusters)
    fixed: dict[str, dict[int, float]]


def fit(frame: pd.DataFrame, cause: str, terms: list[str], effects: list[str]) -> Fit:
    model = pf.fepois(
        f"{cause} ~ {' + '.join(terms)} | {' + '.join(effects)}",
        data=frame,
        offset="log_active",
        vcov="iid",
    )
    coef, se_iid = model.coef().to_dict(), model.se().to_dict()
    fixed = {
        name.removeprefix("C(").removesuffix(")"): {
            int(float(k)): float(v) for k, v in levels.items()
        }
        for name, levels in model.fixef().items()
    }
    # pyfixest leaves out the reference level (effect 0) of each effect after the first; a level with no event
    # of the cause is dropped by the fit (its maximum-likelihood hazard is 0) and stays out
    for name, levels in fixed.items():
        with_events = frame.loc[frame[cause] > 0, name].unique()
        reference = [int(k) for k in with_events if int(k) not in levels]
        assert len(reference) <= (name != effects[0]), f"{name}: levels left out {reference}"
        levels.update(dict.fromkeys(reference, 0.0))
    model.vcov({"CRV1": "tc_code"})
    out = Fit(coef, se_iid, model.se().to_dict(), fixed)
    # the score equations of a Poisson model with effects: fitted events = observed events
    fitted = np.exp(linear(out, frame) + frame.log_active.to_numpy()).sum()
    gap = abs(fitted / frame[cause].sum() - 1)
    assert gap < 1e-4, f"{cause}: fitted events off by {gap:.2e} (effects not rebuilt right)"
    return out


def year_effect(f: Fit, years: np.ndarray) -> tuple[np.ndarray, list[int]]:
    """A filing year with no estimable effect takes the effect of the nearest older year that has one."""
    known = sorted(f.fixed["year"])
    missing = sorted({int(y) for y in years} - set(known))
    effect = {
        y: f.fixed["year"][max((k for k in known if k <= y), default=known[0])]
        for y in set(years.tolist())
    }
    return np.array([effect[int(y)] for y in years]), missing


def linear(f: Fit, x: pd.DataFrame) -> np.ndarray:
    eta = sum(f.coef[k] * x[k].to_numpy() for k in f.coef)
    for name, levels in f.fixed.items():
        if name == "year":
            eta = eta + year_effect(f, x.year.to_numpy())[0]
        else:
            eta = (
                eta + x[name].map(levels).fillna(-np.inf).to_numpy()
            )  # a level with no event: hazard 0
    return np.asarray(eta, dtype=float)


def path(
    c: Cells,
    unit: np.ndarray,
    s0: int,
    inv: np.ndarray,
    stats: tuple[float, float],
    *,
    forecast: bool,
) -> pd.DataFrame:
    """Covariates of the cells (unit, age 0..H) of the scored filing years."""
    u = c.units
    grid_u, grid_a = np.repeat(unit, H + 1), np.tile(np.arange(H + 1), len(unit))
    tc = u.tc_code.to_numpy()[grid_u]
    mi = u.fm.to_numpy()[grid_u] + grid_a
    log_inv = inv[tc, np.clip(mi, MI0, MI0 + inv.shape[1] - 1) - MI0]
    alice = c.alice_tc[tc] & (mi >= check.ALICE)
    if forecast:
        future = mi > s0
        log_inv = np.where(future, inv[:, s0 - 11 - MI0 : s0 + 1 - MI0].mean(axis=1)[tc], log_inv)
        alice = np.where(future, c.alice_tc[tc] & (s0 >= check.ALICE), alice)
    return pd.DataFrame(
        {
            "inv_z": (log_inv - stats[0]) / stats[1],
            "alice": alice.astype(float),
            "aia": u.aia.to_numpy()[grid_u],
            "age_level": np.minimum(grid_a, POOL + 1),
            "tc_code": tc,
            "cls_code": u.cls_code.to_numpy()[grid_u],
            "year": u.year.to_numpy()[grid_u],
        }
    )


def cohort_estimate(
    c: Cells, unit: np.ndarray, s0: int, lam: tuple[np.ndarray, np.ndarray]
) -> float:
    """`lam` = monthly hazards of grant and abandonment [unit, age 0..H]. The formula of 0032 `cohort_estimate`."""
    u = c.units
    a_u = np.clip(s0 - u.fm.to_numpy()[unit], 0, H)
    grant, abandon = lam
    future = np.arange(H + 1)[None, :] > a_u[:, None]
    total = np.where(future, grant + abandon, 0.0)
    before = np.exp(-(np.cumsum(total, axis=1) - total))
    with np.errstate(divide="ignore", invalid="ignore"):
        share = np.where(total > 0, grant / total, 0.0)
    ahead = (before * (1 - np.exp(-total)) * share).sum(axis=1)
    cif = c.aj[unit, a_u] + c.surv[unit, a_u] * ahead
    n = u.n.to_numpy()[unit]
    return float((n * cif).sum() / n.sum())


def self_check(c: Cells, pairs: pd.DataFrame) -> None:
    """With each cell's own empirical hazards in place of the fitted ones, every estimate must be the cohort's
    72-month grant share, which is the stored truth of 0031."""
    for r in pairs.itertuples():
        unit = np.nonzero((c.units.year == r.cohort).to_numpy())[0]
        exits = c.events[unit, : H + 1].sum(axis=2)
        active = c.active[unit, : H + 1]
        gone = np.divide(exits, active, out=np.zeros(exits.shape), where=active > 0)
        total = -np.log(np.maximum(1 - gone, 1e-300))
        share = np.divide(
            c.events[unit, : H + 1, 0], exits, out=np.zeros(exits.shape), where=exits > 0
        )
        est = cohort_estimate(
            c, unit, int(r.snapshot) * 12 + 11, (total * share, total * (1 - share))
        )
        assert abs(est - r.truth) < 1e-9, f"{r.cohort} at {r.snapshot}: {est} vs truth {r.truth}"


def snapshot(
    c: Cells,
    year: int,
    cohorts: list[int],
    inventories: dict[str, np.ndarray],
    stats: tuple[float, float],
) -> tuple[list[dict[str, object]], dict[str, object]]:
    s0 = year * 12 + 11
    young = np.nonzero(c.units.year.isin(cohorts).to_numpy())[0]
    rows: list[dict[str, object]] = []
    info: dict[str, object] = {}
    tables: dict[tuple[str, bool], pd.DataFrame] = {}
    for name, (year_fe, features, own) in MODELS.items():
        inv = inventories["own" if own else "all"]
        key = ("own" if own else "all", features)
        if key not in tables:
            tables[key] = table(c, s0, inv, stats, features=features)
        frame = tables[key]
        terms = ["inv_z"] + (["alice"] if year >= 2014 else [])
        if features and frame.aia.nunique() > 1:
            terms.append("aia")
        effects = (
            ["age_level", "tc_code"]
            + (["year"] if year_fe else [])
            + (["cls_code"] if features else [])
        )
        start = time.perf_counter()
        fits = [fit(frame, cause, terms, effects) for cause in CAUSES]
        seconds = time.perf_counter() - start
        x = {p: path(c, young, s0, inv, stats, forecast=p == "fcst") for p in ("real", "fcst")}
        for p, cov in x.items():
            lam = tuple(np.exp(linear(f, cov)).reshape(len(young), H + 1) for f in fits)
            for cohort in cohorts:
                pick = c.units.year.to_numpy()[young] == cohort
                rows.append(
                    {
                        "snapshot": year,
                        "cohort": float(cohort),
                        "method": f"{name}_{p}",
                        "estimate": cohort_estimate(
                            c, young[pick], s0, (lam[0][pick], lam[1][pick])
                        ),
                    }
                )
        info[name] = {
            "rows": len(frame),
            "seconds": round(seconds, 1),
            **{
                cause: {
                    "coef": {k: round(v, 5) for k, v in f.coef.items()},
                    "se_iid": {k: round(v, 5) for k, v in f.se_iid.items()},
                    "se_centre": {k: round(v, 5) for k, v in f.se_centre.items()},
                    "year_effects": {
                        str(k): round(v, 5) for k, v in sorted(f.fixed.get("year", {}).items())
                    },
                    "years_without_effect": year_effect(f, np.arange(2001, year + 1))[1]
                    if year_fe
                    else [],
                }
                for cause, f in zip(CAUSES, fits, strict=True)
            },
        }
        log.info("%d %s: %d rows, %.0f s", year, name, len(frame), seconds)
    return rows, info


def main() -> None:
    parts = HERE / "artifacts" / "parts"
    parts.mkdir(parents=True, exist_ok=True)
    applications, s = sample()
    c = build(s)
    pairs = pd.read_parquet(PAIRS)
    self_check(c, pairs)
    log.info(
        "cells: %d units, %d applications; self-check passed", len(c.units), int(c.units.n.sum())
    )
    inv_all = inventory_array(check.inventory(applications))
    # z-score of log inventory over the month x centre cells of the sample (scale only: the fit does not depend on it)
    base = table(c, check.CUTOFF, inv_all, (0.0, 1.0), features=False).inv_z
    stats = (float(base.mean()), float(base.std()))
    for year, g in pairs.groupby("snapshot"):
        if (parts / f"{year}.json").exists():
            continue
        cohorts = sorted(int(x) for x in g.cohort)
        others = applications[~applications.year.isin(range(int(year) - 4, int(year)))]
        inventories = {"all": inv_all, "own": inventory_array(check.inventory(others))}
        rows, info = snapshot(c, int(year), cohorts, inventories, stats)
        pd.DataFrame(rows).to_parquet(parts / f"{year}.parquet")
        (parts / f"{year}.json").write_text(json.dumps(info, indent=1))
    years = sorted(int(y) for y in pairs.snapshot.unique())
    wide = pd.concat(pd.read_parquet(parts / f"{y}.parquet") for y in years).pivot(
        index=["snapshot", "cohort"], columns="method", values="estimate"
    )
    out = pairs.merge(wide.reset_index(), on=["snapshot", "cohort"], validate="one_to_one")
    out.to_parquet(HERE / "artifacts" / "pairs_hazard.parquet")
    info = {str(y): json.loads((parts / f"{y}.json").read_text()) for y in years}
    (HERE / "artifacts" / "coefficients.json").write_text(
        json.dumps(
            {"inventory_z_stats": stats, "min_class": MIN_CLASS, "by_snapshot": info}, indent=1
        )
    )


if __name__ == "__main__":
    if os.environ.get("PYTHONHASHSEED") != "0":
        raise SystemExit("run with PYTHONHASHSEED=0 (the fixed-effects fit depends on set order)")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    main()
