"""CONFIRM2 §4: the competing-risks hazard model, M1 of experiment 0066 on A and B (PRE-REGISTERED).

DATASET: the units of data.py; the scored pairs of backtest.py (artifacts/<name>/pairs.parquet).
For each cause k (the event and the competing exit of §2/§3), fitted at the December snapshot S on the unit-months
that S shows (entry month + age <= S):
    events_k ~ Poisson(active units x exp(age effect + group effect + entry-year effect + x'beta_k))
a free monthly age baseline (ages above 96 months pooled into one level), group effects (A: state, B: district),
entry-year effects and x = the calendar covariates of the data set (z-scored): A hpi_yoy, unemp; B log_judges, unemp.
Cells are entry month x group x age. The functions `fit`, `year_effect` and `linear` are the code of 0066 (imported);
`table`, `path` and `cohort_estimate` are its code with the data set's covariates, and the product integral runs to
each pair's own truth horizon h (logged in the paper's table of deviations).
Estimate for a scored cohort: per cell, the incidence by S + the share active at S x the probability of the event
between the age at S and h, from the discrete product integral of the two fitted monthly hazards. Paths after S:
`real` = the realised covariates; `fcst` = each group's mean of the 12 months up to S.
A cause with no event in the cells gets the maximum-likelihood hazard 0 (the rule of 0066 for a level with no event):
this is the exit cause of A at the snapshots 2000-2004, where no payoff is dated yet (deviation P1). POST HOC
variant `hazard_lt` (A, snapshots 2005-2018): the same model fitted only on the calendar months after 2005-07.
Self-check: with each cell's own empirical hazards, every uncut pair's estimate equals its stored truth.
Usage: PYTHONHASHSEED=0 uv run python hazard_m1.py <sba|civil|civil_broad> -> artifacts/<name>/hazard.parquet,
artifacts/<name>/hazard_coefficients.json (one checkpoint per snapshot in artifacts/<name>/parts)
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

import data

HERE: Final = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "0066-patent-hazard-baseline"))
import hazard as m66  # fit, year_effect, linear of the patent model M1

log = logging.getLogger(__name__)
POOL: Final = m66.POOL
CAUSES: Final = ("event", "exit")
EFFECTS: Final = ["age_level", "tc_code", "year"]
# POST HOC variant `hazard_lt` (A only): every SBA payoff before 2005-06 carries the placeholder date 2005-05-31 or
# 2005-07-31, so the risk sets are right only from 2005-08: the model is fitted on the calendar months after 2005-07
LEFT: Final = {"sba": 2005 * 12 + 6}


@dataclass(frozen=True, slots=True)
class Cells:
    units: (
        pd.DataFrame
    )  # one row per cell: year, fm (0-based entry month), tc_code (the group, named as in 0066), n
    active: np.ndarray  # [unit, age] active at the start of the age
    events: np.ndarray  # [unit, age, cause]
    aj: np.ndarray  # [unit, age] event incidence up to the age (Aalen-Johansen)
    surv: np.ndarray  # [unit, age] share active after the age
    groups: list[str]  # tc_code -> group


def build(u: pd.DataFrame) -> Cells:
    """0066 `build` with entry month x group units."""
    groups = sorted(u.group.unique())
    s = u.assign(fm=u.om - 1, tc_code=pd.Categorical(u.group, categories=groups).codes)
    grouped = s.groupby(["year", "fm", "tc_code"])
    uid = grouped.ngroup().to_numpy()
    units = grouped.size().rename("n").reset_index()
    n_units, amax = len(units), int(s.event_age_clean.max())
    kind = s.event_type_clean.map({"default": 0, "prepay": 1, "censored": 2}).to_numpy()
    counts = np.bincount(
        (uid * (amax + 1) + s.event_age_clean.to_numpy()) * 3 + kind,
        minlength=n_units * (amax + 1) * 3,
    ).reshape(n_units, amax + 1, 3)
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
    return Cells(units, active, counts[:, :, :2].astype(float), aj, surv, groups)


@dataclass(frozen=True, slots=True)
class Covariates:
    raw: list[str]
    values: dict[str, np.ndarray]  # [tc_code, month - mi0], raw
    mi0: int
    stats: dict[str, tuple[float, float]]


def covariates(name: str, c: Cells) -> Covariates:
    cov = data.covariates(name)
    cov = cov[cov.group.isin(c.groups)]
    mi0, mi1 = int(cov.mi.min()), int(cov.mi.max())
    raw = data.RAW[name]
    code = pd.Categorical(cov.group, categories=c.groups).codes
    values = {}
    for r in raw:
        a = np.full((len(c.groups), mi1 - mi0 + 1), np.nan)
        a[code, cov.mi.to_numpy() - mi0] = cov[r].to_numpy()
        values[r] = a
    out = Covariates(raw, values, mi0, {r: (0.0, 1.0) for r in raw})
    # z-scores over the unit-month cells of the data (scale only: the fit does not depend on it)
    base = table(c, 10**6, out)
    return Covariates(
        raw, values, mi0, {r: (float(base[r].mean()), float(base[r].std())) for r in raw}
    )


def table(c: Cells, s0: int, cov: Covariates, *, after: int | None = None) -> pd.DataFrame:
    """Cells observable at S (0066 `table` without features); `after` = only calendar months after it."""
    u = c.units
    unit, age = np.nonzero(c.active[:, 1:] > 0)
    age = age + 1
    fm = u.fm.to_numpy()[unit]
    seen = (fm + age <= s0) & (u.year.to_numpy()[unit] <= s0 // 12)
    if after is not None:
        seen &= fm + age > after
    unit, age, fm = unit[seen], age[seen], fm[seen]
    frame = pd.DataFrame(
        {
            "event": c.events[unit, age, 0],
            "exit": c.events[unit, age, 1],
            "active": c.active[unit, age],
            "age": age,
            "year": u.year.to_numpy()[unit],
            "tc_code": u.tc_code.to_numpy()[unit],
        }
    )
    mi = fm + age
    for r in cov.raw:
        frame[r] = (cov.values[r][frame.tc_code.to_numpy(), mi - cov.mi0] - cov.stats[r][0]) / (
            cov.stats[r][1]
        )
    frame = frame.dropna(subset=cov.raw)
    frame["age_level"] = np.minimum(frame.age, POOL + 1)
    frame["log_active"] = np.log(frame.active)
    return frame


def path(
    c: Cells, unit: np.ndarray, s0: int, cov: Covariates, h: int, *, forecast: bool
) -> pd.DataFrame:
    """Covariates of the cells (unit, age 0..h) of the scored cohorts (0066 `path`)."""
    u = c.units
    grid_u, grid_a = np.repeat(unit, h + 1), np.tile(np.arange(h + 1), len(unit))
    g = u.tc_code.to_numpy()[grid_u]
    mi = u.fm.to_numpy()[grid_u] + grid_a
    out = {
        "age_level": np.minimum(grid_a, POOL + 1),
        "tc_code": g,
        "year": u.year.to_numpy()[grid_u],
    }
    for r in cov.raw:
        a = cov.values[r]
        x = a[g, np.clip(mi, cov.mi0, cov.mi0 + a.shape[1] - 1) - cov.mi0]
        if forecast:
            level = np.nanmean(a[:, s0 - 11 - cov.mi0 : s0 + 1 - cov.mi0], axis=1)
            x = np.where(mi > s0, level[g], x)
        out[r] = (x - cov.stats[r][0]) / cov.stats[r][1]
    return pd.DataFrame(out)


def cohort_estimate(
    c: Cells, unit: np.ndarray, s0: int, lam: tuple[np.ndarray, np.ndarray], h: int
) -> float:
    """`lam` = monthly hazards of the event and the exit [unit, age 0..h]. The formula of 0066 `cohort_estimate`,
    to the horizon h."""
    u = c.units
    a_u = np.clip(s0 - u.fm.to_numpy()[unit], 0, h)
    event, other = lam
    future = np.arange(h + 1)[None, :] > a_u[:, None]
    total = np.where(future, event + other, 0.0)
    before = np.exp(-(np.cumsum(total, axis=1) - total))
    with np.errstate(divide="ignore", invalid="ignore"):
        share = np.where(total > 0, event / total, 0.0)
    ahead = (before * (1 - np.exp(-total)) * share).sum(axis=1)
    cif = c.aj[unit, a_u] + c.surv[unit, a_u] * ahead
    n = u.n.to_numpy()[unit]
    return float((n * cif).sum() / n.sum())


def self_check(c: Cells, pairs: pd.DataFrame) -> None:
    """With each cell's own empirical hazards in place of the fitted ones, every estimate is the pair's truth. Only
    the uncut pairs: there no unit is censored before h, so the cohort's truth is the mean of its units' incidences."""
    for r in pairs[~pairs.truncated].itertuples():
        h = int(r.h)
        unit = np.nonzero((c.units.year == r.cohort).to_numpy())[0]
        exits = c.events[unit, : h + 1].sum(axis=2)
        active = c.active[unit, : h + 1]
        gone = np.divide(exits, active, out=np.zeros(exits.shape), where=active > 0)
        total = -np.log(np.maximum(1 - gone, 1e-300))
        share = np.divide(
            c.events[unit, : h + 1, 0], exits, out=np.zeros(exits.shape), where=exits > 0
        )
        s0 = int(r.snapshot) * 12 + 11
        est = cohort_estimate(c, unit, s0, (total * share, total * (1 - share)), h)
        assert abs(est - r.truth) < 1e-9, f"{r.cohort} at {r.snapshot}: {est} vs truth {r.truth}"


def snapshot(
    c: Cells, year: int, pairs: pd.DataFrame, cov: Covariates, name: str
) -> tuple[list[dict[str, object]], dict[str, object]]:
    s0, first = year * 12 + 11, int(c.units.year.min())
    variants: dict[str, int | None] = {"hazard": None}
    if name in LEFT and s0 > LEFT[name]:
        variants["hazard_lt"] = LEFT[name]
    rows: list[dict[str, object]] = []
    info: dict[str, object] = {}
    for variant, after in variants.items():
        frame = table(c, s0, cov, after=after)
        start = time.perf_counter()
        # a cause with no event in the cells has the maximum-likelihood hazard 0 (the rule of 0066 for a level)
        fits = [
            m66.fit(frame, k, list(cov.raw), EFFECTS) if frame[k].sum() > 0 else None
            for k in CAUSES
        ]
        seconds = time.perf_counter() - start
        for r in pairs.itertuples():
            h = int(r.h)
            unit = np.nonzero((c.units.year == r.cohort).to_numpy())[0]
            for p in ("real", "fcst"):
                x = path(c, unit, s0, cov, h, forecast=p == "fcst")
                lam = [_hazard(f, x, (len(unit), h + 1)) for f in fits]
                rows.append(
                    {
                        "snapshot": year,
                        "cohort": float(r.cohort),
                        "method": f"{variant}_{p}",
                        "estimate": cohort_estimate(c, unit, s0, (lam[0], lam[1]), h),
                    }
                )
        info[variant] = {
            "rows": len(frame),
            "seconds": round(seconds, 1),
            **{k: _summary(f, first, year) for k, f in zip(CAUSES, fits, strict=True)},
        }
    return rows, info


def main(name: str) -> None:
    out = data.ART / name
    parts = out / "parts"
    parts.mkdir(parents=True, exist_ok=True)
    c = build(data.units(name))
    pairs = pd.read_parquet(out / "pairs.parquet")
    self_check(c, pairs)
    log.info("%s: %d cells, %d units; self-check passed", name, len(c.units), int(c.units.n.sum()))
    cov = covariates(name, c)
    for year, g in pairs.groupby("snapshot"):
        if (parts / f"hazard_{year}.json").exists():
            continue
        rows, info = snapshot(c, int(year), g, cov, name)
        pd.DataFrame(rows).to_parquet(parts / f"hazard_{year}.parquet")
        (parts / f"hazard_{year}.json").write_text(json.dumps(info, indent=1))
        log.info("%s %d: %s", name, year, {k: v["seconds"] for k, v in info.items()})
    years = sorted(int(y) for y in pairs.snapshot.unique())
    wide = pd.concat(pd.read_parquet(parts / f"hazard_{y}.parquet") for y in years).pivot(
        index=["snapshot", "cohort"], columns="method", values="estimate"
    )
    wide.reset_index().to_parquet(out / "hazard.parquet")
    info = {str(y): json.loads((parts / f"hazard_{y}.json").read_text()) for y in years}
    (out / "hazard_coefficients.json").write_text(
        json.dumps({"z_stats": cov.stats, "by_snapshot": info}, indent=1)
    )


def _hazard(f: m66.Fit | None, x: pd.DataFrame, shape: tuple[int, int]) -> np.ndarray:
    return np.zeros(shape) if f is None else np.exp(m66.linear(f, x)).reshape(shape)


def _summary(f: m66.Fit | None, first: int, year: int) -> dict[str, object]:
    if f is None:
        return {"no_events": True}
    return {
        "coef": {k: round(v, 5) for k, v in f.coef.items()},
        "se_iid": {k: round(v, 5) for k, v in f.se_iid.items()},
        "se_group": {k: round(v, 5) for k, v in f.se_centre.items()},
        "year_effects": {str(k): round(v, 5) for k, v in sorted(f.fixed["year"].items())},
        "years_without_effect": m66.year_effect(f, np.arange(first, year + 1))[1],
    }


if __name__ == "__main__":
    if os.environ.get("PYTHONHASHSEED") != "0":
        raise SystemExit("run with PYTHONHASHSEED=0 (the fixed-effects fit depends on set order)")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    main(sys.argv[1])
