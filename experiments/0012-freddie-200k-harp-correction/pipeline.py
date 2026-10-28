"""Frozen-spec pipeline with the POST HOC correction of T2c section 10 (random censoring). DATASET: 200k loans per
vintage, simple random sample of the full Freddie Mac Standard files (experiments/0010).

Copy of experiments/0011/pipeline.py. Two changes, both in the calendar step (`diff` the two files to see them):
  1. the cells come from cells.py: inverse-probability-of-censoring weights in the sub-distribution risk sets;
  2. plug_in: the observed part of each unit is its Aalen-Johansen incidence, not (defaults seen) / (loans).
Nothing else changes: speed model, guards, baselines, evaluation.
Variants: "ipcw" = HARP parents censored (the spec); "noharp" = HARP terminations are prepayments in the truth and in
every method (robustness check).
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd
import pyfixest as pf

from maturity import data as vdata
from maturity import onestage

HERE: Final = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "0005-freddie-backtests"))
import calendar_methods as cal

LOANS_PATH: Final = vdata.DATA_DIR / "processed" / "freddie_200k_loans.parquet"
CELLS_DIR: Final = HERE / "artifacts"  # written by cells.py
VARIANTS: Final = ("ipcw", "noharp")
AMAX: Final = 200  # loan-level curves
CELL_AMAX: Final = 120
G1_THRESHOLD: Final = (
    100.0  # events x smallest identifying-variation share (simulation 0006, guard1_threshold.csv)
)
METHODS: Final = [
    "naive",
    "drop_recent",
    "fixed_horizon",
    "chain",
    "ours_free",
    "cal_chain",
    "ours",
    "ours_uncond",
]
OMEGA_GRID: Final = (0.0, 0.003, 0.01, 0.03, 0.1, 0.3)  # rolling-origin sensitivity (T2 Update 5)
Step1 = tuple[np.ndarray, pd.Series, np.ndarray]


@dataclass
class Data:
    loans: pd.DataFrame
    cells: pd.DataFrame
    stats: dict[str, tuple[float, float]]
    raw: list[str]
    zcols: list[str]


def month_idx(yyyymm: pd.Series | int) -> pd.Series | int:
    return (yyyymm // 100) * 12 + yyyymm % 100


def load(mode: str = "base", variant: str = "ipcw") -> Data:
    """mode 'base' = state HPI growth + state unemployment (confirmatory); 'mtm' adds log mark-to-market LTV."""
    d = pd.read_parquet(LOANS_PATH)
    if variant == "noharp":
        d["event_type_clean"] = d["event_type_clean_raw"]
    d = d[
        d.event_age_clean.notna() & d.first_payment.notna() & d.orig_year.between(1999, 2019)
    ].copy()
    d["om"] = month_idx(d.first_payment.astype(int)) - 1  # origination month, 1-based month index
    d["ev_mi"] = month_idx(d.event_period_clean.astype(int))
    d["year"] = d.orig_year.astype(float)
    d["quarter"] = ((d.om - 1) // 3) / 4.0
    c = pd.read_parquet(
        CELLS_DIR / f"hazard_cells{'_mtm' if mode == 'mtm' else ''}_{variant}_subdist.parquet"
    )
    raw = ["hpi_yoy", "unemp"] + (["log_mtm"] if mode == "mtm" else [])
    unit_keys = ["vintage", "fp_mi", "state"] + (["ltv_band"] if mode == "mtm" else [])
    c["unit"] = c.groupby(unit_keys).ngroup()
    c["year"] = c.vintage.astype(float)
    c["quarter"] = ((c.fp_mi - 1) // 3) / 4.0
    return Data(
        d,
        c,
        {r: (float(c[r].mean()), float(c[r].std())) for r in raw},
        raw,
        [f"{r}_z" for r in raw],
    )


def aj_default_cif(dur: np.ndarray, kind: np.ndarray) -> np.ndarray:
    dur = np.clip(dur, 0, AMAX).astype(int)
    d, p, c = (
        np.bincount(dur[kind == k], minlength=AMAX + 1) for k in ("default", "prepay", "censored")
    )
    risk = np.maximum(len(dur) - np.concatenate([[0], np.cumsum(d + p + c)[:-1]]), 1)
    surv = np.concatenate([[1.0], np.cumprod(1 - (d + p) / risk)[:-1]])
    return np.cumsum(surv * d / risk)


def observed(data: Data, s: int, cohort: str) -> tuple[pd.DataFrame, dict[float, np.ndarray]]:
    """Curves and follow-up per cohort from loans observable at month s (events after s become censoring at s)."""
    d = data.loans
    x = d[(d.om <= s) & (d.orig_year <= (s - 1) // 12)]
    seen = (x.ev_mi <= s).to_numpy()
    dur = np.where(seen, x.event_age_clean.to_numpy(), (s - x.om).to_numpy())
    kind = np.where(seen, x.event_type_clean.to_numpy(), "censored")
    pot = (s - x.om).to_numpy()
    curves, rows = {}, []
    for key, idx in x.groupby(cohort).indices.items():
        curves[float(key)] = aj_default_cif(dur[idx], kind[idx])
        rows.append(
            {
                "cohort": float(key),
                "A": int(np.median(pot[idx])),
                "n": len(idx),
                "events": int((kind[idx] == "default").sum()),
            }
        )
    return pd.DataFrame(rows).sort_values("cohort", ignore_index=True), curves


def truth(
    data: Data, cohort: str, horizon: int, cutoff: int | None = None
) -> dict[float, tuple[float, int]]:
    """Per cohort: (incidence at its truth horizon, that horizon). cutoff = month index after which nothing counts."""
    info, curves = observed(data, cutoff if cutoff is not None else 10**9, cohort)
    out = {}
    for r in info.itertuples():
        h = horizon if cutoff is None else min(horizon, int(r.A))
        out[float(r.cohort)] = (float(curves[float(r.cohort)][h]), h)
    return out


def step1(data: Data, s0: int) -> Step1:
    """Unpenalized fixed-effects Poisson on the cells observable at S; SE clustered by state."""
    c = data.cells[(data.cells.mi <= s0) & (data.cells.vintage <= s0 // 12)]
    d = c[["events", "exposure", "state", *data.zcols]].copy()
    d["cell"] = (c.vintage.to_numpy() - 1999) * CELL_AMAX + (c.age.to_numpy() - 1)
    d["log_exposure"] = np.log(d.exposure)
    m = pf.fepois(
        f"events ~ {' + '.join(data.zcols)} | cell + state",
        data=d,
        offset="log_exposure",
        vcov={"CRV1": "state"},
    )
    tidy, fe = m.tidy(), m.fixef()
    kappa = pd.Series(fe[next(k for k in fe if "state" in k)], dtype=float)
    wts = d.groupby("state").exposure.sum().reindex(kappa.index).fillna(0.0).to_numpy()
    kappa = kappa - float(np.average(kappa.to_numpy(), weights=wts))
    return (
        tidy.loc[data.zcols, "Estimate"].to_numpy(),
        kappa,
        tidy.loc[data.zcols, "Std. Error"].to_numpy(),
    )


def guard1(data: Data, s0: int) -> tuple[dict[str, float], float]:
    """Identifying-variation share per covariate, and events x smallest share (the G1 statistic)."""
    c = data.cells[
        (data.cells.mi <= s0) & (data.cells.vintage <= s0 // 12) & (data.cells.events > 0)
    ]
    cell = pd.factorize(c.vintage.to_numpy() * 1000 + c.age.to_numpy())[0]
    state = pd.factorize(c.state.to_numpy())[0]
    w = c.events.to_numpy(dtype=float)
    shares = {r: _residual_share(c[r].to_numpy(dtype=float), cell, state, w) for r in data.raw}
    return {k: round(v, 4) for k, v in shares.items()}, float(w.sum() * min(shares.values()))


def _residual_share(
    z: np.ndarray, a: np.ndarray, b: np.ndarray, w: np.ndarray, n_iter: int = 60
) -> float:
    r = z - np.average(z, weights=w)
    total = float(np.sum(w * r**2))
    wa, wb = np.bincount(a, weights=w), np.bincount(b, weights=w)
    for _ in range(n_iter):
        r = r - (np.bincount(a, weights=w * r) / np.maximum(wa, 1e-300))[a]
        r = r - (np.bincount(b, weights=w * r) / np.maximum(wb, 1e-300))[b]
    return float(np.sum(w * r**2) / total) if total > 0 else 0.0


def adjusted_curves(data: Data, s0: int, fit: Step1, cohort: str) -> dict[float, np.ndarray]:
    beta, kappa, _ = fit
    c = data.cells[(data.cells.mi <= s0) & (data.cells.vintage <= s0 // 12)]
    w = c.exposure.to_numpy() * np.exp(
        c[data.zcols].to_numpy() @ beta + c.state.map(kappa).fillna(0.0).to_numpy()
    )
    g = (
        pd.DataFrame(
            {"k": c[cohort].to_numpy(), "age": c.age.to_numpy(), "ev": c.events.to_numpy(), "w": w}
        )
        .groupby(["k", "age"])
        .sum()
    )
    out = {}
    for key, t in g.groupby(level=0):
        inc = np.zeros(AMAX + 1)
        inc[t.index.get_level_values(1).to_numpy()] = t.ev.to_numpy() / np.maximum(
            t.w.to_numpy(), 1e-300
        )
        out[float(key)] = np.cumsum(inc)
    return out


def increments(
    curve: np.ndarray, a: int, speed: float, ugrid: np.ndarray, f0: np.ndarray, horizon: int
) -> np.ndarray:
    """inc[k], k = 0..horizon: observed adjusted increments up to age a, fitted ones (shape F0, speed s) after."""
    inc = np.zeros(horizon + 1)
    a = min(a, horizon)
    inc[1 : a + 1] = np.diff(curve[: a + 1])
    f = np.interp(speed * np.arange(a, horizon + 1), ugrid, f0, right=1.0)
    inc[a + 1 :] = curve[a] * np.diff(f) / max(f[0], 1e-9)
    return inc


def plug_in(
    data: Data,
    sub: pd.DataFrame,
    s0: int,
    fit: Step1,
    inc: np.ndarray,
    horizon: int,
    *,
    conditional: bool,
    support: dict[str, tuple[float, float]] | None = None,
) -> tuple[float, float, float]:
    """T2c section 8 step 5 for one cohort. Returns (incidence at horizon, expected future events, of which outside
    the covariate range observed up to S)."""
    beta, kappa, _ = fit
    sub = sub[sub.age <= horizon]
    units, uid = np.unique(sub.unit.to_numpy(), return_inverse=True)
    n_u, age = len(units), sub.age.to_numpy()
    seen = (sub.mi <= s0).to_numpy()
    first = sub[age == 1]
    n = np.zeros(n_u)
    n[np.searchsorted(units, first.unit.to_numpy())] = first.exposure.to_numpy()
    # Aalen-Johansen incidence of each unit up to its age at S (T2c section 10, step 3)
    aj = np.zeros(n_u)
    np.maximum.at(aj, uid[seen], sub.aj.to_numpy()[seen])
    fp = np.zeros(n_u)
    fp[uid] = sub.fp_mi.to_numpy()
    a_u = np.clip(s0 - fp + 1, 0, horizon).astype(int)
    kap = np.zeros(n_u)
    kap[uid] = sub.state.map(kappa).fillna(0.0).to_numpy()
    z = np.full((n_u, horizon + 1, len(data.raw)), np.nan)
    z[uid, age] = sub[data.raw].to_numpy()
    for j in range(len(data.raw)):
        f = pd.DataFrame(z[:, 1:, j]).ffill(axis=1).bfill(axis=1).to_numpy()
        z[:, 1:, j], z[:, 0, j] = f, f[:, 0]
    if not conditional:
        z = _persistence(data, z, a_u, sub, s0, uid, n_u)
    zs = np.stack(
        [(z[:, :, j] - data.stats[r][0]) / data.stats[r][1] for j, r in enumerate(data.raw)], axis=2
    )
    future = np.arange(horizon + 1)[None, :] > a_u[:, None]
    contrib = np.exp(kap[:, None] + zs @ beta) * inc[None, : horizon + 1] * future
    expected = (n * (1 - aj))[:, None] * contrib
    outside = np.zeros_like(future)
    if support is not None:
        for j, r in enumerate(data.raw):
            outside |= (z[:, :, j] < support[r][0]) | (z[:, :, j] > support[r][1])
    cif = 1 - (1 - aj) * np.exp(-contrib.sum(axis=1))
    p = float((n * cif).sum() / max(n.sum(), 1e-12))
    return p, float(expected.sum()), float((expected * outside).sum())


def _persistence(
    data: Data,
    z: np.ndarray,
    a_u: np.ndarray,
    sub: pd.DataFrame,
    s0: int,
    uid: np.ndarray,
    n_u: int,
) -> np.ndarray:
    """After each unit's age at S: state HPI growth and unemployment at their last-12-month means; log MTM LTV (if
    used) follows the HPI path implied by that growth."""
    last = sub[(sub.mi > s0 - 12) & (sub.mi <= s0)]
    if last.empty:
        last = sub[sub.mi <= s0] if (sub.mi <= s0).any() else sub
    by_state = last.groupby("state")[["hpi_yoy", "unemp"]].mean()
    state = np.empty(n_u, dtype=object)
    state[uid] = sub.state.to_numpy()
    g = np.array([by_state.hpi_yoy.get(s, by_state.hpi_yoy.mean()) for s in state])
    u = np.array([by_state.unemp.get(s, by_state.unemp.mean()) for s in state])
    out = z.copy()
    ages = np.arange(z.shape[1])
    fut = ages[None, :] > a_u[:, None]
    out[:, :, 0] = np.where(fut, g[:, None], z[:, :, 0])
    out[:, :, 1] = np.where(fut, u[:, None], z[:, :, 1])
    if len(data.raw) == 3:
        base = z[np.arange(n_u), np.maximum(a_u, 1), 2]
        out[:, :, 2] = np.where(
            fut, base[:, None] - g[:, None] * (ages[None, :] - a_u[:, None]) / 12, z[:, :, 2]
        )
    return out


def estimate(
    data: Data,
    year: int,
    *,
    cohort: str = "year",
    horizon: int = 84,
    cutoff: int | None = None,
    rolling: bool = False,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """All methods at the December snapshot of `year`. cutoff (month index) = truth is cut there, so a cohort's target
    horizon is min(horizon, its age at the cutoff)."""
    s = int(month_idx(year * 100 + 12))
    s0 = s - 1  # cells use a 0-based month index
    info, curves = observed(data, s, cohort)
    t = info.copy()
    t["h"] = horizon if cutoff is None else np.minimum(horizon, t.A + (cutoff - s))
    t["mature"] = t.A >= horizon
    t["a_used"] = np.minimum(t.A, t.h)
    young = (t.A < t.h).to_numpy()
    g_a = np.array([curves[r.cohort][int(r.a_used)] for r in t.itertuples()])
    mat = t[t.mature].cohort.tolist()
    n_recent = 3 if cohort == "year" else 12
    t["naive"] = g_a
    t["drop_recent"] = [
        float(np.mean([curves[m][int(r.h)] for m in mat[-n_recent:]])) if y else g
        for r, y, g in zip(t.itertuples(), young, g_a, strict=True)
    ]
    t["fixed_horizon"] = [
        float(np.mean([curves[m][int(r.h)] for m in mat[-(n_recent // 3) :]])) if y else g
        for r, y, g in zip(t.itertuples(), young, g_a, strict=True)
    ]
    ages = np.arange(AMAX + 1, dtype=float)

    def cohorts_of(cv: dict[float, np.ndarray]) -> list[onestage.Cohort]:
        return [
            onestage.Cohort(
                r.cohort,
                ages[: int(r.a_used) + 1].copy(),
                cv[r.cohort][: int(r.a_used) + 1].copy(),
                float(r.events),
                float(r.h),
                bool(r.mature),
            )
            for r in t.itertuples()
        ]

    free = cohorts_of(curves)
    t["chain"] = onestage.fit(free, 0.0).level
    phi_free = onestage.dispersion(free)
    sel_free = onestage.select_model(free, phi_free)
    t["ours_free"] = sel_free.fit.level
    # calendar step
    fit1 = step1(data, s0)
    shares, g1_stat = guard1(data, s0)
    adj = adjusted_curves(data, s0, fit1, cohort)
    coh = cohorts_of(adj)
    phi_cal = onestage.dispersion(coh)
    sel = onestage.select_model(coh, phi_cal)
    fit_chain = onestage.fit(coh, 0.0)
    rng = {
        r: (
            float(data.cells.loc[data.cells.mi <= s0, r].min()),
            float(data.cells.loc[data.cells.mi <= s0, r].max()),
        )
        for r in data.raw
    }
    subs = {
        float(k): g for k, g in data.cells[data.cells[cohort].isin(t.cohort[young])].groupby(cohort)
    }
    cols: dict[str, list[float]] = {"cal_chain": [], "ours_cal": [], "ours_cal_uncond": []}
    g2 = {"cond": [0.0, 0.0], "uncond": [0.0, 0.0]}
    for i, r in enumerate(t.itertuples()):
        if not young[i]:
            for c in cols.values():
                c.append(float(g_a[i]))
            continue
        h, a, key = int(r.h), int(r.a_used), r.cohort
        inc_chain = increments(
            adj[key], a, float(fit_chain.speeds[i]), fit_chain.ugrid, fit_chain.f0, h
        )
        inc_ours = increments(adj[key], a, float(sel.fit.speeds[i]), sel.fit.ugrid, sel.fit.f0, h)
        cols["cal_chain"].append(
            plug_in(data, subs[key], s0, fit1, inc_chain, h, conditional=True)[0]
        )
        for name, conditional in (("cond", True), ("uncond", False)):
            p_hat, e_all, e_out = plug_in(
                data, subs[key], s0, fit1, inc_ours, h, conditional=conditional, support=rng
            )
            cols["ours_cal" if conditional else "ours_cal_uncond"].append(p_hat)
            g2[name][0] += e_all
            g2[name][1] += e_out
    for k, c in cols.items():
        t[k] = c
    # T3 diagnostic (not part of the estimate): maturity and variance inflation from smooth fits of the reference
    gg, wb = (
        cal.fit_gengamma(sel.fit.ugrid, sel.fit.f0),
        cal.fit_gengamma(sel.fit.ugrid, sel.fit.f0, weibull=True),
    )
    diag = [
        (
            cal.precision(gg, float(sp) * np.arange(int(a) + 1.0)),
            cal.precision(wb, float(sp) * np.arange(int(a) + 1.0))[1],
        )
        for sp, a in zip(sel.fit.speeds, t.a_used, strict=True)
    ]
    t["maturity_m"] = [x[0][0] for x in diag]
    t["inflation_gg"] = [x[0][1] for x in diag]
    t["inflation_weibull"] = [x[1] for x in diag]
    t["inflation_lower_bound"] = t[["inflation_gg", "inflation_weibull"]].min(axis=1)
    t["own_speed_not_learnable"] = t.maturity_m < 0.3
    calendar_on = g1_stat >= G1_THRESHOLD
    t["ours"] = t.ours_cal if calendar_on else t.ours_free  # G1 switches the calendar step
    t["ours_uncond"] = t.ours_cal_uncond if calendar_on else t.ours_free
    t["speed"] = sel.fit.speeds
    t["young"] = young
    t["g_a"] = g_a
    meta: dict[str, object] = {
        "model": sel.model,
        "phi": round(phi_cal, 2),
        "model_free": sel_free.model,
        "phi_free": round(phi_free, 2),
        "beta": [round(float(b), 3) for b in fit1[0]],
        "beta_se": [round(float(b), 3) for b in fit1[2]],
        "g1_shares": shares,
        "g1_events_x_min_share": round(g1_stat, 1),
        "calendar_on": bool(calendar_on),
        "g2_out_of_support": {k: round(v[1] / v[0], 3) if v[0] > 0 else 0.0 for k, v in g2.items()},
        "f0": (sel.fit.ugrid, sel.fit.f0),
    }
    if rolling:  # sensitivity only: RW(omega2) with omega2 from rolling-origin validation instead of the evidence
        om, h_used = rolling_omega(data, year, cohort, horizon)
        fit_ro = onestage.fit(coh, om)
        vals = []
        for i, r in enumerate(t.itertuples()):
            if not young[i]:
                vals.append(float(g_a[i]))
                continue
            inc = increments(
                adj[r.cohort],
                int(r.a_used),
                float(fit_ro.speeds[i]),
                fit_ro.ugrid,
                fit_ro.f0,
                int(r.h),
            )
            vals.append(plug_in(data, subs[r.cohort], s0, fit1, inc, int(r.h), conditional=True)[0])
        t["ours_rolling"] = vals if calendar_on else t.ours_free
        meta["rolling_omega2"], meta["rolling_h"] = om, h_used
    return t, meta


def rolling_omega(data: Data, year: int, cohort: str, horizon: int) -> tuple[float, int]:
    """Rolling-origin choice of omega2 (T2 Update 5): refit at S - h with data up to S - h, predict each young
    cohort's events in (S - h, S] along the realized path, Poisson deviance, one-standard-error rule toward smaller
    omega2. h = 24 months, 12 if S - 24 has no mature cohort; (0.0, 0) if neither is possible."""
    s = int(month_idx(year * 100 + 12))
    _, curves1 = observed(data, s, cohort)
    for h in (24, 12):
        sp = s - h
        info0, _ = observed(data, sp, cohort)
        if not (info0.A >= horizon).any():
            continue
        t0 = info0.copy()
        t0["h"], t0["mature"] = horizon, t0.A >= horizon
        t0["a_used"] = np.minimum(t0.A, horizon)
        fit0 = step1(data, sp - 1)
        adj0 = adjusted_curves(data, sp - 1, fit0, cohort)
        ages = np.arange(AMAX + 1, dtype=float)
        coh0 = [
            onestage.Cohort(
                r.cohort,
                ages[: int(r.a_used) + 1].copy(),
                adj0[r.cohort][: int(r.a_used) + 1].copy(),
                float(r.events),
                float(horizon),
                bool(r.mature),
            )
            for r in t0.itertuples()
        ]
        yng = [i for i, r in enumerate(t0.itertuples()) if 6 <= r.A < horizon]
        subs0 = {
            float(k): g
            for k, g in data.cells[data.cells[cohort].isin(t0.cohort.iloc[yng])].groupby(cohort)
        }
        totals, per = {}, {}
        for om in OMEGA_GRID:
            f = onestage.fit(coh0, om)
            dev = []
            for i in yng:
                r = t0.iloc[i]
                a1 = int(r.a_used)
                a2 = min(a1 + h, horizon)
                inc = increments(adj0[r.cohort], a1, float(f.speeds[i]), f.ugrid, f.f0, a2)
                p2 = plug_in(data, subs0[r.cohort], sp - 1, fit0, inc, a2, conditional=True)[0]
                p1 = plug_in(data, subs0[r.cohort], sp - 1, fit0, inc, a1, conditional=True)[0]
                e = max(float(r.n) * (p2 - p1), 1e-9)
                o = float(r.n) * float(curves1[r.cohort][a2] - curves1[r.cohort][a1])
                dev.append(2 * (o * np.log(o / e) - (o - e)) if o > 0 else 2 * e)
            totals[om], per[om] = float(np.sum(dev)), np.array(dev)
        best = min(totals, key=lambda k: totals[k])
        se = (
            float(np.std(per[best], ddof=1) * np.sqrt(len(per[best])))
            if len(per[best]) > 1
            else 0.0
        )
        return min(om for om in OMEGA_GRID if totals[om] <= totals[best] + se), h
    return 0.0, 0
