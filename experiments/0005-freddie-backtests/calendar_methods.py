"""Calendar-adjusted backtest methods (update 4, item b). DATASET: official 50k Sample, DEVELOPMENT ONLY.

At snapshot S: fit the default hazard (age bins, vintage level, LTV-band level, state HPI growth, state unemployment,
log mark-to-market LTV) on the cells observable at S (calendar month <= S). The per-cell multiplier exp(beta'z + gamma)
gives each vintage an operational clock T_c(a) = sum of at-risk-weighted multipliers over ages 1..a.
  conditional    the multipliers after S are the REALIZED ones (the real future state-level macro is given)
  unconditional  the multipliers after S are replaced by their historical mean up to S (no future information)
Curves are then handled in operational time: speed fit, empirical-Bayes smoothing and the projection
p = G(A) * F0(s T(H)) / F0(s T(A)) use the vintage's own clock. Calendar-adjusted chain ladder (speed = 1) is the baseline.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd
import pyfixest as pf
from scipy import optimize, stats

from maturity import eb

HERE: Final = Path(__file__).resolve().parents[1] / "0003-freddie-sample"
H: Final = 84
AMAX: Final = 120
BIN: Final = 3
ZCOLS: Final = ["hpi_yoy_z", "unemp_z", "log_mtm_z"]
SPEEDS: Final = np.linspace(0.4, 3.0, 131)


def load_cells() -> pd.DataFrame:
    return pd.read_parquet(HERE / "artifacts" / "hazard_cells_mtm.parquet")


def fit_beta(
    cells: pd.DataFrame, s_mi: int
) -> tuple[np.ndarray, np.ndarray, pd.Series, np.ndarray]:
    """T2c step 1: Poisson fixed-effects regression on cells observable at S (calendar month <= S, vintages up to S):
    vintage x age fixed effects + state fixed effects + z, offset log at-risk. The vintage x age
    effects absorb level, speed and baseline, so beta is learned from same-cohort, same-age contrasts across states."""
    return _fit(cells[(cells.mi <= s_mi) & (cells.vintage <= s_mi // 12)])


def clocks(
    cells: pd.DataFrame,
    s_mi: int,
    fit: tuple[np.ndarray, np.ndarray, pd.Series, np.ndarray],
    mode: str,
) -> dict[int, np.ndarray]:
    """Operational clock T_v[0..AMAX] per vintage from the multipliers exp(kappa_state + gamma_band + beta'z).
    mode: cond (realized future z), uncond_mean (future exp(beta'z) = historical exposure-weighted mean up to S) or
    uncond_last12 (future exp(beta'z) = mean over the last 12 months up to S). State and band effects are kept."""
    beta, gamma, kappa, _se = fit
    zmult = np.exp(cells[ZCOLS].to_numpy() @ beta)
    seen = (cells.mi <= s_mi).to_numpy()
    w = cells.exposure.to_numpy()
    if mode == "uncond_mean":
        zmult = np.where(seen, zmult, np.average(zmult[seen], weights=w[seen]))
    elif mode == "uncond_last12":
        last = seen & (cells.mi > s_mi - 12).to_numpy()
        zmult = np.where(seen, zmult, np.average(zmult[last], weights=w[last]))
    mult = zmult * np.exp(
        cells.state.map(kappa).fillna(0.0).to_numpy() + gamma[cells.ltv_band.to_numpy()]
    )
    g = pd.DataFrame(
        {
            "vintage": cells.vintage,
            "age": cells.age,
            "num": cells.exposure * mult,
            "den": cells.exposure,
        }
    )
    g = g.groupby(["vintage", "age"]).sum()
    mbar = (g.num / g.den).unstack("age").reindex(columns=range(1, AMAX + 1)).fillna(1.0)
    return {
        int(v): np.concatenate([[0.0], np.cumsum(row.to_numpy())]) for v, row in mbar.iterrows()
    }


def _fit(c: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, pd.Series, np.ndarray]:
    """Unpenalized fixed-effects Poisson (pyfixest), standard errors clustered by state.
    Returns (beta, LTV-band effects [zeros: collinear with log MTM LTV], centred state effects, clustered SE of beta)."""
    d = c[["events", "exposure", "state", *ZCOLS]].copy()
    d["cell"] = (c.vintage.to_numpy() - int(c.vintage.min())) * AMAX + (c.age.to_numpy() - 1)
    d["log_exposure"] = np.log(d.exposure)
    m = pf.fepois(
        f"events ~ {' + '.join(ZCOLS)} | cell + state",
        data=d,
        offset="log_exposure",
        vcov={"CRV1": "state"},
    )
    tidy = m.tidy()
    fe = m.fixef()
    kappa = pd.Series(fe[next(k for k in fe if "state" in k)], dtype=float)
    wts = d.groupby("state").exposure.sum().reindex(kappa.index).fillna(0.0).to_numpy()
    kappa = kappa - float(np.average(kappa.to_numpy(), weights=wts))
    return (
        tidy.loc[ZCOLS, "Estimate"].to_numpy(),
        np.zeros(4),
        kappa,
        tidy.loc[ZCOLS, "Std. Error"].to_numpy(),
    )


def estimate(
    curves: dict[int, np.ndarray], info: pd.DataFrame, clk: dict[int, np.ndarray]
) -> tuple[pd.Series, pd.Series, dict[str, object]]:
    """(calendar-adjusted chain ladder, calendar-adjusted EB speed) estimates indexed like info.vintage."""
    mature = [int(v) for v in info[info.A >= H].vintage]
    umax = 3.5 * max(clk[v][H] for v in mature)
    ugrid = np.linspace(0, umax, 1500)
    f0_chain = _pooled(curves, clk, mature, {}, ugrid)
    f0 = f0_chain
    for _ in range(3):
        sp_ = {v: _fit_speed(curves[v], clk[v], H, f0, ugrid) for v in mature}
        valid = {v: s for v, s in sp_.items() if not np.isnan(s)}
        if not valid:
            break
        g = np.exp(np.mean(np.log(list(valid.values()))))
        f0 = _pooled(curves, clk, mature, {v: s / g for v, s in valid.items()}, ugrid)
    chain, shat = {}, {}
    for r in info.itertuples():
        v, a = int(r.vintage), int(r.a_used)
        g_a = curves[v][min(a, 200)]
        if r.A >= H:
            chain[v] = g_a
            shat[v] = _fit_speed(curves[v], clk[v], H, f0, ugrid)
            continue
        chain[v] = min(
            1.0, g_a * _f(f0_chain, ugrid, clk[v][H]) / max(_f(f0_chain, ugrid, clk[v][a]), 1e-9)
        )
        shat[v] = _fit_speed(curves[v], clk[v], a, f0, ugrid)
    y = np.array(
        [
            np.log(shat[int(v)]) if not np.isnan(shat[int(v)]) and e >= 10 else np.nan
            for v, e in zip(info.vintage, info.events, strict=True)
        ]
    )
    fit = eb.eb_log_speeds(y, info.events.to_numpy(dtype=float))
    out = {}
    for r, x in zip(info.itertuples(), fit.x, strict=True):
        v, a, s = int(r.vintage), int(r.a_used), float(np.exp(x))
        g_a = curves[v][min(a, 200)]
        out[v] = (
            g_a
            if r.A >= H
            else min(
                1.0, g_a * _f(f0, ugrid, s * clk[v][H]) / max(_f(f0, ugrid, s * clk[v][a]), 1e-9)
            )
        )
    idx = [int(v) for v in info.vintage]
    speeds = {int(v): float(np.exp(x)) for v, x in zip(info.vintage, fit.x, strict=True)}
    gg, wb = fit_gengamma(ugrid, f0), fit_gengamma(ugrid, f0, weibull=True)
    rows = []
    for r in info.itertuples():
        v = int(r.vintage)
        u = speeds[v] * clk[v][: int(r.a_used) + 1]
        m, infl = precision(gg, u)
        rows.append(
            {
                "vintage": v,
                "maturity_m": m,
                "inflation_gg": infl,
                "inflation_weibull": precision(wb, u)[1],
            }
        )
    extra: dict[str, object] = {
        "omega2": fit.omega2,
        "kappa": fit.kappa,
        "ugrid": ugrid,
        "f0": f0,
        "gg": gg,
        "weibull": wb,
        "diag": pd.DataFrame(rows).set_index("vintage"),
    }
    return (
        pd.Series([chain[v] for v in idx], index=idx),
        pd.Series([out[v] for v in idx], index=idx),
        extra,
    )


def fit_gengamma(
    ugrid: np.ndarray, f0: np.ndarray, *, weibull: bool = False
) -> tuple[float, float, float, float]:
    """Binned, right-truncated maximum likelihood of a generalized gamma (a, c, scale) to the pooled reference F0.
    The reference is normalised to 1 at u_max (first point where F0 >= 0.999), so bin masses are compared with
    [F(u_j) - F(u_{j-1})] / F(u_max). weibull=True fixes a = 1. Returns (a, c, scale, u_max)."""
    top = int(np.argmax(f0 >= 0.999)) if (f0 >= 0.999).any() else len(f0) - 1
    step = max(1, top // 150)
    edges = ugrid[: top + 1 : step]
    mass = np.diff(f0[: top + 1 : step])
    mass = np.maximum(mass, 0) / max(mass.sum(), 1e-12)
    med = float(np.interp(0.5, f0[: top + 1], ugrid[: top + 1]))

    def nll(theta: np.ndarray) -> float:
        a = 1.0 if weibull else float(np.exp(theta[0]))
        c, scale = float(np.exp(theta[-2])), float(np.exp(theta[-1]))
        cdf = stats.gengamma.cdf(edges, a, c, scale=scale)
        prob = np.diff(cdf) / max(cdf[-1], 1e-300)
        return float(-np.sum(mass * np.log(np.maximum(prob, 1e-300))))

    x0 = np.log([1.5, med]) if weibull else np.log([1.0, 1.5, med])
    res = optimize.minimize(
        nll, x0, method="Nelder-Mead", options={"xatol": 1e-6, "fatol": 1e-10, "maxiter": 4000}
    )
    a = 1.0 if weibull else float(np.exp(res.x[0]))
    return a, float(np.exp(res.x[-2])), float(np.exp(res.x[-1])), float(edges[-1])


def precision(par: tuple[float, float, float, float], u: np.ndarray) -> tuple[float, float]:
    """T3 Theorem 3(a) for a cohort whose monthly interval end-points in rescaled time are `u` (u[0] = 0):
    maturity m = F0(u_c) and the inflation factor 1 + e^2 / V, with e and eta from the fitted smooth F0."""
    a, c, scale, u_max = par
    cdf = stats.gengamma.cdf(u, a, c, scale=scale)
    g = u * stats.gengamma.pdf(u, a, c, scale=scale)
    d_f = np.diff(cdf)
    ok = d_f > 1e-300
    m = float(min(1.0, cdf[-1] / max(stats.gengamma.cdf(u_max, a, c, scale=scale), 1e-300)))
    if ok.sum() < 3:
        return m, float("nan")
    eta = np.diff(g)[ok] / d_f[ok]
    w = d_f[ok] / d_f[ok].sum()
    e = float(np.sum(w * eta))
    v = float(np.sum(w * (eta - e) ** 2))
    return m, (1 + e * e / v) if v > 0 else float("inf")


def _f(f0: np.ndarray, ugrid: np.ndarray, u: float) -> float:
    return float(np.interp(u, ugrid, f0, right=1.0))


def _pooled(
    curves: dict[int, np.ndarray],
    clk: dict[int, np.ndarray],
    mature: list[int],
    speed: dict[int, float],
    ugrid: np.ndarray,
) -> np.ndarray:
    """F0(u) = sum_c G_c(H) * min(1, G_c(a_c(u)) / G_c(H)) / sum_c G_c(H), a_c(u) = age whose scaled clock equals u."""
    num, den = np.zeros_like(ugrid), 0.0
    for v in mature:
        s = speed.get(v, 1.0)
        age = np.interp(ugrid / s, clk[v], np.arange(AMAX + 1), right=AMAX)
        g = np.interp(age, np.arange(201), curves[v])
        gh = curves[v][H]
        num += np.minimum(g, gh)
        den += gh
    return np.minimum(num / max(den, 1e-12), 1.0)


def _fit_speed(cif: np.ndarray, t: np.ndarray, a: int, f0: np.ndarray, ugrid: np.ndarray) -> float:
    if a < 12 or cif[a] <= 0:
        return float("nan")
    ages = np.arange(6, a + 1)
    obs = cif[ages] / cif[a]
    best, bs = np.inf, float("nan")
    for s in SPEEDS:
        den = _f(f0, ugrid, s * t[a])
        if den <= 1e-9:
            continue
        loss = float(np.mean((obs - np.interp(s * t[ages], ugrid, f0, right=1.0) / den) ** 2))
        if loss < best:
            best, bs = loss, s
    return bs
