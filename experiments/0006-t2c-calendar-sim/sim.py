"""Synthetic validation of Theorem 2c (spec: the calendar-effects notes, section 6). No dataset: pure simulation.

15 annual cohorts, 10 groups, 4,000 units per cohort x group. Sub-distribution hazard (promotion-time form):
  lambda_i(a) = kappa_g * theta(c) * s(c) * f0(s(c) a) * exp(beta * z_g(c + a)),  f0 = Weibull density (shape 1.5, median 5).
theta falls for recent cohorts (drift about 0.8), speed rises 1.0 -> 1.5, beta = 0.5.
z_g(t) = crisis(t) * (1 + 0.8 u_g) + 0.5 sin(t/2 + phi_g), crisis = +2 in calendar years 10-11.
Truth = realized eventual incidence along the realized path, continued past T with z = 0.
Methods: naive; chain ladder; ours (no calendar); ours + calendar (Poisson fixed effects -> adjusted curves -> T2 -> p(c; z)).
Each T2 method also with log-linear smoothed speeds. 'national' = same shock path for every group (u_g = 0, equal phi).
Usage: sim.py [reps]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Final

import numpy as np
import scipy.sparse as sp
from sklearn.linear_model import PoissonRegressor

HERE: Final = Path(__file__).resolve().parent
T: Final = 15
G: Final = 10
N_UNITS: Final = 4000
BETA: Final = 0.5
SHAPE, SCALE = 1.5, 5.9
DT: Final = 0.25
FINE: Final = 0.05
A: Final = np.arange(T, 0, -1).astype(float)  # follow-up of cohort c
SPEED: Final = np.where(A >= 8, 1.0, 1.0 + 0.5 * (8 - A) / 7)
THETA0: Final = -np.log(1 - 0.04)
THETA: Final = THETA0 * np.where(A >= 8, 1.0, 1.0 - 0.30 * (8 - A) / 7)
SPEEDS: Final = np.linspace(0.4, 3.0, 131)
METHODS: Final = ["naive", "chain", "ours", "ours+smooth", "ours+cal", "ours+cal+smooth"]


def f0(u: np.ndarray) -> np.ndarray:
    u = np.maximum(u, 0)
    return (SHAPE / SCALE) * (u / SCALE) ** (SHAPE - 1) * np.exp(-((u / SCALE) ** SHAPE))


def F0(u: np.ndarray) -> np.ndarray:
    return 1 - np.exp(-((np.maximum(u, 0) / SCALE) ** SHAPE))


def crisis(t: np.ndarray) -> np.ndarray:
    return np.where((t >= 10) & (t < 12), 2.0, 0.0)


def world(rng: np.random.Generator, national: bool) -> dict[str, np.ndarray]:
    u = np.zeros(G) if national else rng.uniform(0, 1, G)
    phi = np.zeros(G) if national else rng.uniform(0, 2 * np.pi, G)
    kappa = np.exp(0.3 * rng.standard_normal(G))
    return {"u": u, "phi": phi, "kappa": kappa / kappa.mean()}


def z(w: dict[str, np.ndarray], g: int, t: np.ndarray) -> np.ndarray:
    return crisis(t) * (1 + 0.8 * w["u"][g]) + 0.5 * np.sin(t / 2 + w["phi"][g])


def cum_hazard(
    w: dict[str, np.ndarray], c: int, g: int, ages: np.ndarray, *, shock: bool = True
) -> np.ndarray:
    """Cumulative hazard H_{c,g}(a) on `ages` (fine grid integration)."""
    grid = np.arange(0, ages.max() + FINE, FINE)
    dens = SPEED[c] * f0(SPEED[c] * (grid + FINE / 2))
    tilt = np.exp(BETA * z(w, g, c + grid + FINE / 2)) if shock else 1.0
    cum = np.concatenate([[0], np.cumsum(dens * tilt * FINE)])
    return (
        w["kappa"][g] * THETA[c] * np.interp(ages, np.concatenate([grid, [grid[-1] + FINE]]), cum)
    )


def truth(w: dict[str, np.ndarray]) -> np.ndarray:
    """Eventual incidence per cohort along the realized path; the calendar path continues after T with z = 0."""
    out = np.empty(T)
    ages = np.arange(0, 60 + FINE, FINE)
    for c in range(T):
        p = []
        for g in range(G):
            t = c + ages[:-1] + FINE / 2
            tilt = np.where(t < T, np.exp(BETA * z(w, g, t)), 1.0)
            h = (
                w["kappa"][g]
                * THETA[c]
                * np.sum(SPEED[c] * f0(SPEED[c] * (ages[:-1] + FINE / 2)) * tilt * FINE)
            )
            p.append(1 - np.exp(-h))
        out[c] = np.mean(p)
    return out


def simulate(rng: np.random.Generator, w: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Cell counts: events[c, g, k] in age bins of DT years (k < A_c / DT) and at-risk counts."""
    nb = int(T / DT)
    ev = np.zeros((T, G, nb))
    risk = np.zeros((T, G, nb))
    for c in range(T):
        edges = np.arange(0, A[c] + DT / 2, DT)
        for g in range(G):
            cif = 1 - np.exp(-cum_hazard(w, c, g, edges))
            p = np.diff(cif)
            counts = rng.multinomial(N_UNITS, np.append(p, 1 - p.sum()))[:-1]
            k = len(counts)
            ev[c, g, :k] = counts
            risk[c, g, :k] = N_UNITS - np.concatenate([[0], np.cumsum(counts)[:-1]])
    return {"ev": ev, "risk": risk}


def shape_fit(curve: np.ndarray, kmax: int, f0_ref: np.ndarray) -> float:
    """Speed matching the normalised curve on bins 1..kmax to F0_ref(s * age) / F0_ref(s * A)."""
    if kmax < 8 or curve[kmax] <= 0:
        return float("nan")
    ages = np.arange(1, kmax + 1) * DT
    obs = curve[1 : kmax + 1] / curve[kmax]
    grid = np.arange(len(f0_ref)) * DT
    best, bs = np.inf, float("nan")
    for s in SPEEDS:
        den = np.interp(s * kmax * DT, grid, f0_ref, right=1.0)
        if den <= 1e-9:
            continue
        loss = float(np.mean((obs - np.interp(s * ages, grid, f0_ref, right=1.0) / den) ** 2))
        if loss < best:
            best, bs = loss, s
    return bs


def t2_estimates(
    curves: list[np.ndarray], n_events: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """T2 on cumulative curves (index = bin edge): returns level (per-cohort speed), level (smoothed), speeds, smoothed speeds."""
    ref = (
        curves[0] / curves[0][-1] if len(curves[0]) else np.array([0.0])
    )  # cohort 0 is the only mature cohort (A = T)
    f0_ref = np.concatenate([ref, [1.0]])
    speeds, level = np.full(T, np.nan), np.full(T, np.nan)
    for c in range(T):
        kmax = round(A[c] / DT)
        speeds[c] = shape_fit(curves[c], kmax, f0_ref)
    grid = np.arange(len(f0_ref)) * DT
    for c in range(T):
        kmax = round(A[c] / DT)
        if not np.isnan(speeds[c]):
            level[c] = curves[c][kmax] / max(
                np.interp(speeds[c] * A[c], grid, f0_ref, right=1.0), 1e-9
            )
    idx = np.where((A >= 3) & (A <= 12) & ~np.isnan(speeds))[0]
    b = np.polyfit(idx, np.log(speeds[idx]), 1, w=np.sqrt(n_events[idx]))
    sm_speed = speeds.copy()
    sm_level = level.copy()
    for c in np.where(A <= 12)[0]:
        sm_speed[c] = np.exp(np.polyval(b, c))
        sm_level[c] = curves[c][round(A[c] / DT)] / max(
            np.interp(sm_speed[c] * A[c], grid, f0_ref, right=1.0), 1e-9
        )
    return level, sm_level, speeds, sm_speed


def fit_calendar(w: dict[str, np.ndarray], sim: dict[str, np.ndarray]) -> tuple[float, np.ndarray]:
    """T2c step 1: Poisson with cohort x age fixed effects + group fixed effects + z, offset log at-risk."""
    cols_fe, cols_g, zval, ev, ex = [], [], [], [], []
    for c in range(T):
        for g in range(G):
            for k in range(round(A[c] / DT)):
                if sim["risk"][c, g, k] <= 0:
                    continue
                t_mid = c + (k + 0.5) * DT
                cols_fe.append(c * int(T / DT) + k)
                cols_g.append(g)
                zval.append(float(z(w, g, np.array([t_mid]))[0]))
                ev.append(sim["ev"][c, g, k])
                ex.append(sim["risk"][c, g, k])
    n = len(ev)
    r = np.arange(n)
    nbin = T * int(T / DT)
    x = sp.hstack(
        [
            sp.csr_matrix((np.ones(n), (r, cols_fe)), shape=(n, nbin)),
            sp.csr_matrix((np.ones(n), (r, cols_g)), shape=(n, G))[:, 1:],
            sp.csr_matrix(np.array(zval).reshape(-1, 1)),
        ]
    ).tocsr()
    ev_a, ex_a = np.array(ev), np.array(ex)
    keep = (
        np.bincount(cols_fe, weights=ev_a, minlength=nbin)[cols_fe] > 0
    )  # cells without any event carry no information
    m = PoissonRegressor(alpha=1e-13, fit_intercept=False, max_iter=5000, tol=1e-10)
    m.fit(x[keep], ev_a[keep] / ex_a[keep], sample_weight=ex_a[keep])
    kappa = np.concatenate([[0.0], m.coef_[nbin : nbin + G - 1]])
    return float(m.coef_[-1]), kappa


def adjusted_curves(
    w: dict[str, np.ndarray], sim: dict[str, np.ndarray], beta: float, kappa: np.ndarray
) -> list[np.ndarray]:
    """Breslow cumulative adjusted hazard per cohort: events / sum_g at-risk * exp(kappa_g + beta z_g), by bin."""
    curves = []
    for c in range(T):
        kmax = round(A[c] / DT)
        inc = np.zeros(kmax)
        for k in range(kmax):
            t_mid = c + (k + 0.5) * DT
            den = sum(
                sim["risk"][c, g, k]
                * np.exp(kappa[g] + beta * float(z(w, g, np.array([t_mid]))[0]))
                for g in range(G)
            )
            inc[k] = sim["ev"][c, :, k].sum() / den if den > 0 else 0.0
        curves.append(np.concatenate([[0.0], np.cumsum(inc)]))
    return curves


def plug_in(
    w: dict[str, np.ndarray],
    beta: float,
    kappa: np.ndarray,
    theta: np.ndarray,
    speed: np.ndarray,
    f0_ref: np.ndarray,
) -> np.ndarray:
    """p(c; z) = mean_g 1 - exp(-kappa_g theta_c E_{U~F0}[exp(beta z_g(c + U / s))]) along the realized path (z = 0 after T)."""
    grid = np.arange(len(f0_ref)) * DT
    atoms = np.diff(np.concatenate([[0.0], f0_ref]))
    out = np.empty(T)
    for c in range(T):
        u_age = (grid - DT / 2).clip(0) / speed[c]
        t = c + u_age
        p = []
        for g in range(G):
            tilt = np.where(t < T, np.exp(beta * z(w, g, t)), 1.0)
            p.append(
                1
                - np.exp(
                    -np.exp(kappa[g])
                    * theta[c]
                    * float(np.sum(atoms * tilt))
                    / max(atoms.sum(), 1e-9)
                )
            )
        out[c] = np.mean(p)
    return out


def one_rep(
    rng: np.random.Generator, national: bool
) -> tuple[np.ndarray, np.ndarray, float, np.ndarray]:
    w = world(rng, national)
    p_true = truth(w)
    sim = simulate(rng, w)
    events = sim["ev"].sum(axis=1)  # cohort x bin
    obs_curves = [
        np.concatenate([[0.0], np.cumsum(events[c, : round(A[c] / DT)]) / (N_UNITS * G)])
        for c in range(T)
    ]
    n_ev = np.array([curve[-1] * N_UNITS * G for curve in obs_curves])
    naive = np.array([curve[-1] for curve in obs_curves])
    ref = obs_curves[0] / obs_curves[0][-1]
    chain = np.array(
        [naive[c] / max(np.interp(A[c], np.arange(len(ref)) * DT, ref), 1e-9) for c in range(T)]
    )
    lvl, lvl_s, _, _ = t2_estimates(obs_curves, n_ev)
    beta_hat, kappa_hat = fit_calendar(w, sim)
    adj = adjusted_curves(w, sim, beta_hat, kappa_hat)
    th, th_s, sp_, sp_s = t2_estimates(adj, n_ev)
    ref_adj = adj[0] / adj[0][-1]
    f0_ref = np.concatenate([ref_adj, [1.0]])
    cal = plug_in(
        w, beta_hat, kappa_hat, np.nan_to_num(th, nan=0.0), np.nan_to_num(sp_, nan=1.0), f0_ref
    )
    cal_s = plug_in(w, beta_hat, kappa_hat, th_s, sp_s, f0_ref)
    return p_true, np.vstack([naive, chain, lvl, lvl_s, cal, cal_s]), beta_hat, kappa_hat


def main(reps: int) -> None:
    results: dict[str, object] = {}
    for name, national in (("multi-state shock", False), ("national shock only", True)):
        rng = np.random.default_rng(11)
        runs = [one_rep(rng, national) for _ in range(reps)]
        p_true = np.array([r[0] for r in runs])
        est = np.array([r[1] for r in runs])
        beta = np.array([r[2] for r in runs])
        rel = np.abs(est - p_true[:, None, :]) / p_true[:, None, :]
        young, old = slice(T - 5, T), A >= 12
        truth_drift = np.mean([p[young].mean() / p[old].mean() for p in p_true])
        show = (T - 1, T - 2, T - 3, T - 5, T - 8)
        err = {
            f"age{int(A[c])}": {
                m: round(float(np.nanmean(rel[:, j, c])), 3) for j, m in enumerate(METHODS)
            }
            for c in show
        }
        drift = {}
        for j, m in enumerate(METHODS):
            d = np.nanmean(est[:, j, young], axis=1) / np.nanmean(est[:, j][:, old], axis=1)
            drift[m] = [
                round(float(np.median(d)), 2),
                *np.round(np.nanpercentile(d, [5, 95]), 2).tolist(),
            ]
        results[name] = {
            "mean_relative_error": err,
            "drift_newest5_vs_old": drift,
            "truth_drift": round(float(truth_drift), 2),
            "beta_hat": {
                "truth": BETA,
                "median": round(float(np.median(beta)), 3),
                "p5_p95": np.round(np.percentile(beta, [5, 95]), 3).tolist(),
            },
        }
        print(
            f"\n== {name} (truth drift {truth_drift:.2f}; beta_hat median {np.median(beta):.3f} [5%,95% {np.percentile(beta, 5):.3f}, {np.percentile(beta, 95):.3f}])",
            flush=True,
        )
        print("age | " + " ".join(f"{m:>16}" for m in METHODS))
        for c in show:
            print(
                f"{int(A[c]):>3} | "
                + " ".join(f"{err[f'age{int(A[c])}'][m]:>16.0%}" for m in METHODS)
            )
        print(
            "drift: "
            + "; ".join(
                f"{m} {drift[m][0]:.2f} [{drift[m][1]:.2f},{drift[m][2]:.2f}]" for m in METHODS
            ),
            flush=True,
        )
    (HERE / "results.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 50)
