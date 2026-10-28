"""T2 simulations with calendar effects and robustness cases (spec: the identification notes, updates 1-2).

Extends an earlier simulation (sim_t2.py). Cohort c (0 = oldest) enters at calendar year c; the snapshot is at year K = 20.
Event-type units (prob p_lat[c]) have operational event age tau ~ F0_c / s(c); a calendar shock is a time change
of the event clock (hazard x5 for one calendar year). Every unit has a competing exit with baseline hazard 8%/year
(x4 in "wave" years). An event is observed iff the event age is before both the exit age and the follow-up A_c.
Truth = realized eventual incidence CIF_event(inf | c) on a separate large population run to the end,
including calendar events AFTER the snapshot.

Scenarios: clean, constant exit, (a) waves, (b) event shock, (c) both, (d) post-snapshot wave + shock hitting young
cohorts, (e) shape change (F0 shape differs for recent cohorts, rescaling violated), (f) abrupt speed jump.
Methods: naive; chain ladder; ours (per-cohort speed); ours+smooth (smoothed speed); ours+cal(cond) and
ours+cal(uncond) = operational clock + exit weights using the REALIZED calendar path / only the path known at the
snapshot. Over-identification check: speed from the first half of a cohort's follow-up vs the second half;
critical value = 95th percentile of |log ratio| in the clean scenario (so false positives are about 5%).
Calibrated bounds (update 3): for each cohort, the shape at its rescaled age u_c may differ from F0 by at most
M x Delta_hist(u_c) on the logit scale, Delta_hist = largest logit deviation among mature cohorts (A >= 15);
p(c) in [obs / F_hi, min(1, obs / F_lo)]. Reported for M in {0.5, 1, 2}, with the breakdown value M* (smallest M at which
the drift interval contains 1).
Usage: uv run python experiments/0004-t2-calendar-simulations/sim.py [reps]
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np

HERE: Final = Path(__file__).resolve().parent
K: Final = 20
A: Final = np.arange(K, 0, -1).astype(float)  # follow-up of cohort c
K_SHAPE, SCALE = 1.5, 5.9
M0: Final = SCALE * np.log(2) ** (1 / K_SHAPE)  # median event age at reference speed
N: Final = 20000
N_TRUTH: Final = 300000
AGE_GRID: Final = np.arange(0, 80.0001, 0.02)
S_SMOOTH: Final = np.where(A >= 12, 1.0, 1.0 + 0.6 * (12 - A) / 11)
S_JUMP: Final = np.where(A >= 6, 1.0, 1.6)
P_LAT: Final = np.where(A >= 12, 0.040, 0.040 - 0.010 * (12 - A) / 11)
SHAPE_C: Final = np.where(A >= 12, K_SHAPE, K_SHAPE + 1.0 * (12 - A) / 11)
SPEEDS: Final = np.linspace(0.3, 3.0, 271)
MS: Final = (0.5, 1.0, 2.0)
M_GRID: Final = np.arange(0, 8.01, 0.1)
QS: Final = np.linspace(0.05, 0.95, 19)
METHODS: Final = ["naive", "chain", "ours", "ours+smooth", "ours+cal(cond)", "ours+cal(uncond)"]


@dataclass(frozen=True)
class Scn:
    exit_base: float = 0.0
    waves: tuple[int, ...] = ()  # calendar years with exit hazard x4
    shocks: tuple[int, ...] = ()  # calendar years with event hazard x5
    shape_change: bool = False
    speed_jump: bool = False


SCENARIOS: Final = {
    "clean": Scn(),
    "constant exit only": Scn(0.08),
    "(a) refinancing waves": Scn(0.08, waves=(8, 15)),
    "(b) event shock": Scn(shocks=(12,)),
    "(c) both": Scn(0.08, (8, 15), (12,)),
    "(d) post-snapshot wave + shock": Scn(0.08, (8, 15, 21), (12, 22)),
    "(e) shape change": Scn(0.08, shape_change=True),
    "(f) abrupt speed jump": Scn(0.08, speed_jump=True),
}


def speed_of(scn: Scn) -> np.ndarray:
    return S_JUMP if scn.speed_jump else S_SMOOTH


def clocks(c: int, scn: Scn, known_only: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Cumulative event clock M_c(a) and exit hazard X_c(a) on AGE_GRID; `known_only` hides calendar years >= K."""
    t = np.floor(c + AGE_GRID[:-1] + 0.01)
    ok = (t < K) if known_only else np.ones_like(t, dtype=bool)
    m_e = np.where(np.isin(t, scn.shocks) & ok, 5.0, 1.0)
    h_x = scn.exit_base * np.where(np.isin(t, scn.waves) & ok, 4.0, 1.0)
    step = np.diff(AGE_GRID)
    return np.concatenate([[0], np.cumsum(m_e * step)]), np.concatenate(
        [[0], np.cumsum(h_x * step)]
    )


def draw(rng: np.random.Generator, c: int, n: int, scn: Scn) -> tuple[np.ndarray, np.ndarray]:
    m, x = clocks(c, scn)
    k = SHAPE_C[c] if scn.shape_change else K_SHAPE
    scale = M0 / np.log(2) ** (1 / k)
    is_event = rng.random(n) < P_LAT[c]
    tau = scale * (-np.log1p(-rng.random(is_event.sum()))) ** (1 / k) / speed_of(scn)[c]
    a_e = np.interp(tau, m, AGE_GRID, right=np.inf)
    e = rng.exponential(size=len(a_e))
    a_x = np.interp(e, x, AGE_GRID, right=np.inf) if x[-1] > 0 else np.full(len(a_e), np.inf)
    return a_e, a_x


def truth(scn: Scn) -> np.ndarray:
    rng = np.random.default_rng(123)
    out = np.empty(K)
    for c in range(K):
        a_e, a_x = draw(rng, c, N_TRUTH, scn)
        out[c] = np.sum(a_e < a_x) / N_TRUTH
    return out


class Ecdf:
    def __init__(self, ref: np.ndarray, w: np.ndarray | None = None) -> None:
        order = np.argsort(ref)
        self.ref = ref[order]
        cw = np.cumsum(np.ones(len(ref)) if w is None else w[order])
        self.cw = np.concatenate([[0.0], cw / cw[-1]])

    def __call__(self, u: np.ndarray | float) -> np.ndarray | float:
        return self.cw[np.searchsorted(self.ref, u, side="right")]


def fit_speed(t: np.ndarray, w: np.ndarray, F0h: Ecdf, lo: float, hi: float) -> float:
    """Speed matching the within-window weighted ECDF of times in (lo, hi] to the reference curve."""
    m = (t > lo) & (t <= hi)
    if m.sum() < 5:
        return float("nan")
    order = np.argsort(t[m])
    tt, ww = t[m][order], w[m][order]
    emp = (np.cumsum(ww) - 0.5 * ww) / ww.sum()
    best, bs = np.inf, float("nan")
    for s in SPEEDS:
        base = F0h(s * lo) if lo > 0 else 0.0
        den = F0h(s * hi) - base
        if den <= 1e-9:
            continue
        d = np.average((emp - (F0h(s * tt) - base) / den) ** 2, weights=ww)
        if d < best:
            best, bs = d, s
    return bs


def one_rep(
    rng: np.random.Generator, scn: Scn, cl: list, cl_known: list
) -> tuple[np.ndarray, np.ndarray]:
    ages, ops, wts = [], [], []
    for c in range(K):
        a_e, a_x = draw(rng, c, N, scn)
        a = np.sort(a_e[(a_e < a_x) & (a_e <= A[c])])
        m, x = cl_known[c]  # events up to the snapshot only need the known path
        ages.append(a)
        ops.append(np.interp(a, AGE_GRID, m))
        wts.append(np.exp(np.interp(a, AGE_GRID, x)))
    obs = np.array([len(a) / N for a in ages])
    nev = np.array([len(a) for a in ages])
    mature = A >= 18
    F0h = Ecdf(np.concatenate([ages[c] for c in range(K) if mature[c]]))
    naive, chain, ours = obs.copy(), np.empty(K), np.full(K, np.nan)
    s_hat, stat = np.full(K, np.nan), np.full(K, np.nan)
    for c in range(K):
        chain[c] = obs[c] / max(F0h(A[c]), 1e-9)
        if nev[c] < 5:
            continue
        s_hat[c] = fit_speed(ages[c], np.ones(nev[c]), F0h, 0, A[c])
        ours[c] = obs[c] / max(F0h(s_hat[c] * A[c]), 1e-9)
        early, late = (
            fit_speed(ages[c], np.ones(nev[c]), F0h, 0, A[c] / 2),
            fit_speed(ages[c], np.ones(nev[c]), F0h, A[c] / 2, A[c]),
        )
        stat[c] = np.log(late / early)
    idx = np.where((A >= 3) & (A <= 12) & ~np.isnan(s_hat))[0]
    b = np.polyfit(idx, np.log(s_hat[idx]), 1, w=np.sqrt(nev[idx]))
    smooth = ours.copy()
    s_final = np.where(np.isnan(s_hat), 1.0, s_hat)
    for c in np.where(A <= 12)[0]:
        s_final[c] = np.exp(np.polyval(b, c))
        smooth[c] = obs[c] / max(F0h(s_final[c] * A[c]), 1e-9)
    bnd, mstar = calibrated_bounds(ages, nev, obs, F0h, s_final)
    cond = _ours_cal(ops, wts, nev, cl_known, cl)
    uncond = _ours_cal(ops, wts, nev, cl_known, cl_known)
    return np.vstack([naive, chain, ours, smooth, cond, uncond]), stat, bnd, mstar


def _logit(p: np.ndarray | float) -> np.ndarray | float:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def calibrated_bounds(
    ages: list, nev: np.ndarray, obs: np.ndarray, F0h: Ecdf, s_final: np.ndarray
) -> tuple[np.ndarray, float]:
    """Bounds (len(MS), 2, K) on each cohort's eventual incidence and the drift breakdown value M*."""
    u_q = F0h.ref[np.minimum(np.searchsorted(F0h.cw[1:], QS), len(F0h.ref) - 1)]
    dev = []
    for c in np.where((A >= 15) & (nev >= 50))[0]:
        sc = fit_speed(ages[c], np.ones(nev[c]), F0h, 0, A[c])
        emp = np.clip(
            np.searchsorted(ages[c], u_q / sc, side="right") / nev[c],
            0.5 / nev[c],
            1 - 0.5 / nev[c],
        )
        dev.append(np.abs(_logit(emp) - _logit(QS)))
    delta = np.max(dev, axis=0)
    q_c = np.clip(np.array([F0h(s_final[c] * A[c]) for c in range(K)]), 0.005, 0.995)
    d_c = np.interp(q_c, QS, delta)

    def interval(m: float) -> np.ndarray:
        f_hi, f_lo = (
            1 / (1 + np.exp(-(_logit(q_c) + m * d_c))),
            1 / (1 + np.exp(-(_logit(q_c) - m * d_c))),
        )
        return np.vstack([obs / f_hi, np.minimum(1.0, obs / f_lo)])

    young, old = slice(K - 5, K), A >= 12

    def drift(iv: np.ndarray) -> tuple[float, float]:
        return iv[0][young].mean() / iv[1][old].mean(), iv[1][young].mean() / iv[0][old].mean()

    includes_one = [drift(interval(m))[0] <= 1 <= drift(interval(m))[1] for m in M_GRID]
    mstar = next((m for m, ok in zip(M_GRID, includes_one, strict=True) if ok), float("inf"))
    return np.array([interval(m) for m in MS]), float(mstar)


def _ours_cal(ops: list, wts: list, nev: np.ndarray, cl_obs: list, cl_proj: list) -> np.ndarray:
    """Operational clock + exit weights. Observed times use `cl_obs`; the projection to the realized CIF uses `cl_proj`."""
    mature = A >= 18
    ref_t = np.concatenate([ops[c] for c in range(K) if mature[c]])
    ref_w = np.concatenate([wts[c] for c in range(K) if mature[c]])
    F0o = Ecdf(ref_t, ref_w)
    order = np.argsort(ref_t)
    atoms, atom_w = ref_t[order], ref_w[order]
    horizon = [cl_obs[c][0][np.searchsorted(AGE_GRID, A[c])] for c in range(K)]
    s_hat, out = np.full(K, np.nan), np.full(K, np.nan)
    for c in range(K):
        if nev[c] >= 5:
            s_hat[c] = fit_speed(ops[c], wts[c], F0o, 0, horizon[c])
    idx = np.where((A >= 3) & (A <= 12) & ~np.isnan(s_hat))[0]
    b = np.polyfit(idx, np.log(s_hat[idx]), 1, w=np.sqrt(nev[idx]))
    for c in range(K):
        s = (
            np.exp(np.polyval(b, c))
            if A[c] <= 12
            else (s_hat[c] if not np.isnan(s_hat[c]) else 1.0)
        )
        p_lat = wts[c].sum() / N / max(F0o(s * horizon[c]), 1e-9)
        m, x = cl_proj[c]
        a_atoms = np.interp(atoms / s, m, AGE_GRID)
        out[c] = p_lat * np.average(np.exp(-np.interp(a_atoms, AGE_GRID, x)), weights=atom_w)
    return out


def main(reps: int) -> None:
    results: dict[str, object] = {}
    crit: np.ndarray | None = None
    for name, scn in SCENARIOS.items():
        cl = [clocks(c, scn) for c in range(K)]
        cl_known = [clocks(c, scn, known_only=True) for c in range(K)]
        p_true = truth(scn)
        rng = np.random.default_rng(7)
        runs = [one_rep(rng, scn, cl, cl_known) for _ in range(reps)]
        res, stat = np.array([r[0] for r in runs]), np.array([r[1] for r in runs])
        bnd, mstar = np.array([r[2] for r in runs]), np.array([r[3] for r in runs])
        if name == "clean":
            crit = np.nanpercentile(np.abs(stat), 95, axis=0)
        det = np.nanmean(np.abs(stat) > crit, axis=0)
        rel = np.abs(res - p_true) / p_true
        young, old = slice(K - 5, K), A >= 12
        ages_show = (K - 1, K - 2, K - 3, K - 5, K - 8, K - 10)
        table = {
            f"age{int(A[c])}": {
                m: round(float(np.nanmean(rel[:, j, c])), 3) for j, m in enumerate(METHODS)
            }
            | {"true_p": round(float(p_true[c]), 4), "overid_detect": round(float(det[c]), 2)}
            for c in ages_show
        }
        drift = {}
        for j, m in enumerate(METHODS):
            r = np.nanmean(res[:, j, young], axis=1) / np.nanmean(res[:, j][:, old], axis=1)
            drift[m] = [
                round(float(np.median(r)), 2),
                *np.round(np.percentile(r, [5, 95]), 2).tolist(),
            ]
        truth_drift = float(p_true[young].mean() / p_true[old].mean())
        bounds_out = {}
        for i, m in enumerate(MS):
            cover = (bnd[:, i, 0, :] <= p_true) & (p_true <= bnd[:, i, 1, :])
            width = (bnd[:, i, 1, :] - bnd[:, i, 0, :]) / p_true
            d_lo = bnd[:, i, 0, young].mean(axis=1) / bnd[:, i, 1, :][:, old].mean(axis=1)
            d_hi = bnd[:, i, 1, young].mean(axis=1) / bnd[:, i, 0, :][:, old].mean(axis=1)
            bounds_out[f"M={m}"] = {
                "coverage_by_age": {
                    f"age{int(A[c])}": round(float(cover[:, c].mean()), 2) for c in ages_show
                },
                "width_over_truth_by_age": {
                    f"age{int(A[c])}": round(float(np.median(width[:, c])), 2) for c in ages_show
                },
                "drift_interval_covers_truth": round(
                    float(((d_lo <= truth_drift) & (truth_drift <= d_hi)).mean()), 2
                ),
                "drift_interval_excludes_1": round(float(((d_lo > 1) | (d_hi < 1)).mean()), 2),
            }
        bounds_out["breakdown_M_star_median"] = round(float(np.median(mstar)), 1)
        bounds_out["breakdown_M_star_never_share"] = round(float(np.isinf(mstar).mean()), 2)
        results[name] = {
            "mean_relative_error": table,
            "relative_drift_newest5_vs_old": drift,
            "truth_drift": round(truth_drift, 2),
            "calibrated_bounds": bounds_out,
        }
        print(f"\n== {name}   (truth drift {truth_drift:.2f})", flush=True)
        print("age true_p | " + " ".join(f"{m:>16}" for m in METHODS) + " | overid")
        for c in ages_show:
            print(
                f"{int(A[c]):>3} {p_true[c]:.4f} | "
                + " ".join(f"{np.nanmean(rel[:, j, c]):>16.0%}" for j in range(len(METHODS)))
                + f" | {det[c]:.0%}"
            )
        b1 = bounds_out["M=1.0"]
        print(
            f"bounds M=1: coverage {b1['coverage_by_age']} width/truth {b1['width_over_truth_by_age']} "
            f"drift-interval covers truth {b1['drift_interval_covers_truth']}, excludes 1: "
            f"{b1['drift_interval_excludes_1']}, M* median {bounds_out['breakdown_M_star_median']}",
            flush=True,
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
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 100)
