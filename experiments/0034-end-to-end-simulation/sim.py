"""End-to-end simulation of Algorithm 1 with a known truth (round 3, item P1.10). POST HOC. DATASET: none (simulated).

World (months; 8 cohort years, 12 entry months per year, 6 groups, n units per entry month x group):
  sub-distribution hazard of a unit of cohort year c, group g, at age a (month e + a - 1, e = entry month)
      kappa_g * theta_c * [F0(s_c a) - F0(s_c (a - 1))] * exp(beta' z_g(month)),
  F0 Weibull (shape 1.5) or log-normal (sigma 0.8), median 36 months; log theta_c a random walk (sd 0.15 per year);
  two covariates z_jg(t) = (1 + 0.8 u_jg) m_j(t) + 0.5 w_jg(t), m and w AR(1) by month (0.97 and 0.90, variance 1).
  Speeds s_c: constant (all 1); step (one jump of +-log 1.4 at a cohort drawn from the 3 youngest); drift (log speed
  rises or falls by log 1.5 over the 5 youngest cohorts, linear). beta = (0, 0) or (0.5, 0.25).
  Competing exits are not drawn: in sub-distribution risk sets a unit with a competing exit stays at risk, so only
  the sub-distribution hazard matters; there is no censoring before the snapshot.
Truth of a cohort = its realised incidence at H = 36 months.

`estimate` is the frozen pipeline (experiments/0012/pipeline.py `estimate`) on arrays instead of loan and cell tables:
Aalen-Johansen curves at exact ages, calendar step (unpenalized fixed-effects Poisson, guard G1 >= 100), speed model
by tempered evidence (`maturity.onestage.select_model`), plug-in on the realised and on the forecast (persistence) path.
`check.py` builds the frozen table layout for some worlds and asserts that both give the same numbers.
One replication = the December snapshot of year 7 (annual cohorts, scored) and the December snapshots of years
3 to 6 (quarterly cohorts, for the conformal calibration errors; the rule of experiments/0011 `conformal`).
Replications (all / with the calibration snapshots, which are 80% of the run time): main settings (Weibull timing,
calendar effect on) 500 / 500; Weibull without calendar effect 500 / 80; log-normal timing 200 / 40 (its
generalized-gamma reference fit is slow).

Usage: PYTHONHASHSEED=0 uv run python sim.py time | run -> artifacts/parts/*.parquet (a stopped run continues)
"""

from __future__ import annotations

import importlib.util
import itertools
import logging
import math
import multiprocessing as mp
import sys
import time
import warnings
from dataclasses import dataclass, replace
from pathlib import Path
from types import ModuleType
from typing import Final

import numpy as np
import pandas as pd
import pyfixest as pf
from numpy.typing import NDArray
from scipy import stats

from maturity import onestage

HERE: Final = Path(__file__).resolve().parent
EXP: Final = HERE.parent
log = logging.getLogger(__name__)
Array = NDArray[np.float64]
Ints = NDArray[np.int64]

SEED: Final = 20261006
WORKERS: Final = 5
# (timing, calendar effect) -> (replications, of which with calibration snapshots and intervals)
REPS: Final = {
    ("weibull", True): (500, 500),
    ("weibull", False): (500, 80),
    ("lognormal", True): (200, 40),
}
K: Final = 8  # cohort years 0..7
H: Final = 36  # horizon in months
G: Final = 6  # groups
P: Final = 2  # covariates
MONTHS: Final = 12 * K
TARGET: Final = K - 1
CALIBRATION: Final = range(H // 12, TARGET)
SHAPE: Final = 1.5  # Weibull
SIGMA: Final = 0.8  # log-normal
MEDIAN: Final = 36.0  # of both timing shapes: F0(H) = 0.5
THETA0: Final = 0.12  # incidence at H about 6% at speed 1 without calendar effect
SD_LEVEL: Final = 0.15
STEP: Final = math.log(1.4)
STEP_LAST: Final = 3
DRIFT: Final = math.log(1.5)
DRIFT_LAST: Final = 5
BETA_ON: Final = (0.5, 0.25)
SIZES: Final = {
    "20k": 280,
    "200k": 2800,
}  # units per entry month x group (x 72 = units per cohort year)
SPEEDS: Final = ("constant", "step", "drift")
ALPHA: Final = 0.10
INTERVALS: Final = ("ours", "ours_uncond", "chain")  # methods with a conformal interval
AGES: Final = np.arange(1, H + 1)
LABELS: Final = np.array([f"g{g}" for g in range(G)])
PARTS: Final = HERE / "artifacts" / "parts"
warnings.filterwarnings("ignore", message=".*removed because of separation.*")


def load_module(name: str, path: Path) -> ModuleType:
    """A module of another experiment, under a name that cannot clash (several experiments have a pipeline.py)."""
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


pl: Final = load_module(
    "frozen_pipeline", EXP / "0012-freddie-200k-harp-correction" / "pipeline.py"
)
extras: Final = load_module("round3_extras", EXP / "0031-round3-backtests" / "extras.py")


@dataclass(frozen=True, slots=True)
class Setting:
    speed: str  # constant | step | drift
    calendar: bool  # beta = BETA_ON, else 0
    size: str  # key of SIZES
    timing: str = "weibull"  # weibull | lognormal

    @property
    def name(self) -> str:
        calendar = "calendar" if self.calendar else "no calendar"
        return f"{self.timing}/{self.speed}/{calendar}/{self.size}"

    @property
    def reps(self) -> tuple[int, int]:
        return REPS[self.timing, self.calendar]


SETTINGS: Final = [
    Setting(speed, calendar, size, timing)
    for timing, calendar in REPS
    for speed, size in itertools.product(SPEEDS, SIZES)
]


@dataclass(frozen=True, slots=True)
class World:
    n: int  # units per entry month x group
    events: Array  # (entry month, group, age - 1): events at ages 1..H
    cum: Array  # (entry month, group, age): events up to and including that age (index 0 = 0)
    z: Array  # (covariate, group, month)
    zs: Array  # (covariate, entry month, group, age - 1): standardised covariate of each cell
    speed: Array  # (cohort year,)
    timing: str


def f0(u: Array, timing: str) -> Array:
    ratio = np.maximum(u, 0) / MEDIAN
    if timing == "weibull":
        return np.asarray(1 - np.exp(-np.log(2) * ratio**SHAPE))
    return np.asarray(stats.norm.cdf(np.log(np.maximum(ratio, 1e-300)) / SIGMA))


def ar1(rng: np.random.Generator, shape: tuple[int, ...], rho: float, length: int) -> Array:
    x = rng.standard_normal((*shape, length))
    x[..., 1:] *= math.sqrt(1 - rho**2)
    for t in range(1, length):
        x[..., t] += rho * x[..., t - 1]
    return x


def speed_path(kind: str, rng: np.random.Generator) -> Array:
    log_s = np.zeros(K)
    sign = float(rng.choice([-1.0, 1.0]))
    start = int(rng.integers(K - STEP_LAST, K))
    if kind == "step":
        log_s[start:] = sign * STEP
    elif kind == "drift":
        log_s[K - DRIFT_LAST :] = sign * DRIFT * np.arange(1, DRIFT_LAST + 1) / DRIFT_LAST
    return np.exp(log_s)


def simulate(setting: Setting, rng: np.random.Generator) -> World:
    n = SIZES[setting.size]
    beta = np.array(BETA_ON if setting.calendar else (0.0, 0.0))
    speed = speed_path(setting.speed, rng)
    theta = THETA0 * np.exp(np.cumsum(SD_LEVEL * rng.standard_normal(K)))
    kappa = np.exp(0.3 * rng.standard_normal(G))
    kappa /= kappa.mean()
    length = MONTHS + H
    load = 1 + 0.8 * rng.uniform(size=(P, G))
    z = load[:, :, None] * ar1(rng, (P, 1), 0.97, length) + 0.5 * ar1(rng, (P, G), 0.90, length)
    entry = np.arange(MONTHS)
    year = entry // 12
    shape = np.diff(
        f0(speed[year][:, None] * np.arange(H + 1.0)[None, :], setting.timing), axis=1
    )  # (entry, age)
    cell_z = z[:, :, entry[:, None] + AGES[None, :] - 1].transpose(0, 2, 1, 3)  # (P, entry, G, age)
    hazard = (
        kappa[None, :, None]
        * (theta[year][:, None] * shape)[:, None, :]
        * np.exp(np.tensordot(beta, cell_z, 1))
    )
    cif = 1 - np.exp(-np.cumsum(hazard, axis=2))
    probs = np.concatenate([np.diff(cif, axis=2, prepend=0.0), 1 - cif[:, :, [-1]]], axis=2)
    events = rng.multinomial(n, probs)[:, :, :-1].astype(float)
    cum = np.concatenate([np.zeros((MONTHS, G, 1)), np.cumsum(events, axis=2)], axis=2)
    flat = cell_z.reshape(P, -1)
    zs = (cell_z - flat.mean(axis=1)[:, None, None, None]) / flat.std(axis=1, ddof=1)[
        :, None, None, None
    ]
    return World(n, events, cum, z, zs, speed, setting.timing)


def truth(w: World, per: int) -> dict[float, float]:
    """Realised incidence at H of every complete cohort (per = 12: years; 3: quarters)."""
    index = cohort_index(np.arange(MONTHS), per)
    done = w.cum[:, :, H].sum(axis=1)
    return {
        float(k) / (12 // per): float(done[index == k].sum() / (w.n * G * (index == k).sum()))
        for k in np.unique(index)
    }


def cohort_index(entry: Ints, per: int) -> Ints:
    """Annual cohorts by entry year; quarterly cohorts straddle the years as in the frozen code ((month - 1) // 3)."""
    return np.asarray(entry // 12 if per == 12 else (entry - 1) // 3, dtype=np.int64)


def calendar_step(w: World, s: int) -> tuple[Array, Array, float]:
    """Fixed-effects Poisson on the cells seen at s (cohort year x age and group effects, offset log exposure):
    coefficients, group effects (exposure-weighted mean 0) and the G1 statistic (events x smallest residual share)."""
    seen = AGES[None, :] <= (s - np.arange(s))[:, None]
    e, g, a = np.nonzero(np.broadcast_to(seen[:, None, :], (s, G, H)))
    exposure = w.n - w.cum[e, g, a]
    frame = pd.DataFrame(
        {
            "events": w.events[e, g, a],
            "log_exposure": np.log(exposure),
            "state": LABELS[g],
            "cell": (e // 12) * H + a,
        }
        | {f"z{j}": w.zs[j, e, g, a] for j in range(P)}
    )
    zcols = [f"z{j}" for j in range(P)]
    m = pf.fepois(
        f"events ~ {' + '.join(zcols)} | cell + state",
        data=frame,
        offset="log_exposure",
        vcov={"CRV1": "state"},
    )
    effects = m.fixef()
    by_state = pd.Series(effects[next(k for k in effects if "state" in k)], dtype=float)
    kappa = by_state.reindex(LABELS).fillna(0.0).to_numpy()
    weights = np.bincount(g, weights=exposure, minlength=G) * by_state.reindex(LABELS).notna()
    kappa = np.where(weights > 0, kappa - float(np.average(kappa, weights=weights)), 0.0)
    has = frame.events.to_numpy() > 0
    cell, state = pd.factorize(frame.cell[has])[0], g[has]
    shares = [
        pl._residual_share(w.z[j, g[has], (e + a)[has]], cell, state, frame.events.to_numpy()[has])
        for j in range(P)
    ]
    return (
        m.tidy().loc[zcols, "Estimate"].to_numpy(),
        kappa,
        float(frame.events.sum() * min(shares)),
    )


def plug_in(
    w: World, s: int, entries: Ints, beta: Array, kappa: Array, inc: Array, *, conditional: bool
) -> float:
    """Incidence at H of one cohort: each unit's incidence so far, then the fitted hazards from its age to H along
    the realised path (conditional) or with the covariates held at the mean of the cohort's cells in the last 12
    months, by group (forecast)."""
    ages = np.arange(H + 1)
    a_u = np.clip(s - entries, 0, H)
    future = ages[None, :] > a_u[:, None]
    zs = w.zs[:, entries][:, :, :, np.maximum(ages, 1) - 1]  # (P, units, G, age)
    if not conditional:
        month = entries[:, None] + AGES[None, :] - 1
        last = (month >= s - 12) & (month <= s - 1)
        mean = (w.zs[:, entries] * last[None, :, None, :]).sum(axis=(1, 3)) / last.sum()
        zs = np.where(future[None, :, None, :], mean[:, None, :, None], zs)
    contrib = (
        np.exp(np.tensordot(beta, zs, 1) + kappa[None, :, None])
        * inc[None, None, :]
        * future[:, None, :]
    )
    aj = w.cum[entries[:, None], np.arange(G)[None, :], a_u[:, None]] / w.n
    return float((1 - (1 - aj) * np.exp(-contrib.sum(axis=2))).mean())


def estimate(w: World, year: int, per: int) -> tuple[pd.DataFrame, dict[str, object]]:
    """All methods at the December snapshot of `year` for annual (per = 12) or quarterly (per = 3) cohorts."""
    s = 12 * (year + 1)
    entry = np.arange(s)
    pot = s - entry
    seen = AGES[None, :] <= pot[:, None]
    keys, inverse = np.unique(cohort_index(entry, per), return_inverse=True)
    member = (inverse[None, :] == np.arange(len(keys))[:, None]).astype(float)
    events = w.events[:s].sum(axis=1) * seen
    risk = (w.n * G - w.cum[:s, :, :-1].sum(axis=1)) * seen
    d = member @ events
    hz = d / np.maximum(member @ risk, 1.0)
    before = np.concatenate([np.ones((len(keys), 1)), np.cumprod(1 - hz, axis=1)[:, :-1]], axis=1)
    curves = np.concatenate([np.zeros((len(keys), 1)), np.cumsum(before * hz, axis=1)], axis=1)
    t = pd.DataFrame(
        {
            "cohort": keys / (12 // per),
            "A": [int(np.median(pot[inverse == i])) for i in range(len(keys))],
            "n": member.sum(axis=1) * G * w.n,
            "events": d.sum(axis=1),
        }
    )
    t["mature"] = t.A >= H
    t["a_used"] = np.minimum(t.A, H)
    t["young"] = ~t.mature
    a_used = t.a_used.to_numpy()
    t["g_a"] = t["naive"] = curves[np.arange(len(keys)), a_used]
    ages = np.arange(H + 1.0)

    def cohorts_of(cv: Array) -> list[onestage.Cohort]:
        return [
            onestage.Cohort(
                float(r.cohort),
                ages[: a + 1].copy(),
                cv[i, : a + 1].copy(),
                float(r.events),
                float(H),
                bool(r.mature),
            )
            for i, (r, a) in enumerate(zip(t.itertuples(), a_used, strict=True))
        ]

    free = cohorts_of(curves)
    t["chain"] = onestage.fit(free, 0.0).level
    beta, kappa, g1 = calendar_step(w, s)
    weight = (w.n - w.cum[:s, :, :-1]) * np.exp(
        np.tensordot(beta, w.zs[:, :s], 1) + kappa[None, :, None]
    )
    adjusted = np.concatenate(
        [
            np.zeros((len(keys), 1)),
            np.cumsum(d / np.maximum(member @ (weight.sum(axis=1) * seen), 1e-300), axis=1),
        ],
        axis=1,
    )
    cohorts = cohorts_of(adjusted)
    selection = onestage.select_model(cohorts, onestage.dispersion(cohorts))
    chain = onestage.fit(cohorts, 0.0)

    world_index = cohort_index(np.arange(MONTHS), per)

    def plugged(fit: onestage.Fit, *, conditional: bool) -> list[float]:
        # as in the frozen plug-in, a cohort keeps its entry months after the snapshot (age 0, nothing observed);
        # this concerns only the last, incomplete quarterly cohort, which is not used (A < 3)
        return [
            float(curves[i, a])
            if a >= H
            else plug_in(
                w,
                s,
                np.flatnonzero(world_index == keys[i]),
                beta,
                kappa,
                pl.increments(adjusted[i], int(a), float(fit.speeds[i]), fit.ugrid, fit.f0, H),
                conditional=conditional,
            )
            for i, a in enumerate(a_used)
        ]

    t["cal_chain"] = plugged(chain, conditional=True)
    t["cal_chain_forecast"] = plugged(chain, conditional=False)
    t["ours"] = plugged(selection.fit, conditional=True)
    t["ours_uncond"] = plugged(selection.fit, conditional=False)
    calendar_on = g1 >= pl.G1_THRESHOLD
    if (
        not calendar_on
    ):  # the guard switches the calendar step off: speed model on the unadjusted curves
        t["ours"] = t["ours_uncond"] = onestage.select_model(
            free, onestage.dispersion(free)
        ).fit.level
    t["speed"] = selection.fit.speeds
    meta: dict[str, object] = {
        "model": selection.model,
        "phi": selection.phi,
        "beta": beta,
        "g1": g1,
        "calendar_on": bool(calendar_on),
    }
    if (
        per == 12
    ):  # not part of the method: own-speed estimate and the shrinkage criterion of the scored cohorts
        own = onestage.fit(
            cohorts, 100.0
        )  # a free speed per cohort (the fit behind the dispersion)
        t["own"], t["own_speed"] = plugged(own, conditional=True), own.speeds
        t = t.join(extras.criterion(t, cohorts, selection))
        t = t.join(extras.criterion(t, cohorts, smooth(selection, chain)).add_suffix("_smooth"))
    return t, meta


def smooth(selection: onestage.Selection, chain: onestage.Fit) -> onestage.Selection:
    """The selection with the generalized-gamma reference shape that the evidence uses, in place of the pooled
    empirical one (which is piecewise linear, so second differences in log speed are noisy)."""
    a, c, scale, top = onestage.reference_parameters(chain)
    f0 = stats.gengamma.cdf(chain.ugrid, a, c, scale=scale) / stats.gengamma.cdf(
        top, a, c, scale=scale
    )
    return replace(selection, fit=replace(selection.fit, ugrid=chain.ugrid, f0=np.asarray(f0)))


def true_information(age: int, speed: float, timing: str, eps: float = 1e-4) -> float:
    """Fisher information about log speed per event, for event ages in monthly bins up to `age` under the true
    shape (no calendar effect)."""

    def log_share(log_s: float) -> Array:
        f = f0(math.exp(log_s) * np.arange(age + 1.0), timing)
        return np.asarray(np.log(np.diff(f) / f[-1]))

    x = math.log(speed)
    score = (log_share(x + eps) - log_share(x - eps)) / (2 * eps)
    return float(np.sum(np.exp(log_share(x)) * score**2))


def conformal(target: pd.DataFrame, calibration: pd.DataFrame, method: str) -> pd.DataFrame:
    """The interval rule of experiments/0011 `conformal` (M = 1): absolute log errors of quarterly cohorts at earlier
    snapshots, age within 6 months, horizon complete at the target snapshot."""
    rows = []
    for r in target.itertuples():
        c = calibration[
            ((calibration.A - r.A).abs() <= 6)
            & (calibration.A + 12 * (TARGET - calibration.snapshot) >= H)
        ]
        errors = np.sort(np.abs(c[f"r_{method}"].to_numpy()))
        j, estimate_ = len(errors), float(getattr(r, method))
        if j == 0 or estimate_ <= 0:
            rows.append({"J": j, "low": np.nan, "high": np.nan})
            continue
        q = float(errors[min(j, math.ceil((j + 1) * (1 - ALPHA))) - 1])
        rows.append(
            {
                "J": j,
                "low": max(float(r.g_a), estimate_ * math.exp(-q)),
                "high": min(1.0, estimate_ * math.exp(q)),
            }
        )
    return pd.DataFrame(rows, index=target.index).add_prefix(f"{method}_")


def calibration_errors(w: World) -> pd.DataFrame:
    """Log errors of the young quarterly cohorts at the earlier December snapshots (experiments/0011)."""
    quarterly = truth(w, 3)
    errors = []
    for year in CALIBRATION:
        t, _ = estimate(w, year, 3)
        y = t[t.young].copy()
        y["truth"] = [quarterly[k] for k in y.cohort]
        y = y[(y.truth > 0) & (y.A >= 3)]
        for m in INTERVALS:
            y[f"r_{m}"] = np.log(np.maximum(y[m], 1e-9)) - np.log(y.truth)
        errors.append(y.assign(snapshot=year)[["snapshot", "A", *(f"r_{m}" for m in INTERVALS)]])
    return pd.concat(errors, ignore_index=True)


def replicate(
    setting: Setting, rep: int, *, intervals: bool
) -> tuple[pd.DataFrame, dict[str, object]]:
    w = simulate(setting, np.random.default_rng([SEED, SETTINGS.index(setting), rep]))
    start = time.perf_counter()
    t, meta = estimate(w, TARGET, 12)
    annual = truth(w, 12)
    t["truth"] = [annual[k] for k in t.cohort]
    t["true_speed"] = w.speed[t.cohort.to_numpy().astype(int)]
    t["V_true"] = [
        true_information(int(a), float(sp), w.timing)
        for a, sp in zip(t.a_used, t.true_speed, strict=True)
    ]
    scored = t[t.young | (t.cohort == t.cohort[t.mature].max())].copy()
    if intervals:
        calibration = calibration_errors(w)
        scored = scored.join([conformal(scored, calibration, m) for m in INTERVALS])
    scored.insert(0, "rep", rep)
    scored.insert(0, "setting", setting.name)
    beta = np.asarray(meta.pop("beta"))
    return scored, meta | {
        "setting": setting.name,
        "rep": rep,
        "intervals": intervals,
        "beta0": float(beta[0]),
        "beta1": float(beta[1]),
        "seconds": time.perf_counter() - start,
    }


def _job(job: tuple[int, int]) -> tuple[pd.DataFrame, dict[str, object]]:
    setting = SETTINGS[job[0]]
    return replicate(setting, job[1], intervals=job[1] < setting.reps[1])


def results() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Scored cohorts and one row per replication, from the parts written so far."""
    pairs, meta = (
        pd.concat([pd.read_parquet(f) for f in sorted(PARTS.glob(f"{kind}_*.parquet"))])
        for kind in ("pairs", "meta")
    )
    return pairs.sort_values(["setting", "rep", "cohort"], ignore_index=True), meta.sort_values(
        ["setting", "rep"], ignore_index=True
    )


def run() -> None:
    """All replications that are not in artifacts/parts yet; a part is written after every 100 (so a stopped run
    continues where it was)."""
    PARTS.mkdir(parents=True, exist_ok=True)
    finished = (
        set(zip(*(results()[1][c] for c in ("setting", "rep")), strict=True))
        if any(PARTS.glob("meta_*.parquet"))
        else set()
    )
    jobs = [
        (i, r)
        for r in range(max(v[0] for v in REPS.values()))
        for i, x in enumerate(SETTINGS)
        if r < x.reps[0] and (x.name, r) not in finished
    ]
    session, start = time.strftime("%Y%m%d_%H%M%S"), time.perf_counter()
    buffer: list[tuple[pd.DataFrame, dict[str, object]]] = []
    with mp.get_context("spawn").Pool(WORKERS) as pool:
        for k, result in enumerate(pool.imap_unordered(_job, jobs, chunksize=2), 1):
            buffer.append(result)
            if k % 100 == 0 or k == len(jobs):
                pd.concat([x for x, _ in buffer]).to_parquet(
                    PARTS / f"pairs_{session}_{k:05d}.parquet", index=False
                )
                pd.DataFrame([m for _, m in buffer]).to_parquet(
                    PARTS / f"meta_{session}_{k:05d}.parquet", index=False
                )
                buffer = []
                (PARTS / f"seconds_{session}.txt").write_text(f"{time.perf_counter() - start:.0f}")
                log.info("%d of %d replications, %.0f s", k, len(jobs), time.perf_counter() - start)


def main(what: str) -> None:
    if what == "time":
        for setting in (SETTINGS[5], SETTINGS[-1]):
            pairs, meta = replicate(setting, 0, intervals=True)
            log.info("%s: %.1f s, model %s", setting.name, meta["seconds"], meta["model"])
            log.info("\n%s", pairs.round(4).T.to_string())
    else:
        run()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    main(sys.argv[1])
