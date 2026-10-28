"""One-stage penalized estimator of cohort speeds and eventual rates (T2 note, Update 5).

Each cohort c is observed as a cumulative-incidence curve G_c on a time axis x (loan age, or a calendar-adjusted
operational clock) up to its follow-up. With a pooled reference shape F0 and a speed s_c, the cohort's event ages
follow the multinomial shape likelihood  l_c(s) = D_c * sum_j q_cj * log[(F0(s x_j) - F0(s x_{j-1})) / F0(s x_A)],
where q_cj is the observed share of the cohort's events in interval j and D_c its number of events (the level p_c is
profiled out: p_c = G_c(A) / F0(s x_A)). Speeds follow a random walk across cohorts:

    maximise  sum_c l_c(s_c)  -  sum_c (log s_c - log s_{c-1})^2 / (2 * omega2 * dt_c).

omega2 = 0 forces every speed to 1, which is chain ladder exactly. F0 is pooled over the cohorts already followed to
the horizon ("mature"), aligned by their speeds, and updated by alternation.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from functools import lru_cache
from typing import Final

import numpy as np
from numpy.typing import NDArray
from scipy import optimize, stats

__all__ = [
    "Cohort",
    "Fit",
    "Selection",
    "dispersion",
    "extended_reference",
    "fit",
    "predict_increment",
    "reference_parameters",
    "select_model",
    "speed_information",
]

Array = NDArray[np.float64]
LOG_S_BOUND: Final = float(np.log(4.0))
N_GRID: Final = 1500
RW_GRID: Final = (0.001, 0.003, 0.01, 0.03, 0.1)
CP_PRIOR_SD: Final = 0.5
CP_LAST: Final = 8


@dataclass(frozen=True, slots=True)
class Cohort:
    key: float  # position on the cohort axis, in years (e.g. 2007.0, 2007.25)
    x: Array  # time axis at the observed ages 0..A (x[0] = 0, increasing)
    g: Array  # cumulative incidence at those ages
    events: float  # number of events behind the curve
    x_h: float  # axis value at the horizon H
    mature: bool  # followed to the horizon (x[-1] == x_h)


@dataclass(frozen=True, slots=True)
class Fit:
    speeds: Array
    level: Array  # estimated incidence at the horizon
    ugrid: Array
    f0: Array
    omega2: float


@dataclass(frozen=True, slots=True)
class Selection:
    fit: Fit
    model: str  # "chain", "rw(<omega2>)" or "cp"
    log_evidence: dict[str, float]
    phi: float = 1.0  # over-dispersion used to temper the evidence


def fit(cohorts: Sequence[Cohort], omega2: float, *, n_alternations: int = 3) -> Fit:
    """Cohorts must be ordered by `key`. Raises ValueError if no cohort is mature."""
    mature = [i for i, c in enumerate(cohorts) if c.mature and c.g[-1] > 0]
    if not mature:
        raise ValueError(
            "no mature cohort: the reference shape cannot be normalised at the horizon"
        )
    ugrid = np.linspace(0.0, 4.0 * max(cohorts[i].x_h for i in mature), N_GRID)
    log_s = np.zeros(len(cohorts))
    f0 = _pooled(cohorts, mature, log_s, ugrid)
    if omega2 > 0:
        for _ in range(n_alternations):
            log_s = _optimise(cohorts, log_s, omega2, ugrid, f0)
            log_s = log_s - float(np.mean(log_s[mature]))  # mature speeds have geometric mean 1
            f0 = _pooled(cohorts, mature, log_s, ugrid)
    speeds = np.exp(log_s)
    level = np.array([_level(c, s, ugrid, f0) for c, s in zip(cohorts, speeds, strict=True)])
    return Fit(speeds, level, ugrid, f0, omega2)


def predict_increment(result: Fit, index: int, cohort: Cohort, x_from: float, x_to: float) -> float:
    """Predicted rise of the cohort's cumulative incidence between two points of its time axis."""
    s = result.speeds[index]
    f_from, f_to = (
        float(np.interp(s * x, result.ugrid, result.f0, right=1.0)) for x in (x_from, x_to)
    )
    base = float(np.interp(s * cohort.x[-1], result.ugrid, result.f0, right=1.0))
    return float(cohort.g[-1]) * (f_to - f_from) / max(base, 1e-9)


def _level(c: Cohort, s: float, ugrid: Array, f0: Array) -> float:
    if c.mature:
        return float(c.g[-1])
    top = float(np.interp(s * c.x_h, ugrid, f0, right=1.0))
    base = float(np.interp(s * c.x[-1], ugrid, f0, right=1.0))
    return float(min(1.0, c.g[-1] * top / max(base, 1e-9)))


def _pooled(cohorts: Sequence[Cohort], mature: list[int], log_s: Array, ugrid: Array) -> Array:
    """F0(u) = sum_c G_c(H) * min(1, G_c(x = u / s_c) / G_c(H)) / sum_c G_c(H) over mature cohorts."""
    num, den = np.zeros_like(ugrid), 0.0
    for i in mature:
        c = cohorts[i]
        g = np.interp(ugrid / np.exp(log_s[i]), c.x, c.g, right=c.g[-1])
        num += np.minimum(g, c.g[-1])
        den += float(c.g[-1])
    return np.minimum(num / den, 1.0)


def _shape_loglik(c: Cohort, log_s: float, ugrid: Array, f0: Array) -> float:
    if c.g[-1] <= 0 or len(c.x) < 3:
        return 0.0
    fx = np.interp(np.exp(log_s) * c.x, ugrid, f0, right=1.0)
    if fx[-1] <= 1e-12:
        return -1e9
    q = np.diff(c.g) / c.g[-1]
    seen = q > 0
    return float(c.events * np.sum(q[seen] * np.log(np.maximum(np.diff(fx)[seen] / fx[-1], 1e-12))))


def _optimise(
    cohorts: Sequence[Cohort],
    start: Array,
    omega2: float,
    ugrid: Array,
    f0: Array,
    *,
    anchored: bool = False,
) -> Array:
    keys = np.array([c.key for c in cohorts])
    scale = 2 * omega2 * np.maximum(np.diff(keys), 1e-6)
    eps = 1e-3

    def objective(v: Array) -> tuple[float, Array]:
        ll = np.array(
            [_shape_loglik(c, float(x), ugrid, f0) for c, x in zip(cohorts, v, strict=True)]
        )
        up = np.array(
            [_shape_loglik(c, float(x) + eps, ugrid, f0) for c, x in zip(cohorts, v, strict=True)]
        )
        dn = np.array(
            [_shape_loglik(c, float(x) - eps, ugrid, f0) for c, x in zip(cohorts, v, strict=True)]
        )
        d = np.diff(v)
        grad = -(up - dn) / (2 * eps)
        grad[1:] += 2 * d / scale
        grad[:-1] -= 2 * d / scale
        penalty = float(np.sum(d**2 / scale))
        if anchored:  # the first cohort is tied to speed 1 with the same variance
            penalty += float(v[0] ** 2 / (2 * omega2))
            grad[0] += v[0] / omega2
        return float(-ll.sum() + penalty), grad

    res = optimize.minimize(
        objective,
        start,
        jac=True,
        method="L-BFGS-B",
        bounds=[(-LOG_S_BOUND, LOG_S_BOUND)] * len(cohorts),
    )
    return np.asarray(res.x, dtype=np.float64)


def dispersion(
    cohorts: Sequence[Cohort], *, bin_width: int = 3, min_expected: float = 1.0
) -> float:
    """Over-dispersion phi = Pearson X^2 / df of the most flexible fit (a free speed per cohort), on cohort x age-bin
    cells (bin_width consecutive ages; 3 = quarters when ages are months). Never below 1."""
    free = fit(cohorts, 100.0)
    chi2, cells, used = 0.0, 0, 0
    for c, s in zip(cohorts, free.speeds, strict=True):
        if c.g[-1] <= 0 or len(c.x) < 2 * bin_width + 1:
            continue
        edges = np.arange(0, len(c.x), bin_width)
        if edges[-1] != len(c.x) - 1:
            edges = np.append(edges, len(c.x) - 1)
        fx = np.interp(s * c.x[edges], free.ugrid, free.f0, right=1.0)
        if fx[-1] <= 1e-12:
            continue
        expected = c.events * np.diff(fx) / fx[-1]
        observed = c.events * np.diff(c.g[edges]) / c.g[-1]
        ok = expected >= min_expected
        chi2 += float(np.sum((observed[ok] - expected[ok]) ** 2 / expected[ok]))
        cells += int(ok.sum())
        used += 1
    df = cells - 2 * used  # per cohort: total fixed (level) and one speed
    return max(1.0, chi2 / df) if df > 0 else 1.0


def select_model(cohorts: Sequence[Cohort], phi: float = 1.0) -> Selection:
    """Choose the speed model by tempered marginal likelihood at the snapshot (T2 note, Updates 7-9).

    Candidates: chain ladder (all speeds 1); random walk RW(omega2) on a grid, Laplace evidence
    l(x) - |Dx|^2 / (2 omega2) - 0.5 log det(omega2 H + D'D) (D = first differences, anchored at speed 1); and a recent
    change point: cohorts after c0 share one shift delta ~ N(0, 0.5^2), c0 uniform over the last 8 cohorts.
    Every candidate is scored against ONE common reference shape: a generalized gamma fitted to the mature reference
    under chain-ladder alignment, held fixed (re-aligning F0 per candidate would give flexible models an in-sample
    advantage the Laplace term does not charge). `phi` > 1 tempers the likelihood (over-dispersed Poisson):
    log-likelihood and Hessian are divided by phi, i.e. each cohort's events count as events / phi.
    The selected model is then refitted with its own pooled reference.
    """
    tempered = [replace(c, events=c.events / phi) for c in cohorts] if phi != 1.0 else list(cohorts)
    chain = fit(cohorts, 0.0)
    common = _smooth_reference(chain)
    evidence = {"chain": _total_loglik(tempered, np.zeros(len(cohorts)), common)}
    for omega2 in RW_GRID:
        log_s = _optimise(
            tempered, np.zeros(len(cohorts)), omega2, common.ugrid, common.f0, anchored=True
        )
        evidence[f"rw({omega2})"] = _rw_evidence(
            tempered, replace(common, speeds=np.exp(log_s), omega2=omega2)
        )
    evidence["cp"] = _change_point(tempered, common)[1]
    best = max(evidence, key=lambda k: evidence[k])
    if best == "chain":
        chosen = chain
    elif best == "cp":
        chosen = _change_point(tempered, chain)[0]
    else:
        chosen = fit(tempered, float(best[3:-1]))
    return Selection(chosen, best, {k: round(v, 3) for k, v in evidence.items()}, phi)


def reference_parameters(chain: Fit) -> tuple[float, float, float, float]:
    """Generalized gamma fitted to the chain-ladder reference shape (binned, right-truncated maximum likelihood):
    (a, c, scale) in the scipy.stats.gengamma form, and the point u where the empirical reference reaches 1."""
    f0, ugrid = chain.f0, chain.ugrid
    top = int(np.argmax(f0 >= 0.999)) if (f0 >= 0.999).any() else len(f0) - 1
    step = max(1, top // 150)
    edges = ugrid[: top + 1 : step]
    mass = np.maximum(np.diff(f0[: top + 1 : step]), 0)
    mass = mass / max(mass.sum(), 1e-12)
    med = float(np.interp(0.5, f0[: top + 1], ugrid[: top + 1]))

    def nll(theta: Array) -> float:
        a, c, scale = np.exp(theta)
        cdf = stats.gengamma.cdf(edges, a, c, scale=scale)
        return float(
            -np.sum(mass * np.log(np.maximum(np.diff(cdf) / max(cdf[-1], 1e-300), 1e-300)))
        )

    res = optimize.minimize(
        nll,
        np.log([1.0, 1.5, med]),
        method="Nelder-Mead",
        options={"xatol": 1e-6, "fatol": 1e-10, "maxiter": 4000},
    )
    a, c, scale = np.exp(res.x)
    return float(a), float(c), float(scale), float(edges[-1])


def _smooth_reference(chain: Fit) -> Fit:
    """The fitted generalized gamma, normalised to 1 at the point where the empirical reference reaches 1; not
    capped beyond it."""
    a, c, scale, u_top = reference_parameters(chain)
    smooth = stats.gengamma.cdf(chain.ugrid, a, c, scale=scale) / stats.gengamma.cdf(
        u_top, a, c, scale=scale
    )
    return replace(chain, f0=np.asarray(smooth, dtype=np.float64))


def extended_reference(ugrid: Array, f0: Array) -> Callable[[Array], Array]:
    """POST HOC (round 4 of the reviews; not used by the frozen estimator). F0 on [0, inf): the pooled reference up
    to its horizon H, and beyond it the generalized gamma G of `reference_parameters`, scaled to continue it:
    F0(u) = F0(H) G(u) / G(H). H is the horizon of the mature cohorts (`fit` builds ugrid on [0, 4H]).
    The frozen plug-in keeps the pooled reference beyond H. Mature cohorts are cut at H, so that reference is flat
    there, and a cohort with speed s > 1, which needs F0(sH), comes out too low."""
    a, c, scale, horizon = _tail_parameters(ugrid.tobytes(), f0.tobytes())
    at_horizon = float(np.interp(horizon, ugrid, f0))
    tail_at_horizon = float(stats.gengamma.cdf(horizon, a, c, scale=scale))

    def reference(u: Array) -> Array:
        u = np.asarray(u, dtype=np.float64)
        inside = np.interp(u, ugrid, f0)
        tail = at_horizon * stats.gengamma.cdf(u, a, c, scale=scale) / max(tail_at_horizon, 1e-300)
        return np.asarray(np.where(u <= horizon, inside, tail), dtype=np.float64)

    return reference


@lru_cache(maxsize=256)
def _tail_parameters(ugrid: bytes, f0: bytes) -> tuple[float, float, float, float]:
    grid, shape = np.frombuffer(ugrid, dtype=np.float64), np.frombuffer(f0, dtype=np.float64)
    a, c, scale, _ = reference_parameters(Fit(np.ones(1), np.ones(1), grid, shape, 0.0))
    return a, c, scale, float(grid[-1]) / 4.0


def speed_information(cohort: Cohort, log_s: float, result: Fit, eps: float = 5e-3) -> float:
    """Observed information of the shape log-likelihood about the cohort's log speed (minus its second derivative,
    with the cohort's event count as weight). 1 / information is the variance of the cohort's own speed estimate."""
    mid, up, down = (
        _shape_loglik(cohort, log_s + step, result.ugrid, result.f0) for step in (0.0, eps, -eps)
    )
    return max(0.0, -(up - 2 * mid + down) / eps**2)


def _total_loglik(cohorts: Sequence[Cohort], log_s: Array, f: Fit) -> float:
    return float(
        sum(_shape_loglik(c, float(x), f.ugrid, f.f0) for c, x in zip(cohorts, log_s, strict=True))
    )


def _rw_evidence(cohorts: Sequence[Cohort], f: Fit) -> float:
    x = np.log(f.speeds)
    eps = 5e-3
    hess = np.empty(len(cohorts))
    for i, c in enumerate(cohorts):
        mid, up, dn = (_shape_loglik(c, float(x[i]) + d, f.ugrid, f.f0) for d in (0.0, eps, -eps))
        hess[i] = max(0.0, -(up - 2 * mid + dn) / eps**2)
    k = len(cohorts)
    d = np.eye(k) - np.eye(k, k=-1)  # first row anchors the first cohort at speed 1
    dx = d @ x
    _, logdet = np.linalg.slogdet(f.omega2 * np.diag(hess) + d.T @ d)
    return float(_total_loglik(cohorts, x, f) - dx @ dx / (2 * f.omega2) - 0.5 * logdet)


def _change_point(cohorts: Sequence[Cohort], chain: Fit) -> tuple[Fit, float]:
    k = len(cohorts)
    young = [i for i, c in enumerate(cohorts) if not c.mature]
    starts = list(
        range(max(min(young, default=k), k - CP_LAST), k)
    )  # c0 = start index of the shifted block
    if not starts:
        return chain, -np.inf
    var = CP_PRIOR_SD**2
    logev, deltas = [], []
    for c0 in starts:
        block = cohorts[c0:]

        def neg(delta: float, block: Sequence[Cohort] = block) -> float:
            return -sum(
                _shape_loglik(c, delta, chain.ugrid, chain.f0) for c in block
            ) + delta**2 / (2 * var)

        res = optimize.minimize_scalar(neg, bounds=(-LOG_S_BOUND, LOG_S_BOUND), method="bounded")
        delta = float(res.x)
        eps = 5e-3
        curv = max(0.0, (neg(delta + eps) - 2 * neg(delta) + neg(delta - eps)) / eps**2 - 1 / var)
        rest = sum(_shape_loglik(c, 0.0, chain.ugrid, chain.f0) for c in cohorts[:c0])
        logev.append(rest - neg(delta) - 0.5 * np.log(var * curv + 1))
        deltas.append(delta)
    logev_arr = np.array(logev)
    top = float(logev_arr.max())
    weights = np.exp(logev_arr - top)
    weights /= weights.sum()
    log_s = np.zeros(k)
    for wgt, c0, delta in zip(weights, starts, deltas, strict=True):
        log_s[c0:] += wgt * delta
    speeds = np.exp(log_s)
    level = np.zeros(k)
    for wgt, c0, delta in zip(weights, starts, deltas, strict=True):
        s_vec = np.ones(k)
        s_vec[c0:] = np.exp(delta)
        level += wgt * np.array(
            [_level(c, s, chain.ugrid, chain.f0) for c, s in zip(cohorts, s_vec, strict=True)]
        )
    evidence = top + float(np.log(np.mean(np.exp(logev_arr - top))))
    return Fit(speeds, level, chain.ugrid, chain.f0, float("nan")), evidence
