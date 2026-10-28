"""One snapshot with the frozen method and, next to it, the round 3 additions (all POST HOC).

`snapshot(pl, d, year, ...)` calls the frozen `pl.estimate` (nothing in it changes) and adds, for the same cohorts:
  bf, cape_cod, gcc_075, gcc_050   classical reserving baselines on the incidence scale, with the chain ladder's
                                   timing curve; BF prior = horizon rate of the last mature cohort
  unit_chain                       chain ladder on cumulative hazards with each unit's own age ("+ exact ages")
  stratified                       the same with group effects
  cal_chain_forecast               calendar-adjusted chain ladder on the forecast path (covariates held at their
                                   mean of the last 12 months); `cal_chain` of the frozen code is the realised path
  D, V, own_var, omega2, tau2      the per-cohort quantities of the shrinkage proposition (see `criterion`)
`pl` is the pipeline module of the run (experiments/0012 for mortgages with corrected weights, patents).
"""

from __future__ import annotations

from types import ModuleType

import numpy as np
import pandas as pd

from maturity import onestage, reserving

Step1 = tuple[np.ndarray, pd.Series, np.ndarray]


def group_effects(pl: ModuleType, d: object, s0: int) -> pd.Series:
    """Group effects of events ~ Poisson(exposure x cell effect x group effect) on the cells seen at S, by iterative
    proportional fitting (the code of experiments/0022)."""
    cells = d.cells  # type: ignore[attr-defined]
    c = cells[(cells.mi <= s0) & (cells.vintage <= s0 // 12)]
    cell = pd.factorize(c.vintage.to_numpy() * 1000 + c.age.to_numpy())[0]
    group, names = pd.factorize(c.state.to_numpy())
    events, exposure = c.events.to_numpy(dtype=float), c.exposure.to_numpy(dtype=float)
    by_cell, by_group = np.bincount(cell, events), np.bincount(group, events)
    k = np.zeros(len(names))
    for _ in range(500):
        alpha = by_cell / np.maximum(np.bincount(cell, exposure * np.exp(k[group])), 1e-300)
        new = np.log(np.maximum(by_group, 1e-300) / np.bincount(group, exposure * alpha[cell]))
        done = float(np.max(np.abs(new - k)[by_group > 0])) < 1e-10
        k = new
        if done:
            break
    k = k - np.average(k[by_group > 0], weights=np.bincount(group, exposure)[by_group > 0])
    return pd.Series(k[by_group > 0], index=names[by_group > 0])


def cohorts_of(t: pd.DataFrame, curves: dict[float, np.ndarray], top: int) -> list[onestage.Cohort]:
    ages = np.arange(top + 1, dtype=float)
    return [
        onestage.Cohort(
            r.cohort,
            ages[: int(r.a_used) + 1].copy(),
            curves[r.cohort][: int(r.a_used) + 1].copy(),
            float(r.events),
            float(r.h),
            bool(r.mature),
        )
        for r in t.itertuples()
    ]


def unit_level(
    pl: ModuleType,
    d: object,
    t: pd.DataFrame,
    s0: int,
    fit1: Step1,
    *,
    conditional: bool = True,
    selection: onestage.Fit | None = None,
) -> tuple[np.ndarray, list[onestage.Cohort], dict[float, np.ndarray]]:
    """Plug-in estimate per cohort from the hazards adjusted by `fit1`, with equal speeds unless a fitted speed
    model is given. Returns the estimates (incidence so far for mature cohorts), the cohort curves and the adjusted
    curves."""
    adjusted = pl.adjusted_curves(d, s0, fit1, "year")
    cohorts = cohorts_of(t, adjusted, pl.AMAX)
    fit = onestage.fit(cohorts, 0.0) if selection is None else selection
    cells = d.cells  # type: ignore[attr-defined]
    young = t.young.to_numpy()
    by_cohort = {float(k): g for k, g in cells[cells["year"].isin(t.cohort[young])].groupby("year")}
    out = t.g_a.to_numpy(dtype=float).copy()
    for i, r in enumerate(t.itertuples()):
        if young[i]:
            inc = pl.increments(
                adjusted[r.cohort], int(r.a_used), float(fit.speeds[i]), fit.ugrid, fit.f0, int(r.h)
            )
            out[i] = pl.plug_in(
                d, by_cohort[r.cohort], s0, fit1, inc, int(r.h), conditional=conditional
            )[0]
    return out, cohorts, adjusted


def reserving_baselines(
    t: pd.DataFrame, curves: dict[float, np.ndarray], top: int
) -> dict[str, np.ndarray]:
    """Bornhuetter-Ferguson and (Generalized) Cape Cod with the timing curve of the cohort-level chain ladder."""
    chain = onestage.fit(cohorts_of(t, curves, top), 0.0)

    def share(age: np.ndarray) -> np.ndarray:
        return np.asarray(np.interp(age, chain.ugrid, chain.f0, right=1.0), dtype=float)

    g, n = t.g_a.to_numpy(dtype=float), t.n.to_numpy(dtype=float)
    f_age, f_target = share(t.a_used.to_numpy(dtype=float)), share(t.h.to_numpy(dtype=float))
    mature = t[t.mature]
    last_mature = float(curves[float(mature.cohort.iloc[-1])][int(mature.h.iloc[-1])])
    keys = t.cohort.to_numpy(dtype=float)
    out = {
        "bf": reserving.bornhuetter_ferguson(g, f_age, f_target, last_mature),
        "cape_cod": reserving.bornhuetter_ferguson(
            g, f_age, f_target, reserving.cape_cod_prior(n, g, f_age)
        ),
    }
    for decay in (0.75, 0.5):
        prior = reserving.cape_cod_prior(n, g, f_age, keys, decay)
        out[f"gcc_{int(100 * decay):03d}"] = reserving.bornhuetter_ferguson(
            g, f_age, f_target, prior
        )
    young = t.young.to_numpy()
    return {k: np.where(young, v, g) for k, v in out.items()}


def criterion(
    t: pd.DataFrame, cohorts: list[onestage.Cohort], selection: onestage.Selection
) -> pd.DataFrame:
    """Per cohort, the quantities of the shrinkage proposition (own speed against a borrowed speed).
    D        events so far
    V        observed information about log speed per event (information / D), at the fitted speed and shape
    own_var  1 / (D V): variance of the cohort's own log-speed estimate (own_var_phi: times the dispersion)
    omega2   increment variance of the random-walk candidate with the largest evidence at this snapshot
    tau2     posterior variance of the older neighbour's log speed under that random walk (Laplace approximation
             with the tempered information, first cohort anchored at speed 1)
    worth    omega2 + tau2 > own_var_phi  (the proposition's condition, with the over-dispersion that the
             selection uses; worth_untempered uses own_var)"""
    fit, phi = selection.fit, selection.phi
    info = np.array(
        [
            onestage.speed_information(c, float(np.log(s)), fit)
            for c, s in zip(cohorts, fit.speeds, strict=True)
        ]
    )
    events = np.array([c.events for c in cohorts])
    rw = {k: v for k, v in selection.log_evidence.items() if k.startswith("rw(")}
    omega2 = float(max(rw, key=lambda k: rw[k])[3:-1])
    k = len(cohorts)
    difference = np.eye(k) - np.eye(k, k=-1)
    posterior = np.linalg.inv(np.diag(info / phi) + difference.T @ difference / omega2)
    tau2 = np.concatenate([[np.nan], np.diag(posterior)[:-1]])
    with np.errstate(divide="ignore", invalid="ignore"):
        own = 1 / info
    return pd.DataFrame(
        {
            "D": events,
            "V": np.divide(info, events, out=np.full(k, np.nan), where=events > 0),
            "own_var": own,
            "own_var_phi": phi * own,
            "omega2": omega2,
            "tau2": tau2,
            "worth": omega2 + tau2 > phi * own,
            "worth_untempered": omega2 + tau2 > own,
        },
        index=t.index,
    )


def snapshot(
    pl: ModuleType, d: object, year: int, *, horizon: int, cutoff: int | None = None
) -> tuple[pd.DataFrame, dict[str, object]]:
    t, meta = pl.estimate(d, year, horizon=horizon, cutoff=cutoff)
    s0 = int(pl.month_idx(year * 100 + 12)) - 1
    _, curves = pl.observed(d, s0 + 1, "year")
    for name, values in reserving_baselines(t, curves, pl.AMAX).items():
        t[name] = values
    k = len(d.zcols)  # type: ignore[attr-defined]
    zero = np.zeros(k)
    t["unit_chain"] = unit_level(pl, d, t, s0, (zero, pd.Series(dtype=float), zero))[0]
    t["stratified"] = unit_level(pl, d, t, s0, (zero, group_effects(pl, d, s0), zero))[0]
    fit1 = pl.step1(d, s0)
    realised, cohorts, _ = unit_level(pl, d, t, s0, fit1)
    assert np.allclose(realised, t.cal_chain.to_numpy(), rtol=1e-9, atol=1e-12), (
        "not the frozen cal_chain"
    )
    t["cal_chain_forecast"] = unit_level(pl, d, t, s0, fit1, conditional=False)[0]
    selection = onestage.select_model(cohorts, onestage.dispersion(cohorts))
    assert selection.model == meta["model"], "not the frozen speed model"
    t = t.join(criterion(t, cohorts, selection))
    a, c, scale, _ = onestage.reference_parameters(onestage.fit(cohorts, 0.0))
    extra = {
        "log_evidence": selection.log_evidence,
        "gengamma_a": round(a, 4),
        "gengamma_c": round(c, 4),
        "gengamma_scale": round(scale, 2),
        "early_exponent_a_times_c": round(a * c, 3),
        "omega2_best_random_walk": float(t.omega2.iloc[0]),
    }
    return t, {key: value for key, value in meta.items() if key != "f0"} | extra
