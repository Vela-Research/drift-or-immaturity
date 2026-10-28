"""Freddie Mac backtests (pipeline development). DATASET: official 50k Sample, DEVELOPMENT ONLY (not paper numbers).

For each snapshot S: keep only what was observable at S (loans originated by S; events after S become censoring at S),
estimate each vintage's eventual default incidence, and score it against the truth from all data to 2026-03.
Eventual incidence of a vintage = Aalen-Johansen cumulative incidence of default at H = 84 months (prepay competes).
Event = the primary definition (first unflagged 180+ month, or credit zero-balance code).
Methods (calendar-free):
  naive          observed incidence at the vintage's follow-up
  chain ladder   observed / F0 at the follow-up; F0 = event-weighted pooled shape of all vintages mature at S
  drop recent    mean incidence of the 3 latest mature vintages (practice: train on mature cohorts only)
  ours           per-vintage speed against the POOLED reference (mature vintages aligned by jointly fitted speeds)
  ours+EB        empirical-Bayes speeds: random walk on log s(c), variance by marginal likelihood (maturity.eb);
                 omega2 = 0 is chain ladder
Usage: backtest.py  -> artifacts/backtest_<snapshot>.csv, artifacts/backtest_summary.json
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Final

import calendar_methods as cal
import numpy as np
import pandas as pd

from maturity import data, eb

HERE: Final = Path(__file__).resolve().parent
H: Final = 84  # truth horizon in months
AMAX: Final = 200
SNAPSHOTS: Final = {"2008-12": 200812, "2012-12": 201212, "2016-12": 201612, "2019-12": 201912}
SPEEDS: Final = np.linspace(0.4, 3.0, 131)
CAL_MODES: Final = {"cond": "cond", "uncond": "uncond_mean", "uncond12": "uncond_last12"}
CAL: Final = [f"cal_{m}_{k}" for m in ("chain", "eb") for k in CAL_MODES]
METHODS: Final = ["naive", "chain", "drop_recent", "ours", "ours_eb", *CAL]
GRID: Final = np.arange(AMAX + 1)


def month_idx(yyyymm: pd.Series | int) -> pd.Series | int:
    return (yyyymm // 100) * 12 + yyyymm % 100


def aj_default_cif(dur: np.ndarray, kind: np.ndarray) -> np.ndarray:
    """Aalen-Johansen cumulative incidence of default by loan age 0..AMAX (prepay competes)."""
    dur = np.clip(dur, 0, AMAX).astype(int)
    d, p, c = (
        np.bincount(dur[kind == k], minlength=AMAX + 1) for k in ("default", "prepay", "censored")
    )
    risk = len(dur) - np.concatenate([[0], np.cumsum(d + p + c)[:-1]])
    r = np.maximum(risk, 1)
    surv = np.concatenate([[1.0], np.cumprod(1 - (d + p) / r)[:-1]])
    return np.cumsum(surv * d / r)


def main() -> None:
    d = pd.read_parquet(data.DATA_DIR / "processed" / "freddie_sample_loans.parquet")
    d = d[
        d.event_age_clean.notna() & d.first_payment.notna() & d.orig_year.between(1999, 2019)
    ].copy()
    d["om"] = month_idx(d.first_payment.astype(int)) - 1  # origination month index
    d["ev_mi"] = month_idx(d.event_period_clean.astype(int))
    truth = {
        v: aj_default_cif(g.event_age_clean.to_numpy(), g.event_type_clean.to_numpy())[H]
        for v, g in d.groupby("orig_year")
    }
    out: dict[str, object] = {}
    cells = cal.load_cells()
    for name, s_yyyymm in SNAPSHOTS.items():
        t, meta = _snapshot(d, month_idx(s_yyyymm), truth, cells, name)
        t.round(4).to_csv(HERE / "artifacts" / f"backtest_{name}.csv", index=False)
        out[name] = _summary(t) | meta
        print(f"\n== snapshot {name}  EB omega2={meta['eb_omega2']} kappa={meta['eb_kappa']}")
        print(
            t[t.A < H][["vintage", "A", "events", "truth", *METHODS]]
            .round(4)
            .to_string(index=False)
        )
        print(json.dumps({k: out[name][k] for k in ("all_young", "best_method")}))  # type: ignore[index]
        print(
            t[t.A < H][
                ["vintage", "A", "events", "maturity_m", "inflation_gg", "inflation_weibull"]
            ]
            .round(3)
            .to_string(index=False)
        )
    (HERE / "artifacts" / "backtest_summary.json").write_text(json.dumps(out, indent=2))


def _snapshot(
    d: pd.DataFrame, s: int, truth: dict[int, float], cells: pd.DataFrame, name: str
) -> tuple[pd.DataFrame, dict[str, object]]:
    x = d[d.om <= s].copy()
    pot = s - x.om  # potential follow-up in months at S
    seen = x.ev_mi <= s
    x["dur"] = np.where(seen, x.event_age_clean, pot)
    x["kind"] = np.where(seen, x.event_type_clean, "censored")
    curves, info = {}, []
    for v, g in x.groupby("orig_year"):
        curves[v] = aj_default_cif(g.dur.to_numpy(), g.kind.to_numpy())
        a_full = int(np.median(s - g.om))  # follow-up A_c: median potential follow-up
        info.append(
            {
                "vintage": v,
                "A": a_full,
                "events": int((g.kind == "default").sum()),
                "a_used": min(a_full, H),
            }
        )
    t = pd.DataFrame(info)
    t["g_a"] = [curves[r.vintage][min(r.a_used, AMAX)] for r in t.itertuples()]
    t["truth"] = t.vintage.map(truth)
    mature = t[t.A >= H].vintage.tolist()
    young = t.A < H
    f0_chain = _pooled(curves, mature, {})
    f0 = _aligned_reference(curves, mature)
    t["naive"] = t.g_a
    t["chain"] = np.where(
        young, _cap(t.g_a / np.maximum(_f0(f0_chain, t.a_used.to_numpy()), 1e-9)), t.g_a
    )
    t["drop_recent"] = np.where(young, np.mean([curves[v][H] for v in sorted(mature)[-3:]]), t.g_a)
    t["s_hat"] = [
        _fit_speed(curves[r.vintage], H if r.A >= H else r.a_used, f0) for r in t.itertuples()
    ]
    t["ours"] = np.where(young, _cap(t.g_a / np.maximum(_f0(f0, t.s_hat * t.a_used), 1e-9)), t.g_a)
    ok = t.s_hat.notna() & (t.events >= 10)
    y = np.where(ok, np.log(t.s_hat.where(ok, 1.0)), np.nan)
    fit = eb.eb_log_speeds(y.astype(float), t.events.to_numpy(dtype=float))
    t["s_eb"] = np.exp(fit.x)
    t["ours_eb"] = np.where(
        young, _cap(t.g_a / np.maximum(_f0(f0, t.s_eb * t.a_used), 1e-9)), t.g_a
    )
    meta: dict[str, object] = {
        "eb_omega2": float(fit.omega2),
        "eb_kappa": float(fit.kappa),
        "mature_vintages": mature,
    }
    fit_cal = cal.fit_beta(cells, s - 1)  # cells use a 0-based month index
    meta["calendar_beta"] = [round(float(b), 3) for b in fit_cal[0]]
    meta["calendar_beta_se_clustered_state"] = [round(float(b), 3) for b in fit_cal[3]]
    for tag, mode in CAL_MODES.items():
        clk = cal.clocks(cells, s - 1, fit_cal, mode)
        chain_cal, eb_cal, extra = cal.estimate(curves, t, clk)
        t[f"cal_chain_{tag}"] = np.where(young, t.vintage.map(chain_cal), t.g_a)
        t[f"cal_eb_{tag}"] = np.where(young, t.vintage.map(eb_cal), t.g_a)
        meta[f"cal_eb_{tag}_hyper"] = {"omega2": extra["omega2"], "kappa": extra["kappa"]}
        if (
            tag == "cond"
        ):  # T3 diagnostic from the calendar-adjusted pooled reference (generalized gamma fit)
            diag: pd.DataFrame = extra["diag"]  # type: ignore[assignment]
            for col in ("maturity_m", "inflation_gg", "inflation_weibull"):
                t[col] = t.vintage.map(diag[col])
            meta["f0_gengamma_a_c_scale"] = [round(float(v), 4) for v in extra["gg"][:3]]  # type: ignore[index]
            meta["f0_weibull_c_scale"] = [round(float(v), 4) for v in extra["weibull"][1:3]]  # type: ignore[index]
            _plot_f0(extra, name)
    return t.drop(columns=["g_a"]), meta


def _plot_f0(extra: dict[str, object], name: str) -> None:
    """Visual check of the smooth F0 fits against the pooled calendar-adjusted reference."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy import stats

    ugrid: np.ndarray = extra["ugrid"]  # type: ignore[assignment]
    f0: np.ndarray = extra["f0"]  # type: ignore[assignment]
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.plot(ugrid, f0, color="#0b0b0b", lw=2, label="pooled reference (empirical)")
    u_top = 1.0
    for key, color, label in (
        ("gg", "#2a78d6", "generalized gamma"),
        ("weibull", "#eb6834", "Weibull"),
    ):
        a, c, scale, u_max = extra[key]  # type: ignore[misc]
        u_top = u_max
        fit = stats.gengamma.cdf(ugrid, a, c, scale=scale) / stats.gengamma.cdf(
            u_max, a, c, scale=scale
        )
        ax.plot(ugrid, np.minimum(1, fit), color=color, lw=1.5, ls="--", label=label)
    ax.set(
        xlim=(0, u_top * 1.05),
        xlabel="operational time (calendar-adjusted months)",
        ylabel="F0",
        title=f"snapshot {name}",
    )
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    (HERE / "figures").mkdir(exist_ok=True)
    fig.savefig(HERE / "figures" / f"f0_fit_{name}.png", dpi=130)
    plt.close(fig)


def _cap(p: pd.Series | np.ndarray) -> np.ndarray:
    return np.minimum(1.0, np.asarray(p, dtype=float))


def _pooled(
    curves: dict[int, np.ndarray], mature: list[int], speed: dict[int, float]
) -> np.ndarray:
    """F0(u), u = 0..H: sum_c G_c(u / s_c) / sum_c G_c(H / s_c) over mature vintages (event-weighted pooling)."""
    num = sum(np.interp(np.arange(H + 1) / speed.get(v, 1.0), GRID, curves[v]) for v in mature)
    den = sum(np.interp(H / speed.get(v, 1.0), GRID, curves[v]) for v in mature)
    return np.minimum(num / den, 1.0)


def _aligned_reference(curves: dict[int, np.ndarray], mature: list[int]) -> np.ndarray:
    """Pooled reference with mature-vintage speeds fitted jointly (alternating, speeds normalised to geometric mean 1)."""
    f0 = _pooled(curves, mature, {})
    for _ in range(3):
        sp = {v: _fit_speed(curves[v], H, f0) for v in mature}
        valid = {v: s for v, s in sp.items() if not np.isnan(s)}
        if not valid:
            break
        g = np.exp(np.mean(np.log(list(valid.values()))))
        f0 = _pooled(curves, mature, {v: s / g for v, s in valid.items()})
    return f0


def _f0(f0: np.ndarray, u: np.ndarray | float) -> np.ndarray:
    return np.interp(u, np.arange(H + 1), f0, right=1.0)


def _fit_speed(cif: np.ndarray, a: int, f0: np.ndarray) -> float:
    if a < 12 or cif[a] <= 0:
        return float("nan")
    ages = np.arange(6, a + 1)
    obs = cif[ages] / cif[a]
    best, bs = np.inf, float("nan")
    for s in SPEEDS:
        den = float(_f0(f0, s * a))
        if den <= 1e-9:
            continue
        loss = float(np.mean((obs - _f0(f0, s * ages) / den) ** 2))
        if loss < best:
            best, bs = loss, s
    return bs


def _summary(t: pd.DataFrame) -> dict[str, object]:
    young = t[(t.A < H) & (t.A >= 12) & t.truth.notna()]
    res: dict[str, object] = {"n_young_scored": len(young)}
    for lo, hi in ((12, 36), (36, 60), (60, H)):
        g = young[(young.A >= lo) & (young.A < hi)]
        if len(g):
            res[f"A_{lo}_{hi}"] = {
                "n": len(g),
                **{c: round(float((np.abs(g[c] - g.truth) / g.truth).mean()), 3) for c in METHODS},
            }
    err = {
        c: round(float((np.abs(young[c] - young.truth) / young.truth).mean()), 3) for c in METHODS
    }
    res["all_young"] = err
    res["signed_bias_all_young"] = {
        c: round(float(((young[c] - young.truth) / young.truth).mean()), 3) for c in METHODS
    }
    res["best_method"] = min(err, key=lambda c: err[c])
    return res


if __name__ == "__main__":
    main()
