"""Fit vintage speeds s(c) on the shock-adjusted curves and run the over-identification check.

Input: artifacts/hazard_cells.parquet (from shock_overlay.py; w_adj = exposure x exp(beta'z), beta fitted on 1999-2012).
Reference shape R(u) = pooled adjusted cumulative hazard of the 1999-2003 vintages (speed 1).
For vintage c, s(c) minimises the squared distance between its normalised adjusted cumulative hazard and
R(s a)/R(s hi) on ages 6..hi (hi = min(72, last well-observed age)).
Over-identification (T2 note): the speed from early ages (6..36) must equal the speed from later ages (36..hi).
Cluster bootstrap over (first-payment month, state) groups within vintage gives 95% intervals.
Writes artifacts/speed_fit.csv, speed_fit.json and figures/6_speed_fit.png.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Final

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

HERE: Final = Path(__file__).resolve().parent
RAMP = LinearSegmentedColormap.from_list("blue", ["#9ec5f4", "#2a78d6", "#0d366b"])
AMAX: Final = 120
HI_MAX: Final = 72
EARLY: Final = 36
LO: Final = 6
MIN_RISK: Final = 2000
SPEEDS: Final = np.linspace(0.5, 2.0, 151)
REF_YEARS: Final = range(1999, 2004)
FIT_YEARS: Final = range(1999, 2013)
B: Final = 200


def main(suffix: str) -> None:
    cells = pd.read_parquet(HERE / "artifacts" / f"hazard_cells{suffix}.parquet")
    mats = {v: _matrices(g) for v, g in cells.groupby("vintage")}
    ev_ref = sum(mats[v][0].sum(axis=0) for v in REF_YEARS)
    w_ref = sum(mats[v][1].sum(axis=0) for v in REF_YEARS)
    ref = np.concatenate([[0.0], np.cumsum(ev_ref / np.maximum(w_ref, 1e-12))])  # R on ages 0..AMAX
    rng = np.random.default_rng(0)
    rows = []
    for v, (ev, w, risk) in sorted(mats.items()):
        hi = _last_age(risk)
        est = _speeds(ev.sum(0), w.sum(0), hi, ref)
        boots = []
        for _ in range(B):
            m = rng.multinomial(len(ev), np.full(len(ev), 1 / len(ev)))
            boots.append(_speeds(m @ ev, m @ w, hi, ref))
        b = np.array(boots)
        lo_ci, hi_ci = np.nanpercentile(b, [2.5, 97.5], axis=0)
        z = np.log(b[:, 2] / b[:, 1])
        rows.append(
            {
                "vintage": v,
                "hi": hi,
                "s_full": est[0],
                "s_early": est[1],
                "s_late": est[2],
                "s_full_lo": lo_ci[0],
                "s_full_hi": hi_ci[0],
                "s_early_lo": lo_ci[1],
                "s_early_hi": hi_ci[1],
                "s_late_lo": lo_ci[2],
                "s_late_hi": hi_ci[2],
                "overid_logratio": np.log(est[2] / est[1]),
                "overid_lo": np.nanpercentile(z, 2.5),
                "overid_hi": np.nanpercentile(z, 97.5),
                "rmse_s1": est[3],
                "rmse_fit": est[4],
                "max_logit_dev": est[5],
            }
        )
    t = pd.DataFrame(rows).set_index("vintage")
    t["rejects"] = (t.overid_lo > 0) | (t.overid_hi < 0)
    t.round(4).to_csv(HERE / "artifacts" / f"speed_fit{suffix}.csv")
    summary = {}
    for name, yrs in (("fit_1999_2012", FIT_YEARS), ("out_of_sample_2013_2019", range(2013, 2020))):
        g = t.loc[[y for y in yrs if y in t.index]]
        g = g[g.index.isin(range(1999, 2020))]
        summary[name] = {
            "vintages": len(g),
            "median_s_full": round(float(g.s_full.median()), 3),
            "range_s_full": [round(float(g.s_full.min()), 3), round(float(g.s_full.max()), 3)],
            "share_squared_error_explained_by_speed": round(
                float(1 - (g.rmse_fit**2).sum() / (g.rmse_s1**2).sum()), 3
            ),
            "overid_reject_share": round(float(g.rejects.mean()), 2),
            "median_abs_overid_logratio": round(float(g.overid_logratio.abs().median()), 3),
            "corr_s_early_s_late": round(float(np.corrcoef(g.s_early, g.s_late)[0, 1]), 3),
        }
    (HERE / "artifacts" / f"speed_fit{suffix}.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(
        t[
            [
                "hi",
                "s_full",
                "s_early",
                "s_late",
                "overid_logratio",
                "overid_lo",
                "overid_hi",
                "rejects",
            ]
        ]
        .round(3)
        .loc[1999:2019]
        .to_string()
    )
    _plot(t.loc[1999:2019], mats, ref, suffix)


def _matrices(g: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-group matrices (groups x ages 1..AMAX) of events, adjusted exposure and loans at risk."""
    keys = g.groupby(["fp_mi", "state"]).ngroup().to_numpy()
    n, ages = keys.max() + 1, g.age.to_numpy() - 1
    out = []
    for col in ("events", "w_adj", "exposure"):
        m = np.zeros((n, AMAX))
        np.add.at(m, (keys, ages), g[col].to_numpy())
        out.append(m)
    return out[0], out[1], out[2]


def _last_age(risk: np.ndarray) -> int:
    ok = np.where(risk.sum(axis=0) >= MIN_RISK)[0]
    return int(min(HI_MAX, ok.max() + 1)) if len(ok) else 0


def _curve(ev: np.ndarray, w: np.ndarray) -> np.ndarray:
    return np.concatenate([[0.0], np.cumsum(ev / np.maximum(w, 1e-12))])


def _speeds(ev: np.ndarray, w: np.ndarray, hi: int, ref: np.ndarray) -> np.ndarray:
    """(s_full, s_early, s_late, rmse at s=1, rmse at fitted s_full, max |logit deviation| from the fitted shape)."""
    h = _curve(ev, w)
    if hi <= EARLY + 6 or h[hi] <= 0:
        return np.full(6, np.nan)
    u = lambda s, a: np.interp(s * a, np.arange(AMAX + 1), ref)

    def fit(lo: int, up: int, base: int) -> tuple[float, float]:
        a = np.arange(max(lo, base + 1), up + 1)
        obs = (h[a] - h[base]) / (h[up] - h[base])
        best, bs = np.inf, np.nan
        for s in SPEEDS:
            den = u(s, up) - u(s, base)
            if den <= 0:
                continue
            d = float(np.mean((obs - (u(s, a) - u(s, base)) / den) ** 2))
            if d < best:
                best, bs = d, s
        return bs, best

    s_full, loss_full = fit(LO, hi, 0)
    s_early, _ = fit(LO, EARLY, 0)
    s_late, _ = fit(EARLY + 1, hi, EARLY)
    a = np.arange(LO, hi + 1)
    obs = h[a] / h[hi]
    rmse1 = float(np.sqrt(np.mean((obs - u(1.0, a) / u(1.0, hi)) ** 2)))
    fit_curve = u(s_full, a) / u(s_full, hi)
    inner = (obs > 0.02) & (obs < 0.98) & (fit_curve > 0.02) & (fit_curve < 0.98)
    logit = lambda x: np.log(x / (1 - x))
    max_dev = (
        float(np.max(np.abs(logit(obs[inner]) - logit(fit_curve[inner]))))
        if inner.any()
        else float("nan")
    )
    return np.array([s_full, s_early, s_late, rmse1, np.sqrt(loss_full), max_dev])


def _plot(
    t: pd.DataFrame,
    mats: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]],
    ref: np.ndarray,
    suffix: str,
) -> None:
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.6))
    v = t.index.to_numpy()
    ax[0].errorbar(
        v,
        t.s_full,
        yerr=[t.s_full - t.s_full_lo, t.s_full_hi - t.s_full],
        fmt="o",
        color="#2a78d6",
        ms=4,
        capsize=2,
    )
    ax[0].axvline(2012.5, color="gray", ls="--", lw=0.8)
    ax[0].axhline(1, color="gray", lw=0.6)
    ax[0].set(
        xlabel="vintage", ylabel="fitted speed s(c)", title="speed by vintage (reference 1999-2003)"
    )
    for y in v:
        ev, w, _ = mats[y]
        h = _curve(ev.sum(0), w.sum(0))
        hi = int(t.loc[y, "hi"])
        if hi < 40 or h[hi] <= 0:
            continue
        a = np.arange(1, hi + 1)
        s = t.loc[y, "s_full"]
        ax[1].plot(
            s * a,
            h[a] / h[hi] * np.interp(s * hi, np.arange(AMAX + 1), ref) / ref[AMAX],
            color=RAMP((y - 1999) / 20),
            lw=1.2,
            ls="-" if y <= 2012 else "--",
        )
    ax[1].set(
        xlabel="rescaled age s(c) x age (months)",
        ylabel="adjusted cumulative hazard (scaled to reference)",
        title="curves on rescaled time",
        xlim=(0, 130),
    )
    ax[2].errorbar(
        t.s_early,
        t.s_late,
        xerr=[t.s_early - t.s_early_lo, t.s_early_hi - t.s_early],
        yerr=[t.s_late - t.s_late_lo, t.s_late_hi - t.s_late],
        fmt="o",
        color="#eb6834",
        ms=4,
        lw=0.8,
        alpha=0.8,
    )
    lim = [0.5, 2.0]
    ax[2].plot(lim, lim, color="gray", lw=0.8)
    ax[2].set(
        xlabel="speed from ages 6-36",
        ylabel="speed from ages 36-72",
        title="over-identification: early vs late speed",
        xlim=lim,
        ylim=lim,
    )
    for a_ in ax:
        a_.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(HERE / "figures" / f"6_speed_fit{suffix}.png", dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    import sys

    main("".join(f"_{a}" for a in sys.argv[1:]))
