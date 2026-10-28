"""Shock-adjusted timing overlay (data specification, update d; MTM variant from T2 Update 3).

Dataset: official 50k Sample (development only).
T2c step 1: unpenalized Poisson with vintage x age and state fixed effects + z, offset log at-risk (pyfixest),
fitted on 1999-2012 vintages (clean event, prepay censors), standard errors clustered by state. Per-vintage Breslow cumulative baseline hazards with and
without the covariate term are then overlaid.
  base: z = state house-price growth (yoy), state unemployment
  mtm : z = base + log mark-to-market LTV;  hinge: mtm + max(0, MTM LTV - 100) and max(0, MTM LTV - 120)
  (mtm detail) = log(orig LTV band midpoint x HPI(state, origination) / HPI(state, t))
Usage: shock_overlay.py [mtm|hinge] [subdist]. Writes figures/5_shock_adjusted_overlay{_mtm}.png, artifacts/shock_overlay{_mtm}.json,
artifacts/hazard_cells{_mtm}.parquet.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Final

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyfixest as pf
from matplotlib.colors import LinearSegmentedColormap

from maturity import data

log = logging.getLogger(__name__)
HERE: Final = Path(__file__).resolve().parent
COV: Final = data.DATA_DIR / "raw" / "covariates"
RAMP = LinearSegmentedColormap.from_list("blue", ["#9ec5f4", "#2a78d6", "#0d366b"])
AMAX: Final = 120
BIN: Final = 3
FIT_YEARS: Final = range(1999, 2013)
SHOW_YEARS: Final = range(1999, 2020)
MIN_RISK: Final = 300
CUTOFF_MI: Final = 2026 * 12 + 2  # 2026-03, the last performance month (0-based month index)
LTV_EDGES: Final = [0, 70, 80, 90, 1000]
LTV_MID: Final = np.array([62.0, 77.0, 87.0, 97.0])


def main(mode: str, subdist: bool = False) -> None:
    mtm, hinge = mode in ("mtm", "hinge"), mode == "hinge"
    suffix = (f"_{mode}" if mode else "") + ("_subdist" if subdist else "")
    d = pd.read_parquet(data.DATA_DIR / "processed" / "freddie_sample_loans.parquet")
    d = d[d.orig_year.isin(SHOW_YEARS) & d.event_age_clean.notna() & d.state.notna()].copy()
    if mtm:
        d = d[d.ltv.notna()].copy()
        d["ltv_band"] = pd.cut(d.ltv, LTV_EDGES, labels=False).astype(int)
    hpi, unemp = _covariates()
    cells = _cells(d, hpi, unemp, mtm, subdist=subdist)
    log.info("cells: %d, events %d", len(cells), cells.events.sum())
    zcols = (
        ["hpi_yoy_z", "unemp_z"]
        + (["log_mtm_z"] if mtm else [])
        + (["h100_z", "h120_z"] if hinge else [])
    )
    beta, kappa, se = _fit(cells[cells.vintage.isin(FIT_YEARS)], zcols)
    log.info(
        "beta: %s  se (clustered by state): %s",
        dict(zip(zcols, beta.round(3), strict=True)),
        se.round(3),
    )
    lin = cells[zcols].to_numpy() @ beta + cells.state.map(kappa).fillna(0.0).to_numpy()
    cells["w_adj"] = cells.exposure * np.exp(lin)
    cells.to_parquet(HERE / "artifacts" / f"hazard_cells{suffix}.parquet")
    raw, adj = _breslow(cells, "exposure"), _breslow(cells, "w_adj")
    stats = _plot_and_stats(raw, adj, suffix)
    stats["beta"] = dict(zip(zcols, beta.round(4).tolist(), strict=True))
    stats["beta_se_clustered_state"] = dict(zip(zcols, se.round(4).tolist(), strict=True))
    stats["state_effect_sd"] = round(float(kappa.std()), 3)
    (HERE / "artifacts" / f"shock_overlay{suffix}.json").write_text(json.dumps(stats, indent=2))


def _covariates() -> tuple[pd.DataFrame, pd.DataFrame]:
    h = pd.read_csv(
        COV / "fhfa_state.csv", header=None, usecols=[0, 1, 2, 3], names=["state", "yr", "q", "hpi"]
    )
    h["qi"] = h.yr * 4 + h.q - 1
    h = h.sort_values(["state", "qi"])
    h["hpi_yoy"] = h.groupby("state").hpi.transform(lambda s: np.log(s / s.shift(4)))
    frames = []
    for f in COV.glob("unrate_*.csv"):
        u = pd.read_csv(f)
        if len(u.columns) != 2 or len(u) < 10:
            continue
        u.columns = ["date", "unemp"]
        dt = pd.to_datetime(u["date"])
        frames.append(
            pd.DataFrame(
                {
                    "state": f.stem.split("_")[1],
                    "mi": dt.dt.year * 12 + dt.dt.month - 1,
                    "unemp": u.unemp,
                }
            )
        )
    return h[["state", "qi", "hpi", "hpi_yoy"]], pd.concat(frames)


def _cells(
    d: pd.DataFrame, hpi: pd.DataFrame, unemp: pd.DataFrame, mtm: bool, *, subdist: bool = False
) -> pd.DataFrame:
    """Loan-month cells. subdist=True uses the sub-distribution risk set (T2c section 8): a prepaid loan stays at risk,
    with zero events, until the data cutoff; a defaulted loan leaves. False = cause-specific (prepay censors)."""
    fp = d.first_payment.astype(int)
    d["fp_mi"] = (fp // 100) * 12 + fp % 100 - 1
    dur = d.event_age_clean
    if subdist:
        dur = dur.where(d.event_type_clean != "prepay", CUTOFF_MI - d["fp_mi"] + 1)
    d["dur"] = dur.clip(1, AMAX).astype(int)
    d["ev"] = (d.event_type_clean == "default") & (d.event_age_clean <= AMAX)
    keys = ["orig_year", "fp_mi", "state"] + (["ltv_band"] if mtm else [])
    ages = np.arange(1, AMAX + 1)
    out = []
    for key, g in d.groupby(keys, observed=True):
        dur, ev = g.dur.to_numpy(), g.ev.to_numpy()
        n_at = len(dur) - np.concatenate(
            [[0], np.cumsum(np.bincount(dur, minlength=AMAX + 1))[:-1]]
        )
        events = np.bincount(dur[ev], minlength=AMAX + 1)
        frame = pd.DataFrame(
            {
                "fp_mi": key[1],
                "vintage": key[0],
                "state": key[2],
                "age": ages,
                "mi": key[1] + ages - 1,
                "exposure": n_at[1:],
                "events": events[1:],
            }
        )
        if mtm:
            frame["ltv_band"] = key[3]
        out.append(frame)
    c = pd.concat(out, ignore_index=True)
    c = c[c.exposure > 0].copy()
    c["qi"] = c.mi // 3
    c = c.merge(hpi, on=["state", "qi"], how="inner").merge(unemp, on=["state", "mi"], how="inner")
    if mtm:
        h0 = hpi.rename(columns={"hpi": "hpi_orig", "qi": "qi_orig"})[
            ["state", "qi_orig", "hpi_orig"]
        ]
        c["qi_orig"] = (c.fp_mi - 1) // 3
        c = c.merge(h0, on=["state", "qi_orig"], how="inner")
        mtm_ltv = LTV_MID[c.ltv_band.to_numpy()] * c.hpi_orig / c.hpi
        c["log_mtm"] = np.log(mtm_ltv)
        c["h100"] = np.maximum(0.0, mtm_ltv - 100)  # negative-equity hinges ("double trigger" view)
        c["h120"] = np.maximum(0.0, mtm_ltv - 120)
    c = c.dropna(subset=["hpi_yoy", "unemp"] + (["log_mtm"] if mtm else []))
    for col in ("hpi_yoy", "unemp") + (("log_mtm", "h100", "h120") if mtm else ()):
        c[f"{col}_z"] = (c[col] - c[col].mean()) / c[col].std()
    return c


def _fit(c: pd.DataFrame, zcols: list[str]) -> tuple[np.ndarray, pd.Series, np.ndarray]:
    """T2c step 1: unpenalized fixed-effects Poisson (vintage x age + state effects), SEs clustered by state."""
    d = c[["events", "exposure", "state", *zcols]].copy()
    d["cell"] = (c.vintage.to_numpy() - FIT_YEARS.start) * AMAX + (c.age.to_numpy() - 1)
    d["log_exposure"] = np.log(d.exposure)
    m = pf.fepois(
        f"events ~ {' + '.join(zcols)} | cell + state",
        data=d,
        offset="log_exposure",
        vcov={"CRV1": "state"},
    )
    tidy = m.tidy()
    fe = m.fixef()
    kappa = pd.Series(fe[next(k for k in fe if "state" in k)], dtype=float)
    wts = d.groupby("state").exposure.sum().reindex(kappa.index).fillna(0.0).to_numpy()
    return (
        tidy.loc[zcols, "Estimate"].to_numpy(),
        kappa - float(np.average(kappa.to_numpy(), weights=wts)),
        tidy.loc[zcols, "Std. Error"].to_numpy(),
    )


def _breslow(c: pd.DataFrame, wcol: str) -> pd.DataFrame:
    g = (
        c.groupby(["vintage", "age"])
        .agg(events=("events", "sum"), w=(wcol, "sum"), risk=("exposure", "sum"))
        .reset_index()
    )
    g["haz"] = g.events / g.w
    g["H"] = g.groupby("vintage").haz.cumsum()
    return g


def _plot_and_stats(raw: pd.DataFrame, adj: pd.DataFrame, suffix: str) -> dict[str, object]:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    stats: dict[str, object] = {}
    for ax, (name, t) in zip(
        axes,
        (
            ("raw (no calendar term)", raw),
            ("shock-adjusted" + (" + MTM LTV" if suffix else ""), adj),
        ),
        strict=True,
    ):
        norm = {}
        for v, g in t.groupby("vintage"):
            h60 = g.loc[g.age == 60, "H"]
            if h60.empty or h60.iloc[0] <= 0:
                continue
            y = (g.H / h60.iloc[0]).where(g.risk >= MIN_RISK)
            norm[v] = y.set_axis(g.age)
            ax.plot(
                g.age, y, color=RAMP((v - 1999) / 20), lw=1.4, ls="-" if v in FIT_YEARS else "--"
            )
        ax.set(title=name, xlabel="loan age (months)", xlim=(0, AMAX), ylim=(0, 2.2))
        ax.axhline(1, color="gray", lw=0.7, ls=":")
        ax.spines[["top", "right"]].set_visible(False)
        wide = pd.DataFrame(norm).loc[[24, 36, 48, 72, 96]]
        fit = wide[[c for c in wide.columns if c in FIT_YEARS]]
        stats[name] = {
            "cv_across_fit_vintages_at_age": (fit.std(axis=1) / fit.mean(axis=1)).round(3).to_dict()
        }
    axes[0].set_ylabel("cumulative baseline hazard / value at 60 months")
    fig.tight_layout()
    fig.savefig(HERE / "figures" / f"5_shock_adjusted_overlay{suffix}.png", dpi=140)
    plt.close(fig)
    return stats


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    args = [a for a in sys.argv[1:] if a != "subdist"]
    main(args[0] if args else "", subdist="subdist" in sys.argv[1:])
