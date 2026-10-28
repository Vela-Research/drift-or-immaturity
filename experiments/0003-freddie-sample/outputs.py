"""Spec section 7 outputs 2-5 from the loan-level table. Writes figures/*.png and artifacts/*.csv."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Final

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

from maturity import data

log = logging.getLogger(__name__)
HERE: Final = Path(__file__).resolve().parent
RAMP = LinearSegmentedColormap.from_list("blue", ["#9ec5f4", "#2a78d6", "#0d366b"])
MAX_AGE: Final = 240
CUTOFF_PERIOD: Final = 202603  # latest performance month in the samples


def main() -> None:
    d = pd.read_parquet(data.DATA_DIR / "processed" / "freddie_sample_loans.parquet")
    (HERE / "figures").mkdir(exist_ok=True)
    (HERE / "artifacts").mkdir(exist_ok=True)
    for tag, suffix in (("primary", ""), ("sens90", "90")):
        tri = pd.DataFrame({y: _cif(g, suffix) for y, g in d.groupby("orig_year")}).T
        tri.round(5).to_csv(HERE / "artifacts" / f"triangle_{tag}.csv")
        _fig_triangle(tri, tag)
    tri = pd.read_csv(HERE / "artifacts" / "triangle_primary.csv", index_col=0)
    _fig_calendar(d)
    _fig_overlay(tri)
    _vintage_table(d, tri).to_csv(HERE / "artifacts" / "vintage_summary.csv")
    _forbearance(d).to_csv(HERE / "artifacts" / "forbearance.csv")


def _cif(g: pd.DataFrame, suffix: str) -> np.ndarray:
    """Aalen-Johansen cumulative incidence of default by loan age in months (prepay = competing risk)."""
    age = g[f"event_age{suffix}"].fillna(0).clip(0, MAX_AGE).astype(int).to_numpy()
    kind = g[f"event_type{suffix}"].to_numpy()
    n = len(g)
    d = np.bincount(age[kind == "default"], minlength=MAX_AGE + 1)
    p = np.bincount(age[kind == "prepay"], minlength=MAX_AGE + 1)
    c = np.bincount(age[kind == "censored"], minlength=MAX_AGE + 1)
    at_risk = n - np.concatenate([[0], np.cumsum(d + p + c)[:-1]])
    haz = np.divide(d + p, at_risk, out=np.zeros(MAX_AGE + 1), where=at_risk > 0)
    surv = np.concatenate([[1.0], np.cumprod(1 - haz)[:-1]])
    cif = np.cumsum(surv * np.divide(d, at_risk, out=np.zeros(MAX_AGE + 1), where=at_risk > 0))
    last_age = int(np.quantile(age, 0.0)) if False else int(g[f"event_age{suffix}"].max())
    cif[last_age + 1 :] = np.nan  # beyond the longest follow-up
    return cif


def _fig_triangle(tri: pd.DataFrame, tag: str) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    ys = list(tri.index)
    for y in ys:
        ax.plot(
            tri.columns.astype(int),
            tri.loc[y] * 100,
            color=RAMP((y - ys[0]) / (ys[-1] - ys[0])),
            lw=1.5,
        )
    ax.set(xlim=(0, 180), xlabel="loan age (months)", ylabel="cumulative default incidence (%)")
    sm = plt.cm.ScalarMappable(cmap=RAMP, norm=plt.Normalize(ys[0], ys[-1]))
    fig.colorbar(sm, ax=ax, label="origination year")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(HERE / "figures" / f"2_triangle_{tag}.png", dpi=140)
    plt.close(fig)


def _fig_calendar(d: pd.DataFrame) -> None:
    ev = d[d.event_type == "default"].copy()
    ev["cal"] = pd.to_datetime(ev.event_period.astype(int).astype(str), format="%Y%m")
    groups = pd.cut(
        ev.orig_year,
        [1998, 2003, 2004, 2005, 2006, 2007, 2008, 2012, 2019, 2026],
        labels=["1999-03", "2004", "2005", "2006", "2007", "2008", "2009-12", "2013-19", "2020+"],
    )
    t = (
        ev.groupby([ev.cal.dt.to_period("M").dt.to_timestamp(), groups], observed=False)
        .size()
        .unstack(fill_value=0)
    )
    fig, ax = plt.subplots(figsize=(10, 4.6))
    cmap = plt.get_cmap("viridis", len(t.columns))
    ax.stackplot(
        t.index, t.T.to_numpy(), labels=t.columns, colors=[cmap(i) for i in range(len(t.columns))]
    )
    ax.set(xlabel="calendar month of default event", ylabel="defaults in sample (count)")
    ax.legend(title="vintage", frameon=False, fontsize=8, ncol=3)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(HERE / "figures" / "3_defaults_by_calendar_month.png", dpi=140)
    plt.close(fig)
    t.to_csv(HERE / "artifacts" / "defaults_by_calendar_month.csv")


def _fig_overlay(tri: pd.DataFrame, ref: int = 60) -> None:
    """CIF(a) / CIF(ref): same shape, different speed? Only vintages observed past `ref` with CIF(ref) >= 0.5%."""
    fig, ax = plt.subplots(figsize=(9, 5))
    tri = tri.copy()
    tri.columns = tri.columns.astype(int)
    ok = [y for y in tri.index if pd.notna(tri.loc[y, ref]) and tri.loc[y, ref] >= 0.005]
    for y in ok:
        ax.plot(
            tri.columns,
            tri.loc[y] / tri.loc[y, ref],
            color=RAMP((y - 1999) / 26),
            lw=1.5,
            label=str(y),
        )
    ax.set(
        xlim=(0, 120),
        ylim=(0, 2),
        xlabel="loan age (months)",
        ylabel=f"CIF(age) / CIF({ref} months)",
    )
    ax.axhline(1, color="gray", lw=0.8, ls="--")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(HERE / "figures" / "4_timing_overlay.png", dpi=140)
    plt.close(fig)


def _vintage_table(d: pd.DataFrame, tri: pd.DataFrame) -> pd.DataFrame:
    g = d.groupby("orig_year")
    t = pd.DataFrame(
        {
            "loans": g.size(),
            "default_ever_pct": g.apply(lambda x: 100 * (x.event_type == "default").mean()),
            "default90_ever_pct": g.apply(lambda x: 100 * (x.event_type90 == "default").mean()),
            "prepay_pct": g.apply(lambda x: 100 * (x.event_type == "prepay").mean()),
            "censored_pct": g.apply(lambda x: 100 * (x.event_type == "censored").mean()),
            "max_follow_up_months": g.event_age.max(),
            "relief_refi_loans": g.is_relief_refi.sum(),
        }
    )
    for a in (24, 60, 120):
        t[f"cif_{a}m_pct"] = (tri[str(a)] * 100).values
    return t.round(2)


def _forbearance(d: pd.DataFrame) -> pd.DataFrame:
    rows = {}
    for label, etype, flag, period in (
        ("180+ (primary)", "event_type", "forb_at_dq6", "dq6_period"),
        ("90+", "event_type90", "forb_at_dq3", "dq3_period"),
    ):
        e = d[d[etype] == "default"]
        first = e[period].notna()
        covid = e[period].between(202003, 202112)
        rows[label] = {
            "default_events": len(e),
            "from_delinquency": int(first.sum()),
            "flagged_forbearance_or_disaster_or_deferral": int(
                e[flag].fillna(False).astype(bool).sum()
            ),
            "event_month_in_2020-03_to_2021-12": int(covid.sum()),
            "flagged_and_in_covid_window": int((e[flag].fillna(False).astype(bool) & covid).sum()),
        }
    return pd.DataFrame(rows).T


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
