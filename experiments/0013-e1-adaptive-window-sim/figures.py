"""Figures for E1 from results.json (reduced grid: shape 1.5, median 7 years, p = 4.5%, K = 15, d = 1).

Usage: figures.py -> figures/1_window_vs_n.png, 2_error_vs_n.png, 3_error_vs_drift.png
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Final

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.axes import Axes

HERE: Final = Path(__file__).resolve().parent
DRIFTS: Final = {
    "none": "no drift",
    "linear_up": "older cohorts higher (2% per year)",
    "linear_down": "older cohorts lower (2% per year)",
    "step": "step 5 cohorts ago (30%)",
}
# colour, marker, label: Okabe-Ito hues (checked for colour-vision separation); the oracle is a grey reference
STYLE: Final = {
    "naive_full": ("#E69F00", "s", "naive, full window"),
    "drop_recent": ("#CC79A7", "D", "naive, drop recent"),
    "naive_lepski": ("#D55E00", "o", "naive Lepski"),
    "naive_adwin": ("#56B4E9", "^", "naive ADWIN"),
    "corrected_full": ("#E69F00", "s", "corrected, full window"),
    "corrected_proven": ("#CC79A7", "D", "corrected Lepski, proven constants"),
    "corrected_z": ("#0072B2", "o", "corrected Lepski, kappa = z"),
    "corrected_1": ("#009E73", "v", "corrected Lepski, kappa = 1"),
    "corrected_rolling": ("#56B4E9", "^", "corrected Lepski, kappa by rolling origin"),
    "corrected_fhat": ("#D55E00", "x", "corrected Lepski, kappa = z, estimated F"),
    "oracle": ("#777777", "", "oracle fixed window"),
}
NAIVE: Final = ("naive_full", "drop_recent", "naive_lepski", "naive_adwin")
CORRECTED: Final = (
    "corrected_full",
    "corrected_proven",
    "corrected_z",
    "corrected_1",
    "corrected_rolling",
    "corrected_fhat",
)


def line(ax: Axes, x: pd.Series, y: pd.Series, method: str) -> None:
    colour, marker, label = STYLE[method]
    dash = "--" if method == "oracle" else "-"
    ax.plot(x, y, dash, color=colour, marker=marker or None, ms=5, lw=2, label=label)


def tidy(ax: Axes, title: str) -> None:
    ax.set_title(title, fontsize=10, loc="left")
    ax.grid(True, color="#e6e6e6", lw=0.6)
    ax.spines[["top", "right"]].set_visible(False)


def window_vs_n(grid: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.6), sharey=True)
    for ax, (drift, title) in zip(axes, DRIFTS.items(), strict=True):
        g = grid[grid.drift == drift]
        for method in ("naive_lepski", "corrected_z", "oracle"):
            t = g[g.method == method]
            line(ax, t.n, t.window, method)
        ax.set_xscale("log")
        ax.set_xlabel("units per cohort (n)")
        tidy(ax, title)
    axes[0].set_ylabel("mean chosen window (cohorts)")
    axes[0].set_ylim(0, 15.5)
    axes[0].legend(frameon=False, fontsize=9, loc="center left")
    fig.suptitle(
        "More data, shorter window: the naive rule cuts to the youngest cohort; the corrected rule shortens only under real drift",
        fontsize=11,
        x=0.01,
        ha="left",
    )
    fig.tight_layout()
    fig.savefig(HERE / "figures" / "1_window_vs_n.png", dpi=150)
    plt.close(fig)


def error_vs_n(grid: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 4, figsize=(15, 7), sharey=True, sharex=True)
    for row, methods in enumerate((NAIVE, CORRECTED)):
        for ax, (drift, title) in zip(axes[row], DRIFTS.items(), strict=True):
            g = grid[grid.drift == drift]
            for method in (*methods, "oracle"):
                t = g[g.method == method]
                line(ax, t.n, 100 * t.rel_error, method)
            ax.set_xscale("log")
            ax.set_yscale("log")
            tidy(ax, title if row == 0 else "")
        axes[row][0].set_ylabel("mean relative error (%)")
        axes[row][-1].legend(
            frameon=False, fontsize=8, loc="center left", bbox_to_anchor=(1.0, 0.5)
        )
    for ax in axes[1]:
        ax.set_xlabel("units per cohort (n)")
    fig.suptitle(
        "Relative error against n. Top: rules that take labels at face value. Bottom: maturity-corrected rules.",
        fontsize=11,
        x=0.01,
        ha="left",
    )
    fig.tight_layout()
    fig.savefig(HERE / "figures" / "2_error_vs_n.png", dpi=150)
    plt.close(fig)


def error_vs_drift(sweep: pd.DataFrame) -> None:
    sweep = sweep.assign(slope=100 * sweep.drift.str.split(":").str[1].astype(float))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), sharey=True)
    methods = ("naive_lepski", "naive_full", "drop_recent", "corrected_z", "corrected_1", "oracle")
    for ax, n in zip(axes, (10_000, 100_000), strict=True):
        g = sweep[sweep.n == n].sort_values("slope")
        for method in methods:
            t = g[g.method == method]
            line(ax, t.slope, 100 * t.rel_error, method)
        ax.set_yscale("log")
        tidy(ax, f"n = {n:,} per cohort")
    axes[0].set_ylabel("mean relative error (%)")
    axes[1].legend(frameon=False, fontsize=8, loc="center left", bbox_to_anchor=(1.0, 0.5))
    fig.supxlabel(
        "drift: change of the eventual rate per year of cohort age (% of the target)", fontsize=10
    )
    fig.suptitle("Relative error against drift size", fontsize=11, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(HERE / "figures" / "3_error_vs_drift.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    results = json.loads((HERE / "results.json").read_text())
    (HERE / "figures").mkdir(exist_ok=True)
    window_vs_n(pd.DataFrame(results["grid"]))
    error_vs_n(pd.DataFrame(results["grid"]))
    error_vs_drift(pd.DataFrame(results["sweep"]))
