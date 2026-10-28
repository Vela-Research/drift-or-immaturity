"""Markdown tables for the README from results.json and artifacts/grid_full.csv (no new simulation).

Usage: summary.py > artifacts/tables.md
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd
from run import METHODS

HERE: Final = Path(__file__).resolve().parent
KEYS: Final = ["shape", "median", "p", "drift", "k", "d"]
DRIFTS: Final = ("none", "linear_up", "linear_down", "step")


def wide(t: pd.DataFrame, value: str, methods: list[str], index: list[str]) -> str:
    w = t.pivot_table(index=index, columns="method", values=value, sort=False)[methods]
    head = [*index, *[METHODS[m] for m in methods]]
    fmt = (lambda x: f"{100 * x:.1f}%") if value == "rel_error" else (lambda x: f"{x:.1f}")
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for key, row in w.iterrows():
        key = key if isinstance(key, tuple) else (key,)
        cells = [f"{k:,}" if isinstance(k, (int, np.integer)) else str(k) for k in key]
        lines.append("| " + " | ".join([*cells, *[fmt(v) for v in row]]) + " |")
    return "\n".join(lines)


def full_grid(t: pd.DataFrame) -> str:
    t = t.assign(f_d=1 - np.exp(-np.log(2) * (t.d / t["median"]) ** t["shape"]))
    t["maturity"] = pd.cut(
        t.f_d,
        [0, 0.25, 0.75, 0.95, 1.01],
        labels=["F(d) < 0.25", "0.25-0.75", "0.75-0.95", "> 0.95"],
    )
    by_n = t.groupby(["method", "n"], sort=False).rel_error.median().unstack()
    parts = [
        "**Median relative error over all 972 settings, by n per cohort**\n\n| method | "
        + " | ".join(f"n = {n:,}" for n in by_n.columns)
        + " |\n|---|"
        + "---|" * len(by_n.columns)
        + "\n"
        + "\n".join(
            f"| {METHODS[m]} | " + " | ".join(f"{100 * v:.1f}%" for v in by_n.loc[m]) + " |"
            for m in METHODS
        )
    ]
    w = t[t.method.isin(["naive_lepski", "naive_adwin", "corrected_z"])].pivot_table(
        index=[*KEYS, "maturity"], columns=["method", "n"], values="window", observed=True
    )
    rows = []
    for band, g in w.groupby(level="maturity", observed=True):
        small, big = g["naive_lepski"][1_000], g["naive_lepski"][1_000_000]
        rows.append(
            f"| {band} | {len(g)} | {small.mean():.1f} | {big.mean():.1f} | {100 * (big < small - 1e-9).mean():.0f}% | "
            f"{100 * (big <= 1.05).mean():.0f}% | {100 * (g['naive_adwin'][1_000_000] <= 1.05).mean():.0f}% |"
        )
    parts.append(
        "**Theorem A on the full grid: window of the naive rules, by maturity of the youngest cohort**\n\n"
        "| F(d) | settings | naive Lepski window, n = 1,000 | n = 1,000,000 | window shorter at 1e6 than at 1e3 | "
        "naive Lepski at window 1 (n = 1e6) | naive ADWIN at window 1 (n = 1e6) |\n|---|---|---|---|---|---|---|\n"
        + "\n".join(rows)
    )
    e = t.pivot_table(
        index=[*KEYS, "n", "maturity"], columns="method", values="rel_error", observed=True
    )
    rows = []
    for (band, n), g in e.groupby(level=["maturity", "n"], observed=True):
        rows.append(
            f"| {band} | {n:,} | {100 * g.naive_lepski.median():.1f}% | {100 * g.corrected_z.median():.1f}% | "
            f"{100 * g.oracle.median():.1f}% | {100 * (g.corrected_z < g.naive_lepski).mean():.0f}% |"
        )
    parts.append(
        "**Naive Lepski against corrected Lepski (kappa = z), full grid**\n\n"
        "| F(d) | n | naive Lepski, median error | corrected Lepski, median error | oracle, median error | "
        "settings where corrected is better |\n|---|---|---|---|---|---|\n" + "\n".join(rows)
    )
    nd = t[(t.drift == "none") & (t.method == "corrected_z")]
    share = nd.assign(full=nd.window >= 0.9 * nd.k).groupby("n").full.mean()
    parts.append(
        "**Corrected Lepski (kappa = z) without drift:** share of settings with a mean window of at least 90% of K: "
        + ", ".join(f"n = {n:,}: {100 * v:.0f}%" for n, v in share.items())
        + "."
    )
    return "\n\n".join(parts)


def main() -> None:
    r = json.loads((HERE / "results.json").read_text())
    grid, check = pd.DataFrame(r["grid"]), pd.DataFrame(r["t4_section5"])
    check["drift"] = check.drift.map(lambda s: f"{0.045 * float(s.split(':')[1]):+.3f} per year")
    methods = list(METHODS)
    adaptive = [m for m in methods if m not in ("naive_full", "drop_recent", "corrected_full")]
    parts = [
        f"## Reduced grid: relative error\n{wide(grid, 'rel_error', methods, ['drift', 'n'])}",
        f"## Reduced grid: mean chosen window\n{wide(grid, 'window', adaptive, ['drift', 'n'])}",
        f"## T4 section 5 check: relative error\n{wide(check, 'rel_error', methods, ['drift', 'n'])}",
        f"## T4 section 5 check: mean chosen window\n{wide(check, 'window', adaptive, ['drift', 'n'])}",
        f"## Full grid\n{full_grid(pd.read_csv(HERE / 'artifacts' / 'grid_full.csv'))}",
    ]
    sys.stdout.write("\n\n".join(parts) + "\n")


if __name__ == "__main__":
    main()
