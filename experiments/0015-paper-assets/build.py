"""Tables and figures for the paper draft (research/drift-or-maturity/paper), built only from existing results.

Sources: 0004 (speed simulation), 0011 (Freddie Mac 200k, as run), 0012 (post hoc corrected run and no-HARP check), 0018 (borrower-followed run),
0013 (E1 simulation; full grid from artifacts/grid_full.csv), 0014 and 0016 (E2 on Freddie Mac 200k). One new diagnostic: the spread of the vintages' timing curves
before and after the calendar adjustment on the 200k cells of 0012 (the 50k version is in 0003).
Figures use the paper's figure style (research/drift-or-maturity/figsrc/style.py) and are drawn at print size; method names are those of paper v3.
Usage: build.py -> paper/figures/*.pdf, paper/tables/*.tex, results.json (every number the paper text quotes)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Final

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter

HERE: Final = Path(__file__).resolve().parent
EXP: Final = HERE.parent
PAPER: Final = EXP.parent / "research" / "drift-or-maturity" / "paper"
sys.path.insert(0, str(EXP / "0003-freddie-sample"))
sys.path.insert(0, str(EXP.parent / "research" / "drift-or-maturity" / "figsrc"))
import shock_overlay as so
from style import (
    C,
)  # the paper's figure style: the import sets the rcParams (8 pt serif, no top and right spines)

# TeX Gyre Termes (the paper's font) is not installed on every machine; Times New Roman has the same metrics
plt.rcParams["font.serif"] = ["TeX Gyre Termes", "Times New Roman", "DejaVu Serif"]
LINEWIDTH: Final = 5.5  # text width of the paper in inches; every figure is drawn at its print size
PERCENT: Final = FuncFormatter(lambda value, _: f"{value:g}%")
# window rule -> (colour, marker, line style, label). Grey = naive rules (events so far), never solid; the
# maturity-weighted rules are solid; the colours of the two headline rules and the oracle are those of the teaser.
E1_STYLE: Final = {
    "naive_lepski": (C["naive"], "o", "--", "naive Lepski"),
    "naive_adwin": (C["naive"], "s", "-.", "naive ADWIN"),
    "naive_full": (C["naive"], "", ":", "naive full window"),
    "drop_recent": (C["drop"], "D", "--", "drop recent (older half)"),
    "corrected_full": (C["clexact"], "s", "-", "maturity-weighted full window"),
    "corrected_proven": (C["cl"], "D", "-", "maturity-weighted Lepski, proven constants"),
    "corrected_z": (C["full"], "o", "-", r"maturity-weighted Lepski, $\kappa = z$"),
    "corrected_1": (C["clcal"], "v", "-", r"maturity-weighted Lepski, $\kappa = 1$"),
    "corrected_rolling": (
        C["lastmature"],
        "x",
        "-",
        r"maturity-weighted Lepski, rolling-origin $\kappa$",
    ),
    "oracle": (C["truth"], "", "-", "oracle fixed window"),
}
E2_ROWS: Final = {
    "naive_lepski": "Naive Lepski",
    "naive_adwin": "Naive ADWIN",
    "corrected_proven": "Maturity-weighted Lepski, proven constants",
    "corrected_z": r"Maturity-weighted Lepski, $\kappa = z$",
    "corrected_1": r"Maturity-weighted Lepski, $\kappa = 1$",
    "corrected_rolling": r"Maturity-weighted Lepski, rolling-origin $\kappa$",
    "hindsight": "Best maturity-weighted window in hindsight",
}


def load(path: str) -> dict:
    return json.loads((EXP / path).read_text())


def ratio2(c: dict) -> str:
    """Error ratio to 2 decimals. The stored ratio has 3 decimals; when its third decimal is 5 (0.4045 is stored as
    0.405 and would print as 0.41) the stored mean log difference decides."""
    stored = c["error_ratio"]
    tie = round(stored * 1000) % 10 == 5
    return f"{np.exp(c['mean_paired_log_difference']) if tie else stored:.2f}"


def pct(x: float, digits: int = 1) -> str:
    return f"{100 * x:.{digits}f}\\%"


def canvas(fraction: float, height: float, **kwargs: object) -> tuple[Figure, Axes | np.ndarray]:
    """A figure as wide as `fraction` of the text width (the width the paper includes it at)."""
    return plt.subplots(figsize=(fraction * LINEWIDTH, height), layout="constrained", **kwargs)  # type: ignore[arg-type]


def save(fig: Figure, name: str) -> None:
    """The whole canvas, not a tight crop: the PDF has the print size, so LaTeX does not rescale it."""
    fig.savefig(PAPER / "figures" / name, bbox_inches=fig.bbox_inches)
    plt.close(fig)


def rule_line(method: str) -> dict[str, object]:
    colour, marker, dash, _ = E1_STYLE[method]
    width = 0.8 if method == "oracle" else 1.2
    return {"color": colour, "marker": marker or None, "ls": dash, "lw": width, "ms": 3}


def ink(colour: str) -> str:
    """Text colour of a direct label: the colour of its line, dark grey for the grey lines."""
    return C["ink"] if colour in (C["naive"], C["muted"]) else colour


def end_labels(ax: Axes, labels: list[tuple[float, str, str]], gap: float = 0.07) -> None:
    """Direct labels right of the axes at the height of each line's last point: (y, text, colour). Labels that
    would overlap are moved apart (gap = height of one text line as a share of the axes). Call it after the axis
    limits and the scale are set."""
    labels = sorted(labels)
    x0 = ax.get_xlim()[0]
    y = np.array(
        [
            ax.transAxes.inverted().transform(ax.transData.transform((x0, v)))[1]
            for v, _, _ in labels
        ]
    )
    height = gap * np.array([text.count("\n") + 1 for _, text, _ in labels])
    for _ in range(200):
        for i in range(len(y) - 1):
            if (short := (height[i] + height[i + 1]) / 2 - (y[i + 1] - y[i])) > 0:
                y[i], y[i + 1] = y[i] - short / 2, y[i + 1] + short / 2
    for position, (_, text, colour) in zip(y, labels, strict=True):
        ax.text(
            1.02,
            position,
            text,
            transform=ax.transAxes,
            color=colour,
            va="center",
            fontsize=7,
            linespacing=0.95,
        )


def fig_e1(full: pd.DataFrame) -> None:
    """Full grid (972 settings per n): mean chosen window as a share of the K cohorts, and median relative error.
    Two files for one figure, side by side: the same axes rectangle in both, and one legend in two halves (left
    file: naive rules, drop recent and the oracle; right file: the maturity-weighted rules)."""
    full = full.assign(share=full.window / full.k)
    window = full.groupby(["method", "n"]).share.mean().unstack(0)
    error = 100 * full.groupby(["method", "n"]).rel_error.median().unstack(0)
    fixed = ("naive_full", "drop_recent", "corrected_full")  # no window choice to show
    weighted = [m for m in E1_STYLE if m.startswith("corrected")]
    others = [m for m in E1_STYLE if m not in weighted]
    panels = (
        ("e1_full_window_vs_n.pdf", window, "mean chosen window (share of $K$)", False, others, ""),
        (
            "e1_full_error_vs_n.pdf",
            error,
            "median relative error",
            True,
            weighted,
            "maturity-weighted",
        ),
    )
    for name, table, label, log_y, legend, title in panels:
        fig = plt.figure(figsize=(0.48 * LINEWIDTH, 2.85))
        ax = fig.add_axes((0.215, 0.44, 0.755, 0.535))
        for method in E1_STYLE:
            if log_y or method not in fixed:
                ax.plot(table.index, table[method], **rule_line(method))
        ax.set_xscale("log")
        ax.set_xlabel("units per cohort $n$")
        ax.set_ylabel(label)
        if log_y:
            ax.set_yscale("log")
            ax.set_ylim(0.3, 150)
            ax.yaxis.set_major_formatter(PERCENT)
        fig.legend(
            handles=[Line2D([], [], **rule_line(m)) for m in legend],
            labels=[E1_STYLE[m][3].removeprefix("maturity-weighted ") for m in legend],
            title=title or None,
            title_fontsize=7,
            alignment="left",
            loc="upper left",
            bbox_to_anchor=(0.04, 0.27),
            frameon=False,
            handlelength=2.6,
            labelspacing=0.25,
            borderpad=0,
        )
        save(fig, name)


def fig_speed(sim: dict) -> None:
    """0004, scenario "clean": recent cohorts are faster and have lower eventual rates."""
    rows = {
        "naive": ("events so far", C["naive"]),
        "chain": ("chain ladder", C["cl"]),
        "ours": ("cohort speed, per cohort", C["full"]),
        "ours+smooth": ("cohort speed, smoothed", C["full"]),
    }
    drift = sim["clean"]["relative_drift_newest5_vs_old"]
    fig, ax = canvas(0.6, 1.55)
    for i, (key, (_, colour)) in enumerate(rows.items()):
        mid, low, high = drift[key]
        ax.plot([low, high], [i, i], color=colour, lw=1.5)
        ax.plot(mid, i, "o", color=colour, ms=4)
    truth = sim["clean"]["truth_drift"]
    ax.axvline(truth, color=C["truth"], lw=1.0)
    ax.axvline(1.0, color=C["muted"], lw=0.6, ls=":")
    ax.text(truth - 0.02, -0.8, f"truth {truth:.2f}", color=C["truth"], ha="right", fontsize=7)
    ax.text(1.02, -0.8, "no drift", color=C["muted"], ha="left", fontsize=7)
    ax.set_yticks(range(len(rows)), [label for label, _ in rows.values()])
    ax.set_ylim(len(rows) - 0.5, -1.1)
    ax.set_xlabel("drift ratio (5 newest cohorts / old cohorts)")
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    save(fig, "speed_chain_ladder.pdf")


def fig_e1_drift(sweep: pd.DataFrame) -> None:
    """E1, reduced setting: relative error against the size of a linear drift, at n = 1e4 and 1e5."""
    sweep = sweep.assign(slope=100 * sweep.drift.str.split(":").str[1].astype(float))
    methods = (
        "naive_lepski",
        "naive_full",
        "drop_recent",
        "corrected_full",
        "corrected_z",
        "corrected_1",
        "oracle",
    )
    fig, axes = canvas(0.75, 2.75, ncols=2, sharey=True)
    for ax, n in zip(axes, (10_000, 100_000), strict=True):
        g = sweep[sweep.n == n].sort_values("slope")
        for method in methods:
            t = g[g.method == method]
            ax.plot(t.slope, 100 * t.rel_error, label=E1_STYLE[method][3], **rule_line(method))
        ax.set_yscale("log")
        ax.set_title(f"$n$ = {n:,} per cohort")
    axes[0].set_ylabel("relative error")
    axes[0].yaxis.set_major_formatter(PERCENT)
    # one label for both panels, centred on the gap between them (a supxlabel would lie under the legend)
    axes[0].set_xlabel(
        "change of the eventual rate per year of cohort age (% of the target)", x=1.04
    )
    fig.legend(
        *axes[0].get_legend_handles_labels(),
        loc="outside lower center",
        ncol=2,
        frameon=False,
        handlelength=2.6,
        labelspacing=0.25,
    )
    save(fig, "e1_error_vs_drift.pdf")


def fig_freddie_snapshots(run: dict) -> None:
    """Pre-registered run: error per snapshot (mean absolute relative error of the six scored vintages)."""
    lines = {
        "naive": (C["naive"], "o", "--", "events so far"),
        "drop_recent": (C["drop"], "D", "--", "drop recent"),
        "chain": (C["cl"], "v", "-", "chain ladder"),
        "cal_chain": (C["clcal"], "s", "-", "calendar-adjusted CL"),
        "ours": (C["full"], "o", "-", "full method"),
    }
    err = {int(y): v["error"] for name in ("E3a", "E3b") for y, v in run[name]["snapshots"].items()}
    years = sorted(err)
    fig, ax = canvas(0.9, 2.3)
    for method, (colour, marker, dash, _) in lines.items():
        values = [100 * err[y][method] for y in years]
        ax.plot(years, values, color=colour, marker=marker, ls=dash, lw=1.2, ms=3)
    ax.axvline(2012.5, color=C["muted"], lw=0.6, ls=":")
    ax.text(2012.4, 1300, "truth: 7-year incidence", ha="right", fontsize=6.5, color=C["muted"])
    ax.text(2012.6, 1300, "truth cut at Feb 2020", ha="left", fontsize=6.5, color=C["muted"])
    ax.set_yscale("log")
    ax.set_ylim(8, 2000)
    ax.yaxis.set_major_formatter(PERCENT)
    ax.set_xticks(years)
    ax.set_xlabel("snapshot (December)")
    ax.set_ylabel("mean relative error")
    end_labels(
        ax,
        [
            (100 * err[years[-1]][m], label, ink(colour))
            for m, (colour, _, _, label) in lines.items()
        ],
    )
    save(fig, "freddie_snapshot_errors.pdf")


def table_overid() -> str:
    """Over-identification test per vintage (base covariates): effect size log(late / early speed), T = LR / phi, p."""
    runs = [
        pd.read_csv(EXP / path).set_index("vintage")
        for path in (
            "0011-freddie-200k-confirmatory/artifacts/overid_test_subdist.csv",
            "0012-freddie-200k-harp-correction/artifacts/overid_test_ipcw_subdist.csv",
        )
    ]
    lines = [
        r"\begin{tabular}{lrcccccc}",
        r"\toprule",
        r" & & \multicolumn{3}{c}{Pre-registered weights} & \multicolumn{3}{c}{Corrected weights} \\",
        r"\cmidrule(lr){3-5}\cmidrule(lr){6-8}",
        r"Vintage & Events & $\log(s_{\text{late}}/s_{\text{early}})$ & $T$ & $p$ & $\log(s_{\text{late}}/s_{\text{early}})$ & $T$ & $p$ \\",
        r"\midrule",
    ]
    for vintage in runs[0].index:
        cells = [f"{int(vintage)}", f"{int(runs[0].events[vintage]):,}"]
        for r in runs:
            star = "$^{*}$" if r.rejects[vintage] else ""
            cells += [
                f"${r.log_ratio[vintage]:+.2f}${star}",
                f"{r['T'][vintage]:.1f}",
                f"{r.p_value[vintage]:.3f}",
            ]
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def table_freddie(as_run: dict, fixed: dict, noharp: dict, linked: dict) -> str:
    def row(label: str, run: dict, method: str) -> str:
        cells = [run[s]["median_error"][method] for s in ("E3a", "E3b")] + [
            run["H10"]["median_error"][method]
        ]
        return f"{label} & " + " & ".join(pct(c) for c in cells) + r" \\"

    lines = [
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r" & \multicolumn{2}{c}{7-year incidence} & 10-year incidence \\",
        r"\cmidrule(lr){2-3}\cmidrule(lr){4-4}",
        r"Method & 2006--2012 (crisis) & 2013--2018 (pre-COVID) & 2009--2016 \\",
        r"\midrule",
        row("Events so far", as_run, "naive"),
        row("Drop recent", as_run, "drop_recent"),
        row("Last mature cohort", as_run, "fixed_horizon"),
        row("Chain ladder", as_run, "chain"),
        row("Full method w/o calendar", as_run, "ours_free"),
        r"\midrule",
        r"\emph{Pre-registered (as run)} \\",
        row("Calendar-adjusted CL", as_run, "cal_chain"),
        row("Full method", as_run, "ours"),
        row("Full method, forecast path", as_run, "ours_uncond"),
        r"\midrule",
        r"\emph{Corrected censoring weights (post hoc)} \\",
        row("Calendar-adjusted CL", fixed, "cal_chain"),
        row("Full method", fixed, "ours"),
        row("Full method, forecast path", fixed, "ours_uncond"),
        r"\midrule",
        r"\emph{Robustness: HARP refinances counted as prepayments} \\",
        row("Chain ladder", noharp, "chain"),
        row("Calendar-adjusted CL", noharp, "cal_chain"),
        row("Full method", noharp, "ours"),
        row("Full method, forecast path", noharp, "ours_uncond"),
        r"\midrule",
        r"\emph{Post hoc: follows borrowers into their relief loans} \\",
        row("Chain ladder", linked, "chain"),
        row("Calendar-adjusted CL", linked, "cal_chain"),
        row("Full method", linked, "ours"),
        row("Full method, forecast path", linked, "ours_uncond"),
        r"\bottomrule",
        r"\end{tabular}",
    ]
    return "\n".join(lines) + "\n"


def table_freddie_tests(runs: dict[str, dict]) -> str:
    def paired(v: dict) -> str:
        p = v["sign_test_p"]
        return f"{v['wins']}/{v['n']} & {'0.0002' if p == 0 else f'{p:.3f}'}"

    def vintage(v: dict) -> str:
        low, high = v["ci95"]
        return f"{ratio2(v)} & $[{low:+.2f},\\,{high:+.2f}]$"

    lines = [
        r"\begin{tabular}{llcccc}",
        r"\toprule",
        r" & & \multicolumn{2}{c}{Snapshots (13)} & \multicolumn{2}{c}{Vintage level (78 pairs)} \\",
        r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}",
        r"Comparison & Run & wins & sign-test $p$ & error ratio & 95\% CI of log ratio \\",
        r"\midrule",
    ]
    key = "{} vs {}, E3a+E3b pooled"
    for label, a, b in (
        ("Full method vs chain ladder (primary)", "ours", "chain"),
        ("Calendar-adjusted CL vs chain ladder", "cal_chain", "chain"),
        ("Full method, forecast path, vs chain ladder", "ours_uncond", "chain"),
        ("Full method vs calendar-adjusted CL", "ours", "cal_chain"),
    ):
        for i, (name, run) in enumerate(runs.items()):
            if (a, b) == ("ours", "chain"):
                first = run["primary"][
                    "P1 (i) ours vs chain, E3a+E3b pooled, sign test over snapshots"
                ]
                second = run["primary"]["P1 (ii) ours vs chain, E3a+E3b pooled, vintage-level"]
            else:
                first = run["secondary"][key.format(a, b)]
                second = first["vintage_level"]
            lines.append(
                f"{label if i == 0 else ''} & {name} & {paired(first)} & {vintage(second)} \\\\"
            )
        lines.append(r"\addlinespace")
    for label, k in (
        ("Full method vs events so far", "P3 ours vs naive, E3a+E3b"),
        ("Full method vs drop recent", "P3 ours vs drop_recent, E3a+E3b"),
    ):
        for i, (name, run) in enumerate(runs.items()):
            lines.append(
                f"{label if i == 0 else ''} & {name} & {paired(run['primary'][k])} & & \\\\"
            )
        lines.append(r"\addlinespace")
    lines[-1:] = [r"\bottomrule", r"\end{tabular}"]  # no rule after the last group
    return "\n".join(lines) + "\n"


def table_e2(e2: dict, calendar: dict) -> str:
    window, error = e2["mean_window_by_n"], e2["median_error_by_n"]
    sizes = ("10000", "50000", "200000")
    lines = [
        r"\begin{tabular}{lcccccc}",
        r"\toprule",
        r" & \multicolumn{3}{c}{Mean chosen window (vintages)} & \multicolumn{3}{c}{Median relative error} \\",
        r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}",
        r"Loans per vintage & 10k & 50k & 200k & 10k & 50k & 200k \\",
        r"\midrule",
    ]
    for method, label in E2_ROWS.items():
        if method == "corrected_proven":
            lines.append(r"\midrule")
        # two decimals, as stored: a second rounding of these means could be wrong by 0.1
        cells = [f"{window[n][method]:.2f}" for n in sizes] + [
            pct(error[n][method], 0) for n in sizes
        ]
        lines.append(f"{label} & " + " & ".join(cells) + r" \\")
    lines.append(r"\midrule")
    for method, label in (
        ("cal_z", r"Calendar-adjusted maturity-weighted Lepski, $\kappa = z$"),
        ("cal_window_1", "Calendar-adjusted CL (window 1)"),
    ):
        lines.append(
            f"{label} & -- & -- & {calendar['mean_window'][method]:.2f} & -- & -- & "
            f"{pct(calendar['median_error'][method], 0)}" + r" \\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def table_harp(check: dict) -> str:
    """0017: default incidence of HARP relief loans against the incidence the censoring assumption implies."""
    lines = [
        r"\begin{tabular}{lrccccc}",
        r"\toprule",
        r" & & \multicolumn{3}{c}{Within 60 months} & \multicolumn{2}{c}{Within 36 months} \\",
        r"\cmidrule(lr){3-5}\cmidrule(lr){6-7}",
        r"Parent vintage & Relief loans in the sample & HARP path & At-risk path & Ratio & HARP path & At-risk path \\",
        r"\midrule",
    ]
    for long, short in zip(check["h60"]["by_vintage"], check["h36"]["by_vintage"], strict=True):
        if long["parent_vintage"].startswith("all"):
            lines.append(r"\midrule")
        lines.append(
            f"{long['parent_vintage'].replace('all 2003-2009', 'All')} & {long['children']:,} & "
            f"{pct(long['harp_path'], 2)} & {pct(long['at_risk_path'], 2)} & {long['ratio_harp_to_at_risk']:.2f} & "
            f"{pct(short['harp_path'], 2)} & {pct(short['at_risk_path'], 2)}" + r" \\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def table_e5(e5: dict) -> str:
    """0019: AUC, level calibration and slow-against-fast under-ranking per snapshot and model."""
    rows = {
        "naive_hgb": "Events so far",
        "drop_recent_hgb": "Drop recent",
        "fixed_horizon_hgb": "Last mature cohort",
        "dfm": "Delayed-feedback model",
        "ours_logit": "Full method, logistic head",
        "ours_hgb": "Full method, boosting head",
    }
    splits = ("under_ranking_fico_band", "under_ranking_purpose", "under_ranking_aft_speed_tercile")
    lines = [
        r"\begin{tabular}{llccccc}",
        r"\toprule",
        r" & & & & \multicolumn{3}{c}{Slow / fast ratio of predicted over realised} \\",
        r"\cmidrule(lr){5-7}",
        r"Snapshot & Model & AUC & Predicted / realised & FICO band & Purpose & AFT speed tercile \\",
    ]
    for year, s in e5.items():
        lines.append(r"\midrule")
        for i, (method, label) in enumerate(rows.items()):
            v = s["methods"][method]
            cells = [f"{year}" if i == 0 else "", label, f"{v['auc_within_vintage']:.3f}"]
            cells += [
                f"{v['predicted_over_realised']:.2f}",
                *[f"{v[k]['relative']:.2f}" for k in splits],
            ]
            lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def table_patex(run: dict) -> str:
    """0021: pre-registered patent backtest, main run: comparisons and wrong direction."""

    def row(label: str, v: dict) -> str:
        p = v["sign_test_p"]
        cells = [label, f"{v['wins']}/{v['n']}", "0.001" if p <= 0.001 else f"{p:.3f}"]
        if "vintage_level" in v:
            low, high = v["vintage_level"]["ci95"]
            cells += [ratio2(v["vintage_level"]), f"$[{low:+.2f},\\,{high:+.2f}]$"]
        else:
            cells += ["", ""]
        return " & ".join(cells) + r" \\"

    p5 = run["P5_wrong_direction"]["wrong_share"]
    lines = [
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r" & \multicolumn{2}{c}{Snapshots (11)} & \multicolumn{2}{c}{Filing-year level (43 pairs)} \\",
        r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}",
        r"Comparison & wins & sign-test $p$ & error ratio & 95\% CI of log ratio \\",
        r"\midrule",
        row("Full method vs chain ladder (primary)", run["primary"]["P1 ours vs chain"]),
        row(
            "Full method vs calendar-adjusted CL (primary)", run["primary"]["P2 ours vs cal_chain"]
        ),
        row("Full method vs events so far", run["P3"]["ours vs naive"]),
        row("Full method vs drop recent", run["P3"]["ours vs drop_recent"]),
        row("Full method vs last mature cohort", run["P3"]["ours vs fixed_horizon"]),
        row("Calendar-adjusted CL vs chain ladder", run["secondary"]["cal_chain vs chain"]),
        row("Full method w/o calendar vs chain ladder", run["secondary"]["ours_free vs chain"]),
        row(
            "Full method, forecast path, vs chain ladder",
            run["secondary"]["ours_uncond vs chain"],
        ),
        r"\midrule",
        f"Wrong direction of change & \\multicolumn{{4}}{{l}}{{events so far {pct(p5['naive'], 0)}, drop recent {pct(p5['drop_recent'], 0)}, chain ladder {pct(p5['chain'], 0)}, "
        f"calendar-adjusted CL {pct(p5['cal_chain'], 0)}, full method {pct(p5['ours'], 0)}}} \\\\",
        r"\bottomrule",
        r"\end{tabular}",
    ]
    return "\n".join(lines) + "\n"


def table_patex_errors(run: dict) -> str:
    """0021: error per snapshot and method (main run)."""
    methods = {
        "naive": "Events so far",
        "drop_recent": "Drop recent",
        "fixed_horizon": "Last mature cohort",
        "chain": "Chain ladder",
        "ours_free": "Full method w/o calendar",
        "cal_chain": "Calendar-adjusted CL",
        "ours": "Full method",
        "ours_uncond": "Full method, forecast path",
    }
    lines = [
        r"\begin{tabular}{l" + "c" * len(methods) + "}",
        r"\toprule",
        "Snapshot & " + " & ".join(methods.values()) + r" \\",
        r"\midrule",
    ]
    for year, s in run["snapshots"].items():
        lines.append(f"{year} & " + " & ".join(pct(s["error"][k]) for k in methods) + r" \\")
    lines += [
        r"\midrule",
        "Median & " + " & ".join(pct(run["median_error"][k]) for k in methods) + r" \\",
        r"\bottomrule",
        r"\end{tabular}",
    ]
    return "\n".join(lines) + "\n"


def fig_patex_speeds(check: dict) -> None:
    """0020: log speed per filing year (free fit), raw and calendar-adjusted, +/- 2 SE, relative to the mean of the
    filing years 2001-2008. The reference shape itself is pooled over the filing years 2001-2014."""
    fig, ax = canvas(0.36, 1.6)
    for key, label, colour, marker, shift in (
        ("raw", "raw", C["ink"], "s", -0.12),
        ("adjusted", "calendar-adjusted", C["clcal"], "o", 0.12),
    ):
        speed, error = check[key]["log_speed_free"], check[key]["log_speed_se"]
        years = np.array([int(y) for y in speed])
        value = np.array(list(speed.values()))
        value = value - value[years <= 2008].mean()
        ax.errorbar(
            years + shift,
            value,
            yerr=2 * np.array(list(error.values())),
            color=colour,
            marker=marker,
            ms=2.2,
            lw=0.9,
            capsize=1.0,
            label=label,
        )
    ax.axhline(0, color=C["muted"], lw=0.6, ls=":")
    ax.set_xticks(range(2001, 2020, 6))
    ax.set_xlabel("filing year")
    ax.set_ylabel("log speed\n(relative to 2001–2008)")
    ax.legend(frameon=False, loc="upper left", handlelength=1.6, borderpad=0, labelspacing=0.2)
    save(fig, "patex_speeds.pdf")


def table_patex_coverage(check: dict) -> str:
    """0020: PatEx counts per filing year and the estimated share of applications that PatEx does not hold."""
    lines = [
        r"\begin{tabular}{lrcccc}",
        r"\toprule",
        r"Filing year & Applications & Granted & Abandoned & Pre-grant publication & Estimated missing share \\",
        r"\midrule",
    ]
    for x in check["coverage"]:
        cells = [str(x["filing_year"]), f"{x['patex_applications']:,}"]
        cells += [
            pct(x[k]) for k in ("granted", "abandoned", "published", "estimated_missing_share")
        ]
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def table_decomposition(d: dict) -> str:
    """0022: where the gain of the calendar step comes from (error ratios; log-scale 95% CI in brackets)."""
    steps = (
        (
            "unit_chain vs chain",
            "CL + exact ages (no group effects, no covariates) vs chain ladder",
        ),
        ("stratified vs unit_chain", "Adding group effects"),
        ("cal_chain vs stratified", "Adding the calendar covariates"),
        ("cal_chain vs chain", "Total: calendar-adjusted CL vs chain ladder"),
    )
    lines = [
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r" & \multicolumn{2}{c}{Patents} & \multicolumn{2}{c}{Freddie Mac (borrower-followed)} \\",
    ]
    lines += [
        r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}",
        r"Step & error ratio & 95\% CI of log ratio & error ratio & 95\% CI of log ratio \\",
        r"\midrule",
    ]
    for key, label in steps:
        cells = [label]
        for name in ("patents", "freddie_borrower_followed"):
            c = d[name]["comparisons"][key]
            cells += [ratio2(c), f"$[{c['ci95'][0]:+.2f},\\,{c['ci95'][1]:+.2f}]$"]
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def timing_spread() -> dict[str, dict[str, float]]:
    """Coefficient of variation across the 1999-2012 vintages of (cumulative hazard / its value at 60 months),
    raw and calendar-adjusted, on the corrected 200k cells. Same definition as experiments/0003 (shock_overlay).
    Also draws the two sets of curves (timing_overlay.pdf)."""
    fig, axes = canvas(0.95, 2.3, ncols=2, sharey=True)
    cells = pd.read_parquet(
        EXP
        / "0012-freddie-200k-harp-correction"
        / "artifacts"
        / "hazard_cells_ipcw_subdist.parquet"
    )
    out = {}
    for name, column in (("raw", "exposure"), ("adjusted", "w_adj")):
        t = so._breslow(cells, column)
        t = t[t.vintage.isin(so.FIT_YEARS)]
        wide = t.pivot(index="age", columns="vintage", values="H").where(
            t.pivot(index="age", columns="vintage", values="risk") >= so.MIN_RISK
        )
        curves = wide / wide.loc[60]
        norm = curves.loc[[24, 36, 48, 72, 96]]
        out[name] = {
            str(a): round(float(v), 3) for a, v in (norm.std(axis=1) / norm.mean(axis=1)).items()
        }
        ax = axes[len(out) - 1]
        for vintage in curves.columns:
            shade = so.RAMP((vintage - so.FIT_YEARS.start) / (len(so.FIT_YEARS) - 1))
            ax.plot(curves.index, curves[vintage], color=shade, lw=1.0)
        ax.axhline(1, color=C["muted"], lw=0.6, ls=":")
        ax.set(xlim=(0, 120), ylim=(0, 2.4), xlabel="loan age (months)")
        ax.set_title("raw" if name == "raw" else "calendar-adjusted")
    axes[0].set_ylabel("cumulative hazard / value at 60 months")
    years = plt.Normalize(so.FIT_YEARS.start, so.FIT_YEARS.stop - 1)
    fig.colorbar(
        plt.cm.ScalarMappable(cmap=so.RAMP, norm=years),
        ax=axes,
        label="vintage",
        fraction=0.03,
        pad=0.02,
    )
    save(fig, "timing_overlay.pdf")
    return out


def main() -> None:
    for sub in ("figures", "tables"):
        (PAPER / sub).mkdir(exist_ok=True)
    as_run = load("0011-freddie-200k-confirmatory/results.json")
    fixed = load("0012-freddie-200k-harp-correction/results_ipcw.json")
    noharp = load("0012-freddie-200k-harp-correction/results_noharp.json")
    e2 = load("0014-e2-freddie-adaptive-windows/results.json")
    fig_e1(pd.read_csv(EXP / "0013-e1-adaptive-window-sim" / "artifacts" / "grid_full.csv"))
    fig_speed(load("0004-t2-calendar-simulations/results.json"))
    fig_e1_drift(pd.DataFrame(load("0013-e1-adaptive-window-sim/results.json")["sweep"]))
    fig_freddie_snapshots(as_run)
    runs = {"as run": as_run, "corrected": fixed, "no HARP rule": noharp}
    linked = load("0018-freddie-200k-follow-borrowers/results_linked.json")
    runs["follows borrowers"] = linked
    (PAPER / "tables" / "freddie.tex").write_text(table_freddie(as_run, fixed, noharp, linked))
    (PAPER / "tables" / "freddie_tests.tex").write_text(table_freddie_tests(runs))
    calendar = load("0016-e2-calendar-adjusted-windows/results.json")
    (PAPER / "tables" / "e2.tex").write_text(table_e2(e2, calendar))
    (PAPER / "tables" / "overid.tex").write_text(table_overid())
    (PAPER / "tables" / "e5.tex").write_text(
        table_e5(load("0019-e5-freddie-features/results.json"))
    )
    patex = load("0021-patex-backtest/results.json")["main"]
    (PAPER / "tables" / "patex.tex").write_text(table_patex(patex))
    (PAPER / "tables" / "patex_errors.tex").write_text(table_patex_errors(patex))
    feasibility = load("0020-patex-feasibility/results.json")
    fig_patex_speeds(feasibility["all_utility"])
    (PAPER / "tables" / "patex_coverage.tex").write_text(table_patex_coverage(feasibility))
    decomposition = load("0022-group-stratified-chain-ladder/results.json")
    (PAPER / "tables" / "decomposition.tex").write_text(table_decomposition(decomposition))
    harp = load("0017-harp-informative-censoring/results.json")
    (PAPER / "tables" / "harp_check.tex").write_text(table_harp(harp))
    overid = {
        f"{name}, {covariates}": json.loads(path.read_text())
        for name, covariates, path in (
            (
                "as run",
                "base",
                EXP / "0011-freddie-200k-confirmatory/artifacts/overid_test_subdist.json",
            ),
            (
                "as run",
                "mtm",
                EXP / "0011-freddie-200k-confirmatory/artifacts/overid_test_mtm_subdist.json",
            ),
            (
                "corrected",
                "base",
                EXP / "0012-freddie-200k-harp-correction/artifacts/overid_test_ipcw_subdist.json",
            ),
            (
                "corrected",
                "mtm",
                EXP
                / "0012-freddie-200k-harp-correction/artifacts/overid_test_mtm_ipcw_subdist.json",
            ),
        )
    }
    out = {"timing_spread_cv_200k_corrected_cells": timing_spread(), "overid_reject_shares": overid}
    (HERE / "results.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
