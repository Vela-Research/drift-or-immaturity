"""Markdown tables for the README, built only from artifacts/ (no new analysis).

Usage: report.py > artifacts/tables.md
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd

ART: Final = Path(__file__).resolve().parent / "artifacts"
COLS: Final = {
    "naive": "naive",
    "drop_recent": "drop recent",
    "fixed_horizon": "fixed horizon",
    "chain": "chain ladder",
    "ours_free": "ours w/o calendar",
    "cal_chain": "calendar + chain",
    "ours": "**ours** (cond.)",
    "ours_uncond": "ours (uncond.)",
    "ours_rolling": "ours, rolling-origin omega2",
}
AGE_BANDS: Final = [0, 12, 24, 36, 48, 60, 84]


def pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def snapshot_table(s: dict, name: str) -> str:
    e = s[name]
    cols = [m for m in COLS if m in e["median_error"]]
    head = ["snapshot", "n", "phi", "model", "G1", "G2 cond / uncond", *[COLS[m] for m in cols]]
    if name != "E3c":
        head.append("rolling omega2 (h)")
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for year, r in e["snapshots"].items():
        g2 = r["g2_out_of_support"]
        row = [
            f"{year}-12",
            str(r["n"]),
            str(r["phi"]),
            r["model"],
            f"{r['g1_events_x_min_share']:,.0f}",
            f"{pct(g2['cond'])} / {pct(g2['uncond'])}",
            *[pct(r["error"][m]) for m in cols],
        ]
        if name != "E3c":
            row.append(f"{r['rolling_omega2']} ({r['rolling_h']})")
        lines.append("| " + " | ".join(row) + " |")
    snaps = list(e["snapshots"].values())
    pad = [""] * (5 + (name != "E3c"))
    med = ["**median**", *pad[:5], *[pct(e["median_error"][m]) for m in cols], *pad[5:]]
    win = [
        "win rate vs chain",
        *pad[:5],
        *[
            "-"
            if m == "chain"
            else f"{100 * np.mean([r['error'][m] < r['error']['chain'] for r in snaps]):.0f}%"
            for m in cols
        ],
        *pad[5:],
    ]
    return "\n".join([*lines, "| " + " | ".join(med) + " |", "| " + " | ".join(win) + " |"])


def paired_table(d: dict) -> str:
    lines = [
        "| comparison | median paired difference | wins / non-tied snapshots | sign test p | mean paired log difference [95% CI] (pairs, vintage clusters) | error ratio |",
        "|---|---|---|---|---|---|",
    ]
    for k, v in d.items():
        vl = v.get("vintage_level") if "wins" in v else v
        sign = (
            f"{100 * v['median_paired_difference']:+.1f} points | {v['wins']} / {v['n']} | {v['sign_test_p']}"
            if "wins" in v
            else " | | "
        )
        vint = (
            f"{vl['mean_paired_log_difference']:+.3f} [{vl['ci95'][0]:+.3f}, {vl['ci95'][1]:+.3f}] ({vl['pairs']}, {vl['vintage_clusters']}) | {vl['error_ratio']}"
            if vl
            else " | "
        )
        lines.append(f"| {k} | {sign} | {vint} |")
    return "\n".join(lines)


def conformal_table(c: dict) -> str:
    lines = [
        "| method | scored vintages | J >= 1 | J >= 9 | **coverage M = 1, J >= 9** | coverage M = 1, J >= 1 | M = 1.5 (J >= 9 / J >= 1) | M = 2 (J >= 9 / J >= 1) | signed (J >= 9 / J >= 1) | median width (upper / lower), M = 1 | median M* |",
        "|" + "---|" * 11,
    ]
    for m in ("ours", "chain"):
        r = c[m]
        lines.append(
            f"| {m} | {r['vintages']} | {r['with_J_ge_1']} | {r['with_J_ge_9']} | **{pct(r['coverage_M1.0_J_ge_9'])}** | {pct(r['coverage_M1.0_all_with_J'])} | "
            f"{pct(r['coverage_M1.5_J_ge_9'])} / {pct(r['coverage_M1.5_all_with_J'])} | {pct(r['coverage_M2.0_J_ge_9'])} / {pct(r['coverage_M2.0_all_with_J'])} | "
            f"{pct(r['coverage_signed_J_ge_9'])} / {pct(r['coverage_signed_all_with_J'])} | {r['median_width_ratio_M1.0']} | {r['m_star_median']} |"
        )
    w = c["width_ratio_ours_over_chain_J_ge_9"]
    notes = [
        f"- Interval-width ratio ours / chain ladder, median over vintages with J >= 9: M = 1: {w['M1.0']}; M = 1.5: {w['M1.5']}; M = 2: {w['M2.0']}.",
        *[
            f"- {m}: coverage at M = 1 by set, J >= 9: {c[m]['coverage_M1_J_ge_9_by_set']}; J >= 1: {c[m]['coverage_M1_by_set']}; median J by set: {c[m]['J_by_set']}."
            for m in ("ours", "chain")
        ],
    ]
    return "\n".join([*lines, "", *notes])


def t3_table() -> str:
    v = pd.concat([pd.read_csv(ART / f"vintages_{n}.csv") for n in ("E3a", "E3b")])
    v["band"] = pd.cut(v.A, AGE_BANDS, right=False)
    lines = [
        "| age at snapshot (months) | scored vintages | median maturity m | median inflation factor (lower bound) | own speed not learnable (m < 0.3) | median error, chain ladder | median error, ours |",
        "|---|---|---|---|---|---|---|",
    ]
    for band, g in v.groupby("band", observed=True):
        err = {m: float((np.abs(g[m] - g.truth) / g.truth).median()) for m in ("chain", "ours")}
        lines.append(
            f"| {band.left}-{band.right - 1} | {len(g)} | {g.maturity_m.median():.2f} | {g.inflation_lower_bound.median():.1f} | "
            f"{pct(float(g.own_speed_not_learnable.mean()))} | {pct(err['chain'])} | {pct(err['ours'])} |"
        )
    return "\n".join(lines)


def overid_table(suffix: str) -> str:
    j = json.loads((ART / f"overid_test{suffix}.json").read_text())
    t = pd.read_csv(ART / f"overid_test{suffix}.csv")
    lines = [
        f"phi = {j['phi']}. Rejection share at 5%: vintages 1999-2012 {pct(j['reject_share_1999_2012'])}, 2013-2019 {pct(j['reject_share_2013_2019'])} "
        f"(without tempering: {pct(j['reject_share_untempered_1999_2012'])} and {pct(j['reject_share_untempered_2013_2019'])}).",
        "",
        "| vintage | events (age <= 72) | early speed | late speed | effect size log(late / early) | T = LR / phi | p | rejects |",
        "|---|---|---|---|---|---|---|---|",
    ]
    lines += [
        f"| {int(r.vintage)} | {int(r.events):,} | {r.s_early:.2f} | {r.s_late:.2f} | {r.log_ratio:+.2f} | {r['T']:.1f} | {r.p_value:.3f} | {'yes' if r.rejects else 'no'} |"
        for _, r in t.iterrows()
    ]
    return "\n".join(lines)


def main() -> None:
    s = json.loads((ART / "summary.json").read_text())
    parts = [
        *[f"## {n}\n{snapshot_table(s, n)}" for n in ("E3a", "E3b", "E3c")],
        f"## Primary\n{paired_table(s['primary'])}",
        f"## P4\n{conformal_table(s['P4_conformal'])}",
        f"## Secondary\n{paired_table(s['secondary'])}",
        "## H10\n"
        + f"median error: { {COLS[k]: pct(v) for k, v in s['H10']['median_error'].items() if k in COLS} }\n\n"
        + paired_table({"ours vs chain, H = 10": s["H10"]["ours vs chain"]}),
        "## MTM\n"
        + "\n".join(
            f"- {n}: median error { {COLS[k]: pct(v) for k, v in s['MTM'][n]['median_error'].items() if k in COLS} }; ours vs chain {s['MTM'][n]['ours vs chain']}"
            for n in ("E3a", "E3b")
        ),
        f"## T3 diagnostic (E3a + E3b scored vintages)\n{t3_table()}",
        f"## Over-identification, base covariates\n{overid_table('_subdist')}",
        f"## Over-identification, MTM\n{overid_table('_mtm_subdist')}",
    ]
    sys.stdout.write("\n\n".join(parts) + "\n")


if __name__ == "__main__":
    main()
