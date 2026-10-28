"""Markdown tables for the README: as run (experiments/0011) next to the post hoc correction and the no-HARP check.

Built only from results files and artifacts (no new analysis). Usage: report.py > artifacts/tables.md
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Final

HERE: Final = Path(__file__).resolve().parent
AS_RUN: Final = HERE.parent / "0011-freddie-200k-confirmatory"
sys.path.insert(0, str(AS_RUN))
import report as rep

RUNS: Final = {
    "as run (0011)": AS_RUN / "results.json",
    "corrected (post hoc)": HERE / "results_ipcw.json",
    "no HARP rule (check)": HERE / "results_noharp.json",
}
METHODS: Final = ("chain", "ours_free", "cal_chain", "ours", "ours_uncond", "ours_rolling")


def sign(v: dict) -> str:
    return f"{v['wins']}/{v['n']}, p = {v['sign_test_p']}; {100 * v['median_paired_difference']:+.1f} pts"


def vint(v: dict) -> str:
    return f"{v['mean_paired_log_difference']:+.3f} [{v['ci95'][0]:+.3f}, {v['ci95'][1]:+.3f}]; ratio {v['error_ratio']}"


def rows(s: dict) -> dict[str, str]:
    p, q, c = s["primary"], s["secondary"], s["P4_conformal"]
    pooled = {k.split(",")[0]: v for k, v in q.items() if k.endswith("pooled")}
    out = {
        "**P1 (i)** ours vs chain, snapshots": sign(
            p["P1 (i) ours vs chain, E3a+E3b pooled, sign test over snapshots"]
        ),
        "**P1 (ii)** ours vs chain, vintage level": vint(
            p["P1 (ii) ours vs chain, E3a+E3b pooled, vintage-level"]
        ),
        "descriptive, E3a": sign(p["descriptive: ours vs chain, E3a"])
        + "; "
        + vint(p["descriptive: ours vs chain, E3a"]["vintage_level"]),
        "descriptive, E3b": sign(p["descriptive: ours vs chain, E3b"])
        + "; "
        + vint(p["descriptive: ours vs chain, E3b"]["vintage_level"]),
        "**P3** ours vs naive": sign(p["P3 ours vs naive, E3a+E3b"]),
        "**P3** ours vs drop recent": sign(p["P3 ours vs drop_recent, E3a+E3b"]),
        "**P4** coverage, J >= 9 (n)": f"{rep.pct(c['ours']['coverage_M1.0_J_ge_9'])} ({c['ours']['with_J_ge_9']})",
        "**P4** coverage, J >= 1 (n)": f"{rep.pct(c['ours']['coverage_M1.0_all_with_J'])} ({c['ours']['with_J_ge_1']})",
        "**P4** width ratio vs chain, J >= 9": str(c["width_ratio_ours_over_chain_J_ge_9"]["M1.0"]),
        "P4 chain ladder coverage, J >= 9": rep.pct(c["chain"]["coverage_M1.0_J_ge_9"]),
        "P4 signed version, J >= 9": rep.pct(c["ours"]["coverage_signed_J_ge_9"]),
    }
    for name, v in pooled.items():
        out[f"secondary: {name}"] = f"{sign(v)}; {vint(v['vintage_level'])}"
    out["secondary: ours vs chain, E3c"] = sign(q["ours vs chain, E3c"])
    for e in ("E3a", "E3b", "E3c"):
        med = s[e]["median_error"]
        out[f"median error {e}: chain / ours w/o cal / cal + chain / ours / ours uncond"] = (
            " / ".join(rep.pct(med[m]) for m in METHODS[:5])
        )
    h = s["H10"]
    out["H = 10: chain / cal + chain / ours / ours uncond; ours vs chain"] = (
        " / ".join(
            rep.pct(h["median_error"][m]) for m in ("chain", "cal_chain", "ours", "ours_uncond")
        )
        + f"; {sign(h['ours vs chain'])}"
    )
    for e in ("E3a", "E3b"):
        m = s["MTM"][e]
        out[f"MTM {e}: chain / cal + chain / ours / ours uncond; ours vs chain"] = (
            " / ".join(
                rep.pct(m["median_error"][k]) for k in ("chain", "cal_chain", "ours", "ours_uncond")
            )
            + f"; {sign(m['ours vs chain'])}"
        )
    return out


def side_by_side(runs: dict[str, dict]) -> str:
    cols = {name: rows(s) for name, s in runs.items()}
    lines = ["| result | " + " | ".join(cols) + " |", "|---|" + "---|" * len(cols)]
    lines += [
        f"| {k} | " + " | ".join(c[k] for c in cols.values()) + " |"
        for k in next(iter(cols.values()))
    ]
    return "\n".join(lines)


def per_snapshot(runs: dict[str, dict]) -> str:
    names = list(runs)
    head = ["set", "snapshot", "chain ladder (as run)", "chain ladder (no HARP rule)"]
    head += [f"{m}, {n}" for m in ("calendar + chain", "ours", "ours uncond.") for n in names]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for e in ("E3a", "E3b"):
        for year in runs[names[0]][e]["snapshots"]:
            err = {n: runs[n][e]["snapshots"][year]["error"] for n in names}
            row = [
                e,
                f"{year}-12",
                rep.pct(err[names[0]]["chain"]),
                rep.pct(err[names[2]]["chain"]),
            ]
            row += [rep.pct(err[n][m]) for m in ("cal_chain", "ours", "ours_uncond") for n in names]
            lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def full_tables(variant: str, s: dict) -> str:
    rep.ART = HERE / "artifacts" / variant  # type: ignore[misc]  # the 0011 report reads from its module-level ART
    parts = [f"### {n}\n{rep.snapshot_table(s, n)}" for n in ("E3a", "E3b", "E3c")]
    parts += [
        f"### Primary\n{rep.paired_table(s['primary'])}",
        f"### P4\n{rep.conformal_table(s['P4_conformal'])}",
        f"### Secondary\n{rep.paired_table(s['secondary'])}",
        f"### T3 diagnostic\n{rep.t3_table()}",
    ]
    rep.ART = HERE / "artifacts"  # type: ignore[misc]
    parts += [
        f"### Over-identification, base covariates\n{rep.overid_table(f'_{variant}_subdist')}",
        f"### Over-identification, MTM\n{rep.overid_table(f'_mtm_{variant}_subdist')}",
    ]
    return "\n\n".join(parts)


def main() -> None:
    runs = {name: json.loads(path.read_text()) for name, path in RUNS.items()}
    parts = [
        f"## Side by side\n{side_by_side(runs)}",
        f"## Per snapshot\n{per_snapshot(runs)}",
        f"## Full tables: corrected\n{full_tables('ipcw', runs['corrected (post hoc)'])}",
        f"## Full tables: no HARP rule\n{full_tables('noharp', runs['no HARP rule (check)'])}",
    ]
    sys.stdout.write("\n\n".join(parts) + "\n")


if __name__ == "__main__":
    main()
