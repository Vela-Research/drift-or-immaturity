"""Run the steps behind a paper table and check the result against the paper.

Usage:
  uv run python reproduce.py list                    the targets
  uv run python reproduce.py <target>                print the steps
  uv run python reproduce.py <target> --run [--from N]   run them (from step N), then render and check
  uv run python reproduce.py <target> --check        only render and check (the steps have run)
  uv run python reproduce.py figdata                 Figure 1(c) and the simulation-by-age data, from finished runs

Targets with a check: `tab:harp-check` (mortgages, about 50 minutes: data build, rebuilt loan ages, the HARP check)
and `tab:replication` (Table 3, about 1.5 hours: downloads and the whole replication). The README maps every other
table and figure to its experiment folder and commands.
"""

from __future__ import annotations

import importlib.util
import json
import os
import shlex
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path
from types import ModuleType
from typing import Final

import pandas as pd

ROOT: Final = Path(__file__).resolve().parent
EXP: Final = ROOT / "experiments"
FIGDATA: Final = ROOT / "research" / "drift-or-maturity" / "figdata"
TREE72: Final = EXP / "0072-auditmp-reruns" / "artifacts" / "tree"
Step = tuple[str, list[str]]  # (experiment folder, command after `python`)


@dataclass(frozen=True)
class Target:
    steps: list[Step]
    check: Callable[[], bool]


def harp_steps() -> list[Step]:
    """Loan ages as recorded -> rebuilt ages (F1) -> the HARP check under F1 and F2 in 0072's tree."""
    return [
        ("0010-freddie-200k-build", ["build.py"]),
        ("0018-freddie-200k-follow-borrowers", ["link.py"]),
        ("0071-mortgage-calendar-age", ["run.py", "check"]),
        ("0071-mortgage-calendar-age", ["run.py", "data"]),
        # 0071's tree copies the patent-only outputs of 0066, which the HARP check does not read: an empty folder will do
        ("0066-patent-hazard-baseline", ["-c", 'import pathlib; pathlib.Path("artifacts").mkdir(exist_ok=True)']),
        ("0071-mortgage-calendar-age", ["run.py", "tree"]),
        ("0072-auditmp-reruns", ["run.py", "check"]),
        ("0072-auditmp-reruns", ["run.py", "data"]),
        ("0072-auditmp-reruns", ["run.py", "tree"]),
        ("0072-auditmp-reruns", ["run.py", "exec", "f1,f2", "0017-harp-informative-censoring", "run.py"]),
    ]


def replication_steps() -> list[Step]:
    """The replication as run (0068), then the audit's corrections (0069), which the paper reports."""
    rep, fix = "0068-confirm2-replication", "0069-confirm2-audit-fixes"
    steps: list[Step] = [(rep, ["download.py"]), (rep, ["build.py"]), (rep, ["gate.py"])]
    for name in ("sba", "civil", "civil_broad"):
        steps += [(rep, ["backtest.py", name]), (rep, ["hazard_m1.py", name])]
    steps += [(rep, ["classify.py", "sba"]), (rep, ["classify.py", "civil"])]
    for name in ("sba", "civil"):
        steps += [(rep, ["secondary.py", name, "windows"]), (rep, ["secondary.py", name, "drift"])]
    steps += [(rep, ["pif_check.py"]), (rep, ["report.py"]), (fix, ["run.py", "link"])]
    for step, name in (
        ("backtest", "civil"),
        ("backtest", "civil_broad"),
        ("hazard", "civil"),
        ("hazard", "civil_broad"),
        ("classify", "sba"),
        ("classify", "civil"),
        ("windows", "sba"),
        ("windows", "civil"),
        ("drift", "civil"),
    ):
        steps.append((fix, ["run.py", step, name]))
    return steps + [(fix, ["run.py", "pif"]), (fix, ["run.py", "report"])]


# ---------------------------------------------------------------- checks against the paper
HARP_PAPER: Final = r"""\begin{tabular}{lrccccc}
\toprule
 & & \multicolumn{3}{c}{Within 60 months} & \multicolumn{2}{c}{Within 36 months} \\
\cmidrule(lr){3-5}\cmidrule(lr){6-7}
Parent vintage & Relief loans in the sample & HARP path & At-risk path & Ratio & HARP path & At-risk path \\
\midrule
2003 & 38,269 & 0.75\% & 2.28\% & 0.33 & 0.48\% & 1.77\% \\
2004 & 25,717 & 1.24\% & 4.10\% & 0.30 & 0.77\% & 3.19\% \\
2005 & 38,635 & 1.95\% & 6.75\% & 0.29 & 1.29\% & 5.28\% \\
2006 & 32,340 & 3.00\% & 10.44\% & 0.29 & 1.97\% & 8.17\% \\
2007 & 39,917 & 3.08\% & 11.11\% & 0.28 & 2.00\% & 8.66\% \\
2008 & 45,754 & 2.29\% & 7.76\% & 0.29 & 1.49\% & 6.01\% \\
2009 & 27,391 & 0.49\% & 1.32\% & 0.37 & 0.32\% & 0.98\% \\
\midrule
All & 248,023 & 1.91\% & 6.56\% & 0.29 & 1.24\% & 5.10\% \\
\bottomrule
\end{tabular}
"""
REPLICATION_PAPER: Final = {  # Table 3: ratio [95% CI], Holm-adjusted one-sided p, verdict
    ("sba", "H1"): "0.27 [0.20, 0.38], <0.001, confirmed",
    ("sba", "H2"): "1.06 [0.85, 1.31], 0.034, confirmed",
    ("sba", "H3"): "0.86 [0.71, 1.05], 0.016, confirmed",
    ("civil", "H1"): "0.09 [0.07, 0.12], <0.001, confirmed",
    ("civil", "H2"): "1.08 [0.78, 1.52], 0.088, inconclusive",
    ("civil", "H3"): "0.62 [0.39, 0.95], 0.011, confirmed",
}


def check_harp() -> bool:
    build = module(EXP / "0015-paper-assets" / "build.py", "build15")
    results = TREE72 / "experiments" / "0017-harp-informative-censoring" / "results.json"
    table = build.table_harp(json.loads(results.read_text()))
    return report("tab:harp-check", table.splitlines(), HARP_PAPER.splitlines())


def check_replication() -> bool:
    r = json.loads((EXP / "0069-confirm2-audit-fixes" / "results.json").read_text())
    ours, paper = [], []
    for (name, h), expected in REPLICATION_PAPER.items():
        x = r[name]["confirmatory"][h]
        lo, hi = x["ci95"]
        p = "<0.001" if x["p_holm"] < 0.0005 else half_up(x["p_holm"], 3)
        ours.append(f"{name} {h}: {half_up(x['error_ratio'], 2)} [{half_up(lo, 2)}, {half_up(hi, 2)}], {p}, {x['verdict']}")
        paper.append(f"{name} {h}: {expected}")
    return report("tab:replication", ours, paper)


def check_downloads() -> None:
    """Compare the downloaded files with the sha256 recorded when the replication was run."""
    reference = json.loads((ROOT / "preregistration" / "confirm2_data_manifest.json").read_text())
    got = json.loads((EXP / "0068-confirm2-replication" / "artifacts" / "manifest.json").read_text())
    differ = [name for name, x in reference.items() if got.get(name, {}).get("sha256") != x["sha256"]]
    print(f"downloads: {len(reference) - len(differ)} of {len(reference)} files equal to the recorded sha256")
    for name in differ:
        print(f"  differs: {name} (the source has changed since the run; results can differ)")


# ---------------------------------------------------------------- figure data from finished runs
TEASER: Final = {  # Figure 1(c): e2.tex row -> series
    "Naive Lepski": "naive_lepski",
    "Naive ADWIN": "naive_adwin",
    "Maturity-weighted Lepski, $\\kappa = z$": "maturity_weighted_kappa_z",
    "Calendar-adjusted maturity-weighted Lepski, $\\kappa = z$": "calendar_adjusted_maturity_weighted_kappa_z",
    "Calendar-adjusted CL (window 1)": "calendar_adjusted_cl",
}


def figdata() -> bool:
    """teaser_mortgage_windows.csv (Figure 1(c), from tab:e2 of 0015 in 0072's tree) and sim_by_age_corrected.csv
    (from the simulation re-run of 0037 in 0072's tree)."""
    e2 = TREE72 / "research" / "drift-or-maturity" / "paper" / "tables" / "e2.tex"
    rows = []
    for line in e2.read_text().splitlines():
        cells = [c.strip() for c in line.removesuffix("\\\\").split("&")]
        if cells[0] in TEASER:
            for n, cell in zip((10_000, 50_000, 200_000), cells[4:7], strict=True):
                if cell != "--":
                    rows.append({"series": TEASER[cells[0]], "loans_per_vintage": n, "median_rel_error_pct": int(cell[:-2])})
    FIGDATA.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(FIGDATA / "teaser_mortgage_windows.csv", index=False)
    sim = json.loads((TREE72 / "experiments" / "0037-corrected-plugin-simulation" / "results.json").read_text())
    sim_by_age(sim["settings"]).to_csv(FIGDATA / "sim_by_age_corrected.csv", index=False)
    print(f"wrote {FIGDATA}/teaser_mortgage_windows.csv ({len(rows)} rows), sim_by_age_corrected.csv")
    return True


def sim_by_age(settings: dict[str, dict[str, dict[str, object]]]) -> pd.DataFrame:
    rows = []
    for s, f in settings["frozen"].items():
        c = settings["corrected"][s]
        timing, speeds, calendar, size = s.split("/")
        for a in ("6", "18", "30"):
            rec = c["speed_recovery_by_age"].get(a, {})  # type: ignore[union-attr]
            lo, hi = c["gain_from_speeds_by_age"][a]["ci95"]  # type: ignore[index]
            rows.append(
                {
                    "timing": timing,
                    "speeds": speeds,
                    "calendar_effect": calendar,
                    "size": size,
                    "age_months": int(a),
                    "events_so_far_median": c["events_so_far_median_by_age"][a],  # type: ignore[index]
                    "error_ratio_frozen": f["gain_from_speeds_by_age"][a]["ratio"],  # type: ignore[index]
                    "error_ratio_corrected": c["gain_from_speeds_by_age"][a]["ratio"],  # type: ignore[index]
                    "ci95_low_corrected": lo,
                    "ci95_high_corrected": hi,
                    "median_abs_error_equal_speeds": c["median_abs_error_by_age"][a]["cal_chain"],  # type: ignore[index]
                    "median_abs_error_full_frozen": f["median_abs_error_by_age"][a]["ours"],  # type: ignore[index]
                    "median_abs_error_full_corrected": c["median_abs_error_by_age"][a]["ours"],  # type: ignore[index]
                    "fitted_over_true_log_speed_median": rec.get("full_median_ratio_to_true_log_speed"),
                    "share_right_direction": rec.get("full_right_direction"),
                    "coverage_frozen": f["coverage"]["ours"][f"age_{a}"]["coverage"],  # type: ignore[index]
                    "coverage_corrected": c["coverage"]["ours"][f"age_{a}"]["coverage"],  # type: ignore[index]
                    "replications": c["replications"],
                }
            )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- helpers
TARGETS: Final = {
    "tab:harp-check": Target(harp_steps(), check_harp),
    "tab:replication": Target(replication_steps(), check_replication),
    "figdata": Target([], figdata),
}


def half_up(x: float, digits: int) -> str:
    return str(Decimal(repr(x)).quantize(Decimal(1).scaleb(-digits), rounding=ROUND_HALF_UP))


def module(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


def report(name: str, ours: list[str], paper: list[str]) -> bool:
    same = [a == b for a, b in zip(ours, paper, strict=True)]
    for a, b, ok in zip(ours, paper, same, strict=True):
        print(f"{'  ' if ok else '!!'} {a}" + ("" if ok else f"\n   paper: {b}"))
    print(f"{name}: {sum(same)} of {len(same)} lines equal to the paper")
    return all(same)


def run(steps: list[Step], start: int) -> None:
    env = {**os.environ, "PYTHONHASHSEED": "0"}
    for i, (folder, argv) in enumerate(steps, 1):
        if i < start:
            continue
        print(f"[{i}/{len(steps)}] {folder}: python {shlex.join(argv)}", flush=True)
        subprocess.run([sys.executable, *argv], cwd=EXP / folder, env=env, check=True)
        if argv == ["download.py"]:
            check_downloads()


def main(args: list[str]) -> int:
    if not args or args[0] == "list":
        for name, t in TARGETS.items():
            print(f"{name}: {len(t.steps)} steps")
        return 0
    target = TARGETS[args[0]]
    if "--run" in args:
        start = int(args[args.index("--from") + 1]) if "--from" in args else 1
        run(target.steps, start)
    elif "--check" not in args and target.steps:
        for i, (folder, argv) in enumerate(target.steps, 1):
            print(f"[{i}] cd experiments/{folder} && PYTHONHASHSEED=0 uv run python {shlex.join(argv)}")
        return 0
    return 0 if target.check() else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
