"""Round 3 tables for the paper (POST HOC analyses of 0031 and 0032), with the v3 names.

Usage: uv run python round3_tables.py -> paper/tables/{baselines,inference,inference_sensitivity,speed_criterion}.tex
Sources: experiments/0031-round3-backtests (results.json, pair files), 0032-mortgage-hazard-baseline (pair file),
0035-speed-criterion-fisher (results.json).
Mortgage rows use the corrected censoring weights (the as-run version has the frozen methods only and is in Table 1).
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd

from maturity import inference

EXP: Final = Path(__file__).resolve().parents[1]
TABLES: Final = EXP.parent / "research" / "drift-or-maturity" / "paper" / "tables"
Block = dict[str, dict]
# key -> (row label, path of the comparison against the chain ladder)
ROWS: Final = (
    ("naive", "Events so far"),
    ("drop_recent", "Drop recent"),
    ("fixed_horizon", "Last mature cohort"),
    ("bf", "Bornhuetter--Ferguson"),
    ("cape_cod", "Cape Cod"),
    ("gcc_075", "Generalized Cape Cod (decay 0.75)"),
    ("chain", "Chain ladder (CL)"),
    ("unit_chain", "CL + exact ages"),
    ("stratified", "CL + exact ages + groups"),
    ("hazard_vintage_real", "Hazard model, realised path"),
    ("hazard_vintage_fcst", "Hazard model, forecast path"),
    ("hazard_real", "Hazard model w/o vintage effects, realised path"),
    ("hazard_fcst", "Hazard model w/o vintage effects, forecast path"),
    ("cal_chain", "Calendar-adjusted CL, realised path"),
    ("cal_chain_forecast", "Calendar-adjusted CL, forecast path"),
    ("ours", "Full method, realised path"),
    ("ours_uncond", "Full method, forecast path"),
)
TESTS: Final = (
    ("ours vs chain", "Full method vs CL"),
    ("ours_uncond vs chain", "Full method, forecast path, vs CL"),
    ("ours vs stratified", "Full method vs CL + exact ages + groups"),
    ("ours_uncond vs stratified", "Full method, forecast path, vs CL + exact ages + groups"),
    ("unit_chain vs chain", "Step: exact ages"),
    ("stratified vs unit_chain", "Step: groups"),
    ("cal_chain vs stratified", "Step: calendar, realised path"),
    ("cal_chain_forecast vs stratified", "Step: calendar, forecast path"),
    ("ours vs cal_chain", "Step: speeds, realised path"),
    ("ours_uncond vs cal_chain_forecast", "Step: speeds, forecast path"),
)


def pct(x: float) -> str:
    return f"{100 * x:.0f}\\%"


def p_value(p: float) -> str:
    return "$<$0.001" if p < 0.0005 else f"{p:.3f}"


def ratio_ci(c: dict, key: str = "ci95_cluster_bootstrap") -> str:
    low, high = (math.exp(x) for x in c[key])
    return f"[{low:.2f}, {high:.2f}]"


def against_chain(pairs: pd.DataFrame, key: str) -> str:
    """Error ratio of a method to the chain ladder with its cohort-cluster interval, and the sign-flip p value."""
    if key == "chain":
        return "1 & "
    if key not in pairs:
        return " & "
    error = {m: ((pairs[m] - pairs.truth) / pairs.truth).to_numpy() for m in (key, "chain")}
    c = inference.compare(
        error[key],
        error["chain"],
        pairs.cohort.to_numpy().astype(np.int64),
        pairs.snapshot.to_numpy().astype(np.int64),
    )
    low, high = (math.exp(x) for x in c.ci95_cluster_bootstrap)
    return f"{c.error_ratio:.2f} [{low:.2f}, {high:.2f}] & {p_value(c.sign_flip_p)}"


def median_error(pairs: pd.DataFrame, key: str) -> str:
    """Median over snapshots of the mean absolute relative error of the scored cohorts."""
    if key not in pairs:
        return ""
    error = ((pairs[key] - pairs.truth) / pairs.truth).abs()
    return pct(float(error.groupby(pairs.snapshot).mean().median()))


def baselines(mortgages: pd.DataFrame, patents: pd.DataFrame) -> str:
    crisis, pre_covid = (mortgages[mortgages.set == k] for k in ("E3a", "E3b"))
    lines = [
        r"\begin{tabular}{@{}lccccccc@{}}",
        r"\toprule",
        rf" & \multicolumn{{4}}{{c}}{{Mortgages ({len(mortgages)} pairs, {mortgages.cohort.nunique()} vintages)}} & "
        rf"\multicolumn{{3}}{{c}}{{Patent applications ({len(patents)} pairs, {patents.cohort.nunique()} filing years)}} \\",
        r"\cmidrule(lr){2-5}\cmidrule(l){6-8}",
        r" & \multicolumn{2}{c}{median error} & ratio to CL & $p$ & median & ratio to CL & $p$ \\",
        r" & crisis & pre-COVID & [95\% CI] & & error & [95\% CI] & \\",
        r"\midrule",
    ]
    for key, label in ROWS:
        cells = [
            label,
            median_error(crisis, key),
            median_error(pre_covid, key),
            against_chain(mortgages, key),
            median_error(patents, key),
            against_chain(patents, key),
        ]
        lines.append(" & ".join(cells) + r" \\")
        if key in ("gcc_075", "stratified", "hazard_fcst"):
            lines.append(r"\addlinespace")
    return "\n".join([*lines, r"\bottomrule", r"\end{tabular}", ""])


def tests(blocks: dict[str, Block], *, sensitivity: bool) -> str:
    head = (
        r" & MAE ratio & $p$ & floor $10^{-4}$ & floor $10^{-3}$ & 5\% trimmed \\"
        if sensitivity
        else r" & ratio [95\% CI] & sign-flip $p$ & wild $p$ & two-way 95\% CI & leave-one-out & worst $p$ \\"
    )
    lines = [
        r"\begin{tabular}{@{}l" + ("ccccc" if sensitivity else "cccccc") + r"@{}}",
        r"\toprule",
        head,
    ]
    for name, block in blocks.items():
        first = next(iter(block["comparisons"].values()))
        lines += [
            r"\midrule",
            rf"\multicolumn{{{6 if sensitivity else 7}}}{{@{{}}l}}{{\emph{{{name}}} "
            rf"({first['pairs']} pairs, {first['clusters']} cohort clusters, {first['snapshots']} snapshots)}} \\",
        ]
        for key, label in TESTS:
            c = block["comparisons"].get(key)
            if c is None:
                continue
            if sensitivity:
                cells = [
                    f"{c['mae_ratio']:.2f}",
                    p_value(c["mae_sign_flip_p"]),
                    f"{c['ratio_floor_1e4']:.2f} ({p_value(c['sign_flip_p_floor_1e4'])})",
                    f"{c['ratio_floor_1e3']:.2f} ({p_value(c['sign_flip_p_floor_1e3'])})",
                    f"{c['ratio_trimmed_5pct']:.2f} ({p_value(c['sign_flip_p_trimmed_5pct'])})",
                ]
            else:
                low, high = c["leave_one_snapshot_out_ratio"]
                cells = [
                    f"{c['error_ratio']:.2f} {ratio_ci(c)}",
                    p_value(c["sign_flip_p"]),
                    p_value(c["wild_cluster_p"]),
                    ratio_ci(c, "ci95_two_way_bootstrap"),
                    f"{low:.2f}--{high:.2f}",
                    p_value(c["leave_one_snapshot_out_max_p"]),
                ]
            lines.append(" & ".join([label, *cells]) + r" \\")
    return "\n".join([*lines, r"\bottomrule", r"\end{tabular}", ""])


def speed_criterion(blocks: dict[str, Block]) -> str:
    """Experiment 0035: the condition of the shrinkage proposition per snapshot, and the gain from speeds."""
    lines = [
        r"\begin{tabular}{@{}lccccccc@{}}",
        r"\toprule",
        r"Snapshot & speed model & $\hat\phi$ & cohorts with & \multicolumn{2}{c}{error ratio, full / equal speeds} & \multicolumn{2}{c}{reference shape} \\",
        r" & & & condition & realised path & forecast path & $a$ & $a c$ \\",
    ]
    for name, block in blocks.items():
        lines += [r"\midrule", rf"\multicolumn{{8}}{{@{{}}l}}{{\emph{{{name}}}}} \\"]
        for year, row in block["by_snapshot"].items():
            model = row["model"].replace("rw(", r"random walk, $\omega^2 = ").replace(")", "$")
            model = {"cp": "change point", "chain": "equal speeds"}.get(model, model)
            lines.append(
                f"{year} & {model} & {row['phi']:.1f} & {pct(row['share_holds'])} & "
                f"{row['error_ratio_full_over_equal_speeds']:.2f} & {row['error_ratio_forecast_path']:.2f} & "
                f"{row['gengamma_a']:.1f} & {row['early_exponent']:.1f}" + r" \\"
            )
        for path in ("realised", "forecast"):
            r = block[f"{path}_path"]
            holds, fails = r["where_the_condition_holds"], r["where_it_fails"]
            lines.append(
                rf"\multicolumn{{8}}{{@{{}}l}}{{{path.capitalize()} path: condition holds, {holds['pairs']} pairs, "
                rf"error ratio {holds['error_ratio']:.2f}; fails, {fails['pairs']} pairs, {fails['error_ratio']:.2f}}} \\"
            )
    return "\n".join([*lines, r"\bottomrule", r"\end{tabular}", ""])


def main() -> None:
    r = json.loads((EXP / "0031-round3-backtests" / "results.json").read_text())
    mortgage_pairs = pd.read_parquet(
        EXP / "0032-mortgage-hazard-baseline" / "artifacts" / "pairs_hazard.parquet"
    )
    patent_pairs = pd.read_parquet(
        EXP / "0031-round3-backtests" / "artifacts" / "pairs_patents_main.parquet"
    )
    mortgages = {
        "Mortgages, as run": r["freddie_as_run"],
        "Mortgages, corrected weights": r["freddie_corrected"],
        "Mortgages, borrower-followed": r["freddie_linked"],
        "Patent applications": r["patents_main"],
    }
    (TABLES / "baselines.tex").write_text(baselines(mortgage_pairs, patent_pairs))
    (TABLES / "inference.tex").write_text(tests(mortgages, sensitivity=False))
    (TABLES / "inference_sensitivity.tex").write_text(tests(mortgages, sensitivity=True))
    criterion = json.loads((EXP / "0035-speed-criterion-fisher" / "results.json").read_text())
    (TABLES / "speed_criterion.tex").write_text(
        speed_criterion(
            {
                "Mortgages, corrected weights": criterion["corrected"],
                "Patent applications": criterion["patents"],
            }
        )
    )


if __name__ == "__main__":
    main()
