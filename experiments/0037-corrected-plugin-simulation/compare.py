"""Frozen against corrected plug-in on the same simulated worlds (round 4, items P0.1a, P0.3, P0.5). POST HOC.

DATASET: none (simulated). Reads the parts of experiments/0034 (frozen code) and artifacts/parts (corrected code,
rerun.py; same seeds), and artifacts/windows_feasible.csv. Every table is made by the functions of 0034 `report.py`,
once per version. Error ratio = exp(mean difference of log absolute relative errors), floor 1e-6, with a 95% interval
from the replication means.
Intervals at a multiplier Gamma: [max(events so far, estimate x exp(-Gamma q)), min(1, estimate x exp(Gamma q))], q =
the conformal quantile of the absolute log calibration errors (Gamma = 1 is the interval of the run; q is read back
from its upper limit).
Usage: uv run python compare.py -> results.json, research/drift-or-maturity/figdata/sim_*_corrected.csv,
sim_interval_gamma.csv, sim_window_rule_feasible.csv
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd

HERE: Final = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "0034-end-to-end-simulation"))
import report  # of 0034
import rerun
import sim

FIGDATA: Final = report.FIGDATA
VERSIONS: Final = {"frozen": rerun.FROZEN_PARTS, "corrected": HERE / "artifacts" / "parts"}
GAMMAS: Final = (1.0, 1.25, 1.5, 2.0, 3.0)
INTERVALS: Final = ("ours", "ours_uncond", "chain")
Frames = dict[str, pd.DataFrame]


def load(folder: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Scored cohorts and replications of one version, with the columns that 0034 `report.main` adds."""
    sim.PARTS = folder
    p, meta = sim.results()
    p = p.merge(meta[["setting", "rep", "phi"]], on=["setting", "rep"], validate="many_to_one")
    with np.errstate(divide="ignore"):
        p["worth_oracle"] = np.log(p.true_speed) ** 2 > 1 / (p.D * p.V_true)
        p["worth_oracle_phi"] = np.log(p.true_speed) ** 2 > p.phi / (p.D * p.V_true)
    p["D_bin"] = pd.cut(p.D, report.D_BINS[0], labels=report.D_BINS[1], right=False)
    return p, meta


def both(function: object, frames: Frames) -> dict[str, object]:
    return {version: function(frame) for version, frame in frames.items()}  # type: ignore[operator]


def gamma_coverage(young: pd.DataFrame, method: str) -> dict[str, object]:
    """Coverage of the interval of `method` at each Gamma, and the smallest Gamma of the grid that gives 90%."""
    q = young[young[f"{method}_J"].notna() & (young[f"{method}_J"] >= 1)]
    if q.empty:
        return {"pairs": 0}
    estimate, truth = q[method].to_numpy(), q.truth.to_numpy()
    floor, low_1, high_1 = (q[c].to_numpy() for c in ("g_a", f"{method}_low", f"{method}_high"))
    stored = (low_1 <= truth) & (truth <= high_1)
    # q from the upper limit; from the lower limit when the upper one is capped at 1; when both limits are at their
    # bounds the interval is [events so far, 1] at every Gamma (q = infinity gives exactly that)
    with np.errstate(divide="ignore", invalid="ignore"):
        quantile = np.where(
            high_1 < 1,
            np.log(high_1 / estimate),
            np.where(low_1 > floor * (1 + 1e-12), np.log(estimate / low_1), np.inf),
        )
    out: dict[str, object] = {
        "pairs": len(q),
        "upper_limit_capped_at_1": int((high_1 >= 1).sum()),
        "both_limits_at_their_bounds": int(np.isinf(quantile).sum()),
    }
    for gamma in GAMMAS:
        with np.errstate(over="ignore"):
            low = np.maximum(floor, estimate * np.exp(-gamma * quantile))
            high = np.minimum(1.0, estimate * np.exp(gamma * quantile))
        covered = (low <= truth) & (truth <= high)
        if gamma == 1.0:
            assert (covered == stored).all(), "the quantile is not read back correctly"
        out[f"coverage_{gamma:g}"] = round(float(covered.mean()), 3)
        out[f"median_width_{gamma:g}"] = round(float(np.median(high / low)), 3)
    enough = [g for g in GAMMAS if out[f"coverage_{g:g}"] >= 0.9]  # type: ignore[operator]
    out["smallest_gamma_with_90"] = enough[0] if enough else "none"
    return out


def gamma_tables(frames: Frames) -> tuple[dict[str, object], pd.DataFrame]:
    """By setting, and pooled over the two settings with a speed change by timing, calendar effect and size: all
    young cohorts, and the cohorts whose true speed is above or below 1."""
    out: dict[str, object] = {}
    rows = []
    for version, p in frames.items():
        young = p[p.young & p.ours_J.notna()]
        parts = young.setting.str.split("/", expand=True)
        young = young.assign(timing=parts[0], speeds=parts[1], calendar=parts[2], size=parts[3])
        changed = young[young.speeds != "constant"]
        groups: dict[str, pd.DataFrame] = {name: g for name, g in young.groupby("setting")}
        for (timing, calendar, size), g in changed.groupby(["timing", "calendar", "size"]):
            label = f"{timing}/step+drift/{calendar}/{size}"
            groups[label] = g
            groups[f"{label} | true speed above 1"] = g[g.true_speed > 1 + 1e-9]
            groups[f"{label} | true speed below 1"] = g[g.true_speed < 1 - 1e-9]
            groups[f"{label} | true speed 1"] = g[(g.true_speed - 1).abs() <= 1e-9]
        block = {name: {m: gamma_coverage(g, m) for m in INTERVALS} for name, g in groups.items()}
        out[version] = block
        rows += [
            {"version": version, "group": name, "method": m} | values
            for name, by_method in block.items()
            for m, values in by_method.items()
            if values["pairs"]
        ]
    return out, pd.DataFrame(rows)


def paired(frames: Frames) -> dict[str, object]:
    """Corrected against frozen estimate of the same cohorts (the full method, both paths, and the own-speed
    plug-in)."""
    frozen, fixed = (frames[v][frames[v].young] for v in VERSIONS)
    out: dict[str, object] = {}
    for name in [s.name for s in sim.SETTINGS]:
        f, c = frozen[frozen.setting == name], fixed[fixed.setting == name]
        block: dict[str, object] = {"pairs": len(c)}
        for method, speed in (("ours", "speed"), ("ours_uncond", "speed"), ("own", "own_speed")):
            q = c.assign(frozen_estimate=f[method].to_numpy())
            above = q[q[speed] > 1]
            block[method] = {
                "share_with_speed_above_1": round(float((q[speed] > 1).mean()), 3),
                "median_change_of_the_estimate_where_above_1": round(
                    float((above[method] / above.frozen_estimate - 1).median()), 4
                )
                if len(above)
                else None,
                "corrected vs frozen, all pairs": report.ratio(q, method, "frozen_estimate"),
                "corrected vs frozen, speed above 1": report.ratio(
                    above, method, "frozen_estimate"
                ),
            }
        out[name] = block
    return out


def fig3c(blocks: dict[str, dict[str, dict[str, object]]]) -> pd.DataFrame:
    """Panel (c) of `fig:speed`: gain from speeds (full method / equal speeds, realised path), calendar effect on."""
    rows = []
    for setting in sim.SETTINGS:
        if not setting.calendar:
            continue
        row: dict[str, object] = {
            "timing": setting.timing,
            "size": setting.size,
            "speeds": setting.speed,
        }
        for version in VERSIONS:
            r = blocks[version][setting.name]["error_ratios"]["ours vs cal_chain"]  # type: ignore[index]
            row |= {
                f"error_ratio_{version}": r["ratio"],
                f"ci95_low_{version}": r["ci95"][0],
                f"ci95_high_{version}": r["ci95"][1],
            }
        rows.append(row)
    return pd.DataFrame(rows)


def windows() -> tuple[dict[str, object], pd.DataFrame]:
    t = pd.read_csv(HERE / "artifacts" / "windows_feasible.csv")
    table = (
        t.groupby(["speeds", "method", "n"])
        .agg(
            median_rel_error=("rel_error", "median"),
            mean_window=("window", "mean"),
            maturity_youngest=("maturity_youngest", "mean"),
            settings=("rel_error", "size"),
            replications=("replications", "first"),
        )
        .reset_index()
        .assign(grid="reduced")
    )
    medians = (100 * t.groupby(["speeds", "method", "n"]).rel_error.median().unstack()).round(1)
    out = {
        "median_error_pct": {f"{a} | {b}": row.to_dict() for (a, b), row in medians.iterrows()},
        "maturity_of_the_youngest_cohort": t.groupby(["speeds", "method"])
        .maturity_youngest.mean()
        .unstack()
        .round(4)
        .T.to_dict(),
        "largest_difference_speed_corrected_vs_speed": float(
            np.abs(
                t[t.method == "weighted_speed_corrected"].rel_error.to_numpy()
                - t[t.method == "weighted_speed"].rel_error.to_numpy()
            ).max()
        ),
        "replications": int(t.replications.iloc[0]),
        "settings_per_cell": int(t.groupby(["speeds", "method", "n"]).size().iloc[0]),
    }
    return out, table


def main() -> None:
    loaded = {version: load(folder) for version, folder in VERSIONS.items()}
    frames = {version: p for version, (p, _) in loaded.items()}
    metas = {version: meta for version, (_, meta) in loaded.items()}
    contract = rerun.compare_with_frozen(
        frames["corrected"].drop(columns="D_bin"),
        frames["frozen"].drop(columns="D_bin"),
        "full run",
    )
    for column in ("model", "phi", "g1", "beta0", "beta1", "calendar_on", "intervals"):
        a, b = metas["corrected"][column].to_numpy(), metas["frozen"][column].to_numpy()
        assert (
            (a == b).all() if a.dtype == object else np.allclose(a.astype(float), b.astype(float))
        )
    order = [s.name for s in sim.SETTINGS]
    blocks = {
        version: {
            name: report.setting_block(
                p[p.setting == name], metas[version][metas[version].setting == name]
            )
            for name in order
        }
        for version, p in frames.items()
    }
    young = {version: p[p.young] for version, p in frames.items()}
    gammas, gamma_table = gamma_tables(frames)
    window_block, window_table = windows()
    meta = metas["corrected"]
    out = {
        "contract_against_the_frozen_run": contract
        | {
            "replications": len(meta),
            "scored_cohorts": len(frames["corrected"]),
            "replications_with_a_guard_off_snapshot": int((meta.guard_off_snapshots > 0).sum()),
            "guard_off_snapshots": int(meta.guard_off_snapshots.sum()),
        },
        "settings": blocks,
        "corrected_against_frozen_estimate": paired(frames),
        "shrinkage_condition": both(report.condition_summary, young),
        "theorem_6_inflation_equal_speeds": both(report.inflation, young),
        "interval_gamma": gammas,
        "windows_feasible": window_block,
        "runtime": {
            "wall_seconds_end_to_end": sum(
                int(f.read_text()) for f in VERSIONS["corrected"].glob("seconds_*.txt")
            ),
            "workers": rerun.WORKERS,
            "seconds_per_replication_with_intervals_median": round(
                float(meta[meta.intervals].seconds.median()), 2
            ),
            "seconds_per_replication_without_intervals_median": round(
                float(meta[~meta.intervals].seconds.median()), 2
            ),
        },
    }
    (HERE / "results.json").write_text(json.dumps(out, indent=1, default=str))
    report.end_to_end_csv(blocks["corrected"]).to_csv(
        FIGDATA / "sim_end_to_end_corrected.csv", index=False
    )
    report.speed_gain(young["corrected"]).to_csv(
        FIGDATA / "sim_speed_gain_corrected.csv", index=False
    )
    fig3c(blocks).to_csv(FIGDATA / "sim_fig3c_corrected.csv", index=False)
    gamma_table.to_csv(FIGDATA / "sim_interval_gamma.csv", index=False)
    window_table.round(5).to_csv(FIGDATA / "sim_window_rule_feasible.csv", index=False)


if __name__ == "__main__":
    main()
