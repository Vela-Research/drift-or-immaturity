"""Answers of the round 4 classifier follow-up from the stored runs (POST HOC; review round 4, item P0.4).

DATASET: result files of experiments/0033 (`corrected` version) and the checkpoints of classify.py (aggregates only).
  auc     per snapshot: AUC over the 6 test vintages pooled and mean AUC within vintage, each PU strategy minus
          events so far; mean, median and range over snapshots
  level   pu_full against the other strategies: exp(mean log ratio) over the 78 vintage-snapshot pairs (the
          statistic of maturity.inference), ratio of the medians over snapshots, ratio of mean absolute level errors
  linked  the summary of report.py of 0033 for the borrower-followed run at all 13 snapshots
Usage: uv run python report.py -> results.json, research/drift-or-maturity/figdata/classifier_{auc_by_snapshot,linked}.csv
"""

from __future__ import annotations

import json
import pickle
import shutil
import sys
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd

HERE: Final = Path(__file__).resolve().parent
ART: Final = HERE / "artifacts"
OLD: Final = HERE.parent / "0033-downstream-classifier"
FIG: Final = HERE.parents[1] / "research" / "drift-or-maturity" / "figdata"
sys.path.insert(0, str(OLD))
import report as old  # summary code of 0033

SETS: Final = old.SETS
PU: Final = ("pu_cl", "pu_cal", "pu_full")
HEADS: Final = ("logit", "gbm")
OTHERS: Final = ("naive", "mature", "drop_recent", "dfm", "esdfm", "pu_cl", "pu_cal")


def stats(x: pd.Series) -> dict[str, float]:
    return {k: round(float(getattr(x, k)()), 4) for k in ("mean", "median", "min", "max")}


def auc_gains(results: dict[str, dict], reps: dict[str, np.ndarray]) -> tuple[dict, pd.DataFrame]:
    """AUC of each PU strategy minus events so far, per snapshot: pooled over the 6 vintages, and mean within
    vintage. `reps` has the paired bootstrap replicates of the pooled AUC."""
    rows = []
    for year, block in results.items():
        m = block["methods"]
        for head in HEADS:
            for strategy in PU:
                a, b = f"{strategy}_{head}", f"naive_{head}"
                low, high = np.percentile(reps[f"{year}_{a}"] - reps[f"{year}_{b}"], [2.5, 97.5])
                rows.append(
                    {
                        "snapshot": int(year),
                        "model": head,
                        "strategy": strategy,
                        "auc_pooled": m[a]["auc"],
                        "auc_pooled_events_so_far": m[b]["auc"],
                        "gain_pooled": round(m[a]["auc"] - m[b]["auc"], 4),
                        "gain_pooled_ci95_low": round(float(low), 4),
                        "gain_pooled_ci95_high": round(float(high), 4),
                        "auc_within": m[a]["auc_within_vintage"],
                        "auc_within_events_so_far": m[b]["auc_within_vintage"],
                        "gain_within": round(
                            m[a]["auc_within_vintage"] - m[b]["auc_within_vintage"], 4
                        ),
                    }
                )
    t = pd.DataFrame(rows)
    out: dict[str, object] = {}
    for (head, strategy), g in t.groupby(["model", "strategy"]):
        g = g.set_index("snapshot")
        out[f"{strategy}_{head} minus naive_{head}"] = {
            "gain_pooled_by_snapshot": g.gain_pooled.to_dict(),
            "gain_within_by_snapshot": g.gain_within.to_dict(),
            "snapshots_with_interval_above_0": int((g.gain_pooled_ci95_low > 0).sum()),
            **{
                f"{kind}, {name}": stats(g.loc[g.index.isin(years), f"gain_{kind}"])
                for kind in ("pooled", "within")
                for name, years in SETS.items()
            },
            "difference_of_the_medians_of_the_table": {
                name: round(
                    float(
                        g.auc_pooled[g.index.isin(years)].median()
                        - g.auc_pooled_events_so_far[g.index.isin(years)].median()
                    ),
                    4,
                )
                for name, years in SETS.items()
            },
        }
    return out, t


def level_ratios(pairs: pd.DataFrame) -> dict[str, object]:
    """pu_full against each other strategy, three statistics of the level error |prediction - realised| / realised
    of a (vintage, snapshot) pair."""
    methods = [m for m in pairs.columns if m not in ("cohort", "snapshot", "realised")]
    error = pairs[methods].sub(pairs.realised, axis=0).div(pairs.realised, axis=0).abs()
    by_snapshot = error.groupby(pairs.snapshot).mean()
    out = {}
    for head in HEADS:
        a = f"pu_full_{head}"
        for other in OTHERS:
            b = f"{other}_{head}" if f"{other}_{head}" in pairs else f"{other}_logit"
            if b not in pairs:
                continue
            test = old.compare(pairs, a, b)
            out[f"{a} vs {b}"] = {
                "pairs": test["pairs"],
                "exp_mean_log_ratio": test["error_ratio"],
                "ci95_of_the_mean_log_ratio": test["ci95_cluster_bootstrap"],
                "sign_flip_p": test["sign_flip_p"],
                "ratio_of_mean_absolute_errors": test["mae_ratio"],
                "ratio_of_medians_over_pairs": round(
                    float(error[a].median() / error[b].median()), 4
                ),
                "ratio_of_medians_over_snapshots": {
                    name: round(
                        float(
                            by_snapshot.loc[by_snapshot.index.isin(years), a].median()
                            / by_snapshot.loc[by_snapshot.index.isin(years), b].median()
                        ),
                        4,
                    )
                    for name, years in SETS.items()
                },
            }
    return out


def assemble(mode: str, name: str) -> dict[str, dict]:
    """Checkpoints of classify.py -> the three files that report.py of 0033 reads, in this folder."""
    parts = [pickle.loads(p.read_bytes()) for p in sorted((ART / "parts").glob(f"{mode}_*.pkl"))]
    pd.concat([p[2] for p in parts], ignore_index=True).to_parquet(ART / f"pairs_{name}.parquet")
    np.savez_compressed(
        ART / f"auc_bootstrap_{name}.npz",
        **{f"{year}_{k}": r for year, _, _, reps in parts for k, r in reps.items()},
    )
    results = {str(year): out for year, out, _, _ in parts}
    (HERE / f"results_{name}.json").write_text(json.dumps(results, indent=1))
    return results


def largest_gap(new: dict[str, dict], stored: dict[str, dict], methods: list[str]) -> float:
    """Largest difference of a level ratio or an AUC between two runs of the same method and snapshot."""
    return max(
        abs(new[y]["methods"][m][k] - stored[y]["methods"][m][k])
        for y in new
        if y in stored
        for m in methods
        for k in ("level_ratio", "auc")
    )


def main() -> None:
    FIG.mkdir(exist_ok=True)
    stored = json.loads((OLD / "results_corrected_vintage.json").read_text())
    reps = dict(np.load(OLD / "artifacts" / "auc_bootstrap_corrected_vintage.npz"))
    pairs = pd.read_parquet(OLD / "artifacts" / "pairs_corrected_vintage.parquet")
    auc, auc_table = auc_gains(stored, reps)
    auc_table.to_csv(FIG / "classifier_auc_by_snapshot.csv", index=False)
    out: dict[str, object] = {
        "auc_gain_over_events_so_far": auc,
        "level_error_ratios": level_ratios(pairs),
    }

    if list((ART / "parts").glob("linked_*.pkl")):
        linked = assemble("linked", "linked_vintage")
        shutil.copy(OLD / "artifacts" / "cohorts_linked.parquet", ART / "cohorts_linked.parquet")
        first = json.loads((OLD / "results_linked_vintage.json").read_text())
        old.HERE, old.ART = HERE, ART  # report.py of 0033 reads the three files from its own folder
        block, table = old.block("linked", "vintage")
        block["largest_gap_to_the_3_snapshots_of_0033"] = largest_gap(
            linked, first, list(next(iter(first.values()))["methods"])
        )
        block["auc_gain_over_events_so_far"] = auc_gains(
            linked, dict(np.load(ART / "auc_bootstrap_linked_vintage.npz"))
        )[0]
        block["level_error_ratios"] = level_ratios(
            pd.read_parquet(ART / "pairs_linked_vintage.parquet")
        )
        out["linked"] = block
        both = [table, old.table(stored).assign(version="corrected", design="vintage")]
        pd.concat(both, ignore_index=True).drop(columns="method").to_csv(
            FIG / "classifier_linked.csv", index=False
        )

    if (HERE / "results_alarms.json").exists():
        out["alarms"] = json.loads((HERE / "results_alarms.json").read_text())
    (HERE / "results.json").write_text(json.dumps(out, indent=1, default=str))


if __name__ == "__main__":
    main()
