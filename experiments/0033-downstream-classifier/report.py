"""Summary of 0033: classifier metrics over snapshots, paired tests, and the figure data.

DATASET: the outputs of run.py (200k loans per vintage, Freddie Mac sample of experiments/0010; aggregates only).
  median over snapshots of each metric, for the snapshots 2006-2012 (the untruncated backtest set of the paper),
  2013-2018 (the 84-month truth includes 2020-21; the paper cuts these backtests at February 2020), and all;
  level error of a (vintage, snapshot) pair = |mean prediction - realised rate| / realised rate on the test loans;
  paired tests of the level error with the declared procedure (maturity.inference; clusters = vintages);
  AUC differences with the paired cluster bootstrap of run.py (clusters = vintage x state), per snapshot.
Usage: uv run python report.py -> results.json (one block per version and design that run.py has written),
       research/drift-or-maturity/figdata/classifier_calibration.csv
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd

from maturity import inference

HERE: Final = Path(__file__).resolve().parent
ART: Final = HERE / "artifacts"
FIG: Final = HERE.parents[1] / "research" / "drift-or-maturity" / "figdata"
SETS: Final = {
    "2006-2012": range(2006, 2013),
    "2013-2018": range(2013, 2019),
    "all": range(2006, 2019),
}
METRICS: Final = ["level_ratio", "ece", "log_loss", "auc", "auc_within_vintage", "slow_over_fast"]
AGAINST: Final = ["final", "naive", "mature", "drop_recent", "dfm", "esdfm", "pu_cl", "pu_cal"]
FROZEN: Final = ["naive", "drop_recent", "fixed_horizon", "chain", "ours_uncond"]


def table(results: dict[str, dict[str, object]]) -> pd.DataFrame:
    """One row per (snapshot, method) with the scalar metrics."""
    rows = []
    for year, block in results.items():
        for method, m in block["methods"].items():  # type: ignore[union-attr]
            strategy, head = method.rsplit("_", 1)
            row = {"snapshot": int(year), "strategy": strategy, "model": head, "method": method}
            row |= {k: m[k] for k in METRICS}
            row |= {"auc_low": m["auc_ci95"][0], "auc_high": m["auc_ci95"][1]}
            rows.append(row | {f"level_{k}": v for k, v in m["level_ratio_by_fico"].items()})
    return pd.DataFrame(rows)


def compare(p: pd.DataFrame, a: str, b: str) -> dict[str, object]:
    c = inference.compare(
        ((p[a] - p.realised) / p.realised).to_numpy(),
        ((p[b] - p.realised) / p.realised).to_numpy(),
        p.cohort.to_numpy().astype(np.int64),
        p.snapshot.to_numpy().astype(np.int64),
    ).as_dict()
    keep = ["pairs", "clusters", "error_ratio", "ci95_cluster_bootstrap", "sign_flip_p"]
    keep += ["wild_cluster_p", "leave_one_snapshot_out_ratio", "mae_ratio", "mae_sign_flip_p"]
    keep += ["snapshots_where_a_is_better", "ratio_floor_1e3", "ratio_trimmed_5pct"]
    return {
        k: round(c[k], 4)
        if isinstance(c[k], float)
        else [round(v, 4) for v in c[k]]  # type: ignore[union-attr]
        if isinstance(c[k], tuple)
        else c[k]
        for k in keep
    }


def auc_differences(
    reps: dict[str, np.ndarray], t: pd.DataFrame, a: str, b: str
) -> dict[str, object]:
    """Per snapshot: AUC of a minus AUC of b, with the paired bootstrap interval of the difference."""
    auc = t.pivot(index="snapshot", columns="method", values="auc")
    low, high = np.array(
        [np.percentile(reps[f"{y}_{a}"] - reps[f"{y}_{b}"], [2.5, 97.5]) for y in auc.index]
    ).T
    return {
        "mean_difference": round(float((auc[a] - auc[b]).mean()), 4),
        "snapshots": len(auc),
        "snapshots_interval_above_0": int((low > 0).sum()),
        "snapshots_interval_below_0": int((high < 0).sum()),
    }


def block(version: str, design: str) -> tuple[dict[str, object], pd.DataFrame]:
    results = json.loads((HERE / f"results_{version}_{design}.json").read_text())
    pairs = pd.read_parquet(ART / f"pairs_{version}_{design}.parquet")
    reps = dict(np.load(ART / f"auc_bootstrap_{version}_{design}.npz"))
    t = table(results)
    methods = [m for m in pairs.columns if m not in ("cohort", "snapshot", "realised")]
    error = pairs[methods].sub(pairs.realised, axis=0).div(pairs.realised, axis=0).abs()
    by_snapshot = error.groupby(pairs.snapshot).mean()
    summary = {}
    for name, years in SETS.items():
        med = t[t.snapshot.isin(years)].groupby("method")[METRICS].median()
        med["level_error"] = by_snapshot[by_snapshot.index.isin(years)].median()
        gap = np.log(t[t.snapshot.isin(years)].set_index("method").level_ratio).abs()
        med["level_ratio_distance"] = np.exp(gap.groupby(level=0).median())
        summary[name] = med.round(4).to_dict("index")
    tests, aucs = {}, {}
    for head in ("logit", "gbm"):
        for b in AGAINST:
            if f"{b}_{head}" in pairs:
                key = f"pu_full_{head} vs {b}_{head}"
                tests[key] = compare(pairs, f"pu_full_{head}", f"{b}_{head}")
                aucs[key] = auc_differences(reps, t, f"pu_full_{head}", f"{b}_{head}")
        tests[f"pu_cal_{head} vs pu_cl_{head}"] = compare(pairs, f"pu_cal_{head}", f"pu_cl_{head}")
        tests[f"pu_cl_{head} vs naive_{head}"] = compare(pairs, f"pu_cl_{head}", f"naive_{head}")
        tests[f"dfm_logit vs naive_{head}"] = compare(pairs, "dfm_logit", f"naive_{head}")
        tests[f"esdfm_{head} vs naive_{head}"] = compare(pairs, f"esdfm_{head}", f"naive_{head}")
    cohorts = pd.read_parquet(ART / f"cohorts_{version}.parquet")
    c = pairs[["cohort", "snapshot", "realised"]].merge(cohorts, on=["cohort", "snapshot"])
    check = (
        c.realised / c.truth
    )  # test loans (weighted) against the backtest's truth (all 200k loans)
    frozen = c[FROZEN].sub(c.truth, axis=0).div(c.truth, axis=0).abs().groupby(c.snapshot).mean()
    info = pd.DataFrame(
        {y: {k: v for k, v in b.items() if k != "methods"} for y, b in results.items()}
    ).T
    out: dict[str, object] = {
        "pairs": len(pairs),
        "snapshots": sorted(int(y) for y in results),
        "realised_rate_of_test_loans_over_backtest_truth": [
            round(float(v), 3) for v in (check.min(), check.median(), check.max())
        ],
        "seconds_per_snapshot_sum": float(info.seconds.sum()),
        "largest_em_change": float(info.filter(like="em_last_change").max().max()),
        "per_snapshot": info.drop(columns=info.filter(like="em_last_change").columns).to_dict(
            "index"
        ),
        "median_over_snapshots": summary,
        "level_error_by_snapshot": by_snapshot.round(4).to_dict("index"),
        "level_tests": tests,
        "auc_differences": aucs,
        "cohort_level_frozen_methods_same_pairs": {
            name: frozen[frozen.index.isin(years)].median().round(4).to_dict()
            for name, years in SETS.items()
        },
    }
    return out, t.assign(version=version, design=design)


def main() -> None:
    """One block per results_<version>_<design>.json of run.py."""
    out: dict[str, dict[str, object]] = {}
    tables = []
    for file in sorted(HERE.glob("results_*_*.json")):
        version, design = file.stem.split("_")[1:]
        if version == "drift":
            continue
        out.setdefault(version, {})[design], t = block(version, design)
        tables.append(t)
    (HERE / "results.json").write_text(json.dumps(out, indent=1, default=str))
    FIG.mkdir(exist_ok=True)
    pd.concat(tables, ignore_index=True).drop(columns="method").to_csv(
        FIG / "classifier_calibration.csv", index=False
    )


if __name__ == "__main__":
    main()
