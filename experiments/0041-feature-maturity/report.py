"""Results of the pre-specified test (FEATURE_MATURITY_SPEC v1): distortion, level error, calibration, AUC.

DATASET: the pair files of real.py (Freddie Mac 200k, corrected weights; 78 vintage-snapshot pairs, 18 vintages).
Primary: |log D| of pu_feat against pu_cl, logistic, exact cluster sign-flip test over vintages (`maturity.inference`).
Usage: uv run python report.py -> results.json, research/drift-or-maturity/figdata/feature_maturity_*.csv
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd

from maturity import inference

HERE: Final = Path(__file__).resolve().parent
PARTS: Final = HERE / "artifacts" / "real"
FIGDATA: Final = HERE.parents[1] / "research" / "drift-or-maturity" / "figdata"
SPLITS: Final = {
    "fico": ("fico_720_up", "fico_below_660"),
    "ltv": ("ltv_le_80", "ltv_gt_90"),
    "purpose": ("purchase", "refinance"),
}
STRATEGIES: Final = (
    "naive",
    "pu_cl",
    "pu_strat",
    "pu_feat",
    "pu_feat_weibull",
    "pu_feat_cal",
    "final",
)
SETS: Final = {"2006-2012": (2006, 2012), "2013-2018": (2013, 2018), "all": (2006, 2018)}
MIN_EVENTS: Final = 5  # a pair needs this many realised defaults in both groups of a split


def load() -> pd.DataFrame:
    p = pd.concat([pd.read_parquet(f) for f in sorted(PARTS.glob("*.parquet"))], ignore_index=True)
    p[["strategy", "head"]] = p.method.str.rsplit("_", n=1, expand=True)
    for split, (slow, fast) in SPLITS.items():
        enough = (p[f"events_{slow}"] >= MIN_EVENTS) & (p[f"events_{fast}"] >= MIN_EVENTS)
        d = p[f"level_ratio_{slow}"] / p[f"level_ratio_{fast}"]
        p[f"D_{split}"] = d.where(enough)
        p[f"abs_log_D_{split}"] = np.log(p[f"D_{split}"]).abs()
    p["level_error"] = (p.level_ratio - 1).abs()
    return p


def paired(
    p: pd.DataFrame, head: str, a: str, b: str, column: str, years: tuple[int, int]
) -> dict[str, object]:
    """a against b on one column, pairs of the snapshots in `years` where both have a value."""
    q = p[(p["head"] == head) & p.snapshot.between(*years)]
    wide = q.pivot(index=["snapshot", "cohort"], columns="strategy", values=column)[[a, b]].dropna()
    cohorts = wide.index.get_level_values("cohort").to_numpy().astype(np.int64)
    difference = (wide[a] - wide[b]).to_numpy()
    out: dict[str, object] = {
        "pairs": len(wide),
        "vintages": len(np.unique(cohorts)),
        f"mean_{a}": round(float(wide[a].mean()), 4),
        f"mean_{b}": round(float(wide[b].mean()), 4),
        "mean_difference": round(float(difference.mean()), 4),
        "sign_flip_p": round(inference.sign_flip_p(difference, cohorts), 5),
        "pairs_where_a_is_lower": int((difference < 0).sum()),
    }
    if column == "level_error":
        d = inference.log_ratio(wide[a].to_numpy(), wide[b].to_numpy())
        out["error_ratio"] = round(float(np.exp(d.mean())), 4)
        out["error_ratio_sign_flip_p"] = round(inference.sign_flip_p(d, cohorts), 5)
    return out


def table(p: pd.DataFrame, head: str) -> dict[str, dict[str, dict[str, float]]]:
    """Per evaluation set and strategy: median D, mean |log D| of each split, level, calibration, AUC."""
    out: dict[str, dict[str, dict[str, float]]] = {}
    for name, years in SETS.items():
        q = p[(p["head"] == head) & p.snapshot.between(*years)]
        rows = {}
        for strategy, g in q.groupby("strategy"):
            row = {"pairs": len(g)}
            for split in SPLITS:
                row[f"median_D_{split}"] = round(float(g[f"D_{split}"].median()), 4)
                row[f"mean_abs_log_D_{split}"] = round(float(g[f"abs_log_D_{split}"].mean()), 4)
                row[f"pairs_with_D_{split}"] = int(g[f"D_{split}"].notna().sum())
            row["median_level_ratio"] = round(float(g.level_ratio.median()), 4)
            row["mean_level_error"] = round(float(g.level_error.mean()), 4)
            row["mean_ece_within_vintage"] = round(float(g.ece.mean()), 5)
            row["mean_auc_within_vintage"] = round(float(g.auc.mean()), 4)
            rows[str(strategy)] = row
        out[name] = rows
    return out


def diagnostics() -> dict[str, object]:
    info = {f.stem: json.loads(f.read_text()) for f in sorted(PARTS.glob("*.json"))}
    return {
        "xi_fico_720_up_by_snapshot": {y: v["xi_pu_feat"]["fico_720_up"] for y, v in info.items()},
        "xi_fico_660_719_by_snapshot": {
            y: v["xi_pu_feat"]["fico_660_719"] for y, v in info.items()
        },
        "xi_weibull_fico_720_up_by_snapshot": {
            y: v["xi_pu_feat_weibull"]["fico_720_up"] for y, v in info.items()
        },
        "xi_all_features_last_snapshot": info[max(info)]["xi_pu_feat"],
        "maturity_by_fico_band_below_660_660_719_720_up": {
            y: v["maturity_by_fico_pu_feat"] for y, v in info.items()
        },
        "xi_fico_720_up_early_and_late_mature_vintages": {
            y: v.get("xi_fico_720_up_early_late") for y, v in info.items()
        },
        "mature_vintages": {y: v["mature_vintages"] for y, v in info.items()},
        "defaults_for_the_timing_model": {
            y: v["defaults_for_the_timing_model"] for y, v in info.items()
        },
        "shape": {y: v["shape_pu_feat"] for y, v in info.items()},
        "mean_completeness_of_young_loans": {
            y: v["mean_completeness_of_young_loans"] for y, v in info.items()
        },
    }


def main() -> None:
    p = load()
    out: dict[str, object] = {"spec_commit": "c5482de", "snapshots": int(p.snapshot.nunique())}
    primary = paired(p, "logit", "pu_feat", "pu_cl", "abs_log_D_fico", SETS["all"])
    level = paired(p, "logit", "pu_feat", "pu_cl", "level_error", SETS["all"])
    out["primary"] = {
        "abs_log_D_fico": primary,
        "level_error": level,
        "H1_supported": bool(
            primary["mean_difference"] < 0
            and primary["sign_flip_p"] < 0.05
            and level["error_ratio"] <= 1.10
        ),
    }
    for head in ("logit", "gbm"):
        block: dict[str, object] = {"table": table(p, head)}
        for a in ("pu_feat", "pu_strat", "pu_feat_cal", "pu_feat_weibull"):
            block[f"{a} vs pu_cl"] = {
                f"{column} {name}": paired(p, head, a, "pu_cl", column, years)
                for name, years in SETS.items()
                for column in (
                    "abs_log_D_fico",
                    "abs_log_D_ltv",
                    "abs_log_D_purpose",
                    "level_error",
                    "ece",
                    "auc",
                )
            }
        out[head] = block
    out["diagnostics"] = diagnostics()
    (HERE / "results.json").write_text(json.dumps(out, indent=1))
    FIGDATA.mkdir(parents=True, exist_ok=True)
    keep = [
        "snapshot",
        "cohort",
        "strategy",
        "head",
        "realised",
        "level_ratio",
        "level_error",
        "ece",
        "auc",
    ]
    keep += [c for c in p.columns if c.startswith(("D_", "level_ratio_"))]
    p[keep].round(5).to_csv(FIGDATA / "feature_maturity_pairs.csv", index=False)
    rows = [
        {"evaluation_set": s, "head": head, "strategy": k} | v
        for head in ("logit", "gbm")
        for s, block in out[head]["table"].items()  # type: ignore[index]
        for k, v in block.items()
    ]
    pd.DataFrame(rows).to_csv(FIGDATA / "feature_maturity_summary.csv", index=False)


if __name__ == "__main__":
    main()
