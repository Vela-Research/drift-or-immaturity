"""Corrected plug-in against the frozen code: comparisons, speed criterion, patents 2011-2012, intervals.

DATASETS: pair and calibration files of backtests.py; frozen pairs of 0031; hazard pairs of 0032 and 0040; criterion
of 0035. Same inference procedure as round 3 (`maturity.inference`).
Usage: uv run python report.py -> results.json
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd

from maturity import inference

HERE: Final = Path(__file__).resolve().parent
EXP: Final = HERE.parent
ART: Final = HERE / "artifacts"
FROZEN: Final = EXP / "0031-round3-backtests" / "artifacts"
HAZARD: Final = {
    "as_run": EXP / "0032-mortgage-hazard-baseline" / "artifacts" / "pairs_hazard.parquet",
    "corrected": EXP / "0032-mortgage-hazard-baseline" / "artifacts" / "pairs_hazard.parquet",
    "noharp": EXP
    / "0040-hazard-other-estimands"
    / "artifacts"
    / "noharp"
    / "artifacts"
    / "pairs_hazard.parquet",
    "linked": EXP
    / "0040-hazard-other-estimands"
    / "artifacts"
    / "linked"
    / "artifacts"
    / "pairs_hazard.parquet",
}
CRITERION: Final = {"patents": "patents", "corrected": "corrected", "linked": "linked"}
# (a, b): a against b; "ours" and "ours_uncond" are replaced by the code under test
TESTS: Final = [
    ("ours", "chain"),
    ("ours_uncond", "chain"),
    ("ours", "cal_chain"),
    ("ours_uncond", "cal_chain_forecast"),
    ("ours", "stratified"),
    ("ours_uncond", "stratified"),
    ("ours", "cape_cod"),
    ("ours_uncond", "cape_cod"),
    ("ours", "bf"),
    ("ours", "hazard_vintage_real"),
    ("ours_uncond", "hazard_vintage_fcst"),
]
GAMMAS: Final = (1.0, 1.25, 1.5, 2.0, 3.0)
ALPHA: Final = 0.10


def load(version: str) -> pd.DataFrame | None:
    """Frozen pairs of 0031 with the corrected estimates as `ours_corr`, `ours_uncond_corr` (and the hazard model)."""
    path = ART / f"pairs_{version}_corrected.parquet"
    if not path.exists():
        return None
    name = "pairs_patents_main" if version == "patents" else f"pairs_freddie_{version}"
    p = pd.read_parquet(FROZEN / f"{name}.parquet")
    new = pd.read_parquet(path)[["snapshot", "cohort", "ours", "ours_uncond"]]
    p = p.merge(new, on=["snapshot", "cohort"], suffixes=("", "_corr"), validate="one_to_one")
    hazard = HAZARD.get(version)
    if hazard is not None and hazard.exists():
        columns = ["snapshot", "cohort", "hazard_vintage_real", "hazard_vintage_fcst"]
        p = p.merge(
            pd.read_parquet(hazard)[columns], on=["snapshot", "cohort"], validate="one_to_one"
        )
    return p


def error(p: pd.DataFrame, m: str) -> np.ndarray:
    return ((p[m] - p.truth) / p.truth).to_numpy()


def compare(p: pd.DataFrame, a: str, b: str) -> dict[str, object]:
    c = inference.compare(
        error(p, a),
        error(p, b),
        p.cohort.to_numpy().astype(np.int64),
        p.snapshot.to_numpy().astype(np.int64),
    )
    low, high = (math.exp(x) for x in c.ci95_cluster_bootstrap)
    two = [round(math.exp(x), 4) for x in c.ci95_two_way_bootstrap]
    return {
        "error_ratio": round(c.error_ratio, 4),
        "ci95": [round(low, 4), round(high, 4)],
        "sign_flip_p": round(c.sign_flip_p, 5),
        "wild_cluster_p": round(c.wild_cluster_p, 5),
        "ci95_two_way": two,
        "leave_one_snapshot_out_max_p": round(c.leave_one_snapshot_out_max_p, 5),
        "snapshots_where_a_is_better": f"{c.snapshots_where_a_is_better}/{c.snapshots}",
        "mae_ratio": round(c.mae_ratio, 4),
        "mae_sign_flip_p": round(c.mae_sign_flip_p, 5),
    }


def median_error(p: pd.DataFrame, m: str) -> float:
    return round(
        float(pd.Series(np.abs(error(p, m))).groupby(p.snapshot.to_numpy()).mean().median()), 4
    )


def comparisons(p: pd.DataFrame) -> dict[str, object]:
    out: dict[str, object] = {
        "pairs": len(p),
        "pairs_changed": int((np.abs(p.ours_corr - p.ours) > 1e-12).sum()),
        "pairs_with_speed_above_1": int((p.speed > 1).sum()),
        "median_change_of_the_changed_estimates": round(
            float((p.ours_corr / p.ours)[np.abs(p.ours_corr - p.ours) > 1e-12].median()), 4
        ),
        "largest_change": round(float((p.ours_corr / p.ours).max()), 4),
        "median_error": {
            m: median_error(p, m)
            for m in (
                "chain",
                "cal_chain",
                "ours",
                "ours_corr",
                "cal_chain_forecast",
                "ours_uncond",
                "ours_uncond_corr",
            )
            if m in p
        },
    }
    for code, suffix in (("frozen", ""), ("corrected", "_corr")):
        out[code] = {f"{a} vs {b}": compare(p, a + suffix, b) for a, b in TESTS if b in p}
    out["corrected vs frozen"] = {
        "ours": compare(p, "ours_corr", "ours"),
        "ours_uncond": compare(p, "ours_uncond_corr", "ours_uncond"),
    }
    return out


def criterion(p: pd.DataFrame, name: str) -> dict[str, object]:
    """The gain from speeds by the condition of the shrinkage proposition (0035), frozen and corrected code."""
    c = pd.DataFrame(
        json.loads((EXP / "0035-speed-criterion-fisher" / "results.json").read_text())[name][
            "per_pair"
        ]
    )[["snapshot", "cohort", "holds"]]
    q = p.merge(c, on=["snapshot", "cohort"], validate="one_to_one")
    bins = pd.cut(q.maturity_m, [0, 0.15, 0.4, 0.7, 1.0], include_lowest=True)
    out: dict[str, object] = {}
    for code, suffix in (("frozen", ""), ("corrected", "_corr")):
        for path, full, equal in (
            ("realised", "ours", "cal_chain"),
            ("forecast", "ours_uncond", "cal_chain_forecast"),
        ):
            gain = pd.Series(
                inference.log_ratio(error(q, full + suffix), error(q, equal)), index=q.index
            )
            out[f"{code}_{path}"] = {
                "all": round(float(np.exp(gain.mean())), 4),
                "holds": round(float(np.exp(gain[q.holds].mean())), 4),
                "fails": round(float(np.exp(gain[~q.holds].mean())), 4),
                "pairs_holds": int(q.holds.sum()),
                "by_maturity": {
                    str(k): round(float(np.exp(g.mean())), 4)
                    for k, g in gain.groupby(bins, observed=True)
                },
                "by_snapshot": {
                    str(int(k)): round(float(np.exp(g.mean())), 4)
                    for k, g in gain.groupby(q.snapshot)
                },
            }
    return out


def intervals(
    scored: pd.DataFrame, cal: pd.DataFrame, method: str, horizon: int, margin: int
) -> pd.DataFrame:
    """The rule of 0011/0021 `conformal` for a grid of multipliers. margin 2 = the strict rule of round 2."""
    rows = []
    for r in scored.itertuples():
        c = cal[
            (cal.snapshot < r.snapshot)
            & ((cal.A - r.A).abs() <= 6)
            & (cal.A + 12 * (r.snapshot - cal.snapshot) >= horizon + margin)
        ]
        res = np.sort(np.abs(c[f"r_{method}"].to_numpy()))
        j, est = len(res), float(getattr(r, method))
        if j == 0 or est <= 0:
            continue
        q = float(res[min(j, math.ceil((j + 1) * (1 - ALPHA))) - 1])
        row: dict[str, object] = {"snapshot": r.snapshot, "cohort": r.cohort, "J": j}
        for g in GAMMAS:
            lo, hi = max(float(r.g_a), est * math.exp(-g * q)), min(1.0, est * math.exp(g * q))
            row[f"covered_{g}"] = lo <= float(r.truth) <= hi
            row[f"log_width_{g}"] = math.log(hi / lo) if lo > 0 else np.inf
        rows.append(row)
    return pd.DataFrame(rows)


def coverage(version: str, horizon: int) -> dict[str, object]:
    out: dict[str, object] = {}
    name = "pairs_patents_main" if version == "patents" else f"pairs_freddie_{version}"
    for code in ("frozen", "corrected"):
        cal_path = ART / f"calibration_{version}_{code}.parquet"
        pair_path = (
            FROZEN / f"{name}.parquet"
            if code == "frozen"
            else ART / f"pairs_{version}_corrected.parquet"
        )
        if not cal_path.exists() or not pair_path.exists():
            continue
        cal, scored = pd.read_parquet(cal_path), pd.read_parquet(pair_path)
        for method in ("ours", "ours_uncond", "chain"):
            for rule, margin in (("as_run", 0), ("strict", 2)):
                t = intervals(scored, cal, method, horizon, margin)
                for label, sub in (("J>=9", t[t.J >= 9]), ("J>=1", t)):
                    share = {g: float(sub[f"covered_{g}"].mean()) for g in GAMMAS}
                    enough = [g for g in GAMMAS if share[g] >= 1 - ALPHA]
                    out[f"{code} {method} {rule} {label}"] = {
                        "pairs": len(sub),
                        "covered": {
                            str(g): f"{int(sub[f'covered_{g}'].sum())}/{len(sub)}" for g in GAMMAS
                        },
                        "smallest_gamma_for_90pct": enough[0] if enough else None,
                        "median_log_width_gamma_1": round(float(sub["log_width_1.0"].median()), 4),
                    }
    return out


def main() -> None:
    out: dict[str, object] = {}
    for version in ("patents", "as_run", "corrected", "noharp", "linked"):
        p = load(version)
        if p is None:
            continue
        block = comparisons(p)
        if version in CRITERION:
            block["speed_criterion"] = criterion(p, CRITERION[version])
        block["intervals"] = coverage(version, 72 if version == "patents" else 84)
        if version == "patents":
            keep = [
                "snapshot",
                "cohort",
                "A",
                "speed",
                "truth",
                "cal_chain",
                "ours",
                "ours_corr",
                "ours_uncond",
                "ours_uncond_corr",
            ]
            block["snapshots_2011_2012"] = json.loads(
                p[p.snapshot.isin([2011, 2012])][keep].round(4).to_json(orient="records")
            )
            block["error_by_snapshot"] = {
                m: pd.Series(np.abs(error(p, m)))
                .groupby(p.snapshot.to_numpy())
                .mean()
                .round(4)
                .to_dict()
                for m in ("cal_chain", "ours", "ours_corr")
            }
        out[version] = block
    (HERE / "results.json").write_text(json.dumps(out, indent=1, default=str))


if __name__ == "__main__":
    main()
