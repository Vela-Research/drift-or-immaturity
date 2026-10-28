"""All comparisons of round 3 with the declared test procedure (maturity.inference), from the full-precision pair files.

DATASETS: the pair files of freddie.py and patents.py (this experiment).
Usage: uv run python report.py -> results.json
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd

from maturity import inference

HERE: Final = Path(__file__).resolve().parent
EXP: Final = HERE.parent
ART: Final = HERE / "artifacts"
NEW: Final = [
    "bf",
    "cape_cod",
    "gcc_075",
    "gcc_050",
    "unit_chain",
    "stratified",
    "cal_chain_forecast",
]
FROZEN: Final = [
    "naive",
    "drop_recent",
    "fixed_horizon",
    "chain",
    "ours_free",
    "cal_chain",
    "ours",
    "ours_uncond",
]
# (a, b): a against b
AGAINST_FULL: Final = [
    "naive",
    "drop_recent",
    "fixed_horizon",
    "chain",
    "bf",
    "cape_cod",
    "gcc_075",
    "unit_chain",
    "stratified",
]
STEPS: Final = (
    ("unit_chain", "chain"),
    ("stratified", "unit_chain"),
    ("cal_chain", "stratified"),
    ("ours", "cal_chain"),
    ("cal_chain_forecast", "stratified"),
    ("ours_uncond", "cal_chain_forecast"),
)


def errors(p: pd.DataFrame, methods: list[str]) -> pd.DataFrame:
    return pd.DataFrame({m: (p[m] - p.truth) / p.truth for m in methods})


def compare(p: pd.DataFrame, a: str, b: str) -> dict[str, object]:
    e = errors(p, [a, b])
    c = inference.compare(
        e[a].to_numpy(),
        e[b].to_numpy(),
        p.cohort.to_numpy().astype(np.int64),
        p.snapshot.to_numpy().astype(np.int64),
    )
    return {
        k: (
            round(v, 5)
            if isinstance(v, float)
            else [round(x, 4) for x in v]
            if isinstance(v, tuple)
            else v
        )
        for k, v in c.as_dict().items()
    }


def holm(p: dict[str, float]) -> dict[str, float]:
    order = sorted(p, key=lambda k: p[k])
    out, running = {}, 0.0
    for i, k in enumerate(order):
        running = max(running, min(1.0, (len(order) - i) * p[k]))
        out[k] = round(running, 5)
    return out


def summary(p: pd.DataFrame, methods: list[str], by: str | None = None) -> dict[str, object]:
    e = errors(p, methods).abs()
    per_snapshot = e.groupby(p.snapshot).mean()
    out: dict[str, object] = {
        "pairs": len(p),
        "cohort_clusters": int(p.cohort.nunique()),
        "snapshots": int(p.snapshot.nunique()),
        "median_over_snapshots": per_snapshot.median().round(4).to_dict(),
        "mean_absolute_error": e.mean().round(4).to_dict(),
        "error_by_snapshot": per_snapshot.round(4).to_dict("index"),
        "signed_error_by_snapshot": errors(p, methods)
        .groupby(p.snapshot)
        .mean()
        .round(4)
        .to_dict("index"),
    }
    if by is not None:
        out["median_over_snapshots_by_set"] = {
            str(k): e[p[by] == k].groupby(p.snapshot[p[by] == k]).mean().median().round(4).to_dict()
            for k in p[by].unique()
        }
    return out


def comparisons(p: pd.DataFrame, methods: list[str]) -> dict[str, object]:
    pairs = [("ours", m) for m in AGAINST_FULL] + [("ours_uncond", m) for m in AGAINST_FULL]
    pairs += [
        (m, "chain")
        for m in (
            "bf",
            "cape_cod",
            "gcc_075",
            "gcc_050",
            "stratified",
            "cal_chain",
            "cal_chain_forecast",
        )
    ]
    pairs += list(STEPS) + [("ours", "ours_uncond")]
    return {
        f"{a} vs {b}": compare(p, a, b)
        for a, b in dict.fromkeys(pairs)
        if a in methods and b in methods
    }


def speed_criterion(
    p: pd.DataFrame,
    meta: dict[str, dict[str, object]],
    path: tuple[str, str] = ("ours", "cal_chain"),
) -> dict[str, object]:
    """The shrinkage proposition per scored pair, and the gain from speeds (full method against the calendar-adjusted
    chain ladder, same calendar path) by whether the condition holds."""
    full, equal = path
    gain = inference.log_ratio(
        ((p[full] - p.truth) / p.truth).to_numpy(), ((p[equal] - p.truth) / p.truth).to_numpy()
    )
    table = p[
        [
            "snapshot",
            "cohort",
            "A",
            "D",
            "V",
            "own_var",
            "own_var_phi",
            "omega2",
            "tau2",
            "worth",
            "worth_untempered",
            "speed",
        ]
    ].copy()
    table["log_error_ratio_full_over_equal_speeds"] = gain
    out: dict[str, object] = {
        "pairs": len(p),
        "share_where_own_speed_is_worth_estimating": round(float(p.worth.mean()), 3),
        "share_without_the_dispersion": round(float(p.worth_untempered.mean()), 3),
        "by_condition": {
            str(flag): {
                "pairs": int((p.worth == flag).sum()),
                "error_ratio_full_over_equal_speeds": round(
                    float(np.exp(gain[(p.worth == flag).to_numpy()].mean())), 3
                ),
            }
            for flag in (True, False)
            if (p.worth == flag).any()
        },
        "by_snapshot": {
            str(int(s)): {
                "share_worth": round(float(g.worth.mean()), 2),
                "error_ratio_full_over_equal_speeds": round(
                    float(np.exp(gain[(p.snapshot == s).to_numpy()].mean())), 3
                ),
                "omega2": float(g.omega2.iloc[0]),
                "model": meta[str(int(s))]["model"],
                "gengamma_a": meta[str(int(s))]["gengamma_a"],
                "gengamma_c": meta[str(int(s))]["gengamma_c"],
                "early_exponent": meta[str(int(s))]["early_exponent_a_times_c"],
                "phi": meta[str(int(s))]["phi"],
            }
            for s, g in p.groupby("snapshot")
        },
        "per_pair": table.round(6).to_dict("records"),
    }
    return out


def wrong_direction(p: pd.DataFrame, methods: list[str]) -> dict[str, object]:
    big = p[(p.truth - p.ref).abs() >= 0.02]
    sign = np.sign(big.truth - big.ref)
    return {
        "pairs": len(big),
        "wrong": {
            m: f"{int((np.sign(big[m] - big.ref) != sign).sum())}/{len(big)}"
            for m in methods
            if m != "fixed_horizon"
        },
    }


def main() -> None:
    out: dict[str, object] = {}
    for version in ("as_run", "corrected", "noharp", "linked"):
        p = pd.read_parquet(ART / f"pairs_freddie_{version}.parquet")
        methods = FROZEN if version == "as_run" else FROZEN + NEW
        block = summary(p, methods, "set") | {"comparisons": comparisons(p, methods)}
        if version != "as_run":
            meta = json.loads((ART / f"meta_freddie_{version}.json").read_text())
            block["speed_criterion"] = speed_criterion(p, meta)
            block["models"] = {y: m["model"] for y, m in meta.items()}
        out[f"freddie_{version}"] = block
    for name in ("main", "no_own"):
        p = pd.read_parquet(ART / f"pairs_patents_{name}.parquet")
        meta = json.loads((ART / f"meta_patents_{name}.json").read_text())
        methods = FROZEN + NEW
        block = summary(p, methods) | {
            "comparisons": comparisons(p, methods),
            "wrong_direction": wrong_direction(p, methods),
        }
        block["speed_criterion"] = speed_criterion(p, meta)
        block["speed_criterion_forecast_path"] = speed_criterion(
            p, meta, ("ours_uncond", "cal_chain_forecast")
        )
        block["models"] = {y: m["model"] for y, m in meta.items()}
        block["beta"] = {y: m["beta"] for y, m in meta.items()}
        c = block["comparisons"]
        block["holm_forecast_path"] = holm(
            {
                k: c[k]["sign_flip_p"]
                for k in ("ours_uncond vs chain", "ours_uncond vs cal_chain_forecast")
            }
        )  # type: ignore[index]
        block["holm_realised_path"] = holm(
            {k: c[k]["sign_flip_p"] for k in ("ours vs chain", "ours vs cal_chain")}
        )  # type: ignore[index]
        out[f"patents_{name}"] = block
    (HERE / "results.json").write_text(json.dumps(out, indent=1, default=str))


if __name__ == "__main__":
    main()
