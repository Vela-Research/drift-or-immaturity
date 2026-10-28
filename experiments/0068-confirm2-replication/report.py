"""CONFIRM2 §6-§8: confirmatory tests, Table 2-format tables, wrong direction, coverage, secondary results.

DATASET: the outputs of backtest.py, hazard_m1.py, classify.py and secondary.py (artifacts/<name>/). No new fits.
  H1-H3      §1/§6: error ratio = exp(mean log ratio of absolute relative errors, floor 1e-6); 95% cohort-cluster
             bootstrap interval (B = 2,000, `maturity.inference.cluster_bootstrap`); one-sided cluster sign-flip test of
             the mean log ratio shifted by log(margin): all 2^C sign vectors when C <= 20, else 10^6 random ones
             (seed 20261025, the first one the identity); Holm over H1-H3 within a data set; verdicts of §1.
             H1 = pu_cl over naive (classifiers); H2 = full method over calendar-adjusted CL (forecast path),
             margin 0.85; H3 = hazard model over full method (forecast path), margin 1.15.
  tables     Table 2 format (`maturity.inference.compare`, as in the paper): median over snapshots of the mean absolute
             relative error, error ratio to CL and to the previous step, two-sided sign-flip p.
  wrong      pairs whose truth differs from the reference (the latest mature cohort at the pair's horizon) by at
             least 10%: the number of pairs where a method is on the other side of the reference.
  coverage   the interval rule of 0036 `report.intervals` (Gamma grid), strict (H + 2) and as run (H + 0).
  secondary  windows (0014 `paired`), drift alarms "rate fell" (0039 `near`, 0067 `one_to_one`), B with the broad
             event, classifier (c).
Usage: uv run python report.py -> results.json
"""

from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path
from types import ModuleType
from typing import Final, Literal

import numpy as np
import pandas as pd

from maturity import inference

HERE: Final = Path(__file__).resolve().parent
EXP: Final = HERE.parent
ART: Final = HERE / "artifacts"
SEED: Final = 20261025
N_RANDOM: Final = 10**6
ALPHA: Final = 0.05
H: Final = {"sba": 72, "civil": 36, "civil_broad": 36}
ROWS: Final = {  # Table 2 rows: (column, label); the ladder rows add one step each
    "forecast": [
        ("naive", "Events so far"),
        ("drop_recent", "Drop recent"),
        ("fixed_horizon", "Last mature cohort"),
        ("bf", "Bornhuetter-Ferguson"),
        ("cape_cod", "Cape Cod, all cohorts"),
        ("chain", "Chain ladder (CL)"),
        ("unit_chain", "CL + exact ages"),
        ("stratified", "CL + exact ages + groups"),
        ("cal_chain_forecast", "Calendar-adjusted CL"),
        ("ours_uncond", "Full method"),
        ("hazard_fcst", "Competing-risks hazard model"),
    ],
    "realised": [
        ("cal_chain", "Calendar-adjusted CL"),
        ("ours", "Full method"),
        ("hazard_real", "Competing-risks hazard model"),
    ],
}
LADDER: Final = ["chain", "unit_chain", "stratified", "cal_chain_forecast", "ours_uncond"]
WRONG: Final = [
    "naive",
    "chain",
    "unit_chain",
    "stratified",
    "cal_chain_forecast",
    "ours_uncond",
    "hazard_fcst",
    "cal_chain",
    "ours",
    "hazard_real",
]


def load(path: Path, name: str) -> ModuleType:
    """A module of another experiment, by path (several experiments have files of the same name)."""
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def pairs(name: str) -> pd.DataFrame:
    p = pd.read_parquet(ART / name / "pairs.parquet")
    h = pd.read_parquet(ART / name / "hazard.parquet")
    return p.merge(h, on=["snapshot", "cohort"], validate="one_to_one")


def error(p: pd.DataFrame, m: str, truth: str = "truth") -> np.ndarray:
    return ((p[m] - p[truth]) / p[truth]).to_numpy()


def median_error(p: pd.DataFrame, m: str, truth: str = "truth") -> float:
    e = pd.Series(np.abs(error(p, m, truth))).groupby(p.snapshot.to_numpy()).mean()
    return round(float(e.median()), 4)


def signs(c: int) -> np.ndarray:
    if c <= inference.MAX_EXACT:
        bits = (np.arange(2**c)[:, None] >> np.arange(c)) & 1
        return np.asarray(1 - 2 * bits, dtype=np.int8)
    random = np.random.default_rng(SEED).integers(0, 2, size=(N_RANDOM, c))
    random[0] = 0
    return np.asarray(1 - 2 * random, dtype=np.int8)


def one_sided(
    d: np.ndarray, clusters: np.ndarray, shift: float, tail: Literal["lower", "upper"]
) -> float:
    """Cluster sign-flip p value of mean(d) - shift against the alternative on the given side."""
    sums = np.bincount(np.unique(clusters, return_inverse=True)[1], weights=d - shift)
    flipped = signs(len(sums)).astype(float) @ sums
    observed = float(sums.sum())
    hits = flipped <= observed + 1e-12 if tail == "lower" else flipped >= observed - 1e-12
    return float(np.mean(hits))


def hypothesis(
    p: pd.DataFrame,
    a: str,
    b: str,
    margin: float,
    tail: Literal["lower", "upper"],
    truth: str = "truth",
) -> dict[str, object]:
    d = inference.log_ratio(error(p, a, truth), error(p, b, truth))
    cohorts = p.cohort.to_numpy().astype(np.int64)
    low, high = inference.cluster_bootstrap(d, cohorts)
    snapshot = pd.DataFrame(
        {"s": p.snapshot, "a": np.abs(error(p, a, truth)), "b": np.abs(error(p, b, truth))}
    )
    means = snapshot.groupby("s")[["a", "b"]].mean()
    out: dict[str, object] = {
        "a": a,
        "b": b,
        "pairs": len(d),
        "clusters": len(np.unique(cohorts)),
        "sign_vectors": "all 2^C"
        if len(np.unique(cohorts)) <= inference.MAX_EXACT
        else f"{N_RANDOM} random",
        "error_ratio": round(float(np.exp(d.mean())), 4),
        "ci95": [round(math.exp(low), 4), round(math.exp(high), 4)],
        "margin": margin,
        "p_one_sided": round(one_sided(d, cohorts, math.log(margin), tail), 6),
        "median_error_a": median_error(p, a, truth),
        "median_error_b": median_error(p, b, truth),
        "snapshots_where_a_is_better": f"{int((means.a < means.b).sum())}/{len(means)}",
        "two_sided_sign_flip_p_of_ratio_1": round(inference.sign_flip_p(d, cohorts), 6),
    }
    if margin != 1.0:
        out["p_one_sided_ratio_below_1"] = round(one_sided(d, cohorts, 0.0, "lower"), 6)
    return out


def holm(p: dict[str, float]) -> dict[str, float]:
    order = sorted(p, key=lambda k: p[k])
    out, running = {}, 0.0
    for i, k in enumerate(order):
        running = max(running, min(1.0, (len(order) - i) * p[k]))
        out[k] = round(running, 6)
    return out


def verdict(key: str, h: dict[str, object], holm_p: float) -> str:
    low, high = h["ci95"]  # type: ignore[misc]
    if holm_p < ALPHA:
        return "confirmed"
    refuted = {"H1": low > 1, "H2": high < 1, "H3": low > 1}[key]
    return "refuted" if refuted else "inconclusive"


def confirmatory(name: str, p: pd.DataFrame, c: pd.DataFrame) -> dict[str, object]:
    tests = {
        "H1": hypothesis(c, "pu_cl", "naive", 1.0, "lower"),
        "H2": hypothesis(p, "ours_uncond", "cal_chain_forecast", 0.85, "upper"),
        "H3": hypothesis(p, "hazard_fcst", "ours_uncond", 1.15, "lower"),
    }
    adjusted = holm({k: float(v["p_one_sided"]) for k, v in tests.items()})  # type: ignore[arg-type]
    for k, v in tests.items():
        v["p_holm"] = adjusted[k]
        v["verdict"] = verdict(k, v, adjusted[k])
        low, high = v["ci95"]  # type: ignore[misc]
        v["two_sided_95_interval_excludes_1"] = (
            "below" if high < 1 else "above" if low > 1 else "no"
        )
    return {"data_set": name, **tests}


def compare(p: pd.DataFrame, a: str, b: str) -> dict[str, object]:
    c = inference.compare(
        error(p, a),
        error(p, b),
        p.cohort.to_numpy().astype(np.int64),
        p.snapshot.to_numpy().astype(np.int64),
    )
    low, high = (math.exp(x) for x in c.ci95_cluster_bootstrap)
    return {
        "error_ratio": round(c.error_ratio, 4),
        "ci95": [round(low, 4), round(high, 4)],
        "sign_flip_p": round(c.sign_flip_p, 5),
        "snapshots_where_a_is_better": f"{c.snapshots_where_a_is_better}/{c.snapshots}",
        "wild_cluster_p": round(c.wild_cluster_p, 5),
        "leave_one_snapshot_out_max_p": round(c.leave_one_snapshot_out_max_p, 5),
    }


def table(p: pd.DataFrame, sets: dict[str, pd.Series]) -> dict[str, object]:
    out: dict[str, object] = {"pairs": len(p), "clusters": int(p.cohort.nunique())}
    for path, rows in ROWS.items():
        block = {}
        for m, label in rows:
            if m not in p:
                continue
            row: dict[str, object] = {"label": label}
            row["median_error"] = {k: median_error(p[s], m) for k, s in sets.items()}
            if m != "chain":
                row["vs_chain"] = compare(p, m, "chain")
            if m in LADDER[1:]:
                row["vs_previous_step"] = compare(p, m, LADDER[LADDER.index(m) - 1])
            block[m] = row
        out[path] = block
    out["full_method_vs_hazard_model"] = {
        "forecast": compare(p, "ours_uncond", "hazard_fcst"),
        "realised": compare(p, "ours", "hazard_real"),
    }
    out["full_method_vs_cal_chain"] = {"realised": compare(p, "ours", "cal_chain")}
    return out


def wrong_direction(p: pd.DataFrame) -> dict[str, object]:
    big = p[((p.truth - p.ref).abs() / p.ref >= 0.10).to_numpy()]
    sign = np.sign(big.truth - big.ref)
    return {
        "pairs": len(p),
        "pairs_with_true_change_ge_10pct": len(big),
        "truth_above_reference": int((sign > 0).sum()),
        "counts": {m: int((np.sign(big[m] - big.ref) != sign).sum()) for m in WRONG if m in big},
    }


def coverage(name: str, p: pd.DataFrame, r36: ModuleType) -> dict[str, object]:
    """The summary of 0036 `report.coverage` for this run's pairs and calibration errors."""
    cal = pd.read_parquet(ART / name / "calibration.parquet")
    out: dict[str, object] = {"calibration_errors": len(cal)}
    for method in ("ours_uncond", "ours", "chain"):
        for rule, margin in (("strict", 2), ("as_run", 0)):
            t = r36.intervals(p, cal, method, H[name], margin)
            for label, sub in (("J>=9", t[t.J >= 9] if len(t) else t), ("J>=1", t)):
                if not len(sub):
                    out[f"{method} {rule} {label}"] = {"pairs": 0}
                    continue
                share = {g: float(sub[f"covered_{g}"].mean()) for g in r36.GAMMAS}
                enough = [g for g in r36.GAMMAS if share[g] >= 1 - r36.ALPHA]
                out[f"{method} {rule} {label}"] = {
                    "pairs": len(sub),
                    "covered": {
                        str(g): f"{int(sub[f'covered_{g}'].sum())}/{len(sub)}" for g in r36.GAMMAS
                    },
                    "smallest_gamma_for_90pct": enough[0] if enough else None,
                    "median_log_width_gamma_1": round(float(sub["log_width_1.0"].median()), 4),
                }
    return out


def windows(name: str, e2: ModuleType) -> dict[str, object]:
    t = pd.read_parquet(ART / name / "windows.parquet")
    out: dict[str, object] = {}
    for fraction, g in t.groupby("fraction"):
        tests = [e2.paired(d, "corrected_z", "naive_lepski") for _, d in g.groupby("draw")]
        out[f"{fraction:.0%}"] = {
            "draws": len(tests),
            "median_error": g.groupby("method").rel_error.median().round(4).to_dict(),
            "mean_window": g.groupby("method").window.mean().round(2).to_dict(),
            "corrected_z_vs_naive_lepski": tests[0]
            if len(tests) == 1
            else {
                "median_error_ratio": round(float(np.median([x["error_ratio"] for x in tests])), 3),
                "draws_where_corrected_z_is_better": int(sum(x["error_ratio"] < 1 for x in tests)),
                "per_draw": tests,
            },
        }
    return out


def alarms(name: str, near: ModuleType, r67: ModuleType) -> dict[str, object]:
    """'Rate fell' alarms at immature months: the paper's rule and the one-to-one matching of 0067."""
    a = pd.read_parquet(ART / name / "alarms.parquet")
    streams = pd.read_parquet(ART / name / "streams.parquet")
    d = a[a.immature & (a.direction == -1)] if len(a) else a
    out: dict[str, object] = {
        "december_snapshots": int(streams.snapshot.nunique()),
        "months_in_the_stream": {
            "min": int(streams.groupby("snapshot").size().min()),
            "max": int(streams.groupby("snapshot").size().max()),
        },
    }
    for detector in ("adwin(0.002)", "ph(0.005,50)"):
        g = d[d.detector == detector] if len(d) else d
        truth = g[g.stream == "truth"] if len(g) else g
        for stream in ("naive", "cl", "full"):
            s = g[g.stream == stream] if len(g) else g
            confirmed = (
                near.near(s, truth) if len(s) and len(truth) else pd.Series(False, index=s.index)
            )
            hit = (
                near.near(truth, s)
                if len(s) and len(truth)
                else pd.Series(False, index=truth.index)
            )
            paired = (
                sum(
                    r67.one_to_one(
                        s[s.snapshot == y].om.to_numpy(), truth[truth.snapshot == y].om.to_numpy()
                    )
                    for y in sorted(g.snapshot.unique())
                )
                if len(g)
                else 0
            )
            out[f"{detector} {stream}"] = {
                "alarms": len(s),
                "truth_alarms": len(truth),
                "rule_of_the_paper": {
                    "false": int((~confirmed).sum()),
                    "truth_alarms_missed": int((~hit).sum()),
                },
                "one_to_one": {
                    "matched": int(paired),
                    "false": len(s) - int(paired),
                    "truth_alarms_missed": len(truth) - int(paired),
                },
            }
    return out


def calendar(name: str) -> dict[str, object]:
    """Guard G1 and the selected speed model per snapshot (meta of backtest.py)."""
    meta = json.loads((ART / name / "meta.json").read_text())
    keys = ("calendar_on", "g1_events_x_min_share", "model", "beta", "beta_se")
    return {year: {k: m[k] for k in keys} for year, m in meta.items()}


def classifier(name: str) -> tuple[pd.DataFrame, dict[str, object]]:
    c = pd.read_parquet(ART / name / "classifier.parquet")
    return c, {
        "pairs": len(c),
        "median_error": {m: median_error(c, m) for m in ("naive", "pu_cl", "pu_full")},
        "pu_full_vs_naive": compare(c, "pu_full", "naive"),
        "pu_full_vs_pu_cl": compare(c, "pu_full", "pu_cl"),
        "pu_cl_vs_naive": compare(c, "pu_cl", "naive"),
    }


def main() -> None:
    r36 = load(EXP / "0036-corrected-plugin-backtests" / "report.py", "report36")
    r67 = load(EXP / "0067-v10-answers" / "report.py", "report67")
    sys.path.insert(0, str(EXP / "0011-freddie-200k-confirmatory"))
    e2 = load(EXP / "0014-e2-freddie-adaptive-windows" / "run.py", "run14")
    out: dict[str, object] = {}
    for name in ("sba", "civil"):
        p = pairs(name)
        c, secondary_c = classifier(name)
        sets = {"all": pd.Series(True, index=p.index)}
        if name == "sba":
            sets |= {"2000-2013": ~p.truncated, "2014-2018 (cut)": p.truncated}
        out[name] = {
            "confirmatory": confirmatory(name, p, c),
            "table2": table(p, sets),
            "wrong_direction": wrong_direction(p),
            "coverage": coverage(name, p, r36),
            "calendar_step": calendar(name),
            "secondary": {
                "windows": windows(name, e2),
                "drift_rate_fell": alarms(name, r67.alarms, r67),
                "classifier_c": secondary_c,
            },
        }
        if name == "sba":
            q = p[p.hazard_lt_fcst.notna()]
            out[name]["hazard_left_truncated_post_hoc"] = {
                "pairs": len(q),
                "snapshots": f"{int(q.snapshot.min())}-{int(q.snapshot.max())}",
                "forecast": hypothesis(q, "hazard_lt_fcst", "ours_uncond", 1.15, "lower"),
                "realised": hypothesis(q, "hazard_lt_real", "ours", 1.15, "lower"),
                "as_specified_on_the_same_pairs": hypothesis(
                    q, "hazard_fcst", "ours_uncond", 1.15, "lower"
                ),
            }
    b = pairs("civil_broad")
    out["civil_broad"] = {
        "H2_procedure": hypothesis(b, "ours_uncond", "cal_chain_forecast", 0.85, "upper"),
        "H3_procedure": hypothesis(b, "hazard_fcst", "ours_uncond", 1.15, "lower"),
        "table2": table(b, {"all": pd.Series(True, index=b.index)}),
        "wrong_direction": wrong_direction(b),
    }
    (HERE / "results.json").write_text(json.dumps(out, indent=1, default=str))


if __name__ == "__main__":
    main()
