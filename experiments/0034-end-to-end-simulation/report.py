"""Tables of the end-to-end simulation (sim.py) and of the window rule with estimated maturities (windows_row.py).

DATASET: none (simulated). Reads artifacts/parts/*.parquet and artifacts/windows_*.csv.
A pair = one young cohort year at the scored snapshot of one replication. Error ratio = exp(mean difference of log
absolute relative errors), floor 1e-6 (`maturity.inference.log_ratio`); its interval uses the standard error of the
replication means (replications are independent).
Usage: uv run python report.py -> results.json, research/drift-or-maturity/figdata/sim_*.csv
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd
import sim

from maturity import inference

HERE: Final = Path(__file__).resolve().parent
ART: Final = HERE / "artifacts"
FIGDATA: Final = HERE.parents[1] / "research" / "drift-or-maturity" / "figdata"
METHODS: Final = {
    "naive": "events so far",
    "chain": "chain ladder",
    "cal_chain": "calendar-adjusted chain ladder (realised path)",
    "ours": "full method (realised path)",
    "own": "calendar-adjusted, own speed per cohort (realised path)",
    "cal_chain_forecast": "calendar-adjusted chain ladder (forecast path)",
    "ours_uncond": "full method (forecast path)",
}
RATIOS: Final = (
    ("ours", "cal_chain"),  # the gain from speeds
    ("ours_uncond", "cal_chain_forecast"),
    ("own", "cal_chain"),
    ("ours", "own"),
    ("ours", "chain"),
    ("ours_uncond", "chain"),
    ("cal_chain", "chain"),
    ("cal_chain_forecast", "chain"),
    ("own", "chain"),
    ("naive", "chain"),
)
SIGN_MIN: Final = (
    0.05  # smallest relative change of the true rate between adjacent cohort years that is scored
)
D_BINS: Final = ([0, 100, 1_000, 10_000, np.inf], ["<100", "100-1k", "1k-10k", ">10k"])
# the condition of the shrinkage proposition, four versions:
#   worth, worth_smooth   omega2 + tau2 > phi / (D V), all estimated (experiments/0031 `criterion`; V on the pooled
#                         empirical reference shape, or on the generalized-gamma reference)
#   worth_oracle(_phi)    (true log speed)^2 > 1 / (D V), V from the true shape: own speed against the reference
#                         speed, which is what equal speeds borrow; _phi: variance times the estimated dispersion
CRITERIA: Final = {
    "worth": "as in 0031",
    "worth_smooth": "smooth reference",
    "worth_oracle": "oracle",
    "worth_oracle_phi": "oracle with dispersion",
}


def relative_error(p: pd.DataFrame, method: str) -> pd.Series:
    return (p[method] - p.truth) / p.truth


def ratio(p: pd.DataFrame, a: str, b: str) -> dict[str, object]:
    """Error ratio of a against b with a 95% interval from the replication means."""
    if p.empty:
        return {"pairs": 0}
    d = pd.Series(
        inference.log_ratio(relative_error(p, a).to_numpy(), relative_error(p, b).to_numpy()),
        index=p.index,
    )
    by_rep = d.groupby([p.setting, p.rep]).mean()
    mean = float(d.mean())
    se = float(by_rep.std(ddof=1) / np.sqrt(len(by_rep))) if len(by_rep) > 1 else float("nan")
    return {
        "pairs": len(d),
        "replications": len(by_rep),
        "ratio": round(float(np.exp(mean)), 4),
        "ci95": [
            round(float(np.exp(mean - 1.96 * se)), 4),
            round(float(np.exp(mean + 1.96 * se)), 4),
        ],
        "share_of_pairs_a_better": round(float((d < 0).mean()), 3),
        "share_of_pairs_equal": round(float((d == 0).mean()), 3),
    }


def coverage(p: pd.DataFrame, method: str) -> dict[str, object]:
    q = p[p[f"{method}_J"].notna() & (p[f"{method}_J"] >= 1)]
    covered = (q[f"{method}_low"] <= q.truth) & (q.truth <= q[f"{method}_high"])
    width = q[f"{method}_high"] / q[f"{method}_low"]

    def part(mask: pd.Series) -> dict[str, object]:
        n = int(mask.sum())
        share = float(covered[mask].mean()) if n else float("nan")
        return {
            "pairs": n,
            "coverage": round(share, 3),
            "se": round(float(np.sqrt(share * (1 - share) / n)), 3) if n else None,
            "median_width_ratio": round(float(width[mask].median()), 3) if n else None,
            "median_log_width": round(float(np.log(width[mask]).median()), 3) if n else None,
        }

    return {"all": part(q.A > 0), "J_ge_9": part(q[f"{method}_J"] >= 9)} | {
        f"age_{int(a)}": part(q.A == a) | {"J": int(q[q.A == a][f"{method}_J"].iloc[0])}
        for a in sorted(q.A.unique())
    }


def sign_recovery(p: pd.DataFrame) -> dict[str, object]:
    """Share of adjacent cohort years (the last mature one and the young ones) whose change of the true rate has the
    estimated sign, for true changes of at least SIGN_MIN (relative)."""
    width = p.groupby(["setting", "rep"]).size().iloc[0]
    truth = p.truth.to_numpy().reshape(-1, width)
    change = np.diff(truth, axis=1)
    scored = np.abs(change / truth[:, :-1]) >= SIGN_MIN
    out: dict[str, object] = {"pairs": int(scored.sum())}
    for m in METHODS:
        estimated = np.diff(p[m].to_numpy().reshape(-1, width), axis=1)
        out[m] = round(float((np.sign(estimated) == np.sign(change))[scored].mean()), 3)
    return out


def speed_recovery(young: pd.DataFrame) -> dict[int, dict[str, float]]:
    """Where the true speed is not 1: does the fitted log speed have the right sign, and how large is it against
    the true one (median ratio; 1 = recovered, 0 = not moved)?"""
    out = {}
    changed = young[np.log(young.true_speed).abs() > 1e-9]
    for age, g in changed.groupby("A"):
        true = np.log(g.true_speed)
        out[int(age)] = {
            f"{label}_{name}": round(float(value), 3)
            for label, column in (("full", "speed"), ("own", "own_speed"))
            for name, value in (
                ("right_direction", (np.sign(np.log(g[column])) == np.sign(true)).mean()),
                ("median_ratio_to_true_log_speed", (np.log(g[column]) / true).median()),
            )
        }
    return out


def by_direction(young: pd.DataFrame) -> dict[str, object]:
    """Young cohorts by their true speed against the reference (faster, slower, equal): mean signed relative error,
    gain from speeds and coverage of the full method's interval."""
    direction = np.where(
        young.true_speed > 1 + 1e-9,
        "faster",
        np.where(young.true_speed < 1 - 1e-9, "slower", "equal"),
    )
    out: dict[str, object] = {}
    for name in ("faster", "slower", "equal"):
        g = young[direction == name]
        if g.empty:
            continue
        with_interval = g[g.ours_J.notna()]
        covered = (with_interval.ours_low <= with_interval.truth) & (
            with_interval.truth <= with_interval.ours_high
        )
        out[name] = {
            "pairs": len(g),
            "mean_signed_error": {
                m: round(float(relative_error(g, m).mean()), 4)
                for m in ("chain", "cal_chain", "ours", "own")
            },
            "median_fitted_speed": round(float(g.speed.median()), 3),
            "median_true_speed": round(float(g.true_speed.median()), 3),
            "full vs cal_chain": ratio(g, "ours", "cal_chain"),
            "coverage_full": round(float(covered.mean()), 3) if len(covered) else None,
            "pairs_with_interval": len(covered),
        }
    return out


def setting_block(p: pd.DataFrame, meta: pd.DataFrame) -> dict[str, object]:
    young = p[p.young]
    errors = pd.DataFrame({m: relative_error(young, m).abs() for m in METHODS})
    with_intervals = p[p.ours_J.notna() & p.young]
    models = meta.model.str.replace(r"\(.*\)", "", regex=True).value_counts(normalize=True)
    return {
        "replications": int(meta.rep.nunique()),
        "replications_with_intervals": int(meta.intervals.sum()),
        "speed_model_selected": {k: round(float(v), 3) for k, v in models.items()},
        "calendar_step_on": round(float(meta.calendar_on.mean()), 3),
        "beta_median": [round(float(meta.beta0.median()), 3), round(float(meta.beta1.median()), 3)],
        "phi_median": round(float(meta.phi.median()), 2),
        "true_rate_median": round(float(young.truth.median()), 4),
        "events_so_far_median_by_age": {
            int(a): int(v) for a, v in young.groupby("A").D.median().items()
        },
        "median_abs_error": errors.median().round(4).to_dict(),
        "mean_abs_error": errors.mean().round(4).to_dict(),
        "median_abs_error_by_age": {
            int(a): g.median().round(4).to_dict() for a, g in errors.groupby(young.A)
        },
        "error_ratios": {f"{a} vs {b}": ratio(young, a, b) for a, b in RATIOS},
        "gain_from_speeds_by_age": {
            int(a): ratio(young[young.A == a], "ours", "cal_chain")
            for a in sorted(young.A.unique())
        },
        "own_speed_vs_equal_speeds_by_age": {
            int(a): ratio(young[young.A == a], "own", "cal_chain") for a in sorted(young.A.unique())
        },
        "speed_recovery_by_age": speed_recovery(young),
        "by_direction_of_true_speed": by_direction(young),
        "sign_recovery": sign_recovery(p),
        "coverage": {m: coverage(with_intervals, m) for m in sim.INTERVALS},
    }


def speed_gain(young: pd.DataFrame) -> pd.DataFrame:
    """Gain from speeds (full method, and own speed, against equal speeds) by setting, age, events so far and the
    condition of the shrinkage proposition (three versions)."""
    rows = []
    cuts = {"all": [("all", young.A > 0)]} | {
        name: [(f"{label}: holds", young[name]), (f"{label}: fails", ~young[name])]
        for name, label in CRITERIA.items()
    }
    for parts in cuts.values():
        for label, mask in parts:
            for keys, g in young[mask].groupby(["setting", "A", "D_bin"], observed=True):
                full, own = ratio(g, "ours", "cal_chain"), ratio(g, "own", "cal_chain")
                rows.append(
                    dict(
                        zip(("setting", "age_months", "events_so_far", "criterion"), (*keys, label))
                    )
                    | {
                        "error_ratio_full_vs_equal_speeds": full["ratio"],
                        "ci95_low": full["ci95"][0],
                        "ci95_high": full["ci95"][1],
                        "error_ratio_own_speed_vs_equal_speeds": own["ratio"],
                        "n": full["pairs"],
                    }
                )
    return pd.DataFrame(rows)


def elasticity(u: float, timing: str, eps: float = 1e-4) -> float:
    up, down = (float(np.log(sim.f0(np.array([u * np.exp(x)]), timing)[0])) for x in (eps, -eps))
    return (up - down) / (2 * eps)


def inflation(young: pd.DataFrame) -> list[dict[str, object]]:
    """Theorem 6 in the settings with equal speeds: mean squared log error with an own speed against equal speeds,
    and the predicted factor 1 + (e(A) - e(H))^2 / V (e = elasticity of F0, V = information per event)."""
    rows = []
    for (setting, age), g in young[young.setting.str.contains("/constant/")].groupby(
        ["setting", "A"]
    ):
        timing = setting.split("/")[0]
        v = sim.true_information(int(age), 1.0, timing)
        gap = elasticity(float(age), timing) - elasticity(float(sim.H), timing)
        mse = {m: float(np.mean(np.log(g[m] / g.truth) ** 2)) for m in ("own", "cal_chain", "ours")}
        rows.append(
            {
                "setting": setting,
                "age_months": int(age),
                "events_so_far_median": int(g.D.median()),
                "information_per_event": round(v, 5),
                "predicted_inflation": round(1 + gap**2 / v, 1),
                "rmse_log_equal_speeds": round(float(np.sqrt(mse["cal_chain"])), 4),
                "rmse_log_own_speed": round(float(np.sqrt(mse["own"])), 4),
                "rmse_log_full_method": round(float(np.sqrt(mse["ours"])), 4),
                "observed_inflation": round(mse["own"] / mse["cal_chain"], 1),
            }
        )
    return rows


def condition_summary(young: pd.DataFrame) -> dict[str, object]:
    """Does the condition of the shrinkage proposition say where an own speed (and the full method) wins?"""
    out: dict[str, object] = {}
    groups = {
        "speeds change (step and drift)": ~young.setting.str.contains("/constant/"),
        "speeds equal": young.setting.str.contains("/constant/"),
    }
    for timing in ("weibull", "lognormal"):
        for label, mask in groups.items():
            g = young[mask & young.setting.str.startswith(timing)]
            block: dict[str, object] = {}
            for name, text in CRITERIA.items():
                block[text] = {
                    "share_holds": round(float(g[name].mean()), 3),
                    "holds": {
                        "own vs equal speeds": ratio(g[g[name]], "own", "cal_chain"),
                        "full vs equal speeds": ratio(g[g[name]], "ours", "cal_chain"),
                    },
                    "fails": {
                        "own vs equal speeds": ratio(g[~g[name]], "own", "cal_chain"),
                        "full vs equal speeds": ratio(g[~g[name]], "ours", "cal_chain"),
                    },
                }
            out[f"{timing}, {label}"] = block
    return out


def end_to_end_csv(blocks: dict[str, dict[str, object]]) -> pd.DataFrame:
    rows = []
    for setting, b in blocks.items():
        for m, label in METHODS.items():
            cover = b["coverage"].get(m, {}).get("all", {})
            ratios = b["error_ratios"]
            rows.append(
                {
                    "setting": setting,
                    "method": label,
                    "replications": b["replications"],
                    "replications_with_intervals": b["replications_with_intervals"],
                    "median_abs_relative_error": b["median_abs_error"][m],
                    "error_ratio_vs_chain_ladder": ratios.get(f"{m} vs chain", {}).get("ratio"),
                    "error_ratio_vs_equal_speeds": ratios.get(
                        f"{m} vs {'cal_chain_forecast' if m == 'ours_uncond' else 'cal_chain'}", {}
                    ).get("ratio"),
                    "sign_recovery": b["sign_recovery"][m],
                    "coverage_90": cover.get("coverage"),
                    "coverage_se": cover.get("se"),
                    "interval_width_ratio_median": cover.get("median_width_ratio"),
                }
            )
    return pd.DataFrame(rows)


def windows() -> dict[str, object]:
    out: dict[str, object] = {}

    def medians(t: pd.DataFrame, by: list[str]) -> pd.DataFrame:
        return (100 * t.groupby([*by, "n"]).rel_error.median().unstack()).round(1)

    if (ART / "windows_check.csv").exists():
        old = pd.read_csv(ART / "windows_check.csv")
        out["existing_rows_by_the_code_of_0013_median_error_pct"] = medians(
            old, ["method"]
        ).T.to_dict()
    frames = []
    for name in ("full", "speed"):
        if (ART / f"windows_{name}.csv").exists():
            t = pd.read_csv(ART / f"windows_{name}.csv")
            m = medians(t, ["speeds", "method"])
            out[f"{name}_grid_median_error_pct"] = {
                f"{a} | {b}": row.to_dict() for (a, b), row in m.iterrows()
            }
            w = t.groupby(["speeds", "method", "n"]).window.mean().unstack().round(2)
            out[f"{name}_grid_mean_window"] = {
                f"{a} | {b}": row.to_dict() for (a, b), row in w.iterrows()
            }
            out[f"{name}_grid_settings_per_cell"] = int(
                t.groupby(["speeds", "method", "n"]).size().iloc[0]
            )
            frames.append(
                t.groupby(["speeds", "method", "n"])
                .agg(
                    median_rel_error=("rel_error", "median"),
                    mean_window=("window", "mean"),
                    settings=("rel_error", "size"),
                )
                .reset_index()
                .assign(grid=name)
            )
    if frames:
        pd.concat(frames).round(5).to_csv(FIGDATA / "sim_window_rule.csv", index=False)
    return out


def main() -> None:
    FIGDATA.mkdir(parents=True, exist_ok=True)
    p, meta = sim.results()
    p = p.merge(meta[["setting", "rep", "phi"]], on=["setting", "rep"], validate="many_to_one")
    with np.errstate(divide="ignore"):
        p["worth_oracle"] = np.log(p.true_speed) ** 2 > 1 / (p.D * p.V_true)
        p["worth_oracle_phi"] = np.log(p.true_speed) ** 2 > p.phi / (p.D * p.V_true)
    p["D_bin"] = pd.cut(p.D, D_BINS[0], labels=D_BINS[1], right=False)
    order = [s.name for s in sim.SETTINGS]
    blocks = {
        name: setting_block(p[p.setting == name], meta[meta.setting == name]) for name in order
    }
    young = p[p.young]
    gain = speed_gain(young)
    out = {
        "design": {
            "cohort_years": sim.K,
            "horizon_months": sim.H,
            "groups": sim.G,
            "units_per_cohort_year": {k: 12 * sim.G * v for k, v in sim.SIZES.items()},
            "scored_snapshot": f"December of year {sim.TARGET}",
            "calibration_snapshots": [int(y) for y in sim.CALIBRATION],
            "replications_all_and_with_intervals": {
                f"{k[0]}, calendar {k[1]}": v for k, v in sim.REPS.items()
            },
            "nominal_coverage": 1 - sim.ALPHA,
            "sign_recovery_min_relative_change": SIGN_MIN,
        },
        "settings": blocks,
        "shrinkage_condition": condition_summary(young),
        "theorem_6_inflation_equal_speeds": inflation(young),
        "windows": windows(),
        "runtime": {
            "wall_seconds_end_to_end": sum(
                int(f.read_text()) for f in sim.PARTS.glob("seconds_*.txt")
            ),
            "workers": sim.WORKERS,
            "seconds_per_replication_with_intervals_median": round(
                float(meta[meta.intervals].seconds.median()), 2
            ),
            "seconds_per_replication_without_intervals_median": round(
                float(meta[~meta.intervals].seconds.median()), 2
            ),
        },
    }
    (HERE / "results.json").write_text(json.dumps(out, indent=1, default=str))
    end_to_end_csv(blocks).to_csv(FIGDATA / "sim_end_to_end.csv", index=False)
    gain.to_csv(FIGDATA / "sim_speed_gain.csv", index=False)


if __name__ == "__main__":
    main()
