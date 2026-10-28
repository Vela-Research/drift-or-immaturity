"""The condition of the shrinkage proposition against the realised gain from speeds, per scored pair.

DATASETS: criterion files of criterion.py; pair files of experiments/0031 (errors of the full method and of the
calendar-adjusted chain ladder on the same pairs).
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
PAIRS: Final = HERE.parent / "0031-round3-backtests" / "artifacts"
# name of the criterion file -> (pair file of 0031, paths: full method and equal speeds)
RUNS: Final = {
    "patents": (
        "pairs_patents_main",
        {"realised": ("ours", "cal_chain"), "forecast": ("ours_uncond", "cal_chain_forecast")},
    ),
    "corrected": (
        "pairs_freddie_corrected",
        {"realised": ("ours", "cal_chain"), "forecast": ("ours_uncond", "cal_chain_forecast")},
    ),
    "linked": (
        "pairs_freddie_linked",
        {"realised": ("ours", "cal_chain"), "forecast": ("ours_uncond", "cal_chain_forecast")},
    ),
}
KEEP: Final = [
    "snapshot",
    "cohort",
    "A",
    "speed",
    "maturity_m",
    "D",
    "V",
    "inflation",
    "phi",
    "own_var",
    "own_var_phi",
    "omega2",
    "tau2",
    "holds",
    "holds_without_phi",
]


def ratio(gain: np.ndarray) -> float | None:
    return round(float(np.exp(gain.mean())), 3) if len(gain) else None


def block(name: str) -> dict[str, object]:
    pair_file, paths = RUNS[name]
    c = pd.read_parquet(HERE / "artifacts" / f"criterion_{name}.parquet")
    p = pd.read_parquet(PAIRS / f"{pair_file}.parquet")
    scored = p[
        ["snapshot", "cohort", "truth", *{m for path in paths.values() for m in path}]
    ].merge(c, on=["snapshot", "cohort"], validate="one_to_one")
    assert len(scored) == len(p), "a scored pair has no criterion row"
    out: dict[str, object] = {
        "pairs": len(scored),
        "share_condition_holds": round(float(scored.holds.mean()), 3),
        "share_condition_holds_without_phi": round(float(scored.holds_without_phi.mean()), 3),
        "pairs_with_zero_information": int((~np.isfinite(scored.own_var)).sum()),
        "median_inflation_factor": round(float(scored.inflation.median()), 1),
        "median_own_sd_of_log_speed_with_phi": round(
            float(np.sqrt(scored.own_var_phi).median()), 3
        ),
        "median_borrowed_sd_of_log_speed": round(
            float(np.sqrt(scored.omega2 + scored.tau2).median()), 3
        ),
    }
    for path, (full, equal) in paths.items():
        error = {m: ((scored[m] - scored.truth) / scored.truth).to_numpy() for m in (full, equal)}
        gain = inference.log_ratio(error[full], error[equal])
        holds = scored.holds.to_numpy()
        margin = np.log((scored.omega2 + scored.tau2) / scored.own_var_phi).to_numpy()
        out[f"{path}_path"] = {
            "error_ratio_full_over_equal_speeds": ratio(gain),
            "where_the_condition_holds": {
                "pairs": int(holds.sum()),
                "error_ratio": ratio(gain[holds]),
            },
            "where_it_fails": {"pairs": int((~holds).sum()), "error_ratio": ratio(gain[~holds])},
            "without_phi_holds": ratio(gain[scored.holds_without_phi.to_numpy()]),
            "without_phi_fails": ratio(gain[~scored.holds_without_phi.to_numpy()]),
            "spearman_of_the_gain_with_the_margin": round(
                float(pd.Series(gain).corr(pd.Series(margin), method="spearman")), 3
            ),
        }
        scored[f"log_error_ratio_{path}"] = gain
    out["by_snapshot"] = {
        str(int(year)): {
            "model": g.model.iloc[0],
            "phi": round(float(g.phi.iloc[0]), 1),
            "omega2": float(g.omega2.iloc[0]),
            "gengamma_a": round(float(g.gengamma_a.iloc[0]), 2),
            "gengamma_c": round(float(g.gengamma_c.iloc[0]), 3),
            "early_exponent": round(float(g.gengamma_a.iloc[0] * g.gengamma_c.iloc[0]), 2),
            "share_holds": round(float(g.holds.mean()), 2),
            "error_ratio_full_over_equal_speeds": ratio(g.log_error_ratio_realised.to_numpy()),
            "error_ratio_forecast_path": ratio(g.log_error_ratio_forecast.to_numpy()),
        }
        for year, g in scored.groupby("snapshot")
    }
    bins = pd.cut(scored.maturity_m, [0, 0.15, 0.4, 0.7, 1.0], include_lowest=True)
    out["by_maturity"] = {
        str(interval): {
            "pairs": len(g),
            "median_age_months": float(g.A.median()),
            "share_holds": round(float(g.holds.mean()), 2),
            "median_inflation_factor": round(float(g.inflation.median()), 1),
            "error_ratio_full_over_equal_speeds": ratio(g.log_error_ratio_realised.to_numpy()),
            "error_ratio_forecast_path": ratio(g.log_error_ratio_forecast.to_numpy()),
        }
        for interval, g in scored.groupby(bins, observed=True)
    }
    columns = [*KEEP, "log_error_ratio_realised", "log_error_ratio_forecast"]
    out["per_pair"] = json.loads(scored[columns].round(6).to_json(orient="records"))
    return out


def main() -> None:
    out = {
        name: block(name)
        for name in RUNS
        if (HERE / "artifacts" / f"criterion_{name}.parquet").exists()
    }
    (HERE / "results.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
