"""Why the calendar simulation of 0006 gives a drift ratio of 0.80 against a truth of 0.75 (review question 6).

DATASET: none (the simulation of experiments/0006, same seed, so the baseline is the same 100 data sets).
The estimate "ours + calendar + smooth" is computed again together with three variants that change one thing each:
  oracle speeds     the true cohort speeds in the place of the fitted and smoothed speeds;
  smooth where the speed changes
                    the log-linear smoothing of 0006 is fitted to the cohorts with 3 to 12 years of follow-up, but
                    the true speed is flat (1.0) for 8 years or more and rises only below 8. Here the line is
                    fitted to the cohorts with 3 to 8 years, and the older cohorts keep their own fitted speed;
  10 x the units    40,000 units per cohort x group (does the gap go away with more data?).
Usage: uv run python calendar_sim_bias.py [reps] -> results_calendar_sim.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Final

import numpy as np

HERE: Final = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "0006-t2c-calendar-sim"))
import sim

YOUNG: Final = slice(sim.T - 5, sim.T)
OLD: Final = sim.A >= 12
VARIANTS: Final = ("as in 0006", "oracle speeds", "smooth where the speed changes")


def level(curve: np.ndarray, c: int, speed: float, f0_ref: np.ndarray) -> float:
    grid = np.arange(len(f0_ref)) * sim.DT
    return float(
        curve[round(sim.A[c] / sim.DT)]
        / max(np.interp(speed * sim.A[c], grid, f0_ref, right=1.0), 1e-9)
    )


def one_rep(rng: np.random.Generator) -> tuple[np.ndarray, dict[str, np.ndarray], np.ndarray]:
    w = sim.world(rng, False)
    p_true = sim.truth(w)
    data = sim.simulate(rng, w)
    events = data["ev"].sum(axis=1)
    n_ev = np.array([events[c, : round(sim.A[c] / sim.DT)].sum() for c in range(sim.T)])
    beta, kappa = sim.fit_calendar(w, data)
    adj = sim.adjusted_curves(w, data, beta, kappa)
    _, theta_smooth, free, smooth = sim.t2_estimates(adj, n_ev)
    f0_ref = np.concatenate([adj[0] / adj[0][-1], [1.0]])

    def estimate(speeds: np.ndarray) -> np.ndarray:
        theta = np.array([level(adj[c], c, speeds[c], f0_ref) for c in range(sim.T)])
        return sim.plug_in(w, beta, kappa, theta, speeds, f0_ref)

    idx = np.where((sim.A >= 3) & (sim.A <= 8) & ~np.isnan(free))[0]
    line = np.polyfit(idx, np.log(free[idx]), 1, w=np.sqrt(n_ev[idx]))
    kink = np.where(
        sim.A <= 8, np.exp(np.polyval(line, np.arange(sim.T))), np.nan_to_num(free, nan=1.0)
    )
    estimates = {
        VARIANTS[0]: sim.plug_in(w, beta, kappa, theta_smooth, smooth, f0_ref),
        VARIANTS[1]: estimate(sim.SPEED),
        VARIANTS[2]: estimate(kink),
    }
    return p_true, estimates, smooth


def run(reps: int, units: int) -> dict[str, object]:
    sim.N_UNITS = units  # type: ignore[misc]
    rng = np.random.default_rng(11)  # the seed of 0006
    runs = [one_rep(rng) for _ in range(reps)]
    p_true = np.array([r[0] for r in runs])
    out: dict[str, object] = {
        "replications": reps,
        "units_per_cohort_and_group": units,
        "truth_drift": round(
            float(np.mean(p_true[:, YOUNG].mean(axis=1) / p_true[:, OLD].mean(axis=1))), 3
        ),
        "true_speed_by_years_of_follow_up": {
            int(a): round(float(s), 3) for a, s in zip(sim.A[YOUNG], sim.SPEED[YOUNG], strict=True)
        },
        "smoothed_speed_median": {
            int(a): round(float(np.median([r[2][c] for r in runs])), 3)
            for c, a in zip(range(sim.T - 5, sim.T), sim.A[YOUNG], strict=True)
        },
    }
    for name in VARIANTS:
        est = np.array([r[1][name] for r in runs])
        drift = est[:, YOUNG].mean(axis=1) / est[:, OLD].mean(axis=1)
        signed = ((est - p_true) / p_true).mean(axis=0)
        out[name] = {
            "drift_median": round(float(np.median(drift)), 3),
            "drift_mean": round(float(drift.mean()), 3),
            "drift_p5_p95": np.round(np.percentile(drift, [5, 95]), 3).tolist(),
            "mean_signed_error_young_by_years_of_follow_up": {
                int(a): round(float(signed[c]), 3)
                for c, a in zip(range(sim.T - 5, sim.T), sim.A[YOUNG], strict=True)
            },
            "mean_signed_error_old_cohorts": round(float(signed[OLD].mean()), 3),
        }
    return out


def main(reps: int) -> None:
    out = {"4,000 units (0006)": run(reps, 4_000), "40,000 units": run(reps, 40_000)}
    (HERE / "results_calendar_sim.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 100)
