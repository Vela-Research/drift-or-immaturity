"""E1: adaptive training windows under delayed labels, with known truth (the window-experiment specification, T4 section 6).

DATASET: none (synthetic, fixed seed).
Setup (T4 section 2): K cohorts aged d, d+1, ..., d+K-1 years at the snapshot, n units each. A unit of the cohort
aged j succeeds eventually with probability p_j; the time to success is Weibull(shape, median), the same for every
cohort. Seen at the snapshot: successes so far, Binomial(n, p_j F(j)). Target: p_d, the youngest cohort's rate.
Drift (p_j = rate of the cohort aged j): none; linear_up = older cohorts higher, p_d (1 + 0.02 (j - d));
linear_down = older cohorts lower, p_d (1 - 0.02 (j - d)); step = cohorts older than the 5 youngest are 30% higher.
Methods: see METHODS. kappa by rolling origin uses the pseudo-snapshots 1, 2 and 3 years before the snapshot.
F-hat = empirical distribution of the event ages of 5 fully mature reference cohorts (T4 Update, part b).
Usage: run.py reduced|full|all -> results.json (reduced grid, drift-size sweep, T4 section 5 check),
artifacts/grid_full.csv, figures/*.png
"""

from __future__ import annotations

import itertools
import json
import logging
import sys
import zlib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from maturity import windows

log = logging.getLogger(__name__)
HERE: Final = Path(__file__).resolve().parent
Array = NDArray[np.float64]
SEED: Final = 20261004
REPS: Final = 500
ORIGINS: Final = (1, 2, 3)  # pseudo-snapshots (years back) for the rolling-origin kappa
N_REFERENCE: Final = 5  # fully mature cohorts behind F-hat
ALPHA_F: Final = 0.05
SLOPE: Final = 0.02
STEP: Final = 0.30
N_GRID: Final = (1_000, 10_000, 100_000, 1_000_000)
METHODS: Final = {
    "naive_full": "naive, full window",
    "drop_recent": "naive, drop recent (older half)",
    "naive_lepski": "naive Lepski (kappa = z)",
    "naive_adwin": "naive ADWIN",
    "corrected_full": "corrected, full window",
    "corrected_proven": "corrected Lepski, proven constants",
    "corrected_z": "corrected Lepski, kappa = z",
    "corrected_1": "corrected Lepski, kappa = 1",
    "corrected_rolling": "corrected Lepski, kappa by rolling origin",
    "corrected_fhat": "corrected Lepski, kappa = z, estimated F",
    "corrected_fhat_proven": "corrected Lepski, proven constants, estimated F",
    "oracle": "oracle fixed window (corrected)",
}
ADAPTIVE: Final = tuple(
    m for m in METHODS if "lepski" in m or "adwin" in m or m.startswith("corrected_")
)


@dataclass(frozen=True, slots=True)
class Scenario:
    shape: float
    median: float
    p: float
    drift: str  # none | linear_up | linear_down | step | slope:<relative slope per year>
    n: int
    k: int
    d: int


def maturity(ages: Array, sc: Scenario) -> Array:
    return np.asarray(1 - np.exp(-np.log(2) * (np.maximum(ages, 0) / sc.median) ** sc.shape))


def rates(sc: Scenario) -> Array:
    gap = np.arange(sc.k, dtype=float)
    if sc.drift == "step":
        return sc.p * np.where(gap >= 5, 1 + STEP, 1.0)
    slope = {"none": 0.0, "linear_up": SLOPE, "linear_down": -SLOPE}.get(sc.drift)
    return sc.p * (1 + (float(sc.drift.split(":")[1]) if slope is None else slope) * gap)


def simulate(sc: Scenario, reps: int, rng: np.random.Generator) -> Array:
    """Successes seen 3, 2, 1 and 0 years before the snapshot: shape (4, reps, cohorts), youngest cohort first."""
    ages = sc.d + np.arange(sc.k, dtype=float)
    back = np.arange(len(ORIGINS), -1, -1)  # 3, 2, 1, 0 years back
    cdf = rates(sc)[:, None] * maturity(ages[:, None] - back[None, :], sc)  # (cohorts, 4)
    probs = np.concatenate([np.diff(cdf, axis=1, prepend=0.0), 1 - cdf[:, [-1]]], axis=1)
    counts = np.stack([rng.multinomial(sc.n, probs[i], size=reps) for i in range(sc.k)], axis=1)
    return np.cumsum(counts[:, :, :-1], axis=2).transpose(2, 0, 1).astype(float)


def estimated_maturity(sc: Scenario, reps: int, rng: np.random.Generator) -> tuple[Array, Array]:
    """F-hat at the cohort ages from the event ages of fully mature reference cohorts, and its DKW error bound."""
    ages = sc.d + np.arange(sc.k, dtype=float)
    n_events = np.maximum(rng.binomial(N_REFERENCE * sc.n, rates(sc)[-1], size=reps), 1)
    cdf = maturity(ages, sc)
    probs = np.concatenate([np.diff(cdf, prepend=0.0), [1 - cdf[-1]]])
    counts = rng.multinomial(n_events, probs)
    f_hat = np.maximum(np.cumsum(counts[:, :-1], axis=1), 0.5) / n_events[:, None]
    return f_hat, np.sqrt(np.log(2 / ALPHA_F) / (2 * n_events))[:, None]


def evaluate(sc: Scenario, reps: int = REPS, seed: int = SEED) -> list[dict[str, object]]:
    rng = np.random.default_rng([seed, zlib.crc32(repr(sc).encode())])
    seen = simulate(sc, reps, rng)
    events, n = seen[-1], np.array(float(sc.n))
    ages = sc.d + np.arange(sc.k, dtype=float)
    f = maturity(ages, sc)
    f_hat, f_error = estimated_maturity(sc, reps, rng)
    origins = [
        windows.Origin(
            seen[-1 - h][:, h:],
            n,
            f[: sc.k - h],
            events[:, h] - seen[-1 - h][:, h],
            np.array(sc.n * (f[h] - f[0])),
        )
        for h in ORIGINS
    ]
    naive, _ = windows.naive_estimates(events, n)
    corrected, _ = windows.corrected_estimates(events, n, f)
    full = np.full(reps, sc.k, dtype=np.int64)
    best = int(np.argmin(np.abs(corrected - sc.p).mean(axis=0))) + 1
    out = {
        "naive_full": (full, naive[:, -1]),
        "drop_recent": (full, events[:, sc.k // 2 :].sum(axis=1) / (sc.n * (sc.k - sc.k // 2))),
        "naive_lepski": windows.naive_lepski(events, n),
        "naive_adwin": windows.adwin(events, n),
        "corrected_full": (full, corrected[:, -1]),
        "corrected_proven": windows.corrected_lepski_proven(events, n, f),
        "corrected_z": windows.corrected_lepski(events, n, f),
        "corrected_1": windows.corrected_lepski(events, n, f, kappa=1.0),
        "corrected_rolling": windows.corrected_lepski(
            events, n, f, kappa=windows.choose_kappa(origins)
        ),
        "corrected_fhat": windows.corrected_lepski(events, n, f_hat),
        "corrected_fhat_proven": windows.corrected_lepski_proven(events, n, f_hat, f_error=f_error),
        "oracle": (np.full(reps, best, dtype=np.int64), corrected[:, best - 1]),
    }
    return [
        asdict(sc)
        | {
            "method": m,
            "rel_error": float(np.abs(est / sc.p - 1).mean()),
            "window": float(window.mean()),
            "window_1_share": float((window == 1).mean()),
        }
        for m, (window, est) in out.items()
    ]


def run(scenarios: list[Scenario]) -> pd.DataFrame:
    return pd.DataFrame([row for sc in scenarios for row in evaluate(sc)])


def reduced() -> dict[str, list[dict[str, object]]]:
    def base(n: int, drift: str) -> Scenario:
        return Scenario(1.5, 7.0, 0.045, drift, n, 15, 1)

    grid = run(
        [base(n, drift) for drift in ("none", "linear_up", "linear_down", "step") for n in N_GRID]
    )
    sizes = (-0.04, -0.03, -0.02, -0.01, -0.005, 0.0, 0.005, 0.01, 0.02, 0.03, 0.04)
    sweep = run([base(n, f"slope:{s}") for n in (10_000, 100_000) for s in sizes])
    # T4 section 5 (the authors' simulation): absolute slopes 0, +0.001 and -0.002 per year at p = 4.5%
    check = run(
        [
            base(n, f"slope:{delta / 0.045}")
            for delta in (0.0, 0.001, -0.002)
            for n in (1_800, 20_000, 200_000, 2_000_000)
        ]
    )
    return {
        k: v.round(5).to_dict("records")
        for k, v in (("grid", grid), ("sweep", sweep), ("t4_section5", check))
    }


def full() -> pd.DataFrame:
    grid = itertools.product(
        (0.8, 1.5, 3.0),
        (3.0, 7.0, 12.0),
        (0.01, 0.045, 0.20),
        ("none", "linear_up", "linear_down", "step"),
        N_GRID,
        (10, 15, 25),
        (1, 3, 8),
    )
    return run([Scenario(*g) for g in grid])


def main(what: str) -> None:
    if what in ("reduced", "all"):
        (HERE / "results.json").write_text(json.dumps(reduced(), indent=1))
        log.info("reduced grid written")
    if what in ("full", "all"):
        (HERE / "artifacts").mkdir(exist_ok=True)
        t = full()
        t.round(5).to_csv(HERE / "artifacts" / "grid_full.csv", index=False)
        log.info("full grid: %d scenarios", len(t) // len(METHODS))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    main(sys.argv[1] if len(sys.argv) > 1 else "reduced")
