"""Window rule with estimated, cohort-specific maturities (round 3, item P1.11). POST HOC. DATASET: none (simulated).

The E1 simulation of experiments/0013 (K cohorts aged d..d+K-1 years, n units each, Weibull timing, target = eventual
rate of the youngest cohort), with one change: the timing of cohort j is F(s_j a). The speed s_j goes log-linearly
from 1 (oldest cohort, and the reference cohorts) to 1.5 ("faster") or 1/1.5 ("slower") at the youngest cohort;
"equal" (all speeds 1) is the setting of 0013 and is the check. The data are now a development triangle: events of each
cohort by quarter of age, and 5 fully mature reference cohorts of n units (speed 1), as in 0013.

Maturity of each cohort, for the maturity-weighted Lepski rule (kappa = z, `maturity.windows.corrected_lepski`):
  true       F(s_j age_j), the true cohort-specific maturity (not available in practice)
  common     F(age_j), the known reference timing: the rule of 0013 when speeds differ
  reference  estimated from the reference cohorts only (the "estimated timing" rule of 0013)
  chain      chain ladder on the triangle (quarterly development factors), tail from the reference cohorts
  speed      the frozen speed model (`maturity.onestage.select_model`) on the reference cohorts and the K cohorts, with
             horizon 40 years: maturity = F0(s_j age_j) / F0(s_j 40). Reduced grid only (one fit per replication).

Usage: uv run python windows_row.py check | full | speed -> artifacts/windows_{check,full,speed}.csv
"""

from __future__ import annotations

import itertools
import logging
import multiprocessing as mp
import sys
import zlib
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from maturity import onestage, windows

HERE: Final = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "0013-e1-adaptive-window-sim"))
import run as e1  # the E1 simulation; a done experiment, used as it is

log = logging.getLogger(__name__)
Array = NDArray[np.float64]
WORKERS: Final = 5
PER_YEAR: Final = 4  # age bins per year
T_REF: Final = 40  # years of follow-up of the reference cohorts in the speed model (F = 0.9999 on the reduced grid)
TOP_SPEED: Final = {"equal": 1.0, "faster": 1.5, "slower": 1 / 1.5}
SPEED_REPS: Final = 60
GRID: Final = (
    (0.8, 1.5, 3.0),
    (3.0, 7.0, 12.0),
    (0.01, 0.045, 0.20),
    ("none", "linear_up", "linear_down", "step"),
    e1.N_GRID,
    (10, 15, 25),
    (1, 3, 8),
)  # the full grid of experiments/0013 `full`


def speeds(sc: e1.Scenario, path: str) -> Array:
    """Speed per cohort, youngest first."""
    return np.asarray(TOP_SPEED[path] ** (1 - np.arange(sc.k) / (sc.k - 1)))


def simulate(
    sc: e1.Scenario, path: str, reps: int, rng: np.random.Generator
) -> tuple[Array, Array, Array]:
    """Events by quarter of age: cohorts (reps, cohorts, bins; zero after a cohort's age), the reference cohorts
    (reps, 5, bins up to T_REF) and the number of eventual events of each reference cohort (reps, 5)."""
    k, d, n = sc.k, sc.d, sc.n
    ages = d + np.arange(k, dtype=float)
    edges = np.arange(PER_YEAR * (d + k - 1) + 1) / PER_YEAR
    cdf = e1.rates(sc)[:, None] * e1.maturity(speeds(sc, path)[:, None] * edges[None, :], sc)
    probs = np.diff(cdf, axis=1) * (edges[None, 1:] <= ages[:, None])
    probs = np.concatenate([probs, 1 - probs.sum(axis=1, keepdims=True)], axis=1)
    counts = np.stack([rng.multinomial(n, probs[i], size=reps) for i in range(k)], axis=1)
    ref_edges = np.arange(PER_YEAR * max(T_REF, d + k - 1) + 1) / PER_YEAR
    ref_cdf = e1.rates(sc)[-1] * e1.maturity(ref_edges, sc)
    rate = float(e1.rates(sc)[-1])
    ref_probs = np.concatenate([np.diff(ref_cdf), [rate - ref_cdf[-1]], [1 - rate]])
    reference = rng.multinomial(n, ref_probs, size=(reps, e1.N_REFERENCE))
    return (
        counts[:, :, :-1].astype(float),
        reference[:, :, :-2].astype(float),
        (n - reference[:, :, -1]).astype(float),
    )


def reference_maturity(reference: Array, eventual: Array, bins: Array) -> Array:
    """Share of the eventual events of the pooled reference cohorts seen by each age bin count in `bins`."""
    seen = np.cumsum(reference.sum(axis=1), axis=1)[:, bins - 1]
    return np.asarray(np.maximum(seen, 0.5) / np.maximum(eventual.sum(axis=1), 1.0)[:, None])


def chain_maturity(counts: Array, reference: Array, eventual: Array, bins: Array) -> Array:
    """Chain ladder: development factors by quarter of age from the cohorts that have reached that age; the share
    seen at the oldest cohort's age comes from the reference cohorts."""
    cum = np.cumsum(counts, axis=2)
    top = int(bins.max())
    reached = (
        np.arange(1, top)[None, :] < bins[:, None]
    )  # cohort has reached bin b + 1, b = 1..top-1
    later = (cum[:, :, 1:top] * reached[None]).sum(axis=1)
    earlier = (cum[:, :, : top - 1] * reached[None]).sum(axis=1)
    log_factor = np.log(np.where(earlier > 0, later / np.maximum(earlier, 1e-300), 1.0))
    to_top = np.concatenate(
        [np.cumsum(log_factor[:, ::-1], axis=1)[:, ::-1], np.zeros((len(cum), 1))], axis=1
    )  # log of the development from bin b to the top, b = 1..top
    tail = reference_maturity(reference, eventual, np.array([top]))
    return np.asarray(tail * np.exp(-to_top[:, bins - 1]))


def speed_maturity(sc: e1.Scenario, counts: Array, reference: Array) -> Array:
    """One replication: maturity of each cohort (youngest first) from the frozen speed model."""
    k, d, n = sc.k, sc.d, sc.n
    grid = np.arange(reference.shape[1] + 1) / PER_YEAR

    def cohort(key: float, events: Array, *, mature: bool) -> onestage.Cohort:
        g = np.concatenate([[0.0], np.cumsum(events)]) / n
        return onestage.Cohort(
            key, grid[: len(g)].copy(), g, float(events.sum()), float(grid[-1]), mature
        )

    cohorts = [
        cohort(-float(d + k + e1.N_REFERENCE - 1 - i), reference[i], mature=True)
        for i in range(e1.N_REFERENCE)
    ] + [
        cohort(-float(d + j), counts[j, : PER_YEAR * (d + j)], mature=False)
        for j in range(k - 1, -1, -1)
    ]
    fit = onestage.select_model(cohorts, onestage.dispersion(cohorts)).fit
    s = fit.speeds[e1.N_REFERENCE :][::-1]
    ages = d + np.arange(k, dtype=float)
    seen, final = (
        np.interp(s * x, fit.ugrid, fit.f0, right=1.0) for x in (ages, np.full(k, grid[-1]))
    )
    return np.asarray(np.maximum(seen, 1e-9) / np.maximum(final, 1e-9))


def evaluate(job: tuple[tuple[object, ...], str, int, bool]) -> list[dict[str, object]]:
    values, path, reps, with_speed = job
    sc = e1.Scenario(*values)
    rng = np.random.default_rng([e1.SEED, zlib.crc32(repr((sc, path)).encode())])
    counts, reference, eventual = simulate(sc, path, reps, rng)
    events, n = counts.sum(axis=2), np.array(float(sc.n))
    ages = sc.d + np.arange(sc.k, dtype=float)
    bins = (PER_YEAR * ages).astype(int)
    maturity = {
        "true": e1.maturity(speeds(sc, path) * ages, sc),
        "common": e1.maturity(ages, sc),
        "reference": reference_maturity(reference, eventual, bins),
        "chain": chain_maturity(counts, reference, eventual, bins),
    }
    if with_speed:
        t_bins = PER_YEAR * T_REF
        maturity["speed"] = np.stack(
            [speed_maturity(sc, counts[r], reference[r, :, :t_bins]) for r in range(reps)]
        )
    out = {"naive_lepski": windows.naive_lepski(events, n)} | {
        f"weighted_{name}": windows.corrected_lepski(events, n, f) for name, f in maturity.items()
    }
    youngest = {
        name: float(np.mean(np.broadcast_to(f, events.shape)[:, 0])) for name, f in maturity.items()
    }
    return [
        {
            "shape": sc.shape,
            "median": sc.median,
            "p": sc.p,
            "drift": sc.drift,
            "n": sc.n,
            "k": sc.k,
            "d": sc.d,
            "speeds": path,
            "method": m,
            "rel_error": float(np.abs(est / sc.p - 1).mean()),
            "window": float(window.mean()),
            "maturity_youngest": youngest.get(m.removeprefix("weighted_"), np.nan),
        }
        for m, (window, est) in out.items()
    ]


def run(jobs: list[tuple[tuple[object, ...], str, int, bool]], name: str) -> pd.DataFrame:
    with mp.get_context("spawn").Pool(WORKERS) as pool:
        rows = [row for part in pool.imap_unordered(evaluate, jobs, chunksize=4) for row in part]
    (HERE / "artifacts").mkdir(exist_ok=True)
    table = pd.DataFrame(rows).sort_values(
        ["speeds", "method", "shape", "median", "p", "drift", "k", "d", "n"]
    )
    table.round(6).to_csv(HERE / "artifacts" / f"windows_{name}.csv", index=False)
    return table


def main(what: str) -> None:
    reduced = [
        (1.5, 7.0, 0.045, drift, n, 15, 1)
        for drift in ("none", "linear_up", "linear_down", "step")
        for n in e1.N_GRID
    ]
    if (
        what == "check"
    ):  # the existing rows, by the code of 0013 itself (exact) for comparison with path "equal"
        old = e1.full()
        old.round(6).to_csv(HERE / "artifacts" / "windows_check.csv", index=False)
        log.info("\n%s", (100 * old.groupby(["method", "n"]).rel_error.median().unstack()).round(1))
    elif what == "full":
        jobs = [(g, path, e1.REPS, False) for g in itertools.product(*GRID) for path in TOP_SPEED]
        t = run(jobs, "full")
        log.info(
            "\n%s",
            (100 * t.groupby(["speeds", "method", "n"]).rel_error.median().unstack()).round(1),
        )
    else:
        t = run([(g, path, SPEED_REPS, True) for g in reduced for path in TOP_SPEED], "speed")
        log.info(
            "\n%s",
            (100 * t.groupby(["speeds", "method", "n"]).rel_error.median().unstack()).round(1),
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    main(sys.argv[1])
