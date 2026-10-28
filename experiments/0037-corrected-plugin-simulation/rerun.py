"""End-to-end simulation of experiments/0034 again with the corrected plug-in (round 4, item P0.1a). POST HOC.

DATASET: none (simulated). Same worlds as 0034: same seeds (`sim.SEED`, setting index, replication), same settings
and replication counts (7,200). The code of 0034 runs as it is (imported, not edited), with one function replaced:
`pipeline.increments` -> `corrected.increments` of experiments/0036 (the reference shape continued beyond its horizon
by the generalized gamma). The calibration snapshots use the corrected plug-in too. Speeds, speed model, calendar
step and the baselines are the frozen ones.

Where the simulation reads F0 beyond the horizon H:
  corrected      `sim.plugged` -> `pl.increments`: full method (both paths) and the own-speed plug-in (`own`)
  no change      chain ladder and calendar-adjusted chain ladder (speed 1: nothing is read beyond H)
  not corrected  the fall-back when guard G1 switches the calendar step off (`onestage._level`, cohort-level plug-in
                 with `right=1.0`); the run counts these snapshots per replication (`guard_off_snapshots`)
  not corrected  the shape likelihood that fits the speeds (`onestage._shape_loglik`): the speeds are the frozen ones
`increments` below calls the frozen function when no rescaled age lies beyond H (the same numbers; it saves a
generalized-gamma fit). `check` proves this and the harness.

Usage: PYTHONHASHSEED=0 uv run python rerun.py check | run   -> artifacts/parts/*.parquet (a stopped run continues)
"""

from __future__ import annotations

import logging
import multiprocessing as mp
import os
import sys
from pathlib import Path
from typing import Final

for _name in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMBA_NUM_THREADS",
):
    os.environ[_name] = (
        "1"  # one thread per process; set before numpy loads, and the workers inherit it
    )

import numpy as np
import pandas as pd

HERE: Final = Path(__file__).resolve().parent
EXP: Final = HERE.parent
FROZEN_PARTS: Final = EXP / "0034-end-to-end-simulation" / "artifacts" / "parts"
sys.path[:0] = [
    str(EXP / "0034-end-to-end-simulation"),
    str(EXP / "0036-corrected-plugin-backtests"),
]
import corrected  # the corrected plug-in (0036)
import sim  # the simulation of 0034; a done experiment, used as it is

log = logging.getLogger(__name__)
WORKERS: Final = 3
FROZEN: Final = sim.pl.increments
GUARD_OFF: Final[
    list[int]
] = []  # snapshots of the running replication where G1 switched the calendar step off
# columns that the corrected plug-in can change; every other column must be equal to the frozen run
PLUG_IN: Final = ("ours", "ours_uncond", "own")
INTERVAL: Final = tuple(f"{m}_{x}" for m in ("ours", "ours_uncond") for x in ("low", "high"))
_estimate: Final = sim.estimate


def increments(
    curve: np.ndarray, a: int, speed: float, ugrid: np.ndarray, f0: np.ndarray, horizon: int
) -> np.ndarray:
    """The corrected plug-in; the frozen function when no rescaled age s * a lies beyond the reference horizon."""
    if speed * horizon <= ugrid[-1] / 4.0:
        return np.asarray(FROZEN(curve, a, speed, ugrid, f0, horizon))
    return corrected.increments(curve, a, speed, ugrid, f0, horizon)


PLUG_INS: Final = {
    "frozen": FROZEN,
    "corrected": increments,
    "corrected_pure": corrected.increments,
}


def estimate(w: sim.World, year: int, per: int) -> tuple[pd.DataFrame, dict[str, object]]:
    t, meta = _estimate(w, year, per)
    if not meta["calendar_on"]:
        GUARD_OFF.append(year)
    return t, meta


def replicate(task: tuple[int, int, str]) -> tuple[pd.DataFrame, dict[str, object]]:
    """One replication of 0034 (setting index, replication) with the named plug-in."""
    index, rep, plug_in = task
    sim.pl.increments = PLUG_INS[plug_in]
    GUARD_OFF.clear()
    setting = sim.SETTINGS[index]
    pairs, meta = sim.replicate(setting, rep, intervals=rep < setting.reps[1])
    return pairs, meta | {"guard_off_snapshots": len(GUARD_OFF)}


def job(task: tuple[int, int]) -> tuple[pd.DataFrame, dict[str, object]]:
    return replicate((*task, "corrected"))


sim.estimate = estimate
sim._job = job
sim.PARTS = HERE / "artifacts" / "parts"
sim.WORKERS = WORKERS


def frozen_pairs() -> pd.DataFrame:
    """The scored cohorts of the frozen run (stored parts of 0034)."""
    return pd.concat(
        [pd.read_parquet(f) for f in sorted(FROZEN_PARTS.glob("pairs_*.parquet"))]
    ).sort_values(["setting", "rep", "cohort"], ignore_index=True)


def assert_equal(a: pd.DataFrame, b: pd.DataFrame, columns: list[str], what: str) -> None:
    for c in columns:
        x, y = a[c].to_numpy(dtype=float), b[c].to_numpy(dtype=float)
        assert np.allclose(x, y, rtol=1e-9, atol=1e-12, equal_nan=True), f"{what}: {c} differs"


def compare_with_frozen(pairs: pd.DataFrame, frozen: pd.DataFrame, what: str) -> dict[str, int]:
    """The contract of the corrected version against the frozen run on the same replications: nothing changes but
    the plug-ins with a speed above 1 (and the intervals of the full method, through the calibration errors)."""
    assert len(pairs) == len(frozen) and (pairs.cohort.to_numpy() == frozen.cohort.to_numpy()).all()
    numeric = [c for c in frozen.columns if c not in ("setting", "mature", "young")]
    unchanged = [c for c in numeric if c not in PLUG_IN + INTERVAL and frozen[c].dtype != object]
    assert_equal(pairs, frozen, unchanged, what)
    out = {}
    for method, speed in (("ours", "speed"), ("ours_uncond", "speed"), ("own", "own_speed")):
        inside = (frozen[speed] * sim.H <= sim.H).to_numpy()
        assert_equal(pairs[inside], frozen[inside], [method], f"{what}, speed <= 1")
        out[f"{method}_changed"] = int(
            (~np.isclose(pairs[method], frozen[method], rtol=1e-9, atol=1e-12)).sum()
        )
        out[f"{method}_speed_above_1"] = int((~inside & frozen.young.to_numpy()).sum())
        out[f"{method}_lower_than_frozen"] = int(
            (pairs[method].to_numpy() < frozen[method].to_numpy() * (1 - 1e-9)).sum()
        )
    return out


def check() -> None:
    """Replication 0 of each of the 18 settings (with calibration snapshots), three times:
    frozen          must give the stored parts of 0034 (the harness runs the code of 0034);
    corrected_pure  `corrected.increments` at every call: cohorts with speed <= 1 keep the frozen estimate;
    corrected       the function of the run: equal to corrected_pure in every column."""
    stored = frozen_pairs()
    stored = stored[stored.rep == 0].reset_index(drop=True)
    tasks = [(i, 0, name) for name in PLUG_INS for i in range(len(sim.SETTINGS))]
    with mp.get_context("spawn").Pool(WORKERS) as pool:
        results = pool.map(replicate, tasks, chunksize=1)
    by_plug_in = {
        name: pd.concat(
            [p for (_, _, n), (p, _) in zip(tasks, results, strict=True) if n == name]
        ).sort_values(["setting", "rep", "cohort"], ignore_index=True)
        for name in PLUG_INS
    }
    numeric = [c for c in stored.columns if c not in ("setting", "mature", "young")]
    assert_equal(by_plug_in["frozen"], stored, numeric, "frozen plug-in against the stored parts")
    log.info("frozen plug-in: %d scored cohorts equal to the stored parts of 0034", len(stored))
    counts = compare_with_frozen(by_plug_in["corrected_pure"], stored, "corrected_pure")
    log.info("corrected_pure against the stored parts: %s", counts)
    assert_equal(by_plug_in["corrected"], by_plug_in["corrected_pure"], numeric, "run function")
    log.info("the run's function is equal to corrected_pure in all %d columns", len(numeric))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    (HERE / "artifacts").mkdir(exist_ok=True)
    if sys.argv[1] == "check":
        check()
    else:
        sim.run()
