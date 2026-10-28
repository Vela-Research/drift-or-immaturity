"""Window rule: the feasible row for `tab:e1` (round 4, item P0.5). POST HOC. DATASET: none (simulated).

The window simulation of experiments/0034 `windows_row.py` (imported, not edited) on the reduced grid of 0013
(Weibull shape 1.5, median 7 years, p 4.5%, K 15, d 1; 4 drift settings x 4 sample sizes x 3 speed paths) with 200
replications per setting (0034: 60; the draws are new, because the generator draws all replications at once).
Maturity of each cohort for the maturity-weighted Lepski rule (kappa = z):
  speed            the frozen speed model: F0(s age) / F0(s x 40 years), F0 = the pooled reference (as in 0034)
  speed_corrected  the same speeds, with the reference continued beyond its horizon of 40 years by the generalized
                   gamma (`onestage.extended_reference`, the corrected code of round 4)
next to the rows of 0034 on the same replications: true, common, reference, chain.
`windows_row.evaluate` runs twice per setting: once as it is (the speed-model fits are recorded), once with
`speed_maturity` replaced by the corrected maturity from the recorded fits. The full 972-setting grid is not run: one
speed-model fit takes about 0.5 s, and the grid has 11,664 settings.

Usage: PYTHONHASHSEED=0 uv run python windows_feasible.py -> artifacts/windows_feasible.csv (a stopped run continues)
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

from maturity import onestage

HERE: Final = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "0034-end-to-end-simulation"))
import windows_row as wr  # of 0034; it imports the E1 simulation of 0013 as `wr.e1`

log = logging.getLogger(__name__)
WORKERS: Final = 3
REPS: Final = 200
PARTS: Final = HERE / "artifacts" / "windows_parts"
REDUCED: Final = [
    (1.5, 7.0, 0.045, drift, n, 15, 1)
    for drift in ("none", "linear_up", "linear_down", "step")
    for n in wr.e1.N_GRID
]
JOBS: Final = [(values, path) for values in REDUCED for path in wr.TOP_SPEED]
FITS: Final[
    list[onestage.Fit]
] = []  # the speed-model fits of the running setting, in the order of the replications
_select: Final = onestage.select_model
_speed_maturity: Final = wr.speed_maturity


def recording(cohorts: list[onestage.Cohort], phi: float = 1.0) -> onestage.Selection:
    selection = _select(cohorts, phi)
    FITS.append(selection.fit)
    return selection


def corrected_maturity(sc: wr.e1.Scenario, counts: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """`windows_row.speed_maturity` for the next recorded fit, with the reference continued beyond its horizon."""
    fit = FITS.pop(0)
    s = fit.speeds[wr.e1.N_REFERENCE :][::-1]
    ages = sc.d + np.arange(sc.k, dtype=float)
    f0 = onestage.extended_reference(fit.ugrid, fit.f0)
    final = f0(s * (reference.shape[1] / wr.PER_YEAR))
    return np.asarray(np.maximum(f0(s * ages), 1e-9) / np.maximum(final, 1e-9))


def evaluate(index: int) -> int:
    """One setting: the rows of 0034 with the frozen speed model, plus the row with the corrected reference."""
    values, path = JOBS[index]
    FITS.clear()
    onestage.select_model, wr.speed_maturity = recording, _speed_maturity
    rows = wr.evaluate((values, path, REPS, True))
    assert len(FITS) == REPS
    onestage.select_model, wr.speed_maturity = _select, corrected_maturity
    again = wr.evaluate((values, path, REPS, True))
    assert not FITS
    for old, new in zip(
        rows, again, strict=True
    ):  # the same replications: every other method is unchanged
        assert old["method"] == new["method"]
        if old["method"] != "weighted_speed":
            assert old["rel_error"] == new["rel_error"]
    rows += [
        r | {"method": "weighted_speed_corrected"} for r in again if r["method"] == "weighted_speed"
    ]
    pd.DataFrame(rows).assign(replications=REPS).to_csv(PARTS / f"{index:02d}.csv", index=False)
    return index


def main() -> None:
    PARTS.mkdir(parents=True, exist_ok=True)
    todo = [i for i in range(len(JOBS)) if not (PARTS / f"{i:02d}.csv").exists()]
    with mp.get_context("spawn").Pool(WORKERS) as pool:
        for k, index in enumerate(pool.imap_unordered(evaluate, todo), 1):
            log.info("setting %d done (%d of %d)", index, k, len(todo))
    table = pd.concat(pd.read_csv(f) for f in sorted(PARTS.glob("*.csv"))).sort_values(
        ["speeds", "method", "drift", "n"]
    )
    table.round(6).to_csv(HERE / "artifacts" / "windows_feasible.csv", index=False)
    log.info(
        "\n%s",
        (100 * table.groupby(["speeds", "method", "n"]).rel_error.median().unstack()).round(1),
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    main()
