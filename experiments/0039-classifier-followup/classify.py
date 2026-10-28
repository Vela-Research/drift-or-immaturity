"""The classifier run of 0033 on the borrower-followed table at all 13 snapshots (POST HOC; round 4, item P0.4).

DATASET: the samples and snapshot files of experiments/0033 (Freddie Mac 200k, experiments/0010), version `linked`
(table of 0018). `run.snapshot` of 0033 is not changed; 0033 ran 3 of the 13 snapshots.
One checkpoint per snapshot in artifacts/parts, so a stopped run continues.
Usage: PYTHONHASHSEED=0 uv run python classify.py linked [workers; default 3]
"""

from __future__ import annotations

import os

for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_key] = "1"  # before numpy is loaded, in the workers too

import logging
import pickle
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Final

HERE: Final = Path(__file__).resolve().parent
PARTS: Final = HERE / "artifacts" / "parts"
sys.path.insert(0, str(HERE.parent / "0033-downstream-classifier"))
import run  # the classifier code of 0033

log = logging.getLogger(__name__)


def checkpointed(task: tuple[str, int]) -> str:
    mode, year = task
    result = run.snapshot((year, mode, "vintage"))
    (PARTS / f"{mode}_{year}.pkl").write_bytes(pickle.dumps(result))
    return f"{mode} {year}: {result[1]['seconds']} s"


def main(mode: str, workers: int) -> None:
    PARTS.mkdir(parents=True, exist_ok=True)
    todo = [(mode, y) for y in run.SNAPSHOTS if not (PARTS / f"{mode}_{y}.pkl").exists()]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for line in pool.map(checkpointed, todo):
            log.info(line)


if __name__ == "__main__":
    if os.environ.get("PYTHONHASHSEED") != "0":
        raise SystemExit("run with PYTHONHASHSEED=0 (as the runs of 0033)")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 3)
