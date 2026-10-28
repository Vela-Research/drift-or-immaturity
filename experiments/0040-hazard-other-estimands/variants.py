"""The cause-specific hazard model of experiments/0032 under two other estimands (POST HOC, round 4, P1a).

DATASETS: Freddie Mac 200k (experiments/0010). `noharp`: HARP refinances are prepayments in the truth and in the
model (event column `event_type_clean_raw`; pairs of the 0031 run `noharp`). `linked`: borrowers are followed into
their relief loans (loan table of 0018; pairs of the 0031 run `linked`). Covariates as in 0032.
The model, the cells, the fit and the product integral are the code of 0032 (imported, not copied).
Usage: PYTHONHASHSEED=0 uv run python variants.py <noharp|linked> -> artifacts/<variant>/artifacts/pairs_hazard.parquet
"""

from __future__ import annotations

import logging
import os
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Final

import pandas as pd

from maturity import data

HERE: Final = Path(__file__).resolve().parent
EXP: Final = HERE.parent
sys.path.insert(0, str(EXP / "0032-mortgage-hazard-baseline"))
import hazard as hz

PAIRS: Final = EXP / "0031-round3-backtests" / "artifacts"
EVENT: Final = "event_type_clean"


def reader(column: str) -> Callable[..., pd.DataFrame]:
    """`pd.read_parquet` that serves `column` under the name of the event column of 0032.
    ponytail: a shim over the loader of 0032, which is done and takes no parameter; give `build` an event-column
    argument if the model moves into `maturity`."""
    original = pd.read_parquet

    def read(path: Path, columns: list[str] | None = None) -> pd.DataFrame:
        if columns is None or EVENT not in columns:
            return original(path, columns=columns)
        swapped = [column if c == EVENT else c for c in columns]
        return original(path, columns=swapped).rename(columns={column: EVENT})

    return read


def main(variant: str) -> None:
    hz.HERE = HERE / "artifacts" / variant  # 0032 writes to HERE / "artifacts"
    hz.PAIRS = PAIRS / f"pairs_freddie_{variant}.parquet"
    if variant == "linked":
        hz.LOANS = data.DATA_DIR / "processed" / "freddie_200k_loans_linked.parquet"
    else:
        pd.read_parquet = reader("event_type_clean_raw")
    hz.main()


if __name__ == "__main__":
    if os.environ.get("PYTHONHASHSEED") != "0":
        raise SystemExit("run with PYTHONHASHSEED=0 (the fixed-effects fit depends on set order)")
    if sys.argv[1] not in ("noharp", "linked"):
        raise SystemExit("usage: variants.py <noharp|linked>")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    main(sys.argv[1])
