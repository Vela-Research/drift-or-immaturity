"""Frozen-spec evaluation on the borrower-followed data (post hoc, 2026-10-04).

Methods unchanged: this runs the code of experiments/0012 (pipeline.py and run.py: corrected calendar step, all
baselines, P1, P3, P4, secondaries) as it is, pointed at the linked loan table and cells of link.py. Truth and inputs
use the same data definition: a HARP parent continues as its relief loan, so there is no HARP censoring.
Usage: link.py first, then PYTHONHASHSEED=0 uv run python evaluate.py -> results_linked.json, artifacts/linked/*
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Final

from maturity import data

HERE: Final = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "0012-freddie-200k-harp-correction"))
import pipeline as pl
import run as frozen  # the evaluation script of 0012

if __name__ == "__main__":
    if os.environ.get("PYTHONHASHSEED") != "0":
        raise SystemExit(
            "run with PYTHONHASHSEED=0 (the fixed-effects fit depends on set order, see 0011)"
        )
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    pl.LOANS_PATH = data.DATA_DIR / "processed" / "freddie_200k_loans_linked.parquet"
    pl.CELLS_DIR = HERE / "artifacts"
    frozen.VARIANT, frozen.HERE, frozen.OUT_DIR = "linked", HERE, HERE / "artifacts" / "linked"
    frozen.OUT_DIR.mkdir(parents=True, exist_ok=True)
    frozen.main()
