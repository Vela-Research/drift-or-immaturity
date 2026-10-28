"""Hazard model under the two other estimands against the chain ladders and the full method (frozen code).

DATASET: pair files of variants.py (Freddie Mac 200k; pairs and truth of the 0031 runs `noharp` and `linked`).
Usage: uv run python report.py -> results.json
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Final

import pandas as pd

HERE: Final = Path(__file__).resolve().parent
ROUND3: Final = HERE.parent / "0031-round3-backtests" / "report.py"
HAZARD: Final = ["hazard_real", "hazard_fcst", "hazard_vintage_real", "hazard_vintage_fcst"]
AGAINST: Final = {
    "real": ["chain", "stratified", "cal_chain", "ours"],
    "fcst": ["chain", "stratified", "cal_chain_forecast", "ours_uncond"],
}


def main() -> None:
    spec = importlib.util.spec_from_file_location("round3_report", ROUND3)
    assert spec is not None and spec.loader is not None
    round3 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(round3)
    out = {}
    for variant in ("noharp", "linked"):
        p = pd.read_parquet(HERE / "artifacts" / variant / "artifacts" / "pairs_hazard.parquet")
        block = round3.summary(p, round3.FROZEN + round3.NEW + HAZARD, "set")
        block["comparisons"] = {
            f"{b} vs {a}": round3.compare(p, b, a)
            for a in HAZARD
            for b in AGAINST[a.rsplit("_", 1)[1]]
        }
        out[variant] = block
    (HERE / "results.json").write_text(json.dumps(out, indent=1, default=str))


if __name__ == "__main__":
    main()
