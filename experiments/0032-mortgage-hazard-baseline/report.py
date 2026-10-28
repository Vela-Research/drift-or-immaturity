"""Hazard-model baseline against the chain ladders and the full method, with the declared test procedure.

DATASET: the pair file of hazard.py (Freddie Mac 200k, corrected weights; same pairs and truth as 0031).
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
# the realised-path model against the realised-path methods, the forecast-path model against the forecast-path ones
AGAINST: Final = {
    "real": ["naive", "chain", "stratified", "cal_chain", "ours"],
    "fcst": ["naive", "chain", "stratified", "cal_chain_forecast", "ours_uncond"],
}


def main() -> None:
    spec = importlib.util.spec_from_file_location("round3_report", ROUND3)
    assert spec is not None and spec.loader is not None
    round3 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(round3)
    p = pd.read_parquet(HERE / "artifacts" / "pairs_hazard.parquet")
    methods = round3.FROZEN + round3.NEW + HAZARD
    out = round3.summary(p, methods, "set")
    out["comparisons"] = {
        f"{b} vs {a}": round3.compare(p, b, a) for a in HAZARD for b in AGAINST[a.rsplit("_", 1)[1]]
    } | {
        "hazard_vintage_real vs hazard_real": round3.compare(
            p, "hazard_vintage_real", "hazard_real"
        ),
        "hazard_fcst vs hazard_real": round3.compare(p, "hazard_fcst", "hazard_real"),
    }
    out["coefficients"] = json.loads((HERE / "artifacts" / "coefficients.json").read_text())
    (HERE / "results.json").write_text(json.dumps(out, indent=1, default=str))


if __name__ == "__main__":
    main()
