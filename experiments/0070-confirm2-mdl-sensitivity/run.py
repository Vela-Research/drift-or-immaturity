"""CONFIRM2, civil cases: POST HOC sensitivity without the MDL-docketed cases (decision 2 of
the decisions after the replication audit; audit 1, finding 36).

DATASET: the parquet files of exp 0068 (FJC civil cumulative file to 2026-06-30; FJC judicial service; BLS LAUS state
unemployment (FRED copies)), with the original-filing records of exp 0069 (C1). Aggregates only.
The primary analysis keeps the direct-filed MDL cases, as decision B3 was written. Here every case whose original
record has an MDL docket number (MDLDOCK a positive integer, whatever the ORIGIN; ORIGIN 13 is dropped already) is
removed, and the corrected pipeline of 0069 (C1-C6) runs on what remains: pairs and calibration, hazard model,
classifiers, windows, detectors and the report. SBA and the broad civil event are 0069's, linked unchanged.
Usage (from this folder, one at a time):
  uv run python run.py link
  PYTHONHASHSEED=0 uv run python run.py <backtest|hazard|classify|windows|drift> civil
  uv run python run.py report      -> results.json (the civil block is the sensitivity)
"""

from __future__ import annotations

import importlib.util
import logging
import sys
from pathlib import Path
from types import ModuleType
from typing import Final

import pandas as pd

HERE: Final = Path(__file__).resolve().parent
FIXES: Final = HERE.parent / "0069-confirm2-audit-fixes"
ART: Final = HERE / "artifacts"
log = logging.getLogger(__name__)


def corrected() -> ModuleType:
    """0069's runner (the 0068 code with C1-C6), with its paths pointed at this experiment."""
    spec = importlib.util.spec_from_file_location("run69", FIXES / "run.py")
    assert spec is not None and spec.loader is not None
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    m.ART, m.HERE = ART, HERE
    return m


def is_mdl(docket: pd.Series) -> pd.Series:
    """An MDL docket number: a positive integer (the other values are -8 = missing and free text)."""
    d = docket.astype("string").str.strip()
    return (d.str.fullmatch(r"\d+") & (pd.to_numeric(d, errors="coerce") > 0)).fillna(False)


def link(fixes: ModuleType) -> None:
    ART.mkdir(parents=True, exist_ok=True)
    for f in fixes.LINKED_INPUTS:
        if not (ART / f).exists():
            (ART / f).symlink_to(fixes.AS_RUN / "artifacts" / f)
    for name in ("sba", "civil_broad"):
        if not (ART / name).exists():
            (ART / name).symlink_to(FIXES / "artifacts" / name, target_is_directory=True)
    first = pd.read_parquet(FIXES / "artifacts" / "civil.parquet")
    mdl = is_mdl(first.mdldock)
    first[~mdl].to_parquet(ART / "civil.parquet")
    data = fixes.frozen_data()
    data.ART = FIXES / "artifacts"
    before = data.civil_units()
    data.ART = ART
    after = data.civil_units()
    log.info(
        "MDL-docketed first records: %d of %d; civil cases %d -> %d",
        int(mdl.sum()),
        len(first),
        len(before),
        len(after),
    )
    n, kept = before.orig_year.value_counts(), after.orig_year.value_counts()
    share = (1 - kept.reindex(n.index, fill_value=0) / n).sort_index().round(4)
    (ART / "removed_share_by_year.json").write_text(share.to_json(indent=1))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    fixes = corrected()
    if sys.argv[1] == "link":
        link(fixes)
    else:
        fixes.main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
