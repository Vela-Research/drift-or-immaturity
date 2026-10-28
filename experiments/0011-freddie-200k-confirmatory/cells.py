"""Sub-distribution hazard cells for the 200k sample (T2c section 8), base and MTM covariate sets.

DATASET: 200k loans per vintage, simple random sample of the full Freddie Mac Standard files (experiments/0010);
HARP parents' prepayments are censoring. Same cell logic as experiments/0003 (shock_overlay._cells). `w_adj` (used
only by the over-identification diagnostic) comes from one fit on the 1999-2012 vintages.
Usage: PYTHONHASHSEED=0 uv run python cells.py -> artifacts/hazard_cells[_mtm]_subdist.parquet
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd

from maturity import data

HERE: Final = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "0003-freddie-sample"))
import shock_overlay as so

log = logging.getLogger(__name__)
LOANS: Final = data.DATA_DIR / "processed" / "freddie_200k_loans.parquet"


def main() -> None:
    cols = ["orig_year", "first_payment", "state", "ltv", "event_age_clean", "event_type_clean"]
    d = pd.read_parquet(LOANS, columns=cols)
    d = d[d.orig_year.isin(so.SHOW_YEARS) & d.event_age_clean.notna() & d.state.notna()]
    hpi, unemp = so._covariates()
    for mtm in (False, True):
        x = d[d.ltv.notna()].copy() if mtm else d.copy()
        if mtm:
            x["ltv_band"] = pd.cut(x.ltv, so.LTV_EDGES, labels=False).astype(int)
        cells = so._cells(x, hpi, unemp, mtm, subdist=True)
        zcols = ["hpi_yoy_z", "unemp_z"] + (["log_mtm_z"] if mtm else [])
        beta, kappa, se = so._fit(cells[cells.vintage.isin(so.FIT_YEARS)], zcols)
        lin = cells[zcols].to_numpy() @ beta + cells.state.map(kappa).fillna(0.0).to_numpy()
        cells["w_adj"] = cells.exposure * np.exp(lin)
        cells.to_parquet(
            HERE / "artifacts" / f"hazard_cells{'_mtm' if mtm else ''}_subdist.parquet"
        )
        log.info(
            "mtm=%s loans=%d cells=%d events=%d beta=%s se=%s",
            mtm,
            len(x),
            len(cells),
            cells.events.sum(),
            beta.round(3),
            se.round(3),
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    main()
