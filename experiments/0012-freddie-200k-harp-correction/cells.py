"""Sub-distribution hazard cells with inverse-probability-of-censoring weights (T2c section 10; Fine & Gray 1999).

DATASET: 200k loans per vintage, simple random sample of the full Freddie Mac Standard files (experiments/0010).
POST HOC: this corrects the cells of experiments/0011, which treated HARP-parent censoring as administrative.

Per unit (vintage x first-payment month x state [x LTV band]), on the loan-age axis:
  - an active loan has weight 1; a censored loan leaves at its censoring age; a defaulted loan leaves;
  - a prepaid loan (prepaid at age T) stays at risk at age a > T with weight G(a - 1) / G(T), where G is the
    Kaplan-Meier survival of the censoring ages of the unit; no weight after the data cutoff.
  - `aj` = Aalen-Johansen default incidence of the unit up to that age (used by the plug-in).
G and `aj` at age a use only months up to the calendar month of that cell, so a snapshot sees no later data.
With administrative censoring only this is the cell logic of experiments/0003 (shock_overlay._cells).
Variants: ipcw = HARP parents censored (the spec); noharp = HARP terminations are prepayments (robustness check).
Usage: PYTHONHASHSEED=0 uv run python cells.py -> artifacts/hazard_cells[_mtm]_{ipcw,noharp}_subdist.parquet
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
AMAX: Final = so.AMAX
EVENT_COLUMN: Final = {"ipcw": "event_type_clean", "noharp": "event_type_clean_raw"}


def unit_arrays(
    dur: np.ndarray, kind: np.ndarray, cutoff_age: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(weighted exposure, default events, Aalen-Johansen default incidence) at ages 1..AMAX for one unit."""
    seen = dur <= AMAX  # later exits: the loan is active at every age in range
    d, p, c = (
        np.bincount(dur[seen & (kind == k)], minlength=AMAX + 1).astype(float)
        for k in ("default", "prepay", "censored")
    )
    active = len(dur) - np.concatenate(
        [[0.0], np.cumsum(d + p + c)[:-1]]
    )  # at the start of each age
    zeros = np.zeros(AMAX + 1)
    g = np.cumprod(1 - np.divide(c, active, out=zeros.copy(), where=active > 0))
    stays = np.cumsum(
        np.divide(p, g, out=zeros.copy(), where=p > 0)
    )  # sum of 1 / G(T) over prepaid loans
    exposure = active.copy()
    exposure[1:] += g[:-1] * stays[:-1]
    exposure[np.arange(AMAX + 1) > cutoff_age] = 0.0
    surv = np.cumprod(1 - np.divide(d + p, active, out=zeros.copy(), where=active > 0))
    aj = np.cumsum(
        np.concatenate([[1.0], surv[:-1]])
        * np.divide(d, active, out=zeros.copy(), where=active > 0)
    )
    return exposure[1:], d[1:], aj[1:]


def cells(
    d: pd.DataFrame, hpi: pd.DataFrame, unemp: pd.DataFrame, event_column: str, mtm: bool
) -> pd.DataFrame:
    fp = d.first_payment.astype(int)
    d = d.assign(
        fp_mi=(fp // 100) * 12 + fp % 100 - 1, dur=d.event_age_clean.clip(lower=1).astype(int)
    )
    keys = ["orig_year", "fp_mi", "state"] + (["ltv_band"] if mtm else [])
    ages = np.arange(1, AMAX + 1)
    out = []
    for key, g in d.groupby(keys, observed=True):
        exposure, events, aj = unit_arrays(
            g.dur.to_numpy(), g[event_column].to_numpy(), so.CUTOFF_MI - key[1] + 1
        )
        frame = pd.DataFrame(
            {
                "fp_mi": key[1],
                "vintage": key[0],
                "state": key[2],
                "age": ages,
                "mi": key[1] + ages - 1,
                "exposure": exposure,
                "events": events,
                "aj": aj,
            }
        )
        if mtm:
            frame["ltv_band"] = key[3]
        out.append(frame)
    c = pd.concat(out, ignore_index=True)
    c = c[c.exposure > 0].copy()
    c["qi"] = c.mi // 3
    c = c.merge(hpi, on=["state", "qi"], how="inner").merge(unemp, on=["state", "mi"], how="inner")
    if mtm:
        h0 = hpi.rename(columns={"hpi": "hpi_orig", "qi": "qi_orig"})[
            ["state", "qi_orig", "hpi_orig"]
        ]
        c["qi_orig"] = (c.fp_mi - 1) // 3
        c = c.merge(h0, on=["state", "qi_orig"], how="inner")
        c["log_mtm"] = np.log(so.LTV_MID[c.ltv_band.to_numpy()] * c.hpi_orig / c.hpi)
    raw = ["hpi_yoy", "unemp"] + (["log_mtm"] if mtm else [])
    c = c.dropna(subset=raw)
    for col in raw:
        c[f"{col}_z"] = (c[col] - c[col].mean()) / c[col].std()
    return c


def main() -> None:
    cols = ["orig_year", "first_payment", "state", "ltv", "event_age_clean", *EVENT_COLUMN.values()]
    d = pd.read_parquet(LOANS, columns=cols)
    d = d[d.orig_year.isin(so.SHOW_YEARS) & d.event_age_clean.notna() & d.state.notna()]
    hpi, unemp = so._covariates()
    for variant, event_column in EVENT_COLUMN.items():
        for mtm in (False, True):
            x = d[d.ltv.notna()].copy() if mtm else d
            if mtm:
                x["ltv_band"] = pd.cut(x.ltv, so.LTV_EDGES, labels=False).astype(int)
            c = cells(x, hpi, unemp, event_column, mtm)
            zcols = ["hpi_yoy_z", "unemp_z"] + (["log_mtm_z"] if mtm else [])
            beta, kappa, se = so._fit(c[c.vintage.isin(so.FIT_YEARS)], zcols)
            lin = c[zcols].to_numpy() @ beta + c.state.map(kappa).fillna(0.0).to_numpy()
            c["w_adj"] = c.exposure * np.exp(lin)  # for the over-identification diagnostic only
            c.to_parquet(
                HERE
                / "artifacts"
                / f"hazard_cells{'_mtm' if mtm else ''}_{variant}_subdist.parquet"
            )
            log.info(
                "%s mtm=%s loans=%d cells=%d events=%d beta=%s se=%s",
                variant,
                mtm,
                len(x),
                len(c),
                c.events.sum(),
                beta.round(3),
                se.round(3),
            )


def _self_check() -> None:
    """Administrative censoring only: a prepaid loan keeps weight 1 up to the cutoff. Random censoring: weight G."""
    dur = np.array([2, 3, 5, 5])
    exposure, events, aj = unit_arrays(
        dur, np.array(["prepay", "default", "censored", "censored"]), 5
    )
    assert exposure[:6].tolist() == [4, 4, 4, 3, 3, 0] and events[2] == 1, exposure[:6]
    assert abs(aj[4] - 0.25) < 1e-12
    # 4 loans: one prepays at age 1; censoring at age 2 (1 of 3 active) and age 4 (1 of 2): G = 2/3, then 1/3
    exposure, _, aj = unit_arrays(
        np.array([1, 2, 4, 4]), np.array(["prepay", "censored", "default", "censored"]), 10
    )
    assert np.allclose(exposure[:5], [4, 3 + 1, 2 + 2 / 3, 2 + 2 / 3, 1 / 3]), exposure[:5]
    assert (
        abs(aj[3] - (3 / 4) * (1 / 2)) < 1e-12
    )  # Aalen-Johansen: survival 3/4, then 1 default of 2 at risk


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    _self_check()
    main()
