"""Exploratory check (2026-10-04): is the censoring of HARP parents informative?

DATASET: 200k loans per vintage, simple random sample of the full Freddie Mac Standard files (experiments/0010),
plus one pass over the full origination files 1999-2010 to read the first-payment month and the original LTV of the
parents of the sampled relief loans.
The 200k sample holds a random sample of the HARP relief loans ("children", vintages 2009-2019). Each child names its
parent (Pre-Relief Refinance Loan Sequence Number). For each child:
  - HARP path: did the child default (clean 180+ DPD or credit termination) within h months?
  - at-risk path: the default incidence within h months, from the parent's age at the refinance, of the loans of the
    parent's vintage x state x original-LTV band that were still at risk at that age. Two versions:
      never_harp = loans that never had a HARP refinance (simple proportions);
      at_risk    = all loans, HARP parents censored at their refinance (Aalen-Johansen, conditional on survival).
Independent censoring (the assumption behind the inverse-probability weights and the Aalen-Johansen truth) says that
the HARP path and the at_risk path have the same incidence. The child has a new loan with a lower rate, so a
difference mixes selection (HARP needs a current loan) and the effect of the programme.
Usage: run.py -> results.json (parents' data cached in data/processed/freddie_200k/harp_child_parents.parquet)
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd

from maturity import data

HERE: Final = Path(__file__).resolve().parent
sys.path[:0] = [
    str(HERE.parent / "0010-freddie-200k-build"),
    str(HERE.parent / "0003-freddie-sample"),
]
import build  # zip access to the full Standard files
import build_loans as bl

log = logging.getLogger(__name__)
LOANS: Final = data.DATA_DIR / "processed" / "freddie_200k_loans.parquet"
CACHE: Final = build.WORK / "harp_child_parents.parquet"
PARENT_YEARS: Final = range(1999, 2011)
HORIZONS: Final = (36, 60)
LTV_EDGES: Final = [0, 70, 80, 90, 1000]
AMAX: Final = 330  # months
MIN_AT_RISK: Final = (
    30  # below this the state cell is replaced by the national cell of the vintage x LTV band
)


def month_idx(yyyymm: pd.Series) -> pd.Series:
    return (yyyymm // 100) * 12 + yyyymm % 100


def parents_of(ids: set[str]) -> pd.DataFrame:
    """First-payment month and original LTV of the named parents, from the full origination files."""
    if CACHE.exists():
        return pd.read_parquet(CACHE)
    cols = {bl.ORIG_COLS.index(c): c for c in ("first_payment", "ltv", "loan_id")}
    frames = []
    for year in PARENT_YEARS:
        for q in build.quarters(year):
            outer, inner, f = build.open_inner(year, q, "orig")
            t = pd.read_csv(f, sep="|", header=None, usecols=list(cols), dtype=str).rename(
                columns=cols
            )
            for h in (f, inner, outer):
                h.close()
            frames.append(t[t.loan_id.isin(ids)])
        log.info("origination files %d read", year)
    out = pd.concat(frames, ignore_index=True)
    out.to_parquet(CACHE)
    return out


def cell_tables(g: pd.DataFrame) -> dict[str, np.ndarray]:
    """Per cell, by age: at-risk counts and cumulative defaults of never-HARP loans, and the Aalen-Johansen default
    incidence and all-cause survival of all loans with HARP parents censored."""
    dur = np.clip(g.event_age_clean.to_numpy(dtype=int), 1, AMAX)
    never = ~g.is_harp_parent.to_numpy()
    raw, clean = g.event_type_clean_raw.to_numpy(), g.event_type_clean.to_numpy()
    exits = np.bincount(dur[never], minlength=AMAX + 1)
    defaults = np.bincount(dur[never & (raw == "default")], minlength=AMAX + 1)
    d, p, c = (
        np.bincount(dur[clean == k], minlength=AMAX + 1) for k in ("default", "prepay", "censored")
    )
    risk = np.maximum(len(dur) - np.concatenate([[0], np.cumsum(d + p + c)[:-1]]), 1)
    surv = np.cumprod(1 - (d + p) / risk)
    return {
        "active": never.sum() - np.cumsum(exits),  # still at risk after age a
        "cum_default": np.cumsum(defaults),
        "cif": np.cumsum(np.concatenate([[1.0], surv[:-1]]) * d / risk),
        "surv": surv,
    }


def matched_rates(children: pd.DataFrame, loans: pd.DataFrame, h: int) -> pd.DataFrame:
    """For each child: the two at-risk-path incidences over h months from the parent's age at the refinance."""
    state_cells = {k: cell_tables(g) for k, g in loans.groupby(["orig_year", "ltv_band", "state"])}
    national = {k: cell_tables(g) for k, g in loans.groupby(["orig_year", "ltv_band"])}
    never, at_risk = np.full(len(children), np.nan), np.full(len(children), np.nan)
    for i, (vintage, band, state, age) in enumerate(
        zip(
            children.parent_year,
            children.ltv_band,
            children.state,
            children.parent_age,
            strict=True,
        )
    ):
        cell = state_cells.get((vintage, band, state))
        if cell is None or cell["active"][age] < MIN_AT_RISK:
            cell = national[(vintage, band)]
        end = min(age + h, AMAX)
        if cell["active"][age] > 0:
            never[i] = (cell["cum_default"][end] - cell["cum_default"][age]) / cell["active"][age]
        if cell["surv"][age] > 0:
            at_risk[i] = (cell["cif"][end] - cell["cif"][age]) / cell["surv"][age]
    return pd.DataFrame({"never_harp": never, "at_risk": at_risk}, index=children.index)


def main() -> None:
    cols = [
        "loan_id",
        "orig_year",
        "first_payment",
        "state",
        "ltv",
        "is_harp_parent",
        "is_relief_refi",
    ]
    cols += ["pre_harp_id", "event_age_clean", "event_type_clean", "event_type_clean_raw"]
    d = pd.read_parquet(LOANS, columns=cols)
    d = d[d.event_age_clean.notna() & d.state.notna()]
    children = d[d.is_relief_refi & d.pre_harp_id.notna()].copy()
    parents = parents_of(set(children.pre_harp_id))
    parents["ltv"] = pd.to_numeric(parents.ltv, errors="coerce").where(lambda x: x < 999)
    children = children.merge(
        parents.rename(
            columns={"loan_id": "pre_harp_id", "first_payment": "parent_fp", "ltv": "parent_ltv"}
        ),
        on="pre_harp_id",
        how="inner",
    ).dropna(subset=["parent_ltv"])
    yy = children.pre_harp_id.str[1:3].astype(int)
    children["parent_year"] = np.where(yy >= 99, 1900 + yy, 2000 + yy)
    gap = month_idx(children.first_payment.astype(int)) - month_idx(children.parent_fp.astype(int))
    # the parent is paid off about two months before the child's first payment (checked on linked pairs below)
    children["parent_age"] = (gap - 1).clip(1, AMAX - max(HORIZONS)).astype(int)
    children["ltv_band"] = pd.cut(children.parent_ltv, LTV_EDGES, labels=False).astype(int)
    loans = d[d.orig_year.isin(children.parent_year.unique()) & d.ltv.notna()].copy()
    loans["ltv_band"] = pd.cut(loans.ltv, LTV_EDGES, labels=False).astype(int)
    pairs = children.merge(
        d[["loan_id", "event_age_clean"]],
        left_on="pre_harp_id",
        right_on="loan_id",
        suffixes=("", "_parent"),
    )
    out: dict[str, object] = {
        "children_in_sample": len(children),
        "linked_pairs_parent_also_sampled": len(pairs),
        "parent_age_check_median_difference_months": float(
            (pairs.parent_age - pairs.event_age_clean_parent).median()
        ),
        "parent_age_check_share_within_2_months": float(
            ((pairs.parent_age - pairs.event_age_clean_parent).abs() <= 2).mean()
        ),
    }
    children = children[children.parent_year.between(2003, 2009)]
    for h in HORIZONS:
        ok = ~((children.event_type_clean_raw == "censored") & (children.event_age_clean <= h))
        c = children[ok].copy()
        c["harp_path"] = ((c.event_type_clean == "default") & (c.event_age_clean <= h)).astype(
            float
        )
        c = c.join(matched_rates(c, loans, h)).dropna(subset=["never_harp", "at_risk"])
        rows = []
        for key, g in [*c.groupby("parent_year"), ("all 2003-2009", c)]:
            rows.append(
                {
                    "parent_vintage": str(key),
                    "children": len(g),
                    "median_parent_age_at_refinance": float(g.parent_age.median()),
                    "harp_path": round(float(g.harp_path.mean()), 5),
                    "at_risk_path": round(float(g.at_risk.mean()), 5),
                    "never_harp_path": round(float(g.never_harp.mean()), 5),
                    "ratio_harp_to_at_risk": round(float(g.harp_path.mean() / g.at_risk.mean()), 3),
                    "ratio_harp_to_never_harp": round(
                        float(g.harp_path.mean() / g.never_harp.mean()), 3
                    ),
                }
            )
        by_band = c.groupby("ltv_band")[["harp_path", "at_risk", "never_harp"]].mean().round(5)
        out[f"h{h}"] = {
            "excluded_censored_children": int((~ok).sum()),
            "by_vintage": rows,
            "by_ltv_band": by_band.to_dict("index"),
        }
        log.info("h = %d: %s", h, rows[-1])
    (HERE / "results.json").write_text(json.dumps(out, indent=1, default=str))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    main()
