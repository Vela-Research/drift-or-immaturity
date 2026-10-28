"""Follow each HARP parent of the 200k sample into its relief loan (2026-10-04; "post hoc, follows borrowers").

DATASET: 200k loans per vintage, simple random sample of the full Freddie Mac Standard files (experiments/0010),
plus the origination and performance records of the relief loans of the sampled HARP parents (full files 2008-2020).
A HARP parent that ends by refinancing into a relief loan continues as that loan: its event is the relief loan's
event, at the age counted from the PARENT's first payment (default = clean 180+ DPD or credit termination;
prepayment; else censored). No relief loan in the sample was refinanced by HARP again, so one step is enough.
All other loans are unchanged. Relief loans also stay in their own vintages, as in every other version.
Output: data/processed/freddie_200k_loans_linked.parquet (event_*_clean replaced for the linked parents) and
artifacts/hazard_cells[_mtm]_linked_subdist.parquet (cell logic of experiments/0012).
Usage: PYTHONHASHSEED=0 uv run python link.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv as pacsv

from maturity import data

HERE: Final = Path(__file__).resolve().parent
sys.path[:0] = [
    str(HERE.parent / "0012-freddie-200k-harp-correction"),
    str(HERE.parent / "0010-freddie-200k-build"),
    str(HERE.parent / "0003-freddie-sample"),
]
import build  # zip access to the full Standard files
import build_loans as bl
import cells as ipcw  # IPCW sub-distribution cells (T2c section 10)
import shock_overlay as so

log = logging.getLogger(__name__)
LOANS: Final = data.DATA_DIR / "processed" / "freddie_200k_loans.parquet"
LINKED: Final = data.DATA_DIR / "processed" / "freddie_200k_loans_linked.parquet"
CHILDREN: Final = build.WORK / "harp_children_of_sample.parquet"
CHILD_EVENTS: Final = build.WORK / "harp_children_events.parquet"
CHILD_YEARS: Final = range(2008, 2021)


def month_idx(yyyymm: pd.Series) -> pd.Series:
    return (yyyymm // 100) * 12 + yyyymm % 100


def find_children(parents: set[str]) -> pd.DataFrame:
    """Relief loans whose named parent is a sampled HARP parent: loan_id, pre_harp_id, file year."""
    if CHILDREN.exists():
        return pd.read_parquet(CHILDREN)
    cols = {bl.ORIG_COLS.index(c): c for c in ("loan_id", "pre_harp_id")}
    frames = []
    for year in CHILD_YEARS:
        for q in build.quarters(year):
            outer, inner, f = build.open_inner(year, q, "orig")
            t = pd.read_csv(f, sep="|", header=None, usecols=list(cols), dtype=str).rename(
                columns=cols
            )
            for h in (f, inner, outer):
                h.close()
            frames.append(t[t.pre_harp_id.isin(parents)].assign(year=year, quarter=q))
        log.info("origination files %d: %d children so far", year, sum(len(x) for x in frames))
    out = pd.concat(frames, ignore_index=True)
    out.to_parquet(CHILDREN)
    return out


def child_events(children: pd.DataFrame) -> pd.DataFrame:
    """Event of each child from its performance records (same rules as the 200k build)."""
    if CHILD_EVENTS.exists():
        return pd.read_parquet(CHILD_EVENTS)
    cols = [f"f{i}" for i in bl.PERF_IDX.values()]
    out = []
    for (year, q), g in children.groupby(["year", "quarter"]):
        ids = pa.array(g.loan_id.to_numpy(), type=pa.string())
        outer, inner, f = build.open_inner(int(year), q, "perf")
        reader = pacsv.open_csv(
            f,
            read_options=pacsv.ReadOptions(autogenerate_column_names=True, block_size=1 << 27),
            parse_options=pacsv.ParseOptions(delimiter="|"),
            convert_options=pacsv.ConvertOptions(
                include_columns=cols, column_types={c: pa.string() for c in cols}
            ),
        )
        parts = [
            hit.to_pandas()
            for batch in reader
            if (hit := batch.filter(pc.is_in(batch.column("f0"), value_set=ids))).num_rows
        ]
        for h in (f, inner, outer):
            h.close()
        if not parts:
            continue
        p = pd.concat(parts, ignore_index=True).rename(
            columns={f"f{i}": k for k, i in bl.PERF_IDX.items()}
        )
        p["period"] = p["period"].astype("int32")
        p["age"] = pd.to_numeric(p["age"], errors="coerce").astype("float32")
        dq = pd.to_numeric(p["dq"], errors="coerce").fillna(-1)
        p["dq_num"] = np.where(p["dq"].eq("RA"), 99, dq).astype("int16")
        p["assist_flag"] = (
            p["assist"].isin(bl.ASSIST_FLAGS) | p["disaster"].eq("Y") | p["deferral"].eq("Y")
        )
        p = p.sort_values(["loan_id", "period"], ignore_index=True)
        out.append(bl._events(p)[["loan_id", "event_type_clean", "event_period_clean"]])
        log.info("performance %s: %d children", q[-10:-4], len(out[-1]))
    events = pd.concat(out, ignore_index=True)
    events.to_parquet(CHILD_EVENTS)
    return events


def linked_loans() -> pd.DataFrame:
    d = pd.read_parquet(LOANS)
    refinanced = d.is_harp_parent & (d.event_type_clean_raw == "prepay")
    children = find_children(set(d.loan_id[refinanced]))
    events = child_events(children).merge(children[["loan_id", "pre_harp_id"]], on="loan_id")
    events = events.drop_duplicates("pre_harp_id").set_index("pre_harp_id")
    kind = d.loan_id.map(events.event_type_clean)
    period = d.loan_id.map(events.event_period_clean)
    found = refinanced & kind.notna()
    age = month_idx(period) - month_idx(d.first_payment.astype(int)) + 1
    d["is_linked"] = found
    d["event_type_clean"] = np.where(found, kind, d.event_type_clean)
    d["event_period_clean"] = np.where(found, period, d.event_period_clean)
    d["event_age_clean"] = np.where(found, age, d.event_age_clean)
    log.info(
        "HARP parents refinanced %d, linked %d (%.1f%%); linked outcomes %s",
        int(refinanced.sum()),
        int(found.sum()),
        100 * found.sum() / refinanced.sum(),
        d.event_type_clean[found].value_counts(normalize=True).round(4).to_dict(),
    )
    d.to_parquet(LINKED)
    return d


def main() -> None:
    d = linked_loans()
    d = d[d.orig_year.isin(so.SHOW_YEARS) & d.event_age_clean.notna() & d.state.notna()]
    hpi, unemp = so._covariates()
    for mtm in (False, True):
        x = d[d.ltv.notna()].copy() if mtm else d
        if mtm:
            x["ltv_band"] = pd.cut(x.ltv, so.LTV_EDGES, labels=False).astype(int)
        c = ipcw.cells(x, hpi, unemp, "event_type_clean", mtm)
        c.to_parquet(
            HERE / "artifacts" / f"hazard_cells{'_mtm' if mtm else ''}_linked_subdist.parquet"
        )
        log.info("cells mtm=%s: %d cells, %d events", mtm, len(c), c.events.sum())


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    (HERE / "artifacts").mkdir(exist_ok=True)
    main()
