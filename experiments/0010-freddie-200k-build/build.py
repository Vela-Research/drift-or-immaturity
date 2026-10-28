"""Build the 200k-per-vintage loan table from the full Freddie Mac Standard files (FROZEN_SPEC_200k.md section 1).

Stage A (origination): read EVERY origination file. Draw a simple random sample of 200,000 loans per origination year
  (all loans if fewer), fixed seed. Collect the set of HARP parent loan ids (Pre-HARP Loan Sequence Number of every
  relief loan, across all years).
Stage B (performance): stream every performance file once and keep the rows of the sampled loans; derive events with
  the same rules as the 50k build (experiments/0003-freddie-sample/build_loans.py).
HARP rule: a sampled loan that is a HARP parent and terminates by prepayment (zero-balance code 01) was refinanced into
  a relief loan; its termination is censoring, not prepayment (columns *_raw keep the unadjusted version).
Output: data/processed/freddie_200k_loans.parquet (same columns as freddie_sample_loans.parquet + is_harp_parent).
Usage: build.py [year ...]   (stages are checkpointed per year in data/processed/freddie_200k/)
"""

from __future__ import annotations

import logging
import sys
import zipfile
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv as pacsv

from maturity import data

HERE: Final = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "0003-freddie-sample"))
import build_loans as bl

log = logging.getLogger(__name__)
RAW: Final = data.DATA_DIR / "raw" / "freddie"
WORK: Final = data.DATA_DIR / "processed" / "freddie_200k"
OUT: Final = data.DATA_DIR / "processed" / "freddie_200k_loans.parquet"
N_SAMPLE: Final = 200_000
SEED: Final = 20261004
YEARS: Final = range(1999, 2027)
EVENT_TYPES: Final = ("event_type", "event_type90", "event_type_clean", "event_type_c202002")


def quarters(year: int) -> list[str]:
    with zipfile.ZipFile(RAW / f"historical_data_{year}.zip") as z:
        return sorted(n for n in z.namelist() if n.endswith(".zip"))


def open_inner(year: int, quarter_zip: str, prefix: str):
    outer = zipfile.ZipFile(RAW / f"historical_data_{year}.zip")
    inner = zipfile.ZipFile(outer.open(quarter_zip))
    name = next(n for n in inner.namelist() if n.startswith(prefix))
    return outer, inner, inner.open(name)


def stage_a(year: int, rng: np.random.Generator) -> set[str]:
    """Sample origination rows for one vintage; return the HARP parent ids named in this year's files.
    Two passes per quarter (ids first, then the sampled rows) so that a whole year is never held in memory."""
    qs = quarters(year)
    counts, parents = [], set()
    for q in qs:
        outer, inner, f = open_inner(year, q, "orig")
        ids = pd.read_csv(
            f, sep="|", header=None, usecols=[19, 25], names=["loan_id", "pre_harp_id"], dtype=str
        )
        for h in (f, inner, outer):
            h.close()
        counts.append(len(ids))
        parents |= set(ids.pre_harp_id.dropna())
    n = int(sum(counts))
    keep = np.sort(rng.choice(n, size=min(N_SAMPLE, n), replace=False))
    frames, start = [], 0
    for q, c in zip(qs, counts, strict=True):
        local = keep[(keep >= start) & (keep < start + c)] - start
        outer, inner, f = open_inner(year, q, "orig")
        df = pd.read_csv(f, sep="|", header=None, names=bl.ORIG_COLS, dtype=str)
        for h in (f, inner, outer):
            h.close()
        frames.append(df.iloc[local])
        start += c
    pd.concat(frames, ignore_index=True).to_parquet(WORK / f"orig_{year}.parquet")
    log.info("A %d: %d loans, sampled %d, parents named %d", year, n, len(keep), len(parents))
    return parents


def stage_b(year: int, pmms: pd.Series, parents: set[str]) -> None:
    orig = pd.read_parquet(WORK / f"orig_{year}.parquet")
    cols = [f"f{i}" for i in bl.PERF_IDX.values()]
    parts: list[pd.DataFrame] = []
    for q in quarters(year):
        ids = pa.array(
            orig.loan_id[orig.loan_id.str[1:5] == q[-10:-4][2:]].to_numpy(), type=pa.string()
        )
        if len(ids) == 0:
            continue
        outer, inner, f = open_inner(year, q, "perf")
        reader = pacsv.open_csv(
            f,
            read_options=pacsv.ReadOptions(autogenerate_column_names=True, block_size=1 << 27),
            parse_options=pacsv.ParseOptions(delimiter="|"),
            convert_options=pacsv.ConvertOptions(
                include_columns=cols, column_types={c: pa.string() for c in cols}
            ),
        )
        for batch in reader:
            hit = batch.filter(pc.is_in(batch.column("f0"), value_set=ids))
            if hit.num_rows:
                parts.append(hit.to_pandas())
        for h in (f, inner, outer):
            h.close()
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
    p = p.sort_values(["loan_id", "period"], ignore_index=True).drop(
        columns=["dq", "assist", "disaster", "deferral"]
    )
    ev = bl._events(p)
    out = _clean_orig(orig, pmms).merge(ev, on="loan_id", how="left")
    out["is_harp_parent"] = out.loan_id.isin(parents)
    for col in (
        EVENT_TYPES
    ):  # HARP parents: a payoff by refinancing into a relief loan is censoring, not prepayment
        out[f"{col}_raw"] = out[col]
        out[col] = np.where(out.is_harp_parent & (out[col] == "prepay"), "censored", out[col])
    out.to_parquet(WORK / f"loans_{year}.parquet")
    log.info(
        "B %d: loans=%d perf rows=%d default(clean)=%.2f%% prepay=%.2f%% harp parents=%d (%.1f%%)",
        year,
        len(out),
        len(p),
        100 * (out.event_type_clean == "default").mean(),
        100 * (out.event_type_clean == "prepay").mean(),
        int(out.is_harp_parent.sum()),
        100 * out.is_harp_parent.mean(),
    )


def _clean_orig(df: pd.DataFrame, pmms: pd.Series) -> pd.DataFrame:
    df = df.copy()
    for c in (
        "fico",
        "mi_pct",
        "units",
        "cltv",
        "dti",
        "upb",
        "ltv",
        "rate",
        "term",
        "n_borrowers",
    ):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    for c, s in bl.SENTINELS.items():
        df[c] = df[c].where(df[c] != s)
    yy = df["loan_id"].str[1:3].astype(int)
    df["orig_year"] = np.where(yy >= 99, 1900 + yy, 2000 + yy)
    df["orig_quarter"] = df["loan_id"].str[4].astype(int)
    prior_month = pd.to_datetime(df["first_payment"], format="%Y%m") - pd.DateOffset(months=1)
    df["pmms30"] = prior_month.dt.to_period("M").map(pmms).astype(float)
    df["rate_spread"] = df["rate"] - df["pmms30"]
    df["is_relief_refi"] = df["harp"].eq("Y")
    return df.drop(columns=["seller", "msa", "zip3", "maturity", "valuation_method", "vantage4"])


def main(years: list[int]) -> None:
    WORK.mkdir(parents=True, exist_ok=True)
    parents_path = WORK / "harp_parents.parquet"
    if parents_path.exists() and all((WORK / f"orig_{y}.parquet").exists() for y in YEARS):
        parents = set(pd.read_parquet(parents_path).loan_id)
    else:  # stage A always covers ALL years: the parent of a relief loan can be in any earlier vintage
        rng = np.random.default_rng(SEED)
        parents = set()
        for y in YEARS:
            parents |= stage_a(y, rng)
        pd.DataFrame({"loan_id": sorted(parents)}).to_parquet(parents_path)
    log.info("HARP parent ids: %d", len(parents))
    pmms = bl._pmms_monthly()
    for y in years:
        if not (WORK / f"loans_{y}.parquet").exists():
            stage_b(y, pmms, parents)
    if all((WORK / f"loans_{y}.parquet").exists() for y in YEARS):
        full = pd.concat(
            [pd.read_parquet(WORK / f"loans_{y}.parquet") for y in YEARS], ignore_index=True
        )
        full.to_parquet(OUT)
        log.info("wrote %s rows=%d", OUT, len(full))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    main([int(a) for a in sys.argv[1:]] or list(YEARS))
