"""Build the loan-level table from Freddie Mac SFLLD zips (spec: the mortgage data specification).

Usage: uv run python experiments/0003-freddie-sample/build_loans.py [year ...]
Reads data/raw/freddie/sample_YYYY.zip, writes data/processed/freddie_sample_loans.parquet.
"""

from __future__ import annotations

import logging
import sys
import zipfile
from typing import Final

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.csv as pacsv

from maturity import data

log = logging.getLogger(__name__)
RAW: Final = data.DATA_DIR / "raw" / "freddie"
COV: Final = data.DATA_DIR / "raw" / "covariates"
OUT: Final = data.DATA_DIR / "processed" / "freddie_sample_loans.parquet"

ORIG_COLS: Final = [  # file_layout_july_2026.xlsx, "Origination Data File"
    "fico",
    "first_payment",
    "first_time_buyer",
    "maturity",
    "msa",
    "mi_pct",
    "units",
    "occupancy",
    "cltv",
    "dti",
    "upb",
    "ltv",
    "rate",
    "channel",
    "prepay_penalty",
    "amortization",
    "state",
    "property_type",
    "zip3",
    "loan_id",
    "purpose",
    "term",
    "n_borrowers",
    "seller",
    "super_conforming",
    "pre_harp_id",
    "special_eligibility",
    "harp",
    "valuation_method",
    "interest_only",
    "vantage4",
]
PERF_IDX: Final = {  # 0-based positions in the "Monthly Performance Data File"
    "loan_id": 0,
    "period": 1,
    "dq": 3,
    "age": 4,
    "zb": 8,
    "zb_date": 9,
    "deferral": 24,
    "disaster": 28,
    "assist": 29,
}
SENTINELS: Final = {
    "fico": 9999,
    "mi_pct": 999,
    "units": 99,
    "cltv": 999,
    "dti": 999,
    "ltv": 999,
    "n_borrowers": 99,
}
CREDIT_ZB: Final = {"02", "03", "09"}  # third-party sale, short sale/charge-off, REO disposition
ASSIST_FLAGS: Final = {
    "F",
    "R",
    "T",
    "O",
}  # forbearance, repayment, trial, other borrower assistance


def main(years: list[int]) -> None:
    pmms = _pmms_monthly()
    frames = [_year(y, pmms) for y in years]
    out = pd.concat(frames, ignore_index=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT)
    log.info("wrote %s rows=%d", OUT, len(out))


def _year(year: int, pmms: pd.Series) -> pd.DataFrame:
    orig = _orig(year, pmms)
    perf = _perf(year)
    ev = _events(perf)
    out = orig.merge(ev, on="loan_id", how="left")
    log.info(
        "%d: loans=%d perf rows=%d default=%.2f%% prepay=%.2f%%",
        year,
        len(out),
        len(perf),
        100 * (out.event_type == "default").mean(),
        100 * (out.event_type == "prepay").mean(),
    )
    return out


def _orig(year: int, pmms: pd.Series) -> pd.DataFrame:
    with zipfile.ZipFile(RAW / f"sample_{year}.zip") as z, z.open(f"sample_orig_{year}.txt") as f:
        df = pd.read_csv(f, sep="|", header=None, names=ORIG_COLS, dtype=str)
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
    for c, s in SENTINELS.items():
        df[c] = df[c].where(df[c] != s)
    yy = df["loan_id"].str[1:3].astype(int)
    df["orig_year"] = np.where(yy >= 99, 1900 + yy, 2000 + yy)
    df["orig_quarter"] = df["loan_id"].str[4].astype(int)
    prior_month = pd.to_datetime(df["first_payment"], format="%Y%m") - pd.DateOffset(months=1)
    df["pmms30"] = prior_month.dt.to_period("M").map(pmms).astype(float)
    df["rate_spread"] = df["rate"] - df["pmms30"]
    df["is_relief_refi"] = df["harp"].eq("Y")
    return df.drop(columns=["seller", "msa", "zip3", "maturity", "valuation_method", "vantage4"])


def _perf(year: int) -> pd.DataFrame:
    cols = [f"f{i}" for i in PERF_IDX.values()]
    parts: list[pd.DataFrame] = []
    with zipfile.ZipFile(RAW / f"sample_{year}.zip") as z, z.open(f"sample_perf_{year}.txt") as f:
        reader = pacsv.open_csv(
            f,
            read_options=pacsv.ReadOptions(autogenerate_column_names=True, block_size=1 << 26),
            parse_options=pacsv.ParseOptions(delimiter="|"),
            convert_options=pacsv.ConvertOptions(
                include_columns=cols, column_types={c: pa.string() for c in cols}
            ),
        )
        for batch in reader:
            parts.append(
                batch.to_pandas().rename(columns={f"f{i}": k for k, i in PERF_IDX.items()})
            )
    df = pd.concat(parts, ignore_index=True)
    df["period"] = df["period"].astype("int32")
    df["age"] = pd.to_numeric(df["age"], errors="coerce").astype("float32")
    dq = pd.to_numeric(df["dq"], errors="coerce").fillna(-1)
    df["dq_num"] = np.where(df["dq"].eq("RA"), 99, dq).astype("int16")  # RA = REO acquisition
    df["assist_flag"] = (
        df["assist"].isin(ASSIST_FLAGS) | df["disaster"].eq("Y") | df["deferral"].eq("Y")
    )
    return df.sort_values(["loan_id", "period"], ignore_index=True).drop(
        columns=["dq", "assist", "disaster", "deferral"]
    )


def _events(p: pd.DataFrame) -> pd.DataFrame:
    """One row per loan: primary (180+ / credit ZB) and sensitivity (90+) event, prepay, censoring."""
    last = p.drop_duplicates("loan_id", keep="last").set_index("loan_id")
    term = (
        p[p["zb"].notna() & (p["zb"] != "")]
        .drop_duplicates("loan_id", keep="first")
        .set_index("loan_id")
    )
    ev = pd.DataFrame(index=last.index)
    ev["last_period"], ev["last_age"] = last["period"], last["age"]
    ev["zb_code"], ev["zb_period"], ev["zb_age"] = term["zb"], term["period"], term["age"]
    for tag, thr, clean in (("", 6, False), ("90", 3, False), ("_clean", 6, True)):
        mask = p["dq_num"] >= thr
        if (
            clean
        ):  # decision: delinquency months flagged forbearance/disaster/deferral do not count
            mask &= ~p["assist_flag"]
        first = p[mask].drop_duplicates("loan_id", keep="first").set_index("loan_id")
        key = f"dq{thr}{'c' if clean else ''}"
        ev[f"{key}_period"], ev[f"{key}_age"] = first["period"], first["age"]
        ev[f"forb_at_{key}"] = first["assist_flag"]
        credit = ev["zb_code"].isin(CREDIT_ZB)
        by_zb = ev["zb_period"].where(credit)
        period = pd.concat([ev[f"{key}_period"], by_zb], axis=1).min(axis=1)
        from_dq = ev[f"{key}_period"].eq(period)
        age = ev[f"{key}_age"].where(from_dq, ev["zb_age"])
        is_def = period.notna()
        prepay = ev["zb_code"].eq("01") & ~is_def
        ev[f"event_type{tag}"] = np.where(is_def, "default", np.where(prepay, "prepay", "censored"))
        ev[f"event_period{tag}"] = np.where(
            is_def, period, np.where(prepay, ev["zb_period"], ev["last_period"])
        )
        ev[f"event_age{tag}"] = np.where(
            is_def, age, np.where(prepay, ev["zb_age"], ev["last_age"])
        )
    _censor_at(ev, 202002)
    return ev.reset_index()


def _months(period: pd.Series) -> pd.Series:
    return (period // 100) * 12 + period % 100


def _censor_at(ev: pd.DataFrame, cutoff: int) -> None:
    """Sensitivity: treat everything after `cutoff` as unobserved (events become censoring at the cutoff)."""
    late = ev["event_period"] > cutoff
    back = (_months(ev["event_period"]) - _months(pd.Series(cutoff, index=ev.index))).where(late, 0)
    ev["event_type_c202002"] = np.where(late, "censored", ev["event_type"])
    ev["event_age_c202002"] = ev["event_age"] - back
    ev["event_period_c202002"] = np.where(late, cutoff, ev["event_period"])


def _pmms_monthly() -> pd.Series:
    d = pd.read_csv(COV / "pmms.csv", usecols=["date", "pmms30"])
    d["date"] = pd.to_datetime(d["date"], format="%m/%d/%Y")
    return d.groupby(d["date"].dt.to_period("M"))["pmms30"].mean()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main([int(a) for a in sys.argv[1:]] or list(range(1999, 2027)))
