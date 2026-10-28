"""CONFIRM2 step 0.2: raw files -> parquet with the columns the spec uses (preregistration/CONFIRM2_SPEC.md sections 2-3).

DATASET: the raw files of download.py (artifacts/manifest.json has URL, date and sha256 of each). No names, street
addresses, ZIP codes, bank or lender fields are kept. Exclusions are NOT applied here: gate.py counts them, and the
method code applies them (`units` in data.py).
  artifacts/sba.parquet        one row per 7(a) loan of the four FOIA files
  artifacts/civil.parquet      one row per record of the FJC civil cumulative file
  artifacts/judges.parquet     one row per Article III appointment: court, commission, senior status, termination
  artifacts/covariates.parquet state x month: unemployment (BLS LAUS, SA) and FHFA HPI growth (log 12-month change)
Usage: uv run python build.py [--delete-raw]
"""

from __future__ import annotations

import logging
import shutil
import sys
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd

HERE: Final = Path(__file__).resolve().parent
ART: Final = HERE / "artifacts"
RAW: Final = ART / "raw"
sys.path.insert(0, str(HERE.parent / "0003-freddie-sample"))
import shock_overlay as so  # the covariate reader of the Freddie runs

log = logging.getLogger(__name__)
SBA_COLUMNS: Final = {
    "ApprovalDate": "approval_date",
    "ApprovalFY": "approval_fy",
    "BorrState": "state",
    "GrossApproval": "gross_approval",
    "SBAGuaranteedApproval": "sba_guaranteed",
    "ProcessingMethod": "subprogram",
    "InitialInterestRate": "interest_rate",
    "TermInMonths": "term_months",
    "NaicsCode": "naics",
    "FranchiseCode": "franchise_code",
    "BusinessType": "business_type",
    "BusinessAge": "business_age",
    "LoanStatus": "status",
    "PaidInFullDate": "paid_in_full_date",
    "ChargeOffDate": "charge_off_date",
    "RevolverStatus": "revolver",
    "JobsSupported": "jobs",
}
SBA_DATES: Final = ("approval_date", "paid_in_full_date", "charge_off_date")
CIVIL_COLUMNS: Final = [
    "CIRCUIT",
    "DISTRICT",
    "OFFICE",
    "DOCKET",
    "ORIGIN",
    "FILEDATE",
    "JURIS",
    "NOS",
    "JURY",
    "CLASSACT",
    "DEMANDED",
    "MDLDOCK",
    "TRANSORG",
    "TERMDATE",
    "DISP",
    "PROSE",
    "STATUSCD",
    "TAPEYEAR",
]  # PLT and DEF (party names) are not read
CIVIL_INTS: Final = (
    "CIRCUIT",
    "ORIGIN",
    "JURIS",
    "NOS",
    "CLASSACT",
    "DEMANDED",
    "TRANSORG",
    "DISP",
    "PROSE",
    "TAPEYEAR",
)
JUDGE_COLUMNS: Final = {
    "Court Type": "court_type",
    "Court Name": "court_name",
    "Appointment Title": "title",
    "Commission Date": "commission_date",
    "Senior Status Date": "senior_date",
    "Termination Date": "termination_date",
}  # no judge names or ids


def sba() -> pd.DataFrame:
    files = sorted((RAW / "sba").glob("FOIA_7a_*.csv"))
    assert len(files) == 4, files
    d = pd.concat(
        (pd.read_csv(f, usecols=list(SBA_COLUMNS), dtype=str) for f in files), ignore_index=True
    ).rename(columns=SBA_COLUMNS)
    for c in SBA_DATES:
        d[c] = pd.to_datetime(d[c], errors="coerce")
    for c in (
        "approval_fy",
        "gross_approval",
        "sba_guaranteed",
        "interest_rate",
        "term_months",
        "jobs",
    ):
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d["status"] = d.status.str.replace(" ", "", regex=False)  # "P I F" -> "PIF"
    return d


def civil() -> pd.DataFrame:
    """The cumulative civil file, read from the zip in chunks (the text file is 2 GB); -8 = missing."""
    chunks = []
    reader = pd.read_csv(
        RAW / "fjc" / "cv88on.zip",
        sep="\t",
        usecols=CIVIL_COLUMNS,
        dtype=str,
        encoding="latin-1",
        chunksize=1_000_000,
    )
    for chunk in reader:
        for c in CIVIL_INTS:
            chunk[c] = pd.to_numeric(chunk[c], errors="coerce").astype("Int32")
        for c in ("FILEDATE", "TERMDATE"):
            chunk[c] = pd.to_datetime(chunk[c], format="%m/%d/%Y", errors="coerce")
        for c in ("DISTRICT", "OFFICE", "JURY", "MDLDOCK", "STATUSCD"):
            chunk[c] = chunk[c].astype("category")
        chunks.append(chunk)
        log.info("civil: %d records", sum(len(c) for c in chunks))
    d = pd.concat(chunks, ignore_index=True)
    for c in ("DISTRICT", "OFFICE", "JURY", "MDLDOCK", "STATUSCD"):
        d[c] = d[c].astype(str).astype("category")
    return d.rename(columns=str.lower)


def judges() -> pd.DataFrame:
    d = pd.read_csv(
        RAW / "fjc" / "federal-judicial-service.csv", usecols=list(JUDGE_COLUMNS), dtype=str
    )
    d = d.rename(columns=JUDGE_COLUMNS)
    for c in ("commission_date", "senior_date", "termination_date"):
        d[c] = pd.to_datetime(d[c], errors="coerce")
    return d


def covariates() -> pd.DataFrame:
    """State x month (0-based month index): unemployment and the log 12-month change of the HPI (quarterly values
    carried forward monthly), with the reader of the Freddie runs."""
    so.COV = RAW / "cov"
    hpi, unemp = so._covariates()
    hpi = hpi.dropna(subset=["hpi_yoy"])
    monthly = hpi.loc[hpi.index.repeat(3), ["state", "qi", "hpi_yoy"]].copy()
    monthly["mi"] = monthly.qi * 3 + np.tile(np.arange(3), len(hpi))
    return unemp.merge(monthly[["state", "mi", "hpi_yoy"]], on=["state", "mi"], how="outer")


def main(*, delete_raw: bool) -> None:
    for name, make in (
        ("sba", sba),
        ("covariates", covariates),
        ("judges", judges),
        ("civil", civil),
    ):
        if (ART / f"{name}.parquet").exists():
            continue
        frame = make()
        frame.to_parquet(ART / f"{name}.parquet")
        log.info("%s: %d rows", name, len(frame))
    if delete_raw:
        shutil.rmtree(RAW)
        log.info("raw files deleted")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    main(delete_raw="--delete-raw" in sys.argv)
