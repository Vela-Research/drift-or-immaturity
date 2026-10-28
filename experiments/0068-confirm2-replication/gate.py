"""CONFIRM2 step 0.3: gate checks of section 5 (structure only: no method runs, no event rates by cohort).

DATASET: the parquet files of build.py (SBA 7(a) FOIA as of 2026-06-30; FJC civil cumulative file to 2026-06-30;
FJC judicial service; BLS LAUS state unemployment (FRED copies) and FHFA state HPI).
Usage: uv run python gate.py -> gate.json (the gate report)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Final

import download  # STATES: the 50 states and DC
import numpy as np
import pandas as pd

HERE: Final = Path(__file__).resolve().parent
ART: Final = HERE / "artifacts"
log = logging.getLogger(__name__)
STATES: Final = frozenset(download.STATES)
# FJC district codes (civil codebook, field DISTRICT) -> (state or territory, district name of the judge directory)
DISTRICTS: Final = {
    "00": ("ME", "Maine"), "01": ("MA", "Massachusetts"), "02": ("NH", "New Hampshire"),
    "03": ("RI", "Rhode Island"), "04": ("PR", "Puerto Rico"), "05": ("CT", "Connecticut"),
    "06": ("NY", "Northern New York"), "07": ("NY", "Eastern New York"), "08": ("NY", "Southern New York"),
    "09": ("NY", "Western New York"), "10": ("VT", "Vermont"), "11": ("DE", "Delaware"),
    "12": ("NJ", "New Jersey"), "13": ("PA", "Eastern Pennsylvania"), "14": ("PA", "Middle Pennsylvania"),
    "15": ("PA", "Western Pennsylvania"), "16": ("MD", "Maryland"), "17": ("NC", "Eastern North Carolina"),
    "18": ("NC", "Middle North Carolina"), "19": ("NC", "Western North Carolina"),
    "20": ("SC", "South Carolina"), "22": ("VA", "Eastern Virginia"), "23": ("VA", "Western Virginia"),
    "24": ("WV", "Northern West Virginia"), "25": ("WV", "Southern West Virginia"),
    "26": ("AL", "Northern Alabama"), "27": ("AL", "Middle Alabama"), "28": ("AL", "Southern Alabama"),
    "29": ("FL", "Northern Florida"), "3A": ("FL", "Middle Florida"), "3C": ("FL", "Southern Florida"),
    "3E": ("GA", "Northern Georgia"), "3G": ("GA", "Middle Georgia"), "3J": ("GA", "Southern Georgia"),
    "3L": ("LA", "Eastern Louisiana"), "3N": ("LA", "Middle Louisiana"), "36": ("LA", "Western Louisiana"),
    "37": ("MS", "Northern Mississippi"), "38": ("MS", "Southern Mississippi"),
    "39": ("TX", "Northern Texas"), "40": ("TX", "Eastern Texas"), "41": ("TX", "Southern Texas"),
    "42": ("TX", "Western Texas"), "43": ("KY", "Eastern Kentucky"), "44": ("KY", "Western Kentucky"),
    "45": ("MI", "Eastern Michigan"), "46": ("MI", "Western Michigan"), "47": ("OH", "Northern Ohio"),
    "48": ("OH", "Southern Ohio"), "49": ("TN", "Eastern Tennessee"), "50": ("TN", "Middle Tennessee"),
    "51": ("TN", "Western Tennessee"), "52": ("IL", "Northern Illinois"), "53": ("IL", "Central Illinois"),
    "54": ("IL", "Southern Illinois"), "55": ("IN", "Northern Indiana"), "56": ("IN", "Southern Indiana"),
    "57": ("WI", "Eastern Wisconsin"), "58": ("WI", "Western Wisconsin"), "60": ("AR", "Eastern Arkansas"),
    "61": ("AR", "Western Arkansas"), "62": ("IA", "Northern Iowa"), "63": ("IA", "Southern Iowa"),
    "64": ("MN", "Minnesota"), "65": ("MO", "Eastern Missouri"), "66": ("MO", "Western Missouri"),
    "67": ("NE", "Nebraska"), "68": ("ND", "North Dakota"), "69": ("SD", "South Dakota"),
    "7-": ("AK", "Alaska"), "70": ("AZ", "Arizona"), "71": ("CA", "Northern California"),
    "72": ("CA", "Eastern California"), "73": ("CA", "Central California"), "74": ("CA", "Southern California"),
    "75": ("HI", "Hawaii"), "76": ("ID", "Idaho"), "77": ("MT", "Montana"), "78": ("NV", "Nevada"),
    "79": ("OR", "Oregon"), "80": ("WA", "Eastern Washington"), "81": ("WA", "Western Washington"),
    "82": ("CO", "Colorado"), "83": ("KS", "Kansas"), "84": ("NM", "New Mexico"),
    "85": ("OK", "Northern Oklahoma"), "86": ("OK", "Eastern Oklahoma"), "87": ("OK", "Western Oklahoma"),
    "88": ("UT", "Utah"), "89": ("WY", "Wyoming"), "90": ("DC", "Columbia"), "91": ("VI", "Virgin Islands"),
    "93": ("GU", "Guam"), "94": ("MP", "Northern Mariana Islands"),
}  # fmt: skip
PARTS: Final = ("Northern", "Eastern", "Southern", "Western", "Middle", "Central")
SBA_FEATURES: Final = [
    "gross_approval", "sba_guaranteed", "term_months", "naics", "business_type", "business_age",
    "franchise_code", "revolver", "jobs", "interest_rate", "subprogram", "state",
]  # fmt: skip
CIVIL_FEATURES: Final = [
    "nos",
    "juris",
    "origin",
    "classact",
    "jury",
    "demanded",
    "prose",
    "district",
]


def court_name(district: str) -> str:
    """Name of the district court in the FJC judge directory."""
    part, _, state = district.partition(" ")
    if part in PARTS:
        return f"U.S. District Court for the {part} District of {state}"
    return f"U.S. District Court for the District of {district}"


def month(dates: pd.Series) -> pd.Series:
    return dates.dt.year * 12 + dates.dt.month - 1


def sba() -> dict[str, object]:
    d = pd.read_parquet(ART / "sba.parquet")
    in_range = d.approval_date.between("1991-01-01", "2019-12-31")
    x = d[in_range]
    out: dict[str, object] = {
        "loans_in_file": len(d),
        "approved_1991_2019": int(in_range.sum()),
        "status_1991_2019": x.status.value_counts().to_dict(),
    }
    x = x[~x.status.isin(["CANCLD", "COMMIT"])]
    in_states = x.state.isin(STATES)
    out["share_with_state_in_50_plus_dc"] = round(float(in_states.mean()), 5)
    out["states_outside"] = x.state[~in_states].value_counts(dropna=False).to_dict()
    x = x[in_states]
    chgoff, pif = x.status == "CHGOFF", x.status == "PIF"
    out["chgoff_share_with_date"] = round(float(x.charge_off_date[chgoff].notna().mean()), 5)
    out["pif_share_with_date"] = round(float(x.paid_in_full_date[pif].notna().mean()), 5)
    missing = (chgoff & x.charge_off_date.isna()) | (pif & x.paid_in_full_date.isna())
    out["excluded_missing_matching_date"] = int(missing.sum())
    out["chgoff_with_pif_date"] = int((chgoff & x.paid_in_full_date.notna()).sum())
    out["pif_with_chgoff_date"] = int((pif & x.charge_off_date.notna()).sum())
    out["other_status_with_event_date"] = int(
        (~chgoff & ~pif & (x.charge_off_date.notna() | x.paid_in_full_date.notna())).sum()
    )
    x = x[~missing]
    event = np.where(chgoff[~missing], x.charge_off_date, x.paid_in_full_date)
    event = pd.Series(event, index=x.index)
    before = (event < x.approval_date) & event.notna()
    out["events_dated_before_approval"] = int(before.sum())
    out["events_dated_in_approval_month"] = int(
        (event.notna() & (month(event) == month(x.approval_date))).sum()
    )
    out["events_after_data_date"] = int((event > pd.Timestamp("2026-06-30")).sum())
    year = x.approval_date.dt.year
    per_year = year.value_counts().sort_index()
    out["loans_per_year_after_exclusions"] = {int(k): int(v) for k, v in per_year.items()}
    out["min_loans_per_year"] = int(per_year.min())
    out["share_missing_by_year"] = {
        c: {int(k): round(float(v), 4) for k, v in _missing(x[c]).groupby(year).mean().items()}
        for c in SBA_FEATURES
    }
    out["business_age_values"] = x.business_age.value_counts(dropna=False).head(12).to_dict()
    out["subprogram_values"] = x.subprogram.value_counts(dropna=False).to_dict()
    return out


def civil() -> dict[str, object]:
    d = pd.read_parquet(ART / "civil.parquet")
    out: dict[str, object] = {"records_in_file": len(d), "fields": list(d.columns)}
    out["disp_values"] = {str(k): int(v) for k, v in d.disp.value_counts(dropna=False).items()}
    in_range = d.filedate.between("1989-01-01", "2016-12-31")
    x = d[in_range]
    out["filed_1989_2016"] = int(in_range.sum())
    steps = {
        "federal question or diversity (JURIS 3, 4)": x.juris.isin([3, 4]),
        "not prisoner petitions (NOS 510-560)": ~x.nos.between(510, 560),
        "not social security (NOS 861-865)": ~x.nos.between(861, 865),
        "not transferred in (ORIGIN 5) or MDL transfer (ORIGIN 6)": ~x.origin.isin([5, 6]),
        "not a reopening or remand (ORIGIN 3, 4, 8-12)": ~x.origin.isin([3, 4, 8, 9, 10, 11, 12]),
    }
    keep = pd.Series(True, index=x.index)
    out["exclusions"] = {}
    for name, m in steps.items():
        m = m.fillna(False).astype(bool)
        out["exclusions"][name] = int((keep & ~m).sum())
        keep &= m
    x = x[keep]
    key = ["district", "office", "docket", "filedate"]
    dup = x.duplicated(key, keep="first")
    out["duplicate_records_same_district_office_docket_filedate"] = int(dup.sum())
    x = x[~dup]
    out["origin_values_kept"] = {
        str(k): int(v) for k, v in x.origin.value_counts(dropna=False).items()
    }
    per_year = x.filedate.dt.year.value_counts().sort_index()
    out["cases_per_year_after_exclusions"] = {int(k): int(v) for k, v in per_year.items()}
    out["min_cases_per_year"] = int(per_year.min())
    terminated = x.tapeyear.ne(2099).fillna(True)
    out["terminated_records"] = int(terminated.sum())
    out["pending_records"] = int((~terminated).sum())
    out["terminated_share_with_termdate"] = round(float(x.termdate[terminated].notna().mean()), 5)
    disp_known = x.disp[terminated].notna() & x.disp[terminated].ne(-8)
    out["terminated_share_with_disp"] = round(float(disp_known.mean()), 5)
    out["termdate_before_filedate"] = int((x.termdate < x.filedate).sum())
    districts = sorted(x.district.astype(str).unique())
    out["districts_in_data"] = len(districts)
    out["districts_not_in_codebook_map"] = [k for k in districts if k not in DISTRICTS]
    out["districts_not_a_state_or_dc"] = [
        f"{k} {DISTRICTS[k][1]}"
        for k in districts
        if k in DISTRICTS and DISTRICTS[k][0] not in STATES
    ]
    year = x.filedate.dt.year
    out["share_missing_by_year"] = {
        c: {
            int(k): round(float(v), 4)
            for k, v in _missing(x[c], minus8=True).groupby(year).mean().items()
        }
        for c in CIVIL_FEATURES
    }
    out["judges"] = judge_coverage(districts)
    return out


def judge_coverage(districts: list[str]) -> dict[str, object]:
    """Active non-senior Article III district judges per district and month, 1985-01 to 2019-12."""
    j = pd.read_parquet(ART / "judges.parquet")
    j = j[j.court_type == "U.S. District Court"]
    names = {court_name(DISTRICTS[k][1]): k for k in districts if k in DISTRICTS}
    j = j.assign(district=j.court_name.map(names))
    months = np.arange(1985 * 12, 2020 * 12)
    start = month(j.commission_date).to_numpy(dtype=float)
    end = np.fmin(
        month(j.senior_date).to_numpy(dtype=float), month(j.termination_date).to_numpy(dtype=float)
    )
    counts = {}
    for k in districts:
        m = (j.district == k).to_numpy()
        active = (start[m][:, None] <= months[None, :]) & ~(end[m][:, None] <= months[None, :])
        counts[k] = active.sum(axis=0)
    zero = {k: int((v == 0).sum()) for k, v in counts.items() if (v == 0).any()}
    return {
        "district_courts_in_directory": int(j.court_name.nunique()),
        "appointments_matched_to_a_district": int(j.district.notna().sum()),
        "districts_without_any_judge_record": [
            f"{k} {DISTRICTS[k][1]}"
            for k in districts
            if k in DISTRICTS and not (j.district == k).any()
        ],
        "districts_with_months_of_zero_active_judges": {
            f"{k} {DISTRICTS[k][1]}": v for k, v in zero.items()
        },
        "min_active_judges_over_districts_with_records": int(
            min(v.min() for k, v in counts.items() if (j.district == k).any())
        ),
    }


def covariates() -> dict[str, object]:
    c = pd.read_parquet(ART / "covariates.parquet")
    out = {}
    for name, (lo, hi) in {
        "sba": (1991 * 12, 2020 * 12 + 1),
        "civil": (1989 * 12, 2019 * 12 + 11),
    }.items():
        need = pd.MultiIndex.from_product(
            [sorted(STATES), range(lo, hi + 1)], names=["state", "mi"]
        )
        have = c.set_index(["state", "mi"])
        for col in ("unemp", "hpi_yoy") if name == "sba" else ("unemp",):
            got = have[col].dropna().reindex(need)
            out[f"{name} {col}: state-months missing of {len(need)}"] = int(got.isna().sum())
    out["unemp_range"] = [
        int(c.dropna(subset=["unemp"]).mi.min()),
        int(c.dropna(subset=["unemp"]).mi.max()),
    ]
    out["hpi_yoy_range"] = [
        int(c.dropna(subset=["hpi_yoy"]).mi.min()),
        int(c.dropna(subset=["hpi_yoy"]).mi.max()),
    ]
    out["territories_with_unemployment"] = sorted(set(c.state) - STATES)
    return out


def main() -> None:
    out = {"sba": sba(), "civil": civil(), "covariates": covariates()}
    (HERE / "gate.json").write_text(json.dumps(out, indent=1, default=str))
    log.info("written")


def _missing(s: pd.Series, *, minus8: bool = False) -> pd.Series:
    m = s.isna()
    if minus8:
        m |= s.astype(str).isin(["-8", "<NA>", "nan"])
    return m


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    main()
