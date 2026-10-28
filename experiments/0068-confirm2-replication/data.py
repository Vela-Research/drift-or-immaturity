"""CONFIRM2: the units of spec sections 2-3 with the decided exclusions, the covariates, and the data object of the
frozen pipeline (experiments/0012 `pipeline.Data`, built as experiments/0021 `build` does for patents).

DATASET: the parquet files of build.py (SBA 7(a) FOIA as of 2026-06-30; FJC civil cumulative file to 2026-06-30; FJC
judicial service; BLS LAUS state unemployment (FRED copies) and FHFA state HPI). Decisions: the paper's table of deviations.
  A  SBA 7(a) loans approved 1991-2019 (not CANCLD or COMMIT, borrower in the 50 states + DC, matching date present,
     no event before approval). Event = charge-off, competing exit = paid in full. No outcome after 2020-02-29 is
     used: a later event is censored at 2020-02 (the truth cut of the spec); open loans are censored there too.
  B  civil cases filed 1989-2016: federal question or diversity, not NOS 510-560 or 861-865, ORIGIN 1 or 2, not in
     the 4 territorial districts, first record per (district, office, docket, filing date), no terminated case that
     ends before filing (a pending case, TAPEYEAR 2099, has the placeholder TERMDATE 1900-01-01 and is kept).
     Event = settled (DISP 13; `broad`: DISP 12 or 13), competing exit = any other termination, pending cases
     censored at the data date 2026-06.
Month indices as in the frozen code: `om` = entry month, 1-based (12 y + m); age a = months since entry; the cell of
age a has the 0-based calendar month index mi = om + a - 1. An event in the entry month counts at age 1.
"""

from __future__ import annotations

import dataclasses
import sys
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd

HERE: Final = Path(__file__).resolve().parent
EXP: Final = HERE.parent
ART: Final = HERE / "artifacts"
sys.path[:0] = [str(EXP / "0012-freddie-200k-harp-correction")]
import cells as ipcw  # unit_arrays of the frozen cells (sub-distribution risk sets, ages 1..120)
import gate  # DISTRICTS, STATES, court_name
import pipeline as pl  # the frozen pipeline

AMAX_CELLS: Final = ipcw.AMAX
CUT_A: Final = 2020 * 12 + 2  # 2020-02, 1-based: no SBA outcome after it
END_B: Final = 2026 * 12 + 6  # 2026-06, 1-based: data date of the civil file
H: Final = {"sba": 72, "civil": 36, "civil_broad": 36}
TERRITORIES: Final = ("04", "91", "93", "94")
RAW: Final = {
    "sba": ["hpi_yoy", "unemp"],
    "civil": ["log_judges", "unemp"],
    "civil_broad": ["log_judges", "unemp"],
}
SBA_FEATURES: Final = [
    "log_gross",
    "guaranteed_share",
    "term_months",
    "naics2",
    "business_type",
    "business_age",
    "franchise",
    "revolver",
    "log1p_jobs",
    "interest_rate",
    "subprogram",
    "state",
]
CIVIL_FEATURES: Final = [
    "nos",
    "juris",
    "origin",
    "classact",
    "jury",
    "log1p_demanded",
    "prose",
    "district",
]


def month(dates: pd.Series) -> np.ndarray:
    """1-based month index 12 y + m."""
    return (dates.dt.year * 12 + dates.dt.month).to_numpy()


def _outcome(units: pd.DataFrame, event_mi: np.ndarray, kind: np.ndarray, end: int) -> pd.DataFrame:
    """Columns of the frozen loan table: entry year, entry month, event month, age at the event, kind."""
    om = units.om.to_numpy()
    late = np.isnan(event_mi) | (event_mi > end)
    kind = np.where(late, "censored", kind)
    dur = np.where(late, end - om, np.maximum(np.nan_to_num(event_mi) - om, 1)).astype(int)
    return units.assign(
        ev_mi=om + dur,
        event_age_clean=dur,
        event_type_clean=kind,
        year=units.orig_year.astype(float),
        quarter=((om - 1) // 3) / 4.0,
    )


def sba_units() -> pd.DataFrame:
    d = pd.read_parquet(ART / "sba.parquet")
    d = d[d.approval_date.between("1991-01-01", "2019-12-31")]
    d = d[~d.status.isin(["CANCLD", "COMMIT"]) & d.state.isin(gate.STATES)]
    chgoff, pif = d.status == "CHGOFF", d.status == "PIF"
    d = d[~((chgoff & d.charge_off_date.isna()) | (pif & d.paid_in_full_date.isna()))]
    chgoff, pif = d.status == "CHGOFF", d.status == "PIF"
    event = pd.Series(
        np.where(chgoff, d.charge_off_date, np.where(pif, d.paid_in_full_date, pd.NaT)),
        index=d.index,
        dtype="datetime64[ns]",
    )
    d = d[~(event < d.approval_date)]
    event, chgoff = event.loc[d.index], chgoff.loc[d.index]
    naics = d.naics.astype("string").str.strip().str[:2]
    franchise = d.franchise_code.astype("string").str.strip()
    u = pd.DataFrame(
        {
            "orig_year": d.approval_date.dt.year.to_numpy(),
            "om": month(d.approval_date),
            "group": d.state.to_numpy(),
            "st": d.state.to_numpy(),
            "log_gross": np.log(d.gross_approval.where(d.gross_approval > 0).to_numpy(dtype=float)),
            "guaranteed_share": (d.sba_guaranteed / d.gross_approval).to_numpy(dtype=float),
            "term_months": d.term_months.to_numpy(dtype=float),
            "naics2": naics.where(naics.str.len() == 2).to_numpy(dtype=object),
            "business_type": d.business_type.to_numpy(dtype=object),
            "business_age": d.business_age.to_numpy(dtype=object),
            "franchise": (franchise.notna() & (franchise != "")).to_numpy(dtype=float),
            "revolver": d.revolver.to_numpy(dtype=object),
            "log1p_jobs": np.log1p(d.jobs.to_numpy(dtype=float)),
            "interest_rate": d.interest_rate.to_numpy(dtype=float),
            "subprogram": d.subprogram.to_numpy(dtype=object),
            "state": d.state.to_numpy(dtype=object),
        }
    )
    ev = event.to_numpy()
    event_mi = np.where(pd.isna(ev), np.nan, month(pd.Series(ev)).astype(float))
    kind = np.where(chgoff.to_numpy(), "default", "prepay")
    return _outcome(u, event_mi, kind, CUT_A).reset_index(drop=True)


def civil_units(*, broad: bool = False) -> pd.DataFrame:
    d = pd.read_parquet(ART / "civil.parquet")
    d = d[d.filedate.between("1989-01-01", "2016-12-31")]
    keep = (
        d.juris.isin([3, 4]).fillna(False)
        & ~d.nos.between(510, 560).fillna(False)
        & ~d.nos.between(861, 865).fillna(False)
        & d.origin.isin([1, 2]).fillna(False)
        & ~d.district.astype(str).isin(TERRITORIES)
    )
    d = d[keep.astype(bool)]
    d = d[~d.duplicated(["district", "office", "docket", "filedate"], keep="first")]
    pending = (
        d.tapeyear.eq(2099).fillna(False).to_numpy(dtype=bool)
    )  # TERMDATE 1900-01-01 is a placeholder
    d = d[~((d.termdate < d.filedate).to_numpy() & ~pending)]
    district = d.district.astype(str)
    demanded = d.demanded.astype(float)
    u = pd.DataFrame(
        {
            "orig_year": d.filedate.dt.year.to_numpy(),
            "om": month(d.filedate),
            "group": district.to_numpy(),
            "st": district.map(lambda k: gate.DISTRICTS[k][0]).to_numpy(),
            "nos": d.nos.astype(float).to_numpy(),
            "juris": d.juris.astype(float).to_numpy(),
            "origin": d.origin.astype(float).to_numpy(),
            "classact": (d.classact == 1).fillna(False).to_numpy(dtype=float),
            "jury": d.jury.astype(str).replace("-8", np.nan).to_numpy(dtype=object),
            "log1p_demanded": np.where(demanded < 0, np.nan, np.log1p(demanded.clip(lower=0))),
            "prose": d.prose.astype(float).where((d.prose >= 0).fillna(False)).to_numpy(),
            "district": district.to_numpy(dtype=object),
        }
    )
    terminated = d.termdate.notna().to_numpy() & ~d.tapeyear.eq(2099).fillna(False).to_numpy(
        dtype=bool
    )
    event_mi = np.where(terminated, month(d.termdate.fillna(pd.Timestamp("2099-01-01"))), np.nan)
    settled = d.disp.isin([12, 13] if broad else [13]).fillna(False).to_numpy()
    kind = np.where(settled, "default", "prepay")
    return _outcome(u, event_mi.astype(float), kind, END_B).reset_index(drop=True)


def judges() -> pd.DataFrame:
    """District x 0-based month: log active non-senior Article III judges, log(max(n, 1)) (decision B2)."""
    j = pd.read_parquet(ART / "judges.parquet")
    j = j[j.court_type == "U.S. District Court"]
    keep = [k for k in gate.DISTRICTS if k not in TERRITORIES]
    names = {gate.court_name(gate.DISTRICTS[k][1]): k for k in keep}
    j = j.assign(district=j.court_name.map(names))
    months = np.arange(1985 * 12, 2027 * 12)
    start = gate.month(j.commission_date).to_numpy(dtype=float)
    end = np.fmin(
        gate.month(j.senior_date).to_numpy(dtype=float),
        gate.month(j.termination_date).to_numpy(dtype=float),
    )
    rows = []
    for k in keep:
        m = (j.district == k).to_numpy()
        active = (start[m][:, None] <= months[None, :]) & ~(end[m][:, None] <= months[None, :])
        n = active.sum(axis=0)
        rows.append(pd.DataFrame({"group": k, "mi": months, "n_judges": n}))
    out = pd.concat(rows, ignore_index=True)
    out["log_judges"] = np.log(np.maximum(out.n_judges, 1))
    return out


def covariates(name: str) -> pd.DataFrame:
    """Group x 0-based month with the raw covariates of the data set."""
    c = pd.read_parquet(ART / "covariates.parquet")
    if name == "sba":
        return c.rename(columns={"state": "group"})[["group", "mi", "hpi_yoy", "unemp"]].dropna()
    j = judges()
    j["st"] = j.group.map(lambda k: gate.DISTRICTS[k][0])
    u = c.rename(columns={"state": "st"})[["st", "mi", "unemp"]].dropna()
    return j.merge(u, on=["st", "mi"], how="inner")[["group", "mi", "log_judges", "unemp"]]


def units(name: str) -> pd.DataFrame:
    return sba_units() if name == "sba" else civil_units(broad=name == "civil_broad")


def build(name: str, u: pd.DataFrame | None = None) -> pl.Data:
    """The frozen pipeline's data object (the construction of experiments/0021 `build`)."""
    u = units(name) if u is None else u
    end = CUT_A if name == "sba" else END_B
    ages = np.arange(1, AMAX_CELLS + 1)
    frames = []
    for (year, om, group), g in u.groupby(["orig_year", "om", "group"]):
        exposure, events, aj = ipcw.unit_arrays(
            g.event_age_clean.to_numpy(), g.event_type_clean.to_numpy(), end - 1 - om + 1
        )
        frames.append(
            pd.DataFrame(
                {
                    "vintage": year,
                    "fp_mi": om,
                    "state": group,
                    "age": ages,
                    "mi": om + ages - 1,
                    "exposure": exposure,
                    "events": events,
                    "aj": aj,
                }
            )
        )
    c = pd.concat(frames, ignore_index=True)
    raw = RAW[name]
    c = c[c.exposure > 0].merge(
        covariates(name).rename(columns={"group": "state"}), on=["state", "mi"], how="inner"
    )
    for r in raw:
        c[f"{r}_z"] = (c[r] - c[r].mean()) / c[r].std()
    c["unit"] = c.groupby(["vintage", "fp_mi", "state"]).ngroup()
    c["year"] = c.vintage.astype(float)
    c["quarter"] = ((c.fp_mi - 1) // 3) / 4.0
    loans = u[
        ["orig_year", "om", "ev_mi", "event_age_clean", "event_type_clean", "year", "quarter"]
    ]
    return pl.Data(
        loans,
        c,
        {r: (float(c[r].mean()), float(c[r].std())) for r in raw},
        list(raw),
        [f"{r}_z" for r in raw],
    )


def persistence(
    data: pl.Data,
    z: np.ndarray,
    a_u: np.ndarray,
    sub: pd.DataFrame,
    s0: int,
    uid: np.ndarray,
    n_u: int,
) -> np.ndarray:
    """Forecast path for B: after each unit's age at S, every covariate at its group's mean of the 12 months up to
    S (the rule of the frozen `pipeline._persistence`, which names the two Freddie covariates)."""
    last = sub[(sub.mi > s0 - 12) & (sub.mi <= s0)]
    if last.empty:
        last = sub[sub.mi <= s0] if (sub.mi <= s0).any() else sub
    by_group = last.groupby("state")[data.raw].mean()
    group = np.empty(n_u, dtype=object)
    group[uid] = sub.state.to_numpy()
    future = np.arange(z.shape[1])[None, :] > a_u[:, None]
    out = z.copy()
    for j, r in enumerate(data.raw):
        level = np.array([by_group[r].get(g, by_group[r].mean()) for g in group])
        out[:, :, j] = np.where(future, level[:, None], z[:, :, j])
    return out


def use(name: str) -> None:
    """Set the forecast path of the data set in the frozen pipeline."""
    if name != "sba":
        pl._persistence = persistence


def subsample(d: pl.Data, keep: np.ndarray) -> pl.Data:
    return dataclasses.replace(d, loans=d.loans[keep])


def self_check() -> None:
    """An event in the entry month counts at age 1; a later event than the cut is censored at the cut."""
    u = pd.DataFrame({"orig_year": [2019, 2019, 2019], "om": [CUT_A - 3] * 3})
    out = _outcome(
        u,
        np.array([CUT_A - 3, CUT_A + 5, np.nan]),
        np.array(["default", "prepay", "prepay"]),
        CUT_A,
    )
    assert out.event_age_clean.tolist() == [1, 3, 3], out
    assert out.event_type_clean.tolist() == ["default", "censored", "censored"], out
    assert out.ev_mi.tolist() == [CUT_A - 2, CUT_A, CUT_A]


if __name__ == "__main__":
    self_check()
    for n in ("sba", "civil"):
        x = units(n)
        print(n, len(x), x.event_type_clean.value_counts().to_dict())
