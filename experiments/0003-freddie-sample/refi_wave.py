"""Refinancing-wave test (proposal agreed in T2 update section 4).

Within vintage x FICO band x LTV band strata, split loans at the median refinancing incentive:
  incentive = PMMS at origination - lowest PMMS 30-year monthly mean in the 36 months after origination
  (depends only on WHEN the loan was originated, not on borrower risk; a first version used the loan's own
  rate and was confounded, because the rate prices risk). Strata also control the rate spread (rate - PMMS).
High incentive = an early rate drop (refinancing wave) hits that loan. Compare, high minus low, at fixed loan age:
  prepay CIF, default CIF (observable eventual incidence), and cause-specific cumulative default hazard
  (prepay as censoring). If waves lower default CIF but not the cause-specific hazard, p(c) is path-dependent.
Writes artifacts/refi_wave.csv and refi_wave.json.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd

from maturity import data

HERE: Final = Path(__file__).resolve().parent
AMAX: Final = 240
HORIZONS: Final = (24, 60, 120)


def main() -> None:
    d = pd.read_parquet(data.DATA_DIR / "processed" / "freddie_sample_loans.parquet")
    d = d[
        d.orig_year.between(2000, 2012)
        & ~d.is_relief_refi
        & d.fico.notna()
        & d.ltv.notna()
        & d.event_age_clean.notna()
    ].copy()
    pmms = _pmms()
    fp = d.first_payment.astype(int)
    d["om"] = (fp // 100) * 12 + fp % 100 - 2  # origination month index (first payment - 1 month)
    d["low"] = d.om.map(lambda m: pmms.reindex(range(m + 1, m + 37)).min())
    d["incentive"] = d.om.map(pmms) - d.low
    d["fico_b"] = pd.cut(d.fico, [0, 740, 900], labels=False)
    d["ltv_b"] = pd.cut(d.ltv, [0, 80, 200], labels=False)
    d["spread_b"] = d.groupby("orig_year").rate_spread.transform(
        lambda s: pd.qcut(s, 3, labels=False, duplicates="drop")
    )
    rows = []
    for key, g in d.groupby(["orig_year", "fico_b", "ltv_b", "spread_b"]):
        hi = g.incentive > g.incentive.median()
        if min(hi.sum(), (~hi).sum()) < 300:
            continue
        a, b = _stats(g[hi]), _stats(g[~hi])
        rows.append(
            {
                "orig_year": key[0],
                "n": len(g),
                **{f"{k}_diff": a[k] - b[k] for k in a},
                **{f"{k}_high": a[k] for k in a},
                **{f"{k}_low": b[k] for k in b},
                "incentive_gap": g[hi].incentive.mean() - g[~hi].incentive.mean(),
            }
        )
    t = pd.DataFrame(rows)
    t.to_csv(HERE / "artifacts" / "refi_wave.csv", index=False)
    out = {
        "strata": len(t),
        "loans": int(t.n.sum()),
        "mean_incentive_gap_pp": round(float(np.average(t.incentive_gap, weights=t.n)), 2),
    }
    for h in HORIZONS:
        for k in ("prepay_cif", "default_cif", "default_cumhaz"):
            col = f"{k}_{h}_diff"
            m = float(np.average(t[col], weights=t.n))
            hi_m = float(np.average(t[f"{k}_{h}_high"], weights=t.n))
            lo_m = float(np.average(t[f"{k}_{h}_low"], weights=t.n))
            out[f"{k}_{h}"] = {
                "high": round(100 * hi_m, 2),
                "low": round(100 * lo_m, 2),
                "diff_pp": round(100 * m, 2),
                "share_strata_negative": round(float((t[col] < 0).mean()), 2),
            }
    rng = np.random.default_rng(0)
    idx = rng.integers(0, len(t), size=(2000, len(t)))  # bootstrap over strata
    for h in HORIZONS:
        for k in ("prepay_cif", "default_cif", "default_cumhaz"):
            col = f"{k}_{h}_diff"
            boot = (t[col].to_numpy()[idx] * t.n.to_numpy()[idx]).sum(axis=1) / t.n.to_numpy()[
                idx
            ].sum(axis=1)
            lo, hi = np.percentile(boot, [2.5, 97.5])
            out[f"{k}_{h}"]["ci95_pp"] = [round(100 * float(lo), 2), round(100 * float(hi), 2)]
    (HERE / "artifacts" / "refi_wave.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


def _stats(g: pd.DataFrame) -> dict[str, float]:
    age = g.event_age_clean.clip(0, AMAX).astype(int).to_numpy()
    kind = g.event_type_clean.to_numpy()
    d, p, c = (
        np.bincount(age[kind == k], minlength=AMAX + 1) for k in ("default", "prepay", "censored")
    )
    risk = len(g) - np.concatenate([[0], np.cumsum(d + p + c)[:-1]])
    div = lambda x: np.divide(x, risk, out=np.zeros(AMAX + 1), where=risk > 0)
    surv = np.concatenate([[1.0], np.cumprod(1 - div(d + p))[:-1]])
    dcif, pcif, chz = np.cumsum(surv * div(d)), np.cumsum(surv * div(p)), np.cumsum(div(d))
    out: dict[str, float] = {}
    for h in HORIZONS:
        out[f"default_cif_{h}"], out[f"prepay_cif_{h}"], out[f"default_cumhaz_{h}"] = (
            float(dcif[h]),
            float(pcif[h]),
            float(chz[h]),
        )
    return out


def _pmms() -> pd.Series:
    x = pd.read_csv(data.DATA_DIR / "raw" / "covariates" / "pmms.csv", usecols=["date", "pmms30"])
    x["date"] = pd.to_datetime(x["date"], format="%m/%d/%Y")
    m = x.groupby(x.date.dt.year * 12 + x.date.dt.month - 1).pmms30.mean()
    return m


if __name__ == "__main__":
    main()
