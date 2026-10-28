"""Composition test for the 2004-2008 front-loading (T5 section 6). DATASET: 50k Sample, development only.

Question: is the front-loading of the crisis vintages a change in borrower mix (Proposition 5a) or a change within
borrower types? Strata: FICO (<660, 660-719, 720+) x original LTV (<=80, 80-90, >90).
Calendar adjustment: T2c step 1 (unpenalized fixed-effects Poisson, vintage x age + state effects, z = state HPI
growth, state unemployment, log mark-to-market LTV), fitted on 1999-2012 vintages; clean default event, prepay censors.
For each vintage 2004-2008: late/early speed log ratio (ages 36-72 vs 6-36) against the 1999-2003 reference shape,
(a) pooled over all loans, (b) within each stratum against that stratum's own reference, then the event-weighted mean
over strata. Cluster bootstrap over (first-payment month, state) groups, shared across strata.
Writes artifacts/composition_test.csv and composition_test.json.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd
import pyfixest as pf
import shock_overlay as so
import speed_fit as sf

from maturity import data

log = logging.getLogger(__name__)
HERE: Final = Path(__file__).resolve().parent
AMAX: Final = 120
HI: Final = 72
YEARS: Final = range(1999, 2013)
REF: Final = range(1999, 2004)
TARGETS: Final = range(2004, 2009)
B: Final = 200
ZCOLS: Final = ["hpi_yoy_z", "unemp_z", "log_mtm_z"]
SUBDIST: Final = "subdist" in __import__("sys").argv[1:]
SUFFIX: Final = "_subdist" if SUBDIST else ""
FICO_LABELS: Final = ["<660", "660-719", "720+"]
LTV_LABELS: Final = ["<=80", "80-90", ">90"]


def main() -> None:
    d = pd.read_parquet(data.DATA_DIR / "processed" / "freddie_sample_loans.parquet")
    d = d[
        d.orig_year.isin(YEARS)
        & d.event_age_clean.notna()
        & d.state.notna()
        & d.ltv.notna()
        & d.fico.notna()
    ].copy()
    d["ltv_band"] = pd.cut(d.ltv, so.LTV_EDGES, labels=False).astype(int)
    d["fico_b"] = pd.cut(d.fico, [0, 659, 719, 900], labels=False).astype(int)
    cells = _cells(d)
    cells["ltv3"] = np.maximum(cells.ltv_band - 1, 0)
    cells["stratum"] = cells.fico_b * 3 + cells.ltv3
    log.info("cells %d, events %d", len(cells), int(cells.events.sum()))
    beta, kappa = _fit(cells)
    log.info("beta %s", dict(zip(ZCOLS, beta.round(3), strict=True)))
    cells["w_adj"] = cells.exposure * np.exp(
        cells[ZCOLS].to_numpy() @ beta + cells.state.map(kappa).fillna(0.0).to_numpy()
    )
    mats = {
        v: _matrices(g) for v, g in cells.groupby("vintage")
    }  # vintage -> (events, w) arrays [stratum, group, age]
    rng = np.random.default_rng(0)
    point = _log_ratios(mats, None)
    boots = np.array(
        [_log_ratios(mats, rng) for _ in range(B)]
    )  # B x (targets) x (pooled, within, 9 strata)
    lo, hi = np.nanpercentile(boots, [2.5, 97.5], axis=0)
    diff = boots[:, :, 1] - boots[:, :, 0]
    rows = []
    for i, v in enumerate(TARGETS):
        ev_s = mats[v][0][:, :, :HI].sum(axis=(1, 2))
        rows.append(
            {
                "vintage": v,
                "pooled": point[i, 0],
                "pooled_lo": lo[i, 0],
                "pooled_hi": hi[i, 0],
                "within": point[i, 1],
                "within_lo": lo[i, 1],
                "within_hi": hi[i, 1],
                "within_minus_pooled": point[i, 1] - point[i, 0],
                "diff_lo": float(np.nanpercentile(diff[:, i], 2.5)),
                "diff_hi": float(np.nanpercentile(diff[:, i], 97.5)),
                **{
                    f"s_{FICO_LABELS[s // 3]}_{LTV_LABELS[s % 3]}": point[i, 2 + s]
                    for s in range(9)
                },
                **{
                    f"events_{FICO_LABELS[s // 3]}_{LTV_LABELS[s % 3]}": int(ev_s[s])
                    for s in range(9)
                },
            }
        )
    t = pd.DataFrame(rows).set_index("vintage")
    t.round(3).to_csv(HERE / "artifacts" / f"composition_test{SUFFIX}.csv")
    summary = {
        "beta": dict(zip(ZCOLS, beta.round(4).tolist(), strict=True)),
        "mean_pooled": round(float(t.pooled.mean()), 3),
        "mean_within": round(float(t.within.mean()), 3),
        "share_of_front_loading_removed_within_strata": round(
            float(1 - t.within.mean() / t.pooled.mean()), 3
        ),
    }
    (HERE / "artifacts" / f"composition_test{SUFFIX}.json").write_text(
        json.dumps(summary, indent=2)
    )
    print(
        t[
            [
                "pooled",
                "pooled_lo",
                "pooled_hi",
                "within",
                "within_lo",
                "within_hi",
                "within_minus_pooled",
                "diff_lo",
                "diff_hi",
            ]
        ]
        .round(2)
        .to_string()
    )
    print(t[[c for c in t.columns if c.startswith("s_")]].round(2).to_string())
    print(t[[c for c in t.columns if c.startswith("events_")]].to_string())
    print(json.dumps(summary))


def _cells(d: pd.DataFrame) -> pd.DataFrame:
    hpi, unemp = so._covariates()
    fp = d.first_payment.astype(int)
    d["fp_mi"] = (fp // 100) * 12 + fp % 100 - 1
    dur = d.event_age_clean
    if SUBDIST:
        dur = dur.where(d.event_type_clean != "prepay", so.CUTOFF_MI - d["fp_mi"] + 1)
    d["dur"] = dur.clip(1, AMAX).astype(int)
    d["ev"] = (d.event_type_clean == "default") & (d.event_age_clean <= AMAX)
    ages = np.arange(1, AMAX + 1)
    out = []
    for key, g in d.groupby(["orig_year", "fp_mi", "state", "ltv_band", "fico_b"], observed=True):
        dur, ev = g.dur.to_numpy(), g.ev.to_numpy()
        n_at = len(dur) - np.concatenate(
            [[0], np.cumsum(np.bincount(dur, minlength=AMAX + 1))[:-1]]
        )
        events = np.bincount(dur[ev], minlength=AMAX + 1)
        keep = n_at[1:] > 0
        out.append(
            pd.DataFrame(
                {
                    "vintage": key[0],
                    "fp_mi": key[1],
                    "state": key[2],
                    "ltv_band": key[3],
                    "fico_b": key[4],
                    "age": ages[keep],
                    "mi": key[1] + ages[keep] - 1,
                    "exposure": n_at[1:][keep],
                    "events": events[1:][keep],
                }
            )
        )
    c = pd.concat(out, ignore_index=True)
    c["qi"] = c.mi // 3
    c = c.merge(hpi, on=["state", "qi"], how="inner").merge(unemp, on=["state", "mi"], how="inner")
    h0 = hpi.rename(columns={"hpi": "hpi_orig", "qi": "qi_orig"})[["state", "qi_orig", "hpi_orig"]]
    c["qi_orig"] = (c.fp_mi - 1) // 3
    c = c.merge(h0, on=["state", "qi_orig"], how="inner")
    c["log_mtm"] = np.log(so.LTV_MID[c.ltv_band.to_numpy()] * c.hpi_orig / c.hpi)
    c = c.dropna(subset=["hpi_yoy", "unemp", "log_mtm"])
    for col in ("hpi_yoy", "unemp", "log_mtm"):
        c[f"{col}_z"] = (c[col] - c[col].mean()) / c[col].std()
    return c


def _fit(c: pd.DataFrame) -> tuple[np.ndarray, pd.Series]:
    d = c[["events", "exposure", "state", *ZCOLS]].copy()
    d["cell"] = (c.vintage.to_numpy() - YEARS.start) * AMAX + (c.age.to_numpy() - 1)
    d["log_exposure"] = np.log(d.exposure)
    m = pf.fepois(
        f"events ~ {' + '.join(ZCOLS)} | cell + state",
        data=d,
        offset="log_exposure",
        vcov={"CRV1": "state"},
    )
    fe = m.fixef()
    kappa = pd.Series(fe[next(k for k in fe if "state" in k)], dtype=float)
    return m.tidy().loc[ZCOLS, "Estimate"].to_numpy(), kappa - float(kappa.mean())


def _matrices(g: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """(events, adjusted exposure) arrays of shape [9 strata, groups, AMAX]; groups = (first-payment month, state)."""
    grp = g.groupby(["fp_mi", "state"]).ngroup().to_numpy()
    shape = (9, grp.max() + 1, AMAX)
    ev, w = np.zeros(shape), np.zeros(shape)
    idx = (g.stratum.to_numpy(), grp, g.age.to_numpy() - 1)
    np.add.at(ev, idx, g.events.to_numpy())
    np.add.at(w, idx, g.w_adj.to_numpy())
    return ev, w


def _log_ratios(
    mats: dict[int, tuple[np.ndarray, np.ndarray]], rng: np.random.Generator | None
) -> np.ndarray:
    """Per target vintage: [pooled log ratio, event-weighted within-strata mean, 9 stratum log ratios]."""
    agg = {}
    for v, (ev, w) in mats.items():
        if v not in REF and v not in TARGETS:
            continue
        n = ev.shape[1]
        m = np.ones(n) if rng is None else rng.multinomial(n, np.full(n, 1 / n)).astype(float)
        agg[v] = (np.einsum("sga,g->sa", ev, m), np.einsum("sga,g->sa", w, m))
    ref_ev = sum(agg[v][0] for v in REF)
    ref_w = sum(agg[v][1] for v in REF)
    ref_pooled = sf._curve(ref_ev.sum(0), ref_w.sum(0))
    ref_s = [sf._curve(ref_ev[s], ref_w[s]) for s in range(9)]
    out = np.full((len(TARGETS), 11), np.nan)
    for i, v in enumerate(TARGETS):
        ev, w = agg[v]
        est = sf._speeds(ev.sum(0), w.sum(0), HI, ref_pooled)
        out[i, 0] = np.log(est[2] / est[1])
        weights = ev[:, :HI].sum(axis=1)
        for s in range(9):
            if weights[s] < 30 or ref_ev[s, :HI].sum() < 60:
                continue
            e = sf._speeds(ev[s], w[s], HI, ref_s[s])
            out[i, 2 + s] = np.log(e[2] / e[1])
        ok = ~np.isnan(out[i, 2:])
        if ok.any():
            out[i, 1] = float(np.average(out[i, 2:][ok], weights=weights[ok]))
    return out


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
