"""1-hour feasibility check on US patent applications (the patent feasibility plan, with its revision of 2026-10-04).

DATASET: USPTO PatEx, release folder "2022" (PEDS data of June 2023), application_data only. Nonprovisional utility
applications filed 2001-2019 in the technology centres 16, 17, 21, 24, 26, 28, 36, 37; 100,000 per filing year
(simple random sample, fixed seed). Primary sample: all such applications in PatEx. Sensitivity: only applications
with a pre-grant publication.
Cohort = filing year. Event = grant (patent_issue_date). Competing exit = abandonment (current status starts with
"Abandoned" or "Expressly Abandoned"; date = appl_status_date). Censoring = the data date (2023-06).
Steps: (1) raw timing curves; (2) calendar adjustment: fixed-effects Poisson on sub-distribution risk sets, with
log pending inventory of the technology centre and an Alice indicator (TC 36 and 21 from 2014-07); (3) one speed per
filing year on the adjusted cumulative hazard, with standard errors, and the tempered evidence for chain ladder,
random walk and change point (maturity.onestage); (4) overlay figure.
Decision rule (fixed before the data was read): go if the evidence prefers a random walk or a change point AND the
adjusted log speeds span at least 0.2. Both samples must agree.
Usage: extract.py, then PYTHONHASHSEED=0 uv run python check.py -> results.json, figures/overlay_<sample>.png
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Final

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyfixest as pf
from numpy.typing import NDArray

from maturity import data, onestage

HERE: Final = Path(__file__).resolve().parent
sys.path[:0] = [
    str(HERE.parent / "0012-freddie-200k-harp-correction"),
    str(HERE.parent / "0003-freddie-sample"),
]
import cells as ipcw  # unit_arrays: sub-distribution risk sets on the age axis (ages 1..120 months)
import pipeline  # _residual_share (guard G1)
import shock_overlay as so  # colour ramp

log = logging.getLogger(__name__)
Array = NDArray[np.float64]
SOURCE: Final = data.DATA_DIR / "processed" / "patex_utility.parquet"
YEARS: Final = range(2001, 2020)
CENTRES: Final = ("16", "17", "21", "24", "26", "28", "36", "37")
N_PER_YEAR: Final = 100_000
SEED: Final = 20261004
H: Final = 96  # months: horizon of the common timing curve
CUTOFF: Final = 2023 * 12 + 5  # month index of 2023-06, the data date
ALICE: Final = 2014 * 12 + 6  # 2014-07
AMAX: Final = ipcw.AMAX


def month(dates: pd.Series) -> pd.Series:
    return dates.dt.year * 12 + dates.dt.month - 1


def load() -> pd.DataFrame:
    d = pd.read_parquet(SOURCE)
    for c in ("filing_date", "appl_status_date", "earliest_pgpub_date", "patent_issue_date"):
        d[c] = pd.to_datetime(d[c], errors="coerce")
    d = d[d.filing_date.notna()].copy()
    d["tc"] = d.examiner_art_unit.str[:2]
    d = d[d.tc.isin(CENTRES)]
    d["fm"] = month(d.filing_date)
    granted = d.patent_issue_date.notna()
    abandoned = (
        d.appl_status_desc.fillna("").str.contains("Abandoned")
        & ~granted
        & d.appl_status_date.notna()
    )
    d["kind"] = np.where(
        granted, "default", np.where(abandoned, "prepay", "censored")
    )  # names of the cell code
    end = np.where(
        granted, month(d.patent_issue_date), np.where(abandoned, month(d.appl_status_date), CUTOFF)
    )
    d["end"] = end
    d["dur"] = np.maximum(end - d.fm, 1).astype(int)
    d["year"] = d.filing_date.dt.year
    d["published"] = d.earliest_pgpub_date.notna()
    return d[["year", "fm", "tc", "kind", "end", "dur", "published"]]


def inventory(d: pd.DataFrame) -> pd.DataFrame:
    """Log pending inventory per technology centre and month: filed, and not yet granted or abandoned, over ALL
    utility applications in the file. Applications filed before 2001 are in PatEx only if they were granted or
    published, so the level before about 2004 is too low."""
    months = np.arange(2001 * 12, CUTOFF + 1)
    rows = []
    for tc, g in d.groupby("tc"):
        filed = np.searchsorted(np.sort(g.fm.to_numpy()), months, side="right")
        done = np.searchsorted(
            np.sort(g.end.to_numpy()[g.kind.to_numpy() != "censored"]), months, side="right"
        )
        rows.append(pd.DataFrame({"tc": tc, "mi": months, "log_inventory": np.log(filed - done)}))
    return pd.concat(rows, ignore_index=True)


def cells(sample: pd.DataFrame, inv: pd.DataFrame) -> pd.DataFrame:
    ages = np.arange(1, AMAX + 1)
    out = []
    for (year, fm, tc), g in sample.groupby(["year", "fm", "tc"]):
        exposure, events, _ = ipcw.unit_arrays(g.dur.to_numpy(), g.kind.to_numpy(), CUTOFF - fm)
        out.append(
            pd.DataFrame(
                {
                    "year": year,
                    "fm": fm,
                    "tc": tc,
                    "age": ages,
                    "mi": fm + ages,
                    "exposure": exposure,
                    "events": events,
                }
            )
        )
    c = pd.concat(out, ignore_index=True)
    c = c[c.exposure > 0].merge(inv, on=["tc", "mi"], how="inner")
    c["alice"] = (c.tc.isin(["36", "21"]) & (c.mi >= ALICE)).astype(float)
    c["inv_z"] = (c.log_inventory - c.log_inventory.mean()) / c.log_inventory.std()
    return c


def calendar_step(c: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object]]:
    """T2c step 1: fixed-effects Poisson (filing year x age-quarter, technology centre), offset log at-risk."""
    f = c[["events", "exposure", "tc", "inv_z", "alice"]].copy()
    f["cell"] = (c.year.to_numpy() - YEARS.start) * 1000 + (c.age.to_numpy() - 1) // 3
    f["log_exposure"] = np.log(f.exposure)
    m = pf.fepois(
        "events ~ inv_z + alice | cell + tc", data=f, offset="log_exposure", vcov={"CRV1": "tc"}
    )
    tidy, fe = m.tidy(), m.fixef()
    kappa = pd.Series(fe[next(k for k in fe if "tc" in k)], dtype=float)
    weights = f.groupby("tc").exposure.sum().reindex(kappa.index).to_numpy()
    kappa = kappa - float(np.average(kappa.to_numpy(), weights=weights))
    lin = (
        c.inv_z * tidy.loc["inv_z", "Estimate"]
        + c.alice * tidy.loc["alice", "Estimate"]
        + c.tc.map(kappa).fillna(0.0)
    )
    c = c.assign(w_adj=c.exposure * np.exp(lin))
    # identifying variation (guard G1 of T2c): share of the covariate's variance left after the fixed effects
    cell, centre = pd.factorize(f.cell)[0], pd.factorize(f.tc)[0]
    share = pipeline._residual_share(
        c.inv_z.to_numpy(), cell, centre, c.events.to_numpy(dtype=float)
    )
    info = {
        "beta_log_inventory_per_sd": round(float(tidy.loc["inv_z", "Estimate"]), 4),
        "se_log_inventory": round(float(tidy.loc["inv_z", "Std. Error"]), 4),
        "sd_of_log_inventory": round(float(c.log_inventory.std()), 4),
        "beta_alice": round(float(tidy.loc["alice", "Estimate"]), 4),
        "se_alice": round(float(tidy.loc["alice", "Std. Error"]), 4),
        "identifying_variation_share_log_inventory": round(share, 4),
        "g1_events_x_share": round(float(c.events.sum() * share), 0),
        "tc_effects": {str(k): round(float(v), 3) for k, v in kappa.items()},
    }
    return c, info


def curves(c: pd.DataFrame, weight: str) -> dict[int, Array]:
    """Cumulative hazard of each filing year on the age axis (ages 0..AMAX months)."""
    g = c.groupby(["year", "age"]).agg(events=("events", "sum"), w=(weight, "sum"))
    out = {}
    for year, t in g.groupby(level=0):
        inc = np.zeros(AMAX + 1)
        inc[t.index.get_level_values(1).to_numpy()] = t.events.to_numpy() / np.maximum(
            t.w.to_numpy(), 1e-300
        )
        out[int(year)] = np.cumsum(inc)
    return out


def cohorts(curve: dict[int, Array], c: pd.DataFrame) -> list[onestage.Cohort]:
    ages = np.arange(AMAX + 1, dtype=float)
    out = []
    for year in YEARS:
        follow = CUTOFF - (year * 12 + 6)  # months of follow-up of an application filed at mid-year
        a = int(min(follow, H))
        events = float(c[(c.year == year) & (c.age <= a)].events.sum())
        out.append(
            onestage.Cohort(
                float(year),
                ages[: a + 1].copy(),
                curve[year][: a + 1].copy(),
                events,
                float(H),
                follow >= H,
            )
        )
    return out


def speed_errors(coh: list[onestage.Cohort], fit: onestage.Fit, phi: float) -> list[float]:
    """Standard error of each log speed from the curvature of its multinomial shape log-likelihood (reference shape
    held fixed), times sqrt(phi)."""

    def loglik(cohort: onestage.Cohort, log_s: float) -> float:
        f = np.interp(np.exp(log_s) * cohort.x, fit.ugrid, fit.f0, right=1.0)
        p = np.maximum(np.diff(f), 1e-300) / max(f[-1] - f[0], 1e-300)
        q = np.diff(cohort.g) / (cohort.g[-1] - cohort.g[0])
        return float(cohort.events * np.sum(q * np.log(p)))

    out = []
    for cohort, s in zip(coh, fit.speeds, strict=True):
        # backward second difference: a faster clock can leave the grid of the reference shape
        x, h = float(np.log(s)), 0.02
        curvature = (
            loglik(cohort, x) - 2 * loglik(cohort, x - h) + loglik(cohort, x - 2 * h)
        ) / h**2
        out.append(float(np.sqrt(phi / max(-curvature, 1e-12))))
    return out


def overlay(
    raw: dict[int, Array], adjusted: dict[int, Array], name: str
) -> dict[str, dict[str, float]]:
    """Figure and spread: F(a | c) = cumulative hazard / its value at 96 months, filing years 2001-2014."""
    years = [y for y in YEARS if CUTOFF - (y * 12 + 6) >= H]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    spread = {}
    for ax, (label, curve) in zip(
        axes, (("raw", raw), ("calendar-adjusted", adjusted)), strict=True
    ):
        norm = pd.DataFrame({y: curve[y][: H + 1] / curve[y][H] for y in years})
        for y in years:
            ax.plot(norm.index, norm[y], color=so.RAMP((y - years[0]) / (len(years) - 1)), lw=1.2)
        ax.set(title=label, xlabel="months since filing", xlim=(0, H))
        ax.spines[["top", "right"]].set_visible(False)
        spread[label] = {
            str(a): round(float(norm.loc[a].std() / norm.loc[a].mean()), 3)
            for a in (24, 36, 48, 60, 72)
        }
    axes[0].set_ylabel("share of the 96-month grants made by this age")
    scale = plt.cm.ScalarMappable(cmap=so.RAMP, norm=plt.Normalize(years[0], years[-1]))
    fig.colorbar(scale, ax=axes, label="filing year", fraction=0.03, pad=0.02)
    fig.savefig(HERE / "figures" / f"overlay_{name}.png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    return spread


def analyse(d: pd.DataFrame, inv: pd.DataFrame, name: str) -> dict[str, object]:
    sample = d[d.year.isin(YEARS)].groupby("year").sample(n=N_PER_YEAR, random_state=SEED)
    c, calendar = calendar_step(cells(sample, inv))
    raw, adjusted = curves(c, "exposure"), curves(c, "w_adj")
    out: dict[str, object] = {
        "applications": len(sample),
        "grants": int(c.events.sum()),
        "calendar_step": calendar,
    }
    out["timing_spread_cv"] = overlay(raw, adjusted, name)
    for label, curve in (("raw", raw), ("adjusted", adjusted)):
        coh = cohorts(curve, c)
        phi = onestage.dispersion(coh)
        free = onestage.fit(
            coh, 100.0
        )  # a very large random-walk variance: one free speed per filing year
        sel = onestage.select_model(coh, phi)
        log_free, log_sel = np.log(free.speeds), np.log(sel.fit.speeds)
        errors = speed_errors(coh, free, phi)
        precise = np.array(errors) < 0.05
        out[label] = {
            "phi": round(phi, 2),
            "selected_model": sel.model,
            "log_evidence": {k: round(v, 1) for k, v in sel.log_evidence.items()},
            "log_speed_free": {
                int(k.key): round(float(v), 3) for k, v in zip(coh, log_free, strict=True)
            },
            "log_speed_se": {
                int(k.key): round(float(v), 3) for k, v in zip(coh, errors, strict=True)
            },
            "log_speed_selected_model": {
                int(k.key): round(float(v), 3) for k, v in zip(coh, log_sel, strict=True)
            },
            "span_free": round(float(log_free.max() - log_free.min()), 3),
            "span_free_se_below_0.05": round(
                float(log_free[precise].max() - log_free[precise].min()), 3
            ),
            "span_selected_model": round(float(log_sel.max() - log_sel.min()), 3),
        }
    a = out["adjusted"]
    assert isinstance(a, dict)
    out["rule_a_evidence_prefers_rw_or_cp"] = a["selected_model"] != "chain"
    out["rule_b_span_at_least_0.2"] = a["span_free_se_below_0.05"] >= 0.2
    out["go"] = bool(out["rule_a_evidence_prefers_rw_or_cp"] and out["rule_b_span_at_least_0.2"])
    log.info(
        "%s: model %s, span %s / %s, go = %s",
        name,
        a["selected_model"],
        a["span_free"],
        a["span_free_se_below_0.05"],
        out["go"],
    )
    return out


def coverage(d: pd.DataFrame) -> list[dict[str, object]]:
    """Counts per filing year, and an internal estimate of the share that PatEx does not hold (abandoned and never
    published), under the assumption that an abandoned application is unpublished as often as a granted one."""
    rows = []
    for year, g in d[d.year.isin(YEARS)].groupby("year"):
        granted, abandoned = g.kind == "default", g.kind == "prepay"
        ratio = (granted & ~g.published).sum() / max((granted & g.published).sum(), 1)
        missing = abandoned.sum() * ratio
        rows.append(
            {
                "filing_year": int(year),
                "patex_applications": len(g),
                "granted": round(float(granted.mean()), 4),
                "abandoned": round(float(abandoned.mean()), 4),
                "pending": round(float((g.kind == "censored").mean()), 4),
                "published": round(float(g.published.mean()), 4),
                "granted_without_publication": round(float((granted & ~g.published).mean()), 4),
                "estimated_missing_share": round(float(missing / (len(g) + missing)), 4),
            }
        )
    return rows


def main() -> None:
    (HERE / "figures").mkdir(exist_ok=True)
    d = load()
    inv = inventory(d)
    out = {
        "coverage": coverage(d),
        "all_utility": analyse(d, inv, "all_utility"),
        "published_only": analyse(d[d.published], inv, "published_only"),
    }
    go = [out[k]["go"] for k in ("all_utility", "published_only")]  # type: ignore[index]
    out["decision"] = "go" if all(go) else "no-go"
    (HERE / "results.json").write_text(json.dumps(out, indent=1, default=str))
    log.info("decision: %s", out["decision"])


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    main()
