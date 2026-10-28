"""Scores of the patent hazard models, wrong-direction counts on patents and mortgages, coefficients (POST HOC).

DATASET: no new data. Patents: artifacts/pairs_hazard.parquet of hazard.py (the 43 pairs of experiment 0031 with the
hazard estimates). Mortgages: the 78 pairs of experiment 0032 (`pairs_hazard.parquet`, corrected weights) and, for
the truth of the reference vintage, the Freddie Mac 200k loan file through the truth code of experiment 0012.
Scoring as Table 2 (maturity.inference.compare): median over snapshots of the mean absolute relative error; error ratio
with a 95% filing-year cluster bootstrap; exact cluster sign-flip p.
Usage: uv run python report.py -> results.json, research/drift-or-maturity/figdata/patent_hazard_by_snapshot.csv
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Final

import numpy as np
import pandas as pd

from maturity import data, inference

HERE: Final = Path(__file__).resolve().parent
EXP: Final = HERE.parent
FIGDATA: Final = EXP.parent / "research" / "drift-or-maturity" / "figdata"
sys.path.insert(0, str(EXP / "0012-freddie-200k-harp-correction"))
import pipeline as pl  # the truth code of the mortgage backtest

MODELS: Final = {
    "m1": "M1",
    "m2": "M2",
    "m3": "M3",
    "m1own": "M1, inventory without the scored years",
}
# path -> (full method, calendar-adjusted chain ladder) on that path
PATHS: Final = {"real": ("ours", "cal_chain"), "fcst": ("ours_uncond", "cal_chain_forecast")}
BASELINES: Final = [
    "naive",
    "chain",
    "cal_chain",
    "cal_chain_forecast",
    "cape_cod",
    "ours",
    "ours_uncond",
]


def error(p: pd.DataFrame, m: str) -> np.ndarray:
    return ((p[m] - p.truth) / p.truth).to_numpy()


def compare(p: pd.DataFrame, a: str, b: str) -> dict[str, object]:
    c = inference.compare(
        error(p, a),
        error(p, b),
        p.cohort.to_numpy().astype(np.int64),
        p.snapshot.to_numpy().astype(np.int64),
    )
    low, high = (math.exp(x) for x in c.ci95_cluster_bootstrap)
    return {
        "ratio": round(c.error_ratio, 3),
        "ci95": [round(low, 3), round(high, 3)],
        "sign_flip_p": round(c.sign_flip_p, 4),
        "snapshots_a_better": c.snapshots_where_a_is_better,
        "snapshots": c.snapshots,
    }


def median_error(p: pd.DataFrame, m: str) -> float:
    return round(
        float(pd.Series(np.abs(error(p, m))).groupby(p.snapshot.to_numpy()).mean().median()), 4
    )


def wrong(p: pd.DataFrame, methods: list[str], keep: np.ndarray) -> dict[str, int]:
    big = p[keep]
    sign = np.sign(big.truth - big.ref)
    return {m: int((np.sign(big[m] - big.ref) != sign).sum()) for m in methods}


def patents(p: pd.DataFrame) -> dict[str, object]:
    rows = {}
    for key, label in MODELS.items():
        for path, (full, cal) in PATHS.items():
            m = f"{key}_{path}"
            rows[m] = {
                "label": label,
                "path": path,
                "median_error": median_error(p, m),
                "vs_chain": compare(p, m, "chain"),
                "full_method_vs_model": compare(p, full, m),
                "cal_chain_vs_model": compare(p, cal, m),
            }
            if (
                key == "m1own"
            ):  # also against the frozen methods with the same inventory (0031 `no_own`)
                rows[m] |= {
                    "full_method_same_inventory": {
                        "median_error": median_error(p, f"{full}_own"),
                        "vs_model": compare(p, f"{full}_own", m),
                    },
                    "cal_chain_same_inventory": {
                        "median_error": median_error(p, f"{cal}_own"),
                        "vs_model": compare(p, f"{cal}_own", m),
                    },
                }
    reference = {
        m: {"median_error": median_error(p, m), "vs_chain": compare(p, m, "chain")}
        for m in BASELINES[1:]
    }
    big = ((p.truth - p.ref).abs() >= 0.02).to_numpy()
    hazard = [f"{k}_{path}" for k in MODELS for path in PATHS]
    return {
        "pairs": len(p),
        "models": rows,
        "reference_methods": reference,
        "wrong_direction": {"pairs": int(big.sum()), "counts": wrong(p, BASELINES + hazard, big)},
    }


def mortgage_refs(p: pd.DataFrame) -> pd.DataFrame:
    """The truth of the latest vintage that is mature (median age >= 84 months) at each snapshot, by the truth code
    of the backtest (set E3b cuts the truth at 2020-02, as experiment 0031 `freddie.py`)."""
    d = pd.read_parquet(
        data.DATA_DIR / "processed" / "freddie_200k_loans.parquet",
        columns=[
            "orig_year",
            "first_payment",
            "event_age_clean",
            "event_type_clean",
            "event_period_clean",
        ],
    )
    d = d[
        d.event_age_clean.notna() & d.first_payment.notna() & d.orig_year.between(1999, 2019)
    ].copy()
    d["om"] = pl.month_idx(d.first_payment.astype(int)) - 1
    d["ev_mi"] = pl.month_idx(d.event_period_clean.astype(int))
    d["year"] = d.orig_year.astype(float)
    loans = SimpleNamespace(loans=d)
    cut = int(pl.month_idx(202002))
    truth = {s: pl.truth(loans, "year", 84, cut if s == "E3b" else None) for s in ("E3a", "E3b")}
    out = p.copy()
    assert np.allclose(
        [truth[r.set][r.cohort][0] for r in p.itertuples()], p.truth, atol=1e-9, rtol=0
    )
    refs = {}
    for year in sorted(p.snapshot.unique()):
        info, _ = pl.observed(loans, int(pl.month_idx(int(year) * 100 + 12)), "year")
        refs[year] = float(info[info.A >= 84].cohort.max())
    out["ref_cohort"] = out.snapshot.map(refs)
    out["ref"] = [truth[r.set][r.ref_cohort][0] for r in out.itertuples()]
    return out


def mortgages() -> dict[str, object]:
    p = mortgage_refs(
        pd.read_parquet(
            EXP / "0032-mortgage-hazard-baseline" / "artifacts" / "pairs_hazard.parquet"
        )
    )
    big = ((p.truth - p.ref).abs() / p.ref >= 0.10).to_numpy()
    methods = ["naive", "chain", "cal_chain", "cal_chain_forecast", "ours", "ours_uncond"]
    methods += ["hazard_vintage_real", "hazard_vintage_fcst"]
    return {
        "pairs": len(p),
        "pairs_with_true_change_ge_10pct": int(big.sum()),
        "reference_vintage_by_snapshot": {
            str(k): int(v) for k, v in p.groupby("snapshot").ref_cohort.first().items()
        },
        "counts": wrong(p, methods, big),
    }


def by_snapshot(p: pd.DataFrame) -> pd.DataFrame:
    methods = BASELINES[1:] + [f"{k}_{path}" for k in MODELS for path in PATHS]
    rows = [
        {
            "snapshot": int(s),
            "method": m,
            "mean_signed_error": round(float(error(g, m).mean()), 5),
            "mean_absolute_error": round(float(np.abs(error(g, m)).mean()), 5),
        }
        for m in methods
        for s, g in p.groupby("snapshot")
    ]
    return pd.DataFrame(rows)


def coefficients() -> dict[str, object]:
    raw = json.loads((HERE / "artifacts" / "coefficients.json").read_text())
    out: dict[str, object] = {"inventory_z_stats": raw["inventory_z_stats"]}
    terms = {}
    for year, models in raw["by_snapshot"].items():
        for key, m in models.items():
            for cause in ("grant", "abandon"):
                for term, value in m[cause]["coef"].items():
                    terms[f"{year} {key} {cause} {term}"] = {
                        "coef": value,
                        "se_iid": m[cause]["se_iid"][term],
                        "se_centre": m[cause]["se_centre"][term],
                    }
    out["terms"] = terms
    out["grant_year_effects_m1"] = {
        year: {
            y: round(v - raw["by_snapshot"][year]["m1"]["grant"]["year_effects"]["2004"], 4)
            for y, v in raw["by_snapshot"][year]["m1"]["grant"]["year_effects"].items()
            if int(y) >= 2004
        }
        for year in ("2011", "2012")
    }
    out["years_without_effect"] = {
        f"{year} {key} {cause}": m[cause]["years_without_effect"]
        for year, models in raw["by_snapshot"].items()
        for key, m in models.items()
        for cause in ("grant", "abandon")
        if m[cause]["years_without_effect"]
    }
    out["fit_seconds"] = {
        f"{y} {k}": m["seconds"] for y, ms in raw["by_snapshot"].items() for k, m in ms.items()
    }
    out["fit_rows"] = {
        f"{y} {k}": m["rows"] for y, ms in raw["by_snapshot"].items() for k, m in ms.items()
    }
    return out


def main() -> None:
    p = pd.read_parquet(HERE / "artifacts" / "pairs_hazard.parquet")
    own = pd.read_parquet(
        EXP / "0031-round3-backtests" / "artifacts" / "pairs_patents_no_own.parquet"
    )
    columns = [m for pair in PATHS.values() for m in pair]
    p = p.merge(
        own[["snapshot", "cohort", *columns]].rename(columns={m: f"{m}_own" for m in columns}),
        on=["snapshot", "cohort"],
        validate="one_to_one",
    )
    by_snapshot(p).to_csv(FIGDATA / "patent_hazard_by_snapshot.csv", index=False)
    out = {"patents": patents(p), "mortgages": mortgages(), "coefficients": coefficients()}
    (HERE / "results.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
