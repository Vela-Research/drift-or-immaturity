"""Mortgage and patent audit corrections that need no new run of a method (POST HOC), as run against corrected.

DATASET: Freddie Mac 200k (`data/processed/freddie_200k_loans.parquet`, and the rebuilt ages of run.py in
`artifacts/data`); the stored simulation results of experiments/0034 and 0037 (no real data). Aggregates only.
  F3 wrong direction on mortgages (audit 2 #2): the reference is the latest mature vintage's incidence at the scored
     pair's own horizon h (0066 `report.mortgage_refs` takes it at 84 months, also for the pre-COVID pairs cut at
     2020-02 whose h is below 84). As run (0032 pairs, stored ages) and corrected (tree pairs, rebuilt ages).
  F4 simulation coverage (audit 2 #3): only pairs with J >= 9 calibration errors get an interval, as the paper's rule
     says; 0034 `report.coverage` stores this share next to the J >= 1 share that the paper used.
  F5 mortgage coverage (audit 1 #9): the calibration errors are 84-month errors, so coverage counts only the pairs
     whose target is the 84-month incidence (h = 84); 0036 `report.intervals` also gave intervals to the pre-COVID
     pairs cut at 2020-02 (h < 84). As run (0036 artifacts) and corrected (tree).
Usage: uv run python report.py -> results.json (after run.py's queue)
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Final

import numpy as np
import pandas as pd

from maturity import data as vdata

HERE: Final = Path(__file__).resolve().parent
EXP: Final = HERE.parent
TREE: Final = HERE / "artifacts" / "tree" / "experiments"
WRONG: Final = [
    "naive",
    "chain",
    "cal_chain",
    "cal_chain_forecast",
    "ours",
    "ours_uncond",
    "hazard_vintage_real",
    "hazard_vintage_fcst",
]


def module(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m  # dataclasses look their module up
    spec.loader.exec_module(m)
    return m


def wrong_direction(pairs: pd.DataFrame, loans: Path, pl: ModuleType) -> dict[str, object]:
    """F3: counts with the reference at 84 months (as 0066) and at the pair's own horizon."""
    d = pd.read_parquet(
        loans,
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
    units = SimpleNamespace(loans=d)
    cut = int(pl.month_idx(202002))
    curves = {
        s: pl.observed(units, cut if s == "E3b" else 10**9, "year")[1] for s in ("E3a", "E3b")
    }
    truth = {s: pl.truth(units, "year", 84, cut if s == "E3b" else None) for s in ("E3a", "E3b")}
    assert np.allclose(
        [truth[r.set][r.cohort][0] for r in pairs.itertuples()], pairs.truth, atol=1e-9
    )
    refs = {}
    for year in sorted(pairs.snapshot.unique()):
        info, _ = pl.observed(units, int(pl.month_idx(int(year) * 100 + 12)), "year")
        refs[year] = float(info[info.A >= 84].cohort.max())
    p = pairs.assign(ref_cohort=pairs.snapshot.map(refs))
    out: dict[str, object] = {}
    for label, ref in (
        (
            "reference at 84 months (as 0066)",
            [truth[r.set][r.ref_cohort][0] for r in p.itertuples()],
        ),
        (
            "reference at the pair's h",
            [curves[r.set][r.ref_cohort][int(r.h)] for r in p.itertuples()],
        ),
    ):
        q = p.assign(ref=ref)
        big = ((q.truth - q.ref).abs() / q.ref >= 0.10).to_numpy()
        sign = np.sign(q.truth - q.ref)[big]
        out[label] = {
            "pairs_with_true_change_ge_10pct": int(big.sum()),
            "counts": {m: int((np.sign(q[m] - q.ref)[big] != sign).sum()) for m in WRONG},
        }
    return out


def simulation_coverage() -> dict[str, object]:
    """F4: coverage of the corrected nominal 90% interval of the full method at Gamma = 1, J >= 1 against J >= 9."""
    r = json.loads((EXP / "0037-corrected-plugin-simulation" / "results.json").read_text())
    out: dict[str, object] = {}
    for path, block in _blocks(r["settings"]["corrected"]):
        c = block["coverage"]["ours"]
        out[path] = {
            "J>=1 (paper)": c["all"]["coverage"],
            "J>=9 (rule)": c["J_ge_9"]["coverage"],
            "pairs J>=9": c["J_ge_9"]["pairs"],
            "by age (coverage, J)": {
                k: [v["coverage"], v["J"]] for k, v in c.items() if k.startswith("age_")
            },
        }
    return out


def coverage_full_horizon(root: Path) -> dict[str, object]:
    """F5: 0036 `report.intervals` (strict rule, J >= 9) on the mortgage pairs with h = 84, Gamma grid."""
    r36 = module(root / "0036-corrected-plugin-backtests" / "report.py", f"report36_{root.name}")
    out: dict[str, object] = {}
    for version in ("as_run", "corrected"):
        for code in ("frozen", "corrected"):
            cal = pd.read_parquet(r36.ART / f"calibration_{version}_{code}.parquet")
            path = (
                r36.FROZEN / f"pairs_freddie_{version}.parquet"
                if code == "frozen"
                else r36.ART / f"pairs_{version}_corrected.parquet"
            )
            scored = pd.read_parquet(path)
            for method in ("ours_uncond", "chain"):
                for label, keep in (
                    ("all pairs (paper)", scored.h > 0),
                    ("h = 84 only", scored.h == 84),
                ):
                    t = r36.intervals(scored[keep], cal, method, 84, 2)
                    t = t[t.J >= 9]
                    out[f"{version} {code} {method} {label}"] = {
                        "pairs": len(t),
                        "covered": {
                            f"{g:g}": f"{int(t[f'covered_{g}'].sum())}/{len(t)}" for g in r36.GAMMAS
                        },
                    }
    return out


def _blocks(tree: dict[str, object], path: str = "") -> list[tuple[str, dict[str, object]]]:
    """The setting blocks (dicts with a coverage entry) of a results tree, with their paths."""
    if "coverage" in tree:
        return [(path.strip("/"), tree)]
    return [b for k, v in tree.items() if isinstance(v, dict) for b in _blocks(v, f"{path}/{k}")]


def main() -> None:
    pl = module(EXP / "0012-freddie-200k-harp-correction" / "pipeline.py", "pipeline12")
    name = "freddie_200k_loans.parquet"
    as_run = pd.read_parquet(
        EXP / "0032-mortgage-hazard-baseline" / "artifacts" / "pairs_hazard.parquet"
    )
    fixed = pd.read_parquet(
        TREE / "0032-mortgage-hazard-baseline" / "artifacts" / "pairs_hazard.parquet"
    )
    out = {
        "F3_wrong_direction_mortgages": {
            "as_run": wrong_direction(as_run, vdata.DATA_DIR / "processed" / name, pl),
            "corrected_ages": wrong_direction(
                fixed, HERE / "artifacts" / "data" / "processed" / name, pl
            ),
        },
        "F4_simulation_coverage": simulation_coverage(),
        "F5_mortgage_coverage_h84": {
            "as_run": coverage_full_horizon(EXP),
            "corrected_ages": coverage_full_horizon(TREE),
        },
    }
    (HERE / "results.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
