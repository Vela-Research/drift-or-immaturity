"""Figure data for the paper as CSV files (aggregates only), for the figures drawn with figsrc/style.py.

Usage: uv run python figdata.py -> research/drift-or-maturity/figdata/*.csv
Sources: results of 0020 (patent speeds), 0031 and 0032 (errors per snapshot and per pair),
0035 (shrinkage criterion per pair).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Final

import pandas as pd

EXP: Final = Path(__file__).resolve().parents[1]
OUT: Final = EXP.parent / "research" / "drift-or-maturity" / "figdata"
NAMES: Final = {
    "naive": "Events so far",
    "drop_recent": "Drop recent",
    "fixed_horizon": "Last mature cohort",
    "chain": "Chain ladder (CL)",
    "bf": "Bornhuetter-Ferguson",
    "cape_cod": "Cape Cod",
    "gcc_075": "Generalized Cape Cod (0.75)",
    "unit_chain": "CL + exact ages",
    "stratified": "CL + exact ages + groups",
    "cal_chain": "Calendar-adjusted CL",
    "cal_chain_forecast": "Calendar-adjusted CL, forecast path",
    "ours": "Full method",
    "ours_uncond": "Full method, forecast path",
    "hazard_vintage_real": "Hazard model",
    "hazard_vintage_fcst": "Hazard model, forecast path",
    "hazard_real": "Hazard model, no vintage effects",
    "hazard_fcst": "Hazard model, no vintage effects, forecast path",
}


CRITERION: Final = [
    "D",
    "V",
    "inflation",
    "phi",
    "own_var",
    "own_var_phi",
    "omega2",
    "tau2",
    "holds",
    "holds_without_phi",
]


def patent_speeds() -> pd.DataFrame:
    """Log speed per filing year (free fit of 0020), raw and calendar-adjusted, with standard errors; `_rel` is
    relative to the mean of the filing years 2001-2008 (the scale of the figure)."""
    check = json.loads((EXP / "0020-patex-feasibility" / "results.json").read_text())["all_utility"]
    out = pd.DataFrame({"filing_year": [int(y) for y in check["raw"]["log_speed_free"]]})
    for key in ("raw", "adjusted"):
        out[f"log_speed_{key}"] = list(check[key]["log_speed_free"].values())
        out[f"se_{key}"] = list(check[key]["log_speed_se"].values())
        base = out.loc[out.filing_year <= 2008, f"log_speed_{key}"].mean()
        out[f"log_speed_{key}_rel"] = out[f"log_speed_{key}"] - base
    return out


def errors_by_snapshot(block: dict[str, dict[str, dict[str, float]]]) -> pd.DataFrame:
    """Long table: snapshot, method, mean signed and mean absolute relative error over the scored cohorts."""
    signed = pd.DataFrame(block["signed_error_by_snapshot"]).T.rename_axis("snapshot")
    absolute = pd.DataFrame(block["error_by_snapshot"]).T.rename_axis("snapshot")
    out = pd.concat(
        {"mean_signed_error": signed.stack(), "mean_abs_error": absolute.stack()}, axis=1
    ).reset_index(names=["snapshot", "key"])
    out.insert(2, "method", out.key.map(NAMES))
    return out[out.method.notna()]


def pairs(path: Path, criterion: list[dict[str, float]]) -> pd.DataFrame:
    """One row per scored (cohort, snapshot) pair: truth, every estimate, and the quantities of the shrinkage
    proposition from experiment 0035 (D events so far, V information per event, inflation factor, own_var =
    1 / (D V), own_var_phi = phi / (D V), omega2, tau2, holds = omega2 + tau2 > own_var_phi)."""
    p = pd.read_parquet(path)
    keep = ["snapshot", "cohort", "A", "h", "n", "events", "truth"] + [k for k in NAMES if k in p]
    out = p[keep + [k for k in ("set", "speed") if k in p]]
    extra = pd.DataFrame(criterion)[["snapshot", "cohort", *CRITERION]]
    return out.merge(extra, on=["snapshot", "cohort"], validate="one_to_one")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    round3 = json.loads((EXP / "0031-round3-backtests" / "results.json").read_text())
    art = EXP / "0031-round3-backtests" / "artifacts"
    hazard = EXP / "0032-mortgage-hazard-baseline"
    criterion = json.loads((EXP / "0035-speed-criterion-fisher" / "results.json").read_text())
    tables = {
        "patent_speeds": patent_speeds(),
        "patent_errors_by_snapshot": errors_by_snapshot(round3["patents_main"]),
        "patent_pairs": pairs(art / "pairs_patents_main.parquet", criterion["patents"]["per_pair"]),
        "mortgage_errors_by_snapshot": errors_by_snapshot(
            json.loads((hazard / "results.json").read_text())
        ),
        "mortgage_pairs": pairs(
            hazard / "artifacts" / "pairs_hazard.parquet", criterion["corrected"]["per_pair"]
        ),
    }
    for name, table in tables.items():
        table.to_csv(OUT / f"{name}.csv", index=False, float_format="%.6g")


if __name__ == "__main__":
    main()
