"""CONFIRM2 audit (the neutral audit of the replication code): POST HOC corrections of exp 0068.

DATASET: the parquet files of exp 0068 (SBA 7(a) FOIA as of 2026-06-30; FJC civil cumulative file to 2026-06-30; FJC
judicial service; BLS LAUS state unemployment (FRED copies) and FHFA state HPI). Aggregates only.
The 0068 code is imported unchanged, one module per process (their frozen imports share module names). This file
replaces only what the audit found wrong (the audit's findings):
  C1 civil units: one record per case = the first record of (district, office, docket, filing date) BEFORE the
     filters (spec §3 "the original filing"). 0068 filtered first, so a case whose original record fails a filter
     kept a later record (audit 1, finding 6). Applied by deduplicating the input file; 0068's own rule is then a
     no-op.
  C2 classifier training set (spec §4.2): every unit of the 10 entry years up to S minus the test half of the
     scored cohorts. 0068 also left out the test half of the cohorts not scored (audit 1 #22, audit 2 #1).
  C3 windows: kappa = z_{alpha/N} = Phi^-1(1 - alpha/N) (spec §7.2), for both Lepski rules. 0068 used the frozen
     multiplier sqrt(2 log(N/alpha)) (audit 2 #3).
  C4 coverage only for pairs whose target is the calibration target (h = H): the calibration errors are 72-month
     errors, and the SBA pairs cut at 2020-02 with h < 72 got intervals from them (audit 2 #2).
  C5 every sign-flip p value with the spec's vectors (all 2^C, else 10^6 random, seed 20261025) (audit 2 #4).
  C6 Holm on unrounded p values (audit 2 #5).
  P1 check (audit 1 #24): the redated payoffs of 0068 `pif_check.py`, compared on every method column.
SBA runs not touched by C1-C3 (pairs, units, calibration, hazard model, detectors) are linked from 0068.
Usage (from this folder, one at a time; the fixed-effects fits need PYTHONHASHSEED=0):
  uv run python run.py link
  PYTHONHASHSEED=0 uv run python run.py <backtest|hazard> <civil|civil_broad>
  PYTHONHASHSEED=0 uv run python run.py <classify|windows> <sba|civil>
  PYTHONHASHSEED=0 uv run python run.py drift civil
  PYTHONHASHSEED=0 uv run python run.py pif
  uv run python run.py report        -> results.json
  uv run python run.py side          -> side_by_side.md (as-run 0068 against corrected)
"""

from __future__ import annotations

import functools
import importlib.util
import json
import logging
import math
import os
import sys
from pathlib import Path
from types import ModuleType
from typing import Final

import numpy as np
import pandas as pd
from scipy.stats import norm

from maturity import inference, windows

HERE: Final = Path(__file__).resolve().parent
AS_RUN: Final = HERE.parent / "0068-confirm2-replication"
ART: Final = HERE / "artifacts"
KEY: Final = ["district", "office", "docket", "filedate"]
LINKED_INPUTS: Final = ("sba.parquet", "judges.parquet", "covariates.parquet")
LINKED_SBA: Final = (
    "pairs.parquet",
    "units.parquet",
    "calibration.parquet",
    "hazard.parquet",
    "meta.json",
    "alarms.parquet",
    "streams.parquet",
)
ESTIMATES: Final = [  # every estimate column of the pairs, and the truths
    "truth",
    "ref",
    "naive",
    "drop_recent",
    "fixed_horizon",
    "bf",
    "cape_cod",
    "gcc_075",
    "gcc_050",
    "chain",
    "unit_chain",
    "stratified",
    "cal_chain",
    "cal_chain_forecast",
    "ours_free",
    "ours_cal",
    "ours_cal_uncond",
    "ours",
    "ours_uncond",
]
TWO_SIDED: Final = ("sign_flip_p", "two_sided_sign_flip_p_of_ratio_1")
sys.path.insert(0, str(AS_RUN))
log = logging.getLogger(__name__)


def frozen_data() -> ModuleType:
    """0068 `data` (it loads the frozen pipeline), pointed at this experiment's artifacts."""
    import data

    data.ART = ART  # type: ignore[misc]  # inputs are linked or corrected copies (see `link`)
    return data


def as_run(module: str) -> ModuleType:
    """One 0068 module, loaded from its file (0012, first on the path after `data`, has a `report.py` too)."""
    frozen_data()
    spec = importlib.util.spec_from_file_location(module, AS_RUN / f"{module}.py")
    assert spec is not None and spec.loader is not None
    m = importlib.util.module_from_spec(spec)
    sys.modules[module] = m
    spec.loader.exec_module(m)
    return m


def link() -> None:
    (ART / "sba" / "parts").mkdir(parents=True, exist_ok=True)
    for f in LINKED_INPUTS:
        if not (ART / f).exists():
            (ART / f).symlink_to(AS_RUN / "artifacts" / f)
    for f in LINKED_SBA:
        if not (ART / "sba" / f).exists():
            (ART / "sba" / f).symlink_to(AS_RUN / "artifacts" / "sba" / f)
    d = pd.read_parquet(AS_RUN / "artifacts" / "civil.parquet")
    first = d[~d.duplicated(KEY, keep="first")]  # C1: the original filing, before any filter
    first.to_parquet(ART / "civil.parquet")
    data = frozen_data()
    data.ART = AS_RUN / "artifacts"
    before = len(data.civil_units())
    data.ART = ART
    after = len(data.civil_units())
    assert not first.duplicated(KEY).any() and after <= before, (before, after)
    log.info(
        "C1: %d records -> %d first records; civil cases %d -> %d",
        len(d),
        len(first),
        before,
        after,
    )


def classify_on_spec_set(m: ModuleType) -> None:
    """C2: only the scored cohorts' test half leaves the training set."""
    original = m.snapshot

    def snapshot(
        u: pd.DataFrame,
        is_test: np.ndarray,
        name: str,
        year: int,
        cohorts: list[float],
        share: np.ndarray,
        units: pd.DataFrame,
    ) -> tuple[pd.DataFrame, dict[str, object]]:
        scored = is_test & u.year.isin(cohorts).to_numpy()
        return original(u, scored, name, year, cohorts, share, units)

    m.snapshot = snapshot


def z_quantile(cohorts: int, alpha: float = 0.05) -> float:
    """C3: kappa = z_{alpha/N}, the upper alpha/N quantile of the standard normal."""
    return float(norm.ppf(1 - alpha / cohorts))


def report() -> None:
    r = as_run("report")
    r.ART, r.HERE = ART, HERE  # results.json is written here
    signs = functools.lru_cache(maxsize=None)(r.signs)
    # C5: the spec's vectors for every sign-flip p value
    inference._signs = lambda c, seed=0: signs(c)
    hypothesis, coverage = r.hypothesis, r.coverage

    def exact(
        p: pd.DataFrame, a: str, b: str, margin: float, tail: str, truth: str = "truth"
    ) -> dict[str, object]:
        out = hypothesis(p, a, b, margin, tail, truth)
        d = inference.log_ratio(r.error(p, a, truth), r.error(p, b, truth))
        cohorts = p.cohort.to_numpy().astype(np.int64)
        out["p_one_sided"] = r.one_sided(d, cohorts, math.log(margin), tail)  # C6: unrounded
        return out

    def full_horizon(name: str, p: pd.DataFrame, r36: ModuleType) -> dict[str, object]:
        return coverage(name, p[p.h == r.H[name]], r36)  # C4

    r.hypothesis, r.coverage = exact, full_horizon
    r.main()


def side() -> None:
    """Key numbers of 0068 (as run) and of this experiment (corrected), side by side."""
    a = json.loads((AS_RUN / "results.json").read_text())
    b = json.loads((HERE / "results.json").read_text())

    def test(x: dict[str, object]) -> str:
        lo, hi = x["ci95"]  # type: ignore[misc]
        holm = f", Holm {x['p_holm']:.4g}, {x['verdict']}" if "p_holm" in x else ""
        return f"{x['error_ratio']:.3f} [{lo:.3f}, {hi:.3f}], p {float(x['p_one_sided']):.4g}{holm}"  # type: ignore[arg-type]

    def two(x: dict[str, object]) -> str:
        lo, hi = x["ci95"]  # type: ignore[misc]
        return f"{x['error_ratio']:.3f} [{lo:.3f}, {hi:.3f}], two-sided p {x['sign_flip_p']}"

    rows = ["| item | as run (0068) | corrected (0069) |", "|---|---|---|"]
    for name in ("sba", "civil"):
        for h in ("H1", "H2", "H3"):
            rows.append(
                f"| {name} {h} | {test(a[name]['confirmatory'][h])} | {test(b[name]['confirmatory'][h])} |"
            )
    for h in ("H2_procedure", "H3_procedure"):
        rows.append(
            f"| civil_broad {h} | {test(a['civil_broad'][h])} | {test(b['civil_broad'][h])} |"
        )
    for name in ("sba", "civil"):
        for method in ("ours_uncond", "chain"):
            key = f"{method} strict J>=9"
            x, y = a[name]["coverage"][key], b[name]["coverage"][key]
            rows.append(
                f"| {name} coverage {key} (Γ=1; smallest Γ for 90%) | {x['covered']['1.0']}; "
                f"{x['smallest_gamma_for_90pct']} | {y['covered']['1.0']}; {y['smallest_gamma_for_90pct']} |"
            )
        for frac in ("10%", "30%", "100%"):
            x, y = (s[name]["secondary"]["windows"][frac]["median_error"] for s in (a, b))
            rows.append(
                f"| {name} windows {frac}: naive Lepski / maturity-weighted (z) | "
                f"{x['naive_lepski']:.3f} / {x['corrected_z']:.3f} | {y['naive_lepski']:.3f} / {y['corrected_z']:.3f} |"
            )
        x, y = (s[name]["secondary"]["classifier_c"]["pu_full_vs_pu_cl"] for s in (a, b))
        rows.append(f"| {name} classifier (c), pu_full / pu_cl | {two(x)} | {two(y)} |")
    # C5 alone: the SBA pairs are unchanged, so only the sign vectors differ there
    sba = (_p_values(s["sba"]["table2"]) for s in (a, b))
    gaps = [abs(x - y) for x, y in zip(*sba, strict=True)]
    rows.append(f"| SBA Table 2: largest change of a two-sided p (C5 alone) | | {max(gaps):.4f} |")
    (HERE / "side_by_side.md").write_text("\n".join(rows) + "\n")
    print("\n".join(rows))


def main(step: str, name: str | None) -> None:
    if step in ("link", "report", "side"):
        {"link": link, "report": report, "side": side}[step]()
        return
    if os.environ.get("PYTHONHASHSEED") != "0":
        raise SystemExit("run with PYTHONHASHSEED=0 (the fixed-effects fit depends on set order)")
    if step == "backtest":
        as_run("backtest").main(name, ["pairs", "calibration"] if name == "civil" else ["pairs"])
    elif step == "hazard":
        as_run("hazard_m1").main(name)
    elif step == "classify":
        m = as_run("classify")
        classify_on_spec_set(m)
        m.main(name)
    elif step in ("windows", "drift"):
        windows.gaussian_multiplier = z_quantile  # C3 (the detectors do not use it)
        as_run("secondary").main(name, step)
    elif step == "pif":
        as_run("backtest")  # first: the frozen pipeline has put 0005's `backtest` under that name
        m = as_run("pif_check")
        m.METHODS = ESTIMATES
        m.main()
    else:
        raise SystemExit(f"unknown step {step}")


def _p_values(tree: object) -> list[float]:
    """Every two-sided sign-flip p value of a results tree, in a fixed order."""
    if isinstance(tree, dict):
        own = [float(v) for k, v in tree.items() if k in TWO_SIDED]
        return own + [p for k, v in tree.items() if k not in TWO_SIDED for p in _p_values(v)]
    return []


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
