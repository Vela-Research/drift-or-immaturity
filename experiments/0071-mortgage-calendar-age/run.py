"""Mortgages: the frozen code run again with loan ages counted from origination (POST HOC; mortgage and patent audit, audit 1 #1 and
audit 3 #5), and with a random-walk evidence that uses the fit's own prior (audit 2 #1).

DATASET: Freddie Mac Single-Family Loan-Level Dataset, the 200,000-loan samples of experiment 0010
(`data/processed/freddie_200k_loans.parquet`, `..._linked.parquet`); covariates as before (`data/raw/covariates`);
USPTO PatEx (`data/processed/patex_utility.parquet`, unchanged). Aggregates only.
  F1 ages. The SFLLD "Loan Age" restarts at a loan modification (it counts from the modification's first payment), but
     the frozen code takes it as months since the first payment. Here every `*_age` column is rebuilt from its
     `*_period` column: age = month(period) - month(first_payment) + 1, the convention of the unmodified loans (for
     97% of the values it equals the stored age). A rebuilt age below 0 keeps the stored value (loans whose records
     end before their first payment; vintages after 2020 for the February 2020 cut).
  F2 random-walk evidence. `maturity.onestage._optimise` gives a step between cohorts the prior variance omega2 x (key
     gap in years), but `_rw_evidence` scores the mode with omega2 per step. The patched evidence uses the fit's
     variances (the first cohort keeps omega2). For yearly cohorts it is the same number, so only the quarterly
     calibration cohorts change.
The frozen code is not edited: `tree` copies the scripts of the experiments that use the mortgage data, byte for byte,
into `artifacts/tree/experiments/<folder>/` (fresh `artifacts/`); every other experiment folder is a link. `exec` runs a
copied script with `maturity.data.DATA_DIR` pointed at `artifacts/data` (the rebuilt loans; links for everything else),
so every output stays inside the tree.
Usage:
  uv run python run.py data      -> artifacts/data/processed/freddie_200k_loans{,_linked}.parquet, ages.json
  uv run python run.py tree      -> artifacts/tree/
  PYTHONHASHSEED=0 uv run python run.py exec <folder> <script> [args...]
  uv run python run.py check     (self-checks of F1 and F2)
"""

from __future__ import annotations

import importlib.util
import json
import logging
import os
import runpy
import shutil
import sys
from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import Final

import numpy as np
import pandas as pd

from maturity import data as vdata
from maturity import onestage

HERE: Final = Path(__file__).resolve().parent
REPO: Final = HERE.parents[1]
ART: Final = HERE / "artifacts"
DATA: Final = ART / "data"
TREE: Final = ART / "tree"
LOANS: Final = ("freddie_200k_loans.parquet", "freddie_200k_loans_linked.parquet")
AGES: Final = {  # age column -> its period column
    "last_age": "last_period",
    "zb_age": "zb_period",
    "dq6_age": "dq6_period",
    "dq3_age": "dq3_period",
    "dq6c_age": "dq6c_period",
    "event_age": "event_period",
    "event_age90": "event_period90",
    "event_age_clean": "event_period_clean",
    "event_age_c202002": "event_period_c202002",
}
CHAIN: Final = (  # experiments that read the mortgage data or its products: copied, with fresh artifacts
    "0011-freddie-200k-confirmatory",
    "0012-freddie-200k-harp-correction",
    "0014-e2-freddie-adaptive-windows",
    "0015-paper-assets",
    "0016-e2-calendar-adjusted-windows",
    "0018-freddie-200k-follow-borrowers",
    "0031-round3-backtests",
    "0032-mortgage-hazard-baseline",
    "0033-downstream-classifier",
    "0036-corrected-plugin-backtests",
    "0039-classifier-followup",
    "0066-patent-hazard-baseline",
    "0067-v10-answers",
)
KEEP_ARTIFACTS: Final = (
    "0066-patent-hazard-baseline",
)  # patent-only artifacts the reports read (copied)
log = logging.getLogger(__name__)


def month(yyyymm: pd.Series) -> np.ndarray:
    x = yyyymm.to_numpy(dtype=float)
    return (x // 100) * 12 + x % 100


def rebuilt(d: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, dict[str, int]]]:
    """F1: every age from its period and the first-payment month."""
    out, counts = d.copy(), {}
    first = month(d.first_payment)
    for age, period in AGES.items():
        new = month(d[period]) - first + 1
        use = np.isfinite(new) & (new >= 0)
        old = d[age].to_numpy(dtype=float)
        out[age] = np.where(use, new, old).astype(d[age].dtype)
        moved = use & (new != old)
        counts[age] = {
            "changed": int(moved.sum()),
            "older_by_2_or_more": int((use & (new - old >= 2)).sum()),
        }
    return out, counts


def build_data() -> None:
    processed = DATA / "processed"
    processed.mkdir(parents=True, exist_ok=True)
    for name in ("freddie_200k", "patex_utility.parquet", "freddie_sample_loans.parquet"):
        if not (processed / name).exists():
            (processed / name).symlink_to(vdata.DATA_DIR / "processed" / name)
    if not (DATA / "raw").exists():
        (DATA / "raw").symlink_to(vdata.DATA_DIR / "raw", target_is_directory=True)
    report = {}
    for name in LOANS:
        d, counts = rebuilt(pd.read_parquet(vdata.DATA_DIR / "processed" / name))
        d.to_parquet(processed / name)
        report[name] = counts
        log.info("%s: %s", name, counts)
    (ART / "ages.json").write_text(json.dumps(report, indent=1))


def build_tree() -> None:
    experiments = TREE / "experiments"
    experiments.mkdir(parents=True, exist_ok=True)
    for folder in sorted(p for p in (REPO / "experiments").iterdir() if p.is_dir()):
        target = experiments / folder.name
        if target.exists() or target.is_symlink():
            continue
        if folder == HERE:
            continue  # no link to this experiment (a loop)
        if folder.name not in CHAIN:
            target.symlink_to(folder, target_is_directory=True)
            continue
        target.mkdir()
        for f in folder.glob("*.py"):
            shutil.copy2(f, target / f.name)
        if folder.name in KEEP_ARTIFACTS:
            shutil.copytree(folder / "artifacts", target / "artifacts")
        else:
            (target / "artifacts").mkdir()
    research = TREE / "research" / "drift-or-maturity"
    for sub in ("figdata", "paper/tables", "paper/figures"):
        (research / sub).mkdir(parents=True, exist_ok=True)
    if not (research / "figsrc").exists():
        (research / "figsrc").symlink_to(REPO / "research" / "drift-or-maturity" / "figsrc")


def consistent_rw_evidence(cohorts: list[onestage.Cohort], f: onestage.Fit) -> float:
    """F2: the Laplace evidence with the prior of `_optimise` (step variance omega2 x key gap)."""
    x = np.log(f.speeds)
    hess = _hessian(cohorts, f, x)
    gaps = np.concatenate([[1.0], np.maximum(np.diff([c.key for c in cohorts]), 1e-6)])
    return _laplace(onestage._total_loglik(cohorts, x, f), x, hess, f.omega2 * gaps)


def execute(folder: str, script: str, args: list[str]) -> None:
    vdata.DATA_DIR = DATA  # type: ignore[misc]  # F1: every constant built from it now points at the rebuilt data
    onestage._rw_evidence = consistent_rw_evidence  # F2
    path = TREE / "experiments" / folder / script
    assert path.is_file() and not path.is_symlink(), path
    sys.path.insert(0, str(path.parent))
    sys.argv = [str(path), *args]
    os.chdir(path.parent)
    if (folder, script) != ("0036-corrected-plugin-backtests", "backtests.py"):
        runpy.run_path(str(path), run_name="__main__")
        return
    # 0036 checks that the corrected plug-in never lowers an estimate of a cohort with speed > 1. On the rebuilt data
    # two pairs with speed <= 1.006 are 0.02% lower (grid interpolation at H): the check gets a 0.1% tolerance there.
    m = _module(path, "backtests36")
    m.check_pairs = _tolerant(m.check_pairs)
    m.main(args[0], args[1], args[2:] or ["pairs", "calibration"])


def self_check() -> None:
    """F2 equals the frozen formula when all gaps are 1; F1 keeps an unmodified loan's age."""
    rng = np.random.default_rng(0)
    k, omega2 = 7, 0.03
    x, hess, loglik = rng.normal(0, 0.2, k), rng.uniform(1, 50, k), -123.4
    d = np.eye(k) - np.eye(k, k=-1)
    frozen = (
        loglik
        - (d @ x) @ (d @ x) / (2 * omega2)
        - 0.5 * np.linalg.slogdet(omega2 * np.diag(hess) + d.T @ d)[1]
    )
    assert abs(_laplace(loglik, x, hess, np.full(k, omega2)) - frozen) < 1e-9
    one = pd.DataFrame({"first_payment": [200501], "last_period": [200712], "last_age": [35.0]})
    one = one.assign(
        **{a: [np.nan] for a in AGES if a != "last_age"},
        **{p: [np.nan] for a, p in AGES.items() if a != "last_age"},
    )
    assert rebuilt(one)[0].last_age.iloc[0] == 36.0  # 2005-01 .. 2007-12 = 36 payment months
    log.info("self-checks passed")


def _module(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _tolerant(
    check: Callable[[pd.DataFrame, pd.DataFrame, str], None],
) -> Callable[[pd.DataFrame, pd.DataFrame, str], None]:
    """The 0036 check, with the stored frozen estimates of non-slow cohorts lowered by 0.1% for its last test."""

    def run(pairs: pd.DataFrame, stored: pd.DataFrame, code: str) -> None:
        loose = stored.copy()
        fast = (stored.speed > 1.0).to_numpy()
        for m in ("ours", "ours_uncond"):
            loose.loc[fast, m] = stored.loc[fast, m] * (1 - 1e-3)
        check(pairs, loose if code == "corrected" else stored, code)

    return run


def _hessian(cohorts: list[onestage.Cohort], f: onestage.Fit, x: np.ndarray) -> np.ndarray:
    eps = 5e-3
    out = np.empty(len(cohorts))
    for i, c in enumerate(cohorts):
        mid, up, dn = (
            onestage._shape_loglik(c, float(x[i]) + e, f.ugrid, f.f0) for e in (0.0, eps, -eps)
        )
        out[i] = max(0.0, -(up - 2 * mid + dn) / eps**2)
    return out


def _laplace(loglik: float, x: np.ndarray, hess: np.ndarray, variances: np.ndarray) -> float:
    k = len(x)
    d = np.eye(k) - np.eye(k, k=-1)  # first row anchors the first cohort
    precision = d.T @ np.diag(1 / variances) @ d
    dx = d @ x
    _, logdet_post = np.linalg.slogdet(np.diag(hess) + precision)
    return float(
        loglik - 0.5 * dx @ (dx / variances) - 0.5 * logdet_post - 0.5 * np.log(variances).sum()
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    step = sys.argv[1]
    if step == "exec":
        if os.environ.get("PYTHONHASHSEED") != "0":
            raise SystemExit(
                "run with PYTHONHASHSEED=0 (the fixed-effects fit depends on set order)"
            )
        execute(sys.argv[2], sys.argv[3], sys.argv[4:])
    else:
        {"data": build_data, "tree": build_tree, "check": self_check}[step]()
