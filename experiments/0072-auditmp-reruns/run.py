"""Decisions after the mortgage and patent audit (POST HOC; the decisions after the mortgage and patent audit): the runs that the
paper still prints from the old loan ages, the replication's calibration with F2, and the end-to-end simulation with
the change point uniform over the last eight cohorts.

DATASET: Freddie Mac Single-Family Loan-Level Dataset, the 200,000-loan samples of experiment 0010 with the ages
rebuilt by experiment 0071 (`0071-mortgage-calendar-age/artifacts/data`); the HARP parents' origination records of
0017 (cached); USPTO PatEx (`data/processed/patex_utility.parquet`); the replication parquet files of 0068/0069 (SBA 7(a) FOIA, FJC civil, BLS LAUS, FHFA HPI); simulated data
(0034, 0037). Aggregates only.
Patches (named in EXP0072_PATCH; `boot/sitecustomize.py` applies them at start-up in every process, workers included):
  f1  `maturity.data.DATA_DIR` -> artifacts/data (links to the rebuilt loans of 0071; links for everything else).
  f2  the random-walk evidence of 0071 (step variance omega2 x key gap; the same number for yearly cohorts).
  cp  the change point of `maturity.onestage.select_model`: start uniform over the last 8 cohorts, as FROZEN_SPEC and the
      paper state. The frozen code allows only the immature cohorts among the last 8 (audit 2 #8). Here every cohort
      is passed as immature; `_level` gives a mature cohort its observed incidence either way (x[-1] = x_h), and
      `_shape_loglik` does not read the flag.
The frozen code is not edited: `tree` builds artifacts/tree/experiments with
  - the 13 experiments of 0071's chain cloned from 0071's tree (scripts and their corrected outputs; APFS clones),
  - the experiments re-run here (NEW) as copies of their scripts with a fresh `artifacts/`,
  - every other experiment as a copy of its top-level files with links to its subfolders,
so that every path a script builds from its own location stays inside the tree.
Usage:
  uv run python run.py check | data | tree
  uv run python run.py exec <patches> <folder> <script> [args...]   (patches: e.g. f1,f2 or f2,cp)
  PYTHONHASHSEED=0 uv run python run.py rep <asrun|corrected> <sba|civil>   (F2 calibration of the replication)
  uv run python run.py simulation <stored|rerun>   -> artifacts/simulation_<label>.json
  uv run python run.py report      -> results.json
"""

from __future__ import annotations

import importlib.util
import json
import logging
import multiprocessing as mp
import os
import shutil
import subprocess
import sys
from collections.abc import Callable, Sequence
from dataclasses import replace
from pathlib import Path
from types import ModuleType
from typing import Final

import numpy as np
import pandas as pd

from maturity import data as vdata
from maturity import inference, onestage

HERE: Final = Path(__file__).resolve().parent
EXP: Final = HERE.parent
ART: Final = HERE / "artifacts"
DATA: Final = ART / "data"
TREE: Final = ART / "tree"
BOOT: Final = HERE / "boot"
MP: Final = EXP / "0071-mortgage-calendar-age"
NEW: Final = (  # re-run here: fresh artifacts
    "0015-paper-assets",
    "0017-harp-informative-censoring",
    "0030-review2-answers",
    "0034-end-to-end-simulation",
    "0035-speed-criterion-fisher",
    "0037-corrected-plugin-simulation",
    "0040-hazard-other-estimands",
    "0041-feature-maturity",
)
REP: Final = {  # replication: where the inputs and the scored pairs of each version are
    "asrun": EXP / "0068-confirm2-replication" / "artifacts",
    "corrected": EXP / "0069-confirm2-audit-fixes" / "artifacts",
}
log = logging.getLogger(__name__)


def module(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


def uniform_change_point(
    frozen: Callable[[Sequence[onestage.Cohort], onestage.Fit], tuple[onestage.Fit, float]],
) -> Callable[[Sequence[onestage.Cohort], onestage.Fit], tuple[onestage.Fit, float]]:
    def run(cohorts: Sequence[onestage.Cohort], chain: onestage.Fit) -> tuple[onestage.Fit, float]:
        return frozen([replace(c, mature=False) for c in cohorts], chain)

    return run


def apply(patches: list[str]) -> None:
    unknown = set(patches) - {"f1", "f2", "cp"}
    assert not unknown, unknown
    if "f1" in patches:
        vdata.DATA_DIR = DATA  # type: ignore[misc]
    if "f2" in patches:
        onestage._rw_evidence = module(MP / "run.py", "run71").consistent_rw_evidence
    if "cp" in patches:
        onestage._change_point = uniform_change_point(onestage._change_point)
    os.environ["EXP0072_APPLIED"] = ",".join(sorted(patches))


def build_data() -> None:
    for sub in ("processed", "raw"):
        source = MP / "artifacts" / "data" / sub
        if sub == "raw":
            if not (DATA / "raw").exists():
                DATA.mkdir(parents=True, exist_ok=True)
                (DATA / "raw").symlink_to(source.resolve(), target_is_directory=True)
            continue
        (DATA / sub).mkdir(parents=True, exist_ok=True)
        for f in source.iterdir():
            if not (DATA / sub / f.name).exists():
                (DATA / sub / f.name).symlink_to(f.resolve())


def build_tree() -> None:
    chain = module(MP / "run.py", "run71").CHAIN
    experiments = TREE / "experiments"
    experiments.mkdir(parents=True, exist_ok=True)
    for folder in sorted(p for p in EXP.iterdir() if p.is_dir()):
        target = experiments / folder.name
        if folder in (HERE, MP) or target.exists() or target.is_symlink():
            continue
        if folder.name in chain and folder.name not in NEW:
            _clone(MP / "artifacts" / "tree" / "experiments" / folder.name, target)
            continue
        target.mkdir()
        if folder.name in NEW:
            for f in folder.glob("*.py"):
                shutil.copy2(f, target / f.name)
            (target / "artifacts").mkdir()
            continue
        for f in folder.iterdir():
            if f.is_dir() and f.name != "__pycache__":
                (target / f.name).symlink_to(f, target_is_directory=True)
            elif f.is_file():
                shutil.copy2(f, target / f.name)
    if not (TREE / "research").exists():
        _clone(MP / "artifacts" / "tree" / "research", TREE / "research")


def execute(patches: str, folder: str, script: str, args: list[str]) -> None:
    path = TREE / "experiments" / folder / script
    assert path.is_file() and not path.is_symlink(), path
    env = {
        **os.environ,
        "PYTHONHASHSEED": "0",
        "PYTHONPATH": str(BOOT),
        "EXP0072_PATCH": patches,
    }
    subprocess.run([sys.executable, script, *args], cwd=path.parent, env=env, check=True)


def replication(version: str, name: str) -> None:
    """F2 calibration errors of the replication (0068 `backtest.run_calibration`), pairs as stored."""
    out = ART / "rep" / version
    if (version, name) == (
        "corrected",
        "sba",
    ):  # 0069 links the SBA inputs and pairs of 0068: the same run
        out.mkdir(parents=True, exist_ok=True)
        (out / name).symlink_to(ART / "rep" / "asrun" / name, target_is_directory=True)
        return
    apply(["f2"])
    (out / name).mkdir(parents=True, exist_ok=True)
    for f in ("sba.parquet", "civil.parquet", "judges.parquet", "covariates.parquet"):
        if not (out / f).exists():
            (out / f).symlink_to((REP[version] / f).resolve())
    if not (out / name / "pairs.parquet").exists():
        (out / name / "pairs.parquet").symlink_to((REP[version] / name / "pairs.parquet").resolve())
    fixes = module(EXP / "0069-confirm2-audit-fixes" / "run.py", "run69")
    fixes.ART = out
    fixes.as_run("backtest").main(name, ["calibration"])


def replication_coverage() -> dict[str, object]:
    """0068 `report.coverage` with the F2 calibration errors: all pairs (as 0068) and h = H only (C4 of 0069)."""
    fixes = module(EXP / "0069-confirm2-audit-fixes" / "run.py", "run69")
    r = fixes.as_run("report")
    r36 = module(EXP / "0036-corrected-plugin-backtests" / "report.py", "report36")
    out: dict[str, object] = {}
    for version in REP:
        r.ART = ART / "rep" / version
        for name in ("sba", "civil"):
            p = pd.read_parquet(r.ART / name / "pairs.parquet")
            for label, keep in (("all pairs (0068)", p.h > 0), ("h = H (C4)", p.h == r.H[name])):
                out[f"{version} {name} {label}"] = {
                    k: v
                    for k, v in r.coverage(name, p[keep], r36).items()
                    if k == "calibration_errors" or "strict J>=9" in k
                }
    return out


def simulation(label: str) -> None:
    """Exact simulation numbers from the parts (`stored`: 0034/0037 as run; `rerun`: the tree): per setting, the error
    ratio of the full method to equal speeds (realised path) and the coverage of its interval at J >= 1 and J >= 9;
    the smallest Gamma with 90% at J >= 9 (F4), by setting and pooled over step and drift. One root per process: the
    0037 modules import 0034's `sim` and `report` by name."""
    root = EXP if label == "stored" else TREE / "experiments"
    sys.path.insert(0, str(root / "0037-corrected-plugin-simulation"))
    compare = module(root / "0037-corrected-plugin-simulation" / "compare.py", "compare37")
    out: dict[str, object] = {}
    for version, folder in compare.VERSIONS.items():
        young = compare.load(folder)[0].query("young")
        parts = young.setting.str.split("/", expand=True)
        speeds = parts[1].where(parts[1] == "constant", "step+drift")
        young = young.assign(group=parts[0] + "/" + speeds + "/" + parts[2] + "/" + parts[3])
        block: dict[str, object] = {}
        for name, g in [*young.groupby("setting"), *young.groupby("group")]:
            d = inference.log_ratio(
                compare.report.relative_error(g, "ours").to_numpy(),
                compare.report.relative_error(g, "cal_chain").to_numpy(),
            )
            covered = (g.ours_low <= g.truth) & (g.truth <= g.ours_high)
            block[name] = {
                "ratio_full_vs_equal": float(np.exp(d.mean())),
                **{
                    f"coverage J>={j}": [
                        int(covered[g.ours_J >= j].sum()),
                        int((g.ours_J >= j).sum()),
                    ]
                    for j in (1, 9)
                },
                "gamma J>=9": compare.gamma_coverage(g[g.ours_J >= 9], "ours"),
            }
        out[version] = block
    (ART / f"simulation_{label}.json").write_text(json.dumps(out, indent=1, default=str))


def report() -> None:
    sims = {
        label: json.loads((ART / f"simulation_{label}.json").read_text())
        for label in ("stored", "rerun")
    }
    (HERE / "results.json").write_text(
        json.dumps(
            {"replication_coverage_f2": replication_coverage(), "simulation": sims}, indent=1
        )
    )


def _clone(source: Path, target: Path) -> None:
    """APFS copy-on-write clone (macOS `cp -c`): an independent copy that costs no space."""
    subprocess.run(["cp", "-Rc", str(source), str(target)], check=True)


def _patched_in_worker() -> tuple[str, bool]:
    return os.environ.get("EXP0072_APPLIED", ""), onestage._change_point.__name__ == "run"


def self_check() -> None:
    """The cp patch equals the frozen change point when the last 8 cohorts are immature, adds the mature ones as
    starts otherwise and keeps their levels; the start-up hook reaches spawned workers."""
    rng = np.random.default_rng(0)
    x = np.linspace(0, 1, 13)

    def cohort(key: int, age: int) -> onestage.Cohort:
        g = np.cumsum(rng.uniform(0, 0.01, age + 1)) * (x[: age + 1] > 0)
        return onestage.Cohort(float(key), x[: age + 1], g, 200.0, 1.0, age == 12)

    cohorts = [cohort(k, a) for k, a in enumerate((12, 12, 12, 12, 12, 12, 9, 6, 3))]
    chain = onestage.fit(cohorts, 0.0)
    frozen, patched = onestage._change_point, uniform_change_point(onestage._change_point)
    young = cohorts[-3:]  # the last 8 (all of them) are immature
    a, b = frozen(young, chain), patched(young, chain)
    assert a[1] == b[1] and np.array_equal(a[0].speeds, b[0].speeds)
    a, b = frozen(cohorts, chain), patched(cohorts, chain)
    assert a[1] != b[1]  # three more starts (the mature cohorts among the last 8)
    assert np.allclose(b[0].level[:6], [c.g[-1] for c in cohorts[:6]], rtol=1e-12, atol=0)
    env = {"PYTHONPATH": str(BOOT), "EXP0072_PATCH": "f2,cp"}
    os.environ.update(env)
    with mp.get_context("spawn").Pool(1) as pool:
        applied, cp = pool.apply(_patched_in_worker)
    assert applied == "cp,f2" and cp, (applied, cp)
    log.info("self-checks passed")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    step = sys.argv[1]
    if step == "exec":
        execute(sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5:])
    elif step == "simulation":
        simulation(sys.argv[2])
    elif step == "rep":
        if os.environ.get("PYTHONHASHSEED") != "0":
            raise SystemExit(
                "run with PYTHONHASHSEED=0 (the fixed-effects fit depends on set order)"
            )
        replication(sys.argv[2], sys.argv[3])
    else:
        {"check": self_check, "data": build_data, "tree": build_tree, "report": report}[step]()
