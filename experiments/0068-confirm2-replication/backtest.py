"""CONFIRM2 §4: the frozen methods on A (SBA) and B (civil), with the corrected plug-in (PRE-REGISTERED).

DATASET: the units of data.py (the paper's table of deviations). The code is frozen and only called:
  methods      `extras.snapshot` of experiments/0031 (the frozen pipeline of 0012 and the round 3 baselines) after
               `corrected.patch` of 0036, the patent run's setting. Realised path: `ours`, `cal_chain`; forecast path
               (covariates at their group's mean of the last 12 months): `ours_uncond`, `cal_chain_forecast`.
  pairs        the rule of 0031 `freddie.py`: young cohorts with a truth, at least `min_ahead` months between S and
               the truth horizon, the N youngest. A: snapshots 2000-2013 with the 72-month truth (min_ahead 1) and
               2014-2019 with the truth cut at 2020-02 (E3b, min_ahead 12), N = 6. B: snapshots 1994-2016, 36-month
               truth, N = 3. `ref` = the latest mature cohort's incidence at the pair's own horizon h (decision A3).
  units        per unit of the young cohorts at the uncut snapshots: loans, incidence seen at S and the full method's
               forecast-path cumulative hazard to H (0033 `prepare.unit_futures`, checked against `ours_uncond`);
               `share_cl` (in meta.json) = the chain ladder's timing curve at ages 0..H, for classifier (b).
  calibration  out-of-sample errors of quarterly cohorts at the December pseudo-snapshots (0036 `errors`).
Checkpoints: artifacts/<name>/parts. Usage: PYTHONHASHSEED=0 uv run python backtest.py <sba|civil|civil_broad>
[pairs] [calibration] -> artifacts/<name>/{pairs,units,calibration}.parquet, meta.json
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd

import data

HERE: Final = Path(__file__).resolve().parent
EXP: Final = HERE.parent
sys.path[:0] = [
    str(EXP / "0033-downstream-classifier"),
    str(EXP / "0036-corrected-plugin-backtests"),
    str(EXP / "0031-round3-backtests"),
]
import backtests as b36  # `errors` of the calibration step
import corrected
import extras
import prepare  # `unit_futures`

from maturity import onestage

log = logging.getLogger(__name__)
pl = data.pl
Plan = list[tuple[int, int | None, int]]  # (snapshot, truth cutoff, least months ahead)
PLANS: Final[dict[str, Plan]] = {
    "sba": [(y, None, 1) for y in range(2000, 2014)]
    + [(y, data.CUT_A, 12) for y in range(2014, 2020)],
    "civil": [(y, None, 1) for y in range(1994, 2017)],
}
N_YOUNG: Final = {"sba": 6, "civil": 3}
CALIBRATION: Final = {"sba": range(2000, 2019), "civil": range(1994, 2016)}  # pseudo-snapshots
TRUTH_END: Final = {"sba": 2020 + 2 / 12, "civil": 2026.5}  # in years: the last month with outcomes


def kind(name: str) -> str:
    return "sba" if name == "sba" else "civil"


def pairs_at(
    d: pl.Data, name: str, year: int, cutoff: int | None, min_ahead: int
) -> tuple[pd.DataFrame, pd.DataFrame | None, dict[str, object]]:
    """Scored pairs of one snapshot; at an uncut snapshot also the units for the completeness and `share_cl`."""
    h_full, k = data.H[name], kind(name)
    seen: dict[str, object] = {}
    frozen_estimate = pl.estimate

    def estimate(*args: object, **kwargs: object) -> tuple[pd.DataFrame, dict[str, object]]:
        t, meta = frozen_estimate(*args, **kwargs)
        seen.update(meta)  # the selected reference F0, which extras.snapshot leaves out of its meta
        return t, meta

    pl.estimate = estimate
    try:
        t, meta = extras.snapshot(pl, d, year, horizon=h_full, cutoff=cutoff)
    finally:
        pl.estimate = frozen_estimate
    info, curves = pl.observed(d, cutoff if cutoff is not None else 10**9, "year")
    horizon = {
        float(r.cohort): h_full if cutoff is None else min(h_full, int(r.A))
        for r in info.itertuples()
    }
    truth = {c: float(curves[c][h]) for c, h in horizon.items()}
    t["truth"] = [
        truth[c]
        if c in truth and (cutoff is not None or c + h_full / 12 <= TRUTH_END[k])
        else np.nan
        for c in t.cohort
    ]
    ref_cohort = float(t[t.mature].cohort.max())
    t["ref_cohort"] = ref_cohort
    t["ref"] = [
        float(curves[ref_cohort][int(h)]) for h in t.h
    ]  # decision A3: at the pair's own horizon
    y = t[t.young & t.truth.notna() & (t.truth > 0) & (t.h - t.A >= min_ahead)]
    y = y.sort_values("cohort").tail(N_YOUNG[k]).assign(snapshot=year, truncated=cutoff is not None)
    meta = meta | {"share_cl": None}
    if cutoff is not None or name == "civil_broad":
        return y, None, meta
    _, now = pl.observed(d, int(pl.month_idx(year * 100 + 12)), "year")
    chain = onestage.fit(extras.cohorts_of(t, now, pl.AMAX), 0.0)
    assert np.allclose(chain.level, t.chain.to_numpy()), "not the frozen chain ladder"
    share = np.interp(np.arange(h_full + 1.0), chain.ugrid, chain.f0, right=1.0)
    meta["share_cl"] = [round(float(x), 8) for x in share]
    return y, unit_table(d, t, year, seen, h_full), meta


def unit_table(
    d: pl.Data, t: pd.DataFrame, year: int, meta: dict[str, object], h: int
) -> pd.DataFrame:
    """0033 `prepare.snapshot` for the full method: per unit of each young cohort, the forecast-path plug-in."""
    s0 = int(pl.month_idx(year * 100 + 12)) - 1
    ugrid, f0 = meta["f0"]
    prepare.H = h
    fit1 = pl.step1(d, s0)
    adjusted = pl.adjusted_curves(d, s0, fit1, "year")
    heads = []
    for r in t[t.young].itertuples():
        sub = d.cells[d.cells.year == r.cohort]
        inc = pl.increments(adjusted[r.cohort], int(r.a_used), float(r.speed), ugrid, f0, h)
        part, estimate = prepare.unit_futures(pl, d, sub, s0, fit1, inc)
        if meta["calendar_on"]:
            assert abs(estimate - float(r.ours_cal_uncond)) < 1e-9, (year, r.cohort, estimate)
        heads.append(part.rename(columns={"future": "future_full"}))
    return pd.concat(heads, ignore_index=True).assign(snapshot=year)


def run_pairs(d: pl.Data, name: str, out: Path) -> None:
    parts = out / "parts"
    parts.mkdir(parents=True, exist_ok=True)
    for year, cutoff, min_ahead in PLANS[kind(name)]:
        if (parts / f"pairs_{year}.parquet").exists():
            continue
        y, units, meta = pairs_at(d, name, year, cutoff, min_ahead)
        if units is not None:
            units.to_parquet(parts / f"units_{year}.parquet")
        (parts / f"meta_{year}.json").write_text(json.dumps(meta, indent=1, default=str))
        y.to_parquet(parts / f"pairs_{year}.parquet")
        log.info(
            "%s %d: %d pairs, model %s, calendar %s",
            name,
            year,
            len(y),
            meta["model"],
            meta["calendar_on"],
        )
    years = [y for y, _, _ in PLANS[kind(name)]]
    for stem in ("pairs", "units"):
        files = [
            parts / f"{stem}_{y}.parquet" for y in years if (parts / f"{stem}_{y}.parquet").exists()
        ]
        if files:
            pd.concat((pd.read_parquet(f) for f in files), ignore_index=True).to_parquet(
                out / f"{stem}.parquet"
            )
    meta_all = {str(y): json.loads((parts / f"meta_{y}.json").read_text()) for y in years}
    (out / "meta.json").write_text(json.dumps(meta_all, indent=1))


def run_calibration(d: pl.Data, name: str, out: Path) -> None:
    """The rule of 0021/0036 `calibration_errors` on quarterly cohorts, uncut truth."""
    h, k = data.H[name], kind(name)
    truth = pl.truth(d, "quarter", h)
    parts = out / "parts"
    parts.mkdir(parents=True, exist_ok=True)
    for year in CALIBRATION[k]:
        if (parts / f"calibration_{year}.parquet").exists():
            continue
        t, _ = pl.estimate(d, year, cohort="quarter", horizon=h)
        y = t[t.young].copy()
        y["truth"] = [
            truth[c][0] if c in truth and c + 0.25 + h / 12 <= TRUTH_END[k] else np.nan
            for c in y.cohort
        ]
        b36.errors(y, year).to_parquet(parts / f"calibration_{year}.parquet")
        log.info("%s calibration %d", name, year)
    pd.concat(pd.read_parquet(f) for f in sorted(parts.glob("calibration_*.parquet"))).to_parquet(
        out / "calibration.parquet"
    )


def main(name: str, steps: list[str]) -> None:
    corrected.patch(pl)
    data.use(name)
    d = data.build(name)
    out = data.ART / name
    log.info("%s: %d units, %d cells", name, len(d.loans), len(d.cells))
    if "pairs" in steps:
        run_pairs(d, name, out)
    if "calibration" in steps:
        run_calibration(d, name, out)


if __name__ == "__main__":
    if os.environ.get("PYTHONHASHSEED") != "0":
        raise SystemExit("run with PYTHONHASHSEED=0 (the fixed-effects fit depends on set order)")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    main(sys.argv[1], sys.argv[2:] or ["pairs", "calibration"])
