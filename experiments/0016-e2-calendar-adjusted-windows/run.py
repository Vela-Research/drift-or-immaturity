"""E2, secondary analysis: the corrected window rule on calendar-adjusted data (the window-experiment specification; 2026-10-04).

DATASET: 200k loans per vintage, simple random sample of the full Freddie Mac Standard files (experiments/0010),
HARP parents censored; corrected cells and plug-in of experiments/0012 (T2c section 10).
Same snapshots, cohorts, target and truth as experiments/0014. At a snapshot S:
  1. calendar step at S (fixed-effects Poisson, state HPI growth and unemployment) -> adjusted cumulative hazard
     H_c(a) of each vintage on the loan-age axis;
  2. timing curve F-hat(a) = pooled H(a) of the vintages mature at S / its value at 84 months;
  3. Cape Cod level of window k: L_k = sum n_c H_c(a_c) / sum n_c F-hat(a_c); the window is chosen by the same rules
     as in 0014 (maturity.windows), with n_c H_c(a_c) in the place of the event counts (an approximation: these
     pseudo-counts are not exactly Poisson);
  4. the target vintage keeps its observed part; its future adjusted increments are L x dF-hat; the T2c plug-in
     gives its 7-year incidence on the realized covariate path (scenario-conditional).
Window 1 is calendar + chain ladder with this timing curve.
Usage: PYTHONHASHSEED=0 uv run python run.py -> results.json
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
from scipy import stats

from maturity import windows

HERE: Final = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "0012-freddie-200k-harp-correction"))
import pipeline as pl  # corrected calendar step (IPCW cells, Aalen-Johansen plug-in)

log = logging.getLogger(__name__)
H: Final = 84
YEARS: Final = range(2006, 2019)
ALPHA_F: Final = 0.05
CALENDAR_FREE: Final = HERE.parent / "0014-e2-freddie-adaptive-windows" / "results.json"


def snapshot(d: pl.Data, year: int, truth: float) -> list[dict[str, object]]:
    s = int(pl.month_idx(year * 100 + 12))
    info, _ = pl.observed(d, s, "year")
    info = info[info.cohort <= year - 1].sort_values("cohort", ascending=False)  # youngest first
    fit = pl.step1(d, s - 1)
    adjusted = pl.adjusted_curves(d, s - 1, fit, "year")
    age, n = np.minimum(info.A.to_numpy(), H), info.n.to_numpy(dtype=float)
    mature = info.A.to_numpy() >= H
    pooled = sum(
        w * adjusted[c][: H + 1] for w, c in zip(n[mature], info.cohort[mature], strict=True)
    )
    curve = pooled / pooled[H]
    f = curve[age]
    events = (n * np.array([adjusted[c][a] for c, a in zip(info.cohort, age, strict=True)]))[
        None, :
    ]
    levels, _ = windows.corrected_estimates(events, n, f)
    target, a0, k = float(info.cohort.iloc[0]), int(age[0]), len(n)
    cells = d.cells[d.cells["year"] == target]
    own = np.diff(adjusted[target][: a0 + 1])

    def predict(window: int) -> float:
        inc = np.zeros(H + 1)
        inc[1 : a0 + 1] = own
        inc[a0 + 1 :] = levels[0, window - 1] * np.diff(curve[a0:])
        return pl.plug_in(d, cells, s - 1, fit, inc, H, conditional=True)[0]

    estimates = {w: predict(w) for w in range(1, k + 1)}
    f_error = float(np.sqrt(np.log(2 / ALPHA_F) / (2 * pooled[H])))
    chosen = {
        "cal_window_1": 1,
        "cal_full": k,
        "cal_proven": int(windows.corrected_lepski_proven(events, n, f, f_error=f_error)[0][0]),
        "cal_z": int(windows.corrected_lepski(events, n, f)[0][0]),
        "cal_1": int(windows.corrected_lepski(events, n, f, kappa=1.0)[0][0]),
        "cal_hindsight": min(estimates, key=lambda w: abs(estimates[w] - truth)),
    }
    return [
        {
            "snapshot": year,
            "cohorts": k,
            "truth": truth,
            "method": m,
            "window": w,
            "estimate": estimates[w],
            "rel_error": abs(estimates[w] / truth - 1),
        }
        for m, w in chosen.items()
    ]


def paired(e: pd.DataFrame, a: str, b: str, n_boot: int = 2000) -> dict[str, object]:
    """Method a against b over snapshots: sign test and mean paired difference of log relative error."""
    diff = (np.log(e[a] + 1e-6) - np.log(e[b] + 1e-6)).to_numpy()
    wins, ties = int((e[a] < e[b]).sum()), int((e[a] == e[b]).sum())
    boot = np.random.default_rng(0).choice(diff, size=(n_boot, len(diff))).mean(axis=1)
    return {
        "wins": wins,
        "n": len(diff) - ties,
        "sign_test_p": round(float(stats.binomtest(wins, len(diff) - ties, 0.5).pvalue), 4),
        "mean_paired_log_difference": round(float(diff.mean()), 3),
        "ci95_bootstrap_over_snapshots": [
            round(float(x), 3) for x in np.percentile(boot, [2.5, 97.5])
        ],
        "error_ratio": round(float(np.exp(diff.mean())), 3),
    }


def main() -> None:
    d = pl.load("base", "ipcw")
    truth = pl.truth(d, "year", H)
    rows = [r for year in YEARS for r in snapshot(d, year, truth[float(year - 1)][0])]
    free = pd.DataFrame(json.loads(CALENDAR_FREE.read_text())["snapshots_200k"])
    free = free[free.method.isin(["naive_lepski", "corrected_z", "corrected_1", "hindsight"])]
    t = pd.concat(
        [
            pd.DataFrame(rows),
            free[["snapshot", "cohorts", "truth", "method", "window", "estimate", "rel_error"]],
        ]
    )
    e = t.pivot(index="snapshot", columns="method", values="rel_error")
    out = {
        "snapshots": t.round(6).to_dict("records"),
        "median_error": e.median().round(4).to_dict(),
        "mean_window": t.groupby("method").window.mean().round(2).to_dict(),
        "paired": {
            f"{a} vs {b}": paired(e, a, b)
            for a, b in (
                ("cal_z", "corrected_z"),
                ("cal_z", "naive_lepski"),
                ("cal_z", "cal_window_1"),
                ("cal_1", "corrected_1"),
                ("cal_window_1", "corrected_z"),
            )
        },
    }
    (HERE / "results.json").write_text(json.dumps(out, indent=1, default=str))
    log.info("median error %s", out["median_error"])


if __name__ == "__main__":
    if os.environ.get("PYTHONHASHSEED") != "0":
        raise SystemExit(
            "run with PYTHONHASHSEED=0 (the fixed-effects fit depends on set order, see 0011)"
        )
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    main()
