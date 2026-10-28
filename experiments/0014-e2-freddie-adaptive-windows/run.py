"""E2: adaptive training windows on real data with delayed labels (the window-experiment specification).

DATASET: 200k loans per vintage, simple random sample of the full Freddie Mac Standard files (experiments/0010),
HARP parents censored. Loan table only (calendar-free, T4's setting).
Cohorts = origination years. Snapshots S = 2006-12 ... 2018-12. At S the cohorts are the vintages 1999 .. year(S) - 1,
youngest first. Target: the 7-year default incidence of the youngest one, vintage year(S) - 1 (about 17 months old).
Truth: its Aalen-Johansen incidence at 84 months in the full 200k sample.
  naive rules     use each vintage's default incidence seen at S as a final label;
  corrected rules use the pooled Cape Cod estimate with F-hat = chain-ladder timing curve of the vintages that are
                  mature (84 months) at S; kappa by rolling origin uses the pseudo-snapshots 1, 2 and 3 years before S
                  that have a mature vintage (else kappa = z).
Theorem A on real data: the same analysis on random subsamples of 10k and 50k loans per vintage (10 seeds each).
Usage: run.py -> results.json
"""

from __future__ import annotations

import dataclasses
import json
import logging
import sys
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy import stats

from maturity import windows

HERE: Final = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "0011-freddie-200k-confirmatory"))
import pipeline as pl  # loan table, observed curves and truth exactly as in the confirmatory run

log = logging.getLogger(__name__)
Array = NDArray[np.float64]
H: Final = 84
YEARS: Final = range(2006, 2019)
SIZES: Final = {10_000: 10, 50_000: 10, 200_000: 1}  # loans per vintage: number of subsamples
ORIGINS: Final = (1, 2, 3)
ALPHA_F: Final = 0.05
SEED: Final = 20261004
METHODS: Final = (
    "naive_full",
    "drop_recent",
    "naive_lepski",
    "naive_adwin",
    "corrected_full",
    "corrected_proven",
    "corrected_z",
    "corrected_1",
    "corrected_rolling",
    "hindsight",
)


@dataclasses.dataclass(frozen=True, slots=True)
class Seen:
    """What a snapshot shows, youngest vintage first."""

    vintage: Array
    age: NDArray[np.int64]  # months of follow-up, capped at 84
    events: Array  # (1, cohorts): loans x incidence seen so far (capped at 84 months)
    n: Array
    f: Array  # F-hat at each vintage's age
    f_error: float  # DKW bound from the events of the mature vintages
    curve: Array  # F-hat at ages 0..84


def seen_at(d: pl.Data, year: int) -> Seen | None:
    """The cohorts at December of `year`; None if no vintage is mature yet."""
    info, curves = pl.observed(d, int(pl.month_idx(year * 100 + 12)), "year")
    info = info[info.cohort <= year - 1]
    mature = info[info.A >= H]
    if mature.empty:
        return None
    weight = mature.n.to_numpy(dtype=float)
    pooled = sum(w * curves[c][: H + 1] for w, c in zip(weight, mature.cohort, strict=True))
    curve = pooled / pooled[H]
    info = info.sort_values("cohort", ascending=False)
    age = np.minimum(info.A.to_numpy(), H)
    rate = np.array([curves[c][a] for c, a in zip(info.cohort, age, strict=True)])
    n = info.n.to_numpy(dtype=float)
    return Seen(
        info.cohort.to_numpy(),
        age,
        (n * rate)[None, :],
        n,
        curve[age],
        float(np.sqrt(np.log(2 / ALPHA_F) / (2 * pooled[H]))),
        curve,
    )


def origins(seen: dict[int, Seen | None], year: int) -> list[windows.Origin]:
    """Pseudo-snapshots before S: what was seen then, and the later events of the then-youngest vintage. The
    expected later events use the timing curve known at S."""
    now, out = seen[year], []
    assert now is not None
    for h in ORIGINS:
        if (then := seen.get(year - h)) is None:
            continue
        i = int(np.argmax(now.vintage == then.vintage[0]))
        later = now.events[0, i] - then.events[0, 0] * now.n[i] / then.n[0]
        exposure = now.n[i] * (now.f[i] - now.curve[then.age[0]])
        out.append(
            windows.Origin(
                then.events, then.n, then.f, np.array([max(later, 0.0)]), np.array(exposure)
            )
        )
    return out


def snapshot(seen: dict[int, Seen | None], year: int, truth: float) -> list[dict[str, object]]:
    now = seen[year]
    assert now is not None
    events, n, f = now.events, now.n, now.f
    k = events.shape[1]
    naive, _ = windows.naive_estimates(events, n)
    corrected, _ = windows.corrected_estimates(events, n, f)
    past = origins(seen, year)
    full = np.array([k])
    best = int(np.argmin(np.abs(corrected[0] - truth))) + 1
    out = {
        "naive_full": (full, naive[:, -1]),
        "drop_recent": (full, events[:, k // 2 :].sum(axis=1) / n[k // 2 :].sum()),
        "naive_lepski": windows.naive_lepski(events, n),
        "naive_adwin": windows.adwin(events, n),
        "corrected_full": (full, corrected[:, -1]),
        "corrected_proven": windows.corrected_lepski_proven(events, n, f, f_error=now.f_error),
        "corrected_z": windows.corrected_lepski(events, n, f),
        "corrected_1": windows.corrected_lepski(events, n, f, kappa=1.0),
        "corrected_rolling": windows.corrected_lepski(
            events, n, f, kappa=windows.choose_kappa(past) if past else None
        ),
        "hindsight": (np.array([best]), corrected[:, best - 1]),
    }
    return [
        {
            "snapshot": year,
            "cohorts": k,
            "truth": truth,
            "maturity_youngest": float(f[0]),
            "method": m,
            "window": int(window[0]),
            "estimate": float(est[0]),
            "rel_error": float(abs(est[0] / truth - 1)),
        }
        for m, (window, est) in out.items()
    ]


def subsample(d: pl.Data, size: int, rng: np.random.Generator) -> pl.Data:
    loans = d.loans
    keep = np.concatenate(
        [
            rng.choice(idx, size=min(size, len(idx)), replace=False)
            for idx in loans.groupby("orig_year").indices.values()
        ]
    )
    return dataclasses.replace(d, loans=loans.iloc[np.sort(keep)])


def paired(t: pd.DataFrame, a: str, b: str, n_boot: int = 2000) -> dict[str, object]:
    """Method a against b over snapshots: sign test and mean paired difference of log relative error."""
    e = t.pivot(index="snapshot", columns="method", values="rel_error")
    diff = (np.log(e[a] + 1e-6) - np.log(e[b] + 1e-6)).to_numpy()
    wins, ties = int((e[a] < e[b]).sum()), int((e[a] == e[b]).sum())
    rng = np.random.default_rng(0)
    boot = rng.choice(diff, size=(n_boot, len(diff))).mean(axis=1)
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
    d = pl.load("base")
    truth = pl.truth(d, "year", H)
    rng = np.random.default_rng(SEED)
    rows = []
    for size, draws in SIZES.items():
        for draw in range(draws):
            sample = d if size == 200_000 else subsample(d, size, rng)
            seen = {y: seen_at(sample, y) for y in range(YEARS.start - max(ORIGINS), YEARS.stop)}
            for year in YEARS:
                rows += [
                    r | {"n_per_vintage": size, "draw": draw}
                    for r in snapshot(seen, year, truth[float(year - 1)][0])
                ]
            log.info("n = %d, draw %d done", size, draw)
    t = pd.DataFrame(rows)
    full = t[t.n_per_vintage == 200_000]
    out = {
        "snapshots_200k": full.drop(columns=["n_per_vintage", "draw"]).round(6).to_dict("records"),
        "median_error_by_n": t.groupby(["n_per_vintage", "method"], sort=False)
        .rel_error.median()
        .round(4)
        .unstack(0)
        .to_dict(),
        "mean_window_by_n": t.groupby(["n_per_vintage", "method"], sort=False)
        .window.mean()
        .round(2)
        .unstack(0)
        .to_dict(),
        "window_by_snapshot_and_n": t[t.method.isin(["naive_lepski", "naive_adwin", "corrected_z"])]
        .groupby(["method", "snapshot", "n_per_vintage"])
        .window.mean()
        .round(2)
        .unstack(2)
        .reset_index()
        .to_dict("records"),
        "corrected_z_vs_naive_lepski_200k": paired(full, "corrected_z", "naive_lepski"),
        "paired_by_n": {
            str(size): {
                str(draw): paired(g, "corrected_z", "naive_lepski")
                for draw, g in t[t.n_per_vintage == size].groupby("draw")
            }
            for size in SIZES
        },
    }
    (HERE / "results.json").write_text(json.dumps(out, indent=1, default=str))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    main()
