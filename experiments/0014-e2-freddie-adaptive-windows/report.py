"""Markdown tables for the README from results.json (no new analysis). Usage: report.py > artifacts/tables.md"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Final

import pandas as pd

HERE: Final = Path(__file__).resolve().parent
NAMES: Final = {
    "naive_full": "naive, full window",
    "drop_recent": "naive, drop recent",
    "naive_lepski": "naive Lepski",
    "naive_adwin": "naive ADWIN",
    "corrected_full": "corrected, full window",
    "corrected_proven": "corrected Lepski, proven constants",
    "corrected_z": "corrected Lepski, kappa = z",
    "corrected_1": "corrected Lepski, kappa = 1",
    "corrected_rolling": "corrected Lepski, kappa by rolling origin",
    "hindsight": "best corrected window in hindsight",
}
ADAPTIVE: Final = [m for m in NAMES if "full" not in m and m != "drop_recent"]


def table(t: pd.DataFrame, fmt: str, index_name: str) -> str:
    head = [index_name, *t.columns]
    lines = ["| " + " | ".join(str(h) for h in head) + " |", "|" + "---|" * len(head)]
    lines += [
        f"| {i} | " + " | ".join(fmt.format(v) for v in row) + " |" for i, row in t.iterrows()
    ]
    return "\n".join(lines)


def main() -> None:
    r = json.loads((HERE / "results.json").read_text())
    s = pd.DataFrame(r["snapshots_200k"])
    info = s.drop_duplicates("snapshot").set_index("snapshot")
    win = s.pivot(index="snapshot", columns="method", values="window")[ADAPTIVE].rename(
        columns=NAMES
    )
    win.insert(0, "cohorts", info.cohorts)
    err = (100 * s.pivot(index="snapshot", columns="method", values="rel_error"))[list(NAMES)]
    err = err.rename(columns=NAMES)
    err.insert(0, "truth (7-year rate, %)", 100 * info.truth)
    err.insert(1, "F-hat of the youngest (%)", 100 * info.maturity_youngest)
    med = (100 * pd.DataFrame(r["median_error_by_n"])).loc[list(NAMES)].rename(index=NAMES)
    mean_w = pd.DataFrame(r["mean_window_by_n"]).loc[ADAPTIVE].rename(index=NAMES)
    by = pd.DataFrame(r["window_by_snapshot_and_n"]).set_index(["snapshot", "method"]).unstack(1)
    by.columns = [f"{NAMES[m]}, n = {int(n):,}" for n, m in by.columns]
    by = by[
        sorted(by.columns, key=lambda c: (c.split(",")[0], int(c.split("= ")[-1].replace(",", ""))))
    ]
    parts = [
        f"## Windows, 200k\n{table(win, '{:.0f}', 'snapshot (December)')}",
        f"## Errors, 200k\n{table(err, '{:.1f}', 'snapshot (December)')}",
        f"## Median error by n\n{table(med.rename(columns=lambda c: f'n = {int(c):,}'), '{:.1f}%', 'method')}",
        f"## Mean window by n\n{table(mean_w.rename(columns=lambda c: f'n = {int(c):,}'), '{:.2f}', 'method')}",
        f"## Window by snapshot and n\n{table(by, '{:.1f}', 'snapshot (December)')}",
    ]
    sys.stdout.write("\n\n".join(parts) + "\n")


if __name__ == "__main__":
    main()
