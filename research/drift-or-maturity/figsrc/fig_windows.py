import numpy as np, matplotlib.pyplot as plt
from style import C, OUT
fig, axs = plt.subplots(1, 2, figsize=(5.5, 1.5), gridspec_kw=dict(wspace=0.55))
# (a) simulation, 972 settings, median relative error (Table: e1)
n = np.array([1e3, 1e4, 1e5, 1e6])
ax = axs[0]
series = [
  ("naive Lepski", [54, 65, 76, 80], C["naive"], "-", "o"),
  ("naive ADWIN", [52, 71, 79, 80], C["naive"], "--", "s"),
  ("naive, full window", [30, 30, 30, 30], C["naive"], ":", None),
  ("corrected Lepski", [15.1, 6.1, 2.2, 0.7], C["full"], "-", "o"),
  ("oracle window", [7.5, 3.1, 1.2, 0.4], C["truth"], "-", None),
]
for lab, y, col, ls, mk in series:
    ax.plot(n, y, color=col, ls=ls, marker=mk, ms=3, lw=1.3 if col != C["truth"] else 0.8)
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_ylim(0.3, 150); ax.set_xlim(7e2, 1.6e6)
ax.set_yticks([1, 10, 100]); ax.set_yticklabels(["1%", "10%", "100%"])
ax.set_xlabel("units per cohort $n$"); ax.set_ylabel("median relative error")
ax.set_title("(a) Simulation (972 settings)")
ax.text(1.15e3, 95, "naive Lepski, ADWIN", fontsize=6.5, color=C["ink"])
ax.text(1.15e3, 20.5, "naive, full window", fontsize=6.5, color=C["ink"])
ax.text(2.5e4, 7.0, "corrected", fontsize=6.5, color=C["full"])
ax.text(2.6e3, 1.1, "oracle window", fontsize=6.5, color=C["truth"])
# (b) Freddie Mac, median relative error (Table: e2)
ax = axs[1]
m = np.array([1e4, 5e4, 2e5])
ax.plot(m, [80, 88, 90], color=C["naive"], marker="o", ms=3, lw=1.3)
ax.plot(m, [84, 90, 90], color=C["naive"], ls="--", marker="s", ms=3, lw=1.3)
ax.plot(m, [46, 38, 43], color=C["full"], marker="o", ms=3, lw=1.3)
ax.plot([2e5], [27], marker="D", ms=4, color=C["clcal"], ls="none")
ax.set_xscale("log"); ax.set_xlim(8e3, 2.6e5); ax.set_ylim(0, 100)
ax.set_xticks([1e4, 5e4, 2e5]); ax.set_xticklabels(["10k", "50k", "200k"])
ax.set_yticks([0, 25, 50, 75, 100]); ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"])
ax.set_xlabel("loans per vintage"); ax.set_ylabel("median relative error")
ax.set_title("(b) Mortgages (13 snapshots)")
ax.text(1.05e4, 93, "naive Lepski, ADWIN", fontsize=6.5, color=C["ink"])
ax.text(1.1e4, 30, "corrected", fontsize=6.5, color=C["full"])
ax.text(1.6e5, 14, "corrected +\ncalendar", fontsize=6.5, color=C["clcal"], ha="right")
fig.savefig(OUT + "/windows.pdf")
fig.savefig(OUT + "/windows.png", dpi=200)
