import numpy as np, matplotlib.pyplot as plt
from style import C, ANN, OUT
g = 1.5; lam = 5/np.log(2)**(1/g)
F = lambda u: 1-np.exp(-(np.asarray(u)/lam)**g)
fig, axs = plt.subplots(1, 3, figsize=(6.5, 1.85), gridspec_kw=dict(wspace=0.52))

# (a) same data, two futures
ax = axs[0]
p, A, Hh = 0.20, 2.0, 7.0
a1 = np.linspace(0, A, 50); a2 = np.linspace(A, Hh, 100)
ax.axvspan(0, A, color=C["shade"], lw=0, zorder=0)
ax.plot(a1, p*F(a1), color=C["truth"], lw=1.8)
ax.plot(a2, p*F(a2), color=C["ink"], ls="--", lw=1.2)
lvl = p*F(A)
ax.plot(a2, lvl*(1+0.12*(1-np.exp(-(a2-A)))), color=C["ink"], ls=":", lw=1.4)
ax.plot([A], [lvl], "o", color=C["truth"], ms=3.5)
ax.axvline(Hh, color=C["muted"], lw=0.6, ls="--")
ax.text(1.0, 0.168, "observed", fontsize=ANN, color=C["muted"], ha="center")
ax.text(Hh-0.12, 0.168, "horizon", fontsize=ANN, color=C["muted"], ha="right")
ax.text(3.3, 0.122, "immature:\nmore to come", fontsize=ANN, color=C["ink"], ha="center")
ax.text(4.9, lvl-0.006, "drifted: rate\nalready low", fontsize=ANN, color=C["ink"], ha="center", va="top")
ax.set_xlim(0, 7.4); ax.set_xticks([0, 2, 4, 6]); ax.set_ylim(0, 0.185); ax.set_yticks([0, 0.05, 0.10, 0.15])
ax.set_xlabel("age (years)"); ax.set_ylabel("cumulative incidence")
ax.set_title("(a) Same data, two futures")

# (b) simulation (Table e1), kappa = z for both Lepski rules
n = np.array([1e3, 1e4, 1e5, 1e6])
ax = axs[1]
ax.plot(n, [54, 65, 76, 80], color=C["naive"], marker="o", ms=3, lw=1.3)
ax.plot(n, [52, 71, 79, 80], color=C["naive"], ls="--", marker="s", ms=3, lw=1.3)
ax.plot(n, [18.2, 12.0, 5.9, 1.0], color=C["full"], marker="o", ms=3.2, lw=1.7)
ax.plot(n, [7.5, 3.1, 1.2, 0.4], color=C["truth"], lw=0.8)
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_ylim(0.3, 220); ax.set_xlim(7e2, 1.6e6)
ax.set_xticks([1e3, 1e4, 1e5, 1e6]); ax.minorticks_off()
ax.set_yticks([1, 10, 100]); ax.set_yticklabels(["1%", "10%", "100%"])
ax.set_xlabel("units per cohort"); ax.set_ylabel("median relative error")
ax.set_title("(b) Simulation")
ax.text(9e2, 118, "naive Lepski, ADWIN", fontsize=ANN, color=C["ink"])
ax.text(9e2, 24, "maturity-weighted", fontsize=ANN, color=C["full"], va="bottom")
ax.text(1.4e3, 0.8, "oracle window", fontsize=ANN, color=C["truth"])

# (c) mortgages (Table e2), kappa = z
ax = axs[2]
m = np.array([1e4, 5e4, 2e5])
import pandas as _pd, os as _os
_tw = _pd.read_csv(_os.environ.get("FIGDATA", "../figdata") + "/teaser_mortgage_windows.csv")
_g = lambda k: _tw[_tw.series == k].sort_values("loans_per_vintage").median_rel_error_pct.values
ax.plot(m, _g("naive_lepski"), color=C["naive"], marker="o", ms=3, lw=1.3)
ax.plot(m, _g("naive_adwin"), color=C["naive"], ls="--", marker="s", ms=3, lw=1.3)
ax.plot(m, _g("maturity_weighted_kappa_z"), color=C["full"], marker="o", ms=3.2, lw=1.7)
_cal = float(_g("calendar_adjusted_maturity_weighted_kappa_z")[0])
ax.plot([2e5], [_cal], marker="D", ms=4.2, color=C["clcal"], ls="none")
ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(8e3, 2.6e5); ax.set_ylim(15, 220)
ax.set_xticks([1e4, 5e4, 2e5]); ax.set_xticklabels(["10k", "50k", "200k"]); ax.minorticks_off()
ax.set_yticks([25, 50, 100]); ax.set_yticklabels(["25%", "50%", "100%"]); ax.minorticks_off()
ax.set_xlabel("loans per vintage"); ax.set_ylabel("median relative error")
ax.set_title("(c) Mortgages")
ax.text(9e3, 118, "naive Lepski, ADWIN", fontsize=ANN, color=C["ink"])
ax.text(9e3, 41, "maturity-weighted", fontsize=ANN, color=C["full"])
ax.text(1.75e5, _cal, "+ calendar\nstep", fontsize=ANN, color=C["clcal"], ha="right", va="center")
fig.savefig(OUT + "/teaser.pdf")
fig.savefig(OUT + "/teaser.png", dpi=220)
