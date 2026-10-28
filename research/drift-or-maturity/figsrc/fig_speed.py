import numpy as np, pandas as pd, matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from style import C, ANN, OUT
import os
D = os.environ.get("FIGDATA", "../figdata") + "/"
fig, axs = plt.subplots(1, 4, figsize=(6.5, 1.85), gridspec_kw=dict(wspace=0.62, width_ratios=[1, 1.2, 1.2, 1]))

# (a) patent speeds, calendar-adjusted, relative to 2001-2008
sp = pd.read_csv(D + "patent_speeds.csv")
ax = axs[0]
ax.axvspan(2000.5, 2008.5, color=C["shade"], lw=0, zorder=0)
y, se = sp.log_speed_adjusted_rel, sp.se_adjusted
ax.fill_between(sp.filing_year, y - 2*se, y + 2*se, color=C["truth"], alpha=0.15, lw=0)
ax.plot(sp.filing_year, y, color=C["truth"], lw=1.4)
ax.axhline(0, color=C["muted"], lw=0.6)
ax.text(2001.0, 0.40, "reference\nyears", fontsize=ANN, color=C["muted"], va="top")
ax.text(2019.3, 0.10, "faster", fontsize=ANN, color=C["ink"], ha="right")
ax.text(2019.3, -0.12, "slower", fontsize=ANN, color=C["ink"], ha="right")
ax.set_xlim(2000.5, 2019.5); ax.set_xticks([2001, 2007, 2013, 2019])
ax.set_ylim(-0.15, 0.42); ax.set_yticks([-0.1, 0, 0.1, 0.2, 0.3, 0.4])
ax.set_xlabel("filing year"); ax.set_ylabel("log grant speed")
ax.set_title("(a) Grant speed shifts")
ax.set_xticks([2001, 2010, 2019])

# (b) mean signed error by snapshot (forecast path)
er = pd.read_csv(D + "patent_errors_by_snapshot.csv")
pv = er.pivot(index="snapshot", columns="key", values="mean_signed_error")
ax = axs[1]
ax.axhline(0, color=C["muted"], lw=0.6)
ax.plot(pv.index, 100*pv["chain"], color=C["cl"], marker="o", ms=2.6, lw=1.3)
ax.plot(pv.index, 100*pv["cal_chain_forecast"], color=C["clcal"], marker="s", ms=2.6, lw=1.3)
ax.plot(pv.index, 100*pv["ours_uncond"], color=C["full"], marker="^", ms=3.0, lw=1.5)
ax.text(2017.5, 100*pv["chain"].iloc[-1], "chain\nladder", fontsize=ANN, color=C["cl"], ha="left", va="center")
ax.text(2017.5, 100*pv["cal_chain_forecast"].iloc[-1], "+ calendar", fontsize=ANN, color=C["clcal"], ha="left", va="center")
ax.text(2017.5, 100*pv["ours_uncond"].iloc[-1], "+ speeds", fontsize=ANN, color=C["full"], ha="left", va="center")
ax.text(2006.8, 42, "over", fontsize=ANN, color=C["muted"], va="center")
ax.text(2010.8, -43, "under", fontsize=ANN, color=C["muted"], va="center")
ax.set_xlim(2006.5, 2022.0); ax.set_xticks([2007, 2012, 2017])
ax.set_ylim(-48, 48); ax.set_yticks([-40, -20, 0, 20, 40])
ax.set_xlabel("snapshot"); ax.set_ylabel("signed error (%)")
ax.set_title("(b) Signed error")

# (c) mean absolute error by snapshot (forecast path)
pa = er.pivot(index="snapshot", columns="key", values="mean_abs_error")
ax = axs[2]
ax.axvspan(2010.5, 2012.5, color=C["shade"], lw=0, zorder=0)
ax.plot(pa.index, 100*pa["cal_chain_forecast"], color=C["clcal"], marker="s", ms=2.6, lw=1.3)
ax.plot(pa.index, 100*pa["ours_uncond"], color=C["full"], marker="^", ms=3.0, lw=1.5)
hz = pd.read_csv(D + "patent_hazard_by_snapshot.csv")
hz = hz[hz.method == "m1_fcst"].set_index("snapshot")
ax.plot(hz.index, 100*hz["mean_absolute_error"], color=C["drop"], marker="D", ms=2.6, lw=1.3)
ax.text(2017.6, 3.2, "hazard\nmodel", fontsize=ANN, color=C["drop"], ha="left", va="center")
ax.text(2014.0, 17.0, "+ calendar", fontsize=ANN, color=C["clcal"], ha="left", va="bottom")
ax.text(2015.6, 8.6, "+ speeds", fontsize=ANN, color=C["full"], ha="left", va="bottom")
ax.text(2012.9, 28.5, "turning point", fontsize=ANN, color=C["muted"], ha="left", va="top")
ax.set_xlim(2006.5, 2022.0); ax.set_xticks([2007, 2012, 2017])
ax.set_ylim(0, 29); ax.set_yticks([0, 10, 20])
ax.set_xlabel("snapshot"); ax.set_ylabel("absolute error (%)")
ax.set_title("(c) Absolute error")

# (d) simulation: error ratio of free speeds to equal speeds (corrected plug-in, calendar effect on)
ax = axs[3]
cats = ["constant", "step", "drift"]
_c = pd.read_csv(D + "sim_fig3c_corrected.csv")
x = np.arange(3)
ax.axhspan(0.2, 1.0, color="#EEF5EE", lw=0, zorder=0)
ax.axhline(1, color=C["muted"], lw=0.6)
for shape, key, mk, off in [("Weibull", "weibull", "o", -0.13), ("log-normal", "lognormal", "^", 0.13)]:
    for n in ["20k", "200k"]:
        sub = _c[(_c.timing == key) & (_c["size"] == n)].set_index("speeds")
        v = [sub.loc[k, "error_ratio_corrected"] for k in cats]
        ax.plot(x + off, v, ls="none", marker=mk, ms=4.0, color=C["truth"],
                mfc=C["truth"] if n == "200k" else "white", mew=1.0, zorder=3)
ax.set_yscale("log"); ax.set_ylim(0.24, 1.55); ax.set_yticks([0.3, 0.5, 0.7, 1.0]); ax.set_yticklabels(["0.3", "0.5", "0.7", "1"])
ax.minorticks_off()
ax.set_xticks(x); ax.set_xticklabels(["const.", "step", "drift"]); ax.set_xlim(-0.5, 2.5)
ax.set_xlabel("true speeds"); ax.set_ylabel("error ratio, full / equal")
ax.text(-0.45, 0.26, "speeds help", fontsize=ANN, color="#2E7D32", ha="left", va="bottom")
h = [Line2D([], [], ls="none", marker="o", mfc="white", mec=C["truth"], ms=4, label="Weibull"),
     Line2D([], [], ls="none", marker="^", mfc="white", mec=C["truth"], ms=4, label="log-normal"),
     Line2D([], [], ls="none", marker="o", mfc=C["truth"], mec=C["truth"], ms=4, label="filled: 200k")]
ax.legend(handles=h, loc="lower left", fontsize=ANN, frameon=False, handletextpad=0.2, borderaxespad=0.0,
          labelspacing=0.2, bbox_to_anchor=(0.0, 0.12))
ax.set_title("(d) Simulation")
fig.savefig(OUT + "/speed.pdf")
fig.savefig(OUT + "/speed.png", dpi=220)
