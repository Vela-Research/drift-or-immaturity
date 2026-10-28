import numpy as np, matplotlib.pyplot as plt
from style import C, ANN, OUT
g = 1.5; lam = 5/np.log(2)**(1/g)
F = lambda u: 1-np.exp(-(np.asarray(u)/lam)**g)
fig, axs = plt.subplots(1, 2, figsize=(6.5, 1.8), gridspec_kw=dict(wspace=0.38, width_ratios=[1, 1.15]))

# (a) Lexis diagram
ax = axs[0]
T, H = 2025, 7
ax.fill([2010, T, T-12, 2010], [0, 0, 12, 12], color=C["shade"], lw=0, zorder=0)
ax.plot([T-12, T], [12, 0], color=C["ink"], lw=1.0)
ax.text(2018.2, 9.6, "snapshot $T$", fontsize=ANN, color=C["ink"])
cs = np.array([[2010, 6], [2010, 7], [2017, 0], [2016, 0]])
ax.fill(cs[:, 0], cs[:, 1], color=C["naive"], alpha=0.35, lw=0)
ax.text(2010.3, 0.5, "calendar\nshock", fontsize=ANN, color=C["ink"], va="bottom")
ax.axhline(H, color=C["muted"], ls="--", lw=0.7)
c0 = 2015
ax.plot([c0, c0], [0, T-c0], color=C["truth"], lw=1.5)
ax.annotate("cohort $c$ seen\nto age $A_c=T-c$", xy=(c0, 3.4), xytext=(2015.7, 2.2), fontsize=ANN, color=C["truth"], va="center",
            arrowprops=dict(arrowstyle="-", lw=0.6, color=C["truth"], shrinkA=0, shrinkB=0))
ax.plot([T-H, T], [-0.35, -0.35], color=C["full"], lw=3.0, solid_capstyle="butt", zorder=3, clip_on=False)
ax.text(2021.4, 0.6, "immature", ha="center", fontsize=ANN, color=C["full"])
ax.set_xlim(2010, T); ax.set_ylim(0, 12)
ax.set_xticks([2010, 2015, 2020, 2025]); ax.set_yticks([0, 7, 12]); ax.set_yticklabels(['0', '$H$', '12'])
ax.set_xlabel("cohort (entry year) $c$"); ax.set_ylabel("age $a$ (years)")
ax.set_title("(a) Young cohorts are seen only young")

# (b) faster cohort and the chain ladder
ax = axs[1]
th, s, A, Hh = 0.20, 1.3, 2.0, 7.0
p_ref = th*F(Hh)
th_c = 0.9*p_ref/F(s*Hh)
aa = np.linspace(0, Hh, 200)
ax.plot(aa, th*F(aa), color=C["naive"], lw=1.2)
a1 = np.linspace(0, A, 60); a2 = np.linspace(A, Hh, 120)
ax.plot(a1, th_c*F(s*a1), color=C["truth"], lw=1.8)
ax.plot(a2, th_c*F(s*a2), color=C["truth"], ls=":", lw=1.4)
Kc = th_c*F(s*A)
ax.plot(a2, Kc*F(a2)/F(A), color=C["cl"], ls="--", lw=1.5)
truth_H, cl_H = th_c*F(s*Hh), Kc*F(Hh)/F(A)
for y, col in [(p_ref, C["naive"]), (truth_H, C["truth"]), (cl_H, C["cl"])]:
    ax.plot([Hh], [y], "o", color=col, ms=3.5, clip_on=False)
ax.axvspan(0, A, color=C["shade"], lw=0, zorder=0)
ax.axvline(Hh, color=C["muted"], lw=0.6, ls="--")
ax.text(Hh+0.25, cl_H+0.003, "chain ladder: +%d%%" % round(100*(cl_H/p_ref-1)), fontsize=ANN, color=C["cl"], va="bottom")
ax.text(Hh+0.25, p_ref, "reference cohort", fontsize=ANN, color=C["muted"], va="center")
ax.text(Hh+0.25, truth_H-0.003, "truth: −%d%%" % round(-100*(truth_H/p_ref-1)), fontsize=ANN, color=C["truth"], va="top")
ax.text(1.0, 0.152, "observed", fontsize=ANN, color=C["muted"], ha="center")
ax.text(2.25, 0.004, "faster cohort (dotted: future)", fontsize=ANN, color=C["truth"])
ax.text(Hh-0.1, 0.152, "$H$", fontsize=ANN, color=C["muted"], ha="right")
ax.set_xlim(0, 7.2); ax.set_xticks([0, 2, 4, 6]); ax.set_ylim(0, 0.165); ax.set_yticks([0, 0.05, 0.10, 0.15])
ax.set_xlabel("age (years)"); ax.set_ylabel("cumulative incidence")
ax.set_title("(b) Chain ladder reports a rise; the rate fell")
print("ref", p_ref, "truth", truth_H, "cl", cl_H, "cl/truth", cl_H/truth_H)
fig.savefig(OUT + "/schematic.pdf")
fig.savefig(OUT + "/schematic.png", dpi=220)
