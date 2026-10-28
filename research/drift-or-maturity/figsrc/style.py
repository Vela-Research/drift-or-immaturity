import matplotlib as mpl
from matplotlib import font_manager as fm
import glob
for f in glob.glob('/usr/share/texmf/fonts/opentype/public/lm/lmroman10-*.otf'):
    fm.fontManager.addfont(f)
import re as _re
fm.fontManager.ttflist = [f for f in fm.fontManager.ttflist if not _re.search(r'lmroman(5|6|7|8|9|12|17)-', f.fname) and 'lmromandunh' not in f.fname and 'lmromanslant' not in f.fname and 'lmromancaps' not in f.fname and 'lmromandemi' not in f.fname]
mpl.rcParams.update({
    "font.family": "serif", "font.serif": ["LM Roman 10", "Latin Modern Roman", "DejaVu Serif"],
    "mathtext.fontset": "cm",
    "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "legend.fontsize": 7.5, "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "lines.linewidth": 1.5, "pdf.fonttype": 42, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
    "axes.titlelocation": "left",
})
C = {
    "naive": "#888888", "drop": "#CC79A7", "lastmature": "#E69F00",
    "cl": "#0072B2", "clexact": "#56B4E9", "clcal": "#009E73", "full": "#D55E00",
    "truth": "#000000", "ink": "#333333", "muted": "#777777", "shade": "#E8E8E8",
}

ANN = 7.2  # annotation font size
import os
OUT = os.environ.get("FIGOUT", ".")
