# Drift or Immaturity? Code

Code for the paper *Drift or Immaturity? Correcting for Maturity Matters, Learning a Young Cohort's Own Timing Rarely Helps*. Every table and figure of the paper comes from the experiments in this repository.

## This is the code as run

- Each folder `experiments/NNNN-name/` is one experiment, with the ID it had in the development repository. Gaps in the numbering (for example 0001–0002, 0007–0009, 0023–0028 and 0042–0065) are experiments that are not part of the paper. 0023–0028 are the startup analyses of earlier drafts; they were removed from this release.
- The code is the code that produced the paper's numbers.
  - Only comments and docstrings were edited, to remove references to internal notes.
  - In 14 files, one string changed in each. Twelve are an output path (`.../from_CC/figdata` became `.../figdata`), one is the text that describes the FRED source in the download log, and one is the same path in the tree builder of 0071. In the figure script `fig_windows.py`, the two save paths of the sandbox where the figure was first drawn became the output folder of the other figure scripts (`OUT`).
  - The shared package is called `maturity` here. In the development repository it had an internal name; the imports changed with it, and so did the names of two environment variables of 0072 (`EXP0072_PATCH`, `EXP0072_APPLIED`).
  - The parts of the startup analyses were removed from five files: 0015 `build.py` and `figdata.py`, 0029 `review.py`, 0030 `review2.py` and 0031 `report.py`. 0030 now imports the patent backtest and the frozen pipeline directly; before, it imported them through the removed code.
  - `maturity/data.py` keeps only `DATA_DIR`. The rest of the original module served other projects.
  - A check compared the abstract syntax tree of every file with its original, docstrings stripped and the package name mapped. All other files are identical.
- **Pre-registration:** the replication protocol in `preregistration/` was committed (commit `b7d4f3d` of the development repository) and time-stamped with OpenTimestamps before any of its data were downloaded. See `preregistration/README.md` to verify the proof.
- **Post hoc changes** are labelled as such in the code (docstrings say POST HOC) and are listed with their effects in the paper's table of deviations (Appendix E).
- **Docstrings** cite the project's design notes, which are not included; the paper's appendix describes each design.
- **One re-implementation:** `experiments/0073-precision-table` makes `tab:precision` again from the table's caption and `thm:precision`. The paper's table comes from this script (10,000 replications, seed 20261027). The earlier 300-replication values (1.60, 0.40, 0.06, 0.53, 0.11) came from a script that was not kept; they lie within the range of 300-replication values over 30 seeds, which `run.py` writes to `results.json`.
- **The corrected mortgage numbers.** The recorded Freddie Mac loan age restarts at a loan modification. Experiment 0071 rebuilds every age from the first-payment month and runs the frozen mortgage code again, unchanged, in a copy of the repository (`run.py tree`, `run.py exec`). Experiment 0072 does the same for the remaining runs and the simulation re-run. All mortgage results in the paper come from these two trees; the as-run values (with the recorded ages) come from the experiments run directly.

## Install

Python 3.13 and [uv](https://docs.astral.sh/uv/):

```
uv sync
```

`uv.lock` pins every package (the main ones are pinned in `pyproject.toml`); `requirements.txt` has the same pins for pip. The replication download needs `curl`.

## Data

Nothing is included: the licences of some data sets forbid redistribution, and the others are public and large. Put the files under `data/` at the repository root:

| data set | where to get it | put it in |
|---|---|---|
| Freddie Mac Single-Family Loan-Level Dataset, standard (full) files | free registration at Freddie Mac (Single-Family Loan-Level Dataset); the runs used the release of July 2026 | `data/raw/freddie/historical_data_YYYY.zip`, 1999 to 2026 (about 41 GB) |
| Freddie Mac 50,000-loan samples (development only, 0003 and 0005) | the same site | `data/raw/freddie/sample_YYYY.zip` |
| USPTO Patent Examination Research Dataset (PatEx), release "2022" (PEDS data of June 2023) | USPTO Office of the Chief Economist, research datasets | `data/raw/patex/application_data.csv.zip` |
| FHFA House Price Index, all transactions, states | `https://www.fhfa.gov/hpi/download/quarterly_datasets/hpi_at_state.csv` | `data/raw/covariates/fhfa_state.csv` |
| BLS LAUS state unemployment rate, seasonally adjusted (the FRED copies `<ST>UR`) | `https://fred.stlouisfed.org/graph/fredgraph.csv?id=<ST>UR` for each state and DC, PR | `data/raw/covariates/unrate_<ST>.csv` |
| Freddie Mac Primary Mortgage Market Survey (PMMS), weekly history | Freddie Mac PMMS page (CSV with the columns `date, pmms30, ...`) | `data/raw/covariates/pmms.csv` |
| SBA 7(a) FOIA loan data (as of 2026-06-30), FJC Integrated Database (civil, 1988 on) and codebook, FJC Biographical Directory of Article III judges (federal judicial service), FRED state unemployment, FHFA state HPI | downloaded by `experiments/0068-confirm2-replication/download.py` | `experiments/0068-confirm2-replication/artifacts/raw/` |

- FRED and FHFA revise their series; a new download can differ slightly from the one used. `preregistration/confirm2_data_manifest.json` has the SHA-256 of every file the replication used, and `reproduce.py` compares a new download with it.

## Running

Each step runs from its experiment folder, for example `cd experiments/0011-freddie-200k-confirmatory && PYTHONHASHSEED=0 uv run python run.py`. `PYTHONHASHSEED=0` is needed wherever a fixed-effects model is fitted (the fits depend on set order). Run one heavy step at a time: some steps need 10–24 GB of memory (0036, 0040), and the timings below are from an 11-core laptop with 36 GB.

### Order of a full reproduction (about 14 hours)

1. **Data (about 40 minutes).** 0010 `build.py` (25 min; the 200,000-loan samples and the HARP relief loans), 0018 `link.py` (8 min), 0020 `extract.py` (10 s). Development sample (optional): 0003 `build_loans.py`.
2. **As run, with the recorded loan ages** (the registered verdict and the "as run" column of `tab:mortgage-ages`):
   - mortgages: 0011 `cells.py`, `run.py` (6 min); 0012 `cells.py`, `run.py ipcw`, `run.py noharp` (15 min); 0014 `run.py` (1 min); 0016 `run.py` (40 s); 0017 `run.py` (1 min); 0018 `evaluate.py` (7 min); 0019 `run.py` (8 min); 0022 `stratified.py` (1 min); 0031 `freddie.py as_run|corrected|noharp|linked` (2–4 min each); 0032 `hazard.py` (46 min); 0033 `prepare.py corrected|linked`, `run.py corrected vintage`, `run.py linked vintage 2008 2012 2016`, `drift.py corrected`, `report.py` (about 1 hour); 0035 `criterion.py patents|corrected|linked`, `report.py` (10 min); 0036 `backtests.py <version> <frozen|corrected>` for every version, `report.py` (20 min); 0039 `prepare_extra.py`, `classify.py linked 3`, `alarms.py`, `report.py` (45 min); 0040 `variants.py linked|noharp`, `report.py` (25 min); 0041 `real.py 3`, `report.py` (10 min); 0067 `hazard_classifier.py`, `report.py` (8 min); 0029 `review.py`; 0030 `review2.py` (2 min);
   - patents: 0020 `check.py`; 0021 `backtest.py` (2 min); 0031 `patents.py` (3 min), `report.py`; 0066 `features.py`, `hazard.py`, `report.py` (13 min).
3. **Simulations:** 0004 `sim.py` (2 min per 100 replications); 0006 `sim.py`; 0013 `run.py all`, `figures.py`, `summary.py` (2 min); 0029 `calendar_sim_bias.py` (5 min); 0038 `run.py` (10 s); 0073 `run.py` (45 s; re-implementation of `tab:precision`); 0041 `timing.py`, `step0.py 200 3` (25 min); 0034 `check.py`, `sim.py run`, `windows_row.py full|speed|check`, `report.py` (1–1.5 hours); 0037 `rerun.py check`, `rerun.py run`, `windows_feasible.py`, `compare.py` (1–1.5 hours).
4. **Corrected loan ages, 0071 (about 2 hours).** In `experiments/0071-mortgage-calendar-age`: `run.py check`, `run.py data`, `run.py tree`; then `PYTHONHASHSEED=0 uv run python run.py exec <folder> <script> [args]` for each step, in this order:
   - 0011: `cells.py`, `run.py`;
   - 0012: `cells.py`, `run.py ipcw`, `run.py noharp`;
   - 0031: `freddie.py as_run|corrected|noharp`; 0018: `link.py`, `evaluate.py`; 0031: `freddie.py linked`, `patents.py`, `report.py`;
   - 0032: `hazard.py`, `report.py`;
   - 0036: `backtests.py` for `as_run|corrected|noharp|linked` with `frozen` and `corrected`, then `report.py`;
   - 0033: `prepare.py corrected|linked`, `run.py corrected vintage`, `run.py linked vintage 2008 2012 2016`, `drift.py corrected`, `report.py`;
   - 0039: `prepare_extra.py`, `classify.py linked 3`, `alarms.py`, `report.py`;
   - 0067: `hazard_classifier.py`; 0066: `report.py`; 0067: `report.py`;
   - 0014: `run.py`, `report.py`; 0016: `run.py`;
   - 0015: `build.py`, `round3_tables.py`, `figdata.py`;
   - then `uv run python report.py` in 0071 (the wrong-direction, simulation-coverage and horizon-84 coverage counts).
5. **0072 (about 4 hours).** In `experiments/0072-auditmp-reruns`: `run.py check`, `run.py data`, `run.py tree`; then `uv run python run.py exec <patches> <folder> <script> [args]`:
   - `f1,f2`: 0017 `run.py`; 0035 `criterion.py patents|corrected|linked`, `report.py`; 0041 `real.py 3`, `report.py`; 0040 `variants.py linked`, `variants.py noharp`, `report.py`; 0036 `report.py`; 0015 `build.py`, `round3_tables.py`, `figdata.py`;
   - `f1`: 0030 `review2.py`;
   - `f2,cp`: 0034 `check.py`, `sim.py run`, `windows_row.py full|speed|check`, `report.py`; 0037 `rerun.py check`, `rerun.py run`, `windows_feasible.py`, `compare.py`;
   - `f1,f2`: 0067 `report.py`;
   - then `PYTHONHASHSEED=0 uv run python run.py rep asrun sba`, `rep asrun civil`, `rep corrected sba`, `rep corrected civil`; `run.py simulation stored`, `run.py simulation rerun`, `run.py report`.
   - Patches: `f1` = rebuilt loan ages; `f2` = the random-walk evidence with the fit's own prior; `cp` = the change point may start at any of the last eight cohorts.
6. **Replication (about 1.5 hours).** 0068: `download.py`, `build.py`, `gate.py`, `backtest.py sba|civil|civil_broad`, `hazard_m1.py sba|civil|civil_broad`, `classify.py sba|civil`, `secondary.py sba|civil windows|drift`, `pif_check.py`, `report.py`; 0069: `run.py link`, then `run.py backtest civil|civil_broad`, `hazard civil|civil_broad`, `classify sba|civil`, `windows sba|civil`, `drift civil`, `pif`, `report`, `side`; 0070: `run.py link`, `backtest|hazard|classify|windows|drift civil`, `report`.
7. **Figures:** `uv run python reproduce.py figdata` (Figure 1(c) and the simulation-by-age data, from 0072's tree); then, in `research/drift-or-maturity/figsrc`, `FIGOUT=../paper/figures FIGDATA=../../../experiments/0072-auditmp-reruns/artifacts/tree/research/drift-or-maturity/figdata uv run python <script>` for `fig_teaser.py`, `fig_speed.py` and `fig_schematic.py`. (`fig_windows.py` draws an older figure that is not in the paper.)

### Where each table and figure comes from

"0071 tree" and "0072 tree" mean the step runs through `run.py exec` of that experiment (corrected loan ages). Generated `.tex` files are written to `research/drift-or-maturity/paper/tables/` of the tree that ran 0015.

| paper | experiment folders | command (see the order above) |
|---|---|---|
| Figure 1 (`fig:teaser`) | (a) illustration; (b) 0013; (c) 0014, 0016 (0071 tree) | `reproduce.py figdata`, then figsrc `fig_teaser.py` |
| Figure 2 (`fig:schematic`) | none (illustration) | figsrc `fig_schematic.py` |
| Table 1 (`tab:classifiers`), `tab:classifiers-full` | 0033, 0039, 0067 (0071 tree) | values in 0033 `results.json` (`corrected/vintage/median_over_snapshots`) and 0067 `results.json` (`hazard_classifier`) |
| Table 2 (`tab:main`) | mortgages: 0031, 0032 (0071 tree); patents: 0031, 0066, 0067 | 0031 `report.py`; the ratios are `maturity.inference.compare` on the pair files |
| Figure 3 (`fig:speed`) | (a–c) 0020, 0031, 0066 via 0015 `figdata.py`; (d) 0034, 0037 (0072 tree) | figsrc `fig_speed.py` |
| Table 3 (`tab:replication`) | 0068, 0069 | `reproduce.py tab:replication --run` |
| `tab:e1` | 0013; rows "speeds differ": 0034 `windows_row.py`; rows "frozen speed model": 0037 `windows_feasible.py` (0072 tree) | 0013 `summary.py` |
| `fig:speed-sim` | 0004 | 0015 `build.py` (`speed_chain_ladder.pdf`) |
| `tab:sim-e2e` | 0034, 0037 (0072 tree) | 0037 `compare.py`; coverage with at least 9 calibration errors: 0072 `run.py simulation rerun` |
| `tab:thm6check` | 0038 | `run.py` |
| `tab:baselines`, `tab:inference`, `tab:inference-sens` | 0031, 0032 (0071 tree) | 0015 `round3_tables.py` (0072 tree) |
| `tab:speed-criterion` | 0035 (0072 tree) | 0015 `round3_tables.py` (0072 tree) |
| `tab:corrected` | 0036, 0040 (0072 tree) | 0036 `report.py` |
| `tab:mortgage-ages` | as run: the experiments run directly; corrected: 0071 and 0072 trees | the same commands in both places |
| `tab:freddie`, `fig:freddie` | 0011, 0012, 0018 (0071 tree) | 0015 `build.py` (0072 tree) |
| `tab:harp-check` | 0017 (0072 tree) | `reproduce.py tab:harp-check --run` |
| `tab:overid` | 0011, 0012 (0071 tree) | 0015 `build.py` (0072 tree) |
| `tab:e2` | 0014, 0016 (0071 tree) | 0015 `build.py` (0072 tree) |
| `tab:alarms` | 0033, 0039, 0067 (0071 tree) | 0067 `results.json` (`alarms`) |
| `tab:wrongdir` | patents: 0031, 0066; mortgages: 0071 `report.py` | 0071 `report.py` |
| `tab:patex`, `tab:patex-errors`, `tab:patex-coverage` | 0020, 0021, 0031 | 0015 `build.py` (`patex_errors.tex`: the paper prints four cells from the full-precision pairs of 0031; the file rounds stored 4-decimal values) |
| `tab:rep-asrun`, `tab:rep-main`, `tab:rep-steps`, `tab:rep-wrongdir`, `tab:rep-windows`, `tab:rep-alarms` | 0068 (as run), 0069 (corrected), 0070 (without MDL cases); coverage with the consistent evidence: 0072 `run.py rep` | `results.json` of each; 0069 `run.py side` |
| `tab:precision` | 0073 (re-implementation; 10,000 replications, seed 20261027) | `run.py` |
| `tab:notation`, `tab:map`, `tab:deviations` | text tables | — |

## Checks

`reproduce.py` runs the steps behind two tables and compares the result with the paper:

```
uv run python reproduce.py tab:harp-check --run      # mortgages: data build, rebuilt ages, HARP check (about 50 minutes)
uv run python reproduce.py tab:replication --run     # Table 3: download and the whole replication (about 1.5 hours)
```

`uv run python reproduce.py <target>` prints the steps without running them; `--from N` restarts at step N; `--check` only compares.

## Licence

Apache License 2.0 (`LICENSE`). The data keep their own licences.
