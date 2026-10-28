# Pre-registration: confirmatory replication on two new data sets

*CD, 2026-10-25. Status: becomes **FROZEN** the moment CC completes step 0.1. After that nothing here changes; any later change goes in `DEVIATIONS.md`, is labelled **post hoc**, and is reported. Every result is reported, whatever it shows.*

**Why:** the paper's headline results are post hoc (the pre-registered mortgage comparison failed; the patent pre-specified result rests on an outcome-driven covariate). This study tests the paper's current claims, fixed in advance, on data none of us has touched.

**Data sets:**
- **A. SBA 7(a) small-business loans:** public domain (US government FOIA).
- **B. Federal civil court cases:** FJC Integrated Database, public domain. B is a different domain from credit and patents; that is the point of B.

SEER was considered and dropped: its data-use agreement bars case-level data from AI tools that send data to an external provider.

---

## 0. Order of work (strict)

**0.1 Freeze.**
1. Commit this file unchanged. Record its sha256.
2. Timestamp it publicly: `pip install opentimestamps-client` → `ots stamp CONFIRM2_SPEC.md` → commit the `.ots` file.
3. Push, and report the commit hash and the stamp time.
4. Vadmin may also register the file on OSF (recommended; an anonymous view-only link can go in the paper).

**0.2 Download (only after 0.1).**
- Record every URL, the download date and the raw-file sha256.
- On ingest, keep only the columns this spec uses, as parquet. Drop names and street addresses.
- Delete the raw files once the parquet is built and checked. Peak disk target: < 3 GB.

**0.3 Gate checks (§5).**
- Structure only: no method runs, and no event rates by cohort.
- Write `GATE_REPORT.md`. If a gate fails, stop that data set and report; CD then decides the fix, which is logged as a deviation before any method run.

**0.4 Run §4–§7 exactly.** Use the frozen paper code (the patent run with the corrected plug-in; the hazard model as patent M1, experiment 0066; the classifiers as 0033/0067).

**0.5 Report per §8.**

## 1. Hypotheses (one family per data set; Holm at α = 0.05 over H1–H3)

All comparisons are error ratios, method X over method Y; below 1 means X is more accurate (§6). Primary path: **forecast**, with covariates held at their last-12-month mean.

**H1. Correcting for maturity helps.** The classifier trained on positive–unlabelled labels with chain-ladder maturity has lower level error than the classifier trained on events so far.
- One-sided test of ratio < 1.

**H2. A young cohort's own speed adds nothing.** Full method over calendar-adjusted chain ladder.
- Non-superiority with margin 0.85. H0: ratio ≤ 0.85, i.e. speeds cut error by at least 15%.
- **Confirmed** if H0 is rejected (one-sided).
- Also report whether the two-sided 95% interval excludes 1, in either direction.

**H3. A standard competing-risks hazard model is at least as accurate as the full method.** Hazard model over full method.
- Non-inferiority with margin 1.15. H0: ratio ≥ 1.15.
- **Confirmed** if H0 is rejected (one-sided).
- Also report superiority (ratio < 1).

**Outcomes and reading across data sets:**
- Each hypothesis per data set is **confirmed**, **refuted** or **inconclusive**:
  - H1 is refuted if its ratio is significantly > 1.
  - H2 is refuted if speeds help significantly (two-sided 95% interval entirely below 1) and H0 is not rejected.
  - H3 is refuted if the hazard model is significantly worse (interval entirely above 1).
- A claim counts as confirmed in the paper only if it is confirmed in both data sets. Each data set is reported on its own.

**Why these margins:** 15% is the smallest gain that would change the paper's advice ("add speeds only if a backtest shows a gain"). The non-inferiority margin is symmetric to it.

## 2. Data set A: SBA 7(a) loans

- **Source:** data.sba.gov "7(a) & 504 FOIA": the 7(a) files FY1991–FY1999, FY2000–FY2009, FY2010–FY2019 and FY2020–present, plus the data dictionary. Use the latest "as of" release, and record its date.
- **Units:** 7(a) loans approved from 1991-01-01 to 2019-12-31.
  - Exclude status CANCLD (never disbursed) and COMMIT.
  - Exclude borrowers outside the 50 states and DC.
  - Exclude loans whose status is CHGOFF or PIF but whose matching date is missing (count reported).
- **Cohort:** approval calendar year (1991–2019). **Entry:** approval date. **Age:** months since approval.
- **Event:** charge-off (charge-off date). **Competing exit:** paid in full (paid-in-full date). All other loans are censored at the snapshot.
- **Horizon:** H = 72 months.
- **Groups:** borrower state (51).
- **Calendar covariates, as in the Freddie run:**
  - state unemployment rate (BLS LAUS, monthly, seasonally adjusted);
  - state house-price growth (FHFA all-transactions index, 12-month log change, quarterly values carried forward monthly).
- **Snapshots:** December 31 of 2000 through 2019 (20). **Scored cohorts:** the six youngest approval years, S−5 to S.
- **Truth:** no outcome after 2020-02-29 is used, because COVID relief (SBA payments on 7(a) loans) broke the link between covariates and losses.
  - Snapshots 2000–2013: truth = CIF of charge-off at 72 months.
  - Snapshots 2014–2019: truth = CIF up to min(72 months, 2020-02-29). Implement this exactly as Freddie E3b, including its rule that keeps a cohort only if at least 12 months lie between S and its truth horizon. Every method estimates this same target.
  - Primary analysis: all scored pairs pooled.
- **Classifier features:** see the list below. All categorical features are one-hot, with missing flags.

| Feature | Form |
|---|---|
| Gross approval | log |
| SBA guaranteed share | as given |
| Term in months | as given |
| NAICS | 2-digit sector |
| Business type | as given |
| Business age | where present |
| Franchise | indicator |
| Revolver status | as given |
| Jobs supported | log1p |
| Initial interest rate | where present |
| Subprogram / delivery method | as given |
| Borrower state | as given |

## 3. Data set B: federal civil cases

- **Source:** FJC Integrated Database, civil, cumulative files from FY1988; the civil codebook; and the FJC Biographical Directory of Article III judges (CSV export).
- **Units:** civil cases filed in US district courts from 1989-01-01 to 2016-12-31, with:
  - federal-question or diversity jurisdiction;
  - prisoner petitions (nature of suit 510–560) and social-security cases (861–865) excluded;
  - cases that arrived by transfer from another district, or by MDL transfer, excluded;
  - one record per case (the original filing; reopenings ignored).
- **Cohort:** filing calendar year. **Entry:** filing date.
- **Event:** termination by settlement (disposition "settled"; code 13 in the FJC scheme, to be confirmed from the codebook at the gate). **Competing exit:** any other termination. All other cases are censored at the snapshot.
- **Horizon:** H = 36 months.
- **Groups:** district (94; mapped to state for the unemployment covariate).
- **Calendar covariates:**
  - log number of active, non-senior Article III district judges in the district, monthly, from the Biographical Directory (commission, senior-status and termination dates). Judicial vacancies are set by the confirmation process, not by case outcomes, so (C3) holds.
  - state unemployment rate.
- **Snapshots:** December 31 of 1994 through 2016 (23). **Scored cohorts:** the three youngest filing years, S−2 to S.
- **Truth:** CIF of settlement at 36 months. All horizons end by December 2019, so no COVID truncation is needed.
- **Classifier features** (all known at filing): nature of suit (3-digit), jurisdiction basis, origin, class-action flag, jury demand, amount demanded (log1p, with a missing flag), pro se flag, district.
- **Sensitivity (secondary):** event = settled or voluntarily dismissed.

## 4. Methods (frozen; no tuning on these data)

- **Simple baselines:** events so far; drop recent; last mature cohort; Bornhuetter–Ferguson; Cape Cod (all cohorts).
- **The ladder, one step at a time:**
  - chain ladder (incidence scale);
  - CL + exact ages (hazard scale);
  - + groups;
  - **calendar-adjusted CL** (+ calendar step, guard G1: events × identifying-variation share ≥ 100);
  - **full method** (+ speed step).
- **Speed step, as frozen for the paper:**
  - tempered evidence over {chain ladder, random walk with ω² ∈ {0.001, 0.003, 0.01, 0.03, 0.1}, change point};
  - one common generalized-gamma F0 per snapshot;
  - **corrected plug-in** (App E.5). It is pre-specified here, so it is not post hoc for this study.
- **Competing-risks hazard model:** exactly as the patent M1 (0066), with units and causes from §2/§3.
  - One Poisson regression per cause on the unit-months visible at S, in cells of entry month × group × age month.
  - Terms: a free monthly age baseline per cause, group effects, cohort (entry-year) effects and the calendar covariates.
  - Estimate = the CIF observed at S plus the product integral of the two cause-specific hazards to H along the covariate path.
- **Paths:** realised and forecast for every method with covariates. The primary path is forecast.
- **Intervals:** calibrated 90% intervals (App C.7, Γ = 1), with quarterly cohorts for calibration. Report coverage for pairs with J ≥ 9.
- **Classifiers (H1).** At each snapshot S:
  1. Split the units of the scored cohorts 50/50 at random (seed 20261025) into training and test parts.
  2. Training set: all units that entered in the 10 years up to S, minus the test part. Features are those of §2/§3, plus an entry-year term.
  3. Model: logistic regression (L2, C = 10).
  4. Labels:
     - (a) events so far;
     - (b) positive–unlabelled EM (Bekker–Davis), with each unit's labelling probability = the chain-ladder maturity of its entry month at S;
     - (c) as (b) with the full method's maturity (secondary).
  5. For each scored cohort: level error = |mean predicted probability of the event by H − truth| / truth, on the test part.

## 5. Gate checks (structure only; write `GATE_REPORT.md`)

**A, SBA:**
- Loans per approval year after exclusions ≥ 10,000 for every year 1991–2019.
- ≥ 95% of CHGOFF loans have a charge-off date, and ≥ 95% of PIF loans a paid-in-full date.
- ≥ 98% of loans have a state among the 50 + DC.
- Number of events dated before approval (must be ~0).
- Feature availability by year (share missing).

**B, civil:**
- All needed fields exist. The codebook value for "settled" is confirmed, and the codebook text is quoted.
- Cases per filing year after exclusions ≥ 20,000 for every year 1989–2016.
- ≥ 99% of terminated cases have a termination date, and ≥ 98% a disposition code.
- Every district maps to a state.
- The judge directory covers every district from 1985 to 2019.

**Both:**
- Covariates cover the full date range.
- Peak disk use.

If all gates pass, continue without waiting for CD.

## 6. Metrics and inference (as in the paper)

- **Pair error:** absolute relative error. **Snapshot error:** the mean over its scored cohorts. **Summary:** the median over snapshots.
- **Error ratio:** geometric mean over pairs of the absolute-error ratios (floor 10⁻⁶).
- **Clusters:** entry-year cohorts.
- **95% intervals:** cluster bootstrap, B = 2,000.
- **Tests:** exact cluster sign-flip test on the mean log ratio, shifted by log(margin) for H2 and H3.
  - Use all 2^C sign vectors when C ≤ 20, otherwise 10⁶ random ones (seed 20261025).
  - One-sided as in §1, with Holm over H1–H3 within each data set.
- **Also report:**
  - wins per snapshot;
  - wrong-direction counts (estimate on the wrong side of the latest mature cohort's rate, among pairs whose truth differs from it by ≥ 10%);
  - interval coverage.

## 7. Secondary analyses (pre-specified, not confirmatory)

1. The full Table 2-format decomposition on both paths, with every baseline.
2. **Windows (E2 analogue):** naive Lepski on events so far vs the maturity-weighted Lepski rule (κ = z_{α/N}).
   - Target: the youngest cohort's horizon rate.
   - Subsamples of 10%, 30% and 100% of units, 10 repetitions each below 100%.
   - Report error and window length.
3. **Drift detectors:** ADWIN (0.002) and Page–Hinkley on the monthly event rate by entry month, as in App E.7.
   - Labels: events so far vs chain-ladder- and full-method-corrected labels.
   - Report false and missed "rate fell" alarms under the any-match and one-to-one rules.
4. B with the broader event (settled or voluntarily dismissed).
5. Classifier (c).

## 8. Reporting

- Write `CONFIRM2_RESULTS.md`:
  - the gate report;
  - an H1–H3 table per data set (ratio, 95% interval, one-sided p, Holm p, verdict);
  - the Table 2-format tables;
  - wrong-direction counts;
  - coverage;
  - the secondary results;
  - every deviation.
- Put machine-readable outputs in `results.json`.
- Everything goes into the paper (main text: a short "Confirmatory replication" subsection and one table; appendix: the rest), whatever it shows.
