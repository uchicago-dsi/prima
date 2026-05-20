# Mirai Longitudinal Trajectory Analysis Plan

## Primary question
Among patients who eventually develop cancer, do Mirai scores increase as exams get closer to diagnosis?

## Secondary question
Is any apparent increase specific to cases, or does a similar drift appear in controls?

## Dataset to build
One row per exam.

Required columns:
- `patient_id`
- `exam_id`
- `study_date`
- `years_to_cancer`
- `years_to_last_followup`
- case/control label
- Mirai risk columns, at minimum `1_year_risk`, `2_year_risk`, `5_year_risk`
- optional covariates: `age_at_exam`, `race_category`, `site`, `device_manufacturer`, `device_model`

Derived columns:
- `is_case_patient`
- `time_to_dx_years`
- `time_from_first_exam_years`
- `time_to_last_exam_years`
- `n_exams_per_patient`
- `exam_order`

## Cohort definitions

### Primary cohort
- case patients only
- pre-diagnostic exams only
- exams with non-missing `study_date`
- for slope analyses: patients with at least 2 pre-diagnostic exams

### Primary score
- `1_year_risk`

### Secondary scores
- `2_year_risk`
- `5_year_risk`

## Recommended time axis
Use `time_to_dx_years` for cases:
- smaller means closer to diagnosis
- easiest clinical interpretation

Preferred definition:
- recompute from raw dates when available: `(dx_date - study_date) / 365.25`

Fallback:
- existing integer `years_to_cancer`

The date-based version is preferred because integer-flooring in `years_to_cancer` can flatten trajectories.

## Core analyses

### 1. Descriptive cohort table
Report:
- number of case patients
- number of pre-diagnostic exams
- exams per patient: median, IQR, range
- span from earliest pre-diagnostic exam to diagnosis: median, IQR
- number of patients with 1, 2, 3, 4+ exams

### 2. Primary plot: patient-weighted binned trajectory
For case patients, bin `time_to_dx_years` into:
- `(5, inf]`
- `(3, 5]`
- `(2, 3]`
- `(1, 2]`
- `(0.5, 1]`
- `(0, 0.5]`

Within each bin:
- compute patient-weighted mean score
- each patient has total weight 1 within that bin
- if a patient has multiple exams in the same bin, divide their weight across those exams

Outputs:
- mean `1_year_risk` by bin
- 95% CI
- counts of patients and exams per bin

### 3. Per-patient slope analysis
Restrict to case patients with at least 2 pre-diagnostic exams.

For each patient:
- regress score on `time_to_dx_years`
- store slope

Interpretation:
- negative slope means score increases as diagnosis approaches

Outputs:
- histogram of slopes
- median slope and IQR
- fraction of patients with negative slope
- sign test or Wilcoxon test against 0

### 4. Mixed-effects model
Fit on case patients:
- `score ~ time_to_dx_years + (1 | patient_id)`

Possible extension:
- `score ~ spline(time_to_dx_years) + (1 | patient_id)`

Outputs:
- coefficient for `time_to_dx_years`
- standard error / CI
- predicted mean trajectory

Interpretation:
- negative coefficient means scores rise toward diagnosis on average

### 5. Spaghetti plot
Sample 30-50 case patients with at least 3 exams.

Plot:
- x = `time_to_dx_years`
- y = `1_year_risk`
- one line per patient

Purpose:
- show whether the average trend reflects many patients or just a few

## Secondary analyses

### 6. Control comparison
Controls do not have diagnosis dates, so they need a different alignment.

Possible control axes:
- `time_from_first_exam_years`
- `time_to_last_observed_exam_years`

Preferred secondary comparison:
- cases aligned to diagnosis
- controls aligned to last observed exam

Interpretation of controls is supportive, not primary.

### 7. Exclusion window near diagnosis
Repeat primary analysis after excluding exams within:
- 90 days
- 180 days

Purpose:
- separate imminent diagnostic workup signal from gradual pre-diagnostic rise

### 8. Horizon comparison
Repeat primary trajectory for:
- `1_year_risk`
- `2_year_risk`
- `5_year_risk`

Expectation:
- shorter-horizon scores should rise more sharply near diagnosis

### 9. Date-based sensitivity analysis
Compare:
- integer `years_to_cancer`
- continuous date-derived `time_to_dx_years`

Purpose:
- check whether discretization distorts the trend

## Decisions fixed before coding
- primary score: `1_year_risk`
- primary cohort: case patients only, pre-diagnostic exams only
- minimum exams for slope analysis: `>= 2`
- near-diagnosis exclusion: none in primary; `<90d` and `<180d` as sensitivity analyses
- weighting rule: patient-weighted within bins
- preferred time variable: date-derived continuous years

## Concrete outputs

Tables:
- `trajectory_cohort_summary.csv`
- `case_binned_trajectory.csv`
- `case_patient_slopes.csv`
- `case_mixed_model_summary.txt`
- `control_binned_trajectory.csv`
- `trajectory_sensitivity_summary.csv`

Figures:
- `case_trajectory_1yr.png`
- `case_spaghetti_sample.png`
- `case_slope_histogram.png`
- `case_vs_control_trajectory_1yr.png`
- `trajectory_by_horizon.png`

## Minimal implementation plan
1. Build exam-level table with scores, dates, and labels.
2. Restrict to case pre-diagnostic exams.
3. Create continuous `time_to_dx_years`.
4. Make primary binned trajectory.
5. Compute per-patient slopes.
6. Fit mixed model.
7. Add sensitivity analyses.
8. Add control comparison.

## First outputs to prioritize
If the goal is the shortest path to an answer, prioritize:
- primary binned trajectory
- slope histogram
- mixed-model coefficient

These are enough to tell whether scores rise toward diagnosis on average.
