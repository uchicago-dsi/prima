# Mirai Validation Extension Checklist

This is the working checklist for turning the corrected Mirai rerun into a validation-paper package before PRS is ready.

Primary reference:

- [lab_meeting_mirai_debug.md](./lab_meeting_mirai_debug.md)

Current corrected run:

- `/gpfs/data/huo-lab/Image/ChiMEC/MG/out_all_modefix`

Primary comparable cohort:

- `time to cancer >= 6 months`

## Must Have

1. Re-run the corrected main metrics after QC is finalized.
2. Re-run the current pre-QC extension analyses after QC:
   - manufacturer robustness
   - calendar-period robustness
   - repeated-exam sensitivity
3. Reproduce the original paper-style subgroup analyses as closely as metadata allows:
   - age
   - race/ethnicity
   - receptor subtype
   - tumor grade
4. Re-run the mirroring / affected-breast analyses on the corrected expanded cohort.
5. Explicitly compare:
   - paper-era curated subset
   - corrected widened cohort
   - overlap-restricted subset

## Strong Additions

1. Calibration by horizon:
   - reliability plots
   - observed-to-expected ratios
   - calibration slope / intercept
2. Threshold-oriented risk enrichment:
   - top `1%`
   - top `5%`
   - top `10%`
   - sensitivity / PPV style summaries
3. Cumulative-incidence or Kaplan-Meier style risk-stratified plots.
4. Clinical-comparison models if metadata are sufficiently complete:
   - BI-RADS alone
   - Mirai + BI-RADS
   - density / BDC comparisons

## Nice To Have

1. Vendor-manufacturer plus calendar-period interaction checks.
2. Technical-subset analyses once QC outputs are ready:
   - scanned film removed
   - implant removed
   - other QC-defined exclusions
3. More formal transportability framing:
   - quantify the paper-era subset advantage after correcting the pipeline
   - show how much of the difference is cohort construction versus technical bug versus QC

## Already Done Pre-QC

1. Corrected the main zarr tensorization bug.
2. Reproduced paper-era performance on Omoleye's exact `yrcut` input table.
3. Ran the first extension pass on the corrected expanded cohort:
   - `/gpfs/data/huo-lab/Image/ChiMEC/MG/out_all_modefix/validation_extensions/manufacturer_auc_ttc6.csv`
   - `/gpfs/data/huo-lab/Image/ChiMEC/MG/out_all_modefix/validation_extensions/calendar_period_auc_ttc6.csv`
   - `/gpfs/data/huo-lab/Image/ChiMEC/MG/out_all_modefix/validation_extensions/exam_selection_auc_ttc6.csv`

## Where To Put Code

- Keep `analysis/analyze_mirai.py` as the main cohort-level evaluator.
- Cohort-level robustness and sensitivity analyses now live there too.
- Keep image-manipulation experiments separate:
  - mirroring
  - affected-breast-specific experiments
  - other view-editing ablations

The main extension outputs are now emitted directly by `analyze_mirai.py` under:

- `/gpfs/data/huo-lab/Image/ChiMEC/MG/out_all_modefix/validation_extensions`
