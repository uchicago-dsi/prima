# Mirai Debugging Update

## Question

Why is the current Mirai pipeline underperforming on ChiMEC?

## Bottom Line

The problem is now localized to the **current input-to-tensor path**, not QC, not the calibrator, and not the Mirai codebase in general.

The key result is:

- current code + current zarr input path: weak
- current code + legacy PNG16 input path: strong
- Omoleye-era code + legacy PNG16 input path: strong
- current code + current zarr input path after PIL mode fix: strong

So the failure follows the **current zarr/current-input representation**, not the model weights or the codebase.

## AUC Comparisons on the Same Shard

All runs below used the same comparison shard of ChiMEC exams.

| Condition | 1y | 2y | 3y | 4y | 5y |
|---|---:|---:|---:|---:|---:|
| Current code + current zarr path + current calibrator | 0.4765 | 0.5320 | 0.5388 | 0.5469 | 0.5793 |
| Current code + current zarr path + no calibrator | 0.4765 | 0.5320 | 0.5388 | 0.5469 | 0.5793 |
| Current code + current zarr path + old Omoleye calibrator | 0.4765 | 0.5320 | 0.5388 | 0.5469 | 0.5793 |
| Omoleye-era code + legacy PNG16 path + old calibrator | 0.6858 | 0.6711 | 0.6678 | 0.6763 | 0.6881 |
| Current code + legacy PNG16 path + current calibrator | 0.6857 | 0.6711 | 0.6678 | 0.6763 | 0.6880 |
| Current code + current zarr path after PIL mode fix | 0.6857 | 0.6711 | 0.6678 | 0.6763 | 0.6880 |

## What These Experiments Rule Out

They rule out:

- the current calibrator as the main cause
- the old calibrator as the missing ingredient
- the current Mirai codebase as the main cause
- the encoder / transformer snapshot family as the main cause

They strongly implicate:

- the current zarr-backed image representation, or
- the current conversion from loaded image to final model tensor

This was then validated directly by patching the zarr loader to match the legacy PNG tensorization convention and rerunning the shard.

## Tensor-Level Audit

I compared matched views from the current zarr path and the legacy PNG16 path under the **same current Mirai test transform stack**.

Audit artifact:

- `/gpfs/data/huo-lab/Image/ChiMEC/MG/out/debug_tensor_audit_quick/summary.json`

Sample used for the quick audit:

- `20` matched views
- `5` exams

### Result

For all sampled matched views:

- raw pixels were **identical**
- post-image-transform arrays were **identical**
- final tensors were **not** identical

Summary from the quick audit:

- raw pixel MAE: `0.0`
- raw equal fraction: `1.0`
- post-transform MAE: `0.0`
- post-transform equal fraction: `1.0`
- final tensor MAE: `0.6300`
- final tensor equal fraction: `0.8846`

This means the divergence is happening **after** pixel loading and **after** the deterministic image transforms.

### Mechanism

The two paths preserve the same pixel values but arrive at `ToTensor()` with different PIL modes:

- zarr path: `I;16`
- legacy PNG16 path: `I`

That difference matters because `torchvision.transforms.ToTensor()` does not treat those modes equivalently.

In a direct matched-view check:

- zarr raw array min/max: `0` to `65535`
- legacy raw array min/max: `0` to `65535`
- zarr post-transform array min/max: identical to legacy
- legacy post-transform array min/max: identical to zarr

But the raw tensor produced by `torchvision.transforms.ToTensor()` differs sharply:

- zarr path raw tensor range: `-32768` to `32767`
- legacy path raw tensor range: `0` to `65535`

So the likely bug is:

- the zarr-backed image arrives as PIL mode `I;16`
- the legacy PNG path arrives as PIL mode `I`
- `ToTensor()` handles those modes differently
- that changes the tensor values sent into Mirai, even though the underlying pixels are the same

So the error is not in the image content. The error is in the **mode / dtype convention used immediately before tensorization**.

## Validation of the Fix

I changed the zarr loader so that it materializes the same PIL mode convention as the legacy PNG16 path:

- before: zarr images were opened as `I;16`
- after: zarr images are opened as `I` while preserving the same `0..65535` pixel values

Rerun artifact:

- `/gpfs/data/huo-lab/Image/ChiMEC/MG/out/debug_zarr_modefix/output_chunk000_modefix.csv`

Rerun AUCs:

- 1y: `0.6857`
- 2y: `0.6711`
- 3y: `0.6678`
- 4y: `0.6763`
- 5y: `0.6880`

Interpretation:

- the fix restores the current zarr-backed path to the same performance as the legacy PNG16 path
- this validates that the main bug was the PIL mode / tensorization mismatch

## Interpretation

This is the strongest result so far because it both explains the large AUC gap mechanistically and resolves it on the test shard.

The problem does **not** appear to be:

- low-level pixel corruption
- a different cohort
- a calibrator problem
- a broad old-vs-new codebase issue

The problem was:

- a mode / dtype / tensorization mismatch in the current zarr-backed path

## Comparison to the Omoleye Validation Subset

We were able to locate the original ChiMEC validation scripts and saved outputs in Omoleye's read-only workspace.

Original launch scripts:

- `/gpfs/data/huo-lab/ojomoleye/repos/MiraiValidation_paper/Mirai/demo/validate_chimec.sh`
- `/gpfs/data/huo-lab/ojomoleye/repos/MiraiValidation_paper/Mirai/demo/validate_chimec_yrcut.sh`
- `/gpfs/data/huo-lab/ojomoleye/repos/MiraiValidation_paper/Mirai/demo/validate_chimec_yrcut2.sh`

These scripts confirm that the paper-era runs used:

- old Mirai code
- old calibrator `MIRAI_FULL_PRED_RF.callibrator.p`
- explicit input files under `data/mirai_validation/cleaned/`
- explicit output files under `data/mirai_validation/raw/`

Located paper-era input files:

- `/gpfs/data/huo-lab/ojomoleye/data/mirai_validation/cleaned/allchimec_eligible_MiraiInput.csv`
- `/gpfs/data/huo-lab/ojomoleye/data/mirai_validation/cleaned/allchimec_eligible_MiraiInput_yrcut.csv`
- `/gpfs/data/huo-lab/ojomoleye/data/mirai_validation/cleaned/allchimec_eligible_MiraiInput_yrcut2.csv`

Located paper-era output files:

- `/gpfs/data/huo-lab/ojomoleye/data/mirai_validation/raw/allchimec_MiraiOutput.csv`
- `/gpfs/data/huo-lab/ojomoleye/data/mirai_validation/raw/allchimec_MiraiOutput_yrcut.csv`
- `/gpfs/data/huo-lab/ojomoleye/data/mirai_validation/raw/allchimec_MiraiOutput_yrcut2.csv`

### Key Counts

For the `yrcut` subset:

- paper-era input exams: `6376`
- paper-era output exams: `6337`
- paper-era output patients: `2053`

Using accession-number mapping, we compared those output exams to the current fixed run:

- overlapping exams: `5437`
- overlapping patients: `1889`
- paper-era `yrcut` output exams missing from our current fixed run: `900`

### Why the first overlap attempt failed

The saved paper-era outputs use accession-style exam IDs like:

- `10930784.2O03570`

Our current pipeline uses UID-style `exam_id` values. The overlap only becomes visible after mapping our current exams through:

- `patient_id + accession_number`

### Score Agreement on Shared Exams

On the `5437` shared exams, the risk scores are correlated but not identical.

Correlation of old vs current fixed scores:

- 1y: `0.8525`
- 2y: `0.8589`
- 3y: `0.8624`
- 4y: `0.8628`
- 5y: `0.8654`

Mean absolute error:

- 1y: `0.00604`
- 2y: `0.00833`
- 3y: `0.00832`
- 4y: `0.00757`
- 5y: `0.00728`

So the fixed current pipeline is now much closer to the paper-era results than before, but it is still not numerically identical on the same exams.

### AUC on the Exact Overlapping `yrcut` Exams

Using the paper-era `yrcut` labels and restricting both models to the exact shared exams:

| Horizon | Eligible exams | Cases | Controls | Omoleye output AUC | Current fixed AUC | Delta |
|---|---:|---:|---:|---:|---:|---:|
| 2y | 5215 | 286 | 4929 | 0.6226 | 0.6212 | -0.0014 |
| 3y | 4721 | 549 | 4172 | 0.6306 | 0.6186 | -0.0120 |
| 4y | 4107 | 745 | 3362 | 0.6250 | 0.6129 | -0.0121 |
| 5y | 3572 | 896 | 2676 | 0.6331 | 0.6202 | -0.0130 |

There were no 1-year cases in the exact overlapping `yrcut` subset, so 1-year AUC is undefined there.

Interpretation:

- the catastrophic AUC failure is fixed
- on the same exams, the current fixed pipeline is now close to the paper-era outputs
- but it remains slightly lower on the exact shared subset
- this means there is likely still some residual paper-vs-current input construction difference, not just a broad model failure

This overlap-restricted AUC comparison is important because it already isolates one specific question:

- these AUCs use **our current chosen views** on the exact accession-mapped overlapping exams
- therefore, they remove the effect of the missing `900` exams
- the remaining delta on this overlap-restricted comparison reflects differences within the shared subset, such as view/duplicate selection and residual input construction differences

So:

- the `900` missing exams explain part of the paper-vs-current gap
- but not all of it
- the overlap-restricted AUC table shows that some smaller gap remains even after conditioning on the shared exams

### What the residual difference now looks like

The next pass narrowed the remaining difference further.

#### 1. The missing `900` exams are upstream subset differences

We classified the `900` paper-era `yrcut` output exams that do not appear in the current fixed overlap.

Result:

- all `900` are missing because there is **no matching `patient_id + accession_number` exam** in the current `sot_all` tables
- they are **not** being dropped later by the fixed Mirai inference step
- they are **not** a downstream QC or AUC-filter artifact

So the `900`-exam gap is primarily a **subset construction / exam identity gap** between the paper-era input table and the current rebuilt SoT tables.

#### 2. Shared exams often do not use the same underlying view files

Among the shared accession-mapped exams, we compared the old PNG view files to the current view table by:

- `patient_id`
- accession number
- laterality
- view

Using the SOP instance UID embedded in the old PNG filename, we asked whether the current pipeline is using the same underlying DICOM for the same `L/R x CC/MLO` slot.

Result:

- unique shared view positions compared: `21825`
- unique shared view positions with an exact SOP UID match: `0`
- unique shared view positions with multiple current matches: `2185`

Interpretation:

- even when the accession-level exam overlaps, the paper-era and current pipelines are typically **not** selecting the same underlying image file for a given view slot
- so the remaining paper-vs-current difference is not just score drift on identical images
- part of it is a real **view-selection / exam-construction difference**
- the differences do **not** look like a simple quality-screening advantage such as “old pipeline kept clean presentation images while current pipeline kept marked-up or odd images”
- in the sampled overlapping view positions, the current candidates are overwhelmingly:
  - `for_presentation = True`
  - `is_marked_up = False`
  - and often look like sibling duplicates from the same acquisition family rather than obviously worse images

There are also `211` suffix-insensitive SOP matches, which suggests part of the old-vs-current mismatch is identifier formatting / instance naming rather than a completely different image in every case.

#### 3. Different selected views can still look very similar after preprocessing

We then ran a small tensor audit on `5` shared exams (`20` view pairs) using:

- the old PNG view file from the paper-era `yrcut` input table
- the current selected zarr-backed view for the same accession / laterality / view slot
- the same current Mirai test transform stack

Result:

- the old and current raw images often had different native shapes
  - current zarr-backed views: typically `2048 x 1664`
  - old PNG views: typically `3328 x 2560` or `4096 x 3328`
- after the current scale-and-align transform stack, the resulting tensors were still **extremely similar**
  - mean final-tensor correlation: about `0.9997`
  - mean final-tensor absolute difference: about `0.0076`

Interpretation:

- the current pipeline is often choosing a different underlying view file
- but those different view files are usually still visually and numerically very similar after Mirai preprocessing
- this is consistent with a **small residual performance gap**, not another catastrophic preprocessing failure

#### 4. Batch size `8` is not feasible on the exact Omoleye PNG subset

We launched the current code on the exact copied `yrcut` input CSV inside the held GPU pilot allocations.

Current status:

- `batch_size = 1`: completed successfully
- `batch_size = 8`: failed with CUDA OOM on the full exact PNG subset

That means the directly comparable Omoleye-style setting remains `batch_size = 1`. The larger-batch exact-subset rerun is not a viable apples-to-apples setting on this input representation.

#### 5. On Omoleye's exact `yrcut` input table, the current fixed pipeline is essentially aligned

We then ran the current fixed pipeline directly on a copied version of Omoleye's exact `yrcut` input CSV using the Omoleye-style `batch_size = 1` setting.

Result:

- 1y: `NA` (`0` one-year cases in this subset)
- 2y: `0.6312`
- 3y: `0.6302`
- 4y: `0.6272`
- 5y: `0.6311`

Compared with the saved Omoleye `yrcut` outputs:

- 2y: `0.6226`
- 3y: `0.6306`
- 4y: `0.6250`
- 5y: `0.6331`

Interpretation:

- this does **not** show that batch size alone changed the answer
- we do not have a successful `batch_size = 8` run on the exact PNG subset because that setting OOMs
- instead, it shows that when we remove the rebuilt-manifest / accession-mapping / missing-exam issues and run the current fixed pipeline on **Omoleye's exact input table**, performance is essentially the same as his

So the closer match is not best interpreted as:

- "batch size 1 rescued the model"

It is better interpreted as:

- "the current fixed pipeline matches Omoleye once we run on the same exact subset definition and image list"

That means the residual difference we saw earlier was mostly due to:

- the large zarr tensorization bug, now fixed
- plus subset construction / view-selection differences between the rebuilt current cohort and the paper-era `yrcut` table

### What remains unexplained

The remaining gap is now much narrower and more specific. At this point, the main unresolved issue is not model failure, but cohort/input bookkeeping:

- differences in exact input set construction before inference
- differences in mirrored / accession-linked / duplicated exam handling
- different underlying SOP/view selection within the same accession
- paper-era cohort exclusions that are not yet replicated exactly in the rebuilt all-available cohort

### Role of QC

QC is still pending and may still move the full-cohort numbers, but it is no longer a good explanation for the original catastrophic failure.

What is already resolved:

- the large underperformance on the current pipeline was driven by the zarr `I;16` vs legacy `I` tensorization mismatch
- fixing that bug restores strong shard-level performance and brings the current pipeline into close agreement with the paper-era subset

What QC could still plausibly explain:

- part of the smaller remaining gap between the rebuilt all-available cohort and the paper-era subset
- performance loss from noisier or less curated exams in the widened cohort
- residual differences between the paper-era subset and the current rebuilt cohort after the main tensorization bug is fixed

So the current state is:

- the main Mirai bug is identified and fixed
- QC remains an important pending refinement step, not the leading explanation for the original failure

## Practical Consequence

This is good news for the longer-term PRS project.

It suggests we do **not** need to build on the old Omoleye codebase just to get working Mirai behavior. Instead, we can:

1. use the corrected current input-to-tensor path,
2. keep the current codebase as the development platform,
3. then build feature extraction / PRS integration on top of the corrected current pipeline.

## Current Full-Cohort Result

After propagating the fix to the widened all-available cohort, the corrected current pipeline now gives:

- 1y AUC: `0.6965`
- 2y AUC: `0.6909`
- 3y AUC: `0.6877`
- 4y AUC: `0.6834`
- 5y AUC: `0.6816`

On the Omoleye-style `TTC ≥ 6 months` row:

- 1y AUC: `0.6231`
- 2y AUC: `0.5970`
- 3y AUC: `0.5912`
- 4y AUC: `0.5909`
- 5y AUC: `0.5874`

## Next Step

The next step is no longer to find the main zarr bug. The remaining work is to explain the smaller residual difference to the paper-era subset:

1. compare Omoleye input subset construction directly against our current manifest
2. decide whether we need to explicitly recreate the paper-era subset definition inside the current pipeline for manuscript-quality replication
3. decide how much of the remaining cohort difference is:
   - missing exams from the rebuilt SoT tables
   - different SOP/view selection within overlapping accessions
   - paper-era cohort bookkeeping that we no longer want to preserve

## Validation-Paper Expansion Plan

Reading the final Omoleye 2023 PDF clarifies which parts of the original paper are most natural to extend in the larger corrected cohort.

What the original paper actually did:

- external validation of Mirai in a high-risk, racially diverse ChiMEC cohort
- compared Mirai with:
  - BI-RADS assessment
  - visually assessed breast density
  - BDC weighted density
- reported:
  - per-horizon AUCs
  - Harrell concordance index
  - subgroup AUCs by age, race/ethnicity, density, receptor subtype, and grade
  - mirroring experiments showing the affected breast drives much of the short-term signal
  - a simple combined Mirai + BI-RADS 1-year logistic model

The best paper-quality extensions in the expanded corrected cohort are therefore:

1. Reproduce the original paper tables and figures on the corrected expanded cohort.
   - This is the cleanest validation contribution.
   - It should include the same TTC `< 6 months` exclusion used in the paper.
   - It should explicitly show old paper cohort versus expanded cohort counts and performance.

2. Directly quantify transport from the paper-era subset to the widened cohort.
   - same institution
   - same model
   - corrected preprocessing
   - larger, less curated cohort
   - This becomes a useful “how much does cohort construction matter?” analysis.

3. Re-run and expand the mirroring / affected-breast analyses.
   - This is the most biologically interesting part of the original paper.
   - It also helps explain whether the enlarged cohort still supports the same “precancerous signal” interpretation.

4. Add calibration and score-distribution analyses.
   - The original paper emphasized discrimination more than calibration.
   - Calibration, observed/expected risk, and threshold behavior would add something genuinely new.

5. Add vendor / calendar-period robustness.
   - The original paper did not make this the main focus.
   - In the expanded cohort, this is a strong practical validation question.

6. Add repeated-exam sensitivity analyses.
   - first exam per patient
   - last eligible prediagnostic exam
   - patient-weighted all exams
   - This will matter because the rebuilt cohort and the expanded cohort both include repeated exams.

7. If BI-RADS and density are sufficiently complete, re-run the combined-model analyses.
   - Mirai alone
   - BI-RADS alone
   - Mirai + BI-RADS
   - density / BDC comparisons
   - This gives the validation paper a concrete clinical-comparison angle beyond “Mirai AUC improved.”

What seems most worth doing before PRS is available:

- exact replication of the paper’s main analyses on the corrected expanded cohort
- explicit comparison between paper-era curated subset and expanded cohort
- mirroring / affected-breast analyses
- calibration
- vendor / time-period robustness

What is probably lower priority:

- spending more time chasing tiny residual paper-era score differences
- old-code-path reproduction beyond what is already enough to show the main bug is fixed
- new modeling before QC is stabilized

## Additional Analyses Worth Doing Before PRS

Some extensions are stronger than a simple "same paper, bigger cohort" update.

High-value additions:

- calibration by horizon
  - reliability plots
  - observed-to-expected ratios by score decile
  - calibration slope / intercept if we want a compact summary
- transportability from the paper-era curated subset to the widened cohort
  - this is now a real story because we found and fixed a pipeline bug and can separate that from cohort construction
- threshold-oriented risk enrichment
  - what fraction of future cancers land in the top `1%`, `5%`, or `10%` of Mirai risk
  - PPV / sensitivity tradeoff at clinically interpretable cut points
- stability across acquisition context
  - manufacturer/vendor
  - calendar period
  - eventually QC-defined technical subsets
- repeated-exam sensitivity
  - all exams patient-weighted
  - first exam per patient
  - last exam per patient
  - random one exam per patient
- mirroring / affected-breast analyses
  - this was one of the most interesting parts of the original paper and is more publishable than a plain AUC update
- cumulative incidence or Kaplan-Meier style stratification by Mirai risk quantile
  - useful if we want a validation-paper figure that reads more clinically than AUC alone

The best overall framing is probably:

- corrected technical validation
- expanded-cohort transportability
- robustness and biological specificity analyses

## Pre-QC Extension Results on the Corrected Expanded Cohort

I ran a first extension pass on the corrected all-available cohort using the same QC filters as the current main run, with the primary paper-matched restriction:

- `time to cancer >= 6 months`

Artifacts:

- `/gpfs/data/huo-lab/Image/ChiMEC/MG/out_all_modefix/validation_extensions/manufacturer_auc_ttc6.csv`
- `/gpfs/data/huo-lab/Image/ChiMEC/MG/out_all_modefix/validation_extensions/calendar_period_auc_ttc6.csv`
- `/gpfs/data/huo-lab/Image/ChiMEC/MG/out_all_modefix/validation_extensions/exam_selection_auc_ttc6.csv`

Primary filtered cohort:

- `13,086` exams
- `2,973` patients

### Manufacturer robustness

Main manufacturers after filtering:

- `HOLOGIC, Inc.`: `9,189` exams
- `GE MEDICAL SYSTEMS`: `3,481` exams
- `LORAD`: `246` exams

Representative AUCs:

- HOLOGIC:
  - 1y `0.632`
  - 2y `0.603`
  - 3y `0.610`
  - 4y `0.610`
  - 5y `0.609`
- GE:
  - 1y `0.587`
  - 2y `0.573`
  - 3y `0.552`
  - 4y `0.555`
  - 5y `0.552`

Interpretation:

- performance is not catastrophically different by manufacturer
- but HOLOGIC is consistently stronger than GE by about `0.03` to `0.06` AUC depending on horizon
- that makes vendor/manufacturer robustness worth keeping in a validation paper

### Calendar-period robustness

Calendar bins:

- `2001-2010`: `3,748` exams
- `2011-2015`: `5,091` exams
- `2016-2020`: `3,049` exams
- `2021-2025`: `1,198` exams

Representative 1-year AUCs:

- `2001-2010`: `0.647`
- `2011-2015`: `0.622`
- `2016-2020`: `0.648`
- `2021-2025`: `0.680`

Representative 5-year AUCs:

- `2001-2010`: `0.629`
- `2011-2015`: `0.614`
- `2016-2020`: `0.626`
- `2021-2025`: `0.647`

Interpretation:

- there is no obvious catastrophic calendar-era collapse after the technical fix
- the newest bin has wide uncertainty because it has very few cases
- the middle eras are fairly consistent, which is reassuring for a wider-cohort validation story

### Repeated-exam sensitivity

This was the most informative extension result so far.

At `TTC >= 6 months`, 1-year / 5-year AUCs were:

- all exams, patient-weighted:
  - 1y `0.623`
  - 5y `0.587`
- first exam per patient:
  - 1y `0.668`
  - 5y `0.623`
- last exam per patient:
  - 1y `0.600`
  - 5y `0.565`
- one random exam per patient:
  - 1y `0.643`
  - 5y `0.601`

Interpretation:

- apparent performance depends materially on how repeated exams are handled
- first-exam-per-patient performance is clearly stronger than last-exam-per-patient performance
- this means the repeated-exam design choice should be explicit in any validation manuscript and ideally treated as a sensitivity analysis, not a hidden implementation detail

This result is useful even before QC is finished because it changes how we should frame the cohort:

- "all available exams" and "one exam per patient" are not interchangeable evaluation targets

#### Repeated-exam sensitivity table

| Evaluation design | 1y AUC | 2y AUC | 3y AUC | 4y AUC | 5y AUC |
|---|---:|---:|---:|---:|---:|
| All exams, patient-weighted | 0.623 | 0.597 | 0.591 | 0.591 | 0.587 |
| First exam per patient | 0.668 | 0.644 | 0.630 | 0.631 | 0.623 |
| Last exam per patient | 0.600 | 0.575 | 0.568 | 0.564 | 0.565 |
| One random exam per patient | 0.643 | 0.608 | 0.604 | 0.603 | 0.601 |

Interpretation:

- first-exam-per-patient is clearly the strongest of the one-exam-per-patient designs
- last-exam-per-patient is the weakest
- the patient-weighted all-exam analysis lands between them, but closer to the lower-performing settings at longer horizons
- this should be treated as a real design choice in the manuscript, not just an implementation detail

## Practical Checklist

### Must-have before writing

1. Re-run the main corrected cohort after QC is finished.
2. Re-run the manufacturer, calendar-period, and repeated-exam sensitivity tables after QC.
3. Reproduce the original paper-style subgroup table as closely as the available metadata allows:
   - age
   - race/ethnicity
   - receptor subtype
   - grade
4. Re-run the mirroring / affected-breast analyses on the corrected expanded cohort.
5. Quantify the paper-era subset versus widened-cohort transport gap explicitly.

### Strong additions if time allows

1. Add calibration by horizon.
2. Add score-quantile enrichment analyses:
   - top `1%`
   - top `5%`
   - top `10%`
3. Add cumulative-incidence style stratification by Mirai risk quantile.
4. Add threshold-oriented tables that read clinically, not just statistically.

### After QC

1. Treat all current pre-QC extension results as provisional and rerun them on the QC-final cohort.
2. Check whether vendor differences shrink after QC.
3. Check whether the repeated-exam sensitivity pattern persists after QC.
4. Reassess whether any residual gap versus the paper-era subset is still attributable to cohort construction versus data quality.

## Key Artifacts

- current calibrated output:
  - `/gpfs/data/huo-lab/Image/ChiMEC/MG/out/mirai_shards/output_000.csv`
- Omoleye-era output:
  - `/gpfs/data/huo-lab/Image/ChiMEC/MG/out/debug_omoleye_chunk000/omoleye_output_chunk000.csv`
- current code on legacy PNG16:
  - `/gpfs/data/huo-lab/Image/ChiMEC/MG/out/debug_currentcode_legacypng/output_chunk000_currentcode_legacypng.csv`
- condition comparison summary:
  - `/gpfs/data/huo-lab/Image/ChiMEC/MG/out/debug_compare_conditions/condition_auc_summary.csv`
- tensor audit summary:
  - `/gpfs/data/huo-lab/Image/ChiMEC/MG/out/debug_tensor_audit_quick/summary.json`
- zarr-path rerun after fix:
  - `/gpfs/data/huo-lab/Image/ChiMEC/MG/out/debug_zarr_modefix/output_chunk000_modefix.csv`
- paper-era input/output subset:
  - `/gpfs/data/huo-lab/ojomoleye/data/mirai_validation/cleaned/allchimec_eligible_MiraiInput_yrcut.csv`
  - `/gpfs/data/huo-lab/ojomoleye/data/mirai_validation/raw/allchimec_MiraiOutput_yrcut.csv`
- exact-subset comparison outputs:
  - `/gpfs/data/huo-lab/Image/ChiMEC/MG/out/debug_omoleye_subset_compare/summary.json`
  - `/gpfs/data/huo-lab/Image/ChiMEC/MG/out/debug_omoleye_subset_compare/omoleye_yrcut_missing_reason_counts.csv`
  - `/gpfs/data/huo-lab/Image/ChiMEC/MG/out/debug_omoleye_subset_compare/omoleye_yrcut_view_identity.csv`
  - `/gpfs/data/huo-lab/Image/ChiMEC/MG/out/debug_omoleye_subset_compare/output_currentcode_omoleye_yrcut_bs1.csv`
- fixed full-cohort outputs:
  - `/gpfs/data/huo-lab/Image/ChiMEC/MG/out_all_modefix/validation_output.csv`
  - `/gpfs/data/huo-lab/Image/ChiMEC/MG/out_all_modefix/validation_output_clinical.csv`
  - `/gpfs/data/huo-lab/Image/ChiMEC/MG/out_all_modefix/model_performance_metrics.csv`
