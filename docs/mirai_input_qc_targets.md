# Mirai input-QC targets

This note separates documented Mirai input requirements from cohort-specific
exclusions and from exploratory image-quality checks. A positive QC label is a
specific routing signal; it does not mean that the entire mammogram is "bad."

## Documented model requirements

The vendored Mirai README requires one L CC, L MLO, R CC, and R MLO image, all
in For Presentation mode. It says that non-standard or unilateral exams, For
Processing images, and images with CAD or human markup are unsupported. It also
notes that the released model was developed on Hologic Selenia and Selenia
Dimensions devices and was not established on other machines.

These requirements imply the following QC layers:

- structural exam checks for exact four-view completeness and valid
  laterality/projection;
- source-header checks that each slot is an unmodified, non-partial CC or MLO
  view. A DICOM whose primary `ViewPosition` is CC/MLO is still non-standard
  when `ViewModifierCodeSequence` says Implant Displaced, rolled, spot
  compression, magnification, or another modifier;
- metadata checks for For Presentation mode;
- a visual check for burned-in CAD or human markup;
- a view-level check that a selected CC/MLO image is not actually a
  non-standard spot-compression or magnification acquisition; and
- device-stratified performance reporting rather than visually rejecting an
  otherwise valid image solely because of manufacturer metadata.

## CHiMEC validation-cohort comparability

The Omoleye et al. cohort note records additional exclusions for implants,
foreign devices, and burned-in annotations. These are cohort definitions, not
proof that the Mirai executable cannot produce a score for such images.

- `visible breast implant`: label at view level, aggregate with ANY across the
  exam, and route the exam outside the paper-comparable analysis until implant
  performance is validated. Implant-displaced views are not valid substitutes
  for the four standard Mirai slots. Do not search for a same-exam non-implant
  fallback merely to erase an exam-level implant finding.
- `large foreign device obscuring the mammogram`: keep separate from implants
  and from small clips. Define and validate this target before using it.
- `burned-in CAD or human markup`: exclude the affected input and seek an
  unmarked standard-view alternative when one exists.

## Artifact-specific fallback targets

Artifacts such as a vertical detector seam are view-local. A positive view can
be replaced only by another source image from the same exam, laterality, and
projection that passes every frozen target. If the exact slot is exhausted,
route the exam to human review rather than silently dropping it.

Digitized hard-copy film appearance is tracked separately because visible film
conversion may create a distribution shift. Metadata such as `DetectorType ==
FILM` is a challenger baseline, not reference truth.

## Findings that are not QC failures by default

Biopsy clips, surgical clips, and routine skin markers are common clinical
content. Neither the vendored Mirai README nor the local Omoleye cohort note
lists small clips as an exclusion. Do not reject them by default. They can be
audited later as potential shortcut or subgroup variables if model behavior
provides a reason.

## Current annotation order

1. Freeze the validated vertical detector-seam classifier.
2. Audit digitized hard-copy film appearance against the FILM metadata rule.
3. Validate `visible breast implant` on a blinded enriched-plus-random panel.
4. Validate `non-standard spot-compression or magnification view` on a separate
   blinded panel.
5. Next, validate burned-in CAD or human markup.
6. Only then consider large obscuring foreign devices, incomplete breast
   coverage/positioning, and severe exposure or contrast failures as distinct
   targets.

References: `vendor/mirai/README.md` and
`docs/papers/omoleye_2023_ryai_220299_reference.md`.
