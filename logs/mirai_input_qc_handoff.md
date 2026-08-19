# Mirai input-QC handoff

Canonical chronology is `logs/lab_notebook_randi.md`. This file holds only
re-entry state; the notebook holds the evidence and reasoning.

## Goal

Finish input QC for the Mirai mammography dataset: every selected L/R CC/MLO
view is a standard, usable acquisition, and every rejected source keeps durable
DICOM lineage plus a reason.

## Current state (2026-08-19)

- The deterministic metadata half is **done and in production**.
  `prima.view_selection.mirai_source_eligibility_reasons` rejects bad
  laterality/projection, any intent other than `FOR PRESENTATION`,
  `DetectorType == FILM`, `PartialView == YES`, and any explicit view modifier.
  Production SoT at `/gpfs/data/huo-lab/Image/ChiMEC/MG/sot`: 40,172 selected
  views, 10,043 exams, 52,699 candidates, 21,591 exclusions. Rollbacks at
  `sot_pre_film_policy` and `sot_pre_current_policy`.
- The visual VLM half is **not done**. The assembled ten-component system has
  failed its registered whole-exam gate three times. Failure is localized to the
  gross-device component: on the production residual audit it produced eight of
  nine false positives and both false negatives, the latter a partially cropped
  device generator in ranks 1 and 2 of one L-MLO slot.
- Validated and retained components: the `DetectorType == FILM` rule (60/60 and
  0/60 on an independent 120-view panel) and the paired SmolVLM MLO-orientation
  component.
- The one-delta gross-device edge-crop challenge is **unblocked and awaiting
  human labels**. Its 2026-07-24 assembly failure was a prompt-hash convention
  defect, now fixed; no GPU work needed repeating. Both arms had already scored
  all 26,450 mining views.

## Evidence paths

- Frozen spec, mining runs, and panel model subsets:
  `qc_redo/auto_qc_development/gross_device_edge_crop_challenge_v1/`
- Blinded 160-view panel awaiting labels:
  `qc_redo/review_batches/gross_device_edge_crop_challenge_v1/`
  (start command is in that directory's `README.md`; loopback port 8767)
- Mining images and manifests:
  `/ess/scratch/scratch1/annawoodard/prima_view_auto_qc/gross_device_edge_crop_challenge_v1/`
- Prior gate result and component attribution:
  `qc_redo/auto_qc_validation/mirai_input_production_residual_v1/`
- Production rebuild provenance:
  `/scratch/annawoodard/prima_film_policy_rebuild/README.md`

## Open uncertainty (decide before spending annotation time)

The registered enriched quota of 40 candidate-only edge-lexicon views could not
be met: only 14 exist in the whole disjoint mining population, 13 are in the
panel, and the frozen `insufficient_reference_rule` cannot help because
expansion draws from the same exhausted population. The mechanism check
therefore needs three human positives out of 13.

Separately, the candidate prompt is globally more conservative than baseline
(1,102 versus 1,427 high-confidence views; +84, -409), the opposite of the
intended recovery direction. These are hidden model dispositions with no truth
meaning, and no human label has been opened, so the stop rule stands: do not
tune either prompt on them.

The decision is whether to label the panel as registered and accept a 13-view
mechanism check, or to mine a population that targets partially cropped devices
directly instead of relying on candidate-only disagreement to surface them.

## Exact first re-entry check

Confirm the panel is still unlabeled and blinded before anything else:

```bash
python -c "import json;s=json.load(open('qc_redo/review_batches/gross_device_edge_crop_challenge_v1/view_qc_state.json'));print(len(s.get('labels',s)))"
```

Expect `0`. If it is nonzero, labeling is already underway: finish all 160
binary labels and every low-confidence adjudication, then score exactly once
with the frozen success rule. Never open the panel model subsets before the
reference is complete and archived.
