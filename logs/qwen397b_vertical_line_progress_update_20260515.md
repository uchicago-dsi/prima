# Qwen3.5-397B Vertical-Line QC Progress Update - 2026-05-15

Status: the vertical-line detector artifact workflow is technically unblocked and has a current best prompt that recovered all repaired-QC positives with zero false positives on the 38-exam development set.

## Goal

We are using a vision-language model to reduce manual QC review burden for mammography artifacts. The immediate target is `vertical line (detector artifact)`: a narrow, low-contrast vertical gray seam or stripe that can appear across mammography views. The desired behavior is high recall for true detector seams without flooding review with horizontal compression artifacts or other non-vertical findings.

## Setup

- Model: `Qwen3.5-397B-A17B-FP8-prima-repair`.
- Runtime path: repaired FP8 model with bf16 eager experts, after native FP8 expert execution repeatedly produced CUDA / CUBLAS / `deep_gemm` failure modes.
- Input: four-view mammography montage per exam.
- Output: exam-level suggested QC tags from the existing 13-tag catalog.
- Evaluation set: 38 reviewed exams from the repaired QC state, with 8 exams labeled positive for `vertical line (detector artifact)`.
- Metric unit: exam-level tag classification.

Progress here means preserving recall on all 8 positives while reducing false positives among the 30 repaired-QC negatives.

## Prompt Arms

Each arm used the same model and the same 38-exam evaluation set. The differences were in the prompt instructions and the examples shown to the model.

- `baseline`: A direct marker-classifier prompt for the target tag. It asked whether `vertical line (detector artifact)` was present, without extra target-positive examples or stronger specificity language. This tested whether the model could detect the artifact from the tag definition alone.
- `fewshot retry3`: The same marker-classifier task, but with few-shot examples added. This tested whether examples could recover sensitivity for subtle vertical seams. It did recover all positives, but it also accepted two negatives as vertical-line artifacts.
- `recall_tilted`: A more permissive prompt variant that biased the model toward calling possible vertical-line artifacts. This tested whether missed positives were mainly a thresholding problem. It preserved recall, but the extra permissiveness increased false positives.
- `confidence_specificity`: A stricter prompt that asked for explicit confidence/review output and emphasized rejecting non-target visual patterns. This tested whether adding uncertainty and specificity language could reduce obvious false positives while preserving recall. It reduced false positives from two to one, but confidence was not a useful separator because predictions were still high-confidence.
- `grayseam_targetshots final`: The final prompt tightened the visual definition to a narrow, low-contrast vertical gray seam and used target-positive examples after fixing exemplar selection. This tested the strongest current hypothesis: the model needed the target phenotype separated from horizontal compression or hardware-like artifacts. This arm preserved all positives and removed the remaining false positives on the development set.

## Results

| Prompt arm | TP | FP | FN | TN | Recall | Precision | FP rate | Interpretation |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| baseline | 1 | 0 | 7 | 30 | 0.125 | 1.000 | 0.000 | Too conservative; misses most true seams. |
| fewshot retry3 | 8 | 2 | 0 | 28 | 1.000 | 0.800 | 0.067 | First viable arm; recovered all positives but kept two false positives. |
| recall_tilted | 8 | 5 | 0 | 25 | 1.000 | 0.615 | 0.167 | Recall preserved, but specificity worsened. |
| confidence_specificity | 8 | 1 | 0 | 29 | 1.000 | 0.889 | 0.033 | Reduced false positives, but confidence did not cleanly separate uncertainty. |
| grayseam_targetshots final | 8 | 0 | 0 | 30 | 1.000 | 1.000 | 0.000 | Current lead on this development set. |

Observation: adding specificity language around a “vertical gray seam” and using target-positive examples moved the detector from high-recall/moderate-FP behavior to perfect separation on this small development set.

Interpretation: the model appears capable of recognizing the intended visual pattern when the prompt distinguishes a vertical gray seam from horizontal compression hardware/artifact cases. The result is encouraging but should be treated as development-set evidence, not generalization evidence.

## Visual Examples

Final gray-seam target-shot detections, all true positives in the repaired-QC labels:

[TP 1](qwen397b_vertical_line_pi_examples/final_grayseam_tp_01.png),
[TP 2](qwen397b_vertical_line_pi_examples/final_grayseam_tp_02.png),
[TP 3](qwen397b_vertical_line_pi_examples/final_grayseam_tp_03.png),
[TP 4](qwen397b_vertical_line_pi_examples/final_grayseam_tp_04.png),
[TP 5](qwen397b_vertical_line_pi_examples/final_grayseam_tp_05.png),
[TP 6](qwen397b_vertical_line_pi_examples/final_grayseam_tp_06.png),
[TP 7](qwen397b_vertical_line_pi_examples/final_grayseam_tp_07.png),
[TP 8](qwen397b_vertical_line_pi_examples/final_grayseam_tp_08.png).

False-positive examples from earlier prompts that the final prompt eliminated:

[Few-shot FP 1](qwen397b_vertical_line_pi_examples/fewshot_fp_retired_01.png),
[Few-shot FP 2](qwen397b_vertical_line_pi_examples/fewshot_fp_retired_02.png),
[Confidence-specificity FP](qwen397b_vertical_line_pi_examples/confidence_fp_retired_01.png).

These example links point to local copied montage PNGs with generic filenames for meeting use.

## What Changed

- The native FP8 expert path was not reliable on this stack. Forcing bf16 eager experts made the repaired Qwen3.5-397B path run end-to-end on the vertical-line workload.
- A plain baseline prompt was not sensitive enough: it identified only 1 of 8 positives.
- A few-shot prompt restored recall but introduced false positives.
- A recall-tilted prompt was worse for specificity.
- A first confidence/specificity follow-on reduced false positives, but the target-positive few-shot condition was initially confounded by exemplar selection.
- After fixing exemplar selection and tightening the visual definition to “vertical gray seam,” the final run preserved all positives and eliminated the remaining false positives on the 38-exam set.

## Caveats

- The evaluation set is small: 38 exams total and only 8 positives.
- The final prompt was developed using this same diagnostic set, so the result is best interpreted as proof of feasibility and prompt direction rather than a held-out estimate.
- The final 8 positives should still be reviewed visually before promoting the prompt, especially because the clinical distinction is qualitative and artifact morphology can vary.
- Broader validation should include more scanners, acquisition dates, and negative examples with visually similar compression or detector hardware patterns.

## Next Decisions

1. Decide whether the 8 final true-positive examples match the intended vertical gray seam phenotype.
2. If yes, run the gray-seam/target-shot prompt on a broader held-out QC set.
3. Use held-out exam-level recall, precision, and false-positive review burden as the promotion readout.
4. If held-out false positives recur, stratify them by visual mechanism first rather than adding broader recall language.

Stop rule for this prompt family: if broader validation preserves recall but produces recurring non-vertical hardware false positives, keep the bf16 runtime repair but revise the visual definition or add a second-stage review filter before deployment.
