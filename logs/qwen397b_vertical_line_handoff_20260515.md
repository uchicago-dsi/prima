# Qwen3.5-397B Vertical-Line Handoff - 2026-05-15

## Goal

Recover the lost conversation state for the Qwen3.5-397B FP8 vertical-line auto-QC campaign and identify the latest results plus the next useful action.

## Current State

Campaign status: completed and idle.

The bf16-expert repair path is the current working runtime path. The completed jobs show that `Qwen3.5-397B-A17B-FP8-prima-repair` with `PRIMA_QWEN35_FP8_FORCE_BF16_EXPERTS=1` avoids the old native FP8 MoE failure family (`layer 14`, CUDA/device-side assert, CUBLAS, `deep_gemm` abort) for this vertical-line workload.

The best prompt result on the 38-exam repaired-QC dev set is the final gray-seam/target-shot rescue artifact:

- Artifact: `/net/projects2/annawoodard/qc_redo/auto_qc_runs/vertical_line_confidence_grayseam_targetshots_rescue6h_20260430/qwen397b_vertical_line_ablation_confidence_grayseam_targetshots_rescue6h_20260430.json`
- Job: `831417`, completed on `p001` at `2026-05-01 20:50:57 CT`, `ExitCode=0:0`
- Metrics from `scripts/evaluate_auto_qc_ablation.py` against `/net/projects2/annawoodard/qc_redo/qc_state_repaired.json`: `TP=8 FP=0 FN=0 TN=30`, recall `1.0`, precision `1.0`, FP rate `0.0`

## Result Sequence

- Retry3 prompt ablation, output root `/net/projects2/annawoodard/qc_redo/auto_qc_runs/vertical_line_prompt_ablation_20260429_launchblocking_retry3`
  - `baseline`: `TP=1 FP=0 FN=7 TN=30`, recall `0.125`, precision `1.0`, FP rate `0.0`
  - `fewshot`: `TP=8 FP=2 FN=0 TN=28`, recall `1.0`, precision `0.8`, FP rate `0.066667`
  - `recall_tilted`: `TP=8 FP=5 FN=0 TN=25`, recall `1.0`, precision `0.615385`, FP rate `0.166667`
- Confidence/specificity follow-on, output root `/net/projects2/annawoodard/qc_redo/auto_qc_runs/vertical_line_confidence_specificity_20260430`
  - `confidence_specificity`: `TP=8 FP=1 FN=0 TN=29`, recall `1.0`, precision `0.888889`, FP rate `0.033333`
  - This run was runtime-clean but confounded for target-positive few-shot selection before commit `ed8d7b4`.
- Corrected gray-seam/target-shot rescue, output root `/net/projects2/annawoodard/qc_redo/auto_qc_runs/vertical_line_confidence_grayseam_targetshots_rescue6h_20260430`
  - `grayseam_targetshots`: `TP=8 FP=0 FN=0 TN=30`, recall `1.0`, precision `1.0`, FP rate `0.0`

## Live Automation

- Timer: `prima-qwen397b-babysitter.timer`, active/waiting, hourly.
- Latest decision checked: `/net/projects2/annawoodard/qc_redo/interactive_debug/qwen397b_babysitter/decisions/20260515_130006.json`
- Latest decision state: `deterministic_completed_idle`, `campaign_status=completed`, `did_act=false`, `needs_human=false`.
- Active matching Slurm jobs: none at the 2026-05-15 re-entry check.
- Babysitter state file: `/net/projects2/annawoodard/qc_redo/interactive_debug/qwen397b_babysitter/state/state.json`

## Evidence Paths

- Canonical notebook: `/net/projects2/annawoodard/qc_redo/interactive_debug/fp8_debug_lab_notebook.md`
- Retry3 metrics: `/net/projects2/annawoodard/qc_redo/auto_qc_runs/vertical_line_prompt_ablation_20260429_launchblocking_retry3/ablation_metrics.csv`
- Confidence/specificity metrics: `/net/projects2/annawoodard/qc_redo/auto_qc_runs/vertical_line_confidence_specificity_20260430/confidence_specificity_metrics.csv`
- Final artifact: `/net/projects2/annawoodard/qc_redo/auto_qc_runs/vertical_line_confidence_grayseam_targetshots_rescue6h_20260430/qwen397b_vertical_line_ablation_confidence_grayseam_targetshots_rescue6h_20260430.json`
- Final job logs: `/net/projects2/annawoodard/qc_redo/submitit_runs/qwen397b_vertical_line_ablation_confidence_grayseam_targetshots_rescue6h_20260430_20260430_190417/submitit_logs`

## Open Uncertainty

The final metrics are against the repaired 38-exam QC dev set. They should not be treated as broad generalization evidence until the 8 final suggestions are qualitatively reviewed and the prompt is tested on a larger or held-out QC set.

## Next Step

First re-entry check:

```bash
micromamba run -p /net/projects2/annawoodard/micromamba/envs/prima python scripts/evaluate_auto_qc_ablation.py \
  --qc-file /net/projects2/annawoodard/qc_redo/qc_state_repaired.json \
  --tag "vertical line (detector artifact)" \
  --run-file fewshot_retry3=/net/projects2/annawoodard/qc_redo/auto_qc_runs/vertical_line_prompt_ablation_20260429_launchblocking_retry3/qwen397b_vertical_line_ablation_launchblocking_retry3_20260429_fewshot_20260429_133321.json \
  --run-file confidence_specificity=/net/projects2/annawoodard/qc_redo/auto_qc_runs/vertical_line_confidence_specificity_20260430/qwen397b_vertical_line_ablation_confidence_specificity_20260430.json \
  --run-file grayseam_targetshots=/net/projects2/annawoodard/qc_redo/auto_qc_runs/vertical_line_confidence_grayseam_targetshots_rescue6h_20260430/qwen397b_vertical_line_ablation_confidence_grayseam_targetshots_rescue6h_20260430.json
```

If continuing the work, qualitatively inspect the 8 final gray-seam/target-shot suggestions. If they are acceptable, promote that prompt configuration or run it on a broader held-out QC set. If no follow-on campaign is planned, the timer can remain enabled because completed-idle ticks are deterministic and cheap; disable it only if hourly idle records become noise.

## Repo State At Re-entry

Pre-existing dirty tracked files were present before this handoff: `AGENTS.md`, `README.md`, `analysis/analyze_mirai.py`, and `configs/analysis.yaml`. This handoff adds `logs/qwen397b_vertical_line_handoff_20260515.md` and updates the external canonical notebook.
