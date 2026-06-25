Goal: keep the Qwen3.5-397B vertical-line campaign moving autonomously and determine whether the `bf16` expert fallback gets the repaired model through a 1-exam smoke so the babysitter can promote to the `reviewed33` control.

Current state: live job `827507` is `RUNNING` on `q001` (`4x h200`, `4:00:00` walltime) and is steadily loading repaired expert cache layers; `squeue` at `2026-04-28T11:16 CDT` showed `1:07:13 / 4:00:00`. The timer-driven babysitter is installed and active via `systemd --user`, waking every 5 minutes.

Evidence:
- Live run log: [/net/projects2/annawoodard/qc_redo/submitit_runs/auto_qc_qwen397b_fp8_vertical_line_oneexam_bf16experts_h200_retry240_20260428_092312/submitit_logs/827507_0_log.err](/net/projects2/annawoodard/qc_redo/submitit_runs/auto_qc_qwen397b_fp8_vertical_line_oneexam_bf16experts_h200_retry240_20260428_092312/submitit_logs/827507_0_log.err)
- Notebook: [/net/projects2/annawoodard/qc_redo/interactive_debug/fp8_debug_lab_notebook.md](/net/projects2/annawoodard/qc_redo/interactive_debug/fp8_debug_lab_notebook.md)
- Babysitter script: [scripts/babysit_qwen397b_vertical_line.py](/home/annawoodard/prima/scripts/babysit_qwen397b_vertical_line.py)
- Babysitter prompt/schema: [scripts/qwen397b_babysitter_prompt.txt](/home/annawoodard/prima/scripts/qwen397b_babysitter_prompt.txt), [scripts/qwen397b_babysitter_output_schema.json](/home/annawoodard/prima/scripts/qwen397b_babysitter_output_schema.json)
- Babysitter artifacts: [/net/projects2/annawoodard/qc_redo/interactive_debug/qwen397b_babysitter/babysitter.log](/net/projects2/annawoodard/qc_redo/interactive_debug/qwen397b_babysitter/babysitter.log), [/net/projects2/annawoodard/qc_redo/interactive_debug/qwen397b_babysitter/state/state.json](/net/projects2/annawoodard/qc_redo/interactive_debug/qwen397b_babysitter/state/state.json), [/net/projects2/annawoodard/qc_redo/interactive_debug/qwen397b_babysitter/decisions](/net/projects2/annawoodard/qc_redo/interactive_debug/qwen397b_babysitter/decisions)
- Relevant commits: `f50de63` (`bf16` expert fallback), `8e6700a` (timer-driven babysitter), `cd8f65b` (generalized babysitter decisions)

Open uncertainty: `827507` has not yet reached first-exam scoring, so we still do not know whether the `bf16` fallback truly stabilizes inference or only delays failure. The current babysitter service tick launched at `2026-04-28 10:40:50 CDT` and was still running a fresh `codex exec` when last checked; if it finishes, it should write a new decision JSON under the decisions directory.

Ruled out / current best explanation: the earlier `826096` failure was timeout during repaired-expert setup, not the old `layer 14` / `deep_gemm` crash. Cache-isolation did not fix the FP8 branch. The best current branch is `bf16` expert fallback plus enough walltime.

Worktree status: current `HEAD` is `cd8f65b`. There are unrelated dirty tracked files the next agent should not touch: [AGENTS.md](/home/annawoodard/prima/AGENTS.md), [README.md](/home/annawoodard/prima/README.md), [analysis/analyze_mirai.py](/home/annawoodard/prima/analysis/analyze_mirai.py), [configs/analysis.yaml](/home/annawoodard/prima/configs/analysis.yaml), [qc/qc_gallery.py](/home/annawoodard/prima/qc/qc_gallery.py).

Next action: first check whether `827507` is still progressing or has finished, with `squeue -j 827507 -o '%i %T %M %l %R %N %j'` and `tail -n 80 /net/projects2/annawoodard/qc_redo/submitit_runs/auto_qc_qwen397b_fp8_vertical_line_oneexam_bf16experts_h200_retry240_20260428_092312/submitit_logs/827507_0_log.err`; then inspect the newest babysitter decision in `/net/projects2/annawoodard/qc_redo/interactive_debug/qwen397b_babysitter/decisions`.
