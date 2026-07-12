# vLLM-backed mammography auto-QC

Prima uses a managed, loopback-only vLLM service for open-weight vision-model
inference. Montage generation, prompts, checkpointable run JSON, pilot queues,
metrics, and the QC gallery remain Prima-owned. Only the model-loading and
generation boundary moved from direct Transformers calls to vLLM's
OpenAI-compatible API.

The portable server lifecycle is provided by BSD-3-Clause dependency
`dsi-local-llms`. The pre-PR development build is from local feature-branch
commit `d4f2d744417252e09bb0a939447a43edbc8c0777`; its EHR extraction pipeline
is not installed or used by Prima. The upstream license is retained at
`docs/licenses/local-llms-BSD-3-Clause.txt`.

## Runtime and model snapshots

Create the isolated serving environment. Do not install vLLM into the general
Prima environment.

```bash
mkdir -p /scratch/annawoodard/tmp/prima-vllm-pip
mkdir -p /scratch/annawoodard/tmp/prima-vllm-runtime
TMPDIR=/scratch/annawoodard/tmp/prima-vllm-pip \
  micromamba create -y -f env-vllm.yaml
PYTHONNOUSERSITE=1 \
TMPDIR=/scratch/annawoodard/tmp/prima-vllm-pip \
UV_CACHE_DIR=/scratch/annawoodard/uv-cache/prima-vllm \
UV_LINK_MODE=copy \
  /gpfs/data/huo-lab/Image/annawoodard/micromamba/envs/prima-vllm/bin/uv \
  pip install \
  --python /gpfs/data/huo-lab/Image/annawoodard/micromamba/envs/prima-vllm/bin/python \
  --torch-backend=cu130 \
  vllm==0.24.0 openai==2.45.0 torch==2.11.0 torchvision==0.26.0 \
  torchaudio==2.11.0 numpy==2.3.5

# Temporary local build before the upstream feature branch is pushed.
LOCAL_LLMS_WHEEL=/scratch/annawoodard/builds/dsi-local-llms/d4f2d744417252e09bb0a939447a43edbc8c0777/dsi_local_llms-0.1.1-py3-none-any.whl
echo '612e137b0b847bbc4a4a01c7ce7c922c4765f1b5587961c22312e7d738ee6ce1  '"$LOCAL_LLMS_WHEEL" | sha256sum --check
/gpfs/data/huo-lab/Image/annawoodard/micromamba/envs/prima-vllm/bin/python \
  -m pip install --no-deps "$LOCAL_LLMS_WHEEL"
```

After the upstream PR is merged, replace the local-wheel step with an immutable
Git dependency pinned to the reviewed reachable commit. Do not preserve the
scratch wheel as the long-term dependency source.

The executable scratch `TMPDIR` avoids the cluster's `noexec` `/tmp` mount.
It must also be exported for submission because Triton compiles and loads
runtime kernels while vLLM starts; the submitter validates this before queueing.
The environment also pins NVIDIA CUDA 13.0's `nvcc`; the system compiler on
the H200 image is not executable by jobs, while vLLM's DeepGEMM path JITs CUDA
kernels during startup. The managed server points `CUDA_HOME` at this
environment and keeps FlashInfer's generated build files under `TMPDIR` so a
bad system-toolchain cache cannot leak into later jobs.
The CUDA backend is explicit because `--torch-backend=auto` selects CPU-only
Torch when installation runs on a GPU-less login node. NumPy is pinned below
2.4 to satisfy vLLM's transitive `mistral-common` constraint.
The download helper disables Hugging Face Xet by default because that transport
stalls on this host; pass `--use-xet` only after independently verifying it.

Model definitions and immutable Hugging Face revisions live in
`qc/auto_qc_models.json`. Downloads fail if a revision is unpinned, and GPU
submission fails if any indexed weight shard is absent or empty.
The configured safetensors prefetch strategy is intentional: vLLM does not
currently classify GPFS as a network filesystem, and its default loader was
measured at 4m42s for the first 397B shard.
Both smoke configurations use one active sequence because Prima currently
classifies exams serially. This also keeps CUDA-graph capture scoped to the
actual request concurrency.
The Qwen image processor is capped at 2,097,152 pixels. This is more than four
times the area of the measured smoke montages, so those inputs are not resized,
while avoiding a synthetic 16.8-megapixel startup profile from the checkpoint
default.
The 27B entry disables FlashInfer's block-scale FP8 GEMM because its
TensorRT-LLM JIT kernel fails during CUDA-graph profiling on the cluster H200
nodes. It also uses vLLM's native sampler and Triton GDN prefill to avoid
optional FlashInfer JIT builds. vLLM selects its supported DeepGEMM backend for
the checkpoint's required FP8 linear kernels.

```bash
micromamba run -p /gpfs/data/huo-lab/Image/annawoodard/micromamba/envs/prima \
  python scripts/download_auto_qc_model.py qwen35_27b_fp8 --dry-run

micromamba run -p /gpfs/data/huo-lab/Image/annawoodard/micromamba/envs/prima \
  python scripts/download_auto_qc_model.py qwen35_27b_fp8
```

The initial decision order is deliberately small:

| Model key | Purpose | GPUs |
|---|---|---:|
| `qwen35_397b_fp8` | Native-vLLM compatibility reference; operationally rejected for slow device loading | 4 H200 |
| `qwen35_27b_fp8` | Active native-vLLM QC model | 1 H200 |

The 397B entry points to the original FP8 snapshot, not the Prima repair
wrapper. Do not use the repair runtime for new QC runs; it remains only to
reproduce the existing unreviewed baseline while human validation is pending.

This is a new experimental condition, not an exact parity test: the prior
240-exam output used the BF16-expert repair wrapper, while vLLM uses upstream
FP8 weights and different kernels and decoding constraints. Agreement is a
useful diagnostic, but only reviewed exam labels can decide accuracy.

## Smoke submission

Run the submitter from `prima-vllm`; it verifies the pinned `dsi-local-llms`,
`vllm`, and `openai` versions before allocating GPUs. Use a new run file for
each model/backend.

```bash
TMPDIR=/scratch/annawoodard/tmp/prima-vllm-runtime \
micromamba run -p /gpfs/data/huo-lab/Image/annawoodard/micromamba/envs/prima-vllm \
  python submit_auto_qc.py \
  --name qwen35_27b_vllm_vertical_line_smoke \
  --log-dir qc_redo/submitit_runs \
  --partition siweiq \
  --qos opportunistic \
  --gpuspec nvidia_h200-141gb \
  --ngpus 1 \
  --cpus-per-task 16 \
  --mem-gb 128 \
  --timeout-min 120 \
  --no-srun \
  --no-wait \
  --views qc_export/views_for_qc.parquet \
  --export-dir qc_export \
  --run-file qc_redo/auto_qc_runs/vllm_migration_smoke/qwen35_27b_fp8_vllm_zero_shot.json \
  --backend vllm \
  --model-key qwen35_27b_fp8 \
  --exam-list qc_redo/review_batches/vllm_migration_smoke/exam_ids.txt \
  --prompt-mode marker_classifier \
  --prompt-variant confidence_specificity \
  --probe-tag 'vertical line (detector artifact)' \
  --few-shot-examples 0 \
  --disable-thinking \
  --debug-dump-dir qc_redo/auto_qc_debug/vllm_migration_smoke/qwen35_27b_fp8
```

`qc_redo/` and `*.log` are ignored because debug responses can contain OCR of
visible image text. Never copy request bodies, base64 images, exam identifiers,
or raw model responses into tracked logs. vLLM binds to `127.0.0.1`, request and
access logging are disabled, and the OpenAI client ignores proxy environment
variables so image payloads stay on-node.

## View-level vertical-line validation

Exam-level montage tags cannot identify which exact L/R CC/MLO source failed
and therefore cannot drive same-slot fallback. The active validation path uses:

- a SHA-keyed individual-view manifest and fresh binary human state;
- `qc/view_qc_gallery.py`, which reports one explicit
  `reviewed / total / remaining` denominator and never exposes sampling strata
  or model suggestions;
- `qc/run_view_auto_qc.py` and `submit_view_auto_qc.py`, which write a separate
  `view_suggestions` schema and use a single-view prompt without the old
  cross-view seam cue;
- `view_candidates.parquet` plus `qc/select_qc_views.py`, which may choose only
  an explicitly passing candidate in the same laterality and projection slot.

The frozen 160-view panel contains 80 heuristic-enriched and 80 random views,
one per exam. Model inference is completed before labels are revealed, but its
run file must not be loaded during human review.

```bash
micromamba run -p /gpfs/data/huo-lab/Image/annawoodard/micromamba/envs/prima \
  python qc/view_qc_gallery.py \
  --manifest qc_redo/review_batches/vertical_line_view_review/manifest.parquet \
  --state qc_redo/review_batches/vertical_line_view_review/view_qc_state.json \
  --port 8765
```

After all labels exist, `qc/evaluate_view_auto_qc.py` computes overall and
stratum-specific confusion matrices and writes every disagreement for visual
adjudication. The continuation gate is sensitivity and specificity of at least
0.90 with no repeated missed morphology. Passing that research gate does not
authorize automatic deployment-grade exclusion.

## Validation decision

The first smoke only answers whether model startup, image transport, constrained
generation, parsing, checkpointing, and teardown work. It does not estimate QC
accuracy. Accuracy requires reviewed positives and negatives, evaluated per
view with the same frozen prompt and rendered inputs. Stop and fix the runtime on any
startup failure, invalid response, missing output row, or orphaned GPU process.
The 397B model proved native TP4 compatibility but loaded too slowly to operate
on this filesystem. Use the validated 27B configuration; do not add another
checkpoint repair path. The decision-changing run is the blinded view-level
panel, not a larger unreviewed montage scan.
