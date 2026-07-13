from __future__ import annotations

import base64
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import auto_annotate_qc
import prima.vllm_server as vllm_server
from prima.auto_qc import AUTO_QC_PROMPT_VERSION, normalize_auto_run
from prima.vllm_server import (
    VLLMModelSpec,
    load_model_registry,
    resolve_model_path,
    validate_executable_tmpdir,
)


def model_payload() -> dict[str, object]:
    return {
        "repo_id": "example/model",
        "revision": "abc123",
        "directory_name": "model",
        "served_model_name": "example_model",
        "tensor_parallel_size": 2,
        "extra_args": ["--max-model-len", "4096"],
    }


def test_registry_and_local_path_are_strict(tmp_path: Path) -> None:
    registry_path = tmp_path / "models.json"
    registry_path.write_text(json.dumps({"example": model_payload()}))
    spec = load_model_registry(registry_path)["example"]
    assert spec.revision == "abc123"

    with pytest.raises(FileNotFoundError, match="local weights"):
        resolve_model_path(spec, tmp_path)

    model_dir = tmp_path / "model"
    model_dir.mkdir()
    with pytest.raises(FileNotFoundError, match="config.json"):
        resolve_model_path(spec, tmp_path)
    (model_dir / "config.json").write_text("{}")
    metadata_dir = model_dir / ".cache" / "huggingface" / "download"
    metadata_dir.mkdir(parents=True)
    (metadata_dir / "config.json.metadata").write_text("abc123\netag\n")
    (model_dir / "model.safetensors").write_bytes(b"weights")
    assert resolve_model_path(spec, tmp_path) == model_dir.resolve()


def test_partial_sharded_snapshot_is_rejected(tmp_path: Path) -> None:
    spec = VLLMModelSpec.from_payload("example", model_payload())
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text("{}")
    metadata_dir = model_dir / ".cache" / "huggingface" / "download"
    metadata_dir.mkdir(parents=True)
    (metadata_dir / "config.json.metadata").write_text("abc123\netag\n")
    (model_dir / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "weight_map": {
                    "layer.0": "model-00001-of-00002.safetensors",
                    "layer.1": "model-00002-of-00002.safetensors",
                }
            }
        )
    )
    (model_dir / "model-00001-of-00002.safetensors").write_bytes(b"weights")
    with pytest.raises(FileNotFoundError, match="incomplete"):
        resolve_model_path(spec, tmp_path)


def test_registry_rejects_launch_argument_override() -> None:
    payload = model_payload()
    payload["extra_args"] = ["--host", "0.0.0.0"]
    with pytest.raises(ValueError, match="reserved"):
        VLLMModelSpec.from_payload("example", payload)


def test_registry_validates_model_environment() -> None:
    payload = model_payload()
    payload["environment"] = {"VLLM_FEATURE": "0"}
    spec = VLLMModelSpec.from_payload("example", payload)
    assert spec.environment == (("VLLM_FEATURE", "0"),)

    payload["environment"] = {"VLLM_FEATURE": 0}
    with pytest.raises(ValueError, match="environment"):
        VLLMModelSpec.from_payload("example", payload)


def test_runtime_tmpdir_must_be_explicit_and_executable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("TMPDIR", raising=False)
    with pytest.raises(RuntimeError, match="TMPDIR is unset"):
        validate_executable_tmpdir()

    monkeypatch.setenv("TMPDIR", str(tmp_path))
    monkeypatch.setattr(vllm_server.os, "ST_NOEXEC", 8, raising=False)
    monkeypatch.setattr(
        vllm_server.os,
        "statvfs",
        lambda _path: SimpleNamespace(f_flag=8),
    )
    with pytest.raises(RuntimeError, match="mounted noexec"):
        validate_executable_tmpdir()

    monkeypatch.setattr(
        vllm_server.os,
        "statvfs",
        lambda _path: SimpleNamespace(f_flag=0),
    )
    assert validate_executable_tmpdir() == tmp_path.resolve()


def test_runtime_requires_nvcc(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        vllm_server.shutil,
        "which",
        lambda command: "/env/bin/vllm" if command == "vllm" else None,
    )
    with pytest.raises(RuntimeError, match="nvcc executable not found"):
        vllm_server.validate_vllm_runtime()


def test_cuda_home_requires_compiler_and_headers(tmp_path: Path) -> None:
    prefix = tmp_path / "env"
    executable = prefix / "bin" / "vllm"
    executable.parent.mkdir(parents=True)
    executable.touch()
    with pytest.raises(RuntimeError, match="complete CUDA toolkit"):
        vllm_server._resolve_cuda_home(str(executable))

    target = prefix / "targets" / "x86_64-linux"
    (target / "include").mkdir(parents=True)
    (target / "include" / "cuda_runtime.h").touch()
    (prefix / "bin" / "nvcc").touch()
    (prefix / "nvvm" / "bin").mkdir(parents=True)
    (prefix / "nvvm" / "bin" / "cicc").touch()
    assert vllm_server._resolve_cuda_home(str(executable)) == prefix


def test_prima_server_delegates_portable_lifecycle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    local_llms = pytest.importorskip("local_llms")
    captured: dict[str, object] = {}

    class FakeManagedServer:
        def __init__(self, config: object, *, log_path: Path) -> None:
            captured["config"] = config
            captured["log_path"] = log_path

        def start(self) -> None:
            captured["started"] = True

        def stop(self) -> None:
            captured["stopped"] = True

    monkeypatch.setattr(local_llms, "ManagedVLLMServer", FakeManagedServer)
    monkeypatch.setattr(
        vllm_server,
        "validate_vllm_runtime",
        lambda: "/runtime/bin/vllm",
    )
    monkeypatch.setattr(
        vllm_server,
        "_resolve_cuda_home",
        lambda _executable: Path("/runtime"),
    )
    monkeypatch.setattr(
        vllm_server,
        "validate_executable_tmpdir",
        lambda: tmp_path,
    )
    monkeypatch.setenv("LIBRARY_PATH", "/existing/link-libraries")
    monkeypatch.setenv("LD_LIBRARY_PATH", "/existing/runtime-libraries")
    payload = model_payload()
    payload["environment"] = {"MODEL_POLICY": "enabled"}
    spec = VLLMModelSpec.from_payload("example", payload)
    server = vllm_server.ManagedVLLMServer(
        spec=spec,
        model_path=Path("/models/example"),
        port=8123,
        log_path=tmp_path / "server.log",
        startup_timeout_seconds=30,
    )

    server.start()
    config = captured["config"]
    assert isinstance(config, local_llms.VLLMServerConfig)
    assert config.host == "127.0.0.1"
    assert config.tensor_parallel_size == 2
    assert config.startup_timeout_seconds == 30
    assert "--no-enable-log-requests" in config.extra_args
    assert "--disable-uvicorn-access-log" in config.extra_args
    environment = dict(config.environment)
    assert environment["CUDA_HOME"] == "/runtime"
    assert environment["LIBRARY_PATH"] == ("/runtime/lib:/existing/link-libraries")
    assert environment["LD_LIBRARY_PATH"] == (
        "/runtime/lib:/existing/runtime-libraries"
    )
    assert environment["FLASHINFER_WORKSPACE_BASE"] == str(tmp_path)
    assert environment["MODEL_POLICY"] == "enabled"
    assert captured["started"] is True

    server.stop()
    assert captured["stopped"] is True


def test_image_data_url_and_tagger_schema(tmp_path: Path) -> None:
    image_path = tmp_path / "montage.png"
    image_bytes = b"not-a-real-png-but-valid-for-encoding"
    image_path.write_bytes(image_bytes)
    data_url = auto_annotate_qc.image_data_url(image_path)
    assert data_url.startswith("data:image/png;base64,")
    assert base64.b64decode(data_url.split(",", 1)[1]) == image_bytes

    response_format = auto_annotate_qc.tagger_json_response_format(
        ["vertical line (detector artifact)"]
    )
    tag_schema = response_format["json_schema"]["schema"]["properties"]["suggestions"][
        "items"
    ]["properties"]["tag"]
    assert tag_schema["enum"] == ["vertical line (detector artifact)"]


def test_vllm_classifier_parse_fails_closed() -> None:
    annotator = auto_annotate_qc.VLLMVisionAnnotator.__new__(
        auto_annotate_qc.VLLMVisionAnnotator
    )
    annotator.prompt_mode = "marker_classifier"
    annotator.probe_tag = "vertical line (detector artifact)"
    payload, suggestions = annotator._parse_response(
        response_text=(
            "EVIDENCE: narrow gray seam\nANSWER: YES\nCONFIDENCE: high\nREVIEW: NO"
        ),
        tag_catalog=["vertical line (detector artifact)"],
    )
    assert payload["present"] is True
    assert suggestions[0]["tag"] == "vertical line (detector artifact)"
    assert suggestions[0]["confidence"] == "high"

    with pytest.raises(ValueError, match="not parseable"):
        annotator._parse_response(
            response_text="uncertain",
            tag_catalog=["vertical line (detector artifact)"],
        )


def test_cli_defaults_to_vllm() -> None:
    parser = auto_annotate_qc.build_arg_parser()
    args = parser.parse_args(
        [
            "--views",
            "views.parquet",
            "--export-dir",
            "qc_export",
            "--run-file",
            "run.json",
        ]
    )
    assert args.backend == "vllm"
    assert args.model_key == "qwen35_397b_fp8"
    assert args.model_path is None
    assert args.vllm_port == 0


def test_system_prompt_matches_output_mode() -> None:
    assert "JSON object" in auto_annotate_qc.build_system_prompt("tagger_json")
    marker_prompt = auto_annotate_qc.build_system_prompt("marker_classifier")
    assert "labeled-line" in marker_prompt
    assert "JSON object" not in marker_prompt
    assert "only yes or no" in auto_annotate_qc.build_system_prompt("binary_tag_probe")


def test_few_shot_payload_matches_target_mode() -> None:
    tag = "vertical line (detector artifact)"
    marker = auto_annotate_qc.build_few_shot_assistant_payload(
        [tag], prompt_mode="marker_classifier", probe_tag=tag
    )
    assert "ANSWER: YES" in marker
    assert not marker.startswith("{")
    binary = auto_annotate_qc.build_few_shot_assistant_payload(
        [], prompt_mode="binary_tag_probe", probe_tag=tag
    )
    assert binary == "no"


def test_prompt_version_and_inference_settings_prevent_mixed_resume(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert AUTO_QC_PROMPT_VERSION == "qc_multimodal_v2"
    args = SimpleNamespace(
        backend="transformers",
        disable_thinking=True,
        few_shot_examples=0,
        few_shot_qc_file=None,
        max_new_tokens=128,
        prompt_mode="marker_classifier",
        prompt_variant="confidence_specificity",
        probe_tag="vertical line (detector artifact)",
        qc_file=None,
        target_prompt_override=None,
        text_only_prompt=None,
    )
    current = auto_annotate_qc.build_run_payload(
        run_id="run",
        model_label="model",
        args=args,
        tag_catalog=[args.probe_tag],
    )
    normalized = normalize_auto_run(current)
    assert normalized["inference_settings"]["max_new_tokens"] == 128

    existing = copy.deepcopy(current)
    existing["exam_suggestions"] = {"private-id": {}}
    changed = copy.deepcopy(current)
    changed["inference_settings"]["max_new_tokens"] = 256
    with pytest.raises(ValueError, match="inference_settings differ"):
        auto_annotate_qc.validate_auto_run_compatible(
            existing=existing,
            current=changed,
            run_file=Path("run.json"),
        )

    vllm_args = SimpleNamespace(
        **{
            **vars(args),
            "backend": "vllm",
            "vllm_request_timeout_seconds": 600,
        }
    )
    model_payload_with_env = model_payload()
    model_payload_with_env["environment"] = {"VLLM_FEATURE": "0"}
    model_spec = VLLMModelSpec.from_payload("example", model_payload_with_env)
    monkeypatch.setattr(
        auto_annotate_qc.metadata,
        "version",
        lambda package: {
            "dsi-local-llms": "0.1.1",
            "openai": "2.45.0",
            "vllm": "0.24.0",
        }[package],
    )
    vllm_settings = auto_annotate_qc.build_inference_settings(
        args=vllm_args,
        model_spec=model_spec,
    )
    assert vllm_settings["serve_environment"] == {"VLLM_FEATURE": "0"}
    assert vllm_settings["runtime_versions"]["dsi-local-llms"] == "0.1.1"


def test_vllm_client_error_is_sanitized() -> None:
    class LeakyError(Exception):
        status_code = 400

    leaked = "data:image/png;base64,private-payload"
    error = auto_annotate_qc.sanitized_vllm_request_error(LeakyError(leaked))
    assert leaked not in str(error)
    assert "status=400" in str(error)
