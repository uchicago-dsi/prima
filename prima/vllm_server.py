"""Prima policy for servers managed by the ``dsi-local-llms`` package."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import metadata
import json
import logging
import os
from pathlib import Path
import shutil
from typing import Any

logger = logging.getLogger(__name__)

PINNED_RUNTIME_PACKAGES = {
    "dsi-local-llms": "0.1.1",
    "openai": "2.45.0",
    "vllm": "0.24.0",
}

_RESERVED_VLLM_ARGS = {
    "--host",
    "--port",
    "--served-model-name",
    "--tensor-parallel-size",
    "-tp",
}


def validate_executable_tmpdir() -> Path:
    """Require an explicit writable, executable runtime directory."""
    raw_tmpdir = os.environ.get("TMPDIR")
    if not raw_tmpdir:
        raise RuntimeError(
            "TMPDIR is unset; vLLM/Triton requires an executable scratch directory"
        )
    tmpdir = Path(raw_tmpdir).expanduser().resolve()
    if not tmpdir.is_dir():
        raise FileNotFoundError(f"TMPDIR does not exist: {tmpdir}")
    if not os.access(tmpdir, os.W_OK | os.X_OK):
        raise RuntimeError(f"TMPDIR is not writable and searchable: {tmpdir}")
    noexec_flag = getattr(os, "ST_NOEXEC", 0)
    if noexec_flag and os.statvfs(tmpdir).f_flag & noexec_flag:
        raise RuntimeError(
            f"TMPDIR is mounted noexec: {tmpdir}; use executable cluster scratch"
        )
    return tmpdir


def validate_vllm_runtime() -> str:
    """Validate the dedicated serving environment and return its CLI path."""
    executable = shutil.which("vllm")
    if executable is None:
        raise RuntimeError(
            "vllm executable not found; run from the pinned Prima vLLM environment"
        )
    if shutil.which("nvcc") is None:
        raise RuntimeError(
            "nvcc executable not found; the pinned Prima vLLM environment must "
            "provide CUDA 13.0 for DeepGEMM JIT compilation"
        )
    _resolve_cuda_home(executable)
    for package, expected_version in PINNED_RUNTIME_PACKAGES.items():
        try:
            installed_version = metadata.version(package)
        except metadata.PackageNotFoundError as exc:
            raise RuntimeError(
                f"required vLLM runtime package is missing: {package}"
            ) from exc
        if installed_version != expected_version:
            raise RuntimeError(
                f"{package} version mismatch: expected {expected_version}, "
                f"found {installed_version}"
            )
    validate_executable_tmpdir()
    return executable


def _resolve_cuda_home(vllm_executable: str) -> Path:
    """Locate a complete CUDA toolkit inside the active serving environment."""
    prefix = Path(vllm_executable).resolve().parent.parent
    conda_target = prefix / "targets" / "x86_64-linux"
    if (
        (prefix / "bin" / "nvcc").is_file()
        and (prefix / "nvvm" / "bin" / "cicc").is_file()
        and (conda_target / "include" / "cuda_runtime.h").is_file()
    ):
        return prefix
    if (prefix / "bin" / "nvcc").is_file() and (
        prefix / "include" / "cuda_runtime.h"
    ).is_file():
        return prefix
    raise RuntimeError(
        f"active vLLM environment has no complete CUDA toolkit under {prefix}"
    )


@dataclass(frozen=True)
class VLLMModelSpec:
    """Validated model-serving configuration from the model registry."""

    key: str
    repo_id: str
    directory_name: str
    served_model_name: str
    tensor_parallel_size: int
    extra_args: tuple[str, ...]
    environment: tuple[tuple[str, str], ...]
    revision: str | None = None

    @classmethod
    def from_payload(cls, key: str, payload: Any) -> "VLLMModelSpec":
        if not isinstance(payload, dict):
            raise ValueError(f"model entry {key!r} must be a JSON object")

        def required_string(field: str) -> str:
            value = payload.get(field)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"model entry {key!r} requires a non-empty {field!r}")
            return value.strip()

        repo_id = required_string("repo_id")
        directory_name = str(payload.get("directory_name", key)).strip()
        if not directory_name or Path(directory_name).name != directory_name:
            raise ValueError(
                f"model entry {key!r} directory_name must be one path component"
            )
        served_model_name = required_string("served_model_name")
        tensor_parallel_size = payload.get("tensor_parallel_size", 1)
        if not isinstance(tensor_parallel_size, int) or tensor_parallel_size <= 0:
            raise ValueError(
                f"model entry {key!r} tensor_parallel_size must be positive"
            )
        raw_extra_args = payload.get("extra_args", [])
        if not isinstance(raw_extra_args, list) or not all(
            isinstance(arg, str) and arg for arg in raw_extra_args
        ):
            raise ValueError(
                f"model entry {key!r} extra_args must be a list of strings"
            )
        normalized_arg_names = {arg.split("=", 1)[0] for arg in raw_extra_args}
        collisions = _RESERVED_VLLM_ARGS.intersection(normalized_arg_names)
        if collisions:
            raise ValueError(
                f"model entry {key!r} extra_args overrides reserved arguments: "
                f"{sorted(collisions)}"
            )
        raw_environment = payload.get("environment", {})
        if not isinstance(raw_environment, dict) or not all(
            isinstance(name, str) and name and isinstance(value, str) and value
            for name, value in raw_environment.items()
        ):
            raise ValueError(
                f"model entry {key!r} environment must map strings to strings"
            )
        raw_revision = payload.get("revision")
        revision = None
        if raw_revision is not None:
            if not isinstance(raw_revision, str) or not raw_revision.strip():
                raise ValueError(
                    f"model entry {key!r} revision must be a non-empty string"
                )
            revision = raw_revision.strip()
        return cls(
            key=key,
            repo_id=repo_id,
            directory_name=directory_name,
            served_model_name=served_model_name,
            tensor_parallel_size=tensor_parallel_size,
            extra_args=tuple(raw_extra_args),
            environment=tuple(sorted(raw_environment.items())),
            revision=revision,
        )


def load_model_registry(path: Path) -> dict[str, VLLMModelSpec]:
    """Load and validate a flat JSON registry keyed by short model name."""
    if not path.is_file():
        raise FileNotFoundError(f"vLLM model registry not found: {path}")
    with open(path) as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict) or not payload:
        raise ValueError(f"vLLM model registry must be a non-empty object: {path}")
    return {
        str(key): VLLMModelSpec.from_payload(str(key), entry)
        for key, entry in payload.items()
    }


def select_model_spec(path: Path, model_key: str) -> VLLMModelSpec:
    """Select one configured model and fail with the available choices."""
    registry = load_model_registry(path)
    try:
        return registry[model_key]
    except KeyError as exc:
        choices = ", ".join(sorted(registry))
        raise ValueError(
            f"unknown vLLM model key {model_key!r}; choose one of: {choices}"
        ) from exc


def resolve_model_path(spec: VLLMModelSpec, models_dir: Path) -> Path:
    """Resolve local weights without silently falling back to the network."""
    model_path = (models_dir / spec.directory_name).resolve()
    if not model_path.is_dir():
        raise FileNotFoundError(
            f"local weights for {spec.key!r} not found at {model_path}; "
            "download the configured Hugging Face snapshot before submitting GPU work"
        )
    config_path = model_path / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(
            f"local model directory is incomplete (missing config.json): {model_path}"
        )
    _validate_model_revision(spec, model_path)
    _validate_model_weights(model_path)
    return model_path


def _validate_model_revision(spec: VLLMModelSpec, model_path: Path) -> None:
    if spec.revision is None:
        raise ValueError(f"model {spec.key!r} has no pinned revision")
    metadata_path = (
        model_path / ".cache" / "huggingface" / "download" / "config.json.metadata"
    )
    provenance_path = model_path / "prima_snapshot.json"
    resolved_revision: str | None = None
    if metadata_path.is_file():
        metadata_lines = metadata_path.read_text(errors="replace").splitlines()
        if metadata_lines:
            resolved_revision = metadata_lines[0].strip()
    elif provenance_path.is_file():
        with open(provenance_path) as handle:
            provenance = json.load(handle)
        raw_revision = (
            provenance.get("revision") if isinstance(provenance, dict) else None
        )
        if isinstance(raw_revision, str):
            resolved_revision = raw_revision.strip()
    if resolved_revision is None:
        raise RuntimeError(
            f"cannot verify pinned revision for local model snapshot: {model_path}"
        )
    if resolved_revision != spec.revision:
        raise RuntimeError(
            f"local model revision mismatch for {spec.key!r}: "
            f"expected {spec.revision}, found {resolved_revision}"
        )


def _validate_model_weights(model_path: Path) -> None:
    index_path = model_path / "model.safetensors.index.json"
    if index_path.is_file():
        with open(index_path) as handle:
            index_payload = json.load(handle)
        weight_map = (
            index_payload.get("weight_map") if isinstance(index_payload, dict) else None
        )
        if not isinstance(weight_map, dict) or not weight_map:
            raise ValueError(f"invalid safetensors weight index: {index_path}")
        filenames = sorted({str(filename) for filename in weight_map.values()})
    else:
        filenames = ["model.safetensors"]

    missing_or_empty = [
        filename
        for filename in filenames
        if not (model_path / filename).is_file()
        or (model_path / filename).stat().st_size <= 0
    ]
    if missing_or_empty:
        preview = ", ".join(missing_or_empty[:5])
        raise FileNotFoundError(
            f"local model snapshot is incomplete at {model_path}: "
            f"{len(missing_or_empty)} missing or empty weight files; first: {preview}"
        )


def find_available_loopback_port() -> int:
    """Delegate ephemeral port selection to ``dsi-local-llms``."""
    try:
        from local_llms import find_available_loopback_port as find_port
    except ImportError as exc:
        raise RuntimeError(
            "dsi-local-llms is missing; install the pinned serving dependency"
        ) from exc
    return find_port()


class ManagedVLLMServer:
    """Apply Prima runtime policy around the upstream managed server."""

    def __init__(
        self,
        *,
        spec: VLLMModelSpec,
        model_path: Path,
        port: int,
        log_path: Path,
        startup_timeout_seconds: int,
    ) -> None:
        if startup_timeout_seconds <= 0:
            raise ValueError("vLLM startup timeout must be positive")
        self.spec = spec
        self.model_path = model_path
        self.port = port
        self.log_path = log_path
        self.startup_timeout_seconds = startup_timeout_seconds
        self._server: Any = None

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}/v1"

    def start(self) -> None:
        if self._server is not None:
            raise RuntimeError("vLLM server is already started")
        executable = validate_vllm_runtime()
        try:
            from local_llms import (
                ManagedVLLMServer as LocalLLMServer,
                VLLMServerConfig,
            )
        except ImportError as exc:
            raise RuntimeError(
                "dsi-local-llms is missing; install the pinned serving dependency"
            ) from exc

        logger.info(
            "starting vLLM model=%s tp=%d log=%s",
            self.spec.key,
            self.spec.tensor_parallel_size,
            self.log_path,
        )
        environment: dict[str, str] = {}
        if "VLLM_LOGGING_LEVEL" not in os.environ:
            environment["VLLM_LOGGING_LEVEL"] = "INFO"
        runtime_prefix = Path(executable).resolve().parent.parent
        environment["CUDA_HOME"] = str(_resolve_cuda_home(executable))
        environment["PATH"] = (
            f"{runtime_prefix / 'nvvm' / 'bin'}{os.pathsep}{os.environ.get('PATH', '')}"
        )
        environment["FLASHINFER_WORKSPACE_BASE"] = str(validate_executable_tmpdir())
        environment.update(self.spec.environment)
        config = VLLMServerConfig(
            model_path=self.model_path,
            served_model_name=self.spec.served_model_name,
            port=self.port,
            executable=executable,
            tensor_parallel_size=self.spec.tensor_parallel_size,
            extra_args=(
                "--no-enable-log-requests",
                "--disable-uvicorn-access-log",
                *self.spec.extra_args,
            ),
            environment=tuple(sorted(environment.items())),
            startup_timeout_seconds=self.startup_timeout_seconds,
        )
        server = LocalLLMServer(config, log_path=self.log_path)
        try:
            server.start()
        except Exception:
            server.stop()
            raise
        self._server = server

    def stop(self) -> None:
        server = self._server
        self._server = None
        if server is not None:
            server.stop()

    def __enter__(self) -> "ManagedVLLMServer":
        self.start()
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        del exc_type, exc, traceback
        self.stop()
