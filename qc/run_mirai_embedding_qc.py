#!/usr/bin/env python3
"""Register, extract, and fit a frozen Mirai-embedding view-QC classifier."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd

from prima.dicom_source import (
    SOURCE_ARCHIVE_COLUMN,
    DicomSource,
    materialize_dicom_sources,
    require_source_columns,
    require_valid_sources,
    validate_materialized_source,
)
from prima.view_auto_qc import new_view_auto_run, save_view_auto_run
from prima.view_qc import (
    VIEW_LABEL_PRESENT,
    load_view_qc_state,
    normalize_view_id,
    summarize_view_qc_state,
    validate_view_manifest_columns,
)

PROTOCOL_SCHEMA_VERSION = 1
MODEL_VARIANT = "frozen_mirai_embedding_logistic_v1"
DEFAULT_C_GRID = (0.0001, 0.001, 0.01, 0.1, 1.0, 10.0)
DEFAULT_SEED = 20260713
DEFAULT_FOLDS = 5
DEFAULT_MINIMUM_OOF_SENSITIVITY = 0.95


def sha256_file(path: Path) -> str:
    """Return a streaming SHA-256 for one file."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _required_file(path: Path, description: str) -> Path:
    resolved = Path(path).resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"{description} not found: {resolved}")
    return resolved


def _write_restricted_json(path: Path, payload: Mapping[str, Any]) -> None:
    path = Path(path).resolve()
    if path.exists():
        raise FileExistsError(f"refusing to overwrite output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        os.chmod(temporary, 0o600)
        os.replace(temporary, path)
        os.chmod(path, 0o600)
    finally:
        if temporary.exists():
            temporary.unlink()


def _manifest(path: Path) -> pd.DataFrame:
    table = pd.read_parquet(path).copy()
    validate_view_manifest_columns(table.columns, str(path))
    table["view_id"] = table["view_id"].map(normalize_view_id)
    if table["view_id"].duplicated().any():
        raise ValueError(f"manifest contains duplicate view IDs: {path}")
    return table.sort_values("review_order", kind="stable").reset_index(drop=True)


def _source_manifest(path: Path) -> pd.DataFrame:
    table = pd.read_parquet(path).copy()
    require_source_columns(table.columns, str(path))
    if "view_id" not in table:
        raise ValueError(f"source manifest is missing view_id: {path}")
    table["view_id"] = table["view_id"].map(normalize_view_id)
    if table["view_id"].duplicated().any():
        raise ValueError(f"source manifest contains duplicate view IDs: {path}")
    require_valid_sources(table.to_dict("records"), str(path))
    return table


def require_matching_views(
    manifest: pd.DataFrame, source: pd.DataFrame, *, context: str
) -> None:
    if set(manifest["view_id"]) != set(source["view_id"]):
        raise ValueError(
            f"{context} manifest and source manifest cover different views"
        )


def require_patient_exam_disjoint(
    development_source: pd.DataFrame, audit_source: pd.DataFrame
) -> dict[str, int]:
    """Fail if the frozen development and audit panels share a patient or exam."""
    required = {"patient_id", "exam_id"}
    for name, table in (
        ("development", development_source),
        ("audit", audit_source),
    ):
        missing = sorted(required - set(table.columns))
        if missing:
            raise ValueError(
                f"{name} source manifest is missing columns: {', '.join(missing)}"
            )
        if table[list(required)].isna().any().any():
            raise ValueError(f"{name} source manifest contains null identifiers")
    development_patients = set(development_source["patient_id"].astype(str))
    audit_patients = set(audit_source["patient_id"].astype(str))
    development_exams = set(development_source["exam_id"].astype(str))
    audit_exams = set(audit_source["exam_id"].astype(str))
    if development_patients & audit_patients:
        raise ValueError("development and audit panels share patients")
    if development_exams & audit_exams:
        raise ValueError("development and audit panels share exams")
    return {
        "development_patients": len(development_patients),
        "development_exams": len(development_exams),
        "audit_patients": len(audit_patients),
        "audit_exams": len(audit_exams),
    }


def register_protocol(args: argparse.Namespace) -> int:
    paths = {
        "development_manifest": _required_file(
            args.development_manifest, "development manifest"
        ),
        "development_source_manifest": _required_file(
            args.development_source_manifest, "development source manifest"
        ),
        "development_state": _required_file(
            args.development_state, "development state"
        ),
        "audit_manifest": _required_file(args.audit_manifest, "audit manifest"),
        "audit_source_manifest": _required_file(
            args.audit_source_manifest, "audit source manifest"
        ),
        "audit_protocol": _required_file(args.audit_protocol, "audit protocol"),
        "encoder_config": _required_file(args.encoder_config, "encoder config"),
        "encoder_snapshot": _required_file(args.encoder_snapshot, "encoder snapshot"),
    }
    development_manifest = _manifest(paths["development_manifest"])
    audit_manifest = _manifest(paths["audit_manifest"])
    development_source = _source_manifest(paths["development_source_manifest"])
    audit_source = _source_manifest(paths["audit_source_manifest"])
    require_matching_views(
        development_manifest, development_source, context="development"
    )
    require_matching_views(audit_manifest, audit_source, context="audit")
    disjoint_counts = require_patient_exam_disjoint(development_source, audit_source)

    state = load_view_qc_state(paths["development_state"])
    progress = summarize_view_qc_state(state, development_manifest["view_id"])
    if progress["remaining"]:
        raise RuntimeError("development reference labels are incomplete")
    if progress["low_confidence"]:
        raise RuntimeError("development reference contains low-confidence labels")
    label_counts = {
        "target_present": sum(
            record["label"] == VIEW_LABEL_PRESENT for record in state["labels"].values()
        ),
        "target_absent": sum(
            record["label"] != VIEW_LABEL_PRESENT for record in state["labels"].values()
        ),
    }
    audit_protocol = json.loads(paths["audit_protocol"].read_text())
    if "gate" not in audit_protocol:
        raise ValueError("audit protocol is missing its gate")

    protocol = {
        "schema_version": PROTOCOL_SCHEMA_VERSION,
        "registered_at": utc_now_iso(),
        "experiment_status": "post_failure_rescue_not_independent_confirmation",
        "purpose": (
            "Test whether a regularized linear head on the frozen open-source "
            "Mirai image encoder can rescue whole-exam input QC."
        ),
        "target": state["target"],
        "development": {
            "manifest": str(paths["development_manifest"]),
            "manifest_sha256": sha256_file(paths["development_manifest"]),
            "source_manifest": str(paths["development_source_manifest"]),
            "source_manifest_sha256": sha256_file(paths["development_source_manifest"]),
            "state": str(paths["development_state"]),
            "state_sha256": sha256_file(paths["development_state"]),
            "views": len(development_manifest),
            "label_counts": label_counts,
            "reference_role": (
                "weak nonexpert development supervision; not clinical ground truth"
            ),
        },
        "audit": {
            "manifest": str(paths["audit_manifest"]),
            "manifest_sha256": sha256_file(paths["audit_manifest"]),
            "source_manifest": str(paths["audit_source_manifest"]),
            "source_manifest_sha256": sha256_file(paths["audit_source_manifest"]),
            "protocol": str(paths["audit_protocol"]),
            "protocol_sha256": sha256_file(paths["audit_protocol"]),
            "views": len(audit_manifest),
            "reference_access": "withheld from fitting and threshold selection",
            "gate": audit_protocol["gate"],
        },
        "disjointness": {
            "rule": "no shared patient_id or exam_id",
            **disjoint_counts,
        },
        "encoder": {
            "name": "Mirai v0.8 open-source image encoder",
            "config": str(paths["encoder_config"]),
            "config_sha256": sha256_file(paths["encoder_config"]),
            "snapshot": str(paths["encoder_snapshot"]),
            "snapshot_sha256": sha256_file(paths["encoder_snapshot"]),
            "representation": "512-dimensional pooled image hidden",
            "preprocessing": (
                "original DICOM -> Mirai pydicom minmax window -> Mirai frozen "
                "test transforms (scale_2d, align_to_left, force_num_chan_2d, "
                "normalize_2d)"
            ),
        },
        "classifier": {
            "variant": MODEL_VARIANT,
            "features": "frozen Mirai pooled image hidden only",
            "pipeline": "StandardScaler then L2 logistic regression",
            "solver": "liblinear",
            "class_weight": "balanced",
            "c_grid": list(DEFAULT_C_GRID),
            "c_selection": ("maximum mean fold ROC AUC; ties choose the smallest C"),
            "cv": {
                "kind": "StratifiedKFold",
                "folds": DEFAULT_FOLDS,
                "shuffle": True,
                "seed": DEFAULT_SEED,
            },
            "threshold_selection": {
                "data": "pooled out-of-fold development predictions only",
                "rule": (
                    "largest inclusive score threshold with sensitivity at least "
                    f"{DEFAULT_MINIMUM_OOF_SENSITIVITY:.2f}"
                ),
                "minimum_sensitivity": DEFAULT_MINIMUM_OOF_SENSITIVITY,
            },
        },
        "frozen_arms": {
            "primary": (
                "deterministic DICOM OR frozen modular visual OR embedding classifier"
            ),
            "secondary_mechanism_check": (
                "deterministic DICOM OR embedding classifier"
            ),
        },
        "interpretation": (
            "A passing audit is hypothesis-generating because the embedding model "
            "class was selected after observing failure of the modular system; one "
            "new patient-disjoint whole-exam panel is required for confirmation."
        ),
    }
    _write_restricted_json(args.output, protocol)
    print(
        "Mirai embedding protocol frozen: "
        f"development_views={len(development_manifest)} "
        f"audit_views={len(audit_manifest)}"
    )
    return 0


def load_protocol(path: Path) -> tuple[Path, dict[str, Any]]:
    resolved = _required_file(path, "embedding protocol")
    payload = json.loads(resolved.read_text())
    if payload.get("schema_version") != PROTOCOL_SCHEMA_VERSION:
        raise ValueError("unsupported Mirai embedding protocol schema")
    if payload.get("classifier", {}).get("variant") != MODEL_VARIANT:
        raise ValueError("embedding protocol classifier variant is unsupported")
    return resolved, payload


def require_protocol_file(
    protocol: Mapping[str, Any], section: str, key: str, actual_path: Path
) -> None:
    expected_path = Path(protocol[section][key]).resolve()
    actual_path = Path(actual_path).resolve()
    if actual_path != expected_path:
        raise ValueError(
            f"{section} {key} path differs from the frozen embedding protocol"
        )
    expected_hash = protocol[section][f"{key}_sha256"]
    if sha256_file(actual_path) != expected_hash:
        raise ValueError(
            f"{section} {key} hash differs from the frozen embedding protocol"
        )


def _load_mirai_encoder(
    *, mirai_repo: Path, config_path: Path, snapshot_path: Path, device_name: str
) -> tuple[Any, Any, Any, int]:
    """Load the official Mirai encoder and test transform lazily."""
    import torch

    repo = Path(mirai_repo).resolve()
    if not (repo / "onconet").is_dir():
        raise FileNotFoundError(f"Mirai repository is invalid: {repo}")
    sys.path.insert(0, str(repo))
    import onconet.models.custom_resnet  # noqa: F401
    import onconet.transformers.image  # noqa: F401
    import onconet.transformers.tensor  # noqa: F401
    from onconet.models.factory import load_model
    from onconet.transformers.basic import ComposeTrans
    from onconet.transformers.factory import get_transformers
    from onconet.utils import parsing

    config = json.loads(Path(config_path).read_text())
    transform_args = argparse.Namespace(**config)
    image_transformers = parsing.parse_transformers(
        transform_args.test_image_transformers
    )
    tensor_transformers = parsing.parse_transformers(
        transform_args.test_tensor_transformers
    )
    transform = ComposeTrans(
        get_transformers(image_transformers, tensor_transformers, transform_args)
    )

    load_args = argparse.Namespace(**config)
    load_args.img_encoder_snapshot = str(snapshot_path)
    load_args.use_precomputed_hiddens = False
    load_args.use_pred_risk_factors_if_unk = True
    load_args.use_spatial_transformer = False
    encoder = load_model(str(snapshot_path), load_args, do_wrap_model=False)
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA embedding extraction requested but CUDA is unavailable"
        )
    device = torch.device(device_name)
    encoder._model.args.cuda = device.type == "cuda"
    encoder._model.args.model_parallel = False
    encoder.eval().to(device)
    dimension = int(encoder._model.args.img_only_dim)
    return torch, encoder, transform, dimension


def _run_embedding_batch(
    *, torch: Any, encoder: Any, tensors: list[Any], device: Any
) -> np.ndarray:
    batch = torch.stack(tensors).to(device)
    encoder_args = encoder._model.args
    risk_factors = [
        torch.zeros(
            len(tensors),
            int(encoder_args.risk_factor_key_to_num_class[key]),
            device=device,
        )
        for key in encoder_args.risk_factor_keys
    ]
    with torch.inference_mode():
        _, hidden, _ = encoder(batch, risk_factors, None)
    image_hidden = hidden[:, : int(encoder_args.img_only_dim)]
    return image_hidden.detach().cpu().numpy().astype(np.float32, copy=False)


def extract_embeddings(args: argparse.Namespace) -> int:
    protocol_path, protocol = load_protocol(args.protocol)
    split = args.split
    manifest_path = _required_file(args.manifest, f"{split} manifest")
    source_path = _required_file(args.source_manifest, f"{split} source manifest")
    require_protocol_file(protocol, split, "manifest", manifest_path)
    require_protocol_file(protocol, split, "source_manifest", source_path)
    config_path = _required_file(args.encoder_config, "encoder config")
    snapshot_path = _required_file(args.encoder_snapshot, "encoder snapshot")
    require_protocol_file(protocol, "encoder", "config", config_path)
    require_protocol_file(protocol, "encoder", "snapshot", snapshot_path)
    raw_root = Path(args.raw_root).resolve()
    temp_root = Path(args.temp_root).resolve()
    if not raw_root.is_dir():
        raise FileNotFoundError(f"raw DICOM root not found: {raw_root}")
    if not temp_root.is_dir():
        raise FileNotFoundError(f"temporary extraction root not found: {temp_root}")
    output_path = Path(args.output).resolve()
    metadata_path = output_path.with_suffix(".metadata.json")
    if output_path.exists() or metadata_path.exists():
        raise FileExistsError("refusing to overwrite embedding outputs")

    manifest = _manifest(manifest_path)
    source = _source_manifest(source_path)
    require_matching_views(manifest, source, context=split)
    source = manifest[["view_id", "review_order"]].merge(
        source, on="view_id", how="left", validate="one_to_one"
    )
    torch, encoder, transform, dimension = _load_mirai_encoder(
        mirai_repo=args.mirai_repo,
        config_path=config_path,
        snapshot_path=snapshot_path,
        device_name=args.device,
    )
    device = next(encoder.parameters()).device

    pending_ids: list[str] = []
    pending_tensors: list[Any] = []
    output_ids: list[str] = []
    output_features: list[np.ndarray] = []

    def flush() -> None:
        if not pending_tensors:
            return
        vectors = _run_embedding_batch(
            torch=torch,
            encoder=encoder,
            tensors=pending_tensors,
            device=device,
        )
        if vectors.shape != (len(pending_ids), dimension):
            raise RuntimeError("Mirai encoder returned an unexpected embedding shape")
        output_ids.extend(pending_ids)
        output_features.extend(vectors)
        pending_ids.clear()
        pending_tensors.clear()

    import pydicom
    from onconet.utils.dicom import dicom_to_arr

    for _, archive_rows in source.groupby(SOURCE_ARCHIVE_COLUMN, sort=False):
        sources = [DicomSource.from_row(row) for row in archive_rows.to_dict("records")]
        with materialize_dicom_sources(
            sources, raw_root=raw_root, temp_root=temp_root
        ) as materialized:
            for (_, row), dicom_source in zip(archive_rows.iterrows(), sources):
                path = materialized[dicom_source.archive_member.as_posix()]
                dataset = pydicom.dcmread(str(path), force=True)
                validate_materialized_source(
                    dicom_source, path, dataset, verify_sha256=True
                )
                image = dicom_to_arr(
                    dataset, window_method="minmax", pillow=True, overlay=False
                )
                pending_ids.append(str(row["view_id"]))
                pending_tensors.append(transform(image))
                if len(pending_tensors) >= args.batch_size:
                    flush()
    flush()
    if set(output_ids) != set(manifest["view_id"]) or len(output_ids) != len(manifest):
        raise RuntimeError("embedding extraction did not cover the frozen manifest")
    matrix = np.stack(output_features)
    features = pd.DataFrame(
        matrix,
        columns=[f"f{index:04d}" for index in range(matrix.shape[1])],
    )
    features.insert(0, "view_id", output_ids)
    features = (
        manifest[["view_id", "review_order"]]
        .merge(features, on="view_id", how="left", validate="one_to_one")
        .drop(columns="review_order")
    )
    output_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    features.to_parquet(output_path, index=False)
    os.chmod(output_path, 0o600)
    metadata = {
        "schema_version": 1,
        "created_at": utc_now_iso(),
        "split": split,
        "protocol": str(protocol_path),
        "protocol_sha256": sha256_file(protocol_path),
        "manifest_sha256": sha256_file(manifest_path),
        "source_manifest_sha256": sha256_file(source_path),
        "encoder_config_sha256": sha256_file(config_path),
        "encoder_snapshot_sha256": sha256_file(snapshot_path),
        "preprocessing": protocol["encoder"]["preprocessing"],
        "views": len(features),
        "dimension": matrix.shape[1],
        "embeddings_sha256": sha256_file(output_path),
    }
    _write_restricted_json(metadata_path, metadata)
    print(
        f"Mirai embeddings extracted: split={split} views={len(features)} "
        f"dimension={matrix.shape[1]} device={device.type}"
    )
    return 0


def select_sensitivity_threshold(
    scores: Iterable[float],
    labels: Iterable[int],
    *,
    minimum_sensitivity: float,
) -> float:
    """Choose the largest inclusive threshold meeting target sensitivity."""
    score_array = np.asarray(list(scores), dtype=float)
    label_array = np.asarray(list(labels), dtype=int)
    if score_array.ndim != 1 or score_array.shape != label_array.shape:
        raise ValueError("scores and labels must be aligned one-dimensional arrays")
    if not 0 < minimum_sensitivity <= 1:
        raise ValueError("minimum sensitivity must be in (0, 1]")
    positive_scores = score_array[label_array == 1]
    if not len(positive_scores):
        raise ValueError("threshold selection requires positive development labels")
    required_true_positives = math.ceil(
        minimum_sensitivity * len(positive_scores) - 1e-12
    )
    ranked = np.sort(positive_scores)[::-1]
    return float(ranked[required_true_positives - 1])


def confusion_counts(
    labels: Iterable[int], scores: Iterable[float], threshold: float
) -> dict[str, int]:
    labels = np.asarray(list(labels), dtype=int)
    predicted = np.asarray(list(scores), dtype=float) >= threshold
    return {
        "true_positive": int(((labels == 1) & predicted).sum()),
        "false_negative": int(((labels == 1) & ~predicted).sum()),
        "false_positive": int(((labels == 0) & predicted).sum()),
        "true_negative": int(((labels == 0) & ~predicted).sum()),
    }


def rates_from_counts(counts: Mapping[str, int]) -> dict[str, float]:
    positive = counts["true_positive"] + counts["false_negative"]
    negative = counts["true_negative"] + counts["false_positive"]
    return {
        "sensitivity": counts["true_positive"] / positive,
        "specificity": counts["true_negative"] / negative,
    }


def choose_c(cv_results: Iterable[Mapping[str, float]]) -> float:
    """Choose maximum mean AUC, with smaller C as the fixed tie-break."""
    rows = list(cv_results)
    if not rows:
        raise ValueError("C selection requires cross-validation results")
    return float(min(rows, key=lambda row: (-row["mean_roc_auc"], row["c"]))["c"])


def _feature_table(
    path: Path, expected_ids: set[str]
) -> tuple[pd.DataFrame, list[str]]:
    table = pd.read_parquet(path).copy()
    if "view_id" not in table:
        raise ValueError(f"embedding table is missing view_id: {path}")
    table["view_id"] = table["view_id"].map(normalize_view_id)
    if table["view_id"].duplicated().any():
        raise ValueError(f"embedding table contains duplicate view IDs: {path}")
    if set(table["view_id"]) != expected_ids:
        raise ValueError(f"embedding table does not cover its frozen manifest: {path}")
    feature_columns = sorted(
        column for column in table.columns if column.startswith("f")
    )
    if not feature_columns:
        raise ValueError(f"embedding table has no feature columns: {path}")
    if table[feature_columns].isna().any().any():
        raise ValueError(f"embedding table contains missing features: {path}")
    return table.set_index("view_id"), feature_columns


def _new_classifier(c_value: float, seed: int) -> Any:
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    return Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    C=c_value,
                    penalty="l2",
                    solver="liblinear",
                    class_weight="balanced",
                    max_iter=10_000,
                    random_state=seed,
                ),
            ),
        ]
    )


def cross_validated_classifier(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    c_grid: Iterable[float],
    folds: int,
    seed: int,
) -> tuple[float, np.ndarray, list[dict[str, Any]]]:
    """Select C and return aligned OOF probabilities for that frozen C."""
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold

    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    splits = list(splitter.split(features, labels))
    results = []
    predictions_by_c: dict[float, np.ndarray] = {}
    for c_value in c_grid:
        oof = np.full(len(labels), np.nan, dtype=float)
        fold_auc = []
        for train_index, validation_index in splits:
            model = _new_classifier(float(c_value), seed)
            model.fit(features[train_index], labels[train_index])
            scores = model.predict_proba(features[validation_index])[:, 1]
            oof[validation_index] = scores
            fold_auc.append(float(roc_auc_score(labels[validation_index], scores)))
        if np.isnan(oof).any():
            raise RuntimeError("cross-validation did not predict every development row")
        c_value = float(c_value)
        predictions_by_c[c_value] = oof
        results.append(
            {
                "c": c_value,
                "mean_roc_auc": float(np.mean(fold_auc)),
                "standard_deviation_roc_auc": float(np.std(fold_auc)),
                "fold_roc_auc": fold_auc,
            }
        )
    selected_c = choose_c(results)
    return selected_c, predictions_by_c[selected_c], results


def _save_linear_model(
    path: Path,
    *,
    model: Any,
    threshold: float,
    feature_columns: list[str],
    protocol_sha256: str,
) -> None:
    path = Path(path).resolve()
    if path.exists():
        raise FileExistsError(f"refusing to overwrite linear model: {path}")
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    scaler = model.named_steps["scale"]
    classifier = model.named_steps["classifier"]
    with path.open("wb") as handle:
        np.savez_compressed(
            handle,
            schema_version=np.asarray([1], dtype=np.int64),
            feature_columns=np.asarray(feature_columns),
            scaler_mean=np.asarray(scaler.mean_, dtype=np.float64),
            scaler_scale=np.asarray(scaler.scale_, dtype=np.float64),
            classifier_coef=np.asarray(classifier.coef_, dtype=np.float64),
            classifier_intercept=np.asarray(classifier.intercept_, dtype=np.float64),
            threshold=np.asarray([threshold], dtype=np.float64),
            protocol_sha256=np.asarray([protocol_sha256]),
        )
    os.chmod(path, 0o600)


def fit_predict(args: argparse.Namespace) -> int:
    from sklearn.metrics import roc_auc_score

    protocol_path, protocol = load_protocol(args.protocol)
    paths = {
        "development_manifest": _required_file(
            args.development_manifest, "development manifest"
        ),
        "development_state": _required_file(
            args.development_state, "development state"
        ),
        "development_embeddings": _required_file(
            args.development_embeddings, "development embeddings"
        ),
        "audit_manifest": _required_file(args.audit_manifest, "audit manifest"),
        "audit_embeddings": _required_file(args.audit_embeddings, "audit embeddings"),
    }
    require_protocol_file(
        protocol, "development", "manifest", paths["development_manifest"]
    )
    require_protocol_file(protocol, "development", "state", paths["development_state"])
    require_protocol_file(protocol, "audit", "manifest", paths["audit_manifest"])
    development_manifest = _manifest(paths["development_manifest"])
    audit_manifest = _manifest(paths["audit_manifest"])
    development_features, feature_columns = _feature_table(
        paths["development_embeddings"], set(development_manifest["view_id"])
    )
    audit_features, audit_feature_columns = _feature_table(
        paths["audit_embeddings"], set(audit_manifest["view_id"])
    )
    if feature_columns != audit_feature_columns:
        raise ValueError("development and audit embedding columns differ")
    state = load_view_qc_state(paths["development_state"])
    progress = summarize_view_qc_state(state, development_manifest["view_id"])
    if progress["remaining"] or progress["low_confidence"]:
        raise RuntimeError("development labels are incomplete or low confidence")

    development_order = development_manifest["view_id"].tolist()
    audit_order = audit_manifest["view_id"].tolist()
    x_development = development_features.loc[
        development_order, feature_columns
    ].to_numpy(dtype=np.float64)
    y_development = np.asarray(
        [
            int(state["labels"][view_id]["label"] == VIEW_LABEL_PRESENT)
            for view_id in development_order
        ],
        dtype=int,
    )
    x_audit = audit_features.loc[audit_order, feature_columns].to_numpy(
        dtype=np.float64
    )
    classifier_spec = protocol["classifier"]
    selected_c, oof_scores, cv_results = cross_validated_classifier(
        x_development,
        y_development,
        c_grid=classifier_spec["c_grid"],
        folds=int(classifier_spec["cv"]["folds"]),
        seed=int(classifier_spec["cv"]["seed"]),
    )
    minimum_sensitivity = float(
        classifier_spec["threshold_selection"]["minimum_sensitivity"]
    )
    threshold = select_sensitivity_threshold(
        oof_scores,
        y_development,
        minimum_sensitivity=minimum_sensitivity,
    )
    oof_counts = confusion_counts(y_development, oof_scores, threshold)
    oof_rates = rates_from_counts(oof_counts)
    if oof_rates["sensitivity"] + 1e-12 < minimum_sensitivity:
        raise RuntimeError(
            "selected OOF threshold does not meet its frozen sensitivity"
        )

    model = _new_classifier(selected_c, int(classifier_spec["cv"]["seed"]))
    model.fit(x_development, y_development)
    audit_scores = model.predict_proba(x_audit)[:, 1]
    protocol_hash = sha256_file(protocol_path)
    model_path = Path(args.model_output).resolve()
    run_path = Path(args.run_output).resolve()
    metrics_path = Path(args.metrics_output).resolve()
    scores_path = Path(args.scores_output).resolve()
    if any(path.exists() for path in (model_path, run_path, metrics_path, scores_path)):
        raise FileExistsError("refusing to overwrite embedding classifier outputs")
    _save_linear_model(
        model_path,
        model=model,
        threshold=threshold,
        feature_columns=feature_columns,
        protocol_sha256=protocol_hash,
    )

    run = new_view_auto_run(
        target=protocol["target"],
        model="Mirai-v0.8-image-encoder+L2-logistic-regression",
        prompt_variant=MODEL_VARIANT,
        inference_settings={
            "protocol_file": str(protocol_path),
            "protocol_sha256": protocol_hash,
            "development_embeddings_sha256": sha256_file(
                paths["development_embeddings"]
            ),
            "audit_embeddings_sha256": sha256_file(paths["audit_embeddings"]),
            "linear_model_file": str(model_path),
            "linear_model_sha256": sha256_file(model_path),
            "selected_c": selected_c,
            "inclusive_probability_threshold": threshold,
            "threshold_selection": classifier_spec["threshold_selection"],
        },
    )
    run["backend"] = "frozen_mirai_embedding_linear"
    run["prompt_mode"] = "supervised_embedding_classifier"
    image_paths = audit_manifest.set_index("view_id")["image_path"].astype(str)
    for view_id, score in zip(audit_order, audit_scores):
        suggestions = []
        if score >= threshold:
            suggestions.append(
                {
                    "tag": protocol["target"],
                    "confidence": "high",
                    "rationale": (
                        "Frozen Mirai embedding score "
                        f"{score:.6f} met the preregistered threshold "
                        f"{threshold:.6f}."
                    ),
                }
            )
        run["view_suggestions"][view_id] = {
            "image_path": image_paths.at[view_id],
            "suggestions": suggestions,
        }
    save_view_auto_run(run_path, run)

    scores = pd.DataFrame(
        {
            "view_id": audit_order,
            "embedding_probability": audit_scores,
            "embedding_target_present": audit_scores >= threshold,
        }
    )
    scores_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    scores.to_parquet(scores_path, index=False)
    os.chmod(scores_path, 0o600)
    metrics = {
        "schema_version": 1,
        "created_at": utc_now_iso(),
        "protocol_sha256": protocol_hash,
        "development": {
            "views": len(y_development),
            "positives": int(y_development.sum()),
            "negatives": int((y_development == 0).sum()),
            "selected_c": selected_c,
            "cv_results": cv_results,
            "oof_roc_auc": float(roc_auc_score(y_development, oof_scores)),
            "inclusive_threshold": threshold,
            "oof_counts": oof_counts,
            "oof_rates": oof_rates,
        },
        "audit_without_reference_access": {
            "views": len(audit_scores),
            "predicted_positive": int((audit_scores >= threshold).sum()),
            "predicted_negative": int((audit_scores < threshold).sum()),
            "score_minimum": float(audit_scores.min()),
            "score_median": float(np.median(audit_scores)),
            "score_maximum": float(audit_scores.max()),
        },
        "outputs": {
            "model_sha256": sha256_file(model_path),
            "run_sha256": sha256_file(run_path),
            "scores_sha256": sha256_file(scores_path),
        },
    }
    _write_restricted_json(metrics_path, metrics)
    print(
        "Mirai embedding classifier fitted without audit references: "
        f"selected_c={selected_c:g} oof_sensitivity={oof_rates['sensitivity']:.4f} "
        f"oof_specificity={oof_rates['specificity']:.4f} "
        f"audit_positive={int((audit_scores >= threshold).sum())}/{len(audit_scores)}"
    )
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    register = subparsers.add_parser(
        "register", help="freeze data, encoder, classifier, threshold, and arm rules"
    )
    register.add_argument("--development-manifest", type=Path, required=True)
    register.add_argument("--development-source-manifest", type=Path, required=True)
    register.add_argument("--development-state", type=Path, required=True)
    register.add_argument("--audit-manifest", type=Path, required=True)
    register.add_argument("--audit-source-manifest", type=Path, required=True)
    register.add_argument("--audit-protocol", type=Path, required=True)
    register.add_argument("--encoder-config", type=Path, required=True)
    register.add_argument("--encoder-snapshot", type=Path, required=True)
    register.add_argument("--output", type=Path, required=True)
    register.set_defaults(func=register_protocol)

    extract = subparsers.add_parser(
        "extract", help="extract frozen Mirai image embeddings from original DICOMs"
    )
    extract.add_argument("--protocol", type=Path, required=True)
    extract.add_argument("--split", choices=("development", "audit"), required=True)
    extract.add_argument("--manifest", type=Path, required=True)
    extract.add_argument("--source-manifest", type=Path, required=True)
    extract.add_argument("--raw-root", type=Path, required=True)
    extract.add_argument("--temp-root", type=Path, required=True)
    extract.add_argument("--mirai-repo", type=Path, required=True)
    extract.add_argument("--encoder-config", type=Path, required=True)
    extract.add_argument("--encoder-snapshot", type=Path, required=True)
    extract.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    extract.add_argument("--batch-size", type=int, default=4)
    extract.add_argument("--output", type=Path, required=True)
    extract.set_defaults(func=extract_embeddings)

    fit = subparsers.add_parser(
        "fit-predict",
        help="select on development OOF predictions, fit, and predict audit views",
    )
    fit.add_argument("--protocol", type=Path, required=True)
    fit.add_argument("--development-manifest", type=Path, required=True)
    fit.add_argument("--development-state", type=Path, required=True)
    fit.add_argument("--development-embeddings", type=Path, required=True)
    fit.add_argument("--audit-manifest", type=Path, required=True)
    fit.add_argument("--audit-embeddings", type=Path, required=True)
    fit.add_argument("--model-output", type=Path, required=True)
    fit.add_argument("--run-output", type=Path, required=True)
    fit.add_argument("--scores-output", type=Path, required=True)
    fit.add_argument("--metrics-output", type=Path, required=True)
    fit.set_defaults(func=fit_predict)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if hasattr(args, "batch_size") and args.batch_size < 1:
        raise ValueError("batch size must be positive")
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
