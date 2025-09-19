#!/usr/bin/env python3
"""prima_pipeline.py

Deterministic one-pass preprocessor for ChiMEC mammograms.

It builds a single source-of-truth (SoT) and a Mirai-compatible pixel cache — without
writing PNGs. Subsequent training/dev reads only Parquet+Zarr; raw DICOMs stay untouched.

Design constraints (matching Mirai exactly):
- accept only the four presentation views per exam: L-CC, L-MLO, R-CC, R-MLO
- mimic DCMTK `dcmj2pnm +on2 --min-max-window` (16-bit min–max VOI), then resize to 1664×2048
- defer normalization (mean/std) to the consumer; store uint16 in Zarr

References: Mirai README + supplementary material
- requires PNG16 generated via `+on2` and `--min-max-window` and uses `--img_size 1664 2048`
  and `--img_mean 7047.99` / `--img_std 12005.5`
- expects exactly four For Presentation views (no implants/markups); CSV keys are patient_id + exam_id

This script does five things:
1) indexes DICOMs and selects the canonical four views per exam
2) performs minimal QC and prints a summary
3) writes SoT Parquet tables (exams, views, cohort) joined with genotype
4) writes a Zarr cache per exam with four uint16 arrays (L_CC, L_MLO, R_CC, R_MLO)
5) emits a manifest.parquet pointing to the per-exam Zarr groups

Usage example
-------------
python prima_pipeline.py preprocess \
  --raw /data/prima/raw \
  --sot /data/prima/sot \
  --out /data/prima/mirai-prep \
  --genotype /data/prima/sot/genotype.parquet \
  --site ucmed \
  --workers 8

Dependencies (install explicitly):
  pip install pydicom numpy pandas pyarrow zarr numcodecs opencv-python-headless einops tqdm

Notes
-----
- no defensive coding; if a required tag is missing or an assumption is violated, we raise
- avoid silent defaults; CLI args are required
- per-image operations are vectorized; per-exam iteration is unavoidable by I/O
- rearrange is used for shape control; never permute/transpose

"""

from __future__ import annotations

import argparse
import hashlib
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import pydicom
import torch
import zarr
from einops import rearrange
from numcodecs import Blosc
from pydicom.dataset import FileDataset
from pydicom.tag import Tag
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# hard requirements mirrored from Mirai
IMG_W = 1664
IMG_H = 2048
IMG_MEAN = 7047.99
IMG_STD = 12005.5

# allowed view labels for canonical four-pack
VIEWS = {
    ("L", "CC"): "L_CC",
    ("L", "MLO"): "L_MLO",
    ("R", "CC"): "R_CC",
    ("R", "MLO"): "R_MLO",
}

# columns for SoT tables
EXAMS_COLS = [
    "patient_id",
    "patient_hash",
    "exam_id",
    "study_date",
    "accession_number",
    "site",
    "device_manufacturer",
    "device_model",
    "presentation_intent",
    "n_views_present",
    "has_full_quad",
    "ibroker_study_id",
    "has_geno",
]

VIEWS_COLS = [
    "patient_id",
    "exam_id",
    "sop_instance_uid",
    "laterality",
    "view",
    "dicom_path",
    "rows",
    "cols",
    "photometric_interpretation",
    "bits_stored",
    "acquisition_time",
    "is_marked_up",
    "has_implant",
    "for_presentation",
    "sha256",
    "device_manufacturer",
    "device_model",
    "study_date",
    "accession_number",
]


@dataclass(frozen=True)
class PreprocessConfig:
    """Immutable configuration for preprocessing.

    All paths are required. No defaults to avoid accidental misconfiguration.
    """

    raw_dir: Path
    sot_dir: Path
    out_dir: Path
    genotype_parquet: Path
    site: str
    workers: int
    summary: bool


# ------------- DICOM helpers -------------


def read_dicom(path: Path) -> FileDataset:
    """Read a DICOM from disk using pydicom without applying VOI LUT.

    Parameters
    ----------
    path : Path
        Filesystem path to the DICOM (.dcm) file.

    Returns
    -------
    FileDataset
        Parsed pydicom dataset. Pixel data is not normalized.
    """
    return pydicom.dcmread(str(path), force=True)


def get_tag(
    ds: FileDataset, tag: Tuple[int, int], default: Optional[str] = None
) -> Optional[str]:
    """Fetch a DICOM tag as string if present.

    parameters
    ----------
    ds : FileDataset
        dicom dataset
    tag : Tuple[int,int]
        group, element of the tag
    default : Optional[str]
        value to return if tag missing

    returns
    -------
    Optional[str]
    """
    t = Tag(tag)
    if t in ds:
        v = ds[t].value
        return str(v)
    return default


def infer_view_fields(ds: FileDataset) -> Tuple[str, str]:
    """Infer laterality (L/R) and view (CC/MLO) from standard tags.

    raises if tags are missing or values are not in the allowed set.
    """
    lat = get_tag(ds, (0x0020, 0x0062)) or get_tag(ds, (0x0020, 0x0060))
    vp = get_tag(ds, (0x0018, 0x5101))
    if lat is None or vp is None:
        raise ValueError("missing laterality or ViewPosition")
    lat = lat.strip().upper()
    vp = vp.strip().upper()
    if lat not in {"L", "R"}:
        raise ValueError(f"unexpected laterality: {lat}")
    if vp not in {"CC", "MLO"}:
        raise ValueError(f"unexpected view position: {vp}")
    return lat, vp


def is_for_presentation(ds: FileDataset) -> bool:
    """True if PresentationIntentType == 'FOR PRESENTATION'."""
    pit = get_tag(ds, (0x0008, 0x0068), "")
    return str(pit).strip().upper() == "FOR PRESENTATION"


def has_implant(ds: FileDataset) -> bool:
    """Return True if BreastImplantPresent (0028,1300) == 'YES'."""
    bip = get_tag(ds, (0x0028, 0x1300))
    return str(bip).strip().upper() == "YES" if bip is not None else False


def is_marked_up(ds: FileDataset) -> bool:
    """Use 'Burned In Annotation' (0028,0301) == 'YES' if present; else False."""
    bia = get_tag(ds, (0x0028, 0x0301))
    return str(bia).strip().upper() == "YES" if bia is not None else False


def to_uint16_minmax(ds: FileDataset) -> np.ndarray:
    """Convert DICOM pixel data to uint16 using min–max VOI (DCMTK --min-max-window).

    Steps
    -----
    - decode PixelData to ndarray
    - apply RescaleSlope/RescaleIntercept if present
    - if PhotometricInterpretation == MONOCHROME1, invert after scaling
    - linearly map min..max → 0..65535, cast to uint16
    - no clipping beyond [min, max] since mapping uses data extents

    returns
    -------
    np.ndarray of shape (H, W), dtype=uint16
    """
    arr = ds.pixel_array.astype(np.float32, copy=False)
    slope = float(ds.get("RescaleSlope", 1.0))
    inter = float(ds.get("RescaleIntercept", 0.0))
    arr = arr * slope + inter
    vmin = float(np.min(arr))
    vmax = float(np.max(arr))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
        raise ValueError("degenerate dynamic range in pixel data")
    arr = (arr - vmin) * (65535.0 / (vmax - vmin))
    if str(ds.get("PhotometricInterpretation", "")).strip().upper() == "MONOCHROME1":
        arr = 65535.0 - arr
    return arr.astype(np.uint16, copy=False)


def resize_uint16(img: np.ndarray, width: int, height: int) -> np.ndarray:
    """Resize a uint16 image with area interpolation for downscale / linear for upscale.

    parameters
    ----------
    img : np.ndarray
        2D uint16 image
    width : int
        target width in pixels
    height : int
        target height in pixels
    """
    h, w = img.shape
    if h == height and w == width:
        return img
    interp = cv2.INTER_AREA if (height < h or width < w) else cv2.INTER_LINEAR
    out = cv2.resize(img, (width, height), interpolation=interp)
    return out.astype(np.uint16, copy=False)


def hash_file_sha256(path: Path, chunk: int = 1024 * 1024) -> str:
    """Compute streaming SHA256 for provenance.

    parameters
    ----------
    path : Path
        file path
    chunk : int
        chunk size in bytes for streaming
    """
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


# ------------- Discovery & selection -------------


def discover_dicoms(raw_dir: Path) -> pd.DataFrame:
    """Scan `raw_dir` recursively for DICOMs and extract minimal tags.

    builds one row per file; heavy pixel operations are deferred.
    """
    rows = []
    for p in raw_dir.rglob("*.dcm"):
        ds = read_dicom(p)
        patient_id = get_tag(ds, (0x0010, 0x0020))
        study_uid = get_tag(ds, (0x0020, 0x000D))
        sop_uid = get_tag(ds, (0x0008, 0x0018))
        if patient_id is None or study_uid is None or sop_uid is None:
            raise ValueError(f"missing key ids in {p}")
        lat, vp = infer_view_fields(ds)
        present = is_for_presentation(ds)
        rows.append(
            {
                "patient_id": patient_id,
                "exam_id": study_uid,
                "sop_instance_uid": sop_uid,
                "laterality": lat,
                "view": vp,
                "dicom_path": str(p.resolve()),
                "rows": int(ds.Rows),
                "cols": int(ds.Columns),
                "photometric_interpretation": str(
                    ds.get("PhotometricInterpretation", "")
                ),
                "bits_stored": int(ds.get("BitsStored", ds.get("BitsAllocated", 16))),
                "acquisition_time": get_tag(ds, (0x0008, 0x0032), ""),
                "is_marked_up": is_marked_up(ds),
                "has_implant": has_implant(ds),
                "for_presentation": present,
                "sha256": hash_file_sha256(p),
                "device_manufacturer": str(ds.get("Manufacturer", "")),
                "device_model": str(ds.get("ManufacturerModelName", "")),
                "study_date": get_tag(ds, (0x0008, 0x0020), ""),
                "accession_number": get_tag(ds, (0x0008, 0x0050), ""),
            }
        )
    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("no DICOMs found")
    return df


def select_full_quad(df_views: pd.DataFrame) -> pd.DataFrame:
    """Return only rows that constitute exactly one L/R × CC/MLO per exam.

    selection policy for duplicates per (exam, laterality, view):
      - prefer not marked up
      - prefer higher bits stored
      - prefer later acquisition_time
    """
    df = df_views.copy()
    df = df[df["for_presentation"]]
    df = df[df["laterality"].isin(["L", "R"]) & df["view"].isin(["CC", "MLO"])]

    df["_rank"] = (
        (~df["is_marked_up"]).astype(int).astype(np.int64) * 1_000_000
        + df["bits_stored"].astype(np.int64) * 1_000
        + pd.to_numeric(df["acquisition_time"].str.replace(":", ""), errors="coerce")
        .fillna(0)
        .astype(np.int64)
    )

    # pick the best per exam+view key
    df = df.sort_values(
        ["exam_id", "laterality", "view", "_rank"], ascending=[True, True, True, False]
    )
    df = df.groupby(["exam_id", "laterality", "view"], as_index=False).head(1)

    # keep only exams that now have all four views
    counts = df.groupby("exam_id").size()
    full_exams = counts[counts == 4].index
    out = (
        df[df["exam_id"].isin(full_exams)]
        .drop(columns=["_rank"])
        .reset_index(drop=True)
    )
    return out


# ------------- Zarr writing -------------


def write_exam_zarr(
    exam_rows: pd.DataFrame, root_dir: Path
) -> Tuple[Path, Dict[str, Tuple[int, int]]]:
    """Write one Zarr group per exam with four uint16 arrays.

    parameters
    ----------
    exam_rows : pd.DataFrame
        exactly 4 rows for a single exam
    root_dir : Path
        base directory for zarr groups; subdirs are /{patient_hash}/{exam_id}.zarr

    returns
    -------
    (zarr_path, shapes)
    """
    assert len(exam_rows) == 4
    first = exam_rows.iloc[0]
    patient_id = first["patient_id"]
    exam_id = first["exam_id"]
    patient_hash = hashlib.sha1(patient_id.encode("utf-8")).hexdigest()[:12]

    zpath = root_dir / patient_hash / f"{exam_id}.zarr"
    zpath.parent.mkdir(parents=True, exist_ok=True)

    # compressor tuned for uint16 imagery
    compressor = Blosc(cname="zstd", clevel=5, shuffle=Blosc.BITSHUFFLE)
    store = zarr.DirectoryStore(str(zpath))
    grp = zarr.group(store=store, overwrite=True)
    grp.attrs.update(
        {
            "patient_id": patient_id,
            "exam_id": exam_id,
            "img_dtype": "uint16",
            "img_size": [IMG_H, IMG_W],
            "transform": "minmax_uint16_then_resize",
            "source": "DICOM",
        }
    )

    shapes: Dict[str, Tuple[int, int]] = {}

    for _, r in exam_rows.iterrows():
        key = VIEWS[(r["laterality"], r["view"])]
        ds = read_dicom(Path(r["dicom_path"]))
        arr16 = to_uint16_minmax(ds)
        arr16 = resize_uint16(arr16, IMG_W, IMG_H)
        # write chunks ~half-size tiles for balance
        chunks = (IMG_H // 2, IMG_W // 2)
        zarr_arr = grp.create_dataset(
            name=key,
            shape=(IMG_H, IMG_W),
            dtype=np.uint16,
            chunks=chunks,
            compressor=compressor,
            overwrite=True,
        )
        zarr_arr[:] = arr16
        zarr_arr.attrs.update(
            {
                "laterality": r["laterality"],
                "view": r["view"],
                "source_sop_instance_uid": r["sop_instance_uid"],
                "photometric_interpretation": r["photometric_interpretation"],
            }
        )
        shapes[key] = (IMG_H, IMG_W)

    return zpath, shapes


# ------------- Orchestration -------------


def preprocess(cfg: PreprocessConfig) -> None:
    """Run the full pipeline: index, select, cache, and emit SoT + manifest.

    prints a concise QC summary at the end.
    """
    raw = cfg.raw_dir
    sot = cfg.sot_dir
    out = cfg.out_dir

    # discovery
    views_df = discover_dicoms(raw)

    # selection to full quad
    sel_df = select_full_quad(views_df)

    # exams table
    exams = (
        sel_df.groupby(["patient_id", "exam_id"], as_index=False)
        .agg(
            {
                "dicom_path": "count",
                "device_manufacturer": "first",
                "device_model": "first",
                "study_date": "first",
                "accession_number": "first",
            }
        )
        .rename(columns={"dicom_path": "n_views_present"})
    )
    exams["has_full_quad"] = exams["n_views_present"] == 4
    exams["patient_hash"] = exams["patient_id"].apply(
        lambda s: hashlib.sha1(s.encode("utf-8")).hexdigest()[:12]
    )
    exams["presentation_intent"] = "FOR PRESENTATION"
    exams["site"] = cfg.site
    exams["ibroker_study_id"] = None

    # genotype join determines cohort
    geno = pd.read_parquet(cfg.genotype_parquet)
    if "patient_id" not in geno.columns:
        raise KeyError("genotype.parquet must have column 'patient_id'")

    exams = exams.merge(
        geno[["patient_id"]].drop_duplicates().assign(has_geno=True),
        on="patient_id",
        how="left",
    )
    exams["has_geno"] = exams["has_geno"].fillna(False)

    # write SoT tables
    sot.mkdir(parents=True, exist_ok=True)
    sel_df[VIEWS_COLS].to_parquet(sot / "views.parquet", index=False)
    exams[EXAMS_COLS].to_parquet(sot / "exams.parquet", index=False)

    cohort = exams[(exams["has_full_quad"]) & (exams["has_geno"])][
        ["patient_id", "patient_hash", "exam_id", "study_date"]
    ]
    cohort.to_parquet(sot / "cohort.parquet", index=False)

    # if summary-only, stop here
    if cfg.summary:
        print("=== summary-only: wrote SoT, skipped zarr cache ===")
        print(f"SoT dir: {sot}")
        return

    # zarr cache + manifest for the cohort
    prep_dir = out / "zarr"
    prep_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows: List[Dict] = []

    grouped = {k: v for k, v in sel_df.groupby("exam_id")}
    targets = [
        (eid, grouped[eid]) for eid in cohort["exam_id"].tolist() if eid in grouped
    ]

    from concurrent.futures import ThreadPoolExecutor, as_completed

    with ThreadPoolExecutor(max_workers=cfg.workers) as ex:
        futs = {
            ex.submit(write_exam_zarr, grp_df, prep_dir): eid for eid, grp_df in targets
        }
        for fut in tqdm(
            as_completed(futs), total=len(futs), desc="writing zarr", unit="exam"
        ):
            zpath, shapes = fut.result()
            eid = futs[fut]
            grp_df = grouped[eid]
            for _, r in grp_df.iterrows():
                key = VIEWS[(r["laterality"], r["view"])]
                manifest_rows.append(
                    {
                        "patient_id": r["patient_id"],
                        "exam_id": r["exam_id"],
                        "laterality": r["laterality"],
                        "view": r["view"],
                        "zarr_uri": str(zpath),
                        "zarr_key": key,
                        "height": shapes[key][0],
                        "width": shapes[key][1],
                    }
                )

    manifest = pd.DataFrame(manifest_rows)
    manifest_path = out / "manifest.parquet"
    manifest.to_parquet(manifest_path, index=False)

    # QC summary
    total_files = len(views_df)
    total_exams = views_df[["exam_id"]].nunique()[0]
    full_exams = exams["has_full_quad"].sum()
    cohort_exams = len(cohort)

    vendor_counts = (
        sel_df.groupby(["device_manufacturer", "device_model"], as_index=False)
        .size()
        .rename(columns={"size": "n_views"})
    )

    print("=== QC summary ===")
    print(f"dicom files indexed: {total_files}")
    print(f"unique exams: {total_exams}")
    print(f"full-quad exams: {full_exams}")
    print(f"cohort (full-quad ∩ has-geno): {cohort_exams}")
    print("views by vendor/model:")
    print(vendor_counts.to_string(index=False))
    print(f"written SoT: {sot}")
    print(f"written cache manifest: {manifest_path}")


# ------------- Consumer helpers -------------


def load_view_from_manifest(
    manifest_row: pd.Series, normalize: bool = True, channels: int = 3
) -> np.ndarray:
    """Load a single view from manifest row, optionally normalize and expand channels.

    returns CHW float32 array ready for pytorch
    """
    z = zarr.open(str(manifest_row["zarr_uri"]), mode="r")
    arr16: np.ndarray = z[manifest_row["zarr_key"]][:]
    x = arr16.astype(np.float32)
    if normalize:
        x = (x - IMG_MEAN) / IMG_STD
    x = rearrange(x, "h w -> 1 h w")
    if channels == 3:
        x = np.repeat(x, 3, axis=0)
    return x


def write_mirai_csv(
    manifest_parquet: Path, labels_parquet: Path, out_csv: Path
) -> None:
    """Create a Mirai-compatible CSV that points to Zarr URIs instead of PNGs.

    The CSV schema mirrors Mirai's documented requirements:
    columns: patient_id, exam_id, laterality, view, file_path,
             years_to_cancer, years_to_last_followup, split_group

    - file_path is encoded as: zarr://<abs_path_to_exam.zarr>#<view_key>
    - strict join on (patient_id, exam_id); raises if any labels missing

    parameters
    ----------
    manifest_parquet : Path
        parquet emitted by preprocess() mapping views to zarr URIs and keys
    labels_parquet : Path
        parquet with columns [patient_id, exam_id, years_to_cancer, years_to_last_followup, split_group]
    out_csv : Path
        destination CSV; parent must exist
    """
    man = pd.read_parquet(manifest_parquet)
    req_m = {"patient_id", "exam_id", "laterality", "view", "zarr_uri", "zarr_key"}
    if req_m - set(man.columns):
        miss = sorted(req_m - set(man.columns))
        raise KeyError(f"manifest missing columns: {miss}")

    lab = pd.read_parquet(labels_parquet)
    req_l = {
        "patient_id",
        "exam_id",
        "years_to_cancer",
        "years_to_last_followup",
        "split_group",
    }
    if req_l - set(lab.columns):
        miss = sorted(req_l - set(lab.columns))
        raise KeyError(f"labels missing columns: {miss}")

    df = man.merge(lab[list(req_l)], on=["patient_id", "exam_id"], how="inner")
    if len(df) != len(man):
        missing = set(man[["patient_id", "exam_id"]].apply(tuple, axis=1)) - set(
            df[["patient_id", "exam_id"]].apply(tuple, axis=1)
        )
        raise ValueError(f"labels missing for {len(missing)} exam(s)")

    df["file_path"] = df.apply(
        lambda r: f"zarr://{Path(r['zarr_uri']).resolve()}#{r['zarr_key']}", axis=1
    )
    out = df[
        [
            "patient_id",
            "exam_id",
            "laterality",
            "view",
            "file_path",
            "years_to_cancer",
            "years_to_last_followup",
            "split_group",
        ]
    ]
    out.to_csv(out_csv, index=False)


def materialize_mirai_embeddings(
    mirai_repo: Path,
    img_encoder_snapshot: Path,
    manifest_parquet: Path,
    out_dir: Path,
    batch_size: int = 8,
    num_workers: int = 4,
) -> None:
    """Dump per-view embeddings using Mirai's encoder weights.

    This implementation loads a torchvision ResNet-18 and attempts to load the
    provided Mirai encoder snapshot into it. If the snapshot format differs, it
    raises loudly so you can adapt the key mapping. Aggregation is left to
    downstream code; we emit per-view 512-d vectors as Parquet.

    parameters
    ----------
    mirai_repo : Path
        path to a local clone of Mirai; added to sys.path for custom loaders if needed
    img_encoder_snapshot : Path
        path to Mirai's ResNet-18 encoder weights (.p / .pt); must be compatible with torchvision resnet18
    manifest_parquet : Path
        our manifest mapping views to zarr; used to build dataset
    out_dir : Path
        base output dir; writes features/mirai_v1/per_view.parquet
    batch_size : int
        dataloader batch size
    num_workers : int
        dataloader workers
    """
    if mirai_repo is not None:
        sys.path.insert(0, str(mirai_repo))

    import torch
    import torch.nn as nn
    import torchvision.models as tv

    enc = tv.resnet18(weights=None)
    enc.fc = nn.Identity()

    sd = torch.load(str(img_encoder_snapshot), map_location="cpu")
    tried = []
    try:
        enc.load_state_dict(sd)
    except Exception as e1:
        tried.append("root")
        ok = False
        for k in (
            "state_dict",
            "model",
            "model_state",
            "encoder_state",
            "net",
            "module",
        ):
            if isinstance(sd, dict) and k in sd:
                try:
                    enc.load_state_dict(sd[k])
                    ok = True
                    break
                except Exception:
                    tried.append(k)
                    continue
        if not ok:
            raise RuntimeError(
                f"cannot load snapshot into resnet18; tried keys {tried}"
            ) from e1

    enc.eval()

    ds = MiraiZarrDataset(manifest_parquet)
    dl = DataLoader(
        ds,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        collate_fn=lambda b: b,
    )

    rows = []
    with torch.no_grad():
        for batch in tqdm(dl, desc="mirai encoder", unit="exams"):
            for sample in batch:
                x = sample["images"]
                v = enc(x)
                v = v.cpu().numpy().astype(np.float32)
                for key, vec in zip(sample["order"], v):
                    rows.append(
                        {
                            "patient_id": sample["patient_id"],
                            "exam_id": sample["exam_id"],
                            "view_key": key,
                            **{f"f{i}": float(vec[i]) for i in range(vec.shape[0])},
                        }
                    )
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    per_view = pd.DataFrame(rows)
    (out_dir / "mirai_v1").mkdir(parents=True, exist_ok=True)
    per_view.to_parquet(out_dir / "mirai_v1" / "per_view.parquet", index=False)


def ingest_mirai_predictions(prediction_csv: Path, out_parquet: Path) -> None:
    """Convert Mirai's validation output CSV to per-exam Parquet.

    expects columns: patient_exam_id (tab-separated patient_id and exam_id),
    and risk predictions for 1-5 years (column names as per demo/validation_output.csv).
    """
    df = pd.read_csv(prediction_csv)
    if "patient_exam_id" not in df.columns:
        raise KeyError("prediction CSV missing 'patient_exam_id'")
    pe = df["patient_exam_id"].astype(str).str.split("	", n=1, expand=True)
    df["patient_id"] = pe[0]
    df["exam_id"] = pe[1]
    keep = [
        c
        for c in df.columns
        if any(
            t in c.lower()
            for t in ("1year", "2year", "3year", "4year", "5year", "risk")
        )
    ]
    out = df[["patient_id", "exam_id"] + keep]
    out.to_parquet(out_parquet, index=False)


# ------------- CLI -------------


def parse_args() -> PreprocessConfig:
    """Parse CLI arguments into a PreprocessConfig.

    all arguments are required; fail if missing.
    """
    p = argparse.ArgumentParser(
        description="Build SoT + Mirai-compatible Zarr cache (no PNGs)"
    )
    p.add_argument("--raw", dest="raw_dir", type=Path, required=True)
    p.add_argument("--sot", dest="sot_dir", type=Path, required=True)
    p.add_argument("--out", dest="out_dir", type=Path, required=True)
    p.add_argument("--genotype", dest="genotype_parquet", type=Path, required=True)
    p.add_argument("--site", dest="site", type=str, required=True)
    p.add_argument("--workers", dest="workers", type=int, required=True)
    p.add_argument("--summary", dest="summary", action="store_true")
    # optional extras
    p.add_argument("--emit-csv", dest="emit_csv", type=Path)
    p.add_argument("--labels", dest="labels_parquet", type=Path)
    p.add_argument("--features-out", dest="features_out", type=Path)
    p.add_argument("--img-encoder-snapshot", dest="img_encoder_snapshot", type=Path)
    p.add_argument("--mirai-repo", dest="mirai_repo", type=Path)
    args = p.parse_args()

    return PreprocessConfig(
        raw_dir=args.raw_dir,
        sot_dir=args.sot_dir,
        out_dir=args.out_dir,
        genotype_parquet=args.genotype_parquet,
        site=args.site,
        workers=args.workers,
        summary=args.summary,
    ), args


def main() -> None:
    """Entrypoint for the CLI."""
    cfg, args = parse_args()
    if cfg.workers <= 0:
        raise ValueError("workers must be > 0")
    if cfg.raw_dir == cfg.out_dir:
        raise ValueError("out_dir must differ from raw_dir")

    preprocess(cfg)

    # emit Mirai CSV if requested
    if args.emit_csv is not None:
        if args.labels_parquet is None:
            raise ValueError(
                "--emit-csv requires --labels with years_to_cancer/years_to_last_followup/split_group"
            )
        write_mirai_csv(
            cfg.out_dir / "manifest.parquet", args.labels_parquet, args.emit_csv
        )
        print(f"wrote Mirai CSV → {args.emit_csv}")

    # materialize per-view embeddings if requested
    if args.features_out is not None:
        if args.img_encoder_snapshot is None:
            raise ValueError(
                "--features-out requires --img-encoder-snapshot (Mirai ResNet18 weights)"
            )
        materialize_mirai_embeddings(
            args.mirai_repo,
            args.img_encoder_snapshot,
            cfg.out_dir / "manifest.parquet",
            args.features_out,
        )
        print(f"wrote per-view embeddings → {args.features_out / 'per_view.parquet'}")


if __name__ == "__main__":
    main()


class MiraiZarrDataset(Dataset):
    """Minimal dataset that reads four-view exams from a manifest and emits tensors.

    It expects a Parquet manifest with columns: patient_id, exam_id, laterality, view,
    zarr_uri, zarr_key. It groups rows by (patient_id, exam_id) and yields a dict with:
        - 'patient_id', 'exam_id'
        - 'images': torch.float32 tensor of shape (4, 3, 2048, 1664), normalized with Mirai mean/std
        - 'order': tuple of view keys in canonical order (L_CC, L_MLO, R_CC, R_MLO)

    Raises if any expected view is missing. Uses rearrange for shape control.
    """

    ORDER = ("L_CC", "L_MLO", "R_CC", "R_MLO")

    def __init__(self, manifest_parquet: Path, channels: int = 3):
        self.manifest = pd.read_parquet(manifest_parquet)
        req = {"patient_id", "exam_id", "laterality", "view", "zarr_uri", "zarr_key"}
        missing = req - set(self.manifest.columns)
        if missing:
            raise KeyError(f"manifest missing columns: {sorted(missing)}")
        self.channels = channels
        self.groups = self.manifest.groupby(["patient_id", "exam_id"], sort=False)
        self.index = list(self.groups.groups.keys())

    def __len__(self) -> int:
        return len(self.index)

    def _load_view(self, row: pd.Series) -> np.ndarray:
        z = zarr.open(str(row["zarr_uri"]), mode="r")
        arr16 = z[row["zarr_key"]][:]
        x = arr16.astype(np.float32)
        x = (x - IMG_MEAN) / IMG_STD
        x = rearrange(x, "h w -> 1 h w")
        if self.channels == 3:
            x = np.repeat(x, 3, axis=0)
        return x

    def __getitem__(self, i: int) -> Dict:
        pid, eid = self.index[i]
        df = self.groups.get_group((pid, eid))
        keyed = {VIEWS[(r["laterality"], r["view"])]: r for _, r in df.iterrows()}
        if any(k not in keyed for k in self.ORDER):
            missing = [k for k in self.ORDER if k not in keyed]
            raise ValueError(f"exam {pid}/{eid} missing views: {missing}")
        imgs = [self._load_view(pd.Series(keyed[k])) for k in self.ORDER]
        x = np.stack(imgs, axis=0)
        return {
            "patient_id": pid,
            "exam_id": eid,
            "images": torch.from_numpy(x),
            "order": self.ORDER,
        }


# --- Patch for Mirai to resolve zarr:// URIs in csv datasets ---
# File: onconet/datasets/csv_mammo.py  (or the module that defines csv_mammo_risk_all_full_future)
# Replace the PNG loader with this function and call it where the code currently opens images


def _load_image_path(file_path: str) -> np.ndarray:
    """Return a CHW float32 tensor from either PNG path or zarr URI.

    Supports two cases:
      - regular PNG path (the original Mirai code path)
      - zarr URI of the form: 'zarr:///abs/path/to/exam.zarr#L_CC'
    """
    if file_path.startswith("zarr://"):
        uri, key = file_path[7:].split("#", 1)
        z = zarr.open(uri, mode="r")
        arr16 = z[key][:]
        x = arr16.astype(np.float32)
        x = (x - IMG_MEAN) / IMG_STD
        x = rearrange(x, "h w -> 1 h w")
        x = np.repeat(x, 3, axis=0)
        return x
    import PIL.Image as Image

    im = Image.open(file_path)
    x = np.array(im, dtype=np.float32)
    x = (x - IMG_MEAN) / IMG_STD
    x = rearrange(x, "h w -> 1 h w")
    x = np.repeat(x, 3, axis=0)
    return x
