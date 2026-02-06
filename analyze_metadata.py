#!/usr/bin/env python3
"""
Script extracted from dev.ipynb
Performs ChiMEC patient data analysis and generates plots
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from prima.filesystem_utils import update_metadata_with_disk_status_by_date
from prima.metadata_utils import extract_base_modality

# argument parser
parser = argparse.ArgumentParser(
    description="Analyze imaging metadata for specified modality"
)
parser.add_argument(
    "--modality",
    type=str,
    default="MG",
    choices=[
        "CR",
        "DX",
        "MG",
        "US",
        "CT",
        "MR",
        "NM",
        "PT",
        "XA",
        "RF",
        "ES",
        "XC",
        "PX",
        "RG",
    ],
    help="Base modality to analyze (default: MG)",
)
parser.add_argument(
    "--dump-screening-patients",
    action="store_true",
    help="Dump CSV of patients with screening scans at least 3 months before diagnosis",
)
args = parser.parse_args()

SELECTED_MODALITY = args.modality

# create plots directory if it doesn't exist
plots_dir = Path("plots")
plots_dir.mkdir(exist_ok=True)

print(f"=== LOADING DATA FOR MODALITY: {SELECTED_MODALITY} ===")
# load data
chimec_patients = pd.read_csv(
    "/gpfs/data/phs/groups/Projects/Huo_projects/SPORE/annawoodard/List_ChiMEC_priority_2025July30.csv"
)
key = pd.read_csv("/gpfs/data/huo-lab/Image/ChiMEC/study-16352a.csv")
all_metadata = pd.read_csv(
    "/gpfs/data/huo-lab/Image/annawoodard/prima/data/imaging_metadata.csv"
)
# add base modality column to metadata (use StudyDescription when Modality is missing)
all_metadata["base_modality"] = all_metadata.apply(
    lambda row: extract_base_modality(row["Modality"], row["StudyDescription"]), axis=1
)

# filter to selected modality FIRST (simplifies all downstream calculations)
metadata = all_metadata[all_metadata["base_modality"] == SELECTED_MODALITY].copy()

print(f"ChiMEC patients: {len(chimec_patients):,}")
print(f"Key table records: {len(key):,}")
print(f"All metadata records: {len(all_metadata):,}")
print(f"{SELECTED_MODALITY} metadata records: {len(metadata):,}")

# update metadata with current disk status using StudyDate matching
# (fingerprint cache is modality-specific, matches filtered metadata)
metadata = update_metadata_with_disk_status_by_date(metadata)

# modality labels for plotting
BASE_MODALITY_LABELS = {
    "CR": "computed radiography",
    "DX": "digital radiography",
    "MG": "mammography",
    "US": "ultrasound",
    "CT": "ct",
    "MR": "mri",
    "NM": "nuclear medicine",
    "PT": "pet",
    "XA": "x-ray angiography",
    "RF": "fluoroscopy",
    "ES": "endoscopy",
    "XC": "external camera",
    "PX": "panoramic x-ray",
    "RG": "radiographic imaging",
}


# helper functions for analysis
def filter_patients_by_chip_status(data, chip_status="all"):
    """
    filter patients by chip status

    Parameters
    ----------
    data : pd.DataFrame
        dataframe with chip column
    chip_status : str
        "all" for all patients, "genotyped" for only patients with non-null chip status

    Returns
    -------
    pd.DataFrame
        filtered dataframe
    """
    if chip_status == "all":
        return data
    elif chip_status == "genotyped":
        return data[data["chip"].notna()]
    else:
        raise ValueError("chip_status must be 'all' or 'genotyped'")


def create_screening_mammograms_plot(
    case_screening_per_patient,
    case_before_dx_month_per_patient,
    control_screening_per_patient,
    case_screening_scans,
    case_before_dx_month_scans,
    control_screening_scans,
    chip_status,
    plots_dir,
    modality="MG",
):
    """create screening scans plot"""
    modality_label = BASE_MODALITY_LABELS.get(modality, modality.lower())
    fig, ((ax1, ax2, ax3), (ax4, ax5, ax6), (ax7, ax8, ax9)) = plt.subplots(
        3, 3, figsize=(24, 18)
    )

    # cases - mammograms >3 months before diagnosis
    if len(case_screening_per_patient) > 0:
        ax1.hist(
            case_screening_per_patient.values,
            bins=range(1, case_screening_per_patient.max() + 2),
            alpha=0.7,
            edgecolor="black",
        )
        ax1.set_xticks(range(1, min(21, case_screening_per_patient.max() + 1)))
    else:
        ax1.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax1.transAxes)

    ax1.set_title(f"cases\n({modality_label} >3 months before diagnosis)")
    ax1.set_xlabel(f"number of screening {modality_label} scans per patient")
    ax1.set_ylabel("number of patients")
    ax1.yaxis.set_major_locator(plt.MaxNLocator(integer=True))

    # cases - mammograms before/including diagnosis month
    if len(case_before_dx_month_per_patient) > 0:
        ax2.hist(
            case_before_dx_month_per_patient.values,
            bins=range(1, case_before_dx_month_per_patient.max() + 2),
            alpha=0.7,
            edgecolor="black",
        )
        ax2.set_xticks(range(1, min(21, case_before_dx_month_per_patient.max() + 1)))
    else:
        ax2.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax2.transAxes)

    ax2.set_title(f"cases\n({modality_label} before/including dx month)")
    ax2.set_xlabel(f"number of {modality_label} scans per patient")
    ax2.set_ylabel("number of patients")
    ax2.yaxis.set_major_locator(plt.MaxNLocator(integer=True))

    # controls
    if len(control_screening_per_patient) > 0:
        ax3.hist(
            control_screening_per_patient.values,
            bins=range(1, control_screening_per_patient.max() + 2),
            alpha=0.7,
            edgecolor="black",
        )
        ax3.set_xticks(range(1, min(21, control_screening_per_patient.max() + 1)))
    else:
        ax3.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax3.transAxes)

    ax3.set_title(f"controls\n(all {modality_label})")
    ax3.set_xlabel(f"number of {modality_label} scans per patient")
    ax3.set_ylabel("number of patients")
    ax3.yaxis.set_major_locator(plt.MaxNLocator(integer=True))

    # fourth panel - collapsed summary
    categories = [
        "cases\n(>3 months before dx)",
        "cases\n(before/including dx month)",
        f"controls\n(all {modality_label})",
    ]
    counts = [
        len(case_screening_per_patient),
        len(case_before_dx_month_per_patient),
        len(control_screening_per_patient),
    ]

    bars4 = ax4.bar(categories, counts, alpha=0.7, edgecolor="black")
    ax4.set_title(f"total patients with {modality_label}")
    ax4.set_ylabel("number of patients")
    ax4.yaxis.set_major_locator(plt.MaxNLocator(integer=True))

    # add value labels on bars
    for bar, count in zip(bars4, counts):
        height = bar.get_height()
        ax4.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.01 * max(counts),
            f"{count}",
            ha="center",
            va="bottom",
        )

    # fifth panel - total scans by category
    categories_scans = [
        "cases\n(>3 months before dx)",
        "cases\n(before/including dx month)",
        f"controls\n(all {modality_label})",
    ]
    total_scans = [
        len(case_screening_scans),
        len(case_before_dx_month_scans),
        len(control_screening_scans),
    ]

    bars5 = ax5.bar(categories_scans, total_scans, alpha=0.7, edgecolor="black")
    ax5.set_title(f"total {modality_label} scans by category")
    ax5.set_ylabel(f"number of {modality_label} scans")
    ax5.yaxis.set_major_locator(plt.MaxNLocator(integer=True))

    # add value labels on bars
    for bar, count in zip(bars5, total_scans):
        height = bar.get_height()
        ax5.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.01 * max(total_scans),
            f"{count}",
            ha="center",
            va="bottom",
        )

    # sixth panel - scan year distribution comparison
    if len(case_screening_scans) > 0 and len(control_screening_scans) > 0:
        # extract years from scan dates
        case_years = case_screening_scans["Study DateTime"].dt.year
        control_years = control_screening_scans["Study DateTime"].dt.year

        # create year bins
        min_year = min(case_years.min(), control_years.min())
        max_year = max(case_years.max(), control_years.max())
        year_bins = range(min_year, max_year + 2)

        # create overlapping histograms with outline style
        ax6.hist(
            case_years,
            bins=year_bins,
            histtype="step",
            label=f"cases (n={len(case_screening_scans)})",
            color="red",
            linewidth=2,
        )
        ax6.hist(
            control_years,
            bins=year_bins,
            histtype="step",
            label=f"controls (n={len(control_screening_scans)})",
            color="blue",
            linewidth=2,
        )

        ax6.set_title(f"screening {modality_label} years\n(cases vs controls)")
        ax6.set_xlabel(f"year of {modality_label}")
        ax6.set_ylabel(f"number of {modality_label} scans")
        ax6.legend()
        ax6.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
        ax6.xaxis.set_major_locator(plt.MaxNLocator(integer=True))

        # rotate x-axis labels if needed
        if max_year - min_year > 10:
            ax6.tick_params(axis="x", rotation=45)
    else:
        ax6.text(
            0.5,
            0.5,
            "insufficient data",
            ha="center",
            va="center",
            transform=ax6.transAxes,
        )
        ax6.set_title(f"screening {modality_label} years\n(cases vs controls)")
        ax6.set_xticks([])
        ax6.set_yticks([])

    # seventh panel - months between screening exam and diagnosis for cases
    if len(case_screening_scans) > 0:
        # calculate months between screening exam and diagnosis
        months_to_dx = (
            case_screening_scans["days_to_dx"] / 30.44
        )  # average days per month

        ax7.hist(months_to_dx, bins=30, alpha=0.7, edgecolor="black")
        ax7.set_title(f"cases\n(months from screening {modality_label} to diagnosis)")
        ax7.set_xlabel("months from screening to diagnosis")
        ax7.set_ylabel(f"number of {modality_label} scans")
        ax7.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
    else:
        ax7.text(
            0.5,
            0.5,
            "no data",
            ha="center",
            va="center",
            transform=ax7.transAxes,
        )
        ax7.set_title(f"cases\n(months from screening {modality_label} to diagnosis)")
        ax7.set_xticks([])
        ax7.set_yticks([])

    # hide unused panels
    ax8.axis("off")
    ax9.axis("off")

    plt.tight_layout()
    plt.savefig(
        plots_dir / f"screening_{modality.lower()}_scans_per_patient_{chip_status}.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


def create_mg_mammograms_per_patient_plot(
    selected_modality_per_patient,
    chimec_with_study_id,
    selected_modality_scans,
    chip_status,
    plots_dir,
    modality="MG",
):
    """create selected modality scans per patient plot"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # get case and control patients
    selected_modality_with_case_control = selected_modality_scans.merge(
        chimec_with_study_id[["AnonymousID", "case_or_control"]],
        left_on="study_id",
        right_on="AnonymousID",
        how="inner",
    )

    # cases
    case_selected_modality_scans = selected_modality_with_case_control[
        selected_modality_with_case_control["case_or_control"] == "Case"
    ]
    case_selected_modality_per_patient = case_selected_modality_scans[
        "study_id"
    ].value_counts()

    modality_label = BASE_MODALITY_LABELS.get(modality, modality.lower())
    ax1.hist(
        case_selected_modality_per_patient.values,
        bins=range(1, max(case_selected_modality_per_patient.max(), 2) + 2),
        alpha=0.7,
        edgecolor="black",
    )
    ax1.set_title("cases")
    ax1.set_xlabel(f"number of {modality_label} scans per patient")
    ax1.set_ylabel("number of patients")
    ax1.set_xticks(range(1, min(21, case_selected_modality_per_patient.max() + 1)))
    ax1.yaxis.set_major_locator(plt.MaxNLocator(integer=True))

    # controls
    control_selected_modality_scans = selected_modality_with_case_control[
        selected_modality_with_case_control["case_or_control"] == "Control"
    ]
    control_selected_modality_per_patient = control_selected_modality_scans[
        "study_id"
    ].value_counts()

    ax2.hist(
        control_selected_modality_per_patient.values,
        bins=range(1, max(control_selected_modality_per_patient.max(), 2) + 2),
        alpha=0.7,
        edgecolor="black",
    )
    ax2.set_title("controls")
    ax2.set_xlabel(f"number of {modality_label} scans per patient")
    ax2.set_ylabel("number of patients")
    ax2.set_xticks(range(1, min(21, control_selected_modality_per_patient.max() + 1)))
    ax2.yaxis.set_major_locator(plt.MaxNLocator(integer=True))

    plt.tight_layout()
    plt.savefig(
        plots_dir / f"{modality.lower()}_scans_per_patient_{chip_status}.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


def create_modality_distribution_plot(metadata, chip_status, plots_dir):
    """create modality distribution plot"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # number of exams by modality
    exam_counts = metadata["base_modality"].value_counts().sort_values(ascending=False)
    # map modality codes to readable labels
    exam_labels = [BASE_MODALITY_LABELS.get(mod, mod) for mod in exam_counts.index]
    ax1.bar(exam_labels, exam_counts.values)
    ax1.set_title("number of exams by modality")
    ax1.set_xlabel("modality")
    ax1.set_ylabel("number of exams")
    ax1.set_xticks(range(len(exam_labels)))
    ax1.set_xticklabels(exam_labels, rotation=45, ha="right")

    # patients with at least one exam in each modality
    patient_counts = (
        metadata.groupby("base_modality")["study_id"]
        .nunique()
        .sort_values(ascending=False)
    )
    # map modality codes to readable labels
    patient_labels = [
        BASE_MODALITY_LABELS.get(mod, mod) for mod in patient_counts.index
    ]
    ax2.bar(patient_labels, patient_counts.values)
    ax2.set_title(
        "patients with ≥1 exam in each modality\n(sum > total patients due to multi-modality)"
    )
    ax2.set_xlabel("modality")
    ax2.set_ylabel("number of patients")
    ax2.set_xticks(range(len(patient_labels)))
    ax2.set_xticklabels(patient_labels, rotation=45, ha="right")
    ax2.yaxis.set_major_locator(plt.MaxNLocator(integer=True))

    plt.tight_layout()
    plt.savefig(
        plots_dir / f"modality_distribution_{chip_status}.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


def create_time_analysis_plot(
    cases_selected_modality,
    controls_selected_modality,
    chip_status,
    plots_dir,
    modality="MG",
):
    """create time analysis plot"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 6))

    modality_label = BASE_MODALITY_LABELS.get(modality, modality.lower())
    # cases
    if len(cases_selected_modality) > 0:
        case_time_to_dx = cases_selected_modality["days_to_dx"].dropna()
        # convert days to years
        case_time_to_dx_years = case_time_to_dx / 365.25
        ax1.hist(case_time_to_dx_years, bins=50, alpha=0.7, edgecolor="black")
        ax1.set_title(f"cases\n(positive = {modality_label} before diagnosis)")
        ax1.set_xlabel(f"years from {modality_label} to diagnosis")
        ax1.set_ylabel(f"number of {modality_label} scans")
        ax1.axvline(x=0, color="red", linestyle="--", alpha=0.7, label="diagnosis date")
        ax1.axvline(
            x=90 / 365.25,
            color="orange",
            linestyle="--",
            alpha=0.7,
            label="3 months before dx (screening threshold)",
        )
        ax1.legend()
    else:
        ax1.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax1.transAxes)
        ax1.set_title(f"cases\n(positive = {modality_label} before diagnosis)")
        ax1.set_xlabel(f"years from {modality_label} to diagnosis")
        ax1.set_ylabel(f"number of {modality_label} scans")

    # controls
    if len(controls_selected_modality) > 0:
        control_time_from_first = controls_selected_modality[
            "days_from_first_scan"
        ].dropna()
        # convert days to years
        control_time_from_first_years = control_time_from_first / 365.25
        ax2.hist(control_time_from_first_years, bins=50, alpha=0.7, edgecolor="black")
        ax2.axvline(
            x=0, color="red", linestyle="--", alpha=0.7, label=f"first {modality_label}"
        )
    else:
        ax2.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax2.transAxes)

    ax2.set_title(f"controls\n(time from first {modality_label})")
    ax2.set_xlabel(f"years from first {modality_label}")
    ax2.set_ylabel(f"number of {modality_label} scans")
    if len(controls_selected_modality) > 0:
        ax2.legend()

    plt.tight_layout()
    plt.savefig(
        plots_dir
        / f"time_analysis_{modality.lower()}_cases_vs_controls_{chip_status}.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


def create_is_on_disk_plot(
    case_screening, case_before_dx, controls, chip_status, plots_dir
):
    """create bar plot showing is_on_disk status by category"""

    # calculate counts for each category
    categories = [
        "cases\n(>3 months before dx)",
        "cases\n(before/including dx month)",
        "controls\n(all)",
    ]

    # count on disk vs not on disk for each category
    case_screening_on_disk = (
        case_screening["is_on_disk"].fillna(False).sum()
        if len(case_screening) > 0
        else 0
    )
    case_screening_not_on_disk = (
        len(case_screening) - case_screening_on_disk if len(case_screening) > 0 else 0
    )

    case_before_dx_on_disk = (
        case_before_dx["is_on_disk"].fillna(False).sum()
        if len(case_before_dx) > 0
        else 0
    )
    case_before_dx_not_on_disk = (
        len(case_before_dx) - case_before_dx_on_disk if len(case_before_dx) > 0 else 0
    )

    controls_on_disk = (
        controls["is_on_disk"].fillna(False).sum() if len(controls) > 0 else 0
    )
    controls_not_on_disk = len(controls) - controls_on_disk if len(controls) > 0 else 0

    on_disk_counts = [case_screening_on_disk, case_before_dx_on_disk, controls_on_disk]
    not_on_disk_counts = [
        case_screening_not_on_disk,
        case_before_dx_not_on_disk,
        controls_not_on_disk,
    ]

    # create figure
    fig, ax = plt.subplots(figsize=(12, 6))

    x = range(len(categories))
    width = 0.35

    # create bars
    bars1 = ax.bar(
        [i - width / 2 for i in x],
        on_disk_counts,
        width,
        label="on disk",
        alpha=0.8,
        color="green",
    )
    bars2 = ax.bar(
        [i + width / 2 for i in x],
        not_on_disk_counts,
        width,
        label="not on disk",
        alpha=0.8,
        color="red",
    )

    # add labels and title
    ax.set_xlabel("category")
    ax.set_ylabel("number of exams")
    ax.set_title(f"exam download status by category ({chip_status} patients)")
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.legend()
    ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))

    # add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height + 0.01 * max(max(on_disk_counts), max(not_on_disk_counts)),
                    f"{int(height):,}",
                    ha="center",
                    va="bottom",
                )

    plt.tight_layout()
    plt.savefig(
        plots_dir / f"download_status_by_category_{chip_status}.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


def create_comprehensive_download_status_plot(
    case_screening, case_before_dx, controls, all_metadata, chip_status, plots_dir
):
    """create comprehensive download status plot with multiple panels"""

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

    # Panel 1: Download status by category (same as before)
    categories = [
        "cases\n(>3 months before dx)",
        "cases\n(before/including dx month)",
        "controls\n(all)",
    ]

    # count on disk vs not on disk for each category
    case_screening_on_disk = (
        case_screening["is_on_disk"].fillna(False).sum()
        if len(case_screening) > 0
        else 0
    )
    case_screening_not_on_disk = (
        len(case_screening) - case_screening_on_disk if len(case_screening) > 0 else 0
    )

    case_before_dx_on_disk = (
        case_before_dx["is_on_disk"].fillna(False).sum()
        if len(case_before_dx) > 0
        else 0
    )
    case_before_dx_not_on_disk = (
        len(case_before_dx) - case_before_dx_on_disk if len(case_before_dx) > 0 else 0
    )

    controls_on_disk = (
        controls["is_on_disk"].fillna(False).sum() if len(controls) > 0 else 0
    )
    controls_not_on_disk = len(controls) - controls_on_disk if len(controls) > 0 else 0

    on_disk_counts = [case_screening_on_disk, case_before_dx_on_disk, controls_on_disk]
    not_on_disk_counts = [
        case_screening_not_on_disk,
        case_before_dx_not_on_disk,
        controls_not_on_disk,
    ]

    x = range(len(categories))
    width = 0.35

    bars1 = ax1.bar(
        [i - width / 2 for i in x],
        on_disk_counts,
        width,
        label="on disk",
        alpha=0.8,
        color="green",
    )
    bars2 = ax1.bar(
        [i + width / 2 for i in x],
        not_on_disk_counts,
        width,
        label="not on disk",
        alpha=0.8,
        color="red",
    )

    ax1.set_xlabel("category")
    ax1.set_ylabel("number of exams")
    ax1.set_title(f"download status by category ({chip_status} patients)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(categories)
    ax1.legend()
    ax1.yaxis.set_major_locator(plt.MaxNLocator(integer=True))

    # add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax1.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height + 0.01 * max(max(on_disk_counts), max(not_on_disk_counts)),
                    f"{int(height):,}",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                )

    # Panel 2: Modality breakdown of downloaded exams
    downloaded_data = all_metadata[all_metadata["is_on_disk"].fillna(False)].copy()
    if len(downloaded_data) > 0:
        modality_counts = downloaded_data["base_modality"].value_counts()
        ax2.bar(
            range(len(modality_counts)),
            modality_counts.values,
            alpha=0.8,
            color="steelblue",
        )
        ax2.set_xlabel("modality")
        ax2.set_ylabel("number of downloaded exams")
        ax2.set_title("downloaded exams by modality")
        ax2.set_xticks(range(len(modality_counts)))
        ax2.set_xticklabels(modality_counts.index, rotation=45)
        ax2.yaxis.set_major_locator(plt.MaxNLocator(integer=True))

        # add value labels
        for i, v in enumerate(modality_counts.values):
            ax2.text(
                i,
                v + 0.01 * max(modality_counts.values),
                f"{v:,}",
                ha="center",
                va="bottom",
                fontsize=9,
            )
    else:
        ax2.text(
            0.5,
            0.5,
            "no downloaded data",
            ha="center",
            va="center",
            transform=ax2.transAxes,
        )
        ax2.set_title("downloaded exams by modality")

    # Panel 3: Download rate by modality
    all_data_combined = pd.concat(
        [case_screening, case_before_dx, controls], ignore_index=True
    )
    if len(all_data_combined) > 0:
        modality_total = all_data_combined["base_modality"].value_counts()
        modality_downloaded = all_data_combined[
            all_data_combined["is_on_disk"].fillna(False)
        ]["base_modality"].value_counts()

        # calculate download rates
        download_rates = []
        modalities = []
        for modality in modality_total.index[:10]:  # top 10 modalities
            total = modality_total[modality]
            downloaded = modality_downloaded.get(modality, 0)
            rate = (downloaded / total) * 100 if total > 0 else 0
            download_rates.append(rate)
            modalities.append(f"{modality}\n({downloaded}/{total})")

        bars = ax3.bar(
            range(len(download_rates)), download_rates, alpha=0.8, color="orange"
        )
        ax3.set_xlabel("modality")
        ax3.set_ylabel("download rate (%)")
        ax3.set_title("download rate by modality")
        ax3.set_xticks(range(len(modalities)))
        ax3.set_xticklabels(modalities, rotation=45, ha="right")
        ax3.yaxis.set_major_locator(plt.MaxNLocator(integer=True))

        # add value labels
        for i, v in enumerate(download_rates):
            ax3.text(i, v + 0.5, f"{v:.1f}%", ha="center", va="bottom", fontsize=9)
    else:
        ax3.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax3.transAxes)
        ax3.set_title("download rate by modality")

    # Panel 4: Export vs Download status
    exported_data = all_metadata[all_metadata["Exported On"].notna()].copy()
    if len(exported_data) > 0:
        exported_on_disk = exported_data["is_on_disk"].fillna(False).sum()
        exported_not_on_disk = len(exported_data) - exported_on_disk

        labels = ["exported & on disk", "exported & not on disk"]
        sizes = [exported_on_disk, exported_not_on_disk]
        colors = ["lightgreen", "lightcoral"]

        ax4.pie(sizes, labels=labels, colors=colors, autopct="%1.1f%%", startangle=90)
        ax4.set_title(
            f"export vs download status\n(total exported: {len(exported_data):,})"
        )
    else:
        ax4.text(
            0.5,
            0.5,
            "no exported data",
            ha="center",
            va="center",
            transform=ax4.transAxes,
        )
        ax4.set_title("export vs download status")

    plt.tight_layout()
    plt.savefig(
        plots_dir / f"comprehensive_download_status_{chip_status}.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


def create_comprehensive_export_analysis_plot(data, plots_dir):
    """create comprehensive export analysis with cumulative progress and velocity"""

    # filter for records with exported_on dates
    exported_data = data[data["Exported On"].notna()].copy()

    if len(exported_data) == 0:
        print("no records with 'Exported On' dates found")
        return

    print(f"records with 'Exported On' dates: {len(exported_data):,}")

    # create 2x2 subplot figure
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))

    # Panel 1: Cumulative Export Progress Over Time
    exported_data_sorted = exported_data.sort_values("Exported On")
    exported_data_sorted["cumulative_count"] = range(1, len(exported_data_sorted) + 1)

    ax1.plot(
        exported_data_sorted["Exported On"],
        exported_data_sorted["cumulative_count"],
        linewidth=2,
        color="steelblue",
    )
    ax1.set_title("cumulative export progress over time")
    ax1.set_xlabel("export date")
    ax1.set_ylabel("cumulative number of exams exported")
    ax1.tick_params(axis="x", rotation=45)
    ax1.grid(True, alpha=0.3)

    # add trend line for recent period (last 30 days of exports)
    if len(exported_data_sorted) > 30:
        recent_data = exported_data_sorted.tail(30)
        if len(recent_data) > 1:
            # calculate daily export rate from recent period
            days_span = (
                recent_data["Exported On"].max() - recent_data["Exported On"].min()
            ).days
            if days_span > 0:
                recent_rate = len(recent_data) / days_span
                ax1.text(
                    0.02,
                    0.98,
                    f"recent rate: {recent_rate:.1f} exams/day",
                    transform=ax1.transAxes,
                    verticalalignment="top",
                    bbox={
                        "boxstyle": "round,pad=0.3",
                        "facecolor": "lightblue",
                        "alpha": 0.7,
                    },
                )

    # Panel 2: Export Velocity (Daily Export Counts)
    exported_data["export_date"] = exported_data["Exported On"].dt.date
    daily_exports = exported_data.groupby("export_date").size()

    ax2.bar(
        daily_exports.index,
        daily_exports.values,
        alpha=0.7,
        edgecolor="black",
        width=0.8,
    )
    ax2.set_title("daily export velocity")
    ax2.set_xlabel("export date")
    ax2.set_ylabel("number of exams exported per day")
    ax2.tick_params(axis="x", rotation=45)
    ax2.yaxis.set_major_locator(plt.MaxNLocator(integer=True))

    # add moving average if enough data points
    if len(daily_exports) >= 7:
        moving_avg = daily_exports.rolling(window=7, center=True).mean()
        ax2.plot(
            moving_avg.index,
            moving_avg.values,
            color="red",
            linewidth=2,
            label="7-day moving average",
        )
        ax2.legend()

    # add summary statistics
    if len(daily_exports) > 0:
        avg_daily = daily_exports.mean()
        max_daily = daily_exports.max()
        ax2.text(
            0.02,
            0.98,
            f"avg: {avg_daily:.1f}/day\nmax: {max_daily}/day",
            transform=ax2.transAxes,
            verticalalignment="top",
            bbox={
                "boxstyle": "round,pad=0.3",
                "facecolor": "lightgreen",
                "alpha": 0.7,
            },
        )

    # Panel 3: Export Distribution by Month
    exported_data["export_month"] = exported_data["Exported On"].dt.to_period("M")
    monthly_exports = exported_data.groupby("export_month").size()

    ax3.bar(
        range(len(monthly_exports)),
        monthly_exports.values,
        alpha=0.7,
        edgecolor="black",
    )
    ax3.set_title("monthly export distribution")
    ax3.set_xlabel("month")
    ax3.set_ylabel("number of exams exported")
    ax3.set_xticks(range(len(monthly_exports)))
    ax3.set_xticklabels([str(m) for m in monthly_exports.index], rotation=45)
    ax3.yaxis.set_major_locator(plt.MaxNLocator(integer=True))

    # add value labels on bars
    for i, v in enumerate(monthly_exports.values):
        ax3.text(
            i,
            v + 0.01 * max(monthly_exports.values),
            f"{v:,}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    # Panel 4: Export Success Rate Analysis (exported vs downloaded)
    if "is_on_disk" in exported_data.columns:
        exported_on_disk = exported_data["is_on_disk"].fillna(False).sum()
        exported_not_on_disk = len(exported_data) - exported_on_disk

        labels = ["exported & downloaded", "exported & not downloaded"]
        sizes = [exported_on_disk, exported_not_on_disk]
        colors = ["lightgreen", "lightcoral"]

        # only create pie chart if there's data
        if sum(sizes) > 0:
            wedges, texts, autotexts = ax4.pie(
                sizes, labels=labels, colors=colors, autopct="%1.1f%%", startangle=90
            )
            ax4.set_title(
                f"export vs download success\n(total exported: {len(exported_data):,})"
            )

            # add absolute numbers to the pie chart
            for autotext, size in zip(autotexts, sizes):
                autotext.set_text(f"{autotext.get_text()}\n({size:,})")
        else:
            ax4.text(
                0.5,
                0.5,
                "no export data",
                ha="center",
                va="center",
                transform=ax4.transAxes,
            )
            ax4.set_title("export vs download success")
    else:
        ax4.text(
            0.5,
            0.5,
            "download status not available",
            ha="center",
            va="center",
            transform=ax4.transAxes,
        )
        ax4.set_title("export vs download success")

    plt.tight_layout()
    plt.savefig(
        plots_dir / "comprehensive_export_analysis.png", dpi=300, bbox_inches="tight"
    )
    plt.close()

    # print summary statistics
    print("\n=== EXPORT ANALYSIS SUMMARY ===")
    if len(daily_exports) > 0:
        print(
            f"export period: {daily_exports.index.min()} to {daily_exports.index.max()}"
        )
        print(f"total days with exports: {len(daily_exports):,}")
        print(f"average exports per day: {daily_exports.mean():.1f}")
        print(f"maximum exports in a day: {daily_exports.max():,}")
        print(f"total exams exported: {len(exported_data):,}")

        # calculate remaining work if we have study data
        total_studies = len(data)
        remaining = total_studies - len(exported_data)
        if remaining > 0 and daily_exports.mean() > 0:
            days_remaining = remaining / daily_exports.mean()
            print(f"estimated days to complete at current rate: {days_remaining:.1f}")

    if "is_on_disk" in exported_data.columns:
        success_rate = (
            exported_data["is_on_disk"].fillna(False).sum() / len(exported_data)
        ) * 100
        print(f"export-to-download success rate: {success_rate:.1f}%")


# find and save missing MRNs
missing_mrns = chimec_patients[~chimec_patients["MRN"].isin(key["MRN"])]["MRN"].tolist()
with open("missing_mrns.txt", "w") as f:
    for mrn in missing_mrns:
        f.write(f"{mrn}\n")
print(f"missing MRNs (not in key): {len(missing_mrns):,} (saved to missing_mrns.txt)")

# connect study_id in metadata to chip column in chimec_patients through key table
# step 1: merge chimec_patients with key table to get AnonymousID for each MRN
chimec_with_study_id = chimec_patients.merge(
    key[["MRN", "AnonymousID"]], on="MRN", how="left"
)

# rename status column to case_or_control
chimec_with_study_id = chimec_with_study_id.rename(
    columns={"status": "case_or_control"}
)

print("\n=== DATA LINKING SUMMARY ===")
print(f"ChiMEC patients after key merge: {chimec_with_study_id.shape[0]:,}")
print(f"patients with chip data: {chimec_with_study_id['chip'].notna().sum():,}")
print(
    f"patients with chip + study_id: {(chimec_with_study_id['chip'].notna() & chimec_with_study_id['AnonymousID'].notna()).sum():,}"
)

# identify chip patients with selected modality records
chip_patients_study_ids = chimec_with_study_id[chimec_with_study_id["chip"].notna()][
    "AnonymousID"
].dropna()
selected_modality_records = metadata[metadata["base_modality"] == SELECTED_MODALITY]
chip_patients_with_selected_modality = chip_patients_study_ids[
    chip_patients_study_ids.isin(selected_modality_records["study_id"])
]

print(
    f"chip patients with {SELECTED_MODALITY} records: {len(chip_patients_with_selected_modality):,} / {len(chip_patients_study_ids):,} ({len(chip_patients_with_selected_modality) / len(chip_patients_study_ids) * 100:.1f}%)"
)

# total cases vs controls breakdown (all ChiMEC patients)
total_cases_count = chimec_with_study_id[
    chimec_with_study_id["case_or_control"] == "Case"
].shape[0]
total_controls_count = chimec_with_study_id[
    chimec_with_study_id["case_or_control"] == "Control"
].shape[0]

# genotyped cases vs controls breakdown
chip_patients_with_status = chimec_with_study_id[chimec_with_study_id["chip"].notna()]
genotyped_cases_count = chip_patients_with_status[
    chip_patients_with_status["case_or_control"] == "Case"
].shape[0]
genotyped_controls_count = chip_patients_with_status[
    chip_patients_with_status["case_or_control"] == "Control"
].shape[0]

print("\n=== PATIENT SUMMARY ===")
print(
    f"total ChiMEC patients: {len(chimec_patients):,} (cases: {total_cases_count:,}, controls: {total_controls_count:,})"
)
print(
    f"genotyped patients: {len(chimec_with_study_id[chimec_with_study_id['chip'].notna()]):,} (cases: {genotyped_cases_count:,}, controls: {genotyped_controls_count:,})"
)
print(f"patients with imaging: {metadata['study_id'].nunique():,}")
print(f"genotyped with imaging: {len(chip_patients_study_ids):,}")
print(
    f"genotyped with {SELECTED_MODALITY} imaging: {len(chip_patients_with_selected_modality):,}"
)

# time-to-diagnosis analysis for cases
print("\n=== TIME-TO-DIAGNOSIS ANALYSIS ===")
cases_with_study_id = chimec_with_study_id[
    chimec_with_study_id["case_or_control"] == "Case"
].copy()

# convert DatedxIndex to datetime
cases_with_study_id["DatedxIndex"] = pd.to_datetime(
    cases_with_study_id["DatedxIndex"], format="%d/%m/%Y"
)

# merge with metadata to get imaging dates
cases_imaging = cases_with_study_id.merge(
    metadata, left_on="AnonymousID", right_on="study_id", how="inner"
)
cases_imaging = cases_imaging[cases_imaging["base_modality"] == SELECTED_MODALITY]

# ensure Study DateTime is datetime type
cases_imaging["Study DateTime"] = pd.to_datetime(cases_imaging["Study DateTime"])

# calculate time-to-diagnosis (days from imaging to diagnosis)
cases_imaging["time_to_diagnosis"] = (
    cases_imaging["DatedxIndex"] - cases_imaging["Study DateTime"]
).dt.days

# convert days to years (approximate, 365.25 days per year)
cases_imaging["time_to_diagnosis_years"] = cases_imaging["time_to_diagnosis"] / 365.25

print(f"total case imaging records: {len(cases_imaging):,}")
print(
    f"records with valid time-to-diagnosis: {cases_imaging['time_to_diagnosis'].notna().sum():,}"
)
print(
    f"range of time-to-diagnosis: {cases_imaging['time_to_diagnosis'].min():,} to {cases_imaging['time_to_diagnosis'].max():,} days"
)

# create histogram including positive times (mammos after dx), x axis in years
plt.figure(figsize=(6, 3))
time_to_dx_years = cases_imaging["time_to_diagnosis_years"].dropna()
plt.hist(time_to_dx_years, bins=50, alpha=0.7, edgecolor="black")
plt.title(
    "cases\n(positive = imaging before diagnosis, negative = imaging after diagnosis)"
)
plt.xlabel("time to diagnosis (years)")
plt.ylabel("number of exams")
plt.axvline(x=0, color="red", linestyle="--", alpha=0.7, label="diagnosis date")
plt.legend()
plt.savefig(plots_dir / "time_to_diagnosis_years.png", dpi=300, bbox_inches="tight")
plt.close()

# summary stats (in years)
print("\ntime-to-diagnosis statistics (years):")
print(f"mean: {time_to_dx_years.mean():.2f} years")
print(f"median: {time_to_dx_years.median():.2f} years")
print(f"exams after diagnosis (negative values): {(time_to_dx_years < 0).sum():,}")
print(f"exams before diagnosis (positive values): {(time_to_dx_years > 0).sum():,}")

# check for extreme time-to-diagnosis values
print("\n=== TIME-TO-DIAGNOSIS VALIDATION ===")
extreme_negative = cases_imaging[
    cases_imaging["time_to_diagnosis"] < -3650
]  # >10 years before
extreme_positive = cases_imaging[
    cases_imaging["time_to_diagnosis"] > 3650
]  # >10 years after
print(f"records >10 years after diagnosis: {len(extreme_negative):,}")
print(f"records >10 years before diagnosis: {len(extreme_positive):,}")
print(
    f"time range: {cases_imaging['time_to_diagnosis'].min():.0f} to {cases_imaging['time_to_diagnosis'].max():.0f} days"
)
print(
    f"mean: {cases_imaging['time_to_diagnosis'].mean():.1f} days, median: {cases_imaging['time_to_diagnosis'].median():.1f} days"
)

if len(extreme_negative) > 0:
    min_record = extreme_negative.loc[extreme_negative["time_to_diagnosis"].idxmin()]
    print(
        f"most extreme case: {min_record['time_to_diagnosis']:.0f} days ({min_record['time_to_diagnosis'] / 365.25:.1f} years) after diagnosis"
    )

print("\n=== BASE MODALITY ANALYSIS ===")
print("unique base modalities:")
print(metadata["base_modality"].value_counts())
print("\nexamples of modality -> base_modality mapping:")
sample_mapping = metadata[["Modality", "base_modality"]].drop_duplicates().head(20)
print(sample_mapping)

BASE_MODALITY_LABELS = {
    "CR": "computed radiography",
    "DX": "digital radiography",
    "MG": "mammography",
    "US": "ultrasound",
    "CT": "ct",
    "MR": "mri",
    "NM": "nuclear medicine",
    "PT": "pet",
    "XA": "x-ray angiography",
    "RF": "fluoroscopy",
    "ES": "endoscopy",
    "XC": "external camera",
    "PX": "panoramic x-ray",
    "RG": "radiographic imaging",
}

print("\n=== MODALITY DISTRIBUTION ===")
total_patients = metadata["study_id"].nunique()
patient_counts = (
    metadata.groupby("base_modality")["study_id"].nunique().sort_values(ascending=False)
)
print(
    f"patients with multi-modality imaging: sum ({patient_counts.sum():,}) > total ({total_patients:,})"
)
create_modality_distribution_plot(metadata, "all", plots_dir)

# selected modality analysis
print(f"\n=== {SELECTED_MODALITY} ANALYSIS ===")
# filter for only selected modality scans
selected_modality_scans = metadata[
    metadata["base_modality"] == SELECTED_MODALITY
].copy()

# additional filtering for MR modality - only include scans with "BREAST" in description
if SELECTED_MODALITY == "MR":
    breast_filter = selected_modality_scans["StudyDescription"].str.contains(
        "BREAST", case=False, na=False
    )
    selected_modality_scans = selected_modality_scans[breast_filter].copy()
    print("filtered MR scans to only include those with 'BREAST' in description")

selected_modality_per_patient = selected_modality_scans["study_id"].value_counts()
print(
    f"total {SELECTED_MODALITY} scans: {len(selected_modality_scans):,} (patients: {selected_modality_scans['study_id'].nunique():,})"
)
print(
    f"scans per patient - range: {selected_modality_per_patient.min()}-{selected_modality_per_patient.max()}, mean: {selected_modality_per_patient.mean():.1f}, median: {selected_modality_per_patient.median():.1f}"
)

# create selected modality scans per patient plot for all patients
create_mg_mammograms_per_patient_plot(
    selected_modality_per_patient,
    chimec_with_study_id,
    selected_modality_scans,
    "all",
    plots_dir,
    SELECTED_MODALITY,
)

print(f"patients with most scans: {selected_modality_per_patient.head(3).to_dict()}")

# screening analysis
print(f"\n=== SCREENING {SELECTED_MODALITY} ANALYSIS ===")
# first merge selected modality scans with patient data to get DatedxIndex
selected_modality_with_patient_data = selected_modality_scans.merge(
    chimec_with_study_id[["AnonymousID", "DatedxIndex", "case_or_control", "chip"]],
    left_on="study_id",
    right_on="AnonymousID",
    how="inner",
)

print(
    f"{SELECTED_MODALITY} scans with patient data: {len(selected_modality_with_patient_data)}"
)

# CUT FLOW ANALYSIS - trace where scans are filtered out
print(f"\n=== {SELECTED_MODALITY} CUT FLOW ANALYSIS ===")
print(f"Starting with: {len(selected_modality_scans):,} {SELECTED_MODALITY} scans")
print(
    f"After merge with patient data: {len(selected_modality_with_patient_data):,} scans"
)

# genotyping coverage in imaging population
total_imaging_patients = selected_modality_with_patient_data["study_id"].nunique()
genotyped_imaging = selected_modality_with_patient_data[
    selected_modality_with_patient_data["chip"].notna()
]["study_id"].nunique()
print(
    f"ChiMEC patients with {SELECTED_MODALITY} imaging: {total_imaging_patients:,} (genotyped: {genotyped_imaging:,}, {genotyped_imaging / total_imaging_patients * 100:.1f}%)"
)
if genotyped_imaging == total_imaging_patients:
    print(
        "all ChiMEC patients with imaging are genotyped (explains identical 'all' vs 'genotyped' plots)"
    )

# convert dates to datetime
selected_modality_with_patient_data["Study DateTime"] = pd.to_datetime(
    selected_modality_with_patient_data["Study DateTime"]
)
selected_modality_with_patient_data["DatedxIndex"] = pd.to_datetime(
    selected_modality_with_patient_data["DatedxIndex"], format="%d/%m/%Y"
)

# find and print 20 earliest exams
print("\n" + "=" * 100)
print("=" * 100)
print(f"20 EARLIEST {SELECTED_MODALITY} EXAMS")
print("=" * 100)
print("=" * 100 + "\n")

earliest_exams = (
    selected_modality_with_patient_data.sort_values("Study DateTime").head(20).copy()
)

# select relevant columns for display
display_cols = [
    "study_id",
    "Accession",
    "Study DateTime",
    "StudyDescription",
    "Modality",
    "case_or_control",
    "DatedxIndex",
    "is_on_disk",
    "is_exported",
]
# only include columns that exist
display_cols = [col for col in display_cols if col in earliest_exams.columns]

for idx, (_, row) in enumerate(earliest_exams.iterrows(), 1):
    print(f"\n--- Exam #{idx} ---")
    for col in display_cols:
        value = row[col]
        if pd.isna(value):
            value = "N/A"
        elif isinstance(value, pd.Timestamp):
            value = value.strftime("%Y-%m-%d %H:%M:%S")
        print(f"  {col:20s}: {value}")
    print()

print("=" * 100)
print("=" * 100 + "\n")

# calculate time difference in days (positive = scan before diagnosis)
selected_modality_with_patient_data["days_to_dx"] = (
    selected_modality_with_patient_data["DatedxIndex"]
    - selected_modality_with_patient_data["Study DateTime"]
).dt.days

# separate cases and controls for screening analysis
cases_selected_modality = selected_modality_with_patient_data[
    selected_modality_with_patient_data["case_or_control"] == "Case"
].copy()
controls_selected_modality = selected_modality_with_patient_data[
    selected_modality_with_patient_data["case_or_control"] == "Control"
].copy()

print("After case/control separation:")
print(f"  - Case scans: {len(cases_selected_modality):,}")
print(f"  - Control scans: {len(controls_selected_modality):,}")
print(
    f"  - Other/Unknown status: {len(selected_modality_with_patient_data) - len(cases_selected_modality) - len(controls_selected_modality):,}"
)

# for cases: filter for screening scans (> 3 months = 90 days before diagnosis)
case_screening_scans = cases_selected_modality[
    cases_selected_modality["days_to_dx"] > 90
].copy()

print("After case screening filter (>90 days before dx):")
print(f"  - Case screening scans: {len(case_screening_scans):,}")
print(
    f"  - Case scans excluded (≤90 days before dx): {len(cases_selected_modality) - len(case_screening_scans):,}"
)

# for cases: filter for scans anytime before and including diagnosis month
# convert to year-month for comparison
cases_selected_modality["scan_ym"] = cases_selected_modality[
    "Study DateTime"
].dt.to_period("M")
cases_selected_modality["diagnosis_ym"] = cases_selected_modality[
    "DatedxIndex"
].dt.to_period("M")
case_before_dx_month_scans = cases_selected_modality[
    cases_selected_modality["scan_ym"] <= cases_selected_modality["diagnosis_ym"]
].copy()

print("After case before/including dx month filter:")
print(f"  - Case before/including dx month scans: {len(case_before_dx_month_scans):,}")
print(
    f"  - Case scans excluded (after dx month): {len(cases_selected_modality) - len(case_before_dx_month_scans):,}"
)

# for controls: all scans are considered (no diagnosis date filter)
control_screening_scans = controls_selected_modality.copy()

print(f"Control scans (no filtering): {len(control_screening_scans):,}")

# SUMMARY OF WHERE SCANS WENT
total_accounted = (
    len(case_screening_scans)
    + len(case_before_dx_month_scans)
    + len(control_screening_scans)
)
print("\nSUMMARY:")
print(f"  - Case screening scans: {len(case_screening_scans):,}")
print(f"  - Case before/including dx month scans: {len(case_before_dx_month_scans):,}")
print(f"  - Control scans: {len(control_screening_scans):,}")
print(f"  - Total accounted for: {total_accounted:,}")
print(
    f"  - Missing from original {len(selected_modality_with_patient_data):,}: {len(selected_modality_with_patient_data) - total_accounted:,}"
)

print(
    f"case screening scans (>90 days before dx): {len(case_screening_scans):,} ({case_screening_scans['study_id'].nunique() if len(case_screening_scans) > 0 else 0:,} patients)"
)

# check genotyping coverage in screening cases
if len(case_screening_scans) > 0:
    screening_patients_genotyped = case_screening_scans[
        case_screening_scans["chip"].notna()
    ]["study_id"].nunique()
    total_screening_patients = case_screening_scans["study_id"].nunique()
    print(
        f"  - screening case patients genotyped: {screening_patients_genotyped:,} / {total_screening_patients:,} ({screening_patients_genotyped / total_screening_patients * 100:.1f}%)"
    )

    # identify any non-genotyped screening patients
    non_genotyped_screening = case_screening_scans[case_screening_scans["chip"].isna()][
        "study_id"
    ].unique()
    if len(non_genotyped_screening) > 0:
        print(
            f"  - non-genotyped screening case patients: {list(non_genotyped_screening)}"
        )
    else:
        print("  - all screening case patients are genotyped")
print(
    f"case scans (before/including dx month): {len(case_before_dx_month_scans):,} ({case_before_dx_month_scans['study_id'].nunique() if len(case_before_dx_month_scans) > 0 else 0:,} patients)"
)
print(
    f"control scans (all): {len(control_screening_scans):,} ({control_screening_scans['study_id'].nunique() if len(control_screening_scans) > 0 else 0:,} patients)"
)

# count screening scans per patient for cases
case_screening_per_patient = (
    case_screening_scans["study_id"].value_counts()
    if len(case_screening_scans) > 0
    else pd.Series(dtype=int)
)

# count scans before/including diagnosis month per patient for cases
case_before_dx_month_per_patient = (
    case_before_dx_month_scans["study_id"].value_counts()
    if len(case_before_dx_month_scans) > 0
    else pd.Series(dtype=int)
)

# count all scans per patient for controls
control_screening_per_patient = (
    control_screening_scans["study_id"].value_counts()
    if len(control_screening_scans) > 0
    else pd.Series(dtype=int)
)

# scans per patient summary
if len(case_screening_per_patient) > 0:
    print(
        f"case screening per patient - range: {case_screening_per_patient.min()}-{case_screening_per_patient.max()}, mean: {case_screening_per_patient.mean():.1f}"
    )
if len(case_before_dx_month_per_patient) > 0:
    print(
        f"case before/including dx month per patient - range: {case_before_dx_month_per_patient.min()}-{case_before_dx_month_per_patient.max()}, mean: {case_before_dx_month_per_patient.mean():.1f}"
    )
if len(control_screening_per_patient) > 0:
    print(
        f"control per patient - range: {control_screening_per_patient.min()}-{control_screening_per_patient.max()}, mean: {control_screening_per_patient.mean():.1f}"
    )

# create screening scans plot for all patients
create_screening_mammograms_plot(
    case_screening_per_patient,
    case_before_dx_month_per_patient,
    control_screening_per_patient,
    case_screening_scans,
    case_before_dx_month_scans,
    control_screening_scans,
    "all",
    plots_dir,
    SELECTED_MODALITY,
)


# histogram of time to DatedxIndex for cases and time from first scan for controls
print("\n=== TIME ANALYSIS ===")

# separate cases and controls
cases_selected_modality_time = selected_modality_with_patient_data[
    selected_modality_with_patient_data["case_or_control"] == "Case"
].copy()
controls_selected_modality_time = selected_modality_with_patient_data[
    selected_modality_with_patient_data["case_or_control"] == "Control"
].copy()

# for controls, calculate time from first scan
if len(controls_selected_modality_time) > 0:
    controls_selected_modality_time = controls_selected_modality_time.sort_values(
        "Study DateTime"
    )
    first_scan_dates = controls_selected_modality_time.groupby("study_id")[
        "Study DateTime"
    ].first()
    controls_selected_modality_time = controls_selected_modality_time.merge(
        first_scan_dates.rename("first_scan_date"),
        left_on="study_id",
        right_index=True,
    )
    controls_selected_modality_time["days_from_first_scan"] = (
        controls_selected_modality_time["Study DateTime"]
        - controls_selected_modality_time["first_scan_date"]
    ).dt.days

# time analysis summary
if len(cases_selected_modality_time) > 0:
    case_time_to_dx = cases_selected_modality_time["days_to_dx"].dropna()
    print(
        f"case {SELECTED_MODALITY} scans: {len(case_time_to_dx):,} (before dx: {(case_time_to_dx > 0).sum():,}, after: {(case_time_to_dx < 0).sum():,}, on dx day: {(case_time_to_dx == 0).sum():,})"
    )
    print(
        f"case time to dx - range: {case_time_to_dx.min():.0f} to {case_time_to_dx.max():.0f} days, mean: {case_time_to_dx.mean():.0f}, median: {case_time_to_dx.median():.0f}"
    )

if len(controls_selected_modality_time) > 0:
    control_time_from_first = controls_selected_modality_time[
        "days_from_first_scan"
    ].dropna()
    print(
        f"control {SELECTED_MODALITY} scans: {len(control_time_from_first):,} with time from first scan"
    )
    print(
        f"control time from first - range: {control_time_from_first.min():.0f} to {control_time_from_first.max():.0f} days, mean: {control_time_from_first.mean():.0f}, median: {control_time_from_first.median():.0f}"
    )

# create time analysis plot for all patients
create_time_analysis_plot(
    cases_selected_modality_time,
    controls_selected_modality_time,
    "all",
    plots_dir,
    SELECTED_MODALITY,
)

# time period breakdown
if len(cases_selected_modality_time) > 0:
    case_time_to_dx = cases_selected_modality_time["days_to_dx"].dropna()
    screening = (case_time_to_dx > 90).sum()
    diagnostic = ((case_time_to_dx >= 0) & (case_time_to_dx <= 90)).sum()
    after = (case_time_to_dx < 0).sum()
    print(
        f"case time periods - screening (>90d): {screening:,}, diagnostic (0-90d): {diagnostic:,}, after dx: {after:,}"
    )

if len(controls_selected_modality_time) > 0:
    control_time_from_first = controls_selected_modality_time[
        "days_from_first_scan"
    ].dropna()
    first = (control_time_from_first == 0).sum()
    followup = (control_time_from_first > 0).sum()
    within_year = (
        (control_time_from_first >= 0) & (control_time_from_first <= 365)
    ).sum()
    print(
        f"control time periods - first: {first:,}, follow-up: {followup:,}, within 1yr: {within_year:,}"
    )

# create genotyped-only versions of all plots
print("\n=== CREATING GENOTYPED-ONLY VERSIONS ===")

# filter for only genotyped patients
genotyped_patients = filter_patients_by_chip_status(chimec_with_study_id, "genotyped")
genotyped_metadata = metadata[
    metadata["study_id"].isin(genotyped_patients["AnonymousID"])
]

print(
    f"genotyped patients: {len(genotyped_patients):,} (metadata records: {len(genotyped_metadata):,})"
)

# create genotyped-only modality distribution plot
create_modality_distribution_plot(genotyped_metadata, "genotyped", plots_dir)

# create genotyped-only selected modality scans per patient plot
genotyped_selected_modality_scans = genotyped_metadata[
    genotyped_metadata["base_modality"] == SELECTED_MODALITY
].copy()

# additional filtering for MR modality - only include scans with "BREAST" in description
if SELECTED_MODALITY == "MR":
    breast_filter = genotyped_selected_modality_scans["StudyDescription"].str.contains(
        "BREAST", case=False, na=False
    )
    genotyped_selected_modality_scans = genotyped_selected_modality_scans[
        breast_filter
    ].copy()
genotyped_selected_modality_per_patient = genotyped_selected_modality_scans[
    "study_id"
].value_counts()
create_mg_mammograms_per_patient_plot(
    genotyped_selected_modality_per_patient,
    genotyped_patients,
    genotyped_selected_modality_scans,
    "genotyped",
    plots_dir,
    SELECTED_MODALITY,
)

# create genotyped-only screening analysis
genotyped_selected_modality_with_patient_data = genotyped_selected_modality_scans.merge(
    genotyped_patients[["AnonymousID", "DatedxIndex", "case_or_control"]],
    left_on="study_id",
    right_on="AnonymousID",
    how="inner",
)

print(
    f"genotyped {SELECTED_MODALITY} scans with patient data: {len(genotyped_selected_modality_with_patient_data):,}"
)

# convert dates to datetime
genotyped_selected_modality_with_patient_data["Study DateTime"] = pd.to_datetime(
    genotyped_selected_modality_with_patient_data["Study DateTime"]
)
genotyped_selected_modality_with_patient_data["DatedxIndex"] = pd.to_datetime(
    genotyped_selected_modality_with_patient_data["DatedxIndex"], format="%d/%m/%Y"
)

# calculate time difference in days (positive = scan before diagnosis)
genotyped_selected_modality_with_patient_data["days_to_dx"] = (
    genotyped_selected_modality_with_patient_data["DatedxIndex"]
    - genotyped_selected_modality_with_patient_data["Study DateTime"]
).dt.days

# separate cases and controls for screening analysis
genotyped_cases_selected_modality = genotyped_selected_modality_with_patient_data[
    genotyped_selected_modality_with_patient_data["case_or_control"] == "Case"
].copy()
genotyped_controls_selected_modality = genotyped_selected_modality_with_patient_data[
    genotyped_selected_modality_with_patient_data["case_or_control"] == "Control"
].copy()

# for cases: filter for screening scans (> 3 months = 90 days before diagnosis)
genotyped_case_screening_scans = genotyped_cases_selected_modality[
    genotyped_cases_selected_modality["days_to_dx"] > 90
].copy()

# for cases: filter for scans anytime before and including diagnosis month
genotyped_cases_selected_modality["scan_ym"] = genotyped_cases_selected_modality[
    "Study DateTime"
].dt.to_period("M")
genotyped_cases_selected_modality["diagnosis_ym"] = genotyped_cases_selected_modality[
    "DatedxIndex"
].dt.to_period("M")
genotyped_case_before_dx_month_scans = genotyped_cases_selected_modality[
    genotyped_cases_selected_modality["scan_ym"]
    <= genotyped_cases_selected_modality["diagnosis_ym"]
].copy()

# for controls: all scans are considered (no diagnosis date filter)
genotyped_control_screening_scans = genotyped_controls_selected_modality.copy()

# count screening scans per patient for cases
genotyped_case_screening_per_patient = (
    genotyped_case_screening_scans["study_id"].value_counts()
    if len(genotyped_case_screening_scans) > 0
    else pd.Series(dtype=int)
)

# count scans before/including diagnosis month per patient for cases
genotyped_case_before_dx_month_per_patient = (
    genotyped_case_before_dx_month_scans["study_id"].value_counts()
    if len(genotyped_case_before_dx_month_scans) > 0
    else pd.Series(dtype=int)
)

# count all scans per patient for controls
genotyped_control_screening_per_patient = (
    genotyped_control_screening_scans["study_id"].value_counts()
    if len(genotyped_control_screening_scans) > 0
    else pd.Series(dtype=int)
)

# create genotyped-only screening scans plot
create_screening_mammograms_plot(
    genotyped_case_screening_per_patient,
    genotyped_case_before_dx_month_per_patient,
    genotyped_control_screening_per_patient,
    genotyped_case_screening_scans,
    genotyped_case_before_dx_month_scans,
    genotyped_control_screening_scans,
    "genotyped",
    plots_dir,
    SELECTED_MODALITY,
)

# create genotyped-only time analysis
genotyped_cases_selected_modality_for_time = (
    genotyped_selected_modality_with_patient_data[
        genotyped_selected_modality_with_patient_data["case_or_control"] == "Case"
    ].copy()
)
genotyped_controls_selected_modality_for_time = (
    genotyped_selected_modality_with_patient_data[
        genotyped_selected_modality_with_patient_data["case_or_control"] == "Control"
    ].copy()
)

# for controls, calculate time from first scan
if len(genotyped_controls_selected_modality_for_time) > 0:
    genotyped_controls_selected_modality_for_time = (
        genotyped_controls_selected_modality_for_time.sort_values("Study DateTime")
    )
    genotyped_first_scan_dates = genotyped_controls_selected_modality_for_time.groupby(
        "study_id"
    )["Study DateTime"].first()
    genotyped_controls_selected_modality_for_time = (
        genotyped_controls_selected_modality_for_time.merge(
            genotyped_first_scan_dates.rename("first_scan_date"),
            left_on="study_id",
            right_index=True,
        )
    )
    genotyped_controls_selected_modality_for_time["days_from_first_scan"] = (
        genotyped_controls_selected_modality_for_time["Study DateTime"]
        - genotyped_controls_selected_modality_for_time["first_scan_date"]
    ).dt.days

# create genotyped-only time analysis plot
create_time_analysis_plot(
    genotyped_cases_selected_modality_for_time,
    genotyped_controls_selected_modality_for_time,
    "genotyped",
    plots_dir,
    SELECTED_MODALITY,
)

# create data exploration plots
print("\n=== CREATING DATA EXPLORATION PLOTS ===")

# convert Exported On column to datetime if it exists
if "Exported On" in metadata.columns:
    metadata["Exported On"] = pd.to_datetime(metadata["Exported On"], errors="coerce")

# use the same filtered data as the screening plots for consistency
# this ensures the download status plots match the screening exam plots
cases_data = cases_selected_modality.copy()
controls_data = controls_selected_modality.copy()

# define case categories for exploration (same as screening analysis)
case_screening_scans_exploration = case_screening_scans.copy()
case_before_dx_month_scans_exploration = case_before_dx_month_scans.copy()
control_scans_exploration = control_screening_scans.copy()

# for exported on plot, use all metadata
metadata_with_patient_data = metadata.merge(
    chimec_with_study_id[["AnonymousID", "DatedxIndex", "case_or_control", "chip"]],
    left_on="study_id",
    right_on="AnonymousID",
    how="inner",
)
# convert Exported On column for the export plot
if "Exported On" in metadata_with_patient_data.columns:
    metadata_with_patient_data["Exported On"] = pd.to_datetime(
        metadata_with_patient_data["Exported On"], errors="coerce"
    )

print(
    f"exploration - cases >3 months before dx: {len(case_screening_scans_exploration):,}"
)
print(
    f"exploration - cases before/including dx month: {len(case_before_dx_month_scans_exploration):,}"
)
print(f"exploration - controls (all): {len(control_scans_exploration):,}")

# create comprehensive download status plots
create_comprehensive_download_status_plot(
    case_screening_scans_exploration,
    case_before_dx_month_scans_exploration,
    control_scans_exploration,
    metadata,
    "all",
    plots_dir,
)

# create comprehensive download status plots for genotyped patients only
genotyped_case_screening_exploration = genotyped_case_screening_scans.copy()
genotyped_case_before_dx_exploration = genotyped_case_before_dx_month_scans.copy()
genotyped_controls_exploration = genotyped_control_screening_scans.copy()

# filter metadata to genotyped patients for the comprehensive plot
genotyped_metadata = metadata[
    metadata["study_id"].isin(
        chimec_with_study_id[chimec_with_study_id["chip"].notna()]["AnonymousID"]
    )
]

create_comprehensive_download_status_plot(
    genotyped_case_screening_exploration,
    genotyped_case_before_dx_exploration,
    genotyped_controls_exploration,
    genotyped_metadata,
    "genotyped",
    plots_dir,
)

# create comprehensive export analysis plot
create_comprehensive_export_analysis_plot(metadata_with_patient_data, plots_dir)

# count and report exams with exported_on date but not on disk
print(f"\n=== EXPORT STATUS ANALYSIS ({SELECTED_MODALITY}) ===")
print(f"Total {SELECTED_MODALITY} exams in iBroker: {len(metadata):,}")

# show Status column breakdown by on_disk status
if "Status" in metadata.columns and "is_on_disk" in metadata.columns:
    exported_count = metadata["Status"].notna().sum()
    not_exported_count = metadata["Status"].isna().sum()
    print(f"  - with export status (in Exported table): {exported_count:,}")
    print(f"  - no export status (Available only):      {not_exported_count:,}")
    print()
    print("Breakdown by export status and disk presence:")

    # get unique status values (excluding NaN)
    status_values = metadata["Status"].dropna().unique()

    is_on_disk = metadata["is_on_disk"].fillna(False)

    # find max status length for alignment
    not_exported_label = "(not exported yet)"
    max_status_len = max(
        max((len(str(s)) for s in status_values), default=0),
        len(not_exported_label),
        len("TOTAL"),
    )
    col_w = max(max_status_len + 2, 50)

    print(f"{'Status':<{col_w}} {'on disk':>12} {'not on disk':>12} {'total':>12}")
    print("-" * (col_w + 40))

    for status in sorted(status_values):
        mask = metadata["Status"] == status
        on_disk = (mask & is_on_disk).sum()
        not_on_disk = (mask & ~is_on_disk).sum()
        total = mask.sum()
        print(f"{status:<{col_w}} {on_disk:>12,} {not_on_disk:>12,} {total:>12,}")

    # also show records with no Status (from Available table, not yet exported)
    no_status = metadata["Status"].isna()
    on_disk_no_status = (no_status & is_on_disk).sum()
    not_on_disk_no_status = (no_status & ~is_on_disk).sum()
    total_no_status = no_status.sum()
    print(
        f"{not_exported_label:<{col_w}} {on_disk_no_status:>12,} {not_on_disk_no_status:>12,} {total_no_status:>12,}"
    )
    print("-" * (col_w + 40))

    # totals
    total_on_disk = is_on_disk.sum()
    total_not_on_disk = (~is_on_disk).sum()
    print(
        f"{'TOTAL':<{col_w}} {total_on_disk:>12,} {total_not_on_disk:>12,} {len(metadata):>12,}"
    )
    print()

    # show sample of records with Status but not on disk
    pending = metadata[metadata["Status"].notna() & ~is_on_disk]
    if len(pending) > 0:
        print(f"sample of {len(pending):,} records with Status but not on disk:")
        display_cols = [
            "study_id",
            "Accession",
            "Status",
            "Exported On",
            "StudyDescription",
        ]
        display_cols = [c for c in display_cols if c in pending.columns]
        print(pending[display_cols].head(20).to_string())
        print()

    # specifically show records with "Completed" status but not on disk
    completed_mask = metadata["Status"].str.lower().str.contains("completed", na=False)
    completed_not_on_disk = metadata[completed_mask & ~is_on_disk]
    if len(completed_not_on_disk) > 0:
        print(
            f"\n=== {len(completed_not_on_disk):,} RECORDS WITH 'COMPLETED' STATUS BUT NOT ON DISK ==="
        )
        display_cols = [
            "study_id",
            "Accession",
            "Status",
            "Exported On",
            "StudyDescription",
        ]
        display_cols = [c for c in display_cols if c in completed_not_on_disk.columns]
        print(completed_not_on_disk[display_cols].to_string())
        print()
else:
    if "Status" not in metadata.columns:
        print("Status column not found in metadata")
    if "is_on_disk" not in metadata.columns:
        print("is_on_disk column not found in metadata")

if (
    "Exported On" in metadata_with_patient_data.columns
    and "is_on_disk" in metadata_with_patient_data.columns
):
    exported_not_on_disk = metadata_with_patient_data[
        (metadata_with_patient_data["Exported On"].notna())
        & (~metadata_with_patient_data["is_on_disk"].fillna(False))
    ]
    print(
        f"exams with 'Exported On' date but not on disk: {len(exported_not_on_disk):,}"
    )

    if len(exported_not_on_disk) > 0:
        # breakdown by case/control
        exported_not_on_disk_cases = exported_not_on_disk[
            exported_not_on_disk["case_or_control"] == "Case"
        ]
        exported_not_on_disk_controls = exported_not_on_disk[
            exported_not_on_disk["case_or_control"] == "Control"
        ]

        print(f"  - cases: {len(exported_not_on_disk_cases):,}")
        print(f"  - controls: {len(exported_not_on_disk_controls):,}")

        # breakdown by modality
        print("\nbreakdown by modality:")
        modality_breakdown = exported_not_on_disk["base_modality"].value_counts()
        for modality, count in modality_breakdown.head(10).items():
            print(f"  - {modality}: {count:,}")
else:
    print("'Exported On' or 'is_on_disk' column not found in metadata")

print("\n=== ANALYSIS COMPLETE ===")
print(f"plots saved to: {plots_dir.absolute()}")
print("created 'all' and 'genotyped' versions of all plots")

# dump screening patients if requested
if args.dump_screening_patients:
    print(f"\n=== DUMPING SCREENING PATIENTS FOR {SELECTED_MODALITY} ===")

    # get case patients with screening scans (>90 days before diagnosis)
    screening_patients_data = case_screening_scans.copy()

    if len(screening_patients_data) > 0:
        # merge with chimec_patients to get MRN
        screening_patients_with_mrn = screening_patients_data.merge(
            key[["AnonymousID", "MRN"]],
            left_on="study_id",
            right_on="AnonymousID",
            how="left",
        )

        # create CSV with requested columns
        csv_data = screening_patients_with_mrn[
            ["MRN", "study_id", "DatedxIndex", "StudyDescription", "Study DateTime"]
        ].copy()

        # rename columns for clarity
        csv_data = csv_data.rename(
            columns={
                "study_id": "study_id",
                "DatedxIndex": "DX_date",
                "StudyDescription": "scan_description",
                "Study DateTime": "scan_date",
            }
        )

        # sort by patient MRN then scan date
        csv_data = csv_data.sort_values(["MRN", "scan_date"])

        # save CSV
        csv_filename = f"screening_patients_{SELECTED_MODALITY.lower()}.csv"
        csv_data.to_csv(csv_filename, index=False)

        # summary statistics
        unique_patients = csv_data["MRN"].nunique()
        total_scans = len(csv_data)

        print(
            f"patients with screening {SELECTED_MODALITY} scans (≥3 months before DX): {unique_patients:,}"
        )
        print(f"total screening {SELECTED_MODALITY} scans: {total_scans:,}")
        print(f"average scans per patient: {total_scans / unique_patients:.1f}")
        print(f"CSV saved to: {csv_filename}")

        # bin patients by scan date (every 2 years)
        print(f"\n--- {SELECTED_MODALITY} screening scans by time period ---")
        csv_data["scan_year"] = pd.to_datetime(csv_data["scan_date"]).dt.year
        min_year = csv_data["scan_year"].min()
        max_year = csv_data["scan_year"].max()

        # create 2-year bins
        bin_edges = list(
            range(min_year, max_year + 3, 2)
        )  # +3 to include the last year
        csv_data["year_bin"] = pd.cut(
            csv_data["scan_year"], bins=bin_edges, right=False, include_lowest=True
        )

        # count scans per bin
        scans_per_bin = csv_data["year_bin"].value_counts().sort_index()

        for bin_range, count in scans_per_bin.items():
            print(f"{bin_range}: {count:,} scans")

    else:
        print(
            f"no case patients found with screening {SELECTED_MODALITY} scans (≥3 months before diagnosis)"
        )
        print("CSV file not created")

# comprehensive data sources analysis
print("\n" + "=" * 80)
print("DATA SOURCES SUMMARY")
print("=" * 80)
print("""
This analysis compares three data sources:
1. iBroker metadata (imaging_metadata.csv) - tracks exports, NOT needed for preprocessing
2. Disk fingerprints - actual imaging data available for training
3. Phenotype labels - case/control status and diagnosis dates

Key insight: Preprocessing works directly on disk DICOMs. Historical data on disk but
missing from iBroker CAN be used for training, as long as patients have phenotype labels.
""")

# load phenotype data
phenotype_path = Path(
    "/gpfs/data/phs/groups/Projects/Huo_projects/SPORE/annawoodard/Phenotype_ChiMEC_2025Oct4.csv"
)
mrn_mapping_path = Path("/gpfs/data/huo-lab/Image/ChiMEC/study-16352a.csv")
fingerprint_cache = Path("data/destination_fingerprints.json")

if phenotype_path.exists() and mrn_mapping_path.exists() and fingerprint_cache.exists():
    # load MRN -> AnonymousID mapping
    mrn_map = pd.read_csv(mrn_mapping_path)
    mrn_map["MRN"] = mrn_map["MRN"].astype(str).str.strip().str.zfill(8)
    mrn_to_anon = dict(zip(mrn_map["MRN"], mrn_map["AnonymousID"].astype(str)))

    # load phenotype
    phenotype_df = pd.read_csv(phenotype_path)
    phenotype_df["MRN"] = phenotype_df["MRN"].astype(str).str.strip().str.zfill(8)
    phenotype_df["patient_id"] = phenotype_df["MRN"].map(mrn_to_anon)

    phenotype_patients = set(phenotype_df["patient_id"].dropna().astype(str))
    phenotype_cases = set(
        phenotype_df[
            phenotype_df["CaseControl"].str.lower().str.contains("case", na=False)
        ]["patient_id"]
        .dropna()
        .astype(str)
    )
    phenotype_controls = set(
        phenotype_df[
            phenotype_df["CaseControl"].str.lower().str.contains("control", na=False)
        ]["patient_id"]
        .dropna()
        .astype(str)
    )

    # load disk fingerprints
    with open(fingerprint_cache) as f:
        fp_data = json.load(f)

    disk_patient_dates = defaultdict(set)
    disk_patients = set(fp_data.keys())
    for pid, exams in fp_data.items():
        for exam_name, data in exams.items():
            uid, hashes, study_date, study_time = data
            if study_date and len(study_date) >= 8:
                date_str = f"{study_date[:4]}-{study_date[4:6]}-{study_date[6:8]}"
                disk_patient_dates[pid].add(date_str)

    disk_exam_keys = set()
    for pid, dates in disk_patient_dates.items():
        for d in dates:
            disk_exam_keys.add((pid, d))

    # get iBroker MG data
    ibroker_patient_dates = defaultdict(set)
    ibroker_patients = set(metadata["study_id"].dropna().astype(str).unique())
    metadata["Study DateTime"] = pd.to_datetime(
        metadata["Study DateTime"], errors="coerce"
    )
    for _, row in metadata.iterrows():
        pid = str(row["study_id"]) if pd.notna(row["study_id"]) else None
        dt = row["Study DateTime"]
        if pid and pd.notna(dt):
            ibroker_patient_dates[pid].add(dt.strftime("%Y-%m-%d"))

    ibroker_exam_keys = set()
    for pid, dates in ibroker_patient_dates.items():
        for d in dates:
            ibroker_exam_keys.add((pid, d))

    # calculate overlaps
    disk_only_keys = disk_exam_keys - ibroker_exam_keys
    disk_only_patients = set(k[0] for k in disk_only_keys)
    ibroker_only_keys = ibroker_exam_keys - disk_exam_keys
    matched_keys = disk_exam_keys & ibroker_exam_keys

    print(f"=== iBROKER {SELECTED_MODALITY} METADATA ===")
    print(f"  patients: {len(ibroker_patients):,}")
    print(f"  unique (patient, date) pairs: {len(ibroker_exam_keys):,}")

    print("\n=== DISK (from fingerprints) ===")
    print(f"  patients: {len(disk_patients):,}")
    print(f"  unique (patient, date) pairs: {len(disk_exam_keys):,}")

    print("\n=== PHENOTYPE (labels) ===")
    print(f"  patients with labels: {len(phenotype_patients):,}")
    print(f"    - cases: {len(phenotype_cases):,}")
    print(f"    - controls: {len(phenotype_controls):,}")

    print("\n=== OVERLAP ANALYSIS ===")
    print(
        f"  matched (on both disk and iBroker): {len(matched_keys):,} (patient, date) pairs"
    )
    print(
        f"  disk-only (historical, not in iBroker): {len(disk_only_keys):,} (patient, date) pairs"
    )
    print(
        f"  iBroker-only (need to download): {len(ibroker_only_keys):,} (patient, date) pairs"
    )

    # disk-only with phenotype
    disk_only_with_labels = sum(
        1 for pid, d in disk_only_keys if pid in phenotype_patients
    )
    disk_only_patients_with_labels = disk_only_patients & phenotype_patients

    print("\n=== DISK-ONLY DATA (historical exams not in iBroker) ===")
    print(f"  total disk-only patients: {len(disk_only_patients):,}")
    print(
        f"  disk-only patients WITH phenotype labels: {len(disk_only_patients_with_labels):,}"
    )
    print(f"  disk-only (patient, date) pairs with labels: {disk_only_with_labels:,}")
    print("  → these CAN be used for training!")

    # remaining to download
    ibroker_only_with_labels = sum(
        1 for pid, d in ibroker_only_keys if pid in phenotype_patients
    )

    print("\n=== REMAINING TO DOWNLOAD ===")
    print(
        f"  (patient, date) pairs in iBroker but NOT on disk: {len(ibroker_only_keys):,}"
    )
    print(
        f"  remaining (patient, date) pairs WITH labels: {ibroker_only_with_labels:,}"
    )

    # training data summary
    all_disk_with_labels = sum(
        1 for pid, d in disk_exam_keys if pid in phenotype_patients
    )
    disk_patients_with_labels = disk_patients & phenotype_patients

    all_potential = disk_exam_keys | ibroker_exam_keys
    all_potential_with_labels = sum(
        1 for pid, d in all_potential if pid in phenotype_patients
    )
    all_potential_patients = (disk_patients | ibroker_patients) & phenotype_patients

    print("\n=== TRAINING DATA SUMMARY ===")
    print("  CURRENT (on disk with labels):")
    print(f"    patients: {len(disk_patients_with_labels):,}")
    print(f"    (patient, date) pairs: {all_disk_with_labels:,}")
    print("\n  POTENTIAL (after downloading remaining):")
    print(f"    patients: {len(all_potential_patients):,}")
    print(f"    (patient, date) pairs: {all_potential_with_labels:,}")
    print(
        f"\n  NOTE: {len(disk_only_keys):,} historical exams on disk are NOT in iBroker"
    )
    print(f"        but {disk_only_with_labels:,} of them have labels and CAN be used!")
else:
    missing = []
    if not phenotype_path.exists():
        missing.append(f"phenotype: {phenotype_path}")
    if not mrn_mapping_path.exists():
        missing.append(f"MRN mapping: {mrn_mapping_path}")
    if not fingerprint_cache.exists():
        missing.append(f"fingerprint cache: {fingerprint_cache}")
    print(f"skipping data sources analysis - missing files: {', '.join(missing)}")

print("\n=== ANALYSIS COMPLETE ===")
print(f"plots saved to: {plots_dir.absolute()}")
print("created 'all' and 'genotyped' versions of all plots")
