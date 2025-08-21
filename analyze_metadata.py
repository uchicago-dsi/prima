#!/usr/bin/env python3
"""
Script extracted from dev.ipynb
Performs ChiMEC patient data analysis and generates plots
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

# create plots directory if it doesn't exist
plots_dir = Path("plots")
plots_dir.mkdir(exist_ok=True)

print("Loading data...")

# load data
chimec_patients = pd.read_csv(
    "/gpfs/data/phs/groups/Projects/Huo_projects/SPORE/annawoodard/List_ChiMEC_priority_2025July30.csv"
)
print(f"Loaded {len(chimec_patients):,} ChiMEC patients")

key = pd.read_csv("/gpfs/data/huo-lab/Image/ChiMEC/study-16352a.csv")
metadata = pd.read_csv(
    "/gpfs/data/huo-lab/Image/annawoodard/prima/data/imaging_metadata.csv"
)

print(f"Loaded key table with {len(key):,} records")
print(f"Loaded metadata with {len(metadata):,} records")


# helper function to extract base modality
def get_base_modality(modality):
    """
    extract base modality from full modality string
    examples: 'MG' -> 'MG', 'MR' -> 'MR', 'CT' -> 'CT', etc.
    """
    if pd.isna(modality):
        return None

    modality_str = str(modality).upper()

    # common base modalities
    if "MG" in modality_str:
        return "MG"
    elif "MR" in modality_str:
        return "MR"
    elif "CT" in modality_str:
        return "CT"
    elif "US" in modality_str:
        return "US"
    elif "CR" in modality_str:
        return "CR"
    elif "DX" in modality_str:
        return "DX"
    elif "NM" in modality_str:
        return "NM"
    elif "PT" in modality_str:
        return "PT"
    else:
        return modality_str  # return as is if no match found


# add base modality column to metadata
metadata["base_modality"] = metadata["Modality"].apply(get_base_modality)

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
    case_screening_mammograms,
    case_before_dx_month_mammograms,
    control_screening_mammograms,
    chip_status,
    plots_dir,
):
    """create screening mammograms plot"""
    fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, figsize=(24, 12))

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

    ax1.set_title("cases\n(mammograms >3 months before diagnosis)")
    ax1.set_xlabel("number of screening mammograms per patient")
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

    ax2.set_title("cases\n(mammograms before/including dx month)")
    ax2.set_xlabel("number of mammograms per patient")
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

    ax3.set_title("controls\n(all mammograms)")
    ax3.set_xlabel("number of mammograms per patient")
    ax3.set_ylabel("number of patients")
    ax3.yaxis.set_major_locator(plt.MaxNLocator(integer=True))

    # fourth panel - collapsed summary
    categories = [
        "cases\n(>3 months before dx)",
        "cases\n(before/including dx month)",
        "controls\n(all mammograms)",
    ]
    counts = [
        len(case_screening_per_patient),
        len(case_before_dx_month_per_patient),
        len(control_screening_per_patient),
    ]

    bars4 = ax4.bar(categories, counts, alpha=0.7, edgecolor="black")
    ax4.set_title("total patients with mammograms")
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

    # fifth panel - total mammograms by category
    categories_mammograms = [
        "cases\n(>3 months before dx)",
        "cases\n(before/including dx month)",
        "controls\n(all mammograms)",
    ]
    total_mammograms = [
        len(case_screening_mammograms),
        len(case_before_dx_month_mammograms),
        len(control_screening_mammograms),
    ]

    bars5 = ax5.bar(
        categories_mammograms, total_mammograms, alpha=0.7, edgecolor="black"
    )
    ax5.set_title("total mammograms by category")
    ax5.set_ylabel("number of mammograms")
    ax5.yaxis.set_major_locator(plt.MaxNLocator(integer=True))

    # add value labels on bars
    for bar, count in zip(bars5, total_mammograms):
        height = bar.get_height()
        ax5.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.01 * max(total_mammograms),
            f"{count}",
            ha="center",
            va="bottom",
        )

    # sixth panel - empty
    ax6.set_xticks([])
    ax6.set_yticks([])

    plt.tight_layout()
    plt.savefig(
        plots_dir / f"screening_mammograms_per_patient_{chip_status}.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


def create_mg_mammograms_per_patient_plot(
    mg_per_patient, chimec_with_study_id, mg_scans, chip_status, plots_dir
):
    """create MG mammograms per patient plot"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # get case and control patients
    mg_with_case_control = mg_scans.merge(
        chimec_with_study_id[["AnonymousID", "case_or_control"]],
        left_on="study_id",
        right_on="AnonymousID",
        how="inner",
    )

    # cases
    case_mg_scans = mg_with_case_control[
        mg_with_case_control["case_or_control"] == "Case"
    ]
    case_mg_per_patient = case_mg_scans["study_id"].value_counts()

    ax1.hist(
        case_mg_per_patient.values,
        bins=range(1, max(case_mg_per_patient.max(), 2) + 2),
        alpha=0.7,
        edgecolor="black",
    )
    ax1.set_title("cases")
    ax1.set_xlabel("number of mammograms per patient")
    ax1.set_ylabel("number of patients")
    ax1.set_xticks(range(1, min(21, case_mg_per_patient.max() + 1)))
    ax1.yaxis.set_major_locator(plt.MaxNLocator(integer=True))

    # controls
    control_mg_scans = mg_with_case_control[
        mg_with_case_control["case_or_control"] == "Control"
    ]
    control_mg_per_patient = control_mg_scans["study_id"].value_counts()

    ax2.hist(
        control_mg_per_patient.values,
        bins=range(1, max(control_mg_per_patient.max(), 2) + 2),
        alpha=0.7,
        edgecolor="black",
    )
    ax2.set_title("controls")
    ax2.set_xlabel("number of mammograms per patient")
    ax2.set_ylabel("number of patients")
    ax2.set_xticks(range(1, min(21, control_mg_per_patient.max() + 1)))
    ax2.yaxis.set_major_locator(plt.MaxNLocator(integer=True))

    plt.tight_layout()
    plt.savefig(
        plots_dir / f"mg_mammograms_per_patient_{chip_status}.png",
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
    total_patients = metadata["study_id"].nunique()
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


def create_time_analysis_plot(cases_mg, controls_mg, chip_status, plots_dir):
    """create time analysis plot"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 6))

    # cases
    if len(cases_mg) > 0:
        case_time_to_dx = cases_mg["days_to_dx"].dropna()
        # convert days to years
        case_time_to_dx_years = case_time_to_dx / 365.25
        ax1.hist(case_time_to_dx_years, bins=50, alpha=0.7, edgecolor="black")
        ax1.set_title("cases\n(positive = mammogram before diagnosis)")
        ax1.set_xlabel("years from mammogram to diagnosis")
        ax1.set_ylabel("number of mammograms")
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
        ax1.set_title("cases\n(positive = mammogram before diagnosis)")
        ax1.set_xlabel("years from mammogram to diagnosis")
        ax1.set_ylabel("number of mammograms")

    # controls
    if len(controls_mg) > 0:
        control_time_from_first = controls_mg["days_from_first_mammogram"].dropna()
        # convert days to years
        control_time_from_first_years = control_time_from_first / 365.25
        ax2.hist(control_time_from_first_years, bins=50, alpha=0.7, edgecolor="black")
        ax2.axvline(
            x=0, color="red", linestyle="--", alpha=0.7, label="first mammogram"
        )
    else:
        ax2.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax2.transAxes)

    ax2.set_title("controls\n(time from first mammogram)")
    ax2.set_xlabel("years from first mammogram")
    ax2.set_ylabel("number of mammograms")
    if len(controls_mg) > 0:
        ax2.legend()

    plt.tight_layout()
    plt.savefig(
        plots_dir / f"time_analysis_cases_vs_controls_{chip_status}.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


print("\n=== BASE MODALITY ANALYSIS ===")
print("unique base modalities:")
print(metadata["base_modality"].value_counts())

print("\nexamples of modality -> base_modality mapping:")
sample_mapping = metadata[["Modality", "base_modality"]].drop_duplicates().head(20)
print(sample_mapping)

# find MRNs in chimec_patients but not in key
missing_mrns = chimec_patients[~chimec_patients["MRN"].isin(key["MRN"])]["MRN"].tolist()
print(f"\nNumber of MRNs in chimec_patients but not in key: {len(missing_mrns)}")

# save missing MRNs to text file
with open("missing_mrns.txt", "w") as f:
    for mrn in missing_mrns:
        f.write(f"{mrn}\n")

print(f"Saved {len(missing_mrns)} missing MRNs to missing_mrns.txt")

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
print(f"Original chimec_patients shape: {chimec_patients.shape}")
print(f"After merging with key table: {chimec_with_study_id.shape}")
print(f"Patients with chip data: {chimec_with_study_id['chip'].notna().sum()}")
print(
    f"Patients with chip data AND study_id: {(chimec_with_study_id['chip'].notna() & chimec_with_study_id['AnonymousID'].notna()).sum()}"
)

# step 2: identify patients with chip data who have at least one MG modality record
# get study_ids for patients with chip data
chip_patients_study_ids = chimec_with_study_id[chimec_with_study_id["chip"].notna()][
    "AnonymousID"
].dropna()

print(
    f"\nNumber of patients with chip data: {len(chimec_with_study_id[chimec_with_study_id['chip'].notna()])}"
)
print(
    f"Number of patients with chip data who have study_id: {len(chip_patients_study_ids)}"
)

# find which of these study_ids have MG modality records
mg_records = metadata[metadata["Modality"].str.contains("MG", na=False)]
chip_patients_with_mg = chip_patients_study_ids[
    chip_patients_study_ids.isin(mg_records["study_id"])
]

print(
    f"Number of chip patients with at least one MG record: {len(chip_patients_with_mg)}"
)
print(
    f"Percentage of chip patients with MG records: {len(chip_patients_with_mg) / len(chip_patients_study_ids) * 100:.1f}%"
)

# enhanced summary with cases vs controls
chip_patients_with_status = chimec_with_study_id[chimec_with_study_id["chip"].notna()]
cases_count = chip_patients_with_status[
    chip_patients_with_status["case_or_control"] == "Case"
].shape[0]
controls_count = chip_patients_with_status[
    chip_patients_with_status["case_or_control"] == "Control"
].shape[0]

print("\n=== ENHANCED SUMMARY ===")
print(f"Total ChiMEC patients: {len(chimec_patients):,}")
print(
    f"ChiMEC patients with chip data (genotyped): {len(chimec_with_study_id[chimec_with_study_id['chip'].notna()]):,} ({len(chimec_with_study_id[chimec_with_study_id['chip'].notna()]) / len(chimec_patients) * 100:.1f}%)"
)
print(f"  - Cases: {cases_count:,}")
print(f"  - Controls: {controls_count:,}")
print(
    f"ChiMEC patients with imaging data available: {metadata['study_id'].nunique():,} ({metadata['study_id'].nunique() / len(chimec_patients) * 100:.1f}%)"
)
print(f"Genotyped ChiMEC patients with imaging data: {len(chip_patients_study_ids):,}")
print(f"Genotyped ChiMEC patients with MG imaging: {len(chip_patients_with_mg):,}")

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
cases_imaging = cases_imaging[cases_imaging["base_modality"] == "MG"]

# ensure Study DateTime is datetime type
cases_imaging["Study DateTime"] = pd.to_datetime(cases_imaging["Study DateTime"])

# calculate time-to-diagnosis (days from imaging to diagnosis)
cases_imaging["time_to_diagnosis"] = (
    cases_imaging["DatedxIndex"] - cases_imaging["Study DateTime"]
).dt.days

# convert days to years (approximate, 365.25 days per year)
cases_imaging["time_to_diagnosis_years"] = cases_imaging["time_to_diagnosis"] / 365.25

print(f"Total case imaging records: {len(cases_imaging)}")
print(
    f"Records with valid time-to-diagnosis: {cases_imaging['time_to_diagnosis'].notna().sum()}"
)
print(
    f"Range of time-to-diagnosis: {cases_imaging['time_to_diagnosis'].min()} to {cases_imaging['time_to_diagnosis'].max()} days"
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
print(f"Mean: {time_to_dx_years.mean():.2f} years")
print(f"Median: {time_to_dx_years.median():.2f} years")
print(f"Exams after diagnosis (negative values): {(time_to_dx_years < 0).sum()}")
print(f"Exams before diagnosis (positive values): {(time_to_dx_years > 0).sum()}")

# debugging extreme time-to-diagnosis values
print("\n=== DEBUGGING EXTREME TIME-TO-DIAGNOSIS VALUES ===")
print("=" * 50)

# find records with extreme negative values (more than 10 years before diagnosis)
extreme_negative = cases_imaging[cases_imaging["time_to_diagnosis"] < -3650]  # 10 years
print(f"Records with >10 years before diagnosis: {len(extreme_negative)}")

if len(extreme_negative) > 0:
    print("\nEXAMPLES OF EXTREME VALUES:")
    extreme_examples = extreme_negative[
        [
            "AnonymousID",
            "DatedxIndex",
            "Study DateTime",
            "time_to_diagnosis",
            "Modality",
        ]
    ].head(10)
    print(extreme_examples)

    print(
        f"\nMINIMUM TIME-TO-DIAGNOSIS: {extreme_negative['time_to_diagnosis'].min()} days"
    )
    print(
        f"This is approximately {extreme_negative['time_to_diagnosis'].min() / 365.25:.1f} years before diagnosis"
    )

    # check if there are any obvious date format issues
    print("\nCHECKING FOR DATE FORMAT ISSUES:")
    min_record = extreme_negative.loc[extreme_negative["time_to_diagnosis"].idxmin()]
    print("Record with minimum value:")
    print(f"  AnonymousID: {min_record['AnonymousID']}")
    print(
        f"  DatedxIndex: {min_record['DatedxIndex']} (type: {type(min_record['DatedxIndex'])})"
    )
    print(
        f"  Study DateTime: {min_record['Study DateTime']} (type: {type(min_record['Study DateTime'])})"
    )
    print(f"  Time to diagnosis: {min_record['time_to_diagnosis']} days")
    print(f"  Modality: {min_record['Modality']}")

    # check if this is a specific patient with multiple extreme values
    print("\nCHECKING IF THIS IS A PATTERN FOR SPECIFIC PATIENTS:")
    patient_extreme_counts = extreme_negative["AnonymousID"].value_counts()
    print(f"Patients with extreme values: {len(patient_extreme_counts)}")
    print("Top 5 patients with most extreme values:")
    print(patient_extreme_counts.head())

# also check for extreme positive values
extreme_positive = cases_imaging[cases_imaging["time_to_diagnosis"] > 3650]  # 10 years
print(f"\nRecords with >10 years after diagnosis: {len(extreme_positive)}")

if len(extreme_positive) > 0:
    print("\nEXAMPLES OF EXTREME POSITIVE VALUES:")
    extreme_pos_examples = extreme_positive[
        [
            "AnonymousID",
            "DatedxIndex",
            "Study DateTime",
            "time_to_diagnosis",
            "Modality",
        ]
    ].head(5)
    print(extreme_pos_examples)

# check the overall distribution
print("\nOVERALL DISTRIBUTION:")
print(f"Total records: {len(cases_imaging)}")
print(
    f"Records with valid time-to-diagnosis: {cases_imaging['time_to_diagnosis'].notna().sum()}"
)
print(
    f"Range: {cases_imaging['time_to_diagnosis'].min()} to {cases_imaging['time_to_diagnosis'].max()} days"
)
print(f"Mean: {cases_imaging['time_to_diagnosis'].mean():.1f} days")
print(f"Median: {cases_imaging['time_to_diagnosis'].median():.1f} days")


def extract_base_modality(modality: str) -> str:
    """
    extract the base imaging modality from a modality string,
    ignoring derived/secondary DICOM object classes (PR, SR, KO, OT, SC, DOC, XC, CADSR, REG, RTSTRUCT, etc.)

    Parameters
    ----------
    modality : str
        modality string (e.g., "CT/PR/SR")

    Returns
    -------
    str
        base modality (e.g., "CT"), or "Other" if no base modality is found
    """
    # set of actual acquisition modalities
    base_modalities = {
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
    }

    tokens = modality.split("/")
    for token in tokens:
        if token in base_modalities:
            return token
    return "Other"


# add base modality column to metadata
metadata["base_modality"] = metadata["Modality"].apply(extract_base_modality)

print("\n=== MODALITY ANALYSIS ===")
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

# create modality distribution plot for all patients
print("\n=== CREATING MODALITY DISTRIBUTION PLOTS ===")
create_modality_distribution_plot(metadata, "all", plots_dir)

# print interpretation
total_patients = metadata["study_id"].nunique()
patient_counts = (
    metadata.groupby("base_modality")["study_id"].nunique().sort_values(ascending=False)
)
print("\nINTERPRETATION:")
print("- Each bar shows how many patients have at least one exam of that modality type")
print("- Patients with multiple modalities appear in multiple bars")
print(
    f"- Sum ({patient_counts.sum():,}) > Total patients ({total_patients:,}) is EXPECTED"
)
print(
    "- This visualization answers: 'How many patients have imaging data for each modality?'"
)

# MG mammogram analysis
print("\n=== MG MAMMOGRAM ANALYSIS ===")
# filter for only MG mammograms
mg_scans = metadata[metadata["base_modality"] == "MG"].copy()

print(f"total MG scans: {len(mg_scans)}")
print(f"unique patients with MG scans: {mg_scans['study_id'].nunique()}")

# count number of MG mammograms per patient
mg_per_patient = mg_scans["study_id"].value_counts()

print("\nsummary statistics:")
print(f"min mammograms per patient: {mg_per_patient.min()}")
print(f"max mammograms per patient: {mg_per_patient.max()}")
print(f"mean mammograms per patient: {mg_per_patient.mean():.2f}")
print(f"median mammograms per patient: {mg_per_patient.median():.1f}")

# create MG mammograms per patient plot for all patients
create_mg_mammograms_per_patient_plot(
    mg_per_patient, chimec_with_study_id, mg_scans, "all", plots_dir
)

print("\ntop 10 patients with most mammograms:")
print(mg_per_patient.head(10))

# screening mammogram analysis
print("\n=== SCREENING MAMMOGRAM ANALYSIS ===")
# first merge MG scans with patient data to get DatedxIndex
mg_with_patient_data = mg_scans.merge(
    chimec_with_study_id[["AnonymousID", "DatedxIndex", "case_or_control"]],
    left_on="study_id",
    right_on="AnonymousID",
    how="inner",
)

print(f"MG scans with patient data: {len(mg_with_patient_data)}")

# key insight about imaging population
imaging_patients_with_chip = mg_with_patient_data["chip"].notna().sum()
total_imaging_patients = mg_with_patient_data["study_id"].nunique()
print(f"ChiMEC patients with MG imaging: {total_imaging_patients:,}")
print(
    f"Of these, genotyped: {mg_with_patient_data[mg_with_patient_data['chip'].notna()]['study_id'].nunique():,} ({mg_with_patient_data[mg_with_patient_data['chip'].notna()]['study_id'].nunique() / total_imaging_patients * 100:.1f}%)"
)
print(
    f"Of these, NOT genotyped: {mg_with_patient_data[mg_with_patient_data['chip'].isna()]['study_id'].nunique():,} ({mg_with_patient_data[mg_with_patient_data['chip'].isna()]['study_id'].nunique() / total_imaging_patients * 100:.1f}%)"
)
print(
    "NOTE: This explains why 'all' and 'genotyped' plots show identical counts - all ChiMEC patients with imaging happen to be genotyped!"
)

# convert dates to datetime
mg_with_patient_data["Study DateTime"] = pd.to_datetime(
    mg_with_patient_data["Study DateTime"]
)
mg_with_patient_data["DatedxIndex"] = pd.to_datetime(
    mg_with_patient_data["DatedxIndex"], format="%d/%m/%Y"
)

# calculate time difference in days (positive = mammogram before diagnosis)
mg_with_patient_data["days_to_dx"] = (
    mg_with_patient_data["DatedxIndex"] - mg_with_patient_data["Study DateTime"]
).dt.days

# separate cases and controls for screening analysis
cases_mg = mg_with_patient_data[
    mg_with_patient_data["case_or_control"] == "Case"
].copy()
controls_mg = mg_with_patient_data[
    mg_with_patient_data["case_or_control"] == "Control"
].copy()

# for cases: filter for screening mammograms (> 3 months = 90 days before diagnosis)
case_screening_mammograms = cases_mg[cases_mg["days_to_dx"] > 90].copy()

# for cases: filter for mammograms anytime before and including diagnosis month
# convert to year-month for comparison
cases_mg["mammogram_ym"] = cases_mg["Study DateTime"].dt.to_period("M")
cases_mg["diagnosis_ym"] = cases_mg["DatedxIndex"].dt.to_period("M")
case_before_dx_month_mammograms = cases_mg[
    cases_mg["mammogram_ym"] <= cases_mg["diagnosis_ym"]
].copy()

# for controls: all mammograms are considered (no diagnosis date filter)
control_screening_mammograms = controls_mg.copy()

print(
    f"case screening mammograms (>90 days before dx): {len(case_screening_mammograms)}"
)
print(
    f"case mammograms (before/including dx month): {len(case_before_dx_month_mammograms)}"
)
print(f"control mammograms (all): {len(control_screening_mammograms)}")
print(
    f"unique case patients with screening mammograms: {case_screening_mammograms['study_id'].nunique() if len(case_screening_mammograms) > 0 else 0}"
)
print(
    f"unique case patients with mammograms before/including dx month: {case_before_dx_month_mammograms['study_id'].nunique() if len(case_before_dx_month_mammograms) > 0 else 0}"
)
print(
    f"unique control patients with mammograms: {control_screening_mammograms['study_id'].nunique() if len(control_screening_mammograms) > 0 else 0}"
)

# count screening mammograms per patient for cases
case_screening_per_patient = (
    case_screening_mammograms["study_id"].value_counts()
    if len(case_screening_mammograms) > 0
    else pd.Series(dtype=int)
)

# count mammograms before/including diagnosis month per patient for cases
case_before_dx_month_per_patient = (
    case_before_dx_month_mammograms["study_id"].value_counts()
    if len(case_before_dx_month_mammograms) > 0
    else pd.Series(dtype=int)
)

# count all mammograms per patient for controls
control_screening_per_patient = (
    control_screening_mammograms["study_id"].value_counts()
    if len(control_screening_mammograms) > 0
    else pd.Series(dtype=int)
)

print("\ncase screening mammogram summary statistics:")
if len(case_screening_per_patient) > 0:
    print(f"min screening mammograms per patient: {case_screening_per_patient.min()}")
    print(f"max screening mammograms per patient: {case_screening_per_patient.max()}")
    print(
        f"mean screening mammograms per patient: {case_screening_per_patient.mean():.2f}"
    )
    print(
        f"median screening mammograms per patient: {case_screening_per_patient.median():.1f}"
    )
else:
    print("no case screening mammograms found")

print("\ncase mammograms before/including dx month summary statistics:")
if len(case_before_dx_month_per_patient) > 0:
    print(f"min mammograms per patient: {case_before_dx_month_per_patient.min()}")
    print(f"max mammograms per patient: {case_before_dx_month_per_patient.max()}")
    print(f"mean mammograms per patient: {case_before_dx_month_per_patient.mean():.2f}")
    print(
        f"median mammograms per patient: {case_before_dx_month_per_patient.median():.1f}"
    )
else:
    print("no case mammograms before/including dx month found")

print("\ncontrol mammogram summary statistics:")
if len(control_screening_per_patient) > 0:
    print(f"min mammograms per patient: {control_screening_per_patient.min()}")
    print(f"max mammograms per patient: {control_screening_per_patient.max()}")
    print(f"mean mammograms per patient: {control_screening_per_patient.mean():.2f}")
    print(
        f"median mammograms per patient: {control_screening_per_patient.median():.1f}"
    )
else:
    print("no control mammograms found")

# create screening mammograms plot for all patients
create_screening_mammograms_plot(
    case_screening_per_patient,
    case_before_dx_month_per_patient,
    control_screening_per_patient,
    case_screening_mammograms,
    case_before_dx_month_mammograms,
    control_screening_mammograms,
    "all",
    plots_dir,
)

print("\ntop 10 case patients with most screening mammograms:")
if len(case_screening_per_patient) > 0:
    print(case_screening_per_patient.head(10))
else:
    print("no case screening mammograms found")

print("\ntop 10 case patients with most mammograms before/including dx month:")
if len(case_before_dx_month_per_patient) > 0:
    print(case_before_dx_month_per_patient.head(10))
else:
    print("no case mammograms before/including dx month found")

print("\ntop 10 control patients with most mammograms:")
if len(control_screening_per_patient) > 0:
    print(control_screening_per_patient.head(10))
else:
    print("no control mammograms found")

# histogram of time to DatedxIndex for cases and time from first mammogram for controls
print("\n=== TIME ANALYSIS ===")

# separate cases and controls
cases_mg = mg_with_patient_data[
    mg_with_patient_data["case_or_control"] == "Case"
].copy()
controls_mg = mg_with_patient_data[
    mg_with_patient_data["case_or_control"] == "Control"
].copy()

# for controls, calculate time from first mammogram
if len(controls_mg) > 0:
    controls_mg = controls_mg.sort_values("Study DateTime")
    first_mammogram_dates = controls_mg.groupby("study_id")["Study DateTime"].first()
    controls_mg = controls_mg.merge(
        first_mammogram_dates.rename("first_mammogram_date"),
        left_on="study_id",
        right_index=True,
    )
    controls_mg["days_from_first_mammogram"] = (
        controls_mg["Study DateTime"] - controls_mg["first_mammogram_date"]
    ).dt.days

# cases time to diagnosis analysis
if len(cases_mg) > 0:
    case_time_to_dx = cases_mg["days_to_dx"].dropna()
    print(f"total case MG scans with valid time to diagnosis: {len(case_time_to_dx)}")
    print("\ncase time to diagnosis summary statistics:")
    print(f"min: {case_time_to_dx.min()} days")
    print(f"max: {case_time_to_dx.max()} days")
    print(f"mean: {case_time_to_dx.mean():.1f} days")
    print(f"median: {case_time_to_dx.median():.1f} days")
    print(
        f"mammograms before diagnosis (positive values): {(case_time_to_dx > 0).sum()}"
    )
    print(
        f"mammograms after diagnosis (negative values): {(case_time_to_dx < 0).sum()}"
    )
    print(f"mammograms on diagnosis day: {(case_time_to_dx == 0).sum()}")

# controls time from first mammogram analysis
if len(controls_mg) > 0:
    control_time_from_first = controls_mg["days_from_first_mammogram"].dropna()
    print(
        f"\ntotal control MG scans with valid time from first mammogram: {len(control_time_from_first)}"
    )
    print("\ncontrol time from first mammogram summary statistics:")
    print(f"min: {control_time_from_first.min()} days")
    print(f"max: {control_time_from_first.max()} days")
    print(f"mean: {control_time_from_first.mean():.1f} days")
    print(f"median: {control_time_from_first.median():.1f} days")

# create time analysis plot for all patients
create_time_analysis_plot(cases_mg, controls_mg, "all", plots_dir)

# additional breakdown for cases
if len(cases_mg) > 0:
    case_time_to_dx = cases_mg["days_to_dx"].dropna()
    print("\ncase breakdown by time periods:")
    print(f"screening mammograms (>90 days before dx): {(case_time_to_dx > 90).sum()}")
    print(
        f"diagnostic mammograms (0-90 days before dx): {((case_time_to_dx >= 0) & (case_time_to_dx <= 90)).sum()}"
    )
    print(f"mammograms after diagnosis: {(case_time_to_dx < 0).sum()}")

# additional breakdown for controls
if len(controls_mg) > 0:
    control_time_from_first = controls_mg["days_from_first_mammogram"].dropna()
    print("\ncontrol breakdown by time periods:")
    print(f"first mammograms: {(control_time_from_first == 0).sum()}")
    print(f"follow-up mammograms (>0 days): {(control_time_from_first > 0).sum()}")
    print(
        f"mammograms within 1 year of first: {((control_time_from_first >= 0) & (control_time_from_first <= 365)).sum()}"
    )
    print(f"mammograms >1 year after first: {(control_time_from_first > 365).sum()}")

# create genotyped-only versions of all plots
print("\n=== CREATING GENOTYPED-ONLY VERSIONS ===")

# filter for only genotyped patients
genotyped_patients = filter_patients_by_chip_status(chimec_with_study_id, "genotyped")
genotyped_metadata = metadata[
    metadata["study_id"].isin(genotyped_patients["AnonymousID"])
]

print(f"genotyped patients: {len(genotyped_patients)}")
print(f"genotyped metadata records: {len(genotyped_metadata)}")

# create genotyped-only modality distribution plot
create_modality_distribution_plot(genotyped_metadata, "genotyped", plots_dir)

# create genotyped-only MG mammograms per patient plot
genotyped_mg_scans = genotyped_metadata[
    genotyped_metadata["base_modality"] == "MG"
].copy()
genotyped_mg_per_patient = genotyped_mg_scans["study_id"].value_counts()
create_mg_mammograms_per_patient_plot(
    genotyped_mg_per_patient,
    genotyped_patients,
    genotyped_mg_scans,
    "genotyped",
    plots_dir,
)

# create genotyped-only screening mammograms analysis
genotyped_mg_with_patient_data = genotyped_mg_scans.merge(
    genotyped_patients[["AnonymousID", "DatedxIndex", "case_or_control"]],
    left_on="study_id",
    right_on="AnonymousID",
    how="inner",
)

print(f"genotyped MG scans with patient data: {len(genotyped_mg_with_patient_data)}")

# convert dates to datetime
genotyped_mg_with_patient_data["Study DateTime"] = pd.to_datetime(
    genotyped_mg_with_patient_data["Study DateTime"]
)
genotyped_mg_with_patient_data["DatedxIndex"] = pd.to_datetime(
    genotyped_mg_with_patient_data["DatedxIndex"], format="%d/%m/%Y"
)

# calculate time difference in days (positive = mammogram before diagnosis)
genotyped_mg_with_patient_data["days_to_dx"] = (
    genotyped_mg_with_patient_data["DatedxIndex"]
    - genotyped_mg_with_patient_data["Study DateTime"]
).dt.days

# separate cases and controls for screening analysis
genotyped_cases_mg = genotyped_mg_with_patient_data[
    genotyped_mg_with_patient_data["case_or_control"] == "Case"
].copy()
genotyped_controls_mg = genotyped_mg_with_patient_data[
    genotyped_mg_with_patient_data["case_or_control"] == "Control"
].copy()

# for cases: filter for screening mammograms (> 3 months = 90 days before diagnosis)
genotyped_case_screening_mammograms = genotyped_cases_mg[
    genotyped_cases_mg["days_to_dx"] > 90
].copy()

# for cases: filter for mammograms anytime before and including diagnosis month
genotyped_cases_mg["mammogram_ym"] = genotyped_cases_mg["Study DateTime"].dt.to_period(
    "M"
)
genotyped_cases_mg["diagnosis_ym"] = genotyped_cases_mg["DatedxIndex"].dt.to_period("M")
genotyped_case_before_dx_month_mammograms = genotyped_cases_mg[
    genotyped_cases_mg["mammogram_ym"] <= genotyped_cases_mg["diagnosis_ym"]
].copy()

# for controls: all mammograms are considered (no diagnosis date filter)
genotyped_control_screening_mammograms = genotyped_controls_mg.copy()

# count screening mammograms per patient for cases
genotyped_case_screening_per_patient = (
    genotyped_case_screening_mammograms["study_id"].value_counts()
    if len(genotyped_case_screening_mammograms) > 0
    else pd.Series(dtype=int)
)

# count mammograms before/including diagnosis month per patient for cases
genotyped_case_before_dx_month_per_patient = (
    genotyped_case_before_dx_month_mammograms["study_id"].value_counts()
    if len(genotyped_case_before_dx_month_mammograms) > 0
    else pd.Series(dtype=int)
)

# count all mammograms per patient for controls
genotyped_control_screening_per_patient = (
    genotyped_control_screening_mammograms["study_id"].value_counts()
    if len(genotyped_control_screening_mammograms) > 0
    else pd.Series(dtype=int)
)

# create genotyped-only screening mammograms plot
create_screening_mammograms_plot(
    genotyped_case_screening_per_patient,
    genotyped_case_before_dx_month_per_patient,
    genotyped_control_screening_per_patient,
    genotyped_case_screening_mammograms,
    genotyped_case_before_dx_month_mammograms,
    genotyped_control_screening_mammograms,
    "genotyped",
    plots_dir,
)

# create genotyped-only time analysis
genotyped_cases_mg_for_time = genotyped_mg_with_patient_data[
    genotyped_mg_with_patient_data["case_or_control"] == "Case"
].copy()
genotyped_controls_mg_for_time = genotyped_mg_with_patient_data[
    genotyped_mg_with_patient_data["case_or_control"] == "Control"
].copy()

# for controls, calculate time from first mammogram
if len(genotyped_controls_mg_for_time) > 0:
    genotyped_controls_mg_for_time = genotyped_controls_mg_for_time.sort_values(
        "Study DateTime"
    )
    genotyped_first_mammogram_dates = genotyped_controls_mg_for_time.groupby(
        "study_id"
    )["Study DateTime"].first()
    genotyped_controls_mg_for_time = genotyped_controls_mg_for_time.merge(
        genotyped_first_mammogram_dates.rename("first_mammogram_date"),
        left_on="study_id",
        right_index=True,
    )
    genotyped_controls_mg_for_time["days_from_first_mammogram"] = (
        genotyped_controls_mg_for_time["Study DateTime"]
        - genotyped_controls_mg_for_time["first_mammogram_date"]
    ).dt.days

# create genotyped-only time analysis plot
create_time_analysis_plot(
    genotyped_cases_mg_for_time, genotyped_controls_mg_for_time, "genotyped", plots_dir
)

print("\n=== SCRIPT COMPLETED ===")
print(f"All plots saved to: {plots_dir.absolute()}")
print("Created both 'all' and 'genotyped' versions of all plots")
