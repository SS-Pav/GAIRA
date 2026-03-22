import io
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd
from scipy.io import loadmat


ZIP_MEMBER_PATIENT = "Diabetes - Raw Data - Codes/Figure 3/data/patient_data.csv"
ZIP_MEMBER_IMPACT = "Diabetes - Raw Data - Codes/Figure 3/data/RawDataImpact.mat"
ZIP_MEMBER_STRONG = "Diabetes - Raw Data - Codes/Figure 3/data/RawDataStrong.mat"
MATLAB_FILES = [
    "Diabetes - Raw Data - Codes/Figure 3/RUN_THIS_FIRST/read_raw_data_impact.m",
    "Diabetes - Raw Data - Codes/Figure 3/RUN_THIS_FIRST/read_raw_data_strong.m",
    "Diabetes - Raw Data - Codes/Figure 3/Fig3A/Code/configure_classes.m",
    "Diabetes - Raw Data - Codes/Figure 3/Fig3A/Code/race_split.m",
    "Diabetes - Raw Data - Codes/Figure 3/Fig3C/Code/configure_classes.m",
    "Diabetes - Raw Data - Codes/Figure 3/Fig3C/Code/race_split.m",
]


def load_patient_table(archive: ZipFile) -> pd.DataFrame:
    patient_df = pd.read_csv(io.BytesIO(archive.read(ZIP_MEMBER_PATIENT)))
    patient_df.columns = [str(column).strip() for column in patient_df.columns]
    patient_df["filename"] = patient_df["filename"].astype(str).str.strip()
    patient_df["group_code"] = patient_df["filename"].str.slice(0, 4)
    patient_df["patient_suffix"] = (
        patient_df["filename"].str.split("-").str[-1].str.lstrip("0").replace("", "0")
    )
    return patient_df


def load_mat_cells(archive: ZipFile, member: str) -> list[np.ndarray]:
    mat = loadmat(io.BytesIO(archive.read(member)), squeeze_me=True, struct_as_record=False)
    return [np.asarray(cell, dtype=float) for cell in list(mat["smoothed_spectra"])]


def extract_code_evidence(archive: ZipFile) -> dict[str, list[str]]:
    snippets: dict[str, list[str]] = {}
    keywords = [
        "dir(",
        "readtable",
        "matched_",
        "unmatched_",
        "race_impact",
        "race_strong",
        "names_imp_org",
        "names_str_org",
        "impact{i-2}",
        "strong{i-2}",
    ]
    for member in MATLAB_FILES:
        text = archive.read(member).decode("utf-8", errors="ignore")
        lines = []
        for line_number, line in enumerate(text.splitlines(), start=1):
            if any(keyword in line for keyword in keywords):
                lines.append(f"{line_number}: {line.strip()}")
        snippets[member] = lines
    return snippets


def summarize_group(
    patient_df: pd.DataFrame,
    group_name: str,
    expected_cells: int,
    scan_counts: list[int],
) -> dict[str, object]:
    group_df = patient_df[patient_df["Group"] == group_name].reset_index(drop=True)
    counts_match = len(group_df) == expected_cells
    missing_rows = len(group_df) - expected_cells

    if group_name == "Strong-D":
        mapping_status = "partially_defensible"
        reason = (
            "CSV and MAT sample counts match exactly, and released MATLAB code maps metadata "
            "to raw-folder order within the Strong group. However, the raw folder-name list used "
            "for that ordering is not present in the release, so exact cell_index -> filename mapping "
            "cannot be reproduced from the archive alone."
        )
    else:
        mapping_status = "not_defensible"
        reason = (
            "CSV and MAT sample counts do not match. One Impact entry is missing after preprocessing "
            "or release filtering, but the missing sample cannot be identified from the released assets "
            "because the raw folder-name list is absent."
        )

    return {
        "group_name": group_name,
        "csv_rows": int(len(group_df)),
        "mat_cells": int(expected_cells),
        "counts_match": bool(counts_match),
        "missing_rows_vs_csv": int(missing_rows),
        "unique_scan_counts": ",".join(str(value) for value in sorted(set(scan_counts))),
        "race_counts_json": json.dumps(group_df["race_ethnicity"].value_counts().to_dict(), sort_keys=True),
        "ordering_hypothesis": (
            "MAT cell order likely follows dir(root) folder order in the author environment."
        ),
        "mapping_status": mapping_status,
        "reason": reason,
    }


def write_summary(
    output_path: Path,
    group_summary_df: pd.DataFrame,
    code_evidence: dict[str, list[str]],
) -> None:
    strong_status = group_summary_df.loc[group_summary_df["group_name"] == "Strong-D", "mapping_status"].iloc[0]
    impact_status = group_summary_df.loc[group_summary_df["group_name"] == "Impact", "mapping_status"].iloc[0]
    overall_status = "not_defensible"

    lines = [
        "Diabetes Plasma EV SERS patient-mapping audit",
        "",
        f"Strong-D ordering status: {strong_status}",
        f"Impact ordering status: {impact_status}",
        f"Overall row-level reconstruction status: {overall_status}",
        "",
        "Conclusion:",
        "The released MATLAB code clearly assumes that metadata is aligned to raw-folder ordering.",
        "That supports an ordering hypothesis, but the release does not include the underlying raw folder names.",
        "Because of that missing intermediate mapping, exact cell_index -> filename reconstruction is not defensible.",
        "Strong-D is closer to defensible because the CSV and MAT counts both equal 24, but it still falls short of strong proof.",
        "Impact is not defensible because the CSV has 40 rows while the MAT archive contains 39 cells.",
        "",
        "Subgroup-label decision:",
        "No automatic A-NWD / A-OWD / W-NWD / W-OWD reconstruction was performed.",
        "Even if BMI and race fields exist in patient_data.csv, the missing cell-to-patient mapping blocks conservative row-level relabeling.",
        "",
        "Exact assumptions used:",
        "1. MATLAB dir(root) ordering is the intended sample ordering for both Strong and Impact groups.",
        "2. race_split.m matches patient_data.csv identifiers against raw folder names, then uses that matched order for race indexing.",
        "3. The released archive does not expose those raw folder names or the matched/unmatched arrays as saved outputs.",
        "",
        "Exact uncertainties remaining:",
        "1. Which specific Impact patient/sample is missing from RawDataImpact.mat.",
        "2. Whether Strong-D MAT cell order exactly matches CSV row order or only folder-order after name matching.",
        "3. The BMI threshold implementation used for the paper subgroup split, as recoverable archive code for that threshold was not found in this pass.",
        "",
        "Key code evidence:",
    ]
    for member, snippets in code_evidence.items():
        lines.append(f"- {member}")
        lines.extend(f"  {snippet}" for snippet in snippets[:8])
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    archive_path = Path("/Volumes/SSD_Rad/GAIRA_DATA/raw/diabetes_plasma_ev_sers/Diabetes_Raw_Data_Codes.zip")
    output_dir = Path("/Volumes/SSD_Rad/GAIRA_DATA/processed")
    output_dir.mkdir(parents=True, exist_ok=True)
    audit_csv = output_dir / "diabetes_plasma_ev_sers_mapping_audit.csv"
    summary_txt = output_dir / "diabetes_plasma_ev_sers_mapping_summary.txt"

    with ZipFile(archive_path, "r") as archive:
        patient_df = load_patient_table(archive)
        impact_cells = load_mat_cells(archive, ZIP_MEMBER_IMPACT)
        strong_cells = load_mat_cells(archive, ZIP_MEMBER_STRONG)
        code_evidence = extract_code_evidence(archive)

    impact_summary = summarize_group(
        patient_df=patient_df,
        group_name="Impact",
        expected_cells=len(impact_cells),
        scan_counts=[matrix.shape[1] for matrix in impact_cells],
    )
    strong_summary = summarize_group(
        patient_df=patient_df,
        group_name="Strong-D",
        expected_cells=len(strong_cells),
        scan_counts=[matrix.shape[1] for matrix in strong_cells],
    )

    group_summary_df = pd.DataFrame([strong_summary, impact_summary])
    group_summary_df.to_csv(audit_csv, index=False)
    write_summary(summary_txt, group_summary_df, code_evidence)

    print("Patient mapping audit written.")
    print(f"Audit CSV: {audit_csv}")
    print(f"Summary TXT: {summary_txt}")
    print()
    print(group_summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
