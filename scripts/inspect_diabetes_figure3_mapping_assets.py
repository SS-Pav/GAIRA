import io
import re
from pathlib import Path
from zipfile import ZipFile

import numpy as np
from scipy.io import loadmat


ARCHIVE_PATH = Path("/Volumes/SSD_SPG/GAIRA_DATA/raw/diabetes_plasma_ev_sers/Diabetes_Raw_Data_Codes.zip")
FIG3_PREFIX = "Diabetes - Raw Data - Codes/Figure 3/"
KEY_TERMS = [
    "a_nwd",
    "a_owd",
    "w_nwd",
    "w_owd",
    "asian",
    "white",
    "nwd",
    "owd",
    "matched",
    "unmatched",
    "filename",
    "file",
    "name",
    "mask",
    "index",
    "indices",
    "label",
    "race",
    "bmi",
    "group",
]


def is_private_mat_key(key: str) -> bool:
    return key.startswith("__") and key.endswith("__")


def inspect_mat_member(archive: ZipFile, member: str) -> dict[str, object]:
    raw = archive.read(member)
    mat = loadmat(io.BytesIO(raw), squeeze_me=True, struct_as_record=False)
    keys = [key for key in mat.keys() if not is_private_mat_key(key)]
    findings = []
    key_hits = []
    for key in keys:
        key_l = key.lower()
        if any(term in key_l for term in KEY_TERMS):
            key_hits.append(key)
        value = mat[key]
        if isinstance(value, str):
            findings.append(f"{key}: string={value!r}")
        elif isinstance(value, np.ndarray):
            if value.dtype.kind in {"U", "S"}:
                findings.append(f"{key}: string_ndarray shape={value.shape} preview={value.flatten()[:10].tolist()}")
            elif value.dtype == object:
                findings.append(f"{key}: object_ndarray shape={value.shape}")
            else:
                findings.append(f"{key}: numeric_array shape={value.shape} dtype={value.dtype}")
        else:
            findings.append(f"{key}: type={type(value).__name__}")
    return {
        "member": member,
        "keys": keys,
        "key_hits": key_hits,
        "findings": findings,
    }


def inspect_code_member(archive: ZipFile, member: str) -> dict[str, object]:
    text = archive.read(member).decode("utf-8", errors="ignore")
    hits = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        line_l = line.lower()
        if any(term in line_l for term in KEY_TERMS):
            hits.append(f"{line_number}: {line.rstrip()}")
    return {
        "member": member,
        "hits": hits,
        "text": text,
    }


def main() -> None:
    print(f"archive: {ARCHIVE_PATH}")
    with ZipFile(ARCHIVE_PATH, "r") as archive:
        members = archive.namelist()
        inspected = [
            member
            for member in members
            if member.startswith(FIG3_PREFIX)
            and (
                member.endswith("configure_classes.m")
                or member.endswith("race_split.m")
                or member.endswith(".mat")
            )
        ]

        print("files inspected:")
        for member in inspected:
            print(f"- {member}")

        code_members = [member for member in inspected if member.endswith(".m")]
        mat_members = [member for member in inspected if member.endswith(".mat")]

        explicit_subgroup_vectors = []
        explicit_filename_lists = []
        cached_mapping_mats = []
        order_reference_hits = []

        print("\nCODE FINDINGS")
        for member in code_members:
            result = inspect_code_member(archive, member)
            print(f"\n=== {member} ===")
            if result["hits"]:
                for hit in result["hits"][:80]:
                    print(hit)
            else:
                print("no relevant hits")

            joined = "\n".join(result["hits"]).lower()
            if re.search(r"\ba_nwd\b|\ba_owd\b|\bw_nwd\b|\bw_owd\b", joined):
                explicit_subgroup_vectors.append(member)
            if "matched_" in joined or "unmatched_" in joined:
                order_reference_hits.append(member)
            if "filename" in joined or "names_org" in joined or "names_imp_org" in joined or "names_str_org" in joined:
                explicit_filename_lists.append(member)

        print("\nMAT FINDINGS")
        for member in mat_members:
            result = inspect_mat_member(archive, member)
            print(f"\n=== {member} ===")
            print(f"keys: {result['keys']}")
            if result["key_hits"]:
                print(f"key hits: {result['key_hits']}")
            for finding in result["findings"][:40]:
                print(finding)
            if result["key_hits"]:
                cached_mapping_mats.append(member)

        print("\nSUMMARY")
        print(f"explicit subgroup index vectors found: {'yes' if explicit_subgroup_vectors else 'no'}")
        if explicit_subgroup_vectors:
            print(f" subgroup vector files: {sorted(set(explicit_subgroup_vectors))}")
        print(f"explicit filename/order lists found in released assets: {'yes' if explicit_filename_lists else 'no'}")
        if explicit_filename_lists:
            print(f" filename/order evidence files: {sorted(set(explicit_filename_lists))}")
        print(f"cached .mat mapping variables found: {'yes' if cached_mapping_mats else 'no'}")
        if cached_mapping_mats:
            print(f" mapping mat files: {sorted(set(cached_mapping_mats))}")
        print(f"ordering-reference code found: {'yes' if order_reference_hits else 'no'}")
        if order_reference_hits:
            print(f" ordering-reference files: {sorted(set(order_reference_hits))}")

        print("\nVERDICT")
        print("cell_index -> patient filename: still not defensible")
        print("patient filename -> subgroup label: partially defensible from patient_data.csv only")
        print("overall archive-based reconstruction: still not defensible")


if __name__ == "__main__":
    main()
