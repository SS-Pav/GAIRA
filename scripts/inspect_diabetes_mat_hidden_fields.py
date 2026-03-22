import io
from pathlib import Path
from zipfile import ZipFile

import numpy as np
from scipy.io import loadmat


ZIP_MEMBERS = {
    "Impact": "Diabetes - Raw Data - Codes/Figure 3/data/RawDataImpact.mat",
    "Strong-D": "Diabetes - Raw Data - Codes/Figure 3/data/RawDataStrong.mat",
}

CLUE_TERMS = [
    "filename",
    "file",
    "name",
    "patient",
    "id",
    "folder",
    "sample",
    "label",
    "metadata",
]


def public_attrs(obj) -> list[str]:
    if hasattr(obj, "__dict__"):
        names = [name for name in vars(obj).keys() if not name.startswith("_")]
    else:
        names = [name for name in dir(obj) if not name.startswith("_")]
    return sorted(set(names))


def looks_private_mat_key(key: str) -> bool:
    return key.startswith("__") and key.endswith("__")


def summarize_numeric_array(value) -> str:
    arr = np.asarray(value)
    return f"numeric array shape={arr.shape} dtype={arr.dtype}"


def inspect_value(value, path: str, depth: int = 0, max_depth: int = 2) -> list[str]:
    lines: list[str] = []
    indent = "  " * depth

    if isinstance(value, str):
        lines.append(f"{indent}{path}: string={value!r}")
        return lines

    if isinstance(value, bytes):
        try:
            decoded = value.decode("utf-8")
        except Exception:
            decoded = repr(value[:80])
        lines.append(f"{indent}{path}: bytes={decoded!r}")
        return lines

    if isinstance(value, np.ndarray):
        if value.dtype.kind in {"U", "S"}:
            preview = value.flatten()[:10].tolist()
            lines.append(f"{indent}{path}: string ndarray shape={value.shape} preview={preview}")
            return lines
        if value.dtype == object:
            lines.append(f"{indent}{path}: object ndarray shape={value.shape}")
            flat = value.flatten()
            for i, item in enumerate(flat[:3]):
                lines.extend(inspect_value(item, f"{path}[{i}]", depth + 1, max_depth))
            return lines
        lines.append(f"{indent}{path}: {summarize_numeric_array(value)}")
        return lines

    attrs = public_attrs(value)
    if attrs:
        lines.append(f"{indent}{path}: object type={type(value).__name__} attrs={attrs}")
        if depth < max_depth:
            for attr in attrs[:20]:
                try:
                    child = getattr(value, attr)
                except Exception as exc:
                    lines.append(f"{indent}  {path}.{attr}: <error {exc}>")
                    continue
                lines.extend(inspect_value(child, f"{path}.{attr}", depth + 1, max_depth))
        return lines

    lines.append(f"{indent}{path}: type={type(value).__name__} repr={repr(value)[:200]}")
    return lines


def find_clues(value, path: str, findings: list[str], depth: int = 0, max_depth: int = 3) -> None:
    path_l = path.lower()
    if any(term in path_l for term in CLUE_TERMS):
        findings.append(f"path clue: {path}")

    if isinstance(value, str):
        if any(term in value.lower() for term in CLUE_TERMS):
            findings.append(f"string clue at {path}: {value!r}")
        return

    if isinstance(value, bytes):
        try:
            text = value.decode("utf-8", errors="ignore")
        except Exception:
            return
        if any(term in text.lower() for term in CLUE_TERMS):
            findings.append(f"bytes clue at {path}: {text!r}")
        return

    if isinstance(value, np.ndarray):
        if value.dtype.kind in {"U", "S"}:
            for item in value.flatten()[:50]:
                if any(term in str(item).lower() for term in CLUE_TERMS):
                    findings.append(f"string ndarray clue at {path}: {item!r}")
            return
        if value.dtype == object and depth < max_depth:
            for i, item in enumerate(value.flatten()[:10]):
                find_clues(item, f"{path}[{i}]", findings, depth + 1, max_depth)
        return

    attrs = public_attrs(value)
    if attrs and depth < max_depth:
        for attr in attrs[:50]:
            try:
                child = getattr(value, attr)
            except Exception:
                continue
            find_clues(child, f"{path}.{attr}", findings, depth + 1, max_depth)


def inspect_member(archive_path: Path, member_name: str, label: str) -> None:
    with ZipFile(archive_path, "r") as archive:
        mat = loadmat(io.BytesIO(archive.read(member_name)), squeeze_me=True, struct_as_record=False)

    print(f"\n=== {label} ===")
    print(f"member: {member_name}")
    print("top-level keys:")
    keys = [key for key in mat.keys() if not looks_private_mat_key(key)]
    print(keys)

    for key in keys:
        value = mat[key]
        print(f"\nkey={key!r} type={type(value).__name__}")
        print("\n".join(inspect_value(value, key, depth=0, max_depth=2)[:80]))

        if isinstance(value, np.ndarray) and value.dtype == object:
            flat = value.flatten()
            print(f"first object elements for {key}:")
            for i, item in enumerate(flat[:3]):
                print(f" element[{i}] type={type(item).__name__}")
                item_lines = inspect_value(item, f"{key}[{i}]", depth=1, max_depth=2)
                print("\n".join(item_lines[:30]))

        findings: list[str] = []
        find_clues(value, key, findings, depth=0, max_depth=3)
        if findings:
            print("clue findings:")
            for finding in sorted(set(findings))[:50]:
                print(f" - {finding}")
        else:
            print("clue findings: none")


def main() -> None:
    archive_path = Path("/Volumes/SSD_Rad/GAIRA_DATA/raw/diabetes_plasma_ev_sers/Diabetes_Raw_Data_Codes.zip")
    print(f"archive: {archive_path}")
    for label, member_name in ZIP_MEMBERS.items():
        inspect_member(archive_path, member_name, label)


if __name__ == "__main__":
    main()
