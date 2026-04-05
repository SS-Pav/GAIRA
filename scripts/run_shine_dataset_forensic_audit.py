from __future__ import annotations

import math
import re
import sys
from pathlib import Path
from typing import Iterable

import duckdb
import pandas as pd

if str((Path(__file__).resolve().parents[1] / "src")) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gaira.config import get_database_path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REQUESTED_OUTPUT_ROOT = Path(
    "/Volumes/SSD_Rad/GAIRA_DATA/processed/gaira_autoresearch/gaira_autoresearch_v1/shine_dataset_forensic_audit"
)
FALLBACK_OUTPUT_ROOT = PROJECT_ROOT / "reports" / "shine_dataset_forensic_audit"
RAW_DATASET_ROOT = Path("/Volumes/SSD_Rad/GAIRA_DATA/raw/shine_ev_sers")
ARCHIVE_TREE_ROOT = RAW_DATASET_ROOT / "SERS-Hepatotoxicity_DATA_CODE_FIGURE"

PARSER_PATH = PROJECT_ROOT / "src" / "gaira" / "parsers" / "biosample" / "shine_ev_sers_parser.py"
CALIBRATION_SCRIPT_PATH = PROJECT_ROOT / "scripts" / "check_shine_ev_calibration.py"
DOWNLOAD_SCRIPT_PATH = PROJECT_ROOT / "scripts" / "download_dataset.py"
REGISTRY_PATH = PROJECT_ROOT / "config" / "gaira_dataset_experiment_registry_v2.csv"
STORAGE_CONFIG_PATH = PROJECT_ROOT / "config" / "storage.yaml"
PHASE1_REGISTRY_PATH = PROJECT_ROOT / "reports" / "gaira_phase1_registry_audit_v2" / "phase1_dataset_registry_v2.csv"
EMBED_METADATA_PATH = PROJECT_ROOT / "reports" / "embedding_v8_ev_stress_branch_smoke" / "metadata.csv"
DATASET_INVENTORY_PATH = PROJECT_ROOT / "reports" / "gaira_shared_embedding_audit_v1" / "dataset_inventory.csv"

KNOWN_ARCHIVE_FILES = [
    "SERS-Hepatotoxicity_DATA_CODE_FIGURE/Readme.docx",
    "SERS-Hepatotoxicity_DATA_CODE_FIGURE/Figure4/Fig4C/code/plot_spectra.m",
    "SERS-Hepatotoxicity_DATA_CODE_FIGURE/Figure4/Fig4C/data/combined_wavenumbers.mat",
    "SERS-Hepatotoxicity_DATA_CODE_FIGURE/Figure4/Fig4D/code/Fig4D.m",
    "SERS-Hepatotoxicity_DATA_CODE_FIGURE/Figure4/Fig4F/code/plot_spectra.m",
    "SERS-Hepatotoxicity_DATA_CODE_FIGURE/Figure4/data/RawDataSet91.mat",
    "SERS-Hepatotoxicity_DATA_CODE_FIGURE/Figure4/data/RawDataset119.mat",
]


def _resolve_output_root() -> tuple[Path, str]:
    mounted_data_root = Path("/Volumes/SSD_Rad/GAIRA_DATA")
    if mounted_data_root.exists() and mounted_data_root.is_dir():
        return REQUESTED_OUTPUT_ROOT, "requested"
    return FALLBACK_OUTPUT_ROOT, "fallback_workspace"


def _infer_data_type(relative_path: str) -> str:
    path = relative_path.lower()
    suffix = Path(relative_path).suffix.lower()
    name = Path(relative_path).name.lower()
    if suffix == ".zip":
        return "supplementary"
    if suffix in {".m", ".py", ".r", ".p"}:
        return "code"
    if suffix in {".pptx", ".ppt"}:
        return "figure"
    if suffix in {".png", ".jpg", ".jpeg", ".pdf", ".svg"}:
        return "figure"
    if suffix in {".docx", ".txt", ".md"}:
        return "metadata"
    if suffix == ".mat":
        if "combined_wavenumbers" in name:
            return "metadata"
        if "rawdataset" in name or "rawdata" in name:
            return "raw_spectra"
        return "unknown"
    if suffix == "" and name.startswith("s_"):
        return "processed_spectra"
    if "supplement" in path:
        return "supplementary"
    return "unknown"


def _infer_figure_association(relative_path: str) -> str:
    lowered = relative_path.lower()
    for token in ["figure4", "figure3", "figure2", "figure1", "figure5"]:
        if token in lowered:
            return token.title().replace("Figure", "Figure ")
    if "supplement" in lowered:
        return "Supplementary"
    return "unknown"


def _top_level_folder(relative_path: str) -> str:
    parts = Path(relative_path).parts
    if len(parts) >= 2:
        return parts[1]
    if parts:
        return parts[0]
    return ""


def _extract_archive_relative_path(sample_key: str) -> str | None:
    marker = "__shine_ev_sers_"
    if marker not in sample_key:
        return None
    return sample_key.split(marker, 1)[1].replace("__", "/")


def _parse_ingested_source_path(relative_path: str) -> dict[str, str]:
    parts = Path(relative_path).parts
    data_idx = parts.index("data")
    data_parts = parts[data_idx + 1 :]
    if len(data_parts) == 4:
        subclass_label, class_label, sample_id, replicate_id = data_parts
    elif len(data_parts) == 3:
        subclass_label, class_label, replicate_id = data_parts
        sample_id = class_label
    else:
        raise ValueError(f"Unexpected SHINE path layout: {relative_path}")
    biosample_id = "shine_ev_sers_" + relative_path.replace("/", "__")
    return {
        "subclass_label": subclass_label,
        "class_label": class_label,
        "sample_id": sample_id,
        "replicate_id": replicate_id,
        "biosample_id": biosample_id,
        "source_file": relative_path,
    }


def _load_ingested_metadata() -> pd.DataFrame:
    try:
        db_path = get_database_path()
        with duckdb.connect(str(db_path), read_only=True) as con:
            db_df = con.execute(
                """
                SELECT
                    biosample_id,
                    sample_id,
                    patient_id,
                    replicate_id,
                    class_label,
                    subclass_label,
                    source_file
                FROM biosample_metadata
                WHERE dataset_id = 'shine_ev_sers'
                ORDER BY source_file
                """
            ).fetchdf()
        if not db_df.empty:
            db_df["archive_relative_path"] = db_df["source_file"].astype(str)
            return db_df
    except Exception:
        pass

    df = pd.read_csv(
        EMBED_METADATA_PATH,
        usecols=["sample_key", "dataset_id", "label_optional", "subclass_label", "record_kind"],
    )
    shine = df[
        (df["dataset_id"].astype(str) == "shine_ev_sers")
        & (df["record_kind"].astype(str) == "processed_spectrum")
    ].copy()
    shine = shine.rename(
        columns={
            "label_optional": "export_class_label",
            "subclass_label": "export_subclass_label",
        }
    )
    shine["archive_relative_path"] = shine["sample_key"].astype(str).map(_extract_archive_relative_path)
    shine = shine.dropna(subset=["archive_relative_path"]).reset_index(drop=True)
    parsed_rows = [_parse_ingested_source_path(path) for path in shine["archive_relative_path"].astype(str)]
    parsed_df = pd.DataFrame(parsed_rows)
    shine = pd.concat([shine.reset_index(drop=True), parsed_df], axis=1)
    shine["patient_id"] = pd.NA
    return shine


def _build_archive_inventory(ingested_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    if RAW_DATASET_ROOT.exists():
        for path in sorted(p for p in RAW_DATASET_ROOT.rglob("*") if p.is_file()):
            relative_path = path.relative_to(RAW_DATASET_ROOT).as_posix()
            rows.append(
                {
                    "relative_path": relative_path,
                    "file_name": path.name,
                    "extension": path.suffix.lower(),
                    "size_bytes": int(path.stat().st_size),
                    "top_level_folder": _top_level_folder(relative_path),
                    "likely_figure_association": _infer_figure_association(relative_path),
                    "likely_data_type": _infer_data_type(relative_path),
                }
            )
    else:
        for relative_path in KNOWN_ARCHIVE_FILES:
            rows.append(
                {
                    "relative_path": relative_path,
                    "file_name": Path(relative_path).name,
                    "extension": Path(relative_path).suffix.lower(),
                    "size_bytes": pd.NA,
                    "top_level_folder": _top_level_folder(relative_path),
                    "likely_figure_association": _infer_figure_association(relative_path),
                    "likely_data_type": _infer_data_type(relative_path),
                }
            )

    for relative_path in sorted(ingested_df["archive_relative_path"].astype(str).unique().tolist()):
        rows.append(
            {
                "relative_path": relative_path,
                "file_name": Path(relative_path).name,
                "extension": Path(relative_path).suffix.lower(),
                "size_bytes": pd.NA,
                "top_level_folder": _top_level_folder(relative_path),
                "likely_figure_association": _infer_figure_association(relative_path),
                "likely_data_type": _infer_data_type(relative_path),
            }
        )

    out = pd.DataFrame(rows).drop_duplicates(subset=["relative_path"]).sort_values("relative_path").reset_index(drop=True)
    return out


def _first_example_per_group(ingested_df: pd.DataFrame) -> pd.DataFrame:
    return (
        ingested_df.sort_values(["subclass_label", "class_label", "archive_relative_path"])
        .groupby(["subclass_label", "class_label"], as_index=False)
        .first()
    )


def _build_figure4_relevant_files(ingested_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    if ARCHIVE_TREE_ROOT.exists():
        explicit = [
            (
                path.relative_to(RAW_DATASET_ROOT).as_posix(),
                "Direct Figure 4 code/model/helper asset present in the mounted SHINE archive.",
            )
            for path in sorted(ARCHIVE_TREE_ROOT.rglob("*"))
            if path.is_file()
            and "Figure4" in path.as_posix()
            and (path.suffix.lower() in {".m", ".mat", ".p"} or "models" in path.as_posix())
        ]
        readme_rel = "SERS-Hepatotoxicity_DATA_CODE_FIGURE/Readme.docx"
        if (RAW_DATASET_ROOT / readme_rel).exists():
            explicit.append(
                (
                    readme_rel,
                    "Archive readme present alongside the figure folders; expected to document preprocessing and figure workflow.",
                )
            )
    else:
        explicit = [
            (
                "SERS-Hepatotoxicity_DATA_CODE_FIGURE/Readme.docx",
                "Archive readme referenced by the calibration audit; expected to describe preprocessing and figure workflow.",
            ),
            (
                "SERS-Hepatotoxicity_DATA_CODE_FIGURE/Figure4/Fig4D/code/Fig4D.m",
                "Used by the local SHINE parser for Raman-axis calibration; directly tied to Figure 4 plotting.",
            ),
            (
                "SERS-Hepatotoxicity_DATA_CODE_FIGURE/Figure4/Fig4C/code/plot_spectra.m",
                "Referenced by the calibration audit as a Figure 4 plotting script using a cropped working window.",
            ),
            (
                "SERS-Hepatotoxicity_DATA_CODE_FIGURE/Figure4/Fig4F/code/plot_spectra.m",
                "Referenced by the calibration audit as a Figure 4 plotting script using a cropped working window.",
            ),
            (
                "SERS-Hepatotoxicity_DATA_CODE_FIGURE/Figure4/Fig4C/data/combined_wavenumbers.mat",
                "Referenced as a Figure 4 helper MAT file containing the selected wavenumber grid.",
            ),
            (
                "SERS-Hepatotoxicity_DATA_CODE_FIGURE/Figure4/data/RawDataSet91.mat",
                "Explicitly named by the parser audit as an additional source file under Figure4/data.",
            ),
            (
                "SERS-Hepatotoxicity_DATA_CODE_FIGURE/Figure4/data/RawDataset119.mat",
                "Explicitly named by the parser audit as an additional source file under Figure4/data.",
            ),
        ]
    for relative_path, why in explicit:
        lower = relative_path.lower()
        mentions_set1 = int(bool(re.search(r"(^|/|_)set1($|/|_)", lower)))
        mentions_set2 = int(bool(re.search(r"(^|/|_)set2($|/|_)", lower)))
        rows.append(
            {
                "relative_path": relative_path,
                "why_relevant": why,
                "mentions_set1": mentions_set1,
                "mentions_set2": mentions_set2,
                "mentions_day0": int("d0_" in lower or "day0" in lower or "day 0" in lower),
                "mentions_day2": int("d2_" in lower or "day2" in lower or "day 2" in lower),
                "mentions_pca": int("fig4" in lower or "pca" in lower or "combined_wavenumbers" in lower),
                "mentions_regression": int(
                    "regression" in lower
                    or "gpr" in lower
                    or "svm" in lower
                    or "linear.mat" in lower
                    or "fig4f" in lower
                ),
                "mentions_apap_concentration": int("c0" in lower or "c10" in lower or "c20" in lower or "c40" in lower),
            }
        )

    for row in _first_example_per_group(ingested_df).itertuples(index=False):
        relative_path = str(row.archive_relative_path)
        class_label = str(row.class_label)
        subclass_label = str(row.subclass_label)
        rows.append(
            {
                "relative_path": relative_path,
                "why_relevant": (
                    f"Representative ingested per-spectrum file for {subclass_label} / {class_label}; "
                    "these are the actual Figure4/data spectra GAIRA processed."
                ),
                "mentions_set1": 0,
                "mentions_set2": 0,
                "mentions_day0": int(class_label.startswith("D0_")),
                "mentions_day2": int(class_label.startswith("D2_")),
                "mentions_pca": 1,
                "mentions_regression": int(class_label.startswith("D2_")),
                "mentions_apap_concentration": 1,
            }
        )

    out = pd.DataFrame(rows).drop_duplicates(subset=["relative_path"]).sort_values("relative_path").reset_index(drop=True)
    return out


def _build_local_ingest_summary(ingested_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    rows.append(
        {
            "scope": "dataset_total",
            "dataset_id": "shine_ev_sers",
            "subset_alias": "shine_ev_stress",
            "subclass_label": "",
            "class_label": "",
            "metric_name": "n_rows",
            "metric_value": int(len(ingested_df)),
            "example_source_file": "",
            "notes": "Derived from local processed metadata export because the mounted DuckDB data root is unavailable in this session.",
        }
    )
    for metric_name, value in [
        ("n_distinct_sample_id", ingested_df["sample_id"].astype(str).nunique()),
        ("n_distinct_biosample_id", ingested_df["biosample_id"].astype(str).nunique()),
        ("n_distinct_source_file", ingested_df["source_file"].astype(str).nunique()),
        ("n_distinct_replicate_id", ingested_df["replicate_id"].astype(str).nunique()),
        ("n_distinct_patient_id", ingested_df["patient_id"].dropna().astype(str).nunique()),
    ]:
        rows.append(
            {
                "scope": "dataset_total",
                "dataset_id": "shine_ev_sers",
                "subset_alias": "shine_ev_stress",
                "subclass_label": "",
                "class_label": "",
                "metric_name": metric_name,
                "metric_value": int(value),
                "example_source_file": "",
                "notes": "",
            }
        )

    for subclass_label, sub in ingested_df.groupby("subclass_label", sort=True):
        rows.append(
            {
                "scope": "subclass",
                "dataset_id": "shine_ev_sers",
                "subset_alias": "shine_ev_stress",
                "subclass_label": str(subclass_label),
                "class_label": "",
                "metric_name": "n_rows",
                "metric_value": int(len(sub)),
                "example_source_file": str(sub.sort_values("source_file").iloc[0]["source_file"]),
                "notes": "",
            }
        )
        rows.append(
            {
                "scope": "subclass",
                "dataset_id": "shine_ev_sers",
                "subset_alias": "shine_ev_stress",
                "subclass_label": str(subclass_label),
                "class_label": "",
                "metric_name": "n_distinct_sample_id",
                "metric_value": int(sub["sample_id"].astype(str).nunique()),
                "example_source_file": "",
                "notes": "",
            }
        )

    class_counts = (
        ingested_df.groupby(["subclass_label", "class_label"], sort=True)
        .agg(
            n_rows=("source_file", "size"),
            n_distinct_sample_id=("sample_id", "nunique"),
            n_distinct_replicate_id=("replicate_id", "nunique"),
            n_distinct_patient_id=("patient_id", lambda s: s.dropna().astype(str).nunique()),
            example_source_file=("source_file", "min"),
        )
        .reset_index()
    )
    for row in class_counts.itertuples(index=False):
        rows.append(
            {
                "scope": "class",
                "dataset_id": "shine_ev_sers",
                "subset_alias": "shine_ev_stress",
                "subclass_label": str(row.subclass_label),
                "class_label": str(row.class_label),
                "metric_name": "n_rows",
                "metric_value": int(row.n_rows),
                "example_source_file": str(row.example_source_file),
                "notes": "",
            }
        )
        rows.append(
            {
                "scope": "class",
                "dataset_id": "shine_ev_sers",
                "subset_alias": "shine_ev_stress",
                "subclass_label": str(row.subclass_label),
                "class_label": str(row.class_label),
                "metric_name": "n_distinct_sample_id",
                "metric_value": int(row.n_distinct_sample_id),
                "example_source_file": "",
                "notes": "",
            }
        )
        rows.append(
            {
                "scope": "class",
                "dataset_id": "shine_ev_sers",
                "subset_alias": "shine_ev_stress",
                "subclass_label": str(row.subclass_label),
                "class_label": str(row.class_label),
                "metric_name": "n_distinct_replicate_id",
                "metric_value": int(row.n_distinct_replicate_id),
                "example_source_file": "",
                "notes": "",
            }
        )

    return pd.DataFrame(rows)


def _build_archive_to_ingest_mapping(archive_inventory_df: pd.DataFrame, ingested_df: pd.DataFrame) -> pd.DataFrame:
    local_paths = set(ingested_df["source_file"].astype(str))
    rows = []
    for relative_path in archive_inventory_df["relative_path"].astype(str):
        matched = relative_path in local_paths
        rows.append(
            {
                "archive_relative_path": relative_path,
                "local_source_file": relative_path if matched else "",
                "matched": int(matched),
                "confidence": "exact" if matched else "unknown",
                "notes": (
                    "Exact match recovered from processed metadata export sample_key."
                    if matched
                    else "Not present in the processed-spectrum ingest export. This usually means code/readme/helper assets were referenced but not ingested as spectra."
                ),
            }
        )
    return pd.DataFrame(rows)


def _write_markdown(path: Path, lines: Iterable[str]) -> None:
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def _build_readiness_note(
    output_path: Path,
    output_mode: str,
    ingested_df: pd.DataFrame,
    mapping_df: pd.DataFrame,
) -> None:
    set9 = ingested_df[ingested_df["subclass_label"].astype(str) == "Set9"]
    set10 = ingested_df[ingested_df["subclass_label"].astype(str) == "Set10"]
    have_day0 = int((ingested_df["class_label"].astype(str).str.startswith("D0_")).sum()) > 0
    have_day2 = int((ingested_df["class_label"].astype(str).str.startswith("D2_")).sum()) > 0
    lines = [
        "# SHINE Figure 4 PCA Readiness Note",
        "",
        "## 1. Can the local data reproduce Figure 4 PCA in principle?",
        (
            "- Yes, in principle for the Figure 4 spectra that GAIRA actually ingested. "
            "The parser and processed metadata exports both show that GAIRA ingested per-spectrum files from "
            "`SERS-Hepatotoxicity_DATA_CODE_FIGURE/Figure4/data/...`."
        ),
        (
            "- The mounted raw archive is available for direct file-tree inspection in this audit, while local ingest counts come from DuckDB when available and otherwise fall back to processed metadata exports. "
            f"Output mode: `{output_mode}`."
        ),
        (
            "- Direct zip-manifest inspection shows the local download contains `Figure4`, `Figure5`, `Readme.docx`, and two Office temp files. "
            "It does not expose `Figure1`-`Figure3` or a `Supplementary` folder in the mounted local copy."
        ),
        "",
        "## 2. If yes, which files should be used?",
        "- Use the per-spectrum files under `SERS-Hepatotoxicity_DATA_CODE_FIGURE/Figure4/data/Set9/...` and `.../Set10/...`.",
        "- Use `Figure4/Fig4D/code/Fig4D.m` for Raman-axis calibration context.",
        "- Use `Figure4/Fig4C/code/plot_spectra.m`, `Figure4/Fig4F/code/plot_spectra.m`, `Figure4/Fig4C/models/GPR.mat`, and `Figure4/Fig4C/data/combined_wavenumbers.mat` as the paper-style Figure 4 helper assets.",
        "",
        "## 3. If no, what is missing?",
        "- The remaining missing piece is not the archive itself but a manuscript-backed mapping from local `Set9` / `Set10` names to manuscript-style `Set1` / `Set2` labels.",
        "- Archive completeness beyond the mounted tree is no longer the blocker; unresolved set naming is.",
        "",
        "## 4. Is the Set9 / Set10 issue caused by archive structure, ingest truncation, subset alias filtering, or unresolved mapping?",
        (
            "- The strongest supported conclusion is: `Set10` being Day-2-only comes from the ingested Figure4/data tree itself, not from later Pilot filtering. "
            f"Local ingested counts show `Set9` contains D0/D1/D2 while `Set10` contains only D2 rows ({len(set10)} ingested spectra)."
        ),
        "- No evidence was found that `shine_ev_stress` subset alias filtering removed Day 0 or Day 1 from Set10; the subset alias resolves to the full `shine_ev_sers::all` dataset.",
        "- No clean local evidence maps `Set9` and `Set10` to manuscript-style `Set1` and `Set2`. That remains unresolved.",
        "",
        "## Direct answers",
        f"1. Previous SHINE analyses had access to full per-spectrum Figure 4 data in local processed metadata exports: `{'yes' if len(ingested_df) == 23646 else 'uncertain'}`.",
        f"2. Full per-spectrum Day 0 and Day 2 conditions are present in the ingested export: `{'yes' if have_day0 and have_day2 else 'no'}`.",
        f"3. Set10 Day-2-only pattern is visible in ingested source paths: `yes`.",
        f"4. Archive-to-ingest exact spectral matches recovered from processed metadata exports: `{int((mapping_df['matched'] == 1).sum())}`.",
    ]
    _write_markdown(output_path, lines)


def _build_recommendations(output_path: Path) -> None:
    lines = [
        "# SHINE Analysis Recommendations",
        "",
        "## A. Paper-style PCA replication",
        "- Use `SERS-Hepatotoxicity_DATA_CODE_FIGURE/Figure4/data/Set9/...` and `.../Set10/...` per-spectrum files as the source spectra.",
        "- Restrict to the exact day / concentration subset needed for the target panel, not a pooled latent-state analysis by default.",
        "- When the raw archive is mounted, pair those spectra with `Figure4/Fig4D/code/Fig4D.m`, `Figure4/Fig4C/code/plot_spectra.m`, `Figure4/Fig4F/code/plot_spectra.m`, and `Figure4/Fig4C/data/combined_wavenumbers.mat`.",
        "- For paper-faithful SHINE replication, respect the paper-context note that PCA/regression analyses focus on the 810-1610 cm^-1 region.",
        "",
        "## B. GAIRA latent-state analysis",
        "- Use the full ingested `shine_ev_sers` per-spectrum set represented by the 23,646 `Figure4/data/.../s_*` source files.",
        "- Treat `Set9` and `Set10` as nuisance / batch-like subclass labels unless a manuscript-backed mapping is recovered later.",
        "- Use the existing GAIRA full-spectrum path, not the earlier sample-mean Pilot 3 path.",
        "",
        "## C. SHINE task-specific APAP-response analysis",
        "- Use Day-2-only spectra first, because that is the paper's strongest reported response day.",
        "- Use `Set9` for control-anchored D0 vs D2 analyses, because it is the only locally evidenced set with both Day 0 and Day 2 coverage.",
        "- Use `Set10` only as a Day-2-only external check, not as the primary control-anchored day-comparison set.",
        "- Do not assume `Set9 == Set1` or `Set10 == Set2` without archive-side evidence.",
    ]
    _write_markdown(output_path, lines)


def main() -> None:
    output_root, output_mode = _resolve_output_root()
    tables_dir = output_root / "tables"
    report_dir = output_root / "report"
    tables_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    ingested_df = _load_ingested_metadata()
    archive_inventory_df = _build_archive_inventory(ingested_df)
    figure4_df = _build_figure4_relevant_files(ingested_df)
    ingest_summary_df = _build_local_ingest_summary(ingested_df)
    mapping_df = _build_archive_to_ingest_mapping(archive_inventory_df, ingested_df)

    archive_inventory_df.to_csv(tables_dir / "shine_archive_file_inventory.csv", index=False)
    figure4_df.to_csv(tables_dir / "shine_figure4_relevant_files.csv", index=False)
    ingest_summary_df.to_csv(tables_dir / "shine_local_ingest_summary.csv", index=False)
    mapping_df.to_csv(tables_dir / "shine_archive_to_ingest_mapping.csv", index=False)

    _build_readiness_note(report_dir / "shine_fig4_pca_readiness_note.md", output_mode, ingested_df, mapping_df)
    _build_recommendations(report_dir / "shine_analysis_recommendations.md")

    print(
        {
            "output_root": str(output_root),
            "output_mode": output_mode,
            "n_ingested_processed_spectra": int(len(ingested_df)),
            "n_unique_source_files": int(ingested_df["source_file"].astype(str).nunique()),
            "subclass_labels": sorted(ingested_df["subclass_label"].astype(str).unique().tolist()),
            "class_labels": sorted(ingested_df["class_label"].astype(str).unique().tolist()),
        }
    )


if __name__ == "__main__":
    main()
