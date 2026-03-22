import csv
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import duckdb
import pandas as pd

PROCESS_BIOSAMPLE_DATASETS = [
    "shine_ev_sers",
    "small2023_ev",
    "diabetes_plasma_ev_sers",
    "serum_ag_colloids",
    "serum_protocol_comparison",
    "cspp_serum",
    "ergothioneine_serum",
]
PROCESS_GROUNDING_DATASETS = [
    "serum_ag_colloids_grounding",
]
LEGACY_VOLUME_TOKEN = "SSD" + "_SPG"
TMP_TOKEN = "/" + "tmp"


def run_python_script(
    project_root: Path,
    log_lines: list[str],
    script_name: str,
    *script_args: str,
    allow_failure: bool = False,
) -> bool:
    # Keep all GAIRA DuckDB writers strictly sequential. Do not parallelize these calls.
    command = [sys.executable, str(project_root / "scripts" / script_name), *script_args]
    log_lines.append(f"$ {' '.join(command)}")
    log_lines.append(f"Starting sequential step: {script_name} {' '.join(script_args)}")
    completed = subprocess.run(
        command,
        cwd=project_root,
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.stdout:
        log_lines.append(completed.stdout.rstrip())
    if completed.stderr:
        log_lines.append(completed.stderr.rstrip())
    if completed.returncode != 0 and not allow_failure:
        log_lines.append(f"[FAIL] {script_name} {' '.join(script_args)}")
        return False
    if completed.returncode != 0:
        log_lines.append(f"[WARN] {script_name} {' '.join(script_args)}")
        return False
    log_lines.append(f"[OK] {script_name} {' '.join(script_args)}")
    return True


def query_count(connection: duckdb.DuckDBPyConnection, sql: str, params: list[str]) -> int:
    return int(connection.execute(sql, params).fetchone()[0])


def collect_dataset_counts(db_path: Path, dataset_id: str) -> dict[str, int]:
    with duckdb.connect(str(db_path), read_only=True) as connection:
        return {
            "reference_metadata": query_count(
                connection,
                "SELECT COUNT(*) FROM reference_metadata WHERE dataset_id = ?",
                [dataset_id],
            ),
            "biosample_metadata": query_count(
                connection,
                "SELECT COUNT(*) FROM biosample_metadata WHERE dataset_id = ?",
                [dataset_id],
            ),
            "biosample_processed_spectra": query_count(
                connection,
                "SELECT COUNT(*) FROM biosample_processed_spectra WHERE dataset_id = ?",
                [dataset_id],
            ),
            "grounding_metadata": query_count(
                connection,
                "SELECT COUNT(*) FROM grounding_metadata WHERE dataset_id = ?",
                [dataset_id],
            ),
            "grounding_processed_spectra": query_count(
                connection,
                "SELECT COUNT(*) FROM grounding_processed_spectra WHERE dataset_id = ?",
                [dataset_id],
            ),
            "grounding_support_documents": query_count(
                connection,
                "SELECT COUNT(*) FROM grounding_support_documents WHERE dataset_id = ?",
                [dataset_id],
            ),
            "knowledge_sources": query_count(
                connection,
                "SELECT COUNT(*) FROM knowledge_sources WHERE dataset_id = ?",
                [dataset_id],
            ),
        }


def append_integrity_summary(project_root: Path, db_path: Path, log_lines: list[str]) -> dict[str, int]:
    with duckdb.connect(str(db_path), read_only=True) as connection:
        totals = {
            "reference_spectra": query_count(connection, "SELECT COUNT(*) FROM reference_spectra", []),
            "biosample_spectra": query_count(connection, "SELECT COUNT(*) FROM biosample_spectra", []),
            "biosample_processed_spectra": query_count(connection, "SELECT COUNT(*) FROM biosample_processed_spectra", []),
            "grounding_spectra": query_count(connection, "SELECT COUNT(*) FROM grounding_spectra", []),
            "grounding_processed_spectra": query_count(connection, "SELECT COUNT(*) FROM grounding_processed_spectra", []),
            "grounding_support_documents": query_count(connection, "SELECT COUNT(*) FROM grounding_support_documents", []),
            "domain_context_documents": query_count(connection, "SELECT COUNT(*) FROM domain_context_documents", []),
            "domain_context_chunks": query_count(connection, "SELECT COUNT(*) FROM domain_context_chunks", []),
        }

        bad_path_hits = {
            "biosample_metadata_notes": query_count(
                connection,
                """
                SELECT COUNT(*) FROM biosample_metadata
                WHERE notes LIKE ? OR notes LIKE ?
                   OR source_file LIKE ? OR source_file LIKE ?
                """,
                [f"%{LEGACY_VOLUME_TOKEN}%", f"%{TMP_TOKEN}%", f"%{LEGACY_VOLUME_TOKEN}%", f"%{TMP_TOKEN}%"],
            ),
            "grounding_support_documents_notes": query_count(
                connection,
                """
                SELECT COUNT(*) FROM grounding_support_documents
                WHERE notes LIKE ? OR notes LIKE ?
                   OR source_file LIKE ? OR source_file LIKE ?
                """,
                [f"%{LEGACY_VOLUME_TOKEN}%", f"%{TMP_TOKEN}%", f"%{LEGACY_VOLUME_TOKEN}%", f"%{TMP_TOKEN}%"],
            ),
            "domain_context_documents_notes": query_count(
                connection,
                """
                SELECT COUNT(*) FROM domain_context_documents
                WHERE notes LIKE ? OR notes LIKE ?
                   OR source_file LIKE ? OR source_file LIKE ?
                """,
                [f"%{LEGACY_VOLUME_TOKEN}%", f"%{TMP_TOKEN}%", f"%{LEGACY_VOLUME_TOKEN}%", f"%{TMP_TOKEN}%"],
            ),
        }

    repo_bad_strings = {}
    for pattern in (LEGACY_VOLUME_TOKEN, TMP_TOKEN):
        completed = subprocess.run(
            ["rg", "-n", pattern, ".", "-g", "!**/__pycache__/**"],
            cwd=project_root,
            text=True,
            capture_output=True,
            check=False,
        )
        repo_bad_strings[pattern] = 0 if completed.returncode == 1 else len(
            [line for line in completed.stdout.splitlines() if line.strip()]
        )

    log_lines.append("Integrity summary:")
    for key, value in totals.items():
        log_lines.append(f"  {key}: {value}")
    for key, value in bad_path_hits.items():
        log_lines.append(f"  {key}: {value}")
    for key, value in repo_bad_strings.items():
        log_lines.append(f"  repo_grep_{key}: {value}")
    return {**totals, **bad_path_hits, **{f"repo_grep_{k}": v for k, v in repo_bad_strings.items()}}


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(project_root / "src"))

    from gaira.config import get_database_path, get_storage_paths, initialize_storage_root

    started_at = datetime.now().astimezone()
    log_lines = [f"GAIRA SSD_Rad rebuild started: {started_at.isoformat()}"]

    storage_paths = initialize_storage_root()
    data_root = storage_paths["data_root"]
    db_path = get_database_path()
    logs_dir = storage_paths["logs"]
    log_path = logs_dir / "rebuild_log.txt"
    registry_path = project_root / "data" / "registry" / "datasets.csv"
    datasets_df = pd.read_csv(registry_path)

    log_lines.append(f"DATA_ROOT: {data_root}")
    log_lines.append(f"Database: {db_path}")
    holdout_datasets = {
        str(row["dataset_id"])
        for row in datasets_df.to_dict(orient="records")
        if str(row.get("status", "")).strip().lower() == "holdout"
    }

    log_lines.append(f"Holdout datasets skipped by default: {', '.join(sorted(holdout_datasets)) or 'none'}")
    log_lines.append("DuckDB write policy: sequential single-writer only. Do not parallelize rebuild steps.")

    if db_path.exists():
        db_path.unlink()
        log_lines.append(f"Removed existing database: {db_path}")

    if not run_python_script(project_root, log_lines, "init_database.py"):
        log_path.write_text("\n".join(log_lines), encoding="utf-8")
        raise SystemExit(1)

    dataset_results: list[dict] = []
    failures: list[str] = []

    for row in datasets_df.to_dict(orient="records"):
        dataset_id = str(row["dataset_id"])
        dataset_family = str(row["dataset_family"])
        if dataset_id in holdout_datasets:
            dataset_results.append({"dataset_id": dataset_id, "status": "skipped"})
            continue

        log_lines.append("")
        log_lines.append(f"=== Dataset: {dataset_id} ({dataset_family}) ===")

        download_ok = True
        if dataset_family != "knowledge":
            download_ok = run_python_script(project_root, log_lines, "download_dataset.py", dataset_id, allow_failure=True)
        else:
            log_lines.append("Skipping download step for local knowledge dataset.")

        ingest_ok = run_python_script(project_root, log_lines, "ingest_dataset.py", dataset_id, allow_failure=True)
        counts = collect_dataset_counts(db_path, dataset_id)
        dataset_results.append(
            {
                "dataset_id": dataset_id,
                "dataset_family": dataset_family,
                "download_ok": download_ok,
                "ingest_ok": ingest_ok,
                **counts,
            }
        )
        log_lines.append(f"Counts: {counts}")
        if not ingest_ok:
            failures.append(f"{dataset_id}: ingest failed")

    for dataset_id in PROCESS_BIOSAMPLE_DATASETS:
        if dataset_id in holdout_datasets:
            continue
        log_lines.append("")
        log_lines.append(f"=== Process biosample: {dataset_id} ===")
        ok = run_python_script(project_root, log_lines, "process_biosample_dataset.py", dataset_id, allow_failure=True)
        if not ok:
            failures.append(f"{dataset_id}: processing failed")

    for dataset_id in PROCESS_GROUNDING_DATASETS:
        log_lines.append("")
        log_lines.append(f"=== Process grounding: {dataset_id} ===")
        ok = run_python_script(project_root, log_lines, "process_grounding_dataset.py", dataset_id, allow_failure=True)
        if not ok:
            failures.append(f"{dataset_id}: grounding processing failed")

    for script_name in (
        "ingest_domain_context.py",
        "ingest_gaira_serum_context.py",
        "ingest_gaira_ev_context.py",
    ):
        log_lines.append("")
        log_lines.append(f"=== Context step: {script_name} ===")
        ok = run_python_script(project_root, log_lines, script_name, allow_failure=True)
        if not ok:
            failures.append(f"{script_name}: failed")

    log_lines.append("")
    log_lines.append("=== Functional inference test ===")
    inference_ok = run_python_script(project_root, log_lines, "run_gaira_inference_reranked_demo.py", allow_failure=True)
    if not inference_ok:
        failures.append("run_gaira_inference_reranked_demo.py: failed")

    integrity = append_integrity_summary(project_root, db_path, log_lines)

    summary_csv_path = storage_paths["exports"] / "rebuild_dataset_summary.csv"
    with summary_csv_path.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = sorted({key for row in dataset_results for key in row.keys()})
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(dataset_results)
    log_lines.append(f"Dataset summary CSV: {summary_csv_path}")

    log_lines.append("")
    log_lines.append("=== Backup snapshot ===")
    backup_ok = run_python_script(project_root, log_lines, "create_backup_snapshot.py", allow_failure=True)
    if not backup_ok:
        failures.append("create_backup_snapshot.py: failed")

    log_lines.append("")
    log_lines.append("Failures:")
    if failures:
        for failure in failures:
            log_lines.append(f"  {failure}")
    else:
        log_lines.append("  none")

    finished_at = datetime.now().astimezone()
    log_lines.append(f"GAIRA SSD_Rad rebuild finished: {finished_at.isoformat()}")
    log_path.write_text("\n".join(log_lines), encoding="utf-8")

    print(f"Rebuild log written to: {log_path}")
    print(f"Dataset summary written to: {summary_csv_path}")
    print("Top-level totals:")
    for key, value in integrity.items():
        print(f"  {key}: {value}")
    if failures:
        print("Failures:")
        for failure in failures:
            print(f"  {failure}")


if __name__ == "__main__":
    main()
