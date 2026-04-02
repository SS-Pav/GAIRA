from __future__ import annotations

from pathlib import Path

import pandas as pd

from gaira.demo.v8_analysis_utils import (
    EV_STRESS_V1_DIR,
    SMALL2023_V1_DIR,
    V5_EVAL_DIR,
    V5_RUN_DIR,
    V6_EVAL_DIR,
    V6_RUN_DIR,
    V7_CLUSTER_DIR,
    V7_EVAL_DIR,
    V7_GROUNDING_DIR,
    V7_RUN_DIR,
)


MASTER_SHARED_DIR = Path("/Volumes/SSD_Rad/GAIRA_DATA/processed/v8_master_shared_backbone_diagnostics")
MASTER_EV_DIR = Path("/Volumes/SSD_Rad/GAIRA_DATA/processed/v8_ev_stress_prep")
MASTER_SMALL_DIR = Path("/Volumes/SSD_Rad/GAIRA_DATA/processed/v8_small2023_specialized_prep")
MASTER_SERUM_DIR = Path("/Volumes/SSD_Rad/GAIRA_DATA/processed/v8_serum_cohort_prep")
MASTER_REPORT_DIR = Path("/Volumes/SSD_Rad/GAIRA_DATA/processed/v8_master_report")

SHARED_V1_DIR = Path("/Volumes/SSD_Rad/GAIRA_DATA/processed/v8_shared_backbone_diagnostics_v1")
SERUM_STRESS_V1_DIR = Path("/Volumes/SSD_Rad/GAIRA_DATA/processed/v8_serum_stress_analysis_v1")
SERUM_DELTA_V1_DIR = Path("/Volumes/SSD_Rad/GAIRA_DATA/processed/v8_serum_delta_analysis_v1")
SMALL2023_BENCHMARK_V1_DIR = Path("/Volumes/SSD_Rad/GAIRA_DATA/processed/v8_small2023_benchmark_v1")


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def safe_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def safe_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore") if path.exists() else ""


def dataset_status_rows() -> pd.DataFrame:
    rows = []
    for name, run_dir, eval_dir in [
        ("v5_shared", V5_RUN_DIR, V5_EVAL_DIR),
        ("v6_shared", V6_RUN_DIR, V6_EVAL_DIR),
        ("v7_shared", V7_RUN_DIR, V7_EVAL_DIR),
    ]:
        rows.append(
            {
                "artifact": name,
                "run_dir": str(run_dir),
                "eval_dir": str(eval_dir),
                "status": "available" if run_dir.exists() and eval_dir.exists() else "missing",
            }
        )
    rows.extend(
        [
            {
                "artifact": "v7_cluster_analysis",
                "run_dir": str(V7_CLUSTER_DIR),
                "eval_dir": "",
                "status": "available" if V7_CLUSTER_DIR.exists() else "missing",
            },
            {
                "artifact": "v7_grounding_analysis",
                "run_dir": str(V7_GROUNDING_DIR),
                "eval_dir": "",
                "status": "available" if V7_GROUNDING_DIR.exists() else "missing",
            },
            {
                "artifact": "ev_stress_v1",
                "run_dir": str(EV_STRESS_V1_DIR),
                "eval_dir": "",
                "status": "available" if EV_STRESS_V1_DIR.exists() else "missing",
            },
            {
                "artifact": "small2023_benchmark_v1",
                "run_dir": str(SMALL2023_BENCHMARK_V1_DIR),
                "eval_dir": "",
                "status": "available" if SMALL2023_BENCHMARK_V1_DIR.exists() else "missing",
            },
            {
                "artifact": "serum_stress_v1",
                "run_dir": str(SERUM_STRESS_V1_DIR),
                "eval_dir": "",
                "status": "available" if SERUM_STRESS_V1_DIR.exists() else "missing",
            },
            {
                "artifact": "serum_delta_v1",
                "run_dir": str(SERUM_DELTA_V1_DIR),
                "eval_dir": "",
                "status": "available" if SERUM_DELTA_V1_DIR.exists() else "missing",
            },
        ]
    )
    return pd.DataFrame(rows)


def copy_if_exists(source: Path, dest: Path) -> bool:
    if not source.exists():
        return False
    dest.write_bytes(source.read_bytes())
    return True


def first_existing(paths: list[Path]) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None
