from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch


LOCAL_OUTPUT_ROOT = Path("/Volumes/SSD_Rad/GAIRA_DATA/processed")
REMOTE_OUTPUT_ROOT = Path.home() / "projects" / "GAIRA" / "data" / "processed"


def embedding_output_dir(
    run_name: str,
    *,
    cli_value: str | None = None,
    env_var: str = "GAIRAM_EMBEDDING_OUTPUT_DIR",
    default_root: Path | None = None,
) -> Path:
    if cli_value:
        return Path(cli_value).expanduser().resolve()
    env_value = os.environ.get(env_var)
    if env_value:
        return Path(env_value).expanduser().resolve()
    root = default_root or REMOTE_OUTPUT_ROOT
    return (root / run_name).resolve()


def embedding_dataset_path(
    output_dir: Path,
    *,
    cli_value: str | None = None,
    env_var: str = "GAIRAM_EMBEDDING_DATASET_PATH",
    filename: str = "embedding_dataset.npz",
) -> Path:
    if cli_value:
        return Path(cli_value).expanduser().resolve()
    env_value = os.environ.get(env_var)
    if env_value:
        return Path(env_value).expanduser().resolve()
    return (output_dir / filename).resolve()


def detect_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    mps_backend = getattr(torch.backends, "mps", None)
    if mps_backend is not None and mps_backend.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def recommended_batch_size(device: torch.device) -> int:
    if device.type == "cuda":
        return 64
    if device.type == "mps":
        return 48
    return 16


def add_common_io_args(
    parser: argparse.ArgumentParser,
    *,
    default_run_name: str,
    default_root: Path | None = None,
    include_dataset_path: bool = True,
) -> None:
    parser.add_argument("--output-dir", default=None, help="Output directory for this run.")
    if include_dataset_path:
        parser.add_argument("--dataset-path", default=None, help="Path to embedding_dataset.npz.")
    parser.set_defaults(_default_run_name=default_run_name, _default_root=default_root)


def resolve_output_dir(args: argparse.Namespace) -> Path:
    return embedding_output_dir(
        getattr(args, "_default_run_name"),
        cli_value=getattr(args, "output_dir", None),
        default_root=getattr(args, "_default_root", None),
    )


def resolve_dataset_path(args: argparse.Namespace, output_dir: Path) -> Path:
    return embedding_dataset_path(output_dir, cli_value=getattr(args, "dataset_path", None))


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
