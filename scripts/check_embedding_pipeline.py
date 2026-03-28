from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch


REQUIRED_FILES = [
    "embedding_dataset.npz",
    "dataset_summary.csv",
    "run_config.json",
    "model.pt",
    "training_log.csv",
    "embeddings.npy",
    "metadata.csv",
    "umap_sample_type.png",
    "umap_dataset.png",
    "umap_class.png",
    "embedding_metrics.csv",
    "embedding_report.md",
    "probe_metrics.csv",
    "probe_report.md",
]


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(project_root / "src"))
    from gaira.embedding.runtime import REMOTE_OUTPUT_ROOT, add_common_io_args

    parser = argparse.ArgumentParser(description="Validate an embedding training run.")
    add_common_io_args(parser, default_run_name="embedding_v2", default_root=REMOTE_OUTPUT_ROOT)
    return parser.parse_args()


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(project_root / "src"))
    from gaira.embedding.model import RamanEncoder
    from gaira.embedding.runtime import resolve_output_dir, resolve_dataset_path

    args = parse_args()
    output_dir = resolve_output_dir(args)
    dataset_path = resolve_dataset_path(args, output_dir)

    missing = [str(output_dir / name) for name in REQUIRED_FILES if not (output_dir / name).exists()]
    if missing:
        raise FileNotFoundError("Missing embedding outputs:\n" + "\n".join(missing))

    dataset = np.load(dataset_path, allow_pickle=True)
    X = dataset["X"]
    if X.ndim != 2 or X.shape[0] == 0:
        raise ValueError("Embedding dataset is empty or malformed.")

    checkpoint = torch.load(output_dir / "model.pt", map_location="cpu")
    model = RamanEncoder(input_len=int(checkpoint["input_len"]))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    with torch.no_grad():
        sample = torch.from_numpy(X[:4].astype(np.float32))
        embedding = model(sample)
    if embedding.shape[0] != min(4, X.shape[0]):
        raise ValueError("Model forward pass returned unexpected shape.")

    log_df = pd.read_csv(output_dir / "training_log.csv")
    if log_df.empty:
        raise ValueError("Training log is empty.")
    loss_decreased = float(log_df["loss"].iloc[-1]) <= float(log_df["loss"].iloc[0])

    metadata_df = pd.read_csv(output_dir / "metadata.csv")
    embeddings = np.load(output_dir / "embeddings.npy")
    if embeddings.shape[0] != len(metadata_df):
        raise ValueError("Embedding rows do not match metadata rows.")

    print("embedding dataset: ok")
    print(f"output_dir={output_dir}")
    print(f"samples={X.shape[0]} input_len={X.shape[1]}")
    print(f"forward_shape={tuple(embedding.shape)}")
    print(f"loss_first={log_df['loss'].iloc[0]:.6f} loss_last={log_df['loss'].iloc[-1]:.6f}")
    print(f"loss_decreased={loss_decreased}")
    print(f"embeddings_shape={embeddings.shape}")


if __name__ == "__main__":
    main()
