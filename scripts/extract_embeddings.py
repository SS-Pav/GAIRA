from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    import sys

    sys.path.insert(0, str(project_root / "src"))
    from gaira.embedding.runtime import REMOTE_OUTPUT_ROOT, add_common_io_args

    parser = argparse.ArgumentParser(description="Extract frozen embeddings from a trained encoder.")
    add_common_io_args(parser, default_run_name="embedding_v2", default_root=REMOTE_OUTPUT_ROOT)
    parser.add_argument("--batch-size", type=int, default=None, help="Override extraction batch size.")
    return parser.parse_args()


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    import sys

    sys.path.insert(0, str(project_root / "src"))
    from gaira.embedding.model import RamanEncoder
    from gaira.embedding.runtime import detect_device, recommended_batch_size, resolve_dataset_path, resolve_output_dir

    args = parse_args()
    output_dir = resolve_output_dir(args)
    dataset_path = resolve_dataset_path(args, output_dir)
    checkpoint_path = output_dir / "model.pt"

    dataset = np.load(dataset_path, allow_pickle=True)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    device = detect_device()
    batch_size = args.batch_size or max(recommended_batch_size(device), 64)
    model = RamanEncoder(input_len=int(checkpoint["input_len"]))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    X = torch.from_numpy(dataset["X"].astype(np.float32))
    loader = DataLoader(TensorDataset(X), batch_size=batch_size, shuffle=False)
    embeddings: list[np.ndarray] = []

    with torch.no_grad():
        for (batch,) in loader:
            batch = batch.to(device)
            embeddings.append(model(batch).cpu().numpy())
    embedding_array = np.vstack(embeddings).astype(np.float32)
    np.save(output_dir / "embeddings.npy", embedding_array)

    metadata_df = pd.DataFrame(
        {
            "sample_key": dataset["sample_keys"].astype(str),
            "dataset_id": dataset["dataset_ids"].astype(str),
            "sample_type": dataset["sample_types"].astype(str),
            "label_optional": dataset["labels_optional"].astype(str),
            "family_label": dataset["family_labels"].astype(str)
            if "family_labels" in dataset.files
            else np.asarray([""] * len(dataset["dataset_ids"]), dtype=object),
            "semantic_group": dataset["semantic_groups"].astype(str)
            if "semantic_groups" in dataset.files
            else np.asarray([""] * len(dataset["dataset_ids"]), dtype=object),
            "hard_negative_scope": dataset["hard_negative_scopes"].astype(str)
            if "hard_negative_scopes" in dataset.files
            else np.asarray([""] * len(dataset["dataset_ids"]), dtype=object),
            "record_kind": dataset["record_kinds"].astype(str),
            "processing_version": dataset["processing_versions"].astype(str),
            "subclass_label": dataset["subclasses"].astype(str),
        }
    )
    metadata_df.to_csv(output_dir / "metadata.csv", index=False)
    print(f"Saved embeddings: {output_dir / 'embeddings.npy'}")


if __name__ == "__main__":
    main()
