from __future__ import annotations

import argparse
import csv
import math
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler


DEFAULT_EPOCHS = 30
TEMPERATURE = 0.2
LEARNING_RATE = 1e-3


class SpectrumDataset(Dataset):
    def __init__(self, spectra: np.ndarray):
        self.spectra = torch.from_numpy(spectra.astype(np.float32))

    def __len__(self) -> int:
        return int(self.spectra.shape[0])

    def __getitem__(self, index: int) -> torch.Tensor:
        return self.spectra[index]


def nt_xent_loss(z1: torch.Tensor, z2: torch.Tensor, temperature: float = TEMPERATURE) -> torch.Tensor:
    batch_size = z1.shape[0]
    z = torch.cat([z1, z2], dim=0)
    similarity = F.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0), dim=2) / temperature
    mask = torch.eye(2 * batch_size, device=z.device, dtype=torch.bool)
    similarity = similarity.masked_fill(mask, -9e15)
    positive_indices = torch.arange(batch_size, device=z.device)
    positive_indices = torch.cat([positive_indices + batch_size, positive_indices], dim=0)
    return F.cross_entropy(similarity, positive_indices)


def build_sample_weights(sample_types: np.ndarray, dataset_ids: np.ndarray) -> np.ndarray:
    sample_type_counts = {key: count for key, count in zip(*np.unique(sample_types.astype(str), return_counts=True))}
    dataset_counts = {key: count for key, count in zip(*np.unique(dataset_ids.astype(str), return_counts=True))}
    weights = []
    for sample_type, dataset_id in zip(sample_types.astype(str), dataset_ids.astype(str), strict=False):
        weights.append(1.0 / math.sqrt(sample_type_counts[sample_type] * dataset_counts[dataset_id]))
    return np.asarray(weights, dtype=np.float32)


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    import sys

    sys.path.insert(0, str(project_root / "src"))
    from gaira.embedding.runtime import REMOTE_OUTPUT_ROOT, add_common_io_args

    parser = argparse.ArgumentParser(description="Train the GAIRAM spectral embedding encoder.")
    add_common_io_args(parser, default_run_name="embedding_v2", default_root=REMOTE_OUTPUT_ROOT)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS, help="Training epochs.")
    parser.add_argument("--batch-size", type=int, default=None, help="Override automatic batch size.")
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE, help="Optimizer learning rate.")
    parser.add_argument("--temperature", type=float, default=TEMPERATURE, help="NT-Xent temperature.")
    parser.add_argument("--seed", type=int, default=7, help="Random seed.")
    parser.add_argument("--max-steps-per-epoch", type=int, default=None, help="Optional step cap for smoke runs.")
    return parser.parse_args()


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    import sys

    sys.path.insert(0, str(project_root / "src"))
    from gaira.embedding.augmentations import augment_spectrum
    from gaira.embedding.model import RamanEncoder
    from gaira.embedding.runtime import (
        detect_device,
        recommended_batch_size,
        resolve_dataset_path,
        resolve_output_dir,
        write_json,
    )

    args = parse_args()
    output_dir = resolve_output_dir(args)
    dataset_path = resolve_dataset_path(args, output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    dataset = np.load(dataset_path, allow_pickle=True)
    X = dataset["X"]
    dataset_ids = dataset["dataset_ids"].astype(str)
    sample_types = dataset["sample_types"].astype(str)
    input_len = int(X.shape[1])
    device = detect_device()
    batch_size = args.batch_size or recommended_batch_size(device)

    training_dataset = SpectrumDataset(X)
    sample_weights = build_sample_weights(sample_types, dataset_ids)
    sampler = WeightedRandomSampler(
        torch.as_tensor(sample_weights, dtype=torch.double),
        num_samples=len(sample_weights),
        replacement=True,
    )
    train_loader = DataLoader(
        training_dataset,
        batch_size=batch_size,
        sampler=sampler,
        drop_last=True,
        num_workers=0,
    )

    model = RamanEncoder(input_len=input_len).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    run_config = {
        "dataset_path": str(dataset_path),
        "output_dir": str(output_dir),
        "device": device.type,
        "batch_size": batch_size,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "temperature": args.temperature,
        "seed": args.seed,
        "samples": int(X.shape[0]),
        "input_len": input_len,
        "sample_type_counts": {key: int(value) for key, value in zip(*np.unique(sample_types, return_counts=True))},
        "dataset_count": int(len(np.unique(dataset_ids))),
        "max_steps_per_epoch": args.max_steps_per_epoch,
    }
    write_json(output_dir / "run_config.json", run_config)
    print(
        f"device={device.type} batch_size={batch_size} epochs={args.epochs} "
        f"samples={X.shape[0]} dataset={dataset_path}"
    )

    log_rows: list[dict[str, float | int | str]] = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        batch_count = 0
        for batch in train_loader:
            if args.max_steps_per_epoch is not None and batch_count >= args.max_steps_per_epoch:
                break
            batch = batch.to(device)
            view1 = torch.stack([augment_spectrum(sample) for sample in batch], dim=0)
            view2 = torch.stack([augment_spectrum(sample) for sample in batch], dim=0)
            optimizer.zero_grad(set_to_none=True)
            z1 = model(view1)
            z2 = model(view2)
            loss = nt_xent_loss(z1, z2, temperature=args.temperature)
            loss.backward()
            optimizer.step()
            running_loss += float(loss.item())
            batch_count += 1

        epoch_loss = running_loss / max(batch_count, 1)
        log_rows.append(
            {
                "epoch": epoch,
                "loss": epoch_loss,
                "device": device.type,
                "batch_size": batch_size,
                "steps": batch_count,
            }
        )
        print(f"epoch={epoch:02d} loss={epoch_loss:.6f} steps={batch_count}")

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "input_len": input_len,
            "epochs": args.epochs,
            "batch_size": batch_size,
            "temperature": args.temperature,
            "device": device.type,
            "learning_rate": args.learning_rate,
        },
        output_dir / "model.pt",
    )
    with (output_dir / "training_log.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["epoch", "loss", "device", "batch_size", "steps"])
        writer.writeheader()
        writer.writerows(log_rows)

    print(f"Saved model: {output_dir / 'model.pt'}")


if __name__ == "__main__":
    main()
