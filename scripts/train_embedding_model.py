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
TRAINING_PRESETS: dict[str, dict[str, object]] = {
    "pass2_baseline": {
        "positive_pair_mode": "instance_only",
        "semantic_positive_weight": 0.0,
        "hard_negative_mode": "off",
        "hard_negative_weight": 0.0,
        "hard_negative_margin": 0.30,
        "augmentation_mode": "pass2",
        "augmentation_strength": 1.0,
    },
    "pass3_aggressive": {
        "positive_pair_mode": "instance_semantic",
        "semantic_positive_weight": 0.35,
        "hard_negative_mode": "same_scope_diff_class",
        "hard_negative_weight": 0.10,
        "hard_negative_margin": 0.25,
        "augmentation_mode": "pass3",
        "augmentation_strength": 1.15,
    },
    "pass3_tempered": {
        "positive_pair_mode": "instance_semantic",
        "semantic_positive_weight": 0.10,
        "hard_negative_mode": "same_scope_diff_class",
        "hard_negative_weight": 0.04,
        "hard_negative_margin": 0.20,
        "augmentation_mode": "pass3",
        "augmentation_strength": 1.00,
    },
}


class SpectrumDataset(Dataset):
    def __init__(self, spectra: np.ndarray):
        self.spectra = torch.from_numpy(spectra.astype(np.float32))

    def __len__(self) -> int:
        return int(self.spectra.shape[0])

    def __getitem__(self, index: int) -> tuple[int, torch.Tensor]:
        return index, self.spectra[index]


def nt_xent_loss(z1: torch.Tensor, z2: torch.Tensor, temperature: float = TEMPERATURE) -> torch.Tensor:
    batch_size = z1.shape[0]
    z = torch.cat([z1, z2], dim=0)
    similarity = F.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0), dim=2) / temperature
    mask = torch.eye(2 * batch_size, device=z.device, dtype=torch.bool)
    similarity = similarity.masked_fill(mask, -9e15)
    positive_indices = torch.arange(batch_size, device=z.device)
    positive_indices = torch.cat([positive_indices + batch_size, positive_indices], dim=0)
    return F.cross_entropy(similarity, positive_indices)


def supervised_contrastive_loss(embeddings: torch.Tensor, labels: list[str], temperature: float) -> torch.Tensor:
    valid_mask = torch.tensor([bool(label) for label in labels], device=embeddings.device, dtype=torch.bool)
    if int(valid_mask.sum().item()) < 2:
        return torch.zeros((), device=embeddings.device)

    z = embeddings[valid_mask]
    valid_labels = [label for label in labels if label]
    label_tensor = torch.tensor(
        [[1 if label_i == label_j else 0 for label_j in valid_labels] for label_i in valid_labels],
        device=embeddings.device,
        dtype=torch.float32,
    )
    logits = torch.matmul(z, z.T) / temperature
    eye = torch.eye(logits.shape[0], device=embeddings.device, dtype=torch.bool)
    logits = logits.masked_fill(eye, -9e15)
    positive_mask = label_tensor.masked_fill(eye, 0.0)
    positive_counts = positive_mask.sum(dim=1)
    valid_rows = positive_counts > 0
    if int(valid_rows.sum().item()) == 0:
        return torch.zeros((), device=embeddings.device)

    log_prob = logits - torch.logsumexp(logits, dim=1, keepdim=True)
    mean_log_prob_pos = (positive_mask * log_prob).sum(dim=1) / positive_counts.clamp_min(1.0)
    return -mean_log_prob_pos[valid_rows].mean()


def hard_negative_penalty(
    embeddings: torch.Tensor,
    scopes: list[str],
    semantic_groups: list[str],
    *,
    margin: float,
) -> torch.Tensor:
    device = embeddings.device
    if len(scopes) < 2:
        return torch.zeros((), device=device)
    scope_tensor = torch.tensor(
        [[1 if scopes[i] and scopes[i] == scopes[j] else 0 for j in range(len(scopes))] for i in range(len(scopes))],
        device=device,
        dtype=torch.bool,
    )
    diff_tensor = torch.tensor(
        [
            [
                1 if scopes[i] and scopes[j] and semantic_groups[i] and semantic_groups[j] and semantic_groups[i] != semantic_groups[j] else 0
                for j in range(len(scopes))
            ]
            for i in range(len(scopes))
        ],
        device=device,
        dtype=torch.bool,
    )
    mask = scope_tensor & diff_tensor
    mask.fill_diagonal_(False)
    if int(mask.sum().item()) == 0:
        return torch.zeros((), device=device)
    sim = torch.matmul(embeddings, embeddings.T)
    penalties = F.relu(sim - margin)
    return penalties[mask].mean()


def build_sample_weights(
    sample_types: np.ndarray,
    dataset_ids: np.ndarray,
    record_kinds: np.ndarray,
    semantic_groups: np.ndarray,
) -> np.ndarray:
    sample_type_counts = {key: count for key, count in zip(*np.unique(sample_types.astype(str), return_counts=True))}
    dataset_counts = {key: count for key, count in zip(*np.unique(dataset_ids.astype(str), return_counts=True))}
    semantic_counts = {
        key: count
        for key, count in zip(*np.unique(semantic_groups[semantic_groups.astype(str) != ""].astype(str), return_counts=True))
    }
    weights = []
    for sample_type, dataset_id, record_kind, semantic_group in zip(
        sample_types.astype(str),
        dataset_ids.astype(str),
        record_kinds.astype(str),
        semantic_groups.astype(str),
        strict=False,
    ):
        base = 1.0 / math.sqrt(sample_type_counts[sample_type] * dataset_counts[dataset_id])
        if semantic_group:
            base *= 1.0 + min(1.5, 1.0 / math.sqrt(max(semantic_counts.get(semantic_group, 1), 1)))
        if record_kind == "class_summary":
            base *= 0.35
        weights.append(base)
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
    parser.add_argument("--preset", choices=sorted(TRAINING_PRESETS), default="pass3_tempered")
    parser.add_argument("--positive-pair-mode", choices=["instance_only", "instance_semantic"], default=None)
    parser.add_argument("--semantic-positive-weight", type=float, default=None)
    parser.add_argument("--hard-negative-mode", choices=["off", "same_scope_diff_class"], default=None)
    parser.add_argument("--hard-negative-weight", type=float, default=None)
    parser.add_argument("--hard-negative-margin", type=float, default=None)
    parser.add_argument("--augmentation-mode", choices=["pass2", "pass3"], default=None)
    parser.add_argument("--augmentation-strength", type=float, default=None)
    return parser.parse_args()


def resolve_training_preset(args: argparse.Namespace) -> dict[str, object]:
    config = dict(TRAINING_PRESETS[args.preset])
    for field in [
        "positive_pair_mode",
        "semantic_positive_weight",
        "hard_negative_mode",
        "hard_negative_weight",
        "hard_negative_margin",
        "augmentation_mode",
        "augmentation_strength",
    ]:
        value = getattr(args, field)
        if value is not None:
            config[field] = value
    return config


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
    preset_config = resolve_training_preset(args)
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
    record_kinds = dataset["record_kinds"].astype(str)
    semantic_groups = dataset["semantic_groups"].astype(str) if "semantic_groups" in dataset.files else np.asarray([""] * len(X))
    hard_negative_scopes = (
        dataset["hard_negative_scopes"].astype(str) if "hard_negative_scopes" in dataset.files else np.asarray([""] * len(X))
    )
    input_len = int(X.shape[1])
    device = detect_device()
    batch_size = args.batch_size or recommended_batch_size(device)

    training_dataset = SpectrumDataset(X)
    sample_weights = build_sample_weights(sample_types, dataset_ids, record_kinds, semantic_groups)
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
        "preset": args.preset,
        "positive_pair_mode": preset_config["positive_pair_mode"],
        "semantic_positive_weight": preset_config["semantic_positive_weight"],
        "hard_negative_mode": preset_config["hard_negative_mode"],
        "hard_negative_weight": preset_config["hard_negative_weight"],
        "hard_negative_margin": preset_config["hard_negative_margin"],
        "augmentation_mode": preset_config["augmentation_mode"],
        "augmentation_strength": preset_config["augmentation_strength"],
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
        running_instance = 0.0
        running_semantic = 0.0
        running_hard_negative = 0.0
        batch_count = 0
        for batch_indices, batch in train_loader:
            if args.max_steps_per_epoch is not None and batch_count >= args.max_steps_per_epoch:
                break
            batch_indices = batch_indices.tolist()
            batch = batch.to(device)
            view1 = torch.stack(
                [
                    augment_spectrum(
                        sample,
                        mode=str(preset_config["augmentation_mode"]),
                        strength=float(preset_config["augmentation_strength"]),
                    )
                    for sample in batch
                ],
                dim=0,
            )
            view2 = torch.stack(
                [
                    augment_spectrum(
                        sample,
                        mode=str(preset_config["augmentation_mode"]),
                        strength=float(preset_config["augmentation_strength"]),
                    )
                    for sample in batch
                ],
                dim=0,
            )
            optimizer.zero_grad(set_to_none=True)
            z1 = model(view1)
            z2 = model(view2)
            instance_loss = nt_xent_loss(z1, z2, temperature=args.temperature)
            total_loss = instance_loss

            semantic_loss = torch.zeros((), device=device)
            if preset_config["positive_pair_mode"] == "instance_semantic" and float(preset_config["semantic_positive_weight"]) > 0.0:
                semantic_labels = [semantic_groups[index] for index in batch_indices]
                semantic_loss = supervised_contrastive_loss(
                    torch.cat([z1, z2], dim=0),
                    semantic_labels + semantic_labels,
                    temperature=args.temperature,
                )
                total_loss = total_loss + float(preset_config["semantic_positive_weight"]) * semantic_loss

            hard_negative_loss = torch.zeros((), device=device)
            if preset_config["hard_negative_mode"] == "same_scope_diff_class" and float(preset_config["hard_negative_weight"]) > 0.0:
                scopes = [hard_negative_scopes[index] for index in batch_indices]
                groups = [semantic_groups[index] for index in batch_indices]
                hard_negative_loss = hard_negative_penalty(
                    torch.cat([z1, z2], dim=0),
                    scopes + scopes,
                    groups + groups,
                    margin=float(preset_config["hard_negative_margin"]),
                )
                total_loss = total_loss + float(preset_config["hard_negative_weight"]) * hard_negative_loss

            total_loss.backward()
            optimizer.step()
            running_loss += float(total_loss.item())
            running_instance += float(instance_loss.item())
            running_semantic += float(semantic_loss.item())
            running_hard_negative += float(hard_negative_loss.item())
            batch_count += 1

        epoch_loss = running_loss / max(batch_count, 1)
        log_rows.append(
            {
                "epoch": epoch,
                "loss": epoch_loss,
                "instance_loss": running_instance / max(batch_count, 1),
                "semantic_loss": running_semantic / max(batch_count, 1),
                "hard_negative_loss": running_hard_negative / max(batch_count, 1),
                "device": device.type,
                "batch_size": batch_size,
                "steps": batch_count,
            }
        )
        print(
            f"epoch={epoch:02d} loss={epoch_loss:.6f} "
            f"instance={running_instance / max(batch_count, 1):.6f} "
            f"semantic={running_semantic / max(batch_count, 1):.6f} "
            f"hardneg={running_hard_negative / max(batch_count, 1):.6f} "
            f"steps={batch_count}"
        )

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "input_len": input_len,
            "epochs": args.epochs,
            "batch_size": batch_size,
            "temperature": args.temperature,
            "device": device.type,
            "learning_rate": args.learning_rate,
            "preset": args.preset,
            "positive_pair_mode": preset_config["positive_pair_mode"],
            "semantic_positive_weight": preset_config["semantic_positive_weight"],
            "hard_negative_mode": preset_config["hard_negative_mode"],
            "hard_negative_weight": preset_config["hard_negative_weight"],
            "hard_negative_margin": preset_config["hard_negative_margin"],
            "augmentation_mode": preset_config["augmentation_mode"],
            "augmentation_strength": preset_config["augmentation_strength"],
        },
        output_dir / "model.pt",
    )
    with (output_dir / "training_log.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["epoch", "loss", "instance_loss", "semantic_loss", "hard_negative_loss", "device", "batch_size", "steps"],
        )
        writer.writeheader()
        writer.writerows(log_rows)

    print(f"Saved model: {output_dir / 'model.pt'}")


if __name__ == "__main__":
    main()
