from __future__ import annotations

import argparse
import csv
import math
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler


DEFAULT_EPOCHS = 30
TEMPERATURE = 0.2
LEARNING_RATE = 1e-3
TRAINING_PRESETS: dict[str, dict[str, object]] = {
    "pass2_baseline": {
        "positive_pair_mode": "instance_only",
        "instance_positive_weight": 1.0,
        "anchor_positive_weight": 0.0,
        "semantic_positive_weight": 0.0,
        "hard_negative_mode": "off",
        "hard_negative_weight": 0.0,
        "hard_negative_margin": 0.30,
        "augmentation_mode": "pass2",
        "augmentation_strength": 1.0,
        "pair_scope_mode": "global",
        "cross_dataset_positive_mode": "off",
        "anchor_variance_weight": 0.0,
        "anchor_variance_target_std": 0.50,
    },
    "pass3_aggressive": {
        "positive_pair_mode": "instance_semantic",
        "instance_positive_weight": 1.0,
        "anchor_positive_weight": 0.0,
        "semantic_positive_weight": 0.35,
        "hard_negative_mode": "same_scope_diff_class",
        "hard_negative_weight": 0.10,
        "hard_negative_margin": 0.25,
        "augmentation_mode": "pass3",
        "augmentation_strength": 1.15,
        "pair_scope_mode": "global",
        "cross_dataset_positive_mode": "off",
        "anchor_variance_weight": 0.0,
        "anchor_variance_target_std": 0.50,
    },
    "pass3_tempered": {
        "positive_pair_mode": "instance_semantic",
        "instance_positive_weight": 1.0,
        "anchor_positive_weight": 0.0,
        "semantic_positive_weight": 0.10,
        "hard_negative_mode": "same_scope_diff_class",
        "hard_negative_weight": 0.04,
        "hard_negative_margin": 0.20,
        "augmentation_mode": "pass3",
        "augmentation_strength": 1.00,
        "pair_scope_mode": "global",
        "cross_dataset_positive_mode": "off",
        "anchor_variance_weight": 0.0,
        "anchor_variance_target_std": 0.50,
    },
    "pass6_within_type": {
        "positive_pair_mode": "instance_semantic",
        "instance_positive_weight": 1.0,
        "anchor_positive_weight": 0.0,
        "semantic_positive_weight": 0.10,
        "hard_negative_mode": "same_scope_diff_class",
        "hard_negative_weight": 0.04,
        "hard_negative_margin": 0.20,
        "augmentation_mode": "pass3",
        "augmentation_strength": 1.00,
        "pair_scope_mode": "within_sample_type",
        "cross_dataset_positive_mode": "prefer",
        "anchor_variance_weight": 0.0,
        "anchor_variance_target_std": 0.50,
    },
    "pass7_anchor_invariance": {
        "positive_pair_mode": "instance_anchor",
        "instance_positive_weight": 0.90,
        "anchor_positive_weight": 0.12,
        "semantic_positive_weight": 0.0,
        "hard_negative_mode": "same_type_diff_anchor",
        "hard_negative_weight": 0.05,
        "hard_negative_margin": 0.20,
        "augmentation_mode": "pass3",
        "augmentation_strength": 1.00,
        "pair_scope_mode": "within_sample_type",
        "cross_dataset_positive_mode": "prefer",
        "anchor_variance_weight": 0.02,
        "anchor_variance_target_std": 0.50,
    },
    "pass8_ev_stress_branch": {
        "positive_pair_mode": "instance_anchor",
        "instance_positive_weight": 0.85,
        "anchor_positive_weight": 0.10,
        "semantic_positive_weight": 0.0,
        "state_positive_weight": 0.14,
        "class_positive_weight": 0.0,
        "probe_confusion_weight": 0.0,
        "cross_probe_positive_boost": 1.0,
        "same_probe_positive_weight": 1.0,
        "class_compactness_weight": 0.0,
        "class_compactness_target_radius": 0.60,
        "hard_negative_mode": "same_type_diff_anchor",
        "hard_negative_weight": 0.05,
        "hard_negative_margin": 0.20,
        "augmentation_mode": "pass3",
        "augmentation_strength": 1.00,
        "pair_scope_mode": "within_sample_type",
        "cross_dataset_positive_mode": "prefer",
        "anchor_variance_weight": 0.02,
        "anchor_variance_target_std": 0.50,
    },
    "pass8_small2023_specialized": {
        "positive_pair_mode": "instance_anchor",
        "instance_positive_weight": 0.85,
        "anchor_positive_weight": 0.0,
        "semantic_positive_weight": 0.0,
        "state_positive_weight": 0.0,
        "class_positive_weight": 0.18,
        "probe_confusion_weight": 0.10,
        "cross_probe_positive_boost": 2.0,
        "same_probe_positive_weight": 0.50,
        "class_compactness_weight": 0.0,
        "class_compactness_target_radius": 0.60,
        "hard_negative_mode": "off",
        "hard_negative_weight": 0.0,
        "hard_negative_margin": 0.20,
        "augmentation_mode": "pass3",
        "augmentation_strength": 1.00,
        "pair_scope_mode": "within_sample_type",
        "cross_dataset_positive_mode": "off",
        "anchor_variance_weight": 0.02,
        "anchor_variance_target_std": 0.50,
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


def weighted_supervised_contrastive_loss(
    embeddings: torch.Tensor,
    labels: list[str],
    *,
    sample_types: list[str],
    dataset_ids: list[str],
    temperature: float,
    pair_scope_mode: str,
    cross_dataset_positive_mode: str,
    confidence_weights: list[float] | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    valid_mask = torch.tensor([bool(label) for label in labels], device=embeddings.device, dtype=torch.bool)
    if int(valid_mask.sum().item()) < 2:
        return torch.zeros((), device=embeddings.device), {
            "positive_pairs": 0.0,
            "positive_pairs_cross_dataset": 0.0,
            "positive_pairs_same_dataset": 0.0,
            "anchored_samples": 0.0,
        }

    z = embeddings[valid_mask]
    valid_labels = [label for label in labels if label]
    valid_sample_types = [sample_type for label, sample_type in zip(labels, sample_types, strict=False) if label]
    valid_dataset_ids = [dataset_id for label, dataset_id in zip(labels, dataset_ids, strict=False) if label]
    valid_confidence_weights = (
        [float(weight) for label, weight in zip(labels, confidence_weights or [], strict=False) if label]
        if confidence_weights is not None
        else [1.0 for _ in valid_labels]
    )
    label_tensor = torch.tensor(
        [[1 if label_i == label_j else 0 for label_j in valid_labels] for label_i in valid_labels],
        device=embeddings.device,
        dtype=torch.bool,
    )
    sample_type_tensor = torch.tensor(
        [[1 if type_i == type_j else 0 for type_j in valid_sample_types] for type_i in valid_sample_types],
        device=embeddings.device,
        dtype=torch.bool,
    )
    cross_dataset_tensor = torch.tensor(
        [[1 if dataset_i != dataset_j else 0 for dataset_j in valid_dataset_ids] for dataset_i in valid_dataset_ids],
        device=embeddings.device,
        dtype=torch.bool,
    )
    logits = torch.matmul(z, z.T) / temperature
    eye = torch.eye(logits.shape[0], device=embeddings.device, dtype=torch.bool)
    logits = logits.masked_fill(eye, -9e15)
    base_positive_mask = label_tensor & ~eye
    if pair_scope_mode == "within_sample_type":
        base_positive_mask = base_positive_mask & sample_type_tensor

    if cross_dataset_positive_mode == "prefer":
        preferred_mask = torch.zeros_like(base_positive_mask)
        same_dataset_mask = base_positive_mask & ~cross_dataset_tensor
        cross_dataset_mask = base_positive_mask & cross_dataset_tensor
        for row_idx in range(base_positive_mask.shape[0]):
            if bool(cross_dataset_mask[row_idx].any().item()):
                preferred_mask[row_idx] = cross_dataset_mask[row_idx]
            else:
                preferred_mask[row_idx] = same_dataset_mask[row_idx]
        positive_mask = preferred_mask.float()
    else:
        positive_mask = base_positive_mask.float()

    confidence_tensor = torch.tensor(valid_confidence_weights, device=embeddings.device, dtype=torch.float32)
    pair_weight_matrix = torch.minimum(confidence_tensor.unsqueeze(1), confidence_tensor.unsqueeze(0))
    weighted_positive_mask = positive_mask * pair_weight_matrix
    positive_counts = weighted_positive_mask.sum(dim=1)
    valid_rows = positive_counts > 0
    if int(valid_rows.sum().item()) == 0:
        return torch.zeros((), device=embeddings.device), {
            "positive_pairs": 0.0,
            "positive_pairs_cross_dataset": 0.0,
            "positive_pairs_same_dataset": 0.0,
            "anchored_samples": 0.0,
        }

    log_prob = logits - torch.logsumexp(logits, dim=1, keepdim=True)
    mean_log_prob_pos = (weighted_positive_mask * log_prob).sum(dim=1) / positive_counts.clamp_min(1.0)
    cross_dataset_count = float((positive_mask.bool() & cross_dataset_tensor).sum().item())
    same_dataset_count = float((positive_mask.bool() & ~cross_dataset_tensor & ~eye).sum().item())
    return -mean_log_prob_pos[valid_rows].mean(), {
        "positive_pairs": float(positive_mask.sum().item()),
        "positive_pairs_cross_dataset": cross_dataset_count,
        "positive_pairs_same_dataset": same_dataset_count,
        "anchored_samples": float(len(valid_labels)),
    }


def class_hard_negative_penalty(
    embeddings: torch.Tensor,
    scopes: list[str],
    semantic_groups: list[str],
    sample_types: list[str],
    *,
    margin: float,
    pair_scope_mode: str,
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
    sample_type_tensor = torch.tensor(
        [[1 if sample_types[i] and sample_types[i] == sample_types[j] else 0 for j in range(len(sample_types))] for i in range(len(sample_types))],
        device=device,
        dtype=torch.bool,
    )
    mask = scope_tensor & diff_tensor
    if pair_scope_mode == "within_sample_type":
        mask = mask & sample_type_tensor
    mask.fill_diagonal_(False)
    if int(mask.sum().item()) == 0:
        return torch.zeros((), device=device)
    sim = torch.matmul(embeddings, embeddings.T)
    penalties = F.relu(sim - margin)
    return penalties[mask].mean()


def anchor_hard_negative_penalty(
    embeddings: torch.Tensor,
    anchor_labels: list[str],
    sample_types: list[str],
    dataset_ids: list[str],
    *,
    margin: float,
) -> torch.Tensor:
    device = embeddings.device
    if len(anchor_labels) < 2:
        return torch.zeros((), device=device)
    anchor_tensor = torch.tensor(
        [[1 if anchor_labels[i] and anchor_labels[j] and anchor_labels[i] != anchor_labels[j] else 0 for j in range(len(anchor_labels))] for i in range(len(anchor_labels))],
        device=device,
        dtype=torch.bool,
    )
    sample_type_tensor = torch.tensor(
        [[1 if sample_types[i] and sample_types[i] == sample_types[j] else 0 for j in range(len(sample_types))] for i in range(len(sample_types))],
        device=device,
        dtype=torch.bool,
    )
    same_dataset_tensor = torch.tensor(
        [[1 if dataset_ids[i] and dataset_ids[i] == dataset_ids[j] else 0 for j in range(len(dataset_ids))] for i in range(len(dataset_ids))],
        device=device,
        dtype=torch.float32,
    )
    mask = anchor_tensor & sample_type_tensor
    mask.fill_diagonal_(False)
    if int(mask.sum().item()) == 0:
        return torch.zeros((), device=device)
    sim = torch.matmul(embeddings, embeddings.T)
    penalties = F.relu(sim - margin)
    weights = torch.where(same_dataset_tensor > 0, torch.full_like(same_dataset_tensor, 1.5), torch.ones_like(same_dataset_tensor))
    weighted = penalties * weights
    return weighted[mask].mean()


def anchor_variance_regularization(
    embeddings: torch.Tensor,
    anchor_labels: list[str],
    sample_types: list[str],
    confidence_weights: list[float],
    *,
    target_std: float,
) -> torch.Tensor:
    device = embeddings.device
    grouped: dict[tuple[str, str], list[int]] = {}
    for idx, (sample_type, anchor) in enumerate(zip(sample_types, anchor_labels, strict=False)):
        if not sample_type or not anchor:
            continue
        grouped.setdefault((sample_type, anchor), []).append(idx)
    penalties = []
    for _, indices in grouped.items():
        if len(indices) < 2:
            continue
        subset = embeddings[indices]
        std = torch.sqrt(subset.var(dim=0, unbiased=False) + 1e-4)
        base_penalty = F.relu(target_std - std).mean()
        mean_conf = float(sum(confidence_weights[i] for i in indices) / max(len(indices), 1))
        penalties.append(base_penalty * mean_conf)
    if not penalties:
        return torch.zeros((), device=device)
    return torch.stack(penalties).mean()


def confidence_multiplier(confidence: str) -> float:
    mapping = {
        "high": 1.0,
        "medium": 0.7,
        "low": 0.4,
    }
    return float(mapping.get(str(confidence).strip().lower(), 0.4))


def dataset_entropy_within_anchor(anchor_labels: list[str], dataset_ids: list[str], sample_types: list[str]) -> float:
    grouped: dict[tuple[str, str], list[str]] = {}
    for sample_type, anchor, dataset_id in zip(sample_types, anchor_labels, dataset_ids, strict=False):
        if not sample_type or not anchor:
            continue
        grouped.setdefault((sample_type, anchor), []).append(dataset_id)
    entropies = []
    for dataset_list in grouped.values():
        if len(dataset_list) < 2:
            continue
        counts = pd.Series(dataset_list).value_counts()
        probs = counts / counts.sum()
        entropy = float(-(probs * np.log2(probs)).sum())
        max_entropy = math.log2(len(counts)) if len(counts) > 1 else 1.0
        entropies.append(entropy / max_entropy if max_entropy > 0 else 0.0)
    return float(np.mean(entropies)) if entropies else 0.0


def build_sample_weights(
    sample_types: np.ndarray,
    dataset_ids: np.ndarray,
    record_kinds: np.ndarray,
    semantic_groups: np.ndarray,
    *,
    branch_mode: str = "none",
    branch_primary_labels: np.ndarray | None = None,
    branch_secondary_labels: np.ndarray | None = None,
) -> np.ndarray:
    sample_type_counts = {key: count for key, count in zip(*np.unique(sample_types.astype(str), return_counts=True))}
    dataset_counts = {key: count for key, count in zip(*np.unique(dataset_ids.astype(str), return_counts=True))}
    semantic_counts = {
        key: count
        for key, count in zip(*np.unique(semantic_groups[semantic_groups.astype(str) != ""].astype(str), return_counts=True))
    }
    primary_counts: dict[str, int] = {}
    secondary_counts: dict[str, int] = {}
    combo_counts: dict[tuple[str, str], int] = {}
    if branch_primary_labels is not None:
        primary_series = pd.Series(branch_primary_labels.astype(str))
        primary_counts = primary_series[primary_series != ""].value_counts().to_dict()
    if branch_secondary_labels is not None:
        secondary_series = pd.Series(branch_secondary_labels.astype(str))
        secondary_counts = secondary_series[secondary_series != ""].value_counts().to_dict()
    if branch_primary_labels is not None and branch_secondary_labels is not None:
        combo_df = pd.DataFrame(
            {
                "primary": branch_primary_labels.astype(str),
                "secondary": branch_secondary_labels.astype(str),
            }
        )
        combo_df = combo_df[(combo_df["primary"] != "") & (combo_df["secondary"] != "")]
        combo_counts = combo_df.value_counts().to_dict()
    weights = []
    for idx, (sample_type, dataset_id, record_kind, semantic_group) in enumerate(
        zip(
            sample_types.astype(str),
            dataset_ids.astype(str),
            record_kinds.astype(str),
            semantic_groups.astype(str),
            strict=False,
        )
    ):
        base = 1.0 / math.sqrt(sample_type_counts[sample_type] * dataset_counts[dataset_id])
        if semantic_group:
            base *= 1.0 + min(1.5, 1.0 / math.sqrt(max(semantic_counts.get(semantic_group, 1), 1)))
        if record_kind == "class_summary":
            base *= 0.35
        if branch_mode in {"small2023_specialized", "small2023_cellline", "small2023_mixture"} and branch_primary_labels is not None:
            primary = str(branch_primary_labels[idx])
            secondary = str(branch_secondary_labels[idx]) if branch_secondary_labels is not None else ""
            if primary:
                base *= 1.0 + min(2.0, 1.0 / math.sqrt(max(primary_counts.get(primary, 1), 1)))
            if branch_mode == "small2023_mixture" and secondary:
                base *= 1.0 + min(2.0, 1.0 / math.sqrt(max(secondary_counts.get(secondary, 1), 1)))
                combo_key = (primary, secondary)
                if primary and combo_key in combo_counts:
                    base *= 1.0 + min(2.5, 1.5 / math.sqrt(max(combo_counts.get(combo_key, 1), 1)))
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
    parser.add_argument("--positive-pair-mode", choices=["instance_only", "instance_semantic", "instance_anchor"], default=None)
    parser.add_argument("--instance-positive-weight", type=float, default=None)
    parser.add_argument("--anchor-positive-weight", type=float, default=None)
    parser.add_argument("--semantic-positive-weight", type=float, default=None)
    parser.add_argument("--hard-negative-mode", choices=["off", "same_scope_diff_class", "same_type_diff_anchor"], default=None)
    parser.add_argument("--hard-negative-weight", type=float, default=None)
    parser.add_argument("--hard-negative-margin", type=float, default=None)
    parser.add_argument("--augmentation-mode", choices=["pass2", "pass3"], default=None)
    parser.add_argument("--augmentation-strength", type=float, default=None)
    parser.add_argument("--pair-scope-mode", choices=["global", "within_sample_type"], default=None)
    parser.add_argument("--cross-dataset-positive-mode", choices=["off", "prefer"], default=None)
    parser.add_argument("--anchor-variance-weight", type=float, default=None)
    parser.add_argument("--anchor-variance-target-std", type=float, default=None)
    parser.add_argument("--state-positive-weight", type=float, default=None)
    parser.add_argument("--class-positive-weight", type=float, default=None)
    parser.add_argument("--probe-confusion-weight", type=float, default=None)
    parser.add_argument("--cross-probe-positive-boost", type=float, default=None)
    parser.add_argument("--same-probe-positive-weight", type=float, default=None)
    parser.add_argument("--class-compactness-weight", type=float, default=None)
    parser.add_argument("--class-compactness-target-radius", type=float, default=None)
    parser.add_argument("--anchor-table-path", default=None)
    parser.add_argument("--branch-mode", choices=["none", "ev_stress", "small2023_specialized", "small2023_cellline", "small2023_mixture"], default="none")
    parser.add_argument("--init-checkpoint", default=None)
    return parser.parse_args()


def resolve_training_preset(args: argparse.Namespace) -> dict[str, object]:
    config = dict(TRAINING_PRESETS[args.preset])
    for field in [
        "positive_pair_mode",
        "instance_positive_weight",
        "anchor_positive_weight",
        "semantic_positive_weight",
        "hard_negative_mode",
        "hard_negative_weight",
        "hard_negative_margin",
        "augmentation_mode",
        "augmentation_strength",
        "pair_scope_mode",
        "cross_dataset_positive_mode",
        "anchor_variance_weight",
        "anchor_variance_target_std",
        "state_positive_weight",
        "class_positive_weight",
        "probe_confusion_weight",
        "cross_probe_positive_boost",
        "same_probe_positive_weight",
        "class_compactness_weight",
        "class_compactness_target_radius",
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
    from gaira.embedding.anchor_loader import aligned_anchor_arrays, resolve_anchor_table_path
    from gaira.embedding.branch_objectives import branch_class_contrastive_loss, class_compactness_loss, probe_alignment_penalty
    from gaira.embedding.branch_sampling import branch_dataset_summary, branch_sample_manifest, filtered_dataset_dict, write_filtered_dataset
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
    effective_dataset_path = dataset_path
    if args.branch_mode != "none":
        filtered = filtered_dataset_dict(dataset, branch_mode=args.branch_mode)
        effective_dataset_path = output_dir / "branch_dataset.npz"
        write_filtered_dataset(filtered, effective_dataset_path)
        manifest_df = branch_sample_manifest(filtered)
        manifest_df.to_csv(output_dir / "branch_sample_manifest.csv", index=False)
        branch_dataset_summary(manifest_df).to_csv(output_dir / "branch_dataset_summary.csv", index=False)
        dataset.close()
        dataset = np.load(effective_dataset_path, allow_pickle=True)

    X = dataset["X"]
    dataset_ids = dataset["dataset_ids"].astype(str)
    sample_types = dataset["sample_types"].astype(str)
    record_kinds = dataset["record_kinds"].astype(str)
    semantic_groups = dataset["semantic_groups"].astype(str) if "semantic_groups" in dataset.files else np.asarray([""] * len(X))
    hard_negative_scopes = (
        dataset["hard_negative_scopes"].astype(str) if "hard_negative_scopes" in dataset.files else np.asarray([""] * len(X))
    )
    sample_keys = dataset["sample_keys"].astype(str) if "sample_keys" in dataset.files else np.asarray([str(i) for i in range(len(X))], dtype=object)
    branch_primary_labels = dataset["branch_primary_label"].astype(str) if "branch_primary_label" in dataset.files else np.asarray([""] * len(X), dtype=object)
    branch_secondary_labels = dataset["branch_secondary_label"].astype(str) if "branch_secondary_label" in dataset.files else np.asarray([""] * len(X), dtype=object)
    branch_state_labels = dataset["branch_state_label"].astype(str) if "branch_state_label" in dataset.files else np.asarray([""] * len(X), dtype=object)
    branch_label_weights = dataset["branch_label_weight"].astype(np.float32) if "branch_label_weight" in dataset.files else np.asarray([0.0] * len(X), dtype=np.float32)
    resolved_anchor_table_path = resolve_anchor_table_path(args.anchor_table_path)
    anchor_arrays = aligned_anchor_arrays(sample_keys, resolved_anchor_table_path)
    harmonized_anchors = anchor_arrays["harmonized_anchor"].astype(str)
    anchor_confidences = anchor_arrays["anchor_confidence"].astype(str)
    anchor_cross_dataset_usable = anchor_arrays["cross_dataset_usable"].astype(bool)
    input_len = int(X.shape[1])
    device = detect_device()
    batch_size = args.batch_size or recommended_batch_size(device)

    training_dataset = SpectrumDataset(X)
    sample_weights = build_sample_weights(
        sample_types,
        dataset_ids,
        record_kinds,
        semantic_groups,
        branch_mode=args.branch_mode,
        branch_primary_labels=branch_primary_labels,
        branch_secondary_labels=branch_secondary_labels,
    )
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
    if args.init_checkpoint:
        checkpoint = torch.load(Path(args.init_checkpoint).expanduser().resolve(), map_location="cpu")
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    run_config = {
        "dataset_path": str(dataset_path),
        "effective_dataset_path": str(effective_dataset_path),
        "output_dir": str(output_dir),
        "device": device.type,
        "batch_size": batch_size,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "temperature": args.temperature,
        "seed": args.seed,
        "samples": int(X.shape[0]),
        "input_len": input_len,
        "anchor_table_path": str(resolved_anchor_table_path),
        "sample_type_counts": {key: int(value) for key, value in zip(*np.unique(sample_types, return_counts=True))},
        "dataset_count": int(len(np.unique(dataset_ids))),
        "max_steps_per_epoch": args.max_steps_per_epoch,
        "preset": args.preset,
        "branch_mode": args.branch_mode,
        "init_checkpoint": str(Path(args.init_checkpoint).expanduser().resolve()) if args.init_checkpoint else "",
        "positive_pair_mode": preset_config["positive_pair_mode"],
        "instance_positive_weight": preset_config["instance_positive_weight"],
        "anchor_positive_weight": preset_config["anchor_positive_weight"],
        "semantic_positive_weight": preset_config["semantic_positive_weight"],
        "state_positive_weight": preset_config.get("state_positive_weight", 0.0),
        "class_positive_weight": preset_config.get("class_positive_weight", 0.0),
        "probe_confusion_weight": preset_config.get("probe_confusion_weight", 0.0),
        "cross_probe_positive_boost": preset_config.get("cross_probe_positive_boost", 1.0),
        "same_probe_positive_weight": preset_config.get("same_probe_positive_weight", 1.0),
        "class_compactness_weight": preset_config.get("class_compactness_weight", 0.0),
        "class_compactness_target_radius": preset_config.get("class_compactness_target_radius", 0.60),
        "hard_negative_mode": preset_config["hard_negative_mode"],
        "hard_negative_weight": preset_config["hard_negative_weight"],
        "hard_negative_margin": preset_config["hard_negative_margin"],
        "augmentation_mode": preset_config["augmentation_mode"],
        "augmentation_strength": preset_config["augmentation_strength"],
        "pair_scope_mode": preset_config["pair_scope_mode"],
        "cross_dataset_positive_mode": preset_config["cross_dataset_positive_mode"],
        "anchor_variance_weight": preset_config["anchor_variance_weight"],
        "anchor_variance_target_std": preset_config["anchor_variance_target_std"],
    }
    write_json(output_dir / "run_config.json", run_config)
    print(
        f"device={device.type} batch_size={batch_size} epochs={args.epochs} "
        f"samples={X.shape[0]} dataset={effective_dataset_path}"
    )

    log_rows: list[dict[str, float | int | str]] = []
    pairing_summary = {
        "semantic_positive_pairs": 0.0,
        "semantic_positive_pairs_cross_dataset": 0.0,
        "semantic_positive_pairs_same_dataset": 0.0,
        "anchor_positive_pairs": 0.0,
        "anchor_positive_pairs_cross_dataset": 0.0,
        "anchor_positive_pairs_same_dataset": 0.0,
        "anchor_anchored_samples": 0.0,
    }
    pairing_epoch_rows: list[dict[str, float | int | str]] = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        running_instance = 0.0
        running_semantic = 0.0
        running_anchor = 0.0
        running_state = 0.0
        running_class = 0.0
        running_probe = 0.0
        running_compactness = 0.0
        running_hard_negative = 0.0
        running_variance = 0.0
        running_anchor_coverage = 0.0
        running_cross_anchor_fraction = 0.0
        running_anchor_entropy = 0.0
        running_state_pair_fraction = 0.0
        running_cross_dataset_state_pair_fraction = 0.0
        running_cross_probe_positive_fraction = 0.0
        running_same_probe_positive_fraction = 0.0
        running_branch_state_distribution: dict[str, int] = {}
        batch_count = 0
        epoch_anchor_pairs = 0.0
        epoch_anchor_cross_pairs = 0.0
        epoch_anchor_same_pairs = 0.0
        epoch_anchor_samples = 0.0
        epoch_state_pairs = 0.0
        epoch_state_cross_pairs = 0.0
        epoch_cross_probe_fraction = 0.0
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
            total_loss = float(preset_config["instance_positive_weight"]) * instance_loss

            semantic_loss = torch.zeros((), device=device)
            state_loss = torch.zeros((), device=device)
            class_loss = torch.zeros((), device=device)
            probe_loss = torch.zeros((), device=device)
            compactness_loss = torch.zeros((), device=device)
            if preset_config["positive_pair_mode"] == "instance_semantic" and float(preset_config["semantic_positive_weight"]) > 0.0:
                semantic_labels = [semantic_groups[index] for index in batch_indices]
                batch_sample_types = [sample_types[index] for index in batch_indices]
                batch_dataset_ids = [dataset_ids[index] for index in batch_indices]
                semantic_loss, semantic_stats = weighted_supervised_contrastive_loss(
                    torch.cat([z1, z2], dim=0),
                    semantic_labels + semantic_labels,
                    sample_types=batch_sample_types + batch_sample_types,
                    dataset_ids=batch_dataset_ids + batch_dataset_ids,
                    temperature=args.temperature,
                    pair_scope_mode=str(preset_config["pair_scope_mode"]),
                    cross_dataset_positive_mode=str(preset_config["cross_dataset_positive_mode"]),
                )
                pairing_summary["semantic_positive_pairs"] += float(semantic_stats["positive_pairs"])
                pairing_summary["semantic_positive_pairs_cross_dataset"] += float(semantic_stats["positive_pairs_cross_dataset"])
                pairing_summary["semantic_positive_pairs_same_dataset"] += float(semantic_stats["positive_pairs_same_dataset"])
                total_loss = total_loss + float(preset_config["semantic_positive_weight"]) * semantic_loss

            anchor_loss = torch.zeros((), device=device)
            anchor_variance_loss = torch.zeros((), device=device)
            batch_anchor_fraction = 0.0
            batch_anchor_cross_fraction = 0.0
            batch_anchor_entropy = 0.0
            if preset_config["positive_pair_mode"] == "instance_anchor" and float(preset_config["anchor_positive_weight"]) > 0.0:
                batch_anchor_labels = [harmonized_anchors[index] for index in batch_indices]
                batch_anchor_confidences = [confidence_multiplier(anchor_confidences[index]) for index in batch_indices]
                batch_sample_types = [sample_types[index] for index in batch_indices]
                batch_dataset_ids = [dataset_ids[index] for index in batch_indices]
                anchor_loss, anchor_stats = weighted_supervised_contrastive_loss(
                    torch.cat([z1, z2], dim=0),
                    batch_anchor_labels + batch_anchor_labels,
                    sample_types=batch_sample_types + batch_sample_types,
                    dataset_ids=batch_dataset_ids + batch_dataset_ids,
                    temperature=args.temperature,
                    pair_scope_mode=str(preset_config["pair_scope_mode"]),
                    cross_dataset_positive_mode=str(preset_config["cross_dataset_positive_mode"]),
                    confidence_weights=batch_anchor_confidences + batch_anchor_confidences,
                )
                pairing_summary["anchor_positive_pairs"] += float(anchor_stats["positive_pairs"])
                pairing_summary["anchor_positive_pairs_cross_dataset"] += float(anchor_stats["positive_pairs_cross_dataset"])
                pairing_summary["anchor_positive_pairs_same_dataset"] += float(anchor_stats["positive_pairs_same_dataset"])
                pairing_summary["anchor_anchored_samples"] += float(anchor_stats["anchored_samples"])
                epoch_anchor_pairs += float(anchor_stats["positive_pairs"])
                epoch_anchor_cross_pairs += float(anchor_stats["positive_pairs_cross_dataset"])
                epoch_anchor_same_pairs += float(anchor_stats["positive_pairs_same_dataset"])
                epoch_anchor_samples += float(anchor_stats["anchored_samples"])
                batch_anchor_fraction = float(np.mean([bool(label) for label in batch_anchor_labels])) if batch_anchor_labels else 0.0
                batch_anchor_cross_fraction = (
                    float(anchor_stats["positive_pairs_cross_dataset"]) / float(anchor_stats["positive_pairs"])
                    if float(anchor_stats["positive_pairs"]) > 0
                    else 0.0
                )
                batch_anchor_entropy = dataset_entropy_within_anchor(
                    batch_anchor_labels,
                    batch_dataset_ids,
                    batch_sample_types,
                )
                total_loss = total_loss + float(preset_config["anchor_positive_weight"]) * anchor_loss
                if float(preset_config["anchor_variance_weight"]) > 0.0:
                    anchor_variance_loss = anchor_variance_regularization(
                        torch.cat([z1, z2], dim=0),
                        batch_anchor_labels + batch_anchor_labels,
                        batch_sample_types + batch_sample_types,
                        batch_anchor_confidences + batch_anchor_confidences,
                        target_std=float(preset_config["anchor_variance_target_std"]),
                    )
                    total_loss = total_loss + float(preset_config["anchor_variance_weight"]) * anchor_variance_loss

            if args.branch_mode == "ev_stress" and float(preset_config.get("state_positive_weight", 0.0)) > 0.0:
                batch_state_labels = [branch_state_labels[index] for index in batch_indices]
                batch_state_weights = [float(branch_label_weights[index]) for index in batch_indices]
                batch_sample_types = [sample_types[index] for index in batch_indices]
                batch_dataset_ids = [dataset_ids[index] for index in batch_indices]
                state_loss, state_stats = weighted_supervised_contrastive_loss(
                    torch.cat([z1, z2], dim=0),
                    batch_state_labels + batch_state_labels,
                    sample_types=batch_sample_types + batch_sample_types,
                    dataset_ids=batch_dataset_ids + batch_dataset_ids,
                    temperature=args.temperature,
                    pair_scope_mode=str(preset_config["pair_scope_mode"]),
                    cross_dataset_positive_mode="prefer",
                    confidence_weights=batch_state_weights + batch_state_weights,
                )
                total_loss = total_loss + float(preset_config["state_positive_weight"]) * state_loss
                epoch_state_pairs += float(state_stats["positive_pairs"])
                epoch_state_cross_pairs += float(state_stats["positive_pairs_cross_dataset"])
                running_state_pair_fraction += float(state_stats["positive_pairs"] > 0)
                running_cross_dataset_state_pair_fraction += (
                    float(state_stats["positive_pairs_cross_dataset"]) / float(state_stats["positive_pairs"])
                    if float(state_stats["positive_pairs"]) > 0
                    else 0.0
                )
                for label in batch_state_labels:
                    if label:
                        running_branch_state_distribution[label] = running_branch_state_distribution.get(label, 0) + 1

            if args.branch_mode in {"small2023_specialized", "small2023_cellline", "small2023_mixture"}:
                batch_class_labels = [branch_primary_labels[index] for index in batch_indices]
                batch_probe_labels = [branch_secondary_labels[index] for index in batch_indices]
                if float(preset_config.get("class_positive_weight", 0.0)) > 0.0:
                    class_loss, class_stats = branch_class_contrastive_loss(
                        torch.cat([z1, z2], dim=0),
                        batch_class_labels + batch_class_labels,
                        batch_probe_labels + batch_probe_labels,
                        temperature=args.temperature,
                        cross_probe_positive_boost=float(preset_config.get("cross_probe_positive_boost", 2.0)),
                        same_probe_positive_weight=float(preset_config.get("same_probe_positive_weight", 0.5)),
                    )
                    total_loss = total_loss + float(preset_config["class_positive_weight"]) * class_loss
                    running_cross_probe_positive_fraction += float(class_stats["cross_probe_positive_fraction"])
                    epoch_cross_probe_fraction += float(class_stats["within_class_cross_probe_fraction"])
                    running_same_probe_positive_fraction += float(class_stats["same_probe_positive_fraction"])
                if float(preset_config.get("probe_confusion_weight", 0.0)) > 0.0:
                    probe_loss, probe_stats = probe_alignment_penalty(
                        torch.cat([z1, z2], dim=0),
                        batch_class_labels + batch_class_labels,
                        batch_probe_labels + batch_probe_labels,
                    )
                    total_loss = total_loss + float(preset_config["probe_confusion_weight"]) * probe_loss
                if float(preset_config.get("class_compactness_weight", 0.0)) > 0.0:
                    compactness_loss, _ = class_compactness_loss(
                        torch.cat([z1, z2], dim=0),
                        batch_class_labels + batch_class_labels,
                        target_radius=float(preset_config.get("class_compactness_target_radius", 0.60)),
                    )
                    total_loss = total_loss + float(preset_config["class_compactness_weight"]) * compactness_loss

            hard_negative_loss = torch.zeros((), device=device)
            if preset_config["hard_negative_mode"] == "same_scope_diff_class" and float(preset_config["hard_negative_weight"]) > 0.0:
                scopes = [hard_negative_scopes[index] for index in batch_indices]
                groups = [semantic_groups[index] for index in batch_indices]
                batch_sample_types = [sample_types[index] for index in batch_indices]
                hard_negative_loss = class_hard_negative_penalty(
                    torch.cat([z1, z2], dim=0),
                    scopes + scopes,
                    groups + groups,
                    batch_sample_types + batch_sample_types,
                    margin=float(preset_config["hard_negative_margin"]),
                    pair_scope_mode=str(preset_config["pair_scope_mode"]),
                )
                total_loss = total_loss + float(preset_config["hard_negative_weight"]) * hard_negative_loss
            elif preset_config["hard_negative_mode"] == "same_type_diff_anchor" and float(preset_config["hard_negative_weight"]) > 0.0:
                batch_anchor_labels = [harmonized_anchors[index] for index in batch_indices]
                batch_sample_types = [sample_types[index] for index in batch_indices]
                batch_dataset_ids = [dataset_ids[index] for index in batch_indices]
                hard_negative_loss = anchor_hard_negative_penalty(
                    torch.cat([z1, z2], dim=0),
                    batch_anchor_labels + batch_anchor_labels,
                    batch_sample_types + batch_sample_types,
                    batch_dataset_ids + batch_dataset_ids,
                    margin=float(preset_config["hard_negative_margin"]),
                )
                total_loss = total_loss + float(preset_config["hard_negative_weight"]) * hard_negative_loss

            total_loss.backward()
            optimizer.step()
            running_loss += float(total_loss.item())
            running_instance += float(instance_loss.item())
            running_semantic += float(semantic_loss.item())
            running_anchor += float(anchor_loss.item())
            running_state += float(state_loss.item())
            running_class += float(class_loss.item())
            running_probe += float(probe_loss.item())
            running_compactness += float(compactness_loss.item())
            running_hard_negative += float(hard_negative_loss.item())
            running_variance += float(anchor_variance_loss.item())
            running_anchor_coverage += batch_anchor_fraction
            running_cross_anchor_fraction += batch_anchor_cross_fraction
            running_anchor_entropy += batch_anchor_entropy
            batch_count += 1

        epoch_loss = running_loss / max(batch_count, 1)
        branch_state_distribution_value = (
            "|".join(f"{k}:{v}" for k, v in sorted(running_branch_state_distribution.items()))
            if running_branch_state_distribution
            else "not_applicable"
        )
        log_rows.append(
            {
                "epoch": epoch,
                "loss": epoch_loss,
                "instance_loss": running_instance / max(batch_count, 1),
                "semantic_loss": running_semantic / max(batch_count, 1),
                "anchor_loss": running_anchor / max(batch_count, 1),
                "state_positive_loss": running_state / max(batch_count, 1),
                "class_positive_loss": running_class / max(batch_count, 1),
                "probe_confusion_loss": running_probe / max(batch_count, 1),
                "class_compactness_loss": running_compactness / max(batch_count, 1),
                "hard_negative_loss": running_hard_negative / max(batch_count, 1),
                "anchor_variance_loss": running_variance / max(batch_count, 1),
                "anchor_coverage_fraction": running_anchor_coverage / max(batch_count, 1),
                "cross_dataset_anchor_fraction": running_cross_anchor_fraction / max(batch_count, 1),
                "state_pair_fraction": running_state_pair_fraction / max(batch_count, 1),
                "cross_dataset_state_pair_fraction": running_cross_dataset_state_pair_fraction / max(batch_count, 1),
                "cross_probe_positive_fraction": running_cross_probe_positive_fraction / max(batch_count, 1),
                "same_probe_positive_fraction": running_same_probe_positive_fraction / max(batch_count, 1),
                "branch_state_distribution": branch_state_distribution_value,
                "dataset_entropy_within_anchor": running_anchor_entropy / max(batch_count, 1),
                "device": device.type,
                "batch_size": batch_size,
                "steps": batch_count,
            }
        )
        pairing_epoch_rows.append(
            {
                "epoch": epoch,
                "cross_dataset_anchor_fraction": epoch_anchor_cross_pairs / epoch_anchor_pairs if epoch_anchor_pairs else 0.0,
                "anchor_coverage_fraction": running_anchor_coverage / max(batch_count, 1),
                "anchor_loss": running_anchor / max(batch_count, 1),
                "instance_loss": running_instance / max(batch_count, 1),
                "state_positive_loss": running_state / max(batch_count, 1),
                "class_positive_loss": running_class / max(batch_count, 1),
                "probe_confusion_loss": running_probe / max(batch_count, 1),
                "class_compactness_loss": running_compactness / max(batch_count, 1),
                "state_pair_fraction": running_state_pair_fraction / max(batch_count, 1),
                "cross_dataset_state_pair_fraction": running_cross_dataset_state_pair_fraction / max(batch_count, 1),
                "cross_probe_positive_fraction": running_cross_probe_positive_fraction / max(batch_count, 1),
                "same_probe_positive_fraction": running_same_probe_positive_fraction / max(batch_count, 1),
                "hard_negative_loss": running_hard_negative / max(batch_count, 1),
                "anchor_variance_loss": running_variance / max(batch_count, 1),
                "dataset_entropy_within_anchor": running_anchor_entropy / max(batch_count, 1),
                "anchor_positive_pairs": epoch_anchor_pairs,
                "anchor_positive_pairs_cross_dataset": epoch_anchor_cross_pairs,
                "anchor_positive_pairs_same_dataset": epoch_anchor_same_pairs,
                "anchor_samples": epoch_anchor_samples,
                "state_positive_pairs": epoch_state_pairs,
                "state_positive_pairs_cross_dataset": epoch_state_cross_pairs,
                "within_class_cross_probe_fraction": epoch_cross_probe_fraction / max(batch_count, 1),
                "branch_state_distribution": branch_state_distribution_value,
            }
        )
        print(
            f"epoch={epoch:02d} loss={epoch_loss:.6f} "
            f"instance={running_instance / max(batch_count, 1):.6f} "
            f"semantic={running_semantic / max(batch_count, 1):.6f} "
            f"anchor={running_anchor / max(batch_count, 1):.6f} "
            f"state={running_state / max(batch_count, 1):.6f} "
            f"class={running_class / max(batch_count, 1):.6f} "
            f"probe={running_probe / max(batch_count, 1):.6f} "
            f"compact={running_compactness / max(batch_count, 1):.6f} "
            f"hardneg={running_hard_negative / max(batch_count, 1):.6f} "
            f"var={running_variance / max(batch_count, 1):.6f} "
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
            "branch_mode": args.branch_mode,
            "init_checkpoint": str(Path(args.init_checkpoint).expanduser().resolve()) if args.init_checkpoint else "",
            "preset": args.preset,
            "positive_pair_mode": preset_config["positive_pair_mode"],
            "instance_positive_weight": preset_config["instance_positive_weight"],
            "anchor_positive_weight": preset_config["anchor_positive_weight"],
            "semantic_positive_weight": preset_config["semantic_positive_weight"],
            "state_positive_weight": preset_config.get("state_positive_weight", 0.0),
            "class_positive_weight": preset_config.get("class_positive_weight", 0.0),
            "probe_confusion_weight": preset_config.get("probe_confusion_weight", 0.0),
            "cross_probe_positive_boost": preset_config.get("cross_probe_positive_boost", 1.0),
            "same_probe_positive_weight": preset_config.get("same_probe_positive_weight", 1.0),
            "class_compactness_weight": preset_config.get("class_compactness_weight", 0.0),
            "class_compactness_target_radius": preset_config.get("class_compactness_target_radius", 0.60),
            "hard_negative_mode": preset_config["hard_negative_mode"],
            "hard_negative_weight": preset_config["hard_negative_weight"],
            "hard_negative_margin": preset_config["hard_negative_margin"],
            "augmentation_mode": preset_config["augmentation_mode"],
            "augmentation_strength": preset_config["augmentation_strength"],
            "pair_scope_mode": preset_config["pair_scope_mode"],
            "cross_dataset_positive_mode": preset_config["cross_dataset_positive_mode"],
            "anchor_variance_weight": preset_config["anchor_variance_weight"],
            "anchor_variance_target_std": preset_config["anchor_variance_target_std"],
        },
        output_dir / "model.pt",
    )
    with (output_dir / "training_log.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "epoch",
                "loss",
                "instance_loss",
                "semantic_loss",
                "anchor_loss",
                "state_positive_loss",
                "class_positive_loss",
                "probe_confusion_loss",
                "class_compactness_loss",
                "hard_negative_loss",
                "anchor_variance_loss",
                "anchor_coverage_fraction",
                "cross_dataset_anchor_fraction",
                "state_pair_fraction",
                "cross_dataset_state_pair_fraction",
                "cross_probe_positive_fraction",
                "same_probe_positive_fraction",
                "branch_state_distribution",
                "dataset_entropy_within_anchor",
                "device",
                "batch_size",
                "steps",
            ],
        )
        writer.writeheader()
        writer.writerows(log_rows)

    with (output_dir / "pairing_summary_v7.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "epoch",
                "cross_dataset_anchor_fraction",
                "anchor_coverage_fraction",
                "anchor_loss",
                "instance_loss",
                "state_positive_loss",
                "class_positive_loss",
                "probe_confusion_loss",
                "class_compactness_loss",
                "state_pair_fraction",
                "cross_dataset_state_pair_fraction",
                "cross_probe_positive_fraction",
                "same_probe_positive_fraction",
                "hard_negative_loss",
                "anchor_variance_loss",
                "dataset_entropy_within_anchor",
                "anchor_positive_pairs",
                "anchor_positive_pairs_cross_dataset",
                "anchor_positive_pairs_same_dataset",
                "anchor_samples",
                "state_positive_pairs",
                "state_positive_pairs_cross_dataset",
                "within_class_cross_probe_fraction",
                "branch_state_distribution",
            ],
        )
        writer.writeheader()
        writer.writerows(pairing_epoch_rows)

    if args.branch_mode == "ev_stress":
        with (output_dir / "pairing_summary_pass8_ev.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(pairing_epoch_rows[0].keys()) if pairing_epoch_rows else ["epoch"])
            writer.writeheader()
            writer.writerows(pairing_epoch_rows)
    if args.branch_mode in {"small2023_specialized", "small2023_cellline", "small2023_mixture"}:
        with (output_dir / "pairing_summary_pass8_small2023.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(pairing_epoch_rows[0].keys()) if pairing_epoch_rows else ["epoch"])
            writer.writeheader()
            writer.writerows(pairing_epoch_rows)

    total_semantic_pairs = pairing_summary["semantic_positive_pairs"]
    cross_pairs = pairing_summary["semantic_positive_pairs_cross_dataset"]
    same_pairs = pairing_summary["semantic_positive_pairs_same_dataset"]
    total_anchor_pairs = pairing_summary["anchor_positive_pairs"]
    anchor_cross_pairs = pairing_summary["anchor_positive_pairs_cross_dataset"]
    anchor_same_pairs = pairing_summary["anchor_positive_pairs_same_dataset"]
    with (output_dir / "pairing_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "semantic_positive_pairs",
                "semantic_positive_pairs_cross_dataset",
                "semantic_positive_pairs_same_dataset",
                "cross_dataset_fraction",
                "same_dataset_fraction",
                "pair_scope_mode",
                "cross_dataset_positive_mode",
                "anchor_positive_pairs",
                "anchor_positive_pairs_cross_dataset",
                "anchor_positive_pairs_same_dataset",
                "anchor_cross_dataset_fraction",
                "anchor_same_dataset_fraction",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "semantic_positive_pairs": total_semantic_pairs,
                "semantic_positive_pairs_cross_dataset": cross_pairs,
                "semantic_positive_pairs_same_dataset": same_pairs,
                "cross_dataset_fraction": cross_pairs / total_semantic_pairs if total_semantic_pairs else 0.0,
                "same_dataset_fraction": same_pairs / total_semantic_pairs if total_semantic_pairs else 0.0,
                "pair_scope_mode": preset_config["pair_scope_mode"],
                "cross_dataset_positive_mode": preset_config["cross_dataset_positive_mode"],
                "anchor_positive_pairs": total_anchor_pairs,
                "anchor_positive_pairs_cross_dataset": anchor_cross_pairs,
                "anchor_positive_pairs_same_dataset": anchor_same_pairs,
                "anchor_cross_dataset_fraction": anchor_cross_pairs / total_anchor_pairs if total_anchor_pairs else 0.0,
                "anchor_same_dataset_fraction": anchor_same_pairs / total_anchor_pairs if total_anchor_pairs else 0.0,
            }
        )
    (output_dir / "pairing_summary.md").write_text(
        (
            "Pairing summary\n\n"
            f"- pair_scope_mode: {preset_config['pair_scope_mode']}\n"
            f"- cross_dataset_positive_mode: {preset_config['cross_dataset_positive_mode']}\n"
            f"- semantic_positive_pairs: {total_semantic_pairs:.0f}\n"
            f"- semantic_positive_pairs_cross_dataset: {cross_pairs:.0f}\n"
            f"- semantic_positive_pairs_same_dataset: {same_pairs:.0f}\n"
            f"- cross_dataset_fraction: {(cross_pairs / total_semantic_pairs if total_semantic_pairs else 0.0):.4f}\n"
            f"- same_dataset_fraction: {(same_pairs / total_semantic_pairs if total_semantic_pairs else 0.0):.4f}\n"
            f"- anchor_positive_pairs: {total_anchor_pairs:.0f}\n"
            f"- anchor_positive_pairs_cross_dataset: {anchor_cross_pairs:.0f}\n"
            f"- anchor_positive_pairs_same_dataset: {anchor_same_pairs:.0f}\n"
            f"- anchor_cross_dataset_fraction: {(anchor_cross_pairs / total_anchor_pairs if total_anchor_pairs else 0.0):.4f}\n"
            f"- anchor_same_dataset_fraction: {(anchor_same_pairs / total_anchor_pairs if total_anchor_pairs else 0.0):.4f}\n"
        ),
        encoding="utf-8",
    )

    print(f"Saved model: {output_dir / 'model.pt'}")


if __name__ == "__main__":
    main()
