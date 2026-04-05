from __future__ import annotations

import torch
import torch.nn.functional as F


def state_confidence_weight(label: str) -> float:
    if label == "intermediate_or_ambiguous":
        return 0.35
    if label:
        return 1.0
    return 0.0


def branch_class_contrastive_loss(
    embeddings: torch.Tensor,
    class_labels: list[str],
    probe_labels: list[str],
    *,
    temperature: float,
    cross_probe_positive_boost: float = 2.0,
    same_probe_positive_weight: float = 0.5,
) -> tuple[torch.Tensor, dict[str, float]]:
    device = embeddings.device
    valid_mask = torch.tensor([bool(label) for label in class_labels], device=device, dtype=torch.bool)
    if int(valid_mask.sum().item()) < 2:
        return torch.zeros((), device=device), {
            "positive_pairs": 0.0,
            "cross_probe_positive_pairs": 0.0,
            "same_probe_positive_pairs": 0.0,
            "cross_probe_positive_fraction": 0.0,
            "within_class_cross_probe_fraction": 0.0,
            "same_probe_positive_fraction": 0.0,
        }

    z = embeddings[valid_mask]
    valid_class_labels = [label for label in class_labels if label]
    valid_probe_labels = [label for class_label, label in zip(class_labels, probe_labels, strict=False) if class_label]
    n = len(valid_class_labels)
    same_class = torch.tensor(
        [[1 if valid_class_labels[i] == valid_class_labels[j] else 0 for j in range(n)] for i in range(n)],
        device=device,
        dtype=torch.bool,
    )
    same_probe = torch.tensor(
        [[1 if valid_probe_labels[i] and valid_probe_labels[i] == valid_probe_labels[j] else 0 for j in range(n)] for i in range(n)],
        device=device,
        dtype=torch.bool,
    )
    eye = torch.eye(n, device=device, dtype=torch.bool)
    base_positive_mask = same_class & ~eye
    cross_probe_mask = base_positive_mask & ~same_probe
    same_probe_mask = base_positive_mask & same_probe

    if int(base_positive_mask.sum().item()) == 0:
        return torch.zeros((), device=device), {
            "positive_pairs": 0.0,
            "cross_probe_positive_pairs": 0.0,
            "same_probe_positive_pairs": 0.0,
            "cross_probe_positive_fraction": 0.0,
            "within_class_cross_probe_fraction": 0.0,
            "same_probe_positive_fraction": 0.0,
        }

    pair_weights = torch.zeros((n, n), device=device, dtype=torch.float32)
    pair_weights[cross_probe_mask] = float(cross_probe_positive_boost)
    pair_weights[same_probe_mask] = float(same_probe_positive_weight)
    positive_counts = pair_weights.sum(dim=1)
    valid_rows = positive_counts > 0
    if int(valid_rows.sum().item()) == 0:
        return torch.zeros((), device=device), {
            "positive_pairs": float(base_positive_mask.sum().item()),
            "cross_probe_positive_pairs": float(cross_probe_mask.sum().item()),
            "same_probe_positive_pairs": float(same_probe_mask.sum().item()),
            "cross_probe_positive_fraction": 0.0,
            "within_class_cross_probe_fraction": 0.0,
            "same_probe_positive_fraction": 0.0,
        }

    logits = torch.matmul(z, z.T) / temperature
    logits = logits.masked_fill(eye, -9e15)
    log_prob = logits - torch.logsumexp(logits, dim=1, keepdim=True)
    mean_log_prob_pos = (pair_weights * log_prob).sum(dim=1) / positive_counts.clamp_min(1.0)
    total_pairs = float(base_positive_mask.sum().item())
    cross_pairs = float(cross_probe_mask.sum().item())
    same_pairs = float(same_probe_mask.sum().item())
    return -mean_log_prob_pos[valid_rows].mean(), {
        "positive_pairs": total_pairs,
        "cross_probe_positive_pairs": cross_pairs,
        "same_probe_positive_pairs": same_pairs,
        "cross_probe_positive_fraction": cross_pairs / total_pairs if total_pairs > 0 else 0.0,
        "within_class_cross_probe_fraction": cross_pairs / total_pairs if total_pairs > 0 else 0.0,
        "same_probe_positive_fraction": same_pairs / total_pairs if total_pairs > 0 else 0.0,
    }


def probe_alignment_penalty(
    embeddings: torch.Tensor,
    class_labels: list[str],
    probe_labels: list[str],
    *,
    margin_same_class: float = 0.05,
    margin_diff_class: float = 0.08,
) -> tuple[torch.Tensor, dict[str, float]]:
    device = embeddings.device
    n = len(class_labels)
    if n < 2:
        return torch.zeros((), device=device), {
            "cross_probe_positive_fraction": 0.0,
            "within_class_cross_probe_fraction": 0.0,
            "same_probe_positive_fraction": 0.0,
        }
    same_class = torch.tensor(
        [[1 if class_labels[i] and class_labels[i] == class_labels[j] else 0 for j in range(n)] for i in range(n)],
        device=device,
        dtype=torch.bool,
    )
    same_probe = torch.tensor(
        [[1 if probe_labels[i] and probe_labels[i] == probe_labels[j] else 0 for j in range(n)] for i in range(n)],
        device=device,
        dtype=torch.bool,
    )
    eye = torch.eye(n, device=device, dtype=torch.bool)
    cross_probe_same_class = same_class & ~same_probe & ~eye
    same_probe_same_class = same_class & same_probe & ~eye
    same_probe_diff_class = ~same_class & same_probe & ~eye
    sim = torch.matmul(embeddings, embeddings.T)
    total_same_class_pairs = int((same_class & ~eye).sum().item())
    if int(cross_probe_same_class.sum().item()) == 0 and int(same_probe_same_class.sum().item()) == 0:
        return torch.zeros((), device=device), {
            "cross_probe_positive_fraction": 0.0,
            "within_class_cross_probe_fraction": 0.0,
            "same_probe_positive_fraction": 0.0,
        }

    if int(cross_probe_same_class.sum().item()) > 0:
        cross_probe_mean = sim[cross_probe_same_class].mean()
    else:
        cross_probe_mean = sim[same_probe_same_class].mean()
    same_probe_same_class_mean = sim[same_probe_same_class].mean() if int(same_probe_same_class.sum().item()) > 0 else cross_probe_mean
    same_probe_diff_class_mean = sim[same_probe_diff_class].mean() if int(same_probe_diff_class.sum().item()) > 0 else same_probe_same_class_mean
    loss = F.relu(margin_same_class + same_probe_same_class_mean - cross_probe_mean)
    loss = loss + F.relu(margin_diff_class + same_probe_diff_class_mean - cross_probe_mean)
    cross_fraction = float(cross_probe_same_class.sum().item()) / float(total_same_class_pairs) if total_same_class_pairs > 0 else 0.0
    same_fraction = float(same_probe_same_class.sum().item()) / float(total_same_class_pairs) if total_same_class_pairs > 0 else 0.0
    return loss, {
        "cross_probe_positive_fraction": cross_fraction,
        "within_class_cross_probe_fraction": cross_fraction,
        "same_probe_positive_fraction": same_fraction,
    }


def class_compactness_loss(
    embeddings: torch.Tensor,
    class_labels: list[str],
    *,
    target_radius: float = 0.60,
) -> tuple[torch.Tensor, dict[str, float]]:
    device = embeddings.device
    grouped: dict[str, list[int]] = {}
    for idx, label in enumerate(class_labels):
        if not label:
            continue
        grouped.setdefault(label, []).append(idx)
    penalties = []
    covered_samples = 0
    for indices in grouped.values():
        if len(indices) < 2:
            continue
        subset = embeddings[indices]
        center = subset.mean(dim=0, keepdim=True)
        distances = torch.norm(subset - center, dim=1)
        penalties.append(F.relu(distances - target_radius).mean())
        covered_samples += len(indices)
    if not penalties:
        return torch.zeros((), device=device), {"compact_classes": 0.0, "compact_samples": 0.0}
    return torch.stack(penalties).mean(), {
        "compact_classes": float(len(penalties)),
        "compact_samples": float(covered_samples),
    }
