"""gaira_base_3 grounding-trained ontology.

Interpretable discriminative learning from grounding spectra. Methods are
deterministic and explicit so that learned motifs/packets/families remain
auditable and exportable to the GAIRA ontology stack.

Pipeline:
  1. class_means(spectra_by_label, master_x) -> dict[label, mean_spectrum]
  2. per-band discriminant ratio (one-vs-rest) per class
  3. extract per-class anchor / support / anti-evidence bands
  4. hierarchical clustering of class means -> prototype groups
  5. score a held-out spectrum against the learned motif set
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Iterable

import numpy as np


# ─────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────

# How many top positive bands per class become anchor-grade
N_ANCHOR_BANDS_PER_CLASS: int = 3
# How many additional positive bands become support-grade
N_SUPPORT_BANDS_PER_CLASS: int = 4
# How many top negative bands become anti-evidence
N_ANTI_BANDS_PER_CLASS: int = 3

# Minimum discriminant ratio for a band to count as a discriminator at all
MIN_DISCRIMINANT_RATIO: float = 0.30
# Tolerance (cm-1) around each band center when scoring a held-out spectrum
DEFAULT_BAND_TOLERANCE_CM1: float = 12.0

# Hierarchical clustering: cut tree at this many clusters
DEFAULT_N_PROTOTYPE_CLUSTERS: int = 24

# Scoring weights
ANCHOR_BAND_WEIGHT:    float = 1.00
SUPPORT_BAND_WEIGHT:   float = 0.50
ANTI_BAND_PENALTY_PER: float = 0.30  # multiplicative penalty per anti-evidence hit


# ─────────────────────────────────────────────────────────────────────
# Data classes
# ─────────────────────────────────────────────────────────────────────

@dataclass
class LearnedBand:
    """A learned discriminative band region."""
    center_cm1: float
    tolerance_cm1: float
    discriminant_ratio: float
    polarity: str   # "positive" or "negative"


@dataclass
class LearnedMotif:
    """An interpretable motif derived from grounding-trained discriminators."""
    learned_motif_id: str
    source_class: str
    anchor_bands: list[LearnedBand]
    support_bands: list[LearnedBand]
    anti_evidence_bands: list[LearnedBand]
    competitor_classes: list[str]
    n_source_spectra: int
    rationale: str = ""


@dataclass
class LearnedPacket:
    """A learned packet (prototype cluster of related classes)."""
    learned_packet_id: str
    member_classes: list[str]
    anchor_motifs: list[str]      # learned_motif_ids
    support_motifs: list[str]
    competitor_packets: list[str]
    rationale: str = ""


# ─────────────────────────────────────────────────────────────────────
# Step A: class-mean spectra
# ─────────────────────────────────────────────────────────────────────

def compute_class_means(
    spectra_by_label: dict[str, list[np.ndarray]],
) -> dict[str, np.ndarray]:
    """Mean spectrum per label. Spectra must already be on common axis +
    L2-normalised (canonical preprocessing is responsible for that)."""
    out = {}
    for label, sps in spectra_by_label.items():
        if not sps:
            continue
        stack = np.vstack(sps)
        out[label] = np.nanmean(stack, axis=0)
    return out


# ─────────────────────────────────────────────────────────────────────
# Step B: per-band discriminant ratio (one-vs-rest)
# ─────────────────────────────────────────────────────────────────────

def compute_discriminant_ratios(
    class_means: dict[str, np.ndarray],
    spectra_by_label: dict[str, list[np.ndarray]],
) -> dict[str, np.ndarray]:
    """For each class, compute per-band discriminant ratio:
        DR_b = (mean_class_b - mean_other_b) / max(pooled_std_b, eps)

    Positive DR -> band fires more in this class than in others.
    Negative DR -> band fires LESS in this class than in others (anti-evidence
    candidate).
    """
    n_bands = next(iter(class_means.values())).shape[0]

    # Pooled std across all spectra (regardless of class)
    all_spectra = []
    for sps in spectra_by_label.values():
        all_spectra.extend(sps)
    if not all_spectra:
        return {}
    all_stack = np.vstack(all_spectra)
    pooled_std = np.nanstd(all_stack, axis=0)
    eps = 1e-6

    # For each class, compute "other" mean = mean over spectra NOT in class
    out = {}
    for label, mean_in in class_means.items():
        spectra_in = spectra_by_label.get(label, [])
        ids_in = set(id(s) for s in spectra_in)
        spectra_out = [s for sps_list in spectra_by_label.values()
                        for s in sps_list if id(s) not in ids_in]
        if not spectra_out:
            out[label] = np.zeros(n_bands)
            continue
        mean_out = np.nanmean(np.vstack(spectra_out), axis=0)
        dr = (mean_in - mean_out) / np.maximum(pooled_std, eps)
        out[label] = dr
    return out


# ─────────────────────────────────────────────────────────────────────
# Step C: extract per-class anchor / support / anti-evidence bands
# ─────────────────────────────────────────────────────────────────────

def _peak_pick(values: np.ndarray, master_x: np.ndarray, n_top: int,
               polarity: str = "positive",
               min_separation_cm1: float = 18.0,
               min_ratio: float = MIN_DISCRIMINANT_RATIO) -> list[int]:
    """Pick top-N discriminative band-CENTERS by sorted discriminant ratio,
    enforcing minimum spacing between picks (so we don't pick 3 adjacent
    bins for what is really one peak)."""
    if polarity == "positive":
        order = np.argsort(-values)  # descending
        threshold = min_ratio
        passing = lambda v: v >= threshold
    else:
        order = np.argsort(values)   # ascending (most negative first)
        threshold = -min_ratio
        passing = lambda v: v <= threshold
    picks: list[int] = []
    for idx in order:
        v = values[idx]
        if not passing(v):
            break
        cm = master_x[idx]
        if any(abs(master_x[p] - cm) < min_separation_cm1 for p in picks):
            continue
        picks.append(int(idx))
        if len(picks) >= n_top:
            break
    return picks


def extract_per_class_motif(
    label: str, dr: np.ndarray, master_x: np.ndarray,
    n_anchor: int = N_ANCHOR_BANDS_PER_CLASS,
    n_support: int = N_SUPPORT_BANDS_PER_CLASS,
    n_anti: int = N_ANTI_BANDS_PER_CLASS,
) -> LearnedMotif:
    """Convert per-band discriminator vector into a LearnedMotif object."""
    pos_idx = _peak_pick(dr, master_x, n_top=n_anchor + n_support,
                          polarity="positive")
    anchor_idx = pos_idx[:n_anchor]
    support_idx = pos_idx[n_anchor:n_anchor + n_support]
    neg_idx = _peak_pick(dr, master_x, n_top=n_anti, polarity="negative")

    anchor_bands = [LearnedBand(center_cm1=float(master_x[i]),
                                tolerance_cm1=DEFAULT_BAND_TOLERANCE_CM1,
                                discriminant_ratio=float(dr[i]),
                                polarity="positive") for i in anchor_idx]
    support_bands = [LearnedBand(center_cm1=float(master_x[i]),
                                  tolerance_cm1=DEFAULT_BAND_TOLERANCE_CM1,
                                  discriminant_ratio=float(dr[i]),
                                  polarity="positive") for i in support_idx]
    anti_bands = [LearnedBand(center_cm1=float(master_x[i]),
                                tolerance_cm1=DEFAULT_BAND_TOLERANCE_CM1,
                                discriminant_ratio=float(dr[i]),
                                polarity="negative") for i in neg_idx]
    return LearnedMotif(
        learned_motif_id=f"learned_motif::{label}",
        source_class=label,
        anchor_bands=anchor_bands,
        support_bands=support_bands,
        anti_evidence_bands=anti_bands,
        competitor_classes=[],
        n_source_spectra=0,
        rationale="",
    )


# ─────────────────────────────────────────────────────────────────────
# Step D: prototype clustering + packet construction
# ─────────────────────────────────────────────────────────────────────

def cluster_class_means(
    class_means: dict[str, np.ndarray],
    n_clusters: int = DEFAULT_N_PROTOTYPE_CLUSTERS,
) -> tuple[dict[str, int], np.ndarray, list[str]]:
    """Hierarchical agglomerative clustering of class-mean spectra by
    correlation distance. Returns:
      - dict[class_label, cluster_id]
      - linkage matrix
      - ordered class labels
    """
    from scipy.cluster.hierarchy import linkage, fcluster
    from scipy.spatial.distance import pdist
    labels = sorted(class_means.keys())
    if len(labels) < 2:
        return {labels[0]: 0} if labels else {}, np.array([]), labels
    X = np.vstack([class_means[l] for l in labels])
    # correlation distance handles overall intensity differences well
    dist = pdist(X, metric="correlation")
    Z = linkage(dist, method="average")
    cluster_ids = fcluster(Z, t=n_clusters, criterion="maxclust")
    return {l: int(cid) for l, cid in zip(labels, cluster_ids)}, Z, labels


def compute_prototype_overlap(
    class_means: dict[str, np.ndarray],
    cluster_assignment: dict[str, int],
) -> tuple[np.ndarray, list[int]]:
    """Compute prototype-level mean spectra and the inter-prototype
    correlation matrix. Returns (correlation_matrix, ordered_cluster_ids)."""
    cluster_to_classes = defaultdict(list)
    for cls, cid in cluster_assignment.items():
        cluster_to_classes[cid].append(cls)
    cluster_ids = sorted(cluster_to_classes.keys())
    proto_means = []
    for cid in cluster_ids:
        members = cluster_to_classes[cid]
        proto_means.append(np.nanmean(np.vstack([class_means[c] for c in members]),
                                        axis=0))
    P = np.vstack(proto_means)
    # correlation matrix between prototypes
    P_centered = P - P.mean(axis=1, keepdims=True)
    norms = np.sqrt((P_centered * P_centered).sum(axis=1, keepdims=True))
    norms = np.maximum(norms, 1e-9)
    P_unit = P_centered / norms
    corr = P_unit @ P_unit.T
    return corr, cluster_ids


def build_packets_from_clusters(
    cluster_assignment: dict[str, int],
    learned_motifs: dict[str, LearnedMotif],
    overlap_matrix: np.ndarray,
    cluster_ids: list[int],
    overlap_competitor_threshold: float = 0.70,
) -> dict[str, LearnedPacket]:
    """Each prototype cluster becomes a learned packet. Anchor motifs =
    the learned motifs of all member classes (chemistry-similar classes
    co-fire). Competitor packets = prototypes whose mean spectrum
    correlates above threshold with this packet's mean spectrum."""
    cluster_to_classes = defaultdict(list)
    for cls, cid in cluster_assignment.items():
        cluster_to_classes[cid].append(cls)

    cid_to_idx = {cid: i for i, cid in enumerate(cluster_ids)}
    packets: dict[str, LearnedPacket] = {}
    for cid in cluster_ids:
        members = cluster_to_classes[cid]
        anchor_motif_ids = [
            learned_motifs[c].learned_motif_id
            for c in members if c in learned_motifs
        ]
        # competitors: other clusters with overlap above threshold
        i = cid_to_idx[cid]
        competitors = []
        for j, other_cid in enumerate(cluster_ids):
            if other_cid == cid:
                continue
            if overlap_matrix[i, j] >= overlap_competitor_threshold:
                competitors.append(f"learned_packet::cluster_{other_cid}")
        pid = f"learned_packet::cluster_{cid}"
        packets[pid] = LearnedPacket(
            learned_packet_id=pid,
            member_classes=members,
            anchor_motifs=anchor_motif_ids,
            support_motifs=[],
            competitor_packets=competitors,
            rationale=(f"Prototype cluster {cid} groups {len(members)} "
                       f"chemistry-similar classes by class-mean correlation."),
        )
    return packets


# ─────────────────────────────────────────────────────────────────────
# Step E: scoring a spectrum against a learned ontology
# ─────────────────────────────────────────────────────────────────────

def _band_intensity(spectrum: np.ndarray, master_x: np.ndarray,
                     band: LearnedBand) -> float:
    mask = ((master_x >= band.center_cm1 - band.tolerance_cm1)
            & (master_x <= band.center_cm1 + band.tolerance_cm1))
    if not mask.any():
        return 0.0
    vals = spectrum[mask]
    fin = np.isfinite(vals)
    if not fin.any():
        return 0.0
    return float(np.max(vals[fin]))


def score_motif_on_spectrum(
    motif: LearnedMotif, spectrum: np.ndarray, master_x: np.ndarray,
    spectrum_max: float | None = None,
) -> float:
    """Symbolic motif score from learned bands. Anchor + support +
    anti-evidence; bounded to [0, 1] roughly via per-band intensities
    that are themselves L2-normalised."""
    if spectrum_max is None:
        fin = np.isfinite(spectrum)
        spectrum_max = float(np.max(spectrum[fin])) if fin.any() else 1.0
    sp_max = max(spectrum_max, 1e-6)

    # Anchor + support: sum of intensities normalised by spectrum max
    anchor_sum = sum(_band_intensity(spectrum, master_x, b) / sp_max
                     for b in motif.anchor_bands)
    support_sum = sum(_band_intensity(spectrum, master_x, b) / sp_max
                       for b in motif.support_bands)
    raw = ANCHOR_BAND_WEIGHT * anchor_sum + SUPPORT_BAND_WEIGHT * support_sum
    # normalise by max possible (n_anchor * 1.0 + n_support * 0.5)
    max_raw = (ANCHOR_BAND_WEIGHT * len(motif.anchor_bands)
               + SUPPORT_BAND_WEIGHT * len(motif.support_bands))
    normalised = raw / max(max_raw, 1e-6)

    # Anti-evidence: each anti band that fires above local threshold
    # multiplies by (1 - ANTI_BAND_PENALTY_PER)
    factor = 1.0
    anti_threshold = 0.10  # 10% of spectrum max
    for b in motif.anti_evidence_bands:
        intensity = _band_intensity(spectrum, master_x, b) / sp_max
        if intensity >= anti_threshold:
            factor *= (1.0 - ANTI_BAND_PENALTY_PER)
    return float(np.clip(normalised * factor, 0.0, 1.0))


def score_packet_from_motifs(
    packet: LearnedPacket,
    motif_scores: dict[str, float],
) -> float:
    """Packet score = MAX over anchor motif scores in the packet
    (chemistry-coherent: one strong anchor wins)."""
    anchor_vals = [motif_scores.get(mid, 0.0) for mid in packet.anchor_motifs]
    if not anchor_vals:
        return 0.0
    return float(max(anchor_vals))


__all__ = [
    "LearnedBand", "LearnedMotif", "LearnedPacket",
    "N_ANCHOR_BANDS_PER_CLASS", "N_SUPPORT_BANDS_PER_CLASS",
    "N_ANTI_BANDS_PER_CLASS", "MIN_DISCRIMINANT_RATIO",
    "DEFAULT_BAND_TOLERANCE_CM1", "DEFAULT_N_PROTOTYPE_CLUSTERS",
    "compute_class_means", "compute_discriminant_ratios",
    "extract_per_class_motif",
    "cluster_class_means", "compute_prototype_overlap",
    "build_packets_from_clusters",
    "score_motif_on_spectrum", "score_packet_from_motifs",
]
