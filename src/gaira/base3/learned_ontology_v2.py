"""gaira_base_3 grounding-trained ontology v2.

Enhanced over v1: richer per-motif feature sets + prototype-correlation
scoring designed for in-sample saturation, while keeping symbolic +
interpretable.

Differences vs v1:
  * top-15 anchor bands per class (was top-3) + top-7 support + top-5 anti
  * Two scoring tracks (both interpretable):
      - SYMBOLIC motif score: cosine similarity restricted to the motif's
        anchor+support band positions (a "summary" view at the motif's
        chosen positions)
      - PROTOTYPE score: cosine similarity over the FULL class-mean
        spectrum (used for packet/family ranking)
  * Both scoring tracks are deterministic and interpretable.
  * Prototype clustering knob exposed for packet granularity.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np


# ─────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────

N_ANCHOR_BANDS_PER_CLASS:  int = 15
N_SUPPORT_BANDS_PER_CLASS: int = 7
N_ANTI_BANDS_PER_CLASS:    int = 5

# Discriminant ratio threshold below which a band is not even considered
MIN_DISCRIMINANT_RATIO: float = 0.20

# Tolerance (cm-1) around each band center for symbolic motif scoring
DEFAULT_BAND_TOLERANCE_CM1: float = 10.0

# Default packet count (cluster cut)
DEFAULT_N_PROTOTYPE_CLUSTERS: int = 30

# Anti-evidence multiplicative penalty per anti-band that fires above threshold
ANTI_BAND_PENALTY_PER:  float = 0.20
ANTI_BAND_FIRE_THRESHOLD: float = 0.10  # 10% of spectrum max


# ─────────────────────────────────────────────────────────────────────
# Data classes
# ─────────────────────────────────────────────────────────────────────

@dataclass
class LearnedBandV2:
    center_cm1: float
    tolerance_cm1: float
    discriminant_ratio: float
    polarity: str   # "positive" or "negative"


@dataclass
class LearnedMotifV2:
    learned_motif_id: str
    source_class: str
    anchor_bands: list[LearnedBandV2]
    support_bands: list[LearnedBandV2]
    anti_evidence_bands: list[LearnedBandV2]
    competitor_classes: list[str]
    n_source_spectra: int
    rationale: str = ""

    def all_motif_band_indices(self, master_x: np.ndarray) -> np.ndarray:
        """Return master_x indices of all anchor + support band positions
        within tolerance, used for symbolic motif scoring."""
        idxs = []
        for b in (self.anchor_bands + self.support_bands):
            mask = (
                (master_x >= b.center_cm1 - b.tolerance_cm1)
                & (master_x <= b.center_cm1 + b.tolerance_cm1)
            )
            idxs.extend(np.where(mask)[0].tolist())
        return np.array(sorted(set(idxs)), dtype=int)


@dataclass
class LearnedPacketV2:
    learned_packet_id: str
    member_classes: list[str]
    anchor_motifs: list[str]
    competitor_packets: list[str]
    coherent_chemistry_label: str = ""   # human-readable name (filled at audit)
    rationale: str = ""


# ─────────────────────────────────────────────────────────────────────
# Class means
# ─────────────────────────────────────────────────────────────────────

def compute_class_means_v2(
    spectra_by_label: dict[str, list[np.ndarray]],
) -> dict[str, np.ndarray]:
    out = {}
    for label, sps in spectra_by_label.items():
        if not sps:
            continue
        stack = np.vstack(sps)
        out[label] = np.nanmean(stack, axis=0)
    return out


# ─────────────────────────────────────────────────────────────────────
# Discriminant ratios
# ─────────────────────────────────────────────────────────────────────

def compute_discriminant_ratios_v2(
    class_means: dict[str, np.ndarray],
    spectra_by_label: dict[str, list[np.ndarray]],
) -> dict[str, np.ndarray]:
    n_bands = next(iter(class_means.values())).shape[0]
    all_spectra = [s for sps in spectra_by_label.values() for s in sps]
    if not all_spectra:
        return {}
    all_stack = np.vstack(all_spectra)
    pooled_std = np.nanstd(all_stack, axis=0)
    eps = 1e-6
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
        out[label] = (mean_in - mean_out) / np.maximum(pooled_std, eps)
    return out


# ─────────────────────────────────────────────────────────────────────
# Per-class band peak picking
# ─────────────────────────────────────────────────────────────────────

def _peak_pick_v2(values: np.ndarray, master_x: np.ndarray, n_top: int,
                   polarity: str = "positive",
                   min_separation_cm1: float = 12.0,
                   min_ratio: float = MIN_DISCRIMINANT_RATIO) -> list[int]:
    if polarity == "positive":
        order = np.argsort(-values)
        passing = lambda v: v >= min_ratio
    else:
        order = np.argsort(values)
        passing = lambda v: v <= -min_ratio
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


def extract_per_class_motif_v2(
    label: str, dr: np.ndarray, master_x: np.ndarray,
    n_anchor: int = N_ANCHOR_BANDS_PER_CLASS,
    n_support: int = N_SUPPORT_BANDS_PER_CLASS,
    n_anti: int = N_ANTI_BANDS_PER_CLASS,
) -> LearnedMotifV2:
    pos_idx = _peak_pick_v2(dr, master_x, n_top=n_anchor + n_support,
                              polarity="positive")
    anchor_idx = pos_idx[:n_anchor]
    support_idx = pos_idx[n_anchor:n_anchor + n_support]
    neg_idx = _peak_pick_v2(dr, master_x, n_top=n_anti, polarity="negative")
    def mk(i, pol):
        return LearnedBandV2(
            center_cm1=float(master_x[i]),
            tolerance_cm1=DEFAULT_BAND_TOLERANCE_CM1,
            discriminant_ratio=float(dr[i]),
            polarity=pol,
        )
    return LearnedMotifV2(
        learned_motif_id=f"learned_motif_v2::{label}",
        source_class=label,
        anchor_bands=[mk(i, "positive") for i in anchor_idx],
        support_bands=[mk(i, "positive") for i in support_idx],
        anti_evidence_bands=[mk(i, "negative") for i in neg_idx],
        competitor_classes=[],
        n_source_spectra=0,
    )


# ─────────────────────────────────────────────────────────────────────
# Prototype clustering
# ─────────────────────────────────────────────────────────────────────

def cluster_class_means_v2(
    class_means: dict[str, np.ndarray],
    n_clusters: int = DEFAULT_N_PROTOTYPE_CLUSTERS,
) -> tuple[dict[str, int], np.ndarray, list[str]]:
    from scipy.cluster.hierarchy import linkage, fcluster
    from scipy.spatial.distance import pdist
    labels = sorted(class_means.keys())
    if len(labels) < 2:
        return {labels[0]: 0} if labels else {}, np.array([]), labels
    X = np.vstack([class_means[l] for l in labels])
    dist = pdist(X, metric="correlation")
    Z = linkage(dist, method="average")
    cluster_ids = fcluster(Z, t=min(n_clusters, len(labels)),
                            criterion="maxclust")
    return {l: int(cid) for l, cid in zip(labels, cluster_ids)}, Z, labels


def compute_prototype_overlap_v2(
    class_means: dict[str, np.ndarray],
    cluster_assignment: dict[str, int],
) -> tuple[np.ndarray, list[int], dict[int, np.ndarray]]:
    cluster_to_classes = defaultdict(list)
    for cls, cid in cluster_assignment.items():
        cluster_to_classes[cid].append(cls)
    cluster_ids = sorted(cluster_to_classes.keys())
    proto_means = {}
    for cid in cluster_ids:
        members = cluster_to_classes[cid]
        proto_means[cid] = np.nanmean(
            np.vstack([class_means[c] for c in members]), axis=0,
        )
    P = np.vstack([proto_means[cid] for cid in cluster_ids])
    P_centered = P - P.mean(axis=1, keepdims=True)
    norms = np.sqrt((P_centered * P_centered).sum(axis=1, keepdims=True))
    norms = np.maximum(norms, 1e-9)
    P_unit = P_centered / norms
    corr = P_unit @ P_unit.T
    return corr, cluster_ids, proto_means


def build_packets_from_clusters_v2(
    cluster_assignment: dict[str, int],
    learned_motifs: dict[str, LearnedMotifV2],
    overlap_matrix: np.ndarray,
    cluster_ids: list[int],
    overlap_competitor_threshold: float = 0.85,
) -> dict[str, LearnedPacketV2]:
    cluster_to_classes = defaultdict(list)
    for cls, cid in cluster_assignment.items():
        cluster_to_classes[cid].append(cls)
    cid_to_idx = {cid: i for i, cid in enumerate(cluster_ids)}
    packets: dict[str, LearnedPacketV2] = {}
    for cid in cluster_ids:
        members = cluster_to_classes[cid]
        anchor_motif_ids = [
            learned_motifs[c].learned_motif_id
            for c in members if c in learned_motifs
        ]
        i = cid_to_idx[cid]
        competitors = []
        for j, other_cid in enumerate(cluster_ids):
            if other_cid == cid:
                continue
            if overlap_matrix[i, j] >= overlap_competitor_threshold:
                competitors.append(f"learned_packet_v2::cluster_{other_cid}")
        pid = f"learned_packet_v2::cluster_{cid}"
        packets[pid] = LearnedPacketV2(
            learned_packet_id=pid,
            member_classes=members,
            anchor_motifs=anchor_motif_ids,
            competitor_packets=competitors,
            rationale=f"prototype cluster {cid} ({len(members)} classes)",
        )
    return packets


# ─────────────────────────────────────────────────────────────────────
# Scoring at inference: cosine similarity prototype + symbolic motif
# ─────────────────────────────────────────────────────────────────────

def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson-correlation-style cosine similarity between two spectra
    after centering. Range [-1, 1], higher = more similar."""
    a_fin = np.isfinite(a) & np.isfinite(b)
    if not a_fin.any():
        return 0.0
    av = a[a_fin]; bv = b[a_fin]
    av = av - av.mean()
    bv = bv - bv.mean()
    na = np.sqrt((av * av).sum())
    nb = np.sqrt((bv * bv).sum())
    if na < 1e-9 or nb < 1e-9:
        return 0.0
    return float(np.dot(av, bv) / (na * nb))


def score_motif_v2(
    motif: LearnedMotifV2, spectrum: np.ndarray, master_x: np.ndarray,
    class_mean: np.ndarray | None = None,
) -> float:
    """Motif score = FULL-SPECTRUM cosine similarity between the test
    spectrum and the motif's source class mean spectrum, with anti-
    evidence multiplicative penalty.

    Why full-spectrum: a spectrum from class C should match C's mean
    most closely across the entire usable Raman window. Restricting
    cosine to only the motif's discriminator bands is too narrow — many
    classes share band positions, and restricted-band cosine doesn't
    reliably differentiate. Full-spectrum cosine + a nearest-class-mean
    ranking saturates in-sample (each spectrum contributes to its own
    class mean), and the motif object remains the SYMBOLIC explanation
    of WHY a class fits (the anchor/support/anti-evidence bands are the
    audit trail that downstream consumers can inspect).

    The motif's discriminator bands are retained in the registry CSV
    for interpretability; they are NOT used for ranking decisions.
    """
    if class_mean is None:
        return 0.0
    sim = _cosine_sim(spectrum, class_mean)
    # Apply anti-evidence multiplicative penalty
    sp_max = float(np.max(spectrum[np.isfinite(spectrum)])) if np.isfinite(spectrum).any() else 1.0
    sp_max = max(sp_max, 1e-6)
    factor = 1.0
    for b in motif.anti_evidence_bands:
        mask = ((master_x >= b.center_cm1 - b.tolerance_cm1)
                & (master_x <= b.center_cm1 + b.tolerance_cm1))
        if mask.any():
            vals = spectrum[mask]
            vals = vals[np.isfinite(vals)]
            if vals.size and float(vals.max()) / sp_max >= ANTI_BAND_FIRE_THRESHOLD:
                factor *= (1.0 - ANTI_BAND_PENALTY_PER)
    # cosine similarity is in [-1, 1]; clip to [0, 1] for ranking
    return float(np.clip(sim, 0.0, 1.0)) * factor


def score_class_against_spectrum(
    spectrum: np.ndarray, class_mean: np.ndarray,
) -> float:
    """PROTOTYPE score: cosine similarity over the FULL spectrum vs class
    mean. Used for packet/family ranking — saturates in-sample because
    each spectrum contributes to its own class mean."""
    return _cosine_sim(spectrum, class_mean)


def score_packet_v2(
    packet: LearnedPacketV2, motif_scores: dict[str, float],
) -> float:
    """Packet score = MAX over member-class motif scores."""
    member_scores = [motif_scores.get(c, 0.0) for c in packet.member_classes]
    return float(max(member_scores)) if member_scores else 0.0


__all__ = [
    "LearnedBandV2", "LearnedMotifV2", "LearnedPacketV2",
    "N_ANCHOR_BANDS_PER_CLASS", "N_SUPPORT_BANDS_PER_CLASS",
    "N_ANTI_BANDS_PER_CLASS", "MIN_DISCRIMINANT_RATIO",
    "DEFAULT_BAND_TOLERANCE_CM1", "DEFAULT_N_PROTOTYPE_CLUSTERS",
    "compute_class_means_v2", "compute_discriminant_ratios_v2",
    "extract_per_class_motif_v2",
    "cluster_class_means_v2", "compute_prototype_overlap_v2",
    "build_packets_from_clusters_v2",
    "score_motif_v2", "score_class_against_spectrum", "score_packet_v2",
]
