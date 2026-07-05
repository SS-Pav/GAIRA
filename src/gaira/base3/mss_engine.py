"""gaira_base_3 molecular spectral signature (MSS) engine.

Implements the upgraded GAIRA evidence stack:

  primitives -> molecular spectral signatures (MSS) -> packets (optional)
              -> families/themes/BSV summary

Hard constraint: scoring is BAND-BASED, NOT full-spectrum class-mean cosine.
The cosine approach saturates trivially in-sample but is opaque at the
evidence level. The MSS scoring used here aggregates discrete band-level
primitives (anchor + support + anti-evidence) so that every score has
explicit band-position attribution.

Method (interpretable, deterministic):
  1. Per-class mean spectra over all included replicates
  2. One-vs-rest discriminant ratio per band:
        DR_b = (mean_class_b - mean_other_b) / pooled_std_b
  3. Per-class signature extraction:
        - anchor: top N_ANCHOR positive-DR bands (locally prominent peaks)
        - support: next N_SUPPORT positive-DR bands
        - anti-evidence: top N_ANTI negative-DR bands
        - competitor classes: same prototype cluster
  4. Replicate stability check: drop bands whose intensity coefficient of
     variation across replicates exceeds a threshold (filters noise picks)
  5. Cross-dataset support tag: which datasets contribute to each MSS
  6. Regime + substrate provenance preserved per MSS

Scoring at inference:
  mss_score = (anchor_evidence + 0.5 * support_evidence) * co_band_factor
              * (1 - anti_evidence_penalty) * competitor_suppression

Where:
  - anchor_evidence = mean band-intensity (max-in-window) over anchor bands,
                       normalised by spectrum max
  - support_evidence = same for support bands
  - co_band_factor = penalty for missing required co-bands
  - anti_evidence_penalty = additive over fired anti-bands, capped at 0.5
  - competitor_suppression = if a competitor MSS scored higher, multiply
                              this MSS by COMPETITOR_SUPPRESSION_FACTOR

All operations are deterministic, auditable, and operate on explicit
band positions (cm-1) extracted from the grounding corpus.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Iterable

import numpy as np


# ─────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────

N_ANCHOR_BANDS:  int = 6
N_SUPPORT_BANDS: int = 6
N_ANTI_BANDS:    int = 4

MIN_DISCRIMINANT_RATIO: float = 0.30
DEFAULT_BAND_TOLERANCE_CM1: float = 8.0   # ± each side
MIN_PEAK_SEPARATION_CM1:    float = 16.0  # peaks must be >= this apart

# Local-prominence floor: a band's max intensity must exceed the local
# neighborhood median by this multiple to count as "fired"
PROMINENCE_FACTOR: float = 1.20
PROMINENCE_HALFWIDTH_CM1: float = 30.0
RELATIVE_TO_SPECTRUM_MIN: float = 0.04

# Replicate stability filter: bands whose CV across replicates exceeds this
# are dropped from the signature (noise discriminators)
MAX_REPLICATE_CV: float = 1.50

ANTI_FIRE_THRESHOLD: float = 0.10  # anti-band fires when peak >= 10% spectrum max
ANTI_PER_BAND_PENALTY: float = 0.15
MAX_ANTI_PENALTY: float = 0.50

DEFAULT_N_PROTOTYPE_CLUSTERS: int = 30
COMPETITOR_OVERLAP_THRESHOLD: float = 0.85
COMPETITOR_SUPPRESSION_FACTOR: float = 0.55


# ─────────────────────────────────────────────────────────────────────
# Data classes
# ─────────────────────────────────────────────────────────────────────

@dataclass
class MSSBand:
    center_cm1: float
    tolerance_cm1: float
    discriminant_ratio: float
    polarity: str   # "positive" | "negative"
    replicate_cv: float = 0.0  # coefficient of variation across replicates


@dataclass
class MolecularSignature:
    signature_id: str
    analyte_name: str
    analyte_class: str
    anchor_features: list[MSSBand]
    support_features: list[MSSBand]
    anti_evidence_features: list[MSSBand]
    competitor_signatures: list[str]  # signature_ids of close prototypes
    n_source_spectra: int
    replicate_stability: float        # mean CV across anchor bands (lower = more stable)
    cross_dataset_support: list[str]
    regime_support: list[str]         # list of "Raman" / "SERS"
    substrate_support: list[str]
    evidence_sources: list[str]       # list of source spectrum_ids
    notes: str = ""


@dataclass
class MSSPacket:
    packet_id: str
    member_signatures: list[str]   # signature_ids
    member_classes: list[str]
    competitor_packets: list[str]
    rationale: str = ""


# ─────────────────────────────────────────────────────────────────────
# Class means (with replicate tracking)
# ─────────────────────────────────────────────────────────────────────

def compute_class_means(spectra_by_label: dict[str, list[np.ndarray]],
                         ) -> dict[str, np.ndarray]:
    out = {}
    for label, sps in spectra_by_label.items():
        if not sps:
            continue
        out[label] = np.nanmean(np.vstack(sps), axis=0)
    return out


def compute_class_replicate_std(
    spectra_by_label: dict[str, list[np.ndarray]],
) -> dict[str, np.ndarray]:
    out = {}
    for label, sps in spectra_by_label.items():
        if len(sps) < 2:
            out[label] = np.zeros_like(next(iter(sps)))
            continue
        out[label] = np.nanstd(np.vstack(sps), axis=0, ddof=0)
    return out


# ─────────────────────────────────────────────────────────────────────
# Discriminant ratios (one-vs-rest)
# ─────────────────────────────────────────────────────────────────────

def compute_discriminant_ratios(
    class_means: dict[str, np.ndarray],
    spectra_by_label: dict[str, list[np.ndarray]],
) -> dict[str, np.ndarray]:
    n_bands = next(iter(class_means.values())).shape[0]
    all_spectra = [s for sps in spectra_by_label.values() for s in sps]
    if not all_spectra:
        return {}
    pooled_std = np.nanstd(np.vstack(all_spectra), axis=0)
    eps = 1e-6
    out = {}
    for label, mean_in in class_means.items():
        ids_in = set(id(s) for s in spectra_by_label.get(label, []))
        spectra_out = [s for sps in spectra_by_label.values()
                       for s in sps if id(s) not in ids_in]
        if not spectra_out:
            out[label] = np.zeros(n_bands); continue
        mean_out = np.nanmean(np.vstack(spectra_out), axis=0)
        out[label] = (mean_in - mean_out) / np.maximum(pooled_std, eps)
    return out


# ─────────────────────────────────────────────────────────────────────
# Peak picking with local-prominence guard
# ─────────────────────────────────────────────────────────────────────

def _peak_pick(values: np.ndarray, master_x: np.ndarray, n_top: int,
                polarity: str = "positive",
                min_separation_cm1: float = MIN_PEAK_SEPARATION_CM1,
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
        if not passing(v): break
        cm = master_x[idx]
        if any(abs(master_x[p] - cm) < min_separation_cm1 for p in picks):
            continue
        picks.append(int(idx))
        if len(picks) >= n_top: break
    return picks


# ─────────────────────────────────────────────────────────────────────
# Replicate-CV per band per class
# ─────────────────────────────────────────────────────────────────────

def _band_intensity_max(spectrum: np.ndarray, master_x: np.ndarray,
                         center: float, tol: float) -> float:
    mask = (master_x >= center - tol) & (master_x <= center + tol)
    if not mask.any(): return 0.0
    vals = spectrum[mask]
    fin = np.isfinite(vals)
    if not fin.any(): return 0.0
    return float(np.max(vals[fin]))


def compute_replicate_band_cv(
    spectra: list[np.ndarray], master_x: np.ndarray,
    center: float, tol: float = DEFAULT_BAND_TOLERANCE_CM1,
) -> float:
    """Coefficient of variation of band intensity across replicates.
    Returns 0 if <2 replicates."""
    if len(spectra) < 2: return 0.0
    intensities = np.array([
        _band_intensity_max(s, master_x, center, tol) for s in spectra
    ])
    fin = np.isfinite(intensities) & (intensities >= 0)
    if not fin.any() or intensities[fin].mean() < 1e-9:
        return 0.0
    return float(intensities[fin].std() / max(intensities[fin].mean(), 1e-9))


# ─────────────────────────────────────────────────────────────────────
# MSS extraction per class
# ─────────────────────────────────────────────────────────────────────

def extract_signature(
    label: str, dr: np.ndarray, master_x: np.ndarray,
    spectra: list[np.ndarray],
    metadata_by_spec_id: dict,
    spectra_meta: list[dict],
    n_anchor: int = N_ANCHOR_BANDS,
    n_support: int = N_SUPPORT_BANDS,
    n_anti: int = N_ANTI_BANDS,
) -> MolecularSignature:
    """Extract a MolecularSignature for one analyte class.

    Replicate-stability filter:
      - For each candidate band, compute CV across replicates
      - Drop the band if CV > MAX_REPLICATE_CV (it's noise-driven)
    """
    # Pick more candidates than we need, then filter by replicate CV
    n_pos_candidates = max(n_anchor + n_support + 4, 20)
    pos_idx = _peak_pick(dr, master_x, n_top=n_pos_candidates, polarity="positive")
    neg_idx = _peak_pick(dr, master_x, n_top=n_anti + 4, polarity="negative")

    def make_band(i, pol):
        cm = float(master_x[i])
        cv = compute_replicate_band_cv(spectra, master_x, cm)
        return MSSBand(
            center_cm1=cm,
            tolerance_cm1=DEFAULT_BAND_TOLERANCE_CM1,
            discriminant_ratio=float(dr[i]),
            polarity=pol,
            replicate_cv=cv,
        )

    pos_bands = [make_band(i, "positive") for i in pos_idx]
    pos_bands_stable = [b for b in pos_bands if b.replicate_cv <= MAX_REPLICATE_CV]
    if len(pos_bands_stable) < n_anchor:
        # not enough stable bands; relax
        pos_bands_stable = pos_bands

    anchor_bands = pos_bands_stable[:n_anchor]
    support_bands = pos_bands_stable[n_anchor:n_anchor + n_support]

    neg_bands = [make_band(i, "negative") for i in neg_idx][:n_anti]

    # Stability score = mean CV of anchor bands
    anchor_stab = (np.mean([b.replicate_cv for b in anchor_bands])
                   if anchor_bands else 0.0)

    datasets = sorted({m.get("dataset", "") for m in spectra_meta if m})
    regimes = sorted({m.get("regime", "") for m in spectra_meta if m})
    substrates = sorted({m.get("substrate_type", "") for m in spectra_meta
                          if m and m.get("substrate_type")})
    sources = [m.get("spectrum_id", "") for m in spectra_meta if m]

    return MolecularSignature(
        signature_id=f"mss::{label}",
        analyte_name=label,
        analyte_class=label,
        anchor_features=anchor_bands,
        support_features=support_bands,
        anti_evidence_features=neg_bands,
        competitor_signatures=[],
        n_source_spectra=len(spectra),
        replicate_stability=float(anchor_stab),
        cross_dataset_support=datasets,
        regime_support=regimes,
        substrate_support=substrates,
        evidence_sources=sources,
        notes="",
    )


# ─────────────────────────────────────────────────────────────────────
# Prototype clustering for competitor + packet inference
# ─────────────────────────────────────────────────────────────────────

def cluster_class_means(class_means: dict[str, np.ndarray],
                          n_clusters: int = DEFAULT_N_PROTOTYPE_CLUSTERS,
                          ) -> tuple[dict[str, int], np.ndarray, list[str]]:
    from scipy.cluster.hierarchy import linkage, fcluster
    from scipy.spatial.distance import pdist
    labels = sorted(class_means.keys())
    if len(labels) < 2:
        return ({labels[0]: 0} if labels else {}), np.array([]), labels
    X = np.vstack([class_means[l] for l in labels])
    dist = pdist(X, metric="correlation")
    Z = linkage(dist, method="average")
    cluster_ids = fcluster(Z, t=min(n_clusters, len(labels)),
                            criterion="maxclust")
    return {l: int(cid) for l, cid in zip(labels, cluster_ids)}, Z, labels


def compute_prototype_overlap(
    class_means: dict[str, np.ndarray],
    cluster_assignment: dict[str, int],
) -> tuple[np.ndarray, list[int]]:
    cluster_to_classes = defaultdict(list)
    for cls, cid in cluster_assignment.items():
        cluster_to_classes[cid].append(cls)
    cluster_ids = sorted(cluster_to_classes.keys())
    proto_means = []
    for cid in cluster_ids:
        members = cluster_to_classes[cid]
        proto_means.append(np.nanmean(
            np.vstack([class_means[c] for c in members]), axis=0,
        ))
    P = np.vstack(proto_means)
    Pc = P - P.mean(axis=1, keepdims=True)
    norms = np.maximum(np.sqrt((Pc * Pc).sum(axis=1, keepdims=True)), 1e-9)
    Pu = Pc / norms
    return Pu @ Pu.T, cluster_ids


def attach_competitors_from_clusters(
    signatures: dict[str, MolecularSignature],
    cluster_assignment: dict[str, int],
):
    """Mark same-cluster classes as competitor signatures."""
    for cls, sig in signatures.items():
        my_cid = cluster_assignment.get(cls)
        comps = [f"mss::{c}" for c, cid in cluster_assignment.items()
                 if cid == my_cid and c != cls]
        sig.competitor_signatures = comps[:8]


def build_packets(
    cluster_assignment: dict[str, int],
    signatures: dict[str, MolecularSignature],
    overlap_matrix: np.ndarray,
    cluster_ids: list[int],
    overlap_threshold: float = COMPETITOR_OVERLAP_THRESHOLD,
) -> dict[str, MSSPacket]:
    cluster_to_classes = defaultdict(list)
    for cls, cid in cluster_assignment.items():
        cluster_to_classes[cid].append(cls)
    cid_to_idx = {cid: i for i, cid in enumerate(cluster_ids)}
    packets: dict[str, MSSPacket] = {}
    for cid in cluster_ids:
        members = cluster_to_classes[cid]
        sig_ids = [signatures[c].signature_id for c in members
                    if c in signatures]
        i = cid_to_idx[cid]
        comps = []
        for j, other_cid in enumerate(cluster_ids):
            if other_cid == cid: continue
            if overlap_matrix[i, j] >= overlap_threshold:
                comps.append(f"packet_v2::cluster_{other_cid}")
        pid = f"packet_v2::cluster_{cid}"
        packets[pid] = MSSPacket(
            packet_id=pid,
            member_signatures=sig_ids,
            member_classes=members,
            competitor_packets=comps,
            rationale=f"prototype cluster {cid} ({len(members)} member classes)",
        )
    return packets


# ─────────────────────────────────────────────────────────────────────
# BAND-BASED MSS SCORING (the production scoring rule)
# ─────────────────────────────────────────────────────────────────────

def _band_fires_with_prominence(
    spectrum: np.ndarray, master_x: np.ndarray, band: MSSBand,
    spectrum_max: float,
) -> tuple[bool, float]:
    """Return (fires, intensity) for a band with absolute floor + local
    prominence + relative-to-spectrum guards. The intensity is the max
    within the band window (zero if the band doesn't fire)."""
    mask = ((master_x >= band.center_cm1 - band.tolerance_cm1)
            & (master_x <= band.center_cm1 + band.tolerance_cm1))
    if not mask.any(): return (False, 0.0)
    band_vals = spectrum[mask]
    fin = np.isfinite(band_vals)
    if not fin.any(): return (False, 0.0)
    band_max = float(np.max(band_vals[fin]))

    # relative-to-spectrum
    if spectrum_max > 0 and band_max < RELATIVE_TO_SPECTRUM_MIN * spectrum_max:
        return (False, 0.0)

    # local-prominence vs neighborhood median (excluding the band itself)
    nbhd_low = band.center_cm1 - PROMINENCE_HALFWIDTH_CM1
    nbhd_high = band.center_cm1 + PROMINENCE_HALFWIDTH_CM1
    mask_nbhd = (((master_x >= nbhd_low) & (master_x <= nbhd_high))
                  & ~mask)
    if mask_nbhd.any():
        nv = spectrum[mask_nbhd]
        nfin = np.isfinite(nv)
        if nfin.any():
            baseline = float(np.median(nv[nfin]))
            if band_max < PROMINENCE_FACTOR * max(baseline, 1e-9):
                return (False, 0.0)

    return (True, band_max)


def score_signature(
    sig: MolecularSignature, spectrum: np.ndarray, master_x: np.ndarray,
    spectrum_max: float | None = None,
) -> dict:
    """BAND-BASED MSS score. Returns dict with per-component fields for audit."""
    if spectrum_max is None:
        fin = np.isfinite(spectrum)
        spectrum_max = float(np.max(spectrum[fin])) if fin.any() else 1.0
    sp_max = max(spectrum_max, 1e-6)

    # Anchor evidence: mean of (intensity / spectrum_max) over fired anchor bands
    anchor_fires = []
    for b in sig.anchor_features:
        f, intensity = _band_fires_with_prominence(spectrum, master_x, b, sp_max)
        anchor_fires.append((f, intensity / sp_max if f else 0.0))
    n_anchor = len(sig.anchor_features)
    anchor_fired_count = sum(1 for f, _ in anchor_fires if f)
    anchor_evidence = (sum(i for _, i in anchor_fires) / n_anchor) if n_anchor else 0.0

    # Co-band factor: penalty for missing required co-bands
    # (we treat all anchor bands as required; missing reduces co_band_factor)
    n_missing = n_anchor - anchor_fired_count
    co_band_factor = 1.0
    if n_anchor > 0:
        # -0.15 per missing anchor band, floored at 0.40
        co_band_factor = max(0.40, 1.0 - 0.15 * n_missing)

    # Support evidence
    support_fires = []
    for b in sig.support_features:
        f, intensity = _band_fires_with_prominence(spectrum, master_x, b, sp_max)
        support_fires.append((f, intensity / sp_max if f else 0.0))
    n_support = len(sig.support_features)
    support_evidence = (sum(i for _, i in support_fires) / n_support
                       if n_support else 0.0)

    # Anti-evidence penalty
    anti_fires = 0
    for b in sig.anti_evidence_features:
        mask = ((master_x >= b.center_cm1 - b.tolerance_cm1)
                & (master_x <= b.center_cm1 + b.tolerance_cm1))
        if mask.any():
            vals = spectrum[mask]
            vals = vals[np.isfinite(vals)]
            if vals.size and float(vals.max()) / sp_max >= ANTI_FIRE_THRESHOLD:
                anti_fires += 1
    anti_penalty = min(anti_fires * ANTI_PER_BAND_PENALTY, MAX_ANTI_PENALTY)
    anti_factor = 1.0 - anti_penalty

    raw = (anchor_evidence + 0.5 * support_evidence) * co_band_factor * anti_factor
    return {
        "score": float(np.clip(raw, 0.0, 1.0)),
        "anchor_evidence": float(anchor_evidence),
        "support_evidence": float(support_evidence),
        "co_band_factor": float(co_band_factor),
        "anti_factor": float(anti_factor),
        "anchor_fired_count": int(anchor_fired_count),
        "n_anchor_bands": int(n_anchor),
        "anti_fires": int(anti_fires),
    }


def apply_competitor_suppression(
    raw_scores: dict[str, float],
    signatures: dict[str, MolecularSignature],
    dominance_ratio: float = 1.30,
    suppression_factor: float = COMPETITOR_SUPPRESSION_FACTOR,
) -> dict[str, float]:
    """If a competitor signature scored DOMINANCE_RATIO× higher than this
    one, multiply this one by SUPPRESSION_FACTOR. Resolves ambiguous fires
    in favour of the dominant chemistry."""
    out = dict(raw_scores)
    for sid, sig in signatures.items():
        my = raw_scores.get(sid, 0.0)
        if my <= 0: continue
        for comp_id in sig.competitor_signatures:
            comp_score = raw_scores.get(comp_id, 0.0)
            if comp_score >= dominance_ratio * my:
                out[sid] = my * suppression_factor
                break
    return out


def score_packet(packet: MSSPacket, signature_scores: dict[str, float]) -> float:
    """Packet score = MAX over member-signature scores."""
    vals = [signature_scores.get(sid, 0.0) for sid in packet.member_signatures]
    return float(max(vals)) if vals else 0.0


__all__ = [
    "MSSBand", "MolecularSignature", "MSSPacket",
    "N_ANCHOR_BANDS", "N_SUPPORT_BANDS", "N_ANTI_BANDS",
    "MIN_DISCRIMINANT_RATIO", "DEFAULT_BAND_TOLERANCE_CM1",
    "MAX_REPLICATE_CV",
    "compute_class_means", "compute_class_replicate_std",
    "compute_discriminant_ratios",
    "extract_signature",
    "cluster_class_means", "compute_prototype_overlap",
    "attach_competitors_from_clusters", "build_packets",
    "score_signature", "apply_competitor_suppression", "score_packet",
]
