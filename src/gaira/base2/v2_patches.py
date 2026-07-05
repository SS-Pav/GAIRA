"""gaira_base_2 v2 patches — patch-layer on top of the v1 engine.

Adds four deterministic scoring patches derived from v1 grounding
behaviour (`gaira_validate_2_grounding_v1`) to reduce broad-motif
dominance, cross-axis routing errors, and ambiguity overfiring.

The patches are **purely additive** — the v1 engine modules are not
modified. Each patch is toggleable; `score_spectrum(apply_patches_v2=False)`
reproduces the v1 engine exactly.

Patches:

  A. Specificity weighting: multiply each motif's self-weight by a
     specificity factor derived from v1 grounding breadth (fraction of
     reference spectra where the motif activated > 0.05). Broader motifs
     get smaller specificity factors.

  B. Competitor-set dampening: within chemistry-related competitor sets,
     apply sqrt-relative dampening so only the strongest 1-2 motifs retain
     full weight and weaker competitors are dampened proportionally.

  C. Ambiguity gating: the ambiguity lane fires only when ≥ 2 distinct
     biology axes carry non-trivial evidence from the ambiguity motif's
     candidate set. A single chemistry candidate firing does NOT trigger
     ambiguity.

  D. Sparse-axis mapping boost: PRIMARY mapping_weight boosted from 1.0
     to 1.2 for motifs mapped to axes with fewer than 3 PRIMARY
     contributors (metabolic_small_molecule, phosphate_nucleic_adjacent,
     and the less-active sterol/sulfur motifs), so sparse axes are not
     out-scored by broad axes with many PRIMARY contributors.

All constants are documented inline with provenance.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from gaira.base2.ambiguity import _noisy_or as _noisy_or_ambig  # reuse
from gaira.base2.motif_engine import (
    compute_motif_activation,
    motif_belongs_to_ambiguity_lane,
    resolve_mapping_weight,
    resolve_status_calibration_weight,
    resolve_status_core_weight,
)
from gaira.base2.projection import project_to_8_axes
from gaira.base2.schema import (
    AmbiguityLane,
    AxisMapping,
    AxisScore,
    BIOLOGY_AXES_V11,
    MotifDualStatus,
    MotifScore,
    MotifSpec,
    SpectrumResult,
)


# ──────────────────────────────────────────────────────────────────────
# PATCH A — Specificity weights (from v1 grounding breadth)
# ──────────────────────────────────────────────────────────────────────
#
# Breadth = fraction of 377 v1 grounding references with motif_core > 0.05.
# specificity_weight = 1 / (1 + 2 × breadth)
# Derived from `grounding_per_spectrum_scores_v1.csv` — deterministic.
#
# The k=2 scaling gives:
#   breadth 0.60 → specificity 0.45
#   breadth 0.30 → specificity 0.63
#   breadth 0.10 → specificity 0.83
#   breadth 0.00 → specificity 1.00

SPECIFICITY_WEIGHTS: dict[str, float] = {
    "lipid_acyl_C_C_str_1060_1130":            0.45,  # breadth 0.613
    "lipid_C_H_bend_1440_1460":                0.50,  # breadth 0.493
    "lipid_methylene_twist_1300":              0.50,  # breadth 0.491
    "collision_1300_1400_multi_candidate_motif": 0.52,  # breadth 0.472
    "amide_III_protein_backbone_1230_1280":    0.52,  # breadth 0.469
    "cholesterol_signature":                   0.54,  # breadth 0.427
    "amide_I_alpha_helix_beta_sheet_motif":    0.54,  # breadth 0.422
    "free_saccharide_motif":                   0.55,  # breadth 0.414
    "neutral_lipid_triglyceride_motif":        0.58,  # breadth 0.369
    "glycan_glycosidic_C_O_C_1020_1100":       0.60,  # breadth 0.340
    "phosphate_PO_asym_str_1240":              0.63,  # breadth 0.294
    "sialic_acid_signature":                   0.63,  # breadth 0.289
    "glycan_pyranose_ring_skeletal_850_950":   0.65,  # breadth 0.265
    "disulfide_S_S_str_500_550":               0.66,  # breadth 0.260
    "phenylalanine_ring_1003":                 0.67,  # breadth 0.241
    "thiol_C_S_str_660_motif":                 0.68,  # breadth 0.236
    "tyrosine_doublet_830_850":                0.69,  # breadth 0.220
    "dna_composite_motif":                     0.70,  # breadth 0.212
    "ergothioneine_signature":                 0.71,  # breadth 0.202
    "phosphate_PO2_sym_str_1080":              0.73,  # breadth 0.188
    "uric_acid_full_signature":                0.73,  # breadth 0.186
    "creatine_creatinine_motif":               0.76,  # breadth 0.156
    "pyrimidine_ring_breathing_780_800":       0.77,  # breadth 0.151
    "glutathione_GSH_motif":                   0.77,  # breadth 0.151
    "purine_ring_breathing_720_735":           0.78,  # breadth 0.141
    "hypoxanthine_signature":                  0.85,  # breadth ~0.09
    "xanthine_signature":                      0.90,  # breadth ~0.05
    "guanine_specific_motif":                  0.85,  # breadth ~0.09
    "thymine_specific_motif":                  0.88,  # breadth ~0.07
    "cytosine_specific_motif":                 0.88,  # breadth ~0.07
    "dna_methylation_marker_790":              0.85,
    "purine_HX_lipid_choline_715_overlap_ambiguity": 0.80,
    "collision_1020_1080_multi_candidate":     0.70,  # HELD_V2 but still scored
    "nucleobase_in_plane_ring_1320_1340":      0.70,  # HELD_V2 but still scored
    "amide_I_lipid_carbonyl_partial_panel_motif": 0.70,  # HELD_V2 but still scored
    "amide_II_motif":                          0.80,
    "phosphatidylcholine_choline_head_715":    0.80,
    "citrate_baseline_artifact_motif":         0.80,
    "cytochrome_c_resonance_motif":            0.90,
}

# Default for any motif not in the table above
DEFAULT_SPECIFICITY_WEIGHT = 0.80


def specificity_weight(motif_id: str) -> float:
    return SPECIFICITY_WEIGHTS.get(motif_id, DEFAULT_SPECIFICITY_WEIGHT)


# ──────────────────────────────────────────────────────────────────────
# PATCH B — Competitor-set dampening
# ──────────────────────────────────────────────────────────────────────
#
# Within each competitor set (chemistry-related motifs that tend to
# co-fire on grounding references), apply sqrt-relative dampening:
#
#   for each motif m in set:
#     if weight[m] < max(weights in set):
#         patched_weight[m] = weight[m] × sqrt(weight[m] / max_weight)
#     else:
#         patched_weight[m] = weight[m]  (winner retained)
#
# This creates soft winner-take-most without hard zeroing. Effect on
# v1 grounding:
#   - purine: UA/HX/Xanth compete with purine_ring_breathing +
#     guanine-specific; stronger metabolite motif wins on UA spectra,
#     stronger nucleobase motif wins on adenine spectra.
#   - sterol vs acyl lipid: cholesterol_signature's 4-band structure
#     competes with the generic lipid CH bend / methylene twist when
#     both fire; on pure cholesterol powder, the sterol-specific motif
#     wins.

COMPETITOR_SETS: list[list[str]] = [
    # Purine family — nucleobase vs metabolite competition
    [
        "purine_ring_breathing_720_735",
        "guanine_specific_motif",
        "uric_acid_full_signature",
        "hypoxanthine_signature",
        "xanthine_signature",
    ],
    # Sterol vs acyl lipid
    [
        "cholesterol_signature",
        "neutral_lipid_triglyceride_motif",
        "lipid_acyl_C_C_str_1060_1130",
        "lipid_C_H_bend_1440_1460",
        "lipid_methylene_twist_1300",
    ],
    # Nucleobase vs phosphate backbone (for DNA/RNA references)
    [
        "guanine_specific_motif",
        "purine_ring_breathing_720_735",
        "phosphate_PO2_sym_str_1080",
        "phosphate_PO_asym_str_1240",
        "dna_composite_motif",
    ],
    # Amide I/II/III vs lipid C=O (protein vs triglyceride carbonyl)
    [
        "amide_I_alpha_helix_beta_sheet_motif",
        "amide_II_motif",
        "amide_III_protein_backbone_1230_1280",
        "neutral_lipid_triglyceride_motif",
    ],
    # Glycan vs phosphate (1020-1080 cm⁻¹ overlap)
    [
        "glycan_glycosidic_C_O_C_1020_1100",
        "glycan_pyranose_ring_skeletal_850_950",
        "free_saccharide_motif",
        "phosphate_PO2_sym_str_1080",
    ],
    # Aromatic residue vs purine nucleobase (both have ring modes)
    [
        "phenylalanine_ring_1003",
        "tyrosine_doublet_830_850",
        "purine_ring_breathing_720_735",
        "guanine_specific_motif",
    ],
]


def apply_competitor_dampening(
    self_core_weights: dict[str, float],
) -> dict[str, float]:
    """Apply sqrt-relative dampening within each competitor set."""
    out = dict(self_core_weights)
    for cset in COMPETITOR_SETS:
        vals = [(m, out.get(m, 0.0)) for m in cset if m in self_core_weights]
        if len(vals) < 2:
            continue
        max_w = max(w for _, w in vals)
        if max_w <= 1e-9:
            continue
        for m, w in vals:
            if w >= max_w - 1e-9:
                # winner — full weight
                continue
            # dampen by sqrt(w / max_w)
            ratio = w / max_w
            out[m] = w * float(np.sqrt(ratio))
    return out


# ──────────────────────────────────────────────────────────────────────
# PATCH C — Ambiguity gating (multi-axis agreement required)
# ──────────────────────────────────────────────────────────────────────
#
# Rule: an ambiguity motif contributes to the ambiguity lane only if
# AT LEAST TWO distinct biology axes are also carrying non-trivial
# evidence (axis_core > 0.10). A single clean chemistry class firing
# an ambiguity motif should NOT trigger ambiguity.
#
# Implementation: after computing 11-axis scores, count how many biology
# axes have evidence > AMBIGUITY_AXIS_COFIRE_THRESHOLD. If < 2, dampen
# the ambiguity lane to zero.

AMBIGUITY_AXIS_COFIRE_THRESHOLD: float = 0.10
AMBIGUITY_MIN_CANDIDATE_AXES: int = 2


def compute_gated_ambiguity_lane(
    motif_scores: dict[str, MotifScore],
    mappings: dict[str, AxisMapping],
    axis11_core: dict[str, float],
) -> AmbiguityLane:
    """PATCH C: compute ambiguity lane with multi-axis-agreement gating.

    Similar to the v1 compute_ambiguity_lane but additionally requires
    at least AMBIGUITY_MIN_CANDIDATE_AXES biology axes to have evidence
    above AMBIGUITY_AXIS_COFIRE_THRESHOLD before the lane is permitted
    to activate.
    """
    # Count biology axes above the co-fire threshold
    cofiring_axes = sum(
        1 for v in axis11_core.values() if v > AMBIGUITY_AXIS_COFIRE_THRESHOLD
    )
    gate_open = cofiring_axes >= AMBIGUITY_MIN_CANDIDATE_AXES

    contrib_ids: list[str] = []
    core_terms: list[float] = []
    regime_terms: list[float] = []
    for mid, mscore in motif_scores.items():
        if not mscore.contributes_to_ambiguity:
            continue
        mapping = mappings.get(mid)
        if mapping is None:
            continue
        if mapping.mapping_type == "AMBIGUITY_ONLY":
            mw = 1.0
        elif "ambiguity_artifact" in (mapping.primary_axis, *mapping.secondary_axes):
            mw = 0.70
        else:
            mw = 0.70
        contrib_ids.append(mid)
        core_terms.append(mscore.core_weight * mw)
        regime_terms.append(mscore.regime_weight * mw)

    if not gate_open:
        # gate closed → ambiguity lane silenced
        return AmbiguityLane(
            core_evidence=0.0,
            regime_evidence=0.0,
            contributing_motifs=tuple(contrib_ids),
        )

    return AmbiguityLane(
        core_evidence=_noisy_or_ambig(core_terms),
        regime_evidence=_noisy_or_ambig(regime_terms),
        contributing_motifs=tuple(contrib_ids),
    )


# ──────────────────────────────────────────────────────────────────────
# PATCH D — Sparse-axis mapping boost
# ──────────────────────────────────────────────────────────────────────
#
# Axes with fewer than 3 active PRIMARY motifs are "sparse" and get
# their PRIMARY mapping_weight multiplied by 1.2 so they compete
# meaningfully against axes with many PRIMARY contributors.
#
# Current sparse axes (from v1.1 mapping skeleton):
#   - metabolic_small_molecule: 1 PRIMARY (creatine) + 1 CROSS
#   - phosphate_nucleic_adjacent: 3 PRIMARY (marginal — include)
#
# Also promote specific motifs that are out-competed despite being
# chemistry-specific (sterol axis).

SPARSE_AXIS_BOOST: dict[str, float] = {
    # Axis → PRIMARY mapping_weight multiplier
    "metabolic_small_molecule":     1.3,
    "phosphate_nucleic_adjacent":   1.2,
    "purine_metabolite":            1.15,  # competes with purine_nucleotide via shared motifs
    "sterol_neutral_lipid":         1.2,   # out-competed by generic lipid
}


def apply_sparse_axis_boost(mapping_weight: float, target_axis: str,
                             mapping_type: str) -> float:
    """Apply PATCH D boost to PRIMARY mapping weights on sparse axes."""
    if mapping_type != "PRIMARY":
        return mapping_weight
    boost = SPARSE_AXIS_BOOST.get(target_axis, 1.0)
    return mapping_weight * boost


# ──────────────────────────────────────────────────────────────────────
# Patched motif-score + axis-aggregation + scoring pipeline
# ──────────────────────────────────────────────────────────────────────

def _resolve_mapping_weight_patched(
    mapping: AxisMapping, target_axis: str,
) -> float:
    """PATCH D-aware wrapper over resolve_mapping_weight."""
    base = resolve_mapping_weight(mapping, target_axis)
    return apply_sparse_axis_boost(base, target_axis, mapping.mapping_type)


def _noisy_or(weights: Iterable[float]) -> float:
    p = 1.0
    any_contrib = False
    for w in weights:
        wc = float(np.clip(w, 0.0, 1.0))
        if wc > 0:
            any_contrib = True
        p *= (1.0 - wc)
    return 0.0 if not any_contrib else float(1.0 - p)


def patched_score_spectrum(
    spectrum: np.ndarray,
    master_x: np.ndarray,
    motifs: dict[str, MotifSpec],
    mappings: dict[str, AxisMapping],
    dual_status: dict[str, MotifDualStatus],
    spectrum_id: str = "",
    apply_a: bool = True,  # specificity
    apply_b: bool = True,  # competitor dampening
    apply_c: bool = True,  # ambiguity gating
    apply_d: bool = True,  # sparse axis boost
) -> SpectrumResult:
    """Patched score_spectrum applying the four v2 patches.

    Each patch is independently toggleable for ablation analysis.
    With all four OFF, this reproduces the v1 engine exactly.
    """
    # ── Compute motif activations + per-motif "self" weights ─────────
    per_motif_self_core: dict[str, float] = {}
    per_motif_self_regime: dict[str, float] = {}
    per_motif_contrib_ambig: dict[str, bool] = {}
    per_motif_activation: dict[str, float] = {}

    for mid, spec in motifs.items():
        activation = compute_motif_activation(spec, spectrum, master_x)
        per_motif_activation[mid] = activation
        status = dual_status.get(mid)
        core_status_w = resolve_status_core_weight(status)
        cal_w = resolve_status_calibration_weight(status)

        # PATCH A — specificity weighting
        spec_w = specificity_weight(mid) if apply_a else 1.0

        self_core = float(np.clip(
            activation * core_status_w * spec_w, 0.0, 1.0,
        ))
        self_regime = float(np.clip(self_core * cal_w, 0.0, 1.0))
        per_motif_self_core[mid] = self_core
        per_motif_self_regime[mid] = self_regime
        per_motif_contrib_ambig[mid] = motif_belongs_to_ambiguity_lane(
            spec, mappings.get(mid),
        )

    # ── PATCH B — competitor-set dampening ────────────────────────────
    if apply_b:
        per_motif_self_core = apply_competitor_dampening(per_motif_self_core)
        per_motif_self_regime = apply_competitor_dampening(per_motif_self_regime)

    # ── Motif scores (for output) ────────────────────────────────────
    motif_scores: dict[str, MotifScore] = {}
    for mid in motifs:
        motif_scores[mid] = MotifScore(
            motif_id=mid,
            activation=float(per_motif_activation[mid]),
            core_weight=float(per_motif_self_core[mid]),
            regime_weight=float(per_motif_self_regime[mid]),
            contributes_to_ambiguity=per_motif_contrib_ambig[mid],
        )

    # ── 11-axis aggregation (noisy-OR with patched mapping weights) ──
    axis11: list[AxisScore] = []
    axis11_core_dict: dict[str, float] = {}
    resolver = _resolve_mapping_weight_patched if apply_d else resolve_mapping_weight

    for axis_id in BIOLOGY_AXES_V11:
        contrib_ids: list[str] = []
        core_terms: list[float] = []
        regime_terms: list[float] = []
        for mid in motifs:
            mapping = mappings.get(mid)
            if mapping is None:
                continue
            mw = resolver(mapping, axis_id)
            if mw <= 0:
                continue
            contrib_ids.append(mid)
            core_terms.append(per_motif_self_core[mid] * mw)
            regime_terms.append(per_motif_self_regime[mid] * mw)
        core_ev = _noisy_or(core_terms)
        regime_ev = _noisy_or(regime_terms)
        axis11_core_dict[axis_id] = core_ev
        axis11.append(AxisScore(
            axis_id=axis_id,
            core_evidence=core_ev,
            regime_evidence=regime_ev,
            contributing_motifs=tuple(contrib_ids),
        ))

    # ── 8-axis projection (MAX combiner, unchanged) ──────────────────
    axis8 = project_to_8_axes(axis11)

    # ── Ambiguity lane (PATCH C — multi-axis gate) ────────────────────
    if apply_c:
        ambiguity = compute_gated_ambiguity_lane(
            motif_scores, mappings, axis11_core_dict,
        )
    else:
        from gaira.base2.ambiguity import compute_ambiguity_lane
        ambiguity = compute_ambiguity_lane(motif_scores, mappings)

    return SpectrumResult(
        spectrum_id=spectrum_id,
        motif_scores=tuple(motif_scores.values()),
        axis11_scores=tuple(axis11),
        axis8_projection=tuple(axis8),
        ambiguity=ambiguity,
        metadata={
            "n_motifs_evaluated": len(motif_scores),
            "patches_applied": {
                "A_specificity":       apply_a,
                "B_competitor":        apply_b,
                "C_ambiguity_gated":   apply_c,
                "D_sparse_axis_boost": apply_d,
            },
        },
    )


# ──────────────────────────────────────────────────────────────────────
# Summary of patches (for documentation / UI disclosure)
# ──────────────────────────────────────────────────────────────────────

PATCH_DOC = {
    "A_specificity": {
        "name": "Specificity weighting",
        "source": "v1 grounding breadth statistics",
        "effect": "broader motifs get smaller self-weights",
        "breadth_formula": "specificity = 1 / (1 + 2 × breadth)",
        "breadth_threshold": 0.05,
    },
    "B_competitor": {
        "name": "Competitor-set sqrt-relative dampening",
        "source": "chemistry-related motif pairs; 6 competitor sets",
        "effect": "within each set, weaker motifs are dampened by sqrt(w / max_w)",
    },
    "C_ambiguity_gated": {
        "name": "Ambiguity multi-axis gating",
        "source": "v1 observation: ambiguity lane fires on 96% of refs",
        "effect": "lane silenced unless ≥ 2 biology axes > 0.10",
        "threshold": AMBIGUITY_AXIS_COFIRE_THRESHOLD,
        "min_candidate_axes": AMBIGUITY_MIN_CANDIDATE_AXES,
    },
    "D_sparse_axis_boost": {
        "name": "Sparse-axis mapping boost",
        "source": "v1 observation: sparse axes out-competed by dense axes",
        "effect": "PRIMARY mapping_weight × multiplier for sparse axes",
        "multipliers": SPARSE_AXIS_BOOST,
    },
}
