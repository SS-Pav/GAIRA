"""gaira_base_2 v2_patches rescue variant (coverage_rescue_v1).

Imports the v2 patches module and overrides the COMPETITOR_SETS
list to remove the 'glycan vs phosphate' pair that caused a 30%
glycan top-1 regression in the v2 retest.

Also extends SPECIFICITY_WEIGHTS with an entry for the newly added
`sterol_skeletal_motif` (conservative 0.85 — new motif, low breadth
assumption).

This is additive only. The original v2_patches module is NOT modified.
The rescue variant is used by the focused grounding retest script.
"""
from __future__ import annotations

import copy

from gaira.base2 import v2_patches as _v2


# Remove the glycan-vs-phosphate competitor set (set #5 in the original list)
COMPETITOR_SETS_RESCUE = [
    cset for cset in _v2.COMPETITOR_SETS
    if "glycan_glycosidic_C_O_C_1020_1100" not in cset
    or "phosphate_PO2_sym_str_1080" not in cset
]

# Sanity: must have dropped exactly one set (the glycan-vs-phosphate set)
assert len(COMPETITOR_SETS_RESCUE) == len(_v2.COMPETITOR_SETS) - 1, (
    "rescue competitor-set patch unexpectedly removed a different number of sets"
)

# Extended specificity weights — add sterol_skeletal_motif + the promoted
# metabolite motifs. Conservative initial values.
SPECIFICITY_WEIGHTS_RESCUE = {
    **_v2.SPECIFICITY_WEIGHTS,
    "sterol_skeletal_motif":       0.85,  # new motif, chemistry-specific
    "glutamate_glutamine_motif":   0.80,  # promoted; not yet in breadth table
    "citrate_as_biology_motif":    0.80,  # promoted; not yet in breadth table
    "sugar_phosphate_skeletal_870_900": 0.75,
}


def specificity_weight_rescue(motif_id: str) -> float:
    return SPECIFICITY_WEIGHTS_RESCUE.get(
        motif_id, _v2.DEFAULT_SPECIFICITY_WEIGHT,
    )


def apply_competitor_dampening_rescue(self_core_weights):
    """PATCH B with the glycan-vs-phosphate set removed."""
    import numpy as np
    out = dict(self_core_weights)
    for cset in COMPETITOR_SETS_RESCUE:
        vals = [(m, out.get(m, 0.0)) for m in cset if m in self_core_weights]
        if len(vals) < 2:
            continue
        max_w = max(w for _, w in vals)
        if max_w <= 1e-9:
            continue
        for m, w in vals:
            if w >= max_w - 1e-9:
                continue
            ratio = w / max_w
            out[m] = w * float(np.sqrt(ratio))
    return out


def patched_score_spectrum_rescue(
    spectrum, master_x, motifs, mappings, dual_status,
    spectrum_id="",
    apply_a=True, apply_b=True, apply_c=True, apply_d=True,
):
    """Rescue-variant patched_score_spectrum.

    - PATCH A: uses SPECIFICITY_WEIGHTS_RESCUE (includes new/promoted motifs)
    - PATCH B: uses COMPETITOR_SETS_RESCUE (glycan-vs-phosphate removed)
    - PATCH C: unchanged (gated ambiguity)
    - PATCH D: unchanged (sparse-axis boost)
    """
    import numpy as np
    from gaira.base2.motif_engine import (
        compute_motif_activation, motif_belongs_to_ambiguity_lane,
        resolve_mapping_weight, resolve_status_calibration_weight,
        resolve_status_core_weight,
    )
    from gaira.base2.projection import project_to_8_axes
    from gaira.base2.schema import (
        AxisScore, BIOLOGY_AXES_V11, MotifScore, SpectrumResult,
    )

    per_motif_self_core = {}
    per_motif_self_regime = {}
    per_motif_contrib_ambig = {}
    per_motif_activation = {}

    for mid, spec in motifs.items():
        activation = compute_motif_activation(spec, spectrum, master_x)
        per_motif_activation[mid] = activation
        status = dual_status.get(mid)
        core_status_w = resolve_status_core_weight(status)
        cal_w = resolve_status_calibration_weight(status)
        spec_w = specificity_weight_rescue(mid) if apply_a else 1.0
        self_core = float(np.clip(activation * core_status_w * spec_w, 0.0, 1.0))
        self_regime = float(np.clip(self_core * cal_w, 0.0, 1.0))
        per_motif_self_core[mid] = self_core
        per_motif_self_regime[mid] = self_regime
        per_motif_contrib_ambig[mid] = motif_belongs_to_ambiguity_lane(
            spec, mappings.get(mid),
        )

    if apply_b:
        per_motif_self_core = apply_competitor_dampening_rescue(per_motif_self_core)
        per_motif_self_regime = apply_competitor_dampening_rescue(per_motif_self_regime)

    motif_scores = {}
    for mid in motifs:
        motif_scores[mid] = MotifScore(
            motif_id=mid,
            activation=float(per_motif_activation[mid]),
            core_weight=float(per_motif_self_core[mid]),
            regime_weight=float(per_motif_self_regime[mid]),
            contributes_to_ambiguity=per_motif_contrib_ambig[mid],
        )

    axis11 = []
    axis11_core_dict = {}
    resolver = _v2._resolve_mapping_weight_patched if apply_d else resolve_mapping_weight

    for axis_id in BIOLOGY_AXES_V11:
        contrib_ids = []
        core_terms = []
        regime_terms = []
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
        core_ev = _v2._noisy_or(core_terms)
        regime_ev = _v2._noisy_or(regime_terms)
        axis11_core_dict[axis_id] = core_ev
        axis11.append(AxisScore(
            axis_id=axis_id,
            core_evidence=core_ev,
            regime_evidence=regime_ev,
            contributing_motifs=tuple(contrib_ids),
        ))

    axis8 = project_to_8_axes(axis11)

    if apply_c:
        ambiguity = _v2.compute_gated_ambiguity_lane(
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
            "variant": "rescue_v1",
            "patches_applied": {
                "A_specificity": apply_a,
                "B_competitor_rescue": apply_b,  # glycan-vs-phos removed
                "C_ambiguity_gated": apply_c,
                "D_sparse_axis_boost": apply_d,
            },
        },
    )
