"""gaira_base_2 final ranking repair (v_rankfix).

ADDITIVE wrapper around v2_patches_discriminative. Implements the four
ranking-layer repairs identified after the targeted anchor acquisition
phase showed top-3 family hit at 72.2% but family top-1 stuck at 36.9%
(the canonical sign of a ranking problem, not an ontology problem).

Design rules (locked):

  * No engine-module modification.
  * No discriminative-module modification — this module is byte-additive
    on top of v2_patches_discriminative.
  * No new motifs added.
  * No registry / mapping changes.
  * Anti-evidence added to v2_patches_discriminative.ANTI_EVIDENCE at
    runtime in the driver, NOT in this module file (so the discriminative
    module file stays byte-identical).
  * Family scoring rule changed in this module (rank-fix family scoring)
    but the per-motif discriminative scoring is unchanged.

Four ranking patches implemented here:

  REPAIR 1 — ANCHOR_VALID_THRESHOLD: an ANCHOR's contribution is "valid"
             only when its discriminative weight >= threshold. Weakly-
             firing ANCHORs (typical when band noise satisfies the
             REQUIRED 3-band check at the engine's BAND_FLOOR=1e-3) are
             treated as weak-anchor and downgraded for ranking.

  REPAIR 2 — Hard anchor-gated family scoring: families WITHOUT a valid
             ANCHOR motif or active CO_FIRE_ANCHOR_GROUP get their score
             capped at NON_ANCHOR_FAMILY_CAP × sum(motif contributions).
             Families WITH a valid anchor sum the anchor contributions
             plus a discounted (×0.5) sum of non-anchor contributions.

  REPAIR 3 — Strengthened anti-evidence: applied via the driver as runtime
             updates to v2_patches_discriminative.ANTI_EVIDENCE (not
             encoded in this module file). Documented in the driver.

  REPAIR 4 — Weak-anchor motif-rank downgrade: ANCHOR motifs below the
             validity threshold have their RANKED weight multiplied by
             WEAK_ANCHOR_DOWNGRADE so they don't win top-1 motif over
             genuinely-firing motifs.

  REPAIR 5 — Ambiguity routing preserved unchanged.

The original `discriminative_weights` dict from
score_spectrum_discriminative is left untouched (for audit). A separate
`rankfix_motif_weights` dict carries the post-rank-fix weights used for
top-1/3/5 ranking. Family scores are recomputed under the new rule.
"""
from __future__ import annotations

from typing import Iterable

import numpy as np

from gaira.base2 import v2_patches_discriminative as _disc


# ─────────────────────────────────────────────────────────────────────
# Constants — calibrated against the actual discriminative_weight scale
# (typical 0.01-0.15 with peaks ~0.20). Diagnostics from the targeted-
# anchor phase showed the new ANCHORs (monosaccharide / free_FA / adenine)
# fire on 96%+ of spectra at low weights because the engine's BAND_FLOOR
# is permissive — these constants apply the ranking-layer correction.
# ─────────────────────────────────────────────────────────────────────

# An ANCHOR's discriminative_weight must be >= this to count as valid.
# 0.015 captures genuine chemistry (peak weights for true UA/glucose/
# adenine fires are 0.02-0.10) while excluding the BAND_FLOOR noise
# (typical noise-driven weights are 0.005-0.012).
ANCHOR_VALID_THRESHOLD: float = 0.015

# Weak ANCHORs (below threshold) have their ranked weight multiplied by this
# in BOTH the rankfix_motif_weights (for motif top-1 ranking) and in the
# rankfix family-scoring rule (counted as a non-anchor contribution).
WEAK_ANCHOR_DOWNGRADE: float = 0.50

# Families WITHOUT anchor-grade evidence have their family score
# multiplied by this cap. Prevents broad/background-only families from
# outranking properly-anchored chemistry. NOT too aggressive (0.65 instead
# of 0.50) — first-iteration 0.50 was too crushing for families with weak
# anchor fires that nonetheless represent real chemistry.
NON_ANCHOR_FAMILY_CAP: float = 0.65

# Within an anchored family, non-anchor motifs (SUPPORT/BACKGROUND) still
# contribute, but at this discount on top of their role-gated weight.
# Keeps the multi-band evidence story while not letting BG accumulation
# dominate the anchor contribution.
ANCHORED_FAMILY_NON_ANCHOR_DISCOUNT: float = 0.75


# ─────────────────────────────────────────────────────────────────────
# Validity check helpers
# ─────────────────────────────────────────────────────────────────────

def is_valid_anchor_fire(motif_id: str, disc_weight: float) -> bool:
    """True iff the motif is ANCHOR-role AND its weight >= validity threshold."""
    return (
        _disc.ROLE_TABLE.get(motif_id) == "ANCHOR"
        and disc_weight >= ANCHOR_VALID_THRESHOLD
    )


def family_has_active_cofire_group(
    family: str, base_weights: dict[str, float],
) -> bool:
    """True iff some CO_FIRE_ANCHOR_GROUP for this family has all members
    firing above their min_weight. Reuses the discriminative module's
    cofire-group definitions."""
    for grp in _disc.CO_FIRE_ANCHOR_GROUPS:
        if family not in grp.get("anchor_for_families", []):
            continue
        if all(base_weights.get(m, 0.0) >= grp["min_weight"]
               for m in grp["members"]):
            return True
    return False


def active_cofire_member_ids(
    family: str, base_weights: dict[str, float],
) -> set[str]:
    """Return set of motif_ids that are members of some CO_FIRE_ANCHOR_GROUP
    that is currently active for this family. These motifs' contributions
    are counted as anchor-grade in family scoring."""
    members: set[str] = set()
    for grp in _disc.CO_FIRE_ANCHOR_GROUPS:
        if family not in grp.get("anchor_for_families", []):
            continue
        if all(base_weights.get(m, 0.0) >= grp["min_weight"]
               for m in grp["members"]):
            members.update(grp["members"])
    return members


# ─────────────────────────────────────────────────────────────────────
# Rank-fix family scoring (REPAIR 2)
# ─────────────────────────────────────────────────────────────────────

def family_score_rankfix(
    discriminative_weights: dict[str, float],
    base_weights: dict[str, float],
    mappings: dict, family: str,
) -> tuple[float, list[str], bool]:
    """Apply REPAIR 1 + REPAIR 2 to compute a rank-fixed family score.

    A motif's contribution is "anchor-grade" if EITHER:
      - role is ANCHOR AND weight >= ANCHOR_VALID_THRESHOLD, OR
      - it is a member of some active CO_FIRE_ANCHOR_GROUP for this family.

    Then:
      score = anchor_grade_sum + ANCHORED_FAMILY_NON_ANCHOR_DISCOUNT × other_sum
            (when at least one anchor-grade contribution exists)
      score = NON_ANCHOR_FAMILY_CAP × other_sum
            (when no anchor-grade contribution exists)

    Returns:
        (score, contributing motif_ids, family_has_anchor_grade_evidence)
    """
    from gaira.base2.motif_engine import resolve_mapping_weight

    cofire_member_ids = active_cofire_member_ids(family, base_weights)
    anchor_grade_contribs: list[tuple[str, float]] = []
    other_contribs: list[tuple[str, float]] = []

    for mid, w in discriminative_weights.items():
        if w <= 0:
            continue
        mp = mappings.get(mid)
        if mp is None:
            continue
        mw = resolve_mapping_weight(mp, family)
        if mw <= 0:
            continue
        contrib = w * mw
        role = _disc.ROLE_TABLE.get(mid, "SUPPORT")
        if role == "ANCHOR" and is_valid_anchor_fire(mid, w):
            anchor_grade_contribs.append((mid, contrib))
        elif mid in cofire_member_ids:
            # cofire member counts as anchor-grade
            anchor_grade_contribs.append((mid, contrib))
        elif role == "ANCHOR":
            # weak ANCHOR (below threshold) — downgraded
            other_contribs.append((mid, contrib * WEAK_ANCHOR_DOWNGRADE))
        else:
            other_contribs.append((mid, contrib))

    has_anchor_grade = bool(anchor_grade_contribs)
    other_sum = sum(c for _, c in other_contribs)

    if has_anchor_grade:
        anchor_sum = sum(c for _, c in anchor_grade_contribs)
        score = anchor_sum + ANCHORED_FAMILY_NON_ANCHOR_DISCOUNT * other_sum
    else:
        score = NON_ANCHOR_FAMILY_CAP * other_sum

    contribs = [m for m, _ in anchor_grade_contribs + other_contribs]
    return float(score), contribs, has_anchor_grade


# ─────────────────────────────────────────────────────────────────────
# Rank-fix motif weights (REPAIR 4)
# ─────────────────────────────────────────────────────────────────────

def rankfix_motif_weight(motif_id: str, disc_weight: float) -> float:
    """Apply weak-anchor downgrade to the ranked motif weight."""
    role = _disc.ROLE_TABLE.get(motif_id, "SUPPORT")
    if role == "ANCHOR" and disc_weight > 0 and disc_weight < ANCHOR_VALID_THRESHOLD:
        return float(disc_weight * WEAK_ANCHOR_DOWNGRADE)
    return float(disc_weight)


# ─────────────────────────────────────────────────────────────────────
# Main scoring entry point
# ─────────────────────────────────────────────────────────────────────

def score_spectrum_rankfix(
    spectrum, master_x, motifs, mappings, dual_status,
    spectrum_id: str = "",
):
    """Score a spectrum through the discriminative engine and apply
    rank-fix gating to family scores + motif ranking."""
    out = _disc.score_spectrum_discriminative(
        spectrum, master_x, motifs, mappings, dual_status, spectrum_id,
    )
    disc_w = out["discriminative_weights"]
    base_w = out["base_weights"]

    # Rank-fix per-motif weights (for motif ranking)
    rankfix_motif_weights = {
        mid: rankfix_motif_weight(mid, w) for mid, w in disc_w.items()
    }

    # Rank-fix family scores
    from gaira.base2.schema import BIOLOGY_AXES_V11
    rankfix_family_scores: dict[str, tuple[float, list[str], bool]] = {}
    for fam in BIOLOGY_AXES_V11:
        score, contribs, has_anchor = family_score_rankfix(
            disc_w, base_w, mappings, fam,
        )
        rankfix_family_scores[fam] = (score, contribs, has_anchor)

    out["rankfix_motif_weights"] = rankfix_motif_weights
    out["rankfix_family_scores"] = rankfix_family_scores
    return out


__all__ = [
    "ANCHOR_VALID_THRESHOLD", "WEAK_ANCHOR_DOWNGRADE",
    "NON_ANCHOR_FAMILY_CAP", "ANCHORED_FAMILY_NON_ANCHOR_DISCOUNT",
    "is_valid_anchor_fire", "family_has_active_cofire_group",
    "family_score_rankfix", "rankfix_motif_weight",
    "score_spectrum_rankfix",
]
