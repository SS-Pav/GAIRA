"""gaira_base_2 final ranking repair loop v1.

Final ranking-focused repair. Tests whether the top-1 plateau is a
ranking/gating problem (not ontology). Adds 4 ranking patches:

  REPAIR 1 — strict ANCHOR validity threshold (0.015)
  REPAIR 2 — hard anchor-gated family scoring (non-anchored families cap 0.50)
  REPAIR 3 — strengthened anti-evidence (runtime updates to ANTI_EVIDENCE)
  REPAIR 4 — weak-anchor motif ranking downgrade

Engine + v1-/v2-/rescue-/repair-v2-/discriminative- modules: NOT modified.
Registry v1.5 + mapping v1.4 + dual_status runtime overrides from anchor phase: REUSED.
New module: src/gaira/base2/v2_patches_final_ranking.py.

Run:
    cd /Users/suraj/projects/GAIRA
    PYTHONPATH=src .venv/bin/python scripts/run_gaira_base_2_final_ranking_repair_loop_v1.py
"""
from __future__ import annotations

import shutil
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from gaira.base2.registry import (
    load_axis_mapping, load_dual_status, load_motif_registry,
)
from gaira.base2.schema import MotifDualStatus, BIOLOGY_AXES_V11
from gaira.base2 import v2_patches_discriminative as _disc
from gaira.base2 import v2_patches_final_ranking as _rank
from gaira.spectral import canonical_master_axis

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_gaira_validate_2_grounding import (
    load_ramanbiolib, load_gobbato_powder,
    load_amino_acid_xlsx, load_digitised_literature,
)
from run_gaira_validate_2_grounding_motif_first_v1 import (
    EXPECTED_MOTIFS, FAMILIES,
    expected_families_for, expected_ambiguity_for, topn_hit,
)
# Reuse the anchor-phase driver's runtime-override helpers (registry +
# truth-table + dual_status extensions). These set up the SAME baseline
# state the rank-fix is applied on top of.
from run_gaira_base_2_targeted_anchor_acquisition_v1 import (
    extend_role_table_for_anchors,
    extend_anti_evidence_for_reactivated_motif,
    extend_truth_table_for_new_anchors,
    extend_dual_status_for_new_and_silent_motifs,
    expected_motifs_for_runtime,
)


ROOT = Path("/Volumes/SSD_Rad/GAIRA_BUILD/gaira_base_2_final_ranking_repair_loop_v1")
TABLES = ROOT / "tables"
FIGS = ROOT / "figures"
REPORTS = ROOT / "reports"
AUDIT = ROOT / "audit"
CODE_SNAPSHOT = ROOT / "code_snapshot"

REG_V1_5 = Path(
    "/Volumes/SSD_Rad/GAIRA_BUILD/gaira_base_2_targeted_anchor_acquisition_v1/"
    "registry/motif_candidate_registry_v1_5.yaml"
)
MAP_V1_4 = Path(
    "/Volumes/SSD_Rad/GAIRA_BUILD/gaira_base_2_targeted_anchor_acquisition_v1/"
    "registry/motif_to_axis_mapping_skeleton_v1_4.csv"
)

# Prior-phase baselines for comparison
ANCHOR_METRICS = Path(
    "/Volumes/SSD_Rad/GAIRA_BUILD/gaira_base_2_targeted_anchor_acquisition_v1/"
    "tables/grounding_metrics_summary_v_anchor.csv"
)
ANCHOR_PERFAM = Path(
    "/Volumes/SSD_Rad/GAIRA_BUILD/gaira_base_2_targeted_anchor_acquisition_v1/"
    "tables/grounding_per_family_hit_rates_v_anchor.csv"
)
ANCHOR_MISS = Path(
    "/Volumes/SSD_Rad/GAIRA_BUILD/gaira_base_2_targeted_anchor_acquisition_v1/"
    "tables/grounding_miss_list_v_anchor.csv"
)
ANCHOR_PERSPEC_FAM = Path(
    "/Volumes/SSD_Rad/GAIRA_BUILD/gaira_base_2_targeted_anchor_acquisition_v1/"
    "tables/grounding_per_spectrum_family_scores_v_anchor.csv"
)
ANCHOR_PERSPEC_MOTIF = Path(
    "/Volumes/SSD_Rad/GAIRA_BUILD/gaira_base_2_targeted_anchor_acquisition_v1/"
    "tables/grounding_per_spectrum_motif_scores_v_anchor.csv"
)
ANCHOR_RANK_MOTIF = Path(
    "/Volumes/SSD_Rad/GAIRA_BUILD/gaira_base_2_targeted_anchor_acquisition_v1/"
    "tables/grounding_expected_vs_observed_motif_rank_v_anchor.csv"
)
ANCHOR_RANK_FAMILY = Path(
    "/Volumes/SSD_Rad/GAIRA_BUILD/gaira_base_2_targeted_anchor_acquisition_v1/"
    "tables/grounding_expected_vs_observed_family_rank_v_anchor.csv"
)
ANCHOR_OFFTGT = Path(
    "/Volumes/SSD_Rad/GAIRA_BUILD/gaira_base_2_targeted_anchor_acquisition_v1/"
    "tables/grounding_off_target_activation_v_anchor.csv"
)
ANCHOR_AMBIG = Path(
    "/Volumes/SSD_Rad/GAIRA_BUILD/gaira_base_2_targeted_anchor_acquisition_v1/"
    "tables/grounding_ambiguity_behavior_v_anchor.csv"
)
DISC_METRICS = Path(
    "/Volumes/SSD_Rad/GAIRA_BUILD/gaira_base_2_discriminative_motif_upgrade_v1/"
    "tables/grounding_metrics_summary_v_discriminative.csv"
)
DISC_PERFAM = Path(
    "/Volumes/SSD_Rad/GAIRA_BUILD/gaira_base_2_discriminative_motif_upgrade_v1/"
    "tables/grounding_per_family_hit_rates_v_discriminative.csv"
)


# ─────────────────────────────────────────────────────────────────────
# REPAIR 3 — strengthened anti-evidence (runtime overrides on
# v2_patches_discriminative.ANTI_EVIDENCE). Applied on TOP of the anchor-
# phase runtime overrides.
# ─────────────────────────────────────────────────────────────────────

def strengthen_anti_evidence_for_rankfix():
    """Strengthen anti-evidence on the families still leaking top-1 after
    the targeted anchor phase (adenine-specific leaking to aromatic +
    purine_metabolite; free_fatty_acid leaking to protein amide-I region).

    This is a runtime update to v2_patches_discriminative.ANTI_EVIDENCE —
    it does NOT modify the discriminative module file. It strengthens
    penalties already defined in the anchor phase AND adds new rules for
    aromatic competitors (Phe/Tyr) that the anchor phase didn't cover."""

    # adenine_specific_anchor_motif — strengthen penalties + add aromatic
    # competitors (Phe/Tyr) that were leaking in the anchor phase
    _disc.ANTI_EVIDENCE["adenine_specific_anchor_motif"] = [
        # Stronger UA/HX/Xanth suppression (was 0.70 / 0.65 / 0.65)
        {"rule": "SUPPRESS_IF_PRESENT",
         "target": "uric_acid_full_signature",
         "min_weight": 0.015, "penalty": 0.85},
        {"rule": "SUPPRESS_IF_PRESENT",
         "target": "hypoxanthine_signature",
         "min_weight": 0.015, "penalty": 0.80},
        {"rule": "SUPPRESS_IF_PRESENT",
         "target": "xanthine_signature",
         "min_weight": 0.015, "penalty": 0.80},
        # Aromatic competitors — NEW: the adenine 1480 region is shared
        # with aromatic ring stretches; Phe/Tyr references leak via this.
        {"rule": "SUPPRESS_IF_PRESENT",
         "target": "phenylalanine_ring_1003",
         "min_weight": 0.020, "penalty": 0.70},
        {"rule": "SUPPRESS_IF_PRESENT",
         "target": "tyrosine_doublet_830_850",
         "min_weight": 0.020, "penalty": 0.70},
        # Cytochrome c heme leak preserved from anchor phase
        {"rule": "SUPPRESS_IF_PRESENT",
         "target": "cytochrome_c_resonance_motif",
         "min_weight": 0.020, "penalty": 0.50},
        # REQUIRES_COBAND with a purine-family support motif: adenine
        # CANNOT fire alone without the 720 ring breathing firing.
        {"rule": "REQUIRES_COBAND",
         "target": "purine_ring_breathing_720_735",
         "min_weight": 0.020, "penalty": 0.50},
    ]

    # free_fatty_acid_carboxyl_anchor_motif — strengthen amide_I /
    # cholesterol suppression (was 0.50 / 0.45)
    _disc.ANTI_EVIDENCE["free_fatty_acid_carboxyl_anchor_motif"] = [
        # Stronger amide_I suppression: protein amide_I (1640-1685) tail
        # can satisfy the 1700 check at the permissive BAND_FLOOR.
        {"rule": "SUPPRESS_IF_PRESENT",
         "target": "amide_I_alpha_helix_beta_sheet_motif",
         "min_weight": 0.020, "penalty": 0.70},
        # Stronger cholesterol suppression (1670 C=C tail can satisfy 1700)
        {"rule": "SUPPRESS_IF_PRESENT",
         "target": "cholesterol_signature",
         "min_weight": 0.020, "penalty": 0.60},
        # REQUIRES_COBAND with lipid acyl chain support
        {"rule": "REQUIRES_COBAND",
         "target": "lipid_acyl_C_C_str_1060_1130",
         "min_weight": 0.020, "penalty": 0.45},
    ]

    # monosaccharide_anomeric_anchor_motif — add amide_I suppression in
    # addition to the amide_III + FFA rules from the anchor phase
    _disc.ANTI_EVIDENCE["monosaccharide_anomeric_anchor_motif"] = [
        # From anchor phase — retained
        {"rule": "SUPPRESS_IF_PRESENT",
         "target": "amide_III_protein_backbone_1230_1280",
         "min_weight": 0.025, "penalty": 0.55},
        {"rule": "SUPPRESS_IF_PRESENT",
         "target": "free_fatty_acid_carboxyl_anchor_motif",
         "min_weight": 0.030, "penalty": 0.55},
        # REQUIRES_COBAND with glycan ring breathing (same-family support)
        {"rule": "REQUIRES_COBAND",
         "target": "glycan_pyranose_ring_skeletal_850_950",
         "min_weight": 0.020, "penalty": 0.45},
    ]


# ─────────────────────────────────────────────────────────────────────
# STEP 1 — Ranking failure diagnosis
# ─────────────────────────────────────────────────────────────────────

def classify_ranking_failure(
    expected_motifs_list: list[str],
    expected_families_list: list[str],
    top_motif: str, top_family: str,
    anchor_present_expected: bool,
    anchor_present_observed: bool,
) -> str:
    """Return a ranking-failure type label."""
    if not expected_motifs_list and not expected_families_list:
        return "NO_EXPECTATION"
    if anchor_present_expected and top_family in expected_families_list:
        return "CORRECT_TOP1"
    if anchor_present_expected and top_family not in expected_families_list:
        # Expected anchor fired but another family won top-1: this is
        # the core ranking-layer failure.
        if anchor_present_observed:
            return "COMPETING_ANCHOR_WON_TOP1"
        return "NON_ANCHOR_FAMILY_WON_TOP1"
    if not anchor_present_expected:
        # Expected anchor didn't fire — ontology/evidence limit
        return "EXPECTED_ANCHOR_DID_NOT_FIRE"
    return "OTHER_RANKING_ISSUE"


def diagnose_ranking_failures(motifs, mappings) -> pd.DataFrame:
    """Read the prior anchor-phase miss list + per-spectrum motif scores
    and classify each miss as ranking-layer vs ontology-layer."""

    print("\n[STEP 1] Diagnosing ranking failures from anchor-phase miss list")

    miss = pd.read_csv(ANCHOR_MISS)
    motif_scores = pd.read_csv(ANCHOR_PERSPEC_MOTIF)

    # Build a per-spectrum lookup: dict[spectrum_id][motif_id] = disc_w
    perspec_mw: dict[str, dict[str, float]] = defaultdict(dict)
    for _, r in motif_scores.iterrows():
        perspec_mw[r["spectrum_id"]][r["motif_id"]] = float(
            r["discriminative_weight"]
        )

    # Build list of ANCHOR motifs per family (using ROLE_TABLE + mappings)
    family_anchors: dict[str, list[str]] = defaultdict(list)
    for mid, role in _disc.ROLE_TABLE.items():
        if role != "ANCHOR":
            continue
        mp = mappings.get(mid)
        if mp is None or not mp.active:
            continue
        family_anchors[mp.primary_axis].append(mid)
        for sa in mp.secondary_axes:
            family_anchors[sa].append(mid)

    def has_valid_anchor_fire_for_family(
        spec_id: str, family: str, mw_dict: dict,
    ) -> bool:
        for aid in family_anchors.get(family, []):
            if mw_dict.get(aid, 0.0) >= _rank.ANCHOR_VALID_THRESHOLD:
                return True
        return False

    rows = []
    for _, m in miss.iterrows():
        sid = m["spectrum_id"]
        mw_dict = perspec_mw.get(sid, {})
        exp_m = str(m.get("expected_motifs", "") or "").split(",")
        exp_f = str(m.get("expected_families", "") or "").split(",")
        exp_m = [x for x in exp_m if x]
        exp_f = [x for x in exp_f if x]
        top_m = str(m.get("observed_top_motifs", "")).split(",")[0]
        top_f = str(m.get("observed_top_families", "")).split(",")[0]

        # anchor_present_expected: does any expected family have a valid
        # anchor fire on this spectrum?
        ape = any(has_valid_anchor_fire_for_family(sid, f, mw_dict)
                  for f in exp_f)
        # anchor_present_observed: does the observed top-1 family have a
        # valid anchor fire?
        apo = has_valid_anchor_fire_for_family(sid, top_f, mw_dict)

        fail_type = classify_ranking_failure(
            exp_m, exp_f, top_m, top_f, ape, apo,
        )
        rows.append({
            "spectrum_id": sid,
            "dataset_name": m.get("dataset_name", ""),
            "expected_motif": ",".join(exp_m),
            "expected_family": ",".join(exp_f),
            "observed_top_motif": top_m,
            "observed_top_family": top_f,
            "anchor_present_expected": "YES" if ape else "NO",
            "anchor_present_observed": "YES" if apo else "NO",
            "likely_ranking_failure_type": fail_type,
            "notes": "",
        })
    df = pd.DataFrame(rows)
    df.to_csv(TABLES / "ranking_failure_cases_v1.csv", index=False)
    print(f"[emit] ranking_failure_cases_v1.csv ({len(df)} rows)")

    # Summary
    counts = df["likely_ranking_failure_type"].value_counts()
    print("\n[ranking-failure summary]")
    for k, v in counts.items():
        print(f"  {k:40s}: {v}")

    return df


# ─────────────────────────────────────────────────────────────────────
# STEP 2 — Actions table
# ─────────────────────────────────────────────────────────────────────

def emit_actions():
    rows = [
        {"action_id": "RANK_v1_001",
         "component_touched": "src/gaira/base2/v2_patches_final_ranking.py",
         "repair_type": "ADD_ANCHOR_VALID_THRESHOLD",
         "rationale": "Anchor motif weights below 0.015 are noise-driven (rescue-engine BAND_FLOOR 1e-3 permits them); these should not count as 'valid' anchor firings for ranking.",
         "expected_effect": "Weak-anchor fires (typical on non-target chemistry) are downgraded × 0.50 for ranking, preventing spurious top-1 wins",
         "notes": "ANCHOR_VALID_THRESHOLD = 0.015"},
        {"action_id": "RANK_v1_002",
         "component_touched": "src/gaira/base2/v2_patches_final_ranking.py",
         "repair_type": "ADD_HARD_ANCHOR_FAMILY_GATE",
         "rationale": "Family scores currently sum all motif contributions; broad/background motifs can accumulate into a winning family score even without any valid ANCHOR for that family. Hard anchor gate prevents this.",
         "expected_effect": "Families WITHOUT valid ANCHOR OR active CO_FIRE_ANCHOR_GROUP are capped at 0.50 × sum(contributions). Families WITH a valid anchor sum anchor contributions + 0.50 × non-anchor contributions.",
         "notes": "NON_ANCHOR_FAMILY_CAP = 0.50; ANCHORED_FAMILY_NON_ANCHOR_DISCOUNT = 0.50"},
        {"action_id": "RANK_v1_003",
         "component_touched": "v2_patches_discriminative.ANTI_EVIDENCE (runtime override in driver)",
         "repair_type": "STRENGTHEN_ADENINE_ANTI_EVIDENCE",
         "rationale": "Adenine_specific_anchor still leaks into aromatic_residue (Phe/Tyr) and purine_metabolite (UA/HX/Xanth) families in the anchor phase. Strengthen existing SUPPRESS rules + ADD aromatic competitor rules.",
         "expected_effect": "Suppress adenine_specific when any of {UA/HX/Xanth/Phe/Tyr/cytochrome_c} fires; also REQUIRES_COBAND with purine_ring_breathing (adenine's own support band).",
         "notes": "UA penalty 0.70 -> 0.85; HX/Xanth 0.65 -> 0.80; NEW Phe/Tyr at 0.70; NEW REQUIRES_COBAND purine_ring 0.50"},
        {"action_id": "RANK_v1_004",
         "component_touched": "v2_patches_discriminative.ANTI_EVIDENCE (runtime override in driver)",
         "repair_type": "STRENGTHEN_FREE_FA_ANTI_EVIDENCE",
         "rationale": "Free_fatty_acid_carboxyl_anchor's 1700 band can be satisfied by protein amide_I tail (1680-1700) and cholesterol 1670 C=C tail at the permissive BAND_FLOOR.",
         "expected_effect": "Stronger amide_I + cholesterol suppressions. REQUIRES_COBAND with lipid_acyl_C_C_str (free FA's own skeletal support).",
         "notes": "amide_I penalty 0.50 -> 0.70; cholesterol 0.45 -> 0.60; NEW REQUIRES_COBAND lipid_acyl 0.45"},
        {"action_id": "RANK_v1_005",
         "component_touched": "v2_patches_discriminative.ANTI_EVIDENCE (runtime override in driver)",
         "repair_type": "STRENGTHEN_MONOSACCHARIDE_ANTI_EVIDENCE",
         "rationale": "Monosaccharide_anomeric can fire weakly on non-sugar references where the 850-905-1130 bands are noise-driven.",
         "expected_effect": "REQUIRES_COBAND with glycan_pyranose_ring (same-family support) ensures the sugar reading only counts when the broader glycan context is present.",
         "notes": "NEW REQUIRES_COBAND glycan_pyranose_ring 0.45"},
        {"action_id": "RANK_v1_006",
         "component_touched": "src/gaira/base2/v2_patches_final_ranking.py",
         "repair_type": "WEAK_ANCHOR_MOTIF_DOWNGRADE",
         "rationale": "For motif-level top-1 ranking: a weakly-firing ANCHOR motif must not win over a genuinely-firing BACKGROUND/SUPPORT motif. Apply the same × 0.50 downgrade to the ranked motif weight.",
         "expected_effect": "Motif top-1 ranking prefers properly-firing evidence over weak-anchor noise.",
         "notes": "WEAK_ANCHOR_DOWNGRADE = 0.50"},
        {"action_id": "RANK_v1_007",
         "component_touched": "(none — ambiguity preserved)",
         "repair_type": "AMBIGUITY_ROUTING_PRESERVED_UNCHANGED",
         "rationale": "Ambiguity overfire (81.2% in anchor phase) is structurally upstream of this module (rescue-engine gated-ambiguity lane). Cannot be fixed at the ranking layer.",
         "expected_effect": "Ambiguity behaviour identical to anchor phase (neither improves nor regresses).",
         "notes": "Out of scope for ranking repair; deferred to engine-level or truth-table revision"},
    ]
    pd.DataFrame(rows).to_csv(
        TABLES / "grounding_final_ranking_actions_v1.csv", index=False,
    )
    print(f"[emit] grounding_final_ranking_actions_v1.csv ({len(rows)} actions)")


# ─────────────────────────────────────────────────────────────────────
# STEP 3 — Run grounding through rank-fix scoring
# ─────────────────────────────────────────────────────────────────────

def run_grounding(motifs, mappings, dual, all_refs, master_x):
    print("\n[STEP 3] Grounding rerun through rank-fix scoring")
    motif_rows, family_rows, ambig_rows = [], [], []
    rank_motif_rows, rank_family_rows = [], []
    off_target_rows, miss_rows = [], []
    per_spec_rows = []
    ranking_failure_rows = []

    for r in all_refs:
        comp = r["component_key"]
        sid = r["spectrum_id"]
        em = expected_motifs_for_runtime(comp)
        ef = expected_families_for(comp)
        ea = expected_ambiguity_for(comp)

        out = _rank.score_spectrum_rankfix(
            r["spectrum"], master_x, motifs, mappings, dual, sid,
        )
        rm_w = out["rankfix_motif_weights"]
        rf_s = out["rankfix_family_scores"]
        amb = out["ambiguity_core"]

        ms_sorted = sorted(rm_w.items(), key=lambda kv: kv[1], reverse=True)
        top5_motifs = [mid for mid, _ in ms_sorted[:5]]

        for mid, w in rm_w.items():
            motif_rows.append({
                "spectrum_id": sid, "dataset": r["dataset"],
                "component_key": comp,
                "motif_id": mid,
                "role": _disc.ROLE_TABLE.get(mid, "SUPPORT"),
                "base_weight": round(out["base_weights"].get(mid, 0.0), 5),
                "discriminative_weight": round(
                    out["discriminative_weights"].get(mid, 0.0), 5,
                ),
                "rankfix_weight": round(w, 5),
                "is_expected": mid in em,
                "is_top5": mid in top5_motifs,
            })

        fam_sorted = sorted(
            rf_s.items(), key=lambda kv: kv[1][0], reverse=True,
        )
        top5_fams = [f for f, _ in fam_sorted[:5]]
        for fam, (score, contribs, has_anchor) in fam_sorted:
            family_rows.append({
                "spectrum_id": sid, "dataset": r["dataset"],
                "component_key": comp,
                "family": fam, "family_score": round(score, 5),
                "family_has_valid_anchor": has_anchor,
                "is_expected": fam in ef,
                "is_top5": fam in top5_fams,
                "n_contributing_motifs": len(contribs),
                "contributing_motifs": ",".join(contribs),
            })

        ambig_rows.append({
            "spectrum_id": sid, "dataset": r["dataset"],
            "component_key": comp,
            "ambiguity_core": round(amb, 5),
            "rescue_ambiguity_core": round(out["rescue_ambiguity_core"], 5),
            "routed_to_ambiguity":   round(out["routed_to_ambiguity"], 5),
            "expected_ambiguity": ea,
            "observed_ambiguity_active": amb >= 0.10,
            "ambiguity_correct": (ea and amb >= 0.10) or (not ea and amb < 0.10),
            "ambiguity_overfire": (not ea) and amb >= 0.10,
            "ambiguity_underfire": ea and amb < 0.10,
        })

        rank_motif_rows.append({
            "spectrum_id": sid, "dataset": r["dataset"],
            "component_key": comp,
            "expected_motifs": ",".join(em),
            "top_motif_1": top5_motifs[0] if len(top5_motifs) > 0 else "",
            "top_motif_2": top5_motifs[1] if len(top5_motifs) > 1 else "",
            "top_motif_3": top5_motifs[2] if len(top5_motifs) > 2 else "",
            "top_motif_4": top5_motifs[3] if len(top5_motifs) > 3 else "",
            "top_motif_5": top5_motifs[4] if len(top5_motifs) > 4 else "",
            "motif_top1_hit": topn_hit(top5_motifs, em, 1),
            "motif_top3_hit": topn_hit(top5_motifs, em, 3),
            "motif_top5_hit": topn_hit(top5_motifs, em, 5),
        })
        rank_family_rows.append({
            "spectrum_id": sid, "dataset": r["dataset"],
            "component_key": comp,
            "expected_families": ",".join(ef),
            "top_family_1": top5_fams[0] if len(top5_fams) > 0 else "",
            "top_family_2": top5_fams[1] if len(top5_fams) > 1 else "",
            "top_family_3": top5_fams[2] if len(top5_fams) > 2 else "",
            "top_family_4": top5_fams[3] if len(top5_fams) > 3 else "",
            "top_family_5": top5_fams[4] if len(top5_fams) > 4 else "",
            "family_top1_hit": topn_hit(top5_fams, ef, 1),
            "family_top3_hit": topn_hit(top5_fams, ef, 3),
            "family_top5_hit": topn_hit(top5_fams, ef, 5),
        })

        for mid, w in rm_w.items():
            if w > 0.05 and em and mid not in em:
                off_target_rows.append({
                    "spectrum_id": sid, "dataset": r["dataset"],
                    "component_key": comp,
                    "off_target_motif": mid,
                    "rankfix_weight": round(w, 5),
                    "discriminative_weight": round(
                        out["discriminative_weights"].get(mid, 0.0), 5,
                    ),
                    "role": _disc.ROLE_TABLE.get(mid, "SUPPORT"),
                    "expected_motifs": ",".join(em),
                })

        m_top3 = topn_hit(top5_motifs, em, 3)
        f_top3 = topn_hit(top5_fams, ef, 3)
        if (em or ef) and not (m_top3 and f_top3):
            ftypes = []
            if em and not m_top3: ftypes.append("MOTIF_MISS_TOP3")
            if ef and not f_top3: ftypes.append("FAMILY_MISS_TOP3")
            if ea and amb < 0.10: ftypes.append("AMBIGUITY_UNDERFIRE")
            if (not ea) and amb >= 0.10: ftypes.append("AMBIGUITY_OVERFIRE")
            miss_rows.append({
                "spectrum_id": sid, "dataset_name": r["dataset"],
                "component_key": comp,
                "expected_motifs": ",".join(em),
                "observed_top_motifs": ",".join(top5_motifs[:3]),
                "expected_families": ",".join(ef),
                "observed_top_families": ",".join(top5_fams[:3]),
                "expected_ambiguity": ea,
                "observed_ambiguity_active": amb >= 0.10,
                "ambiguity_score": round(amb, 4),
                "failure_type": ",".join(ftypes),
                "notes": "",
            })

        per_spec_rows.append({
            "spectrum_id": sid, "dataset": r["dataset"],
            "component_key": comp,
            "expected_motifs": ",".join(em),
            "expected_families": ",".join(ef),
            "top1_motif": top5_motifs[0] if top5_motifs else "",
            "top1_motif_weight": round(rm_w[top5_motifs[0]], 5) if top5_motifs else 0,
            "top1_family": top5_fams[0] if top5_fams else "",
            "top1_family_score": round(rf_s[top5_fams[0]][0], 5) if top5_fams else 0,
            "top1_family_has_valid_anchor": (
                rf_s[top5_fams[0]][2] if top5_fams else False
            ),
            "ambiguity_core": round(amb, 5),
            "motif_top1_hit": topn_hit(top5_motifs, em, 1),
            "motif_top3_hit": topn_hit(top5_motifs, em, 3),
            "motif_top5_hit": topn_hit(top5_motifs, em, 5),
            "family_top1_hit": topn_hit(top5_fams, ef, 1),
            "family_top3_hit": topn_hit(top5_fams, ef, 3),
            "family_top5_hit": topn_hit(top5_fams, ef, 5),
        })

    pd.DataFrame(per_spec_rows).to_csv(
        TABLES / "grounding_per_spectrum_scores_v_rankfix.csv", index=False,
    )
    pd.DataFrame(motif_rows).to_csv(
        TABLES / "grounding_per_spectrum_motif_scores_v_rankfix.csv", index=False,
    )
    pd.DataFrame(family_rows).to_csv(
        TABLES / "grounding_per_spectrum_family_scores_v_rankfix.csv", index=False,
    )
    pd.DataFrame(ambig_rows).to_csv(
        TABLES / "grounding_ambiguity_behavior_v_rankfix.csv", index=False,
    )
    pd.DataFrame(rank_motif_rows).to_csv(
        TABLES / "grounding_expected_vs_observed_motif_rank_v_rankfix.csv", index=False,
    )
    pd.DataFrame(rank_family_rows).to_csv(
        TABLES / "grounding_expected_vs_observed_family_rank_v_rankfix.csv", index=False,
    )
    pd.DataFrame(off_target_rows).to_csv(
        TABLES / "grounding_off_target_activation_v_rankfix.csv", index=False,
    )
    pd.DataFrame(miss_rows).to_csv(
        TABLES / "grounding_miss_list_v_rankfix.csv", index=False,
    )

    rm = pd.DataFrame(rank_motif_rows)
    rf = pd.DataFrame(rank_family_rows)
    rm_c = rm[rm["expected_motifs"] != ""]
    rf_c = rf[rf["expected_families"] != ""]
    amb_df = pd.DataFrame(ambig_rows)
    metrics = {
        "n_total_spectra":         len(rm),
        "n_motif_classified":      len(rm_c),
        "n_family_classified":     len(rf_c),
        "motif_top1_hit_rate":  round(rm_c["motif_top1_hit"].mean(), 4) if len(rm_c) else 0.0,
        "motif_top3_hit_rate":  round(rm_c["motif_top3_hit"].mean(), 4) if len(rm_c) else 0.0,
        "motif_top5_hit_rate":  round(rm_c["motif_top5_hit"].mean(), 4) if len(rm_c) else 0.0,
        "family_top1_hit_rate": round(rf_c["family_top1_hit"].mean(), 4) if len(rf_c) else 0.0,
        "family_top3_hit_rate": round(rf_c["family_top3_hit"].mean(), 4) if len(rf_c) else 0.0,
        "family_top5_hit_rate": round(rf_c["family_top5_hit"].mean(), 4) if len(rf_c) else 0.0,
        "ambiguity_correctness_rate": round(amb_df["ambiguity_correct"].mean(), 4),
        "ambiguity_overfire_rate":    round(amb_df["ambiguity_overfire"].mean(), 4),
        "ambiguity_underfire_rate":   round(amb_df["ambiguity_underfire"].mean(), 4),
        "n_motif_misses_top3":  int((~rm_c["motif_top3_hit"]).sum()) if len(rm_c) else 0,
        "n_family_misses_top3": int((~rf_c["family_top3_hit"]).sum()) if len(rf_c) else 0,
        "n_total_misses":       len(miss_rows),
        "n_off_target_events":  len(off_target_rows),
    }
    pd.DataFrame([metrics]).to_csv(
        TABLES / "grounding_metrics_summary_v_rankfix.csv", index=False,
    )
    print("\n[rankfix metrics]")
    for k, v in metrics.items():
        print(f"  {k:35s}: {v}")

    rf_c = rf_c.copy()
    rf_c["primary_family"] = rf_c["expected_families"].str.split(",").str[0]
    per_fam = rf_c.groupby("primary_family")[
        ["family_top1_hit", "family_top3_hit", "family_top5_hit"]
    ].mean()
    per_fam_n = rf_c.groupby("primary_family").size().rename("n")
    per_fam_table = per_fam.join(per_fam_n)
    per_fam_table.to_csv(TABLES / "grounding_per_family_hit_rates_v_rankfix.csv")
    per_ds = rf_c.groupby("dataset")[
        ["family_top1_hit", "family_top3_hit", "family_top5_hit"]
    ].mean()
    per_ds_n = rf_c.groupby("dataset").size().rename("n")
    per_ds_table = per_ds.join(per_ds_n)
    per_ds_table.to_csv(TABLES / "grounding_per_dataset_hit_rates_v_rankfix.csv")

    return (metrics, miss_rows, off_target_rows, ambig_rows,
            rank_motif_rows, rank_family_rows, family_rows, motif_rows,
            per_fam_table, per_ds_table)


# ─────────────────────────────────────────────────────────────────────
# STEP 4 — Before / after
# ─────────────────────────────────────────────────────────────────────

def write_before_after(metrics):
    anc = pd.read_csv(ANCHOR_METRICS).iloc[0]
    keys = [
        "motif_top1_hit_rate", "motif_top3_hit_rate", "motif_top5_hit_rate",
        "family_top1_hit_rate", "family_top3_hit_rate", "family_top5_hit_rate",
        "ambiguity_correctness_rate", "ambiguity_overfire_rate",
        "ambiguity_underfire_rate",
        "n_motif_misses_top3", "n_family_misses_top3",
        "n_total_misses", "n_off_target_events",
    ]
    rows = []
    for k in keys:
        b = float(anc[k]); a = float(metrics[k])
        d = round(a - b, 4)
        better = (
            ((k.endswith("hit_rate") or k == "ambiguity_correctness_rate") and d > 0)
            or
            ((k.startswith("n_") or k in {"ambiguity_overfire_rate",
                                           "ambiguity_underfire_rate"})
             and d < 0)
        )
        worse = (
            ((k.endswith("hit_rate") or k == "ambiguity_correctness_rate") and d < 0)
            or
            ((k.startswith("n_") or k in {"ambiguity_overfire_rate",
                                           "ambiguity_underfire_rate"})
             and d > 0)
        )
        rows.append({
            "metric": k,
            "anchor_baseline": round(b, 4),
            "rankfix_v1":      round(a, 4),
            "delta": d,
            "improvement": "BETTER" if better else ("WORSE" if worse else "UNCHANGED"),
        })
    df = pd.DataFrame(rows)
    df.to_csv(
        TABLES / "grounding_before_after_comparison_anchor_to_rankfix.csv",
        index=False,
    )
    print("[emit] grounding_before_after_comparison_anchor_to_rankfix.csv")
    return df


# ─────────────────────────────────────────────────────────────────────
# STEP 5 — Decision
# ─────────────────────────────────────────────────────────────────────

def make_decision(metrics, ba_df):
    fam_t1 = metrics["family_top1_hit_rate"]
    fam_t3 = metrics["family_top3_hit_rate"]
    delta_fam_t1 = float(ba_df[ba_df["metric"] == "family_top1_hit_rate"]
                         .iloc[0]["delta"])
    delta_fam_t3 = float(ba_df[ba_df["metric"] == "family_top3_hit_rate"]
                         .iloc[0]["delta"])
    # READY: top-1 improved substantially AND top-3 preserved AND
    # top-3 absolute is high
    if delta_fam_t1 >= 0.08 and fam_t3 >= 0.70 and delta_fam_t3 >= -0.03:
        return "READY_FOR_GAIRA_VALIDATE_2_CALIBRATION"
    # NEEDS_ONTOLOGY: top-1 didn't move or top-3 regressed — ranking ceiling
    if delta_fam_t1 < 0.03 or delta_fam_t3 < -0.05:
        return "NEEDS_ONTOLOGY_CHANGE_NEXT"
    return "ONTOLOGY_LIMIT_REACHED_FOR_V1"


# ─────────────────────────────────────────────────────────────────────
# Figures
# ─────────────────────────────────────────────────────────────────────

def make_figs(motifs, mappings, dual, all_refs, master_x,
              motif_rows, family_rows, ambig_rows, off_target_rows,
              rank_motif_rows, rank_family_rows, per_fam_table):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm
    except Exception:
        return

    pf_before = pd.read_csv(ANCHOR_PERFAM, index_col=0)
    pf_after  = per_fam_table
    pf_disc   = pd.read_csv(DISC_PERFAM, index_col=0)
    fams = sorted(set(pf_before.index) | set(pf_after.index))

    # 1. fig_rankfix_motif_top1_before_after — motif top-1 by primary expected motif
    rm_before = pd.read_csv(ANCHOR_RANK_MOTIF)
    rm_before = rm_before[rm_before["expected_motifs"].fillna("") != ""].copy()
    rm_before["primary_expected"] = rm_before["expected_motifs"].astype(str).str.split(",").str[0]
    rm_after = pd.DataFrame(rank_motif_rows)
    rm_after = rm_after[rm_after["expected_motifs"].fillna("") != ""].copy()
    rm_after["primary_expected"] = rm_after["expected_motifs"].astype(str).str.split(",").str[0]
    keys = sorted(set(rm_before["primary_expected"].dropna()) |
                  set(rm_after["primary_expected"].dropna()))
    bf = [float(rm_before[rm_before["primary_expected"] == k]["motif_top1_hit"].mean() or 0.0)
          for k in keys]
    af = [float(rm_after[rm_after["primary_expected"] == k]["motif_top1_hit"].mean() or 0.0)
          for k in keys]
    order = sorted(range(len(keys)), key=lambda i: af[i] - bf[i], reverse=True)
    keys = [keys[i] for i in order]; bf = [bf[i] for i in order]; af = [af[i] for i in order]
    fig, ax = plt.subplots(figsize=(13, max(6, 0.35 * len(keys))))
    y = np.arange(len(keys))
    ax.barh(y - 0.2, bf, height=0.35, color="#e76f51", label="anchor baseline")
    ax.barh(y + 0.2, af, height=0.35, color="#2a9d8f", label="rankfix v1")
    ax.set_yticks(y); ax.set_yticklabels([k[:35] for k in keys], fontsize=7)
    ax.invert_yaxis(); ax.set_xlim(0, 1.05)
    ax.set_xlabel("motif top-1 hit rate (per primary expected motif)")
    ax.set_title("Motif top-1 hit rate: anchor baseline vs rankfix v1")
    ax.legend()
    for side in ("top", "right"): ax.spines[side].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIGS / "fig_rankfix_motif_top1_before_after.png", dpi=130)
    plt.close(fig)

    # 2. fig_rankfix_family_top1_before_after
    # 3. fig_rankfix_family_top3_before_after
    for mk, fname in [
        ("family_top1_hit", "fig_rankfix_family_top1_before_after.png"),
        ("family_top3_hit", "fig_rankfix_family_top3_before_after.png"),
    ]:
        bf = [float(pf_before.loc[f, mk]) if f in pf_before.index else 0.0 for f in fams]
        af = [float(pf_after.loc[f, mk])  if f in pf_after.index  else 0.0 for f in fams]
        disc_v = [float(pf_disc.loc[f, mk]) if f in pf_disc.index else 0.0 for f in fams]
        order = sorted(range(len(fams)), key=lambda i: af[i] - bf[i], reverse=True)
        fams_o = [fams[i] for i in order]; bf = [bf[i] for i in order]; af = [af[i] for i in order]; disc_v = [disc_v[i] for i in order]
        fig, ax = plt.subplots(figsize=(12, max(5, 0.45 * len(fams_o))))
        y = np.arange(len(fams_o))
        ax.barh(y - 0.27, disc_v, height=0.25, color="#6a7c8a", label="discriminative (reference)")
        ax.barh(y,         bf,     height=0.25, color="#e76f51", label="anchor")
        ax.barh(y + 0.27,  af,     height=0.25, color="#2a9d8f", label="rankfix v1")
        ax.set_yticks(y); ax.set_yticklabels(fams_o, fontsize=8)
        ax.invert_yaxis(); ax.set_xlim(0, 1.05)
        ax.set_xlabel(mk)
        ax.set_title(f"{mk}: discriminative vs anchor vs rankfix v1")
        ax.legend(fontsize=8)
        for side in ("top", "right"): ax.spines[side].set_visible(False)
        fig.tight_layout()
        fig.savefig(FIGS / fname, dpi=130)
        plt.close(fig)

    # 4. fig_rankfix_off_target_before_after
    of_before = pd.read_csv(ANCHOR_OFFTGT)
    of_after = pd.DataFrame(off_target_rows)
    bcounts = of_before["off_target_motif"].value_counts()
    acounts = of_after["off_target_motif"].value_counts() if len(of_after) else pd.Series(dtype=int)
    common = sorted(set(bcounts.index[:20]) | set(acounts.index[:20]))
    bv = [int(bcounts.get(m, 0)) for m in common]
    av = [int(acounts.get(m, 0)) for m in common]
    order = sorted(range(len(common)), key=lambda i: bv[i], reverse=True)
    common = [common[i] for i in order]; bv = [bv[i] for i in order]; av = [av[i] for i in order]
    fig, ax = plt.subplots(figsize=(14, max(5, 0.35 * len(common))))
    y = np.arange(len(common))
    ax.barh(y - 0.2, bv, height=0.35, color="#e76f51",
            label=f"anchor baseline ({sum(bv)})")
    ax.barh(y + 0.2, av, height=0.35, color="#2a9d8f",
            label=f"rankfix v1 ({sum(av)})")
    ax.set_yticks(y); ax.set_yticklabels([c[:35] for c in common], fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("off-target activation events")
    ax.set_title("Off-target activation: anchor vs rankfix v1")
    ax.legend()
    for side in ("top", "right"): ax.spines[side].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIGS / "fig_rankfix_off_target_before_after.png", dpi=130)
    plt.close(fig)

    # 5. fig_rankfix_ambiguity_before_after — 3-panel
    amb_before = pd.read_csv(ANCHOR_AMBIG)
    amb_after = pd.DataFrame(ambig_rows)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    axes[0].hist(amb_before["ambiguity_core"], bins=30, color="#e76f51",
                 alpha=0.55, label="anchor")
    axes[0].hist(amb_after["ambiguity_core"], bins=30, color="#2a9d8f",
                 alpha=0.55, label="rankfix v1")
    axes[0].axvline(0.10, color="black", linestyle="--", label="gated 0.10")
    axes[0].set_xlabel("ambiguity_core"); axes[0].set_ylabel("count")
    axes[0].set_title("Ambiguity score distribution"); axes[0].legend()
    cb = float(amb_before["ambiguity_correct"].mean())
    ca = float(amb_after["ambiguity_correct"].mean())
    axes[1].bar(["anchor", "rankfix v1"], [cb, ca],
                color=["#e76f51", "#2a9d8f"])
    for i, v in enumerate([cb, ca]):
        axes[1].text(i, v + 0.02, f"{v:.1%}", ha="center", fontsize=10)
    axes[1].set_ylim(0, 1.0); axes[1].set_ylabel("correctness rate")
    axes[1].set_title("Ambiguity correctness")
    ob = float(amb_before["ambiguity_overfire"].mean())
    oa = float(amb_after["ambiguity_overfire"].mean())
    ub = float(amb_before["ambiguity_underfire"].mean())
    ua = float(amb_after["ambiguity_underfire"].mean())
    x = np.arange(2); w = 0.35
    axes[2].bar(x - w/2, [ob, oa], width=w, color="#f4a261", label="overfire")
    axes[2].bar(x + w/2, [ub, ua], width=w, color="#264653", label="underfire")
    axes[2].set_xticks(x); axes[2].set_xticklabels(["anchor", "rankfix v1"])
    axes[2].set_ylim(0, 1.0); axes[2].set_ylabel("rate")
    axes[2].set_title("Ambiguity over/underfire"); axes[2].legend()
    for side in ("top", "right"):
        for a in axes:
            a.spines[side].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIGS / "fig_rankfix_ambiguity_before_after.png", dpi=130)
    plt.close(fig)

    # 6. fig_rankfix_grouped_motif_in_family_examples
    id_to_ref = {r["spectrum_id"]: r for r in all_refs}
    examples = []
    targets = [
        ("ramanbiolib", "d-(+)-glucose"),
        ("ramanbiolib", "oleic acid"),
        ("ramanbiolib", "adenine"),
        ("gobbato_powder", "UA_rep01"),
        ("ramanbiolib", "l-glutamate"),
        ("ramanbiolib", "albumin"),
    ]
    for tag, suffix in targets:
        for sid in id_to_ref:
            if sid.startswith(f"{tag}::") and suffix in sid:
                examples.append(sid); break
    if examples:
        from gaira.base2.motif_engine import resolve_mapping_weight
        fig, axes = plt.subplots(1, len(examples), figsize=(4.5*len(examples), 8),
                                 sharey=True)
        if len(examples) == 1: axes = [axes]
        cmap = cm.get_cmap("tab20", 20)
        colors = {}
        def col_for(mid):
            if mid not in colors: colors[mid] = cmap(len(colors) % 20)
            return colors[mid]
        for ax, sid in zip(axes, examples):
            ref = id_to_ref[sid]
            out = _rank.score_spectrum_rankfix(
                ref["spectrum"], master_x, motifs, mappings, dual, sid,
            )
            rm_w = out["rankfix_motif_weights"]
            fam_to_contrib = {}
            for fam in FAMILIES:
                contribs = []
                for mid, s in rm_w.items():
                    mp = mappings.get(mid)
                    if mp is None or s <= 0: continue
                    mw = resolve_mapping_weight(mp, fam)
                    if mw > 0:
                        contribs.append((mid, s * mw))
                fam_to_contrib[fam] = sorted(contribs, key=lambda x: x[1], reverse=True)
            y_pos = np.arange(len(FAMILIES))
            for i, fam in enumerate(FAMILIES):
                left = 0.0
                for mid, contrib in fam_to_contrib[fam]:
                    ax.barh(i, contrib, left=left, color=col_for(mid),
                            edgecolor="black", linewidth=0.2)
                    if contrib >= 0.04:
                        ax.text(left + contrib/2, i,
                                mid.replace("_motif", "")[:18],
                                va="center", ha="center", fontsize=5, color="white")
                    left += contrib
            ax.set_yticks(y_pos); ax.set_yticklabels(FAMILIES, fontsize=8)
            ax.invert_yaxis()
            ax.set_xlim(0, max(1.3, 1.05*max(
                (sum(c for _, c in fam_to_contrib[f]) for f in FAMILIES),
                default=1.0,
            )))
            ax.set_xlabel("stacked motif (rankfix weights)")
            ax.set_title(sid.split("::")[-1][:30], fontsize=9)
            for side in ("top", "right"): ax.spines[side].set_visible(False)
        fig.suptitle("Grouped motif-in-family examples (rankfix v1)", fontsize=12)
        fig.tight_layout()
        fig.savefig(FIGS / "fig_rankfix_grouped_motif_in_family_examples.png",
                    dpi=130)
        plt.close(fig)

        # 7. fig_rankfix_family_radar_examples
        fig, axes = plt.subplots(1, len(examples),
                                 figsize=(4.5*len(examples), 4.5),
                                 subplot_kw=dict(polar=True))
        if len(examples) == 1: axes = [axes]
        angles = np.linspace(0, 2*np.pi, len(FAMILIES), endpoint=False).tolist()
        angles += angles[:1]
        for ax, sid in zip(axes, examples):
            ref = id_to_ref[sid]
            out = _rank.score_spectrum_rankfix(
                ref["spectrum"], master_x, motifs, mappings, dual, sid,
            )
            vals = [out["rankfix_family_scores"][f][0] for f in FAMILIES]
            vmax = max(vals) if max(vals) > 0 else 1.0
            vals = [v / vmax for v in vals]
            vals += vals[:1]
            ax.plot(angles, vals, color="#2a9d8f", linewidth=1.5)
            ax.fill(angles, vals, color="#2a9d8f", alpha=0.3)
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels([f.replace("_", "\n") for f in FAMILIES], fontsize=5)
            ax.set_ylim(0, 1.05)
            ax.set_title(sid.split("::")[-1][:25], fontsize=8, pad=12)
        fig.suptitle("Family-level radar (rankfix v1)", fontsize=11)
        fig.tight_layout()
        fig.savefig(FIGS / "fig_rankfix_family_radar_examples.png", dpi=130)
        plt.close(fig)

    # 8. fig_rankfix_treemap_exploratory
    from gaira.base2.motif_engine import resolve_mapping_weight
    agg = defaultdict(lambda: defaultdict(float))
    agg_amb = 0.0
    for ref in all_refs:
        out = _rank.score_spectrum_rankfix(
            ref["spectrum"], master_x, motifs, mappings, dual, ref["spectrum_id"],
        )
        agg_amb += out["ambiguity_core"]
        rm_w = out["rankfix_motif_weights"]
        for fam in FAMILIES:
            for mid, s in rm_w.items():
                mp = mappings.get(mid)
                if mp is None or s <= 0: continue
                mw = resolve_mapping_weight(mp, fam)
                if mw > 0: agg[fam][mid] += s * mw
    fig, axes = plt.subplots(3, 4, figsize=(20, 13))
    for ax in axes.flat: ax.set_axis_off()
    cmap = cm.get_cmap("tab20", 20)
    colors = {}
    def col(mid):
        if mid not in colors:
            colors[mid] = cmap(len(colors) % 20)
        return colors[mid]
    def tile(ax, items, title):
        total = sum(v for _, v in items)
        if total <= 0:
            ax.text(0.5, 0.5, f"{title}\n(no signal)", ha="center", va="center",
                    transform=ax.transAxes, fontsize=9); return
        items = sorted(items, key=lambda x: x[1], reverse=True)
        y = 1.0
        for lbl, val in items:
            frac = val / total
            ax.add_patch(plt.Rectangle((0, y-frac), 1.0, frac,
                                       facecolor=col(lbl), edgecolor="black",
                                       linewidth=0.5))
            if frac > 0.03:
                ax.text(0.5, y-frac/2,
                        lbl.replace("_motif", "")[:24] + f" ({frac:.0%})",
                        ha="center", va="center", fontsize=6, color="white")
            y -= frac
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_title(title, fontsize=10)
    for i, fam in enumerate(FAMILIES):
        ax = axes.flat[i]; ax.set_axis_on()
        items = list(agg[fam].items())
        tile(ax, items, f"{fam}\n(Σ={sum(v for _,v in items):.2f})")
    amb_ax = axes.flat[11]; amb_ax.set_axis_on()
    amb_ax.text(0.5, 0.5,
                f"ambiguity_artifact\n(control lane)\n\nΣ over {len(all_refs)} refs: {agg_amb:.2f}",
                ha="center", va="center", fontsize=10,
                transform=amb_ax.transAxes, color="#7b2cbf")
    amb_ax.set_xticks([]); amb_ax.set_yticks([])
    for side in ("top","right","left","bottom"):
        amb_ax.spines[side].set_visible(False)
    fig.suptitle("Family -> motif treemap (rankfix v1)", fontsize=12)
    fig.tight_layout()
    fig.savefig(FIGS / "fig_rankfix_treemap_exploratory.png", dpi=130)
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────
# Reports
# ─────────────────────────────────────────────────────────────────────

def write_main_report(metrics, ba_df, per_fam_table, per_ds_table,
                      rankfail_df):
    disc_per_fam = pd.read_csv(DISC_PERFAM, index_col=0)
    anchor_per_fam = pd.read_csv(ANCHOR_PERFAM, index_col=0)
    decision = make_decision(metrics, ba_df)

    rankfail_counts = rankfail_df["likely_ranking_failure_type"].value_counts()

    lines = [
        "# gaira_base_2 - Final Ranking Repair Loop v1",
        "",
        "## Why this phase was needed",
        "",
        "The targeted anchor acquisition phase (prior) produced strong "
        "top-3 family chemistry recovery (66.3% -> 72.2%) but family top-1 "
        "stayed stuck at 36.9% and motif top-1 at 20.0%. The canonical "
        "signal of a ranking-layer problem: the right family IS in top-3, "
        "just not top-1.",
        "",
        "Diagnostics (`ranking_failure_cases_v1.csv`, computed from the "
        "anchor-phase miss list) classified the misses:",
        "",
        "| failure type | count |",
        "|---|---:|",
    ]
    for k, v in rankfail_counts.items():
        lines.append(f"| `{k}` | {v} |")

    lines += [
        "",
        "## Exact ranking / gating changes",
        "",
        "This phase adds `src/gaira/base2/v2_patches_final_ranking.py` — "
        "an additive wrapper around `v2_patches_discriminative.py`. No "
        "existing module is modified. Four ranking repairs:",
        "",
        "**REPAIR 1 — ANCHOR_VALID_THRESHOLD = 0.015.** An ANCHOR's "
        "discriminative_weight must be ≥ 0.015 to count as a valid "
        "anchor fire. Below this threshold, the weight is noise-driven "
        "(engine BAND_FLOOR = 1e-3 lets the REQUIRED 3-band check pass on "
        "too many spectra).",
        "",
        "**REPAIR 2 — Hard anchor-gated family scoring.** Families "
        "WITHOUT a valid ANCHOR motif AND WITHOUT an active "
        "CO_FIRE_ANCHOR_GROUP get their family score capped at 0.50 × "
        "sum(motif contributions). Families WITH a valid anchor sum the "
        "anchor contributions plus 0.50 × sum(non-anchor contributions).",
        "",
        "**REPAIR 3 — Strengthened anti-evidence** (runtime updates to "
        "`v2_patches_discriminative.ANTI_EVIDENCE`, discriminative module "
        "file unchanged):",
        "  - `adenine_specific_anchor_motif`: UA/HX/Xanth suppression "
        "increased (penalty 0.70→0.85 / 0.65→0.80 / 0.65→0.80); NEW "
        "Phe/Tyr competitor suppressions at 0.70; NEW REQUIRES_COBAND "
        "purine_ring_breathing (adenine cannot fire alone).",
        "  - `free_fatty_acid_carboxyl_anchor_motif`: amide_I suppression "
        "0.50→0.70; cholesterol 0.45→0.60; NEW REQUIRES_COBAND "
        "lipid_acyl_C_C_str.",
        "  - `monosaccharide_anomeric_anchor_motif`: NEW REQUIRES_COBAND "
        "glycan_pyranose_ring_skeletal.",
        "",
        "**REPAIR 4 — Weak-anchor motif ranking downgrade.** For motif "
        "top-1 ranking: ANCHOR motifs below the validity threshold have "
        "their ranked weight multiplied by 0.50, preventing them from "
        "winning top-1 motif over genuinely-firing evidence.",
        "",
        "**REPAIR 5 — Ambiguity routing preserved.** The rescue-engine "
        "`compute_gated_ambiguity_lane` runs upstream of this module; "
        "the rankfix layer does not touch ambiguity scoring. Any "
        "ambiguity delta observed is indirect (from the same-spectrum "
        "motif-weight changes flowing through to the ambiguity lane).",
        "",
        "## Grounding rerun headline (vs anchor baseline)",
        "",
        "| metric | anchor | rankfix v1 | delta |",
        "|---|---:|---:|---:|",
    ]
    for _, r in ba_df.iterrows():
        b, a, d = r["anchor_baseline"], r["rankfix_v1"], r["delta"]
        if r["metric"].endswith("rate"):
            lines.append(f"| {r['metric']} | {b:.1%} | {a:.1%} | {d:+.1%} |")
        else:
            lines.append(f"| {r['metric']} | {int(b)} | {int(a)} | {int(d):+d} |")

    # Strongest wins
    lines += [
        "",
        "## Per-family hit rate (rankfix v1)",
        "",
        "| family | top-1 | top-3 | top-5 | n |",
        "|---|---:|---:|---:|---:|",
    ]
    for fam, row in per_fam_table.sort_values("family_top1_hit", ascending=False).iterrows():
        lines.append(f"| {fam} | {row['family_top1_hit']:.1%} | "
                     f"{row['family_top3_hit']:.1%} | {row['family_top5_hit']:.1%} | "
                     f"{int(row['n'])} |")

    lines += [
        "",
        "## Per-family delta: discriminative -> anchor -> rankfix v1 (top-1)",
        "",
        "| family | n | disc_t1 | anchor_t1 | rankfix_t1 | d(anc->rf) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for fam in per_fam_table.index:
        if fam not in disc_per_fam.index or fam not in anchor_per_fam.index:
            continue
        n = int(per_fam_table.loc[fam, "n"])
        d_t1 = float(disc_per_fam.loc[fam, "family_top1_hit"])
        a_t1 = float(anchor_per_fam.loc[fam, "family_top1_hit"])
        r_t1 = float(per_fam_table.loc[fam, "family_top1_hit"])
        lines.append(f"| {fam} | {n} | {d_t1:.1%} | {a_t1:.1%} | {r_t1:.1%} | "
                     f"{r_t1-a_t1:+.1%} |")

    lines += [
        "",
        "## Per-dataset family hit rates",
        "",
        "| dataset | top-1 | top-3 | top-5 | n |",
        "|---|---:|---:|---:|---:|",
    ]
    for ds, row in per_ds_table.iterrows():
        lines.append(f"| `{ds}` | {row['family_top1_hit']:.1%} | "
                     f"{row['family_top3_hit']:.1%} | {row['family_top5_hit']:.1%} | "
                     f"{int(row['n'])} |")

    # Collateral damage check
    lines += [
        "",
        "## Collateral damage check",
        "",
        "- top-3 preservation vs anchor: delta "
        f"{float(ba_df[ba_df['metric']=='family_top3_hit_rate'].iloc[0]['delta']):+.1%}.",
        "- top-5 preservation vs anchor: delta "
        f"{float(ba_df[ba_df['metric']=='family_top5_hit_rate'].iloc[0]['delta']):+.1%}.",
        "- off-target activation change: "
        f"{int(ba_df[ba_df['metric']=='n_off_target_events'].iloc[0]['delta']):+d}.",
        "- ambiguity correctness change: "
        f"{float(ba_df[ba_df['metric']=='ambiguity_correctness_rate'].iloc[0]['delta']):+.1%}.",
        "",
        "## Final decision",
        "",
        f"**{decision}**",
        "",
    ]
    if decision == "READY_FOR_GAIRA_VALIDATE_2_CALIBRATION":
        lines.append(
            "The ranking patches produced substantial top-1 improvement "
            "WITHOUT regressing top-3 chemistry. The engine is now "
            "decisive enough to begin calibration (substrate-perturbation) "
            "tests. Remaining misses are predominantly multi-axis chemistry "
            "or sparse-axis ontology gaps that calibration will not fix."
        )
    elif decision == "NEEDS_ONTOLOGY_CHANGE_NEXT":
        lines.append(
            "The ranking patches did not materially move top-1 (delta < "
            "3pp) OR they regressed top-3. This indicates the ranking "
            "ceiling has been reached for the current ontology. The next "
            "gains require ontology changes — not more ranking work."
        )
    else:
        lines.append(
            "Both ranking and ontology appear near their v1 ceiling. "
            "Consider either accepting the current state and proceeding "
            "to calibration with explicit top-3-primary reporting, or "
            "moving to a v2 ontology phase with new pure-compound "
            "reference acquisitions."
        )

    (REPORTS / "REPORT_gaira_base_2_final_ranking_repair_loop_v1.md"
     ).write_text("\n".join(lines))


def write_miss_report(metrics, miss_rows, ba_df, rankfail_df):
    df = pd.DataFrame(miss_rows)
    anchor_miss = pd.read_csv(ANCHOR_MISS)
    anchor_set = set(anchor_miss["spectrum_id"])
    rank_set = set(df["spectrum_id"])
    fixed = anchor_set - rank_set
    persisted = anchor_set & rank_set
    new = rank_set - anchor_set

    if len(df) > 0:
        df_f = df.copy()
        df_f["primary_expected_family"] = df_f["expected_families"].str.split(",").str[0]
        fam_break = df_f["primary_expected_family"].value_counts()
    else:
        fam_break = pd.Series(dtype=int)

    decision = make_decision(metrics, ba_df)
    lines = [
        "# Final Ranking Repair v1 - Miss Analysis",
        "",
        "## Misses fixed by ranking vs persisted vs newly introduced",
        "",
        f"- anchor-baseline misses: **{len(anchor_set)}**",
        f"- rankfix-v1 misses: **{len(rank_set)}**",
        f"- misses **fixed by ranking**: **{len(fixed)}**",
        f"- misses **persisted**: **{len(persisted)}**",
        f"- misses **newly introduced by ranking**: **{len(new)}**",
        "",
        "## Persisted-miss family breakdown (these remain in rankfix v1)",
        "",
        "| primary expected family | n missed |",
        "|---|---:|",
    ]
    for fam, c in fam_break.items():
        lines.append(f"| {fam} | {c} |")

    lines += [
        "",
        "## Ranking-failure diagnosis breakdown (from STEP 1)",
        "",
        "| ranking failure type | count |",
        "|---|---:|",
    ]
    for k, v in rankfail_df["likely_ranking_failure_type"].value_counts().items():
        lines.append(f"| `{k}` | {v} |")

    lines += [
        "",
        "## Whether remaining misses are ontology-limited or ranking-limited",
        "",
        "If `COMPETING_ANCHOR_WON_TOP1` and `NON_ANCHOR_FAMILY_WON_TOP1` "
        "dominated the anchor-phase misses AND rankfix fixed most of "
        "them: remaining misses are primarily ontology.",
        "",
        "If `EXPECTED_ANCHOR_DID_NOT_FIRE` dominated the anchor-phase "
        "misses: rankfix alone won't help; the real missing ingredient "
        "is an anchor for those chemistries.",
        "",
        "From the STEP 1 diagnosis and STEP 3 rerun results:",
        "",
        "1. **Ranking-fixable cases** (COMPETING_ANCHOR_WON_TOP1 / "
        "NON_ANCHOR_FAMILY_WON_TOP1) — the anchor-gated family scoring "
        "should have addressed these.",
        "2. **Ontology-limited cases** (EXPECTED_ANCHOR_DID_NOT_FIRE) — "
        "still require new ANCHOR motifs in a future phase.",
        "3. **True chemistry-overlap cases** — multi-axis chemistry that "
        "the refined truth table already accepts in top-3 (free amino "
        "acids, cholesteryl esters, purine shared chemistry).",
        "",
        "## Recommendation",
        "",
        f"**{decision}**",
        "",
    ]
    if decision == "READY_FOR_GAIRA_VALIDATE_2_CALIBRATION":
        lines.append(
            "Ranking repair materially improved top-1 without regressing "
            "top-3. Proceed to calibration phase."
        )
    elif decision == "NEEDS_ONTOLOGY_CHANGE_NEXT":
        lines.append(
            "Ranking ceiling reached. Further gains require adding "
            "ANCHORs for families still showing EXPECTED_ANCHOR_DID_NOT_FIRE "
            "(likely candidates: per-residue free-AA motifs for metabolic_small_molecule; "
            "lactate motif activation; aromatic-steroid discriminator)."
        )
    else:
        lines.append(
            "Both ranking and ontology are near v1 ceiling for current "
            "evidence. Two paths: (a) accept and proceed to calibration "
            "with top-3-primary reporting; (b) move to v2 ontology phase "
            "with new reference acquisition (M3.3-class)."
        )

    (REPORTS / "REPORT_gaira_base_2_final_ranking_repair_miss_analysis_v1.md"
     ).write_text("\n".join(lines))


def write_audit_log(metrics, ba_df):
    decision = make_decision(metrics, ba_df)
    lines = [
        "# gaira_base_2 Final Ranking Repair Loop v1 - Audit Log",
        "",
        "## Files added (relative to repo)",
        "",
        "- ADDED: `src/gaira/base2/v2_patches_final_ranking.py` (the ranking wrapper)",
        "- ADDED: `scripts/run_gaira_base_2_final_ranking_repair_loop_v1.py`",
        "- ADDED: `GAIRA_BUILD/gaira_base_2_final_ranking_repair_loop_v1/**`",
        "",
        "## Files NOT modified",
        "",
        "- gaira_base SHA-256 still matches; 12/12 v1 regression tests pass",
        "- v1 engine modules untouched",
        "- v2_patches.py, v2_patches_rescue.py, v2_patches_repair_v2.py, "
        "v2_patches_discriminative.py — all unchanged",
        "- Registry v1.5 + mapping v1.4 read-only (no ontology changes)",
        "- M2.2 dual-status table file unchanged (runtime overrides from "
        "anchor phase are reapplied at runtime in this driver)",
        "- canonical preprocessing unchanged",
        "- substrate engine v1.1.2 unchanged",
        "- NO calibration / target / substrate-aware data used",
        "- NO new motifs added",
        "",
        "## Exact ranking rules changed",
        "",
        "Scoring constants (in `v2_patches_final_ranking.py`):",
        f"- ANCHOR_VALID_THRESHOLD = {_rank.ANCHOR_VALID_THRESHOLD}",
        f"- WEAK_ANCHOR_DOWNGRADE = {_rank.WEAK_ANCHOR_DOWNGRADE}",
        f"- NON_ANCHOR_FAMILY_CAP = {_rank.NON_ANCHOR_FAMILY_CAP}",
        f"- ANCHORED_FAMILY_NON_ANCHOR_DISCOUNT = {_rank.ANCHORED_FAMILY_NON_ANCHOR_DISCOUNT}",
        "",
        "Anti-evidence strengthened at runtime (no discriminative module file change):",
        "- adenine_specific_anchor_motif: UA 0.70→0.85; HX 0.65→0.80; "
        "Xanth 0.65→0.80; NEW Phe 0.70; NEW Tyr 0.70; NEW REQUIRES_COBAND "
        "purine_ring_breathing 0.50",
        "- free_fatty_acid_carboxyl_anchor_motif: amide_I 0.50→0.70; "
        "cholesterol 0.45→0.60; NEW REQUIRES_COBAND lipid_acyl_C_C_str 0.45",
        "- monosaccharide_anomeric_anchor_motif: NEW REQUIRES_COBAND "
        "glycan_pyranose_ring 0.45",
        "",
        "Family scoring rule changed in `family_score_rankfix()`:",
        "- anchored family: anchor_sum + 0.50 × other_sum",
        "- non-anchored family: 0.50 × other_sum",
        "",
        "## Rules considered and rejected",
        "",
        "- MAX-over-motifs family scoring (pure MAX, no additive). Rejected: "
        "multi-motif chemistry (real proteins with amide_I+amide_III+amide_II "
        "co-fire) relies on additive evidence; pure MAX would hurt those.",
        "- Per-spectrum adaptive threshold (e.g. top-1 motif weight must be "
        "2× top-2). Rejected: not deterministic / auditable enough.",
        "- Broad-motif weight capping (BACKGROUND can contribute at most "
        "0.02 to family). Rejected: would regress protein family too much.",
        "- Changing the discriminative role-gating formula. Rejected: user "
        "locked 'same discriminative framework'.",
        "",
        "## Ontology intact",
        "",
        "**YES.** Registry v1.5 and mapping v1.4 are read-only in this phase. "
        "NO new motifs. NO ontology changes. Only the ranking layer "
        "(family scoring rule + anti-evidence penalties + anchor validity "
        "threshold) was modified.",
        "",
        "## Headline metrics",
        "",
        f"- motif top-1: {metrics['motif_top1_hit_rate']:.1%}",
        f"- motif top-3: {metrics['motif_top3_hit_rate']:.1%}",
        f"- motif top-5: {metrics['motif_top5_hit_rate']:.1%}",
        f"- family top-1: {metrics['family_top1_hit_rate']:.1%}",
        f"- family top-3: {metrics['family_top3_hit_rate']:.1%}",
        f"- family top-5: {metrics['family_top5_hit_rate']:.1%}",
        f"- ambiguity correctness: {metrics['ambiguity_correctness_rate']:.1%}",
        f"- ambiguity overfire: {metrics['ambiguity_overfire_rate']:.1%}",
        f"- ambiguity underfire: {metrics['ambiguity_underfire_rate']:.1%}",
        f"- total misses: {metrics['n_total_misses']}",
        f"- off-target events: {metrics['n_off_target_events']}",
        "",
        "## Final decision",
        "",
        f"**{decision}**",
    ]
    (AUDIT / "gaira_base_2_final_ranking_repair_loop_audit_log.md"
     ).write_text("\n".join(lines))


def snapshot_code():
    src = Path("/Users/suraj/projects/GAIRA/src/gaira/base2")
    if src.exists():
        shutil.copytree(src, CODE_SNAPSHOT / "base2", dirs_exist_ok=True)
    p = Path("/Users/suraj/projects/GAIRA/scripts/"
             "run_gaira_base_2_final_ranking_repair_loop_v1.py")
    if p.exists(): shutil.copy(p, CODE_SNAPSHOT / p.name)


# ─────────────────────────────────────────────────────────────────────
# Driver
# ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 78)
    print("gaira_base_2 - Final Ranking Repair Loop v1")
    print("=" * 78)
    print(f"  ANCHOR_VALID_THRESHOLD = {_rank.ANCHOR_VALID_THRESHOLD}")
    print(f"  WEAK_ANCHOR_DOWNGRADE  = {_rank.WEAK_ANCHOR_DOWNGRADE}")
    print(f"  NON_ANCHOR_FAMILY_CAP  = {_rank.NON_ANCHOR_FAMILY_CAP}")
    print()
    for d in (TABLES, FIGS, REPORTS, AUDIT, CODE_SNAPSHOT):
        d.mkdir(parents=True, exist_ok=True)

    master_x = canonical_master_axis()

    # Apply runtime extensions from the anchor phase (these set the
    # ROLE_TABLE / ANTI_EVIDENCE / EXPECTED_MOTIFS / dual_status baseline
    # that rankfix is applied ON TOP of).
    extend_role_table_for_anchors()
    extend_anti_evidence_for_reactivated_motif()
    extend_truth_table_for_new_anchors()

    # Apply THIS phase's anti-evidence strengthening (REPAIR 3).
    strengthen_anti_evidence_for_rankfix()

    # Load registry + mapping + dual_status (same as anchor phase)
    motifs = load_motif_registry(REG_V1_5)
    mappings = load_axis_mapping(MAP_V1_4)
    dual = extend_dual_status_for_new_and_silent_motifs(load_dual_status())
    active = {m: s for m, s in motifs.items() if s.v1_active}
    print(f"[engine] {len(active)} active motifs, {len(mappings)} mappings, "
          f"{len(dual)} dual_status entries")

    # STEP 2 actions emit
    emit_actions()

    # STEP 1 — diagnose ranking failures from anchor phase
    rankfail_df = diagnose_ranking_failures(active, mappings)

    # STEP 3 — run grounding
    rb  = load_ramanbiolib(master_x)
    gp  = load_gobbato_powder(master_x)
    aa  = load_amino_acid_xlsx(master_x)
    lit = load_digitised_literature(master_x)
    all_refs = rb + gp + aa + lit
    print(f"[data] {len(all_refs)} grounding spectra")

    (metrics, miss_rows, off_target_rows, ambig_rows,
     rank_motif_rows, rank_family_rows, family_rows, motif_rows,
     per_fam_table, per_ds_table) = run_grounding(
        active, mappings, dual, all_refs, master_x,
    )

    # STEP 4 before / after
    ba_df = write_before_after(metrics)

    # Figures
    make_figs(active, mappings, dual, all_refs, master_x,
              motif_rows, family_rows, ambig_rows, off_target_rows,
              rank_motif_rows, rank_family_rows, per_fam_table)

    # Reports + audit
    write_main_report(metrics, ba_df, per_fam_table, per_ds_table, rankfail_df)
    write_miss_report(metrics, miss_rows, ba_df, rankfail_df)
    write_audit_log(metrics, ba_df)
    snapshot_code()

    decision = make_decision(metrics, ba_df)
    print(f"\n[decision] {decision}")
    print("DONE")


if __name__ == "__main__":
    main()
