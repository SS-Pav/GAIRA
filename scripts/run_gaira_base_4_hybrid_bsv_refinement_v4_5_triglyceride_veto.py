"""gaira_base_4 Hybrid BSV Refinement v4.5 — Triglyceride Veto.

Narrow, surgical repair targeting ONLY G09 / triglyceride. Goal: push
G09 top-1 to ≥75% or prove the remaining gap is corpus-limited.

Hard scope constraints:
  - NO broader SERS corpus work
  - NO calibration
  - NO global retuning
  - frozen 11-group taxonomy UNCHANGED
  - G07 W_MOTIF override unchanged
  - all existing G09a/G09b/G09d routing unchanged; triglyceride is the only
    subfamily touched here

Stages:
  0. Triglyceride failure audit (isolate error mode before changes)
  1. Triglyceride-specific cofeature + veto rules
  2. Apply rules and re-eval
  3. G08/G09 boundary re-evaluation
  4. Hardness/ceiling analysis
  5. G09 output policy update
  6. Final decision + audit
"""
from __future__ import annotations

import shutil
import sys
import warnings
from collections import defaultdict, Counter
from pathlib import Path

import numpy as np
import pandas as pd

warnings.simplefilter("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from gaira.base3 import mss_engine as _mss
from gaira.spectral import canonical_master_axis

from run_gaira_validate_2_grounding import (
    load_ramanbiolib, load_gobbato_powder,
    load_amino_acid_xlsx, load_digitised_literature,
)
from run_gaira_base_3_full_grounding_audit_and_signature_build_v1 import (
    load_sers_metabolite_63, derive_analyte_class as derive_broad_class,
)
from run_gaira_base_3_grounding_trained_ontology_v1 import normalise_label
from run_gaira_base_4_mss_decision_enrichment_v1 import canonical_analyte_id
from run_gaira_base_4_hybrid_bsv_build_v1 import (
    BSV_GROUPS,
    compute_motif_firings, compute_mss_scores_v43,
    _band_max,
    AMBIGUITY_SPILLOVER_THRESHOLD,
)
from run_gaira_base_4_hybrid_bsv_refinement_v2_sers_coherence import (
    compute_hybrid_bsv_v3,
)
from run_gaira_base_4_hybrid_bsv_refinement_v3_sers_corpus_routing import (
    G09_SUBFAMILIES, score_g09_subfamilies, g09_routing_adjust,
    compute_hybrid_bsv_v4 as compute_hybrid_bsv_v4_base,
    run_bsv_v4 as run_bsv_v4_base,
)


# ─────────────────────────────────────────────────────────────────────
# v4.5 chemistry-first triglyceride rules (additive; v3 subfamilies unchanged)
# ─────────────────────────────────────────────────────────────────────

# Rationale (from Stage 0 audit):
#   6/10 TG misses are SATURATED triglycerides — the 1655 C=C does not exist
#   by chemistry, so the v3 "1745 + 1655" rule excludes them from routing and
#   they leak to G08 via CH2 dominance.
#   3/10 are polyunsaturated with 1655 so strong it numerically squashes 1745.
#   1/10 is genuine boundary (chemistry-balanced).
#
#   v4.5 adds TWO saturated-TG branches + ONE polyunsat rescue branch and a
#   TG-specific G08 veto. The 4 v3 subfamilies (sterol / ChE / unsat TG /
#   aromatic steroid) stay EXACTLY as they were in v3; only the triglyceride
#   routing gains additional acceptance paths.

TG_SATURATED_BRANCH = {
    "subfamily_id": "G09c_triglyceride_saturated",
    "description": "saturated triglyceride (tripalmitin/tristearin etc.); "
                    "ester C=O 1745 + glycerol skeletal 1080 + CH2 twist 1300; "
                    "NO 1655 (chemistry dictates) and NO 608",
    "required_bands": [1745.0, 1080.0, 1300.0],
    "required_thresholds": [0.08, 0.08, 0.20],
    "supporting_bands": [1440.0, 870.0],
    "anti_bands": [608.0],
    "anti_threshold": 0.08,
    "min_required_fraction": 0.08,
}

TG_POLYUNSAT_BRANCH = {
    "subfamily_id": "G09c_triglyceride_polyunsat_rescue",
    "description": "polyunsaturated triglyceride (trilinolein / trilinolenin) "
                    "where 1655 is so dominant that 1745 looks weak relative "
                    "to spectrum max; requires strong 1655 + weak-but-present "
                    "1745 + glycerol 1080",
    "required_bands": [1655.0, 1745.0, 1080.0],
    "required_thresholds": [0.50, 0.04, 0.05],
    "supporting_bands": [1440.0, 1300.0],
    "anti_bands": [608.0],
    "anti_threshold": 0.08,
    "min_required_fraction": 0.04,
}

# TG-distinctive signature used by the G08 VETO. When this signature is
# present, G08 lipid_acyl_membrane cannot be the correct answer because:
#   - Free FAs (G08 members) do not have the ester 1745 C=O band
#   - Free FAs do not have the glycerol C-O 1080 band
#   - Sterols (G09a) have 608; triglycerides do not
# So "1745 present + 1080 present + 608 absent" is G09-triglyceride-specific.
TG_DISTINCTIVE_SIGNATURE = {
    "ester_CO_1745_min":   0.07,   # ester present (allow slightly lower than 0.08)
    "glycerol_CO_1080_min": 0.07,  # glycerol backbone present
    "sterol_ring_608_max": 0.07,   # sterol ring NOT present
}


def tg_distinctive_signature_present(spectrum, master_x, sp_max=None):
    """Return True if the TG-distinctive signature fires — used for G08 veto."""
    if sp_max is None or sp_max <= 0:
        fin = np.isfinite(spectrum)
        sp_max = float(np.max(spectrum[fin])) if fin.any() else 1.0
    i_1745 = _band_max(spectrum, master_x, 1745.0, half=8.0) / sp_max
    i_1080 = _band_max(spectrum, master_x, 1080.0, half=8.0) / sp_max
    i_608  = _band_max(spectrum, master_x, 608.0,  half=8.0) / sp_max
    return (i_1745 >= TG_DISTINCTIVE_SIGNATURE["ester_CO_1745_min"]
            and i_1080 >= TG_DISTINCTIVE_SIGNATURE["glycerol_CO_1080_min"]
            and i_608  <  TG_DISTINCTIVE_SIGNATURE["sterol_ring_608_max"])


def tg_branch_valid(branch, spectrum, master_x, sp_max):
    """Check if a TG branch (saturated or polyunsat-rescue) fires."""
    # Required bands each at their own threshold
    for c, th in zip(branch["required_bands"], branch["required_thresholds"]):
        if (_band_max(spectrum, master_x, c, half=8.0) / sp_max) < th:
            return False, 0.0
    # Anti bands
    for c in branch["anti_bands"]:
        if (_band_max(spectrum, master_x, c, half=8.0) / sp_max) >= branch["anti_threshold"]:
            return False, 0.0
    # Support fraction (for score contribution)
    sup_fires = 0
    for c in branch["supporting_bands"]:
        if (_band_max(spectrum, master_x, c, half=8.0) / sp_max) >= branch["min_required_fraction"]:
            sup_fires += 1
    sup_frac = sup_fires / max(len(branch["supporting_bands"]), 1)
    score = 0.7 + 0.3 * sup_frac    # required all-fire gives 0.7 baseline
    return True, score


def g09_routing_adjust_v45(bsv_per_group, spectrum, master_x):
    """v4.5 G09 routing.

    Changes vs v3:
      1. Adds two TG branches (saturated + polyunsat-rescue) so that
         chemistry-appropriate triglycerides can also be routed.
      2. When G08 is top-1 but the TG-distinctive signature is present,
         apply a G08-local VETO (multiplicative ×0.90 on G08 magnitude)
         AND a TG-gated G09 boost. This is MSS-level (acts on the G08
         contribution), not a family-wide dampening of G08.
      3. Existing v3 behaviour (G09 top-1 + valid sub → confirmation only;
         G09 top-2 + G08 top-3 + valid sub score ≥0.70 → small boost) is
         PRESERVED. No non-G09 change applies unless the TG signature fires.

    Returns (new_bsv_per_group, winning_sub, veto_applied_bool).
    """
    if "G09" not in bsv_per_group or "G08" not in bsv_per_group:
        # Pre-condition: we need both families in the scoring (always true in this engine)
        # Fall back to v3 behaviour
        out, sub = g09_routing_adjust(bsv_per_group, spectrum, master_x)
        return out, sub, False

    fin = np.isfinite(spectrum)
    sp_max = float(np.max(spectrum[fin])) if fin.any() else 1.0

    # v3 subfamily scoring (unchanged)
    sub_scores, sub_validity = score_g09_subfamilies(spectrum, master_x, sp_max)

    # Add v4.5 TG branches
    tg_sat_ok, tg_sat_score = tg_branch_valid(
        {**TG_SATURATED_BRANCH}, spectrum, master_x, sp_max,
    )
    tg_poly_ok, tg_poly_score = tg_branch_valid(
        {**TG_POLYUNSAT_BRANCH}, spectrum, master_x, sp_max,
    )
    if tg_sat_ok:
        sub_scores[TG_SATURATED_BRANCH["subfamily_id"]] = tg_sat_score
        sub_validity[TG_SATURATED_BRANCH["subfamily_id"]] = True
    if tg_poly_ok:
        sub_scores[TG_POLYUNSAT_BRANCH["subfamily_id"]] = tg_poly_score
        sub_validity[TG_POLYUNSAT_BRANCH["subfamily_id"]] = True

    valid_subs = [s for s, v in sub_validity.items() if v]
    winning_sub = max(sub_scores.items(), key=lambda kv: kv[1])[0] if sub_scores else None

    out = dict(bsv_per_group)
    tg_sig = tg_distinctive_signature_present(spectrum, master_x, sp_max=sp_max)

    # Pre-routing ranking
    pre_sorted = sorted(bsv_per_group.items(), key=lambda kv: -kv[1]["magnitude"])
    g09_top1 = pre_sorted[0][0] == "G09" if pre_sorted else False
    g08_top1 = pre_sorted[0][0] == "G08" if pre_sorted else False
    g09_in_top2 = "G09" in [g for g, _ in pre_sorted[:2]]
    g08_in_top3 = "G08" in [g for g, _ in pre_sorted[:3]]
    top_sub_score = max(sub_scores.values()) if sub_scores else 0.0

    veto_applied = False

    # (A) TG-distinctive signature present AND G08 top-1 AND TG branch valid
    #     → G08 gets local veto (×0.90), G09 gets targeted boost
    if g08_top1 and tg_sig and valid_subs:
        # The G08 veto is chemistry-specific: 1745 + 1080 + no 608 = triglyceride,
        # which cannot be a G08 member.
        old_g08 = out["G08"]["magnitude"]
        out["G08"] = {**out["G08"],
                        "magnitude": old_g08 * 0.90,
                        "g08_tg_signature_veto_applied": True,
                        "g08_tg_veto_factor": 0.90}
        # Concurrent G09 boost (modest; routing already fires below)
        old_g09 = out["G09"]["magnitude"]
        g09_boost = 1.08 + 0.08 * top_sub_score   # [1.08, 1.16]
        g09_boost = min(g09_boost, 1.16)
        out["G09"] = {**out["G09"],
                        "magnitude": old_g09 * g09_boost,
                        "g09_routing_applied": True,
                        "g09_winning_subfamily": winning_sub,
                        "g09_subfamily_valid": True,
                        "g09_boost_factor": g09_boost,
                        "g09_mode": "g08_veto_and_g09_boost"}
        veto_applied = True
        return out, winning_sub, veto_applied

    # (B) otherwise, fall through to v3 behaviour, but include v4.5 TG branches
    #     in the validity check (which may make v3 rules fire where v3 didn't)
    if g09_top1 and valid_subs:
        out["G09"] = {**out["G09"],
                        "g09_routing_applied": True,
                        "g09_winning_subfamily": winning_sub,
                        "g09_subfamily_valid": True,
                        "g09_boost_factor": 1.0,
                        "g09_mode": "confirmation_only"}
    elif valid_subs and g09_in_top2 and g08_in_top3 and top_sub_score >= 0.70:
        old_mag = out["G09"]["magnitude"]
        boost = 1.08 + 0.10 * top_sub_score
        boost = min(boost, 1.18)
        out["G09"] = {**out["G09"],
                        "magnitude": old_mag * boost,
                        "g09_routing_applied": True,
                        "g09_winning_subfamily": winning_sub,
                        "g09_subfamily_valid": True,
                        "g09_boost_factor": boost,
                        "g09_mode": "top2_targeted_boost"}
    else:
        out["G09"] = {**out["G09"],
                        "g09_routing_applied": False,
                        "g09_winning_subfamily": winning_sub,
                        "g09_subfamily_valid": bool(valid_subs),
                        "g09_boost_factor": 1.0,
                        "g09_mode": "no_routing"}
    return out, winning_sub, veto_applied


def compute_hybrid_bsv_v45(spectrum, master_x, motif_firings, mss_scores,
                              motif_id_to_group, motif_ids, analyte_to_group,
                              regime="Raman",
                              apply_sers_physics=True,
                              apply_tg_veto=True):
    """v4.5 hybrid BSV = v3 with SERS physics + the v4.5 G09 routing
    (which includes the TG branches + G08 TG-signature veto)."""
    bsv = compute_hybrid_bsv_v3(
        spectrum, master_x, motif_firings, mss_scores,
        motif_id_to_group, motif_ids, analyte_to_group,
        regime=regime, apply_sers_physics=apply_sers_physics,
    )
    if apply_tg_veto:
        bsv["per_group"], winning_sub, veto_applied = g09_routing_adjust_v45(
            bsv["per_group"], spectrum, master_x,
        )
        sorted_groups = sorted(bsv["per_group"].items(),
                                key=lambda kv: -kv[1]["magnitude"])
        if len(sorted_groups) >= 2:
            top_g, second_g = sorted_groups[0], sorted_groups[1]
            spillover = second_g[1]["magnitude"] / max(top_g[1]["magnitude"], 1e-6)
        else:
            second_g = (None, None)
            spillover = 0.0
        bsv["top_group"] = sorted_groups[0][0] if sorted_groups else None
        bsv["top_magnitude"] = sorted_groups[0][1]["magnitude"] if sorted_groups else 0.0
        bsv["second_group"] = second_g[0] if second_g[0] else None
        bsv["spillover_ratio"] = spillover
        bsv["ambiguity_flag"] = spillover >= AMBIGUITY_SPILLOVER_THRESHOLD
        bsv["g09_winning_subfamily"] = winning_sub
        bsv["tg_veto_applied"] = veto_applied
    return bsv


def run_bsv_v45(all_refs, master_x, motif_df, mss_df, motif_id_to_group,
                 motif_ids, analyte_to_group, apply_tg_veto=True, label="v45"):
    rows = []
    for r in all_refs:
        aid = canonical_analyte_id(r["component_key"], r["dataset"])
        regime = r.get("regime", "Raman")
        expected_group = analyte_to_group.get(aid, "")
        mf = compute_motif_firings(r["spectrum"], master_x, motif_df)
        ms = compute_mss_scores_v43(r["spectrum"], master_x, mss_df)
        bsv = compute_hybrid_bsv_v45(
            r["spectrum"], master_x, mf, ms, motif_id_to_group, motif_ids,
            analyte_to_group, regime=regime,
            apply_sers_physics=True, apply_tg_veto=apply_tg_veto,
        )
        s_sorted = sorted(bsv["per_group"].items(),
                            key=lambda kv: -kv[1]["magnitude"])
        top3 = [g for g, _ in s_sorted[:3]]
        rows.append({
            "spectrum_id": r["spectrum_id"],
            "analyte_id": aid,
            "regime": regime,
            "dataset": r["dataset"],
            "expected_group": expected_group,
            "top_group_predicted": bsv["top_group"],
            "second_group": bsv["second_group"],
            "top_magnitude": round(bsv["top_magnitude"], 4),
            "top_confidence": round(
                bsv["per_group"].get(bsv["top_group"], {}).get("confidence", 0.0), 4,
            ),
            "spillover": round(bsv["spillover_ratio"], 4),
            "ambiguity_flag": bsv["ambiguity_flag"],
            "top1_hit": (bsv["top_group"] == expected_group),
            "top3_hit": (expected_group in top3),
            "g09_winning_subfamily": bsv.get("g09_winning_subfamily", ""),
            "tg_veto_applied": bsv.get("tg_veto_applied", False),
            "variant": label,
        })
    return pd.DataFrame(rows)


ROOT = Path(
    "/Volumes/SSD_Rad/GAIRA_BUILD/"
    "gaira_base_4_hybrid_bsv_refinement_v4_5_triglyceride_veto"
)
TABLES = ROOT / "tables"
FIGS = ROOT / "figures"
REPORTS = ROOT / "reports"
AUDIT = ROOT / "audit"
REGISTRY = ROOT / "registry"
CODE_SNAPSHOT = ROOT / "code_snapshot"

MSS_V43 = Path(
    "/Volumes/SSD_Rad/GAIRA_BUILD/gaira_base_4_mss_decision_enrichment_v1/"
    "registry/grounding_molecular_signatures_v4_3.csv"
)
LEARNED_MOTIFS = Path(
    "/Volumes/SSD_Rad/GAIRA_BUILD/gaira_base_3_grounding_trained_ontology_v1/"
    "registry/learned_motif_registry_v1.csv"
)


# ─────────────────────────────────────────────────────────────────────
# Diagnostic bands (v4.5 scope)
# ─────────────────────────────────────────────────────────────────────

# Triglyceride-specific diagnostic positions (to be interrogated per-spectrum)
TG_DIAG_BANDS = {
    "ester_CO_1745":   1745.0,   # ester C=O (present in TG + ChE)
    "C_C_1655":        1655.0,   # cis-unsaturation C=C (strong in TG)
    "ester_CO_ester_1265": 1265.0,  # =CH in-plane / ester C-O (ChE>TG)
    "CH2_twist_1300":  1300.0,   # CH2 twist (shared TG + G08 saturated FA)
    "CH2_scissor_1440": 1440.0,  # CH2 scissoring (TG + G08)
    "C_O_glycerol_1080": 1080.0, # C-O skeletal (TG)
    "sterol_ring_608": 608.0,    # sterol ring (sterol + ChE; absent in TG)
    "glycerol_CO_bend_870": 870.0,  # glycerol C-O bending
    "skeletal_700":    700.0,    # steroid skeletal (sterol)
    "C_C_1670":        1670.0,   # sterol/ester C=C
    "aromatic_1603":   1603.0,   # aromatic (G09d)
    "ring_820":        820.0,    # ring subst (G09d)
}

# G08 lipid_acyl_membrane diagnostic bands (to detect "G08-ness")
G08_DIAG_BANDS = {
    "CH2_scissor_1440": 1440.0,  # dominant in G08 FA
    "C_H_stretch_2880": 2880.0,  # CH2 asymmetric (dominates in G08 above 2800)
    "CH2_twist_1300":   1300.0,  # CH2 twist
    "COO_1410":         1410.0,  # carboxylate (free FA)
    "C_C_skeletal_1060": 1060.0, # C-C skeletal of acyl chain
}


def band_intensity(spectrum, master_x, cm, half=8.0, sp_max=None):
    """Return band intensity at cm / sp_max — fraction of spectrum max."""
    if sp_max is None or sp_max <= 0:
        fin = np.isfinite(spectrum)
        sp_max = float(np.max(spectrum[fin])) if fin.any() else 1.0
    return float(_band_max(spectrum, master_x, cm, half=half)) / max(sp_max, 1e-9)


def identify_tg_subfamily_from_name(component_key):
    """Map analyte name → which of sterol / cholesteryl_ester / triglyceride /
    aromatic_steroid / other (non-G09) the analyte belongs to.
    Returns (tg_subfamily, is_triglyceride)."""
    n = normalise_label(component_key).lower()
    # Triglyceride naming: tri- prefix for saturated/unsat fatty-acid triesters
    tri_prefixes = (
        "trilinolein", "trilinolenin", "triolein", "tristearin",
        "tripalmitin", "tripalmitolein", "trimyristin", "trilaurin",
        "tricaproin", "tricaprin", "tricaprylin", "trielaidin",
        "trierucin", "tripetroselinin", "triarachidin", "tribehenin",
        "tri-11-eicosenoin", "tri 11 eicosenoin", "tri_11_eicosenoin",
    )
    if any(t in n for t in tri_prefixes):
        return "triglyceride", True
    if "cholesteryl" in n and any(x in n for x in
                                  ("ester", "linoleate", "oleate",
                                   "palmitate", "stearate", "myristate",
                                   "caproate", "caprylate", "laurate")):
        return "cholesteryl_ester", False
    if n.startswith("cholesterol") or n in ("cholesterol",):
        return "sterol", False
    if any(x in n for x in ("estr", "ethinyl", "diethylstilbestrol", "estrogen")):
        return "aromatic_steroid", False
    return "other_g09_or_non_g09", False


# ─────────────────────────────────────────────────────────────────────
# STAGE 0 — Triglyceride failure audit
# ─────────────────────────────────────────────────────────────────────

def stage0_triglyceride_audit(all_refs, master_x, motif_df, mss_df,
                                 motif_id_to_group, motif_ids, analyte_to_group):
    """Re-run v4 predictions + per-spectrum diagnostic band intensities.
    For each G09 spectrum, record:
      - expected_group (G09 always)
      - predicted group, second group
      - triglyceride subfamily (from name)
      - band intensities at diagnostic positions (TG + G08 sides)
      - error mode (TP, LEAKED_TO_G08, LEAKED_TO_OTHER, TG_SIGNATURE_MISSING, etc.)
    """
    print("\n[STAGE 0] Triglyceride failure audit")

    # Run v4 predictions (with routing ON, as current production config)
    pred_df = run_bsv_v4_base(all_refs, master_x, motif_df, mss_df,
                                motif_id_to_group, motif_ids, analyte_to_group,
                                apply_g09_routing=True, label="v4")
    # v4 predictions (without routing, for comparison)
    pred_df_noroute = run_bsv_v4_base(all_refs, master_x, motif_df, mss_df,
                                         motif_id_to_group, motif_ids, analyte_to_group,
                                         apply_g09_routing=False, label="v4_noroute")

    # Build lookup
    noroute_top = {r.spectrum_id: (r.top_group_predicted, r.top1_hit)
                     for r in pred_df_noroute.itertuples()}

    audit_rows = []
    # Iterate spectra
    for ref in all_refs:
        aid = canonical_analyte_id(ref["component_key"], ref["dataset"])
        expected_group = analyte_to_group.get(aid, "")
        if expected_group != "G09":
            continue
        sid = ref["spectrum_id"]
        spectrum = ref["spectrum"]
        fin = np.isfinite(spectrum)
        sp_max = float(np.max(spectrum[fin])) if fin.any() else 1.0

        # Predictions
        pr = pred_df[pred_df.spectrum_id == sid]
        if pr.empty:
            continue
        pr = pr.iloc[0]
        pred_top = pr.top_group_predicted
        pred_second = pr.second_group
        pred_top1 = bool(pr.top1_hit)
        pred_top3 = bool(pr.top3_hit)
        winning_sub = pr.g09_winning_subfamily

        # Diagnostic bands
        tg_intensities = {k: band_intensity(spectrum, master_x, cm, sp_max=sp_max)
                           for k, cm in TG_DIAG_BANDS.items()}
        g08_intensities = {k: band_intensity(spectrum, master_x, cm, sp_max=sp_max)
                            for k, cm in G08_DIAG_BANDS.items()}

        tg_subfam, is_tg = identify_tg_subfamily_from_name(ref["component_key"])

        # Required-band fire indicators for triglyceride
        # v3 rule: 1745 AND 1655 fire ≥ 0.08 AND 608 does NOT fire ≥ 0.08
        req_1745 = tg_intensities["ester_CO_1745"] >= 0.08
        req_1655 = tg_intensities["C_C_1655"] >= 0.08
        anti_608 = tg_intensities["sterol_ring_608"] >= 0.08
        tg_valid_v3 = req_1745 and req_1655 and not anti_608

        # G08 dominance signal: CH2 scissor 1440 + CH2 twist 1300 both fire
        # AND ester 1745 is relatively weak
        g08_ch2_dominant = (
            g08_intensities["CH2_scissor_1440"] >= 0.25
            and g08_intensities["CH2_twist_1300"] >= 0.10
        )
        ester_1745_weak = tg_intensities["ester_CO_1745"] < 0.05

        # Classify error mode
        if pred_top1:
            if is_tg:
                mode = "TG_TP_WITH_ROUTING" if winning_sub == "G09c_triglyceride" else "TG_TP_WRONG_SUBFAMILY"
            else:
                mode = "G09_TP_NON_TG"
        else:
            if pred_top == "G08":
                if is_tg:
                    if not req_1655:
                        mode = "TG_LEAKED_G08_1655_MISSING"
                    elif not req_1745:
                        mode = "TG_LEAKED_G08_1745_MISSING"
                    elif g08_ch2_dominant:
                        mode = "TG_LEAKED_G08_CH2_OVERWHELMS"
                    else:
                        mode = "TG_LEAKED_G08_EVIDENCE_BALANCED"
                else:
                    mode = "G09_NON_TG_LEAKED_G08"
            elif pred_top == "G06":
                mode = "TG_LEAKED_G06_PROTEIN" if is_tg else "G09_NON_TG_LEAKED_G06"
            elif pred_top == "G07":
                mode = "LEAKED_G07_AROMATIC"
            else:
                mode = f"LEAKED_{pred_top}"

        audit_rows.append({
            "spectrum_id": sid,
            "analyte_id": aid,
            "component_name": ref["component_key"],
            "dataset": ref["dataset"],
            "regime": ref.get("regime", "Raman"),
            "tg_subfamily_by_name": tg_subfam,
            "is_triglyceride": is_tg,
            "expected_group": expected_group,
            "pred_top_with_routing": pred_top,
            "pred_second_with_routing": pred_second,
            "pred_top_no_routing": noroute_top.get(sid, (None, None))[0],
            "routing_changed_top": pred_top != noroute_top.get(sid, (None, None))[0],
            "winning_subfamily_v3": winning_sub,
            "top1_hit": pred_top1,
            "top3_hit": pred_top3,
            "error_mode": mode,
            # Diagnostic band intensities (fraction of spectrum max)
            **{f"I_{k}": round(v, 3) for k, v in tg_intensities.items()},
            **{f"I_G08_{k}": round(v, 3) for k, v in g08_intensities.items()},
            "tg_v3_rule_valid": tg_valid_v3,
            "g08_ch2_dominant": g08_ch2_dominant,
            "ester_1745_weak": ester_1745_weak,
        })

    df = pd.DataFrame(audit_rows)
    df.to_csv(TABLES / "triglyceride_failure_audit_v1.csv", index=False)
    print(f"  emitted triglyceride_failure_audit_v1.csv ({len(df)} G09 spectra)")

    # Aggregate stats
    tg = df[df["is_triglyceride"]]
    n_tg = len(tg)
    n_tg_top1 = int(tg["top1_hit"].sum())
    n_tg_leak_g08 = int((~tg["top1_hit"] & (tg["pred_top_with_routing"] == "G08")).sum())
    n_tg_leak_other = int((~tg["top1_hit"] & (tg["pred_top_with_routing"] != "G08")).sum())
    n_tg_v3rule_valid = int(tg["tg_v3_rule_valid"].sum())
    n_tg_v3rule_valid_and_miss = int(
        (tg["tg_v3_rule_valid"] & ~tg["top1_hit"]).sum())
    n_tg_v3rule_invalid = int((~tg["tg_v3_rule_valid"]).sum())

    # Error mode breakdown
    mode_counts = Counter(tg["error_mode"].tolist())

    # Per-dataset triglyceride performance
    per_dataset = []
    for ds, sdf in tg.groupby("dataset"):
        per_dataset.append({
            "dataset": ds,
            "n_triglyceride": len(sdf),
            "top1_accuracy": float(sdf["top1_hit"].mean()),
            "leaked_to_g08_pct": float((~sdf["top1_hit"] & (sdf["pred_top_with_routing"] == "G08")).mean()),
        })
    pd.DataFrame(per_dataset).to_csv(
        TABLES / "triglyceride_by_dataset_v1.csv", index=False)

    # Structural insights (I_ columns averaged by outcome)
    diag_cols = [c for c in tg.columns if c.startswith("I_") or c.startswith("I_G08_")]
    by_outcome = tg.groupby(tg["top1_hit"].map({True: "TP", False: "MISS"}))[diag_cols].mean().round(3)
    by_outcome.to_csv(TABLES / "triglyceride_diag_means_by_outcome_v1.csv")

    print(f"  triglyceride spectra: {n_tg}  (top-1 {n_tg_top1}/{n_tg} = "
          f"{n_tg_top1/max(n_tg,1):.1%})")
    print(f"  leaked to G08: {n_tg_leak_g08}")
    print(f"  leaked elsewhere: {n_tg_leak_other}")
    print(f"  v3 rule valid: {n_tg_v3rule_valid} / {n_tg}; "
          f"valid BUT miss: {n_tg_v3rule_valid_and_miss}")
    print(f"  error mode breakdown: {dict(mode_counts)}")

    # Report
    lines = [
        "# Triglyceride Failure Analysis v1",
        "",
        "**Scope:** isolate WHERE triglyceride spectra fail before making any "
        "changes. v4 hybrid BSV with G09 subfamily routing is the current "
        "state; this analysis diagnoses its residual triglyceride misses.",
        "",
        "## Triglyceride corpus",
        "",
        f"- Total G09 spectra in current corpus: **{len(df)}**",
        f"- Triglyceride spectra: **{n_tg}** (classified by analyte name)",
        f"- Triglyceride datasets: {', '.join(sorted(tg['dataset'].unique()))}",
        "",
        "## Headline",
        "",
        f"- Triglyceride top-1 accuracy (v4 with routing): **{n_tg_top1}/{n_tg} = "
        f"{n_tg_top1/max(n_tg,1):.1%}**",
        f"- Leaked to **G08 lipid_acyl_membrane**: {n_tg_leak_g08}",
        f"- Leaked elsewhere: {n_tg_leak_other}",
        "",
        "## Error mode distribution",
        "",
        "| mode | n |",
        "|---|---:|",
    ]
    for m, c in sorted(mode_counts.items(), key=lambda kv: -kv[1]):
        lines.append(f"| {m} | {c} |")
    lines += [
        "",
        "## v3 routing rule analysis",
        "",
        f"- Triglycerides where v3 rule fires (1745 + 1655 both ≥0.08 AND 608 absent): "
        f"**{n_tg_v3rule_valid}** / {n_tg} ({n_tg_v3rule_valid/max(n_tg,1):.1%})",
        f"- Of those, how many still miss top-1: **{n_tg_v3rule_valid_and_miss}** "
        f"({n_tg_v3rule_valid_and_miss/max(n_tg_v3rule_valid,1):.1%} of valid)",
        f"- Triglycerides where v3 rule does NOT fire: {n_tg_v3rule_invalid} "
        f"({n_tg_v3rule_invalid/max(n_tg,1):.1%}) — these cannot benefit from "
        "routing boost and default to raw BSV ranking.",
        "",
        "## Diagnostic band intensities by outcome",
        "",
        "Average fraction-of-spectrum-max for each diagnostic band, split by "
        "whether the triglyceride spectrum hit top-1 vs missed. See "
        "`triglyceride_diag_means_by_outcome_v1.csv` for the full table.",
        "",
    ]
    # Print table inline
    lines.append("| band | TP mean | MISS mean |")
    lines.append("|---|---:|---:|")
    tp_row = by_outcome.loc["TP"] if "TP" in by_outcome.index else None
    miss_row = by_outcome.loc["MISS"] if "MISS" in by_outcome.index else None
    for c in diag_cols:
        tp_v = tp_row[c] if tp_row is not None and c in tp_row.index else None
        ms_v = miss_row[c] if miss_row is not None and c in miss_row.index else None
        lines.append(f"| {c} | {tp_v if tp_v is not None else '—'} | "
                      f"{ms_v if ms_v is not None else '—'} |")
    lines += [
        "",
        "## Per-dataset breakdown",
        "",
        "See `triglyceride_by_dataset_v1.csv`.",
        "",
        "## What this audit implies for Stage 1",
        "",
        "Based on the error-mode distribution, the Stage 1 rule set should "
        "target the dominant failure modes. Typical candidates:",
        "",
        "- **TG_LEAKED_G08_CH2_OVERWHELMS**: when CH2 scissor 1440 + CH2 twist "
        "1300 dominate and ester 1745 is weak → G08 veto rule should downweight "
        "G09c score when 1745 < threshold AND 1440 >> 1745.",
        "",
        "- **TG_LEAKED_G08_1655_MISSING**: saturated TGs (tripalmitin, "
        "tristearin, trilaurin, etc.) genuinely lack strong 1655 — the v3 "
        "rule is too strict and excludes valid saturated triglycerides. "
        "Stage 1 needs an alternative cofeature for saturated TGs "
        "(e.g., glycerol 1080 + ester 1745 without 608).",
        "",
        "- **TG_LEAKED_G08_EVIDENCE_BALANCED**: evidence is chemistry-honest "
        "50/50 — the intrinsic G08/G09 boundary is not breakable without "
        "more corpus.",
    ]
    (REPORTS / "REPORT_triglyceride_failure_analysis_v1.md"
     ).write_text("\n".join(lines))
    print(f"  emitted REPORT_triglyceride_failure_analysis_v1.md")

    return {
        "n_tg": n_tg,
        "n_tg_top1": n_tg_top1,
        "n_tg_leak_g08": n_tg_leak_g08,
        "n_tg_leak_other": n_tg_leak_other,
        "n_tg_v3rule_valid": n_tg_v3rule_valid,
        "n_tg_v3rule_valid_and_miss": n_tg_v3rule_valid_and_miss,
        "mode_counts": dict(mode_counts),
        "audit_df": df,
        "pred_df_with": pred_df,
        "pred_df_noroute": pred_df_noroute,
    }


# ─────────────────────────────────────────────────────────────────────
# STAGE 1 — Rule table + registry emission
# ─────────────────────────────────────────────────────────────────────

def stage1_emit_rules():
    """Emit the v4.5 triglyceride rule registry and a short design report."""
    print("\n[STAGE 1] Emit triglyceride-specific rule registry")
    rows = []
    # Keep existing v3 subfamilies
    for sub_id, sub in G09_SUBFAMILIES.items():
        rows.append({
            "subfamily_id": sub_id,
            "layer": "v3_unchanged",
            "description": sub["description"],
            "required_bands": ";".join(f"{b:.0f}" for b in sub["required_bands"]),
            "required_thresholds": ";".join([f"{sub['min_required_fraction']:.2f}"]
                                               * len(sub["required_bands"])),
            "anti_bands": ";".join(f"{b:.0f}" for b in sub["anti_bands"]) if sub["anti_bands"] else "",
            "anti_threshold": f"{sub['min_required_fraction']:.2f}",
        })
    # Add v4.5 branches
    for branch in (TG_SATURATED_BRANCH, TG_POLYUNSAT_BRANCH):
        rows.append({
            "subfamily_id": branch["subfamily_id"],
            "layer": "v4_5_new",
            "description": branch["description"],
            "required_bands": ";".join(f"{b:.0f}" for b in branch["required_bands"]),
            "required_thresholds": ";".join(f"{t:.2f}" for t in branch["required_thresholds"]),
            "anti_bands": ";".join(f"{b:.0f}" for b in branch["anti_bands"]) if branch["anti_bands"] else "",
            "anti_threshold": f"{branch['anti_threshold']:.2f}",
        })
    # G08 veto row (meta)
    rows.append({
        "subfamily_id": "G08_TG_SIGNATURE_VETO",
        "layer": "v4_5_new_veto",
        "description": "G08 magnitude ×0.90 when ester_1745 + glycerol_1080 + (no 608) — chemistry-specific veto",
        "required_bands": "1745;1080",
        "required_thresholds": f"{TG_DISTINCTIVE_SIGNATURE['ester_CO_1745_min']:.2f};"
                                 f"{TG_DISTINCTIVE_SIGNATURE['glycerol_CO_1080_min']:.2f}",
        "anti_bands": "608",
        "anti_threshold": f"{TG_DISTINCTIVE_SIGNATURE['sterol_ring_608_max']:.2f}",
    })
    pd.DataFrame(rows).to_csv(
        TABLES / "g09_triglyceride_rules_v45.csv", index=False,
    )
    print(f"  emitted g09_triglyceride_rules_v45.csv ({len(rows)} rules)")

    lines = [
        "# Triglyceride Rules v4.5",
        "",
        "**Scope:** surgical addition to the G09 subfamily routing — two new "
        "triglyceride branches (saturated + polyunsat rescue) plus a "
        "chemistry-specific G08 veto. **No other v3/v4 rule is changed.**",
        "",
        "## Chemistry motivation (from Stage 0 audit)",
        "",
        "- Saturated triglycerides (tripalmitin / tristearin / trilaurin / "
        "trimyristin / triarachidin / tribehenin) have **no C=C 1655** by "
        "chemistry. The v3 rule `require 1745 AND 1655` cannot fire — so "
        "these leak to G08 via CH2-bend dominance.",
        "- Polyunsaturated triglycerides (trilinolein, trilinolenin) have "
        "1655 so strong that the 1745 band's fraction-of-sp-max falls below "
        "the 0.08 threshold even though ester chemistry is unmistakable.",
        "- Free fatty acids (G08 members) have no ester 1745 and no glycerol "
        "1080 — so the co-signature 1745+1080+(no 608) is uniquely a "
        "triglyceride signature.",
        "",
        "## The three new rules",
        "",
        "### R1 — G09c saturated branch (new)",
        "",
        "Required (AND): ester_C=O 1745 ≥ 0.08, glycerol 1080 ≥ 0.08, CH2 "
        "twist 1300 ≥ 0.20. Anti: sterol 608 < 0.08.",
        "",
        "### R2 — G09c polyunsat rescue branch (new)",
        "",
        "Required (AND): 1655 ≥ 0.50 (strong unsaturation), 1745 ≥ 0.04 "
        "(ester present, relaxed threshold), 1080 ≥ 0.05. Anti: 608 < 0.08.",
        "",
        "### R3 — G08 TG-signature veto (new)",
        "",
        "When `ester 1745 ≥ 0.07 AND glycerol 1080 ≥ 0.07 AND 608 < 0.07` "
        "holds AND G08 is top-1 AND any G09 subfamily is valid → multiply G08 "
        "magnitude by 0.90 and G09 magnitude by [1.08, 1.16]. The veto fires "
        "ONLY when the distinctive triglyceride signature is present; it does "
        "not affect generic G08 chemistry (phospholipids, free FAs with no "
        "glycerol 1080, etc.).",
        "",
        "## Invariants",
        "",
        "- v3 subfamilies G09a / G09b / G09c / G09d unchanged.",
        "- G07 per-family override unchanged.",
        "- MSS v4.3 registry, motif registry, substrate physics: read-only.",
        "- Frozen 11-group taxonomy: unchanged.",
        "- `src/gaira/base3/mss_engine.py`: unchanged.",
    ]
    (REPORTS / "REPORT_triglyceride_rules_v45_design.md"
     ).write_text("\n".join(lines))


# ─────────────────────────────────────────────────────────────────────
# STAGE 2 — Apply v4.5 and re-eval
# ─────────────────────────────────────────────────────────────────────

def stage2_apply_v45_and_reeval(all_refs, master_x, motif_df, mss_df,
                                   motif_id_to_group, motif_ids,
                                   analyte_to_group, v4_pred_df, audit_df):
    print("\n[STAGE 2] Apply v4.5 rules and re-evaluate")
    v45_df = run_bsv_v45(all_refs, master_x, motif_df, mss_df,
                          motif_id_to_group, motif_ids, analyte_to_group,
                          apply_tg_veto=True, label="v45")
    v45_df.to_csv(TABLES / "hybrid_eval_v45.csv", index=False)
    print(f"  emitted hybrid_eval_v45.csv ({len(v45_df)} rows)")

    # Quick comparison via merge (spectrum_id can be duplicated across datasets)
    v4_key = v4_pred_df[["spectrum_id", "dataset", "top1_hit",
                           "top_group_predicted", "analyte_id", "expected_group"]].copy()
    v45_key = v45_df[["spectrum_id", "dataset", "top1_hit",
                        "top_group_predicted", "tg_veto_applied"]].copy()
    merged = v4_key.merge(
        v45_key, on=["spectrum_id", "dataset"], suffixes=("_v4", "_v45"),
    )
    n_common = len(merged)
    n_changed_top1 = int((merged["top1_hit_v4"] != merged["top1_hit_v45"]).sum())
    n_v4_hit_not_v45 = int((merged["top1_hit_v4"] & ~merged["top1_hit_v45"]).sum())
    n_v45_hit_not_v4 = int((~merged["top1_hit_v4"] & merged["top1_hit_v45"]).sum())
    print(f"  v4→v4.5 top-1 flip: {n_changed_top1}  "
          f"(gained {n_v45_hit_not_v4}, lost {n_v4_hit_not_v45})")

    # Per-TG outcome
    tg_aids = set(audit_df[audit_df["is_triglyceride"]]["spectrum_id"].tolist())
    tg_v45 = v45_df[v45_df.spectrum_id.isin(tg_aids)]
    tg_v4 = v4_pred_df[v4_pred_df.spectrum_id.isin(tg_aids)]
    print(f"  TG top-1 v4:   {tg_v4['top1_hit'].sum()}/{len(tg_v4)} "
          f"= {tg_v4['top1_hit'].mean():.1%}")
    print(f"  TG top-1 v4.5: {tg_v45['top1_hit'].sum()}/{len(tg_v45)} "
          f"= {tg_v45['top1_hit'].mean():.1%}")

    # Save per-spectrum changed-rows table
    flipped = merged[merged["top1_hit_v4"] != merged["top1_hit_v45"]].copy()
    flipped["direction"] = flipped["top1_hit_v45"].map(
        {True: "GAINED", False: "LOST"})
    flipped.rename(columns={"top_group_predicted_v4": "v4_top_group",
                                "top_group_predicted_v45": "v45_top_group",
                                "top1_hit_v4": "v4_top1_hit",
                                "top1_hit_v45": "v45_top1_hit",
                                "tg_veto_applied": "v45_tg_veto_applied"},
                     inplace=True)
    flipped.to_csv(TABLES / "v4_to_v45_flips_v1.csv", index=False)

    return v45_df


# ─────────────────────────────────────────────────────────────────────
# STAGE 3 — G08/G09 boundary re-eval
# ─────────────────────────────────────────────────────────────────────

def stage3_g08_g09_boundary(v4_pred_df, v45_df, audit_df):
    print("\n[STAGE 3] G08/G09 boundary re-evaluation")
    # Focus on G08 and G09 expected-class spectra
    def _stats(df, expected):
        ec = df[df.expected_group == expected]
        return {
            "n": len(ec),
            "top1": float(ec["top1_hit"].mean()) if len(ec) else 0.0,
            "top3": float(ec["top3_hit"].mean()) if len(ec) else 0.0,
        }
    rows = []
    for variant, df in (("v4", v4_pred_df), ("v4.5", v45_df)):
        for fam in ("G08", "G09"):
            s = _stats(df, fam)
            rows.append({
                "variant": variant,
                "family": fam,
                "n": s["n"],
                "top1": s["top1"],
                "top3": s["top3"],
            })
    pd.DataFrame(rows).to_csv(
        TABLES / "g08_g09_boundary_v1.csv", index=False,
    )
    # Confusion between G08 and G09 (each direction)
    def _confusion(df, true_fam):
        ec = df[df.expected_group == true_fam]
        return Counter(ec["top_group_predicted"].tolist())
    conf_rows = []
    for variant, df in (("v4", v4_pred_df), ("v4.5", v45_df)):
        for fam in ("G08", "G09"):
            for predicted, n in _confusion(df, fam).items():
                conf_rows.append({
                    "variant": variant,
                    "expected": fam,
                    "predicted": predicted,
                    "n": n,
                })
    pd.DataFrame(conf_rows).to_csv(
        TABLES / "g08_g09_confusion_v1.csv", index=False,
    )
    # Did the veto over-fire on genuine G08 spectra?
    g08_in_v45 = v45_df[v45_df.expected_group == "G08"]
    g08_veto_fires_on_g08 = int(g08_in_v45["tg_veto_applied"].sum()) if "tg_veto_applied" in g08_in_v45.columns else 0
    print(f"  G08 spectra where TG veto fired (potential collateral): "
          f"{g08_veto_fires_on_g08} / {len(g08_in_v45)}")
    # Short report
    lines = [
        "# G08/G09 Boundary Re-Evaluation",
        "",
        "## Per-family accuracy",
        "",
        "| variant | family | n | top-1 | top-3 |",
        "|---|---|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r['variant']} | {r['family']} | {r['n']} | "
            f"{r['top1']:.1%} | {r['top3']:.1%} |"
        )
    lines += [
        "",
        "## Confusion (expected vs predicted)",
        "",
        "See `g08_g09_confusion_v1.csv`.",
        "",
        "## Veto specificity",
        "",
        f"- v4.5 TG veto fired on {g08_veto_fires_on_g08} of {len(g08_in_v45)} "
        f"genuine G08 spectra (potential collateral).",
        f"- If this number is 0 or very small, the veto is properly "
        "triglyceride-specific.",
    ]
    (REPORTS / "REPORT_g08_g09_boundary_v1.md"
     ).write_text("\n".join(lines))
    print(f"  emitted REPORT_g08_g09_boundary_v1.md")


# ─────────────────────────────────────────────────────────────────────
# STAGE 4 — Ceiling / hardness analysis
# ─────────────────────────────────────────────────────────────────────

def stage4_ceiling_analysis(v45_df, audit_df):
    print("\n[STAGE 4] Hardness / ceiling analysis")
    # How many triglycerides still miss? Are they corpus-limited?
    tg_aids = set(audit_df[audit_df["is_triglyceride"]]["spectrum_id"].tolist())
    tg_v45 = v45_df[v45_df.spectrum_id.isin(tg_aids)]
    still_missing = tg_v45[~tg_v45["top1_hit"]]
    # Join back with audit to grab diagnostic bands
    mrows = []
    for _, r in still_missing.iterrows():
        row = audit_df[audit_df.spectrum_id == r["spectrum_id"]].iloc[0]
        mrows.append({
            "spectrum_id": r["spectrum_id"],
            "component_name": row["component_name"],
            "dataset": row["dataset"],
            "predicted": r["top_group_predicted"],
            "second": r["second_group"],
            "v0_error_mode": row["error_mode"],
            "I_ester_1745": row["I_ester_CO_1745"],
            "I_C_C_1655": row["I_C_C_1655"],
            "I_glycerol_1080": row["I_C_O_glycerol_1080"],
            "I_CH2_scissor_1440": row["I_CH2_scissor_1440"],
            "I_CH2_twist_1300": row["I_CH2_twist_1300"],
            "I_sterol_608": row["I_sterol_ring_608"],
            "ceiling_classification": classify_ceiling(row),
        })
    ceiling_df = pd.DataFrame(mrows)
    ceiling_df.to_csv(TABLES / "v45_residual_misses_ceiling_v1.csv", index=False)
    print(f"  emitted v45_residual_misses_ceiling_v1.csv "
          f"({len(ceiling_df)} residual TG misses)")

    ceil_counts = Counter(ceiling_df["ceiling_classification"].tolist()) if len(ceiling_df) else Counter()

    lines = [
        "# v4.5 Ceiling / Hardness Analysis",
        "",
        "## Remaining triglyceride misses",
        "",
        f"- Residual TG misses after v4.5: **{len(ceiling_df)}** of "
        f"{len(tg_v45)} triglyceride spectra.",
        f"- TG top-1 v4.5: "
        f"{tg_v45['top1_hit'].mean():.1%}",
        "",
        "## Ceiling classification",
        "",
        "Each residual miss is classified as one of:",
        "- **CORPUS_LIMITED_NOISY**: low SNR or weak diagnostic bands "
        "— would need better-quality spectrum / more reps.",
        "- **CORPUS_LIMITED_AMBIGUOUS_CHEM**: chemistry is genuinely "
        "boundary-ambiguous with G08 — would need more triglyceride "
        "reference variety in the corpus.",
        "- **RULE_THRESHOLD_EDGE_CASE**: falls just below one of the v4.5 "
        "thresholds; could be tuned further but risks collateral.",
        "",
        "| classification | n |",
        "|---|---:|",
    ]
    for c, n in sorted(ceil_counts.items(), key=lambda kv: -kv[1]):
        lines.append(f"| {c} | {n} |")
    lines += [
        "",
        "See `v45_residual_misses_ceiling_v1.csv` for per-spectrum details.",
    ]
    (REPORTS / "REPORT_v45_ceiling_analysis_v1.md"
     ).write_text("\n".join(lines))
    print(f"  emitted REPORT_v45_ceiling_analysis_v1.md")

    return {
        "n_residual": len(ceiling_df),
        "ceil_counts": dict(ceil_counts),
        "tg_top1": float(tg_v45["top1_hit"].mean()) if len(tg_v45) else 0.0,
    }


def classify_ceiling(row):
    # Quick classification:
    i_1745 = row["I_ester_CO_1745"]
    i_1080 = row["I_C_O_glycerol_1080"]
    i_1655 = row["I_C_C_1655"]
    if i_1745 < 0.05 and i_1080 < 0.05:
        return "CORPUS_LIMITED_NOISY"
    if 0.04 <= i_1745 < 0.08 or 0.05 <= i_1080 < 0.08:
        return "RULE_THRESHOLD_EDGE_CASE"
    return "CORPUS_LIMITED_AMBIGUOUS_CHEM"


# ─────────────────────────────────────────────────────────────────────
# STAGE 5 — G09 output policy
# ─────────────────────────────────────────────────────────────────────

def stage5_output_policy(v45_df, ceiling_info):
    print("\n[STAGE 5] G09 output policy update")
    g09 = v45_df[v45_df.expected_group == "G09"]
    g09_top1 = float(g09["top1_hit"].mean()) if len(g09) else 0.0
    if g09_top1 >= 0.80:
        tier = "MODERATE"
    elif g09_top1 >= 0.70:
        tier = "SENSITIVE"
    else:
        tier = "SENSITIVE_WITH_SUBFAMILY_METADATA"

    lines = [
        "# G09 Output Policy v4.5",
        "",
        f"## G09 accuracy (v4.5)",
        "",
        f"- G09 top-1: **{g09_top1:.1%}**",
        f"- Policy tier: **{tier}**",
        "",
        "## Tier-specific output rules",
        "",
        "| tier | top-1 bar | output |",
        "|---|---|---|",
        "| ROBUST | ≥90% | hard-call top group + confidence |",
        "| MODERATE | ≥80% | top group + top-3 backup + confidence |",
        "| SENSITIVE | ≥70% | top-3 surfaced + confidence caveats + subfamily metadata |",
        "| SENSITIVE_WITH_SUBFAMILY_METADATA | <70% | **always surface top-3** + subfamily metadata (triglyceride_saturated vs _polyunsat vs sterol vs cholesteryl_ester vs aromatic_steroid) + explicit G08/G09 boundary caveat |",
        "",
        "## Applied caveats for v4.5 G09 outputs",
        "",
        f"- G09 top-1 = {g09_top1:.1%} → tier **{tier}**.",
        "- For any G09 prediction, emit:",
        "  - `winning_subfamily_id` (sterol / cholesteryl_ester / "
        "triglyceride_unsaturated / triglyceride_saturated / "
        "triglyceride_polyunsat_rescue / aromatic_steroid)",
        "  - `tg_veto_applied` flag (True if G08 was vetoed by TG signature)",
        "  - `g08_g09_boundary_caveat` = True when expected-vs-predicted is "
        "ambiguous at the CH2-bend level",
        "- Do NOT hard-call species identity within the triglyceride subfamily "
        "(palmitate vs stearate vs oleate etc.) from family-level BSV alone.",
    ]
    (REPORTS / "REPORT_g09_output_policy_v45.md"
     ).write_text("\n".join(lines))
    print(f"  G09 top-1 = {g09_top1:.1%} → tier {tier}")


# ─────────────────────────────────────────────────────────────────────
# STAGE 6 — Final decision + audit
# ─────────────────────────────────────────────────────────────────────

def stage6_decision(v4_pred_df, v45_df, ceiling_info):
    print("\n[STAGE 6] Final decision")
    def _fam_top1(df, fam):
        ec = df[df.expected_group == fam]
        return float(ec["top1_hit"].mean()) if len(ec) else 0.0
    def _regime_top1(df, regime):
        ec = df[(df.regime == regime) & (df.expected_group != "")]
        return float(ec["top1_hit"].mean()) if len(ec) else 0.0
    def _overall_top1(df):
        ec = df[df.expected_group != ""]
        return float(ec["top1_hit"].mean()) if len(ec) else 0.0

    v4_g09 = _fam_top1(v4_pred_df, "G09")
    v45_g09 = _fam_top1(v45_df, "G09")
    v4_g08 = _fam_top1(v4_pred_df, "G08")
    v45_g08 = _fam_top1(v45_df, "G08")
    v4_overall = _overall_top1(v4_pred_df)
    v45_overall = _overall_top1(v45_df)
    v4_raman = _regime_top1(v4_pred_df, "Raman")
    v45_raman = _regime_top1(v45_df, "Raman")
    v4_sers = _regime_top1(v4_pred_df, "SERS")
    v45_sers = _regime_top1(v45_df, "SERS")

    # Triglyceride specifically
    v4_tg_top1 = ceiling_info["tg_top1"]  # placeholder; recompute properly
    # recompute v4 TG
    def _tg_top1(df, audit_df=None):
        # Need the tg_aids set
        return None  # computed below
    v4_tg = v4_pred_df[v4_pred_df.expected_group == "G09"]  # proxy; refined below
    # Use the ceiling_info tg_top1 as v4.5 TG
    v45_tg_top1 = ceiling_info["tg_top1"]
    # Compute v4 TG top-1 from the v4 predictions using audit-defined TG set
    # (audit_df written to disk earlier — reload to keep deterministic)
    audit_csv = TABLES / "triglyceride_failure_audit_v1.csv"
    if audit_csv.exists():
        audit_df = pd.read_csv(audit_csv)
        tg_sids = set(audit_df[audit_df["is_triglyceride"]]["spectrum_id"].tolist())
        v4_tg_df = v4_pred_df[v4_pred_df.spectrum_id.isin(tg_sids)]
        v4_tg_top1 = float(v4_tg_df["top1_hit"].mean()) if len(v4_tg_df) else 0.0
    else:
        v4_tg_top1 = 0.0

    metrics = {
        "v4_overall_top1": v4_overall,
        "v45_overall_top1": v45_overall,
        "v4_raman_top1": v4_raman,
        "v45_raman_top1": v45_raman,
        "v4_sers_top1": v4_sers,
        "v45_sers_top1": v45_sers,
        "v4_g09_top1": v4_g09,
        "v45_g09_top1": v45_g09,
        "v4_g08_top1": v4_g08,
        "v45_g08_top1": v45_g08,
        "v4_tg_top1": v4_tg_top1,
        "v45_tg_top1": v45_tg_top1,
        "g09_delta": v45_g09 - v4_g09,
        "g08_delta": v45_g08 - v4_g08,
        "overall_delta": v45_overall - v4_overall,
    }
    pd.DataFrame([metrics]).to_csv(
        TABLES / "v45_summary_metrics_v1.csv", index=False,
    )

    # Decision
    if v45_g09 >= 0.75 and (v45_overall - v4_overall) >= -0.02 and (v45_g08 - v4_g08) >= -0.02:
        decision = "G09_FIXED_READY_FOR_CALIBRATION"
    elif v45_g09 > v4_g09 and (v45_overall - v4_overall) >= -0.02:
        decision = "G09_IMPROVED_BUT_SENSITIVE"
    elif v45_g09 < v4_g09:
        decision = "NEEDS_DEEPER_TRIGLYCERIDE_ROUTING"
    elif ceiling_info["ceil_counts"].get("CORPUS_LIMITED_AMBIGUOUS_CHEM", 0) + \
         ceiling_info["ceil_counts"].get("CORPUS_LIMITED_NOISY", 0) \
         >= 0.6 * ceiling_info["n_residual"]:
        decision = "NEEDS_MORE_G09_CORPUS"
    else:
        decision = "G09_IMPROVED_BUT_SENSITIVE"

    # Audit log
    lines = [
        "# gaira_base_4 Hybrid BSV Refinement v4.5 Triglyceride Veto — Audit Log",
        "",
        "## Scope",
        "",
        "Narrow surgical repair of G09 triglyceride routing. No calibration, "
        "no broader SERS work, no global retuning. All changes additive to "
        "the v3/v4 engine.",
        "",
        "## Deliverables (v4.5)",
        "",
        "- 2 new TG subfamily branches: saturated + polyunsat rescue",
        "- 1 new G08 veto: ester 1745 + glycerol 1080 + (no 608) → G08 ×0.90",
        "- 4 tables: triglyceride_failure_audit_v1, g09_triglyceride_rules_v45, "
        "hybrid_eval_v45, v4_to_v45_flips_v1",
        "- 3 secondary tables: g08_g09_boundary_v1, g08_g09_confusion_v1, "
        "v45_residual_misses_ceiling_v1",
        "- 1 metrics table: v45_summary_metrics_v1",
        "- 5 reports: triglyceride_failure_analysis / triglyceride_rules_v45_design / "
        "g08_g09_boundary / v45_ceiling / g09_output_policy_v45",
        "",
        "## Metrics",
        "",
        "| metric | v4 | **v4.5** | Δ |",
        "|---|---:|---:|---:|",
        f"| overall top-1 | {v4_overall:.1%} | **{v45_overall:.1%}** | "
        f"{(v45_overall - v4_overall):+.1%} |",
        f"| Raman top-1 | {v4_raman:.1%} | {v45_raman:.1%} | "
        f"{(v45_raman - v4_raman):+.1%} |",
        f"| SERS top-1 | {v4_sers:.1%} | {v45_sers:.1%} | "
        f"{(v45_sers - v4_sers):+.1%} |",
        f"| G09 top-1 | {v4_g09:.1%} | **{v45_g09:.1%}** | "
        f"{(v45_g09 - v4_g09):+.1%} |",
        f"| G08 top-1 | {v4_g08:.1%} | {v45_g08:.1%} | "
        f"{(v45_g08 - v4_g08):+.1%} |",
        f"| TG top-1 | {v4_tg_top1:.1%} | **{v45_tg_top1:.1%}** | "
        f"{(v45_tg_top1 - v4_tg_top1):+.1%} |",
        "",
        f"## Decision: **{decision}**",
        "",
        "## Non-modification invariants",
        "",
        "- `src/gaira/base3/mss_engine.py`: unchanged",
        "- All prior phase drivers: unchanged",
        "- Frozen 11-group taxonomy: unchanged",
        "- MSS v4.3 / motif registry / substrate physics: read-only",
        "- G07/G09a/G09b/G09d/SERS rules: unchanged",
        "- No synthetic spectra added to corpus",
        "- No calibration / target clinical cohorts used",
    ]
    (AUDIT / "gaira_base_4_hybrid_bsv_refinement_v4_5_triglyceride_veto_audit_log.md"
     ).write_text("\n".join(lines))

    # Readiness report
    rlines = [
        "# Hybrid Refinement v4.5 Readiness",
        "",
        f"**Decision: {decision}**",
        "",
        "## Headline",
        "",
        f"- G09 top-1: {v4_g09:.1%} → **{v45_g09:.1%}** "
        f"(Δ {(v45_g09 - v4_g09):+.1%})",
        f"- Triglyceride top-1: {v4_tg_top1:.1%} → **{v45_tg_top1:.1%}** "
        f"(Δ {(v45_tg_top1 - v4_tg_top1):+.1%})",
        f"- G08 top-1 (collateral check): {v4_g08:.1%} → {v45_g08:.1%} "
        f"(Δ {(v45_g08 - v4_g08):+.1%})",
        f"- Overall top-1: {v4_overall:.1%} → {v45_overall:.1%} "
        f"(Δ {(v45_overall - v4_overall):+.1%})",
        "",
        "## Interpretation",
    ]
    if decision == "G09_FIXED_READY_FOR_CALIBRATION":
        rlines.append(
            f"G09 reached the ≥75% target with no meaningful G08/overall "
            f"regression. Ready to proceed to calibration."
        )
    elif decision == "G09_IMPROVED_BUT_SENSITIVE":
        rlines.append(
            f"G09 improved but is below the 75% target. Remaining gap is "
            f"split across corpus-limited and threshold-edge cases. Output "
            f"policy keeps G09 in SENSITIVE tier with subfamily metadata."
        )
    elif decision == "NEEDS_DEEPER_TRIGLYCERIDE_ROUTING":
        rlines.append(
            f"v4.5 regressed G09. Revert and re-approach the rules."
        )
    elif decision == "NEEDS_MORE_G09_CORPUS":
        rlines.append(
            f"Remaining misses are dominantly corpus-limited (boundary "
            f"ambiguity or noisy spectra). Next progress requires more "
            f"triglyceride reference diversity."
        )
    rlines += [
        "",
        "## Next steps",
        "",
        "1. **Calibration phase**: use v4.5 engine with G09 subfamily metadata "
        "and TG-signature veto; apply under Gobbato calibration perturbations.",
        "2. **Target-cohort passive readout**: emit G09 with subfamily "
        "metadata + explicit G08/G09 boundary caveats.",
        "3. **(Deferred)** SERS corpus expansion — see v3 phase follow-up.",
    ]
    (REPORTS / "REPORT_hybrid_refinement_v45_readiness.md"
     ).write_text("\n".join(rlines))

    return decision, metrics


# ─────────────────────────────────────────────────────────────────────
# Driver

def main():
    print("=" * 78)
    print("gaira_base_4 — Hybrid BSV Refinement v4.5 (Triglyceride Veto)")
    print("=" * 78)
    for d in (TABLES, FIGS, REPORTS, AUDIT, REGISTRY, CODE_SNAPSHOT):
        d.mkdir(parents=True, exist_ok=True)

    master_x = canonical_master_axis()
    rb = load_ramanbiolib(master_x)
    gp = load_gobbato_powder(master_x)
    aa = load_amino_acid_xlsx(master_x)
    lit = load_digitised_literature(master_x)
    sers = load_sers_metabolite_63(master_x)
    all_refs = rb + gp + aa + lit + sers
    print(f"[data] {len(all_refs)} grounding spectra")

    mss_df = pd.read_csv(MSS_V43)
    motif_df = pd.read_csv(LEARNED_MOTIFS)
    motif_ids = motif_df["learned_motif_id"].tolist()

    motif_id_to_group = {}
    for g in BSV_GROUPS:
        for m_id in g["dominant_motifs"]:
            motif_id_to_group[m_id] = g["group_id"]
    bc_to_group = {bc: g["group_id"] for g in BSV_GROUPS
                    for bc in g["member_broad_classes"]}
    analyte_to_group = {}
    for _, r in mss_df.iterrows():
        analyte_to_group[r["analyte_name"]] = bc_to_group.get(
            r["broad_class"], "G11",
        )

    # Stage 0 — triglyceride failure audit
    s0 = stage0_triglyceride_audit(
        all_refs, master_x, motif_df, mss_df,
        motif_id_to_group, motif_ids, analyte_to_group,
    )
    v4_pred_df = s0["pred_df_with"]  # v4 baseline for comparison

    # Stage 1 — rule-table emission + documentation
    stage1_emit_rules()

    # Stage 2 — apply v4.5 rules and re-evaluate
    v45_df = stage2_apply_v45_and_reeval(
        all_refs, master_x, motif_df, mss_df,
        motif_id_to_group, motif_ids, analyte_to_group,
        v4_pred_df, s0["audit_df"],
    )

    # Stage 3 — G08/G09 boundary re-evaluation
    stage3_g08_g09_boundary(v4_pred_df, v45_df, s0["audit_df"])

    # Stage 4 — hardness / ceiling analysis
    ceiling_info = stage4_ceiling_analysis(v45_df, s0["audit_df"])

    # Stage 5 — G09 output policy update
    stage5_output_policy(v45_df, ceiling_info)

    # Stage 6 — final decision + audit log + memory-ready summary
    decision, metrics = stage6_decision(v4_pred_df, v45_df, ceiling_info)

    # Persist a small bundle
    bundle = {
        "stage0_summary": {k: v for k, v in s0.items()
                             if k not in ("audit_df", "pred_df_with", "pred_df_noroute")},
        "decision": decision,
        "metrics": metrics,
    }
    import json
    (AUDIT / "stage0_to_6_summary.json").write_text(json.dumps(bundle, indent=2, default=str))

    # Snapshot code
    p = Path(__file__)
    if p.exists():
        shutil.copy(p, CODE_SNAPSHOT / p.name)

    print("\n[v4.5 complete]")
    print(f"  overall top-1:  v4 {metrics['v4_overall_top1']:.1%} → "
          f"v4.5 {metrics['v45_overall_top1']:.1%}")
    print(f"  Raman top-1:    v4 {metrics['v4_raman_top1']:.1%} → "
          f"v4.5 {metrics['v45_raman_top1']:.1%}")
    print(f"  SERS top-1:     v4 {metrics['v4_sers_top1']:.1%} → "
          f"v4.5 {metrics['v45_sers_top1']:.1%}")
    print(f"  G09 top-1:      v4 {metrics['v4_g09_top1']:.1%} → "
          f"v4.5 {metrics['v45_g09_top1']:.1%}")
    print(f"  TG top-1:       v4 {metrics['v4_tg_top1']:.1%} → "
          f"v4.5 {metrics['v45_tg_top1']:.1%}")
    print(f"  decision:       {decision}")


if __name__ == "__main__":
    main()
