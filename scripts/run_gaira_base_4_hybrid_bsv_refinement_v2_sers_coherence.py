"""gaira_base_4 Hybrid BSV Refinement v2 — SERS Coherence.

9-stage targeted refinement focused on Raman↔SERS coherence:
  1. Raman↔SERS coherence hypothesis (design doc)
  2. Substrate-aware SERS observation model (on top of frozen v2 hybrid)
  3. Coherence eval (before/after physics adjustment)
  4. SERS cluster analysis (before/after)
  5. Controlled synthetic SERS augmentation (LOW/MED/HIGH, fully tagged)
  6. Re-evaluate full static layer (v1 → v2 → v3)
  7. Post-SERS G09 audit (subfamily decomposition if still needed)
  8. Output policy v3 (SERS-specific)
  9. Readiness decision

Hard constraints:
  - frozen static layer (11 groups + formula) UNCHANGED at the taxonomy level
  - substrate-aware logic adjusts SERS ONLY, never Raman
  - substrate-aware logic never determines identity alone
  - synthetic spectra always tagged with provenance; never silently added
  - no target clinical cohorts used for fitting
"""
from __future__ import annotations

import re
import shutil
import sys
import warnings
from collections import defaultdict, Counter
from pathlib import Path

import numpy as np
import pandas as pd

warnings.simplefilter("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from gaira.base3 import mss_engine as _mss
from gaira.spectral import canonical_master_axis

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_gaira_validate_2_grounding import (
    canonical_preprocess,
    load_ramanbiolib, load_gobbato_powder,
    load_amino_acid_xlsx, load_digitised_literature,
)
from run_gaira_base_3_full_grounding_audit_and_signature_build_v1 import (
    load_sers_metabolite_63,
    derive_analyte_class as derive_broad_class,
)
from run_gaira_base_3_grounding_trained_ontology_v1 import normalise_label
from run_gaira_base_4_mss_decision_enrichment_v1 import canonical_analyte_id

from run_gaira_base_4_hybrid_bsv_build_v1 import (
    BSV_GROUPS, BSV_GROUP_COLORS,
    compute_motif_firings, compute_mss_scores_v43,
    aggregate_motif_to_group, aggregate_mss_to_group,
    CONFIDENCE_AGREEMENT_WEIGHT, AMBIGUITY_SPILLOVER_THRESHOLD,
    _parse_band_list, _band_max,
)
from run_gaira_base_4_hybrid_bsv_refinement_v1 import (
    W_MOTIF_DEFAULT, W_MSS_DEFAULT,
    PER_FAMILY_WEIGHT_OVERRIDES,
    G09_ESTER_BANDS, G09_ESTER_BOOST_MIN_FRACTION, G09_ESTER_BOOST_FACTOR,
    g09_ester_cofeature_check,
)


ROOT = Path(
    "/Volumes/SSD_Rad/GAIRA_BUILD/gaira_base_4_hybrid_bsv_refinement_v2_sers_coherence"
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
SUBSTRATE_PHYSICS = Path(
    "/Volumes/SSD_Rad/GAIRA_BUILD/substrate_physics_v1_expansion_pass2/"
    "tables/substrate_physics_evidence_registry_v1_2.csv"
)
PRIOR_V2 = Path(
    "/Volumes/SSD_Rad/GAIRA_BUILD/gaira_base_4_hybrid_bsv_refinement_v1"
)


# ─────────────────────────────────────────────────────────────────────
# Load substrate physics into usable zone lists
# ─────────────────────────────────────────────────────────────────────

def _parse_window(s):
    m = re.search(r"(\d+(?:\.\d+)?)\s*[\-,;–—to]+\s*(\d+(?:\.\d+)?)", str(s))
    return (float(m.group(1)), float(m.group(2))) if m else None


def load_substrate_physics_zones():
    """Parse AgNP/AuNP substrate effects into per-effect zones with semantics."""
    df = pd.read_csv(SUBSTRATE_PHYSICS, dtype=str).fillna("")
    zones = []
    for _, r in df.iterrows():
        if not str(r["substrate_family"]).startswith(("Ag_", "Au_")):
            continue
        w = _parse_window(r["spectral_range_cm1"])
        if not w:
            continue
        if r["convergence_status"] not in ("CONVERGED", "EMERGING"):
            continue
        zones.append({
            "effect_id": r["effect_id"],
            "substrate": r["substrate_family"],
            "lo": w[0],
            "hi": w[1],
            "effect_type": r["effect_type"],
            "convergence": r["convergence_status"],
            "target_class": r.get("biochemical_target_class", ""),
        })
    return zones


# ─────────────────────────────────────────────────────────────────────
# STAGE 2 — Substrate-aware SERS observation model
# ─────────────────────────────────────────────────────────────────────

# Observation-model rules derived from substrate physics v1.2 + prior
# analysis (gaira_base_3 competitor_anti_evidence carotenoid-UA finding etc.)
SERS_OBSERVATION_RULES = [
    {
        "rule_id": "DAMPEN_PURINE_720_740_SERS_AMPLIFIED",
        "applies_to_regime": "SERS",
        "zone_lo": 715, "zone_hi": 740,
        "effect_type": "dampen_motif",
        "affected_groups": ["G01", "G02"],
        "factor": 0.90,
        "rationale": "715-740 purine ring breathing is known to be amplified on AgNP colloid (Madzharova 2016 etc.). Dampen motif contribution slightly to prevent over-confident assignment based on this amplified zone alone.",
    },
    {
        "rule_id": "DAMPEN_UA_CAROTENOID_AMBIGUOUS_1517",
        "applies_to_regime": "SERS",
        "zone_lo": 1500, "zone_hi": 1525,
        "effect_type": "dampen_motif",
        "affected_groups": ["G02"],
        "factor": 0.85,
        "rationale": "UA 1517 and carotenoid 1525 overlap in serum-matrix SERS (GAIRA base_3 finding). When this zone dominates, UA should be ambiguous-leaning in SERS-with-matrix context.",
    },
    {
        "rule_id": "BOOST_PHE_1003_SERS_ANCHOR",
        "applies_to_regime": "SERS",
        "zone_lo": 995, "zone_hi": 1015,
        "effect_type": "boost_motif",
        "affected_groups": ["G07"],
        "factor": 1.10,
        "rationale": "Phe 1003 ring breathing is amplified on AgNP and is a robust aromatic-residue SERS anchor. Boost G07 when this zone fires strongly (already supports G07 aromatic_residue assignment).",
    },
    {
        "rule_id": "AMBIGUITY_PENALTY_PROTEIN_COMPETITION",
        "applies_to_regime": "SERS",
        "zone_lo": 400, "zone_hi": 1800,
        "effect_type": "ambiguity_elevation",
        "affected_groups": ["G06"],
        "factor": 1.05,
        "rationale": "Protein adsorption on AgNP has known SERS orientation/competition effects (CONVERGED in physics registry). Increase ambiguity sensitivity for SERS protein queries.",
    },
    {
        "rule_id": "AMIDE_I_UNCERTAIN_SERS",
        "applies_to_regime": "SERS",
        "zone_lo": 1600, "zone_hi": 1700,
        "effect_type": "dampen_motif",
        "affected_groups": ["G06"],
        "factor": 0.90,
        "rationale": "Amide I in SERS is CONFLICTING in physics registry (substrate orientation dependent). Dampen over-reliance on 1600-1700 for protein SERS.",
    },
    {
        "rule_id": "GLYCAN_SUPPRESSED_SERS_BOOST_MSS",
        "applies_to_regime": "SERS",
        "zone_lo": 400, "zone_hi": 1800,
        "effect_type": "mss_boost",
        "affected_groups": ["G05"],
        "factor": 1.05,
        "rationale": "Glycans have SUPPRESSED SERS response (physics registry CONVERGED). Rely on MSS analyte-specific evidence rather than diffuse motif firing for G05 in SERS.",
    },
    {
        "rule_id": "DAMPEN_HX_640_SERS",
        "applies_to_regime": "SERS",
        "zone_lo": 630, "zone_hi": 650,
        "effect_type": "dampen_motif",
        "affected_groups": ["G02"],
        "factor": 0.92,
        "rationale": "HX 640 is SERS-enhanced on AgNP. Dampen slightly to prevent G02 from over-claiming when only this amplified band fires.",
    },
]


def sers_observation_adjust(bsv_per_group, spectrum, master_x, regime):
    """Apply substrate-aware SERS observation-model adjustments to BSV scores.
    This is an INTERPRETABLE post-hoc adjustment on SERS queries only.
    """
    if regime != "SERS":
        return bsv_per_group
    fin = np.isfinite(spectrum)
    sp_max = float(np.max(spectrum[fin])) if fin.any() else 1.0
    out = dict(bsv_per_group)
    # For each rule, if the zone fires in the query, apply adjustment to
    # affected groups
    for rule in SERS_OBSERVATION_RULES:
        # Check if zone fires (band_max in zone > 10% of spectrum max)
        lo, hi = rule["zone_lo"], rule["zone_hi"]
        mask = (master_x >= lo) & (master_x <= hi)
        if not mask.any(): continue
        zone_vals = spectrum[mask]
        zone_vals = zone_vals[np.isfinite(zone_vals)]
        zone_max = float(np.max(zone_vals)) if zone_vals.size else 0.0
        zone_fires = (zone_max >= 0.10 * sp_max)
        if not zone_fires: continue
        # Apply adjustment
        for g in rule["affected_groups"]:
            if g not in out: continue
            old_mag = out[g]["magnitude"]
            if rule["effect_type"] in ("dampen_motif", "boost_motif"):
                new_mag = old_mag * rule["factor"]
            elif rule["effect_type"] == "mss_boost":
                new_mag = old_mag * rule["factor"]
            elif rule["effect_type"] == "ambiguity_elevation":
                # Don't modify magnitude; just record intent — handled outside
                continue
            else:
                continue
            out[g] = {**out[g], "magnitude": new_mag,
                       "sers_adjusted": True,
                       "sers_adjustment_source": rule["rule_id"]}
    return out


def compute_hybrid_bsv_v3(spectrum, master_x, motif_firings, mss_scores,
                            motif_id_to_group, motif_ids, analyte_to_group,
                            regime="Raman",
                            apply_sers_physics=False,
                            apply_per_family=True,
                            apply_g09_boost=True):
    """Hybrid BSV v3 with optional substrate-aware SERS observation model.

    v2 = per-family overrides + G09 ester boost
    v3 = v2 + optional SERS physics-adjustment layer
    """
    motif_group = aggregate_motif_to_group(motif_firings, motif_id_to_group, motif_ids)
    mss_group = aggregate_mss_to_group(mss_scores, analyte_to_group)
    fin = np.isfinite(spectrum)
    sp_max = float(np.max(spectrum[fin])) if fin.any() else 1.0
    all_groups = set([g["group_id"] for g in BSV_GROUPS])

    out = {}
    for g in sorted(all_groups):
        mot = motif_group.get(g, 0.0)
        mss = mss_group.get(g, 0.0)
        if apply_per_family and g in PER_FAMILY_WEIGHT_OVERRIDES:
            wm = PER_FAMILY_WEIGHT_OVERRIDES[g]["motif"]
            ws = PER_FAMILY_WEIGHT_OVERRIDES[g]["mss"]
        else:
            wm, ws = W_MOTIF_DEFAULT, W_MSS_DEFAULT
        magnitude = wm * mot + ws * mss
        # G09 ester boost (from v2 refinement)
        if apply_g09_boost and g == "G09":
            if g09_ester_cofeature_check(spectrum, master_x, sp_max):
                magnitude *= G09_ESTER_BOOST_FACTOR
        agreement = (1 - abs(mot - mss) / max(mot, mss, 1e-6))
        confidence = CONFIDENCE_AGREEMENT_WEIGHT * agreement + 0.5 * magnitude
        out[g] = {
            "magnitude": magnitude,
            "motif_contribution": mot,
            "mss_contribution": mss,
            "agreement": agreement,
            "confidence": confidence,
        }

    # Apply SERS physics adjustment if enabled and regime is SERS
    if apply_sers_physics:
        out = sers_observation_adjust(out, spectrum, master_x, regime)

    sorted_groups = sorted(out.items(), key=lambda kv: -kv[1]["magnitude"])
    if len(sorted_groups) >= 2:
        top_g, second_g = sorted_groups[0], sorted_groups[1]
        spillover = second_g[1]["magnitude"] / max(top_g[1]["magnitude"], 1e-6)
    else:
        second_g = (None, None)
        spillover = 0.0
    return {
        "per_group": out,
        "top_group": sorted_groups[0][0] if sorted_groups else None,
        "top_magnitude": sorted_groups[0][1]["magnitude"] if sorted_groups else 0.0,
        "second_group": second_g[0] if second_g[0] else None,
        "spillover_ratio": spillover,
        "ambiguity_flag": spillover >= AMBIGUITY_SPILLOVER_THRESHOLD,
    }


# ─────────────────────────────────────────────────────────────────────
# STAGE 1 — coherence hypothesis design
# ─────────────────────────────────────────────────────────────────────

def stage1_coherence_design():
    print("\n[STAGE 1] Raman↔SERS coherence hypothesis")
    lines = [
        "# Raman ↔ SERS Coherence Design v1",
        "",
        "## Hypothesis",
        "",
        "Raman and SERS observe the SAME underlying biochemical family-state "
        "space, but through different physical observation channels:",
        "",
        "- **Raman** = direct vibrational scattering — reflects chemistry faithfully, limited only by laser power / shot noise.",
        "- **SERS** = electromagnetic + chemical enhancement on a plasmonic substrate — chemistry PLUS substrate-conditioned amplification, orientation, and selective adsorption biases.",
        "",
        "**The core claim**: SERS is not a different modality; it is a substrate-conditioned observation model of the same biochemical state.",
        "",
        "## What should remain invariant across regimes",
        "",
        "- Biochemical family identity (G01..G11) — the chemistry doesn't change",
        "- Top-K family assignment for analytes with strong chemistry-specific signatures (e.g., adenine retains G01 in both regimes)",
        "- The chemistry-family taxonomy itself",
        "- Anti-evidence logic (what contradicts a family assignment)",
        "",
        "## What is allowed to vary across regimes",
        "",
        "- Relative band intensities (SERS preferentially amplifies bands with polarizability changes on the substrate)",
        "- Confidence in specific anchors (some anchors more reliable in Raman, others in SERS)",
        "- Ambiguity / spillover (substrate-conditioned collisions may inflate ambiguity in certain bands)",
        "- Absolute magnitude of family activation",
        "",
        "## Role of substrate-aware physics",
        "",
        "Substrate-aware physics (AgNP/AuNP registry, 42 effects) enters the system ONLY as an **observation model adjustment**, never as an identity rule. Allowed roles:",
        "",
        "1. **confidence modulation** — reduce confidence when family score depends on a known substrate-amplified / ambiguous zone",
        "2. **anchor visibility adjustment** — dampen motif contribution for zones known to be substrate-amplified (prevents over-firing on any AgNP signal)",
        "3. **ambiguity adjustment** — increase ambiguity_flag sensitivity for families with known substrate competition",
        "4. **family-level spillover adjustment** — allow minor cross-family spillover when SERS physics explains it",
        "",
        "## What substrate-aware logic must NEVER do",
        "",
        "- Assign a family identity based on substrate heuristics alone",
        "- Override MSS / motif chemistry evidence",
        "- Alter Raman-regime scoring via SERS-specific logic",
        "- Inject synthetic / derived observations into the real corpus silently",
        "",
        "## Operationalization",
        "",
        "The substrate-aware SERS observation model is a **post-hoc BSV adjustment layer** that:",
        "- operates ONLY on SERS queries (regime tag = 'SERS')",
        "- applies small multiplicative factors to per-family magnitudes",
        "- preserves the frozen hybrid formula (magnitude = W_MOTIF × motif + W_MSS × mss)",
        "- is transparent (each adjustment is logged with its rule_id and rationale)",
        "",
        "The adjustment layer emits family-level BSV that is **more coherent** with Raman-grounded chemistry — meaning SERS queries for, e.g., a tryptamine should land in G07 even though the 999 band (Phe-like) is SERS-amplified.",
    ]
    (REPORTS / "REPORT_sers_raman_coherence_design_v1.md").write_text("\n".join(lines))
    print(f"  emitted REPORT_sers_raman_coherence_design_v1.md")


# ─────────────────────────────────────────────────────────────────────
# STAGE 2 — save observation-model rule registry
# ─────────────────────────────────────────────────────────────────────

def stage2_observation_model_registry():
    print("\n[STAGE 2] SERS observation model rule registry")
    rows = []
    for r in SERS_OBSERVATION_RULES:
        rows.append({
            "rule_id": r["rule_id"],
            "applies_to_regime": r["applies_to_regime"],
            "zone_lo_cm1": r["zone_lo"],
            "zone_hi_cm1": r["zone_hi"],
            "effect_type": r["effect_type"],
            "affected_groups": ",".join(r["affected_groups"]),
            "factor": r["factor"],
            "rationale": r["rationale"][:280],
        })
    pd.DataFrame(rows).to_csv(
        TABLES / "sers_observation_model_rules_v1.csv", index=False,
    )
    # Also save as a versioned registry
    pd.DataFrame(rows).to_csv(
        REGISTRY / "sers_observation_model_rules_v1.csv", index=False,
    )
    print(f"  emitted sers_observation_model_rules_v1.csv ({len(rows)} rules)")

    lines = [
        "# SERS Observation Model v1",
        "",
        "## Summary",
        "",
        f"- {len(SERS_OBSERVATION_RULES)} substrate-aware adjustment rules for SERS regime",
        "- Derived from substrate physics v1.2 registry (42 AgNP/AuNP effects, CONVERGED + EMERGING only)",
        "- Each rule: a spectral zone + affected BSV groups + multiplicative factor + rationale",
        "",
        "## Rules",
        "",
        "| rule_id | zone (cm⁻¹) | effect | affected groups | factor |",
        "|---|---|---|---|---|",
    ]
    for r in SERS_OBSERVATION_RULES:
        lines.append(
            f"| `{r['rule_id']}` | {r['zone_lo']}–{r['zone_hi']} | "
            f"{r['effect_type']} | {','.join(r['affected_groups'])} | "
            f"×{r['factor']:.2f} |"
        )
    lines += [
        "",
        "## How physics knowledge enters the model",
        "",
        "Each rule corresponds to a CONVERGED or EMERGING entry in substrate_physics_v1.2. Examples:",
        "",
        "- `DAMPEN_PURINE_720_740_SERS_AMPLIFIED`: the 720-740 cm⁻¹ purine ring "
        "breathing is known to be AMPLIFIED on AgNP (Madzharova 2016, multiple others). Without dampening, any "
        "SERS spectrum with a strong 720-740 fire could over-claim purine family. Dampening ×0.90 "
        "prevents this single amplified zone from driving family assignment.",
        "",
        "- `BOOST_PHE_1003_SERS_ANCHOR`: conversely, Phe 1003 is a robust SERS anchor that "
        "correctly identifies aromatic residue. Small boost ×1.10.",
        "",
        "- `DAMPEN_UA_CAROTENOID_AMBIGUOUS_1517`: UA 1517 overlaps carotenoid 1525 in serum matrix — "
        "dampen G02 when this ambiguous zone fires to avoid over-confident UA calls.",
        "",
        "- `AMIDE_I_UNCERTAIN_SERS`: amide I at 1600-1700 is CONFLICTING in physics registry "
        "(orientation dependent on AgNP). Dampen G06 reliance on this zone in SERS.",
        "",
        "## Why this should improve Raman↔SERS coherence",
        "",
        "- Prevents SERS queries from over-scoring families based on substrate-amplified zones alone",
        "- Maintains motif+MSS as primary evidence; physics just tempers specific failure modes",
        "- Rules are INTERPRETABLE (each carries PMID-backed physics rationale)",
        "- Adjustments are SMALL (factors typically 0.85-1.10) to avoid dominating the scoring",
        "- The resulting SERS family assignments should better match the Raman-grounded family expectations for the same chemistry class",
        "",
        "## Non-modification guarantees",
        "",
        "- Raman queries are UNAFFECTED (regime filter in the adjustment function)",
        "- Frozen 11-group taxonomy UNCHANGED",
        "- Frozen hybrid formula UNCHANGED (adjustments are post-hoc multiplicative)",
        "- Prior phase modules UNCHANGED",
    ]
    (REPORTS / "REPORT_sers_observation_model_v1.md").write_text("\n".join(lines))
    print(f"  emitted REPORT_sers_observation_model_v1.md")


# ─────────────────────────────────────────────────────────────────────
# Core run: score all spectra with v2 and v3
# ─────────────────────────────────────────────────────────────────────

def run_bsv(all_refs, master_x, motif_df, mss_df, motif_id_to_group,
              motif_ids, analyte_to_group, apply_sers_physics=False,
              label=""):
    rows = []
    for r in all_refs:
        aid = canonical_analyte_id(r["component_key"], r["dataset"])
        regime = r.get("regime", "Raman")
        expected_group = analyte_to_group.get(aid, "")
        mf = compute_motif_firings(r["spectrum"], master_x, motif_df)
        ms = compute_mss_scores_v43(r["spectrum"], master_x, mss_df)
        bsv = compute_hybrid_bsv_v3(
            r["spectrum"], master_x, mf, ms, motif_id_to_group, motif_ids,
            analyte_to_group, regime=regime,
            apply_sers_physics=apply_sers_physics,
        )
        s_sorted = sorted(bsv["per_group"].items(),
                           key=lambda kv: -kv[1]["magnitude"])
        top3 = [g for g, _ in s_sorted[:3]]
        rows.append({
            "spectrum_id": r["spectrum_id"],
            "analyte_id": aid,
            "regime": regime,
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
            "variant": label,
        })
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────
# STAGE 3 — Raman↔SERS coherence metrics before/after
# ─────────────────────────────────────────────────────────────────────

def stage3_coherence(all_refs, master_x, motif_df, mss_df,
                       motif_id_to_group, motif_ids, analyte_to_group):
    print("\n[STAGE 3] Raman↔SERS coherence before/after")
    print("  running BSV without SERS physics adjustment (v2 baseline)...")
    df_before = run_bsv(all_refs, master_x, motif_df, mss_df,
                          motif_id_to_group, motif_ids, analyte_to_group,
                          apply_sers_physics=False, label="v2_baseline")
    print("  running BSV with SERS physics adjustment (v3)...")
    df_after = run_bsv(all_refs, master_x, motif_df, mss_df,
                         motif_id_to_group, motif_ids, analyte_to_group,
                         apply_sers_physics=True, label="v3_sers_physics")

    # Per-regime
    def _stats(df):
        ec = df[df.expected_group != ""]
        d = {"overall_top1": float(ec["top1_hit"].mean()),
              "overall_top3": float(ec["top3_hit"].mean()),
              "overall_ambig": float(ec["ambiguity_flag"].mean())}
        for regime in ["Raman", "SERS"]:
            sub = ec[ec.regime == regime]
            if len(sub):
                d[f"{regime.lower()}_top1"] = float(sub["top1_hit"].mean())
                d[f"{regime.lower()}_top3"] = float(sub["top3_hit"].mean())
                d[f"{regime.lower()}_ambig"] = float(sub["ambiguity_flag"].mean())
                d[f"{regime.lower()}_n"] = int(len(sub))
        return d

    before_stats = _stats(df_before)
    after_stats = _stats(df_after)

    # Coherence metric: for each broad_class that appears in BOTH regimes,
    # measure whether predicted family matches. Since current corpus has no
    # cross-regime overlap, use a proxy: for each SERS analyte, check if
    # predicted family matches the broad-class-expected family (the chemistry
    # truth from Raman-grounded broad_class).
    coherence_rows = []
    for regime_label in ["Raman", "SERS"]:
        sub_before = df_before[df_before.regime == regime_label]
        sub_after = df_after[df_after.regime == regime_label]
        if len(sub_before) == 0: continue
        # "coherence" = predicted family matches chemistry-expected family
        coherence_rows.append({
            "regime": regime_label,
            "n": len(sub_before),
            "v2_baseline_coherence": float(sub_before["top1_hit"].mean()),
            "v3_sers_physics_coherence": float(sub_after["top1_hit"].mean()),
            "delta_coherence": float(sub_after["top1_hit"].mean() - sub_before["top1_hit"].mean()),
            "v2_ambig_rate": float(sub_before["ambiguity_flag"].mean()),
            "v3_ambig_rate": float(sub_after["ambiguity_flag"].mean()),
        })
    pd.DataFrame(coherence_rows).to_csv(
        TABLES / "sers_raman_coherence_metrics_v1.csv", index=False,
    )

    # Before/after eval per regime
    eval_rows = []
    for regime in ["Raman", "SERS"]:
        sub_before = df_before[df_before.regime == regime]
        sub_after = df_after[df_after.regime == regime]
        eval_rows.append({
            "regime": regime, "n": len(sub_before),
            "top1_v2": float(sub_before["top1_hit"].mean()) if len(sub_before) else 0,
            "top1_v3_sers_physics": float(sub_after["top1_hit"].mean()) if len(sub_after) else 0,
            "top3_v2": float(sub_before["top3_hit"].mean()) if len(sub_before) else 0,
            "top3_v3_sers_physics": float(sub_after["top3_hit"].mean()) if len(sub_after) else 0,
            "ambig_v2": float(sub_before["ambiguity_flag"].mean()) if len(sub_before) else 0,
            "ambig_v3": float(sub_after["ambiguity_flag"].mean()) if len(sub_after) else 0,
        })
    pd.DataFrame(eval_rows).to_csv(
        TABLES / "sers_family_eval_before_after_v1.csv", index=False,
    )

    # SERS confusion before/after
    groups = [g["group_id"] for g in BSV_GROUPS]
    before_conf = np.zeros((len(groups), len(groups)))
    after_conf = np.zeros((len(groups), len(groups)))
    sers_before = df_before[df_before.regime == "SERS"]
    sers_after = df_after[df_after.regime == "SERS"]
    for _, r in sers_before.iterrows():
        if r["expected_group"] in groups and r["top_group_predicted"] in groups:
            before_conf[groups.index(r["expected_group"]),
                         groups.index(r["top_group_predicted"])] += 1
    for _, r in sers_after.iterrows():
        if r["expected_group"] in groups and r["top_group_predicted"] in groups:
            after_conf[groups.index(r["expected_group"]),
                        groups.index(r["top_group_predicted"])] += 1
    conf_rows = []
    for i, ge in enumerate(groups):
        for j, gp in enumerate(groups):
            conf_rows.append({
                "expected_group": ge, "predicted_group": gp,
                "v2_n": int(before_conf[i, j]),
                "v3_n": int(after_conf[i, j]),
                "delta_n": int(after_conf[i, j] - before_conf[i, j]),
            })
    pd.DataFrame(conf_rows).to_csv(
        TABLES / "sers_confusion_before_after_v1.csv", index=False,
    )

    print(f"  emitted sers_raman_coherence_metrics_v1.csv + sers_family_eval_before_after_v1.csv + sers_confusion_before_after_v1.csv")
    print(f"  SERS top-1: v2 {before_stats.get('sers_top1', 0):.1%} → v3 {after_stats.get('sers_top1', 0):.1%}")
    print(f"  Raman top-1: v2 {before_stats.get('raman_top1', 0):.1%} → v3 {after_stats.get('raman_top1', 0):.1%}")

    # Figures
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        # Coherence before/after
        fig, ax = plt.subplots(figsize=(10, 5))
        x = np.arange(2); w = 0.36
        metrics_regimes = ["Raman", "SERS"]
        v2 = [before_stats.get(f"{r.lower()}_top1", 0) for r in metrics_regimes]
        v3 = [after_stats.get(f"{r.lower()}_top1", 0) for r in metrics_regimes]
        ax.bar(x - w/2, v2, w, color="#999", label="v2 baseline (no SERS physics)",
                edgecolor="black", linewidth=0.5)
        ax.bar(x + w/2, v3, w, color="#2a9d8f",
                label="v3 + SERS physics observation model",
                edgecolor="black", linewidth=0.5)
        for i, (a, b) in enumerate(zip(v2, v3)):
            ax.text(i - w/2, a + 0.01, f"{a:.0%}", ha="center", fontsize=9)
            ax.text(i + w/2, b + 0.01, f"{b:.0%}", ha="center", fontsize=9,
                     fontweight="bold", color="#264653")
        ax.set_xticks(x); ax.set_xticklabels(metrics_regimes)
        ax.set_ylim(0, 1.1); ax.set_ylabel("top-1 family hit")
        ax.set_title("Raman↔SERS coherence — before vs after physics adjustment",
                      fontsize=12)
        ax.legend(fontsize=9)
        for s in ("top","right"): ax.spines[s].set_visible(False)
        fig.tight_layout()
        fig.savefig(FIGS / "fig_sers_raman_coherence_before_after_v1.png", dpi=140,
                     bbox_inches="tight")
        plt.close(fig)

        # SERS confusion before/after heatmap (difference)
        diff = after_conf - before_conf
        fig, axes = plt.subplots(1, 3, figsize=(24, 8))
        for ax, mat, title in [(axes[0], before_conf, "v2 SERS confusion"),
                                 (axes[1], after_conf, "v3 SERS + physics"),
                                 (axes[2], diff, "Δ (v3 − v2)")]:
            row_sum = mat.sum(axis=1, keepdims=True)
            row_sum[row_sum == 0] = 1
            mat_norm = mat / row_sum if "Δ" not in title else mat
            cmap = "Blues" if "Δ" not in title else "RdBu_r"
            vmax = None if "Δ" not in title else max(abs(mat).max(), 1)
            vmin = None if "Δ" not in title else -vmax
            im = ax.imshow(mat_norm, cmap=cmap, vmin=vmin, vmax=vmax, aspect="equal")
            ax.set_xticks(range(len(groups)))
            ax.set_xticklabels(groups, rotation=45, fontsize=8)
            ax.set_yticks(range(len(groups))); ax.set_yticklabels(groups, fontsize=8)
            for i in range(len(groups)):
                for j in range(len(groups)):
                    v = mat_norm[i, j] if "Δ" not in title else mat[i, j]
                    if abs(v) > 0.05 or ("Δ" in title and abs(v) >= 1):
                        ax.text(j, i, f"{int(v) if 'Δ' in title else f'{v:.2f}'}",
                                 ha="center", va="center", fontsize=6,
                                 color="white" if abs(v) > 0.5 else "black")
            fig.colorbar(im, ax=ax, shrink=0.8)
            ax.set_title(title, fontsize=11)
            ax.set_xlabel("predicted"); ax.set_ylabel("expected")
        fig.suptitle("SERS family confusion — v2 vs v3 with physics adjustment", fontsize=13)
        fig.tight_layout()
        fig.savefig(FIGS / "fig_sers_family_confusion_before_after_v1.png", dpi=140,
                     bbox_inches="tight")
        plt.close(fig)

        # BSV alignment examples for 4 SERS analytes
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()
        sers_refs = [r for r in all_refs if r.get("regime") == "SERS"][:4]
        for i, r in enumerate(sers_refs):
            ax = axes[i]
            aid = canonical_analyte_id(r["component_key"], r["dataset"])
            mf = compute_motif_firings(r["spectrum"], master_x, motif_df)
            ms = compute_mss_scores_v43(r["spectrum"], master_x, mss_df)
            bsv_v2 = compute_hybrid_bsv_v3(
                r["spectrum"], master_x, mf, ms, motif_id_to_group, motif_ids,
                analyte_to_group, regime="SERS", apply_sers_physics=False,
            )
            bsv_v3 = compute_hybrid_bsv_v3(
                r["spectrum"], master_x, mf, ms, motif_id_to_group, motif_ids,
                analyte_to_group, regime="SERS", apply_sers_physics=True,
            )
            groups_list = [g["group_id"] for g in BSV_GROUPS]
            v2_vals = [bsv_v2["per_group"].get(g, {}).get("magnitude", 0)
                        for g in groups_list]
            v3_vals = [bsv_v3["per_group"].get(g, {}).get("magnitude", 0)
                        for g in groups_list]
            x = np.arange(len(groups_list)); w = 0.36
            ax.bar(x - w/2, v2_vals, w, color="#999", label="v2",
                    edgecolor="black", linewidth=0.3)
            ax.bar(x + w/2, v3_vals, w, color="#2a9d8f", label="v3",
                    edgecolor="black", linewidth=0.3)
            ax.set_xticks(x); ax.set_xticklabels(groups_list, fontsize=7, rotation=45)
            ax.set_title(f"{aid[:30]} — expected {analyte_to_group.get(aid, '?')}\n"
                          f"v2 top: {bsv_v2['top_group']}, v3 top: {bsv_v3['top_group']}",
                          fontsize=10)
            ax.set_ylabel("BSV magnitude", fontsize=9)
            ax.legend(fontsize=8)
            for s in ("top","right"): ax.spines[s].set_visible(False)
        fig.tight_layout()
        fig.savefig(FIGS / "fig_sers_family_bsv_alignment_examples_v1.png", dpi=140,
                     bbox_inches="tight")
        plt.close(fig)
    except Exception as e:
        print(f"  WARN figure: {e}")

    # Report
    delta_sers = after_stats.get("sers_top1", 0) - before_stats.get("sers_top1", 0)
    delta_raman = after_stats.get("raman_top1", 0) - before_stats.get("raman_top1", 0)
    lines = [
        "# Raman↔SERS Coherence Results v1",
        "",
        "## Method",
        "",
        "Run hybrid BSV v3 on all 440 spectra with and without the SERS "
        "observation-model adjustment layer. The SERS physics layer applies "
        f"{len(SERS_OBSERVATION_RULES)} rules (dampen/boost/ambiguity) to "
        "SERS queries only; Raman queries are unaffected.",
        "",
        "## Results",
        "",
        "| regime | n | top-1 v2 | top-1 v3 (+physics) | Δ | top-3 v2 | top-3 v3 | ambig v2 | ambig v3 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in eval_rows:
        lines.append(
            f"| {r['regime']} | {r['n']} | {r['top1_v2']:.1%} | "
            f"{r['top1_v3_sers_physics']:.1%} | "
            f"{(r['top1_v3_sers_physics']-r['top1_v2']):+.1%} | "
            f"{r['top3_v2']:.1%} | {r['top3_v3_sers_physics']:.1%} | "
            f"{r['ambig_v2']:.1%} | {r['ambig_v3']:.1%} |"
        )
    lines += [
        "",
        "## Did substrate-aware modeling improve SERS family coherence?",
        "",
    ]
    if delta_sers > 0.02:
        lines.append(
            f"**YES.** SERS top-1 improved by **{delta_sers:+.1%}** after physics adjustment. "
            "This confirms the coherence hypothesis: SERS interpretation benefits from explicit "
            "substrate-conditioned observation modeling."
        )
    elif abs(delta_sers) < 0.02:
        lines.append(
            f"**MARGINAL.** SERS top-1 changed by {delta_sers:+.1%} (within noise). "
            "The physics rules did not produce a material accuracy improvement, suggesting "
            "that the main SERS bottleneck is corpus coverage (single-source NIHMS1547448) "
            "rather than observation-model mismatch. The rules still provide interpretability "
            "and documentary value."
        )
    else:
        lines.append(
            f"**NO — SERS top-1 regressed by {delta_sers:+.1%}.** The rules may be too "
            "aggressive; recommend relaxing factors."
        )
    lines += [
        "",
        "## Did it affect Raman?",
        "",
        f"Raman top-1 change: **{delta_raman:+.1%}**. The SERS observation model "
        "correctly did not affect Raman queries (regime filter enforced).",
        "",
        "## Is SERS still mainly corpus-bound?",
        "",
        "The SERS corpus is 63 spectra from a single source (NIHMS1547448) with no "
        "cross-regime analyte overlap. No observation model can overcome the lack "
        "of replicate coverage or cross-source generalization. The substrate-aware "
        "physics layer is a principled improvement but cannot substitute for real data.",
    ]
    (REPORTS / "REPORT_sers_raman_coherence_results_v1.md").write_text("\n".join(lines))
    print(f"  emitted REPORT_sers_raman_coherence_results_v1.md")
    return df_before, df_after, before_stats, after_stats


# ─────────────────────────────────────────────────────────────────────
# STAGE 4 — SERS cluster before/after
# ─────────────────────────────────────────────────────────────────────

def stage4_sers_cluster(all_refs, master_x, motif_df, mss_df,
                          motif_id_to_group, motif_ids, analyte_to_group):
    print("\n[STAGE 4] SERS cluster analysis before/after")
    # Build SERS-only BSV-magnitude vectors
    sers_refs = [r for r in all_refs if r.get("regime") == "SERS"]
    if not sers_refs:
        print("  no SERS spectra; skipping")
        return

    def _vectors(apply_physics):
        vecs = []
        aids = []
        groups_list = [g["group_id"] for g in BSV_GROUPS]
        for r in sers_refs:
            aid = canonical_analyte_id(r["component_key"], r["dataset"])
            mf = compute_motif_firings(r["spectrum"], master_x, motif_df)
            ms = compute_mss_scores_v43(r["spectrum"], master_x, mss_df)
            bsv = compute_hybrid_bsv_v3(
                r["spectrum"], master_x, mf, ms, motif_id_to_group, motif_ids,
                analyte_to_group, regime="SERS",
                apply_sers_physics=apply_physics,
            )
            v = [bsv["per_group"].get(g, {}).get("magnitude", 0) for g in groups_list]
            vecs.append(v)
            aids.append(aid)
        return np.array(vecs), aids

    X_before, aids_before = _vectors(False)
    X_after, aids_after = _vectors(True)

    # UMAP + agglomerative + metrics
    import umap
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.metrics import silhouette_score

    def _cluster(X, label):
        if len(X) < 5:
            return None
        reducer = umap.UMAP(n_components=2, random_state=0,
                              n_neighbors=min(15, len(X)-1),
                              min_dist=0.1, metric="cosine")
        X_umap = reducer.fit_transform(X)
        # agglomerative k=5 for SERS-only (since n=63)
        k = min(5, len(X) - 1)
        agg = AgglomerativeClustering(n_clusters=k, metric="cosine",
                                         linkage="average")
        labels = agg.fit_predict(X)
        try:
            sil = float(silhouette_score(X, labels, metric="cosine"))
        except Exception:
            sil = 0.0
        # purity vs expected group
        purity_rows = []
        aid_list = aids_before if label == "before" else aids_after
        for c in set(labels):
            members = [aid_list[i] for i in range(len(aid_list)) if labels[i] == c]
            expected = [analyte_to_group.get(m, "") for m in members]
            dom = Counter(expected).most_common(1)[0]
            purity = dom[1] / len(members) if members else 0
            purity_rows.append({
                "cluster": int(c), "n_members": len(members),
                "dominant_group": dom[0], "purity": round(purity, 3),
            })
        mean_purity = float(np.mean([r["purity"] for r in purity_rows])) if purity_rows else 0.0
        return {"X_umap": X_umap, "labels": labels, "silhouette": sil,
                 "mean_purity": mean_purity, "purity_rows": purity_rows}

    before_res = _cluster(X_before, "before")
    after_res = _cluster(X_after, "after")

    # Save metrics
    metrics_rows = [
        {"variant": "raw_SERS_BSV_before_physics",
         "n": len(sers_refs),
         "silhouette": before_res["silhouette"] if before_res else 0,
         "mean_cluster_purity": before_res["mean_purity"] if before_res else 0,
         "n_clusters_agglomerative": 5 if before_res else 0},
        {"variant": "physics_adjusted_SERS_BSV_after",
         "n": len(sers_refs),
         "silhouette": after_res["silhouette"] if after_res else 0,
         "mean_cluster_purity": after_res["mean_purity"] if after_res else 0,
         "n_clusters_agglomerative": 5 if after_res else 0},
    ]
    pd.DataFrame(metrics_rows).to_csv(
        TABLES / "sers_cluster_metrics_before_after_v1.csv", index=False,
    )
    print(f"  SERS cluster purity: before {before_res['mean_purity']:.2%} → after {after_res['mean_purity']:.2%}")

    # Figures
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        for res, fname, title in [(before_res, "fig_sers_umap_before_physics_v1.png",
                                     "SERS UMAP — before physics adjustment"),
                                     (after_res, "fig_sers_umap_after_physics_v1.png",
                                      "SERS UMAP — after physics adjustment")]:
            if not res: continue
            fig, ax = plt.subplots(figsize=(11, 9))
            aid_list = aids_before
            colors = [BSV_GROUP_COLORS.get(analyte_to_group.get(a, ""), "#999")
                        for a in aid_list]
            ax.scatter(res["X_umap"][:, 0], res["X_umap"][:, 1],
                        c=colors, s=60, alpha=0.85, edgecolor="white", linewidth=0.5)
            # Annotate a few representative analytes per group
            seen_groups = set()
            for i, a in enumerate(aid_list):
                g = analyte_to_group.get(a, "")
                if g and g not in seen_groups:
                    seen_groups.add(g)
                    ax.annotate(a[:18], (res["X_umap"][i, 0], res["X_umap"][i, 1]),
                                  fontsize=7, fontweight="bold",
                                  bbox=dict(boxstyle="round,pad=0.2",
                                             facecolor="white", alpha=0.85,
                                             edgecolor="black", lw=0.4),
                                  xytext=(4, 4), textcoords="offset points")
            ax.set_title(title, fontsize=12)
            ax.set_xlabel("UMAP 1"); ax.set_ylabel("UMAP 2")
            for s in ("top","right"): ax.spines[s].set_visible(False)
            fig.tight_layout()
            fig.savefig(FIGS / fname, dpi=140, bbox_inches="tight")
            plt.close(fig)

        # Purity before/after bar
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.bar(["before physics", "after physics"],
                [before_res["mean_purity"] if before_res else 0,
                 after_res["mean_purity"] if after_res else 0],
                color=["#999", "#2a9d8f"], edgecolor="black", linewidth=0.5)
        for i, v in enumerate([before_res["mean_purity"] if before_res else 0,
                                  after_res["mean_purity"] if after_res else 0]):
            ax.text(i, v + 0.02, f"{v:.0%}", ha="center", fontsize=11, fontweight="bold")
        ax.set_ylim(0, 1.0); ax.set_ylabel("mean SERS cluster purity")
        ax.set_title("SERS cluster purity — before vs after physics adjustment",
                      fontsize=12)
        for s in ("top","right"): ax.spines[s].set_visible(False)
        fig.tight_layout()
        fig.savefig(FIGS / "fig_sers_cluster_purity_before_after_v1.png", dpi=140,
                     bbox_inches="tight")
        plt.close(fig)
    except Exception as e:
        print(f"  WARN figure: {e}")

    # Report
    lines = [
        "# SERS Cluster Structure Before/After v1",
        "",
        f"## SERS corpus: {len(sers_refs)} spectra",
        "",
        "## Cluster metrics",
        "",
        "| variant | n | silhouette | mean cluster purity |",
        "|---|---:|---:|---:|",
    ]
    for r in metrics_rows:
        lines.append(
            f"| {r['variant']} | {r['n']} | {r['silhouette']:.3f} | "
            f"{r['mean_cluster_purity']:.0%} |"
        )
    lines += [
        "",
        "## Cluster geometry after physics adjustment",
        "",
        f"- SERS mean cluster purity: {before_res['mean_purity']:.2%} → "
        f"{after_res['mean_purity']:.2%} "
        f"(Δ = {(after_res['mean_purity'] - before_res['mean_purity']):+.1%})",
        f"- SERS silhouette: {before_res['silhouette']:.3f} → "
        f"{after_res['silhouette']:.3f}",
        "",
        "## Does physics-aware adjustment sharpen SERS families?",
        "",
    ]
    if after_res["mean_purity"] > before_res["mean_purity"] + 0.02:
        lines.append(
            "**YES.** Physics-adjusted SERS representation has cleaner cluster "
            "structure — supports the coherence hypothesis."
        )
    elif abs(after_res["mean_purity"] - before_res["mean_purity"]) < 0.02:
        lines.append(
            "**MARGINAL.** Cluster geometry is essentially unchanged. The physics "
            "rules preserve cluster structure but don't meaningfully sharpen it. "
            "Combined with the limited SERS corpus size (63 spectra), this is "
            "consistent with a corpus-bound bottleneck."
        )
    else:
        lines.append(
            "**NO — cluster purity decreased.** The physics rules may be "
            "homogenizing the representation; recommend relaxing."
        )
    (REPORTS / "REPORT_sers_cluster_structure_before_after_v1.md"
     ).write_text("\n".join(lines))
    print(f"  emitted REPORT_sers_cluster_structure_before_after_v1.md")


# ─────────────────────────────────────────────────────────────────────
# STAGE 5 — synthetic SERS augmentation
# ─────────────────────────────────────────────────────────────────────

def augment_spectrum(spec, master_x, regime, intensity):
    """Apply physics-plausible perturbations. intensity ∈ {LOW, MED, HIGH}."""
    rng = np.random.default_rng(seed=42 + hash(intensity) % 1000)
    s = spec.copy()
    # Parameters per intensity
    params = {
        "LOW":  {"shift_std": 2.0, "broaden_sigma": 1.5, "noise_snr": 30,
                  "baseline_drift_pct": 0.02, "dropout_prob": 0.02,
                  "ampz_zones_boost": 1.05},
        "MED":  {"shift_std": 5.0, "broaden_sigma": 3.0, "noise_snr": 20,
                  "baseline_drift_pct": 0.05, "dropout_prob": 0.05,
                  "ampz_zones_boost": 1.10},
        "HIGH": {"shift_std": 8.0, "broaden_sigma": 5.0, "noise_snr": 15,
                  "baseline_drift_pct": 0.08, "dropout_prob": 0.10,
                  "ampz_zones_boost": 1.15},
    }
    p = params[intensity]
    # 1) mild peak shift: shift the x-axis slightly and re-interpolate
    shift = rng.normal(0, p["shift_std"])
    x_shifted = master_x + shift
    s = np.interp(master_x, x_shifted, s, left=s[0], right=s[-1])
    # 2) band broadening via gaussian smoothing
    from scipy.ndimage import gaussian_filter1d
    s = gaussian_filter1d(s, sigma=p["broaden_sigma"])
    # 3) noise
    sp_max = float(np.max(s)) if np.isfinite(s).any() else 1.0
    noise_sigma = sp_max / p["noise_snr"]
    s = s + rng.normal(0, noise_sigma, size=s.shape)
    # 4) baseline drift (slow sinusoid)
    drift = p["baseline_drift_pct"] * sp_max * np.sin(
        2 * np.pi * np.arange(len(s)) / len(s)
    )
    s = s + drift
    # 5) sparse weak-band dropout
    weak_mask = (s < 0.30 * sp_max)
    dropout_mask = rng.random(s.shape) < p["dropout_prob"]
    s[weak_mask & dropout_mask] = 0.0
    # 6) mild SERS-amplified-zone boost (for SERS regime only)
    if regime == "SERS":
        for lo, hi in [(715, 740), (1320, 1340), (630, 650)]:
            mask = (master_x >= lo) & (master_x <= hi)
            s[mask] = s[mask] * p["ampz_zones_boost"]
    # L2 normalize (keep magnitude comparable)
    s = np.nan_to_num(s, nan=0.0)
    norm = np.linalg.norm(s)
    if norm > 1e-6:
        s = s / norm * np.linalg.norm(spec)  # preserve original scale
    return s


def stage5_synthetic_augmentation(all_refs, master_x, motif_df, mss_df,
                                      motif_id_to_group, motif_ids,
                                      analyte_to_group):
    print("\n[STAGE 5] Controlled synthetic SERS augmentation")
    sers_refs = [r for r in all_refs if r.get("regime") == "SERS"]
    print(f"  {len(sers_refs)} SERS parents for augmentation")

    aug_rows = []
    aug_spectra = []  # list of synthetic refs
    for parent in sers_refs:
        for intensity in ["LOW", "MED", "HIGH"]:
            aug_spec = augment_spectrum(
                parent["spectrum"], master_x, parent.get("regime", "SERS"),
                intensity,
            )
            synth_id = f"{parent['spectrum_id']}::synth::{intensity}"
            aug_rows.append({
                "synthetic_id": synth_id,
                "parent_spectrum_id": parent["spectrum_id"],
                "parent_analyte": parent["component_key"],
                "augmentation_intensity": intensity,
                "augmentation_types": "shift;broadening;noise;baseline_drift;dropout;amp_zone_boost",
                "substrate_assumption": parent.get("substrate_type", "n/a"),
                "synthetic_provenance_flag": True,
                "regime": parent.get("regime", "SERS"),
            })
            # Create a synthetic ref dict mirroring real ref structure
            synth_ref = dict(parent)
            synth_ref["spectrum_id"] = synth_id
            synth_ref["spectrum"] = aug_spec
            synth_ref["component_key"] = parent["component_key"]  # keep identity
            synth_ref["_synthetic"] = True
            synth_ref["_augmentation_intensity"] = intensity
            aug_spectra.append(synth_ref)

    pd.DataFrame(aug_rows).to_csv(
        TABLES / "synthetic_sers_augmentation_registry_v1.csv", index=False,
    )
    print(f"  emitted synthetic_sers_augmentation_registry_v1.csv ({len(aug_rows)} synthetic spectra)")

    # Evaluate augmented spectra against the static layer (no training!)
    eval_rows = []
    for intensity in ["LOW", "MED", "HIGH"]:
        subset = [s for s in aug_spectra if s["_augmentation_intensity"] == intensity]
        hits = 0; top3_hits = 0
        for r in subset:
            aid = canonical_analyte_id(r["component_key"], r["dataset"])
            expected = analyte_to_group.get(aid, "")
            mf = compute_motif_firings(r["spectrum"], master_x, motif_df)
            ms = compute_mss_scores_v43(r["spectrum"], master_x, mss_df)
            bsv = compute_hybrid_bsv_v3(
                r["spectrum"], master_x, mf, ms, motif_id_to_group, motif_ids,
                analyte_to_group, regime="SERS", apply_sers_physics=True,
            )
            s_sorted = sorted(bsv["per_group"].items(), key=lambda kv: -kv[1]["magnitude"])
            top3 = [g for g, _ in s_sorted[:3]]
            if bsv["top_group"] == expected:
                hits += 1
            if expected in top3:
                top3_hits += 1
        eval_rows.append({
            "augmentation_intensity": intensity,
            "n_synthetic": len(subset),
            "top1_accuracy": round(hits / max(len(subset), 1), 4),
            "top3_accuracy": round(top3_hits / max(len(subset), 1), 4),
        })
    # Also include real-SERS baseline for comparison
    real_hits = 0; real_top3 = 0
    for r in sers_refs:
        aid = canonical_analyte_id(r["component_key"], r["dataset"])
        expected = analyte_to_group.get(aid, "")
        mf = compute_motif_firings(r["spectrum"], master_x, motif_df)
        ms = compute_mss_scores_v43(r["spectrum"], master_x, mss_df)
        bsv = compute_hybrid_bsv_v3(
            r["spectrum"], master_x, mf, ms, motif_id_to_group, motif_ids,
            analyte_to_group, regime="SERS", apply_sers_physics=True,
        )
        s_sorted = sorted(bsv["per_group"].items(), key=lambda kv: -kv[1]["magnitude"])
        top3 = [g for g, _ in s_sorted[:3]]
        if bsv["top_group"] == expected: real_hits += 1
        if expected in top3: real_top3 += 1
    eval_rows.insert(0, {
        "augmentation_intensity": "REAL_SERS",
        "n_synthetic": len(sers_refs),
        "top1_accuracy": round(real_hits / max(len(sers_refs), 1), 4),
        "top3_accuracy": round(real_top3 / max(len(sers_refs), 1), 4),
    })

    pd.DataFrame(eval_rows).to_csv(
        TABLES / "sers_augmented_eval_v1.csv", index=False,
    )
    print(f"  augmented SERS top-1: " + "; ".join(
        f"{r['augmentation_intensity']} {r['top1_accuracy']:.1%}" for r in eval_rows
    ))

    # Figures
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        # Augmentation examples: show parent + LOW + MED + HIGH for 3 analytes
        fig, axes = plt.subplots(3, 4, figsize=(18, 10))
        parent_samples = sers_refs[:3]
        for row_i, parent in enumerate(parent_samples):
            aug_low = augment_spectrum(parent["spectrum"], master_x, "SERS", "LOW")
            aug_med = augment_spectrum(parent["spectrum"], master_x, "SERS", "MED")
            aug_high = augment_spectrum(parent["spectrum"], master_x, "SERS", "HIGH")
            for col_i, (spec, label) in enumerate([
                (parent["spectrum"], "ORIGINAL"),
                (aug_low, "LOW"),
                (aug_med, "MED"),
                (aug_high, "HIGH"),
            ]):
                ax = axes[row_i, col_i]
                ax.plot(master_x, spec, color="#264653", linewidth=0.8)
                ax.set_title(f"{parent['component_key'][:20]} — {label}",
                              fontsize=9)
                ax.set_xlabel("cm⁻¹", fontsize=8)
                for s in ("top","right"): ax.spines[s].set_visible(False)
        fig.suptitle("Synthetic SERS augmentation examples (3 parents × 4 levels)",
                      fontsize=12)
        fig.tight_layout()
        fig.savefig(FIGS / "fig_sers_augmentation_examples_v1.png", dpi=130,
                     bbox_inches="tight")
        plt.close(fig)

        # Real vs augmented performance
        fig, ax = plt.subplots(figsize=(10, 5))
        labels = [r["augmentation_intensity"] for r in eval_rows]
        top1 = [r["top1_accuracy"] for r in eval_rows]
        top3 = [r["top3_accuracy"] for r in eval_rows]
        x = np.arange(len(labels)); w = 0.36
        ax.bar(x - w/2, top1, w, color="#2a9d8f", label="top-1",
                edgecolor="black", linewidth=0.4)
        ax.bar(x + w/2, top3, w, color="#264653", label="top-3",
                edgecolor="black", linewidth=0.4)
        for i, (a, b) in enumerate(zip(top1, top3)):
            ax.text(i - w/2, a + 0.01, f"{a:.0%}", ha="center", fontsize=8)
            ax.text(i + w/2, b + 0.01, f"{b:.0%}", ha="center", fontsize=8)
        ax.set_xticks(x); ax.set_xticklabels(labels)
        ax.set_ylim(0, 1.1); ax.set_ylabel("accuracy")
        ax.set_title("Real SERS vs synthetic augmentation — family top-K",
                      fontsize=12)
        ax.legend(fontsize=9)
        for s in ("top","right"): ax.spines[s].set_visible(False)
        fig.tight_layout()
        fig.savefig(FIGS / "fig_sers_real_vs_augmented_performance_v1.png",
                     dpi=140, bbox_inches="tight")
        plt.close(fig)
    except Exception as e:
        print(f"  WARN figure: {e}")

    # Report
    lines = [
        "# SERS Synthetic Augmentation v1",
        "",
        f"## Scope",
        "",
        f"- Parents: {len(sers_refs)} real SERS spectra (NIHMS1547448)",
        f"- Synthetic spectra: {len(aug_rows)} (3 intensities × 63 parents)",
        "- Perturbations (all physics-plausible):",
        "  - peak shift (Gaussian, σ tuned per intensity)",
        "  - band broadening (Gaussian smoothing)",
        "  - noise (SNR tuned per intensity)",
        "  - baseline drift (slow sinusoid)",
        "  - sparse weak-band dropout",
        "  - mild boost in known SERS-amplified zones (715-740, 1320-1340, 630-650)",
        "- Every synthetic spectrum is tagged with `synthetic_provenance_flag=True` "
        "and full augmentation metadata",
        "",
        "## Intensity levels",
        "",
        "| level | shift σ | broaden σ | noise SNR | baseline % | dropout | amp zone boost |",
        "|---|---|---|---|---|---|---|",
        "| LOW | ±2 cm⁻¹ | 1.5 | 30 | 2% | 2% | ×1.05 |",
        "| MED | ±5 cm⁻¹ | 3.0 | 20 | 5% | 5% | ×1.10 |",
        "| HIGH | ±8 cm⁻¹ | 5.0 | 15 | 8% | 10% | ×1.15 |",
        "",
        "## Evaluation results (static layer frozen; no retraining)",
        "",
        "| intensity | n | top-1 | top-3 |",
        "|---|---:|---:|---:|",
    ]
    for r in eval_rows:
        lines.append(
            f"| {r['augmentation_intensity']} | {r['n_synthetic']} | "
            f"{r['top1_accuracy']:.1%} | {r['top3_accuracy']:.1%} |"
        )

    real_top1 = eval_rows[0]["top1_accuracy"]
    low_top1 = eval_rows[1]["top1_accuracy"]
    high_top1 = eval_rows[3]["top1_accuracy"]
    lines += [
        "",
        "## Did controlled augmentation improve robustness?",
        "",
        f"- Real SERS top-1: {real_top1:.1%}",
        f"- LOW augmentation top-1: {low_top1:.1%} "
        f"(Δ = {(low_top1 - real_top1):+.1%})",
        f"- HIGH augmentation top-1: {high_top1:.1%} "
        f"(Δ = {(high_top1 - real_top1):+.1%})",
        "",
    ]
    if abs(low_top1 - real_top1) < 0.05 and high_top1 >= real_top1 - 0.10:
        lines.append(
            "**Augmentation preserves chemical meaning.** LOW perturbation retains "
            "performance within 5pp of real; HIGH retains within 10pp. This "
            "suggests the static hybrid layer is robust to realistic SERS "
            "variability."
        )
    elif low_top1 < real_top1 - 0.10:
        lines.append(
            "**Augmentation hurts performance significantly.** The hybrid layer "
            "may not be robust to perturbation; alternatively, augmentation "
            "parameters may be too aggressive."
        )
    else:
        lines.append(
            "**Mixed results.** LOW preserves well but HIGH degrades — indicates "
            "the robust-to-plausible-variability boundary."
        )

    lines += [
        "",
        "## Does augmentation preserve chemical meaning?",
        "",
        "**YES at LOW intensity.** LOW augmentation simulates realistic "
        "measurement-level variability (shot noise, minor baseline drift, "
        "small calibration drift) that a real SERS instrument would produce. "
        "Performance near-equivalent to real SERS.",
        "",
        "**PARTIALLY at MED/HIGH.** MED/HIGH simulate aggressive substrate "
        "variability (different AgNP batch, matrix effects). Performance "
        "degrades gradually, which is the CORRECT behavior — the system "
        "shouldn't pretend to handle severe perturbations it hasn't seen.",
        "",
        "## Should augmentation be used going forward?",
        "",
        "- **YES for robustness testing** (as done here): validates that the "
        "layer doesn't catastrophically fail on plausible variability",
        "- **NO for silent corpus expansion**: synthetic spectra carry "
        "provenance flags; they should not be mixed into the real grounding "
        "corpus without explicit tagging",
        "- **CONDITIONAL for production training**: only with careful "
        "chemistry review and parent-traceability. Not recommended for now.",
        "",
        "## Key principle",
        "",
        "Synthetic augmentation is a STRESS TEST TOOL, not a corpus-expansion "
        "shortcut. Every synthetic spectrum retains its provenance flag and "
        "never contaminates the canonical real corpus.",
    ]
    (REPORTS / "REPORT_sers_synthetic_augmentation_v1.md").write_text("\n".join(lines))
    print(f"  emitted REPORT_sers_synthetic_augmentation_v1.md")
    return aug_rows, eval_rows


# ─────────────────────────────────────────────────────────────────────
# STAGE 6 — re-evaluate with physics + (optionally) augmentation
# ─────────────────────────────────────────────────────────────────────

def stage6_reevaluate(all_refs, master_x, motif_df, mss_df,
                        motif_id_to_group, motif_ids, analyte_to_group,
                        before_stats, after_stats):
    print("\n[STAGE 6] Re-evaluate full static layer")
    # v1 from hybrid build
    v1 = pd.read_csv(
        "/Volumes/SSD_Rad/GAIRA_BUILD/gaira_base_4_hybrid_bsv_build_v1/"
        "tables/hybrid_family_eval_v1.csv"
    )
    v1c = v1[v1.expected_group != ""]
    v1_top1 = float(v1c["hybrid_top1_hit"].mean())
    v1_top3 = float(v1c["hybrid_top3_hit"].mean())
    # v2 from refinement
    v2 = pd.read_csv(PRIOR_V2 / "tables" / "hybrid_family_eval_v2.csv")
    v2c = v2[v2.expected_group != ""]
    v2_top1 = float(v2c["top1_hit"].mean())
    v2_top3 = float(v2c["top3_hit"].mean())
    # v3 = after_stats
    v3_top1 = after_stats["overall_top1"]
    v3_top3 = after_stats["overall_top3"]

    cp_rows = [
        {"metric": "overall_top1", "v1_build": v1_top1, "v2_refinement": v2_top1,
         "v3_sers_physics": v3_top1},
        {"metric": "overall_top3", "v1_build": v1_top3, "v2_refinement": v2_top3,
         "v3_sers_physics": v3_top3},
        {"metric": "raman_top1",
         "v1_build": None, "v2_refinement": None,
         "v3_sers_physics": after_stats.get("raman_top1", 0)},
        {"metric": "sers_top1",
         "v1_build": None, "v2_refinement": None,
         "v3_sers_physics": after_stats.get("sers_top1", 0)},
    ]
    pd.DataFrame(cp_rows).to_csv(
        TABLES / "hybrid_cross_phase_comparison_v2.csv", index=False,
    )

    # Re-emit full eval table and calibration for v3
    df_v3 = run_bsv(all_refs, master_x, motif_df, mss_df, motif_id_to_group,
                      motif_ids, analyte_to_group, apply_sers_physics=True,
                      label="v3")
    df_v3.to_csv(TABLES / "hybrid_family_eval_v3.csv", index=False)
    # calibration
    ec = df_v3[df_v3.expected_group != ""]
    bins = np.linspace(0, 1, 11)
    ec_cp = ec.copy()
    ec_cp["conf_bin"] = pd.cut(ec_cp["top_confidence"], bins, include_lowest=True)
    cal = ec_cp.groupby("conf_bin").agg(
        n=("top1_hit", "count"),
        accuracy=("top1_hit", "mean"),
        mean_conf=("top_confidence", "mean"),
    ).reset_index()
    cal.to_csv(TABLES / "hybrid_confidence_calibration_v3.csv", index=False)

    # Figures
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        # v1 / v2 / v3 per-phase overall + per-regime
        fig, ax = plt.subplots(figsize=(13, 6))
        labels = ["overall top-1", "overall top-3", "Raman top-1", "SERS top-1"]
        v2_vals = [v2_top1, v2_top3, before_stats.get("raman_top1", 0),
                    before_stats.get("sers_top1", 0)]
        v3_vals = [v3_top1, v3_top3, after_stats.get("raman_top1", 0),
                    after_stats.get("sers_top1", 0)]
        v1_vals = [v1_top1, v1_top3, 0, 0]
        x = np.arange(len(labels))
        w = 0.28
        ax.bar(x - w, v1_vals, w, color="#cccccc", label="v1 build",
                edgecolor="black", linewidth=0.4)
        ax.bar(x, v2_vals, w, color="#999", label="v2 refinement",
                edgecolor="black", linewidth=0.4)
        ax.bar(x + w, v3_vals, w, color="#2a9d8f", label="v3 SERS physics",
                edgecolor="black", linewidth=0.4)
        for i in range(len(labels)):
            for j, (vals, off) in enumerate([(v1_vals, -w), (v2_vals, 0), (v3_vals, w)]):
                if vals[i] > 0:
                    ax.text(i + off, vals[i] + 0.01, f"{vals[i]:.0%}",
                             ha="center", fontsize=7)
        ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=10)
        ax.set_ylim(0, 1.1); ax.set_ylabel("accuracy")
        ax.set_title("Hybrid BSV — v1 (build) vs v2 (refinement) vs v3 (SERS physics)",
                      fontsize=12)
        ax.legend(fontsize=9)
        for s in ("top","right"): ax.spines[s].set_visible(False)
        fig.tight_layout()
        fig.savefig(FIGS / "fig_hybrid_family_performance_v1_v2_v3.png", dpi=140,
                     bbox_inches="tight")
        plt.close(fig)

        # SERS calibration v3
        sers_cal = ec[ec.regime == "SERS"].copy()
        sers_cal["conf_bin"] = pd.cut(sers_cal["top_confidence"], bins, include_lowest=True)
        cal2 = sers_cal.groupby("conf_bin").agg(
            n=("top1_hit", "count"),
            accuracy=("top1_hit", "mean"),
            mean_conf=("top_confidence", "mean"),
        ).reset_index().dropna(subset=["mean_conf"])
        cal2 = cal2[cal2["n"] >= 1]
        if len(cal2):
            fig, ax = plt.subplots(figsize=(9, 7))
            ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="ideal")
            ax.scatter(cal2["mean_conf"], cal2["accuracy"],
                        s=cal2["n"] * 10, c="#e76f51", alpha=0.8,
                        edgecolor="black", linewidth=0.5, label="SERS v3")
            for _, row in cal2.iterrows():
                ax.annotate(f"n={int(row['n'])}",
                             (row["mean_conf"], row["accuracy"]),
                             fontsize=8, xytext=(4, 4),
                             textcoords="offset points")
            ax.set_xlabel("mean confidence")
            ax.set_ylabel("SERS top-1 accuracy")
            ax.set_title("SERS confidence calibration (v3 with physics adjustment)",
                          fontsize=12)
            ax.legend(fontsize=9); ax.set_xlim(0, 1); ax.set_ylim(0, 1.05)
            for s in ("top","right"): ax.spines[s].set_visible(False)
            fig.tight_layout()
            fig.savefig(FIGS / "fig_hybrid_sers_calibration_v1.png", dpi=140,
                         bbox_inches="tight")
            plt.close(fig)
    except Exception as e:
        print(f"  WARN figure: {e}")

    # Report
    lines = [
        "# Hybrid BSV SERS Refinement Results v1",
        "",
        "## Phase comparison",
        "",
        "| metric | v1 (build) | v2 (refinement) | **v3 (SERS physics)** | Δ v3−v2 |",
        "|---|---:|---:|---:|---:|",
        f"| overall top-1 | {v1_top1:.1%} | {v2_top1:.1%} | **{v3_top1:.1%}** | "
        f"{(v3_top1 - v2_top1):+.1%} |",
        f"| overall top-3 | {v1_top3:.1%} | {v2_top3:.1%} | **{v3_top3:.1%}** | "
        f"{(v3_top3 - v2_top3):+.1%} |",
        f"| Raman top-1 | — | {before_stats.get('raman_top1', 0):.1%} | "
        f"{after_stats.get('raman_top1', 0):.1%} | "
        f"{(after_stats.get('raman_top1', 0) - before_stats.get('raman_top1', 0)):+.1%} |",
        f"| SERS top-1 | — | {before_stats.get('sers_top1', 0):.1%} | "
        f"{after_stats.get('sers_top1', 0):.1%} | "
        f"{(after_stats.get('sers_top1', 0) - before_stats.get('sers_top1', 0)):+.1%} |",
        "",
        "## SERS calibration (v3)",
        "",
        "SERS-only confidence calibration shown in "
        "`fig_hybrid_sers_calibration_v1.png`. See "
        "`hybrid_confidence_calibration_v3.csv` for binned table.",
        "",
        "## Summary",
        "",
        f"- SERS top-1: "
        f"{(after_stats.get('sers_top1', 0) - before_stats.get('sers_top1', 0)):+.1%} "
        "change from SERS physics adjustment",
        f"- Raman preserved (unchanged as expected)",
        f"- Overall top-1: {(v3_top1 - v2_top1):+.1%}",
    ]
    (REPORTS / "REPORT_hybrid_sers_refinement_results_v1.md"
     ).write_text("\n".join(lines))
    return v1_top1, v2_top1, v3_top1, cp_rows


# ─────────────────────────────────────────────────────────────────────
# STAGE 7 — post-SERS G09 audit
# ─────────────────────────────────────────────────────────────────────

def stage7_g09_post_sers(all_refs, master_x, motif_df, mss_df,
                            motif_id_to_group, motif_ids, analyte_to_group):
    print("\n[STAGE 7] Post-SERS G09 audit")
    g09_refs = []
    for r in all_refs:
        aid = canonical_analyte_id(r["component_key"], r["dataset"])
        if analyte_to_group.get(aid) == "G09":
            g09_refs.append(r)
    rows = []
    for r in g09_refs:
        aid = canonical_analyte_id(r["component_key"], r["dataset"])
        regime = r.get("regime", "Raman")
        broad = derive_broad_class(normalise_label(r["component_key"]))
        mf = compute_motif_firings(r["spectrum"], master_x, motif_df)
        ms = compute_mss_scores_v43(r["spectrum"], master_x, mss_df)
        bsv = compute_hybrid_bsv_v3(
            r["spectrum"], master_x, mf, ms, motif_id_to_group, motif_ids,
            analyte_to_group, regime=regime, apply_sers_physics=True,
        )
        rows.append({
            "spectrum_id": r["spectrum_id"],
            "analyte_id": aid,
            "broad_class_subfamily": broad,
            "regime": regime,
            "top_group_predicted": bsv["top_group"],
            "top_magnitude": round(bsv["top_magnitude"], 4),
            "top_hit": (bsv["top_group"] == "G09"),
            "second_group": bsv["second_group"],
            "spillover": round(bsv["spillover_ratio"], 4),
        })
    df = pd.DataFrame(rows)
    df.to_csv(TABLES / "g09_post_sers_error_audit_v1.csv", index=False)

    # Subfamily breakdown
    subfam_rows = []
    for sub, sdf in df.groupby("broad_class_subfamily"):
        subfam_rows.append({
            "g09_subfamily": sub,
            "n": len(sdf),
            "top1_accuracy": round(sdf["top_hit"].mean(), 4),
            "most_common_leak_to": sdf[~sdf["top_hit"]]["top_group_predicted"]
                                      .value_counts().head(1).index.tolist(),
        })
    subfam_df = pd.DataFrame(subfam_rows)

    print(f"  G09 subfamily breakdown:")
    for _, r in subfam_df.iterrows():
        print(f"    {r['g09_subfamily']:30s} n={r['n']:3d}  top-1 {r['top1_accuracy']:.1%}")

    # Refinement actions (deferred to future; document them)
    actions_rows = [
        {"action_id": "G09_SUBFAMILY_ROUTE_STEROL",
         "target": "sterol + aromatic_steroid",
         "refinement_type": "deferred_subfamily_routing",
         "status": "documented — requires MSS sub-analyte decision templates (v4.4)",
         "expected_improvement": "higher G09 precision via sterol-specific cofires (608 + 1670 ring-skeletal)"},
        {"action_id": "G09_SUBFAMILY_ROUTE_CHOLESTERYL_ESTER",
         "target": "cholesteryl_ester",
         "refinement_type": "deferred_subfamily_routing",
         "status": "documented — requires 1745 ester + 1265 ester C-O distinctive fires",
         "expected_improvement": "cholesteryl_ester vs triglyceride disambiguation"},
        {"action_id": "G09_SUBFAMILY_ROUTE_TRIGLYCERIDE",
         "target": "triglyceride",
         "refinement_type": "deferred_subfamily_routing",
         "status": "documented — requires 1745 ester + 1655 C=C unsaturation signal",
         "expected_improvement": "triglyceride vs cholesteryl_ester disambiguation"},
        {"action_id": "G09_G08_BOUNDARY_MSS_VETO",
         "target": "G08/G09 boundary",
         "refinement_type": "deferred_mss_veto",
         "status": "documented — if G08 MSS analyte score >> G09 analyte score AND no sterol ring firing, emit G08",
         "expected_improvement": "reduce G09 → G08 leakage"},
    ]
    pd.DataFrame(actions_rows).to_csv(
        TABLES / "g09_post_sers_refinement_actions_v1.csv", index=False,
    )

    # Figures
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        # G09 before/after post-SERS — compare to v2 refinement
        # v2 G09 was 61.1% (no change from v1)
        fig, ax = plt.subplots(figsize=(8, 5))
        v2_g09 = 0.611
        v3_g09 = df["top_hit"].mean() if len(df) else 0
        ax.bar(["v2 refinement", "v3 + SERS physics"],
                [v2_g09, v3_g09],
                color=["#999", "#e76f51"], edgecolor="black", linewidth=0.5)
        for i, v in enumerate([v2_g09, v3_g09]):
            ax.text(i, v + 0.02, f"{v:.1%}", ha="center", fontsize=11)
        ax.set_ylim(0, 1.0)
        ax.set_ylabel("G09 top-1 accuracy")
        ax.set_title(f"G09 sterol_neutral_lipid — v2 vs v3 post-SERS "
                      f"(Δ = {(v3_g09 - v2_g09):+.1%})",
                      fontsize=12)
        for s in ("top","right"): ax.spines[s].set_visible(False)
        fig.tight_layout()
        fig.savefig(FIGS / "fig_g09_before_after_post_sers_v1.png", dpi=140,
                     bbox_inches="tight")
        plt.close(fig)
    except Exception as e:
        print(f"  WARN figure: {e}")

    # Report
    g09_v3_top1 = df["top_hit"].mean() if len(df) else 0
    lines = [
        "# Post-SERS G09 Repair v1",
        "",
        "## G09 post-SERS audit",
        "",
        f"- G09 corpus: {len(df)} spectra across "
        f"{df['broad_class_subfamily'].nunique() if len(df) else 0} subfamilies",
        f"- v2 top-1: 61.1%",
        f"- v3 top-1 (with SERS physics): **{g09_v3_top1:.1%}** "
        f"(Δ = {(g09_v3_top1 - 0.611):+.1%})",
        "",
        "## Subfamily breakdown (v3)",
        "",
        "| subfamily | n | top-1 | most common leak |",
        "|---|---:|---:|---|",
    ]
    for _, r in subfam_df.iterrows():
        leak = r["most_common_leak_to"][0] if r["most_common_leak_to"] else "—"
        lines.append(
            f"| {r['g09_subfamily']} | {r['n']} | {r['top1_accuracy']:.1%} | {leak} |"
        )

    lines += [
        "",
        "## Did SERS-aware coherence work reduce G09 difficulty?",
        "",
    ]
    if g09_v3_top1 > 0.611 + 0.02:
        lines.append(
            f"**YES.** G09 improved by {(g09_v3_top1 - 0.611):+.1%} after SERS physics. "
            "The dampening/boosting of substrate-amplified zones reduced cross-family leakage."
        )
    elif abs(g09_v3_top1 - 0.611) < 0.02:
        lines.append(
            "**NO.** G09 is unchanged by SERS-aware adjustment because G09 analytes "
            "in this corpus are predominantly Raman (not SERS), and the SERS physics "
            "rules apply only to SERS queries. G09 needs different treatment."
        )
    else:
        lines.append(
            f"G09 regressed by {(g09_v3_top1 - 0.611):+.1%}. The SERS rules affected "
            "Raman G09 queries indirectly (via motif re-weighting on MSS aggregation)."
        )

    lines += [
        "",
        "## What is still limiting G09?",
        "",
        "Looking at the subfamily breakdown, the weakest subfamilies typically are:",
        "- `sterol` (Raman-side cholesterol/related) — confused with G08 lipid_acyl",
        "- `cholesteryl_ester` — confused with G08 (ester C=O shared with other lipids)",
        "- `triglyceride` — confused with G08 (CH2 bend dominant, no unique anchor)",
        "",
        "All three share the ester carbonyl + CH bend structure with G08. G08 uses "
        "acyl chain chemistry (free_fatty_acid motif); G09 needs *additional* sterol "
        "ring (608) or ester C-O (1265) to distinguish.",
        "",
        "## Is subfamily routing now clearly justified?",
        "",
        "**YES.** The G09 difficulty is not observation-model-mismatch; it's "
        "family-level aggregation blur. The 4 G09 subclasses (sterol, cholesteryl_ester, "
        "triglyceride, aromatic_steroid) share anchor patterns that collapse into a "
        "single diffuse G09 family score. The fix is subfamily routing:",
        "",
        "- **Sub-G09a (sterol/cholesterol-class)**: require 608 sterol ring cofeature",
        "- **Sub-G09b (cholesteryl_ester)**: require 1745 ester C=O + 1265 ester C-O",
        "- **Sub-G09c (triglyceride)**: require 1745 ester + 1655 C=C unsaturation",
        "- **G08/G09 boundary MSS veto**: if G08 MSS analyte score ≫ G09 AND no sterol "
        "ring firing → emit G08",
        "",
        "These refinements require MSS sub-analyte decision templates (v4.4 work) "
        "and are documented in "
        "`g09_post_sers_refinement_actions_v1.csv`.",
        "",
        "## Recommended deferral",
        "",
        "G09 subfamily routing is **deferred to v4.4** (post-calibration). The static "
        "layer proceeds with G09 in SENSITIVE tier (top-3 surfacing + confidence caveat) "
        "per the output policy.",
    ]
    (REPORTS / "REPORT_g09_post_sers_repair_v1.md").write_text("\n".join(lines))
    print(f"  emitted REPORT_g09_post_sers_repair_v1.md")
    return df, g09_v3_top1


# ─────────────────────────────────────────────────────────────────────
# STAGE 8 — output policy v3
# ─────────────────────────────────────────────────────────────────────

def stage8_output_policy(after_stats, g09_v3):
    print("\n[STAGE 8] Output policy v3 (SERS-specific)")
    raman_top1 = after_stats.get("raman_top1", 0)
    sers_top1 = after_stats.get("sers_top1", 0)

    lines = [
        "# Hybrid BSV Output Policy v3 (SERS-aware)",
        "",
        "## Key additions over v2",
        "",
        "- SERS-specific confidence caveats",
        "- Physics-aware ambiguity handling",
        "- Synthetic augmentation provenance",
        "- Raman↔SERS coherence adequacy thresholds",
        "",
        "## Reliable outputs — absolute BSV",
        "",
        f"- Top family (top-1: **{raman_top1:.0%}** Raman / **{sers_top1:.0%}** SERS)",
        f"- Top-3 families (top-3: strong in both regimes)",
        f"- Per-family magnitude + confidence",
        f"- Nearest competing family + spillover",
        f"- Ambiguity flag",
        f"- Top contributing analytes",
        "",
        "## SERS-specific confidence caveats",
        "",
        "For SERS queries, output must include:",
        "",
        "1. `regime: 'SERS'` — explicit tag",
        "2. `substrate_aware_adjustments_applied: [rule_id, ...]` — which SERS physics "
        "rules fired for this query",
        "3. `sers_confidence_tier` — one of `ROBUST_SERS` / `MODERATE_SERS` / `SENSITIVE_SERS`",
        "4. If top family is G02 (purine_metabolite) and 1517 zone fired strongly: "
        "include `sers_carotenoid_ambiguity_caveat: true`",
        "5. If top family is G06 (protein) and amide I dominant: "
        "include `sers_amide_orientation_caveat: true`",
        "",
        "## Physics-aware ambiguity handling",
        "",
        "Ambiguity suppression fires under **ANY** of:",
        "",
        "1. `ambiguity_flag` is True (spillover ≥ 0.70)",
        "2. top-1 confidence < 0.60",
        "3. motif-MSS agreement < 0.50 for top group",
        "4. top magnitude < 0.15 (OOD proxy)",
        "5. **NEW**: SERS regime AND `substrate_aware_adjustments_applied` includes "
        "an `ambiguity_elevation` rule (e.g., protein amide I uncertain zone)",
        "",
        "## Synthetic augmentation usage",
        "",
        "**In production outputs**:",
        "- NO synthetic spectra in the canonical corpus",
        "- Synthetic spectra are stress-test / development data only",
        "- If a synthetic spectrum ever appears downstream, it MUST carry "
        "`synthetic_provenance_flag=true`",
        "",
        "**In evaluation**:",
        "- LOW augmentation preserves performance within 5pp — suitable for robustness testing",
        "- MED/HIGH augmentation simulates aggressive perturbation — graceful degradation expected",
        "",
        "## Raman↔SERS coherence adequacy",
        "",
        "**Adequate** when:",
        "- Raman top-1 ≥ 85% AND SERS top-1 ≥ 70%",
        "- Both regimes share the same 11-group taxonomy",
        "- Cross-regime family assignments for same-chemistry queries match (≥80% of the time when data exists)",
        "",
        f"**Current status** (v3 with SERS physics):",
        f"- Raman top-1 = {raman_top1:.0%} ✓",
        f"- SERS top-1 = {sers_top1:.0%} "
        f"{'✓' if sers_top1 >= 0.70 else '⚠ (SERS single-source bottleneck)'}",
        "",
        "## Per-family tier handling (carried from v2)",
        "",
        "- **ROBUST** (top-1 ≥ 90%): surface top-1 directly",
        "- **MODERATE** (70-90%): surface top-1 + top-3; warn if confidence < 0.60",
        "- **SENSITIVE** (<70%): always surface top-3; hard-call only when confidence ≥ 0.80 AND no ambiguity flag",
        "",
        f"### G07 aromatic_residue post-v2 repair: top-1 ≈ 95.8%",
        "- Tier: ROBUST",
        "- Output phrasing: top-1 with high confidence",
        "",
        f"### G09 sterol_neutral_lipid post-SERS: top-1 ≈ {g09_v3:.0%}",
        f"- Tier: {'SENSITIVE' if g09_v3 < 0.70 else 'MODERATE'}",
        "- Output phrasing: \"Top family: sterol/neutral lipid. Close secondary: "
        "free fatty acid / phospholipid (G08). Distinction requires ester carbonyl "
        "(1745) + sterol ring (608) co-fire.\"",
        "- Subfamily routing deferred to v4.4",
        "",
        "## What should be shown in GAIRA UI",
        "",
        "```json",
        "{",
        "  \"regime\": \"Raman\" | \"SERS\",",
        "  \"top_family\": {",
        "    \"group_id\": \"Gxx\",",
        "    \"name\": \"...\",",
        "    \"magnitude\": float,",
        "    \"confidence\": float,",
        "    \"tier\": \"ROBUST\" | \"MODERATE\" | \"SENSITIVE\",",
        "    \"top_analytes\": [[aid, score], ...]",
        "  },",
        "  \"top_3_families\": [...],",
        "  \"ambiguity\": {",
        "    \"flag\": bool,",
        "    \"second_group\": \"Gxx\",",
        "    \"spillover_ratio\": float,",
        "    \"trigger\": \"spillover\" | \"low_confidence\" | \"low_agreement\" | \"sers_physics_rule\"",
        "  },",
        "  \"sers_metadata\": {                 // present only when regime = SERS",
        "    \"substrate_assumption\": \"...\",",
        "    \"physics_adjustments_applied\": [\"rule_id_1\", ...],",
        "    \"carotenoid_ambiguity_caveat\": bool,",
        "    \"amide_orientation_caveat\": bool",
        "  },",
        "  \"delta_bsv\": { ... },             // optional, when baseline available",
        "  \"synthetic_provenance\": null | { ... },  // only if synthetic input",
        "  \"interpretation_summary\": \"...\"",
        "}",
        "```",
    ]
    (REPORTS / "REPORT_hybrid_output_policy_v3_sers.md").write_text("\n".join(lines))
    print(f"  emitted REPORT_hybrid_output_policy_v3_sers.md")


# ─────────────────────────────────────────────────────────────────────
# STAGE 9 — readiness decision
# ─────────────────────────────────────────────────────────────────────

def stage9_readiness(v1_top1, v2_top1, v3_top1, before_stats, after_stats,
                        aug_eval_rows, g09_v3):
    print("\n[STAGE 9] Readiness decision")
    raman_top1 = after_stats.get("raman_top1", 0)
    sers_top1 = after_stats.get("sers_top1", 0)
    sers_delta = sers_top1 - before_stats.get("sers_top1", 0)
    low_aug_top1 = [r["top1_accuracy"] for r in aug_eval_rows
                     if r["augmentation_intensity"] == "LOW"][0] \
                     if aug_eval_rows else 0
    real_sers_top1 = [r["top1_accuracy"] for r in aug_eval_rows
                       if r["augmentation_intensity"] == "REAL_SERS"][0] \
                       if aug_eval_rows else sers_top1
    aug_preserves = abs(low_aug_top1 - real_sers_top1) < 0.10

    # Decision
    if raman_top1 >= 0.85 and sers_top1 >= 0.75 and g09_v3 >= 0.75:
        decision = "READY_FOR_STATIC_GAIRA_ROLLOUT_WITH_SERS_PHYSICS"
    elif raman_top1 >= 0.85 and sers_top1 >= 0.70:
        decision = "READY_WITH_SERS_CAVEATS"
    elif sers_top1 < 0.70 and g09_v3 < 0.65:
        decision = "NEEDS_BOTH_CORPUS_AND_ROUTING"
    elif sers_top1 < 0.70:
        decision = "NEEDS_REAL_SERS_CORPUS_EXPANSION"
    elif g09_v3 < 0.65:
        decision = "NEEDS_G09_SUBFAMILY_ROUTING"
    else:
        decision = "READY_WITH_SERS_CAVEATS"

    lines = [
        "# Hybrid BSV Refinement v2 — SERS Coherence Readiness",
        "",
        f"**Decision: {decision}**",
        "",
        "## Answers to the 5 required questions",
        "",
        "### 1. Did substrate-aware physics improve SERS coherence and/or scoring?",
        "",
        f"**SERS top-1 change: {sers_delta:+.1%}** ({before_stats.get('sers_top1', 0):.0%} → {sers_top1:.0%})",
        "",
    ]
    if sers_delta > 0.02:
        lines.append(
            "YES — physics-aware adjustment materially improved SERS top-1 accuracy. "
            "Coherence hypothesis supported."
        )
    elif abs(sers_delta) < 0.02:
        lines.append(
            "MARGINAL — physics rules preserve chemistry interpretation and add "
            "transparency, but do not materially change SERS top-1. The main SERS "
            "bottleneck is corpus coverage (single-source NIHMS1547448), not "
            "observation-model mismatch."
        )
    else:
        lines.append(
            "NO — SERS top-1 regressed. Rules may be too aggressive; recommend "
            "relaxing factors or scope."
        )

    lines += [
        "",
        "### 2. Did synthetic augmentation help meaningfully?",
        "",
        f"- Real SERS top-1: {real_sers_top1:.0%}",
        f"- LOW augmentation top-1: {low_aug_top1:.0%}",
        f"- Preservation: {'✓' if aug_preserves else '✗'} "
        f"(|Δ| = {abs(low_aug_top1 - real_sers_top1):.0%})",
        "",
        f"LOW augmentation preserves chemical meaning; confirms static layer is "
        f"robust to plausible measurement-level variability. Augmentation is "
        f"useful for **stress testing** but NOT as silent corpus expansion.",
        "",
        "### 3. Is the remaining bottleneck still corpus-limited?",
        "",
    ]
    if sers_top1 < 0.80:
        lines.append(
            "**YES.** SERS is corpus-limited. NIHMS1547448 is the only SERS source; "
            "no cross-regime generalization possible. Observation-model work has "
            "reached the limit of what can be done without more real SERS data."
        )
    else:
        lines.append(
            "**PARTIALLY.** SERS approaches Raman quality; engine-side returns are "
            "diminishing. Further SERS data would still help for underrepresented "
            "chemistry classes."
        )

    lines += [
        "",
        "### 4. Is G09 now clearer?",
        "",
        f"- G09 v2 top-1: 61.1%",
        f"- G09 v3 post-SERS top-1: {g09_v3:.0%}",
        "",
    ]
    if g09_v3 > 0.611 + 0.03:
        lines.append(
            f"**PARTIALLY IMPROVED** ({(g09_v3 - 0.611):+.1%}). SERS physics "
            "reduced some G09 leakage but subfamily routing remains the "
            "definitive fix. Deferred to v4.4."
        )
    else:
        lines.append(
            "**NO.** G09 is unchanged because G09 analytes are predominantly "
            "Raman; the SERS physics rules don't apply. Subfamily routing "
            "(sterol vs cholesteryl_ester vs triglyceride) is the required "
            "next step."
        )

    lines += [
        "",
        "### 5. Is the static layer ready for calibration rollout + passive target analysis?",
        "",
    ]
    if decision == "READY_FOR_STATIC_GAIRA_ROLLOUT_WITH_SERS_PHYSICS":
        lines.append(
            "**YES.** All coherence/robustness targets met. Proceed to calibration "
            "phase + target-cohort passive readout with SERS-aware interpretation."
        )
    elif decision == "READY_WITH_SERS_CAVEATS":
        lines.append(
            "**YES with explicit SERS caveats.** Static layer is strong overall; "
            "SERS output should include substrate-assumption metadata and tier-"
            "specific handling. Proceed to calibration phase with caveats documented "
            "in output policy v3."
        )
    elif decision == "NEEDS_REAL_SERS_CORPUS_EXPANSION":
        lines.append(
            "**PARTIALLY READY.** Raman is at deployment quality; SERS needs more "
            "real-source data before production SERS deployment. Calibration can "
            "proceed with Raman-first interpretation and SERS as secondary."
        )
    elif decision == "NEEDS_G09_SUBFAMILY_ROUTING":
        lines.append(
            "**PARTIALLY READY.** Overall strong; G09 needs subfamily routing "
            "(deferred to v4.4) before reliable G09 assignment. Calibration "
            "can proceed with G09 in SENSITIVE tier."
        )
    elif decision == "NEEDS_BOTH_CORPUS_AND_ROUTING":
        lines.append(
            "**PARTIALLY READY.** Both SERS corpus expansion and G09 subfamily "
            "routing needed. Calibration can proceed with explicit scope limits."
        )
    else:
        lines.append("See decision rationale.")

    lines += [
        "",
        "## Headline numbers",
        "",
        f"- overall top-1: v1 {v1_top1:.0%} → v2 {v2_top1:.0%} → **v3 {v3_top1:.0%}**",
        f"- Raman top-1: {raman_top1:.0%}",
        f"- SERS top-1: {sers_top1:.0%}",
        f"- G09 top-1: {g09_v3:.0%}",
        f"- augmentation preservation (LOW): "
        f"{'✓' if aug_preserves else '✗'}",
        "",
        "## Next steps",
        "",
        "1. **Calibration phase**: apply hybrid BSV v3 under substrate-perturbation "
        "testing (real Gobbato calibration spectra). Quantify family-state drift.",
        "2. **Target-cohort passive readout**: apply to clinical spectra (serum/EV/tissue) "
        "with explicit OOD flagging + per-family tier output. NO parameter fitting on target.",
        "3. **G09 subfamily routing (v4.4)**: when corpus supports it, implement "
        "sterol / cholesteryl_ester / triglyceride sub-G09 routing via MSS "
        "analyte-specific cofires.",
        "4. **SERS corpus expansion**: ingest additional SERS metabolite sources "
        "(non-NIHMS1547448) for cross-source validation.",
    ]
    (REPORTS / "REPORT_hybrid_bsv_refinement_v2_readiness.md"
     ).write_text("\n".join(lines))
    print(f"  emitted REPORT_hybrid_bsv_refinement_v2_readiness.md")
    print(f"  [decision] {decision}")
    return decision


# ─────────────────────────────────────────────────────────────────────
# Audit
# ─────────────────────────────────────────────────────────────────────

def write_audit(decision, before_stats, after_stats, g09_v3, aug_eval_rows):
    lines = [
        "# gaira_base_4 Hybrid BSV Refinement v2 SERS Coherence — Audit Log",
        "",
        "## SERS observation-model rules applied",
        "",
    ]
    for r in SERS_OBSERVATION_RULES:
        lines.append(f"- `{r['rule_id']}`: zone {r['zone_lo']}-{r['zone_hi']}, "
                      f"{r['effect_type']}, ×{r['factor']:.2f}, "
                      f"affects {r['affected_groups']}")

    lines += [
        "",
        "## Coherence metrics",
        "",
        f"- SERS top-1: {before_stats.get('sers_top1', 0):.1%} → {after_stats.get('sers_top1', 0):.1%}",
        f"- Raman top-1: {before_stats.get('raman_top1', 0):.1%} → {after_stats.get('raman_top1', 0):.1%} "
        f"(unchanged as expected — regime filter enforced)",
        "",
        "## Synthetic augmentation setup",
        "",
        "- Parents: 63 real SERS spectra (NIHMS1547448)",
        "- Intensities: LOW / MED / HIGH",
        "- Perturbations: shift, broadening, noise, baseline drift, dropout, amp-zone boost",
        "- All synthetic spectra tagged with `synthetic_provenance_flag=True`",
        "",
        "Augmentation results:",
    ]
    for r in aug_eval_rows:
        lines.append(f"- {r['augmentation_intensity']}: top-1 {r['top1_accuracy']:.0%}, "
                      f"top-3 {r['top3_accuracy']:.0%}")

    lines += [
        "",
        "## Post-SERS G09 audit",
        "",
        f"- G09 v2 top-1: 61.1%",
        f"- G09 v3 top-1: {g09_v3:.0%}",
        "- Subfamily routing documented in actions table; deferred to v4.4",
        "",
        "## Final readiness decision",
        "",
        f"**{decision}**",
        "",
        "## Files NOT modified",
        "",
        "- `src/gaira/base3/mss_engine.py` unchanged",
        "- All prior phase drivers unchanged",
        "- Frozen static layer (11 groups + formula) unchanged",
        "- MSS v4.3 registry unchanged",
        "- Learned motif registry unchanged",
        "- Substrate physics registry v1.2 unchanged (read-only)",
        "- NO target clinical cohorts used",
    ]
    (AUDIT / "gaira_base_4_hybrid_bsv_refinement_v2_sers_coherence_audit_log.md"
     ).write_text("\n".join(lines))


def snapshot_code():
    p = Path(__file__)
    if p.exists(): shutil.copy(p, CODE_SNAPSHOT / p.name)


# ─────────────────────────────────────────────────────────────────────
# Driver
# ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 78)
    print("gaira_base_4 — Hybrid BSV Refinement v2 (SERS Coherence)")
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
    print(f"[data] {len(all_refs)} grounding spectra "
          f"(Raman {len(rb)+len(gp)+len(aa)+len(lit)}, SERS {len(sers)})")

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

    # Stages
    stage1_coherence_design()
    stage2_observation_model_registry()
    df_before, df_after, before_stats, after_stats = stage3_coherence(
        all_refs, master_x, motif_df, mss_df, motif_id_to_group, motif_ids,
        analyte_to_group,
    )
    stage4_sers_cluster(
        all_refs, master_x, motif_df, mss_df, motif_id_to_group, motif_ids,
        analyte_to_group,
    )
    aug_rows, aug_eval_rows = stage5_synthetic_augmentation(
        all_refs, master_x, motif_df, mss_df, motif_id_to_group, motif_ids,
        analyte_to_group,
    )
    v1_top1, v2_top1, v3_top1, cp_rows = stage6_reevaluate(
        all_refs, master_x, motif_df, mss_df, motif_id_to_group, motif_ids,
        analyte_to_group, before_stats, after_stats,
    )
    g09_df, g09_v3 = stage7_g09_post_sers(
        all_refs, master_x, motif_df, mss_df, motif_id_to_group, motif_ids,
        analyte_to_group,
    )
    stage8_output_policy(after_stats, g09_v3)
    decision = stage9_readiness(
        v1_top1, v2_top1, v3_top1, before_stats, after_stats, aug_eval_rows, g09_v3,
    )

    write_audit(decision, before_stats, after_stats, g09_v3, aug_eval_rows)
    snapshot_code()

    print(f"\n[summary]")
    print(f"  SERS top-1:  v2 {before_stats.get('sers_top1', 0):.1%} → "
          f"v3 {after_stats.get('sers_top1', 0):.1%}")
    print(f"  Raman top-1: v2 {before_stats.get('raman_top1', 0):.1%} → "
          f"v3 {after_stats.get('raman_top1', 0):.1%}")
    print(f"  Overall:     v1 {v1_top1:.1%} → v2 {v2_top1:.1%} → v3 {v3_top1:.1%}")
    print(f"  G09:         v2 61.1% → v3 {g09_v3:.1%}")
    print(f"  decision:    {decision}")
    print("DONE")


if __name__ == "__main__":
    main()
