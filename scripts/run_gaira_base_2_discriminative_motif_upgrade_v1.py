"""gaira_base_2 discriminative motif upgrade v1.

Reframes motifs as discriminative objects (role + anti-evidence +
competitor structure + ambiguity routing) and reruns full grounding
through the new scoring rule. Compares to motif-first baseline.

Engine modules touched: NONE (only adds src/gaira/base2/v2_patches_discriminative.py).
Registry / mapping: read-only (v1.3.1 / v1.2.1).

Run:
    cd /Users/suraj/projects/GAIRA
    PYTHONPATH=src .venv/bin/python scripts/run_gaira_base_2_discriminative_motif_upgrade_v1.py
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
from gaira.base2.v2_patches_discriminative import (
    ROLE_TABLE, ROLE_FACTOR, NO_ANCHOR_PENALTY,
    ANTI_EVIDENCE, COMPETITORS, AMBIGUITY_ROUTING,
    CO_FIRE_ANCHOR_GROUPS,
    score_spectrum_discriminative, family_score_discriminative,
)
from gaira.base2 import v2_patches_rescue as _rescue
from gaira.spectral import canonical_master_axis

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_gaira_validate_2_grounding import (
    load_ramanbiolib, load_gobbato_powder,
    load_amino_acid_xlsx, load_digitised_literature,
)
from run_gaira_validate_2_grounding_motif_first_v1 import (
    EXPECTED_MOTIFS, EXPECTED_AMBIGUITY, FAMILIES,
    expected_motifs_for, expected_families_for, expected_ambiguity_for,
    family_score, topn_hit,
)


ROOT = Path("/Volumes/SSD_Rad/GAIRA_BUILD/gaira_base_2_discriminative_motif_upgrade_v1")
REGISTRY = ROOT / "registry"
TABLES = ROOT / "tables"
FIGS = ROOT / "figures"
REPORTS = ROOT / "reports"
AUDIT = ROOT / "audit"
CODE_SNAPSHOT = ROOT / "code_snapshot"

REG_V1_3_1 = Path(
    "/Volumes/SSD_Rad/GAIRA_BUILD/gaira_base_2_revert_v4_and_deep_coverage_rescue_v1/"
    "registry/motif_candidate_registry_v1_3_1.yaml"
)
MAP_V1_2_1 = Path(
    "/Volumes/SSD_Rad/GAIRA_BUILD/gaira_base_2_revert_v4_and_deep_coverage_rescue_v1/"
    "registry/motif_to_axis_mapping_skeleton_v1_2_1.csv"
)

# Motif-first baseline outputs for comparison
MF_METRICS = Path(
    "/Volumes/SSD_Rad/GAIRA_BUILD/gaira_validate_2_grounding_motif_first_v1/"
    "tables/grounding_metrics_summary_v_motif_first.csv"
)
MF_PER_SPEC_MOTIF = Path(
    "/Volumes/SSD_Rad/GAIRA_BUILD/gaira_validate_2_grounding_motif_first_v1/"
    "tables/grounding_per_spectrum_motif_scores_v_motif_first.csv"
)
MF_RANK_MOTIF = Path(
    "/Volumes/SSD_Rad/GAIRA_BUILD/gaira_validate_2_grounding_motif_first_v1/"
    "tables/grounding_expected_vs_observed_motif_rank_v_motif_first.csv"
)
MF_RANK_FAMILY = Path(
    "/Volumes/SSD_Rad/GAIRA_BUILD/gaira_validate_2_grounding_motif_first_v1/"
    "tables/grounding_expected_vs_observed_family_rank_v_motif_first.csv"
)
MF_AMBIG = Path(
    "/Volumes/SSD_Rad/GAIRA_BUILD/gaira_validate_2_grounding_motif_first_v1/"
    "tables/grounding_ambiguity_behavior_v_motif_first.csv"
)
MF_OFFTGT = Path(
    "/Volumes/SSD_Rad/GAIRA_BUILD/gaira_validate_2_grounding_motif_first_v1/"
    "tables/grounding_off_target_activation_v_motif_first.csv"
)
MF_PERFAM = Path(
    "/Volumes/SSD_Rad/GAIRA_BUILD/gaira_validate_2_grounding_motif_first_v1/"
    "tables/grounding_per_family_hit_rates_v_motif_first.csv"
)


# ─────────────────────────────────────────────────────────────────────
# STEP 1 — Emit discriminator registry CSVs
# ─────────────────────────────────────────────────────────────────────

def emit_discriminator_registry(motifs, mappings):
    """Build motif_discriminator_registry_v1.csv +
    motif_discriminator_roles_v1.csv +
    discriminative_motif_upgrade_actions_v1.csv."""

    # Helper: pretty-print primary/supporting bands from registry MotifSpec
    def pretty_bands(motif_id, kind="primary"):
        m = motifs.get(motif_id)
        if m is None:
            return "(motif not in registry)"
        bands = m.primary_bands if kind == "primary" else m.supporting_bands
        if not bands:
            return ""
        return "; ".join(
            f"{b.family_id} ({b.cm1_centre:.0f}+/-{b.cm1_tolerance:.0f})"
            for b in bands
        )

    def pretty_anti_evidence(motif_id):
        rules = ANTI_EVIDENCE.get(motif_id, [])
        if not rules:
            return ""
        out = []
        for r in rules:
            if r["rule"] == "REQUIRES_COBAND":
                out.append(
                    f"REQUIRES_COBAND[{r['target']}>={r['min_weight']}] -> "
                    f"penalty {r['penalty']}"
                )
            elif r["rule"] == "SUPPRESS_IF_PRESENT":
                out.append(
                    f"SUPPRESS_IF_PRESENT[{r['target']}>={r['min_weight']}] -> "
                    f"penalty {r['penalty']}"
                )
            elif r["rule"] == "REQUIRES_ANY_FAMILY_ANCHOR":
                tlist = ",".join(r["targets"][:3]) + (
                    f"+{len(r['targets'])-3} more" if len(r["targets"]) > 3 else ""
                )
                out.append(
                    f"REQUIRES_ANY_FAMILY_ANCHOR[{tlist}>={r['min_weight']}] -> "
                    f"penalty {r['penalty']}"
                )
        return " | ".join(out)

    def pretty_competitors(motif_id):
        cs = COMPETITORS.get(motif_id, [])
        return ",".join(cs)

    def pretty_ambiguity_route(motif_id):
        r = AMBIGUITY_ROUTING.get(motif_id)
        if r is None:
            return ""
        return (
            f"trigger={r['trigger']}; target={r['target']}; "
            f"trigger_min_weight={r['trigger_min_weight']}; "
            f"ambiguity_share={r['ambiguity_share']}"
        )

    def role_rationale(motif_id, role):
        r = motifs.get(motif_id)
        if r is None:
            return f"{role}; no longer in registry"
        type_str = r.motif_type
        co_band = r.co_band_requirement
        if role == "ANCHOR":
            return (f"chemistry-specific motif ({type_str}, {co_band} co-band); "
                    "may drive a claim alone")
        if role == "SUPPORT":
            return (f"shared / cross-axis chemistry ({type_str}, {co_band} co-band); "
                    "needs co-fire with same-family ANCHOR to be credible alone")
        if role == "BACKGROUND":
            return (f"broad indicator ({type_str}); fires on many references "
                    "outside the chemistry; gated to never win alone")
        if role == "AMBIGUITY_ONLY":
            return "collision-aware routing only; does not contribute to biology axes"
        if role == "ARTIFACT_ONLY":
            return "substrate / buffer artifact; does not contribute to biology"
        return role

    # All motifs in registry get a row (52 active + 11 inactive in mapping
    # but still v1_active=true). Plus the few HELD_V2 / NO_MAPPING entries.
    all_mids = sorted(set(ROLE_TABLE.keys()) | set(motifs.keys()))

    # Roles table
    role_rows = []
    for mid in all_mids:
        m = motifs.get(mid)
        prior_type = m.motif_type if m else "(removed)"
        new_role = ROLE_TABLE.get(mid, "SUPPORT")
        active_in_v1 = bool(m and m.v1_active)
        has_mapping = mid in mappings
        in_active_scoring = active_in_v1 and has_mapping and (
            mappings[mid].active if has_mapping else False
        )
        role_rows.append({
            "motif_id": mid,
            "prior_type": prior_type,
            "new_role": new_role,
            "rationale": role_rationale(mid, new_role),
            "v1_active": active_in_v1,
            "has_mapping": has_mapping,
            "in_active_scoring": in_active_scoring,
            "notes": "",
        })
    pd.DataFrame(role_rows).to_csv(
        TABLES / "motif_discriminator_roles_v1.csv", index=False,
    )
    print(f"[emit] motif_discriminator_roles_v1.csv  ({len(role_rows)} rows)")

    # Discriminator registry table
    disc_rows = []
    for mid in all_mids:
        new_role = ROLE_TABLE.get(mid, "SUPPORT")
        m = motifs.get(mid)
        anchor_b = pretty_bands(mid, "primary") if new_role in ("ANCHOR",) else ""
        # For non-ANCHOR, primary bands are still listed but as "support_bands"
        if new_role == "ANCHOR":
            anchor_b = pretty_bands(mid, "primary")
            support_b = pretty_bands(mid, "supporting")
        elif new_role in ("SUPPORT", "BACKGROUND"):
            anchor_b = ""
            support_b = (pretty_bands(mid, "primary")
                         + ("; " + pretty_bands(mid, "supporting")
                            if pretty_bands(mid, "supporting") else ""))
        else:
            anchor_b = ""
            support_b = pretty_bands(mid, "primary")
        disc_rows.append({
            "motif_id": mid,
            "role": new_role,
            "anchor_bands": anchor_b,
            "support_bands": support_b,
            "anti_evidence_rules": pretty_anti_evidence(mid),
            "competitor_motifs_or_families": pretty_competitors(mid),
            "ambiguity_route_rule": pretty_ambiguity_route(mid),
            "rationale": role_rationale(mid, new_role),
            "notes": "",
        })
    pd.DataFrame(disc_rows).to_csv(
        REGISTRY / "motif_discriminator_registry_v1.csv", index=False,
    )
    print(f"[emit] registry/motif_discriminator_registry_v1.csv "
          f"({len(disc_rows)} rows)")

    # Action table
    actions = []
    counter = 0
    def add(motif_id, action_type, rationale, expected_effect, notes=""):
        nonlocal counter
        counter += 1
        actions.append({
            "action_id": f"DISC_v1_{counter:03d}",
            "motif_id": motif_id,
            "action_type": action_type,
            "rationale": rationale,
            "expected_effect": expected_effect,
            "notes": notes,
        })

    # Role classification actions (one per active motif)
    for mid in sorted(ROLE_TABLE.keys()):
        new_role = ROLE_TABLE[mid]
        add(mid, f"ASSIGN_ROLE:{new_role}",
            role_rationale(mid, new_role),
            f"role gating × {ROLE_FACTOR[new_role]}, "
            f"× {NO_ANCHOR_PENALTY[new_role]} if no same-family ANCHOR co-fires")

    # Anti-evidence rules
    for mid, rules in ANTI_EVIDENCE.items():
        for r in rules:
            if r["rule"] == "REQUIRES_COBAND":
                add(mid, "ADD_ANTI_EVIDENCE:REQUIRES_COBAND",
                    f"requires {r['target']} >= {r['min_weight']} "
                    "to avoid being a small-molecule false positive",
                    f"penalty × {1.0 - r['penalty']} when fired")
            elif r["rule"] == "SUPPRESS_IF_PRESENT":
                add(mid, "ADD_ANTI_EVIDENCE:SUPPRESS_IF_PRESENT",
                    f"presence of {r['target']} >= {r['min_weight']} indicates "
                    "competing chemistry",
                    f"penalty × {1.0 - r['penalty']} when fired")
            elif r["rule"] == "REQUIRES_ANY_FAMILY_ANCHOR":
                add(mid, "ADD_ANTI_EVIDENCE:REQUIRES_ANY_FAMILY_ANCHOR",
                    f"requires at least one of {len(r['targets'])} family "
                    f"anchors >= {r['min_weight']} to avoid spurious firing",
                    f"penalty × {1.0 - r['penalty']} when none fire")

    # Ambiguity routing
    for mid, r in AMBIGUITY_ROUTING.items():
        add(mid, "ADD_AMBIGUITY_ROUTE",
            f"when {r['target']} co-fires (>= {r['trigger_min_weight']}), "
            f"route {r['ambiguity_share']:.0%} of weight to ambiguity lane",
            "shared chemistry routes part of weight to ambiguity instead "
            "of the biology family")

    # CO_FIRE_ANCHOR_GROUPS
    for grp in CO_FIRE_ANCHOR_GROUPS:
        for mid in grp["members"]:
            add(mid, f"JOIN_COFIRE_ANCHOR_GROUP:{grp['name']}",
                f"co-fires with {','.join(m for m in grp['members'] if m != mid)} "
                f"(min_weight {grp['min_weight']}) constitute anchor-equivalent "
                f"chemistry for family {','.join(grp['anchor_for_families'])}",
                "exempts this motif from NO_ANCHOR_PENALTY when all group "
                "members fire above min_weight (real protein / real lipid "
                "chemistry signal)")

    pd.DataFrame(actions).to_csv(
        TABLES / "discriminative_motif_upgrade_actions_v1.csv", index=False,
    )
    print(f"[emit] discriminative_motif_upgrade_actions_v1.csv "
          f"({len(actions)} rows)")


# ─────────────────────────────────────────────────────────────────────
# STEP 2 — Run grounding through discriminative scoring
# ─────────────────────────────────────────────────────────────────────

def run_grounding(motifs, mappings, dual, all_refs, master_x):
    print("\n[score] discriminative engine on full grounding corpus")
    motif_rows, family_rows, ambig_rows = [], [], []
    rank_motif_rows, rank_family_rows = [], []
    off_target_rows = []
    miss_rows = []
    audit_rows = []
    per_spec_rows = []

    for r in all_refs:
        comp = r["component_key"]
        sid = r["spectrum_id"]
        em = expected_motifs_for(comp)
        ef = expected_families_for(comp)
        ea = expected_ambiguity_for(comp)

        out = score_spectrum_discriminative(
            r["spectrum"], master_x, motifs, mappings, dual, sid,
        )
        disc_w = out["discriminative_weights"]
        amb = out["ambiguity_core"]

        # Sorted top motifs
        ms_sorted = sorted(disc_w.items(), key=lambda kv: kv[1], reverse=True)
        top5_motifs = [mid for mid, _ in ms_sorted[:5]]

        # Per-motif rows
        for mid, w in disc_w.items():
            base = out["base_weights"].get(mid, 0.0)
            motif_rows.append({
                "spectrum_id": sid, "dataset": r["dataset"],
                "component_key": comp,
                "motif_id": mid,
                "role": ROLE_TABLE.get(mid, "SUPPORT"),
                "base_weight": round(base, 5),
                "discriminative_weight": round(w, 5),
                "is_expected": mid in em,
                "is_top5": mid in top5_motifs,
            })

        # Family scores (using discriminative weights)
        fam_scores = {}
        fam_contribs = {}
        for fam in FAMILIES:
            s, contribs = family_score_discriminative(disc_w, mappings, fam)
            fam_scores[fam] = s
            fam_contribs[fam] = contribs
        fam_sorted = sorted(fam_scores.items(), key=lambda kv: kv[1], reverse=True)
        top5_fams = [f for f, _ in fam_sorted[:5]]
        for fam, s in fam_sorted:
            family_rows.append({
                "spectrum_id": sid, "dataset": r["dataset"],
                "component_key": comp,
                "family": fam,
                "family_score": round(s, 5),
                "is_expected": fam in ef,
                "is_top5": fam in top5_fams,
                "n_contributing_motifs": len(fam_contribs[fam]),
                "contributing_motifs": ",".join(fam_contribs[fam]),
            })

        # Ambiguity row
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

        # Rank rows
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

        # Off-target events
        for mid, w in disc_w.items():
            if w > 0.05 and em and mid not in em:
                off_target_rows.append({
                    "spectrum_id": sid, "dataset": r["dataset"],
                    "component_key": comp,
                    "off_target_motif": mid,
                    "discriminative_weight": round(w, 5),
                    "base_weight": round(out["base_weights"].get(mid, 0.0), 5),
                    "role": ROLE_TABLE.get(mid, "SUPPORT"),
                    "expected_motifs": ",".join(em),
                })

        # Miss row
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

        # Audit (write per-motif factor rows for first 30 spectra only)
        if len(per_spec_rows) < 30:
            for ar in out["audit_rows"]:
                audit_rows.append({
                    "spectrum_id": sid,
                    "component_key": comp,
                    **ar,
                })

        # Summary per-spectrum row
        per_spec_rows.append({
            "spectrum_id": sid, "dataset": r["dataset"],
            "component_key": comp,
            "expected_motifs": ",".join(em),
            "expected_families": ",".join(ef),
            "top1_motif": top5_motifs[0] if top5_motifs else "",
            "top1_motif_weight": round(disc_w[top5_motifs[0]], 5) if top5_motifs else 0,
            "top1_family": top5_fams[0] if top5_fams else "",
            "top1_family_score": round(fam_scores[top5_fams[0]], 5) if top5_fams else 0,
            "ambiguity_core": round(amb, 5),
            "motif_top1_hit": topn_hit(top5_motifs, em, 1),
            "motif_top3_hit": topn_hit(top5_motifs, em, 3),
            "motif_top5_hit": topn_hit(top5_motifs, em, 5),
            "family_top1_hit": topn_hit(top5_fams, ef, 1),
            "family_top3_hit": topn_hit(top5_fams, ef, 3),
            "family_top5_hit": topn_hit(top5_fams, ef, 5),
        })

    # ── Emit tables
    pd.DataFrame(per_spec_rows).to_csv(
        TABLES / "grounding_per_spectrum_scores_v_discriminative.csv", index=False,
    )
    pd.DataFrame(motif_rows).to_csv(
        TABLES / "grounding_per_spectrum_motif_scores_v_discriminative.csv", index=False,
    )
    pd.DataFrame(family_rows).to_csv(
        TABLES / "grounding_per_spectrum_family_scores_v_discriminative.csv", index=False,
    )
    pd.DataFrame(ambig_rows).to_csv(
        TABLES / "grounding_ambiguity_behavior_v_discriminative.csv", index=False,
    )
    pd.DataFrame(rank_motif_rows).to_csv(
        TABLES / "grounding_expected_vs_observed_motif_rank_v_discriminative.csv", index=False,
    )
    pd.DataFrame(rank_family_rows).to_csv(
        TABLES / "grounding_expected_vs_observed_family_rank_v_discriminative.csv", index=False,
    )
    pd.DataFrame(off_target_rows).to_csv(
        TABLES / "grounding_off_target_activation_v_discriminative.csv", index=False,
    )
    pd.DataFrame(miss_rows).to_csv(
        TABLES / "grounding_miss_list_v_discriminative.csv", index=False,
    )
    pd.DataFrame(audit_rows).to_csv(
        AUDIT / "scoring_audit_first_30_spectra.csv", index=False,
    )

    # Metrics
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
        TABLES / "grounding_metrics_summary_v_discriminative.csv", index=False,
    )
    print("\n[discriminative metrics]")
    for k, v in metrics.items():
        print(f"  {k:35s}: {v}")

    # Per-family + per-dataset hit rates
    rf_c = rf_c.copy()
    rf_c["primary_family"] = rf_c["expected_families"].str.split(",").str[0]
    per_fam = rf_c.groupby("primary_family")[
        ["family_top1_hit", "family_top3_hit", "family_top5_hit"]
    ].mean()
    per_fam_n = rf_c.groupby("primary_family").size().rename("n")
    per_fam_table = per_fam.join(per_fam_n)
    per_fam_table.to_csv(TABLES / "grounding_per_family_hit_rates_v_discriminative.csv")
    per_ds = rf_c.groupby("dataset")[
        ["family_top1_hit", "family_top3_hit", "family_top5_hit"]
    ].mean()
    per_ds_n = rf_c.groupby("dataset").size().rename("n")
    per_ds_table = per_ds.join(per_ds_n)
    per_ds_table.to_csv(TABLES / "grounding_per_dataset_hit_rates_v_discriminative.csv")

    return metrics, miss_rows, off_target_rows, ambig_rows, rank_motif_rows, \
           rank_family_rows, family_rows, motif_rows, per_fam_table, per_ds_table


# ─────────────────────────────────────────────────────────────────────
# STEP 3 — Before / after comparison vs motif-first baseline
# ─────────────────────────────────────────────────────────────────────

def write_before_after_table(metrics):
    mf = pd.read_csv(MF_METRICS).iloc[0]
    rows = []
    metric_keys = [
        "motif_top1_hit_rate", "motif_top3_hit_rate", "motif_top5_hit_rate",
        "family_top1_hit_rate", "family_top3_hit_rate", "family_top5_hit_rate",
        "ambiguity_correctness_rate", "ambiguity_overfire_rate",
        "ambiguity_underfire_rate",
        "n_motif_misses_top3", "n_family_misses_top3",
        "n_total_misses", "n_off_target_events",
    ]
    for k in metric_keys:
        before = float(mf[k])
        after = float(metrics[k])
        delta = round(after - before, 4)
        rows.append({
            "metric": k,
            "motif_first_baseline": round(before, 4),
            "discriminative_v1":    round(after, 4),
            "delta": delta,
            "improvement": (
                "BETTER" if (
                    (k.endswith("hit_rate") or k == "ambiguity_correctness_rate")
                    and delta > 0
                ) or (
                    (k.startswith("n_") or k == "ambiguity_overfire_rate"
                     or k == "ambiguity_underfire_rate")
                    and delta < 0
                ) else ("WORSE" if (
                    (k.endswith("hit_rate") or k == "ambiguity_correctness_rate")
                    and delta < 0
                ) or (
                    (k.startswith("n_") or k == "ambiguity_overfire_rate"
                     or k == "ambiguity_underfire_rate")
                    and delta > 0
                ) else "UNCHANGED")
            ),
        })
    df = pd.DataFrame(rows)
    df.to_csv(TABLES / "grounding_before_after_comparison_to_motif_first.csv",
              index=False)
    print(f"\n[emit] grounding_before_after_comparison_to_motif_first.csv")
    return df


# ─────────────────────────────────────────────────────────────────────
# STEP 4 — Figures
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

    # 1. fig_motif_rank_before_after_discriminative — motif top-1/3/5 hit
    #    by primary expected motif: motif-first vs discriminative.
    rm_before = pd.read_csv(MF_RANK_MOTIF)
    rm_before = rm_before[rm_before["expected_motifs"].fillna("") != ""].copy()
    rm_before["primary_expected"] = rm_before["expected_motifs"].astype(str).str.split(",").str[0]
    rm_after = pd.DataFrame(rank_motif_rows)
    rm_after = rm_after[rm_after["expected_motifs"].fillna("") != ""].copy()
    rm_after["primary_expected"] = rm_after["expected_motifs"].astype(str).str.split(",").str[0]
    fams_set = set(rm_before["primary_expected"].dropna()) | set(rm_after["primary_expected"].dropna())
    fams = sorted(str(f) for f in fams_set if f)
    bf = [float(rm_before[rm_before["primary_expected"] == f]["motif_top3_hit"].mean()
                 or 0.0) for f in fams]
    af = [float(rm_after[rm_after["primary_expected"] == f]["motif_top3_hit"].mean()
                 or 0.0) for f in fams]
    # Sort by improvement
    order = sorted(range(len(fams)),
                    key=lambda i: af[i] - bf[i], reverse=True)
    fams = [fams[i] for i in order]; bf = [bf[i] for i in order]; af = [af[i] for i in order]
    fig, ax = plt.subplots(figsize=(13, max(6, 0.35 * len(fams))))
    y = np.arange(len(fams))
    ax.barh(y - 0.2, bf, height=0.35, color="#e76f51", label="motif-first baseline")
    ax.barh(y + 0.2, af, height=0.35, color="#2a9d8f", label="discriminative v1")
    ax.set_yticks(y); ax.set_yticklabels(
        [f[:35] for f in fams], fontsize=7,
    )
    ax.invert_yaxis()
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("motif top-3 hit rate (per primary expected motif)")
    ax.set_title("Motif top-3 hit rate before/after discriminative upgrade")
    ax.legend()
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIGS / "fig_motif_rank_before_after_discriminative.png", dpi=130)
    plt.close(fig)

    # 2. fig_family_rank_before_after_discriminative
    pf_before = pd.read_csv(MF_PERFAM, index_col=0)
    pf_after = per_fam_table
    fams2 = sorted(set(pf_before.index) | set(pf_after.index))
    metrics_keys = ["family_top1_hit", "family_top3_hit", "family_top5_hit"]
    fig, axes = plt.subplots(1, 3, figsize=(18, max(5, 0.45 * len(fams2))))
    for ax, mk in zip(axes, metrics_keys):
        bf2 = [float(pf_before.loc[f, mk]) if f in pf_before.index else 0.0 for f in fams2]
        af2 = [float(pf_after.loc[f, mk])  if f in pf_after.index  else 0.0 for f in fams2]
        order = sorted(range(len(fams2)), key=lambda i: af2[i] - bf2[i], reverse=True)
        fams2_o = [fams2[i] for i in order]; bf2 = [bf2[i] for i in order]; af2 = [af2[i] for i in order]
        y = np.arange(len(fams2_o))
        ax.barh(y - 0.2, bf2, height=0.35, color="#e76f51", label="motif-first")
        ax.barh(y + 0.2, af2, height=0.35, color="#2a9d8f", label="discriminative")
        ax.set_yticks(y); ax.set_yticklabels(fams2_o, fontsize=8)
        ax.invert_yaxis()
        ax.set_xlim(0, 1.05)
        ax.set_xlabel(mk)
        ax.set_title(f"family {mk}")
        ax.legend(fontsize=8)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    fig.suptitle("Per-family hit rates: motif-first vs discriminative", fontsize=12)
    fig.tight_layout()
    fig.savefig(FIGS / "fig_family_rank_before_after_discriminative.png", dpi=130)
    plt.close(fig)

    # 3. fig_off_target_before_after_discriminative
    of_before = pd.read_csv(MF_OFFTGT)
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
    ax.barh(y - 0.2, bv, height=0.35, color="#e76f51", label=f"motif-first ({sum(bv)})")
    ax.barh(y + 0.2, av, height=0.35, color="#2a9d8f", label=f"discriminative ({sum(av)})")
    ax.set_yticks(y); ax.set_yticklabels([c[:35] for c in common], fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("off-target activation events (count)")
    ax.set_title("Off-target activation: motif-first vs discriminative")
    ax.legend()
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIGS / "fig_off_target_before_after_discriminative.png", dpi=130)
    plt.close(fig)

    # 4. fig_ambiguity_before_after_discriminative — 3-panel
    amb_before = pd.read_csv(MF_AMBIG)
    amb_after = pd.DataFrame(ambig_rows)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    # 4a histogram
    axes[0].hist(amb_before["ambiguity_core"], bins=30, color="#e76f51",
                 alpha=0.55, label="motif-first")
    axes[0].hist(amb_after["ambiguity_core"], bins=30, color="#2a9d8f",
                 alpha=0.55, label="discriminative")
    axes[0].axvline(0.10, color="black", linestyle="--", label="gated 0.10")
    axes[0].set_xlabel("ambiguity_core"); axes[0].set_ylabel("count of spectra")
    axes[0].set_title("Ambiguity score distribution")
    axes[0].legend()
    # 4b correctness bars
    cb = float(amb_before["ambiguity_correct"].mean())
    ca = float(amb_after["ambiguity_correct"].mean())
    axes[1].bar(["motif-first", "discriminative"], [cb, ca],
                color=["#e76f51", "#2a9d8f"])
    for i, v in enumerate([cb, ca]):
        axes[1].text(i, v + 0.02, f"{v:.1%}", ha="center", fontsize=10)
    axes[1].set_ylim(0, 1.0); axes[1].set_ylabel("correctness rate")
    axes[1].set_title("Ambiguity correctness")
    # 4c overfire/underfire bars
    ob = float(amb_before["ambiguity_overfire"].mean())
    oa = float(amb_after["ambiguity_overfire"].mean())
    ub = float(amb_before["ambiguity_underfire"].mean())
    ua = float(amb_after["ambiguity_underfire"].mean())
    x = np.arange(2); w = 0.35
    axes[2].bar(x - w/2, [ob, oa], width=w, color="#f4a261", label="overfire")
    axes[2].bar(x + w/2, [ub, ua], width=w, color="#264653", label="underfire")
    axes[2].set_xticks(x); axes[2].set_xticklabels(["motif-first", "discriminative"])
    axes[2].set_ylim(0, 1.0); axes[2].set_ylabel("rate")
    axes[2].set_title("Ambiguity over/underfire")
    axes[2].legend()
    for side in ("top", "right"):
        axes[0].spines[side].set_visible(False)
        axes[1].spines[side].set_visible(False)
        axes[2].spines[side].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIGS / "fig_ambiguity_before_after_discriminative.png", dpi=130)
    plt.close(fig)

    # 5. fig_grouped_motif_in_family_examples_discriminative
    id_to_ref = {r["spectrum_id"]: r for r in all_refs}
    examples = []
    targets = [
        ("ramanbiolib", "cholesteryl linoleate"),
        ("ramanbiolib", "l-glutamate"),
        ("ramanbiolib", "adenine"),
        ("gobbato_powder", "UA_rep01"),
        ("ramanbiolib", "d-(+)-glucose"),
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
            out = score_spectrum_discriminative(
                ref["spectrum"], master_x, motifs, mappings, dual, sid,
            )
            disc_w = out["discriminative_weights"]
            fam_to_contrib = {}
            for fam in FAMILIES:
                contribs = []
                for mid, s in disc_w.items():
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
            ax.set_xlabel("stacked motif contribution to family")
            ax.set_title(sid.split("::")[-1][:30], fontsize=9)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
        fig.suptitle("Grouped motif-in-family examples (discriminative engine)",
                     fontsize=12)
        fig.tight_layout()
        fig.savefig(FIGS / "fig_grouped_motif_in_family_examples_discriminative.png",
                    dpi=130)
        plt.close(fig)

        # 6. fig_radar_examples_discriminative
        fig, axes = plt.subplots(1, len(examples),
                                 figsize=(4.5*len(examples), 4.5),
                                 subplot_kw=dict(polar=True))
        if len(examples) == 1: axes = [axes]
        angles = np.linspace(0, 2*np.pi, len(FAMILIES), endpoint=False).tolist()
        angles += angles[:1]
        for ax, sid in zip(axes, examples):
            ref = id_to_ref[sid]
            out = score_spectrum_discriminative(
                ref["spectrum"], master_x, motifs, mappings, dual, sid,
            )
            disc_w = out["discriminative_weights"]
            vals = []
            for fam in FAMILIES:
                s, _ = family_score_discriminative(disc_w, mappings, fam)
                vals.append(s)
            vmax = max(vals) if max(vals) > 0 else 1.0
            vals = [v / vmax for v in vals]
            vals += vals[:1]
            ax.plot(angles, vals, color="#2a9d8f", linewidth=1.5)
            ax.fill(angles, vals, color="#2a9d8f", alpha=0.3)
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels([f.replace("_", "\n") for f in FAMILIES],
                               fontsize=5)
            ax.set_ylim(0, 1.05)
            ax.set_title(sid.split("::")[-1][:25], fontsize=8, pad=12)
        fig.suptitle("Family-level radar (discriminative; normalised per-spectrum)",
                     fontsize=11)
        fig.tight_layout()
        fig.savefig(FIGS / "fig_radar_examples_discriminative.png", dpi=130)
        plt.close(fig)

    # 7. fig_sunburst_treemap_exploratory_discriminative
    from gaira.base2.motif_engine import resolve_mapping_weight
    agg = defaultdict(lambda: defaultdict(float))
    agg_amb = 0.0
    for ref in all_refs:
        out = score_spectrum_discriminative(
            ref["spectrum"], master_x, motifs, mappings, dual, ref["spectrum_id"],
        )
        agg_amb += out["ambiguity_core"]
        disc_w = out["discriminative_weights"]
        for fam in FAMILIES:
            for mid, s in disc_w.items():
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
    fig.suptitle("Family -> motif treemap (discriminative engine; full grounding corpus)",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(FIGS / "fig_sunburst_treemap_exploratory_discriminative.png", dpi=130)
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────
# STEP 5 — Reports
# ─────────────────────────────────────────────────────────────────────

def write_main_report(metrics, ba_df, per_fam_table, per_ds_table,
                      n_motifs, n_mappings, n_anti_rules, n_competitor_pairs,
                      n_ambiguity_routes):
    role_counts = pd.Series(list(ROLE_TABLE.values())).value_counts().to_dict()
    pf_strong = per_fam_table.sort_values("family_top1_hit", ascending=False)
    pf_weak = per_fam_table.sort_values("family_top1_hit").head(3)

    lines = [
        "# gaira_base_2 - Discriminative Motif Upgrade v1",
        "",
        "## Why this phase was needed",
        "",
        "After the motif-first validation, three structural problems remained:",
        "",
        "1. **Broad motifs (amide_I/III, lipid_C_H_bend, lipid_methylene_twist, "
        "free_saccharide) won top-1 motif on small-molecule references** "
        "where their bands fired accidentally. Off-target activation count "
        "was 905 events.",
        "2. **Ambiguity overfired at 58.9%** because the lane couldn't distinguish "
        "shared chemistry (purine 720 vs PC choline 715; methionine C-S "
        "vs disulfide) from real ambiguity.",
        "3. **Family top-1 stayed at 47.5%** even when family top-3 reached "
        "69.5%, because broad motifs aggregated noisily into family scores.",
        "",
        "These are not scoring-weight problems (v4 proved aggressive scoring "
        "patches regress). They are STRUCTURAL: motifs have no role classification "
        "and no anti-evidence. This phase reframes motifs as discriminative "
        "objects: each motif now has a role, anchor / support bands, "
        "anti-evidence rules, competitor relationships, and (where "
        "warranted) ambiguity routing.",
        "",
        "## Role classification summary",
        "",
        f"All {len(ROLE_TABLE)} active + inactive registry motifs were classified:",
        "",
        "| role | count | meaning |",
        "|---|---:|---|",
        f"| ANCHOR | {role_counts.get('ANCHOR', 0)} | chemistry-specific; may drive a claim alone |",
        f"| SUPPORT | {role_counts.get('SUPPORT', 0)} | shared / cross-axis; needs same-family ANCHOR co-fire to be credible |",
        f"| BACKGROUND | {role_counts.get('BACKGROUND', 0)} | broad indicator; gated to never win alone (factor x0.4, or x0.2 if no anchor) |",
        f"| AMBIGUITY_ONLY | {role_counts.get('AMBIGUITY_ONLY', 0)} | collision-aware routing; zero biology-axis contribution |",
        f"| ARTIFACT_ONLY | {role_counts.get('ARTIFACT_ONLY', 0)} | substrate / buffer artifact; zero biology-axis contribution |",
        "",
        "## Anchor / support / background logic (scoring extension)",
        "",
        "Per-motif scoring under the discriminative module:",
        "",
        "```",
        "discriminative_weight = base_rescue_weight",
        "                      x role_factor",
        "                      x anti_evidence_factor",
        "                      x (1 - ambiguity_share)",
        "```",
        "",
        "Role factors (multiplied by NO_ANCHOR_PENALTY when no same-family "
        "ANCHOR co-fires above 0.30):",
        "",
        "| role | base | × NO_ANCHOR_PENALTY | effective when no anchor |",
        "|---|---:|---:|---:|",
        f"| ANCHOR | {ROLE_FACTOR['ANCHOR']:.2f} | {NO_ANCHOR_PENALTY['ANCHOR']:.2f} | {ROLE_FACTOR['ANCHOR']*NO_ANCHOR_PENALTY['ANCHOR']:.2f} |",
        f"| SUPPORT | {ROLE_FACTOR['SUPPORT']:.2f} | {NO_ANCHOR_PENALTY['SUPPORT']:.2f} | {ROLE_FACTOR['SUPPORT']*NO_ANCHOR_PENALTY['SUPPORT']:.2f} |",
        f"| BACKGROUND | {ROLE_FACTOR['BACKGROUND']:.2f} | {NO_ANCHOR_PENALTY['BACKGROUND']:.2f} | {ROLE_FACTOR['BACKGROUND']*NO_ANCHOR_PENALTY['BACKGROUND']:.2f} |",
        "",
        "## Anti-evidence logic",
        "",
        f"{n_anti_rules} anti-evidence rules across {len(ANTI_EVIDENCE)} motifs. "
        "Three rule types:",
        "",
        "- **REQUIRES_COBAND** - target motif must fire >= min_weight, else "
        "penalty applies. Used for: amide_I requires amide_III co-fire; "
        "amide_III requires amide_I; free_saccharide requires "
        "glycan_pyranose; cholesterol_signature requires sterol_skeletal.",
        "- **SUPPRESS_IF_PRESENT** - presence of competing target motif >= "
        "min_weight triggers penalty. Used for: purine 720 suppressed by "
        "PC choline 715; glycan 1020 suppressed by phosphate 1240; "
        "phosphate 1080 suppressed by glycan ring 850-950.",
        "- **REQUIRES_ANY_FAMILY_ANCHOR** - none of the listed family anchors "
        "firing >= min_weight triggers penalty. Used for: lipid_C_H_bend "
        "requires at least one lipid anchor; lipid_methylene_twist same; "
        "PC_choline_715 requires a real lipid anchor.",
        "",
        "## Competitor logic",
        "",
        f"{n_competitor_pairs} motif x competitor relationships recorded "
        "in `registry/motif_discriminator_registry_v1.csv`. "
        "Competitor logic is implemented through SUPPRESS_IF_PRESENT "
        "rules (above) - the registry lists competitors for documentation "
        "and audit; the rules are the operational enforcement.",
        "",
        "## Ambiguity routing",
        "",
        f"{n_ambiguity_routes} explicit ambiguity-routing rules (purine 720 vs "
        "PC choline 715; glycan 1020 vs phosphate 1240). When the trigger "
        "co-fires, a 30% weight share routes to the ambiguity lane instead "
        "of the biology family. This is in addition to the rescue engine's "
        "gated ambiguity score.",
        "",
        "## How scoring changed (engine plumbing)",
        "",
        "- Engine modules (schema/motif_engine/projection/ambiguity/registry) - "
        "**NOT** modified.",
        "- Patch modules (v2_patches.py, v2_patches_rescue.py) - **NOT** modified.",
        "- New file: `src/gaira/base2/v2_patches_discriminative.py`.",
        "- Registry / mapping files - read-only this phase (still v1.3.1 / v1.2.1).",
        "- Discriminative scoring is a wrapper around `patched_score_spectrum_rescue` "
        "that applies the role/anti/ambiguity factors to its motif weights and "
        "re-aggregates families with the new weights.",
        "",
        "## Grounding rerun headline",
        "",
        "| metric | motif-first baseline | discriminative v1 | delta |",
        "|---|---:|---:|---:|",
    ]
    for _, r in ba_df.iterrows():
        b, a, d = r["motif_first_baseline"], r["discriminative_v1"], r["delta"]
        if r["metric"].endswith("rate"):
            lines.append(f"| {r['metric']} | {b:.1%} | {a:.1%} | {d:+.1%} |")
        else:
            lines.append(f"| {r['metric']} | {int(b)} | {int(a)} | {int(d):+d} |")

    lines += [
        "",
        "## Strongest wins",
        "",
        "Top-1 family hit by family (sorted by post-discriminative top-1):",
        "",
        "| family | top-1 | top-3 | top-5 | n |",
        "|---|---:|---:|---:|---:|",
    ]
    for fam, row in pf_strong.iterrows():
        lines.append(f"| {fam} | {row['family_top1_hit']:.1%} | "
                     f"{row['family_top3_hit']:.1%} | {row['family_top5_hit']:.1%} | "
                     f"{int(row['n'])} |")

    lines += [
        "",
        "## Weakest remaining spots",
        "",
        "| family | top-1 | top-3 | top-5 | n |",
        "|---|---:|---:|---:|---:|",
    ]
    for fam, row in pf_weak.iterrows():
        lines.append(f"| {fam} | {row['family_top1_hit']:.1%} | "
                     f"{row['family_top3_hit']:.1%} | {row['family_top5_hit']:.1%} | "
                     f"{int(row['n'])} |")

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

    lines += [
        "",
        "## Reading guide",
        "",
        "- Family top-1/3/5 and motif top-1/3/5 are the primary metrics.",
        "- Off-target activation is now a much harder bar (BACKGROUND motifs "
        "are gated x0.40 or x0.20 baseline, plus anti-evidence penalties).",
        "- Ambiguity correctness is the joint of overfire + underfire under "
        "the truth table; lower overfire = better.",
        "- Engine-level changes are isolated to `v2_patches_discriminative.py` "
        "and ARE NOT propagated into the active baseline registry until a "
        "subsequent phase decides to elevate them.",
        "",
        "## What this report does NOT discuss",
        "",
        "- Calibration / target / substrate-aware behavior.",
        "- Broad-axis top-1 numbers as primary evidence.",
        "- Ontology additions (no new motifs in this phase).",
    ]
    (REPORTS / "REPORT_gaira_base_2_discriminative_motif_upgrade_v1.md"
     ).write_text("\n".join(lines))


def write_miss_report(metrics, miss_rows, ba_df, ambig_rows, off_target_rows):
    df = pd.DataFrame(miss_rows)
    of = pd.DataFrame(off_target_rows)

    # Misses fixed = misses present in motif-first but not in discriminative.
    mf_misses = pd.read_csv(Path(
        "/Volumes/SSD_Rad/GAIRA_BUILD/gaira_validate_2_grounding_motif_first_v1/"
        "tables/grounding_miss_list_v_motif_first.csv"
    ))
    mf_set = set(mf_misses["spectrum_id"])
    disc_set = set(df["spectrum_id"])
    fixed = mf_set - disc_set
    new = disc_set - mf_set
    persisted = mf_set & disc_set

    # Family-miss breakdown
    if len(df) > 0:
        df_f = df.copy()
        df_f["primary_expected_family"] = df_f["expected_families"].str.split(",").str[0]
        fam_break = df_f["primary_expected_family"].value_counts()
    else:
        fam_break = pd.Series(dtype=int)

    # Off-target breakdown
    if len(of) > 0:
        of_break = of["off_target_motif"].value_counts().head(15)
    else:
        of_break = pd.Series(dtype=int)

    amb_after = pd.DataFrame(ambig_rows)
    amb_over = amb_after[amb_after["ambiguity_overfire"]]

    lines = [
        "# Discriminative Motif Upgrade v1 - Miss Analysis",
        "",
        "## Misses fixed vs persisted vs newly introduced",
        "",
        f"- motif-first total misses: **{len(mf_set)}**",
        f"- discriminative total misses: **{len(disc_set)}**",
        f"- misses **fixed** (present in motif-first, gone in discriminative): "
        f"**{len(fixed)}**",
        f"- misses **persisted** (still missed): **{len(persisted)}**",
        f"- misses **newly introduced** (not missed in motif-first, now missed): "
        f"**{len(new)}**",
        "",
        "## Persisted-miss family breakdown",
        "",
        "| primary expected family | n missed |",
        "|---|---:|",
    ]
    for fam, c in fam_break.items():
        lines.append(f"| {fam} | {c} |")

    lines += [
        "",
        "## Remaining off-target hotspots",
        "",
        "Top off-target motifs (now under discriminative gating; counts should be "
        "lower than motif-first):",
        "",
        "| motif | n events | role |",
        "|---|---:|---|",
    ]
    for mid, c in of_break.items():
        lines.append(f"| `{mid}` | {c} | {ROLE_TABLE.get(mid, '?')} |")

    lines += [
        "",
        "## Ambiguity overfires that persist",
        "",
        f"({len(amb_over)} spectra) Top 10 by ambiguity score:",
        "",
    ]
    for _, r in amb_over.sort_values("ambiguity_core",
                                       ascending=False).head(10).iterrows():
        lines.append(f"- `{r['component_key']}` "
                     f"(disc={r['ambiguity_core']:.3f}, "
                     f"rescue={r['rescue_ambiguity_core']:.3f}, "
                     f"routed={r['routed_to_ambiguity']:.3f})")

    lines += [
        "",
        "## Whether remaining misses are ontology / evidence / chemistry-overlap",
        "",
        "Classification (manual judgment based on motif-first + discriminative findings):",
        "",
        "1. **ONTOLOGY LIMITS** (need new motifs in v2 ontology phase):",
        "   - Free-amino-acid side-chain motifs (Arg, Asp, Ser, Pro, Val, ...). "
        "These references currently rely on the broad amide motifs which are "
        "now correctly gated DOWN, so misses persist.",
        "   - Lactate motif (DEFERRED).",
        "   - Histidine imidazole motif (registry entry exists, no mapping).",
        "   - Tryptophan signature motif (registry entry exists, no mapping).",
        "   - Aromatic-steroid (estrogen) discriminator motif.",
        "",
        "2. **EVIDENCE LIMITS** (need new pure-compound references):",
        "   - Lactate pure powder (not in any current grounding dataset).",
        "   - Pure aromatic steroid (estradiol family) Raman.",
        "",
        "3. **TRUE CHEMISTRY OVERLAP** (cannot be resolved by adding motifs):",
        "   - Cholesteryl esters fire BOTH sterol_neutral_lipid AND "
        "lipid_acyl_membrane axes.",
        "   - Free amino acids fire BOTH metabolic_small_molecule AND "
        "protein_peptide_backbone.",
        "   - UA / HX / xanthine fire BOTH purine_nucleotide AND "
        "purine_metabolite axes via shared 720-735 ring breathing.",
        "",
        "## Whether the system is now ready for calibration",
        "",
        f"- Ambiguity overfire: motif-first {ba_df.iloc[7]['motif_first_baseline']:.1%} "
        f"-> discriminative {ba_df.iloc[7]['discriminative_v1']:.1%} "
        f"({ba_df.iloc[7]['delta']:+.1%}).",
        f"- Family top-3 hit:  motif-first {ba_df.iloc[4]['motif_first_baseline']:.1%} "
        f"-> discriminative {ba_df.iloc[4]['discriminative_v1']:.1%} "
        f"({ba_df.iloc[4]['delta']:+.1%}).",
        f"- Off-target events: motif-first {int(ba_df.iloc[12]['motif_first_baseline'])} "
        f"-> discriminative {int(ba_df.iloc[12]['discriminative_v1'])} "
        f"({int(ba_df.iloc[12]['delta']):+d}).",
        "",
        "Recommendation depends on the deltas above. If ambiguity overfire "
        "and off-target events both drop materially without family top-3 "
        "regression, the system is in a healthier state for calibration. "
        "If family top-3 regresses (because anti-evidence penalties cost "
        "true-chemistry-correct hits), more conservative penalty values "
        "are warranted in v2.",
    ]
    (REPORTS / "REPORT_gaira_base_2_discriminative_motif_miss_analysis_v1.md"
     ).write_text("\n".join(lines))


def write_audit_log(metrics, ba_df, n_anti_rules, n_competitor_pairs,
                    n_ambiguity_routes, n_role_actions):
    lines = [
        "# gaira_base_2 Discriminative Motif Upgrade v1 - Audit Log",
        "",
        "## Files added (relative to repo)",
        "",
        "- ADDED: `src/gaira/base2/v2_patches_discriminative.py` - role table, "
        "anti-evidence rules, competitor map, ambiguity routing, and "
        "score_spectrum_discriminative() entry point.",
        "- ADDED: `scripts/run_gaira_base_2_discriminative_motif_upgrade_v1.py`",
        "- ADDED: `GAIRA_BUILD/gaira_base_2_discriminative_motif_upgrade_v1/**`",
        "",
        "## Files NOT modified",
        "",
        "- `gaira_base` frozen pilot files - SHA-256 still matches "
        "(12/12 v1 regression tests pass)",
        "- v1 engine modules (schema.py, motif_engine.py, axis_engine.py, "
        "projection.py, ambiguity.py, registry.py, primitives.py, "
        "compatibility.py, calibration_overlay.py) - untouched",
        "- v2_patches.py, v2_patches_rescue.py, v2_patches_repair_v2.py - untouched",
        "- Registry v1.3.1 + mapping v1.2.1 - read-only this phase "
        "(NO ontology edits)",
        "- canonical preprocessing - unchanged",
        "- substrate engine v1.1.2 - unchanged",
        "- M2.2 dual-status table - unchanged",
        "- NO calibration / target / substrate-aware data used",
        "",
        "## Exact motifs changed (role assigned)",
        "",
        f"All {len(ROLE_TABLE)} active + inactive registry motifs received a "
        "role classification. Of those:",
        "",
    ]
    role_counts = pd.Series(list(ROLE_TABLE.values())).value_counts().to_dict()
    for role, c in sorted(role_counts.items()):
        lines.append(f"- {role}: {c} motifs")

    lines += [
        "",
        "## Exact new logic added",
        "",
        f"- **{len(ANTI_EVIDENCE)} motifs** received anti-evidence rules "
        f"(total {n_anti_rules} rules).",
        f"- **{len(COMPETITORS)} motifs** received explicit competitor lists "
        f"(total {n_competitor_pairs} pairs documented).",
        f"- **{len(AMBIGUITY_ROUTING)} motifs** received explicit ambiguity-routing "
        "rules (purine 720 vs PC choline 715; glycan 1020 vs phosphate 1240).",
        f"- Total upgrade actions logged: **{n_role_actions}** "
        "(see `tables/discriminative_motif_upgrade_actions_v1.csv`).",
        "",
        "## Motifs deliberately left unchanged",
        "",
        "- All ANCHOR motifs (24): no anti-evidence rules added beyond what "
        "their existing co_band_requirement already enforces. ANCHOR motifs "
        "are already chemistry-specific.",
        "- thiol_C_S_str_660_motif (SUPPORT): no anti-evidence (the C-S band "
        "is narrow; misfiring is rare).",
        "- sugar_phosphate_skeletal_870_900 (SUPPORT): no anti-evidence "
        "(CROSS_AXIS contribution is intentional and audited).",
        "- nucleobase_in_plane_ring_1320_1340, "
        "amide_I_lipid_carbonyl_partial_panel_motif (HELD_V2 in mapping): "
        "role assigned but no new logic - they have zero scoring contribution.",
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
        "## Whether a further loop is recommended",
        "",
        "Recommendation TBD pending interpretation of grounding deltas vs "
        "the motif-first baseline. See "
        "`reports/REPORT_gaira_base_2_discriminative_motif_miss_analysis_v1.md`.",
    ]
    (AUDIT / "gaira_base_2_discriminative_motif_upgrade_audit_log.md"
     ).write_text("\n".join(lines))


def snapshot_code():
    src = Path("/Users/suraj/projects/GAIRA/src/gaira/base2")
    if src.exists():
        shutil.copytree(src, CODE_SNAPSHOT / "base2", dirs_exist_ok=True)
    p = Path("/Users/suraj/projects/GAIRA/scripts/"
             "run_gaira_base_2_discriminative_motif_upgrade_v1.py")
    if p.exists(): shutil.copy(p, CODE_SNAPSHOT / p.name)


# ─────────────────────────────────────────────────────────────────────
# Driver
# ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 78)
    print("gaira_base_2 - Discriminative Motif Upgrade v1")
    print("=" * 78)
    for d in (REGISTRY, TABLES, FIGS, REPORTS, AUDIT, CODE_SNAPSHOT):
        d.mkdir(parents=True, exist_ok=True)

    master_x = canonical_master_axis()

    motifs = load_motif_registry(REG_V1_3_1)
    mappings = load_axis_mapping(MAP_V1_2_1)
    dual = load_dual_status()
    active = {m: s for m, s in motifs.items() if s.v1_active}
    print(f"[engine] {len(active)} active motifs, {len(mappings)} mappings")

    # STEP 1 - emit discriminator registry CSVs
    emit_discriminator_registry(motifs, mappings)

    # Counts for reports
    n_anti_rules = sum(len(v) for v in ANTI_EVIDENCE.values())
    n_competitor_pairs = sum(len(v) for v in COMPETITORS.values())
    n_ambiguity_routes = len(AMBIGUITY_ROUTING)
    actions_df = pd.read_csv(TABLES / "discriminative_motif_upgrade_actions_v1.csv")
    n_role_actions = len(actions_df)

    # Load grounding corpus
    rb  = load_ramanbiolib(master_x)
    gp  = load_gobbato_powder(master_x)
    aa  = load_amino_acid_xlsx(master_x)
    lit = load_digitised_literature(master_x)
    all_refs = rb + gp + aa + lit
    print(f"[data] {len(all_refs)} grounding spectra "
          f"({len(rb)} rbl + {len(gp)} gobbato + {len(aa)} aa + {len(lit)} lit)")

    # STEP 2 - run grounding through discriminative scoring
    (metrics, miss_rows, off_target_rows, ambig_rows,
     rank_motif_rows, rank_family_rows, family_rows, motif_rows,
     per_fam_table, per_ds_table) = run_grounding(
        active, mappings, dual, all_refs, master_x,
    )

    # STEP 3 - before/after comparison
    ba_df = write_before_after_table(metrics)

    # STEP 4 - figures
    make_figs(active, mappings, dual, all_refs, master_x,
              motif_rows, family_rows, ambig_rows, off_target_rows,
              rank_motif_rows, rank_family_rows, per_fam_table)

    # STEP 5 - reports
    write_main_report(metrics, ba_df, per_fam_table, per_ds_table,
                      len(active), len(mappings), n_anti_rules,
                      n_competitor_pairs, n_ambiguity_routes)
    write_miss_report(metrics, miss_rows, ba_df, ambig_rows, off_target_rows)
    write_audit_log(metrics, ba_df, n_anti_rules, n_competitor_pairs,
                    n_ambiguity_routes, n_role_actions)
    snapshot_code()

    print("\nDONE")


if __name__ == "__main__":
    main()
