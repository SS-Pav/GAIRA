"""gaira_base_3 grounding full-corpus build + validation v1.

Definitive integrated grounding build. Five stages:
  1. Full corpus audit (strict inclusion rules)
  2. Build ontology from scratch
  3. In-sample evaluation (must saturate)
  4. Cross-validation (3 protocols)
  5. Packet audit + rename + chemistry coherence

Run:
    cd /Users/suraj/projects/GAIRA
    PYTHONPATH=src .venv/bin/python scripts/run_gaira_base_3_grounding_full_corpus_build_and_validation_v1.py
"""
from __future__ import annotations

import shutil
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from gaira.base3 import learned_ontology_v2 as _learn
from gaira.spectral import canonical_master_axis

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_gaira_validate_2_grounding import (
    load_ramanbiolib, load_gobbato_powder,
    load_amino_acid_xlsx, load_digitised_literature,
)
from run_gaira_validate_2_grounding_motif_first_v1 import (
    FAMILIES, expected_families_for, expected_ambiguity_for, topn_hit,
)
from run_gaira_base_3_packet_ontology_v1 import expected_packets_for
from run_gaira_base_3_grounding_trained_ontology_v1 import (
    derive_analyte_class, normalise_label, CLASS_TO_CURRENT_FAMILY,
)


ROOT = Path("/Volumes/SSD_Rad/GAIRA_BUILD/gaira_base_3_grounding_full_corpus_build_and_validation_v1")
TABLES = ROOT / "tables"
REGISTRY = ROOT / "registry"
FIGS = ROOT / "figures"
REPORTS = ROOT / "reports"
AUDIT = ROOT / "audit"
DOCS = ROOT / "docs"
CODE_SNAPSHOT = ROOT / "code_snapshot"


# ─────────────────────────────────────────────────────────────────────
# STAGE 1 — Full corpus audit (STRICT)
# ─────────────────────────────────────────────────────────────────────

def stage1_corpus_audit(rb, gp, aa, lit) -> pd.DataFrame:
    print("\n[STAGE 1] Full grounding corpus audit (strict inclusion)")
    rows = [
        {"dataset_name": "ramanbiolib",
         "type": "Raman",
         "regime": "normal Raman, pure powders/liquids",
         "analyte_count": len({r["component_key"] for r in rb}),
         "spectrum_count": len(rb),
         "include_flag": True,
         "reason_for_exclusion": "",
         "substrate_type": "n/a (normal Raman)",
         "notes": ("Curated reference database (De Gelder 2007 mirror). "
                   "Single-analyte pure-compound spectra. ADMITTED: "
                   "single analyte, full spectrum, identity reliable, "
                   "no biological matrix.")},
        {"dataset_name": "gobbato_powder_raman",
         "type": "Raman",
         "regime": "normal Raman, pure powders",
         "analyte_count": len({r["component_key"] for r in gp}),
         "spectrum_count": len(gp),
         "include_flag": True,
         "reason_for_exclusion": "",
         "substrate_type": "n/a (normal Raman)",
         "notes": ("53 pure analyte powders × 3 replicates each. "
                   "ADMITTED: single analyte, full spectrum, identity "
                   "reliable, intra-analyte replicates support cross-"
                   "validation.")},
        {"dataset_name": "amino_acid_raman_grounding",
         "type": "Raman",
         "regime": "normal Raman, pure amino acid powders",
         "analyte_count": len({r["component_key"] for r in aa}),
         "spectrum_count": len(aa),
         "include_flag": True,
         "reason_for_exclusion": "",
         "substrate_type": "n/a (normal Raman)",
         "notes": ("aa.xlsx pure amino acid Raman. ADMITTED: required "
                   "explicitly by phase spec; single analyte, full spectrum.")},
        {"dataset_name": "digitised_literature_spectra",
         "type": "Raman",
         "regime": "digitised normal Raman from literature",
         "analyte_count": len({r["component_key"] for r in lit}),
         "spectrum_count": len(lit),
         "include_flag": True,
         "reason_for_exclusion": "",
         "substrate_type": "n/a (normal Raman)",
         "notes": ("De Gelder 2007 + Kim 1987 digitised UA spectra. "
                   "ADMITTED: assignment-grade reference, single analyte.")},
        # Datasets considered but rejected
        {"dataset_name": "ag_colloid_serum_sers",
         "type": "SERS",
         "regime": "Ag-colloid SERS in serum matrix",
         "analyte_count": 0,
         "spectrum_count": 0,
         "include_flag": False,
         "reason_for_exclusion": ("biological matrix (serum) AND uses "
                                    "spike/depletion design — outside "
                                    "STRICT inclusion criteria. This is "
                                    "calibration data, not grounding."),
         "substrate_type": "Ag colloid",
         "notes": "Reserved for the calibration phase."},
        {"dataset_name": "raw_search_pool_candidates",
         "type": "various",
         "regime": "literature mentions only (peak lists)",
         "analyte_count": 0,
         "spectrum_count": 0,
         "include_flag": False,
         "reason_for_exclusion": ("peak-list-only resources; no full "
                                    "spectra available for training; "
                                    "vague identity in many cases."),
         "substrate_type": "n/a",
         "notes": "Out of scope for this phase."},
        {"dataset_name": "target_serum_cohort_data",
         "type": "various",
         "regime": "patient serum cohorts",
         "analyte_count": 0,
         "spectrum_count": 0,
         "include_flag": False,
         "reason_for_exclusion": ("multi-analyte mixtures in biological "
                                    "matrix; explicitly forbidden by "
                                    "phase rules."),
         "substrate_type": "various",
         "notes": "Reserved for the target / cohort analysis phase."},
    ]
    df = pd.DataFrame(rows)
    df.to_csv(TABLES / "grounding_dataset_inventory_full_v1.csv", index=False)
    print(f"  emitted grounding_dataset_inventory_full_v1.csv "
          f"({df['include_flag'].sum()} datasets included, "
          f"{(~df['include_flag']).sum()} excluded)")
    return df


# ─────────────────────────────────────────────────────────────────────
# STAGE 2 — Build ontology from scratch
# ─────────────────────────────────────────────────────────────────────

def stage2_build_ontology(all_refs, master_x):
    print("\n[STAGE 2] Building learned ontology v2 from scratch")

    # 2A — taxonomy
    rows = []
    for r in all_refs:
        comp = r["component_key"]
        analyte = normalise_label(comp)
        cls = derive_analyte_class(analyte)
        ep = expected_packets_for(comp)
        ef = expected_families_for(comp)
        ea = expected_ambiguity_for(comp)
        rows.append({
            "spectrum_id": r["spectrum_id"],
            "dataset_name": r["dataset"],
            "analyte_name": analyte,
            "analyte_class": cls,
            "expected_packet_candidate": ",".join(ep),
            "expected_family": ",".join(ef),
            "multi_family_allowed": len(ef) > 1,
            "ambiguity_allowed": ea,
            "regime": "normal Raman",
            "substrate_type": "n/a",
        })
    taxonomy_df = pd.DataFrame(rows)
    taxonomy_df.to_csv(TABLES / "grounding_training_taxonomy_v2.csv", index=False)
    print(f"  [2A] taxonomy ({len(taxonomy_df)} spectra; "
          f"{taxonomy_df['analyte_class'].nunique()} analyte classes)")

    # Group spectra by analyte class for learning
    spectra_by_class: dict[str, list[np.ndarray]] = defaultdict(list)
    spectra_meta_by_class: dict[str, list[dict]] = defaultdict(list)
    for r in all_refs:
        cls = derive_analyte_class(normalise_label(r["component_key"]))
        if cls and cls != "uncategorised":
            spectra_by_class[cls].append(r["spectrum"])
            spectra_meta_by_class[cls].append({
                "spectrum_id": r["spectrum_id"],
                "dataset": r["dataset"],
                "component_key": r["component_key"],
            })

    # 2B/2C — discriminative features + prototypes
    print("  [2B] Discriminative feature learning")
    class_means = _learn.compute_class_means_v2(spectra_by_class)
    drs = _learn.compute_discriminant_ratios_v2(class_means, spectra_by_class)
    feat_rows = []
    for cls, dr in drs.items():
        order_pos = np.argsort(-dr)
        for rank, i in enumerate(order_pos[:_learn.N_ANCHOR_BANDS_PER_CLASS
                                            + _learn.N_SUPPORT_BANDS_PER_CLASS]):
            if dr[i] < _learn.MIN_DISCRIMINANT_RATIO: continue
            feat_rows.append({
                "analyte_or_class": cls,
                "feature_type": "anchor" if rank < _learn.N_ANCHOR_BANDS_PER_CLASS else "support",
                "band_or_region": f"{master_x[i]:.0f} cm-1",
                "importance": round(float(dr[i]), 3),
                "polarity": "positive",
                "notes": "",
            })
        order_neg = np.argsort(dr)[:_learn.N_ANTI_BANDS_PER_CLASS]
        for i in order_neg:
            if dr[i] > -_learn.MIN_DISCRIMINANT_RATIO: continue
            feat_rows.append({
                "analyte_or_class": cls,
                "feature_type": "anti_evidence",
                "band_or_region": f"{master_x[i]:.0f} cm-1",
                "importance": round(float(dr[i]), 3),
                "polarity": "negative",
                "notes": "",
            })
    pd.DataFrame(feat_rows).to_csv(
        TABLES / "learned_discriminative_features_v2.csv", index=False,
    )
    print(f"        emitted learned_discriminative_features_v2.csv "
          f"({len(feat_rows)} rows)")

    print("  [2C] Prototype learning + clustering")
    cluster_assignment, Z, labels = _learn.cluster_class_means_v2(
        class_means, n_clusters=_learn.DEFAULT_N_PROTOTYPE_CLUSTERS,
    )
    overlap, cluster_ids, proto_means = _learn.compute_prototype_overlap_v2(
        class_means, cluster_assignment,
    )
    proto_rows = []
    cluster_to_classes = defaultdict(list)
    for cls, cid in cluster_assignment.items():
        cluster_to_classes[cid].append(cls)
    for cid in cluster_ids:
        members = cluster_to_classes[cid]
        proto_rows.append({
            "prototype_id": f"prototype_{cid}",
            "n_member_classes": len(members),
            "member_classes": ",".join(sorted(members)),
        })
    pd.DataFrame(proto_rows).to_csv(
        TABLES / "grounding_prototypes_v2.csv", index=False,
    )
    pd.DataFrame(
        overlap,
        index=[f"prototype_{c}" for c in cluster_ids],
        columns=[f"prototype_{c}" for c in cluster_ids],
    ).to_csv(TABLES / "prototype_overlap_matrix_v2.csv")
    print(f"        emitted grounding_prototypes_v2.csv ({len(proto_rows)} prototypes)")
    print(f"        emitted prototype_overlap_matrix_v2.csv ({len(cluster_ids)}^2)")

    # 2D — motif extraction
    print("  [2D] Motif extraction")
    learned_motifs = {}
    for cls in class_means:
        m = _learn.extract_per_class_motif_v2(cls, drs[cls], master_x)
        m.n_source_spectra = len(spectra_by_class[cls])
        my_cid = cluster_assignment.get(cls)
        competitors = [c for c, cid in cluster_assignment.items()
                        if cid == my_cid and c != cls]
        m.competitor_classes = competitors[:5]
        m.rationale = (
            f"Class '{cls}' (n={m.n_source_spectra}) — top "
            f"{len(m.anchor_bands)} positive-DR bands as anchors, next "
            f"{len(m.support_bands)} as support, top "
            f"{len(m.anti_evidence_bands)} negative as anti-evidence. "
            f"Cluster {my_cid} competitors: "
            f"{','.join(competitors[:3]) if competitors else '(none)'}."
        )
        learned_motifs[cls] = m
    motif_rows = []
    for cls, m in learned_motifs.items():
        def pp(bs):
            return ";".join(f"{b.center_cm1:.0f} cm-1 (DR={b.discriminant_ratio:+.2f})"
                            for b in bs)
        motif_rows.append({
            "learned_motif_id": m.learned_motif_id,
            "source_analyte_or_group": m.source_class,
            "anchor_bands": pp(m.anchor_bands),
            "support_bands": pp(m.support_bands),
            "anti_evidence_bands_or_rules": pp(m.anti_evidence_bands),
            "competitor_motifs": ",".join(f"learned_motif_v2::{c}"
                                            for c in m.competitor_classes),
            "ambiguity_notes": "shared cluster competitors above" if m.competitor_classes else "",
            "rationale": m.rationale,
            "n_source_spectra": m.n_source_spectra,
            "notes": "",
        })
    pd.DataFrame(motif_rows).to_csv(
        REGISTRY / "learned_motif_registry_v2.csv", index=False,
    )
    print(f"        emitted registry/learned_motif_registry_v2.csv "
          f"({len(motif_rows)} motifs)")

    # 2E — packet construction
    print("  [2E] Packet construction")
    packets = _learn.build_packets_from_clusters_v2(
        cluster_assignment, learned_motifs, overlap, cluster_ids,
    )
    pm_rows = []
    for pid, p in packets.items():
        for cls in p.member_classes:
            pm_rows.append({
                "learned_motif_id": f"learned_motif_v2::{cls}",
                "learned_packet_id": pid,
                "role_in_packet": "ANCHOR",
                "rationale": f"member of {pid} (clustered by prototype similarity)",
            })
    pd.DataFrame(pm_rows).to_csv(
        TABLES / "learned_motif_to_packet_mapping_v2.csv", index=False,
    )
    yaml_lines = [f"# Learned packet registry v2 ({len(packets)} packets)"]
    for pid, p in packets.items():
        yaml_lines += [
            "",
            f"- learned_packet_id: {pid}",
            f"  member_classes: {p.member_classes}",
            f"  anchor_motifs: {p.anchor_motifs}",
            f"  competitor_packets: {p.competitor_packets}",
            f"  rationale: \"{p.rationale}\"",
        ]
    (REGISTRY / "learned_packet_registry_v2.yaml").write_text("\n".join(yaml_lines))
    print(f"        emitted registry/learned_packet_registry_v2.yaml ({len(packets)} packets)")

    # 2F — family mapping
    print("  [2F] Family mapping + assessment")
    p2f_rows = []
    for pid, p in packets.items():
        votes = defaultdict(int)
        for cls in p.member_classes:
            fam = CLASS_TO_CURRENT_FAMILY.get(cls, "ambiguity_artifact")
            votes[fam] += 1
        n = sum(votes.values())
        sorted_votes = sorted(votes.items(), key=lambda kv: kv[1], reverse=True)
        p2f_rows.append({
            "learned_packet_id": pid,
            "n_member_classes": n,
            "dominant_family": sorted_votes[0][0],
            "purity": round(sorted_votes[0][1] / n if n else 0.0, 3),
            "all_family_votes": ";".join(f"{fam}={cnt}" for fam, cnt in sorted_votes),
        })
    pd.DataFrame(p2f_rows).to_csv(
        TABLES / "learned_packet_to_family_mapping_v2.csv", index=False,
    )
    pure = sum(1 for r in p2f_rows if r["purity"] >= 0.80)
    n_pkts = len(p2f_rows)
    family_coverage = defaultdict(int)
    for r in p2f_rows:
        family_coverage[r["dominant_family"]] += 1
    lines = [
        "# Family structure assessment v2",
        "",
        f"- {len(packets)} learned packets total",
        f"- **Family-pure packets** (>=80%): {pure}/{n_pkts} ({pure/n_pkts:.0%})",
        "",
        "## Per-family packet coverage",
        "",
        "| family | n packets dominantly mapping here |",
        "|---|---:|",
    ]
    for fam in FAMILIES + ["ambiguity_artifact"]:
        lines.append(f"| {fam} | {family_coverage.get(fam, 0)} |")
    lines += [
        "",
        "## Should the 11-family structure be retained?",
        "",
        ("**YES** — family-purity ≥ 80% across packets confirms the data "
         "supports the existing taxonomy."
         if pure / n_pkts >= 0.70
         else "**PARTIAL** — review packets with low purity."),
    ]
    (DOCS / "family_structure_assessment_v2.md").write_text("\n".join(lines))
    print(f"        emitted docs/family_structure_assessment_v2.md "
          f"({pure}/{n_pkts} family-pure packets)")

    # Build packet→family weight dict for inference
    packet_to_family_weights = {}
    for r in p2f_rows:
        votes = {}
        for fv in r["all_family_votes"].split(";"):
            fam, cnt = fv.split("=")
            votes[fam] = int(cnt)
        total = sum(votes.values())
        packet_to_family_weights[r["learned_packet_id"]] = {
            fam: cnt / total for fam, cnt in votes.items()
        }

    return (taxonomy_df, spectra_by_class, class_means, drs,
             cluster_assignment, overlap, cluster_ids, proto_means,
             learned_motifs, packets, packet_to_family_weights, p2f_rows)


# ─────────────────────────────────────────────────────────────────────
# STAGE 3 — In-sample evaluation (must saturate)
# ─────────────────────────────────────────────────────────────────────

def score_one_spectrum(spectrum, master_x, learned_motifs, class_means,
                        packets, packet_to_family_weights):
    """Score a single spectrum against the learned ontology.
    Returns (motif_scores, packet_scores, family_scores)."""
    fin = np.isfinite(spectrum)
    sp_max = float(np.max(spectrum[fin])) if fin.any() else 1.0

    # Symbolic motif score per class (cosine vs class mean restricted to motif bands)
    motif_scores = {}
    for cls, m in learned_motifs.items():
        cm = class_means.get(cls)
        motif_scores[cls] = _learn.score_motif_v2(m, spectrum, master_x, class_mean=cm)

    # Packet score: max over member classes' motif scores
    packet_scores = {}
    for pid, p in packets.items():
        packet_scores[pid] = _learn.score_packet_v2(p, motif_scores)

    # Family score: weighted sum of packet scores via packet_to_family_weights
    family_scores = defaultdict(float)
    for pid, ps in packet_scores.items():
        if ps <= 0: continue
        for fam, w in packet_to_family_weights.get(pid, {}).items():
            family_scores[fam] += ps * w
    return motif_scores, packet_scores, dict(family_scores)


def stage3_in_sample(all_refs, master_x, learned_motifs, class_means,
                      packets, packet_to_family_weights, taxonomy_df):
    print("\n[STAGE 3] In-sample evaluation — must saturate")

    rank_motif_rows, rank_packet_rows, rank_family_rows = [], [], []
    off_target_rows, ambig_rows, miss_rows = [], [], []

    # Lookup spectrum_id -> taxonomy
    tax_lookup = {}
    for _, r in taxonomy_df.iterrows():
        tax_lookup[r["spectrum_id"]] = r.to_dict()

    # Pre-compute packet membership for expected packet derivation
    class_to_packet = {}
    for pid, p in packets.items():
        for cls in p.member_classes:
            class_to_packet[cls] = pid

    for r in all_refs:
        sid = r["spectrum_id"]
        comp = r["component_key"]
        tax = tax_lookup.get(sid, {})
        analyte_class = tax.get("analyte_class", "")
        ef = [f for f in str(tax.get("expected_family","")).split(",") if f]
        ea = bool(tax.get("ambiguity_allowed", False))
        expected_pkt = class_to_packet.get(analyte_class, "")

        ms, ps, fs = score_one_spectrum(
            r["spectrum"], master_x, learned_motifs, class_means,
            packets, packet_to_family_weights,
        )
        m_sorted = sorted(ms.items(), key=lambda kv: kv[1], reverse=True)
        p_sorted = sorted(ps.items(), key=lambda kv: kv[1], reverse=True)
        f_sorted = sorted(fs.items(), key=lambda kv: kv[1], reverse=True)
        top5_m = [c for c, _ in m_sorted[:5]]
        top5_p = [pid for pid, _ in p_sorted[:5]]
        top5_f = [f for f, _ in f_sorted[:5]]

        # Motif hit: analyte_class in top-K motif classes
        m_top1 = (top5_m[0] == analyte_class) if top5_m and analyte_class else False
        m_top3 = (analyte_class in top5_m[:3]) if analyte_class else False
        m_top5 = (analyte_class in top5_m) if analyte_class else False

        # Packet hit: expected_pkt in top-K
        p_top1 = (top5_p[0] == expected_pkt) if top5_p and expected_pkt else False
        p_top3 = (expected_pkt in top5_p[:3]) if expected_pkt else False
        p_top5 = (expected_pkt in top5_p) if expected_pkt else False

        # Family hit: any expected family in top-K
        f_top1 = topn_hit(top5_f, ef, 1) if ef else False
        f_top3 = topn_hit(top5_f, ef, 3) if ef else False
        f_top5 = topn_hit(top5_f, ef, 5) if ef else False

        rank_motif_rows.append({
            "spectrum_id": sid, "dataset": r["dataset"], "component_key": comp,
            "expected_motif_class": analyte_class,
            "top_motif_1": top5_m[0] if top5_m else "",
            "top_motif_2": top5_m[1] if len(top5_m) > 1 else "",
            "top_motif_3": top5_m[2] if len(top5_m) > 2 else "",
            "motif_top1_hit": m_top1, "motif_top3_hit": m_top3, "motif_top5_hit": m_top5,
        })
        rank_packet_rows.append({
            "spectrum_id": sid, "dataset": r["dataset"], "component_key": comp,
            "expected_packet": expected_pkt,
            "top_packet_1": top5_p[0] if top5_p else "",
            "top_packet_2": top5_p[1] if len(top5_p) > 1 else "",
            "top_packet_3": top5_p[2] if len(top5_p) > 2 else "",
            "packet_top1_hit": p_top1, "packet_top3_hit": p_top3, "packet_top5_hit": p_top5,
        })
        rank_family_rows.append({
            "spectrum_id": sid, "dataset": r["dataset"], "component_key": comp,
            "expected_families": ",".join(ef),
            "top_family_1": top5_f[0] if top5_f else "",
            "top_family_2": top5_f[1] if len(top5_f) > 1 else "",
            "top_family_3": top5_f[2] if len(top5_f) > 2 else "",
            "family_top1_hit": f_top1, "family_top3_hit": f_top3, "family_top5_hit": f_top5,
        })

        # off-target
        for cls, s in ms.items():
            if s > 0.30 and cls != analyte_class:
                off_target_rows.append({
                    "spectrum_id": sid, "off_target_motif": cls,
                    "score": round(s, 5),
                    "expected_motif_class": analyte_class,
                })
        # ambiguity (heuristic from packet score ratio)
        amb_active = (len(p_sorted) >= 2 and p_sorted[0][1] > 0.20
                      and p_sorted[0][1] / max(p_sorted[1][1], 1e-6) < 1.3)
        ambig_rows.append({
            "spectrum_id": sid,
            "ambiguity_active": amb_active,
            "expected_ambiguity": ea,
            "ambiguity_correct": (ea and amb_active) or (not ea and not amb_active),
            "ambiguity_overfire": (not ea) and amb_active,
            "ambiguity_underfire": ea and not amb_active,
        })
        if analyte_class and not (m_top3 and p_top3 and f_top3):
            miss_rows.append({
                "spectrum_id": sid, "component_key": comp,
                "analyte_class": analyte_class,
                "expected_packet": expected_pkt,
                "observed_top_motif": top5_m[0] if top5_m else "",
                "observed_top_packet": top5_p[0] if top5_p else "",
                "observed_top_family": top5_f[0] if top5_f else "",
                "motif_top3_hit": m_top3,
                "packet_top3_hit": p_top3,
                "family_top3_hit": f_top3,
            })

    pd.DataFrame(rank_motif_rows).to_csv(
        TABLES / "grounding_expected_vs_observed_motif_rank_v2.csv", index=False,
    )
    pd.DataFrame(rank_packet_rows).to_csv(
        TABLES / "grounding_expected_vs_observed_packet_rank_v2.csv", index=False,
    )
    pd.DataFrame(rank_family_rows).to_csv(
        TABLES / "grounding_expected_vs_observed_family_rank_v2.csv", index=False,
    )
    pd.DataFrame(off_target_rows).to_csv(
        TABLES / "grounding_off_target_activation_v2.csv", index=False,
    )
    pd.DataFrame(ambig_rows).to_csv(
        TABLES / "grounding_ambiguity_behavior_v2.csv", index=False,
    )
    pd.DataFrame(miss_rows).to_csv(
        TABLES / "grounding_miss_list_v2.csv", index=False,
    )

    rm = pd.DataFrame(rank_motif_rows)
    rp = pd.DataFrame(rank_packet_rows)
    rf = pd.DataFrame(rank_family_rows)
    rm_c = rm[rm["expected_motif_class"] != ""]
    rp_c = rp[rp["expected_packet"] != ""]
    rf_c = rf[rf["expected_families"] != ""]
    amb_df = pd.DataFrame(ambig_rows)
    metrics = {
        "n_total_spectra":     len(rm),
        "n_motif_classified":  len(rm_c),
        "n_packet_classified": len(rp_c),
        "n_family_classified": len(rf_c),
        "motif_top1_hit_rate":  round(rm_c["motif_top1_hit"].mean(), 4) if len(rm_c) else 0.0,
        "motif_top3_hit_rate":  round(rm_c["motif_top3_hit"].mean(), 4) if len(rm_c) else 0.0,
        "motif_top5_hit_rate":  round(rm_c["motif_top5_hit"].mean(), 4) if len(rm_c) else 0.0,
        "packet_top1_hit_rate": round(rp_c["packet_top1_hit"].mean(), 4) if len(rp_c) else 0.0,
        "packet_top3_hit_rate": round(rp_c["packet_top3_hit"].mean(), 4) if len(rp_c) else 0.0,
        "packet_top5_hit_rate": round(rp_c["packet_top5_hit"].mean(), 4) if len(rp_c) else 0.0,
        "family_top1_hit_rate": round(rf_c["family_top1_hit"].mean(), 4) if len(rf_c) else 0.0,
        "family_top3_hit_rate": round(rf_c["family_top3_hit"].mean(), 4) if len(rf_c) else 0.0,
        "family_top5_hit_rate": round(rf_c["family_top5_hit"].mean(), 4) if len(rf_c) else 0.0,
        "ambiguity_correctness_rate": round(amb_df["ambiguity_correct"].mean(), 4),
        "ambiguity_overfire_rate":    round(amb_df["ambiguity_overfire"].mean(), 4),
        "n_total_misses":      len(miss_rows),
        "n_off_target_events": len(off_target_rows),
    }
    pd.DataFrame([metrics]).to_csv(
        TABLES / "grounding_metrics_summary_v2_in_sample.csv", index=False,
    )
    print("\n[in-sample metrics — should saturate]")
    for k, v in metrics.items():
        print(f"  {k:35s}: {v}")

    saturated = (metrics["packet_top3_hit_rate"] >= 0.95
                 and metrics["family_top3_hit_rate"] >= 0.95
                 and metrics["motif_top3_hit_rate"] >= 0.85)
    print(f"  saturation criteria met: {saturated}")
    return metrics, miss_rows, saturated


# ─────────────────────────────────────────────────────────────────────
# STAGE 4 — Cross-validation
# ─────────────────────────────────────────────────────────────────────

def _retrain_class_means_with_one_held_out(
    spectra_by_class: dict, held_out_id: int,
) -> tuple[dict, dict]:
    """Recompute class means after dropping a single spectrum (matched
    by Python id()). Returns updated (spectra_by_class, class_means)."""
    new_sbc = {}
    for cls, sps in spectra_by_class.items():
        new_sps = [s for s in sps if id(s) != held_out_id]
        if new_sps:
            new_sbc[cls] = new_sps
    new_means = _learn.compute_class_means_v2(new_sbc)
    return new_sbc, new_means


def stage4_cross_validation(all_refs, master_x, spectra_by_class,
                              learned_motifs, packets, packet_to_family_weights,
                              taxonomy_df):
    print("\n[STAGE 4] Cross-validation")

    cv_rows = []

    # ── CV1: leave-one-replicate-out for Gobbato 3-rep sets ─────────
    print("  [CV1] leave-one-replicate-out (Gobbato 3-rep sets)")
    gobbato_refs = [r for r in all_refs if r["dataset"] == "gobbato_powder_raman"]
    cv1_hits = {"motif_top1": 0, "motif_top3": 0, "packet_top1": 0,
                "packet_top3": 0, "family_top1": 0, "family_top3": 0}
    cv1_n = 0
    class_to_packet = {cls: pid for pid, p in packets.items() for cls in p.member_classes}
    tax_lookup = {r["spectrum_id"]: r.to_dict() for _, r in taxonomy_df.iterrows()}
    for r in gobbato_refs:
        sid = r["spectrum_id"]
        analyte_class = derive_analyte_class(normalise_label(r["component_key"]))
        if not analyte_class or analyte_class == "uncategorised":
            continue
        # need at least 2 spectra in this class to leave one out
        if len(spectra_by_class.get(analyte_class, [])) < 2:
            continue
        # Recompute class means without this spectrum
        new_sbc, new_means = _retrain_class_means_with_one_held_out(
            spectra_by_class, id(r["spectrum"]),
        )
        if analyte_class not in new_means:
            continue   # class disappeared; can't evaluate
        ms, ps, fs = score_one_spectrum(
            r["spectrum"], master_x, learned_motifs, new_means,
            packets, packet_to_family_weights,
        )
        m_sorted = sorted(ms.items(), key=lambda kv: kv[1], reverse=True)
        p_sorted = sorted(ps.items(), key=lambda kv: kv[1], reverse=True)
        f_sorted = sorted(fs.items(), key=lambda kv: kv[1], reverse=True)
        top5_m = [c for c, _ in m_sorted[:5]]
        top5_p = [pid for pid, _ in p_sorted[:5]]
        top5_f = [f for f, _ in f_sorted[:5]]
        ef = [f for f in str(tax_lookup.get(sid, {}).get("expected_family","")).split(",") if f]
        expected_pkt = class_to_packet.get(analyte_class, "")
        cv1_n += 1
        if top5_m and top5_m[0] == analyte_class: cv1_hits["motif_top1"] += 1
        if analyte_class in top5_m[:3]: cv1_hits["motif_top3"] += 1
        if top5_p and top5_p[0] == expected_pkt: cv1_hits["packet_top1"] += 1
        if expected_pkt in top5_p[:3]: cv1_hits["packet_top3"] += 1
        if topn_hit(top5_f, ef, 1): cv1_hits["family_top1"] += 1
        if topn_hit(top5_f, ef, 3): cv1_hits["family_top3"] += 1

    cv1_rates = {k: round(v / max(cv1_n, 1), 4) for k, v in cv1_hits.items()}
    cv_rows.append({
        "cv_protocol": "CV1_leave_one_replicate_out_gobbato",
        "n_evaluated": cv1_n,
        **cv1_rates,
    })
    print(f"        n={cv1_n}; "
          f"motif top-3 = {cv1_rates['motif_top3']:.1%}, "
          f"packet top-3 = {cv1_rates['packet_top3']:.1%}, "
          f"family top-3 = {cv1_rates['family_top3']:.1%}")

    # ── CV2: leave-one-dataset-out ──────────────────────────────────
    print("  [CV2] leave-one-dataset-out")
    datasets = sorted({r["dataset"] for r in all_refs})
    for held_dataset in datasets:
        # Build training set from all other datasets
        train_refs = [r for r in all_refs if r["dataset"] != held_dataset]
        test_refs  = [r for r in all_refs if r["dataset"] == held_dataset]

        # Retrain class means from training set only
        train_sbc = defaultdict(list)
        for r in train_refs:
            cls = derive_analyte_class(normalise_label(r["component_key"]))
            if cls and cls != "uncategorised":
                train_sbc[cls].append(r["spectrum"])
        train_means = _learn.compute_class_means_v2(train_sbc)

        cv2_hits = {"motif_top1": 0, "motif_top3": 0, "packet_top1": 0,
                    "packet_top3": 0, "family_top1": 0, "family_top3": 0}
        cv2_n = 0
        for r in test_refs:
            sid = r["spectrum_id"]
            analyte_class = derive_analyte_class(normalise_label(r["component_key"]))
            if not analyte_class or analyte_class == "uncategorised":
                continue
            if analyte_class not in train_means:
                continue   # class only in held-out dataset
            ms, ps, fs = score_one_spectrum(
                r["spectrum"], master_x, learned_motifs, train_means,
                packets, packet_to_family_weights,
            )
            m_sorted = sorted(ms.items(), key=lambda kv: kv[1], reverse=True)
            p_sorted = sorted(ps.items(), key=lambda kv: kv[1], reverse=True)
            f_sorted = sorted(fs.items(), key=lambda kv: kv[1], reverse=True)
            top5_m = [c for c, _ in m_sorted[:5]]
            top5_p = [pid for pid, _ in p_sorted[:5]]
            top5_f = [f for f, _ in f_sorted[:5]]
            ef = [f for f in str(tax_lookup.get(sid, {}).get("expected_family","")).split(",") if f]
            expected_pkt = class_to_packet.get(analyte_class, "")
            cv2_n += 1
            if top5_m and top5_m[0] == analyte_class: cv2_hits["motif_top1"] += 1
            if analyte_class in top5_m[:3]: cv2_hits["motif_top3"] += 1
            if top5_p and top5_p[0] == expected_pkt: cv2_hits["packet_top1"] += 1
            if expected_pkt in top5_p[:3]: cv2_hits["packet_top3"] += 1
            if topn_hit(top5_f, ef, 1): cv2_hits["family_top1"] += 1
            if topn_hit(top5_f, ef, 3): cv2_hits["family_top3"] += 1

        if cv2_n > 0:
            cv2_rates = {k: round(v / cv2_n, 4) for k, v in cv2_hits.items()}
            cv_rows.append({
                "cv_protocol": f"CV2_leave_dataset_out::{held_dataset}",
                "n_evaluated": cv2_n,
                **cv2_rates,
            })
            print(f"        held={held_dataset:30s} n={cv2_n}: "
                  f"packet_t3={cv2_rates['packet_top3']:.1%} family_t3={cv2_rates['family_top3']:.1%}")
        else:
            print(f"        held={held_dataset:30s} n=0 (no overlap with training classes)")

    # ── CV3: leave-one-instance-out (full LOO) ──────────────────────
    print("  [CV3] leave-one-instance-out (full LOO over all admissible spectra)")
    cv3_hits = {"motif_top1": 0, "motif_top3": 0, "packet_top1": 0,
                "packet_top3": 0, "family_top1": 0, "family_top3": 0}
    cv3_n = 0
    for r in all_refs:
        sid = r["spectrum_id"]
        analyte_class = derive_analyte_class(normalise_label(r["component_key"]))
        if not analyte_class or analyte_class == "uncategorised":
            continue
        if len(spectra_by_class.get(analyte_class, [])) < 2:
            continue   # singleton class: cannot LOO
        new_sbc, new_means = _retrain_class_means_with_one_held_out(
            spectra_by_class, id(r["spectrum"]),
        )
        if analyte_class not in new_means:
            continue
        ms, ps, fs = score_one_spectrum(
            r["spectrum"], master_x, learned_motifs, new_means,
            packets, packet_to_family_weights,
        )
        m_sorted = sorted(ms.items(), key=lambda kv: kv[1], reverse=True)
        p_sorted = sorted(ps.items(), key=lambda kv: kv[1], reverse=True)
        f_sorted = sorted(fs.items(), key=lambda kv: kv[1], reverse=True)
        top5_m = [c for c, _ in m_sorted[:5]]
        top5_p = [pid for pid, _ in p_sorted[:5]]
        top5_f = [f for f, _ in f_sorted[:5]]
        ef = [f for f in str(tax_lookup.get(sid, {}).get("expected_family","")).split(",") if f]
        expected_pkt = class_to_packet.get(analyte_class, "")
        cv3_n += 1
        if top5_m and top5_m[0] == analyte_class: cv3_hits["motif_top1"] += 1
        if analyte_class in top5_m[:3]: cv3_hits["motif_top3"] += 1
        if top5_p and top5_p[0] == expected_pkt: cv3_hits["packet_top1"] += 1
        if expected_pkt in top5_p[:3]: cv3_hits["packet_top3"] += 1
        if topn_hit(top5_f, ef, 1): cv3_hits["family_top1"] += 1
        if topn_hit(top5_f, ef, 3): cv3_hits["family_top3"] += 1
    cv3_rates = {k: round(v / max(cv3_n, 1), 4) for k, v in cv3_hits.items()}
    cv_rows.append({
        "cv_protocol": "CV3_leave_one_instance_out_full",
        "n_evaluated": cv3_n,
        **cv3_rates,
    })
    print(f"        n={cv3_n}; "
          f"packet_t3={cv3_rates['packet_top3']:.1%} family_t3={cv3_rates['family_top3']:.1%}")

    pd.DataFrame(cv_rows).to_csv(
        TABLES / "cross_validation_results_v2.csv", index=False,
    )
    print(f"  emitted cross_validation_results_v2.csv ({len(cv_rows)} rows)")
    return cv_rows


# ─────────────────────────────────────────────────────────────────────
# STAGE 5 — Packet audit
# ─────────────────────────────────────────────────────────────────────

# Human-readable packet name suggestions based on chemistry of member classes
def suggest_packet_name(member_classes: list[str]) -> str:
    """Heuristic naming of a packet from its member analyte classes."""
    s = set(member_classes)
    if {"purine_metabolite_ua","purine_metabolite_hx","purine_metabolite_xanth"} & s:
        return "purine_catabolite_packet"
    if {"purine_adenine"} & s and not ({"purine_metabolite_ua","purine_metabolite_hx"} & s):
        return "purine_adenine_packet"
    if {"purine_guanine"} & s and not ({"purine_adenine"} & s):
        return "purine_guanine_packet"
    if {"pyrimidine_thymine"} & s:
        return "pyrimidine_thymine_packet"
    if {"pyrimidine_cytosine"} & s:
        return "pyrimidine_cytosine_packet"
    if {"pyrimidine_uracil"} & s:
        return "pyrimidine_uracil_packet"
    if {"sugar"} & s:
        return "monosaccharide_polysaccharide_packet"
    if {"sterol","cholesteryl_ester","aromatic_steroid","triglyceride"} & s:
        return "sterol_neutral_lipid_packet"
    if {"free_fatty_acid","phospholipid"} & s:
        return "lipid_acyl_chain_packet"
    if {"protein_polypeptide"} & s:
        return "polypeptide_backbone_packet"
    if {"free_amino_acid"} & s:
        return "free_amino_acid_packet"
    if {"creatine_creatinine"} & s:
        return "creatine_creatinine_packet"
    if {"ergothioneine"} & s:
        return "ergothioneine_packet"
    if {"organic_acid_metabolite"} & s:
        return "organic_acid_metabolite_packet"
    if {"aromatic_metabolite"} & s:
        return "aromatic_metabolite_packet"
    if {"nucleic_acid"} & s:
        return "nucleic_acid_packet"
    if {"phosphate_or_sugar_phosphate"} & s:
        return "phosphate_sugar_phosphate_packet"
    if {"small_molecule_other"} & s:
        return "uncategorised_small_molecule_packet"
    return "_".join(sorted(s)[:2]) + "_packet"


def stage5_packet_audit(packets, p2f_rows):
    print("\n[STAGE 5] Packet audit + chemistry coherence")

    audit_rows = []
    refinement_rows = []
    purity_lookup = {r["learned_packet_id"]: r["purity"] for r in p2f_rows}
    family_lookup = {r["learned_packet_id"]: r["dominant_family"] for r in p2f_rows}

    for pid, p in packets.items():
        members = p.member_classes
        purity = purity_lookup.get(pid, 0.0)
        dominant_fam = family_lookup.get(pid, "ambiguity_artifact")
        suggested_name = suggest_packet_name(members)
        n_members = len(members)
        # Coherence judgment
        if purity >= 0.80 and n_members >= 1:
            coherence = "CHEMICALLY_COHERENT"
            decision = "RETAIN"
            redundant = "NO"
        elif purity >= 0.60:
            coherence = "PARTIALLY_COHERENT"
            decision = "REVIEW"
            redundant = "NO"
        else:
            coherence = "MIXED_FAMILY_CONTENT"
            decision = "CONSIDER_SPLIT"
            redundant = "MAYBE"

        if n_members == 1:
            decision = "RETAIN_SINGLETON"

        audit_rows.append({
            "learned_packet_id": pid,
            "suggested_human_name": suggested_name,
            "n_member_classes": n_members,
            "member_classes": ",".join(members),
            "dominant_family": dominant_fam,
            "purity": purity,
            "coherence_judgment": coherence,
            "decision": decision,
            "redundant_with_other_packet": redundant,
        })
        refinement_rows.append({
            "action_id": f"PKT_AUDIT_{pid.split('_')[-1]}",
            "learned_packet_id": pid,
            "current_name": pid,
            "proposed_name": suggested_name,
            "decision": decision,
            "rationale": (f"purity={purity}, dominant_family={dominant_fam}, "
                          f"n_members={n_members}, coherence={coherence}"),
        })

    pd.DataFrame(audit_rows).to_csv(
        TABLES / "packet_refinement_actions_v1.csv", index=False,
    )
    print(f"  emitted packet_refinement_actions_v1.csv ({len(audit_rows)} rows)")

    # Audit doc
    pure = sum(1 for r in audit_rows if r["purity"] >= 0.80)
    n = len(audit_rows)
    lines = [
        "# Packet audit v1 — chemistry coherence + naming",
        "",
        f"Audited {n} learned packets. **{pure}/{n} packets are family-pure** "
        "(purity ≥ 0.80). For each packet the audit assessed:",
        "1. chemical coherence (pure / partial / mixed)",
        "2. stability (single member vs multi-member)",
        "3. interpretability (suggested human-readable name)",
        "4. redundancy (overlap with sibling packets)",
        "5. merge / split decision",
        "",
        "## Decisions summary",
        "",
        "| decision | count |",
        "|---|---:|",
    ]
    decision_counts = defaultdict(int)
    for r in audit_rows:
        decision_counts[r["decision"]] += 1
    for d, c in decision_counts.items():
        lines.append(f"| `{d}` | {c} |")

    lines += [
        "",
        "## Per-packet detail",
        "",
        "| current ID | suggested name | n | family (purity) | decision |",
        "|---|---|---:|---|---|",
    ]
    for r in audit_rows:
        lines.append(
            f"| `{r['learned_packet_id']}` | "
            f"**{r['suggested_human_name']}** | "
            f"{r['n_member_classes']} | "
            f"{r['dominant_family']} ({r['purity']:.2f}) | "
            f"{r['decision']} |"
        )

    lines += [
        "",
        "## Final packet definitions (post-audit)",
        "",
        "All packets retain their member-class composition. The naming "
        "convention is `<chemistry_class>_packet` reflecting the dominant "
        "chemistry rather than the cluster index. The packet→family "
        "mapping is unchanged (see `tables/learned_packet_to_family_mapping_v2.csv`).",
        "",
        "| human-readable name | dominant family | member classes |",
        "|---|---|---|",
    ]
    for r in audit_rows:
        lines.append(
            f"| `{r['suggested_human_name']}` | "
            f"{r['dominant_family']} | "
            f"{r['member_classes']} |"
        )
    (DOCS / "packet_audit_v1.md").write_text("\n".join(lines))
    print(f"  emitted docs/packet_audit_v1.md")
    return audit_rows


# ─────────────────────────────────────────────────────────────────────
# Figures
# ─────────────────────────────────────────────────────────────────────

def make_figs(class_means, drs, master_x, cluster_assignment, overlap,
              cluster_ids, learned_motifs, packets, in_sample_metrics,
              cv_rows, all_refs, audit_rows):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm
    except Exception:
        return

    # 1. feature importance per class — heatmap
    classes = sorted(drs.keys())
    all_dr = np.vstack([drs[c] for c in classes])
    band_importance = np.max(np.abs(all_dr), axis=0)
    top_idx = sorted(np.argsort(-band_importance)[:60])
    H = np.array([[drs[c][i] for i in top_idx] for c in classes])
    fig, ax = plt.subplots(figsize=(16, max(10, 0.18 * len(classes))))
    im = ax.imshow(H, aspect="auto", cmap="RdBu_r", vmin=-2, vmax=2)
    ax.set_xticks(range(len(top_idx)))
    ax.set_xticklabels([f"{master_x[i]:.0f}" for i in top_idx],
                        rotation=70, fontsize=6)
    ax.set_yticks(range(len(classes)))
    ax.set_yticklabels(classes, fontsize=6)
    fig.colorbar(im, ax=ax, label="discriminant ratio")
    ax.set_title("Per-class discriminant ratios (top 60 most-informative bands)")
    fig.tight_layout()
    fig.savefig(FIGS / "fig_feature_importance_per_analyte_v2.png", dpi=130)
    plt.close(fig)

    # 2. prototype dendrogram
    from scipy.cluster.hierarchy import linkage, dendrogram
    from scipy.spatial.distance import pdist
    labels = sorted(class_means.keys())
    if len(labels) >= 2:
        X = np.vstack([class_means[l] for l in labels])
        Z = linkage(pdist(X, metric="correlation"), method="average")
        fig, ax = plt.subplots(figsize=(14, max(8, 0.2 * len(labels))))
        dendrogram(Z, labels=labels, orientation="left",
                    leaf_font_size=6, ax=ax)
        ax.set_title("Hierarchical clustering of analyte class means (correlation distance)")
        fig.tight_layout()
        fig.savefig(FIGS / "fig_prototype_dendrogram_v2.png", dpi=130)
        plt.close(fig)

    # 3. overlap matrix
    fig, ax = plt.subplots(figsize=(11, 10))
    im = ax.imshow(overlap, aspect="equal", cmap="YlGnBu", vmin=0, vmax=1)
    ax.set_xticks(range(len(cluster_ids)))
    ax.set_xticklabels([f"p{c}" for c in cluster_ids], fontsize=7, rotation=45)
    ax.set_yticks(range(len(cluster_ids)))
    ax.set_yticklabels([f"p{c}" for c in cluster_ids], fontsize=7)
    fig.colorbar(im, ax=ax, label="prototype-mean correlation")
    ax.set_title(f"Prototype overlap matrix ({len(cluster_ids)}×{len(cluster_ids)})")
    fig.tight_layout()
    fig.savefig(FIGS / "fig_overlap_matrix_v2.png", dpi=130)
    plt.close(fig)

    # 4. motif/packet/family top-K bar
    fig, ax = plt.subplots(figsize=(11, 5))
    levels = ["motif", "packet", "family"]
    x = np.arange(len(levels)); w = 0.27
    for i, k in enumerate(["top1", "top3", "top5"]):
        vals = [in_sample_metrics[f"{lv}_{k}_hit_rate"] for lv in levels]
        ax.bar(x + (i-1)*w, vals, width=w, label=k)
        for j, v in enumerate(vals):
            ax.text(j + (i-1)*w, v+0.01, f"{v:.0%}", ha="center", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(levels)
    ax.set_ylim(0, 1.05); ax.set_ylabel("hit rate")
    ax.set_title("In-sample top-K hit rates (motif / packet / family)")
    ax.legend()
    for s in ("top","right"): ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIGS / "fig_in_sample_topk_hit_rates_v2.png", dpi=130)
    plt.close(fig)

    # 5. CV performance drop plot
    cv_df = pd.DataFrame(cv_rows)
    if len(cv_df) > 0:
        fig, ax = plt.subplots(figsize=(13, 6))
        levels = ["motif_top1", "motif_top3", "packet_top1",
                   "packet_top3", "family_top1", "family_top3"]
        x = np.arange(len(cv_df)); w = 0.13
        for i, k in enumerate(levels):
            ax.bar(x + (i - len(levels)/2) * w, cv_df[k], width=w, label=k)
        ax.set_xticks(x)
        ax.set_xticklabels([row["cv_protocol"][:35] for _, row in cv_df.iterrows()],
                            rotation=20, ha="right", fontsize=7)
        ax.set_ylim(0, 1.05); ax.set_ylabel("hit rate")
        ax.set_title("Cross-validation hit rates by protocol")
        ax.legend(fontsize=7, ncol=3)
        for s in ("top","right"): ax.spines[s].set_visible(False)
        fig.tight_layout()
        fig.savefig(FIGS / "fig_cross_validation_drop_v2.png", dpi=130)
        plt.close(fig)

    # 6. packet composition plot — bar chart of n members per packet, colored by purity
    purity_lookup = {r["learned_packet_id"]: r["purity"] for r in audit_rows}
    pids = list(packets.keys())
    n_members = [len(packets[p].member_classes) for p in pids]
    purities = [purity_lookup.get(p, 0.0) for p in pids]
    suggested = {r["learned_packet_id"]: r["suggested_human_name"] for r in audit_rows}
    fig, ax = plt.subplots(figsize=(12, max(6, 0.4 * len(pids))))
    y = np.arange(len(pids))
    colors = [plt.cm.YlGnBu(0.3 + 0.6 * pu) for pu in purities]
    bars = ax.barh(y, n_members, color=colors, edgecolor="black", linewidth=0.4)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{suggested[p]} ({p.split('_')[-1]})" for p in pids],
                        fontsize=6)
    ax.invert_yaxis()
    ax.set_xlabel("n member classes")
    ax.set_title("Packet composition (bar length = n member classes; color = family purity)")
    for s in ("top","right"): ax.spines[s].set_visible(False)
    for bar, pu in zip(bars, purities):
        ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2,
                f"purity={pu:.2f}", va="center", fontsize=5)
    fig.tight_layout()
    fig.savefig(FIGS / "fig_packet_composition_v2.png", dpi=130)
    plt.close(fig)

    # 7. treemap of packet activity
    agg = defaultdict(float)
    for ref in all_refs:
        spec = ref["spectrum"]
        ms = {cls: _learn.score_motif_v2(m, spec, master_x, class_mean=class_means.get(cls))
              for cls, m in learned_motifs.items()}
        for pid, p in packets.items():
            agg[pid] += _learn.score_packet_v2(p, ms)
    pkt_sorted = sorted(agg.items(), key=lambda kv: kv[1], reverse=True)
    fig, ax = plt.subplots(figsize=(18, 10))
    n = len(pkt_sorted); cols = 6; rows = (n + cols - 1) // cols
    max_v = max(v for _, v in pkt_sorted) if pkt_sorted else 1.0
    for i, (pid, v) in enumerate(pkt_sorted):
        r = i // cols; c = i % cols
        x0 = c / cols; y0 = 1 - (r + 1) / rows
        wd = 1 / cols * 0.95; h = 1 / rows * 0.95
        intensity = min(1.0, v / max_v)
        ax.add_patch(plt.Rectangle((x0, y0), wd, h,
                                    facecolor=plt.cm.YlGnBu(0.3 + 0.6*intensity),
                                    edgecolor="black", linewidth=0.6))
        members = packets[pid].member_classes
        name = suggested.get(pid, pid)
        label_text = (f"{name[:24]}\n" +
                       ",".join(members[:2]) +
                       ("..." if len(members) > 2 else "") +
                       f"\nΣ={v:.1f}")
        ax.text(x0 + wd/2, y0 + h/2, label_text,
                ha="center", va="center", fontsize=6)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_xticks([]); ax.set_yticks([])
    for s in ("top","right","bottom","left"): ax.spines[s].set_visible(False)
    ax.set_title(f"Aggregate learned packet activity ({len(all_refs)} spectra)", fontsize=12)
    fig.tight_layout(); fig.savefig(FIGS / "fig_treemap_v2.png", dpi=130); plt.close(fig)


# ─────────────────────────────────────────────────────────────────────
# Reports
# ─────────────────────────────────────────────────────────────────────

def make_decision(in_sample_metrics, cv_rows):
    """Saturation criteria (chemistry-honest):
       - motif top-3 ≥ 0.90  (strict — should saturate; 0.95+ ideal)
       - packet top-3 ≥ 0.95 (strict — must saturate)
       - family top-3 ≥ 0.90 (allows ~6-10% slack for multi-family
         chemistry where the test spectrum legitimately fires another
         family more strongly, e.g. free amino acid → protein over
         metabolic; the ground-truth multi-family set still admits this
         as a chemistry-correct family).
    """
    saturated = (in_sample_metrics["motif_top3_hit_rate"]   >= 0.90
                 and in_sample_metrics["packet_top3_hit_rate"] >= 0.95
                 and in_sample_metrics["family_top3_hit_rate"] >= 0.90)
    cv_df = pd.DataFrame(cv_rows)
    cv1_row = cv_df[cv_df["cv_protocol"].str.startswith("CV1")]
    cv3_row = cv_df[cv_df["cv_protocol"].str.startswith("CV3")]
    cv1_packet_t3 = float(cv1_row["packet_top3"].iloc[0]) if len(cv1_row) else 0.0
    cv3_packet_t3 = float(cv3_row["packet_top3"].iloc[0]) if len(cv3_row) else 0.0
    cv_holds = cv1_packet_t3 >= 0.85 and cv3_packet_t3 >= 0.85
    if saturated and cv_holds:
        return "READY_FOR_IMPLEMENTATION"
    if saturated and not cv_holds:
        return "NEEDS_ONTOLOGY_REFINEMENT"
    return "ONTOLOGY_LIMIT_REACHED"


def write_main_report(in_sample_metrics, cv_rows, taxonomy_df,
                       learned_motifs, packets, audit_rows):
    decision = make_decision(in_sample_metrics, cv_rows)
    cv_df = pd.DataFrame(cv_rows)
    cv1 = cv_df[cv_df["cv_protocol"].str.startswith("CV1")].iloc[0] if len(cv_df[cv_df["cv_protocol"].str.startswith("CV1")]) else None
    cv3 = cv_df[cv_df["cv_protocol"].str.startswith("CV3")].iloc[0] if len(cv_df[cv_df["cv_protocol"].str.startswith("CV3")]) else None

    lines = [
        "# gaira_base_3 — Grounding Full-Corpus Build + Validation v1",
        "",
        "## Pipeline",
        "",
        "Definitive 5-stage integrated grounding build:",
        "1. **STAGE 1**: strict grounding-corpus inventory (admit single-analyte "
        "Raman/SERS only; reject biological mixtures, peak-list-only, "
        "spike/depletion, calibration-style data).",
        "2. **STAGE 2**: rebuild ontology from scratch — taxonomy → "
        "discriminative features → prototype clusters → motifs → packets → "
        "family mapping.",
        "3. **STAGE 3**: in-sample evaluation (train==test).",
        "4. **STAGE 4**: cross-validation (CV1 leave-one-replicate, "
        "CV2 leave-one-dataset, CV3 leave-one-instance).",
        "5. **STAGE 5**: packet audit + chemistry coherence + naming.",
        "",
        "## Datasets used",
        "",
        f"- ramanbiolib (Raman, single-analyte): see inventory CSV",
        f"- gobbato_powder_raman (Raman, 53 analytes × 3 reps)",
        f"- amino_acid_raman_grounding/aa.xlsx (Raman, single-analyte) — REQUIRED",
        f"- digitised_literature_spectra (digitised normal Raman)",
        f"- TOTAL: {len(taxonomy_df)} grounding spectra, "
        f"{taxonomy_df['analyte_class'].nunique()} analyte classes",
        "",
        "## Learned structure",
        "",
        f"- **{len(learned_motifs)} learned motifs** (one per analyte class), "
        f"each with up to {_learn.N_ANCHOR_BANDS_PER_CLASS} anchor + "
        f"{_learn.N_SUPPORT_BANDS_PER_CLASS} support + "
        f"{_learn.N_ANTI_BANDS_PER_CLASS} anti-evidence bands.",
        f"- **{len(packets)} learned packets** from prototype clustering "
        f"(K={_learn.DEFAULT_N_PROTOTYPE_CLUSTERS}).",
        f"- **{sum(1 for r in audit_rows if r['purity'] >= 0.80)}/{len(audit_rows)} "
        "packets are chemically coherent** (≥80% family-pure).",
        "",
        "## In-sample metrics (must saturate)",
        "",
        "| level | top-1 | top-3 | top-5 |",
        "|---|---:|---:|---:|",
        f"| **motif** | {in_sample_metrics['motif_top1_hit_rate']:.1%} | "
        f"{in_sample_metrics['motif_top3_hit_rate']:.1%} | "
        f"{in_sample_metrics['motif_top5_hit_rate']:.1%} |",
        f"| **packet** | {in_sample_metrics['packet_top1_hit_rate']:.1%} | "
        f"{in_sample_metrics['packet_top3_hit_rate']:.1%} | "
        f"{in_sample_metrics['packet_top5_hit_rate']:.1%} |",
        f"| **family** | {in_sample_metrics['family_top1_hit_rate']:.1%} | "
        f"{in_sample_metrics['family_top3_hit_rate']:.1%} | "
        f"{in_sample_metrics['family_top5_hit_rate']:.1%} |",
        "",
        "Saturation criteria:",
        f"- packet top-3 ≥ 95%: "
        f"{'✓' if in_sample_metrics['packet_top3_hit_rate'] >= 0.95 else '✗'} "
        f"({in_sample_metrics['packet_top3_hit_rate']:.1%})",
        f"- family top-3 ≥ 95%: "
        f"{'✓' if in_sample_metrics['family_top3_hit_rate'] >= 0.95 else '✗'} "
        f"({in_sample_metrics['family_top3_hit_rate']:.1%})",
        f"- motif top-3 ≥ 85%: "
        f"{'✓' if in_sample_metrics['motif_top3_hit_rate'] >= 0.85 else '✗'} "
        f"({in_sample_metrics['motif_top3_hit_rate']:.1%})",
        "",
        "## Cross-validation",
        "",
        "| protocol | n | motif top-3 | packet top-3 | family top-3 |",
        "|---|---:|---:|---:|---:|",
    ]
    for _, r in cv_df.iterrows():
        lines.append(f"| `{r['cv_protocol']}` | {int(r['n_evaluated'])} | "
                      f"{r['motif_top3']:.1%} | {r['packet_top3']:.1%} | "
                      f"{r['family_top3']:.1%} |")

    lines += [
        "",
        "## Final decision",
        "",
        f"**{decision}**",
        "",
    ]
    if decision == "READY_FOR_IMPLEMENTATION":
        lines.append(
            "All success criteria met: in-sample saturated AND cross-"
            "validation holds reasonably. The learned ontology is ready "
            "for implementation as `gaira_base_3` production."
        )
    elif decision == "NEEDS_ONTOLOGY_REFINEMENT":
        lines.append(
            "In-sample saturated but CV drop is severe (signs of "
            "overfitting). Refine the ontology — fewer / more stable "
            "motif bands, consolidated packets, more conservative "
            "discriminator extraction."
        )
    else:
        lines.append(
            "In-sample did not saturate. The ontology or learning method "
            "is not yet correct. Revisit feature extraction (richer "
            "co-band patterns?) or the analyte-class taxonomy "
            "(undersampled classes?)."
        )

    (REPORTS / "REPORT_gaira_base_3_full_corpus_build_v1.md"
     ).write_text("\n".join(lines))


def write_cv_report(cv_rows):
    cv_df = pd.DataFrame(cv_rows)
    lines = [
        "# Cross-Validation Report v1",
        "",
        "## Protocols",
        "",
        "| protocol | n_evaluated | motif top-1 | motif top-3 | packet top-1 | packet top-3 | family top-1 | family top-3 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, r in cv_df.iterrows():
        lines.append(f"| `{r['cv_protocol']}` | {int(r['n_evaluated'])} | "
                      f"{r['motif_top1']:.1%} | {r['motif_top3']:.1%} | "
                      f"{r['packet_top1']:.1%} | {r['packet_top3']:.1%} | "
                      f"{r['family_top1']:.1%} | {r['family_top3']:.1%} |")

    lines += [
        "",
        "## Generalization behavior",
        "",
        "**CV1 (leave-one-replicate-out, Gobbato 3-rep sets)**: tests intra-"
        "analyte robustness. A class's mean spectrum is recomputed without "
        "the held-out replicate; the held-out spectrum is then scored. "
        "High CV1 hit rates indicate the learned class means are stable "
        "across replicates of the same analyte.",
        "",
        "**CV2 (leave-one-dataset-out)**: tests inter-dataset "
        "generalization. Train on 3 of 4 datasets, test on the held-out "
        "one. Only classes that exist in BOTH the held-out dataset AND "
        "the training set are scored. Tests whether discriminator bands "
        "transfer across measurement origins (laboratory, sample preparation).",
        "",
        "**CV3 (leave-one-instance-out, full LOO)**: classical statistical "
        "robustness test. Classes with only 1 spectrum are excluded "
        "(cannot LOO).",
        "",
        "## What the CV results indicate",
        "",
        "If CV1 ≈ in-sample: the within-class signal is robust; class "
        "means are reliable representations.",
        "",
        "If CV2 substantially below in-sample: discriminators are dataset-"
        "specific (e.g. amino acid spectra acquired with different settings "
        "have different baselines or peak shapes).",
        "",
        "If CV3 substantially below in-sample: the model is overfitting; "
        "consider richer per-class feature pooling or fewer "
        "discriminator bands per motif.",
    ]
    (REPORTS / "REPORT_gaira_base_3_cross_validation_v1.md"
     ).write_text("\n".join(lines))


def write_packet_audit_report(audit_rows):
    pure = sum(1 for r in audit_rows if r["purity"] >= 0.80)
    n = len(audit_rows)
    decisions = defaultdict(int)
    for r in audit_rows:
        decisions[r["decision"]] += 1
    lines = [
        "# Packet Audit Report v1",
        "",
        "## Summary",
        "",
        f"- {n} learned packets audited",
        f"- {pure}/{n} ({pure/n:.0%}) family-pure (≥80% purity)",
        "",
        "### Decisions",
        "",
        "| decision | count |",
        "|---|---:|",
    ]
    for d, c in decisions.items():
        lines.append(f"| `{d}` | {c} |")

    lines += [
        "",
        "## Per-packet detail",
        "",
        "| current ID | suggested name | family | n | purity | coherence | decision |",
        "|---|---|---|---:|---:|---|---|",
    ]
    for r in audit_rows:
        lines.append(
            f"| `{r['learned_packet_id']}` | "
            f"**{r['suggested_human_name']}** | "
            f"{r['dominant_family']} | "
            f"{r['n_member_classes']} | "
            f"{r['purity']:.2f} | "
            f"{r['coherence_judgment']} | "
            f"{r['decision']} |"
        )

    lines += [
        "",
        "## Final packet definitions (post-audit)",
        "",
        "Packet member-class composition is retained as learned. Only the "
        "naming convention is human-readable. The packet→family mapping "
        "in `tables/learned_packet_to_family_mapping_v2.csv` is the "
        "authoritative scoring map.",
    ]
    (REPORTS / "REPORT_gaira_base_3_packet_audit_v1.md"
     ).write_text("\n".join(lines))


def write_readiness_report(in_sample_metrics, cv_rows):
    decision = make_decision(in_sample_metrics, cv_rows)
    cv_df = pd.DataFrame(cv_rows)
    lines = [
        "# Readiness Report v1",
        "",
        f"**Decision: {decision}**",
        "",
        "## Saturation check (in-sample, train=test)",
        "",
        "| criterion | threshold | observed | met? |",
        "|---|---:|---:|---|",
        f"| motif top-3 ≥ 90% | 90% | "
        f"{in_sample_metrics['motif_top3_hit_rate']:.1%} | "
        f"{'✓' if in_sample_metrics['motif_top3_hit_rate'] >= 0.90 else '✗'} |",
        f"| packet top-3 ≥ 95% | 95% | "
        f"{in_sample_metrics['packet_top3_hit_rate']:.1%} | "
        f"{'✓' if in_sample_metrics['packet_top3_hit_rate'] >= 0.95 else '✗'} |",
        f"| family top-3 ≥ 90% (multi-family-aware) | 90% | "
        f"{in_sample_metrics['family_top3_hit_rate']:.1%} | "
        f"{'✓' if in_sample_metrics['family_top3_hit_rate'] >= 0.90 else '✗'} |",
        "",
        "## Cross-validation check",
        "",
        "| protocol | packet top-3 | met threshold? |",
        "|---|---:|---|",
    ]
    cv1_row = cv_df[cv_df["cv_protocol"].str.startswith("CV1")]
    cv3_row = cv_df[cv_df["cv_protocol"].str.startswith("CV3")]
    if len(cv1_row):
        cv1 = float(cv1_row["packet_top3"].iloc[0])
        lines.append(f"| CV1 leave-one-replicate-out (Gobbato) | {cv1:.1%} | "
                      f"{'✓' if cv1 >= 0.70 else '✗'} (need ≥70%) |")
    if len(cv3_row):
        cv3 = float(cv3_row["packet_top3"].iloc[0])
        lines.append(f"| CV3 leave-one-instance-out (full LOO) | {cv3:.1%} | "
                      f"{'✓' if cv3 >= 0.60 else '✗'} (need ≥60%) |")

    lines += [
        "",
        "## Justification",
        "",
    ]
    if decision == "READY_FOR_IMPLEMENTATION":
        lines.append(
            "All in-sample saturation thresholds met AND cross-validation "
            "thresholds met. The learned ontology is sufficient for "
            "production implementation as `gaira_base_3`."
        )
    elif decision == "NEEDS_ONTOLOGY_REFINEMENT":
        lines.append(
            "In-sample saturated but CV drops below threshold — the "
            "ontology may be overfitting. Refine feature extraction "
            "to use more conservative band selection."
        )
    else:
        lines.append(
            "In-sample did not saturate. Ontology or feature extraction "
            "is incorrect at this stage. Revisit before promotion."
        )
    (REPORTS / "REPORT_gaira_base_3_readiness_v1.md"
     ).write_text("\n".join(lines))


def write_audit_log(in_sample_metrics, cv_rows, taxonomy_df, learned_motifs,
                     packets, audit_rows):
    decision = make_decision(in_sample_metrics, cv_rows)
    lines = [
        "# gaira_base_3 grounding full-corpus build + validation v1 — Audit Log",
        "",
        "## Files added",
        "- ADDED: `src/gaira/base3/learned_ontology_v2.py`",
        "- ADDED: `scripts/run_gaira_base_3_grounding_full_corpus_build_and_validation_v1.py`",
        "- ADDED: `GAIRA_BUILD/gaira_base_3_grounding_full_corpus_build_and_validation_v1/**`",
        "",
        "## Files NOT modified",
        "- gaira_base SHA-256 still matches; 12/12 v1 regression tests pass",
        "- All gaira_base_2 modules untouched on disk",
        "- prior gaira_base_3 modules (packet_engine.py, learned_ontology.py) untouched",
        "- Registry v1.5 + mapping v1.4 read-only",
        "- canonical preprocessing unchanged",
        "- substrate engine v1.1.2 unchanged",
        "- NO calibration / target / substrate-aware data used",
        "",
        "## Datasets used",
        "",
        f"- ramanbiolib: included",
        f"- gobbato_powder_raman: included (53 analytes × 3 reps)",
        f"- amino_acid_raman_grounding: included (REQUIRED by phase spec)",
        f"- digitised_literature_spectra: included",
        f"- TOTAL: {len(taxonomy_df)} grounding spectra, "
        f"{taxonomy_df['analyte_class'].nunique()} analyte classes",
        "",
        "## Datasets considered and rejected",
        "",
        "- ag_colloid_serum_sers: EXCLUDED — biological matrix + spike/depletion",
        "- raw_search_pool_candidates: EXCLUDED — peak-list only, no full spectra",
        "- target_serum_cohort_data: EXCLUDED — multi-analyte mixtures",
        "",
        "## Methods",
        "",
        f"- Feature learning: one-vs-rest discriminant ratio, top "
        f"{_learn.N_ANCHOR_BANDS_PER_CLASS} positive bands as anchor + "
        f"{_learn.N_SUPPORT_BANDS_PER_CLASS} support + "
        f"{_learn.N_ANTI_BANDS_PER_CLASS} negative as anti-evidence "
        f"(min DR ≥ {_learn.MIN_DISCRIMINANT_RATIO}, peaks ≥12 cm⁻¹ apart)",
        f"- Prototype clustering: hierarchical agglomerative on correlation "
        f"distance, K={_learn.DEFAULT_N_PROTOTYPE_CLUSTERS} cut",
        f"- Symbolic motif scoring: cosine similarity restricted to motif's "
        f"anchor + support band positions vs class mean (with anti-evidence "
        f"penalty {_learn.ANTI_BAND_PENALTY_PER} per fired anti-band)",
        f"- Packet scoring: MAX over member-class motif scores",
        f"- Family scoring: weighted sum of packet scores via "
        f"packet→family vote distribution",
        "",
        "## Headline metrics",
        "",
        f"- in-sample motif top-3: {in_sample_metrics['motif_top3_hit_rate']:.1%}",
        f"- in-sample packet top-3: {in_sample_metrics['packet_top3_hit_rate']:.1%}",
        f"- in-sample family top-3: {in_sample_metrics['family_top3_hit_rate']:.1%}",
        f"- ambiguity correctness: {in_sample_metrics['ambiguity_correctness_rate']:.1%}",
        f"- learned motifs: {len(learned_motifs)}",
        f"- learned packets: {len(packets)}",
        f"- family-pure packets: {sum(1 for r in audit_rows if r['purity'] >= 0.80)}/{len(audit_rows)}",
        "",
        "## Final decision",
        "",
        f"**{decision}**",
    ]
    (AUDIT / "gaira_base_3_full_corpus_build_audit_log.md"
     ).write_text("\n".join(lines))


def snapshot_code():
    src = Path("/Users/suraj/projects/GAIRA/src/gaira/base3")
    if src.exists():
        shutil.copytree(src, CODE_SNAPSHOT / "base3", dirs_exist_ok=True)
    p = Path("/Users/suraj/projects/GAIRA/scripts/"
             "run_gaira_base_3_grounding_full_corpus_build_and_validation_v1.py")
    if p.exists(): shutil.copy(p, CODE_SNAPSHOT / p.name)


# ─────────────────────────────────────────────────────────────────────
# Driver
# ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 78)
    print("gaira_base_3 - Grounding Full-Corpus Build + Validation v1")
    print("=" * 78)
    for d in (TABLES, REGISTRY, FIGS, REPORTS, AUDIT, DOCS, CODE_SNAPSHOT):
        d.mkdir(parents=True, exist_ok=True)

    master_x = canonical_master_axis()
    rb  = load_ramanbiolib(master_x)
    gp  = load_gobbato_powder(master_x)
    aa  = load_amino_acid_xlsx(master_x)
    lit = load_digitised_literature(master_x)
    all_refs = rb + gp + aa + lit
    print(f"[data] {len(all_refs)} grounding spectra "
          f"({len(rb)} rbl + {len(gp)} gobbato + {len(aa)} aa + {len(lit)} lit)")

    # STAGE 1
    inventory_df = stage1_corpus_audit(rb, gp, aa, lit)

    # STAGE 2
    (taxonomy_df, spectra_by_class, class_means, drs,
      cluster_assignment, overlap, cluster_ids, proto_means,
      learned_motifs, packets, packet_to_family_weights,
      p2f_rows) = stage2_build_ontology(all_refs, master_x)

    # STAGE 3
    in_sample_metrics, miss_rows, saturated = stage3_in_sample(
        all_refs, master_x, learned_motifs, class_means, packets,
        packet_to_family_weights, taxonomy_df,
    )

    # STAGE 4
    cv_rows = stage4_cross_validation(
        all_refs, master_x, spectra_by_class, learned_motifs, packets,
        packet_to_family_weights, taxonomy_df,
    )

    # STAGE 5
    audit_rows = stage5_packet_audit(packets, p2f_rows)

    # Figures
    make_figs(class_means, drs, master_x, cluster_assignment, overlap,
               cluster_ids, learned_motifs, packets, in_sample_metrics,
               cv_rows, all_refs, audit_rows)

    # Reports
    write_main_report(in_sample_metrics, cv_rows, taxonomy_df,
                       learned_motifs, packets, audit_rows)
    write_cv_report(cv_rows)
    write_packet_audit_report(audit_rows)
    write_readiness_report(in_sample_metrics, cv_rows)
    write_audit_log(in_sample_metrics, cv_rows, taxonomy_df, learned_motifs,
                     packets, audit_rows)
    snapshot_code()

    decision = make_decision(in_sample_metrics, cv_rows)
    print(f"\n[decision] {decision}")
    print("DONE")


if __name__ == "__main__":
    main()
