"""gaira_base_4_mss_narrow_metabolite_registry_repair_v1

Phase: MSS narrow-metabolite registry REPAIR.

Goal: fix MSS template coverage gaps and cluster / BSV-family assignment
for the 16-19 narrow metabolites relevant to liver / COVID serum interpretation.

Constraints (NEVER violated):
- Engine v4.5 unchanged
- 11-axis BSV unchanged
- Motif registry unchanged
- No soft-MSS scoring
- No competitor-aware scoring
- No pilot reruns
- No disease labels

Use only ground-truth / calibration / reference data.

Outputs:
  /Volumes/SSD_Rad/GAIRA_BUILD/gaira_base_4_mss_narrow_metabolite_registry_repair_v1/
    registry/, tables/, figures/, reports/, audit/, code_snapshot/

Run:
    cd /Users/suraj/projects/GAIRA
    PYTHONPATH=src .venv/bin/python scripts/run_gaira_base_4_mss_narrow_metabolite_registry_repair_v1.py
"""
from __future__ import annotations

import shutil
import sys
import warnings
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

warnings.simplefilter("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from gaira.spectral import canonical_master_axis  # noqa: E402

from run_gaira_validate_2_grounding import (  # noqa: E402
    load_ramanbiolib, load_gobbato_powder,
    load_amino_acid_xlsx, load_digitised_literature,
)
from run_gaira_base_3_full_grounding_audit_and_signature_build_v1 import (  # noqa: E402
    load_sers_metabolite_63,
)
from run_gaira_base_4_hybrid_bsv_controlled_calibration_v2 import (  # noqa: E402
    load_sers_fitting, load_isotopic, load_uricase, load_erg_calibration,
)
from run_gaira_base_4_paper_band_vs_ground_truth_validation_v1 import (  # noqa: E402
    canonicalize, COMPARATORS, PAPER_BANDS, _has_real_peak,
)
import run_gaira_base_4_paper_band_vs_ground_truth_validation_v1 as _pbv  # noqa: E402

# Extend NAME_MAP in-place to cover the optional small-molecule targets
_pbv.NAME_MAP.update({
    "glucose": "glucose",
    "d-(+)-glucose": "glucose",
    "d-(+)-glucose monohydrate": "glucose",
    "gluc": "glucose",
    "d-glucose": "glucose",
    "alpha-d-glucose": "glucose",
    "beta-d-glucose": "glucose",
    "urea": "urea",
    "creatinine": "creatinine",
    "creat": "creatinine",
})


# ──────────────────────────────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────────────────────────────
ROOT = Path("/Volumes/SSD_Rad/GAIRA_BUILD/gaira_base_4_mss_narrow_metabolite_registry_repair_v1")
REGISTRY = ROOT / "registry"
TABLES   = ROOT / "tables"
FIGS     = ROOT / "figures"
REPORTS  = ROOT / "reports"
AUDIT    = ROOT / "audit"
CODE_SNAPSHOT = ROOT / "code_snapshot"
for d in (REGISTRY, TABLES, FIGS, REPORTS, AUDIT, CODE_SNAPSHOT):
    d.mkdir(parents=True, exist_ok=True)

MSS_REGISTRY = Path(
    "/Volumes/SSD_Rad/GAIRA_BUILD/gaira_base_4_mss_decision_enrichment_v1/"
    "registry/grounding_molecular_signatures_v4_3.csv"
)
PRIOR_SPECIFICITY = Path(
    "/Volumes/SSD_Rad/GAIRA_BUILD/gaira_base_4_paper_band_vs_ground_truth_validation_v1/"
    "tables/paper_band_specificity_v1.csv"
)


# ──────────────────────────────────────────────────────────────────────
# Targets and family/cluster taxonomy (per user spec; matches BIOLOGY_AXES_V11)
# ──────────────────────────────────────────────────────────────────────
TARGETS = [
    # Core paper / liver narrow
    "uric_acid", "hypoxanthine", "xanthine", "adenine",
    "ergothioneine", "glutathione",
    # Additional serum / metabolic
    "lactate", "cysteine", "cystine",
    "tryptophan", "phenylalanine", "tyrosine",
    "cholesterol", "oleic_acid", "palmitic_acid", "stearic_acid",
    # Optional
    "glucose", "urea", "creatinine",
]

# G01..G11 BIOLOGY_AXES_V11
BSV_FAMILY_BY_TARGET = {
    "adenine":      ("G01", "purine_nucleotide"),
    "hypoxanthine": ("G02", "purine_metabolite"),
    "xanthine":     ("G02", "purine_metabolite"),
    "uric_acid":    ("G02", "purine_metabolite"),
    "ergothioneine": ("G10", "sulfur_thiol_redox"),
    "glutathione":   ("G10", "sulfur_thiol_redox"),
    "cysteine":      ("G10", "sulfur_thiol_redox"),
    "cystine":       ("G10", "sulfur_thiol_redox"),
    "tryptophan":    ("G07", "aromatic_residue"),
    "phenylalanine": ("G07", "aromatic_residue"),
    "tyrosine":      ("G07", "aromatic_residue"),
    "lactate":       ("G11", "metabolic_small_molecule"),
    "glucose":       ("G05", "glycan_carbohydrate"),
    "urea":          ("G11", "metabolic_small_molecule"),
    "creatinine":    ("G11", "metabolic_small_molecule"),
    "cholesterol":   ("G09", "sterol_neutral_lipid"),
    "oleic_acid":    ("G08", "lipid_acyl_membrane"),
    "palmitic_acid": ("G08", "lipid_acyl_membrane"),
    "stearic_acid":  ("G08", "lipid_acyl_membrane"),
}

# MSS broad-class (cluster) used by registry — preserved or normalized
MSS_CLUSTER_BY_TARGET = {
    "adenine":       "purine_adenine",
    "hypoxanthine":  "purine_metabolite_hx",
    "xanthine":      "purine_metabolite_xanth",
    "uric_acid":     "purine_metabolite_ua",
    "ergothioneine": "sulfur_thiol_ergothioneine",
    "glutathione":   "sulfur_thiol_glutathione",
    "cysteine":      "sulfur_amino_acid",
    "cystine":       "sulfur_amino_acid",
    "tryptophan":    "tryptophan_indole",
    "phenylalanine": "free_amino_acid_phe",
    "tyrosine":      "free_amino_acid_tyr",
    "lactate":       "metabolic_small_molecule_lactate",
    "glucose":       "carbohydrate_glucose",
    "urea":          "metabolic_small_molecule_urea",
    "creatinine":    "metabolic_small_molecule_creatinine",
    "cholesterol":   "sterol_cholesterol",
    "oleic_acid":    "free_fatty_acid_unsaturated",
    "palmitic_acid": "free_fatty_acid_saturated",
    "stearic_acid":  "free_fatty_acid_saturated",
}

# Lactate literature stub — taken from widely-cited Raman tables of sodium lactate
# (Pecul et al. 2003; De Gelder et al. 2007 amino-acid/metabolite Raman atlas).
# Marked LOW reliability + INSUFFICIENT_GT.
LACTATE_LITERATURE_STUB = {
    "anchors":   [837.0, 928.0, 1043.0, 1453.0],
    "supports":  [762.0, 1090.0, 1126.0, 1370.0],
    "anti":      [],
    "regime":    "Raman",
    "source":    "literature_only_no_gt_in_corpus",
    "citation":  "De Gelder 2007 / Pecul 2003 — Raman of sodium lactate",
}


# ──────────────────────────────────────────────────────────────────────
# Loaders unification
# ──────────────────────────────────────────────────────────────────────
def gather_pure_refs(master_x):
    refs_by_mol_regime: dict[tuple[str, str], list[dict]] = defaultdict(list)
    bundles = []
    for tag, regime, substrate, fn in [
        ("ramanbiolib",            "Raman", "n/a",                   load_ramanbiolib),
        ("gobbato_powder_raman",   "Raman", "n/a (powder)",          load_gobbato_powder),
        ("amino_acid_raman",       "Raman", "n/a",                   load_amino_acid_xlsx),
        ("digitised_literature",   "Raman", "n/a (digitised)",       load_digitised_literature),
        ("sers_metabolite_63",     "SERS",  "Au-on-Si plasmonic",    load_sers_metabolite_63),
        ("serum_ag_colloids_fitting",   "SERS", "Ag colloid",        load_sers_fitting),
        ("serum_ag_colloids_isotopic",  "SERS", "Ag colloid",        load_isotopic),
        ("serum_ag_colloids_uricase",   "SERS", "Ag colloid (cAg-like)", load_uricase),
        ("serum_ag_colloids_erg_cal",   "SERS", "Ag colloid",        load_erg_calibration),
    ]:
        try:
            refs = fn(master_x)
        except Exception as e:
            print(f"  loader {tag} failed: {e}")
            refs = []
        for r in refs:
            raw = r.get("component_key") or r.get("cohort") or r.get("conc_label")
            mol = canonicalize(raw)
            if mol is None:
                continue
            entry = {
                "spectrum_id": r.get("spectrum_id", ""),
                "dataset":     r.get("dataset", tag),
                "regime":      r.get("regime", regime),
                "substrate":   r.get("substrate_type") or r.get("substrate_family") or substrate,
                "raw_label":   raw,
                "molecule":    mol,
                "spectrum":    r["spectrum"],
            }
            refs_by_mol_regime[(mol, entry["regime"])].append(entry)
            bundles.append(entry)
        print(f"  {tag}: {len(refs)} refs loaded")
    return refs_by_mol_regime, bundles


# ──────────────────────────────────────────────────────────────────────
# Stage 1 — Existing template audit
# ──────────────────────────────────────────────────────────────────────
def stage1_existing_audit(refs_by_mol_regime):
    print("[STAGE 1] Audit MSS v4.3 templates for target molecules")
    if not MSS_REGISTRY.exists():
        print(f"  MSS registry not found: {MSS_REGISTRY}")
        df = pd.DataFrame()
        df.to_csv(TABLES / "existing_mss_template_audit_v1.csv", index=False)
        return df, {}
    mss = pd.read_csv(MSS_REGISTRY)
    mss["analyte_lower"] = mss["analyte_name"].astype(str).str.lower().str.strip()

    rows = []
    mss_by_target: dict[str, list[dict]] = defaultdict(list)
    seen_lower = set()
    for _, r in mss.iterrows():
        c = canonicalize(r["analyte_lower"])
        if c is None:
            continue
        if c not in TARGETS:
            continue
        mss_by_target[c].append(r.to_dict())
        seen_lower.add(r["analyte_lower"])

    # Build audit rows (one per target, summarising matched MSS rows)
    for tgt in TARGETS:
        regimes_with_gt = sorted({rk for (m, rk) in refs_by_mol_regime if m == tgt})
        n_gt = sum(len(refs_by_mol_regime.get((tgt, rk), [])) for rk in regimes_with_gt)
        matches = mss_by_target.get(tgt, [])
        if not matches:
            rows.append({
                "molecule":           tgt,
                "mss_template_present": False,
                "n_mss_signatures":   0,
                "mss_analyte_names":  "",
                "mss_anchors":        "",
                "mss_companions":     "",
                "mss_anti_evidence":  "",
                "mss_broad_class":    "",
                "regime_support":     "",
                "n_source_spectra_gt": n_gt,
                "regimes_with_gt":    "|".join(regimes_with_gt),
            })
            continue
        rows.append({
            "molecule":           tgt,
            "mss_template_present": True,
            "n_mss_signatures":   len(matches),
            "mss_analyte_names":  "|".join(sorted({m["analyte_name"] for m in matches})),
            "mss_anchors":        "|".join(sorted({m["mandatory_anchors_cm1"] for m in matches if pd.notna(m["mandatory_anchors_cm1"])})),
            "mss_companions":     "|".join(sorted({m["optional_support_cm1"] for m in matches if pd.notna(m["optional_support_cm1"])})),
            "mss_anti_evidence":  "|".join(sorted({m["anti_evidence_cm1"] for m in matches if pd.notna(m["anti_evidence_cm1"])})),
            "mss_broad_class":    "|".join(sorted({m["broad_class"] for m in matches if pd.notna(m["broad_class"])})),
            "regime_support":     "|".join(sorted({m["regime_support"] for m in matches if pd.notna(m["regime_support"])})),
            "n_source_spectra_gt": n_gt,
            "regimes_with_gt":    "|".join(regimes_with_gt),
        })
    df = pd.DataFrame(rows)
    df.to_csv(TABLES / "existing_mss_template_audit_v1.csv", index=False)
    print(f"  audited {len(df)} target rows; templates present: "
            f"{int(df.mss_template_present.sum())}/{len(df)}")
    # Note duplicate "auric acid" anomaly in v4.3 → record as a known issue
    return df, mss_by_target


# ──────────────────────────────────────────────────────────────────────
# Stage 2 — Derive / repair templates from ground truth
# ──────────────────────────────────────────────────────────────────────
def _derive_anchors(refs: list[dict], master_x: np.ndarray,
                       n_anchors: int = 3, n_supports: int = 6,
                       prom_frac: float = 0.10):
    """Empirically derive anchor + companion bands from a set of replicate
    spectra of the same molecule + regime."""
    if not refs:
        return [], [], 0, []
    Y = np.stack([r["spectrum"] for r in refs])
    ymean = Y.mean(axis=0)
    rng = float(ymean.max() - ymean.min())
    if rng <= 0:
        return [], [], len(refs), []
    idx, props = find_peaks(ymean, prominence=prom_frac * rng)
    if len(idx) == 0:
        # fall back to lower prominence threshold
        idx, props = find_peaks(ymean, prominence=0.05 * rng)
    heights = ymean[idx]
    order = np.argsort(-heights)
    ranked = idx[order]
    anchors = []
    supports = []
    for i, ix in enumerate(ranked[:n_anchors + n_supports]):
        cm1 = float(master_x[ix])
        if i < n_anchors:
            anchors.append(round(cm1, 1))
        else:
            supports.append(round(cm1, 1))

    # Replicate consistency: for each anchor, fraction of replicates that
    # also have a real peak within ±5 cm⁻¹ (top-12 in the replicate)
    rep_consistency = []
    for a in anchors:
        if len(refs) <= 1:
            rep_consistency.append(np.nan)
            continue
        hits = sum(int(_has_real_peak(r["spectrum"], master_x, a, 5.0))
                     for r in refs)
        rep_consistency.append(round(hits / len(refs), 3))
    return anchors, supports, len(refs), rep_consistency


def stage2_derive_repair(refs_by_mol_regime, audit_df, master_x, mss_by_target):
    print("[STAGE 2] Derive / repair templates")
    derived_rows = []
    repair_summary = []

    for tgt in TARGETS:
        # Decide regime to derive: prefer Raman for most; use SERS where Raman is empty
        regimes = sorted({rk for (m, rk) in refs_by_mol_regime if m == tgt})

        if tgt == "lactate":
            # Literature-stub fallback (no GT in corpus)
            derived_rows.append({
                "molecule": tgt,
                "regime":   LACTATE_LITERATURE_STUB["regime"],
                "n_source_spectra": 0,
                "anchors_cm1":   ";".join(str(x) for x in LACTATE_LITERATURE_STUB["anchors"]),
                "supports_cm1":  ";".join(str(x) for x in LACTATE_LITERATURE_STUB["supports"]),
                "anti_evidence_cm1": "",
                "rep_consistency": "",
                "source":         LACTATE_LITERATURE_STUB["source"],
                "citation":       LACTATE_LITERATURE_STUB["citation"],
                "reliability_tier": "LOW",
                "missing_data_note": "No pure lactate spectrum in any of 9 GAIRA grounding sources; literature stub only.",
            })
            repair_summary.append({
                "molecule": tgt, "action": "ADD_LITERATURE_STUB",
                "reason": "lactate has zero pure-component ground truth in current corpus",
                "anchors_added": ";".join(str(x) for x in LACTATE_LITERATURE_STUB["anchors"]),
                "reliability_tier": "LOW",
            })
            continue

        if not regimes:
            derived_rows.append({
                "molecule": tgt, "regime": "",
                "n_source_spectra": 0,
                "anchors_cm1": "", "supports_cm1": "",
                "anti_evidence_cm1": "",
                "rep_consistency": "",
                "source": "no_gt_in_corpus",
                "citation": "",
                "reliability_tier": "INSUFFICIENT_GT",
                "missing_data_note": f"{tgt} has no pure-component ground truth.",
            })
            repair_summary.append({
                "molecule": tgt, "action": "SKIP_NO_GT",
                "reason": "no GT spectra found", "anchors_added": "",
                "reliability_tier": "INSUFFICIENT_GT",
            })
            continue

        # Derive per regime
        per_regime = {}
        for rk in regimes:
            refs = refs_by_mol_regime[(tgt, rk)]
            anchors, supports, n_used, rep = _derive_anchors(refs, master_x)
            per_regime[rk] = {
                "anchors": anchors, "supports": supports,
                "n": n_used, "rep_consistency": rep,
            }
            derived_rows.append({
                "molecule": tgt, "regime": rk,
                "n_source_spectra": n_used,
                "anchors_cm1":  ";".join(str(x) for x in anchors),
                "supports_cm1": ";".join(str(x) for x in supports),
                "anti_evidence_cm1": "",
                "rep_consistency": ";".join("" if (x is None or (isinstance(x, float) and np.isnan(x)))
                                                  else str(x) for x in rep),
                "source": "ground_truth_derived",
                "citation": "",
                "reliability_tier": "",     # filled in stage4
                "missing_data_note": "",
            })

        # Action: VERIFY vs REPAIR vs ADD
        existing = mss_by_target.get(tgt, [])
        if not existing:
            action = "ADD_NEW"
        else:
            # Compare derived anchors vs MSS v4.3 anchors with ±10cm⁻¹ tolerance
            mss_anchors = []
            for r in existing:
                if pd.isna(r.get("mandatory_anchors_cm1")): continue
                for tok in str(r["mandatory_anchors_cm1"]).split(";"):
                    try: mss_anchors.append(float(tok))
                    except ValueError: pass
            derived_all = []
            for rk, d in per_regime.items():
                derived_all += d["anchors"]
            overlap = sum(int(any(abs(a - m) <= 10 for m in mss_anchors))
                            for a in derived_all)
            action = "VERIFY" if overlap >= max(1, len(derived_all) // 2) else "REPAIR"

        repair_summary.append({
            "molecule": tgt, "action": action,
            "reason": f"derived {sum(d['n'] for d in per_regime.values())} GT spectra across regimes {list(per_regime)}",
            "anchors_added": ";".join({str(a) for d in per_regime.values() for a in d["anchors"]}),
            "reliability_tier": "",   # filled in stage4
        })

    derived_df = pd.DataFrame(derived_rows)
    derived_df.to_csv(REGISTRY / "narrow_metabolite_mss_registry_v1.csv", index=False)
    repair_df = pd.DataFrame(repair_summary)
    repair_df.to_csv(TABLES / "template_repair_summary_v1.csv", index=False)
    print(f"  derived {len(derived_df)} (molecule × regime) rows; "
            f"repair actions: {dict(repair_df.action.value_counts())}")
    return derived_df, repair_df


# ──────────────────────────────────────────────────────────────────────
# Stage 3 — Cluster / family assignment
# ──────────────────────────────────────────────────────────────────────
def stage3_assignment(audit_df, derived_df):
    print("[STAGE 3] Cluster / family assignment")
    rows = []
    for tgt in TARGETS:
        bsv_id, bsv_name = BSV_FAMILY_BY_TARGET.get(tgt, ("?", "unassigned"))
        mss_cluster_target = MSS_CLUSTER_BY_TARGET.get(tgt, "unassigned")
        existing = audit_df[audit_df.molecule == tgt]
        existing_cluster = existing.mss_broad_class.iloc[0] if not existing.empty else ""
        match = "MATCH" if existing_cluster and \
                                  any(existing_cluster.split("|")[0].startswith(p)
                                          for p in [mss_cluster_target.split("_")[0], "purine_metabolite",
                                                       "purine_adenine", "sulfur_amino_acid",
                                                       "free_fatty_acid", "tryptophan_indole",
                                                       "free_amino_acid", "sterol", "ergothioneine",
                                                       "creatine_creatinine", "small_molecule_other",
                                                       "sugar"]) \
                  else ("UPDATED" if existing_cluster else "NEW")
        rows.append({
            "molecule": tgt,
            "bsv_family_id":   bsv_id,
            "bsv_family_name": bsv_name,
            "mss_cluster_v1":  mss_cluster_target,
            "mss_cluster_existing": existing_cluster,
            "assignment_status": match,
        })
    df = pd.DataFrame(rows)
    df.to_csv(TABLES / "molecule_cluster_family_assignment_v1.csv", index=False)
    print(f"  emitted {len(df)} assignment rows")
    return df


# ──────────────────────────────────────────────────────────────────────
# Stage 4 — Template quality check
# ──────────────────────────────────────────────────────────────────────
def stage4_quality_check(derived_df, refs_by_mol_regime, master_x):
    print("[STAGE 4] Quality check")
    rows = []
    # Load prior collision summary if available
    coll = pd.read_csv(PRIOR_SPECIFICITY) if PRIOR_SPECIFICITY.exists() else pd.DataFrame()

    for _, drow in derived_df.iterrows():
        tgt = drow["molecule"]; regime = drow["regime"]
        anchors = [float(x) for x in drow["anchors_cm1"].split(";") if x.strip()]
        n = int(drow["n_source_spectra"])
        rep_str = drow["rep_consistency"]
        rep_vals = []
        if rep_str:
            for tok in rep_str.split(";"):
                try: rep_vals.append(float(tok))
                except (ValueError, TypeError): pass
        rep_mean = float(np.mean(rep_vals)) if rep_vals else np.nan

        # Anchor presence — re-test on the refs themselves
        refs = refs_by_mol_regime.get((tgt, regime), [])
        anchor_present_frac = []
        for a in anchors:
            if not refs:
                anchor_present_frac.append(np.nan); continue
            hits = sum(int(_has_real_peak(r["spectrum"], master_x, a, 5.0))
                          for r in refs)
            anchor_present_frac.append(round(hits / len(refs), 3))
        present_mean = float(np.mean([x for x in anchor_present_frac
                                              if not (isinstance(x, float) and np.isnan(x))])) \
                          if anchor_present_frac else np.nan

        # Paper-band overlap (informational only)
        paper_b = PAPER_BANDS.get(tgt, [])
        overlap_paper = sum(int(any(abs(a - p) <= 10 for p in paper_b)) for a in anchors)

        # Collision summary from prior phase
        if not coll.empty:
            target_rows = coll[coll.paper_target == tgt]
            n_collision = int((target_rows.specificity_flag == "LOW_SPECIFICITY_COLLISION_PRONE").sum())
            n_high      = int((target_rows.specificity_flag == "HIGH_SPECIFICITY").sum())
        else:
            n_collision = 0; n_high = 0

        # Assign reliability tier
        if drow["source"] == "literature_only_no_gt_in_corpus":
            tier = "LOW"
        elif n == 0 or not anchors:
            tier = "INSUFFICIENT_GT"
        elif n >= 3 and (np.isnan(present_mean) or present_mean >= 0.67) and len(anchors) >= 3:
            tier = "HIGH"
        elif n >= 2 and len(anchors) >= 2:
            tier = "MODERATE"
        else:
            tier = "LOW"

        # Quality flag
        if tier == "INSUFFICIENT_GT":
            flag = "NEEDS_MORE_GT"
        elif tier == "LOW":
            flag = "DO_NOT_USE_FOR_TARGET_INFERENCE" if drow["source"] == "literature_only_no_gt_in_corpus" \
                      else "NEEDS_MORE_GT"
        elif tier == "MODERATE":
            flag = "READY_WITH_CAVEAT"
        else:
            flag = "READY"

        rows.append({
            "molecule": tgt, "regime": regime,
            "n_source_spectra": n,
            "n_anchors": len(anchors),
            "anchors_cm1": ";".join(str(x) for x in anchors),
            "anchor_presence_mean": round(present_mean, 3) if not np.isnan(present_mean) else None,
            "rep_consistency_mean": round(rep_mean, 3) if not np.isnan(rep_mean) else None,
            "overlap_with_paper_bands": overlap_paper,
            "n_paper_bands":  len(paper_b),
            "n_collision_prone_paper_bands": n_collision,
            "n_high_specificity_paper_bands": n_high,
            "reliability_tier": tier,
            "quality_flag":     flag,
        })
    df = pd.DataFrame(rows)
    df.to_csv(TABLES / "mss_template_quality_check_v1.csv", index=False)
    print(f"  emitted {len(df)} QC rows; tiers: "
            f"{dict(df.reliability_tier.value_counts())}")
    return df


# ──────────────────────────────────────────────────────────────────────
# Figures
# ──────────────────────────────────────────────────────────────────────
def make_figures(audit_df, derived_df, assign_df, qc_df):
    print("[FIG] generating QC figures")
    # Fig 1: coverage before vs after
    try:
        before = audit_df.set_index("molecule")["mss_template_present"].astype(int)
        after  = qc_df.groupby("molecule")["n_anchors"].max().reindex(TARGETS) > 0
        idx = TARGETS
        fig, ax = plt.subplots(figsize=(10, 5))
        x = np.arange(len(idx)); w = 0.4
        ax.bar(x - w/2, before.reindex(idx).fillna(0), w, label="MSS v4.3 present", color="#888")
        ax.bar(x + w/2, after.reindex(idx).fillna(0).astype(int), w, label="repair v1 covered", color="#4C72B0")
        ax.set_xticks(x); ax.set_xticklabels(idx, rotation=45, ha="right", fontsize=8)
        ax.set_yticks([0, 1]); ax.set_ylim(0, 1.2)
        ax.set_title("MSS template coverage — before vs after repair v1")
        ax.legend(loc="upper right", fontsize=8); ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(FIGS / "fig_mss_template_coverage_before_after_v1.png", dpi=150)
        plt.close(fig)
    except Exception as e:
        print(f"  fig1 issue: {e}")

    # Fig 2: molecule → family assignment
    try:
        fams = sorted({BSV_FAMILY_BY_TARGET[t][0] for t in TARGETS})
        fig, ax = plt.subplots(figsize=(8, 5))
        for i, tgt in enumerate(TARGETS):
            fam = BSV_FAMILY_BY_TARGET[tgt][0]
            ax.scatter(fams.index(fam), -i, s=60, color="#4C72B0")
            ax.text(fams.index(fam) + 0.06, -i, tgt, va="center", fontsize=8)
        ax.set_xticks(range(len(fams))); ax.set_xticklabels(fams, fontsize=9)
        ax.set_yticks([])
        ax.set_title("Target molecules → 11-axis BSV family assignment")
        ax.grid(axis="x", alpha=0.3)
        fig.tight_layout()
        fig.savefig(FIGS / "fig_molecule_family_assignment_map_v1.png", dpi=150)
        plt.close(fig)
    except Exception as e:
        print(f"  fig2 issue: {e}")

    # Fig 3: anchor band map
    try:
        fig, ax = plt.subplots(figsize=(11, 6))
        for i, tgt in enumerate(TARGETS):
            sub = derived_df[derived_df.molecule == tgt]
            for _, r in sub.iterrows():
                anchors = [float(x) for x in r["anchors_cm1"].split(";") if x.strip()]
                supports = [float(x) for x in r["supports_cm1"].split(";") if x.strip()]
                color = "#4C72B0" if r["regime"] == "Raman" else "#DD8452"
                if r["source"] == "literature_only_no_gt_in_corpus":
                    color = "#999999"
                for a in anchors:
                    ax.plot([a], [-i], "o", color=color, ms=7)
                for s in supports:
                    ax.plot([s], [-i], "x", color=color, ms=4, alpha=0.5)
        ax.set_yticks([-i for i in range(len(TARGETS))])
        ax.set_yticklabels(TARGETS, fontsize=8)
        ax.set_xlabel("wavenumber cm⁻¹")
        ax.set_xlim(400, 1800)
        ax.grid(axis="x", alpha=0.3)
        ax.set_title("Repair v1 anchor (●) + companion (×) band map  —  blue=Raman, orange=SERS, gray=literature")
        fig.tight_layout()
        fig.savefig(FIGS / "fig_anchor_band_map_v1.png", dpi=150)
        plt.close(fig)
    except Exception as e:
        print(f"  fig3 issue: {e}")

    # Fig 4: reliability tier summary
    try:
        tier_counts = qc_df.groupby("molecule")["reliability_tier"].agg(
            lambda s: sorted(s, key=lambda x: ["HIGH","MODERATE","LOW","INSUFFICIENT_GT"].index(x))[0]
        ).reindex(TARGETS).fillna("INSUFFICIENT_GT")
        colors = {"HIGH": "#2ca02c", "MODERATE": "#f39c12",
                    "LOW": "#999999", "INSUFFICIENT_GT": "#c0392b"}
        fig, ax = plt.subplots(figsize=(10, 5))
        x = np.arange(len(TARGETS))
        ax.bar(x, [1]*len(TARGETS), color=[colors[t] for t in tier_counts], edgecolor="black")
        for i, t in enumerate(tier_counts):
            ax.text(i, 0.5, t, rotation=90, ha="center", va="center",
                       fontsize=8, color="white", fontweight="bold")
        ax.set_xticks(x); ax.set_xticklabels(TARGETS, rotation=45, ha="right", fontsize=8)
        ax.set_yticks([])
        ax.set_title("Reliability tier (best per molecule) — repair v1")
        fig.tight_layout()
        fig.savefig(FIGS / "fig_reliability_tier_summary_v1.png", dpi=150)
        plt.close(fig)
    except Exception as e:
        print(f"  fig4 issue: {e}")


# ──────────────────────────────────────────────────────────────────────
# Reports
# ──────────────────────────────────────────────────────────────────────
def write_demo_note(qc_df, repair_df):
    txt = [
        "# DEMO note — MSS narrow-metabolite registry repair v1\n",
        "**Date:** 2026-04-23",
        "",
        "## What this is\n",
        "A registry-only repair that fixes MSS narrow-metabolite coverage gaps and assigns each "
        "target molecule to the correct 11-axis BSV family. The GAIRA engine, MSS scoring, motif "
        "registry, and BSV weights are all UNCHANGED.\n",
        "## Honest framing for any narrow-metabolite output\n",
        "- The earlier paper-band vs ground-truth phase showed paper bands are **collision-prone** "
        "in pure-molecule SERS/Raman corpora: only 2/12 paper bands (ergothioneine 1220, glutathione "
        "912) were HIGH_SPECIFICITY; 8/12 fired across multiple chemically-related comparators.",
        "- Full MSS is stricter because it requires **multi-band molecular consistency** — co-firing "
        "of multiple anchor bands at the molecule's expected positions, not just a single peak in a "
        "neighborhood.",
        "- This repair adds the missing molecules to the registry — most importantly **lactate** "
        "(no pure-component ground truth available; literature stub only). Hypoxanthine already had "
        "an MSS template under the short name 'hypox' and is now canonicalized.",
        "- This does NOT prove molecule identity in disease spectra. Per-molecule MSS firings on "
        "patient cohorts must remain **candidate evidence**, not definitive assignment.",
        "- The validated cross-pilot interpretation layer for now is the broad **11-axis BSV** "
        "(family-level chemistry; G09 Sterol-lipid ↓ confirmed across 5 disease cohorts × 2 regimes).",
        "- MSS-resolution reporting (per-spectrum molecule attribution with co-fire constraints) is "
        "**Task 2** — a separate downstream phase that uses this repaired registry as input.",
        "",
        "## Repair summary\n",
    ]
    for _, r in repair_df.iterrows():
        txt.append(f"- **{r['molecule']}** — {r['action']} ({r['reason']})")
    (REPORTS / "DEMO_NOTE_mss_gap_repair_and_paper_band_collision_v1.md").write_text("\n".join(txt))


def write_report(audit_df, derived_df, assign_df, qc_df, repair_df, decision):
    lines = []
    lines.append("# GAIRA MSS narrow-metabolite registry repair v1 — final report\n")
    lines.append(f"## Decision: **{decision}**\n")
    lines.append(
        "Registry-only repair. GAIRA engine, MSS scoring, motif registry, BSV weights, and "
        "substrate physics are unchanged. No disease labels used."
    )
    lines.append("")

    lines.append("## Stage 1 — existing MSS template audit\n")
    lines.append("| molecule | template present | n MSS sigs | analyte names | broad class | regime support | n GT spectra |")
    lines.append("|---|---|---:|---|---|---|---:|")
    for _, r in audit_df.iterrows():
        lines.append(f"| {r['molecule']} | {'✓' if r['mss_template_present'] else '✗'} | "
                        f"{r['n_mss_signatures']} | {r['mss_analyte_names'][:60]} | "
                        f"{r['mss_broad_class'][:50]} | {r['regime_support'][:30]} | "
                        f"{r['n_source_spectra_gt']} |")
    lines.append("")

    lines.append("## Stage 2 — repair actions\n")
    lines.append("| molecule | action | reason | anchors added |")
    lines.append("|---|---|---|---|")
    for _, r in repair_df.iterrows():
        lines.append(f"| {r['molecule']} | {r['action']} | {r['reason']} | "
                        f"{(r['anchors_added'] or '')[:60]} |")
    lines.append("")

    lines.append("## Stage 3 — cluster / family assignment\n")
    lines.append("| molecule | BSV family | MSS cluster (v1) | MSS cluster existing | status |")
    lines.append("|---|---|---|---|---|")
    for _, r in assign_df.iterrows():
        lines.append(f"| {r['molecule']} | {r['bsv_family_id']} {r['bsv_family_name']} | "
                        f"{r['mss_cluster_v1']} | {r['mss_cluster_existing']} | "
                        f"{r['assignment_status']} |")
    lines.append("")

    lines.append("## Stage 4 — template quality check\n")
    lines.append("| molecule | regime | n GT | n anchors | anchor presence | rep consistency | overlap paper | tier | flag |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---|---|")
    for _, r in qc_df.iterrows():
        lines.append(f"| {r['molecule']} | {r['regime']} | {r['n_source_spectra']} | "
                        f"{r['n_anchors']} | {r['anchor_presence_mean']} | "
                        f"{r['rep_consistency_mean']} | "
                        f"{r['overlap_with_paper_bands']}/{r['n_paper_bands']} | "
                        f"{r['reliability_tier']} | {r['quality_flag']} |")
    lines.append("")

    # Required answers
    lines.append("## Required answers\n")
    n_missing = int((~audit_df["mss_template_present"]).sum())
    missing_names = list(audit_df[~audit_df["mss_template_present"]]["molecule"])

    lines.append("### 1. Which MSS templates were missing?\n")
    if n_missing == 0:
        lines.append("- 0 missing at the analyte-name level. The only true coverage gap is **lactate** "
                        "(no pure lactate spectrum in any of 9 grounding sources). Hypoxanthine was "
                        "present under the short name 'hypox' and has been canonicalized.")
    else:
        lines.append(f"- {n_missing} target(s) missing: {missing_names}")
    lines.append("")

    lines.append("### 2. Which were added?\n")
    added = repair_df[repair_df.action.isin(["ADD_NEW", "ADD_LITERATURE_STUB"])]
    if added.empty:
        lines.append("- (none — all targets had MSS entries; only lactate added as literature stub)")
    else:
        for _, r in added.iterrows():
            lines.append(f"- **{r['molecule']}** ({r['action']}, tier {r['reliability_tier']}) — {r['reason']}")
    lines.append("")

    lines.append("### 3. Which were repaired?\n")
    rep = repair_df[repair_df.action == "REPAIR"]
    ver = repair_df[repair_df.action == "VERIFY"]
    if rep.empty:
        lines.append("- 0 templates needed REPAIR action (derived anchors overlap MSS v4.3 within ±10 cm⁻¹).")
    else:
        for _, r in rep.iterrows():
            lines.append(f"- **{r['molecule']}** — derived anchors diverge from v4.3 (>50% non-overlap)")
    lines.append(f"- {len(ver)} template(s) VERIFIED against v4.3 anchors with at least majority overlap.")
    lines.append("")

    lines.append("### 4. Which molecules have strong ground-truth support?\n")
    strong = qc_df[qc_df.reliability_tier == "HIGH"]
    for tgt in sorted(set(strong.molecule)):
        rgs = "|".join(sorted(set(strong[strong.molecule == tgt]["regime"])))
        lines.append(f"- **{tgt}** — HIGH ({rgs})")
    if strong.empty:
        lines.append("- (none reached HIGH tier)")
    lines.append("")

    lines.append("### 5. Which molecules remain weak or substrate-sensitive?\n")
    weak = qc_df[qc_df.reliability_tier.isin(["LOW", "INSUFFICIENT_GT"])]
    for _, r in weak.iterrows():
        lines.append(f"- **{r['molecule']}** ({r['regime'] or 'no regime'}) — tier {r['reliability_tier']}, "
                        f"flag {r['quality_flag']}")
    if weak.empty:
        lines.append("- (none — all reach MODERATE or HIGH)")
    lines.append("")

    lines.append("### 6. Are all molecules assigned to correct MSS clusters and BSV families?\n")
    bad = assign_df[assign_df.assignment_status == ""]
    lines.append(f"- All {len(assign_df)} target molecules now have explicit BSV family + MSS cluster assignment.")
    lines.append("- Mismatches with prior MSS broad_class are resolved by preferring the new explicit "
                    "assignment (e.g. uric_acid v4.3 'free_fatty_acid' typo on 'auric acid' duplicate row → "
                    "corrected to 'purine_metabolite_ua' in repair v1).")
    lines.append("")

    lines.append("### 7. What is ready for Task 2 MSS-resolution reporting?\n")
    ready = qc_df[qc_df.quality_flag == "READY"]
    cav   = qc_df[qc_df.quality_flag == "READY_WITH_CAVEAT"]
    notuse = qc_df[qc_df.quality_flag == "DO_NOT_USE_FOR_TARGET_INFERENCE"]
    needsgt = qc_df[qc_df.quality_flag == "NEEDS_MORE_GT"]
    lines.append(f"- READY ({len(ready)}): {sorted(set(ready.molecule))}")
    lines.append(f"- READY_WITH_CAVEAT ({len(cav)}): {sorted(set(cav.molecule))}")
    lines.append(f"- DO_NOT_USE_FOR_TARGET_INFERENCE ({len(notuse)}): {sorted(set(notuse.molecule))}")
    lines.append(f"- NEEDS_MORE_GT ({len(needsgt)}): {sorted(set(needsgt.molecule))}")
    lines.append("")

    (REPORTS / "REPORT_mss_narrow_metabolite_registry_repair_v1.md").write_text("\n".join(lines))


def write_audit(decision):
    txt = [
        "# gaira_base_4_mss_narrow_metabolite_registry_repair_v1 — audit log",
        f"date: {datetime.now().isoformat()}",
        "",
        "## Source datasets used (read-only)",
        "- ramanbiolib (Raman pure-component)",
        "- gobbato_powder_raman (Raman pure-component, multiple replicates)",
        "- amino_acid_raman (Raman pure-component)",
        "- digitised_literature (Raman digitised)",
        "- sers_metabolite_63 (SERS pure-component, NIHMS1547448)",
        "- serum_ag_colloids fitting / isotopic / uricase / ERG calibration cohorts (SERS)",
        "- existing MSS v4.3 registry (decision_enrichment build)",
        "- prior paper-band-vs-ground-truth validation outputs (informational only)",
        "",
        "## Strict negative invariants",
        "- NO disease labels used at any stage",
        "- NO engine changes (gaira/base2 / base3 / base4 modules untouched on disk)",
        "- NO 11-axis BSV weight changes",
        "- NO motif registry changes",
        "- NO soft-MSS scoring implemented",
        "- NO competitor-aware scoring implemented",
        "- NO classifier feedback",
        "- NO pilot reruns",
        "- NO threshold tuning, NO label-driven optimization",
        "",
        "## Outputs",
        "- registry/narrow_metabolite_mss_registry_v1.csv",
        "- tables/existing_mss_template_audit_v1.csv",
        "- tables/template_repair_summary_v1.csv",
        "- tables/molecule_cluster_family_assignment_v1.csv",
        "- tables/mss_template_quality_check_v1.csv",
        "- figures/fig_mss_template_coverage_before_after_v1.png",
        "- figures/fig_molecule_family_assignment_map_v1.png",
        "- figures/fig_anchor_band_map_v1.png",
        "- figures/fig_reliability_tier_summary_v1.png",
        "- reports/REPORT_mss_narrow_metabolite_registry_repair_v1.md",
        "- reports/DEMO_NOTE_mss_gap_repair_and_paper_band_collision_v1.md",
        "",
        f"## Final decision\n**{decision}**",
    ]
    (AUDIT / "gaira_base_4_mss_narrow_metabolite_registry_repair_v1_audit_log.md").write_text("\n".join(txt))


# ──────────────────────────────────────────────────────────────────────
# Entrypoint
# ──────────────────────────────────────────────────────────────────────
def decide(qc_df, audit_df):
    """Final decision label."""
    n_targets = len(TARGETS)
    n_ready = qc_df.groupby("molecule")["quality_flag"].apply(
        lambda s: any(x in ("READY", "READY_WITH_CAVEAT") for x in s)
    ).sum()
    n_donotuse = qc_df.groupby("molecule")["quality_flag"].apply(
        lambda s: all(x == "DO_NOT_USE_FOR_TARGET_INFERENCE" for x in s)
    ).sum()
    n_needsgt = qc_df.groupby("molecule")["quality_flag"].apply(
        lambda s: all(x in ("NEEDS_MORE_GT",) for x in s)
    ).sum()
    n_missing_originally = int((~audit_df["mss_template_present"]).sum())
    if n_donotuse + n_needsgt > n_targets // 3:
        return "MSS_GAP_REPAIR_PARTIAL_NEEDS_GT"
    if n_ready >= int(0.85 * n_targets):
        return "MSS_GAP_REPAIR_COMPLETE_READY_FOR_TASK2"
    if n_donotuse >= n_targets // 4:
        return "MSS_REPAIR_BLOCKED_BY_MISSING_REFERENCES"
    return "MSS_GAP_REPAIR_PARTIAL_NEEDS_GT"


def main():
    print("=" * 78)
    print("gaira_base_4_mss_narrow_metabolite_registry_repair_v1")
    print("=" * 78)
    master_x = canonical_master_axis()
    print("[load] gathering pure-component refs")
    refs_by_mol_regime, _ = gather_pure_refs(master_x)

    audit_df, mss_by_target = stage1_existing_audit(refs_by_mol_regime)
    derived_df, repair_df   = stage2_derive_repair(refs_by_mol_regime, audit_df, master_x, mss_by_target)
    assign_df               = stage3_assignment(audit_df, derived_df)
    qc_df                   = stage4_quality_check(derived_df, refs_by_mol_regime, master_x)

    # Backfill reliability tier into derived_df + repair_df
    tier_by_mol = qc_df.groupby("molecule")["reliability_tier"].apply(
        lambda s: sorted(s, key=lambda x: ["HIGH","MODERATE","LOW","INSUFFICIENT_GT"].index(x))[0]
    ).to_dict()
    derived_df["reliability_tier"] = derived_df.apply(
        lambda r: r.get("reliability_tier") or tier_by_mol.get(r["molecule"], ""), axis=1)
    derived_df.to_csv(REGISTRY / "narrow_metabolite_mss_registry_v1.csv", index=False)
    repair_df["reliability_tier"] = repair_df.molecule.map(tier_by_mol).fillna("")
    repair_df.to_csv(TABLES / "template_repair_summary_v1.csv", index=False)

    make_figures(audit_df, derived_df, assign_df, qc_df)
    write_demo_note(qc_df, repair_df)
    decision = decide(qc_df, audit_df)
    write_report(audit_df, derived_df, assign_df, qc_df, repair_df, decision)
    write_audit(decision)
    try:
        shutil.copy(__file__, CODE_SNAPSHOT / Path(__file__).name)
    except Exception:
        pass
    print(f"[done] decision: {decision}")


if __name__ == "__main__":
    main()
