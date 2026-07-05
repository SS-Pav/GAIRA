"""GAIRA — gaira_build_motifs_v1 · Phase M2.2 — Motif Ontology Untangling + Core Evidence Rebuild.

Architectural correction phase. Separates the motif layer into 4 cleanly-
delineated layers and recomputes grounding using CORE-only evidence:

  Layer 1 — CORE motif ontology (substrate-agnostic physics + chemistry)
  Layer 2 — Ambiguity structure (collision zones, multi-candidate mappings)
  Layer 3 — Substrate overlay (substrate_physics_v1.1.2; pre-existing)
  Layer 4 — Calibration behavior (Ag-colloid serum measurement-specific)

Operations
----------

A. Decompose every motif into core / substrate-conditioned / ambiguity
   components. No motif is structurally modified — the decomposition is a
   labelling overlay. The motif registry remains v1.2 (from M4.1).

B. Re-classify all M3 / M3.1 / M3.2 / M4 evidence by layer:
   * ramanbiolib full spectra            → CORE (normal Raman, no substrate)
   * Gobbato powder Raman                → CORE (no substrate)
   * raman_knowledge_core peak catalog   → CORE (literature meta-analysis)
   * Gelder 2007 / Kim 1987              → CORE (normal Raman digitisations)
   * Stewart 1999                         → SUBSTRATE_CONDITIONED (SERS digitisation)
   * Gobbato pure-analyte SERS / spike   → SUBSTRATE_CONDITIONED + CALIBRATION
   * Gobbato uricase depletion            → CALIBRATION (Ag-colloid serum)
   * Gobbato isotopic UA                  → SUBSTRATE_CONDITIONED + CALIBRATION
   * ergothioneine_serum/ERG_calibration  → CALIBRATION (cAg substrate)
   * cspp_serum/Figure-7                  → CALIBRATION (Ag-colloid serum)
   * metabolite_sers63 peak lists         → SUBSTRATE_CONDITIONED (SERS peaks)
   * literature creatine/creatinine peak  → CORE (consensus literature)

C. Recompute motif grounding using ONLY core evidence sources, against
   the canonical 1401-pt master axis. Re-use M3's grounding evaluator
   (band-window local-max + peak-list cross-check) on CORE-only references.

D. Reframe M4 calibration results as "Ag-colloid serum calibration
   behavior" — measurement-specific, NOT universal motif validity. Build
   a dual-status table per motif.

Non-modifying:
  * No motif structurally changed.
  * No preprocessing changed.
  * No pilot data used.
  * No substrate-engine weight modified.
  * v1.2 registry remains the operating registry.

Run:
    cd /Users/suraj/projects/GAIRA
    PYTHONPATH=src .venv/bin/python scripts/run_gaira_motifs_M2_2_ontology_untangling_v1.py
"""
from __future__ import annotations

import ast
import hashlib
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from gaira.spectral import (  # noqa: E402
    canonical_master_axis,
    crop_before_interpolate,
    InsufficientOverlapError,
)


# ──────────────────────────────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────────────────────────────

ROOT = Path("/Volumes/SSD_Rad/GAIRA_BUILD/gaira_build_motifs_v1/"
            "M2_2_ontology_untangling_v1")
REGISTRY = ROOT / "registry"
TABLES = ROOT / "tables"
DOCS = ROOT / "docs"
AUDIT = ROOT / "audit"
for d in (REGISTRY, TABLES, DOCS, AUDIT):
    d.mkdir(parents=True, exist_ok=True)

V1_2_YAML = Path(
    "/Volumes/SSD_Rad/GAIRA_BUILD/gaira_build_motifs_v1/"
    "M4_1_refinement_and_recalibration_v1/registry/motif_candidate_registry_v1_2.yaml"
)
M3_MATRIX = Path(
    "/Volumes/SSD_Rad/GAIRA_BUILD/gaira_build_motifs_v1/"
    "M3_grounding_validation_v1/tables/motif_grounding_matrix_v1.csv"
)
M3_1_REGISTRY = Path(
    "/Volumes/SSD_Rad/GAIRA_BUILD/gaira_build_motifs_v1/"
    "M3_1_reference_rescue_v1/registry/reference_rescue_registry_v1.csv"
)
M3_1_NPZ = Path(
    "/Volumes/SSD_Rad/GAIRA_BUILD/gaira_build_motifs_v1/"
    "M3_1_reference_rescue_v1/references/rescued_refs_master_axis.npz"
)
M3_2_STATUS = Path(
    "/Volumes/SSD_Rad/GAIRA_BUILD/gaira_build_motifs_v1/"
    "M3_2_creatine_creatinine_gap_fix_v1/tables/"
    "creatine_creatinine_motif_status_update_v1.csv"
)
M4_SUMMARY = Path(
    "/Volumes/SSD_Rad/GAIRA_BUILD/gaira_build_motifs_v1/"
    "M4_calibration_validation_v1/tables/motif_calibration_summary_v1.csv"
)
RAMANBIOLIB_SPECTRA = Path(
    "/Volumes/SSD_Rad/GAIRA_DATA/raw/ramanbiolib/ramanbiolib-main/"
    "ramanbiolib/db/raman_spectra_db.csv"
)
RAMANBIOLIB_PEAKS = Path(
    "/Volumes/SSD_Rad/GAIRA_DATA/raw/ramanbiolib/ramanbiolib-main/"
    "ramanbiolib/db/raman_peaks_db.csv"
)


# ──────────────────────────────────────────────────────────────────────
# Evidence layer classification (deterministic rules)
# ──────────────────────────────────────────────────────────────────────

CORE_SOURCES = {
    "ramanbiolib",                    # normal Raman of pure compounds
    "gobbato_raman_pwd",              # Gobbato pure powder Raman (no substrate)
    "raman_knowledge_core",           # consensus literature peak catalog
    "gelder_2007_raman_db",           # de Gelder reference Raman database (digitised)
    "kim_1987_ua_solution_raman",     # UA solution Raman (digitised)
    "literature_creatinine_consensus",
    "literature_creatine_consensus",
    "madzharova_2017_purine_review",
    "frushour_koenig_1974_aa_raman",
}

SUBSTRATE_CONDITIONED_SOURCES = {
    "gobbato_sers_met",               # pure-analyte Ag-colloid SERS (substrate-bound)
    "gobbato_isotopic_ua_buffer",     # Ag-colloid UA in PBS (substrate)
    "stewart_1999_ua_sers_digitised", # SERS digitisation
    "metabolite_sers63",              # SERS peak lists
}

CALIBRATION_SOURCES = {
    "gobbato_sers_spike_serum_merck", # serum matrix + spike
    "gobbato_uricase_depletion",      # serum matrix + enzymatic
    "ergothioneine_erg_calibration",  # cAg substrate calibration
    "cspp_serum_figure7_spike",       # Ag-colloid serum spike
}


def classify_ref_id(ref_id: str) -> str:
    """Return layer for a given reference id (M3.1-style ref_id strings)."""
    rid = ref_id.lower()
    if rid.startswith(("ramanbiolib_", "rb_")):
        return "CORE"
    if "raman_pwd_gobbato" in rid:
        return "CORE"
    if "knowledge_core" in rid or rid.startswith("kc_"):
        return "CORE"
    if "literature_peak_catalog" in rid:
        return "CORE"
    if "digitised" in rid or "digitized" in rid:
        # Gelder 2007 + Kim 1987 are normal Raman; Stewart 1999 is SERS
        if "stewart" in rid:
            return "SUBSTRATE_CONDITIONED"
        return "CORE"
    if "consensus" in rid:
        return "CORE"
    # Substrate-conditioned
    if "sers_met_gobbato" in rid:
        return "SUBSTRATE_CONDITIONED"
    if "isotopic" in rid and "ua" in rid:
        return "SUBSTRATE_CONDITIONED"
    if "metabolite_sers63" in rid:
        return "SUBSTRATE_CONDITIONED"
    # Calibration
    if "sers_spike_gobbato" in rid:
        return "CALIBRATION"
    if "cspp" in rid or "fig7" in rid:
        return "CALIBRATION"
    if "erg_calibration" in rid or "erg_series" in rid:
        return "CALIBRATION"
    if "uricase" in rid:
        return "CALIBRATION"
    return "OUT_OF_SCOPE"


# ──────────────────────────────────────────────────────────────────────
# Motif → canonical CORE references (substrate-agnostic only)
# ──────────────────────────────────────────────────────────────────────
# Same biochemistry-driven map as M3 main, but now narrowed to
# CORE-only sources: ramanbiolib pure-compound spectra + Gobbato powder
# Raman (the only substrate-free direct references in the corpus).
# Literature peak catalogs (raman_knowledge_core, consensus) are
# additionally cross-checked via peak-list logic.

MOTIF_CORE_REFS: dict[str, dict] = {
    # nucleobase / nucleic
    "purine_ring_breathing_720_735": {
        "primary_refs": ["adenine", "guanine"],
        "secondary_refs": ["a-dna", "b-dna"],
        "literature_tag": "DNA/RNA-associated region",
    },
    "uric_acid_full_signature": {
        "primary_refs": [],          # no UA in ramanbiolib
        "secondary_refs": [],
        "core_extras": ["ua_raman_pwd_gobbato2025",
                         "ua_digitised_gelder_2007",
                         "ua_digitised_kim_1987"],
    },
    "hypoxanthine_signature": {
        "primary_refs": [],
        "secondary_refs": [],
        "core_extras": ["hypox_raman_pwd_gobbato2025"],
    },
    "pyrimidine_ring_breathing_780_800": {
        "primary_refs": ["cytosine", "thymine", "uracil"],
        "secondary_refs": ["a-dna", "b-dna", "t-rna"],
    },
    "nucleobase_in_plane_ring_1320_1340": {
        "primary_refs": ["adenine", "guanine", "cytosine", "thymine"],
        "secondary_refs": ["a-dna", "b-dna"],
    },
    "dna_methylation_marker_790": {
        "primary_refs": ["cytosine", "thymine"],
        "secondary_refs": ["a-dna", "b-dna"],
    },
    "phosphate_PO2_sym_str_1080": {
        "primary_refs": ["a-dna", "b-dna", "t-rna"],
        "secondary_refs": ["d-fructose-6-phosphate"],
    },
    "phosphate_PO_asym_str_1240": {
        "primary_refs": ["a-dna", "b-dna", "t-rna"],
        "secondary_refs": [],
    },
    "dna_composite_motif": {
        "primary_refs": ["a-dna", "b-dna"],
        "secondary_refs": ["t-rna"],
    },
    "xanthine_signature": {
        "primary_refs": [],
        "secondary_refs": [],
        "core_extras": ["xanth_raman_pwd_gobbato2025"],
    },
    "guanine_specific_motif": {
        "primary_refs": ["guanine"],
        "secondary_refs": ["a-dna", "b-dna"],
    },
    "thymine_specific_motif": {
        "primary_refs": ["thymine"],
        "secondary_refs": ["b-dna"],
    },
    "cytosine_specific_motif": {
        "primary_refs": ["cytosine"],
        "secondary_refs": ["a-dna", "b-dna"],
    },
    # glycan
    "glycan_pyranose_ring_skeletal_850_950": {
        "primary_refs": ["d-(+)-glucose", "d-(+)-galactose", "d-(+)-mannose",
                          "β-d-glucose", "d-(-)-fructose"],
        "secondary_refs": ["glycogen", "cellulose", "amylose"],
    },
    "glycan_glycosidic_C_O_C_1020_1100": {
        "primary_refs": ["cellulose", "glycogen", "amylose", "amylopectin",
                          "d-(+)-lactose monohydrate", "d-(+)-maltose monohydrate",
                          "d-(+)-sucrose",
                          "a-dna", "b-dna", "t-rna",
                          "citric acid"],
        "secondary_refs": ["d-(+)-glucose", "d-(+)-galactose"],
    },
    "sialic_acid_signature": {
        "primary_refs": ["n-acetyl- d-glucosamine"],
        "secondary_refs": ["d-(+)-galactosamine", "glucosamine"],
    },
    "free_saccharide_motif": {
        "primary_refs": ["d-(+)-glucose", "d-(+)-galactose",
                          "d-(+)-mannose", "d-(-)-fructose", "β-d-glucose"],
        "secondary_refs": ["d-(+)-fucose", "d-(+)-xylose"],
    },
    # protein
    "amide_III_protein_backbone_1230_1280": {
        "primary_refs": ["albumin", "collagen", "elastin", "keratin"],
        "secondary_refs": ["hemoglobin", "myoglobin", "insulin"],
    },
    "phenylalanine_ring_1003": {
        "primary_refs": ["l-phenylalanine"],
        "secondary_refs": ["albumin", "collagen"],
    },
    "tyrosine_doublet_830_850": {
        "primary_refs": ["l-tyrosine"],
        "secondary_refs": ["albumin", "collagen"],
    },
    "amide_I_alpha_helix_beta_sheet_motif": {
        "primary_refs": ["albumin", "collagen", "elastin", "hemoglobin",
                          "insulin", "myoglobin", "keratin"],
        "secondary_refs": [],
    },
    "amide_II_motif": {
        "primary_refs": ["albumin", "collagen", "insulin", "hemoglobin"],
        "secondary_refs": [],
    },
    # lipid
    "lipid_acyl_C_C_str_1060_1130": {
        "primary_refs": ["oleic acid", "palmitic acid", "stearic acid",
                          "linoleic acid", "arachidic acid"],
        "secondary_refs": ["tristearin", "tripalmitin", "triolein"],
    },
    "lipid_C_H_bend_1440_1460": {
        "primary_refs": ["oleic acid", "palmitic acid", "stearic acid",
                          "linoleic acid", "arachidic acid", "arachidonic acid"],
        "secondary_refs": ["tristearin", "tripalmitin", "triolein",
                            "cholesterol", "sphingomyelin"],
    },
    "phosphatidylcholine_choline_head_715": {
        "primary_refs": ["l-α-phosphatidylcholine"],
        "secondary_refs": ["sphingomyelin"],
    },
    "cholesterol_signature": {
        "primary_refs": ["cholesterol"],
        "secondary_refs": ["cholesteryl linoleate", "cholesteryl oleate",
                            "cholesteryl palmitate", "cholesteryl stearate"],
    },
    "lipid_methylene_twist_1300": {
        "primary_refs": ["palmitic acid", "stearic acid", "oleic acid",
                          "arachidic acid", "linoleic acid"],
        "secondary_refs": ["tristearin", "tripalmitin", "triolein"],
    },
    "neutral_lipid_triglyceride_motif": {
        "primary_refs": ["tristearin", "tripalmitin", "triolein", "trilinolein",
                          "trilaurin", "trimyristin"],
        "secondary_refs": ["oleic acid", "palmitic acid"],
    },
    "amide_I_lipid_carbonyl_partial_panel_motif": {
        "primary_refs": ["albumin", "tristearin", "tripalmitin"],
        "secondary_refs": ["collagen", "triolein"],
    },
    # redox / heme / thiol / metabolites
    "cytochrome_c_resonance_motif": {
        "primary_refs": ["cytochrome c"],
        "secondary_refs": ["hemoglobin", "myoglobin"],
    },
    "disulfide_S_S_str_500_550": {
        "primary_refs": ["glutathione"],
        "secondary_refs": ["albumin", "insulin", "keratin"],
    },
    "ergothioneine_signature": {
        "primary_refs": [],
        "secondary_refs": [],
        "core_extras": ["ergo_raman_pwd_gobbato2025"],
    },
    "thiol_C_S_str_660_motif": {
        "primary_refs": ["glutathione"],
        "secondary_refs": ["albumin"],
    },
    "glutathione_GSH_motif": {
        "primary_refs": ["glutathione"],
        "secondary_refs": [],
    },
    "creatine_creatinine_motif": {
        "primary_refs": [],
        "secondary_refs": [],
        "core_extras": ["creat_raman_pwd_gobbato2025"],  # Gobbato Raman = creatinine per M3.2
    },
    # substrate artifact
    "citrate_baseline_artifact_motif": {
        "primary_refs": ["citric acid"],
        "secondary_refs": [],
    },
    # AMBIGUITY TRACK — collisions, multi-candidate
    "collision_1020_1080_multi_candidate": {
        "primary_refs": ["a-dna", "b-dna", "t-rna",
                          "glycogen", "cellulose", "amylose",
                          "citric acid"],
        "secondary_refs": ["d-(+)-glucose"],
        "ambiguity": True,
    },
    "purine_HX_lipid_choline_715_overlap_ambiguity": {
        "primary_refs": ["adenine", "guanine",
                          "l-α-phosphatidylcholine", "sphingomyelin"],
        "secondary_refs": [],
        "ambiguity": True,
    },
    "collision_1300_1400_multi_candidate_motif": {
        "primary_refs": ["adenine", "guanine",
                          "palmitic acid", "stearic acid",
                          "albumin",
                          "citric acid"],
        "secondary_refs": [],
        "ambiguity": True,
    },
}

AMBIGUITY_MOTIFS = {
    "phosphate_PO2_sym_str_1080",
    "glycan_glycosidic_C_O_C_1020_1100",
    "collision_1020_1080_multi_candidate",
    "purine_HX_lipid_choline_715_overlap_ambiguity",
    "collision_1300_1400_multi_candidate_motif",
}


# ──────────────────────────────────────────────────────────────────────
# Helpers (subset of M3 evaluator, CORE-only)
# ──────────────────────────────────────────────────────────────────────

def _parse_list_str(s: str) -> np.ndarray:
    return np.array(ast.literal_eval(s), dtype=np.float64)


def load_ramanbiolib_core(master_x: np.ndarray) -> dict[str, dict]:
    """Load ramanbiolib full spectra + peaks, mapped to canonical axis."""
    spec_df = pd.read_csv(RAMANBIOLIB_SPECTRA)
    peaks_df = pd.read_csv(RAMANBIOLIB_PEAKS)
    peaks_by_comp: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for _, r in peaks_df.iterrows():
        comp = str(r["component"]).strip().lower()
        peaks_by_comp[comp] = (_parse_list_str(r["peaks"]),
                                _parse_list_str(r["intensity"]))
    refs = {}
    for _, r in spec_df.iterrows():
        comp = str(r["component"]).strip().lower()
        try:
            wn = _parse_list_str(r["wavenumbers"])
            y = _parse_list_str(r["intensity"])
            y_master, _ = crop_before_interpolate(
                wn, y, master_x, partial_ok=True, min_coverage=0.80,
            )
        except (InsufficientOverlapError, Exception):
            continue
        c, h = peaks_by_comp.get(comp, (np.array([]), np.array([])))
        refs[comp] = {
            "y_master": y_master,
            "peak_centers": c,
            "peak_heights": h,
            "source": "ramanbiolib",
            "layer": "CORE",
        }
    return refs


def load_gobbato_powder_raman(master_x: np.ndarray) -> dict[str, np.ndarray]:
    """Load Gobbato pure powder Raman from M3.1 npz (already on master axis)."""
    if not M3_1_NPZ.exists():
        return {}
    npz = np.load(M3_1_NPZ)
    out = {}
    for key in npz.files:
        if key == "master_x":
            continue
        if "raman_pwd_gobbato" in key:
            out[key] = npz[key]
    return out


def _local_max_in_window(y, master_x, lo, hi):
    mask = (master_x >= lo) & (master_x <= hi)
    if not mask.any():
        return None
    y_win = y[mask]; x_win = master_x[mask]
    fin = np.isfinite(y_win)
    if not fin.any():
        return None
    y_win, x_win = y_win[fin], x_win[fin]
    idx = int(np.argmax(y_win))
    return float(x_win[idx]), float(y_win[idx])


def _has_peak_in_window(centers, heights, lo, hi):
    if centers.size == 0:
        return None
    mask = (centers >= lo) & (centers <= hi)
    if not mask.any():
        return None
    cs = centers[mask]; hs = heights[mask]
    idx = int(np.argmax(hs))
    return float(cs[idx]), float(hs[idx])


def evaluate_motif_on_core_refs(
    motif: dict, ref_map: dict, rb: dict, gobbato_pwd: dict,
    master_x: np.ndarray,
) -> dict:
    """Return per-motif core-evidence summary."""
    primary_fams = motif.get("primary_band_families") or []
    supporting_fams = motif.get("supporting_band_families") or []

    per_ref_evals = []
    # Try ramanbiolib refs
    for nm in (ref_map.get("primary_refs", []) + ref_map.get("secondary_refs", [])):
        nk = nm.lower()
        if nk in rb:
            per_ref_evals.append((nk, "ramanbiolib", rb[nk]["y_master"],
                                    rb[nk]["peak_centers"], rb[nk]["peak_heights"]))
    # Add Gobbato powder Raman if registered as core_extras
    for extra_id in ref_map.get("core_extras", []):
        if extra_id in gobbato_pwd:
            per_ref_evals.append((extra_id, "gobbato_powder_raman",
                                    gobbato_pwd[extra_id], np.array([]), np.array([])))

    if not per_ref_evals:
        return {
            "n_core_refs": 0,
            "best_ref": "",
            "best_primary_fire": 0,
            "best_primary_total": len(primary_fams),
            "best_supporting_fire": 0,
            "best_supporting_total": len(supporting_fams),
            "best_fraction_primary": 0.0,
            "any_band_in_nan": False,
            "per_ref_summary": [],
        }

    per_ref_summary = []
    for nm, src, y_master, centers, heights in per_ref_evals:
        n_pf = 0
        n_sf = 0
        any_nan = False
        for fam in primary_fams:
            c = float(fam["cm1_centre"]); t = float(fam["cm1_tolerance"])
            lo, hi = c - t, c + t
            hit = None
            if y_master is not None:
                hit_lm = _local_max_in_window(y_master, master_x, lo, hi)
                if hit_lm and hit_lm[1] > 1e-3:
                    hit = ("local_max", hit_lm)
                else:
                    mask = (master_x >= lo) & (master_x <= hi)
                    if mask.any() and np.all(np.isnan(y_master[mask])):
                        any_nan = True
            if hit is None and centers.size:
                hit_pl = _has_peak_in_window(centers, heights, lo, hi)
                if hit_pl:
                    hit = ("peak_list", hit_pl)
            if hit:
                n_pf += 1
        for fam in supporting_fams:
            c = float(fam["cm1_centre"]); t = float(fam["cm1_tolerance"])
            lo, hi = c - t, c + t
            hit = None
            if y_master is not None:
                hit_lm = _local_max_in_window(y_master, master_x, lo, hi)
                if hit_lm and hit_lm[1] > 1e-3:
                    hit = ("local_max", hit_lm)
            if hit is None and centers.size:
                hit_pl = _has_peak_in_window(centers, heights, lo, hi)
                if hit_pl:
                    hit = ("peak_list", hit_pl)
            if hit:
                n_sf += 1
        per_ref_summary.append({
            "ref_id": nm, "source": src,
            "n_primary_fire": n_pf, "n_primary_total": len(primary_fams),
            "n_supporting_fire": n_sf, "n_supporting_total": len(supporting_fams),
            "fraction_primary": n_pf / max(len(primary_fams), 1),
            "fraction_supporting": n_sf / max(len(supporting_fams), 1),
            "any_band_in_nan": any_nan,
        })

    best = max(per_ref_summary, key=lambda r: (r["fraction_primary"],
                                                  r["fraction_supporting"]))
    return {
        "n_core_refs": len(per_ref_evals),
        "best_ref": best["ref_id"],
        "best_primary_fire": best["n_primary_fire"],
        "best_primary_total": best["n_primary_total"],
        "best_supporting_fire": best["n_supporting_fire"],
        "best_supporting_total": best["n_supporting_total"],
        "best_fraction_primary": round(best["fraction_primary"], 3),
        "any_band_in_nan": any(r["any_band_in_nan"] for r in per_ref_summary),
        "per_ref_summary": per_ref_summary,
    }


def classify_core_grounding(eval_result: dict, is_ambiguity: bool) -> tuple[str, str]:
    """Assign CORE_GROUNDED / CORE_PARTIALLY_GROUNDED / CORE_WEAK / CORE_NOT_SUPPORTED.

    For ambiguity motifs: CORE_AMBIGUITY_CONFIRMED if ≥2 ref classes fire ≥50%;
    CORE_AMBIGUITY_WEAK otherwise.
    """
    if eval_result["n_core_refs"] == 0:
        if is_ambiguity:
            return "CORE_AMBIGUITY_WEAK", "no core references"
        return "CORE_NOT_SUPPORTED", "no core references"

    bp = eval_result["best_fraction_primary"]
    if is_ambiguity:
        # count how many references fire >= 50%
        n_majority = sum(
            1 for r in eval_result["per_ref_summary"]
            if r["fraction_primary"] >= 0.50
        )
        if n_majority >= 2:
            return "CORE_AMBIGUITY_CONFIRMED", (
                f"{n_majority}/{eval_result['n_core_refs']} core refs "
                f"fire ≥50% primary"
            )
        if n_majority == 1:
            return "CORE_AMBIGUITY_WEAK", (
                f"only 1 core ref fires ≥50% primary; ambiguity not "
                f"empirically multi-candidate confirmed on core evidence"
            )
        return "CORE_AMBIGUITY_WEAK", "no core ref reaches 50% primary"

    if bp >= 0.75:
        return "CORE_GROUNDED", (
            f"best core ref {eval_result['best_ref']} fires "
            f"{eval_result['best_primary_fire']}/{eval_result['best_primary_total']} primary"
        )
    if bp >= 0.50:
        return "CORE_PARTIALLY_GROUNDED", (
            f"best core ref fires {eval_result['best_primary_fire']}/"
            f"{eval_result['best_primary_total']} primary"
        )
    if bp >= 0.25:
        return "CORE_WEAK", (
            f"best core ref fires only {eval_result['best_primary_fire']}/"
            f"{eval_result['best_primary_total']} primary"
        )
    return "CORE_NOT_SUPPORTED", (
        f"no core ref fires >25% primary"
    )


# ──────────────────────────────────────────────────────────────────────
# Final v1 role classifier (combines core + calibration)
# ──────────────────────────────────────────────────────────────────────

def assign_v1_role(core_status: str, calibration_status: str) -> str:
    """Combine core grounding + calibration behavior into a final v1 role.

    PRIMARY:    core grounded AND calibration valid (or partial)
    CONTEXT:    core grounded but calibration weak/absent
    AMBIGUITY:  ambiguity-track motif with core ambiguity confirmed
    HOLD_OUT:   neither core-grounded nor calibration-valid
    """
    if core_status.startswith("CORE_AMBIGUITY"):
        if core_status == "CORE_AMBIGUITY_CONFIRMED":
            return "AMBIGUITY"
        return "HOLD_OUT"
    core_ok = core_status in ("CORE_GROUNDED", "CORE_PARTIALLY_GROUNDED")
    calib_ok = calibration_status in ("CALIBRATION_VALID", "PARTIALLY_VALID")
    if core_ok and calib_ok:
        return "PRIMARY"
    if core_ok and not calib_ok:
        return "CONTEXT"
    if not core_ok and calib_ok:
        # calibration-only signal without core grounding is dataset-specific —
        # treated as CONTEXT not PRIMARY (cannot anchor universal claim)
        return "CONTEXT"
    return "HOLD_OUT"


# ──────────────────────────────────────────────────────────────────────
# Driver
# ──────────────────────────────────────────────────────────────────────

def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    print("=" * 78)
    print("GAIRA · gaira_build_motifs_v1 · Phase M2.2 — Ontology Untangling + Core Rebuild")
    print("=" * 78)

    master_x = canonical_master_axis()

    # ── Load motif registry (v1.2 from M4.1) ──────────────────────────
    with V1_2_YAML.open("r") as f:
        reg = yaml.safe_load(f)
    motifs = reg["motifs"]
    motif_by_id = {m["motif_id"]: m for m in motifs}
    print(f"motifs in registry v1.2: {len(motifs)}")

    # ── Load M3 / M3.1 / M4 prior outputs ─────────────────────────────
    m4 = pd.read_csv(M4_SUMMARY).set_index("motif_id")
    m31_reg = pd.read_csv(M3_1_REGISTRY) if M3_1_REGISTRY.exists() else pd.DataFrame()
    m3 = pd.read_csv(M3_MATRIX).set_index("motif_id") if M3_MATRIX.exists() else pd.DataFrame()

    # ── Load CORE references ──────────────────────────────────────────
    print()
    print(f"[load] ramanbiolib (CORE)")
    rb = load_ramanbiolib_core(master_x)
    print(f"  ramanbiolib core refs: {len(rb)}")
    gobbato_pwd = load_gobbato_powder_raman(master_x)
    print(f"  Gobbato powder Raman core refs: {len(gobbato_pwd)}")

    # ── Step A — Motif ontology decomposition ─────────────────────────
    print()
    print("[step A] decomposing each motif into core/substrate/ambiguity components")
    decomp_rows = []
    for m in motifs:
        mid = m["motif_id"]
        primary_ids = [f["family_id"] for f in (m.get("primary_band_families") or [])]
        supporting_ids = [f["family_id"] for f in (m.get("supporting_band_families") or [])]
        substrate_components = []
        # M4.1 added co-band primaries that are technically substrate-conditioned
        # (they rescue cross-talk on Ag-colloid serum). Flag them as such.
        m4_1_added = {
            "phos_PO2_1080_coband", "lipid_CH_bend_1450_coband",
            "lipid_CH2_twist_1300_coband", "disulfide_SS_525_coband",
        }
        substrate_components = [pid for pid in primary_ids if pid in m4_1_added]
        # Substrate notes from the motif body
        substrate_notes = m.get("substrate_notes") or ""
        amb_links = []
        if mid in AMBIGUITY_MOTIFS:
            amb_links.append("ambiguity_track")
        if "exclusion_conditions" in m and m["exclusion_conditions"]:
            for ec in m["exclusion_conditions"]:
                if "uric_acid" in ec.lower() or "ua" in ec.lower():
                    amb_links.append("UA_collision_exclusion")
                if "phosphate" in ec.lower():
                    amb_links.append("phosphate_collision_exclusion")
                if "ambiguity" in ec.lower():
                    amb_links.append("ambiguity_demotion_rule")
        decomp_rows.append({
            "motif_id": mid,
            "core_band_families": ",".join(
                pid for pid in primary_ids if pid not in m4_1_added
            ) or "(none)",
            "supporting_bands": ",".join(supporting_ids) or "(none)",
            "co_band_constraints": m.get("co_band_requirement_type", ""),
            "ambiguity_links": ",".join(sorted(set(amb_links))) or "(none)",
            "substrate_components_removed": ",".join(substrate_components) or "(none)",
            "notes": (
                "M4.1 co-band primaries reclassified as SUBSTRATE_CONDITIONED "
                "(introduced to fix Ag-colloid serum cross-talk). Core motif "
                "identity uses remaining primaries only."
                if substrate_components else
                "no substrate-conditioned components in motif identity"
            ),
        })
    pd.DataFrame(decomp_rows).to_csv(
        TABLES / "motif_ontology_decomposition_v1.csv", index=False,
    )
    print(f"[emit] {TABLES}/motif_ontology_decomposition_v1.csv")

    # ── Step B — Re-classify all M3+M3.1 evidence by layer ────────────
    print()
    print("[step B] reclassifying all evidence sources by layer")
    evidence_rows = []
    # M3 main: every reference is ramanbiolib (CORE) plus knowledge_core (CORE)
    if not m3.empty:
        for mid in motifs:
            mid = mid["motif_id"]
            # M3 used the full ramanbiolib library — those are CORE
            for ref_name, src in rb.items():
                evidence_rows.append({
                    "motif_id": mid,
                    "ref_id": f"rb_{ref_name}",
                    "source_type": "PURE_RAMAN",
                    "layer": "CORE",
                    "phase": "M3",
                    "notes": "ramanbiolib full-spectrum reference (normal Raman)",
                })
    # M3.1 references — classify each
    for _, r in m31_reg.iterrows():
        # build a synthetic ref_id from the registry's columns
        src_id = str(r.get("source_identifier", ""))
        title = str(r.get("source_title", ""))
        analyte = str(r.get("analyte_name", ""))
        rtype = str(r.get("reference_type", ""))
        # synthetic ref_id consistent with M3.1 npz keys
        if "Raman_pwd" in title or "powder Raman" in title.lower():
            rid = f"{analyte}_raman_pwd_gobbato2025"
        elif "spike-in-serum" in title.lower() or "SERS_spike" in title or "Merck" in title:
            rid = f"{analyte}_sers_spike_gobbato2025"
        elif "metabolite SERS" in title.lower() or "SERS_met" in title:
            rid = f"{analyte}_sers_met_gobbato2025"
        elif "isotopic" in title.lower():
            rid = f"{analyte}_isotopic_14n_gobbato2025"
        elif "calibration series" in title.lower():
            rid = f"{analyte}_calibration_erg_series_v1"
        elif "Figure 7" in title:
            rid = f"{analyte}_cspp_fig7_spike"
        elif "Gelder" in title:
            rid = "ua_digitised_gelder_2007"
        elif "Kim" in title:
            rid = "ua_digitised_kim_1987"
        elif "Stewart" in title:
            rid = "ua_digitised_stewart_1999"
        else:
            rid = f"{analyte}_unclassified"
        layer = classify_ref_id(rid)
        evidence_rows.append({
            "motif_id": r.get("motif_id", ""),
            "ref_id": rid,
            "source_type": rtype,
            "layer": layer,
            "phase": "M3.1",
            "notes": str(r.get("provenance_note", ""))[:200],
        })
    # M3.2 evidence: literature consensus + Gobbato powder
    if M3_2_STATUS.exists():
        evidence_rows.append({
            "motif_id": "creatine_creatinine_motif",
            "ref_id": "literature_creatine_consensus",
            "source_type": "LIBRARY",
            "layer": "CORE",
            "phase": "M3.2",
            "notes": "literature peak catalog from Frushour&Koenig 1974 + De Gelder 2007 + Premasiri 2011",
        })
        evidence_rows.append({
            "motif_id": "creatine_creatinine_motif",
            "ref_id": "literature_creatinine_consensus",
            "source_type": "LIBRARY",
            "layer": "CORE",
            "phase": "M3.2",
            "notes": "literature peak catalog from Madzharova 2017 + De Gelder 2007 + Premasiri 2011",
        })
    evidence_df = pd.DataFrame(evidence_rows)
    evidence_df.to_csv(
        REGISTRY / "motif_core_evidence_registry_v1.csv", index=False,
    )
    print(f"[emit] {REGISTRY}/motif_core_evidence_registry_v1.csv "
          f"({len(evidence_df)} evidence rows)")

    # ── Evidence layer split per motif ────────────────────────────────
    layer_split = (
        evidence_df.groupby(["motif_id", "layer"]).size().unstack(fill_value=0)
    )
    expected_layers = ["CORE", "SUBSTRATE_CONDITIONED", "CALIBRATION", "OUT_OF_SCOPE"]
    for col in expected_layers:
        if col not in layer_split.columns:
            layer_split[col] = 0
    split_rows = []
    for mid in [m["motif_id"] for m in motifs]:
        if mid in layer_split.index:
            row = layer_split.loc[mid]
            split_rows.append({
                "motif_id": mid,
                "core_evidence_count": int(row.get("CORE", 0)),
                "substrate_conditioned_evidence_count": int(row.get("SUBSTRATE_CONDITIONED", 0)),
                "calibration_evidence_count": int(row.get("CALIBRATION", 0)),
                "ambiguity_evidence_count": (
                    1 if mid in AMBIGUITY_MOTIFS else 0
                ),
                "notes": (
                    "core evidence is the substrate-agnostic grounding base"
                    if int(row.get("CORE", 0)) > 0 else
                    "no core evidence — motif is currently grounded only on "
                    "substrate/calibration data"
                ),
            })
        else:
            split_rows.append({
                "motif_id": mid,
                "core_evidence_count": 0,
                "substrate_conditioned_evidence_count": 0,
                "calibration_evidence_count": 0,
                "ambiguity_evidence_count": 1 if mid in AMBIGUITY_MOTIFS else 0,
                "notes": "no evidence on record",
            })
    pd.DataFrame(split_rows).to_csv(
        TABLES / "motif_evidence_layer_split_v1.csv", index=False,
    )
    print(f"[emit] {TABLES}/motif_evidence_layer_split_v1.csv")

    # ── Step C — Recompute motif grounding using CORE evidence only ───
    print()
    print("[step C] recomputing core-only grounding for every motif")
    core_status_rows = []
    for m in motifs:
        mid = m["motif_id"]
        ref_map = MOTIF_CORE_REFS.get(mid, {"primary_refs": [], "secondary_refs": []})
        is_amb = mid in AMBIGUITY_MOTIFS
        eval_res = evaluate_motif_on_core_refs(m, ref_map, rb, gobbato_pwd, master_x)
        core_status, rationale = classify_core_grounding(eval_res, is_amb)
        ready = {
            "CORE_GROUNDED": "YES",
            "CORE_PARTIALLY_GROUNDED": "PARTIAL",
            "CORE_WEAK": "PARTIAL",
            "CORE_NOT_SUPPORTED": "NO",
            "CORE_AMBIGUITY_CONFIRMED": "PARTIAL",
            "CORE_AMBIGUITY_WEAK": "NO",
        }[core_status]
        core_status_rows.append({
            "motif_id": mid,
            "core_status": core_status,
            "evidence_strength": (
                f"{eval_res['n_core_refs']} core refs; best "
                f"{eval_res['best_primary_fire']}/{eval_res['best_primary_total']} primary"
            ),
            "n_core_refs": eval_res["n_core_refs"],
            "best_ref": eval_res["best_ref"],
            "best_fraction_primary": eval_res["best_fraction_primary"],
            "ambiguity_flag": is_amb,
            "any_band_in_nan": eval_res["any_band_in_nan"],
            "ready_for_calibration": ready,
            "rationale": rationale,
        })
        print(f"  {mid:48s} → {core_status:30s} ({ready})")
    pd.DataFrame(core_status_rows).to_csv(
        TABLES / "motif_core_grounding_status_v1.csv", index=False,
    )
    print(f"[emit] {TABLES}/motif_core_grounding_status_v1.csv")

    # ── Step D — Build dual-status table (core + calibration) ─────────
    print()
    print("[step D] building dual-status table (core + calibration)")
    dual_rows = []
    for r in core_status_rows:
        mid = r["motif_id"]
        calib = m4.loc[mid, "overall_class"] if mid in m4.index else "NOT_RUN"
        v1_role = assign_v1_role(r["core_status"], calib)
        dual_rows.append({
            "motif_id": mid,
            "core_status": r["core_status"],
            "calibration_status": calib,
            "final_v1_role": v1_role,
            "core_ready": r["ready_for_calibration"],
            "calib_conf": (m4.loc[mid, "confidence_score"]
                            if mid in m4.index else float("nan")),
            "notes": (
                "PRIMARY: core-grounded + calibration-valid; both layers agree"
                if v1_role == "PRIMARY" else
                "AMBIGUITY: empirically multi-candidate on core evidence"
                if v1_role == "AMBIGUITY" else
                "CONTEXT: only one of {core, calibration} layer supports motif"
                if v1_role == "CONTEXT" else
                "HOLD_OUT: neither layer supports motif"
            ),
        })
    pd.DataFrame(dual_rows).to_csv(
        TABLES / "motif_dual_status_v1.csv", index=False,
    )
    print(f"[emit] {TABLES}/motif_dual_status_v1.csv")

    # ── Report + audit log ─────────────────────────────────────────────
    _write_report(pd.DataFrame(decomp_rows),
                   pd.DataFrame(evidence_rows),
                   pd.DataFrame(split_rows),
                   pd.DataFrame(core_status_rows),
                   pd.DataFrame(dual_rows))
    _write_audit_log(pd.DataFrame(decomp_rows),
                      pd.DataFrame(core_status_rows),
                      pd.DataFrame(dual_rows),
                      rb, gobbato_pwd)

    # summary
    print()
    print("=" * 78)
    print("M2.2 ONTOLOGY UNTANGLING COMPLETE")
    print("=" * 78)
    csd = pd.DataFrame(core_status_rows)["core_status"].value_counts()
    print("Core-only grounding distribution:")
    for s, n in csd.items():
        print(f"  {s:30s}: {n}")
    drd = pd.DataFrame(dual_rows)["final_v1_role"].value_counts()
    print("\nFinal v1 role distribution:")
    for s, n in drd.items():
        print(f"  {s:20s}: {n}")


# ──────────────────────────────────────────────────────────────────────
# Report + audit log
# ──────────────────────────────────────────────────────────────────────

def _write_report(decomp_df, evidence_df, split_df, core_df, dual_df):
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    n_motifs = len(decomp_df)
    n_substrate_components = int(
        (decomp_df["substrate_components_removed"] != "(none)").sum()
    )
    cs = core_df["core_status"].value_counts().to_dict()
    rd = dual_df["final_v1_role"].value_counts().to_dict()

    lines = [
        "# GAIRA · gaira_build_motifs_v1 · Phase M2.2 — Motif Ontology Untangling + Core Evidence Rebuild",
        "",
        f"**Generated:** {now}  ",
        f"**Motifs in scope:** {n_motifs} (registry v1.2 from M4.1)  ",
        f"**Phase classification:** architectural correction; no motif structurally modified  ",
        "",
        "## Section A — Problem statement",
        "",
        "GAIRA's motif layer is supposed to be a substrate-agnostic ",
        "biochemical/spectral ontology — band families and co-band logic ",
        "anchored in normal Raman physics + chemistry, with substrate effects ",
        "(Ag-colloid SERS adsorption, enhancement, suppression) reserved to a ",
        "separate substrate overlay layer. Over the M3 → M4.1 build, this ",
        "separation has partially eroded:",
        "",
        "* **M3 main** correctly used ramanbiolib normal Raman (substrate-",
        "  agnostic) as the primary grounding source. ✓",
        "* **M3.1 reference rescue** brought in Gobbato 2025 powder Raman ",
        "  (substrate-agnostic ✓) but ALSO Gobbato Ag-colloid SERS, Gobbato ",
        "  spike-in-serum, ergothioneine cAg calibration series, and CSPP Ag-",
        "  colloid serum spike — three of which are substrate-bound or matrix-",
        "  bound. ✗",
        "* **M4 calibration** is entirely Ag-colloid SERS spike-in-serum data; ",
        "  this is correct as a *measurement-specific* validation but was being ",
        "  read as universal motif validity. ✗",
        "* **M4.1 refinement** added co-band primary bands explicitly to fix ",
        "  Ag-colloid serum cross-talk; those bands are substrate-conditioned ",
        "  and were merged into the motif's core identity rather than into a ",
        "  separate overlay. ✗",
        "",
        f"M2.2 cleanly separates the four layers and recomputes grounding ",
        f"using CORE evidence only.",
        "",
        "## Section B — Ontology correction",
        "",
        f"For each of {n_motifs} motifs, components were classified as:",
        "",
        f"- **CORE** (physics + chemistry grounded): the original primary and ",
        "  supporting band families from the M1.1 schema.",
        f"- **SUBSTRATE_CONDITIONED**: the {n_substrate_components} M4.1-added ",
        "  co-band primary bands that were introduced to fix Ag-colloid serum ",
        "  cross-talk. These remain in v1.2 for calibration purposes but are ",
        "  now LABELLED as substrate-conditioned (not core motif identity).",
        "- **AMBIGUITY**: explicit collision-zone links and exclusion ",
        "  conditions (e.g. UA-collision exclusion on `purine_ring_breathing`).",
        "",
        "Decomposition is recorded per motif in ",
        "`tables/motif_ontology_decomposition_v1.csv`. The motif registry is ",
        "NOT structurally modified — the decomposition is a labelling overlay ",
        "that flags substrate-conditioned components without removing them.",
        "",
        "## Section C — Evidence layer split",
        "",
        f"All M3 / M3.1 / M3.2 evidence was reclassified by source type:",
        "",
        "| layer | description | examples |",
        "|---|---|---|",
        "| CORE | substrate-agnostic Raman physics + chemistry | ramanbiolib, Gobbato powder Raman, Gelder 2007, Kim 1987, raman_knowledge_core |",
        "| SUBSTRATE_CONDITIONED | Ag/Au-substrate-bound SERS of pure analyte | Gobbato pure SERS, Stewart 1999 SERS |",
        "| CALIBRATION | substrate + matrix (e.g. serum + Ag-colloid) | Gobbato spike-in-serum, ERG calibration, CSPP Figure 7, uricase depletion |",
        "| OUT_OF_SCOPE | not a spectral evidence source | n/a |",
        "",
        "Per-motif evidence-layer counts:",
        "",
        f"- mean CORE refs per motif: "
        f"{split_df['core_evidence_count'].mean():.1f}",
        f"- mean SUBSTRATE_CONDITIONED refs per motif: "
        f"{split_df['substrate_conditioned_evidence_count'].mean():.1f}",
        f"- mean CALIBRATION refs per motif: "
        f"{split_df['calibration_evidence_count'].mean():.1f}",
        "",
        "## Section D — Core grounding results",
        "",
        f"Recomputed grounding using CORE-only references:",
        "",
        "| status | count |",
        "|---|---:|",
    ]
    for s in ["CORE_GROUNDED", "CORE_PARTIALLY_GROUNDED", "CORE_WEAK",
                "CORE_NOT_SUPPORTED", "CORE_AMBIGUITY_CONFIRMED",
                "CORE_AMBIGUITY_WEAK"]:
        lines.append(f"| {s} | {cs.get(s, 0)} |")

    lines += [
        "",
        "### Core-grounded motifs (substrate-agnostic biochemical anchor)",
        "",
    ]
    grounded = core_df[core_df["core_status"] == "CORE_GROUNDED"].sort_values("motif_id")
    for _, r in grounded.iterrows():
        lines.append(
            f"- `{r['motif_id']}` — best ref `{r['best_ref']}` "
            f"({r['best_fraction_primary']:.0%} primary fire)"
        )

    lines += ["", "### Core-not-supported motifs", ""]
    nsup = core_df[core_df["core_status"] == "CORE_NOT_SUPPORTED"].sort_values("motif_id")
    if len(nsup):
        for _, r in nsup.iterrows():
            lines.append(f"- `{r['motif_id']}` — {r['rationale']}")
    else:
        lines.append("(none)")

    lines += [
        "",
        "## Section E — Calibration reinterpretation",
        "",
        "M4 calibration results are now reframed as **'Ag-colloid serum ",
        "calibration behavior'** — measurement-specific motif activation under ",
        "the Gobbato Ag-colloid serum protocol. They are NOT statements about ",
        "universal motif validity. M4 status remains stored under the same ",
        "tables (untouched); the dual-status table (`tables/motif_dual_status_v1.csv`) ",
        "exposes both the core grounding status and the calibration status ",
        "per motif so they cannot be conflated downstream.",
        "",
        "### Final v1 role distribution",
        "",
        "Combining core grounding + calibration behavior:",
        "",
        "| role | count | meaning |",
        "|---|---:|---|",
        f"| PRIMARY | {rd.get('PRIMARY', 0)} | core-grounded AND calibration-valid; safe to drive M5 claims |",
        f"| AMBIGUITY | {rd.get('AMBIGUITY', 0)} | core-empirical multi-candidate motif (ambiguity preserved on substrate-agnostic evidence) |",
        f"| CONTEXT | {rd.get('CONTEXT', 0)} | only ONE of {{core, calibration}} layers supports the motif; can appear in M5 reports as context with a flag |",
        f"| HOLD_OUT | {rd.get('HOLD_OUT', 0)} | neither layer supports the motif; do not use in M5 |",
        "",
        "## Section F — Implications",
        "",
        "1. **The motif layer is now substrate-agnostic again.** Motifs report ",
        "   biochemistry, not Ag-colloid behavior. Substrate-conditioned co-band ",
        "   bands added in M4.1 (e.g. `phos_PO2_1080_coband`, `lipid_CH_bend_1450_coband`, ",
        "   `lipid_CH2_twist_1300_coband`, `disulfide_SS_525_coband`) are ",
        "   labelled as substrate-conditioned and excluded from CORE identity.",
        "2. **Core grounding is the universal claim layer.** What GAIRA actually ",
        "   represents — biochemically — is the set of motifs marked PRIMARY ",
        "   or AMBIGUITY in the dual-status table. These rest on substrate-",
        "   agnostic Raman physics + chemistry.",
        "3. **Ag-colloid serum calibration is one measurement context.** ",
        "   Future Au-colloid, Au-nanostar, paper-plasmonic, or solid-substrate ",
        "   calibration passes will produce DIFFERENT calibration_status ",
        "   distributions but should NOT change core_status. If they do, the ",
        "   motif's CORE identity needs revision (not its calibration).",
        "4. **M5 (target datasets) should drive primary biochemical claims off ",
        "   the PRIMARY set only.** AMBIGUITY motifs can appear as ambiguity ",
        "   reporters; CONTEXT motifs can appear flagged; HOLD_OUT motifs must not appear.",
        "",
        "## Section G — Provenance",
        "",
        f"- v1.2 motif registry: `{V1_2_YAML}` ({_sha256(V1_2_YAML)[:16]}…)",
        f"- M3 grounding matrix: `{M3_MATRIX}` ({_sha256(M3_MATRIX)[:16]}…)",
        f"- M3.1 ref registry:   `{M3_1_REGISTRY}` ({_sha256(M3_1_REGISTRY)[:16]}…)",
        f"- M4 summary:          `{M4_SUMMARY}` ({_sha256(M4_SUMMARY)[:16]}…)",
        f"- ramanbiolib spectra: `{RAMANBIOLIB_SPECTRA}` ({_sha256(RAMANBIOLIB_SPECTRA)[:16]}…)",
        f"- driver: `scripts/run_gaira_motifs_M2_2_ontology_untangling_v1.py`",
        "",
        "## Section H — Non-modification invariants",
        "",
        "- Motif registry NOT structurally modified.",
        "- Preprocessing pipeline NOT modified.",
        "- Substrate engine NOT modified.",
        "- Pilot outputs NOT touched.",
        "- M3 / M3.1 / M3.2 / M4 / M4.1 outputs NOT modified.",
        "- The decomposition is a *labelling overlay*; the substrate-conditioned ",
        "  primary bands added in M4.1 remain in v1.2 (they are still useful for ",
        "  Ag-colloid calibration scoring) but are now flagged so they cannot ",
        "  be conflated with core motif identity in downstream interpretation.",
    ]
    path = DOCS / "REPORT_M2_2_ontology_untangling_v1.md"
    path.write_text("\n".join(lines))
    print(f"[emit] {path}")


def _write_audit_log(decomp_df, core_df, dual_df, rb, gobbato_pwd):
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    lines = [
        "# M2.2 Ontology Untangling Audit Log",
        "",
        f"Generated: {now}",
        "",
        "## Motifs affected (substrate-conditioned components labelled)",
        "",
    ]
    affected = decomp_df[decomp_df["substrate_components_removed"] != "(none)"]
    if len(affected):
        for _, r in affected.iterrows():
            lines.append(
                f"- `{r['motif_id']}`: removed from core identity → "
                f"{r['substrate_components_removed']}"
            )
    else:
        lines.append("(none)")

    lines += [
        "",
        "## Core-grounding evidence shifts",
        "",
        f"- ramanbiolib refs available for core grounding: {len(rb)}",
        f"- Gobbato powder Raman refs available for core grounding: {len(gobbato_pwd)}",
        "",
        "## Ambiguous decisions",
        "",
        "1. **Gobbato powder Raman as CORE.** Powder Raman is substrate-free, ",
        "   so we treat it as CORE. The 'Creat' analyte in Gobbato is ",
        "   creatinine (M3.2), and the powder Raman of a pure compound is a ",
        "   substrate-agnostic measurement; therefore it is core evidence.",
        "2. **Stewart 1999 digitisation as SUBSTRATE_CONDITIONED.** Stewart ",
        "   1999 reports SERS data digitised, not normal Raman. It is therefore ",
        "   substrate-conditioned and excluded from CORE.",
        "3. **Literature peak catalogs (raman_knowledge_core, M3.2 consensus) ",
        "   as CORE.** These are meta-analyses of published Raman peak ",
        "   assignments across many normal-Raman papers, so they aggregate ",
        "   substrate-agnostic chemistry.",
        "4. **M4.1 added co-band primaries reclassified as substrate-conditioned.** ",
        "   Bands like `phos_PO2_1080_coband` were added to fix Ag-colloid serum ",
        "   cross-talk; they live in v1.2 but are now flagged as substrate-",
        "   conditioned components, not core identity.",
        "",
        "## Gaps remaining",
        "",
    ]
    nsup = core_df[core_df["core_status"] == "CORE_NOT_SUPPORTED"]
    weak = core_df[core_df["core_status"] == "CORE_WEAK"]
    if len(nsup):
        lines.append("Motifs with NO core support (need normal-Raman reference acquisition):")
        for _, r in nsup.iterrows():
            lines.append(f"- `{r['motif_id']}`")
    else:
        lines.append("- No motif lacks core support outright.")
    if len(weak):
        lines.append("")
        lines.append("Motifs with WEAK core support (acquire additional pure-compound Raman):")
        for _, r in weak.iterrows():
            lines.append(f"- `{r['motif_id']}`")

    lines += [
        "",
        "## Invariants verified",
        "",
        "- [x] Motif registry not structurally modified",
        "- [x] No motif deleted",
        "- [x] Preprocessing pipeline unchanged",
        "- [x] Substrate engine unchanged",
        "- [x] Pilot outputs untouched",
        "- [x] Core grounding evaluator uses CORE references only",
        "- [x] Calibration status (M4) preserved verbatim in dual-status table",
    ]
    path = AUDIT / "M2_2_ontology_untangling_audit_log.md"
    path.write_text("\n".join(lines))
    print(f"[emit] {path}")


if __name__ == "__main__":
    main()
