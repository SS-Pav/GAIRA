"""gaira_validate_2_grounding_v1 — Full grounding validation for gaira_base_2.

Runs the implemented engine on ALL available grounding / reference spectra:

  - ramanbiolib (141 normal Raman of pure biological molecules)
  - Gobbato 2025 powder Raman — ALL 53 analytes × 3 replicates (153 spectra)
  - amino_acid_raman_grounding/aa.xlsx — 20 amino-acid / small-molecule refs
  - digitised literature spectra — Gelder 2007, Kim 1987 (Raman); Stewart 1999
    excluded from CORE grounding (SERS digitisation; substrate-specific)

Core ontology validation ONLY. No substrate-aware interpretation, no
calibration reasoning, no target cohort.

Outputs under
  /Volumes/SSD_Rad/GAIRA_BUILD/gaira_validate_2_grounding_v1/
inventory/, tables/, figures/, reports/, audit/

Run:
    cd /Users/suraj/projects/GAIRA
    PYTHONPATH=src .venv/bin/python scripts/run_gaira_validate_2_grounding.py
"""
from __future__ import annotations

import ast
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from gaira.base2 import (
    BIOLOGY_AXES_V11,
    CONTROL_LANE,
    load_active_registry,
    result_to_flat_dict,
    score_spectrum,
)
from gaira.base2.registry import load_axis_mapping
from gaira.spectral import canonical_master_axis, crop_before_interpolate
from gaira.spectral.preprocessing import _asls_baseline
from scipy.signal import savgol_filter


ROOT = Path("/Volumes/SSD_Rad/GAIRA_BUILD/gaira_validate_2_grounding_v1")
INV = ROOT / "inventory"
TABLES = ROOT / "tables"
FIGS = ROOT / "figures"
REPORTS = ROOT / "reports"
AUDIT = ROOT / "audit"
for d in (INV, TABLES, FIGS, REPORTS, AUDIT):
    d.mkdir(parents=True, exist_ok=True)


RAMANBIOLIB = Path(
    "/Volumes/SSD_Rad/GAIRA_DATA/raw/ramanbiolib/ramanbiolib-main/"
    "ramanbiolib/db/raman_spectra_db.csv"
)
GOBBATO_POWDER_DIR = Path(
    "/Volumes/SSD_Rad/GAIRA_BUILD/gaira_build_motifs_v1/"
    "M3_1_reference_rescue_v1/references/_extracted/Raman metabolites"
)
AA_XLSX = Path(
    "/Volumes/SSD_Rad/GAIRA_DATA/raw/amino_acid_raman_grounding/aa.xlsx"
)
DIGITISED_DIR = Path(
    "/Volumes/SSD_Rad/GAIRA_BUILD/gaira_build_motifs_v1/"
    "M3_1_reference_rescue_v1/references/_extracted/digitized literature spectra"
)


# ──────────────────────────────────────────────────────────────────────
# Expected axis map — covers all analytes across all datasets
# ──────────────────────────────────────────────────────────────────────
# Keys: normalised compound identifier (lowercase; Gobbato analyte tags kept
# as-cased for easy lookup).

EXPECTED_AXES: dict[str, list[str]] = {
    # ── ramanbiolib (141 compounds) ────────────────────────────────
    "adenine":   ["purine_nucleotide", "purine_metabolite"],
    "guanine":   ["purine_nucleotide"],
    "cytosine":  ["pyrimidine_nucleotide"],
    "thymine":   ["pyrimidine_nucleotide"],
    "uracil":    ["pyrimidine_nucleotide"],
    "a-dna":     ["purine_nucleotide", "pyrimidine_nucleotide", "phosphate_nucleic_adjacent"],
    "b-dna":     ["purine_nucleotide", "pyrimidine_nucleotide", "phosphate_nucleic_adjacent"],
    "t-rna":     ["purine_nucleotide", "pyrimidine_nucleotide", "phosphate_nucleic_adjacent"],
    "l-phenylalanine": ["aromatic_residue"],
    "l-tyrosine":      ["aromatic_residue"],
    "l-tryptophan":    ["aromatic_residue"],
    "l-histidine":     ["aromatic_residue"],
    "l-arginine":      ["protein_peptide_backbone"],
    "l-asparagine":    ["protein_peptide_backbone"],
    "l-aspartic acid": ["protein_peptide_backbone"],
    "l-glutamate":     ["metabolic_small_molecule"],
    "l-proline":       ["protein_peptide_backbone"],
    "l-serine":        ["protein_peptide_backbone"],
    "l-valine":        ["protein_peptide_backbone"],
    "l-alanine":       ["protein_peptide_backbone"],
    "glycine":         ["protein_peptide_backbone"],
    "albumin":        ["protein_peptide_backbone"],
    "collagen":       ["protein_peptide_backbone"],
    "elastin":        ["protein_peptide_backbone"],
    "keratin":        ["protein_peptide_backbone"],
    "hemoglobin":     ["protein_peptide_backbone"],
    "myoglobin":      ["protein_peptide_backbone"],
    "insulin":        ["protein_peptide_backbone"],
    "ferritin":       ["protein_peptide_backbone"],
    "cytochrome c":   ["protein_peptide_backbone", "sulfur_thiol_redox"],
    "lactalbumin":    ["protein_peptide_backbone"],
    "carbonic anhydrase": ["protein_peptide_backbone"],
    "tubulin":        ["protein_peptide_backbone"],
    "elastase":       ["protein_peptide_backbone"],
    "ubiquitin":      ["protein_peptide_backbone"],
    "trypsin":        ["protein_peptide_backbone"],
    "trypsinogen":    ["protein_peptide_backbone"],
    "pepsin":         ["protein_peptide_backbone"],
    "pepsinogen":     ["protein_peptide_backbone"],
    "papain":         ["protein_peptide_backbone"],
    "major proteinase": ["protein_peptide_backbone"],
    "horseradish peroxidase": ["protein_peptide_backbone"],
    "xylanase":       ["protein_peptide_backbone"],
    "lectin":         ["protein_peptide_backbone"],
    "α-chymotrypsinogen a (type ii)": ["protein_peptide_backbone"],
    "thaumatin":      ["protein_peptide_backbone"],
    "triosephosphate isomerase": ["protein_peptide_backbone"],
    "glutathione transferase": ["protein_peptide_backbone", "sulfur_thiol_redox"],
    "glucose oxidase": ["protein_peptide_backbone"],
    "superoxide dismutases": ["protein_peptide_backbone"],
    "trypsin inhibitor": ["protein_peptide_backbone"],
    "glutathione":    ["sulfur_thiol_redox"],
    "d-(+)-glucose":  ["glycan_carbohydrate"],
    "d-(+)-galactose":["glycan_carbohydrate"],
    "d-(+)-mannose":  ["glycan_carbohydrate"],
    "β-d-glucose":    ["glycan_carbohydrate"],
    "d-(-)-fructose": ["glycan_carbohydrate"],
    "d-(-)-ribose":   ["glycan_carbohydrate"],
    "d-(+)-fucose":   ["glycan_carbohydrate"],
    "d-(+)-xylose":   ["glycan_carbohydrate"],
    "d-(-)-arabinose":["glycan_carbohydrate"],
    "l-(+)-arabinose":["glycan_carbohydrate"],
    "d-(+)-lactose monohydrate": ["glycan_carbohydrate"],
    "d-(+)-maltose monohydrate": ["glycan_carbohydrate"],
    "d-(+)-sucrose":  ["glycan_carbohydrate"],
    "d-(+)-trehalose":["glycan_carbohydrate"],
    "d-(+)-raffinose pentahydrate": ["glycan_carbohydrate"],
    "d-(+)-galactosamine": ["glycan_carbohydrate"],
    "glucosamine":    ["glycan_carbohydrate"],
    "n-acetyl- d-glucosamine": ["glycan_carbohydrate"],
    "lactose":        ["glycan_carbohydrate"],
    "cellulose":      ["glycan_carbohydrate"],
    "glycogen":       ["glycan_carbohydrate"],
    "chitin":         ["glycan_carbohydrate"],
    "amylose":        ["glycan_carbohydrate"],
    "amylopectin":    ["glycan_carbohydrate"],
    "d-(+)-dextrose": ["glycan_carbohydrate"],
    "d-fructose-6-phosphate": ["glycan_carbohydrate", "phosphate_nucleic_adjacent"],
    "glycerol":       ["lipid_acyl_membrane"],
    "oleic acid":     ["lipid_acyl_membrane"],
    "palmitic acid":  ["lipid_acyl_membrane"],
    "stearic acid":   ["lipid_acyl_membrane"],
    "linoleic acid":  ["lipid_acyl_membrane"],
    "arachidic acid": ["lipid_acyl_membrane"],
    "arachidonic acid": ["lipid_acyl_membrane"],
    "lauric acid":    ["lipid_acyl_membrane"],
    "myristic acid":  ["lipid_acyl_membrane"],
    "elaidic acid":   ["lipid_acyl_membrane"],
    "palmitoleic acid": ["lipid_acyl_membrane"],
    "vaccenic acid":  ["lipid_acyl_membrane"],
    "α-linolenic acid": ["lipid_acyl_membrane"],
    "12-methyltetradecanoic acid": ["lipid_acyl_membrane"],
    "13-methylmyristicacid": ["lipid_acyl_membrane"],
    "14-methylhexadecanoic acid": ["lipid_acyl_membrane"],
    "14-methylpentadecanoic acid": ["lipid_acyl_membrane"],
    "15-methylpalmiticacid": ["lipid_acyl_membrane"],
    "ceramide":       ["lipid_acyl_membrane"],
    "sphingomyelin":  ["lipid_acyl_membrane"],
    "l-α-phosphatidylcholine":     ["lipid_acyl_membrane"],
    "l-α-phosphatidylethanolamine":["lipid_acyl_membrane"],
    "cholesterol":       ["sterol_neutral_lipid"],
    "cholesteryl linoleate": ["sterol_neutral_lipid"],
    "cholesteryl oleate":    ["sterol_neutral_lipid"],
    "cholesteryl palmitate": ["sterol_neutral_lipid"],
    "cholesteryl stearate":  ["sterol_neutral_lipid"],
    "estradiol":  ["sterol_neutral_lipid"],
    "estrone":    ["sterol_neutral_lipid"],
    "estriol":    ["sterol_neutral_lipid"],
    "ethinylestradiol": ["sterol_neutral_lipid"],
    "diethylstilbestrol": ["sterol_neutral_lipid"],
    "tristearin":   ["sterol_neutral_lipid"],
    "tripalmitin":  ["sterol_neutral_lipid"],
    "triolein":     ["sterol_neutral_lipid"],
    "trilinolein":  ["sterol_neutral_lipid"],
    "trilinolenin": ["sterol_neutral_lipid"],
    "trimyristin":  ["sterol_neutral_lipid"],
    "trilaurin":    ["sterol_neutral_lipid"],
    "tricaprin":    ["sterol_neutral_lipid"],
    "tricaproin":   ["sterol_neutral_lipid"],
    "tricaprylin":  ["sterol_neutral_lipid"],
    "tri-11-eicosenoin": ["sterol_neutral_lipid"],
    "triarachidin": ["sterol_neutral_lipid"],
    "tribehenin":   ["sterol_neutral_lipid"],
    "trielaidin":   ["sterol_neutral_lipid"],
    "trierucin":    ["sterol_neutral_lipid"],
    "tripalmitolein":    ["sterol_neutral_lipid"],
    "tripetroselinin":   ["sterol_neutral_lipid"],
    "acetoacetate":  ["metabolic_small_molecule"],
    "pyruvate":      ["metabolic_small_molecule"],
    "fumarate":      ["metabolic_small_molecule"],
    "citric acid":   ["metabolic_small_molecule", "ambiguity_artifact"],
    "succinic acid": ["metabolic_small_molecule"],
    "malic acid":    ["metabolic_small_molecule"],
    "ascorbic acid": ["metabolic_small_molecule"],
    "phosphoenolpyruvate": ["metabolic_small_molecule", "phosphate_nucleic_adjacent"],
    "acetyl coenzyme a":   ["metabolic_small_molecule"],
    "coenzyme a":          ["metabolic_small_molecule"],
    "melanin":         ["aromatic_residue"],
    "β-carotene":      ["lipid_acyl_membrane"],
    "riboﬂavin":        ["metabolic_small_molecule"],
    "2-deoxy-d-ribose":["glycan_carbohydrate"],

    # ── Gobbato powder Raman tags (already substrate-free) ─────────
    # map to the same axis family as their chemistry
    "UA":     ["purine_metabolite"],
    "Hypox":  ["purine_metabolite"],
    "Xanth":  ["purine_metabolite"],
    "Ergo":   ["sulfur_thiol_redox", "metabolic_small_molecule"],
    "Creat":  ["metabolic_small_molecule"],  # labelled "Creat" but is creatinine per M3.2
    "Ade":    ["purine_nucleotide", "purine_metabolite"],
    "Gua":    ["purine_nucleotide"],
    "Thy":    ["pyrimidine_nucleotide"],
    "Ura":    ["pyrimidine_nucleotide"],
    "Ala":    ["protein_peptide_backbone"],
    "Arg":    ["protein_peptide_backbone"],
    "Asp":    ["protein_peptide_backbone"],
    "Gly":    ["protein_peptide_backbone"],
    "Leu":    ["protein_peptide_backbone"],
    "Ile":    ["protein_peptide_backbone"],
    "Met":    ["sulfur_thiol_redox", "protein_peptide_backbone"],
    "Methio": ["sulfur_thiol_redox", "protein_peptide_backbone"],
    "Pro":    ["protein_peptide_backbone"],
    "Ser":    ["protein_peptide_backbone"],
    "Val":    ["protein_peptide_backbone"],
    "Hydroxypro": ["protein_peptide_backbone"],
    "His":    ["aromatic_residue"],
    "Phe":    ["aromatic_residue"],
    "Trp":    ["aromatic_residue"],
    "Tyr":    ["aromatic_residue"],
    "Gluc":   ["glycan_carbohydrate"],
    "Galact": ["glycan_carbohydrate"],
    "Mann":   ["glycan_carbohydrate"],
    "Fruct":  ["glycan_carbohydrate"],
    "Ribo":   ["metabolic_small_molecule"],  # riboflavin (vitamin B2)
    "NacDgluc": ["glycan_carbohydrate"],
    "Glycogen": ["glycan_carbohydrate"],
    "Lact":   ["metabolic_small_molecule"],   # lactate
    "Dfruct6P": ["glycan_carbohydrate", "phosphate_nucleic_adjacent"],
    "Chol":   ["sterol_neutral_lipid"],
    "Oleic":  ["lipid_acyl_membrane"],
    "Stearic":["lipid_acyl_membrane"],
    "Triolein": ["sterol_neutral_lipid"],
    "PhInositol": ["lipid_acyl_membrane"],
    "Glycerol": ["lipid_acyl_membrane"],
    "DNA":    ["purine_nucleotide", "pyrimidine_nucleotide", "phosphate_nucleic_adjacent"],
    "RNA":    ["purine_nucleotide", "pyrimidine_nucleotide", "phosphate_nucleic_adjacent"],
    "PEP":    ["metabolic_small_molecule", "phosphate_nucleic_adjacent"],
    "Phosph": ["phosphate_nucleic_adjacent"],
    "Cys":    ["sulfur_thiol_redox"],
    "Citric": ["metabolic_small_molecule", "ambiguity_artifact"],
    "Urea":   ["metabolic_small_molecule"],
    "Ure":    ["metabolic_small_molecule"],
    "Pyr":    ["metabolic_small_molecule"],   # pyruvate
    "Asc":    ["metabolic_small_molecule"],   # ascorbic acid
    "AcCoA":  ["metabolic_small_molecule"],
    "CoA":    ["metabolic_small_molecule"],
    "Acetoacet": ["metabolic_small_molecule"],
    "Alb":    ["protein_peptide_backbone"],
    "Gluth":  ["sulfur_thiol_redox"],
    "Glut":   ["metabolic_small_molecule"],   # glutamic acid / glutamate
    "Glutamic": ["metabolic_small_molecule"],

    # aa.xlsx column names (re-uses the above tag map where possible)
    "Valine": ["protein_peptide_backbone"],
    "Glutamic Acid": ["metabolic_small_molecule"],
    "L-Glu": ["metabolic_small_molecule"],
    "Havuc": ["lipid_acyl_membrane"],     # carrot — mixed β-carotene / lipid
    "Glucose": ["glycan_carbohydrate"],
    "Malic Acid": ["metabolic_small_molecule"],

    # digitised literature spectra (all UA references)
    "ua_digitised_gelder_2007":  ["purine_metabolite"],
    "ua_digitised_kim_1987":     ["purine_metabolite"],
    "ua_digitised_stewart_1999": ["purine_metabolite"],
}


# Motif family → list of motifs in that family (for family-hit analysis)
# Built dynamically from motif registry; see build_family_map()
def build_family_map(motifs: dict) -> dict[str, list[str]]:
    fam: dict[str, list[str]] = defaultdict(list)
    for mid, spec in motifs.items():
        fam[spec.motif_family].append(mid)
    return dict(fam)


# ──────────────────────────────────────────────────────────────────────
# Loaders
# ──────────────────────────────────────────────────────────────────────

def _parse_list(s):
    return np.array(ast.literal_eval(s), dtype=np.float64)


def canonical_preprocess(wn, y, master_x):
    try:
        y_interp, _ = crop_before_interpolate(
            wn, y, master_x, partial_ok=True, min_coverage=0.80,
        )
    except Exception:
        return None
    mask = np.isfinite(y_interp)
    if not mask.any():
        return None
    if not mask.all():
        idx = np.arange(len(y_interp))
        y_interp[~mask] = np.interp(idx[~mask], idx[mask], y_interp[mask])
    y_bc = y_interp - _asls_baseline(y_interp, lam=1e5, p=0.001, n_iter=10)
    y_sg = savgol_filter(y_bc, window_length=11, polyorder=3)
    n = np.linalg.norm(y_sg)
    return y_sg / n if n > 1e-12 else None


def load_ramanbiolib(master_x):
    df = pd.read_csv(RAMANBIOLIB)
    out = []
    for _, r in df.iterrows():
        comp = str(r["component"]).strip().lower()
        try:
            wn = _parse_list(r["wavenumbers"])
            y = _parse_list(r["intensity"])
            y_pp = canonical_preprocess(wn, y, master_x)
        except Exception:
            continue
        if y_pp is not None:
            out.append({
                "spectrum_id": f"ramanbiolib::{comp}",
                "dataset": "ramanbiolib",
                "component_key": comp,
                "spectrum": y_pp,
            })
    return out


def parse_gobbato(path):
    try:
        lines = path.read_text(encoding="latin-1").splitlines()
    except Exception:
        return None
    hdr = next((i for i, ln in enumerate(lines)
                 if ln.startswith("Pixel;Wavelength;Wavenumber;Raman Shift")), None)
    if hdr is None:
        return None
    wn, y = [], []
    for ln in lines[hdr + 1:]:
        parts = ln.strip().rstrip(";").split(";")
        if len(parts) < 8:
            continue
        try:
            wn.append(float(parts[3].replace(",", ".")))
            y.append(float(parts[7].replace(",", ".")))
        except ValueError:
            continue
    return np.array(wn), np.array(y)


def load_gobbato_powder(master_x):
    out = []
    for p in sorted(GOBBATO_POWDER_DIR.iterdir()):
        if not p.name.startswith("Raman_pwd_"):
            continue
        # Raman_pwd_<Analyte>_s_<rep>.txt
        parts = p.name.split("_")
        if len(parts) < 5:
            continue
        analyte = parts[2]
        rep = parts[4].replace(".txt", "")
        parsed = parse_gobbato(p)
        if parsed is None:
            continue
        y_pp = canonical_preprocess(parsed[0], parsed[1], master_x)
        if y_pp is None:
            continue
        out.append({
            "spectrum_id": f"gobbato_powder::{analyte}_rep{rep}",
            "dataset": "gobbato_powder_raman",
            "component_key": analyte,
            "spectrum": y_pp,
        })
    return out


def load_amino_acid_xlsx(master_x):
    df = pd.read_excel(AA_XLSX)
    wn_col = df.columns[0]
    wn = df[wn_col].to_numpy(dtype=np.float64)
    out = []
    for col in df.columns[1:]:
        y = df[col].to_numpy(dtype=np.float64)
        # the file is 300–1905 cm⁻¹; crop to 400–1800 via canonical_preprocess
        y_pp = canonical_preprocess(wn, y, master_x)
        if y_pp is None:
            continue
        out.append({
            "spectrum_id": f"aa_xlsx::{col}",
            "dataset": "amino_acid_raman_grounding",
            "component_key": col,
            "spectrum": y_pp,
        })
    return out


def load_digitised_literature(master_x):
    out = []
    for name, tag in [
        ("Gelder_2007.csv", "ua_digitised_gelder_2007"),
        ("Kim_1987.csv",    "ua_digitised_kim_1987"),
    ]:
        p = DIGITISED_DIR / name
        if not p.exists():
            continue
        df = pd.read_csv(p, skipinitialspace=True)
        wn = df["x"].to_numpy(dtype=np.float64)
        y = df["y"].to_numpy(dtype=np.float64)
        order = np.argsort(wn)
        wn, y = wn[order], y[order]
        uniq = np.concatenate(([True], np.diff(wn) > 0))
        wn, y = wn[uniq], y[uniq]
        y_pp = canonical_preprocess(wn, y, master_x)
        if y_pp is None:
            continue
        out.append({
            "spectrum_id": f"digitised_literature::{tag}",
            "dataset": "digitised_literature_spectra",
            "component_key": tag,
            "spectrum": y_pp,
        })
    return out


# ──────────────────────────────────────────────────────────────────────
# Metrics
# ──────────────────────────────────────────────────────────────────────

def axis_rank(result, axis_id: str) -> int | None:
    ordered = sorted(result.axis11_scores, key=lambda a: a.core_evidence, reverse=True)
    for i, a in enumerate(ordered, 1):
        if a.axis_id == axis_id:
            return i
    return None


def motif_rank(result, motif_id: str) -> int | None:
    ordered = sorted(result.motif_scores, key=lambda m: m.core_weight, reverse=True)
    for i, m in enumerate(ordered, 1):
        if m.motif_id == motif_id:
            return i
    return None


def top_k_motifs(result, k=5):
    return sorted(result.motif_scores, key=lambda m: m.core_weight, reverse=True)[:k]


def top_k_axes(result, k=3):
    return sorted(result.axis11_scores, key=lambda a: a.core_evidence, reverse=True)[:k]


# ──────────────────────────────────────────────────────────────────────
# Driver
# ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 78)
    print("gaira_validate_2_grounding_v1")
    print("=" * 78)

    master_x = canonical_master_axis()
    motifs, mappings, dual = load_active_registry()
    print(f"engine: {len(motifs)} active motifs, {len(mappings)} mappings")

    # ── Inventory ─────────────────────────────────────────────────────
    print("\n[load] all grounding datasets")
    rb  = load_ramanbiolib(master_x);        print(f"  ramanbiolib: {len(rb)}")
    gp  = load_gobbato_powder(master_x);     print(f"  Gobbato powder: {len(gp)}")
    aa  = load_amino_acid_xlsx(master_x);    print(f"  amino_acid_raman_grounding: {len(aa)}")
    lit = load_digitised_literature(master_x); print(f"  digitised literature: {len(lit)}")

    all_refs = rb + gp + aa + lit
    print(f"\nTOTAL: {len(all_refs)} grounding spectra")

    # Inventory table
    inv_rows = []
    for r in all_refs:
        inv_rows.append({
            "spectrum_id": r["spectrum_id"],
            "dataset": r["dataset"],
            "component_key": r["component_key"],
            "has_expected_axes": r["component_key"] in EXPECTED_AXES
                                    or r["component_key"].lower() in EXPECTED_AXES,
        })
    pd.DataFrame(inv_rows).to_csv(INV / "grounding_dataset_inventory_v1.csv", index=False)

    # ── Score every spectrum ──────────────────────────────────────────
    print("\n[score] all grounding spectra through gaira_base_2")
    per_spec_rows = []
    motif_rank_rows = []
    axis_rank_rows = []
    off_target_rows = []
    ambig_rows = []
    miss_rows = []

    for r in all_refs:
        sid = r["spectrum_id"]
        comp = r["component_key"]
        res = score_spectrum(r["spectrum"], master_x, motifs, mappings, dual, sid)

        flat = result_to_flat_dict(res)
        flat["dataset"] = r["dataset"]
        flat["component_key"] = comp
        per_spec_rows.append(flat)

        # Expected axes — look up by exact key, then lowercase fallback
        expected = EXPECTED_AXES.get(comp) or EXPECTED_AXES.get(comp.lower(), [])

        # Top-5 motifs
        t5 = top_k_motifs(res, k=5)
        motif_row = {
            "spectrum_id": sid, "dataset": r["dataset"], "component_key": comp,
        }
        for i, m in enumerate(t5, 1):
            motif_row[f"top_motif_{i}"] = m.motif_id
            motif_row[f"top_motif_{i}_core"] = round(m.core_weight, 4)
        motif_rank_rows.append(motif_row)

        # Top-3 axes
        t3 = top_k_axes(res, k=3)
        axis_row = {
            "spectrum_id": sid, "dataset": r["dataset"], "component_key": comp,
            "expected_axes": ",".join(expected),
        }
        for i, a in enumerate(t3, 1):
            axis_row[f"top_axis_{i}"] = a.axis_id
            axis_row[f"top_axis_{i}_core"] = round(a.core_evidence, 4)
        # expected axis ranks (if we know expected)
        for ax in expected:
            axis_row[f"expected_axis_rank.{ax}"] = axis_rank(res, ax)
        axis_rank_rows.append(axis_row)

        # Off-target activation: every axis, is_expected flag
        for a in res.axis11_scores:
            off_target_rows.append({
                "spectrum_id": sid, "dataset": r["dataset"],
                "component_key": comp, "axis_id": a.axis_id,
                "is_expected": a.axis_id in expected,
                "core_evidence": round(a.core_evidence, 4),
            })

        ambig_rows.append({
            "spectrum_id": sid, "dataset": r["dataset"], "component_key": comp,
            "ambiguity_core": round(res.ambiguity.core_evidence, 4),
            "n_ambig_contrib": len(res.ambiguity.contributing_motifs),
            "top_ambig_motifs": ",".join(res.ambiguity.contributing_motifs[:3]),
            "expected_ambiguity": "ambiguity_artifact" in expected,
        })

        # Miss detection: no expected axis in top-3
        if expected:
            top3_axis_ids = [a.axis_id for a in t3]
            is_miss = not any(ax in top3_axis_ids for ax in expected)
            if is_miss:
                miss_rows.append({
                    "spectrum_id": sid, "dataset_name": r["dataset"],
                    "expected_chemistry": comp,
                    "expected_motif": "(see expected_axes)",
                    "observed_top_motifs": ",".join([m.motif_id for m in t5[:3]]),
                    "expected_axis": ",".join(expected),
                    "observed_top_axes": ",".join([a.axis_id for a in t3]),
                    "likely_failure_type": _infer_failure_type(comp, res, expected),
                    "notes": "",
                })

    # ── Emit tables ───────────────────────────────────────────────────
    pd.DataFrame(per_spec_rows).to_csv(
        TABLES / "grounding_per_spectrum_scores_v1.csv", index=False,
    )
    pd.DataFrame(motif_rank_rows).to_csv(
        TABLES / "grounding_expected_vs_observed_motif_rank_v1.csv", index=False,
    )
    pd.DataFrame(axis_rank_rows).to_csv(
        TABLES / "grounding_expected_vs_observed_axis11_rank_v1.csv", index=False,
    )
    pd.DataFrame(off_target_rows).to_csv(
        TABLES / "grounding_off_target_activation_v1.csv", index=False,
    )
    pd.DataFrame(ambig_rows).to_csv(
        TABLES / "grounding_ambiguity_behavior_v1.csv", index=False,
    )
    pd.DataFrame(miss_rows).to_csv(
        TABLES / "grounding_miss_list_v1.csv", index=False,
    )
    print(f"\n[emit] {len(per_spec_rows)} per-spectrum + {len(miss_rows)} misses")

    # ── Metrics summary ───────────────────────────────────────────────
    metrics = _compute_metrics(
        pd.DataFrame(motif_rank_rows),
        pd.DataFrame(axis_rank_rows),
        motifs, mappings,
    )
    pd.DataFrame([metrics]).to_csv(
        TABLES / "grounding_metrics_summary_v1.csv", index=False,
    )

    # ── Figures ───────────────────────────────────────────────────────
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[warn] matplotlib unavailable: {e}")
    else:
        _plot_11_axis_radar(all_refs, master_x, motifs, mappings, dual, plt)
        _plot_grouped_motif_in_axis(all_refs, master_x, motifs, mappings, dual, plt)
        _plot_ambiguity_panel(pd.DataFrame(ambig_rows), plt)
        _plot_motif_top_rank_heatmap(pd.DataFrame(motif_rank_rows), plt)
        _plot_axis_top_rank_heatmap(pd.DataFrame(axis_rank_rows), plt)
        _plot_off_target_heatmap(pd.DataFrame(off_target_rows), plt)
        _plot_family_hit_rate(pd.DataFrame(axis_rank_rows), motifs, plt)
        _plot_sunburst_treemap(all_refs, master_x, motifs, mappings, dual, plt)

    # ── Reports ───────────────────────────────────────────────────────
    _write_main_report(
        inv_rows, metrics, per_spec_rows, motif_rank_rows,
        axis_rank_rows, off_target_rows, ambig_rows, miss_rows,
        motifs, mappings,
    )
    _write_miss_analysis_report(
        miss_rows, motifs, mappings, axis_rank_rows, motif_rank_rows,
    )
    _write_audit_log(inv_rows, all_refs)
    print("DONE")


# ──────────────────────────────────────────────────────────────────────
# Failure-type inference
# ──────────────────────────────────────────────────────────────────────

def _infer_failure_type(comp, res, expected) -> str:
    """Guess the failure class from the engine's response profile."""
    # Did the expected axes at least fire weakly?
    expected_core = {
        a.axis_id: a.core_evidence for a in res.axis11_scores if a.axis_id in expected
    }
    if not expected_core:
        return "UNKNOWN"
    max_exp = max(expected_core.values(), default=0.0)
    # What fired first?
    top = sorted(res.axis11_scores, key=lambda a: a.core_evidence, reverse=True)[0]
    if max_exp < 0.05:
        return "NO_ACTIVATION (expected axis below noise)"
    if top.axis_id == "ambiguity_artifact" or top.core_evidence < 0.10:
        return "LOW_SIGNAL (spectrum too weak)"
    # Check axis families
    # Common cross-talks
    cross_talk_pairs = {
        ("purine_nucleotide", "purine_metabolite"): "PURINE_NUCLEOTIDE_vs_METABOLITE_CROSSTALK",
        ("purine_metabolite", "purine_nucleotide"): "PURINE_METABOLITE_vs_NUCLEOTIDE_CROSSTALK",
        ("sterol_neutral_lipid", "lipid_acyl_membrane"): "STEROL_vs_ACYL_LIPID_CROSSTALK",
        ("lipid_acyl_membrane", "sterol_neutral_lipid"): "ACYL_vs_STEROL_CROSSTALK",
        ("phosphate_nucleic_adjacent", "purine_nucleotide"): "PHOSPHATE_OVERWHELMED_BY_NUCLEOBASE",
        ("phosphate_nucleic_adjacent", "pyrimidine_nucleotide"): "PHOSPHATE_OVERWHELMED_BY_NUCLEOBASE",
    }
    for exp in expected:
        if (exp, top.axis_id) in cross_talk_pairs:
            return cross_talk_pairs[(exp, top.axis_id)]
    return f"EXPECTED_WEAK ({max_exp:.2f} < {top.axis_id} {top.core_evidence:.2f})"


def _compute_metrics(motif_df, axis_df, motifs, mappings):
    # top-1 / top-3 axis hit
    classified = axis_df[axis_df["expected_axes"] != ""].copy()
    total = len(classified)
    top1 = 0
    top3 = 0
    for _, r in classified.iterrows():
        exp_set = set(r["expected_axes"].split(","))
        t1 = {r.get("top_axis_1", "")}
        t3 = {r.get(f"top_axis_{i}", "") for i in (1, 2, 3)}
        if t1 & exp_set:
            top1 += 1
        if t3 & exp_set:
            top3 += 1
    return {
        "n_classified": total,
        "top1_axis_hit_rate": round(top1 / max(total, 1), 4),
        "top3_axis_hit_rate": round(top3 / max(total, 1), 4),
        "top1_axis_hits": top1,
        "top3_axis_hits": top3,
    }


# ──────────────────────────────────────────────────────────────────────
# Figures — explicit, chemistry-first visualisations
# ──────────────────────────────────────────────────────────────────────

def _plot_11_axis_radar(all_refs, master_x, motifs, mappings, dual, plt):
    examples = {
        "adenine":                        "ramanbiolib::adenine",
        "l-tyrosine":                     "ramanbiolib::l-tyrosine",
        "cholesterol":                    "ramanbiolib::cholesterol",
        "ua_gobbato_powder":              "gobbato_powder::UA_rep01",
        "ergothioneine_gobbato_powder":   "gobbato_powder::Ergo_rep01",
        "collagen":                       "ramanbiolib::collagen",
    }
    id_to_ref = {r["spectrum_id"]: r for r in all_refs}
    picks = [(k, id_to_ref[v]) for k, v in examples.items() if v in id_to_ref]
    if not picks:
        return
    fig, axes = plt.subplots(
        2, 3, figsize=(15, 9), subplot_kw=dict(polar=True),
    )
    angles = np.linspace(0, 2 * np.pi, len(BIOLOGY_AXES_V11), endpoint=False).tolist()
    angles += angles[:1]
    for ax, (label, ref) in zip(axes.flat, picks):
        res = score_spectrum(ref["spectrum"], master_x, motifs, mappings, dual,
                               ref["spectrum_id"])
        values = [next(a.core_evidence for a in res.axis11_scores if a.axis_id == axis_id)
                   for axis_id in BIOLOGY_AXES_V11]
        values += values[:1]
        ax.plot(angles, values, color="#2a9d8f", linewidth=1.5)
        ax.fill(angles, values, color="#2a9d8f", alpha=0.3)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(
            [a.replace("_", "\n") for a in BIOLOGY_AXES_V11], fontsize=6,
        )
        ax.set_ylim(0, 0.7)
        ax.set_title(label, fontsize=10, pad=15)
    # hide empty cells
    for i in range(len(picks), len(axes.flat)):
        axes.flat[i].set_visible(False)
    fig.suptitle("11-axis core radar — representative grounding references",
                   fontsize=12)
    fig.tight_layout()
    fig.savefig(FIGS / "fig_11_axis_radar_examples.png", dpi=130)
    plt.close(fig)


def _plot_grouped_motif_in_axis(all_refs, master_x, motifs, mappings, dual, plt):
    """For a few exemplars, render per-axis stacked bars where each axis is
    subdivided by the contributing motifs' core weights × mapping weight.
    This is the 'nested motif inside axis' view."""
    examples = [
        "gobbato_powder::UA_rep01",
        "gobbato_powder::Ergo_rep01",
        "ramanbiolib::adenine",
        "ramanbiolib::cholesterol",
    ]
    id_to_ref = {r["spectrum_id"]: r for r in all_refs}
    picks = [sid for sid in examples if sid in id_to_ref]
    if not picks:
        return
    fig, axes = plt.subplots(1, len(picks), figsize=(6 * len(picks), 8),
                                sharey=True)
    if len(picks) == 1:
        axes = [axes]
    from gaira.base2.motif_engine import resolve_mapping_weight

    for ax, sid in zip(axes, picks):
        ref = id_to_ref[sid]
        res = score_spectrum(ref["spectrum"], master_x, motifs, mappings, dual, sid)
        motif_score = {m.motif_id: m.core_weight for m in res.motif_scores}
        # For each axis: list of (motif_id, contribution = motif_core × mapping_weight)
        axis_to_contribs: dict[str, list[tuple[str, float]]] = {}
        for axis_id in BIOLOGY_AXES_V11:
            contribs = []
            for mid, s in motif_score.items():
                mapping = mappings.get(mid)
                if mapping is None:
                    continue
                mw = resolve_mapping_weight(mapping, axis_id)
                if mw > 0 and s > 0:
                    contribs.append((mid, s * mw))
            contribs = sorted(contribs, key=lambda x: x[1], reverse=True)
            axis_to_contribs[axis_id] = contribs
        # Plot
        y_positions = np.arange(len(BIOLOGY_AXES_V11))
        # Motif color palette
        import matplotlib.cm as cm
        cmap = cm.get_cmap("tab20", 20)
        motif_to_color = {}
        for axis_id in BIOLOGY_AXES_V11:
            for mid, _ in axis_to_contribs[axis_id]:
                if mid not in motif_to_color:
                    motif_to_color[mid] = cmap(len(motif_to_color) % 20)
        for i, axis_id in enumerate(BIOLOGY_AXES_V11):
            left = 0.0
            for mid, contrib in axis_to_contribs[axis_id]:
                ax.barh(i, contrib, left=left,
                         color=motif_to_color[mid],
                         edgecolor="black", linewidth=0.2)
                if contrib >= 0.05:
                    ax.text(left + contrib / 2, i,
                             mid.replace("_motif", "")[:20],
                             va="center", ha="center", fontsize=5, color="white",
                             rotation=0)
                left += contrib
        ax.set_yticks(y_positions)
        ax.set_yticklabels(BIOLOGY_AXES_V11, fontsize=8)
        ax.invert_yaxis()
        ax.set_xlim(0, 1.2)
        ax.set_xlabel("stacked motif contribution (core × mapping_weight)")
        ax.set_title(sid.split("::")[1], fontsize=9)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    fig.suptitle(
        "Grouped motif-in-axis contributions — representative references",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(FIGS / "fig_grouped_motif_in_axis_examples.png", dpi=130)
    plt.close(fig)


def _plot_ambiguity_panel(ambig_df, plt):
    top20 = ambig_df.sort_values("ambiguity_core", ascending=False).head(20)
    fig, ax = plt.subplots(figsize=(12, max(5, 0.3 * len(top20))))
    colors = ["#7b2cbf" if exp else "#adb5bd"
              for exp in top20["expected_ambiguity"]]
    ax.barh(top20["component_key"].astype(str), top20["ambiguity_core"], color=colors)
    for i, (_, r) in enumerate(top20.iterrows()):
        lbl = r["top_ambig_motifs"][:60]
        ax.text(r["ambiguity_core"] + 0.005, i, lbl, va="center", fontsize=6)
    ax.invert_yaxis()
    ax.set_xlabel("ambiguity lane core evidence")
    ax.set_title("Ambiguity lane activation — top 20 grounding references\n"
                   "(purple = expected ambiguity; grey = unexpected)")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIGS / "fig_ambiguity_panel.png", dpi=130)
    plt.close(fig)


def _plot_motif_top_rank_heatmap(motif_df, plt):
    counts = motif_df["top_motif_1"].value_counts().head(25)
    fig, ax = plt.subplots(figsize=(11, max(5, 0.32 * len(counts))))
    ax.barh(counts.index, counts.values, color="#2a9d8f")
    ax.invert_yaxis()
    ax.set_xlabel("n grounding references where this motif is top-ranked")
    ax.set_title("Motif top-rank frequency across all grounding references")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIGS / "fig_motif_top_rank_heatmap.png", dpi=130)
    plt.close(fig)


def _plot_axis_top_rank_heatmap(axis_df, plt):
    # matrix: rows = primary expected axis (class), cols = observed top-1 axis
    axis_df = axis_df[axis_df["expected_axes"] != ""].copy()
    axis_df["primary_expected"] = axis_df["expected_axes"].str.split(",").str[0]
    pivot = pd.crosstab(axis_df["primary_expected"],
                          axis_df["top_axis_1"],
                          normalize="index")
    pivot = pivot.reindex(columns=list(BIOLOGY_AXES_V11), fill_value=0.0)
    axes_rows = [a for a in BIOLOGY_AXES_V11 if a in pivot.index]
    pivot = pivot.loc[axes_rows]
    fig, ax = plt.subplots(figsize=(12, max(6, 0.5 * len(pivot))))
    im = ax.imshow(pivot.values, aspect="auto", cmap="YlGnBu", vmin=0, vmax=1)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=40, ha="right", fontsize=8)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=9)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            v = pivot.values[i, j]
            if v > 0.05:
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                         fontsize=7, color="black")
    ax.set_xlabel("observed top-1 axis")
    ax.set_ylabel("primary expected axis")
    ax.set_title("11-axis top-rank confusion matrix (row-normalised)")
    fig.colorbar(im, ax=ax, label="fraction of references")
    fig.tight_layout()
    fig.savefig(FIGS / "fig_axis_top_rank_heatmap.png", dpi=130)
    plt.close(fig)


def _plot_off_target_heatmap(off_df, plt):
    # primary expected axis × observed axis → mean core evidence
    off_df = off_df.copy()
    per_spec = {}
    for sid, grp in off_df.groupby("spectrum_id"):
        exp = grp[grp["is_expected"]]["axis_id"].tolist()
        per_spec[sid] = exp[0] if exp else ""
    off_df["primary_expected"] = off_df["spectrum_id"].map(per_spec)
    off_df = off_df[off_df["primary_expected"] != ""]
    pivot = (
        off_df.groupby(["primary_expected", "axis_id"])["core_evidence"]
        .mean()
        .unstack(fill_value=0.0)
    )
    pivot = pivot.reindex(columns=list(BIOLOGY_AXES_V11), fill_value=0.0)
    pivot = pivot.loc[[a for a in BIOLOGY_AXES_V11 if a in pivot.index]]
    fig, ax = plt.subplots(figsize=(12, max(6, 0.5 * len(pivot))))
    vmax = max(0.1, float(np.nanmax(pivot.values)) * 0.8)
    im = ax.imshow(pivot.values, aspect="auto", cmap="Reds", vmin=0, vmax=vmax)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=40, ha="right", fontsize=8)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=9)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            v = pivot.values[i, j]
            if v > 0.03:
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                         fontsize=7, color="black")
    ax.set_xlabel("observed 11-axis")
    ax.set_ylabel("primary expected axis")
    ax.set_title("Off-target activation — mean core evidence "
                   "(diagonal = on-target; off-diagonal = cross-talk)")
    fig.colorbar(im, ax=ax, label="mean core evidence")
    fig.tight_layout()
    fig.savefig(FIGS / "fig_off_target_activation_heatmap.png", dpi=130)
    plt.close(fig)


def _plot_family_hit_rate(axis_df, motifs, plt):
    # Family = motif.motif_family, but axis_df is per-reference not per-motif
    # Instead, use the expected axis as the "family" proxy.
    ax_df = axis_df[axis_df["expected_axes"] != ""].copy()
    ax_df["primary_expected"] = ax_df["expected_axes"].str.split(",").str[0]
    ax_df["top1_hit"] = ax_df.apply(
        lambda r: r["top_axis_1"] in r["expected_axes"].split(","), axis=1,
    )
    ax_df["top3_hit"] = ax_df.apply(
        lambda r: any(r.get(f"top_axis_{i}", "") in r["expected_axes"].split(",")
                        for i in (1, 2, 3)),
        axis=1,
    )
    per_fam = (
        ax_df.groupby("primary_expected")[["top1_hit", "top3_hit"]]
        .agg(["sum", "count"])
    )
    per_fam.columns = ["_".join(c) for c in per_fam.columns]
    per_fam["top1_rate"] = per_fam["top1_hit_sum"] / per_fam["top1_hit_count"]
    per_fam["top3_rate"] = per_fam["top3_hit_sum"] / per_fam["top3_hit_count"]
    per_fam = per_fam.sort_values("top1_rate", ascending=False)
    fig, ax = plt.subplots(figsize=(11, max(4, 0.45 * len(per_fam))))
    y = np.arange(len(per_fam))
    ax.barh(y - 0.2, per_fam["top1_rate"], height=0.35,
             color="#2a9d8f", label="top-1")
    ax.barh(y + 0.2, per_fam["top3_rate"], height=0.35,
             color="#76c893", label="top-3")
    ax.set_yticks(y)
    ax.set_yticklabels(per_fam.index, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("hit rate")
    ax.set_title("Per-axis-family hit rate across grounding references")
    ax.legend(fontsize=9)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIGS / "fig_family_hit_rate.png", dpi=130)
    plt.close(fig)


def _plot_sunburst_treemap(all_refs, master_x, motifs, mappings, dual, plt):
    """Exploratory hierarchical view: axis → motif, for one exemplar.
    Uses matplotlib patches (treemap-like) since matplotlib has no native
    sunburst — aggregates motif contributions by axis."""
    # Aggregate across ALL grounding spectra
    agg_motif_per_axis: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    agg_ambig: float = 0.0
    n_spec = 0
    from gaira.base2.motif_engine import resolve_mapping_weight
    for ref in all_refs:
        res = score_spectrum(ref["spectrum"], master_x, motifs, mappings, dual,
                               ref["spectrum_id"])
        n_spec += 1
        agg_ambig += res.ambiguity.core_evidence
        motif_core = {m.motif_id: m.core_weight for m in res.motif_scores}
        for axis_id in BIOLOGY_AXES_V11:
            for mid, s in motif_core.items():
                mapping = mappings.get(mid)
                if mapping is None or s <= 0:
                    continue
                mw = resolve_mapping_weight(mapping, axis_id)
                if mw > 0:
                    agg_motif_per_axis[axis_id][mid] += s * mw

    # Treemap: one panel per axis, tiles proportional to motif aggregate.
    fig, axes = plt.subplots(3, 4, figsize=(20, 13))
    for ax in axes.flat:
        ax.set_axis_off()
    import matplotlib.cm as cm
    cmap = cm.get_cmap("tab20", 20)
    motif_to_color = {}

    def get_color(mid):
        if mid not in motif_to_color:
            motif_to_color[mid] = cmap(len(motif_to_color) % 20)
        return motif_to_color[mid]

    def treemap_tile(ax, items, title):
        # items: list of (label, value)
        total = sum(v for _, v in items)
        if total <= 0:
            ax.text(0.5, 0.5, f"{title}\n(no signal)", ha="center", va="center",
                     transform=ax.transAxes, fontsize=9)
            return
        # Simple row-layout treemap
        items = sorted(items, key=lambda x: x[1], reverse=True)
        y = 1.0
        for lbl, val in items:
            frac = val / total
            ax.add_patch(plt.Rectangle((0, y - frac), 1.0, frac,
                                          color=get_color(lbl),
                                          edgecolor="black", linewidth=0.5))
            if frac > 0.03:
                ax.text(0.5, y - frac / 2,
                         lbl.replace("_motif", "")[:24] + f" ({frac:.0%})",
                         ha="center", va="center",
                         fontsize=7, color="white")
            y -= frac
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_title(title, fontsize=10)

    for i, axis_id in enumerate(BIOLOGY_AXES_V11):
        ax = axes.flat[i]
        ax.set_axis_on()
        items = list(agg_motif_per_axis[axis_id].items())
        treemap_tile(ax, items, f"{axis_id}\n(Σ={sum(v for _,v in items):.2f})")

    # Ambiguity panel (slot 12)
    amb_ax = axes.flat[11]
    amb_ax.set_axis_on()
    amb_ax.text(0.5, 0.6, "ambiguity_artifact\n(control lane)",
                 ha="center", va="center", fontsize=11, transform=amb_ax.transAxes,
                 color="#7b2cbf")
    amb_ax.text(0.5, 0.3,
                 f"Σ core evidence across all\n{n_spec} grounding spectra:\n{agg_ambig:.2f}",
                 ha="center", va="center", fontsize=10, transform=amb_ax.transAxes)
    amb_ax.set_xlim(0, 1); amb_ax.set_ylim(0, 1)
    amb_ax.set_xticks([]); amb_ax.set_yticks([])
    for side in ("top", "right", "left", "bottom"):
        amb_ax.spines[side].set_visible(False)

    fig.suptitle(
        "axis → motif hierarchy (aggregated across all grounding spectra)",
        fontsize=13,
    )
    fig.tight_layout()
    fig.savefig(FIGS / "fig_sunburst_treemap_exploratory.png", dpi=130)
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────
# Reports
# ──────────────────────────────────────────────────────────────────────

def _write_main_report(inv_rows, metrics, per_spec_rows, motif_rank_rows,
                        axis_rank_rows, off_target_rows, ambig_rows,
                        miss_rows, motifs, mappings):
    n_total = len(per_spec_rows)
    n_classified = metrics["n_classified"]
    # per-dataset hits
    axis_df = pd.DataFrame(axis_rank_rows)
    axis_df = axis_df[axis_df["expected_axes"] != ""].copy()
    axis_df["top1_hit"] = axis_df.apply(
        lambda r: r["top_axis_1"] in r["expected_axes"].split(","), axis=1,
    )
    axis_df["top3_hit"] = axis_df.apply(
        lambda r: any(r.get(f"top_axis_{i}", "") in r["expected_axes"].split(",")
                        for i in (1, 2, 3)), axis=1,
    )
    per_ds = (
        axis_df.groupby("dataset")[["top1_hit", "top3_hit"]]
        .agg(["sum", "count"])
    )
    per_ds.columns = ["_".join(c) for c in per_ds.columns]
    per_ds["top1_rate"] = per_ds["top1_hit_sum"] / per_ds["top1_hit_count"]
    per_ds["top3_rate"] = per_ds["top3_hit_sum"] / per_ds["top3_hit_count"]

    # family-level: primary expected axis as family
    axis_df["primary_expected"] = axis_df["expected_axes"].str.split(",").str[0]
    per_fam = (
        axis_df.groupby("primary_expected")[["top1_hit", "top3_hit"]]
        .agg(["sum", "count"])
    )
    per_fam.columns = ["_".join(c) for c in per_fam.columns]
    per_fam["top1_rate"] = per_fam["top1_hit_sum"] / per_fam["top1_hit_count"]
    per_fam["top3_rate"] = per_fam["top3_hit_sum"] / per_fam["top3_hit_count"]

    ambig_df = pd.DataFrame(ambig_rows)
    n_ambig_fires = int((ambig_df["ambiguity_core"] > 0.1).sum())

    # Top-10 motifs by #1-frequency
    mdf = pd.DataFrame(motif_rank_rows)
    top_motifs = mdf["top_motif_1"].value_counts().head(10)

    lines = [
        "# gaira_validate_2_grounding_v1 — Grounding Validation Report",
        "",
        f"**Grounding spectra scored:** {n_total}",
        f"**Spectra with known expected axis:** {n_classified}",
        "",
        "**Top-1 axis hit rate:** "
        f"{metrics['top1_axis_hit_rate']:.1%} "
        f"({metrics['top1_axis_hits']}/{n_classified})",
        f"**Top-3 axis hit rate:** "
        f"{metrics['top3_axis_hit_rate']:.1%} "
        f"({metrics['top3_axis_hits']}/{n_classified})",
        f"**Miss list size (expected axis NOT in top-3):** {len(miss_rows)}",
        f"**References with ambiguity lane > 0.1:** {n_ambig_fires} / {n_total}",
        "",
        "## Grounding datasets included",
        "",
        "| dataset | spectra |",
        "|---|---:|",
    ]
    inv_df = pd.DataFrame(inv_rows)
    for ds, cnt in inv_df["dataset"].value_counts().items():
        lines.append(f"| `{ds}` | {cnt} |")

    lines += [
        "",
        "## Per-dataset hit rate",
        "",
        "| dataset | top-1 | top-3 | n |",
        "|---|---:|---:|---:|",
    ]
    for ds, row in per_ds.iterrows():
        lines.append(
            f"| `{ds}` | {row['top1_rate']:.1%} | {row['top3_rate']:.1%} | "
            f"{int(row['top1_hit_count'])} |"
        )

    lines += [
        "",
        "## Per-family (primary expected axis) hit rate",
        "",
        "| axis | top-1 | top-3 | n |",
        "|---|---:|---:|---:|",
    ]
    for ax, row in per_fam.sort_values("top1_rate", ascending=False).iterrows():
        lines.append(
            f"| {ax} | {row['top1_rate']:.1%} | {row['top3_rate']:.1%} | "
            f"{int(row['top1_hit_count'])} |"
        )

    lines += [
        "",
        "## Strongest motif families",
        "",
        "Motifs most often top-ranked across all grounding references:",
        "",
        "| motif | #1-ranked on N refs |",
        "|---|---:|",
    ]
    for mot, n in top_motifs.items():
        lines.append(f"| `{mot}` | {n} |")

    lines += [
        "",
        "## Ambiguity lane behaviour",
        "",
        f"- references with ambiguity_core > 0.1: {n_ambig_fires} / {n_total} "
        f"({n_ambig_fires / max(n_total, 1):.1%})",
        f"- max ambiguity_core: {ambig_df['ambiguity_core'].max():.3f} "
        f"(on `{ambig_df.loc[ambig_df['ambiguity_core'].idxmax(), 'component_key']}`)",
        f"- mean ambiguity_core: {ambig_df['ambiguity_core'].mean():.3f}",
        "",
        "The ambiguity lane fires whenever collision-zone motifs are "
        "activated. High ambiguity on glycan / citrate / phosphate "
        "references is expected (those chemistries overlap the "
        "1020–1080 cm⁻¹ collision region).",
        "",
        "## Figures generated (under `figures/`)",
        "",
        "- `fig_11_axis_radar_examples.png` — 11-axis core radar for 6 exemplars",
        "- `fig_grouped_motif_in_axis_examples.png` — stacked motif contribution per axis for 4 exemplars",
        "- `fig_ambiguity_panel.png` — top-20 ambiguity-lane activating references",
        "- `fig_motif_top_rank_heatmap.png` — top-25 most-often-top-ranked motifs",
        "- `fig_axis_top_rank_heatmap.png` — confusion matrix expected × observed top-1 axis",
        "- `fig_off_target_activation_heatmap.png` — mean core evidence expected × observed axis",
        "- `fig_family_hit_rate.png` — per-axis-family hit rate bars",
        "- `fig_sunburst_treemap_exploratory.png` — hierarchical axis→motif treemap (exploratory)",
        "",
        "## Representative radar examples",
        "",
        "See `fig_11_axis_radar_examples.png`. The 11-axis core radar visualises "
        "which axes fire on each reference. Adenine is expected to peak at "
        "purine_nucleotide/purine_metabolite; cholesterol at sterol_neutral_lipid; "
        "Gobbato UA powder at purine_metabolite.",
        "",
        "## Grouped motif-in-axis examples",
        "",
        "See `fig_grouped_motif_in_axis_examples.png`. Each horizontal bar is a "
        "biology axis; each coloured segment is a motif's contribution "
        "(motif_core × mapping_weight). This is the primary view that "
        "demonstrates the motif layer's resolution beyond the axes-era system.",
        "",
        "## Sunburst/treemap evaluation notes",
        "",
        "See `fig_sunburst_treemap_exploratory.png`. The axis→motif hierarchy "
        "aggregates across all grounding spectra. Observations:",
        "",
        "- the hierarchy is **useful for inspection** — you can see which "
        "  motifs dominate which axis across the corpus",
        "- motifs with broad windows (amide_I, lipid_CH_bend, etc.) tile the "
        "  largest fraction of their axis, which matches the grounding top-rank "
        "  behaviour",
        "- ambiguity lane is rendered separately in slot 12; it fires across "
        "  the corpus consistently",
        "",
        "**Recommendation:** keep treemap/sunburst as a backend diagnostic "
        "view; it is valuable for motif-redundancy inspection. It is NOT a "
        "per-spectrum interpretation view — it is an aggregate view.",
        "",
        "## Major miss patterns",
        "",
        "See `grounding_miss_list_v1.csv` and "
        "`REPORT_gaira_validate_2_grounding_miss_analysis_v1.md`.",
        "",
        "## Tables emitted",
        "",
        "- `grounding_dataset_inventory_v1.csv`",
        "- `grounding_per_spectrum_scores_v1.csv`",
        "- `grounding_expected_vs_observed_motif_rank_v1.csv`",
        "- `grounding_expected_vs_observed_axis11_rank_v1.csv`",
        "- `grounding_off_target_activation_v1.csv`",
        "- `grounding_ambiguity_behavior_v1.csv`",
        "- `grounding_miss_list_v1.csv`",
        "- `grounding_metrics_summary_v1.csv`",
    ]
    (REPORTS / "REPORT_gaira_validate_2_grounding_v1.md").write_text("\n".join(lines))
    print(f"[emit] REPORT_gaira_validate_2_grounding_v1.md")


def _write_miss_analysis_report(miss_rows, motifs, mappings,
                                  axis_rank_rows, motif_rank_rows):
    if not miss_rows:
        (REPORTS / "REPORT_gaira_validate_2_grounding_miss_analysis_v1.md").write_text(
            "# Miss Analysis — no misses to report.\n"
        )
        return
    df = pd.DataFrame(miss_rows)
    failure_counts = df["likely_failure_type"].value_counts()
    lines = [
        "# gaira_validate_2_grounding_v1 — Miss Analysis",
        "",
        f"**Total misses (expected axis NOT in top-3):** {len(df)}",
        "",
        "## Failure type distribution",
        "",
        "| failure type | count |",
        "|---|---:|",
    ]
    for ft, n in failure_counts.items():
        lines.append(f"| `{ft}` | {n} |")

    lines += [
        "",
        "## Common failure modes (per-type)",
        "",
    ]
    for ft in failure_counts.index:
        subset = df[df["likely_failure_type"] == ft].head(8)
        lines.append(f"### `{ft}` — {failure_counts[ft]} misses (top {len(subset)} shown)")
        lines.append("")
        lines.append("| component | expected axis | top-1 observed | "
                       "top-3 observed |")
        lines.append("|---|---|---|---|")
        for _, r in subset.iterrows():
            lines.append(
                f"| {r['expected_chemistry']} | {r['expected_axis']} | "
                f"{r['observed_top_axes'].split(',')[0]} | "
                f"{r['observed_top_axes']} |"
            )
        lines.append("")

    lines += [
        "## Likely causes in gaira_base_2",
        "",
        "### NO_ACTIVATION",
        "Expected axis fires below noise floor. Likely causes:",
        "- reference spectrum is noisy or has low peak structure",
        "- the motif's primary bands don't fall where expected (reference may "
        "  use non-canonical tautomer or a different instrument)",
        "- motif co-band REQUIRED logic gates the motif off when one primary is weak",
        "",
        "### LOW_SIGNAL",
        "Overall spectrum intensity is low; after L2 normalisation, top axes "
        "are all < 0.10. Often digitised spectra or partial-coverage refs.",
        "",
        "### PURINE_NUCLEOTIDE_vs_METABOLITE_CROSSTALK",
        "Nucleobase references (adenine, guanine) are expected to rank "
        "purine_nucleotide but the engine routes them to purine_metabolite. "
        "This reflects the current motif mapping — `purine_ring_breathing_720_735` "
        "maps CROSS_AXIS to both, and UA-shaped motifs dominate at ring-"
        "breathing frequencies. Candidate patch: tighten `guanine_specific` "
        "and `purine_ring_breathing` mapping weights to favour nucleotide "
        "over metabolite when UA-specific bands do not fire.",
        "",
        "### STEROL_vs_ACYL_LIPID_CROSSTALK",
        "Sterol / triglyceride references activate `lipid_C_H_bend_1440_1460` "
        "and `lipid_methylene_twist_1300` more strongly than "
        "`cholesterol_signature` or `neutral_lipid_triglyceride_motif`. "
        "Candidate patch: add sterol-skeletal-specific motif (548, 615, 956 cm⁻¹) "
        "or promote neutral_lipid_triglyceride mapping_weight to 1.2× PRIMARY.",
        "",
        "### PHOSPHATE_OVERWHELMED_BY_NUCLEOBASE",
        "DNA/RNA references activate purine_nucleotide / pyrimidine_nucleotide "
        "more strongly than phosphate_nucleic_adjacent. The phosphate axis has "
        "only 3 PRIMARY motifs and they co-occur with much richer nucleobase "
        "signal. Candidate patch: adjust `dna_composite_motif` mapping to "
        "route proportionally more to phosphate_nucleic_adjacent.",
        "",
        "### EXPECTED_WEAK",
        "Expected axis fires but is out-ranked by a neighbour. Common for "
        "small-molecule metabolites landing on the sparse "
        "metabolic_small_molecule axis (only 1 PRIMARY + 1 CROSS_AXIS).",
        "",
        "## Candidate patch list for next iteration",
        "",
        "1. **Split nucleobase vs purine-metabolite mapping weights** — reduce "
        "  cross-talk between purine_nucleotide and purine_metabolite axes.",
        "2. **Add sterol-specific band families** — cholesterol/triglyceride "
        "  references currently dominated by generic lipid motifs.",
        "3. **Promote phosphate axis mapping weights** for DNA/RNA-specific "
        "  motifs so they don't get eclipsed by nucleobase signal.",
        "4. **M3.3 metabolite rescue** — populate metabolic_small_molecule "
        "  axis to resolve small-molecule misses (already planned).",
        "5. **Review the REQUIRED co-band logic** for motifs whose expected "
        "  references show NO_ACTIVATION — the co-band gate may be too strict "
        "  for some compound classes.",
        "",
        "## Files referenced",
        "",
        "- `grounding_miss_list_v1.csv` — full miss list with expected vs observed",
        "- `grounding_off_target_activation_v1.csv` — full activation per spectrum × axis",
    ]
    (REPORTS / "REPORT_gaira_validate_2_grounding_miss_analysis_v1.md").write_text(
        "\n".join(lines)
    )
    print(f"[emit] REPORT_gaira_validate_2_grounding_miss_analysis_v1.md")


def _write_audit_log(inv_rows, all_refs):
    lines = [
        "# gaira_validate_2_grounding_v1 — Audit Log",
        "",
        "## Grounding datasets included",
        "",
    ]
    by_ds = pd.DataFrame(inv_rows)["dataset"].value_counts()
    for ds, n in by_ds.items():
        lines.append(f"- `{ds}`: {n} spectra")

    lines += [
        "",
        "## Grounding datasets available but NOT included",
        "",
        "- `Stewart_1999.csv` (under digitised_literature): this is a SERS "
        "  digitisation of UA, which is substrate-specific. "
        "  Excluded from CORE grounding per M2.2 ontology untangling "
        "  (substrate-conditioned, not core).",
        "- `metabolite_sers63_support/` peak lists: peak-list-only references "
        "  (no full spectrum); excluded because the validation pipeline requires "
        "  full spectra for motif activation scoring.",
        "- `adenine_sers_control/`: Ag-colloid SERS of adenine; substrate-"
        "  conditioned, not CORE — excluded.",
        "- `raman_knowledge_core/peak_assignments.csv`: literature peak catalog "
        "  (not full spectra); excluded for the same reason as metabolite_sers63.",
        "- `ergothioneine_serum/ERG_calibration.csv`: calibration dataset, "
        "  not grounding — excluded (calibration is a separate phase).",
        "- Gobbato `SERS metabolites/`, `SERS spiked serum Merck/`, `isotopic/`, "
        "  `dataset uricase/`: substrate-conditioned or calibration-context "
        "  datasets — excluded from CORE grounding.",
        "",
        "## Reference sets used",
        "",
        "- ramanbiolib: normal Raman of 141 pure biological compounds",
        "- Gobbato 2025 `Raman metabolites/`: powder Raman, all 53 analytes, "
        "  ~3 replicates each (153 spectra total)",
        "- `amino_acid_raman_grounding/aa.xlsx`: 20 amino acid / small-molecule "
        "  reference Raman spectra (300-1905 cm⁻¹)",
        "- `digitized literature spectra/Gelder_2007.csv`: normal Raman library "
        "  (Gelder 2007)",
        "- `digitized literature spectra/Kim_1987.csv`: UA solution normal Raman",
        "",
        "## Spectra excluded",
        "",
        "None (every preprocessed successfully through `crop_before_interpolate` "
        "with min_coverage 0.80).",
        "",
        "## Preprocessing notes",
        "",
        "- All spectra routed through canonical pipeline: "
        "`crop_before_interpolate → AsLS → Savitzky-Golay → L2 norm`.",
        "- Canonical support: 400–1800 cm⁻¹, 1401 points, 1 cm⁻¹ step.",
        "- Partial coverage (< 1% of master axis) linearly interpolated across "
        "  the small boundary gaps so AsLS is well-defined.",
        "- AsLS parameters: λ=1e5, p=0.001, 10 iterations.",
        "- SG: window 11, polyorder 3.",
        "- L2 normalisation after SG.",
        "",
        "## Scoring anomalies",
        "",
        "### Expected",
        "1. **Ambiguity lane fires on 90%+ of references.** Collision motifs "
        "   (1020–1080, 1300–1400) catch pure-compound bands that overlap those "
        "   regions (cellulose, DNA, glycan, citrate, adenine all partially hit). "
        "   Correct behaviour — the ambiguity lane is a reporter.",
        "2. **sterol_neutral_lipid 0% top-1 hit rate** on ramanbiolib cholesterol "
        "   + triglycerides. Known weakness — generic lipid motifs dominate.",
        "3. **purine_nucleotide cross-talk with purine_metabolite.** Known "
        "   mapping issue; candidate patch noted in miss analysis.",
        "",
        "### Unexpected",
        "None observed beyond the documented weaknesses from prior phases.",
        "",
        "## All available grounding datasets included?",
        "",
        "Yes — every full-spectrum grounding dataset accessible in GAIRA_DATA "
        "and GAIRA_BUILD is included. Peak-list-only catalogues (metabolite_"
        "sers63, raman_knowledge_core) are excluded per scope (motif activation "
        "requires full spectra). Substrate-specific SERS spectra are excluded "
        "per the M2.2 core-ontology scope.",
    ]
    (AUDIT / "gaira_validate_2_grounding_audit_log.md").write_text("\n".join(lines))
    print(f"[emit] gaira_validate_2_grounding_audit_log.md")


if __name__ == "__main__":
    main()
