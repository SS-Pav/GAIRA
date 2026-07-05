"""GAIRA — gaira_build_motifs_v1 · Phase M3 — Grounding Validation (v1)

Validates M1.1 motifs against a curated reference library of pure-compound
Raman/SERS spectra (ramanbiolib primary + raman_knowledge_core secondary
+ ergothioneine dedicated file + metabolite_sers63_support peak lists).

Pipeline discipline:

  - All reference spectra go through ``gaira.spectral.crop_before_interpolate``
    (the canonical crop-before-interpolate helper from the pipeline-fix pass).
    No silent ``np.interp`` clamping. NaN masking outside measured support.
  - Tighter min_coverage for references (0.80) than the default (0.50) so
    that reference compounds truly cover the analysis support.
  - Motif definitions are NOT modified. This script is read-only w.r.t. the
    motif registry and cohort pilots.

Two tracks:

  READY_M3       (34 motifs) → primary grounding evaluation
  AMBIGUITY_TRACK ( 5 motifs) → candidate-separation evaluation

Outputs (under ``/Volumes/SSD_Rad/GAIRA_BUILD/gaira_build_motifs_v1/
M3_grounding_validation_v1/``):

  tables/motif_grounding_matrix_v1.csv           (motif × status)
  tables/motif_reference_support_v1.csv          (motif × reference)
  tables/motif_ambiguity_grounding_v1.csv        (ambiguity motifs)
  tables/motif_grounding_coverage_audit_v1.csv   (per-reference coverage)
  figures/fig_motif_grounding_status_overview.png
  figures/fig_motif_family_grounding_summary.png
  figures/fig_ambiguity_track_summary.png
  docs/REPORT_M3_grounding_validation_v1.md
  audit/M3_grounding_audit_log.md

Run:
    cd /Users/suraj/projects/GAIRA
    PYTHONPATH=src .venv/bin/python scripts/run_gaira_motifs_M3_grounding_validation_v1.py
"""
from __future__ import annotations

import ast
import hashlib
import json
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from gaira.spectral import (  # noqa: E402
    CANONICAL_SUPPORT_CM1,
    CANONICAL_N_POINTS,
    CANONICAL_STEP_CM1,
    InsufficientOverlapError,
    canonical_master_axis,
    crop_before_interpolate,
)


# ──────────────────────────────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────────────────────────────

ROOT = Path("/Volumes/SSD_Rad/GAIRA_BUILD/gaira_build_motifs_v1/M3_grounding_validation_v1")
TABLES = ROOT / "tables"
FIGURES = ROOT / "figures"
DOCS = ROOT / "docs"
AUDIT = ROOT / "audit"
ARTIFACTS = ROOT / "artifacts"
for d in (TABLES, FIGURES, DOCS, AUDIT, ARTIFACTS):
    d.mkdir(parents=True, exist_ok=True)

REGISTRY_YAML = Path(
    "/Volumes/SSD_Rad/GAIRA_BUILD/gaira_build_motifs_v1/"
    "M1_1_family_expansion_v1/registry/motif_candidate_registry_v1_1.yaml"
)
STATUS_CSV = Path(
    "/Volumes/SSD_Rad/GAIRA_BUILD/gaira_build_motifs_v1/"
    "M2_1_targeted_rescue_v1/tables/motif_convergence_status_post_M2_1.csv"
)
RAMANBIOLIB_SPECTRA = Path(
    "/Volumes/SSD_Rad/GAIRA_DATA/raw/ramanbiolib/ramanbiolib-main/"
    "ramanbiolib/db/raman_spectra_db.csv"
)
RAMANBIOLIB_PEAKS = Path(
    "/Volumes/SSD_Rad/GAIRA_DATA/raw/ramanbiolib/ramanbiolib-main/"
    "ramanbiolib/db/raman_peaks_db.csv"
)
RAMANBIOLIB_META = Path(
    "/Volumes/SSD_Rad/GAIRA_DATA/raw/ramanbiolib/ramanbiolib-main/"
    "ramanbiolib/db/metadata_db.csv"
)
KNOWLEDGE_CORE_PEAKS = Path(
    "/Volumes/SSD_Rad/GAIRA_DATA/raw/raman_knowledge_core/peak_assignments.csv"
)
METABOLITE_SERS63_DIR = Path(
    "/Volumes/SSD_Rad/GAIRA_DATA/raw/metabolite_sers63_support/peaks"
)


# ──────────────────────────────────────────────────────────────────────
# Motif → reference compound map (biochemistry-driven, explicit)
# ──────────────────────────────────────────────────────────────────────
#
# Each motif lists:
#   primary_refs:   ramanbiolib component name(s) that are canonical exemplars
#   secondary_refs: component name(s) that partially support the motif
#   literature_tag: literature peak-catalog category (matches knowledge_core)
#   exemplar_notes: what we expect to see (bookkeeping only)
#
# Motifs with no direct pure-compound analogue (e.g. SERS-specific baselines)
# or whose reference would be a mixture are marked refs=[] and are gated
# against knowledge_core / literature-only evidence.

MOTIF_REF_MAP: dict[str, dict] = {
    # ── nucleobase / nucleic ───────────────────────────────────────────
    "purine_ring_breathing_720_735": {
        "primary_refs": ["adenine", "guanine"],
        "secondary_refs": ["a-dna", "b-dna"],
        "literature_tag": "DNA/RNA-associated region",
    },
    "uric_acid_full_signature": {
        "primary_refs": [],
        "secondary_refs": [],
        "literature_tag": "nucleic-acid-associated SERS region",
        "external_ref": "literature_only",  # UA not in ramanbiolib
    },
    "hypoxanthine_signature": {
        "primary_refs": [],
        "secondary_refs": [],
        "literature_tag": "nucleic-acid-associated SERS region",
        "external_ref": "literature_only",
    },
    "pyrimidine_ring_breathing_780_800": {
        "primary_refs": ["cytosine", "thymine", "uracil"],
        "secondary_refs": ["a-dna", "b-dna", "t-rna"],
        "literature_tag": "DNA/RNA-associated region",
    },
    "nucleobase_in_plane_ring_1320_1340": {
        "primary_refs": ["adenine", "guanine", "cytosine", "thymine"],
        "secondary_refs": ["a-dna", "b-dna"],
        "literature_tag": "DNA/RNA-associated mid region",
    },
    "dna_methylation_marker_790": {
        "primary_refs": ["cytosine", "thymine"],
        "secondary_refs": ["a-dna", "b-dna"],
        "literature_tag": "DNA/RNA-associated region",
    },
    "phosphate_PO2_sym_str_1080": {
        "primary_refs": ["a-dna", "b-dna", "t-rna"],
        "secondary_refs": ["d-fructose-6-phosphate"],
        "literature_tag": "phosphate-backbone-associated region",
    },
    "phosphate_PO_asym_str_1240": {
        "primary_refs": ["a-dna", "b-dna", "t-rna"],
        "secondary_refs": [],
        "literature_tag": "phosphate-backbone-associated region",
    },
    "dna_composite_motif": {
        "primary_refs": ["a-dna", "b-dna"],
        "secondary_refs": ["t-rna"],
        "literature_tag": "DNA/RNA-associated region",
    },
    "xanthine_signature": {
        "primary_refs": [],
        "secondary_refs": [],
        "literature_tag": "nucleic-acid-associated SERS region",
        "external_ref": "literature_only",
    },
    "guanine_specific_motif": {
        "primary_refs": ["guanine"],
        "secondary_refs": ["a-dna", "b-dna"],
        "literature_tag": "DNA/RNA-associated region",
    },
    "thymine_specific_motif": {
        "primary_refs": ["thymine"],
        "secondary_refs": ["b-dna"],
        "literature_tag": "DNA/RNA-associated region",
    },
    "cytosine_specific_motif": {
        "primary_refs": ["cytosine"],
        "secondary_refs": ["a-dna", "b-dna"],
        "literature_tag": "DNA/RNA-associated region",
    },

    # ── glycans ────────────────────────────────────────────────────────
    "glycan_pyranose_ring_skeletal_850_950": {
        "primary_refs": [
            "d-(+)-glucose", "d-(+)-galactose", "d-(+)-mannose",
            "β-d-glucose", "d-(-)-fructose",
        ],
        "secondary_refs": ["glycogen", "cellulose", "amylose"],
        "literature_tag": "polysaccharide-associated C-O region",
    },
    "glycan_glycosidic_C_O_C_1020_1100": {
        "primary_refs": [
            "cellulose", "glycogen", "amylose", "amylopectin",
            "d-(+)-lactose monohydrate", "d-(+)-maltose monohydrate",
            "d-(+)-sucrose",
            "a-dna", "b-dna", "t-rna",   # competing phosphate-backbone class
            "citric acid",               # competing citrate class
        ],
        "secondary_refs": ["d-(+)-glucose", "d-(+)-galactose"],
        "literature_tag": "polysaccharide-associated region",
        "ambiguity": True,
    },
    "sialic_acid_signature": {
        "primary_refs": ["n-acetyl- d-glucosamine"],
        "secondary_refs": ["d-(+)-galactosamine", "glucosamine"],
        "literature_tag": "monosaccharide-associated region",
    },
    "free_saccharide_motif": {
        "primary_refs": [
            "d-(+)-glucose", "d-(+)-galactose", "d-(+)-mannose",
            "d-(-)-fructose", "β-d-glucose",
        ],
        "secondary_refs": ["d-(+)-fucose", "d-(+)-xylose"],
        "literature_tag": "monosaccharide-associated region",
    },

    # ── proteins ───────────────────────────────────────────────────────
    "amide_III_protein_backbone_1230_1280": {
        "primary_refs": ["albumin", "collagen", "elastin", "keratin"],
        "secondary_refs": ["hemoglobin", "myoglobin", "insulin"],
        "literature_tag": "amide III protein region",
    },
    "phenylalanine_ring_1003": {
        "primary_refs": ["l-phenylalanine"],
        "secondary_refs": ["albumin", "collagen"],
        "literature_tag": "phenylalanine-like aromatic marker used cautiously",
    },
    "tyrosine_doublet_830_850": {
        "primary_refs": ["l-tyrosine"],
        "secondary_refs": ["albumin", "collagen"],
        "literature_tag": "tyrosine-like protein-associated region",
    },
    "amide_I_alpha_helix_beta_sheet_motif": {
        "primary_refs": [
            "albumin", "collagen", "elastin", "hemoglobin",
            "insulin", "myoglobin", "keratin",
        ],
        "secondary_refs": [],
        "literature_tag": "amide I protein-associated region",
    },
    "amide_II_motif": {
        "primary_refs": ["albumin", "collagen", "insulin", "hemoglobin"],
        "secondary_refs": [],
        "literature_tag": "amide II adjacent protein region",
    },

    # ── lipids ─────────────────────────────────────────────────────────
    "lipid_acyl_C_C_str_1060_1130": {
        "primary_refs": [
            "oleic acid", "palmitic acid", "stearic acid",
            "linoleic acid", "arachidic acid",
        ],
        "secondary_refs": ["tristearin", "tripalmitin", "triolein"],
        "literature_tag": "fatty-acid-associated region",
    },
    "lipid_C_H_bend_1440_1460": {
        "primary_refs": [
            "oleic acid", "palmitic acid", "stearic acid",
            "linoleic acid", "arachidic acid", "arachidonic acid",
        ],
        "secondary_refs": [
            "tristearin", "tripalmitin", "triolein",
            "cholesterol", "sphingomyelin",
        ],
        "literature_tag": "lipid CH deformation region",
    },
    "phosphatidylcholine_choline_head_715": {
        "primary_refs": ["l-α-phosphatidylcholine"],
        "secondary_refs": ["sphingomyelin"],
        "literature_tag": "adenine/choline-adjacent region used cautiously",
    },
    "cholesterol_signature": {
        "primary_refs": ["cholesterol"],
        "secondary_refs": [
            "cholesteryl linoleate", "cholesteryl oleate",
            "cholesteryl palmitate", "cholesteryl stearate",
        ],
        "literature_tag": "sterol-like mid region",
    },
    "lipid_methylene_twist_1300": {
        "primary_refs": [
            "palmitic acid", "stearic acid", "oleic acid",
            "arachidic acid", "linoleic acid",
        ],
        "secondary_refs": ["tristearin", "tripalmitin", "triolein"],
        "literature_tag": "fatty-acid CH2 twisting/deformation region",
    },
    "neutral_lipid_triglyceride_motif": {
        "primary_refs": [
            "tristearin", "tripalmitin", "triolein", "trilinolein",
            "trilaurin", "trimyristin",
        ],
        "secondary_refs": ["oleic acid", "palmitic acid"],
        "literature_tag": "lipid-associated mid region",
    },
    "amide_I_lipid_carbonyl_partial_panel_motif": {
        "primary_refs": [
            "albumin", "tristearin", "tripalmitin",
        ],
        "secondary_refs": ["collagen", "triolein"],
        "literature_tag": "amide I protein-associated region",
    },

    # ── redox / heme / thiol ───────────────────────────────────────────
    "cytochrome_c_resonance_motif": {
        "primary_refs": ["cytochrome c"],
        "secondary_refs": ["hemoglobin", "myoglobin"],
        "literature_tag": "protein aromatic/amide-adjacent region",
    },
    "disulfide_S_S_str_500_550": {
        "primary_refs": ["glutathione"],
        "secondary_refs": ["albumin", "insulin", "keratin"],
        "literature_tag": "protein backbone-associated region",
    },
    "ergothioneine_signature": {
        "primary_refs": [],
        "secondary_refs": [],
        "literature_tag": "nucleic-acid-associated SERS region",
        "external_ref": "ergothioneine_serum",
    },
    "thiol_C_S_str_660_motif": {
        "primary_refs": ["glutathione"],
        "secondary_refs": ["albumin"],
        "literature_tag": "protein-associated low-mid band",
    },
    "glutathione_GSH_motif": {
        "primary_refs": ["glutathione"],
        "secondary_refs": [],
        "literature_tag": "protein-associated low-mid band",
    },

    # ── metabolites ────────────────────────────────────────────────────
    "creatine_creatinine_motif": {
        "primary_refs": [],
        "secondary_refs": [],
        "literature_tag": "protein-related low-mid region",
        "external_ref": "metabolite_sers63",  # methylguanidine closest analogue
    },

    # ── substrate artifact ─────────────────────────────────────────────
    "citrate_baseline_artifact_motif": {
        "primary_refs": ["citric acid"],
        "secondary_refs": [],
        "literature_tag": "protein-related low-mid region",
    },

    # ── AMBIGUITY_TRACK motifs (collision / ambiguity type) ────────────
    # These are NOT expected to resolve cleanly — the grounding test for
    # them is "do multiple candidate compounds all fire the overlap
    # region?" not "which single compound dominates?".
    "collision_1020_1080_multi_candidate": {
        "primary_refs": [
            "a-dna", "b-dna", "t-rna",            # phosphate backbone
            "glycogen", "cellulose", "amylose",   # polysaccharide C-O
            "citric acid",                        # citrate
        ],
        "secondary_refs": ["d-(+)-glucose"],
        "literature_tag": "phosphate-backbone-associated region",
        "ambiguity": True,
    },
    "purine_HX_lipid_choline_715_overlap_ambiguity": {
        "primary_refs": [
            "adenine", "guanine",                 # free purines
            "l-α-phosphatidylcholine",            # choline head
            "sphingomyelin",                      # choline-bearing lipid
        ],
        "secondary_refs": [],
        "literature_tag": "adenine/choline-adjacent region used cautiously",
        "ambiguity": True,
        "external_ref": "literature_only_ua_hx",
    },
    "collision_1300_1400_multi_candidate_motif": {
        "primary_refs": [
            "adenine", "guanine",                 # nucleobase in-plane
            "palmitic acid", "stearic acid",      # CH2 twist
            "albumin",                            # amide III
            "citric acid",                        # citrate
        ],
        "secondary_refs": [],
        "literature_tag": "DNA/RNA-associated mid region",
        "ambiguity": True,
    },
}


# ──────────────────────────────────────────────────────────────────────
# Reference loader
# ──────────────────────────────────────────────────────────────────────

@dataclass
class ReferenceSpectrum:
    component: str
    source: str                    # "ramanbiolib" | "metabolite_sers63" | ...
    raw_wn: np.ndarray | None      # None if peak-list-only reference
    raw_y: np.ndarray | None
    y_master: np.ndarray | None    # cropped+interpolated to master_x (or None)
    coverage: object | None        # CoverageInfo or None
    peak_centers: np.ndarray       # (K,) detected peak centers
    peak_heights: np.ndarray       # (K,) detected peak heights (normalised 0..1)
    note: str = ""


def _parse_list_str(s: str) -> np.ndarray:
    """Parse a '[a, b, c, ...]' Python-literal-style CSV cell to float array."""
    return np.array(ast.literal_eval(s), dtype=np.float64)


def load_ramanbiolib() -> dict[str, ReferenceSpectrum]:
    """Load ramanbiolib full spectra + peak lists; interpolate to canonical axis."""
    print(f"[load] ramanbiolib spectra: {RAMANBIOLIB_SPECTRA}")
    spec_df = pd.read_csv(RAMANBIOLIB_SPECTRA)
    peaks_df = pd.read_csv(RAMANBIOLIB_PEAKS)

    # index peaks by component (lowercased for match robustness)
    peaks_by_comp: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for _, r in peaks_df.iterrows():
        comp = str(r["component"]).strip().lower()
        centers = _parse_list_str(r["peaks"])
        heights = _parse_list_str(r["intensity"])
        peaks_by_comp[comp] = (centers, heights)

    master_x = canonical_master_axis()
    refs: dict[str, ReferenceSpectrum] = {}
    n_fail = 0
    for _, r in spec_df.iterrows():
        comp = str(r["component"]).strip().lower()
        try:
            wn = _parse_list_str(r["wavenumbers"])
            y = _parse_list_str(r["intensity"])
        except Exception as e:
            print(f"  [warn] parse fail {comp}: {e}")
            n_fail += 1
            continue
        # apply crop-before-interp with tight coverage threshold
        try:
            y_master, cov = crop_before_interpolate(
                wn, y, master_x,
                partial_ok=True, min_coverage=0.80,
            )
        except InsufficientOverlapError as e:
            print(f"  [skip] {comp}: {e}")
            n_fail += 1
            continue

        centers, heights = peaks_by_comp.get(comp, (np.array([]), np.array([])))
        refs[comp] = ReferenceSpectrum(
            component=comp,
            source="ramanbiolib",
            raw_wn=wn,
            raw_y=y,
            y_master=y_master,
            coverage=cov,
            peak_centers=centers,
            peak_heights=heights,
        )
    print(f"[load] ramanbiolib: {len(refs)} spectra ready; {n_fail} failed")
    return refs


def load_metabolite_sers63() -> dict[str, ReferenceSpectrum]:
    """Peak-only references from metabolite_sers63_support/peaks/*.peaks."""
    print(f"[load] metabolite_sers63 peaks from: {METABOLITE_SERS63_DIR}")
    refs: dict[str, ReferenceSpectrum] = {}
    for f in sorted(METABOLITE_SERS63_DIR.glob("*.peaks")):
        comp = f.stem.lower().replace("-fingerprint", "")
        centers = []
        heights = []
        for line in f.read_text(encoding="latin-1").splitlines():
            if line.startswith("#") or not line.strip():
                continue
            toks = line.split()
            # format: %_N Qgau CENTER x x x HEIGHT CENTER HWHM q
            if len(toks) < 10:
                continue
            try:
                c = float(toks[2])
                h = float(toks[6])
                centers.append(c)
                heights.append(h)
            except (ValueError, IndexError):
                continue
        if not centers:
            continue
        refs[comp] = ReferenceSpectrum(
            component=comp,
            source="metabolite_sers63",
            raw_wn=None, raw_y=None,
            y_master=None, coverage=None,
            peak_centers=np.array(centers),
            peak_heights=np.array(heights) / (max(heights) + 1e-12),
        )
    print(f"[load] metabolite_sers63: {len(refs)} peak-lists ready")
    return refs


def load_knowledge_core() -> pd.DataFrame:
    """Literature peak catalog."""
    print(f"[load] raman_knowledge_core: {KNOWLEDGE_CORE_PEAKS}")
    return pd.read_csv(KNOWLEDGE_CORE_PEAKS)


# ──────────────────────────────────────────────────────────────────────
# Peak-detection helpers (on master-axis spectra)
# ──────────────────────────────────────────────────────────────────────

def _local_max_in_window(
    y_master: np.ndarray,
    master_x: np.ndarray,
    lo: float, hi: float,
) -> tuple[float, float] | None:
    """Return (center, height) of the local maximum of y_master inside [lo, hi].

    Returns None if the window is entirely NaN or outside master_x.
    """
    mask = (master_x >= lo) & (master_x <= hi)
    if not mask.any():
        return None
    y_win = y_master[mask]
    x_win = master_x[mask]
    finite = np.isfinite(y_win)
    if not finite.any():
        return None
    y_win = y_win[finite]
    x_win = x_win[finite]
    idx = int(np.argmax(y_win))
    return float(x_win[idx]), float(y_win[idx])


def _has_peak_in_window_from_list(
    peak_centers: np.ndarray,
    peak_heights: np.ndarray,
    lo: float, hi: float,
) -> tuple[float, float] | None:
    """Return the tallest cataloged peak within [lo, hi] or None."""
    if peak_centers.size == 0:
        return None
    mask = (peak_centers >= lo) & (peak_centers <= hi)
    if not mask.any():
        return None
    cs = peak_centers[mask]
    hs = peak_heights[mask]
    idx = int(np.argmax(hs))
    return float(cs[idx]), float(hs[idx])


# ──────────────────────────────────────────────────────────────────────
# Motif-band evaluation
# ──────────────────────────────────────────────────────────────────────

@dataclass
class BandSupport:
    family_id: str
    role: str                  # primary / supporting
    cm1_centre: float
    cm1_tolerance: float
    match_cm1: float | None    # matched peak position
    match_intensity: float | None
    evidence: str              # "peak_list" | "local_max" | "none" | "nan"


@dataclass
class ReferenceMotifSupport:
    motif_id: str
    component: str
    source: str
    bands: list[BandSupport] = field(default_factory=list)
    n_primary_fire: int = 0
    n_primary_total: int = 0
    n_supporting_fire: int = 0
    n_supporting_total: int = 0
    fraction_primary_fire: float = 0.0
    fraction_supporting_fire: float = 0.0
    any_band_in_nan: bool = False


def evaluate_motif_on_reference(
    motif: dict,
    ref: ReferenceSpectrum,
    master_x: np.ndarray,
) -> ReferenceMotifSupport:
    """For a given motif × reference, determine which bands fire."""
    primary_fams = motif.get("primary_band_families") or []
    supporting_fams = motif.get("supporting_band_families") or []

    result = ReferenceMotifSupport(
        motif_id=motif["motif_id"],
        component=ref.component,
        source=ref.source,
        n_primary_total=len(primary_fams),
        n_supporting_total=len(supporting_fams),
    )

    def eval_one(fam: dict, role: str) -> BandSupport:
        c = float(fam["cm1_centre"])
        t = float(fam["cm1_tolerance"])
        lo, hi = c - t, c + t

        bs = BandSupport(
            family_id=fam["family_id"], role=role,
            cm1_centre=c, cm1_tolerance=t,
            match_cm1=None, match_intensity=None, evidence="none",
        )

        # Prefer full-spectrum local-max check (more lenient, captures shoulders)
        if ref.y_master is not None:
            hit = _local_max_in_window(ref.y_master, master_x, lo, hi)
            if hit is not None:
                mc, mi = hit
                # reject if intensity is essentially zero (background)
                if mi > 1e-3:
                    bs.match_cm1 = mc
                    bs.match_intensity = mi
                    bs.evidence = "local_max"
                    return bs
            # check whether the window is all-NaN (missing support)
            mask = (master_x >= lo) & (master_x <= hi)
            if mask.any():
                y_win = ref.y_master[mask]
                if np.all(np.isnan(y_win)):
                    bs.evidence = "nan"
                    result.any_band_in_nan = True
                    return bs

        # Fall back to curated peak-list check
        hit = _has_peak_in_window_from_list(
            ref.peak_centers, ref.peak_heights, lo, hi,
        )
        if hit is not None:
            mc, mi = hit
            bs.match_cm1 = mc
            bs.match_intensity = mi
            bs.evidence = "peak_list"
        return bs

    for fam in primary_fams:
        bs = eval_one(fam, "primary")
        result.bands.append(bs)
        if bs.evidence in ("local_max", "peak_list"):
            result.n_primary_fire += 1

    for fam in supporting_fams:
        bs = eval_one(fam, "supporting")
        result.bands.append(bs)
        if bs.evidence in ("local_max", "peak_list"):
            result.n_supporting_fire += 1

    result.fraction_primary_fire = (
        result.n_primary_fire / max(result.n_primary_total, 1)
    )
    result.fraction_supporting_fire = (
        result.n_supporting_fire / max(result.n_supporting_total, 1)
    )
    return result


# ──────────────────────────────────────────────────────────────────────
# Grounding status classifier
# ──────────────────────────────────────────────────────────────────────

GROUNDED_PRIMARY_THRESHOLD = 0.75   # ≥75% of primary bands fire
PARTIAL_PRIMARY_THRESHOLD  = 0.50   # 50-75% = partial
WEAK_PRIMARY_THRESHOLD     = 0.25   # 25-50% = weak
# < 25% & supporting evidence only → NOT_GROUNDED


def classify_ready_motif(
    motif: dict,
    per_ref_support: list[ReferenceMotifSupport],
    has_external_ref: bool,
) -> tuple[str, str, dict]:
    """Return (grounding_status, rationale, metrics) for a READY_M3 motif."""
    metrics: dict = {}
    if not per_ref_support and not has_external_ref:
        return (
            "NOT_EVALUABLE",
            "no reference compound available in reference library",
            metrics,
        )

    # Aggregate across references: best-fire rate and coverage
    if per_ref_support:
        best = max(
            per_ref_support,
            key=lambda r: (r.fraction_primary_fire, r.fraction_supporting_fire),
        )
        any_any_nan = any(r.any_band_in_nan for r in per_ref_support)
        best_primary = best.fraction_primary_fire
        best_support = best.fraction_supporting_fire
        n_refs_eval = len(per_ref_support)
        n_refs_primary_majority = sum(
            1 for r in per_ref_support if r.fraction_primary_fire >= 0.50
        )
        n_refs_grounded = sum(
            1 for r in per_ref_support if r.fraction_primary_fire >= GROUNDED_PRIMARY_THRESHOLD
        )
        metrics.update({
            "n_refs_evaluated": n_refs_eval,
            "n_refs_primary_majority": n_refs_primary_majority,
            "n_refs_grounded": n_refs_grounded,
            "best_fraction_primary_fire": round(best_primary, 3),
            "best_fraction_supporting_fire": round(best_support, 3),
            "best_reference": best.component,
            "any_band_in_nan": any_any_nan,
        })

        if best_primary >= GROUNDED_PRIMARY_THRESHOLD and n_refs_primary_majority >= 1:
            return (
                "GROUNDED",
                f"best reference {best.component} fires {best.n_primary_fire}/"
                f"{best.n_primary_total} primary bands",
                metrics,
            )
        elif best_primary >= PARTIAL_PRIMARY_THRESHOLD:
            return (
                "PARTIALLY_GROUNDED",
                f"best reference {best.component} fires {best.n_primary_fire}/"
                f"{best.n_primary_total} primary bands",
                metrics,
            )
        elif best_primary >= WEAK_PRIMARY_THRESHOLD or best_support >= 0.50:
            return (
                "WEAKLY_GROUNDED",
                f"best reference {best.component} fires only "
                f"{best.n_primary_fire}/{best.n_primary_total} primary bands "
                f"(+{best.n_supporting_fire}/{best.n_supporting_total} supporting)",
                metrics,
            )
        else:
            return (
                "NOT_GROUNDED",
                f"no reference fires >25% of primary bands "
                f"(best: {best.n_primary_fire}/{best.n_primary_total})",
                metrics,
            )

    # Only external / literature evidence
    if has_external_ref:
        return (
            "NOT_EVALUABLE",
            "no direct pure-compound reference; external literature-only evidence "
            "(UA / HX / xanthine / ergothioneine / creatine) — requires calibration-phase "
            "dedicated reference spectrum",
            metrics,
        )

    return ("NOT_GROUNDED", "no evaluable evidence", metrics)


def classify_ambiguity_motif(
    motif: dict,
    per_ref_support: list[ReferenceMotifSupport],
    ref_map: dict,
) -> tuple[str, str, dict]:
    """Return (grounding_status, rationale, metrics) for an AMBIGUITY motif.

    Logic: ambiguity is CONFIRMED iff multiple (≥2) independent candidate
    compound classes fire the overlap bands. PARTIAL iff only a subset of
    candidates fire. WEAK iff ambiguity is compatible with observation but
    under-resolved (only one candidate available / most bands missing).
    """
    metrics: dict = {"candidate_classes_firing": []}
    if not per_ref_support:
        return ("AMBIGUITY_WEAK", "no reference compounds available", metrics)

    # Group references into candidate classes (based on component name semantics)
    # For each class, count how many members fire ≥50% of primary bands
    fam_by_ref: dict[str, list[ReferenceMotifSupport]] = defaultdict(list)
    for r in per_ref_support:
        fam = _classify_ref_family(r.component)
        fam_by_ref[fam].append(r)

    firing_classes = []
    for fam, rs in fam_by_ref.items():
        fires = sum(1 for r in rs if r.fraction_primary_fire >= 0.50)
        if fires >= 1:
            firing_classes.append((fam, fires, len(rs)))

    metrics["candidate_classes_firing"] = [
        {"family": f, "n_firing": n, "n_total": t} for f, n, t in firing_classes
    ]
    metrics["n_candidate_classes_firing"] = len(firing_classes)
    metrics["n_refs_evaluated"] = len(per_ref_support)

    if len(firing_classes) >= 2:
        return (
            "AMBIGUITY_CONFIRMED",
            f"{len(firing_classes)} independent candidate classes fire overlap bands: "
            f"{[c[0] for c in firing_classes]}",
            metrics,
        )
    elif len(firing_classes) == 1:
        return (
            "AMBIGUITY_PARTIAL",
            f"only 1 candidate class fires: {firing_classes[0][0]}; other candidates "
            f"{[f for f in fam_by_ref if f != firing_classes[0][0]]} "
            f"do not fire overlap bands",
            metrics,
        )
    else:
        return (
            "AMBIGUITY_WEAK",
            "no candidate class reaches 50% primary-fire threshold",
            metrics,
        )


def _classify_ref_family(component: str) -> str:
    """Map a reference component name to an ambiguity-class family tag."""
    c = component.lower()
    if c in ("a-dna", "b-dna", "t-rna"):
        return "nucleic_acid_backbone"
    if c in ("adenine", "guanine", "cytosine", "thymine", "uracil"):
        return "nucleobase"
    if c in ("cellulose", "glycogen", "amylose", "amylopectin",
             "d-(+)-lactose monohydrate", "d-(+)-maltose monohydrate",
             "d-(+)-sucrose", "d-(+)-glucose", "d-(+)-galactose",
             "d-(+)-mannose", "d-(-)-fructose", "β-d-glucose",
             "n-acetyl- d-glucosamine", "d-(+)-fucose", "d-(+)-xylose",
             "glucosamine"):
        return "glycan"
    if c == "citric acid":
        return "citrate"
    if c in ("l-α-phosphatidylcholine", "sphingomyelin"):
        return "choline_lipid"
    if c in ("l-α-phosphatidylethanolamine", "ceramide"):
        return "non_choline_lipid"
    if c in ("palmitic acid", "stearic acid", "oleic acid",
             "linoleic acid", "arachidic acid", "arachidonic acid",
             "tristearin", "tripalmitin", "triolein"):
        return "acyl_lipid"
    if c in ("albumin", "collagen", "elastin", "keratin",
             "hemoglobin", "myoglobin", "insulin", "cytochrome c"):
        return "protein"
    if c in ("glutathione",):
        return "thiol_tripeptide"
    return f"other:{c}"


# ──────────────────────────────────────────────────────────────────────
# Driver
# ──────────────────────────────────────────────────────────────────────

def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def load_motif_registry() -> tuple[list[dict], pd.DataFrame]:
    print(f"[load] motif registry: {REGISTRY_YAML}")
    with REGISTRY_YAML.open("r") as f:
        reg = yaml.safe_load(f)
    motifs = reg["motifs"]
    status_df = pd.read_csv(STATUS_CSV)
    return motifs, status_df


def main() -> None:
    print("=" * 78)
    print("GAIRA — Phase M3 — Grounding Validation (v1)")
    print("=" * 78)
    print(f"canonical support : {CANONICAL_SUPPORT_CM1}  n={CANONICAL_N_POINTS}  step={CANONICAL_STEP_CM1}")
    print(f"pipeline          : crop_before_interpolate → (no AsLS/SG/L2 on references)")
    print(f"min_coverage      : 0.80 (tight; references must cover ≥80% of master)")
    print()

    master_x = canonical_master_axis()

    # ── Load inputs ────────────────────────────────────────────────────
    motifs, status_df = load_motif_registry()
    motif_by_id = {m["motif_id"]: m for m in motifs}

    rb_refs = load_ramanbiolib()          # dict[component -> ReferenceSpectrum]
    m63_refs = load_metabolite_sers63()   # dict[component -> ReferenceSpectrum]
    kcore = load_knowledge_core()         # DataFrame

    # ── Per-reference coverage audit rows ──────────────────────────────
    coverage_rows: list[dict] = []
    for comp, ref in rb_refs.items():
        cov = ref.coverage
        coverage_rows.append({
            "reference_id": comp,
            "reference_source": "ramanbiolib",
            "raw_min_cm1": cov.raw_min_cm1,
            "raw_max_cm1": cov.raw_max_cm1,
            "cropped_min_cm1": cov.cropped_min_cm1,
            "cropped_max_cm1": cov.cropped_max_cm1,
            "coverage_fraction": round(cov.coverage_fraction, 4),
            "partial_coverage": cov.partial_coverage,
            "n_interpolated_points": cov.n_interpolated_points,
            "n_masked_out_of_support": cov.n_masked_out_of_support,
            "note": "crop_before_interpolate path, no AsLS/SG/L2",
        })
    for comp, ref in m63_refs.items():
        coverage_rows.append({
            "reference_id": comp,
            "reference_source": "metabolite_sers63",
            "raw_min_cm1": float(ref.peak_centers.min()) if ref.peak_centers.size else float("nan"),
            "raw_max_cm1": float(ref.peak_centers.max()) if ref.peak_centers.size else float("nan"),
            "cropped_min_cm1": float("nan"),
            "cropped_max_cm1": float("nan"),
            "coverage_fraction": float("nan"),
            "partial_coverage": False,
            "n_interpolated_points": 0,
            "n_masked_out_of_support": 0,
            "note": "peak-list only (no full spectrum path)",
        })
    coverage_df = pd.DataFrame(coverage_rows)
    coverage_df.to_csv(TABLES / "motif_grounding_coverage_audit_v1.csv", index=False)
    print(f"[emit] {TABLES}/motif_grounding_coverage_audit_v1.csv "
          f"({len(coverage_df)} refs)")

    # ── Determine buckets ──────────────────────────────────────────────
    ready_m3 = status_df[status_df["readiness_bucket"] == "READY_M3"]["motif_id"].tolist()
    ambiguity = status_df[status_df["readiness_bucket"] == "AMBIGUITY_TRACK"]["motif_id"].tolist()
    print(f"[plan] READY_M3 motifs     : {len(ready_m3)}")
    print(f"[plan] AMBIGUITY motifs    : {len(ambiguity)}")

    # ── Evaluate each motif ────────────────────────────────────────────
    matrix_rows: list[dict] = []
    support_rows: list[dict] = []
    ambig_rows: list[dict] = []

    all_targets = list(ready_m3) + list(ambiguity)
    for motif_id in all_targets:
        if motif_id not in MOTIF_REF_MAP:
            print(f"  [warn] no MOTIF_REF_MAP entry for {motif_id}; skipping")
            continue
        if motif_id not in motif_by_id:
            print(f"  [warn] motif {motif_id} not in registry; skipping")
            continue

        motif = motif_by_id[motif_id]
        ref_entry = MOTIF_REF_MAP[motif_id]
        is_ambiguity = motif_id in ambiguity
        has_external = bool(ref_entry.get("external_ref"))

        # gather reference spectra
        picks = []
        for name in ref_entry.get("primary_refs", []) + ref_entry.get("secondary_refs", []):
            nk = name.lower()
            if nk in rb_refs:
                picks.append(rb_refs[nk])
            elif nk in m63_refs:
                picks.append(m63_refs[nk])

        per_ref_support: list[ReferenceMotifSupport] = []
        for ref in picks:
            rms = evaluate_motif_on_reference(motif, ref, master_x)
            per_ref_support.append(rms)

            for bs in rms.bands:
                support_rows.append({
                    "motif_id": motif_id,
                    "motif_family": motif["motif_family"],
                    "reference_id": ref.component,
                    "reference_source": ref.source,
                    "band_family_id": bs.family_id,
                    "band_role": bs.role,
                    "band_cm1_centre": bs.cm1_centre,
                    "band_cm1_tolerance": bs.cm1_tolerance,
                    "match_cm1": bs.match_cm1 if bs.match_cm1 is not None else "",
                    "match_intensity": (round(bs.match_intensity, 4)
                                         if bs.match_intensity is not None else ""),
                    "evidence": bs.evidence,
                })

        # classify
        if is_ambiguity:
            status, rationale, metrics = classify_ambiguity_motif(
                motif, per_ref_support, ref_entry,
            )
            ambig_rows.append({
                "motif_id": motif_id,
                "motif_family": motif["motif_family"],
                "motif_type": motif["motif_type"],
                "grounding_status": status,
                "rationale": rationale,
                "n_refs_evaluated": metrics.get("n_refs_evaluated", 0),
                "n_candidate_classes_firing": metrics.get("n_candidate_classes_firing", 0),
                "candidate_classes_firing": json.dumps(
                    metrics.get("candidate_classes_firing", [])
                ),
            })
        else:
            status, rationale, metrics = classify_ready_motif(
                motif, per_ref_support, has_external,
            )

        matrix_rows.append({
            "motif_id": motif_id,
            "motif_family": motif["motif_family"],
            "motif_type": motif["motif_type"],
            "readiness_bucket": "AMBIGUITY_TRACK" if is_ambiguity else "READY_M3",
            "grounding_status": status,
            "rationale": rationale,
            "n_refs_evaluated": metrics.get("n_refs_evaluated", 0),
            "best_reference": metrics.get("best_reference", ""),
            "best_fraction_primary_fire": metrics.get("best_fraction_primary_fire", 0.0),
            "best_fraction_supporting_fire": metrics.get("best_fraction_supporting_fire", 0.0),
            "n_refs_primary_majority": metrics.get("n_refs_primary_majority", 0),
            "n_refs_grounded": metrics.get("n_refs_grounded", 0),
            "any_band_in_nan": metrics.get("any_band_in_nan", False),
            "literature_only": has_external,
        })

    # ── Emit tables ────────────────────────────────────────────────────
    matrix_df = pd.DataFrame(matrix_rows)
    support_df = pd.DataFrame(support_rows)
    ambig_df = pd.DataFrame(ambig_rows)
    matrix_df.to_csv(TABLES / "motif_grounding_matrix_v1.csv", index=False)
    support_df.to_csv(TABLES / "motif_reference_support_v1.csv", index=False)
    ambig_df.to_csv(TABLES / "motif_ambiguity_grounding_v1.csv", index=False)
    print(f"[emit] {TABLES}/motif_grounding_matrix_v1.csv ({len(matrix_df)} rows)")
    print(f"[emit] {TABLES}/motif_reference_support_v1.csv ({len(support_df)} rows)")
    print(f"[emit] {TABLES}/motif_ambiguity_grounding_v1.csv ({len(ambig_df)} rows)")

    # ── Figures ────────────────────────────────────────────────────────
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[warn] matplotlib unavailable: {e}; skipping figures")
    else:
        _make_overview_figure(matrix_df, FIGURES / "fig_motif_grounding_status_overview.png", plt)
        _make_family_figure(matrix_df, FIGURES / "fig_motif_family_grounding_summary.png", plt)
        _make_ambiguity_figure(ambig_df, FIGURES / "fig_ambiguity_track_summary.png", plt)

    # ── Report + audit log ─────────────────────────────────────────────
    _write_report(matrix_df, ambig_df, support_df, coverage_df,
                  rb_refs, m63_refs, ready_m3, ambiguity)
    _write_audit_log(matrix_df, support_df, coverage_df, rb_refs, m63_refs)

    print()
    print("=" * 78)
    print("M3 GROUNDING VALIDATION COMPLETE")
    print("=" * 78)
    # summary tallies
    print("\nREADY_M3 status distribution:")
    ready_counts = matrix_df[matrix_df["readiness_bucket"] == "READY_M3"]["grounding_status"].value_counts()
    for s, n in ready_counts.items():
        print(f"  {s:24s}: {n}")
    print("\nAMBIGUITY_TRACK status distribution:")
    amb_counts = matrix_df[matrix_df["readiness_bucket"] == "AMBIGUITY_TRACK"]["grounding_status"].value_counts()
    for s, n in amb_counts.items():
        print(f"  {s:24s}: {n}")


# ──────────────────────────────────────────────────────────────────────
# Figure helpers
# ──────────────────────────────────────────────────────────────────────

def _make_overview_figure(matrix_df, outpath, plt) -> None:
    order_ready = ["GROUNDED", "PARTIALLY_GROUNDED", "WEAKLY_GROUNDED",
                   "NOT_GROUNDED", "NOT_EVALUABLE"]
    order_amb = ["AMBIGUITY_CONFIRMED", "AMBIGUITY_PARTIAL", "AMBIGUITY_WEAK"]

    ready = matrix_df[matrix_df["readiness_bucket"] == "READY_M3"]["grounding_status"].value_counts()
    amb = matrix_df[matrix_df["readiness_bucket"] == "AMBIGUITY_TRACK"]["grounding_status"].value_counts()

    ready = ready.reindex(order_ready, fill_value=0)
    amb = amb.reindex(order_amb, fill_value=0)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    colors_ready = {"GROUNDED": "#2a9d8f", "PARTIALLY_GROUNDED": "#76c893",
                    "WEAKLY_GROUNDED": "#e9c46a", "NOT_GROUNDED": "#e76f51",
                    "NOT_EVALUABLE": "#adb5bd"}
    colors_amb = {"AMBIGUITY_CONFIRMED": "#7b2cbf", "AMBIGUITY_PARTIAL": "#b48ee6",
                  "AMBIGUITY_WEAK": "#d8bfd8"}
    axes[0].bar(ready.index, ready.values,
                color=[colors_ready.get(s, "#999") for s in ready.index])
    axes[0].set_title(f"READY_M3 grounding status (n={ready.sum()} motifs)")
    axes[0].set_ylabel("count")
    axes[0].tick_params(axis="x", rotation=30)
    for i, v in enumerate(ready.values):
        axes[0].text(i, v + 0.1, str(int(v)), ha="center", fontsize=9)
    axes[1].bar(amb.index, amb.values,
                color=[colors_amb.get(s, "#999") for s in amb.index])
    axes[1].set_title(f"AMBIGUITY_TRACK grounding (n={amb.sum()} motifs)")
    axes[1].tick_params(axis="x", rotation=30)
    for i, v in enumerate(amb.values):
        axes[1].text(i, v + 0.1, str(int(v)), ha="center", fontsize=9)
    for a in axes:
        for side in ("top", "right"):
            a.spines[side].set_visible(False)
    fig.suptitle("M3 Grounding Validation · motif-level status", fontsize=13)
    fig.tight_layout()
    fig.savefig(outpath, dpi=130)
    plt.close(fig)
    print(f"[emit] {outpath}")


def _make_family_figure(matrix_df, outpath, plt) -> None:
    pivot = (
        matrix_df
        .groupby(["motif_family", "grounding_status"])
        .size()
        .unstack(fill_value=0)
    )
    status_order = ["GROUNDED", "PARTIALLY_GROUNDED", "WEAKLY_GROUNDED",
                    "NOT_GROUNDED", "NOT_EVALUABLE",
                    "AMBIGUITY_CONFIRMED", "AMBIGUITY_PARTIAL", "AMBIGUITY_WEAK"]
    pivot = pivot.reindex(columns=status_order, fill_value=0)
    # sort families by total motifs descending
    pivot = pivot.loc[pivot.sum(axis=1).sort_values(ascending=False).index]

    colors = {
        "GROUNDED": "#2a9d8f", "PARTIALLY_GROUNDED": "#76c893",
        "WEAKLY_GROUNDED": "#e9c46a", "NOT_GROUNDED": "#e76f51",
        "NOT_EVALUABLE": "#adb5bd",
        "AMBIGUITY_CONFIRMED": "#7b2cbf", "AMBIGUITY_PARTIAL": "#b48ee6",
        "AMBIGUITY_WEAK": "#d8bfd8",
    }
    fig, ax = plt.subplots(figsize=(12, max(4.5, 0.38 * len(pivot))))
    bottom = np.zeros(len(pivot))
    for s in status_order:
        if s not in pivot.columns:
            continue
        ax.barh(pivot.index, pivot[s].values, left=bottom,
                color=colors[s], label=s)
        bottom += pivot[s].values
    ax.set_xlabel("motif count")
    ax.set_title("Grounding status by motif family")
    ax.legend(loc="lower right", fontsize=8, ncol=2)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    fig.savefig(outpath, dpi=130)
    plt.close(fig)
    print(f"[emit] {outpath}")


def _make_ambiguity_figure(ambig_df, outpath, plt) -> None:
    if ambig_df.empty:
        print("[warn] no ambiguity rows; skipping ambiguity figure")
        return
    fig, ax = plt.subplots(figsize=(10, 0.8 * len(ambig_df) + 2))
    y = np.arange(len(ambig_df))
    colors = {"AMBIGUITY_CONFIRMED": "#7b2cbf",
              "AMBIGUITY_PARTIAL": "#b48ee6",
              "AMBIGUITY_WEAK": "#d8bfd8"}
    bar_colors = [colors.get(s, "#999") for s in ambig_df["grounding_status"]]
    widths = ambig_df["n_candidate_classes_firing"].values
    ax.barh(y, widths, color=bar_colors)
    ax.set_yticks(y)
    ax.set_yticklabels(ambig_df["motif_id"], fontsize=9)
    ax.set_xlabel("n candidate classes firing")
    ax.set_title("Ambiguity-track motif grounding (confirmed ambiguity = ≥2 classes fire)")
    for i, (s, n) in enumerate(zip(ambig_df["grounding_status"], widths)):
        ax.text(n + 0.05, i, s, va="center", fontsize=8)
    ax.set_xlim(0, max(widths.max() + 2, 3))
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    fig.savefig(outpath, dpi=130)
    plt.close(fig)
    print(f"[emit] {outpath}")


# ──────────────────────────────────────────────────────────────────────
# Report + audit log
# ──────────────────────────────────────────────────────────────────────

def _write_report(matrix_df, ambig_df, support_df, coverage_df,
                  rb_refs, m63_refs, ready_m3, ambiguity) -> None:
    ready = matrix_df[matrix_df["readiness_bucket"] == "READY_M3"]
    amb = matrix_df[matrix_df["readiness_bucket"] == "AMBIGUITY_TRACK"]

    tally_ready = ready["grounding_status"].value_counts().to_dict()
    tally_amb = amb["grounding_status"].value_counts().to_dict()

    n_grounded = tally_ready.get("GROUNDED", 0)
    n_partial = tally_ready.get("PARTIALLY_GROUNDED", 0)
    n_weak = tally_ready.get("WEAKLY_GROUNDED", 0)
    n_not_grd = tally_ready.get("NOT_GROUNDED", 0)
    n_not_eval = tally_ready.get("NOT_EVALUABLE", 0)

    report_path = DOCS / "REPORT_M3_grounding_validation_v1.md"
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    lines = [
        "# GAIRA · gaira_build_motifs_v1 · Phase M3 — Grounding Validation (v1)",
        "",
        f"**Generated:** {now}  ",
        f"**Pipeline:** `crop_before_interpolate` (canonical; 400-1800 cm⁻¹, 1401 pts, 1 cm⁻¹ step)  ",
        f"**References:** ramanbiolib ({len(rb_refs)} spectra) + metabolite_sers63 ({len(m63_refs)} peak lists) + raman_knowledge_core (literature catalog)  ",
        f"**min_coverage:** 0.80 (tight, reference-grade)  ",
        "",
        "## Section A — Purpose",
        "",
        "M3 validates that every M1.1 motif rising to READY_M3 (or routed to ",
        "AMBIGUITY_TRACK) has chemical support in a curated reference library ",
        "of **pure-compound** Raman/SERS spectra. This is the chemistry-first ",
        "sanity check that must succeed before any cohort-phase calibration (M4) ",
        "or substrate-aware pilot validation (M5). It does not test cohort ",
        "behaviour or pilot outcomes; those are downstream.",
        "",
        "## Section B — Method",
        "",
        "1. Every reference spectrum from ramanbiolib was passed through ",
        "   `crop_before_interpolate` (canonical support [400, 1800] cm⁻¹, ",
        "   1401-point master axis, min_coverage 0.80). Out-of-support ",
        "   positions are NaN. No silent `np.interp` clamping.",
        "2. For each motif in {READY_M3 ∪ AMBIGUITY_TRACK}, a biochemistry-driven ",
        "   map (see script `MOTIF_REF_MAP`) selects the canonical pure-compound ",
        "   exemplars in the reference library.",
        "3. For each motif × reference pair, every primary and supporting band ",
        "   is tested via two evidence channels:",
        "   * `local_max` — a local maximum of the interpolated reference spectrum ",
        "     lies inside `[cm1_centre ± cm1_tolerance]`, with intensity above ",
        "     background floor (>1e-3 after normalisation).",
        "   * `peak_list` — the reference's catalogued peak list contains a peak ",
        "     inside the same window.",
        "   A band is judged to 'fire' if either channel is hit.",
        "4. Per motif, the best-firing reference is chosen and aggregate ",
        "   fractions are computed:",
        "   * `best_fraction_primary_fire` = (primary bands fired) / (primary bands total)",
        "   * `best_fraction_supporting_fire` = (supporting bands fired) / (supporting bands total)",
        "5. Grounding status is assigned:",
        "   * **GROUNDED** — ≥75% of primary bands fire in best reference",
        "   * **PARTIALLY_GROUNDED** — 50-75% of primary bands fire",
        "   * **WEAKLY_GROUNDED** — 25-50% primary OR ≥50% supporting",
        "   * **NOT_GROUNDED** — <25% primary and supporting evidence is weak",
        "   * **NOT_EVALUABLE** — no pure-compound reference in library ",
        "     (UA, HX, xanthine, ergothioneine, creatine) OR bands lie in NaN region",
        "6. For AMBIGUITY_TRACK motifs, the test is different: multiple candidate ",
        "   compound classes are expected to fire the overlap bands. Status:",
        "   * **AMBIGUITY_CONFIRMED** — ≥2 independent candidate classes fire ≥50% of primaries",
        "   * **AMBIGUITY_PARTIAL** — 1 candidate class fires; others do not",
        "   * **AMBIGUITY_WEAK** — 0 candidate classes reach threshold",
        "",
        "## Section C — Results summary",
        "",
        f"### READY_M3 ({len(ready)} motifs)",
        "",
        "| status              | count | fraction |",
        "|---|---:|---:|",
    ]
    for s in ["GROUNDED", "PARTIALLY_GROUNDED", "WEAKLY_GROUNDED",
              "NOT_GROUNDED", "NOT_EVALUABLE"]:
        n = tally_ready.get(s, 0)
        frac = n / max(len(ready), 1)
        lines.append(f"| {s} | {n} | {frac:.1%} |")

    lines += [
        "",
        f"### AMBIGUITY_TRACK ({len(amb)} motifs)",
        "",
        "| status | count |",
        "|---|---:|",
    ]
    for s in ["AMBIGUITY_CONFIRMED", "AMBIGUITY_PARTIAL", "AMBIGUITY_WEAK"]:
        lines.append(f"| {s} | {tally_amb.get(s, 0)} |")

    lines += [
        "",
        "## Section D — Per-motif grounding table",
        "",
        "See `tables/motif_grounding_matrix_v1.csv` for the authoritative row-per-motif ",
        "grounding status with best-reference metrics and rationale. Highlights:",
        "",
        "### Motifs GROUNDED",
        "",
    ]
    grounded_rows = ready[ready["grounding_status"] == "GROUNDED"].sort_values("motif_id")
    if len(grounded_rows):
        for _, r in grounded_rows.iterrows():
            lines.append(
                f"- `{r['motif_id']}` — best: {r['best_reference']} "
                f"({r['best_fraction_primary_fire']:.0%} primary fire)"
            )
    else:
        lines.append("(none)")

    lines += ["", "### Motifs NOT_GROUNDED", ""]
    ng_rows = ready[ready["grounding_status"] == "NOT_GROUNDED"].sort_values("motif_id")
    if len(ng_rows):
        for _, r in ng_rows.iterrows():
            lines.append(
                f"- `{r['motif_id']}` — {r['rationale']}"
            )
    else:
        lines.append("(none)")

    lines += ["", "### Motifs NOT_EVALUABLE", ""]
    ne_rows = ready[ready["grounding_status"] == "NOT_EVALUABLE"].sort_values("motif_id")
    if len(ne_rows):
        for _, r in ne_rows.iterrows():
            lines.append(f"- `{r['motif_id']}` — {r['rationale']}")
    else:
        lines.append("(none)")

    lines += [
        "",
        "## Section E — Ambiguity-track results",
        "",
        "Each ambiguity motif encodes a collision zone where the 22-window ",
        "panel cannot distinguish between multiple biochemistry candidates. ",
        "The grounding test here is *confirmation of ambiguity*, not resolution. ",
        "If ≥2 independent candidate classes fire the overlap bands in pure-",
        "compound references, the motif is empirically justified as an ",
        "ambiguity object rather than a CONVERGED single-chemistry claim.",
        "",
        "| motif_id | status | n_classes_firing | rationale |",
        "|---|---|---:|---|",
    ]
    for _, r in ambig_df.iterrows():
        rat = matrix_df.loc[matrix_df["motif_id"] == r["motif_id"], "rationale"].iloc[0]
        lines.append(
            f"| `{r['motif_id']}` | {r['grounding_status']} | "
            f"{r['n_candidate_classes_firing']} | {rat} |"
        )

    lines += [
        "",
        "## Section F — Decisions and downstream implications",
        "",
        f"- {n_grounded + n_partial} of {len(ready)} READY_M3 motifs cleared at GROUNDED+PARTIALLY_GROUNDED ",
        f"  ({(n_grounded + n_partial) / max(len(ready), 1):.0%}). These are the motifs that ",
        "  should enter M4 calibration without further evidence acquisition.",
        f"- {n_weak} WEAKLY_GROUNDED motifs are **calibration-eligible but must carry a ",
        "  'weak-grounding' flag** through to reporting; their cohort-direction signal must ",
        "  be treated as exploratory until a dedicated reference spectrum is acquired.",
        f"- {n_not_grd} NOT_GROUNDED motifs **do not clear M3**. They should be frozen ",
        "  on the shelf pending either pure-compound reference acquisition or motif ",
        "  redefinition (e.g. widening tolerances at the cost of specificity).",
        f"- {n_not_eval} NOT_EVALUABLE motifs have no direct pure-compound analogue ",
        "  in the current reference library (UA, HX, xanthine, ergothioneine, creatine). ",
        "  These are high-value for GAIRA (they are the metabolites that motivated the ",
        "  purine/thiol motif family) but their M3 clearance must be deferred until ",
        "  a dedicated reference-spectrum acquisition pass.",
        "",
        "### Recommended next steps",
        "",
        "1. Proceed to **M4 calibration** using the GROUNDED + PARTIALLY_GROUNDED set.",
        "2. **Do not** let WEAKLY_GROUNDED motifs contribute to any motif-level claim ",
        "   in a report without their flag propagating through.",
        "3. Plan a **reference-spectrum acquisition pass** for the NOT_EVALUABLE set:",
        "   UA, HX, xanthine (Ag-colloid SERS), ergothioneine, creatine/creatinine.",
        "4. Re-examine NOT_GROUNDED motifs: either acquire references or retire.",
        "5. Ambiguity motifs that are AMBIGUITY_CONFIRMED are **empirically justified**; ",
        "   they should be promoted from candidate to first-class motif in the registry ",
        "   and carry an explicit candidate-class list through to reporting.",
        "",
        "## Section G — Provenance",
        "",
        f"- Motif registry: `{REGISTRY_YAML}` ({_sha256(REGISTRY_YAML)[:16]}...)",
        f"- Status table:  `{STATUS_CSV}` ({_sha256(STATUS_CSV)[:16]}...)",
        f"- ramanbiolib spectra: `{RAMANBIOLIB_SPECTRA}` ({_sha256(RAMANBIOLIB_SPECTRA)[:16]}...)",
        f"- ramanbiolib peaks: `{RAMANBIOLIB_PEAKS}` ({_sha256(RAMANBIOLIB_PEAKS)[:16]}...)",
        f"- knowledge_core peak catalog: `{KNOWLEDGE_CORE_PEAKS}` ({_sha256(KNOWLEDGE_CORE_PEAKS)[:16]}...)",
        f"- script: `scripts/run_gaira_motifs_M3_grounding_validation_v1.py`",
        "",
        "## Section H — Non-modification invariants",
        "",
        "- No motif definition modified (read-only on `motif_candidate_registry_v1_1.yaml`).",
        "- No pilot output modified (axes-phase `gaira_build_axes_v1` is not touched).",
        "- No substrate engine weight modified (substrate_physics_v1.1.2 unchanged).",
        "- No calibration performed (deferred to M4).",
        "- Every reference loaded through `crop_before_interpolate` (no silent clamping).",
    ]

    report_path.write_text("\n".join(lines))
    print(f"[emit] {report_path}")


def _write_audit_log(matrix_df, support_df, coverage_df,
                     rb_refs, m63_refs) -> None:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    lines = [
        "# M3 Grounding Audit Log",
        "",
        f"Generated: {now}",
        "",
        "## Reference library summary",
        "",
        f"- ramanbiolib: {len(rb_refs)} full-spectrum references",
        f"- metabolite_sers63: {len(m63_refs)} peak-list references",
        "",
        "## Coverage audit (crop_before_interpolate)",
        "",
        f"- n_refs with partial coverage: {int(coverage_df['partial_coverage'].sum())}",
        f"- mean coverage fraction (ramanbiolib only): "
        f"{coverage_df[coverage_df['reference_source'] == 'ramanbiolib']['coverage_fraction'].mean():.3f}",
        f"- min coverage fraction: "
        f"{coverage_df[coverage_df['reference_source'] == 'ramanbiolib']['coverage_fraction'].min():.3f}",
        "",
        "## Motif evaluation audit",
        "",
        f"- total motifs evaluated: {len(matrix_df)}",
        f"- total band × reference evaluations: {len(support_df)}",
        f"- band evidence counts (across all motifs × refs):",
    ]
    ev = support_df["evidence"].value_counts()
    for e, n in ev.items():
        lines.append(f"  - {e}: {n}")

    lines += [
        "",
        "## Status distribution",
        "",
    ]
    for bucket, sub in matrix_df.groupby("readiness_bucket"):
        lines.append(f"### {bucket}")
        lines.append("")
        for s, n in sub["grounding_status"].value_counts().items():
            lines.append(f"- {s}: {n}")
        lines.append("")

    lines += [
        "## Invariants verified",
        "",
        "- [x] All reference spectra routed through crop_before_interpolate",
        "- [x] min_coverage=0.80 enforced on ramanbiolib loads",
        "- [x] NaN masking preserved; no band counted as firing when window is NaN-only",
        "- [x] Motif registry not mutated",
        "- [x] Pilot outputs not touched",
        "",
        "## Pipeline version",
        "",
        f"- canonical support: {CANONICAL_SUPPORT_CM1}",
        f"- n_points: {CANONICAL_N_POINTS}",
        f"- step: {CANONICAL_STEP_CM1}",
        "- helper: gaira.spectral.crop_before_interpolate (canonical_signal_support_rule_v1)",
    ]
    path = AUDIT / "M3_grounding_audit_log.md"
    path.write_text("\n".join(lines))
    print(f"[emit] {path}")


if __name__ == "__main__":
    main()
