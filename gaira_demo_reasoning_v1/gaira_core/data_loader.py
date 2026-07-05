"""GAIRA Demo v1 — data loaders.

Loaders try real cached data first (under /data/cached or the legacy demo
folder), and fall back to clearly-labelled demo placeholders.

Every loader returns a tuple (df_or_obj, is_placeholder) so the UI can
render a "Demo placeholder" badge whenever real cached data is missing.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from . import config as cfg


# ─────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────

def _exists(path: Path) -> bool:
    try:
        return path.exists() and path.stat().st_size > 0
    except Exception:
        return False


def _read_csv(path: Path) -> pd.DataFrame | None:
    if _exists(path):
        try:
            return pd.read_csv(path)
        except Exception:
            return None
    return None


def _read_parquet(path: Path) -> pd.DataFrame | None:
    if _exists(path):
        try:
            return pd.read_parquet(path)
        except Exception:
            return None
    return None


# ─────────────────────────────────────────────────────────────────────
# Reference spectra (synthetic but spectroscopically plausible)
# ─────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class MoleculeRef:
    name: str
    primary_axis: str
    anchors: tuple[float, ...]
    supports: tuple[float, ...]
    anti_evidence: tuple[float, ...] = ()
    summary: str = ""
    domain_notes: str = ""


# Curated reference set used across MSS explorer, collision viewer, and end-to-end demo.
MOLECULES: dict[str, MoleculeRef] = {
    "adenine": MoleculeRef(
        name="Adenine",
        primary_axis="G01_purine_nucleotide",
        anchors=(725.0, 1335.0, 1485.0),
        supports=(1245.0, 1380.0, 1580.0),
        anti_evidence=(640.0,),
        summary="Purine nucleobase. Ring-breathing near 720–735 cm⁻¹ is the dominant anchor.",
        domain_notes="Ag-SERS amplifies the 720–740 band markedly; molecule-level call requires co-bands at 1335 & 1485.",
    ),
    "uric_acid": MoleculeRef(
        name="Uric acid",
        primary_axis="G02_purine_metabolite",
        anchors=(640.0, 891.0, 1130.0),
        supports=(998.0, 1207.0, 1495.0, 1517.0),
        anti_evidence=(725.0,),
        summary="Oxidized purine metabolite. 640+891 doublet is highly distinctive.",
        domain_notes="In Ag-SERS serum, 1517 cm⁻¹ overlaps carotenoid C=C — substrate caveat applies.",
    ),
    "hypoxanthine": MoleculeRef(
        name="Hypoxanthine",
        primary_axis="G02_purine_metabolite",
        anchors=(720.0, 1230.0),
        supports=(645.0, 870.0, 1140.0, 1390.0),
        summary="Purine catabolite, intermediate between adenine and uric acid.",
        domain_notes="Shares 720 cm⁻¹ purine ring-breathing with adenine — class-level call only.",
    ),
    "ergothioneine": MoleculeRef(
        name="Ergothioneine",
        primary_axis="G10_sulfur_thiol_redox",
        anchors=(495.0, 1265.0, 1430.0),
        supports=(720.0, 1015.0, 1605.0),
        summary="Thione-bearing imidazole, dietary redox-protective small molecule.",
        domain_notes="C–S thione stretch near 495 cm⁻¹ is the redox-axis anchor; imidazole adds 720 cm⁻¹ which can collide with adenine.",
    ),
    "lactate": MoleculeRef(
        name="Lactate",
        primary_axis="G11_metabolic_small_molecule",
        anchors=(845.0, 925.0),
        supports=(1043.0, 1090.0, 1455.0),
        summary="Glycolytic end-product, common biofluid metabolite.",
        domain_notes="C–C–O stretch near 845 cm⁻¹ overlaps glycan ring features — class-level only.",
    ),
    "glutathione": MoleculeRef(
        name="Glutathione",
        primary_axis="G10_sulfur_thiol_redox",
        anchors=(510.0, 670.0, 2575.0),  # 2575 is out-of-window but listed
        supports=(870.0, 1300.0, 1410.0),
        summary="Cellular thiol redox buffer.",
        domain_notes="S–H stretch at 2575 cm⁻¹ is outside the demo window; redox call rests on 500–670 region.",
    ),
    "oleic_acid": MoleculeRef(
        name="Oleic acid",
        primary_axis="G08_lipid_acyl_membrane",
        anchors=(1265.0, 1440.0, 1655.0),
        supports=(890.0, 1080.0, 1300.0, 1745.0),
        summary="Unsaturated free fatty acid, common membrane / serum lipid component.",
        domain_notes="C=C stretch at 1655 cm⁻¹ overlaps amide-I — substrate-aware routing required in protein-rich matrices.",
    ),
    "phenylalanine": MoleculeRef(
        name="Phenylalanine",
        primary_axis="G07_aromatic_residue",
        anchors=(1003.0,),
        supports=(622.0, 1031.0, 1208.0, 1605.0),
        summary="Aromatic amino acid, sharp 1003 cm⁻¹ ring-breathing band.",
        domain_notes="1003 cm⁻¹ is the most reliable aromatic anchor in serum; SERS may down-weight intensity.",
    ),
    "tyrosine": MoleculeRef(
        name="Tyrosine",
        primary_axis="G07_aromatic_residue",
        anchors=(830.0, 850.0),
        supports=(643.0, 1175.0, 1615.0),
        summary="Aromatic amino acid with characteristic 830/850 Fermi doublet.",
        domain_notes="Doublet ratio is environment-sensitive (H-bonding); useful for protein conformation context.",
    ),
    "glucose": MoleculeRef(
        name="Glucose",
        primary_axis="G05_glycan_carbohydrate",
        anchors=(517.0, 845.0, 1125.0),
        supports=(914.0, 1065.0, 1370.0, 1460.0),
        summary="Reference monosaccharide. C–C/C–O stretches dominate.",
        domain_notes="Anomeric C–H near 845 cm⁻¹ collides with lactate C–C–O — needs co-bands at 1125 cm⁻¹.",
    ),
    "cholesterol": MoleculeRef(
        name="Cholesterol",
        primary_axis="G09_sterol_neutral_lipid",
        anchors=(548.0, 700.0, 1440.0),
        supports=(1130.0, 1300.0, 1665.0),
        summary="Sterol skeletal motif, characteristic 548 cm⁻¹ ring deformation.",
        domain_notes="Sterol-specific 548 cm⁻¹ separates from general lipid; 1665 cm⁻¹ shoulder marks the C=C of the steroid ring.",
    ),
}


def molecule_list() -> list[str]:
    return list(MOLECULES.keys())


# ─────────────────────────────────────────────────────────────────────
# Synthesised reference spectrum
# ─────────────────────────────────────────────────────────────────────

def _gaussian_bumps(
    wavenumber: np.ndarray,
    centers: list[float],
    heights: list[float],
    widths: list[float],
) -> np.ndarray:
    y = np.zeros_like(wavenumber, dtype=float)
    for c, h, w in zip(centers, heights, widths):
        if not (cfg.WAVENUMBER_MIN <= c <= cfg.WAVENUMBER_MAX):
            continue
        y += h * np.exp(-((wavenumber - c) ** 2) / (2.0 * (w ** 2)))
    return y


def synth_reference_spectrum(mol_id: str, noise: float = 0.005, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Return (wavenumber, intensity) for a curated molecule's reference."""
    ref = MOLECULES[mol_id]
    wn = np.linspace(cfg.WAVENUMBER_MIN, cfg.WAVENUMBER_MAX, cfg.WAVENUMBER_N)

    centers = list(ref.anchors) + list(ref.supports)
    heights = [0.9] * len(ref.anchors) + [0.4] * len(ref.supports)
    widths  = [8.0] * len(ref.anchors) + [12.0] * len(ref.supports)

    y = _gaussian_bumps(wn, centers, heights, widths)
    # Soft baseline
    y += 0.05 * np.exp(-((wn - 600.0) ** 2) / (2.0 * 250.0 ** 2))
    if noise > 0:
        rng = np.random.default_rng(seed if seed else hash(mol_id) % (2**31))
        y = y + rng.normal(0.0, noise, size=y.shape)
    # Normalize to [0, 1]
    y = (y - y.min()) / max(y.max() - y.min(), 1e-9)
    return wn, y


# ─────────────────────────────────────────────────────────────────────
# Grounding corpus map
# ─────────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def load_grounding_corpus() -> tuple[pd.DataFrame, bool]:
    """Return grounding source table and is_placeholder flag.

    Real loader: pulls the autoresearch v1 `warehouse_source_registry.csv`
    (43 sources across reference / disease+stress / serum-grounding) and
    joins per-source spectra/peak counts from `grounding_peak_support_summary.csv`
    when available.
    """
    real = _load_grounding_corpus_real()
    if real is not None:
        return real, False

    # Fallback: curated demo summary
    rows = [
        {"source": "RamanBioLib",            "regime": "Raman", "tier": 1, "role": "direct molecular reference",  "n_spectra": 202, "n_analytes": 78},
        {"source": "Gobbato (powder Raman)", "regime": "Raman", "tier": 1, "role": "calibration ground truth",     "n_spectra": 153, "n_analytes": 53},
        {"source": "Stanford AA panel",      "regime": "Raman", "tier": 1, "role": "amino-acid reference",         "n_spectra":  20, "n_analytes":  20},
        {"source": "NIHMS1547448 Ag-SERS",   "regime": "SERS",  "tier": 1, "role": "serum Ag-colloid metabolites", "n_spectra":  63, "n_analytes":  25},
        {"source": "Isotope validation",     "regime": "Raman", "tier": 1, "role": "isotope-shift validation",     "n_spectra":  12, "n_analytes":   6},
        {"source": "Spike-in calibration",   "regime": "Raman", "tier": 1, "role": "dose-response anchors",        "n_spectra":  30, "n_analytes":   3},
        {"source": "Raman atlas (literature)", "regime": "both", "tier": 2, "role": "band-position context",      "n_spectra":   0, "n_analytes":   0},
        {"source": "Substrate physics rules",  "regime": "SERS", "tier": 2, "role": "substrate-aware caveats",    "n_spectra":   0, "n_analytes":   0},
        {"source": "Domain caveat library",   "regime": "both", "tier": 2, "role": "interpretation rules",        "n_spectra":   0, "n_analytes":   0},
    ]
    return pd.DataFrame(rows), True


def _load_grounding_corpus_real() -> pd.DataFrame | None:
    """Read warehouse_source_registry + per-source counts and produce a UI-ready
    grounding-map DataFrame with columns:
        source, source_family, regime, tier, role,
        n_spectra, n_peaks, n_classes, biosample_type, notes
    """
    reg = _read_csv(cfg.GROUNDING_REGISTRY_TABLES / "warehouse_source_registry.csv")
    if reg is None:
        return None
    counts = _read_csv(cfg.GROUNDING_REGISTRY_TABLES / "grounding_peak_support_summary.csv")

    # Tier rule: reference_molecule + serum_grounding → tier 1 (direct);
    # disease_or_stress_paper → tier 2 (literature-backed evidence)
    def _tier(fam: str) -> int:
        return 1 if fam in ("reference_molecule", "serum_grounding") else 2

    def _regime(s: str) -> str:
        s = str(s or "").lower()
        if "raman" in s and "sers" in s: return "both"
        if "sers" in s: return "SERS"
        if "raman" in s: return "Raman"
        if "mixed" in s: return "both"
        return s or "unknown"

    def _role(fam: str, ref: str | bool, evid: str | bool) -> str:
        if fam == "reference_molecule": return "direct molecular reference"
        if fam == "serum_grounding":    return "serum reference grounding"
        return "literature evidence (disease/stress paper)"

    cnt_map = {}
    if counts is not None and "source_id" in counts.columns:
        for _, r in counts.iterrows():
            cnt_map[str(r["source_id"])] = {
                "n_spectra": int(r.get("summary_spectra_count", 0) or 0),
                "n_peaks":   int(r.get("detected_peak_count", 0) or 0),
                "n_classes": int(r.get("class_count", 0) or 0),
            }

    rows = []
    for _, r in reg.iterrows():
        sid = str(r.get("source_id", ""))
        fam = str(r.get("source_family", ""))
        c   = cnt_map.get(sid, {})
        rows.append({
            "source":        str(r.get("source_name") or sid),
            "source_id":     sid,
            "source_family": fam,
            "regime":        _regime(r.get("modality")),
            "tier":          _tier(fam),
            "role":          _role(fam, r.get("is_reference_grounding"),
                                       r.get("is_disease_or_stress_evidence")),
            "biosample_type": str(r.get("biosample_type") or ""),
            "n_spectra":     int(c.get("n_spectra", 0) or 0),
            "n_peaks":       int(c.get("n_peaks", 0) or 0),
            "n_classes":     int(c.get("n_classes", 0) or 0),
            "notes":         str(r.get("notes") or "")[:200],
        })
    return pd.DataFrame(rows)


@st.cache_data(show_spinner=False)
def load_family_counts() -> tuple[pd.DataFrame, bool]:
    """Per-axis grounding analyte counts (demo placeholders)."""
    counts = {
        "G01_purine_nucleotide":      9,
        "G02_purine_metabolite":      7,
        "G03_pyrimidine_nucleotide":  8,
        "G04_nucleic_acid_phosphate": 5,
        "G05_glycan_carbohydrate":   15,
        "G06_protein_peptide_backbone": 18,
        "G07_aromatic_residue":      10,
        "G08_lipid_acyl_membrane":   16,
        "G09_sterol_neutral_lipid":   6,
        "G10_sulfur_thiol_redox":     8,
        "G11_metabolic_small_molecule": 14,
    }
    rows = [{"axis": a, "n_analytes": counts.get(a, 0)} for a in cfg.BSV_AXES]
    return pd.DataFrame(rows), True


# ─────────────────────────────────────────────────────────────────────
# 11-axis biochemical space: reference points
# ─────────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def load_reference_points() -> tuple[pd.DataFrame, bool]:
    """Return per-molecule reference axis contributions for the 11-axis map.

    Tries the legacy 8-axis grounding_molecule_bsv.csv first; if found,
    remaps to v11 (replicating shared axes). Otherwise builds a small
    curated demo set from MOLECULES.
    """
    legacy_path = cfg.LEGACY_DEMO_DATA / "grounding_molecule_bsv.csv"
    idx_path = cfg.LEGACY_DEMO_DATA / "grounding_molecule_index.csv"
    df = _read_csv(legacy_path)
    idx = _read_csv(idx_path)
    if df is not None and idx is not None:
        try:
            df = df.merge(idx[["id", "type", "sample_substrate"]], on="id", how="left")
            # Remap to v11 via cfg.LEGACY8_TO_V11
            for a11 in cfg.BSV_AXES:
                df[a11] = 0.0
            for legacy_axis, v11_children in cfg.LEGACY8_TO_V11.items():
                if legacy_axis in df.columns:
                    for child in v11_children:
                        df[child] = df[child] + df[legacy_axis] / len(v11_children)
            df = df.rename(columns={"component": "name", "type": "category",
                                     "sample_substrate": "substrate"})
            keep = ["id", "name", "category", "substrate"] + list(cfg.BSV_AXES)
            keep = [c for c in keep if c in df.columns]
            df = df[keep].copy()
            # Tag regime crudely (Raman by default for legacy)
            df["regime"] = "Raman"
            return df, False
        except Exception:
            pass

    # Fallback: build from curated MOLECULES
    rows = []
    for mol_id, ref in MOLECULES.items():
        row = {"id": mol_id, "name": ref.name, "category": "curated",
               "substrate": "Raman", "regime": "Raman"}
        for a in cfg.BSV_AXES:
            row[a] = 0.0
        row[ref.primary_axis] = 0.7
        # Light cross-axis spread
        for a in cfg.BSV_AXES:
            if a != ref.primary_axis:
                row[a] = float(np.random.default_rng(hash((mol_id, a)) % 2**31).uniform(0.0, 0.18))
        rows.append(row)
    return pd.DataFrame(rows), True


# ─────────────────────────────────────────────────────────────────────
# Ergothioneine dose-response
# ─────────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def load_ergothioneine_dose() -> tuple[pd.DataFrame, bool]:
    """Dose-response with per-axis BSV at each concentration.

    Tries legacy 8-axis CSV; remaps to v11 if found. Otherwise synthesises
    a clearly-labelled sigmoid placeholder dominated by G10/G11.
    """
    legacy_path = cfg.LEGACY_DEMO_DATA / "ergothioneine_dose_response.csv"
    raw = _read_csv(legacy_path)
    if raw is not None and {"concentration_uM", "axis", "bsv_mean", "delta_bsv"}.issubset(raw.columns):
        wide = raw.pivot_table(index="concentration_uM", columns="axis",
                                values="bsv_mean", aggfunc="mean").reset_index()
        delta = raw.pivot_table(index="concentration_uM", columns="axis",
                                 values="delta_bsv", aggfunc="mean").reset_index()
        # Remap to v11
        out_rows = []
        for _, row in wide.iterrows():
            conc = float(row["concentration_uM"])
            v11 = {a: 0.0 for a in cfg.BSV_AXES}
            for legacy_axis, v11_children in cfg.LEGACY8_TO_V11.items():
                if legacy_axis in row.index:
                    val = float(row[legacy_axis])
                    for child in v11_children:
                        v11[child] += val / len(v11_children)
            out_rows.append({"concentration_uM": conc, **v11})
        df = pd.DataFrame(out_rows)
        return df, False

    # Placeholder: sigmoid in G10 + smaller rise in G11
    concs = np.array([0.0, 0.1, 0.25, 0.5, 1.0, 1.5, 2.0])
    rng = np.random.default_rng(42)
    rows = []
    for c in concs:
        row = {"concentration_uM": float(c)}
        for a in cfg.BSV_AXES:
            row[a] = float(rng.uniform(0.02, 0.06))
        # Sigmoidal G10 climb
        row["G10_sulfur_thiol_redox"] = float(0.04 + 0.45 / (1.0 + np.exp(-3.5 * (c - 0.5))))
        # Smaller G11 rise
        row["G11_metabolic_small_molecule"] = float(0.04 + 0.22 / (1.0 + np.exp(-3.0 * (c - 0.6))))
        # Tiny G07 trickle (imidazole 720 nudge → routed mostly to G10 by motif logic)
        row["G07_aromatic_residue"] = float(0.04 + 0.05 * (c / 2.0))
        rows.append(row)
    return pd.DataFrame(rows), True


# ─────────────────────────────────────────────────────────────────────
# Adenine + Uric acid calibration placeholders
# ─────────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def load_adenine_calibration() -> tuple[pd.DataFrame, bool]:
    """Per-concentration adenine BSV summary.

    Real path: read the bAgNPs concentration series CSVs from
    `/Volumes/SSD_Rad/GAIRA_DATA/raw/adenine_sers_control/`, preprocess each
    (crop 400–1800 cm⁻¹, interpolate to 1 cm⁻¹ grid), call `build_report`
    per spectrum with substrate='Ag colloid SERS', and return one row per
    concentration with the 11-axis BSV.

    Falls back to a clearly-labelled sigmoid placeholder if the raw CSVs are
    not on disk.
    """
    real = _load_adenine_real()
    if real is not None:
        return real, False

    # Fallback: curated sigmoid placeholder
    rng = np.random.default_rng(7)
    rows = []
    for cond, conc in [("blank", 0.0), ("low", 5e-6), ("mid", 5e-5), ("high", 5e-4)]:
        row = {"condition": cond, "concentration_M": conc}
        for a in cfg.BSV_AXES:
            row[a] = float(rng.uniform(0.02, 0.07))
        scale = {"blank": 0.05, "low": 0.18, "mid": 0.45, "high": 0.72}[cond]
        row["G01_purine_nucleotide"] = scale + float(rng.uniform(-0.02, 0.02))
        row["G02_purine_metabolite"] = 0.18 * scale + float(rng.uniform(-0.01, 0.02))
        row["G04_nucleic_acid_phosphate"] = 0.12 * scale + float(rng.uniform(-0.01, 0.02))
        rows.append(row)
    return pd.DataFrame(rows), True


# Map filename → (concentration_label, concentration_ng_per_mL).
# Sorted from low to high; bAgNPs substrate only.
_ADENINE_FILES: tuple[tuple[str, str, float], ...] = (
    ("Adenine_bAgNPs_10pg.CSV",   "10 pg/mL",   0.01),
    ("Adenine_bAgNPs_100pg.CSV",  "100 pg/mL",  0.1),
    ("Adenine_bAgNPs_10nano.CSV", "10 ng/mL",   10.0),
    ("Adenine_bAgNPs_100nano.CSV","100 ng/mL",  100.0),
    ("Adenine_bAgNPs_1micro.CSV", "1 µg/mL",    1000.0),
    ("Adenine_bAgNPs_10micro.CSV","10 µg/mL",   10000.0),
)


def _read_adenine_csv(path) -> tuple[np.ndarray, np.ndarray] | None:
    """Parse one raw adenine CSV. Format: cp1252 + ';' separator + ',' decimal,
    no header, two columns (wavenumber, intensity), 99–3500 cm⁻¹."""
    try:
        df = pd.read_csv(path, sep=";", decimal=",", encoding="cp1252",
                          header=None, names=["wn", "intensity"])
        wn = df["wn"].to_numpy(dtype=float)
        y  = df["intensity"].to_numpy(dtype=float)
        return wn, y
    except Exception:
        return None


def _crop_and_interp(wn: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Crop to demo window and interpolate to 1 cm⁻¹ grid."""
    grid = np.linspace(cfg.WAVENUMBER_MIN, cfg.WAVENUMBER_MAX, cfg.WAVENUMBER_N)
    # Sort by wavenumber in case the file isn't strictly monotonic
    order = np.argsort(wn)
    wn_sorted, y_sorted = wn[order], y[order]
    # numpy.interp clamps outside-range to edge values; we want zero outside.
    interp = np.interp(grid, wn_sorted, y_sorted, left=0.0, right=0.0)
    return grid, interp


def _load_adenine_real():
    """Real adenine calibration loader. Returns DataFrame or None."""
    if not cfg.ADENINE_RAW_DIR.exists():
        return None

    # Lazy import to avoid circular dependency at module import time
    from .report_builder import build_report

    rows = []
    n_loaded = 0
    for fname, label, ng_per_ml in _ADENINE_FILES:
        path = cfg.ADENINE_RAW_DIR / fname
        if not _exists(path):
            continue
        parsed = _read_adenine_csv(path)
        if parsed is None:
            continue
        wn_raw, y_raw = parsed
        # Only use the demo wavenumber window; spectra here go 99–3500 cm⁻¹
        wn, y = _crop_and_interp(wn_raw, y_raw)
        if not np.any(y):
            continue
        report = build_report(
            sample_id=f"adenine_{fname}",
            title=f"Adenine {label}",
            domain="calibration",
            substrate="Ag colloid SERS",
            wavenumber=wn, intensity=y,
        )
        row = {
            "condition":           label,
            "concentration_ng_mL": ng_per_ml,
            "source_file":         fname,
            "n_motifs_firing":     int(sum(1 for v in report["motif_scores_adjusted"].values() if v >= 0.05)),
        }
        for a in cfg.BSV_AXES:
            row[a] = float(report["bsv"].get(a, 0.0))
        rows.append(row)
        n_loaded += 1

    if n_loaded < 2:                 # not enough concentrations for a demo curve
        return None
    return pd.DataFrame(rows)


@st.cache_data(show_spinner=False)
def load_uric_acid_validation() -> tuple[pd.DataFrame, bool]:
    """Real uricase / hypoxanthine spike conditions from the calibration layer.

    Returns wide-format Δ BSV (11-axis remapped from legacy 8) keyed by
    condition_id. Columns:
        - condition_id (cspp_fig7_hypoxanthine_spike, uricase_sigma_depletion,
            uricase_spiked_hypoxanthine_serum)
        - display_name
        - n_control, n_perturbed
        - sael_overall_confidence, overall_label
        - delta_<G0X_*>  per the 11 axes
        - expected_direction_<G0X_*> per axis (up/down/unknown)
        - verdict_<G0X_*> per axis (agree/disagree/flat/not_testable)

    NOTE: No isotope (¹⁵N) data exists in the GAIRA corpus. The previous
    placeholder included a fabricated `uric_isotope_15N` condition — it has
    been removed.
    """
    cond_path = cfg.LEGACY_DEMO_DATA / "calibration_conditions.csv"
    delta_path = cfg.LEGACY_DEMO_DATA / "calibration_delta_bsv.csv"
    conds = _read_csv(cond_path)
    deltas = _read_csv(delta_path)
    if conds is None or deltas is None:
        return _uric_acid_placeholder(), True

    target_ids = {
        "cspp_fig7_hypoxanthine_spike":     "Hypoxanthine spike — serum",
        "uricase_sigma_depletion":          "Uricase depletion — Sigma serum",
        "uricase_spiked_hypoxanthine_serum": "Hypoxanthine spike + uricase",
    }
    conds = conds[conds["contrast_id"].isin(target_ids)].copy()
    if conds.empty:
        return _uric_acid_placeholder(), True

    rows = []
    for _, c in conds.iterrows():
        row: dict = {
            "condition_id":   c["contrast_id"],
            "display_name":   target_ids.get(c["contrast_id"], c.get("display_name", c["contrast_id"])),
            "n_control":      int(c.get("n_control", 0) or 0),
            "n_perturbed":    int(c.get("n_perturbed", 0) or 0),
            "confidence":     str(c.get("sael_overall_confidence", "")),
            "label":          str(c.get("overall_label", "")),
        }
        # Initialise all 11 axes
        for a in cfg.BSV_AXES:
            row[f"delta_{a}"] = 0.0
            row[f"expected_direction_{a}"] = "unknown"
            row[f"verdict_{a}"] = "not_testable"

        sub = deltas[deltas["contrast_id"] == c["contrast_id"]]
        for _, dr in sub.iterrows():
            legacy_axis = str(dr["axis"])
            v11_children = cfg.LEGACY8_TO_V11.get(legacy_axis, ())
            if not v11_children:
                continue
            d_val = float(dr.get("observed_delta", 0.0) or 0.0) / len(v11_children)
            exp_dir = str(dr.get("expected_direction", "unknown"))
            verdict = str(dr.get("verdict", "not_testable"))
            for child in v11_children:
                row[f"delta_{child}"] += d_val
                row[f"expected_direction_{child}"] = exp_dir
                row[f"verdict_{child}"] = verdict
        rows.append(row)
    return pd.DataFrame(rows), False


def _uric_acid_placeholder() -> pd.DataFrame:
    rng = np.random.default_rng(11)
    rows = []
    for cond in ["serum_unspiked", "serum_uric_spiked", "uricase_depleted"]:
        row = {"condition_id": cond, "display_name": cond,
                "n_control": 0, "n_perturbed": 0, "confidence": "low", "label": "demo"}
        for a in cfg.BSV_AXES:
            row[f"delta_{a}"] = float(rng.uniform(-0.01, 0.03))
            row[f"expected_direction_{a}"] = "unknown"
            row[f"verdict_{a}"] = "not_testable"
        rows.append(row)
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────
# Biological pilot cohorts (placeholders, but structured)
# ─────────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def load_pilot_cohorts(pilot: str) -> tuple[pd.DataFrame, bool]:
    """Return cohort-level BSV averages with n per cohort.

    Pilots: 'serum_liver', 'ev_diabetes', 'shine_liver_injury'.

    - 'shine_liver_injury' tries the autoresearch v1 outputs first and returns
        the real Day 0 + Day 2 × C0/C10/C20/C40 cohorts (NO Day 3/Day 7 — that
        data does not exist).
    - 'serum_liver' and 'ev_diabetes' currently have no cached cohort BSVs and
        return clearly-labelled demo placeholders.
    """
    if pilot == "shine_liver_injury":
        real = _load_shine_real()
        if real is not None:
            return real, False
        return _shine_placeholder(), True

    if pilot == "serum_liver":
        # Prefer real-spectra → demo 11-axis pipeline (lights all 11 axes).
        # Fall back to autoresearch BSV (3-axis sparse, kept honest as legacy).
        real = _load_serum_liver_from_spectra(n_per_cohort=4)
        if real is not None:
            return real, False
        real = _load_serum_liver_real()
        if real is not None:
            return real, False
        return _serum_liver_placeholder(), True

    if pilot == "ev_diabetes":
        real = _load_ev_diabetes_from_spectra(n_per_cohort=8)
        if real is not None:
            return real, False
        real = _load_ev_diabetes_real()
        if real is not None:
            return real, False
        return _ev_diabetes_placeholder(), True

    raise KeyError(pilot)


# ── EV diabetes real loader ─────────────────────────────────────────

def _load_ev_diabetes_real() -> pd.DataFrame | None:
    """Real EV-diabetes cohort BSVs from `pilot2_target_validation_v1`.

    Cohort labels are the ones the autoresearch pilot defined for this
    figure: `Impact` (the diabetes-impact group, n=39) and `Strong-D`
    (the strong-discrimination subgroup, n=24). These are NOT a generic
    "Normal vs Diabetic" split — they are project-specific labels and must
    be displayed verbatim in the UI so users know what they're looking at.
    """
    cmean = _read_csv(cfg.EV_DIABETES_TABLES / "class_mean_bsv.csv")
    if cmean is None:
        return None
    # Cohort sample counts from the parent pilot (documented in the
    # warehouse audit; no per-sample file in this tables/ dir).
    n_lookup = {"Impact": 39, "Strong-D": 24}

    rows = []
    for _, r in cmean.iterrows():
        label = str(r["class_label"])
        out = {"cohort": label, "n": int(n_lookup.get(label, 0))}
        # Autoresearch raw values for transparency
        for ax in cfg.AUTORESEARCH8_TO_V11:
            out[f"autoresearch_{ax}"] = float(r.get(ax, 0.0) or 0.0)
        for ax in cfg.AUTORESEARCH_NON_BIOLOGY:
            if ax in r:
                out[f"autoresearch_{ax}"] = float(r.get(ax, 0.0) or 0.0)
        # Remap into 11-axis BSV
        for a in cfg.BSV_AXES:
            out[a] = 0.0
        for ar_axis, v11_children in cfg.AUTORESEARCH8_TO_V11.items():
            v = float(r.get(ar_axis, 0.0) or 0.0)
            for child in v11_children:
                out[child] += v / len(v11_children)
        rows.append(out)
    return pd.DataFrame(rows)


# ── Serum liver real loader ─────────────────────────────────────────

def _load_serum_liver_real() -> pd.DataFrame | None:
    """Real serum-liver per-cohort BSVs from `pilot4_1_cca_hcc_lm_serum_patient_level`.

    Source is patient-level (one row per patient). We groupby class_label
    to produce cohort means + sample counts. Labels are: CCA, HA (healthy
    adult), HCC, LM. Autoresearch ontology; 6 of 8 axes carry signal.
    """
    pat = _read_csv(cfg.LIVER_PATIENT_TABLES / "patient_level_bsv.csv")
    if pat is None:
        return None
    auto_biology = [c for c in cfg.AUTORESEARCH8_TO_V11 if c in pat.columns]
    auto_non_bio = [c for c in cfg.AUTORESEARCH_NON_BIOLOGY if c in pat.columns]
    if not auto_biology:
        return None

    grouped = pat.groupby("class_label_display")[auto_biology + auto_non_bio].mean()
    counts = pat.groupby("class_label_display").size().to_dict()

    display = {
        "CCA": "CCA (cholangiocarcinoma)",
        "HCC": "HCC (hepatocellular carcinoma)",
        "LM":  "LM (liver metastases)",
        "HA":  "Healthy adult",
    }

    rows = []
    for label, row in grouped.iterrows():
        out = {"cohort": display.get(label, str(label)), "n": int(counts.get(label, 0)),
                "raw_label": label}
        for ax in auto_biology + auto_non_bio:
            out[f"autoresearch_{ax}"] = float(row.get(ax, 0.0) or 0.0)
        for a in cfg.BSV_AXES:
            out[a] = 0.0
        for ar_axis, v11_children in cfg.AUTORESEARCH8_TO_V11.items():
            v = float(row.get(ar_axis, 0.0) or 0.0)
            for child in v11_children:
                out[child] += v / len(v11_children)
        rows.append(out)
    # Preferred display order: Healthy first, then cancers
    df = pd.DataFrame(rows)
    order = {"HA": 0, "CCA": 1, "HCC": 2, "LM": 3}
    df["_order"] = df["raw_label"].map(order).fillna(99)
    return df.sort_values("_order").drop(columns="_order").reset_index(drop=True)


# ── SHINE real-data loader ──────────────────────────────────────────

def _load_shine_real() -> pd.DataFrame | None:
    """Load the real SHINE per-class BSV from the autoresearch v1 outputs.

    Source files (autoresearch 8-axis ontology):
      - class_mean_bsv_day0_day2.csv   (per-class means, Day 0 and Day 2 × C0/C10/C20/C40)
      - pilot3_shine_ev_sers/tables/per_sample_bsv.csv (per-sample BSVs covering
            Day 0 + Day 1 + Day 2 across Set 9 and Set 10 — used for n-per-class
            and for the per-scan total surfaced in the UI)

    Returns wide-format 11-axis cohort DataFrame with columns:
        cohort, n, day, dose, autoresearch_<axis>..., <G0X axis>...
    or None when the autoresearch outputs cannot be read.
    """
    cmean_path = cfg.SHINE_DAY02_TABLES / "class_mean_bsv_day0_day2.csv"
    sample_path = cfg.SHINE_EV_SERS_TABLES / "per_sample_bsv.csv"
    cmean = _read_csv(cmean_path)
    if cmean is None:
        return None

    # Autoresearch axes present in real file
    auto_axes = [c for c in cmean.columns
                    if c in cfg.AUTORESEARCH8_TO_V11 or c in cfg.AUTORESEARCH_NON_BIOLOGY]

    # n per class via per_sample_bsv. The pilot3_shine_ev_sers file is the
    # right source — it includes Day 0 + Day 1 + Day 2, keyed by full
    # class_label (e.g. D0_C0). Previously the demo used per_sample_bsv_day2.csv
    # which only carried Day 2 and forced Day 0 cohorts to display n=0.
    n_per_class: dict[str, int] = {}
    scans_per_class: dict[str, int] = {}
    samples = _read_csv(sample_path)
    if samples is not None and "class_label" in samples.columns:
        n_per_class = samples.groupby("class_label").size().to_dict()
        if "n_scans" in samples.columns:
            scans_per_class = (samples.groupby("class_label")["n_scans"]
                                  .sum(min_count=1).fillna(0).astype(int).to_dict())

    out_rows = []
    for _, row in cmean.iterrows():
        label = str(row["class_label"])               # e.g. D0_C10 / D2_C40
        day_str, dose_str = label.split("_", 1)
        day = int(day_str[1:])
        dose = int(dose_str[1:])
        cohort_label = f"Day {day} · C{dose}"
        out = {
            "cohort":      cohort_label,
            "n":           int(n_per_class.get(label, 0)),
            "n_scans":     int(scans_per_class.get(label, 0)),
            "day":         day,
            "dose":        dose,
            "class_label": label,
        }
        # Carry the raw autoresearch axes (with prefix) so UI can show them transparently
        for ax in auto_axes:
            out[f"autoresearch_{ax}"] = float(row.get(ax, 0.0) or 0.0)

        # Remap into 11-axis BSV
        for a in cfg.BSV_AXES:
            out[a] = 0.0
        for ar_axis, v11_children in cfg.AUTORESEARCH8_TO_V11.items():
            v = float(row.get(ar_axis, 0.0) or 0.0)
            for child in v11_children:
                out[child] += v / len(v11_children)
        out_rows.append(out)

    df = pd.DataFrame(out_rows)
    # Sort by (day, dose) so UI displays in trajectory order
    df = df.sort_values(["day", "dose"]).reset_index(drop=True)
    return df


def _shine_placeholder() -> pd.DataFrame:
    """Fallback if autoresearch volume is unavailable. Only Day 0 + Day 2 to
    match real data shape — no fabricated Day 3 / Day 7 timepoints."""
    rng = np.random.default_rng(3)
    rows = []
    for day in (0, 2):
        for dose in (0, 10, 20, 40):
            row = {
                "cohort":      f"Day {day} · C{dose}",
                "n":           0,
                "day":         day,
                "dose":        dose,
                "class_label": f"D{day}_C{dose}",
            }
            for a in cfg.BSV_AXES:
                row[a] = float(rng.uniform(0.04, 0.10))
            rows.append(row)
    return pd.DataFrame(rows)


def _serum_liver_placeholder() -> pd.DataFrame:
    rng = np.random.default_rng(1)
    cohorts = [("Healthy", 32), ("HCC", 28), ("CCA", 18), ("LM (liver metastases)", 22)]
    bases = {
        "Healthy":  {"G06_protein_peptide_backbone": 0.42, "G07_aromatic_residue": 0.35,
                       "G02_purine_metabolite": 0.22, "G08_lipid_acyl_membrane": 0.30},
        "HCC":      {"G06_protein_peptide_backbone": 0.36, "G07_aromatic_residue": 0.28,
                       "G02_purine_metabolite": 0.41, "G08_lipid_acyl_membrane": 0.34,
                       "G11_metabolic_small_molecule": 0.31},
        "CCA":      {"G06_protein_peptide_backbone": 0.34, "G07_aromatic_residue": 0.27,
                       "G02_purine_metabolite": 0.36, "G05_glycan_carbohydrate": 0.31,
                       "G08_lipid_acyl_membrane": 0.32},
        "LM (liver metastases)": {"G06_protein_peptide_backbone": 0.32, "G02_purine_metabolite": 0.39,
                                    "G08_lipid_acyl_membrane": 0.38, "G09_sterol_neutral_lipid": 0.21,
                                    "G11_metabolic_small_molecule": 0.29},
    }
    rows = []
    for name, n in cohorts:
        row = {"cohort": name, "n": n}
        base = bases.get(name, {})
        for a in cfg.BSV_AXES:
            row[a] = float(base.get(a, rng.uniform(0.06, 0.14)))
        rows.append(row)
    return pd.DataFrame(rows)


def _ev_diabetes_placeholder() -> pd.DataFrame:
    rng = np.random.default_rng(2)
    cohorts = [("Normal-Weight Control", 24), ("Overweight Diabetic", 21),
                 ("Normal-Weight Diabetic", 18)]
    bases = {
        "Normal-Weight Control":  {"G08_lipid_acyl_membrane": 0.42, "G06_protein_peptide_backbone": 0.31,
                                     "G05_glycan_carbohydrate": 0.25},
        "Overweight Diabetic":    {"G08_lipid_acyl_membrane": 0.55, "G09_sterol_neutral_lipid": 0.32,
                                     "G11_metabolic_small_molecule": 0.34, "G05_glycan_carbohydrate": 0.30},
        "Normal-Weight Diabetic": {"G08_lipid_acyl_membrane": 0.45, "G11_metabolic_small_molecule": 0.37,
                                     "G05_glycan_carbohydrate": 0.31},
    }
    rows = []
    for name, n in cohorts:
        row = {"cohort": name, "n": n}
        base = bases.get(name, {})
        for a in cfg.BSV_AXES:
            row[a] = float(base.get(a, rng.uniform(0.06, 0.14)))
        rows.append(row)
    return pd.DataFrame(rows)


# ────────────────────────────────────────────────────────────────────
# Real-spectra → demo 11-axis pipeline loaders
#
# The autoresearch v1 BSV exports are structurally a 3-axis projection
# (verified: 100% of 15,027 SHINE spectra fire on exactly 3 of 8 axes;
# all liver/EV pilots show the same pattern). Remapping that 3-axis
# signal into 11 axes still leaves 8 axes at zero by construction.
#
# These loaders bypass the autoresearch BSV entirely. They read the
# autoresearch's preprocessed mean spectra (still real GAIRA data —
# the same spectra the autoresearch pipeline consumed) and run them
# through this demo's own motif/MSS/substrate pipeline via build_report.
# That pipeline is designed to fire on all 11 axes when biology is
# present, so the resulting radars carry real multi-axis biology.
#
# Sampling: we take N patients/samples per cohort to keep first-load
# fast (~16s first load for liver, cached after).
# ────────────────────────────────────────────────────────────────────

def _build_cohort_means_from_spectra(spectra_rows, wn, source_label,
                                       substrate, n_per_cohort, domain):
    """Helper: iterate (sample_id, cohort, intensity vector) tuples, call
    build_report on the first n_per_cohort per cohort, aggregate per cohort.

    Returns a wide DataFrame keyed by cohort with all 11 BSV axes + metadata.
    """
    from .report_builder import build_report

    cohort_to_rows: dict[str, list[tuple[str, np.ndarray]]] = {}
    cohort_totals:  dict[str, int]                          = {}
    for sample_id, cohort, y in spectra_rows:
        cohort_totals[cohort] = cohort_totals.get(cohort, 0) + 1
        if len(cohort_to_rows.get(cohort, [])) < n_per_cohort:
            cohort_to_rows.setdefault(cohort, []).append((sample_id, y))

    out_rows = []
    for cohort, samples in cohort_to_rows.items():
        per_sample_bsv = []
        for sample_id, y in samples:
            y_clip = np.clip(y, 0, None)
            rep = build_report(
                sample_id=str(sample_id), title=f"{source_label}::{cohort}",
                domain=domain, substrate=substrate,
                wavenumber=wn, intensity=y_clip,
            )
            per_sample_bsv.append([float(rep["bsv"].get(a, 0.0)) for a in cfg.BSV_AXES])
        arr = np.asarray(per_sample_bsv)
        mean = arr.mean(axis=0)
        out = {
            "cohort":          cohort,
            "n_sampled":       len(samples),
            "n_cohort_total":  cohort_totals[cohort],
            "n":               cohort_totals[cohort],   # mirror legacy `n` col
            "source":          source_label,
            "n_nonzero_axes":  int((mean > 1e-4).sum()),
        }
        for a, v in zip(cfg.BSV_AXES, mean):
            out[a] = float(v)
        out["dominant_axis"] = cfg.BSV_AXES[int(np.argmax(mean))]
        out_rows.append(out)
    return pd.DataFrame(out_rows)


@st.cache_data(show_spinner="Computing real 11-axis BSV from liver patient spectra…")
def _load_serum_liver_from_spectra(n_per_cohort: int = 4) -> pd.DataFrame | None:
    """Real serum-liver 11-axis BSV from the demo pipeline applied to the
    autoresearch's processed patient mean spectra."""
    path = cfg.LIVER_PATIENT_TABLES / "patient_level_mean_spectra.csv"
    if not _exists(path):
        return None
    try:
        sp = pd.read_csv(path)
    except Exception:
        return None
    wn_cols = [c for c in sp.columns if c.startswith("wn_")]
    if not wn_cols or "class_label_display" not in sp.columns:
        return None
    wn = np.array([int(c[3:]) for c in wn_cols], dtype=float)

    spectra_rows = []
    for _, r in sp.iterrows():
        sample_id = str(r.get("sample_id", ""))
        cohort    = str(r["class_label_display"])
        y = r[wn_cols].to_numpy(dtype=float)
        spectra_rows.append((sample_id, cohort, y))

    df = _build_cohort_means_from_spectra(
        spectra_rows, wn,
        source_label="patient_level_mean_spectra",
        substrate="Ag colloid SERS",
        n_per_cohort=n_per_cohort, domain="serum",
    )
    if df is None or df.empty:
        return None
    display = {
        "CCA": "CCA (cholangiocarcinoma)",
        "HCC": "HCC (hepatocellular carcinoma)",
        "LM":  "LM (liver metastases)",
        "HA":  "Healthy adult",
    }
    df["raw_label"] = df["cohort"]
    df["cohort"] = df["cohort"].map(lambda c: display.get(c, c))
    order = {"Healthy adult": 0, "CCA (cholangiocarcinoma)": 1,
              "HCC (hepatocellular carcinoma)": 2, "LM (liver metastases)": 3}
    df["_order"] = df["cohort"].map(order).fillna(99)
    return df.sort_values("_order").drop(columns="_order").reset_index(drop=True)


@st.cache_data(show_spinner="Computing real 11-axis BSV from EV-diabetes spectra…")
def _load_ev_diabetes_from_spectra(n_per_cohort: int = 8) -> pd.DataFrame | None:
    """Real EV-diabetes 11-axis BSV via the demo pipeline.

    Source: `sample_query_spectra.csv` — 63 samples with wavenumbers_json +
    intensity_json columns. Each sample is interpolated onto the demo's
    400–1800 1 cm⁻¹ grid before scoring.
    """
    import json
    path = cfg.EV_DIABETES_TABLES / "sample_query_spectra.csv"
    if not _exists(path):
        return None
    try:
        sp = pd.read_csv(path)
    except Exception:
        return None
    need = {"wavenumbers_json", "intensity_json", "class_label", "sample_id"}
    if not need.issubset(sp.columns):
        return None

    grid = np.linspace(cfg.WAVENUMBER_MIN, cfg.WAVENUMBER_MAX, cfg.WAVENUMBER_N)

    spectra_rows = []
    for _, r in sp.iterrows():
        try:
            wn_raw = np.asarray(json.loads(r["wavenumbers_json"]), dtype=float)
            y_raw  = np.asarray(json.loads(r["intensity_json"]),  dtype=float)
        except Exception:
            continue
        order = np.argsort(wn_raw)
        y_on_grid = np.interp(grid, wn_raw[order], y_raw[order], left=0.0, right=0.0)
        spectra_rows.append((str(r["sample_id"]), str(r["class_label"]), y_on_grid))

    if not spectra_rows:
        return None
    df = _build_cohort_means_from_spectra(
        spectra_rows, grid,
        source_label="sample_query_spectra",
        substrate="Ag colloid SERS",
        n_per_cohort=n_per_cohort, domain="extracellular_vesicle",
    )
    return df


# ─── Family-fingerprint loaders (complementary view, NOT remapped to v11) ───

def _load_family_fingerprint_long(path) -> pd.DataFrame | None:
    df = _read_csv(path)
    if df is None or df.empty:
        return None
    cohort_col = next((c for c in ("class_label", "class_label_display", "cohort")
                          if c in df.columns), None)
    if cohort_col is None or "family" not in df.columns or "family_fraction" not in df.columns:
        return None
    pivot = (df.groupby([cohort_col, "family"])["family_fraction"]
                .mean().unstack(fill_value=0.0).reset_index()
                .rename(columns={cohort_col: "cohort"}))
    return pivot


def load_shine_family_fingerprint() -> tuple[pd.DataFrame, bool]:
    df = _load_family_fingerprint_long(cfg.SHINE_FAMILY_FINGERPRINT)
    return (df if df is not None else pd.DataFrame()), df is None


def load_ev_diabetes_family_fingerprint() -> tuple[pd.DataFrame, bool]:
    df = _load_family_fingerprint_long(cfg.EV_FAMILY_FINGERPRINT)
    return (df if df is not None else pd.DataFrame()), df is None


def load_liver_family_fingerprint() -> tuple[pd.DataFrame, bool]:
    df = _read_csv(cfg.LIVER_FAMILY_FINGERPRINT)
    if df is None or df.empty:
        return pd.DataFrame(), True
    family_cols = [c for c in df.columns
                      if c.endswith("_like") or c == "generic_other_metabolite"]
    if not family_cols or "class_label_display" not in df.columns:
        return pd.DataFrame(), True
    agg = (df.groupby("class_label_display")[family_cols].mean()
              .reset_index().rename(columns={"class_label_display": "cohort"}))
    return agg, False
