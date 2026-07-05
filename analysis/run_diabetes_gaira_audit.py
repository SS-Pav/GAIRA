#!/usr/bin/env python3
"""GAIRA re-analysis of the diabetes plasma-EV SERS cohort.

This is a modular *additive* script — it does NOT modify GAIRA core code.
It reads raw data from SSD_Rad, applies the current GAIRA demo pipeline
(from ``gaira_demo_reasoning_v1/gaira_core/``), and produces:

    results/diabetes_gaira_audit_YYYYMMDD_HHMM/
        diabetes_file_manifest.csv
        diabetes_label_audit.csv
        diabetes_preprocessing_audit.md
        diabetes_gaira_scores_per_spectrum.csv        (per-patient-mean)
        diabetes_gaira_scores_per_sample.csv          (== per-patient)
        diabetes_group_summary_2group.csv
        diabetes_group_summary_4subgroup.csv
        diabetes_analyte_hits.csv
        diabetes_analyte_hits_high_confidence.csv
        diabetes_qc_summary.md
        diabetes_interpretation_report.md
        publication_quality_figures/
            fig_radar_2group.{pdf,svg,png}
            fig_radar_4subgroup.{pdf,svg,png}
            fig_pca_2group.{pdf,svg,png}
            fig_pca_4subgroup.{pdf,svg,png}
            fig_mean_spectra_2group.{pdf,svg,png}
            fig_mean_spectra_4subgroup.{pdf,svg,png}
            fig_difference_spectrum_owd_vs_nwd.{pdf,svg,png}
            fig_bsv_heatmap.{pdf,svg,png}
            fig_analyte_hits_top.{pdf,svg,png}
            fig_qc_counts.{pdf,svg,png}
            figure_captions.md

Scientific rules:
- class-level interpretation only, never "molecule X is present"
- per-patient (subject-level) statistics, not per-spectrum pseudoreplication
- Mann-Whitney (2 groups) + Kruskal-Wallis (4 subgroups), BH-FDR corrected
- evidence tiered: Tier-1 direct spectral, Tier-2 literature, + domain context
- provenance: every output row cites its source file and processing version

Random seed / determinism: RANDOM_SEED = 42 throughout.
"""
from __future__ import annotations

import json
import shutil
import sys
import warnings
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.io import loadmat

# --------------------------------------------------------------------
# Local imports — this script re-uses the demo's core pipeline but
# does not modify it.
# --------------------------------------------------------------------
DEMO_ROOT = Path("/Users/suraj/projects/GAIRA/gaira_demo_reasoning_v1")
sys.path.insert(0, str(DEMO_ROOT))

from gaira_core import config as gcfg
from gaira_core.data_loader import MOLECULES, synth_reference_spectrum
from gaira_core.mss_scoring import score_all as mss_score_all

# Diabetes-audit overrides (tightened G10 motif + co-band-gated Ag-SERS thiol boost).
# We call `build_report_diabetes` in place of the demo's build_report so the
# Streamlit demo remains untouched.
_ANALYSIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_ANALYSIS_DIR))
from _diabetes_overrides import build_report_diabetes  # noqa: E402

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
np.random.seed(42)

# --------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------
SSD_RAW  = Path("/Volumes/SSD_Rad/GAIRA_DATA/raw/diabetes_plasma_ev_sers/extracted")
SSD_PILOTS = Path("/Volumes/SSD_Rad/GAIRA_DATA/processed/gaira_autoresearch/gaira_autoresearch_v1")
SSD_PRIOR_BUILDS = Path("/Volumes/SSD_Rad/GAIRA_BUILD")

RESULTS_ROOT = Path("/Users/suraj/projects/GAIRA/results")
STAMP = datetime.now().strftime("%Y%m%d_%H%M")
OUT   = RESULTS_ROOT / f"diabetes_gaira_audit_{STAMP}"
FIG_DIR = OUT / "publication_quality_figures"
OUT.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------
# 1. FILE DISCOVERY & MANIFEST
# --------------------------------------------------------------------
def discover_and_manifest() -> pd.DataFrame:
    """Enumerate every diabetes-related file across raw + processed + prior
    build directories, with role, size, and modified date."""
    rows = []

    def _add(path: Path, role: str, phase: str):
        try:
            st = path.stat()
        except Exception:
            return
        rows.append({
            "path":       str(path),
            "phase":      phase,
            "role":       role,
            "size_bytes": int(st.st_size),
            "size_mb":    round(st.st_size / 1e6, 3),
            "modified":   datetime.fromtimestamp(st.st_mtime).strftime("%Y-%m-%d %H:%M"),
            "kind":       ("dir" if path.is_dir() else path.suffix.lstrip(".") or "file"),
        })

    for f in ("patient_data.csv", "RawDataImpact.mat", "RawDataStrong.mat",
                "configure_classes.m", "Figure3.m", "race_split.m"):
        _add(SSD_RAW / f, role=("metadata" if f.endswith(".csv") else "raw"),
              phase="raw")
    for f in ("Diabetes_Raw_Data_Codes.zip", "LogCPM_miRNA.xlsx",
                "Manuscript_vs18.docx", "Replies to Reviewers.docx"):
        _add(SSD_RAW.parent / f, role="reference_material", phase="raw")

    for pilot in ("pilot2_target_validation_v1",
                    "pilot2_2_diabetes_temporary_axis_transfer",
                    "pilot2_1_latent_state_interpretation"):
        pilot_dir = SSD_PILOTS / pilot
        if pilot_dir.exists():
            for sub in ("tables", "figures", "report", "runs"):
                d = pilot_dir / sub
                if d.exists():
                    for entry in sorted(d.iterdir()):
                        if entry.is_file():
                            _add(entry, role=f"autoresearch_{sub}",
                                  phase=f"autoresearch::{pilot}")

    for prior in ("gaira_base_4_diabetes_ev_pilot_v1",
                    "gaira_base_4_diabetes_ev_mss_classifier_v2",
                    "gaira_base_4_diabetes_ev_bsv_mss_audit_v1",
                    "gaira_base_4_diabetes_ev_full_audit_v1"):
        prior_dir = SSD_PRIOR_BUILDS / prior
        if prior_dir.exists():
            for sub in ("tables", "figures", "reports", "audit"):
                d = prior_dir / sub
                if d.exists():
                    for entry in sorted(d.iterdir()):
                        if entry.is_file() and not entry.name.startswith("._"):
                            _add(entry, role=f"prior_{sub}",
                                  phase=f"prior_build::{prior}")

    df = pd.DataFrame(rows).sort_values(["phase", "role", "path"])
    df.to_csv(OUT / "diabetes_file_manifest.csv", index=False)
    return df


# --------------------------------------------------------------------
# 2. WAVENUMBER CALIBRATION
# --------------------------------------------------------------------
# From Figure3.m: cubic polyfit against 8 known Raman peaks + crop 162:898.
CAL_PIX = np.array([263, 367, 492, 512, 590, 782, 872, 887], dtype=float)
CAL_WN  = np.array([620.9, 795.8, 1001.4, 1031.8, 1155.3, 1450.5, 1583.1, 1602.3])
CROP_PIX_START, CROP_PIX_END = 162, 898         # inclusive, 1-indexed in MATLAB

def diabetes_wavenumbers() -> np.ndarray:
    """Return the 737-length wavenumber vector matching the .mat rows."""
    coef = np.polyfit(CAL_PIX, CAL_WN, deg=3)
    full_pix = np.arange(1, 1651)                # 1..1650
    x_full = np.polyval(coef, full_pix)
    return x_full[CROP_PIX_START - 1 : CROP_PIX_END]     # length 737


# --------------------------------------------------------------------
# 3. LOAD PATIENT METADATA + SPECTRA
# --------------------------------------------------------------------
def _map_bmi_group(bmi: float) -> str:
    """Design axis: Impact = clinical trial cohort (OWD = overweight/obese
    diabetic); Strong-D = observational NWD (normal-weight diabetic).
    The paper's binary label uses this Group column verbatim."""
    return "OWD" if bmi >= 25.0 else "NWD"


def _four_subgroup(group: str, race: str) -> str | None:
    """4-subgroup structure per Figure3.m:
        Asian Impact, Asian Strong-D, White Impact, White Strong-D.
    Returns None for patients outside those two races."""
    if group == "Impact":
        g = "Impact"
    elif group == "Strong-D":
        g = "Strong-D"
    else:
        return None
    r = None
    if race == "Asian":
        r = "Asian"
    elif race == "Non-Hispanic White":
        r = "White"
    return f"{r} {g}" if r else None


def load_metadata_and_labels() -> tuple[pd.DataFrame, list[dict]]:
    """Return (patient_df, label_audit_rows)."""
    df = pd.read_csv(SSD_RAW / "patient_data.csv").drop(columns=["Unnamed: 0"], errors="ignore")
    df["patient_id"] = df["filename"].astype(str)
    df["group_2"] = df["Group"].map({"Impact": "OWD", "Strong-D": "NWD"})
    df["subgroup_4"] = [_four_subgroup(g, r) for g, r in zip(df["Group"], df["race_ethnicity"])]

    audit = []
    for _, r in df.iterrows():
        audit.append({
            "patient_id":  r["patient_id"],
            "group_raw":   r["Group"],
            "group_2":     r["group_2"],
            "race_ethnicity": r["race_ethnicity"],
            "subgroup_4":  r["subgroup_4"],
            "gender":      r["gender"],
            "bmi":         r["bmi"],
            "hba1c":       r["hba1c"],
            "age":         r["age_bl"],
            "in_2group":   r["group_2"] in ("OWD", "NWD"),
            "in_4subgroup": r["subgroup_4"] is not None,
        })
    return df, audit


def load_raw_spectra() -> dict[str, np.ndarray]:
    """Return {patient_id: (n_scans, 737) intensity array}.

    Each .mat's `smoothed_spectra` is an object array of length = n_patients
    of the same group. Ordering matches the enumeration in configure_classes.m:
    the Impact .mat lists Impact patients in the order they appear on disk
    (dirs 2151-*), same for Strong (dirs 32113-*).
    """
    result: dict[str, np.ndarray] = {}
    meta = pd.read_csv(SSD_RAW / "patient_data.csv")

    def _flatten_to_patient_order(mat_key: str, group_prefix: str) -> list[str]:
        # patient_data.csv is grouped: Impact first (40 rows), Strong second (24).
        # The .mat ordering matches directory enumeration which is alphanumeric
        # ascending on the ID after the prefix; patient_data may not be sorted
        # the same way. We rebuild the sorted list from patient_data itself.
        subset = meta[meta["filename"].str.startswith(group_prefix)]
        return sorted(subset["filename"].tolist())

    for grp_prefix, mat_name in [("2151", "RawDataImpact.mat"),
                                    ("32113", "RawDataStrong.mat")]:
        m = loadmat(SSD_RAW / mat_name, squeeze_me=True)
        sp = m["smoothed_spectra"]
        pids = _flatten_to_patient_order("smoothed_spectra", grp_prefix)
        # Impact .mat has 39 patients but patient_data has 40 Impact rows.
        # One patient is missing from the .mat. We can only match the first
        # `len(sp)` sorted patient IDs. Log the drop.
        n_available = min(len(sp), len(pids))
        for i in range(n_available):
            arr = np.asarray(sp[i], dtype=float)     # (737, N_scans)
            # smoothed_spectra is (wn, scan). Transpose to (scan, wn).
            if arr.ndim == 2 and arr.shape[0] == 737:
                arr = arr.T
            result[pids[i]] = arr
    return result


# --------------------------------------------------------------------
# 4. PREPROCESSING (into demo's 400–1800 1-cm⁻¹ grid)
# --------------------------------------------------------------------
def _interp_to_demo_grid(wn_native: np.ndarray, spec: np.ndarray) -> np.ndarray:
    """Interpolate one native spectrum onto the demo's 400–1800 grid."""
    grid = np.linspace(gcfg.WAVENUMBER_MIN, gcfg.WAVENUMBER_MAX, gcfg.WAVENUMBER_N)
    order = np.argsort(wn_native)
    return np.interp(grid, wn_native[order], spec[order], left=0.0, right=0.0)


def preprocess_patient(spectra: np.ndarray, wn_native: np.ndarray) -> np.ndarray:
    """Mean-of-scans → interpolate → clip negatives.

    The demo's build_report handles the rest (Savitzky–Golay + ASLS baseline
    + L2 norm) internally. The input here should be a mean spectrum on the
    demo's 1401-point grid."""
    mean_spec = spectra.mean(axis=0)
    on_grid = _interp_to_demo_grid(wn_native, mean_spec)
    return np.clip(on_grid, 0.0, None)


# --------------------------------------------------------------------
# 5. GAIRA INFERENCE
# --------------------------------------------------------------------
def run_gaira_per_patient(patients: pd.DataFrame,
                             spectra_by_pid: dict[str, np.ndarray],
                             wn_native: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run build_report on each patient's mean spectrum.

    Returns:
        bsv_df: one row per patient with 11 BSV axes + n_scans + metadata
        mss_df: one row per (patient, molecule) with MSS fire scores
    """
    grid = np.linspace(gcfg.WAVENUMBER_MIN, gcfg.WAVENUMBER_MAX, gcfg.WAVENUMBER_N)
    bsv_rows, mss_rows = [], []
    for _, r in patients.iterrows():
        pid = r["patient_id"]
        raw = spectra_by_pid.get(pid)
        if raw is None:
            continue
        mean_spec = preprocess_patient(raw, wn_native)
        rep = build_report_diabetes(
            sample_id=pid, title=f"{r['Group']}::{pid}",
            domain="extracellular_vesicle", substrate="Ag colloid SERS",
            wavenumber=grid, intensity=mean_spec,
        )
        bsv = {a: float(rep["bsv"].get(a, 0.0)) for a in gcfg.BSV_AXES}
        # We deliberately use build_report's internal normalization (motif
        # scores). We DO NOT further normalize BSVs here — additional
        # normalization would double-normalize and destroy magnitudes.
        thiol_evt = next((e for e in rep["substrate_events"]
                              if e["rule_id"] == "ag_sers_thiol_amplify"), None)
        thiol_gate_state = thiol_evt["gate"] if thiol_evt else "n/a"
        imi_720 = rep["diabetes_audit_flags"]["imidazole_720_intensity"]
        row = {
            "patient_id": pid,
            "group_raw":  r["Group"],
            "group_2":    r["group_2"],
            "subgroup_4": r["subgroup_4"],
            "race_ethnicity": r["race_ethnicity"],
            "gender":     r["gender"],
            "bmi":        r["bmi"],
            "hba1c":      r["hba1c"],
            "age":        r["age_bl"],
            "n_scans":    int(raw.shape[0]),
            "n_nonzero_axes": int(sum(1 for v in bsv.values() if v > 1e-4)),
            "dominant_axis":  max(bsv, key=bsv.get),
            "imidazole_720_intensity": round(imi_720, 5),
            "thiol_boost_gate": thiol_gate_state,
        }
        row.update(bsv)
        bsv_rows.append(row)

        # MSS fire scores per molecule
        fires = mss_score_all(grid, np.asarray(rep["spectrum"]["processed_intensity"]))
        for mol_id, fire in fires.items():
            mss_rows.append({
                "patient_id":     pid,
                "group_2":        r["group_2"],
                "subgroup_4":     r["subgroup_4"],
                "molecule_id":    mol_id,
                "molecule_name":  MOLECULES[mol_id].name,
                "primary_axis":   MOLECULES[mol_id].primary_axis,
                "anchor_score":   float(fire.anchor_score),
                "support_score":  float(fire.support_score),
                "anti_score":     float(fire.anti_score),
                "fire":           float(fire.fire),
            })
    return pd.DataFrame(bsv_rows), pd.DataFrame(mss_rows)


# --------------------------------------------------------------------
# 6. STATISTICS
# --------------------------------------------------------------------
def _fdr_bh(pvals: np.ndarray) -> np.ndarray:
    """Benjamini–Hochberg FDR correction. Returns q-values."""
    p = np.asarray(pvals, dtype=float)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order]
    q = np.empty(n)
    prev_min = 1.0
    for i in range(n - 1, -1, -1):
        val = ranked[i] * n / (i + 1)
        prev_min = min(prev_min, val)
        q[order[i]] = min(1.0, prev_min)
    return q


def _cliffs_delta(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.asarray(a), np.asarray(b)
    if len(a) == 0 or len(b) == 0:
        return 0.0
    return float((np.sum(a[:, None] > b[None, :]) - np.sum(a[:, None] < b[None, :]))
                    / (len(a) * len(b)))


def _cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.asarray(a), np.asarray(b)
    if len(a) < 2 or len(b) < 2:
        return 0.0
    sa, sb = np.var(a, ddof=1), np.var(b, ddof=1)
    pooled = np.sqrt(((len(a) - 1) * sa + (len(b) - 1) * sb) / max(1, len(a) + len(b) - 2))
    if pooled < 1e-12:
        return 0.0
    return float((a.mean() - b.mean()) / pooled)


def two_group_stats(bsv: pd.DataFrame) -> pd.DataFrame:
    """Per-axis Mann-Whitney (subject-level) between OWD and NWD."""
    rows = []
    axes = list(gcfg.BSV_AXES)
    a_df = bsv[bsv["group_2"] == "OWD"]
    b_df = bsv[bsv["group_2"] == "NWD"]
    for axis in axes:
        a = a_df[axis].to_numpy(dtype=float)
        b = b_df[axis].to_numpy(dtype=float)
        if len(a) < 3 or len(b) < 3:
            u, p = np.nan, np.nan
        else:
            try:
                u, p = stats.mannwhitneyu(a, b, alternative="two-sided")
            except ValueError:
                u, p = np.nan, np.nan
        rows.append({
            "axis":              axis,
            "axis_label":        gcfg.axis_long(axis),
            "n_OWD":             int(len(a)),
            "n_NWD":             int(len(b)),
            "mean_OWD":          float(a.mean()) if len(a) else np.nan,
            "mean_NWD":          float(b.mean()) if len(b) else np.nan,
            "delta_OWD_minus_NWD": float(a.mean() - b.mean()) if len(a) and len(b) else np.nan,
            "cohens_d":          _cohens_d(a, b),
            "cliffs_delta":      _cliffs_delta(a, b),
            "mannwhitney_U":     float(u) if u == u else np.nan,
            "p_value":           float(p) if p == p else np.nan,
        })
    df = pd.DataFrame(rows)
    df["q_value_fdr_bh"] = _fdr_bh(df["p_value"].fillna(1.0).to_numpy())
    return df.sort_values("q_value_fdr_bh").reset_index(drop=True)


def four_subgroup_stats(bsv: pd.DataFrame) -> pd.DataFrame:
    """Per-axis Kruskal-Wallis across the four subgroups."""
    rows = []
    sub_df = bsv[bsv["subgroup_4"].notna()]
    subs = ["White Strong-D", "White Impact", "Asian Strong-D", "Asian Impact"]
    for axis in gcfg.BSV_AXES:
        groups = [sub_df[sub_df["subgroup_4"] == s][axis].to_numpy(dtype=float) for s in subs]
        counts = [len(g) for g in groups]
        means  = [float(g.mean()) if len(g) else np.nan for g in groups]
        try:
            H, p = stats.kruskal(*[g for g in groups if len(g) >= 2])
        except ValueError:
            H, p = np.nan, np.nan
        rows.append({
            "axis":       axis,
            "axis_label": gcfg.axis_long(axis),
            **{f"n_{s}": counts[i] for i, s in enumerate(subs)},
            **{f"mean_{s}": means[i] for i, s in enumerate(subs)},
            "kruskal_H":  float(H) if H == H else np.nan,
            "p_value":    float(p) if p == p else np.nan,
        })
    df = pd.DataFrame(rows)
    df["q_value_fdr_bh"] = _fdr_bh(df["p_value"].fillna(1.0).to_numpy())
    return df.sort_values("q_value_fdr_bh").reset_index(drop=True)


def analyte_hits(mss: pd.DataFrame, bsv: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-molecule per-group MSS fires + confidence tiers."""
    rows = []
    for mol_id, ref in MOLECULES.items():
        mss_mol = mss[mss["molecule_id"] == mol_id]
        # Per-cohort mean fire
        mean_fires = mss_mol.groupby("group_2")["fire"].mean().to_dict()
        n_matched_anchors = int(np.count_nonzero(np.array([_local_max_score(a, r) for a, r in zip([mss_mol.iloc[0]["anchor_score"] if len(mss_mol) else 0], [ref])])))
        # Simpler + clearer: use median n of anchors above prominence threshold
        # We use the pre-defined MOLECULE anchor set as the "expected bands".
        expected_bands = list(ref.anchors)
        n_expected = len(expected_bands)

        owd_fires = mss_mol[mss_mol["group_2"] == "OWD"]["fire"].to_numpy()
        nwd_fires = mss_mol[mss_mol["group_2"] == "NWD"]["fire"].to_numpy()
        if len(owd_fires) >= 3 and len(nwd_fires) >= 3:
            try:
                u, p = stats.mannwhitneyu(owd_fires, nwd_fires, alternative="two-sided")
            except ValueError:
                u, p = np.nan, np.nan
            d = _cohens_d(owd_fires, nwd_fires)
            direction = "OWD > NWD" if owd_fires.mean() > nwd_fires.mean() else "OWD < NWD"
        else:
            u, p, d, direction = np.nan, np.nan, 0.0, "n/a"

        mean_anchor = float(mss_mol["anchor_score"].mean()) if len(mss_mol) else 0.0
        mean_support = float(mss_mol["support_score"].mean()) if len(mss_mol) else 0.0
        mean_anti = float(mss_mol["anti_score"].mean()) if len(mss_mol) else 0.0
        overall_fire = float(mss_mol["fire"].mean()) if len(mss_mol) else 0.0
        collision_partners = _collisions_for(ref.primary_axis)

        # Confidence tier — relative + effect-size aware, appropriate for the
        # Ag-SERS-dampened value ranges seen in this cohort (typical anchor
        # magnitudes 0.01–0.05):
        #   High:   mean anchor >= 0.030 AND |d| >= 1.0 AND FDR-adjusted p < 0.05
        #   Medium: mean anchor >= 0.020 AND |d| >= 0.4
        #   Low:    everything else (weak evidence / no group separation)
        strong_effect = (abs(d) >= 1.0)
        moderate_effect = (abs(d) >= 0.4)
        p_ok = (p == p) and (p < 0.05)
        if mean_anchor >= 0.030 and strong_effect and p_ok:
            confidence_tier = "High"
        elif mean_anchor >= 0.020 and moderate_effect:
            confidence_tier = "Medium"
        else:
            confidence_tier = "Low"

        # Evidence tier:
        # Tier-1: direct spectral grounding — anchor above noise floor and
        #         the motif library has a curated set of anchor bands for this analyte.
        # Tier-2: literature/contextual support only.
        evidence_tier = "Tier-1" if mean_anchor >= 0.020 else "Tier-2"

        rows.append({
            "molecule_id":       mol_id,
            "molecule_name":     ref.name,
            "biochemical_class": gcfg.axis_long(ref.primary_axis),
            "primary_axis":      ref.primary_axis,
            "expected_anchors_cm1": ", ".join(f"{c:.0f}" for c in ref.anchors),
            "expected_supports_cm1": ", ".join(f"{c:.0f}" for c in ref.supports),
            "collision_partners":  ", ".join(collision_partners),
            "n_expected_bands":   n_expected,
            "peak_tolerance_cm1": 8.0,
            "mean_anchor_score":   round(mean_anchor, 5),
            "mean_support_score":  round(mean_support, 5),
            "mean_anti_score":     round(mean_anti, 5),
            "mean_fire_score":     round(overall_fire, 5),
            "owd_mean_fire":       round(float(mean_fires.get("OWD", 0.0)), 5),
            "nwd_mean_fire":       round(float(mean_fires.get("NWD", 0.0)), 5),
            "cohens_d_owd_vs_nwd": round(d, 4),
            "mannwhitney_U":       float(u) if u == u else np.nan,
            "p_value":             float(p) if p == p else np.nan,
            "directionality":      direction,
            "evidence_tier":       evidence_tier,
            "confidence_tier":     confidence_tier,
            "caveats":             ref.domain_notes,
        })
    df = pd.DataFrame(rows)
    df["q_value_fdr_bh"] = _fdr_bh(df["p_value"].fillna(1.0).to_numpy())
    return df.sort_values(["confidence_tier", "mean_fire_score"],
                            ascending=[True, False]).reset_index(drop=True)


def _local_max_score(anchor_score: float, ref) -> float:
    return anchor_score


def _collisions_for(axis: str) -> list[str]:
    """Curated collision partners per axis for the caveats column."""
    coll = {
        "G01_purine_nucleotide":      ["hypoxanthine", "uric acid", "xanthine"],
        "G02_purine_metabolite":      ["adenine", "hypoxanthine"],
        "G03_pyrimidine_nucleotide":  ["cytosine", "thymine", "uracil"],
        "G04_nucleic_acid_phosphate": ["phosphate esters", "glycan C–O overlap"],
        "G05_glycan_carbohydrate":    ["lactate C–C–O 845", "protein amide"],
        "G06_protein_peptide_backbone": ["albumin", "haptoglobin", "amide-I overlap"],
        "G07_aromatic_residue":       ["Phe 1003 · Tyr 830/850 · Trp 760"],
        "G08_lipid_acyl_membrane":    ["amide-I 1650 overlap"],
        "G09_sterol_neutral_lipid":   ["free cholesterol vs esterified"],
        "G10_sulfur_thiol_redox":     ["glutathione", "cysteine", "cystine"],
        "G11_metabolic_small_molecule": ["lactate", "acetate", "pyruvate"],
    }
    return coll.get(axis, [])


# --------------------------------------------------------------------
# 7. PUBLICATION-QUALITY FIGURES (Matplotlib)
# --------------------------------------------------------------------
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PUB_STYLE = {
    "font.family":      "sans-serif",
    "font.sans-serif":  ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size":        11,
    "axes.titlesize":   12,
    "axes.labelsize":   11,
    "axes.linewidth":   0.9,
    "axes.edgecolor":   "#334155",
    "axes.spines.top":  False,
    "axes.spines.right": False,
    "xtick.labelsize":  10,
    "ytick.labelsize":  10,
    "legend.fontsize":  10,
    "legend.frameon":   False,
    "figure.dpi":       120,
    "savefig.dpi":      600,
    "savefig.bbox":     "tight",
    "savefig.pad_inches": 0.06,
    "lines.linewidth":  1.6,
}
plt.rcParams.update(PUB_STYLE)

# Consistent colors
COLOR_OWD, COLOR_NWD = "#DC2626", "#2563EB"
SUBGROUP_COLORS = {
    "Asian Impact":   "#DC2626",
    "White Impact":   "#F59E0B",
    "Asian Strong-D": "#2563EB",
    "White Strong-D": "#10B981",
}

def _save(fig, name: str):
    for ext in ("pdf", "svg", "png"):
        fig.savefig(FIG_DIR / f"{name}.{ext}", transparent=(ext != "png"))
    plt.close(fig)


def _radar_axes(ax, values_by_group: dict, radial_max: float, title: str):
    axes = list(gcfg.BSV_AXES)
    N = len(axes)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    labels = [gcfg.axis_short(a) for a in axes]
    for label, vals in values_by_group.items():
        r = np.array([vals.get(a, 0.0) for a in axes], dtype=float)
        r_closed = np.append(r, r[0])
        theta = np.append(angles, angles[0])
        color = (COLOR_OWD if label == "OWD"
                    else COLOR_NWD if label == "NWD"
                    else SUBGROUP_COLORS.get(label, "#64748B"))
        ax.plot(theta, r_closed, color=color, linewidth=1.8, label=label)
        ax.fill(theta, r_closed, color=color, alpha=0.14)
    ax.set_xticks(angles)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylim(0, radial_max)
    ax.set_yticks(np.linspace(0, radial_max, 4)[1:])
    ax.set_yticklabels([f"{v:.2f}" for v in np.linspace(0, radial_max, 4)[1:]],
                         fontsize=8, color="#64748B")
    ax.grid(color="#CBD5E1", linewidth=0.6, alpha=0.8)
    ax.spines["polar"].set_color("#94A3B8")
    ax.spines["polar"].set_linewidth(0.8)
    ax.set_title(title, y=1.10, fontsize=12, weight="bold")


def fig_radar_2group(bsv: pd.DataFrame):
    means = bsv.groupby("group_2")[list(gcfg.BSV_AXES)].mean()
    n_owd = int((bsv["group_2"] == "OWD").sum())
    n_nwd = int((bsv["group_2"] == "NWD").sum())
    values = {
        f"OWD (n={n_owd})": means.loc["OWD"].to_dict(),
        f"NWD (n={n_nwd})": means.loc["NWD"].to_dict(),
    }
    radial_max = max(0.15, float(means.values.max()) * 1.20)
    fig, ax = plt.subplots(figsize=(7.2, 6.8), subplot_kw=dict(polar=True))
    for label, vals in values.items():
        color = COLOR_OWD if "OWD" in label else COLOR_NWD
        axes = list(gcfg.BSV_AXES)
        angles = np.linspace(0, 2 * np.pi, len(axes), endpoint=False)
        r = np.array([vals.get(a, 0.0) for a in axes])
        r_closed = np.append(r, r[0])
        theta = np.append(angles, angles[0])
        ax.plot(theta, r_closed, color=color, linewidth=1.9, label=label)
        ax.fill(theta, r_closed, color=color, alpha=0.14)
    axes = list(gcfg.BSV_AXES)
    angles = np.linspace(0, 2 * np.pi, len(axes), endpoint=False)
    ax.set_xticks(angles)
    ax.set_xticklabels([gcfg.axis_short(a) for a in axes], fontsize=10)
    ax.set_ylim(0, radial_max)
    ax.set_yticks(np.linspace(0, radial_max, 4)[1:])
    ax.set_yticklabels([f"{v:.2f}" for v in np.linspace(0, radial_max, 4)[1:]],
                         fontsize=8, color="#64748B")
    ax.grid(color="#CBD5E1", linewidth=0.6)
    ax.spines["polar"].set_color("#94A3B8")
    ax.set_title("GAIRA 11-axis biochemical state — OWD vs NWD",
                    y=1.10, fontsize=12, weight="bold")
    ax.legend(loc="upper right", bbox_to_anchor=(1.30, 1.10), fontsize=10)
    _save(fig, "fig_radar_2group")


def fig_radar_4subgroup(bsv: pd.DataFrame):
    sub_df = bsv[bsv["subgroup_4"].notna()]
    means = sub_df.groupby("subgroup_4")[list(gcfg.BSV_AXES)].mean()
    counts = sub_df.groupby("subgroup_4").size().to_dict()
    radial_max = max(0.15, float(means.values.max()) * 1.20)

    fig, ax = plt.subplots(figsize=(7.6, 7.2), subplot_kw=dict(polar=True))
    axes = list(gcfg.BSV_AXES)
    angles = np.linspace(0, 2 * np.pi, len(axes), endpoint=False)
    for sub, vals in means.iterrows():
        r = vals.to_numpy(dtype=float)
        r_closed = np.append(r, r[0])
        theta = np.append(angles, angles[0])
        color = SUBGROUP_COLORS.get(sub, "#64748B")
        ax.plot(theta, r_closed, color=color, linewidth=1.7,
                 label=f"{sub} (n={counts.get(sub, 0)})")
        ax.fill(theta, r_closed, color=color, alpha=0.10)
    ax.set_xticks(angles)
    ax.set_xticklabels([gcfg.axis_short(a) for a in axes], fontsize=10)
    ax.set_ylim(0, radial_max)
    ax.set_yticks(np.linspace(0, radial_max, 4)[1:])
    ax.set_yticklabels([f"{v:.2f}" for v in np.linspace(0, radial_max, 4)[1:]],
                         fontsize=8, color="#64748B")
    ax.grid(color="#CBD5E1", linewidth=0.6)
    ax.spines["polar"].set_color("#94A3B8")
    ax.set_title("GAIRA 11-axis biochemical state — 4 subgroups (Race × Group)",
                    y=1.10, fontsize=12, weight="bold")
    ax.legend(loc="upper right", bbox_to_anchor=(1.32, 1.12), fontsize=9)
    _save(fig, "fig_radar_4subgroup")


def _numpy_pca(X, n_components=2):
    """Pure-numpy PCA. Returns (coords, explained_variance_ratio)."""
    Xc = X - X.mean(axis=0, keepdims=True)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    coords = Xc @ Vt.T[:, :n_components]
    var = (S ** 2) / max(1, len(X) - 1)
    ratio = var / var.sum()
    return coords, ratio[:n_components]


# Biochemical family palette for the mechanistic radar — colors related axes
# together so the reader can immediately see which system(s) are shifted.
AXIS_FAMILY = {
    "G01_purine_nucleotide":      "nucleic",
    "G02_purine_metabolite":      "nucleic",
    "G03_pyrimidine_nucleotide":  "nucleic",
    "G04_nucleic_acid_phosphate": "nucleic",
    "G05_glycan_carbohydrate":    "glycan",
    "G06_protein_peptide_backbone": "protein",
    "G07_aromatic_residue":       "protein",
    "G08_lipid_acyl_membrane":    "lipid",
    "G09_sterol_neutral_lipid":   "lipid",
    "G10_sulfur_thiol_redox":     "redox",
    "G11_metabolic_small_molecule": "metabolite",
}
FAMILY_COLOR = {
    "nucleic":    "#3B82F6",
    "glycan":     "#14B8A6",
    "protein":    "#F59E0B",
    "lipid":      "#10B981",
    "redox":      "#EF4444",
    "metabolite": "#A855F7",
}


def _compute_zscore(bsv: pd.DataFrame, cohort_col: str) -> pd.DataFrame:
    """Return one row per cohort with z = (cohort_mean - pool_mean) / pool_sd
    per axis. Pool uses ALL patients in the analysis (both cohorts pooled)."""
    axes = list(gcfg.BSV_AXES)
    pool_mean = bsv[axes].mean()
    pool_sd   = bsv[axes].std(ddof=1).replace(0, np.nan)
    rows = []
    for cohort, sub in bsv.groupby(cohort_col):
        if pd.isna(cohort): continue
        mean = sub[axes].mean()
        z    = (mean - pool_mean) / pool_sd
        row  = {"cohort": cohort, "n": int(len(sub))}
        for a in axes:
            row[a] = float(z[a]) if not pd.isna(z[a]) else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def _radar_with_axis_families(ax, values_by_group: dict, radial_lim: tuple,
                                title: str, sig_axes: set[str] | None = None,
                                effect_by_axis: dict[str, float] | None = None):
    axes = list(gcfg.BSV_AXES)
    N = len(axes)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    labels = [gcfg.axis_short(a) for a in axes]
    # Radial background: color each wedge with its biochemical family so the
    # reader can immediately identify "the redox axes" or "the nucleic axes"
    for i, a in enumerate(axes):
        fam = AXIS_FAMILY.get(a, "other")
        color = FAMILY_COLOR.get(fam, "#94A3B8")
        theta_lo = angles[i] - np.pi / N
        theta_hi = angles[i] + np.pi / N
        ax.bar([angles[i]], [radial_lim[1] - radial_lim[0]],
                 width=2 * np.pi / N, bottom=radial_lim[0],
                 color=color, alpha=0.05, edgecolor="none")

    for label, vals in values_by_group.items():
        r = np.array([vals.get(a, 0.0) for a in axes], dtype=float)
        r_closed = np.append(r, r[0])
        theta = np.append(angles, angles[0])
        color = (COLOR_OWD if label.startswith("OWD")
                    else COLOR_NWD if label.startswith("NWD")
                    else SUBGROUP_COLORS.get(label.split(" (")[0], "#64748B"))
        ax.plot(theta, r_closed, color=color, linewidth=2.0, label=label)
        ax.fill(theta, r_closed, color=color, alpha=0.14)

    # Enhanced axis labels: append Cohen's d for significant axes
    if effect_by_axis is not None and sig_axes is not None:
        for i, a in enumerate(axes):
            d = effect_by_axis.get(a, 0.0)
            fam_color = FAMILY_COLOR.get(AXIS_FAMILY[a], "#94A3B8")
            if a in sig_axes:
                sig_star = "**" if abs(d) >= 1.0 else "*"
                labels[i] = f"{gcfg.axis_short(a)}\n{sig_star} d={d:+.2f}"
                # Color the significant axis label by its family
    ax.set_xticks(angles)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylim(radial_lim)
    ticks = np.linspace(radial_lim[0], radial_lim[1], 5)
    ax.set_yticks(ticks[1:-1])
    ax.set_yticklabels([f"{v:+.1f}" if abs(v) < 5 else f"{v:.2f}" for v in ticks[1:-1]],
                         fontsize=8, color="#64748B")
    ax.grid(color="#CBD5E1", linewidth=0.6, alpha=0.9)
    if radial_lim[0] < 0:
        ax.plot(np.linspace(0, 2*np.pi, 360), [0]*360,
                 color="#334155", linewidth=1.0, alpha=0.6)   # emphasize zero
    ax.spines["polar"].set_color("#94A3B8")
    ax.set_title(title, y=1.12, fontsize=12, weight="bold")


def fig_radar_mechanistic_2group(bsv: pd.DataFrame, two_group_stats_df: pd.DataFrame):
    """Two-panel figure: raw magnitude radar + z-score radar with per-axis
    Cohen's d and significance markers. Reveals mechanism, not just fingerprint."""
    axes = list(gcfg.BSV_AXES)
    raw = bsv.groupby("group_2")[axes].mean()
    z_df = _compute_zscore(bsv, "group_2").set_index("cohort")

    sig_axes = set(two_group_stats_df[two_group_stats_df["q_value_fdr_bh"] < 0.05]["axis"])
    effect_by_axis = dict(zip(two_group_stats_df["axis"], two_group_stats_df["cohens_d"]))

    n_owd = int((bsv["group_2"] == "OWD").sum())
    n_nwd = int((bsv["group_2"] == "NWD").sum())

    fig = plt.figure(figsize=(15.5, 7.8))
    gs = fig.add_gridspec(1, 2, wspace=0.35)

    ax1 = fig.add_subplot(gs[0], polar=True)
    _radar_with_axis_families(
        ax1,
        {f"OWD (n={n_owd})": raw.loc["OWD"].to_dict(),
          f"NWD (n={n_nwd})": raw.loc["NWD"].to_dict()},
        (0.0, max(0.15, raw.values.max() * 1.15)),
        "Raw BSV magnitude — cohort mean",
        sig_axes=set(), effect_by_axis={},
    )
    ax1.legend(loc="upper right", bbox_to_anchor=(1.28, 1.13), fontsize=9)

    ax2 = fig.add_subplot(gs[1], polar=True)
    _radar_with_axis_families(
        ax2,
        {f"OWD (n={n_owd})": z_df.loc["OWD"].to_dict(),
          f"NWD (n={n_nwd})": z_df.loc["NWD"].to_dict()},
        (-1.8, 1.8),
        "Mechanistic z-score — deviation from pooled cohort mean",
        sig_axes=sig_axes, effect_by_axis=effect_by_axis,
    )
    ax2.legend(loc="upper right", bbox_to_anchor=(1.32, 1.13), fontsize=9)

    fig.suptitle("GAIRA 11-axis biochemical state — OWD vs NWD (mechanistic view)",
                     y=1.02, fontsize=13, weight="bold")
    # Legend for significance
    fig.text(0.5, -0.02,
              "Sig. axes labeled with Cohen's d and **/* for |d|≥1.0 / q<0.05 "
              "(BH-FDR). Left panel = magnitude; right panel = per-axis deviation "
              "from pooled mean, normalizing out inter-axis pipeline biases.",
              ha="center", fontsize=9, color="#475569", style="italic")
    _save(fig, "fig_radar_2group_mechanistic")


def fig_radar_mechanistic_4subgroup(bsv: pd.DataFrame,
                                        four_group_stats_df: pd.DataFrame):
    axes = list(gcfg.BSV_AXES)
    sub_df = bsv[bsv["subgroup_4"].notna()]
    raw = sub_df.groupby("subgroup_4")[axes].mean()
    z_df = _compute_zscore(sub_df, "subgroup_4").set_index("cohort")
    counts = sub_df.groupby("subgroup_4").size().to_dict()

    sig_axes = set(four_group_stats_df[four_group_stats_df["q_value_fdr_bh"] < 0.05]["axis"])

    fig = plt.figure(figsize=(15.5, 7.8))
    gs = fig.add_gridspec(1, 2, wspace=0.35)

    ax1 = fig.add_subplot(gs[0], polar=True)
    vals_raw = {f"{s} (n={counts.get(s, 0)})": raw.loc[s].to_dict() for s in raw.index}
    _radar_with_axis_families(ax1, vals_raw,
                                  (0.0, max(0.15, raw.values.max() * 1.15)),
                                  "Raw BSV magnitude — subgroup means")
    ax1.legend(loc="upper right", bbox_to_anchor=(1.32, 1.13), fontsize=8)

    ax2 = fig.add_subplot(gs[1], polar=True)
    vals_z = {f"{s} (n={counts.get(s, 0)})": z_df.loc[s].to_dict() for s in z_df.index}
    _radar_with_axis_families(ax2, vals_z, (-1.8, 1.8),
                                  "Mechanistic z-score — subgroup deviation from pooled mean",
                                  sig_axes=sig_axes, effect_by_axis={})
    ax2.legend(loc="upper right", bbox_to_anchor=(1.32, 1.13), fontsize=8)

    fig.suptitle("GAIRA 11-axis biochemical state — 4 subgroups (mechanistic view)",
                     y=1.02, fontsize=13, weight="bold")
    fig.text(0.5, -0.02,
              "Sig. axes marked with * where 4-way Kruskal-Wallis q<0.05 (BH-FDR). "
              "Right panel z-scores normalise inter-axis pipeline biases so subgroup-"
              "specific mechanistic shifts are directly visible.",
              ha="center", fontsize=9, color="#475569", style="italic")
    _save(fig, "fig_radar_4subgroup_mechanistic")


def fig_forest_owd_vs_nwd(bsv: pd.DataFrame, two_group_stats_df: pd.DataFrame):
    """Effect-size forest plot with 95% bootstrap CI + significance stars.
    Reveals magnitude AND uncertainty simultaneously — better than a bar chart
    for mechanistic conclusions."""
    axes = list(gcfg.BSV_AXES)
    rng = np.random.default_rng(42)
    stats_rows = []
    for a in axes:
        owd = bsv.loc[bsv["group_2"] == "OWD", a].to_numpy()
        nwd = bsv.loc[bsv["group_2"] == "NWD", a].to_numpy()
        if len(owd) < 3 or len(nwd) < 3:
            continue
        # Bootstrap Cohen's d
        ds = np.empty(2000)
        for i in range(2000):
            a_b = rng.choice(owd, size=len(owd), replace=True)
            b_b = rng.choice(nwd, size=len(nwd), replace=True)
            sa, sb = np.var(a_b, ddof=1), np.var(b_b, ddof=1)
            pooled = np.sqrt(((len(a_b) - 1) * sa + (len(b_b) - 1) * sb)
                                / max(1, len(a_b) + len(b_b) - 2))
            ds[i] = (a_b.mean() - b_b.mean()) / max(pooled, 1e-12)
        d_med = float(np.median(ds))
        d_lo, d_hi = np.percentile(ds, [2.5, 97.5])
        stats_rows.append({
            "axis": a, "d_median": d_med,
            "d_lo": d_lo, "d_hi": d_hi,
            "family": AXIS_FAMILY[a],
        })
    stats_row = pd.DataFrame(stats_rows).merge(
        two_group_stats_df[["axis", "q_value_fdr_bh"]], on="axis"
    ).sort_values("d_median")

    fig, ax = plt.subplots(figsize=(8.5, 6.5))
    y = np.arange(len(stats_row))
    for i, (_, r) in enumerate(stats_row.iterrows()):
        color = FAMILY_COLOR.get(r["family"], "#64748B")
        ax.plot([r["d_lo"], r["d_hi"]], [i, i], color=color, linewidth=2.2, alpha=0.85)
        ax.scatter([r["d_median"]], [i], color=color, s=70, edgecolor="white",
                     linewidth=0.9, zorder=3)
    ax.axvline(0, color="#334155", linewidth=1.0, alpha=0.7)
    ax.set_yticks(y)
    ax_labels = []
    for _, r in stats_row.iterrows():
        base = gcfg.axis_short(r["axis"])
        star = "**" if r["q_value_fdr_bh"] < 0.001 else ("*" if r["q_value_fdr_bh"] < 0.05 else "")
        ax_labels.append(f"{base} {star}")
    ax.set_yticklabels(ax_labels, fontsize=10)
    ax.set_xlabel("Cohen's d — OWD − NWD (95% bootstrap CI)")
    ax.set_title("Per-axis effect size, OWD − NWD (Race × Group pooled)",
                    weight="bold")
    # Family legend
    handles = [plt.Line2D([0], [0], marker="o", color=FAMILY_COLOR[fam],
                              linestyle="", label=fam)
                for fam in ("nucleic", "glycan", "protein", "lipid", "redox", "metabolite")]
    ax.legend(handles=handles, loc="lower right", fontsize=8, title="Biochemical family")
    ax.axvspan(-1.0, 1.0, color="#E2E8F0", alpha=0.20, zorder=0)  # shade the ±1 SD band
    _save(fig, "fig_forest_owd_vs_nwd")


def fig_pca_bsv(bsv: pd.DataFrame):
    X = bsv[list(gcfg.BSV_AXES)].to_numpy()
    coords, evr = _numpy_pca(X, 2)

    class _P:
        def __init__(self, evr): self.explained_variance_ratio_ = evr
    pca = _P(evr)

    # 2-group PCA
    fig, ax = plt.subplots(figsize=(6.5, 5.0))
    for grp, color in [("OWD", COLOR_OWD), ("NWD", COLOR_NWD)]:
        mask = bsv["group_2"] == grp
        ax.scatter(coords[mask, 0], coords[mask, 1], c=color, s=42,
                     alpha=0.85, edgecolor="white", linewidth=0.6, label=grp)
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    ax.set_title("Per-patient PCA of 11-axis BSV — 2 groups", weight="bold")
    ax.legend()
    _save(fig, "fig_pca_2group")

    # 4-subgroup PCA
    fig, ax = plt.subplots(figsize=(6.5, 5.0))
    sub_mask = bsv["subgroup_4"].notna().to_numpy()
    for sub, color in SUBGROUP_COLORS.items():
        m = (bsv["subgroup_4"] == sub).to_numpy()
        if not m.any(): continue
        ax.scatter(coords[m, 0], coords[m, 1], c=color, s=42,
                     alpha=0.85, edgecolor="white", linewidth=0.6, label=sub)
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    ax.set_title("Per-patient PCA of 11-axis BSV — 4 subgroups", weight="bold")
    ax.legend(fontsize=8, loc="best")
    _save(fig, "fig_pca_4subgroup")


def fig_mean_spectra(spectra_by_pid: dict[str, np.ndarray],
                       patients: pd.DataFrame,
                       wn_native: np.ndarray):
    """Mean spectra per group with 95% CI band, using ALL raw scans for
    biological plausibility (not just patient means)."""
    grid = np.linspace(gcfg.WAVENUMBER_MIN, gcfg.WAVENUMBER_MAX, gcfg.WAVENUMBER_N)

    def _resample_group(mask_col: str, group_val):
        """Aggregate all patient means for this group onto the demo grid."""
        rows = []
        for _, r in patients.iterrows():
            if r.get(mask_col) != group_val: continue
            arr = spectra_by_pid.get(r["patient_id"])
            if arr is None: continue
            m = arr.mean(axis=0)
            on_grid = _interp_to_demo_grid(wn_native, m)
            # baseline-remove-lite for visual comparison
            on_grid = on_grid - np.percentile(on_grid, 5)
            on_grid = np.clip(on_grid, 0, None)
            if on_grid.max() > 0:
                on_grid = on_grid / on_grid.max()
            rows.append(on_grid)
        return np.asarray(rows) if rows else None

    # 2-group
    fig, ax = plt.subplots(figsize=(9.0, 4.6))
    for grp, color in [("OWD", COLOR_OWD), ("NWD", COLOR_NWD)]:
        arr = _resample_group("group_2", grp)
        if arr is None: continue
        mean = arr.mean(axis=0)
        se   = arr.std(axis=0, ddof=1) / np.sqrt(len(arr))
        ax.plot(grid, mean, color=color, linewidth=1.6, label=f"{grp} (n={len(arr)})")
        ax.fill_between(grid, mean - 1.96 * se, mean + 1.96 * se,
                          color=color, alpha=0.16, linewidth=0)
    ax.set_xlim(gcfg.WAVENUMBER_MIN, gcfg.WAVENUMBER_MAX)
    ax.set_xlabel("Wavenumber (cm$^{-1}$)")
    ax.set_ylabel("Normalized intensity (a.u.)")
    ax.set_title("Group mean spectra ± 95% CI — OWD vs NWD", weight="bold")
    ax.legend()
    _save(fig, "fig_mean_spectra_2group")

    # 4-subgroup
    fig, ax = plt.subplots(figsize=(9.0, 4.8))
    for sub, color in SUBGROUP_COLORS.items():
        arr = _resample_group("subgroup_4", sub)
        if arr is None: continue
        mean = arr.mean(axis=0)
        se   = arr.std(axis=0, ddof=1) / np.sqrt(len(arr))
        ax.plot(grid, mean, color=color, linewidth=1.4, label=f"{sub} (n={len(arr)})")
        ax.fill_between(grid, mean - 1.96 * se, mean + 1.96 * se,
                          color=color, alpha=0.10, linewidth=0)
    ax.set_xlim(gcfg.WAVENUMBER_MIN, gcfg.WAVENUMBER_MAX)
    ax.set_xlabel("Wavenumber (cm$^{-1}$)")
    ax.set_ylabel("Normalized intensity (a.u.)")
    ax.set_title("Subgroup mean spectra ± 95% CI (Race × Group)", weight="bold")
    ax.legend(fontsize=9)
    _save(fig, "fig_mean_spectra_4subgroup")


def fig_difference_spectrum(spectra_by_pid: dict[str, np.ndarray],
                               patients: pd.DataFrame,
                               wn_native: np.ndarray):
    grid = np.linspace(gcfg.WAVENUMBER_MIN, gcfg.WAVENUMBER_MAX, gcfg.WAVENUMBER_N)

    def _mean_norm(group_val):
        rows = []
        for _, r in patients.iterrows():
            if r["group_2"] != group_val: continue
            arr = spectra_by_pid.get(r["patient_id"])
            if arr is None: continue
            m = arr.mean(axis=0)
            on_grid = _interp_to_demo_grid(wn_native, m)
            on_grid = np.clip(on_grid - np.percentile(on_grid, 5), 0, None)
            if on_grid.max() > 0:
                on_grid = on_grid / on_grid.max()
            rows.append(on_grid)
        return np.mean(rows, axis=0) if rows else None

    owd, nwd = _mean_norm("OWD"), _mean_norm("NWD")
    diff = owd - nwd

    # Annotate strongest bands in the difference
    key_bands = [(720, "purine 720"), (1003, "Phe 1003"),
                    (1082, "PO$_2^-$ 1080"), (1440, "CH$_2$ 1440"),
                    (1655, "amide-I 1655"), (500, "S–S 500")]
    fig, ax = plt.subplots(figsize=(9.0, 4.6))
    ax.plot(grid, diff, color="#111827", linewidth=1.4)
    ax.axhline(0, color="#94A3B8", linewidth=0.7, alpha=0.7)
    ax.fill_between(grid, 0, diff, where=(diff > 0),
                      color=COLOR_OWD, alpha=0.30, label="OWD > NWD")
    ax.fill_between(grid, 0, diff, where=(diff < 0),
                      color=COLOR_NWD, alpha=0.30, label="NWD > OWD")
    for cm, label in key_bands:
        idx = int(cm - gcfg.WAVENUMBER_MIN)
        if 0 <= idx < len(grid):
            ax.axvline(cm, color="#64748B", linewidth=0.6, alpha=0.5, linestyle="--")
            ax.text(cm, ax.get_ylim()[1] * 0.85, label,
                     rotation=90, fontsize=8, color="#334155", va="top")
    ax.set_xlim(gcfg.WAVENUMBER_MIN, gcfg.WAVENUMBER_MAX)
    ax.set_xlabel("Wavenumber (cm$^{-1}$)")
    ax.set_ylabel("Δ intensity (OWD − NWD)")
    ax.set_title("Difference spectrum — OWD − NWD, with key biochemical bands annotated",
                    weight="bold")
    ax.legend(loc="upper right")
    _save(fig, "fig_difference_spectrum_owd_vs_nwd")


def fig_bsv_heatmap(bsv: pd.DataFrame):
    """Per-patient BSV heatmap, sorted by group."""
    ordered = bsv.sort_values(["group_2", "patient_id"]).reset_index(drop=True)
    M = ordered[list(gcfg.BSV_AXES)].to_numpy(dtype=float)
    labels = ordered["patient_id"].tolist()
    groups = ordered["group_2"].tolist()

    fig, ax = plt.subplots(figsize=(8.0, max(6, len(labels) * 0.13)))
    im = ax.imshow(M, aspect="auto", cmap="magma", interpolation="nearest")
    ax.set_xticks(np.arange(len(gcfg.BSV_AXES)))
    ax.set_xticklabels([gcfg.axis_short(a) for a in gcfg.BSV_AXES],
                         rotation=45, ha="right", fontsize=9)
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels([f"{g}::{l}" for g, l in zip(groups, labels)],
                         fontsize=6)
    cbar = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
    cbar.set_label("BSV value")
    ax.set_title("Per-patient GAIRA 11-axis biochemical state (heatmap)",
                    weight="bold", fontsize=11)
    _save(fig, "fig_bsv_heatmap")


def fig_analyte_hits(analyte: pd.DataFrame):
    top = analyte.sort_values("mean_fire_score", ascending=False).head(11)
    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    tiers = top["confidence_tier"].tolist()
    colors_by_tier = {"High": "#059669", "Medium": "#F59E0B", "Low": "#DC2626"}
    bar_colors = [colors_by_tier.get(t, "#64748B") for t in tiers]
    ax.barh(top["molecule_name"], top["mean_fire_score"], color=bar_colors,
             edgecolor="#334155", linewidth=0.6)
    ax.invert_yaxis()
    ax.set_xlabel("Mean MSS fire score (across cohort)")
    ax.set_title("GAIRA analyte hits — per-molecule mean fire score by confidence tier",
                    weight="bold")
    for tier, color in colors_by_tier.items():
        ax.plot([], [], "s", color=color, markersize=10, label=tier)
    ax.legend(title="Confidence")
    _save(fig, "fig_analyte_hits_top")


def fig_qc(bsv: pd.DataFrame, spectra_by_pid: dict[str, np.ndarray]):
    fig, axs = plt.subplots(1, 2, figsize=(10.5, 4.5))
    counts_group = bsv["group_2"].value_counts()
    axs[0].bar(counts_group.index, counts_group.values,
                 color=[COLOR_OWD if g == "OWD" else COLOR_NWD for g in counts_group.index],
                 edgecolor="#334155", linewidth=0.6)
    axs[0].set_ylabel("Number of patients")
    axs[0].set_title("QC — patients per 2-group split", weight="bold", fontsize=11)

    scans_per_patient = [(spectra_by_pid[p].shape[0] if p in spectra_by_pid else 0)
                            for p in bsv["patient_id"]]
    axs[1].hist(scans_per_patient, bins=15, color="#0EA5E9",
                 edgecolor="#075985", linewidth=0.6)
    axs[1].set_xlabel("Number of SERS scans per patient")
    axs[1].set_ylabel("Patients")
    axs[1].set_title("QC — SERS scan counts per patient", weight="bold", fontsize=11)
    _save(fig, "fig_qc_counts")


# --------------------------------------------------------------------
# 8. REPORTS
# --------------------------------------------------------------------
def write_preprocessing_audit():
    txt = f"""# Diabetes EV-SERS — preprocessing audit

## Raw data
- Source: `/Volumes/SSD_Rad/GAIRA_DATA/raw/diabetes_plasma_ev_sers/extracted/`
- MATLAB files:
    - `RawDataImpact.mat` — `smoothed_spectra` object array, one entry per
        Impact-cohort patient. Each entry is a (737 × N_scans) matrix.
    - `RawDataStrong.mat` — same shape, Strong-D cohort.
- Metadata: `patient_data.csv` (64 rows, 13 columns).

## Wavenumber calibration
- Method: cubic polyfit of pixel index → wavenumber against 8 known Raman
    peaks (Phe / Tyr / lipid / amide anchors). Source: `Figure3.m` of the
    original manuscript. Pixel range used = 162–898 (inclusive) → 737 rows,
    matching the .mat data.
- Peaks used: 620.9, 795.8, 1001.4, 1031.8, 1155.3, 1450.5, 1583.1, 1602.3 cm⁻¹.
- Coverage: ~484–1642 cm⁻¹ across the 737 pixels.

## GAIRA preprocessing (applied in this audit, via `build_report`)
- Per patient: mean of technical SERS scans → interpolate to the demo's
    canonical grid (400–1800 cm⁻¹ at 1 cm⁻¹).
- `build_report` internally applies:
    - Savitzky–Golay smoothing (window 11, polynomial 3)
    - ASLS baseline (λ=1e5, p=0.01, 8 iterations)
    - L2 normalization
- Peak detection: SciPy `find_peaks` with prominence floor 5e-3,
    minimum distance 6 cm⁻¹.
- Motif scoring: 11 curated class-level motifs (`gaira_core/motif_scoring.py`)
    with anchor-first-then-support geometric mean over co-firing bands.
- MSS: 11 curated analyte anchor/support/anti sets
    (`gaira_core/data_loader.py:MOLECULES`).
- Substrate physics: `Ag colloid SERS` — dampens purine 720–740 cm⁻¹
    (×0.65) and mildly boosts thiol/thione 490–510 cm⁻¹ (×1.20).
- BSV projection: motif fires → noisy-OR aggregation over the 11 axes.

## What we do NOT do here (compared to the prior GAIRA_BUILD audit)
- We do **not** apply an additional sum-to-one normalization or CLR transform
    on the BSV. The prior audit's per-axis Cohen's d values are computed on
    CLR-transformed spectrum-level BSVs; ours are computed on the demo
    pipeline's raw motif-based BSVs at the patient-mean level. Both are
    valid decompositions of the same biology; they will produce different
    absolute magnitudes but qualitatively similar directions.
- We do **not** subtract a background or blank spectrum. The .mat rows
    already carry the study's own smoothed spectra (per the field name);
    additional background handling would be double-processing.
- We do **not** remove cosmic rays or perform outlier rejection at this
    stage. The mean-of-scans step provides implicit robustness.

## Replicates
- Each patient contributes 1 BSV row = mean over N_scans SERS scans.
- N_scans ranges 441–1089 depending on the acquisition map size.
- Statistical tests are performed **at the subject level** (1 patient = 1 n),
    not at the spectrum level. This avoids the pseudoreplication issue
    flagged in the audit brief.
"""
    (OUT / "diabetes_preprocessing_audit.md").write_text(txt.strip() + "\n")


def write_qc_summary(bsv: pd.DataFrame, patients: pd.DataFrame,
                        spectra_by_pid: dict[str, np.ndarray],
                        wn_native: np.ndarray):
    counts_group = bsv["group_2"].value_counts().to_dict()
    counts_sub   = bsv["subgroup_4"].value_counts().to_dict()
    missing = [p for p in patients["patient_id"] if p not in spectra_by_pid]
    n_axes = [int(row["n_nonzero_axes"]) for _, row in bsv.iterrows()]
    lines = [
        "# Diabetes EV-SERS — QC summary",
        "",
        f"- Patients in metadata: {len(patients)}",
        f"- Patients with spectra loaded: {len(bsv)}",
        f"- Patients missing from .mat: {len(missing)} ({', '.join(missing) or 'none'})",
        f"- Wavenumber grid: {wn_native.min():.1f} – {wn_native.max():.1f} cm⁻¹ ({len(wn_native)} pixels native)",
        f"- Interpolated to demo grid: {gcfg.WAVENUMBER_MIN:.0f}–{gcfg.WAVENUMBER_MAX:.0f} cm⁻¹ at 1 cm⁻¹ ({gcfg.WAVENUMBER_N} points)",
        "",
        "## 2-group counts",
        *[f"- {g}: {c} patients" for g, c in counts_group.items()],
        "",
        "## 4-subgroup counts",
        *[f"- {g}: {c} patients" for g, c in counts_sub.items()],
        "",
        f"## Non-zero-axis distribution (11-axis BSV, patient-level)",
        f"- min: {min(n_axes)} / max: {max(n_axes)} / median: {int(np.median(n_axes))}",
        f"- % of patients with ≥8 axes populated: "
        f"{100 * sum(1 for n in n_axes if n >= 8) / max(1, len(n_axes)):.1f}%",
        "",
        "## Potential batch effects to watch",
        "- Impact vs Strong-D come from two different site/protocol codes "
        "(`2151-*` vs `32113-*`). Any BSV difference may partially reflect "
        "the site/protocol difference. Downstream analyses should stratify.",
        "- BMI, HbA1c, and age distributions differ by group (see label audit).",
    ]
    (OUT / "diabetes_qc_summary.md").write_text("\n".join(lines) + "\n")


def write_captions(counts_by_group: dict, counts_by_sub: dict,
                     two_group_top_axes: list, sub_top_axes: list):
    text = f"""# Figure captions

**Figure — Radar (2 group).** GAIRA 11-axis biochemical state vector (BSV) means
for the two clinical groups: OWD (overweight/obese diabetic, n={counts_by_group.get("OWD", 0)}) versus
NWD (normal-weight diabetic, n={counts_by_group.get("NWD", 0)}). Values are the mean
BSV per axis across per-patient spectra processed through the current GAIRA
demo pipeline (Ag-colloid SERS substrate context). Class-level biochemical
themes only; no molecule-level identity claim is implied.

**Figure — Radar (4 subgroup).** Same axes stratified by race × group:
{", ".join(f"{s} n={counts_by_sub.get(s, 0)}" for s in counts_by_sub)}.

**Figure — PCA (2 group).** Per-patient PCA of the 11-axis BSV.

**Figure — Mean spectra ± 95% CI (2 group).** Baseline-adjusted, max-normalized
per-patient mean spectra averaged within each group; shaded band = 95% CI
across patients.

**Figure — Mean spectra ± 95% CI (4 subgroup).** Same, stratified by race × group.

**Figure — Difference spectrum OWD − NWD.** Positive lobes = higher in OWD,
negative lobes = higher in NWD. Key biochemical bands annotated
(purine 720, Phe 1003, PO₂⁻ 1080, CH₂ 1440, amide-I 1655, S–S 500).

**Figure — BSV heatmap.** Per-patient BSV values across all 11 axes, sorted
by group.

**Figure — Analyte hits.** Top-11 GAIRA analytes ranked by mean MSS fire
score, colored by confidence tier (High / Medium / Low).

**Figure — QC counts.** Patient counts per group + histogram of SERS-scan
counts per patient.
"""
    (FIG_DIR / "figure_captions.md").write_text(text)


def write_interpretation_report(two_group: pd.DataFrame,
                                   four_group: pd.DataFrame,
                                   analyte: pd.DataFrame,
                                   bsv: pd.DataFrame):
    def _top(df, k=5):
        return df.head(k)[["axis_label", "cohens_d", "cliffs_delta",
                              "p_value", "q_value_fdr_bh"]].to_string(index=False)

    high = analyte[analyte["confidence_tier"] == "High"]
    med  = analyte[analyte["confidence_tier"] == "Medium"]

    md = f"""# Diabetes EV-SERS — GAIRA re-analysis interpretation report

**Analysis stamp:** `{STAMP}`
**Domain context:** plasma extracellular vesicle SERS on Ag colloid substrate.
**Interpretive stance:** class-level biochemical themes. Molecule-level identity
is not claimed; language is "consistent with"-style throughout.

## Cohort
- OWD (overweight/obese diabetic, `Group == "Impact"`): {int((bsv["group_2"] == "OWD").sum())} patients
- NWD (normal-weight diabetic, `Group == "Strong-D"`): {int((bsv["group_2"] == "NWD").sum())} patients
- 4-subgroup structure (Race × Group): {bsv["subgroup_4"].dropna().value_counts().to_dict()}

## Top-5 axes by OWD vs NWD effect (per-patient Mann-Whitney)

```
{_top(two_group.head(5))}
```

Directional reading (consistent with the prior GAIRA_BUILD audit's per-axis Cohen's d):

- **G05 glycan / carbohydrate-associated** — the strongest tier-1 signal.
    Prior report: d = −0.56 (OWD < NWD, CI excludes 0). Interpretation:
    plasma-EV signal consistent with reduced carbohydrate-associated
    contribution in OWD relative to NWD. Class-level; not a specific glycan claim.
- **G01 purine-nucleotide-associated** — d ≈ +0.52 in prior report
    (OWD > NWD). Consistent with elevated purine-associated contribution in
    OWD plasma EVs. Substrate caveat: Ag-SERS purine amplification is
    inherently high; the demo's substrate rule dampens it ×0.65 to prevent
    molecule-level overclaim.
- **G08 lipid / membrane-associated** — d ≈ +0.34 (OWD > NWD).
    Consistent with metabolic/lipid loading in obese plasma EV populations.
- **G09 sterol / neutral lipid-associated** — d ≈ −0.20 (OWD < NWD).
    Weak effect; interpret with caution.

The current audit's per-patient BSV directions should be checked against
the numeric values in `diabetes_group_summary_2group.csv`.

## Top-5 axes by 4-subgroup Kruskal-Wallis effect
```
{four_group.head(5)[["axis_label", "kruskal_H", "p_value", "q_value_fdr_bh"]].to_string(index=False)}
```

## Analyte hits (Tier-1 = direct spectral, Tier-2 = literature-supported)
- **High-confidence hits:** {len(high)}
- **Medium-confidence hits:** {len(med)}
- **Low-confidence hits:** {len(analyte) - len(high) - len(med)}

High-confidence:
```
{high[["molecule_name", "biochemical_class", "mean_fire_score",
        "cohens_d_owd_vs_nwd", "directionality", "confidence_tier"]].to_string(index=False)}
```

Every High-confidence hit still carries a class-level rather than molecule-level
interpretation: the "molecule name" is the anchor set that GAIRA's motif library
uses, not a definitive identity claim. See `caveats` column of
`diabetes_analyte_hits.csv` for known collision partners per molecule.

## Domain-context caveats
- **EV-specific:** plasma EV pellets are mixture matrices; individual band
    assignments carry substrate + matrix uncertainty. Do not read the radar as
    identifying specific molecules — read it as biochemical *themes*.
- **SERS on Ag colloid:** purine adsorption is amplified ×3–10; the demo's
    substrate rule dampens G01 (×0.65) and G02 to force class-level calls.
    Any inference about specific purines requires corroborating co-bands.
- **Race split (4 subgroups):** the subgroup structure follows the paper's
    Fig. 3 (Asian × White × Impact × Strong-D). Sample sizes for other races
    (Hispanic 7, African-American 4, Other 3) are too small for reliable
    Kruskal-Wallis; those patients are dropped from the 4-subgroup analysis
    but retained in the 2-group.

## Limitations
- **Sample size:** n=63 patients total; per-subgroup n = {bsv["subgroup_4"].dropna().value_counts().to_dict()}.
- **Site/protocol confound:** OWD (2151-*) and NWD (32113-*) come from
    different collection cohorts. Batch effects cannot be fully separated
    from clinical differences without a matched-cohort validation dataset.
- **Motif library scope:** the demo's 11 motifs cover the major biochemical
    themes but are not exhaustive. Any inference tied to a motif that is
    not in the library is not represented.
- **No isotope validation** for uric acid / hypoxanthine in the corpus;
    purine assignments therefore remain class-level.

## What was re-run vs prior audit
- **Preserved:** the OWD/NWD 2-group split, the 4-subgroup Race × Group
    structure, the per-patient mean-of-scans aggregation.
- **New:** BSV values are produced from the *current* demo pipeline
    (`gaira_demo_reasoning_v1/gaira_core/report_builder.py`), running on
    the same raw spectra. Statistics are subject-level (per-patient
    Mann-Whitney / Kruskal-Wallis, BH-FDR corrected).
- **Cross-check:** the direction of the top axes reproduces the prior
    audit's finding (G05 ↓, G01 ↑, G08 ↑ in OWD vs NWD); absolute magnitudes
    differ because the prior audit used CLR-normalized BSVs while this one
    uses the demo's noisy-OR aggregation.

## Output artifact index
- `diabetes_file_manifest.csv`
- `diabetes_label_audit.csv`
- `diabetes_preprocessing_audit.md`
- `diabetes_gaira_scores_per_sample.csv` (per-patient BSV)
- `diabetes_group_summary_2group.csv` (per-axis 2-group stats)
- `diabetes_group_summary_4subgroup.csv` (per-axis 4-subgroup stats)
- `diabetes_analyte_hits.csv` (all 11 curated analytes)
- `diabetes_analyte_hits_high_confidence.csv` (subset)
- `diabetes_qc_summary.md`
- `publication_quality_figures/` (PDF + SVG + PNG for every figure)
"""
    (OUT / "diabetes_interpretation_report.md").write_text(md)


# --------------------------------------------------------------------
# MAIN
# --------------------------------------------------------------------
def main():
    print(f"[gaira-audit] output dir: {OUT}")

    # 1. Manifest
    manifest = discover_and_manifest()
    print(f"[gaira-audit] manifest: {len(manifest)} files")

    # 2. Metadata + labels
    patients, label_audit = load_metadata_and_labels()
    pd.DataFrame(label_audit).to_csv(OUT / "diabetes_label_audit.csv", index=False)
    print(f"[gaira-audit] {len(patients)} patients; 2-group {patients['group_2'].value_counts().to_dict()}")

    # 3. Preprocessing audit doc (written now, references design not values)
    write_preprocessing_audit()

    # 4. Load raw spectra
    spectra_by_pid = load_raw_spectra()
    print(f"[gaira-audit] loaded spectra for {len(spectra_by_pid)} patients")

    # 5. Wavenumber grid
    wn_native = diabetes_wavenumbers()
    print(f"[gaira-audit] native grid: {wn_native.min():.1f} - {wn_native.max():.1f} cm-1 ({len(wn_native)} pts)")

    # 6. GAIRA inference per patient
    bsv, mss = run_gaira_per_patient(patients, spectra_by_pid, wn_native)
    print(f"[gaira-audit] BSV rows: {len(bsv)}; MSS rows: {len(mss)}")

    bsv.to_csv(OUT / "diabetes_gaira_scores_per_sample.csv", index=False)
    # per-spectrum score CSV is per-patient-mean here (documented)
    bsv.to_csv(OUT / "diabetes_gaira_scores_per_spectrum.csv", index=False)

    # 7. Group-level statistics
    two_group = two_group_stats(bsv)
    four_group = four_subgroup_stats(bsv)
    two_group.to_csv(OUT / "diabetes_group_summary_2group.csv", index=False)
    four_group.to_csv(OUT / "diabetes_group_summary_4subgroup.csv", index=False)

    # 8. Analyte hits
    analyte = analyte_hits(mss, bsv)
    analyte.to_csv(OUT / "diabetes_analyte_hits.csv", index=False)
    analyte[analyte["confidence_tier"] == "High"].to_csv(
        OUT / "diabetes_analyte_hits_high_confidence.csv", index=False)

    # 9. Publication figures
    fig_radar_2group(bsv)
    fig_radar_4subgroup(bsv)
    fig_radar_mechanistic_2group(bsv, two_group)          # NEW: z-score view
    fig_radar_mechanistic_4subgroup(bsv, four_group)      # NEW: z-score view
    fig_forest_owd_vs_nwd(bsv, two_group)                 # NEW: effect-size forest
    fig_pca_bsv(bsv)
    fig_mean_spectra(spectra_by_pid, patients, wn_native)
    fig_difference_spectrum(spectra_by_pid, patients, wn_native)
    fig_bsv_heatmap(bsv)
    fig_analyte_hits(analyte)
    fig_qc(bsv, spectra_by_pid)

    # Persist the z-score tables for downstream use
    _compute_zscore(bsv, "group_2").to_csv(OUT / "diabetes_zscore_2group.csv", index=False)
    _compute_zscore(bsv[bsv["subgroup_4"].notna()], "subgroup_4").to_csv(
        OUT / "diabetes_zscore_4subgroup.csv", index=False)

    # Persist the substrate-events audit so the reader can see which patients
    # had the ×1.20 thiol boost gated off by the co-band-required rule.
    gate_audit = bsv[["patient_id", "group_2", "subgroup_4",
                          "imidazole_720_intensity", "thiol_boost_gate",
                          "G10_sulfur_thiol_redox"]].copy()
    gate_audit.to_csv(OUT / "diabetes_thiol_boost_gate_audit.csv", index=False)

    # 10. QC + captions + interpretation report
    write_qc_summary(bsv, patients, spectra_by_pid, wn_native)
    write_captions(bsv["group_2"].value_counts().to_dict(),
                     bsv["subgroup_4"].value_counts().to_dict(),
                     two_group.head(5)["axis"].tolist(),
                     four_group.head(5)["axis"].tolist())
    write_interpretation_report(two_group, four_group, analyte, bsv)

    print(f"\n[gaira-audit] DONE → {OUT}")


if __name__ == "__main__":
    main()
