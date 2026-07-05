"""gaira_base_4_shine_otc_dataset_audit_v1

Strict PRE-INGESTION AUDIT + INGESTION PLAN for:
  1) SHINE EV SERS hepatotoxicity dataset (APAP dose-response)
  2) OTC drugs SERS dataset

No GAIRA scoring, no classifier, no molecule-level claims.
Emits audit tables + ingestion-pipeline design + evaluation plan.

Run:
    cd /Users/suraj/projects/GAIRA
    PYTHONPATH=src .venv/bin/python scripts/run_gaira_base_4_shine_otc_dataset_audit_v1.py
"""
from __future__ import annotations

import shutil
import sys
import warnings
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as sio
import matplotlib.pyplot as plt

warnings.simplefilter("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


# ──────────────────────────────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────────────────────────────
ROOT = Path("/Volumes/SSD_Rad/GAIRA_BUILD/gaira_base_4_shine_otc_dataset_audit_v1")
TABLES  = ROOT / "tables"
REPORTS = ROOT / "reports"
AUDIT   = ROOT / "audit"
CODE_SNAPSHOT = ROOT / "code_snapshot"
for d in (TABLES, REPORTS, AUDIT, CODE_SNAPSHOT):
    d.mkdir(parents=True, exist_ok=True)

SHINE = Path("/Volumes/SSD_Rad/GAIRA_DATA/raw/shine_ev_sers/SERS-Hepatotoxicity_DATA_CODE_FIGURE")
SHINE_SET9  = SHINE / "Figure4/data/Set9"
SHINE_SET10 = SHINE / "Figure4/data/Set10"
SHINE_WN    = SHINE / "Figure4/Fig4C/data/combined_wavenumbers.mat"
SHINE_MAT91 = SHINE / "Figure4/data/RawDataSet91.mat"
SHINE_MAT119 = SHINE / "Figure4/data/RawDataset119.mat"
OTC   = Path("/Volumes/SSD_Rad/GAIRA_DATA/raw/otc_drugs")


# Pixel→wavenumber calibration extracted verbatim from Fig4D.m (SHINE paper)
SHINE_CAL_PIX = np.array([263, 367, 492, 512, 590, 782, 872, 887])
SHINE_CAL_CM  = np.array([620.9, 795.8, 1001.4, 1031.8, 1155.3, 1450.5, 1583.1, 1602.3])
SHINE_N_PIXELS = 1650
SHINE_ANALYSIS_PIXEL_RANGE = (162, 898)  # from paper code; crops to ~400-1700 cm⁻¹


# ──────────────────────────────────────────────────────────────────────
# SHINE audit
# ──────────────────────────────────────────────────────────────────────
def shine_file_structure():
    print("[SHINE] file structure + spectra count")
    rows = []
    for set_name, set_path in [("Set9", SHINE_SET9), ("Set10", SHINE_SET10)]:
        if not set_path.exists(): continue
        for cond_dir in sorted(set_path.iterdir()):
            if not cond_dir.is_dir(): continue
            cond = cond_dir.name  # e.g. "D0_C0"
            try:
                day_part, conc_part = cond.split("_")
                day = int(day_part.lstrip("D"))
                conc_mM = int(conc_part.lstrip("C"))
            except Exception:
                day, conc_mM = -1, -1
            n_spectra = sum(1 for _ in cond_dir.rglob("s_*") if _.is_file())
            subject_dirs = [p for p in cond_dir.iterdir() if p.is_dir()]
            # Some conditions flatten subjects: files are directly s_* under the condition folder
            has_subject_folders = len(subject_dirs) > 0 and \
                                          any(d.name.startswith("s_") for d in subject_dirs) is False
            n_subjects = len(subject_dirs) if has_subject_folders else 1
            rows.append({
                "dataset":         "SHINE_EV_SERS",
                "set":             set_name,
                "condition":       cond,
                "day":             day,
                "APAP_conc_mM":    conc_mM,
                "n_subjects_or_groups": n_subjects,
                "n_spectra":       n_spectra,
                "has_subject_folders":  has_subject_folders,
            })
    df = pd.DataFrame(rows)
    df.to_csv(TABLES / "shine_file_structure_v1.csv", index=False)
    return df


def shine_spectral_format():
    print("[SHINE] spectral format")
    # A single spectrum sample
    sample_file = next((f for f in SHINE_SET9.rglob("s_*") if f.is_file()), None)
    n_points = 0
    if sample_file:
        with open(sample_file) as f:
            n_points = sum(1 for _ in f)
    # Pixel→wavenumber polynomial (3rd order, from paper MATLAB)
    p = np.polyfit(SHINE_CAL_PIX, SHINE_CAL_CM, 3)
    wn_axis = np.polyval(p, np.arange(1, SHINE_N_PIXELS + 1))
    full_min, full_max = float(wn_axis.min()), float(wn_axis.max())
    analysis_wn = wn_axis[SHINE_ANALYSIS_PIXEL_RANGE[0]-1:SHINE_ANALYSIS_PIXEL_RANGE[1]]
    ana_min, ana_max = float(analysis_wn.min()), float(analysis_wn.max())

    rows = [{
        "dataset":                     "SHINE_EV_SERS",
        "file_format":                 "CSV (pixel_idx,intensity); 1650 rows per spectrum",
        "raw_x_axis":                  "pixel_index (1..1650) — NOT wavenumber",
        "wavenumber_calibration":      "3rd-order polynomial on 8 paper-provided pixel↔Raman-shift pairs (from Fig4D.m)",
        "calibration_reference_bands": ";".join(f"{int(p)}px→{c:.1f}cm⁻¹" for p, c in zip(SHINE_CAL_PIX, SHINE_CAL_CM)),
        "full_pixel_range_cm1":        f"{full_min:.1f} - {full_max:.1f}",
        "paper_analysis_pixel_range":  f"{SHINE_ANALYSIS_PIXEL_RANGE[0]}..{SHINE_ANALYSIS_PIXEL_RANGE[1]}",
        "paper_analysis_cm1_range":    f"{ana_min:.1f} - {ana_max:.1f}",
        "intensity_units":             "raw counts (integer)",
        "n_points_per_spectrum_observed": n_points,
    }]
    df = pd.DataFrame(rows)
    df.to_csv(TABLES / "shine_spectral_format_v1.csv", index=False)
    return df, wn_axis


def shine_experimental_variables(struct_df):
    print("[SHINE] experimental variables")
    rows = []
    # Days
    for day in sorted(struct_df["day"].unique()):
        sub = struct_df[struct_df.day == day]
        rows.append({"variable": "day", "value": day,
                        "n_conditions": int(len(sub)),
                        "n_spectra":    int(sub.n_spectra.sum())})
    # Concentrations
    for conc in sorted(struct_df["APAP_conc_mM"].unique()):
        sub = struct_df[struct_df.APAP_conc_mM == conc]
        rows.append({"variable": "APAP_conc_mM", "value": int(conc),
                        "n_conditions": int(len(sub)),
                        "n_spectra":    int(sub.n_spectra.sum())})
    # Sets
    for set_name in sorted(struct_df["set"].unique()):
        sub = struct_df[struct_df.set == set_name]
        rows.append({"variable": "set", "value": set_name,
                        "n_conditions": int(len(sub)),
                        "n_spectra":    int(sub.n_spectra.sum())})
    df = pd.DataFrame(rows)
    df.to_csv(TABLES / "shine_experimental_variables_v1.csv", index=False)
    return df


def shine_qc_probe():
    """Lightweight QC probe on a small random sample — detect obvious issues."""
    print("[SHINE] QC probe on a sampled subset")
    rng = np.random.default_rng(42)
    sample_files = list(SHINE_SET9.rglob("s_*")) + list(SHINE_SET10.rglob("s_*"))
    sample_files = [f for f in sample_files if f.is_file()]
    if len(sample_files) > 500:
        sample_files = rng.choice(np.array(sample_files, dtype=object), 500, replace=False).tolist()
    qc_rows = []
    for f in sample_files:
        try:
            arr = np.loadtxt(f, delimiter=",")
            y = arr[:, 1]
        except Exception:
            qc_rows.append({"file": str(f.relative_to(SHINE)),
                               "status": "READ_ERROR", "n_points": 0,
                               "max": np.nan, "median": np.nan, "std": np.nan})
            continue
        is_flat = bool(np.std(y) < 1.0)
        is_saturated = bool(np.max(y) >= 65500)
        is_empty    = bool(np.all(y == 0))
        qc_rows.append({
            "file":     str(f.relative_to(SHINE)),
            "status":   "FLAT" if is_flat else
                          "SATURATED" if is_saturated else
                          "EMPTY" if is_empty else "OK",
            "n_points": int(len(y)),
            "max":      float(np.max(y)) if len(y) else np.nan,
            "median":   float(np.median(y)) if len(y) else np.nan,
            "std":      float(np.std(y)) if len(y) else np.nan,
        })
    df = pd.DataFrame(qc_rows)
    df.to_csv(TABLES / "shine_qc_sample_v1.csv", index=False)
    print(f"  sampled {len(df)} spectra, status dist: {dict(Counter(df.status))}")
    return df


# ──────────────────────────────────────────────────────────────────────
# OTC audit
# ──────────────────────────────────────────────────────────────────────
def otc_file_structure_and_spectral_format():
    print("[OTC] file structure + spectral format")
    rows_struct = []; rows_fmt = []
    files = [p for p in sorted(OTC.iterdir())
             if p.suffix == ".xlsx" and not p.name.startswith("._")]
    for f in files:
        df = pd.read_excel(f, sheet_name=0, header=0)
        rs = pd.to_numeric(df.iloc[:, 0], errors="coerce").dropna()
        cols = df.columns.tolist()
        labels = [str(c).replace("\n", "").replace("\t", "").strip() for c in cols[1:]]
        label_bases = [l.split(".")[0] for l in labels]
        counter = Counter(label_bases)
        rows_struct.append({
            "dataset":      "OTC_drugs",
            "file":         f.name,
            "n_rows":       int(df.shape[0]),
            "n_cols":       int(df.shape[1]),
            "n_spectra":    int(df.shape[1] - 1),
            "n_unique_label_groups": len(counter),
            "label_groups_counts":   ";".join(f"{k}={v}" for k, v in counter.items()),
            "first_col_header": str(df.columns[0]),
        })
        steps = np.diff(rs.values)
        rows_fmt.append({
            "dataset":          "OTC_drugs",
            "file":             f.name,
            "file_format":      "xlsx (first col Raman Shift cm⁻¹, each subsequent col = one spectrum)",
            "n_spectral_points": int(len(rs)),
            "x_axis_type":      "wavenumber_cm1",
            "wn_min_cm1":       float(rs.min()),
            "wn_max_cm1":       float(rs.max()),
            "wn_step_median":   float(np.median(steps)),
            "wn_step_min":      float(np.min(steps)),
            "wn_step_max":      float(np.max(steps)),
            "intensity_units":  "arbitrary (likely already baseline-corrected Raman counts)",
        })
    pd.DataFrame(rows_struct).to_csv(TABLES / "otc_file_structure_v1.csv", index=False)
    pd.DataFrame(rows_fmt).to_csv(TABLES / "otc_spectral_format_v1.csv", index=False)
    return pd.DataFrame(rows_struct), pd.DataFrame(rows_fmt)


def otc_experimental_variables(struct_df):
    print("[OTC] experimental variables")
    rows = []
    # Identify drug identity + pure vs trademark
    for _, r in struct_df.iterrows():
        f = r["file"]
        if "trademark" in f.lower():
            variant = "trademark_brand_variants"
        elif f.lower().startswith("all spectra"):
            variant = "combined_pure_drugs"
        else:
            variant = "pure_drug"
        # Drug identity: look at file name base
        base = f.replace("-trademark.xlsx", "").replace(".xlsx", "")
        rows.append({
            "file":       f,
            "variant":    variant,
            "drug_base":  base,
            "n_spectra":  int(r["n_spectra"]),
            "label_groups_counts": r["label_groups_counts"],
        })
    df = pd.DataFrame(rows)
    df.to_csv(TABLES / "otc_experimental_variables_v1.csv", index=False)
    return df


def otc_qc_probe():
    print("[OTC] QC probe on all xlsx files")
    files = [p for p in sorted(OTC.iterdir())
             if p.suffix == ".xlsx" and not p.name.startswith("._")]
    qc_rows = []
    for f in files:
        df = pd.read_excel(f, sheet_name=0, header=0)
        Y = df.iloc[:, 1:].values.astype(float)
        for j in range(Y.shape[1]):
            y = Y[:, j]
            y_valid = y[np.isfinite(y)]
            if len(y_valid) == 0:
                status = "EMPTY"
            elif np.std(y_valid) < 1e-6:
                status = "FLAT"
            elif np.all(y_valid == 0):
                status = "EMPTY"
            else:
                status = "OK"
            qc_rows.append({
                "file":       f.name,
                "spectrum_idx": j,
                "status":     status,
                "n_points":   int(len(y_valid)),
                "median":     float(np.median(y_valid)) if len(y_valid) else np.nan,
                "std":        float(np.std(y_valid)) if len(y_valid) else np.nan,
            })
    df = pd.DataFrame(qc_rows)
    df.to_csv(TABLES / "otc_qc_per_spectrum_v1.csv", index=False)
    print(f"  checked {len(df)} OTC spectra, status dist: {dict(Counter(df.status))}")
    return df


# ──────────────────────────────────────────────────────────────────────
# Preprocessing comparison (GAIRA canonical vs dataset)
# ──────────────────────────────────────────────────────────────────────
def preprocessing_comparison():
    print("[preprocess] comparison GAIRA canonical vs dataset-native")
    rows = [
        {
            "stage":              "raw_input",
            "gaira_canonical":    "per-spectrum raw counts, pixel→wavenumber calibration required",
            "shine_native":       "CSV pixel,intensity (1650 pixels); wn calibration is 3rd-order polynomial in Fig4D.m",
            "otc_native":         "xlsx (Raman Shift cm⁻¹, 1823 points at ~2.4 cm⁻¹ step, 148-3199 cm⁻¹)",
            "action_for_gaira":   "use paper polynomial for SHINE; OTC already in cm⁻¹",
        },
        {
            "stage":              "baseline_correction",
            "gaira_canonical":    "AsLS (lam=1e5, p=0.001, 10 iter)",
            "shine_native":       "paper applies baseline correction (produces *_bc variants inside clustered{})",
            "otc_native":         "intensity magnitudes look already pre-processed; re-run AsLS defensively",
            "action_for_gaira":   "apply GAIRA canonical AsLS to the raw SHINE counts; re-apply AsLS to OTC for consistency",
        },
        {
            "stage":              "spectral_crop",
            "gaira_canonical":    "crop to master axis 400-1800 cm⁻¹",
            "shine_native":       "paper crops to pixel range 162..898 ≈ 400-1700 cm⁻¹; further text mentions 810-1610 for ML",
            "otc_native":         "full 148-3199 cm⁻¹ retained by default",
            "action_for_gaira":   "crop both to GAIRA master 400-1800 cm⁻¹; record native crop as annotation; do NOT enforce paper's 810-1610 crop for BSV (loses 400-810 band info)",
        },
        {
            "stage":              "smoothing",
            "gaira_canonical":    "Savitzky-Golay window=11 polyorder=3",
            "shine_native":       "not explicitly smoothed in Fig4D.m after BC",
            "otc_native":         "unknown smoothing state",
            "action_for_gaira":   "apply GAIRA SG to both",
        },
        {
            "stage":              "normalization",
            "gaira_canonical":    "L2 norm per spectrum",
            "shine_native":       "paper normalizes by mean of D0_C0 (Day 0) or D2_C0 (Day 2) × 100 — label-LEAKING normalization",
            "otc_native":         "none documented",
            "action_for_gaira":   "apply GAIRA L2 per spectrum; IGNORE paper's D0_C0/D2_C0 relative normalization — this leaks control-cohort information",
        },
        {
            "stage":              "k_means_blank_filtering",
            "gaira_canonical":    "not part of canonical pipeline",
            "shine_native":       "Fig4A uses clustered{} which implies a prior k-means filter of blank/dead pixels",
            "otc_native":         "n/a",
            "action_for_gaira":   "bypass k-means filtering — operate on all non-corrupt spectra and flag low-SNR spectra via GAIRA QC instead (label-free)",
        },
        {
            "stage":              "si_peak_642_normalization",
            "gaira_canonical":    "not part of canonical pipeline",
            "shine_native":       "paper mentions 642 cm⁻¹ Si peak normalization in some variants; the code Fig4D.m does NOT apply it (uses mean-based within-day scaling instead)",
            "otc_native":         "n/a",
            "action_for_gaira":   "skip Si peak normalization — would bias BSV toward Si-adjacent bands and remove intensity info we want",
        },
    ]
    df = pd.DataFrame(rows)
    df.to_csv(TABLES / "preprocessing_comparison_v1.csv", index=False)
    return df


# ──────────────────────────────────────────────────────────────────────
# Ingestion pipeline design
# ──────────────────────────────────────────────────────────────────────
def ingestion_pipeline():
    print("[ingestion] pipeline design")
    steps = [
        {
            "step": 1, "name": "raw_loader",
            "shine":   "iterate Set9/{D0,D1,D2}_C{0,10,20,40}/{subject?}/s_* → load pixel_idx,intensity CSV → 1650-pt array",
            "otc":     "per xlsx: first col=Raman_Shift, subsequent cols=spectra → (1823, N)",
            "output":  "per-spectrum dict with {dataset, set, day, conc_mM, subject, rep, y_raw, wn_raw}",
        },
        {
            "step": 2, "name": "wavenumber_alignment",
            "shine":   "apply Fig4D.m polynomial fit → 1650-pt wn axis; interpolate to GAIRA master_x (400-1800 step 1)",
            "otc":     "already in cm⁻¹; interpolate to GAIRA master_x (400-1800 step 1) — OTC covers well beyond 1800, so crop happens here",
            "output":  "y_master_x (1401-pt array); NaN where not covered",
        },
        {
            "step": 3, "name": "gaira_canonical_preprocess",
            "shine":   "AsLS → SG (win=11, order=3) → L2 normalize",
            "otc":     "AsLS → SG (win=11, order=3) → L2 normalize",
            "output":  "y_pp (1401-pt canonical spectrum)",
        },
        {
            "step": 4, "name": "qc_gate_unlabeled",
            "shine":   "flag FLAT (std<1e-3) / SATURATED (max>0.95 × 99th-pctile) / NAN-MAJORITY (<50% finite points)",
            "otc":     "same criteria",
            "output":  "qc_status in {OK, FLAT, SATURATED, NAN_MAJ}; exclude non-OK from BSV",
        },
        {
            "step": 5, "name": "bsv_scoring",
            "shine":   "11-axis BSV (sumnorm + CLR) via family-aggregated MSS-anchor score kernel (same as MSS resolution layer v1)",
            "otc":     "same",
            "output":  "per-spectrum (11,) raw / sumnorm / CLR vectors",
        },
        {
            "step": 6, "name": "delta_bsv",
            "shine":   "ΔBSV = per-spectrum CLR - reference (reference = C0 mean of that day+set); ΔBSV is label-aware at the REFERENCE-COHORT level only (no dose label inside BSV)",
            "otc":     "ΔBSV = per-spectrum CLR - ALL-drug reference (e.g. all-drug mean) — reference is GROUP-of-all not per-drug; labels used only for evaluation",
            "output":  "per-spectrum ΔBSV vector for downstream evaluation",
        },
        {
            "step": 7, "name": "mss_reporting_layer_optional",
            "shine":   "per-spectrum top-5 MSS molecule candidates (constrained to narrow registry); top-k frequency by cohort",
            "otc":     "same, with the caveat: no overlap between target narrow registry (adenine/UA/HX/ERG/…) and OTC drugs (ibuprofen / aspirin / paracetamol); MSS layer is likely INDETERMINATE for OTC",
            "output":  "per-spectrum top-k molecule candidates (flagged candidate-level evidence only)",
        },
    ]
    df = pd.DataFrame(steps)
    df.to_csv(TABLES / "ingestion_pipeline_v1.csv", index=False)
    return df


# ──────────────────────────────────────────────────────────────────────
# Evaluation plan
# ──────────────────────────────────────────────────────────────────────
def evaluation_plan():
    print("[eval] evaluation plan")
    shine_rows = [
        {
            "metric":        "ΔBSV vs dose monotonicity",
            "scope":         "per axis × per day (D0, D1, D2)",
            "implementation": "Spearman ρ(APAP_conc_mM, mean(ΔBSV_axis)) across {C0, C10, C20, C40}",
            "expected":      "flat on D0 (no biology yet); monotonic on D2; D1 intermediate",
            "outputs":       "table: axis × day × ρ; figure: 11-axis trajectory per day",
        },
        {
            "metric":        "Day 0 vs Day 2 separability",
            "scope":         "per axis",
            "implementation": "Cohen's d(D2, D0) per axis within each concentration cohort",
            "expected":      "near-zero on C0 (control vs control); growing with dose on D2",
            "outputs":       "table: axis × conc × d; figure: d-heatmap axis × conc",
        },
        {
            "metric":        "axis-level effect sizes",
            "scope":         "per axis, C40 vs C0 on D2",
            "implementation": "Cohen's d(C40_D2, C0_D2) per axis; bootstrap CI",
            "expected":      "largest effects on protein/nucleic-acid/aromatic axes per paper priors",
            "outputs":       "ranked table + CI plot",
        },
        {
            "metric":        "radar morphing consistency",
            "scope":         "per dose, per day",
            "implementation": "per-cohort mean 11-axis CLR vector; radar plot overlay C0/10/20/40 by day",
            "expected":      "C0 → C40 morphs in consistent direction across 11 axes on D2",
            "outputs":       "3 radar figures (D0, D1, D2)",
        },
        {
            "metric":        "set-level transferability (Set9 vs Set10)",
            "scope":         "D2 only (Set10 has only D2)",
            "implementation": "per-axis mean Set9_D2 vs Set10_D2; cross-set Pearson",
            "expected":      "high cross-set correlation if biology is real",
            "outputs":       "scatter axis × axis; cross-set rank correlation",
        },
        {
            "metric":        "subject-level variance decomposition (η²)",
            "scope":         "Set9 D0+D2; factors = {day, conc, subject, set}",
            "implementation": "single-factor ANOVA on ΔBSV per axis",
            "expected":      "conc + day should dominate; subject should be smaller",
            "outputs":       "η² table axis × factor",
        },
    ]
    otc_rows = [
        {
            "metric":        "BSV clustering in 2D PCA",
            "scope":         "pure-drugs (All spectra.xlsx)",
            "implementation": "PCA on CLR BSV 11-axis vectors; color by drug label",
            "expected":      "3 drug clusters separable; axes dominated by their distinctive chemistry regions",
            "outputs":       "PCA scatter + loadings",
        },
        {
            "metric":        "drug-specific BSV-axis signatures",
            "scope":         "per drug × per axis",
            "implementation": "per-drug mean BSV CLR vector; 11-axis radar",
            "expected":      "ibuprofen / ASA / paracetamol show distinct BSV profiles",
            "outputs":       "3 radar figures",
        },
        {
            "metric":        "pure vs trademark stability",
            "scope":         "per drug × per {pure, trademark} split",
            "implementation": "per-drug × per-variant mean BSV CLR; cross-variant Cohen's d",
            "expected":      "within-drug BSV stable across pure → trademark (same active ingredient)",
            "outputs":       "table: drug × axis × d(pure-trademark); d should be small",
        },
        {
            "metric":        "clustering silhouette (by drug identity)",
            "scope":         "All pure + trademark spectra",
            "implementation": "silhouette score on BSV-CLR space with drug_base as label",
            "expected":      "score > 0.3 indicates drug-identifiable structure at BSV level",
            "outputs":       "silhouette number + figure",
        },
        {
            "metric":        "molecule-level MSS expectation",
            "scope":         "per spectrum, against narrow registry",
            "implementation": "top-5 MSS candidates; compute top-K frequency by drug cohort",
            "expected":      "LIKELY NO MATCH — narrow registry has no ibuprofen/ASA/paracetamol templates. OTC spectra will surface whichever narrow-registry molecules happen to share band positions. Report as candidate-level noise, NOT molecule identity.",
            "outputs":       "honest MSS frequency table with explicit caveat",
        },
    ]
    shine_df = pd.DataFrame(shine_rows); shine_df["dataset"] = "SHINE_EV_SERS"
    otc_df   = pd.DataFrame(otc_rows);   otc_df["dataset"] = "OTC_drugs"
    out = pd.concat([shine_df, otc_df], ignore_index=True)
    out.to_csv(TABLES / "evaluation_plan_v1.csv", index=False)
    return out


# ──────────────────────────────────────────────────────────────────────
# Expected biochemical signal mapping (validation reference only)
# ──────────────────────────────────────────────────────────────────────
def expected_biochemical_mapping():
    print("[priors] expected biochemical signal (SHINE; validation reference only)")
    rows = [
        {"band_cm1": 739,  "paper_assignment": "DNA/nucleic ring breathing (T/A)",
         "gaira_axis": "G04 phosphate_nucleic_adjacent / G01 purine_nucleotide (nearby)"},
        {"band_cm1": 960,  "paper_assignment": "DNA/phosphate backbone",
         "gaira_axis": "G04 phosphate_nucleic_adjacent"},
        {"band_cm1": 1250, "paper_assignment": "amide III protein",
         "gaira_axis": "G06 protein_peptide_backbone"},
        {"band_cm1": 1525, "paper_assignment": "carotenoid / conjugated lipid (context-dependent)",
         "gaira_axis": "G07 aromatic_residue or G08 lipid_acyl_membrane (context)"},
        {"band_cm1": 1576, "paper_assignment": "purine ring-mode (G, A)",
         "gaira_axis": "G01 purine_nucleotide"},
        {"band_cm1": 1602, "paper_assignment": "phenylalanine / aromatic AA ring breathing",
         "gaira_axis": "G07 aromatic_residue"},
        {"band_cm1": "(dose-direction)", "paper_assignment": "intensity ↓ with dose on Day 2",
         "gaira_axis": "expect ΔBSV magnitude ↓ on protein/nucleic/aromatic axes at higher doses on D2"},
        {"band_cm1": "(day-direction)",  "paper_assignment": "Day 0 shows no separation by dose",
         "gaira_axis": "expect flat ΔBSV on D0; separation emerging on D1; clearest on D2"},
    ]
    df = pd.DataFrame(rows)
    df.to_csv(TABLES / "expected_biochemistry_reference_v1.csv", index=False)
    return df


# ──────────────────────────────────────────────────────────────────────
# Risks / issues
# ──────────────────────────────────────────────────────────────────────
def risks_list(shine_struct, otc_struct):
    print("[risks] issues + risks")
    rows = [
        {
            "dataset": "SHINE",
            "risk":    "Set9 D1 lacks subject subfolders (flat s_* files); Set9 D0/D2 have subjects — mixed hierarchy",
            "impact":  "cannot compute subject-level variance on D1; plan must treat D1 as 'group-level' only",
            "mitigation": "compute subject-level variance only on D0 / D2; report D1 as aggregate",
        },
        {
            "dataset": "SHINE",
            "risk":    "Set10 contains only D2 (no D0 baseline)",
            "impact":  "cross-set transferability test is D2-only",
            "mitigation": "use Set9 D2 ↔ Set10 D2 comparison; explicit scope note in report",
        },
        {
            "dataset": "SHINE",
            "risk":    "Paper normalizes Day 0 cohorts by D0_C0 mean and Day 2 cohorts by D2_C0 mean — this LEAKS the control cohort into every normalized spectrum",
            "impact":  "paper's PCA/GPR results bake in C0 normalization; GAIRA should NOT replicate this without per-day reference acknowledgement",
            "mitigation": "apply GAIRA L2 per-spectrum normalization instead; use C0 ONLY as evaluation reference, not as a normalization anchor",
        },
        {
            "dataset": "SHINE",
            "risk":    "Paper also mentions k-means filtering of blank spectra; the clustered{} variable hides this step",
            "impact":  "raw Set9 / Set10 folders contain the pre-k-means raw CSVs; GAIRA QC must be a label-free alternative",
            "mitigation": "use GAIRA QC gates (flat/saturated/NaN-majority) and record filter-rate; report as dataset-native cleanup not biology filter",
        },
        {
            "dataset": "SHINE",
            "risk":    "Pixel→wavenumber calibration is a 3rd-order polynomial from 8 reference points; calibration error grows outside the fit range",
            "impact":  "bands near pixel 1 or pixel 1650 have higher wn uncertainty (the paper's reference points span 263-887)",
            "mitigation": "restrict analysis to pixel range 162-898 (~400-1700 cm⁻¹) per paper; flag anything outside as extrapolated",
        },
        {
            "dataset": "OTC",
            "risk":    "Narrow MSS registry has NO ibuprofen / acetylsalicylic-acid / paracetamol templates",
            "impact":  "MSS top-K on OTC spectra will fire incidental near-band matches (e.g. aromatic ring ~1003 cm⁻¹) from registry targets that are NOT the actual drug",
            "mitigation": "Report MSS as candidate-level noise, NOT identity. Lead evaluation with BSV-axis clustering + per-drug BSV signature stability. Explicit caveat in every OTC MSS output.",
        },
        {
            "dataset": "OTC",
            "risk":    "Trademark files contain brand variants with multiple codes per file (e.g. Paracetamol-trademark: 1 Para-D + 49 Para-B)",
            "impact":  "'pure vs trademark' stability test may actually compare pure Paracetamol vs brand-B-only (not all trademark variants represented)",
            "mitigation": "split trademark files by brand code; report per-brand n; avoid aggregating over brand codes",
        },
        {
            "dataset": "OTC",
            "risk":    "OTC spectral range 148-3199 cm⁻¹ extends well beyond GAIRA master_x (400-1800 cm⁻¹)",
            "impact":  "high-wn modes (C-H 2800-3000 cm⁻¹) are dropped from BSV scoring",
            "mitigation": "acceptable — BSV operates on 400-1800; document that C-H region is discarded and is NOT part of GAIRA cross-dataset comparison",
        },
        {
            "dataset": "OTC",
            "risk":    "Step size ~2.4 cm⁻¹ (not 1.0 cm⁻¹ as master_x); interpolation is required",
            "impact":  "minor smoothing of fine features near 1 cm⁻¹",
            "mitigation": "linear interpolation to master 1 cm⁻¹ grid; standard practice across GAIRA",
        },
    ]
    df = pd.DataFrame(rows)
    df.to_csv(TABLES / "risks_and_issues_v1.csv", index=False)
    return df


# ──────────────────────────────────────────────────────────────────────
# Next steps
# ──────────────────────────────────────────────────────────────────────
def next_steps():
    rows = [
        {"order": 1, "action": "Implement SHINE loader: iterate Set9+Set10 tree, produce (spectrum_id, set, day, conc_mM, subject, rep, y_raw_1650pt)"},
        {"order": 2, "action": "Implement pixel→wavenumber transform (3rd-order polynomial from Fig4D.m) + interp to master_x (400-1800)"},
        {"order": 3, "action": "Implement OTC loader: iterate 6 xlsx files + 'All spectra.xlsx', parse per-column spectrum + drug_base + variant label"},
        {"order": 4, "action": "Apply GAIRA canonical preprocessing (AsLS + SG + L2) per spectrum"},
        {"order": 5, "action": "Apply QC gates (flat/saturated/NaN-majority); exclude non-OK from BSV scoring; record exclusion log"},
        {"order": 6, "action": "Compute 11-axis BSV (raw / sumnorm / CLR) per spectrum — reuse family-aggregated MSS-anchor kernel from MSS resolution layer v1"},
        {"order": 7, "action": "SHINE evaluation: ΔBSV-vs-dose ρ per axis per day; D0-vs-D2 Cohen's d; axis effect sizes; radar plots; set-level transferability; subject-level η² decomposition"},
        {"order": 8, "action": "OTC evaluation: BSV PCA colored by drug; per-drug BSV radar; pure-vs-trademark cross-variant d; clustering silhouette by drug identity"},
        {"order": 9, "action": "Emit MSS candidate layer with explicit caveats (SHINE: candidate-level, unreliable per prior phases; OTC: LIKELY NO MATCH in registry)"},
        {"order": 10, "action": "Write demo narrative: SHINE ΔBSV dose-response + Day 0/Day 2 separation + cross-set stability; OTC BSV-level drug-identity clustering + caveat that molecule-level claims require OTC-specific registry (not yet in GAIRA)"},
    ]
    df = pd.DataFrame(rows)
    df.to_csv(TABLES / "next_steps_v1.csv", index=False)
    return df


# ──────────────────────────────────────────────────────────────────────
# Report
# ──────────────────────────────────────────────────────────────────────
def write_report(shine_struct, shine_fmt, shine_vars, shine_qc,
                     otc_struct, otc_fmt, otc_vars, otc_qc,
                     preproc_df, ingest_df, eval_df, priors_df, risks_df, next_df):
    lines = [
        "# SHINE + OTC dataset audit + ingestion plan v1",
        f"date: {datetime.now().isoformat()}",
        "",
        "## Scope",
        "- STRICT pre-ingestion audit. NO GAIRA scoring was performed in this phase.",
        "- NO classifier, NO molecule-level claims, NO engine / MSS / BSV / preprocessing changes.",
        "",
        "## 1. Dataset audit summary\n",
    ]

    # SHINE inventory
    n_shine = int(shine_struct.n_spectra.sum())
    n_shine_subj = int(shine_struct[shine_struct.has_subject_folders].n_subjects_or_groups.sum())
    lines.append("### 1.1 SHINE EV SERS hepatotoxicity")
    lines.append(f"- **Total spectra: {n_shine:,}** "
                    f"(Set9={int(shine_struct[shine_struct.set=='Set9'].n_spectra.sum()):,}, "
                    f"Set10={int(shine_struct[shine_struct.set=='Set10'].n_spectra.sum()):,})")
    lines.append(f"- Conditions: 3 days × 4 concentrations × 2 sets (Set10 is D2-only) = 16 cohort directories")
    lines.append(f"- Subject structure: Set9 D0/D2 have subject subfolders; Set9 D1 has flat spectrum files; Set10 D2 has subject subfolders")
    lines.append(f"- Spectral format: CSV (pixel_idx, intensity) — {shine_fmt['n_points_per_spectrum_observed'].iloc[0]} pixels per spectrum")
    lines.append(f"- Wavenumber axis: {shine_fmt['full_pixel_range_cm1'].iloc[0]} cm⁻¹ via 3rd-order polynomial on 8 reference points")
    lines.append(f"- Paper analysis pixel range: {shine_fmt['paper_analysis_pixel_range'].iloc[0]} → "
                    f"{shine_fmt['paper_analysis_cm1_range'].iloc[0]} cm⁻¹")
    lines.append(f"- QC probe ({len(shine_qc)} sampled): "
                    f"{dict(Counter(shine_qc.status))}")
    lines.append("")

    lines.append("### SHINE experimental variables (counts)")
    lines.append("| variable | value | n_conditions | n_spectra |")
    lines.append("|---|---|---:|---:|")
    for _, r in shine_vars.iterrows():
        lines.append(f"| {r['variable']} | {r['value']} | {int(r['n_conditions'])} | {int(r['n_spectra']):,} |")
    lines.append("")

    # OTC inventory
    n_otc = int(otc_struct.n_spectra.sum())
    lines.append("### 1.2 OTC drugs SERS")
    lines.append(f"- **Total spectra: {n_otc:,}** across {len(otc_struct)} xlsx files (one \"All spectra.xlsx\" aggregate + 6 single-drug files)")
    lines.append(f"- Drugs represented: ibuprofen, acetylsalicylic-acid (ASA), paracetamol — "
                    f"each with both pure and trademark-brand variants (~50 spectra per drug × variant)")
    lines.append(f"- Spectral format: xlsx (first column Raman Shift cm⁻¹, each subsequent column = one spectrum)")
    lines.append(f"- Wavenumber axis: **{otc_fmt['wn_min_cm1'].iloc[0]:.1f} — {otc_fmt['wn_max_cm1'].iloc[0]:.1f} cm⁻¹**, "
                    f"step ~{otc_fmt['wn_step_median'].iloc[0]:.2f} cm⁻¹, {int(otc_fmt['n_spectral_points'].iloc[0])} points")
    lines.append(f"- QC: {dict(Counter(otc_qc.status))}")
    lines.append("")
    lines.append("### OTC variants")
    lines.append("| file | variant | drug_base | n_spectra | label_group_counts |")
    lines.append("|---|---|---|---:|---|")
    for _, r in otc_vars.iterrows():
        lines.append(f"| {r['file']} | {r['variant']} | {r['drug_base']} | {int(r['n_spectra'])} | {r['label_groups_counts']} |")
    lines.append("")

    # Issues / risks
    lines.append("## 2. Issues / risks")
    lines.append("| dataset | risk | impact | mitigation |")
    lines.append("|---|---|---|---|")
    for _, r in risks_df.iterrows():
        lines.append(f"| {r['dataset']} | {r['risk']} | {r['impact']} | {r['mitigation']} |")
    lines.append("")

    # Preprocessing comparison
    lines.append("## 3. Preprocessing compatibility (GAIRA canonical vs dataset-native)")
    lines.append("| stage | gaira_canonical | shine_native | otc_native | action_for_gaira |")
    lines.append("|---|---|---|---|---|")
    for _, r in preproc_df.iterrows():
        lines.append(f"| {r['stage']} | {r['gaira_canonical']} | {r['shine_native']} | {r['otc_native']} | {r['action_for_gaira']} |")
    lines.append("")

    # Ingestion pipeline
    lines.append("## 4. Ingestion pipeline (design — not yet executed)")
    lines.append("| step | name | shine | otc | output |")
    lines.append("|---:|---|---|---|---|")
    for _, r in ingest_df.iterrows():
        lines.append(f"| {int(r['step'])} | {r['name']} | {r['shine']} | {r['otc']} | {r['output']} |")
    lines.append("")

    # Evaluation plan
    lines.append("## 5. Evaluation plan")
    lines.append("### 5.1 SHINE")
    lines.append("| metric | scope | implementation | expected | outputs |")
    lines.append("|---|---|---|---|---|")
    for _, r in eval_df[eval_df.dataset == "SHINE_EV_SERS"].iterrows():
        lines.append(f"| {r['metric']} | {r['scope']} | {r['implementation']} | {r['expected']} | {r['outputs']} |")
    lines.append("")
    lines.append("### 5.2 OTC")
    lines.append("| metric | scope | implementation | expected | outputs |")
    lines.append("|---|---|---|---|---|")
    for _, r in eval_df[eval_df.dataset == "OTC_drugs"].iterrows():
        lines.append(f"| {r['metric']} | {r['scope']} | {r['implementation']} | {r['expected']} | {r['outputs']} |")
    lines.append("")

    # Expected biochem signal (validation reference)
    lines.append("## 6. Expected biochemical signal for SHINE (validation reference ONLY, not an input)")
    lines.append("| band (cm⁻¹) | paper_assignment | gaira_axis |")
    lines.append("|---|---|---|")
    for _, r in priors_df.iterrows():
        lines.append(f"| {r['band_cm1']} | {r['paper_assignment']} | {r['gaira_axis']} |")
    lines.append("")

    # Next steps
    lines.append("## 7. Immediate next steps (code-level)")
    lines.append("| order | action |")
    lines.append("|---:|---|")
    for _, r in next_df.iterrows():
        lines.append(f"| {int(r['order'])} | {r['action']} |")
    lines.append("")

    lines.append("## Principles honored")
    lines.append("- spectroscopy-first, biochemical-theme level (no molecule overclaiming)")
    lines.append("- ΔBSV-first analysis (labels used ONLY for evaluation, not in BSV computation)")
    lines.append("- GAIRA canonical preprocessing preserved; paper-specific normalizations (D0_C0 / Si 642 / k-means filter) deliberately not replicated")
    lines.append("- OTC MSS layer flagged as candidate-level noise (registry lacks ibuprofen / ASA / paracetamol templates)")
    lines.append("")
    (REPORTS / "REPORT_shine_otc_dataset_audit_v1.md").write_text("\n".join(lines))


def write_audit():
    txt = [
        "# gaira_base_4_shine_otc_dataset_audit_v1 — audit log",
        f"date: {datetime.now().isoformat()}",
        "",
        "## Scope",
        "- STRICT pre-ingestion audit + ingestion-pipeline + evaluation plan",
        "- NO GAIRA scoring performed",
        "- NO classifier trained",
        "- NO molecule-level claims made",
        "- NO engine / MSS / motif / BSV / preprocessing changes",
        "",
        "## Inputs (read-only)",
        "- /Volumes/SSD_Rad/GAIRA_DATA/raw/shine_ev_sers/SERS-Hepatotoxicity_DATA_CODE_FIGURE/",
        "    Figure4/data/Set9/ (18,871 spectra across 12 cohort dirs)",
        "    Figure4/data/Set10/ (4,775 spectra; D2 only)",
        "    Figure4/Fig4D/code/Fig4D.m (pixel→wavenumber calibration source)",
        "    Figure4/Fig4C/data/combined_wavenumbers.mat (189-point unit16 vector; unclear role)",
        "- /Volumes/SSD_Rad/GAIRA_DATA/raw/otc_drugs/ (7 xlsx files, 300 total spectra)",
        "",
        "## Outputs",
        "- tables/shine_file_structure_v1.csv",
        "- tables/shine_spectral_format_v1.csv",
        "- tables/shine_experimental_variables_v1.csv",
        "- tables/shine_qc_sample_v1.csv",
        "- tables/otc_file_structure_v1.csv",
        "- tables/otc_spectral_format_v1.csv",
        "- tables/otc_experimental_variables_v1.csv",
        "- tables/otc_qc_per_spectrum_v1.csv",
        "- tables/preprocessing_comparison_v1.csv",
        "- tables/ingestion_pipeline_v1.csv",
        "- tables/evaluation_plan_v1.csv",
        "- tables/expected_biochemistry_reference_v1.csv",
        "- tables/risks_and_issues_v1.csv",
        "- tables/next_steps_v1.csv",
        "- reports/REPORT_shine_otc_dataset_audit_v1.md",
    ]
    (AUDIT / "gaira_base_4_shine_otc_dataset_audit_v1_audit_log.md").write_text("\n".join(txt))


# ──────────────────────────────────────────────────────────────────────
# Entrypoint
# ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 78)
    print("gaira_base_4_shine_otc_dataset_audit_v1 — PRE-INGESTION AUDIT")
    print("=" * 78)
    shine_struct = shine_file_structure()
    shine_fmt, wn_axis = shine_spectral_format()
    shine_vars = shine_experimental_variables(shine_struct)
    shine_qc   = shine_qc_probe()

    otc_struct, otc_fmt = otc_file_structure_and_spectral_format()
    otc_vars = otc_experimental_variables(otc_struct)
    otc_qc   = otc_qc_probe()

    preproc_df = preprocessing_comparison()
    ingest_df  = ingestion_pipeline()
    eval_df    = evaluation_plan()
    priors_df  = expected_biochemical_mapping()
    risks_df   = risks_list(shine_struct, otc_struct)
    next_df    = next_steps()

    write_report(shine_struct, shine_fmt, shine_vars, shine_qc,
                    otc_struct, otc_fmt, otc_vars, otc_qc,
                    preproc_df, ingest_df, eval_df, priors_df, risks_df, next_df)
    write_audit()
    try:
        shutil.copy(__file__, CODE_SNAPSHOT / Path(__file__).name)
    except Exception:
        pass
    print("[done] pre-ingestion audit complete")


if __name__ == "__main__":
    main()
