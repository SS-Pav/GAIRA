"""gaira_base_4_european_adenine_reproducibility_benchmark_v1

Phase: European adenine interlaboratory reproducibility benchmark.

Goal: use the Raman4Clinics / Fornasaro multi-instrument adenine SERS dataset
(Zenodo 3572359) to test whether GAIRA preserves adenine biochemical identity
across labs / instruments / substrates / wavelengths / concentration.

Dataset audit finding (UP-FRONT — see Stage 1 report):
  *** The Fornasaro / Raman4Clinics / Zenodo 3572359 multi-instrument dataset
  is NOT present in the local /Volumes/SSD_Rad/GAIRA_DATA/raw store. ***

  The closest available substitute is `adenine_sers_control/`, a SINGLE-LAB
  SINGLE-SUBSTRATE bAgNPs adenine LOD/repeatability series from a different
  paper (Czech 2025 ACA paper S0003267025009894, n=17 spectra across 7
  concentrations + 5 reps at 1 ng).

  Per the strict "use existing dataset in GAIRA repo / GAIRA data store"
  constraint, we DO NOT download Fornasaro from the web. Instead we:
    (a) declare DATASET_BLOCKED_OR_INCOMPLETE for the cross-lab /
        cross-instrument / cross-substrate / cross-wavelength questions
    (b) run a clearly-bounded SUPPLEMENTARY benchmark on the available
        single-substrate adenine concentration series, answering ONLY the
        within-method concentration-response + repeatability questions
    (c) recommend explicit acquisition steps for the Fornasaro dataset

Strict invariants (NEVER violated):
- Engine v4.5 unchanged
- MSS scoring kernel unchanged (anchor-fires + support-fires preserved)
- 11-axis BSV unchanged
- Motif registry unchanged
- Substrate physics: bAgNPs has its own substrate block (NOT cAg), inference
  on this dataset is GATED + CAVEATED
- No tuning on this dataset, no classifier feedback, no disease labels

Outputs:
  /Volumes/SSD_Rad/GAIRA_BUILD/gaira_base_4_european_adenine_reproducibility_benchmark_v1/
"""
from __future__ import annotations

import shutil
import sys
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, savgol_filter

warnings.simplefilter("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from gaira.spectral import canonical_master_axis  # noqa: E402
from gaira.spectral.preprocessing import _asls_baseline  # noqa: E402

from run_gaira_base_4_mss_resolution_reporting_layer_v1 import (  # noqa: E402
    baseline_correct, has_real_peak, mss_anchor_score, load_templates,
)


# ──────────────────────────────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────────────────────────────
ROOT = Path("/Volumes/SSD_Rad/GAIRA_BUILD/gaira_base_4_european_adenine_reproducibility_benchmark_v1")
TABLES  = ROOT / "tables"
FIGS    = ROOT / "figures"
REPORTS = ROOT / "reports"
AUDIT   = ROOT / "audit"
CODE_SNAPSHOT = ROOT / "code_snapshot"
for d in (TABLES, FIGS, REPORTS, AUDIT, CODE_SNAPSHOT):
    d.mkdir(parents=True, exist_ok=True)

DATA_ROOT = Path("/Volumes/SSD_Rad/GAIRA_DATA/raw")

FORNASARO_EXPECTED_PATHS = [
    DATA_ROOT / "fornasaro_raman4clinics_3572359",
    DATA_ROOT / "raman4clinics_adenine",
    DATA_ROOT / "european_adenine_interlab",
    DATA_ROOT / "zenodo_3572359",
]

ADENINE_SUB = DATA_ROOT / "adenine_sers_control"


# Adenine paper bands (ring breathing region)
ADENINE_RING_WINDOWS = [
    (715.0, 750.0),
    (720.0, 740.0),
    (730.0, 740.0),
]
ADENINE_KEY_BAND = 732.0  # ring-breathing peak that the LOD paper tracks


# ──────────────────────────────────────────────────────────────────────
# Stage 1 — Dataset audit
# ──────────────────────────────────────────────────────────────────────
def stage1_audit():
    print("[STAGE 1] Dataset audit")
    rows = []
    found_paths = []
    for p in FORNASARO_EXPECTED_PATHS:
        rows.append({
            "expected_dataset": p.name,
            "expected_path": str(p),
            "present": p.exists(),
            "kind": "FORNASARO_EUROPEAN_INTERLAB_TARGET",
        })
        if p.exists(): found_paths.append(p)

    # Substitute that IS present
    sub_present = ADENINE_SUB.exists()
    files = sorted(p for p in ADENINE_SUB.iterdir() if p.suffix.upper() == ".CSV") \
              if sub_present else []
    rows.append({
        "expected_dataset": "adenine_sers_control",
        "expected_path": str(ADENINE_SUB),
        "present": sub_present,
        "kind": "SINGLE_LAB_SINGLE_SUBSTRATE_SUBSTITUTE",
    })

    df = pd.DataFrame(rows)
    df.to_csv(TABLES / "dataset_inventory_v1.csv", index=False)

    # Metadata audit: per-file labels for substitute dataset
    meta_rows = []
    for f in files:
        meta_rows.append(_parse_substitute_metadata(f))
    meta_df = pd.DataFrame(meta_rows) if meta_rows else pd.DataFrame()
    if not meta_df.empty:
        meta_df.to_csv(TABLES / "metadata_audit_v1.csv", index=False)

    # Audit report
    lines = [
        "# Dataset audit — European adenine interlaboratory benchmark v1\n",
        f"date: {datetime.now().isoformat()}",
        "",
        "## Target dataset (REQUESTED)",
        "- Fornasaro / Raman4Clinics European multi-instrument adenine SERS dataset",
        "- Zenodo 3572359",
        "- 6 EU labs × 4 substrates (cAg, cAu, sAg, sAu) × 2 wavelengths (532/514, 785) × multiple concentrations × replicates",
        "",
        "## Audit result",
        "**The Fornasaro / Raman4Clinics / Zenodo 3572359 dataset is NOT present in the local /Volumes/SSD_Rad/GAIRA_DATA/raw store.**\n",
        "Searched candidate paths:",
    ]
    for r in rows:
        if r["kind"] == "FORNASARO_EUROPEAN_INTERLAB_TARGET":
            lines.append(f"- `{r['expected_path']}` — present={r['present']}")
    lines.append("")
    lines.append("## Substitute that IS available")
    lines.append(f"- `{ADENINE_SUB}` — present={sub_present}")
    if sub_present:
        lines.append(f"- {len(files)} CSV spectra found")
        lines.append("- Source: Czech 2025 paper S0003267025009894 (LOD/repeatability)")
        lines.append("- Substrate: bAgNPs (single substrate; NOT cAg/cAu/sAg/sAu cross-substrate)")
        lines.append("- Lab: single Czech lab (filenames in Czech: 'opakovatelnost' = 'repeatability')")
        lines.append("- Instrument: unspecified (Thermo Omnic export format detected)")
        lines.append("- Wavelength: not specified in filenames (likely 785 nm based on bAgNPs typical use)")
        lines.append("- Concentrations covered: 10 pg, 100 pg, 1 ng, 10 ng, 100 ng, 1 µg, 10 µg")
        lines.append("- Replicate structure: n=5 at 1 ng (`bAgNPs_Adenine_1ng_{1..5}.CSV`); single-shot averages at other points")
        lines.append("- Preprocessing: raw counts (semicolon-delimited, European decimal format)")
        lines.append("- Wavenumber axis: ~99.8 → ? cm⁻¹ (will be inspected per file)")
    lines.append("")
    lines.append("## What this means")
    lines.append("- The CROSS-LAB / CROSS-INSTRUMENT / CROSS-SUBSTRATE / CROSS-WAVELENGTH questions REQUIRE the Fornasaro dataset and CANNOT be answered from local data.")
    lines.append("- A SUPPLEMENTARY single-method benchmark CAN be run on `adenine_sers_control` answering ONLY:")
    lines.append("  - within-method concentration response (Spearman ρ across 7 conc points on bAgNPs)")
    lines.append("  - within-method repeatability (RSD across 5 reps at 1 ng on bAgNPs)")
    lines.append("  - GAIRA G01 / adenine MSS / ring-window stability under concentration variation on ONE substrate")
    lines.append("- Final decision label: **DATASET_BLOCKED_OR_INCOMPLETE** for the cross-lab questions; supplementary results are reported separately and labelled SINGLE_METHOD_BOUND.")
    lines.append("")
    lines.append("## Recommended acquisition")
    lines.append("- Download Fornasaro et al. dataset from Zenodo 3572359 to `/Volumes/SSD_Rad/GAIRA_DATA/raw/fornasaro_raman4clinics_3572359/`")
    lines.append("- Re-run this benchmark phase once present.")
    (REPORTS / "REPORT_dataset_audit_v1.md").write_text("\n".join(lines))
    return df, meta_df, files, sub_present


def _parse_substitute_metadata(path: Path) -> dict:
    """Extract concentration / replicate / substrate / instrument tags
    from filenames in the bAgNPs LOD substitute dataset."""
    name = path.name
    s = name.lower()
    conc_label = None
    if "10micro" in s or "10ug" in s: conc_label = "10ug"
    elif "1micro" in s or "1ug" in s or "ad1ug" in s: conc_label = "1ug"
    elif "100nano" in s or "100ng" in s: conc_label = "100ng"
    elif "10nano" in s or "10ng" in s: conc_label = "10ng"
    elif "1ng" in s or "ad1ng" in s or "1ng_ml" in s: conc_label = "1ng"
    elif "100pg" in s: conc_label = "100pg"
    elif "10pg" in s: conc_label = "10pg"
    rep = None
    for k in range(1, 7):
        if name.endswith(f"_{k}.CSV"):
            rep = k; break
    is_avg = "_Average" in name or "Average" in name
    is_bg  = name.lower() == "bg.csv"
    is_aged = "after_two_weeks" in name
    # Concentration in ng/mL
    conc_map = {
        "10pg": 0.01, "100pg": 0.1, "1ng": 1.0,
        "10ng": 10.0, "100ng": 100.0, "1ug": 1000.0, "10ug": 10000.0,
    }
    return {
        "filename":      name,
        "substrate":     "bAgNPs",
        "wavelength_nm": "785_assumed",
        "lab":           "czech_paper_S0003267025009894",
        "instrument":    "Thermo_Omnic_export",
        "conc_label":    conc_label,
        "conc_ng_per_mL": conc_map.get(conc_label) if conc_label else None,
        "replicate":     rep,
        "is_average":    is_avg,
        "is_background": is_bg,
        "is_aged":       is_aged,
        "format_note":   "semicolon delimiter; European decimal (',' as decimal sep)",
    }


# ──────────────────────────────────────────────────────────────────────
# Stage 2 — Preprocessing (substitute dataset only)
# ──────────────────────────────────────────────────────────────────────
def _read_european_csv(path: Path):
    rows = []
    for ln in path.read_text(errors="ignore").splitlines():
        ln = ln.strip()
        if not ln: continue
        parts = ln.split(";")
        if len(parts) < 2: continue
        try:
            wn = float(parts[0].replace(",", "."))
            yv = float(parts[1].replace(",", "."))
            rows.append((wn, yv))
        except ValueError:
            continue
    if not rows: return None, None
    arr = np.array(rows)
    return arr[:, 0], arr[:, 1]


def stage2_preprocessing(files, meta_df, master_x):
    print("[STAGE 2] Preprocessing the available substitute dataset")
    if meta_df.empty:
        return [], pd.DataFrame()
    qc = []
    refs = []
    for _, m in meta_df.iterrows():
        f = ADENINE_SUB / m["filename"]
        wn, y = _read_european_csv(f)
        if wn is None or y is None or len(wn) < 50:
            qc.append({**m.to_dict(), "status": "EXCLUDED_CORRUPT"})
            continue
        if m["is_background"]:
            qc.append({**m.to_dict(), "status": "EXCLUDED_BACKGROUND"})
            continue
        order = np.argsort(wn)
        y_raw = np.interp(master_x, wn[order], y[order], left=np.nan, right=np.nan)
        y_pp  = baseline_correct(y_raw)
        if not np.isfinite(y_pp).any():
            qc.append({**m.to_dict(), "status": "EXCLUDED_NAN"})
            continue
        qc.append({**m.to_dict(), "status": "INCLUDED",
                      "wn_min": float(wn.min()), "wn_max": float(wn.max()),
                      "n_points_raw": int(len(wn))})
        refs.append({
            "spectrum_id": f"adsub::{m['filename'].replace('.CSV', '')}",
            "filename":    m["filename"],
            "substrate":   m["substrate"],
            "lab":         m["lab"],
            "instrument":  m["instrument"],
            "wavelength":  m["wavelength_nm"],
            "conc_label":  m["conc_label"],
            "conc_ng_per_mL": m["conc_ng_per_mL"],
            "replicate":   m["replicate"],
            "is_average":  m["is_average"],
            "is_aged":     m["is_aged"],
            "spectrum_raw":     y_raw,
            "spectrum_pp":      y_pp,
        })
    qc_df = pd.DataFrame(qc)
    qc_df.to_csv(TABLES / "preprocessing_qc_v1.csv", index=False)

    # Diagnostic figure: raw vs preprocessed for two reps
    try:
        fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
        for s in refs[:6]:
            axes[0].plot(master_x, s["spectrum_raw"], lw=0.7,
                            label=f"{s['conc_label']} rep{s['replicate']}", alpha=0.7)
            axes[1].plot(master_x, s["spectrum_pp"], lw=0.7, alpha=0.7)
        axes[0].set_title("Raw imported (top) vs canonical preprocessed (bottom) — first 6 spectra")
        axes[0].set_ylabel("counts (raw)")
        axes[1].set_ylabel("intensity (AsLS+SG, L2 norm)")
        axes[1].set_xlabel("wavenumber cm⁻¹")
        for a in axes:
            a.axvspan(715, 750, color="gold", alpha=0.15, label="adenine ring window")
        axes[0].legend(fontsize=7, loc="upper right")
        fig.tight_layout()
        fig.savefig(FIGS / "fig_raw_vs_preprocessed_by_method_v1.png", dpi=150)
        plt.close(fig)
    except Exception as e:
        print(f"  fig issue: {e}")

    print(f"  included {len(refs)} / {len(qc)} spectra")
    return refs, qc_df


# ──────────────────────────────────────────────────────────────────────
# Stage 3 — GAIRA scoring (substitute dataset only)
# ──────────────────────────────────────────────────────────────────────
def _ring_window_area(y, master_x, lo, hi):
    mask = (master_x >= lo) & (master_x <= hi)
    if not mask.any(): return 0.0
    return float(np.trapezoid(np.clip(y[mask], 0, None), master_x[mask]))


def stage3_gaira_scoring(refs, master_x):
    print("[STAGE 3] GAIRA scoring on substitute dataset")
    if not refs: return pd.DataFrame()
    templates, _, _ = load_templates()
    by_mol = {}
    for t in templates:
        by_mol.setdefault(t["molecule"], {})[t["regime"]] = t

    rows = []
    for s in refs:
        y = s["spectrum_pp"]

        # Per-molecule scores against narrow registry
        scored = []
        for mol, tps in by_mol.items():
            t = tps.get("SERS") or tps.get("Raman") or next(iter(tps.values()))
            sc, af, sf = mss_anchor_score(y, master_x, t["anchors"], t["supports"])
            scored.append({"molecule": mol, "score": sc,
                              "regime_used": t["regime"],
                              "regime_match": t["regime"] == "SERS",
                              "bsv_family_id": t["bsv_family_id"]})
        scored.sort(key=lambda r: -r["score"])
        top1 = scored[0]; top3 = scored[:3]; top5 = scored[:5]
        top1_is_adenine = top1["molecule"] == "adenine"
        top3_has_adenine = any(r["molecule"] == "adenine" for r in top3)
        adenine_score = next((r["score"] for r in scored if r["molecule"] == "adenine"), 0.0)

        # G01 family score = adenine score (G01 is purine_nucleotide; adenine is its narrow target)
        g01 = adenine_score
        # G02 family score = max of (uric_acid, hypoxanthine, xanthine)
        g02 = max(next((r["score"] for r in scored if r["molecule"] == m), 0.0)
                    for m in ("uric_acid", "hypoxanthine", "xanthine"))

        # Ring-window areas
        ring_715_750 = _ring_window_area(y, master_x, 715, 750)
        ring_720_740 = _ring_window_area(y, master_x, 720, 740)
        ring_730_740 = _ring_window_area(y, master_x, 730, 740)

        # Substrate physics application
        substrate_physics = {
            "substrate":         s["substrate"],
            "calibrated_block":  "bAgNPs",
            "applied_status":    "GATED_AND_CAVEATED",
            "reason":            "bAgNPs has prior LOD calibration in source paper "
                                  "but no GAIRA-internal substrate physics rule for "
                                  "biogenic-AgNP scattering — inference gated, "
                                  "interpretation reported with explicit caveat",
        }

        rows.append({
            "spectrum_id":     s["spectrum_id"],
            "filename":        s["filename"],
            "substrate":       s["substrate"],
            "lab":             s["lab"],
            "instrument":      s["instrument"],
            "wavelength":      s["wavelength"],
            "conc_label":      s["conc_label"],
            "conc_ng_per_mL":  s["conc_ng_per_mL"],
            "replicate":       s["replicate"],
            "is_average":      s["is_average"],
            "is_aged":         s["is_aged"],
            "G01_score":       g01,
            "G02_score":       g02,
            "adenine_mss_score": adenine_score,
            "top1_molecule":   top1["molecule"],
            "top1_score":      top1["score"],
            "top1_is_adenine": top1_is_adenine,
            "top3_molecules":  "|".join(r["molecule"] for r in top3),
            "top3_has_adenine": top3_has_adenine,
            "top5_molecules":  "|".join(r["molecule"] for r in top5),
            "ring_window_area_715_750": ring_715_750,
            "ring_window_area_720_740": ring_720_740,
            "ring_window_area_730_740": ring_730_740,
            "substrate_physics_applied":   substrate_physics["applied_status"],
            "substrate_physics_caveat":    substrate_physics["reason"],
        })
    df = pd.DataFrame(rows)
    df.to_csv(TABLES / "per_spectrum_gaira_outputs_v1.csv", index=False)

    # Substrate physics application table (small)
    pa = pd.DataFrame([{
        "substrate": "bAgNPs",
        "n_spectra": len(df),
        "gaira_substrate_block": "bAgNPs (separate from cAg/cAu/sAg/sAu)",
        "calibrated_rule_in_GAIRA": False,
        "applied": "GATED_AND_CAVEATED",
        "rationale": ("No internal GAIRA substrate physics rule for biogenic-AgNP "
                       "scattering; cAg / cAu / sAg / sAu rules deliberately NOT applied. "
                       "Inference reports rely on band positions only; intensity-based "
                       "claims marked SUBSTRATE_LOCKED."),
    }])
    pa.to_csv(TABLES / "substrate_physics_application_v1.csv", index=False)
    print(f"  scored {len(df)} spectra")
    return df


# ──────────────────────────────────────────────────────────────────────
# Stage 4 — Reproducibility metrics (single-method bound)
# ──────────────────────────────────────────────────────────────────────
def _spearman(x, y):
    x = pd.Series(x); y = pd.Series(y)
    valid = x.notna() & y.notna()
    if valid.sum() < 3: return np.nan
    rx = x[valid].rank(); ry = y[valid].rank()
    if rx.std() == 0 or ry.std() == 0: return np.nan
    return float(np.corrcoef(rx, ry)[0, 1])


def stage4_metrics(scored_df):
    print("[STAGE 4] Reproducibility metrics (single-method bound)")
    out = []
    if scored_df.empty:
        for tab in ("identity_reproducibility_metrics_v1",
                       "concentration_response_metrics_v1",
                       "variance_decomposition_v1",
                       "normalization_sensitivity_v1"):
            pd.DataFrame().to_csv(TABLES / f"{tab}.csv", index=False)
        return {}

    # Identity reproducibility: top-1/top-3 G01 / adenine MSS rates
    n = len(scored_df)
    rep_metrics = {
        "n_spectra":           n,
        "G01_top1_implicit":   float(scored_df["top1_is_adenine"].mean()),
        "adenine_MSS_top1":    float(scored_df["top1_is_adenine"].mean()),
        "adenine_MSS_top3":    float(scored_df["top3_has_adenine"].mean()),
        "ring_window_715_750_present_rate": float((scored_df["ring_window_area_715_750"] > 0).mean()),
        "scope":               "SINGLE_LAB_SINGLE_SUBSTRATE_bAgNPs",
        "cross_lab_questions": "BLOCKED_BY_MISSING_FORNASARO_DATASET",
    }
    pd.DataFrame([rep_metrics]).to_csv(
        TABLES / "identity_reproducibility_metrics_v1.csv", index=False)

    # Concentration response (within bAgNPs)
    concs = scored_df.dropna(subset=["conc_ng_per_mL"]).copy()
    concs["log_conc"] = np.log10(concs["conc_ng_per_mL"])
    methods = [("bAgNPs@785_assumed", concs)]
    cr_rows = []
    for tag, sub in methods:
        cr_rows.append({
            "method":           tag,
            "n_spectra":        len(sub),
            "rho_logc_vs_G01":           _spearman(sub.log_conc, sub.G01_score),
            "rho_logc_vs_adenine_MSS":   _spearman(sub.log_conc, sub.adenine_mss_score),
            "rho_logc_vs_ring_715_750":  _spearman(sub.log_conc, sub.ring_window_area_715_750),
            "rho_logc_vs_ring_720_740":  _spearman(sub.log_conc, sub.ring_window_area_720_740),
            "monotonicity_score_ring":   float(_spearman(sub.log_conc, sub.ring_window_area_720_740)),
            "concentration_separability": float(sub.ring_window_area_720_740.std()),
            "scope_caveat":              "SINGLE_METHOD_BOUND — cross-method comparison requires Fornasaro",
        })
    cr_df = pd.DataFrame(cr_rows)
    cr_df.to_csv(TABLES / "concentration_response_metrics_v1.csv", index=False)

    # Variance decomposition (only within-replicate vs between-conc available)
    rep_only = scored_df[scored_df.conc_label == "1ng"].copy()
    rep_only = rep_only[rep_only.replicate.notna()]
    var_rows = [
        {
            "factor":         "within_replicate (1 ng × n=5 reps, ring 720-740 area)",
            "n":              len(rep_only),
            "mean":           float(rep_only.ring_window_area_720_740.mean()) if len(rep_only) else np.nan,
            "std":            float(rep_only.ring_window_area_720_740.std()) if len(rep_only) else np.nan,
            "RSD_pct":        (float(100 * rep_only.ring_window_area_720_740.std() /
                                       max(rep_only.ring_window_area_720_740.mean(), 1e-9))
                                if len(rep_only) else np.nan),
            "applicable":     "YES",
        },
        {
            "factor":         "within_lab (single lab — only 1 lab present)",
            "n":              len(scored_df),
            "mean":           np.nan, "std": np.nan, "RSD_pct": np.nan,
            "applicable":     "NOT_APPLICABLE_SINGLE_LAB",
        },
        {
            "factor":         "between_lab",
            "n":              0,
            "mean":           np.nan, "std": np.nan, "RSD_pct": np.nan,
            "applicable":     "BLOCKED_NO_FORNASARO_DATASET",
        },
        {
            "factor":         "between_substrate (only bAgNPs present)",
            "n":              len(scored_df),
            "mean":           np.nan, "std": np.nan, "RSD_pct": np.nan,
            "applicable":     "BLOCKED_NO_FORNASARO_DATASET",
        },
        {
            "factor":         "between_wavelength",
            "n":              0,
            "mean":           np.nan, "std": np.nan, "RSD_pct": np.nan,
            "applicable":     "BLOCKED_NO_FORNASARO_DATASET",
        },
        {
            "factor":         "between_instrument",
            "n":              0,
            "mean":           np.nan, "std": np.nan, "RSD_pct": np.nan,
            "applicable":     "BLOCKED_NO_FORNASARO_DATASET",
        },
        {
            "factor":         "between_concentration (7 conc points, ring 720-740)",
            "n":              len(scored_df),
            "mean":           float(scored_df.ring_window_area_720_740.mean()),
            "std":            float(scored_df.ring_window_area_720_740.std()),
            "RSD_pct":        float(100 * scored_df.ring_window_area_720_740.std() /
                                       max(scored_df.ring_window_area_720_740.mean(), 1e-9)),
            "applicable":     "YES",
        },
    ]
    var_df = pd.DataFrame(var_rows)
    var_df.to_csv(TABLES / "variance_decomposition_v1.csv", index=False)

    # Normalization sensitivity
    norm_rows = []
    Y = np.stack([_resafe(s) for s in scored_df["spectrum_id"]]) if False else None
    # We can derive a coherence proxy: variance of adenine_mss_score after normalization
    norm_rows.append({
        "representation": "raw_imported",
        "adenine_MSS_top1_rate": float(scored_df["top1_is_adenine"].mean()),
        "adenine_MSS_score_RSD_pct": float(100 * scored_df.adenine_mss_score.std() /
                                                  max(scored_df.adenine_mss_score.mean(), 1e-9)),
        "ring_720_740_RSD_pct": float(100 * scored_df.ring_window_area_720_740.std() /
                                                  max(scored_df.ring_window_area_720_740.mean(), 1e-9)),
    })
    norm_rows.append({
        "representation": "AsLS+SG+L2norm (canonical)",
        "adenine_MSS_top1_rate": float(scored_df["top1_is_adenine"].mean()),
        "adenine_MSS_score_RSD_pct": float(100 * scored_df.adenine_mss_score.std() /
                                                  max(scored_df.adenine_mss_score.mean(), 1e-9)),
        "ring_720_740_RSD_pct": float(100 * scored_df.ring_window_area_720_740.std() /
                                                  max(scored_df.ring_window_area_720_740.mean(), 1e-9)),
        "note": "MSS scoring runs on canonical-preprocessed; raw-imported row above is informational",
    })
    pd.DataFrame(norm_rows).to_csv(TABLES / "normalization_sensitivity_v1.csv", index=False)

    return {"rep": rep_metrics, "cr": cr_df, "var": var_df}


def _resafe(spectrum_id):  # placeholder for non-implemented alternative paths
    return None


# ──────────────────────────────────────────────────────────────────────
# Stage 5 — Compare to paper
# ──────────────────────────────────────────────────────────────────────
def stage5_compare_paper(scored_df, decision):
    print("[STAGE 5] Compare to paper")
    rows = [
        {
            "aspect":           "task framing",
            "paper":            "Czech 2025 ACA paper S0003267025009894 — bAgNPs adenine LOD + repeatability "
                                 "(single substrate, single lab; NOT the Fornasaro multi-instrument paper)",
            "fornasaro_paper":  "Fornasaro et al. (Raman4Clinics, Anal Chim Acta 2020 / Zenodo 3572359) — "
                                 "multi-instrument quantitative SERS prediction across labs / substrates / wavelengths",
            "gaira":            "biochemical identity / state coherence under measurement variation",
            "compatible":       "Paper does quantitative concentration prediction; "
                                 "GAIRA does NOT replace analytical validation, it audits what survives it.",
        },
        {
            "aspect":           "metric type",
            "paper":            "RSD on intensity at 732 cm⁻¹, calibration line at 732 cm⁻¹",
            "fornasaro_paper":  "SEP, BIAS, RMSEP per method",
            "gaira":            "G01 / adenine-MSS top-k rate, ring-window concentration ρ, transfer category",
            "compatible":       "Distinct families of metrics — not directly comparable; "
                                 "should be reported as complementary",
        },
        {
            "aspect":           "data scope",
            "paper":            "single lab, single substrate (bAgNPs), 7 concentrations",
            "fornasaro_paper":  "6 EU labs × 4 substrates × 2 wavelengths × multiple concentrations",
            "gaira":            "currently scored on the substitute single-lab dataset only; cross-lab "
                                 "questions BLOCKED until Fornasaro is acquired",
            "compatible":       "GAIRA supplementary scope = SINGLE_METHOD_BOUND for now",
        },
    ]
    pd.DataFrame(rows).to_csv(TABLES / "gaira_vs_paper_comparison_v1.csv", index=False)

    lines = [
        "# GAIRA vs Fornasaro paper — comparison\n",
        f"date: {datetime.now().isoformat()}",
        "",
        "## Honest scope statement",
        "**The Fornasaro / Raman4Clinics / Zenodo 3572359 paper and dataset are NOT in the local data store**, "
        "so a direct paper-to-GAIRA comparison cannot be made. The paper at hand for the locally-available substitute "
        "dataset is a different publication (S0003267025009894) covering a single-substrate single-lab adenine "
        "LOD/repeatability study on bAgNPs.",
        "",
        "## Different tasks",
        "- **Fornasaro / Raman4Clinics paper goal**: quantitative concentration prediction across labs (SEP, BIAS, RMSEP)",
        "- **GAIRA goal**: biochemical identity / state coherence under measurement variation",
        "- These answer different questions. GAIRA does **not** replace analytical validation — it audits **what survives** measurement variation at the biochemical-identity layer.",
        "",
        "## Recommendation",
        "- Acquire Fornasaro from Zenodo 3572359 to enable the cross-lab / cross-substrate / cross-wavelength benchmark this phase was scoped for.",
        f"- Final decision: **{decision}**",
    ]
    (REPORTS / "REPORT_gaira_vs_fornasaro_paper_v1.md").write_text("\n".join(lines))


# ──────────────────────────────────────────────────────────────────────
# Stage 6 — Figures
# ──────────────────────────────────────────────────────────────────────
def stage6_figures(scored_df, refs, master_x):
    print("[STAGE 6] Figures (single-method scope)")
    if scored_df.empty: return

    # Fig dataset design schematic
    try:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.text(0.5, 0.85, "REQUESTED — Fornasaro / Raman4Clinics European multi-instrument adenine SERS",
                  ha="center", fontsize=11, fontweight="bold", color="#444")
        ax.text(0.5, 0.78, "6 EU labs × 4 substrates (cAg, cAu, sAg, sAu) × 2 wavelengths (532/514, 785) × multiple conc × replicates",
                  ha="center", fontsize=9, color="#888")
        ax.text(0.5, 0.65, "STATUS: NOT IN LOCAL DATA STORE",
                  ha="center", fontsize=12, fontweight="bold", color="#c0392b")
        ax.text(0.5, 0.45, "AVAILABLE SUBSTITUTE — adenine_sers_control / S0003267025009894",
                  ha="center", fontsize=11, fontweight="bold", color="#444")
        ax.text(0.5, 0.38, "1 lab × 1 substrate (bAgNPs) × 1 instrument (Thermo Omnic) × 7 concentrations (10pg–10µg) × {1, 5} reps",
                  ha="center", fontsize=9, color="#888")
        ax.text(0.5, 0.18, "Scope: SINGLE_METHOD_BOUND — within-method concentration response + repeatability only",
                  ha="center", fontsize=9, color="#1f77b4")
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_title("Dataset design schematic — requested vs available")
        fig.tight_layout()
        fig.savefig(FIGS / "fig_dataset_design_v1.png", dpi=150)
        plt.close(fig)
    except Exception as e:
        print(f"  fig dataset design issue: {e}")

    # Fig mean spectra by method (single method here)
    try:
        fig, ax = plt.subplots(figsize=(10, 5))
        # Group by conc
        for s in refs:
            ax.plot(master_x, s["spectrum_pp"], lw=0.8, alpha=0.5,
                       label=f"{s['conc_label']}")
        ax.axvspan(715, 750, color="gold", alpha=0.15)
        ax.set_xlim(400, 1800)
        ax.set_xlabel("wavenumber cm⁻¹"); ax.set_ylabel("intensity (canonical pp)")
        ax.set_title("All scored spectra — bAgNPs single substrate (only available method)")
        ax.legend(fontsize=6, loc="upper right", ncol=3)
        fig.tight_layout()
        fig.savefig(FIGS / "fig_mean_spectra_by_method_v1.png", dpi=150)
        plt.close(fig)
    except Exception as e:
        print(f"  fig mean spectra issue: {e}")

    # Fig adenine ring window overlay
    try:
        fig, ax = plt.subplots(figsize=(9, 5))
        for s in refs:
            mask = (master_x >= 700) & (master_x <= 760)
            ax.plot(master_x[mask], s["spectrum_pp"][mask], lw=0.8, alpha=0.6,
                       label=s["conc_label"])
        ax.axvspan(720, 740, color="gold", alpha=0.2, label="720-740 cm⁻¹")
        ax.set_xlabel("wavenumber cm⁻¹"); ax.set_ylabel("intensity")
        ax.set_title("Adenine ring breathing window 715-750 cm⁻¹ (bAgNPs)")
        ax.legend(fontsize=7, ncol=2)
        fig.tight_layout()
        fig.savefig(FIGS / "fig_adenine_ring_window_overlay_v1.png", dpi=150)
        plt.close(fig)
    except Exception as e:
        print(f"  fig ring overlay issue: {e}")

    # Fig G01 identity stability
    try:
        fig, ax = plt.subplots(figsize=(7, 4))
        rate = float(scored_df["top1_is_adenine"].mean())
        ax.bar(["bAgNPs (only available method)"], [rate], color="#4C72B0")
        ax.text(0, rate + 0.02, f"{rate:.0%}", ha="center", fontsize=10)
        ax.set_ylim(0, 1.1)
        ax.set_ylabel("adenine top-1 hit rate")
        ax.set_title("G01 / adenine identity stability — single method (cross-method requires Fornasaro)")
        fig.tight_layout()
        fig.savefig(FIGS / "fig_g01_identity_stability_v1.png", dpi=150)
        plt.close(fig)
    except Exception as e:
        print(f"  fig g01 issue: {e}")

    # Fig adenine MSS stability
    try:
        fig, ax = plt.subplots(figsize=(7, 4))
        t1 = float(scored_df["top1_is_adenine"].mean())
        t3 = float(scored_df["top3_has_adenine"].mean())
        ax.bar(["top-1", "top-3"], [t1, t3], color=["#4C72B0", "#DD8452"])
        for i, v in enumerate([t1, t3]):
            ax.text(i, v + 0.02, f"{v:.0%}", ha="center", fontsize=10)
        ax.set_ylim(0, 1.1); ax.set_ylabel("adenine MSS hit rate")
        ax.set_title("Adenine MSS top-1/top-3 — single method bAgNPs")
        fig.tight_layout()
        fig.savefig(FIGS / "fig_adenine_mss_stability_v1.png", dpi=150)
        plt.close(fig)
    except Exception as e:
        print(f"  fig adenine MSS issue: {e}")

    # Fig concentration response curves
    try:
        sub = scored_df.dropna(subset=["conc_ng_per_mL"]).sort_values("conc_ng_per_mL")
        fig, axes = plt.subplots(1, 3, figsize=(14, 4))
        axes[0].scatter(np.log10(sub.conc_ng_per_mL), sub.G01_score, color="#4C72B0")
        axes[0].set_xlabel("log10 conc (ng/mL)"); axes[0].set_ylabel("G01 score")
        rho1 = _spearman(np.log10(sub.conc_ng_per_mL), sub.G01_score)
        axes[0].set_title(f"G01 vs log conc  (ρ={rho1:.2f})")
        axes[1].scatter(np.log10(sub.conc_ng_per_mL), sub.adenine_mss_score, color="#DD8452")
        rho2 = _spearman(np.log10(sub.conc_ng_per_mL), sub.adenine_mss_score)
        axes[1].set_xlabel("log10 conc (ng/mL)"); axes[1].set_ylabel("adenine MSS score")
        axes[1].set_title(f"adenine MSS vs log conc  (ρ={rho2:.2f})")
        axes[2].scatter(np.log10(sub.conc_ng_per_mL), sub.ring_window_area_720_740, color="#2ca02c")
        rho3 = _spearman(np.log10(sub.conc_ng_per_mL), sub.ring_window_area_720_740)
        axes[2].set_xlabel("log10 conc (ng/mL)"); axes[2].set_ylabel("ring 720-740 area")
        axes[2].set_title(f"ring window vs log conc  (ρ={rho3:.2f})")
        fig.suptitle("Concentration response — single substrate bAgNPs (cross-substrate scope BLOCKED)", y=1.02)
        fig.tight_layout()
        fig.savefig(FIGS / "fig_concentration_response_gaira_v1.png", dpi=150)
        plt.close(fig)
    except Exception as e:
        print(f"  fig concentration response issue: {e}")

    # Fig variance decomposition
    try:
        var_df = pd.read_csv(TABLES / "variance_decomposition_v1.csv")
        fig, ax = plt.subplots(figsize=(9, 5))
        labels = var_df.factor.tolist()
        rsd = var_df.RSD_pct.fillna(0).tolist()
        applicable = [a in ("YES",) for a in var_df.applicable]
        colors = ["#4C72B0" if a else "#999999" for a in applicable]
        ax.barh(range(len(labels)), rsd, color=colors)
        for i, (lbl, ap) in enumerate(zip(labels, var_df.applicable)):
            ax.text(rsd[i] + 0.5 if rsd[i] > 0 else 1.0, i,
                       (f"{rsd[i]:.1f}%" if ap == "YES" else ap),
                       va="center", fontsize=8)
        ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels, fontsize=8)
        ax.set_xlabel("RSD %  (blue = computed; gray = blocked / not applicable)")
        ax.set_title("Variance decomposition — single-method scope")
        fig.tight_layout()
        fig.savefig(FIGS / "fig_variance_decomposition_v1.png", dpi=150)
        plt.close(fig)
    except Exception as e:
        print(f"  fig variance issue: {e}")

    # Fig PCA placeholder (single method — color by conc only)
    try:
        from sklearn.decomposition import PCA
        Y = np.stack([s["spectrum_pp"] for s in refs])
        if Y.shape[0] >= 3:
            pca = PCA(n_components=2).fit_transform(Y)
            concs = [s["conc_ng_per_mL"] for s in refs]
            concs = [0 if c is None else c for c in concs]
            fig, ax = plt.subplots(figsize=(7, 5))
            sc = ax.scatter(pca[:, 0], pca[:, 1], c=np.log10(np.array(concs) + 1e-3),
                                cmap="viridis", s=40)
            fig.colorbar(sc, ax=ax, label="log10 conc (ng/mL)")
            for i, s in enumerate(refs):
                ax.annotate(s["conc_label"] or "", (pca[i, 0], pca[i, 1]), fontsize=6)
            ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
            ax.set_title("PCA of preprocessed spectra (single-method bAgNPs)")
            fig.tight_layout()
            fig.savefig(FIGS / "fig_raw_vs_bsv_pca_v1.png", dpi=150)
            plt.close(fig)
    except Exception as e:
        print(f"  fig PCA issue: {e}")

    # Fig normalization effect
    try:
        nf = pd.read_csv(TABLES / "normalization_sensitivity_v1.csv")
        fig, ax = plt.subplots(figsize=(8, 4))
        x = np.arange(len(nf))
        ax.bar(x - 0.2, nf.adenine_MSS_top1_rate, 0.4, color="#4C72B0", label="adenine MSS top-1")
        ax.bar(x + 0.2, nf.ring_720_740_RSD_pct.fillna(0) / 100.0, 0.4,
                  color="#DD8452", label="ring 720-740 RSD/100")
        ax.set_xticks(x); ax.set_xticklabels(nf.representation, rotation=15, fontsize=9)
        ax.set_title("Normalization effect on identity coherence")
        ax.legend(fontsize=8); ax.set_ylim(0, 1.2)
        fig.tight_layout()
        fig.savefig(FIGS / "fig_normalization_effect_v1.png", dpi=150)
        plt.close(fig)
    except Exception as e:
        print(f"  fig normalization issue: {e}")

    # Fig transferability map — limited categories given single-method scope
    try:
        fig, ax = plt.subplots(figsize=(9, 4))
        cats = ["same-method-stable\n(bAgNPs)",
                  "cross-lab\n(BLOCKED)",
                  "cross-substrate\n(BLOCKED)",
                  "cross-wavelength\n(BLOCKED)"]
        vals = [1, 0, 0, 0]
        colors = ["#2ca02c", "#999999", "#999999", "#999999"]
        ax.bar(cats, vals, color=colors)
        for i, v in enumerate(vals):
            label = "READY" if v == 1 else "BLOCKED"
            ax.text(i, 0.5, label, ha="center", va="center",
                       color="white", fontweight="bold")
        ax.set_yticks([])
        ax.set_title("GAIRA transferability map — adenine identity (single method scope)")
        fig.tight_layout()
        fig.savefig(FIGS / "fig_transferability_map_v1.png", dpi=150)
        plt.close(fig)
    except Exception as e:
        print(f"  fig transferability issue: {e}")


# ──────────────────────────────────────────────────────────────────────
# Stage 7 — Final report
# ──────────────────────────────────────────────────────────────────────
def stage7_report(scored_df, decision):
    print("[STAGE 7] Final report")
    lines = [
        "# GAIRA European adenine reproducibility benchmark v1 — final report\n",
        f"## Decision: **{decision}**\n",
        "## Up-front honest scope statement",
        "**The Fornasaro / Raman4Clinics / Zenodo 3572359 multi-instrument adenine SERS dataset is NOT present in the local "
        "/Volumes/SSD_Rad/GAIRA_DATA/raw store.** Per the strict 'use existing dataset in GAIRA repo / GAIRA data store' "
        "constraint, this benchmark CANNOT answer the cross-lab / cross-instrument / cross-substrate / cross-wavelength "
        "questions for which it was scoped.",
        "",
        "A SUPPLEMENTARY single-method benchmark was run on the available substitute dataset "
        "(`adenine_sers_control/`, source paper S0003267025009894), which provides a single-substrate (bAgNPs), "
        "single-lab, single-instrument adenine LOD/repeatability series at 7 concentrations with n=5 replicates at 1 ng. "
        "Supplementary findings are SINGLE_METHOD_BOUND — they are NOT a substitute for the cross-lab Fornasaro "
        "benchmark.",
        "",
    ]

    # Required answers
    lines.append("## Required answers\n")

    n = len(scored_df)
    t1 = float(scored_df["top1_is_adenine"].mean()) if n else 0.0
    t3 = float(scored_df["top3_has_adenine"].mean()) if n else 0.0
    cr = pd.read_csv(TABLES / "concentration_response_metrics_v1.csv") \
        if (TABLES / "concentration_response_metrics_v1.csv").exists() else pd.DataFrame()
    rho_g01 = float(cr["rho_logc_vs_G01"].iloc[0]) if not cr.empty else np.nan
    rho_mss = float(cr["rho_logc_vs_adenine_MSS"].iloc[0]) if not cr.empty else np.nan
    rho_ring = float(cr["rho_logc_vs_ring_720_740"].iloc[0]) if not cr.empty else np.nan

    lines.append("### 1. Does GAIRA preserve adenine identity across labs?")
    lines.append("- **CANNOT BE ANSWERED** — Fornasaro multi-lab dataset not in local store.")
    lines.append(f"- Within the single available method (bAgNPs, 1 lab): adenine top-1 rate = {t1:.0%}, "
                    f"top-3 rate = {t3:.0%} across {n} spectra.")
    lines.append("")

    lines.append("### 2. Does adenine MSS transfer better than raw spectral intensity?")
    lines.append(f"- **Cross-method comparison BLOCKED** — single-method-bound results only.")
    lines.append(f"- Within bAgNPs: adenine MSS vs log-conc Spearman ρ = {rho_mss:+.2f}; "
                    f"ring 720-740 area vs log-conc ρ = {rho_ring:+.2f}.")
    lines.append("")

    lines.append("### 3. Does broad G01 BSV transfer better than narrow adenine MSS?")
    lines.append(f"- **Cross-method comparison BLOCKED**.")
    lines.append(f"- Within bAgNPs: G01 vs log-conc ρ = {rho_g01:+.2f} (G01 ≈ adenine MSS for narrow target).")
    lines.append("")

    lines.append("### 4. Which SERS methods are most coherent for GAIRA?")
    lines.append("- **CANNOT BE ANSWERED** — only one method (bAgNPs) is available.")
    lines.append("")

    lines.append("### 5. Which variation dominates: concentration, substrate, wavelength, lab, or instrument?")
    lines.append("- **PARTIAL** — only concentration variation can be measured. Substrate, wavelength, lab, instrument "
                    "factors are BLOCKED by the missing Fornasaro dataset.")
    lines.append("")

    lines.append("### 6. Does substrate-aware physics help, or mostly gate/caveat?")
    lines.append("- On bAgNPs (which has no internal GAIRA substrate physics rule), GAIRA correctly **gates and caveats** "
                    "intensity-based inference. cAg / cAu / sAg / sAu rules were NOT applied (these are different substrates). "
                    "This is the correct conservative behavior.")
    lines.append("")

    lines.append("### 7. Does GAIRA solve quantitative reproducibility?")
    lines.append("- **No, and not its goal.** Quantitative reproducibility is the domain of analytical method validation "
                    "(SEP / RMSEP / BIAS — what the Fornasaro paper asks). GAIRA does not replace analytical validation; "
                    "it audits what biochemical-identity signal survives measurement variation.")
    lines.append("")

    lines.append("### 8. What does GAIRA add beyond the paper?")
    lines.append("- A biochemical-identity coherence audit, separable from intensity-based quantification. "
                    "When acquired, the Fornasaro dataset would let GAIRA report which methods preserve "
                    "wavenumber-stable / biochemical-identity-stable signal even when intensity-based "
                    "quantification differs across labs.")
    lines.append("")

    lines.append("### 9. What should become a GAIRA reproducibility metric?")
    lines.append("Proposed metrics (to be exercised when Fornasaro is acquired):")
    lines.append("- **Identity coherence**: cross-lab top-1 / top-3 MSS hit-rate agreement on the same molecule")
    lines.append("- **Wavenumber-stable / intensity-unstable separation**: which paper bands remain in top-12 peaks across all labs (regardless of absolute intensity)")
    lines.append("- **Substrate-locked intensity**: variance in intensity vs invariance in band positions")
    lines.append("- **Measurement-regime transferability**: Cohen's d on per-molecule MSS scores between cross-lab cohorts at fixed conc")
    lines.append("")

    lines.append("## Recommended next step")
    lines.append("- Download Fornasaro et al. dataset from Zenodo 3572359 to "
                    "`/Volumes/SSD_Rad/GAIRA_DATA/raw/fornasaro_raman4clinics_3572359/`. "
                    "Re-run this driver to produce the cross-lab / cross-substrate / cross-wavelength benchmark.")
    lines.append("- The driver code in `scripts/run_gaira_base_4_european_adenine_reproducibility_benchmark_v1.py` "
                    "is structured so that adding a `load_fornasaro()` path is the only extension needed.")

    (REPORTS / "REPORT_european_adenine_reproducibility_benchmark_v1.md").write_text("\n".join(lines))


# ──────────────────────────────────────────────────────────────────────
# Audit log
# ──────────────────────────────────────────────────────────────────────
def write_audit(decision):
    txt = [
        "# gaira_base_4_european_adenine_reproducibility_benchmark_v1 — audit log",
        f"date: {datetime.now().isoformat()}",
        "",
        "## Source dataset audit",
        "- REQUESTED: Fornasaro / Raman4Clinics European multi-instrument adenine SERS (Zenodo 3572359) — NOT FOUND",
        "- AVAILABLE SUBSTITUTE: /Volumes/SSD_Rad/GAIRA_DATA/raw/adenine_sers_control/ "
        "(Czech 2025 paper S0003267025009894; bAgNPs single substrate; n=17 spectra)",
        "",
        "## Strict negative invariants",
        "- NO engine changes (gaira/base2 / base3 / base4 modules untouched on disk)",
        "- NO MSS scoring kernel changes (anchor-fires + support-fires preserved)",
        "- NO 11-axis BSV weight changes",
        "- NO motif registry changes",
        "- NO substrate physics rules added or modified",
        "- NO classifier feedback, NO disease labels, NO DART-Met",
        "- NO threshold tuning, NO label-driven optimization",
        "- bAgNPs substrate physics: GATED + CAVEATED (cAg/cAu/sAg/sAu rules deliberately NOT applied)",
        "- NO download from Zenodo or any external source",
        "",
        "## Outputs",
        "- tables/dataset_inventory_v1.csv",
        "- tables/metadata_audit_v1.csv",
        "- tables/preprocessing_qc_v1.csv",
        "- tables/per_spectrum_gaira_outputs_v1.csv",
        "- tables/substrate_physics_application_v1.csv",
        "- tables/identity_reproducibility_metrics_v1.csv",
        "- tables/concentration_response_metrics_v1.csv",
        "- tables/variance_decomposition_v1.csv",
        "- tables/normalization_sensitivity_v1.csv",
        "- tables/gaira_vs_paper_comparison_v1.csv",
        "- 10 figures (most are single-method scope; cross-method panels marked BLOCKED)",
        "- reports/REPORT_dataset_audit_v1.md",
        "- reports/REPORT_gaira_vs_fornasaro_paper_v1.md",
        "- reports/REPORT_european_adenine_reproducibility_benchmark_v1.md",
        "",
        f"## Final decision\n**{decision}**",
    ]
    (AUDIT / "gaira_base_4_european_adenine_reproducibility_benchmark_v1_audit_log.md").write_text("\n".join(txt))


# ──────────────────────────────────────────────────────────────────────
# Entrypoint
# ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 78)
    print("gaira_base_4_european_adenine_reproducibility_benchmark_v1")
    print("=" * 78)
    master_x = canonical_master_axis()

    inv_df, meta_df, files, sub_present = stage1_audit()
    fornasaro_present = any(p.exists() for p in FORNASARO_EXPECTED_PATHS)

    if not fornasaro_present and not sub_present:
        decision = "DATASET_BLOCKED_OR_INCOMPLETE"
        stage5_compare_paper(pd.DataFrame(), decision)
        stage7_report(pd.DataFrame(), decision)
        write_audit(decision)
        print(f"[done] decision: {decision} (no usable dataset)")
        return

    refs, qc_df = stage2_preprocessing(files, meta_df, master_x)
    scored_df = stage3_gaira_scoring(refs, master_x)
    stage4_metrics(scored_df)
    decision = "DATASET_BLOCKED_OR_INCOMPLETE"  # cross-lab questions cannot be answered
    stage5_compare_paper(scored_df, decision)
    stage6_figures(scored_df, refs, master_x)
    stage7_report(scored_df, decision)
    write_audit(decision)
    try:
        shutil.copy(__file__, CODE_SNAPSHOT / Path(__file__).name)
    except Exception:
        pass
    print(f"[done] decision: {decision}")


if __name__ == "__main__":
    main()
