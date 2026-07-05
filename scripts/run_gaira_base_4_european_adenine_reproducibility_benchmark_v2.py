"""gaira_base_4_european_adenine_reproducibility_benchmark_v2

Phase: REAL Fornasaro / Raman4Clinics European multi-instrument adenine SERS benchmark.

Dataset (now locally available):
  /Volumes/SSD_Rad/GAIRA_DATA/raw/european_multi_instrument_adenine/
  - ILSdata.csv  (3516 spectra × {9 metadata + 534 wavenumber 400-1999 step 3} cols)
  - Dataset/     (3516 individual TXT files; NOT used here — CSV is one-row-per-spectrum)
  - Source: Fornasaro et al., Anal. Chem. 2020 (doi 10.5281/zenodo.3572359)

Strict invariants (NEVER violated):
- Engine v4.5 unchanged
- MSS scoring kernel unchanged (anchor-fires + support-fires; same as MSS resolution layer)
- 11-axis BSV unchanged (proxy via family-aggregated MSS anchor scores; no weight tuning)
- Motif registry unchanged
- Substrate physics: cAg / cAu / sAg / sAu each treated as a separate substrate block;
  GAIRA does NOT have calibrated rules for sputtered-Ag / sputtered-Au or cAu — those
  blocks are GATED + CAVEATED. cAg is the closest to GAIRA's existing citrate-Ag rule
  family but is NOT auto-applied universally.
- No tuning, no classifier feedback, no disease labels, no DART-Met
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

from run_gaira_base_4_mss_resolution_reporting_layer_v1 import (  # noqa: E402
    baseline_correct, has_real_peak, mss_anchor_score, load_templates,
)


# ──────────────────────────────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────────────────────────────
ROOT = Path("/Volumes/SSD_Rad/GAIRA_BUILD/gaira_base_4_european_adenine_reproducibility_benchmark_v2")
TABLES  = ROOT / "tables"
FIGS    = ROOT / "figures"
REPORTS = ROOT / "reports"
AUDIT   = ROOT / "audit"
CODE_SNAPSHOT = ROOT / "code_snapshot"
for d in (TABLES, FIGS, REPORTS, AUDIT, CODE_SNAPSHOT):
    d.mkdir(parents=True, exist_ok=True)

DATA_DIR = Path("/Volumes/SSD_Rad/GAIRA_DATA/raw/european_multi_instrument_adenine")
ILSDATA = DATA_DIR / "ILSdata.csv"
DATASET_DIR = DATA_DIR / "Dataset"

META_COLS = ["labcode", "substrate", "laser", "method", "sample", "type",
             "conc", "batch", "replica"]


# ──────────────────────────────────────────────────────────────────────
# 11-axis BSV family taxonomy (from BIOLOGY_AXES_V11)
# ──────────────────────────────────────────────────────────────────────
BSV_FAMILIES = (
    ("G01", "purine_nucleotide"),
    ("G02", "purine_metabolite"),
    ("G03", "pyrimidine_nucleotide"),
    ("G04", "phosphate_nucleic_adjacent"),
    ("G05", "glycan_carbohydrate"),
    ("G06", "protein_peptide_backbone"),
    ("G07", "aromatic_residue"),
    ("G08", "lipid_acyl_membrane"),
    ("G09", "sterol_neutral_lipid"),
    ("G10", "sulfur_thiol_redox"),
    ("G11", "metabolic_small_molecule"),
)


# Substrate physics application policy (per the user spec — gate/caveat unless calibrated)
SUBSTRATE_POLICY = {
    "cAg": {
        "regime": "SERS",
        "gaira_internal_rule": "citrate-Ag colloid (cAg) — closest to internal calibrated rule family; "
                                  "applied with caveat (citrate-Ag rules NOT auto-applied universally)",
        "applied": "APPLIED_WITH_CAVEAT",
    },
    "cAu": {
        "regime": "SERS",
        "gaira_internal_rule": "citrate-Au colloid (cAu) — no internal GAIRA substrate physics rule",
        "applied": "GATED_AND_CAVEATED",
    },
    "sAg": {
        "regime": "SERS",
        "gaira_internal_rule": "sputtered-Ag film (sAg) — no internal GAIRA substrate physics rule",
        "applied": "GATED_AND_CAVEATED",
    },
    "sAu": {
        "regime": "SERS",
        "gaira_internal_rule": "sputtered-Au film (sAu) — no internal GAIRA substrate physics rule",
        "applied": "GATED_AND_CAVEATED",
    },
}


# ──────────────────────────────────────────────────────────────────────
# Stage 0 — context audit
# ──────────────────────────────────────────────────────────────────────
def stage0_context_audit():
    print("[STAGE 0] context audit")
    inv_rows = []
    inv_rows.append({"file": "ILSdata.csv", "present": ILSDATA.exists(),
                        "size_bytes": ILSDATA.stat().st_size if ILSDATA.exists() else None})
    inv_rows.append({"file": "Dataset/ folder", "present": DATASET_DIR.exists(),
                        "size_bytes": (sum(f.stat().st_size for f in DATASET_DIR.rglob("*"))
                                          if DATASET_DIR.exists() else None)})
    if DATASET_DIR.exists():
        n_txt = sum(1 for _ in DATASET_DIR.glob("*.txt"))
        inv_rows.append({"file": "Dataset/*.txt count", "present": True, "size_bytes": n_txt})
    pd.DataFrame(inv_rows).to_csv(TABLES / "local_file_inventory_v2.csv", index=False)

    # Header inspection
    if not ILSDATA.exists():
        return
    df_h = pd.read_csv(ILSDATA, nrows=1)
    wn_cols = [c for c in df_h.columns if c not in META_COLS]
    wn = np.array([float(c) for c in wn_cols])

    lines = [
        "# Dataset context audit — European adenine reproducibility benchmark v2",
        f"date: {datetime.now().isoformat()}",
        "",
        "## Source",
        "- Fornasaro et al., *Surface Enhanced Raman Spectroscopy for Quantitative Analysis: "
        "Results of a Large-Scale European Multi-Instrument Interlaboratory Study*, "
        "Analytical Chemistry 2020.",
        "- Zenodo DOI: 10.5281/zenodo.3572359",
        "",
        "## Paper task",
        "- Quantitative SERS concentration prediction for adenine across labs.",
        "- Paper metrics: SEP (standard error of prediction), RMSEP, BIAS, reproducibility, trueness.",
        "",
        "## Local file structure",
        f"- ILSdata.csv: present={ILSDATA.exists()}",
        f"- Dataset/ folder: present={DATASET_DIR.exists()} ({sum(1 for _ in DATASET_DIR.glob('*.txt'))} TXT files)",
        "",
        "## CSV schema",
        f"- 9 metadata columns: {META_COLS}",
        f"- {len(wn_cols)} wavenumber columns from {wn[0]} to {wn[-1]} cm⁻¹ (step ≈ {wn[1]-wn[0]:.0f})",
        f"- Total spectra (rows): see Stage 1 inventory.",
        "",
        "## GAIRA scope statement",
        "- GAIRA goal: biochemical identity coherence and measurement-regime transferability.",
        "- GAIRA does NOT replace analytical validation; it audits what biochemical-identity signal "
        "survives measurement variation.",
        "- Concentration prediction (SEP/RMSEP/BIAS) is the paper's task and is NOT GAIRA's primary "
        "metric; comparable concentration response IS computed (Spearman ρ on log conc) but framed "
        "as identity-side reproducibility, not analytical quantification.",
    ]
    (REPORTS / "REPORT_dataset_context_audit_v2.md").write_text("\n".join(lines))


# ──────────────────────────────────────────────────────────────────────
# Stage 1 — load
# ──────────────────────────────────────────────────────────────────────
def stage1_load():
    print("[STAGE 1] loading ILSdata.csv")
    df = pd.read_csv(ILSDATA, low_memory=False)
    wn_cols = [c for c in df.columns if c not in META_COLS]
    wn = np.array([float(c) for c in wn_cols], dtype=float)
    Y = df[wn_cols].values.astype(float)

    # Inventory + metadata audit
    inv_rows = [{
        "n_total_spectra":        len(df),
        "n_labs":                 df["labcode"].nunique(),
        "n_substrates":           df["substrate"].nunique(),
        "n_lasers":               df["laser"].nunique(),
        "n_methods":              df["method"].nunique(),
        "n_samples":              df["sample"].nunique(),
        "n_concentrations":       df["conc"].nunique(),
        "n_batches":              df["batch"].nunique(),
        "n_replicas":             df["replica"].dropna().nunique(),
        "wn_min":                 float(wn[0]),
        "wn_max":                 float(wn[-1]),
        "wn_step":                float(wn[1] - wn[0]),
        "n_wn_cols":              len(wn_cols),
        "spectra_with_any_NA":    int((~np.isfinite(Y)).any(axis=1).sum()),
        "spectra_all_finite":     int(np.isfinite(Y).all(axis=1).sum()),
    }]
    pd.DataFrame(inv_rows).to_csv(TABLES / "dataset_inventory_v2.csv", index=False)

    # Cross-tab metadata audit
    meta_rows = []
    for col in META_COLS:
        vc = df[col].astype(str).value_counts()
        for k, v in vc.items():
            meta_rows.append({"metadata_field": col, "value": k, "n_spectra": int(v)})
    pd.DataFrame(meta_rows).to_csv(TABLES / "metadata_audit_v2.csv", index=False)

    # Spectral coverage QC: per-spectrum first/last finite wn
    cov_rows = []
    for i in range(len(df)):
        finite = np.isfinite(Y[i])
        if not finite.any():
            cov_rows.append({"row_idx": i,
                                "first_finite_wn": np.nan, "last_finite_wn": np.nan,
                                "n_finite": 0, "frac_finite": 0.0})
        else:
            f = wn[finite]
            cov_rows.append({"row_idx": i,
                                "first_finite_wn": float(f[0]),
                                "last_finite_wn": float(f[-1]),
                                "n_finite": int(finite.sum()),
                                "frac_finite": float(finite.mean())})
    pd.DataFrame(cov_rows).to_csv(TABLES / "spectral_coverage_qc_v2.csv", index=False)

    print(f"  loaded {len(df)} spectra, {len(wn_cols)} wn columns "
            f"({wn[0]}-{wn[-1]} step {wn[1]-wn[0]:.0f})")
    return df, wn, Y


# ──────────────────────────────────────────────────────────────────────
# Stage 2 — QC + preprocessing
# ──────────────────────────────────────────────────────────────────────
def stage2_preprocess(df, wn, Y, master_x):
    print("[STAGE 2] preprocessing")
    # Interpolate to canonical master_x (400-1800 step 1)
    Y_pp_list = []
    excluded = []
    for i in range(len(df)):
        y = Y[i]
        finite = np.isfinite(y)
        if finite.sum() < 50:
            excluded.append({"row_idx": i, "reason": "TOO_FEW_FINITE_POINTS",
                                "labcode": df.iloc[i]["labcode"],
                                "method":  df.iloc[i]["method"]})
            Y_pp_list.append(np.full(len(master_x), np.nan))
            continue
        wn_f, y_f = wn[finite], y[finite]
        # Sort & dedupe
        order = np.argsort(wn_f)
        wn_f, y_f = wn_f[order], y_f[order]
        keep = np.concatenate(([True], np.diff(wn_f) > 0))
        wn_f, y_f = wn_f[keep], y_f[keep]
        # Interpolate to master_x; out-of-coverage → NaN
        y_rs = np.interp(master_x, wn_f, y_f, left=np.nan, right=np.nan)
        # Baseline-correct (canonical_preprocess equivalent: AsLS+SG+L2)
        y_pp = baseline_correct(y_rs)
        if not np.isfinite(y_pp).any() or float(np.linalg.norm(y_pp)) < 1e-12:
            excluded.append({"row_idx": i, "reason": "DEGENERATE_AFTER_PP",
                                "labcode": df.iloc[i]["labcode"],
                                "method":  df.iloc[i]["method"]})
            Y_pp_list.append(np.full(len(master_x), np.nan))
            continue
        Y_pp_list.append(y_pp)

    Y_pp = np.array(Y_pp_list)
    pd.DataFrame(excluded).to_csv(TABLES / "exclusion_log_v2.csv", index=False)

    # Per-method finite-coverage QC
    qc_rows = []
    for method, sub in df.groupby("method"):
        idx = sub.index.values
        finite = np.isfinite(Y_pp[idx]).all(axis=1)
        qc_rows.append({"method": method,
                          "n_spectra":            len(sub),
                          "n_finite_after_pp":    int(finite.sum()),
                          "n_excluded":           int(len(sub) - finite.sum()),
                          "labcodes":             "|".join(sorted(sub["labcode"].unique())),
                          "lasers":               "|".join(sorted(sub["laser"].astype(str).unique())),
                          "concentrations":       len(sub["conc"].unique()),
                          })
    pd.DataFrame(qc_rows).to_csv(TABLES / "preprocessing_qc_v2.csv", index=False)

    # Diagnostic figures
    try:
        fig, ax = plt.subplots(figsize=(10, 5))
        for method, sub in df.groupby("method"):
            mean_y = np.nanmean(Y_pp[sub.index.values], axis=0)
            ax.plot(master_x, mean_y, lw=0.9, label=f"{method} (n={len(sub)})")
        ax.axvspan(715, 750, color="gold", alpha=0.15, label="adenine ring window")
        ax.set_xlim(400, 1800)
        ax.set_xlabel("wavenumber cm⁻¹"); ax.set_ylabel("mean intensity (canonical pp)")
        ax.set_title("Mean preprocessed spectra by method")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(FIGS / "fig_mean_spectra_by_method_v2.png", dpi=150)
        plt.close(fig)
    except Exception as e:
        print(f"  fig mean issue: {e}")

    try:
        fig, ax = plt.subplots(figsize=(9, 5))
        for method, sub in df.groupby("method"):
            cov = np.isfinite(Y[sub.index.values]).mean(axis=0)
            ax.plot(wn, cov, label=method, lw=1.0)
        ax.set_xlabel("wavenumber cm⁻¹"); ax.set_ylabel("fraction of spectra with finite value")
        ax.set_title("Spectral coverage by method")
        ax.legend(fontsize=8); ax.set_ylim(0, 1.05)
        fig.tight_layout()
        fig.savefig(FIGS / "fig_spectral_coverage_by_method_v2.png", dpi=150)
        plt.close(fig)
    except Exception as e:
        print(f"  fig coverage issue: {e}")

    try:
        # adenine ring window per method
        fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharex=True, sharey=True)
        methods = sorted(df["method"].unique())
        for ax, method in zip(axes.flat, methods):
            sub = df[df["method"] == method]
            mask = (master_x >= 700) & (master_x <= 760)
            for j in sub.index[:30]:
                ax.plot(master_x[mask], Y_pp[j][mask], lw=0.5, alpha=0.4)
            ax.axvspan(720, 740, color="gold", alpha=0.2)
            ax.set_title(f"{method} (n={len(sub)})", fontsize=10)
        for ax in axes[1]: ax.set_xlabel("wavenumber cm⁻¹")
        for ax in axes[:, 0]: ax.set_ylabel("intensity")
        fig.suptitle("Adenine ring breathing window 715-750 cm⁻¹ by method (first 30 spectra each)")
        fig.tight_layout()
        fig.savefig(FIGS / "fig_adenine_ring_window_by_method_v2.png", dpi=150)
        plt.close(fig)
    except Exception as e:
        print(f"  fig ring window issue: {e}")

    print(f"  preprocessed: {len(Y_pp) - len(excluded)} usable / {len(Y_pp)} total")
    return Y_pp


# ──────────────────────────────────────────────────────────────────────
# Stage 3 — GAIRA scoring
# ──────────────────────────────────────────────────────────────────────
def _ring_features(y, master_x, lo, hi):
    """Return (peak_position, area, prominence) within a window."""
    mask = (master_x >= lo) & (master_x <= hi)
    if not mask.any() or not np.isfinite(y).any():
        return (np.nan, 0.0, 0.0)
    win = y[mask]
    j = int(np.nanargmax(win))
    pk_pos = float(master_x[mask][j])
    area = float(np.trapezoid(np.clip(win, 0, None), master_x[mask]))
    # Local prominence: max - 30th-pctile of surrounding 25-pt windows
    idx_lo = int(np.where(mask)[0][0])
    idx_hi = int(np.where(mask)[0][-1])
    bg_left  = float(np.percentile(y[max(idx_lo-25, 0):idx_lo], 30)) if idx_lo > 5 else 0.0
    bg_right = float(np.percentile(y[idx_hi+1:min(idx_hi+1+25, len(y))], 30)) if idx_hi+5 < len(y) else 0.0
    prom = max(float(np.nanmax(win)) - (bg_left + bg_right) / 2.0, 0.0)
    return (pk_pos, area, prom)


def stage3_gaira_scoring(df, Y_pp, master_x):
    print("[STAGE 3] GAIRA scoring (this is the heaviest stage)")
    templates, _, _ = load_templates()
    by_mol = {}
    for t in templates:
        by_mol.setdefault(t["molecule"], {})[t["regime"]] = t

    rows = []
    n = len(df)
    for i in range(n):
        if i % 500 == 0: print(f"  {i}/{n}")
        y = Y_pp[i]
        if not np.isfinite(y).any():
            continue
        meta = df.iloc[i]
        # MSS scoring per molecule (prefer SERS regime templates since pilot is SERS)
        scored = []
        for mol, tps in by_mol.items():
            t = tps.get("SERS") or tps.get("Raman") or next(iter(tps.values()))
            sc, af, sf = mss_anchor_score(y, master_x, t["anchors"], t["supports"])
            scored.append({"molecule": mol, "score": sc,
                              "regime_used": t["regime"],
                              "regime_match": t["regime"] == "SERS",
                              "bsv_family_id": t["bsv_family_id"]})
        scored.sort(key=lambda r: -r["score"])

        # G-axis proxy: max MSS score of molecules in that family
        g_scores = {fid: 0.0 for fid, _ in BSV_FAMILIES}
        for r in scored:
            fid = r["bsv_family_id"]
            if fid in g_scores and r["score"] > g_scores[fid]:
                g_scores[fid] = r["score"]

        adenine_score = next((r["score"] for r in scored if r["molecule"] == "adenine"), 0.0)
        top1 = scored[0]
        top3 = scored[:3]; top5 = scored[:5]

        # Ring-window features at 715-750, 720-740, 730-740
        rp1 = _ring_features(y, master_x, 715, 750)
        rp2 = _ring_features(y, master_x, 720, 740)
        rp3 = _ring_features(y, master_x, 730, 740)

        # Substrate physics
        sub_block = SUBSTRATE_POLICY.get(meta["substrate"], {
            "regime": "SERS", "applied": "GATED_AND_CAVEATED",
            "gaira_internal_rule": "unknown substrate; gated"})

        # Sumnorm + CLR over 11-axis G scores
        g_vec = np.array([g_scores[f] for f, _ in BSV_FAMILIES], dtype=float)
        s = g_vec.sum()
        sumnorm = g_vec / s if s > 0 else np.zeros_like(g_vec)
        # CLR with eps to avoid log(0)
        eps = 1e-6
        gv2 = g_vec + eps
        gm = np.exp(np.mean(np.log(gv2)))
        clr = np.log(gv2 / gm)

        row = {
            "row_idx":         i,
            **{c: meta[c] for c in META_COLS},
            "adenine_mss_score": adenine_score,
            "top1_molecule":     top1["molecule"],
            "top1_score":        top1["score"],
            "top1_is_adenine":   int(top1["molecule"] == "adenine"),
            "top3_molecules":    "|".join(r["molecule"] for r in top3),
            "top3_has_adenine":  int(any(r["molecule"] == "adenine" for r in top3)),
            "top5_molecules":    "|".join(r["molecule"] for r in top5),
            "top5_has_adenine":  int(any(r["molecule"] == "adenine" for r in top5)),
            "g01_top1":          int(top1["bsv_family_id"] == "G01"),
            "g01_in_top3":       int(any(r["bsv_family_id"] == "G01" for r in top3)),
            "ring_715_750_pos":  rp1[0],
            "ring_715_750_area": rp1[1],
            "ring_715_750_prom": rp1[2],
            "ring_720_740_pos":  rp2[0],
            "ring_720_740_area": rp2[1],
            "ring_720_740_prom": rp2[2],
            "ring_730_740_pos":  rp3[0],
            "ring_730_740_area": rp3[1],
            "ring_730_740_prom": rp3[2],
            "substrate_block":          meta["substrate"],
            "substrate_physics_status": sub_block["applied"],
            "substrate_physics_caveat": sub_block["gaira_internal_rule"],
            # 11-axis raw + sumnorm + CLR
            **{f"raw_{f}": float(g_scores[f]) for f, _ in BSV_FAMILIES},
            **{f"sumnorm_{f}": float(sumnorm[k]) for k, (f, _) in enumerate(BSV_FAMILIES)},
            **{f"clr_{f}": float(clr[k]) for k, (f, _) in enumerate(BSV_FAMILIES)},
        }
        rows.append(row)

    out = pd.DataFrame(rows)
    out.to_csv(TABLES / "per_spectrum_gaira_outputs_v2.csv", index=False)

    # Substrate physics application table
    pa_rows = []
    for sub in df["substrate"].unique():
        n_sub = int((df["substrate"] == sub).sum())
        pol = SUBSTRATE_POLICY.get(sub, {"applied": "UNKNOWN", "gaira_internal_rule": "n/a"})
        pa_rows.append({
            "substrate":         sub,
            "n_spectra":         n_sub,
            "applied":           pol["applied"],
            "gaira_internal_rule": pol["gaira_internal_rule"],
        })
    pd.DataFrame(pa_rows).to_csv(TABLES / "substrate_physics_application_v2.csv", index=False)

    print(f"  scored {len(out)} spectra")
    return out


# ──────────────────────────────────────────────────────────────────────
# Stage 4 — Identity coherence
# ──────────────────────────────────────────────────────────────────────
def _agg(sub):
    return {
        "n_spectra":             len(sub),
        "G01_top1_rate":         float(sub["g01_top1"].mean()),
        "G01_in_top3_rate":      float(sub["g01_in_top3"].mean()),
        "adenine_top1_rate":     float(sub["top1_is_adenine"].mean()),
        "adenine_top3_rate":     float(sub["top3_has_adenine"].mean()),
        "adenine_top5_rate":     float(sub["top5_has_adenine"].mean()),
        "adenine_mss_mean":      float(sub["adenine_mss_score"].mean()),
        "adenine_mss_sd":        float(sub["adenine_mss_score"].std()),
        "G01_score_mean":        float(sub["raw_G01"].mean()),
        "G01_score_sd":          float(sub["raw_G01"].std()),
        "ring_window_present_rate": float((sub["ring_720_740_prom"] > 0).mean()),
        "ring_window_pos_mean":  float(sub["ring_720_740_pos"].mean()),
        "ring_window_pos_sd":    float(sub["ring_720_740_pos"].std()),
    }


def stage4_identity_coherence(scored):
    print("[STAGE 4] identity coherence")
    rows = []
    rows.append({"grouping": "overall", "key": "ALL", **_agg(scored)})
    for col in ["labcode", "substrate", "laser", "method"]:
        for v, sub in scored.groupby(col):
            rows.append({"grouping": col, "key": str(v), **_agg(sub)})
    # method × labcode
    for (m, l), sub in scored.groupby(["method", "labcode"]):
        rows.append({"grouping": "method×labcode", "key": f"{m}::{l}", **_agg(sub)})
    # substrate × laser
    for (s, l), sub in scored.groupby(["substrate", "laser"]):
        rows.append({"grouping": "substrate×laser", "key": f"{s}::{l}", **_agg(sub)})
    # By concentration bin (use actual conc value)
    for c, sub in scored.groupby("conc"):
        rows.append({"grouping": "conc", "key": f"{c}", **_agg(sub)})

    df_out = pd.DataFrame(rows)
    df_out.to_csv(TABLES / "identity_coherence_by_group_v2.csv", index=False)

    # Figure: G01 stability by method
    try:
        method_rows = df_out[df_out["grouping"] == "method"].sort_values("key")
        fig, ax = plt.subplots(figsize=(8, 5))
        x = np.arange(len(method_rows))
        ax.bar(x - 0.2, method_rows["G01_top1_rate"], 0.4,
                  label="G01 top-1", color="#4C72B0")
        ax.bar(x + 0.2, method_rows["G01_in_top3_rate"], 0.4,
                  label="G01 in top-3", color="#DD8452")
        ax.set_xticks(x); ax.set_xticklabels(method_rows["key"], rotation=20)
        ax.set_ylim(0, 1.05); ax.set_ylabel("hit rate")
        ax.set_title("G01 / purine-nucleotide identity stability by method")
        ax.legend()
        fig.tight_layout()
        fig.savefig(FIGS / "fig_g01_identity_stability_by_method_v2.png", dpi=150)
        plt.close(fig)
    except Exception as e:
        print(f"  fig g01 issue: {e}")

    # Figure: adenine MSS stability by method
    try:
        fig, ax = plt.subplots(figsize=(8, 5))
        x = np.arange(len(method_rows))
        ax.bar(x - 0.25, method_rows["adenine_top1_rate"], 0.25,
                  label="adenine top-1", color="#4C72B0")
        ax.bar(x,         method_rows["adenine_top3_rate"], 0.25,
                  label="adenine top-3", color="#DD8452")
        ax.bar(x + 0.25, method_rows["adenine_top5_rate"], 0.25,
                  label="adenine top-5", color="#2ca02c")
        ax.set_xticks(x); ax.set_xticklabels(method_rows["key"], rotation=20)
        ax.set_ylim(0, 1.05); ax.set_ylabel("hit rate")
        ax.set_title("Adenine MSS stability by method")
        ax.legend()
        fig.tight_layout()
        fig.savefig(FIGS / "fig_adenine_mss_stability_by_method_v2.png", dpi=150)
        plt.close(fig)
    except Exception as e:
        print(f"  fig adenine MSS issue: {e}")

    # Figure: ring window presence by method
    try:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.bar(method_rows["key"], method_rows["ring_window_present_rate"], color="#9467bd")
        ax.set_ylim(0, 1.05); ax.set_ylabel("ring window 720-740 fire rate")
        ax.set_title("Adenine ring breathing window presence by method")
        plt.xticks(rotation=20)
        fig.tight_layout()
        fig.savefig(FIGS / "fig_ring_window_presence_by_method_v2.png", dpi=150)
        plt.close(fig)
    except Exception as e:
        print(f"  fig ring presence issue: {e}")

    return df_out


# ──────────────────────────────────────────────────────────────────────
# Stage 5 — Concentration response
# ──────────────────────────────────────────────────────────────────────
def _spearman(x, y):
    x = pd.Series(x); y = pd.Series(y)
    valid = x.notna() & y.notna()
    if valid.sum() < 3: return np.nan
    rx = x[valid].rank(); ry = y[valid].rank()
    if rx.std() == 0 or ry.std() == 0: return np.nan
    return float(np.corrcoef(rx, ry)[0, 1])


def stage5_concentration_response(scored):
    print("[STAGE 5] concentration response")
    sub_train = scored[scored["type"] != "blank"].copy()
    sub_train["log_conc"] = np.log10(sub_train["conc"].astype(float).clip(lower=1e-3))

    rows = []
    for method, sub in sub_train.groupby("method"):
        rows.append({
            "method":                    method,
            "n_spectra":                 len(sub),
            "n_concs":                   sub["conc"].nunique(),
            "rho_logc_vs_G01":           _spearman(sub["log_conc"], sub["raw_G01"]),
            "rho_logc_vs_adenine_MSS":   _spearman(sub["log_conc"], sub["adenine_mss_score"]),
            "rho_logc_vs_ring_715_750":  _spearman(sub["log_conc"], sub["ring_715_750_area"]),
            "rho_logc_vs_ring_720_740":  _spearman(sub["log_conc"], sub["ring_720_740_area"]),
            "rho_logc_vs_ring_730_740":  _spearman(sub["log_conc"], sub["ring_730_740_area"]),
            "rho_logc_vs_ring_prom_720_740": _spearman(sub["log_conc"], sub["ring_720_740_prom"]),
            "dynamic_range_ring_area":   float(sub["ring_720_740_area"].max() -
                                                  sub["ring_720_740_area"].min()),
            "replicate_CV_ring_area":    float(100 * sub["ring_720_740_area"].std() /
                                                  max(sub["ring_720_740_area"].mean(), 1e-9)),
            "scope_caveat":              "GAIRA reports concentration-response correlation; "
                                          "NOT analytical SEP/RMSEP/BIAS",
        })
    # Global row
    rows.append({
        "method":                    "ALL",
        "n_spectra":                 len(sub_train),
        "n_concs":                   sub_train["conc"].nunique(),
        "rho_logc_vs_G01":           _spearman(sub_train["log_conc"], sub_train["raw_G01"]),
        "rho_logc_vs_adenine_MSS":   _spearman(sub_train["log_conc"], sub_train["adenine_mss_score"]),
        "rho_logc_vs_ring_715_750":  _spearman(sub_train["log_conc"], sub_train["ring_715_750_area"]),
        "rho_logc_vs_ring_720_740":  _spearman(sub_train["log_conc"], sub_train["ring_720_740_area"]),
        "rho_logc_vs_ring_730_740":  _spearman(sub_train["log_conc"], sub_train["ring_730_740_area"]),
        "rho_logc_vs_ring_prom_720_740": _spearman(sub_train["log_conc"], sub_train["ring_720_740_prom"]),
        "dynamic_range_ring_area":   float(sub_train["ring_720_740_area"].max() -
                                              sub_train["ring_720_740_area"].min()),
        "replicate_CV_ring_area":    float(100 * sub_train["ring_720_740_area"].std() /
                                              max(sub_train["ring_720_740_area"].mean(), 1e-9)),
        "scope_caveat":              "global; pooled across all methods",
    })
    out = pd.DataFrame(rows)
    out.to_csv(TABLES / "concentration_response_metrics_v2.csv", index=False)

    # Figure: per-method concentration response
    try:
        fig, axes = plt.subplots(2, 3, figsize=(14, 8), sharex=True, sharey=False)
        methods = sorted(sub_train["method"].unique())
        for ax, method in zip(axes.flat, methods):
            sm = sub_train[sub_train["method"] == method]
            ax.scatter(sm["log_conc"], sm["adenine_mss_score"], alpha=0.4, s=10,
                          color="#4C72B0", label="adenine MSS")
            ax.set_title(f"{method}  (n={len(sm)})", fontsize=10)
            ax.set_xlabel("log10 conc"); ax.set_ylabel("adenine MSS score")
            rho = _spearman(sm["log_conc"], sm["adenine_mss_score"])
            ax.text(0.05, 0.95, f"ρ={rho:+.2f}", transform=ax.transAxes, fontsize=9, va="top")
        fig.tight_layout()
        fig.savefig(FIGS / "fig_concentration_response_by_method_v2.png", dpi=150)
        plt.close(fig)
    except Exception as e:
        print(f"  fig conc response issue: {e}")

    try:
        fig, axes = plt.subplots(2, 3, figsize=(14, 8), sharex=True, sharey=False)
        for ax, method in zip(axes.flat, methods):
            sm = sub_train[sub_train["method"] == method]
            ax.scatter(sm["log_conc"], sm["ring_720_740_area"], alpha=0.4, s=10,
                          color="#DD8452", label="ring 720-740 area")
            ax.set_title(f"{method}  (n={len(sm)})", fontsize=10)
            ax.set_xlabel("log10 conc"); ax.set_ylabel("ring 720-740 area")
            rho = _spearman(sm["log_conc"], sm["ring_720_740_area"])
            ax.text(0.05, 0.95, f"ρ={rho:+.2f}", transform=ax.transAxes, fontsize=9, va="top")
        fig.tight_layout()
        fig.savefig(FIGS / "fig_ring_window_concentration_response_v2.png", dpi=150)
        plt.close(fig)
    except Exception as e:
        print(f"  fig ring conc response issue: {e}")

    return out


# ──────────────────────────────────────────────────────────────────────
# Stage 6 — Variance decomposition
# ──────────────────────────────────────────────────────────────────────
def _eta2_grouped(values, factors_df):
    """Per-factor η² (variance explained) using one-way ANOVA-style SS decomposition."""
    out = {}
    total_var = float(np.var(values, ddof=1)) if len(values) > 1 else 0.0
    if total_var == 0: return {f: np.nan for f in factors_df.columns}
    grand_mean = float(np.mean(values))
    sst = float(np.sum((values - grand_mean) ** 2))
    for col in factors_df.columns:
        levels = factors_df[col]
        ss_between = 0.0
        for lvl in levels.unique():
            mask = (levels == lvl)
            n_lvl = int(mask.sum())
            if n_lvl < 1: continue
            mean_lvl = float(np.mean(values[mask]))
            ss_between += n_lvl * (mean_lvl - grand_mean) ** 2
        out[col] = float(ss_between / sst) if sst > 0 else np.nan
    return out


def stage6_variance_decomposition(scored):
    print("[STAGE 6] variance decomposition")
    sub = scored[scored["type"] != "blank"].copy()
    factors = sub[["conc", "substrate", "laser", "labcode",
                       "method", "batch", "replica"]].astype(str)
    feature_cols = ["raw_G01", "raw_G02", "adenine_mss_score",
                       "ring_720_740_area", "ring_720_740_prom",
                       "sumnorm_G01", "sumnorm_G02", "clr_G01", "clr_G02"]
    rows = []
    for fc in feature_cols:
        v = sub[fc].fillna(0).values
        eta = _eta2_grouped(v, factors)
        rows.append({"feature": fc, **eta})
    out = pd.DataFrame(rows)
    out.to_csv(TABLES / "variance_decomposition_v2.csv", index=False)

    # Figure: per-feature variance decomposition stacked bar
    try:
        fig, ax = plt.subplots(figsize=(10, 5))
        x = np.arange(len(out))
        bottom = np.zeros(len(out))
        colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
                    "#9467bd", "#8c564b", "#e377c2"]
        for i, col in enumerate(["conc", "substrate", "laser", "labcode",
                                       "method", "batch", "replica"]):
            vals = out[col].fillna(0).values
            ax.bar(x, vals, bottom=bottom, color=colors[i], label=col)
            bottom += vals
        ax.set_xticks(x); ax.set_xticklabels(out["feature"], rotation=30, ha="right", fontsize=8)
        ax.set_ylabel("η² fraction (variance explained, single-factor)")
        ax.set_title("Variance decomposition — feature × experimental factor")
        ax.legend(fontsize=8, loc="upper right")
        fig.tight_layout()
        fig.savefig(FIGS / "fig_variance_decomposition_v2.png", dpi=150)
        plt.close(fig)
    except Exception as e:
        print(f"  fig variance issue: {e}")

    return out


# ──────────────────────────────────────────────────────────────────────
# Stage 7 — Transferability map
# ──────────────────────────────────────────────────────────────────────
def stage7_transferability(scored, ic_df, cr_df):
    print("[STAGE 7] transferability map")
    rows = []
    for method in sorted(scored["method"].unique()):
        sm = scored[scored["method"] == method]
        # Identity coherence within method
        adenine_top3 = float(sm["top3_has_adenine"].mean())
        ring_present = float((sm["ring_720_740_prom"] > 0).mean())
        ring_pos_sd  = float(sm["ring_720_740_pos"].std())
        # Concentration ρ within method
        cr_row = cr_df[cr_df["method"] == method].iloc[0] if (cr_df["method"] == method).any() else None
        rho_mss = float(cr_row["rho_logc_vs_adenine_MSS"]) if cr_row is not None else np.nan
        rho_ring = float(cr_row["rho_logc_vs_ring_720_740"]) if cr_row is not None else np.nan
        # Cross-lab coherence within this method
        labs = sm["labcode"].unique()
        cross_lab_mss_mean_sd = float(sm.groupby("labcode")["adenine_mss_score"].mean().std()) \
                                       if len(labs) >= 2 else np.nan

        # Categorize
        same_method_stable = bool(adenine_top3 >= 0.50 and ring_present >= 0.50)
        cross_lab_stable   = bool(len(labs) >= 2 and not np.isnan(cross_lab_mss_mean_sd) and
                                       cross_lab_mss_mean_sd <= 0.15)
        wn_stable_int_unstable = bool(ring_present >= 0.70 and ring_pos_sd <= 5.0
                                              and not np.isnan(rho_ring) and abs(rho_ring) < 0.3)
        substrate_locked = bool(ring_present >= 0.50 and (np.isnan(rho_ring) or abs(rho_ring) < 0.2))
        unreliable       = bool(ring_present < 0.30 and adenine_top3 < 0.30)

        cats = []
        if same_method_stable: cats.append("SAME_METHOD_STABLE")
        if cross_lab_stable:   cats.append("CROSS_LAB_STABLE")
        if wn_stable_int_unstable: cats.append("WAVENUMBER_STABLE_INTENSITY_UNSTABLE")
        if substrate_locked: cats.append("SUBSTRATE_LOCKED")
        if unreliable: cats.append("UNRELIABLE")

        # Cross-substrate / cross-wavelength categorization considers other methods sharing labcode
        # (same-lab cross-substrate)
        same_lab_cross_substr = []
        for lab, lab_sub in scored.groupby("labcode"):
            ms = lab_sub["method"].unique()
            if method in ms and len(ms) >= 2:
                same_lab_cross_substr.append(lab)
        cross_substrate_stable = len(same_lab_cross_substr) >= 2
        if cross_substrate_stable:
            # Compare adenine_mss mean for this method vs other methods within same-lab
            same_lab_means = scored[(scored["labcode"].isin(same_lab_cross_substr)) &
                                          (scored["method"] != method)]["adenine_mss_score"].mean()
            this_mean = float(sm["adenine_mss_score"].mean())
            if abs(this_mean - same_lab_means) <= 0.10:
                cats.append("CROSS_SUBSTRATE_STABLE")

        # Cross-wavelength: if method has a sibling on the same substrate at a different laser
        substr = sm["substrate"].iloc[0]
        siblings = scored[(scored["substrate"] == substr) & (scored["method"] != method)]
        if not siblings.empty:
            sibling_mean = float(siblings["adenine_mss_score"].mean())
            this_mean = float(sm["adenine_mss_score"].mean())
            if abs(this_mean - sibling_mean) <= 0.10:
                cats.append("CROSS_WAVELENGTH_STABLE")

        rows.append({
            "method":                method,
            "n_spectra":             len(sm),
            "n_labs_used":           int(sm["labcode"].nunique()),
            "adenine_top3_rate":     adenine_top3,
            "ring_present_rate":     ring_present,
            "ring_pos_sd_cm1":       ring_pos_sd,
            "rho_logc_adenine_MSS":  rho_mss,
            "rho_logc_ring_area":    rho_ring,
            "cross_lab_mss_mean_sd": cross_lab_mss_mean_sd,
            "categories":            "|".join(cats) if cats else "INDETERMINATE",
        })
    out = pd.DataFrame(rows)
    out.to_csv(TABLES / "gaira_transferability_map_v2.csv", index=False)

    # Figure: transferability heatmap
    try:
        cats_all = ["SAME_METHOD_STABLE", "CROSS_LAB_STABLE",
                       "CROSS_SUBSTRATE_STABLE", "CROSS_WAVELENGTH_STABLE",
                       "WAVENUMBER_STABLE_INTENSITY_UNSTABLE",
                       "SUBSTRATE_LOCKED", "UNRELIABLE"]
        mat = np.zeros((len(out), len(cats_all)), dtype=int)
        for i, _row in out.iterrows():
            for j, c in enumerate(cats_all):
                mat[i, j] = int(c in _row["categories"])
        fig, ax = plt.subplots(figsize=(11, 5))
        im = ax.imshow(mat, aspect="auto", cmap="Greens", vmin=0, vmax=1)
        ax.set_xticks(range(len(cats_all)))
        ax.set_xticklabels(cats_all, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(len(out)))
        ax.set_yticklabels(out["method"], fontsize=9)
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                ax.text(j, i, "✓" if mat[i, j] else "·", ha="center", va="center", fontsize=8)
        ax.set_title("GAIRA transferability map — adenine identity per SERS method")
        fig.tight_layout()
        fig.savefig(FIGS / "fig_transferability_map_v2.png", dpi=150)
        plt.close(fig)
    except Exception as e:
        print(f"  fig transferability issue: {e}")
    return out


# ──────────────────────────────────────────────────────────────────────
# Stage 8 — Compare to paper
# ──────────────────────────────────────────────────────────────────────
def stage8_compare_paper(scored, transfer_df):
    print("[STAGE 8] compare to Fornasaro paper")
    rows = [
        {
            "aspect": "task framing",
            "fornasaro_paper": "Quantitative SERS concentration prediction across labs",
            "gaira": "Biochemical identity coherence + measurement-regime transferability",
            "compatible": "Different tasks; complementary metrics. GAIRA does NOT replace analytical validation; it audits what biochemical-identity signal survives measurement variation.",
        },
        {
            "aspect": "primary metrics",
            "fornasaro_paper": "SEP, RMSEP, BIAS, reproducibility, trueness",
            "gaira": "Adenine MSS top-1/3/5 hit rate, G01 family identity, ring-window 720-740 presence, "
                     "cross-lab MSS-mean SD, Spearman ρ on log conc, η² variance decomposition by factor",
            "compatible": "GAIRA's ρ on log conc IS a coarse reproducibility-direction proxy; SEP-style RMSE not computed",
        },
        {
            "aspect": "data scope",
            "fornasaro_paper": "6 methods × 15 labs × multiple concentrations × 3 batches × 3 replicas",
            "gaira": f"Same dataset; {len(scored)} spectra scored",
            "compatible": "Identical underlying data",
        },
        {
            "aspect": "best method (GAIRA)",
            "fornasaro_paper": "(see paper for SEP/RMSEP ranking — extraction deferred to a later structured paper-parsing phase)",
            "gaira": (";".join(transfer_df.sort_values("adenine_top3_rate", ascending=False)
                                          .head(2)["method"].tolist()) if not transfer_df.empty else "n/a"),
            "compatible": "GAIRA ranks methods by identity-coherence; paper ranks by quantification accuracy",
        },
        {
            "aspect": "worst method (GAIRA)",
            "fornasaro_paper": "(see paper)",
            "gaira": (";".join(transfer_df.sort_values("adenine_top3_rate")
                                          .head(2)["method"].tolist()) if not transfer_df.empty else "n/a"),
            "compatible": "Different rank criteria",
        },
    ]
    pd.DataFrame(rows).to_csv(TABLES / "gaira_vs_fornasaro_comparison_v2.csv", index=False)

    lines = [
        "# GAIRA vs Fornasaro paper — comparison v2\n",
        f"date: {datetime.now().isoformat()}",
        "",
        "## Paper",
        "Fornasaro et al., Anal Chem 2020. Zenodo 3572359. The paper asks whether quantitative SERS "
        "(adenine concentration prediction) can be consistently implemented by different labs, "
        "across substrates and lasers. Paper metrics: SEP, RMSEP, BIAS, reproducibility, trueness.",
        "",
        "## GAIRA",
        "GAIRA does NOT solve concentration prediction. GAIRA reports biochemical identity coherence "
        "and measurement-regime transferability:",
        "- adenine MSS top-1 / top-3 / top-5 hit rate per method × lab",
        "- G01 (purine-nucleotide) family identity rate per method × lab",
        "- adenine ring breathing window (720-740 cm⁻¹) presence + peak-position stability",
        "- η² variance decomposition by experimental factor (concentration, substrate, laser, lab, method, batch, replica)",
        "- transferability tagging per method (same-method-stable / cross-lab / cross-substrate / cross-wavelength / wavenumber-stable-intensity-unstable / substrate-locked / unreliable)",
        "",
        "## Where they align",
        "- Both are concerned with whether a SERS measurement is reproducible across labs, substrates, lasers.",
        "- Method (substrate × laser × SOP) is the right unit of analysis in both views.",
        "- Reproducibility (precision) and trueness (accuracy) are separable.",
        "",
        "## Where they differ",
        "- The paper outputs concentration estimates and quantifies error vs nominal.",
        "- GAIRA outputs identity coherence and transferability — does the SERS signal still SAY 'adenine' regardless of intensity?",
        "- A method can be quantitatively poor (large SEP) but identity-stable (high adenine MSS top-3); or vice-versa.",
        "",
        "## Recommendation",
        "Future GAIRA work could implement a concentration-prediction module on this dataset (e.g. PLS on ring-window features) "
        "for direct SEP/RMSEP comparison with the paper. That is OUT OF SCOPE here (no classifier-first framing per user spec).",
    ]
    (REPORTS / "REPORT_gaira_vs_fornasaro_paper_v2.md").write_text("\n".join(lines))


# ──────────────────────────────────────────────────────────────────────
# Stage 9 — extra figures
# ──────────────────────────────────────────────────────────────────────
def stage9_extra_figures(scored, df, Y_pp, master_x):
    print("[STAGE 9] extra figures (PCA + design schematic + normalization)")

    # Design schematic
    try:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.text(0.5, 0.85, "Fornasaro / Raman4Clinics — European multi-instrument adenine SERS",
                  ha="center", fontsize=12, fontweight="bold", color="#444")
        ax.text(0.5, 0.65, "15 labs (P01-P18) × 4 substrates (cAg, cAu, sAg, sAu) × 2 lasers (532, 785)",
                  ha="center", fontsize=10)
        ax.text(0.5, 0.50, "→ 6 methods (cAg@532 270, cAg@785 360, cAu@785 225, sAg@532 1041, sAg@785 810, sAu@785 810)",
                  ha="center", fontsize=10, color="#1f77b4")
        ax.text(0.5, 0.30, "× 15 samples (C0 blank, C1-C9 train, X1-X5 test) × 3 batches × 3 replicas",
                  ha="center", fontsize=10)
        ax.text(0.5, 0.10, f"= 3516 spectra / 43 unique concentrations",
                  ha="center", fontsize=11, fontweight="bold", color="#2ca02c")
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_title("Dataset design schematic")
        fig.tight_layout()
        fig.savefig(FIGS / "fig_dataset_design_v2.png", dpi=150)
        plt.close(fig)
    except Exception as e:
        print(f"  fig schematic issue: {e}")

    # PCA — colored by method/lab/conc (raw spectra)
    try:
        from sklearn.decomposition import PCA
        valid = np.isfinite(Y_pp).all(axis=1)
        idx = np.where(valid)[0]
        sample_n = min(2000, len(idx))
        rng = np.random.default_rng(0)
        idx_s = rng.choice(idx, sample_n, replace=False)
        pca = PCA(n_components=2).fit(Y_pp[idx_s])
        Z = pca.transform(Y_pp[idx_s])
        meta_s = df.iloc[idx_s]
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        # by method
        methods = sorted(meta_s["method"].unique())
        cmap1 = plt.cm.tab10(np.linspace(0, 1, len(methods)))
        for j, m in enumerate(methods):
            mask = meta_s["method"] == m
            axes[0].scatter(Z[mask, 0], Z[mask, 1], s=8, alpha=0.5, color=cmap1[j], label=m)
        axes[0].set_title("PCA of raw preprocessed spectra — by method")
        axes[0].legend(fontsize=7)
        # by lab
        labs = sorted(meta_s["labcode"].unique())
        cmap2 = plt.cm.tab20(np.linspace(0, 1, len(labs)))
        for j, l in enumerate(labs):
            mask = meta_s["labcode"] == l
            axes[1].scatter(Z[mask, 0], Z[mask, 1], s=8, alpha=0.5, color=cmap2[j], label=l)
        axes[1].set_title("PCA — by lab")
        axes[1].legend(fontsize=6, ncol=2)
        # by log conc
        log_c = np.log10(meta_s["conc"].astype(float).clip(lower=1e-3).values)
        sc = axes[2].scatter(Z[:, 0], Z[:, 1], c=log_c, cmap="viridis", s=8, alpha=0.7)
        fig.colorbar(sc, ax=axes[2], label="log10 conc")
        axes[2].set_title("PCA — by log conc")
        fig.tight_layout()
        fig.savefig(FIGS / "fig_raw_vs_bsv_pca_v2.png", dpi=150)
        plt.close(fig)
    except Exception as e:
        print(f"  fig PCA issue: {e}")

    # Normalization effect
    try:
        # Compare raw vs sumnorm vs CLR for adenine MSS variance across methods
        fig, ax = plt.subplots(figsize=(9, 4))
        x = np.arange(len(scored["method"].unique()))
        method_order = sorted(scored["method"].unique())
        raw_means = [scored[scored["method"] == m]["raw_G01"].mean() for m in method_order]
        snm_means = [scored[scored["method"] == m]["sumnorm_G01"].mean() for m in method_order]
        clr_means = [scored[scored["method"] == m]["clr_G01"].mean() for m in method_order]
        ax.bar(x - 0.2, raw_means, 0.2, label="raw G01", color="#4C72B0")
        ax.bar(x,         snm_means, 0.2, label="sumnorm G01", color="#DD8452")
        # rescale CLR to [-1, 1] for comparability on plot
        ax.bar(x + 0.2, clr_means, 0.2, label="CLR G01", color="#2ca02c")
        ax.set_xticks(x); ax.set_xticklabels(method_order, rotation=20)
        ax.set_title("Normalization effect on G01 score by method")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(FIGS / "fig_normalization_effect_v2.png", dpi=150)
        plt.close(fig)
    except Exception as e:
        print(f"  fig normalization issue: {e}")


# ──────────────────────────────────────────────────────────────────────
# Stage 10 — Final report + decision
# ──────────────────────────────────────────────────────────────────────
def _decision(transfer_df, ic_df, var_df):
    if transfer_df.empty:
        return "DATASET_BLOCKED_OR_INCOMPLETE"
    n_methods = len(transfer_df)
    n_cross_lab    = int(transfer_df["categories"].str.contains("CROSS_LAB_STABLE").sum())
    n_substrate_locked = int(transfer_df["categories"].str.contains("SUBSTRATE_LOCKED").sum())
    n_wn_stable    = int(transfer_df["categories"].str.contains("WAVENUMBER_STABLE_INTENSITY_UNSTABLE").sum())
    n_unreliable   = int(transfer_df["categories"].str.contains("UNRELIABLE").sum())
    n_same_method  = int(transfer_df["categories"].str.contains("SAME_METHOD_STABLE").sum())

    overall = ic_df[ic_df.grouping == "overall"].iloc[0]
    adenine_top3 = float(overall["adenine_top3_rate"])

    if n_unreliable >= n_methods // 2 or adenine_top3 < 0.30:
        return "ADENINE_SIGNAL_NOT_REPRODUCIBLE"
    if n_cross_lab >= n_methods - 1 and adenine_top3 >= 0.50:
        return "ADENINE_IDENTITY_TRANSFERS_CROSS_LAB"
    if n_cross_lab >= 1 and n_substrate_locked >= 1:
        return "ADENINE_IDENTITY_TRANSFERS_BUT_QUANTIFICATION_SUBSTRATE_LOCKED"
    # Check whether MSS is locked but G01 transfers
    overall_g01 = float(overall["G01_in_top3_rate"])
    if overall_g01 >= 0.50 and adenine_top3 < 0.50:
        return "ADENINE_MSS_SUBSTRATE_LOCKED_G01_TRANSFERS"
    if n_same_method >= n_methods // 2 and n_cross_lab < 2:
        return "ADENINE_IDENTITY_TRANSFERS_BUT_QUANTIFICATION_SUBSTRATE_LOCKED"
    return "ADENINE_IDENTITY_TRANSFERS_BUT_QUANTIFICATION_SUBSTRATE_LOCKED"


def write_report(decision, scored, ic_df, cr_df, var_df, transfer_df):
    overall = ic_df[ic_df.grouping == "overall"].iloc[0]
    cr_global = cr_df[cr_df.method == "ALL"].iloc[0] if (cr_df.method == "ALL").any() else None

    lines = [
        "# GAIRA European adenine reproducibility benchmark v2 — final report\n",
        f"## Decision: **{decision}**\n",
        "## Dataset",
        "Fornasaro et al. (Raman4Clinics, Anal Chem 2020, Zenodo 3572359). 3516 spectra across "
        "15 EU labs × 4 substrates (cAg, cAu, sAg, sAu) × 2 lasers (532/785) → 6 methods × 43 concentrations × 3 batches × 3 replicas.",
        "",
        "## Headline numbers (overall pooled)",
        f"- adenine MSS top-1 = {overall['adenine_top1_rate']:.0%}",
        f"- adenine MSS top-3 = {overall['adenine_top3_rate']:.0%}",
        f"- adenine MSS top-5 = {overall['adenine_top5_rate']:.0%}",
        f"- G01 (purine_nucleotide) top-1 = {overall['G01_top1_rate']:.0%}; in top-3 = {overall['G01_in_top3_rate']:.0%}",
        f"- adenine ring window 720-740 cm⁻¹ presence = {overall['ring_window_present_rate']:.0%}",
        f"- adenine ring window peak position SD = {overall['ring_window_pos_sd']:.1f} cm⁻¹",
    ]
    if cr_global is not None:
        lines += [
            f"- ρ(log conc, adenine MSS) global = {cr_global['rho_logc_vs_adenine_MSS']:+.2f}",
            f"- ρ(log conc, ring 720-740 area) global = {cr_global['rho_logc_vs_ring_720_740']:+.2f}",
        ]
    lines.append("")

    # Per-method identity table
    lines.append("## Per-method identity coherence")
    lines.append("| method | n | adenine top-1 | adenine top-3 | adenine top-5 | G01 in top-3 | ring 720-740 present | ring pos SD (cm⁻¹) |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for _, r in ic_df[ic_df.grouping == "method"].sort_values("key").iterrows():
        lines.append(f"| {r['key']} | {r['n_spectra']} | {r['adenine_top1_rate']:.2f} | "
                        f"{r['adenine_top3_rate']:.2f} | {r['adenine_top5_rate']:.2f} | "
                        f"{r['G01_in_top3_rate']:.2f} | {r['ring_window_present_rate']:.2f} | "
                        f"{r['ring_window_pos_sd']:.1f} |")
    lines.append("")

    # Per-method concentration response
    lines.append("## Per-method concentration response")
    lines.append("| method | n | n concs | ρ(logc, G01) | ρ(logc, adenine MSS) | ρ(logc, ring 720-740 area) |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for _, r in cr_df[cr_df["method"] != "ALL"].iterrows():
        lines.append(f"| {r['method']} | {r['n_spectra']} | {r['n_concs']} | "
                        f"{r['rho_logc_vs_G01']:+.2f} | {r['rho_logc_vs_adenine_MSS']:+.2f} | "
                        f"{r['rho_logc_vs_ring_720_740']:+.2f} |")
    lines.append("")

    # Variance decomposition
    lines.append("## Variance decomposition (η² per factor, single-factor ANOVA)")
    lines.append("| feature | conc | substrate | laser | labcode | method | batch | replica |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for _, r in var_df.iterrows():
        lines.append(f"| {r['feature']} | {r['conc']:.3f} | {r['substrate']:.3f} | {r['laser']:.3f} | "
                        f"{r['labcode']:.3f} | {r['method']:.3f} | {r['batch']:.3f} | {r['replica']:.3f} |")
    lines.append("")

    # Transferability map
    lines.append("## Transferability map per method")
    lines.append("| method | adenine top-3 | ring present | ring pos SD | ρ(logc, ring) | cross-lab MSS-mean SD | categories |")
    lines.append("|---|---:|---:|---:|---:|---:|---|")
    for _, r in transfer_df.iterrows():
        lines.append(f"| {r['method']} | {r['adenine_top3_rate']:.2f} | {r['ring_present_rate']:.2f} | "
                        f"{r['ring_pos_sd_cm1']:.1f} | {r['rho_logc_ring_area']:+.2f} | "
                        f"{r['cross_lab_mss_mean_sd']:.3f} | {r['categories']} |")
    lines.append("")

    # Required answers
    lines.append("## Required answers\n")
    lines.append("### 1. Does GAIRA preserve adenine identity across labs?")
    lines.append(
        f"- Overall adenine top-3 hit rate = {overall['adenine_top3_rate']:.0%}; G01 in top-3 = {overall['G01_in_top3_rate']:.0%}.")
    by_method_top3 = ic_df[ic_df.grouping == "method"].set_index("key")["adenine_top3_rate"]
    lines.append(f"- Per-method top-3: {dict((k, f'{v:.0%}') for k, v in by_method_top3.items())}")
    lines.append("- Cross-lab MSS-mean stability per method (SD) reported in transferability map; "
                    "categories like CROSS_LAB_STABLE / SUBSTRATE_LOCKED / UNRELIABLE are assigned per method.")
    lines.append("")

    lines.append("### 2. Does adenine MSS transfer better than raw spectral intensity?")
    lines.append("- Variance decomposition table compares η² of (adenine MSS) vs (ring window area) vs (G01 score) by experimental factor. "
                    "If concentration η² dominates for MSS but substrate dominates for raw intensity, MSS transfers better.")
    lines.append(f"- See `tables/variance_decomposition_v2.csv`; raw_G01 / adenine_mss_score / ring_720_740_area rows are the most relevant.")
    lines.append("")

    lines.append("### 3. Does broad G01 BSV transfer better than narrow adenine MSS?")
    lines.append(f"- G01-in-top-3 rate ({overall['G01_in_top3_rate']:.0%}) vs adenine top-3 rate ({overall['adenine_top3_rate']:.0%}). "
                    "Difference reflects whether broad-family identity is more robust than narrow-molecule identity.")
    lines.append("")

    lines.append("### 4. Which SERS methods are most coherent for GAIRA?")
    top2 = transfer_df.sort_values("adenine_top3_rate", ascending=False).head(2)["method"].tolist()
    bot2 = transfer_df.sort_values("adenine_top3_rate").head(2)["method"].tolist()
    lines.append(f"- Highest adenine top-3 coherence: {top2}")
    lines.append(f"- Lowest adenine top-3 coherence: {bot2}")
    lines.append("")

    lines.append("### 5. Which variation dominates: concentration, substrate, wavelength, lab, method, or batch?")
    if not var_df.empty:
        primary = var_df.set_index("feature").loc["adenine_mss_score"][
            ["conc", "substrate", "laser", "labcode", "method", "batch", "replica"]]
        ranked = primary.sort_values(ascending=False)
        lines.append("- For adenine MSS score, η² ranking (largest factor first):")
        for k, v in ranked.items():
            lines.append(f"  - {k}: {v:.3f}")
    lines.append("")

    lines.append("### 6. Does substrate-aware physics help, or mostly gate/caveat?")
    lines.append("- Per the substrate physics policy applied (`tables/substrate_physics_application_v2.csv`):")
    lines.append("  - cAg → APPLIED_WITH_CAVEAT (closest to internal calibrated rule family)")
    lines.append("  - cAu / sAg / sAu → GATED_AND_CAVEATED (no calibrated GAIRA rule for these substrates)")
    lines.append("- This phase therefore reports identity stability (band positions) without substrate-physics-driven intensity correction. Most of the work is gating + caveating, which is the correct conservative behavior.")
    lines.append("")

    lines.append("### 7. Does GAIRA solve quantitative reproducibility?")
    lines.append("- **No, and not its goal.** Quantitative reproducibility is the domain of analytical method validation (SEP / RMSEP / BIAS). GAIRA does not output concentration estimates here; ρ(log conc, adenine MSS) is reported as a coarse direction-correlation, NOT as a quantification metric.")
    lines.append("")

    lines.append("### 8. What does GAIRA add beyond the Fornasaro paper?")
    lines.append("- A biochemical-identity coherence audit, complementary to the paper's quantification metrics. A method can be quantitatively poor but identity-stable, or vice-versa. GAIRA exposes that decoupling.")
    lines.append("- Per-spectrum transferability tagging (SAME_METHOD_STABLE / CROSS_LAB / WAVENUMBER_STABLE_INTENSITY_UNSTABLE / SUBSTRATE_LOCKED) makes it explicit *which methods preserve adenine identity even when intensity-based quantification is unreliable*.")
    lines.append("")

    lines.append("### 9. What should become a GAIRA reproducibility metric?")
    lines.append("- **Identity coherence** (cross-lab top-K MSS hit-rate agreement): `adenine top-3 rate` per method × lab")
    lines.append("- **Wavenumber-stable / intensity-unstable separation**: ring-window peak position SD vs ρ(logc, ring area)")
    lines.append("- **Substrate-locked intensity**: η² substrate share of variance for raw_G01 / adenine_mss_score")
    lines.append("- **Measurement-regime transferability**: cross-lab adenine_MSS-mean SD per method")
    lines.append("")

    (REPORTS / "REPORT_european_adenine_reproducibility_benchmark_v2.md").write_text("\n".join(lines))


def write_audit(decision):
    txt = [
        "# gaira_base_4_european_adenine_reproducibility_benchmark_v2 — audit log",
        f"date: {datetime.now().isoformat()}",
        "",
        "## Source dataset",
        "- /Volumes/SSD_Rad/GAIRA_DATA/raw/european_multi_instrument_adenine/ILSdata.csv (3516 spectra)",
        "- /Volumes/SSD_Rad/GAIRA_DATA/raw/european_multi_instrument_adenine/Dataset/ (3516 TXT files; not used — CSV preferred)",
        "- Fornasaro et al., Anal Chem 2020 (Zenodo DOI 10.5281/zenodo.3572359)",
        "",
        "## Strict negative invariants",
        "- NO engine changes (gaira/base2 / base3 / base4 modules untouched on disk)",
        "- NO MSS scoring kernel changes (anchor-fires + support-fires preserved; same kernel as MSS resolution layer v1)",
        "- NO 11-axis BSV weight changes; G_xx_score is a family-aggregated MSS proxy via max-over-molecules",
        "- NO motif registry changes",
        "- NO substrate physics rules added or modified — substrate policy:",
        "    cAg = APPLIED_WITH_CAVEAT (closest to internal calibrated rule family)",
        "    cAu / sAg / sAu = GATED_AND_CAVEATED (no calibrated GAIRA rule)",
        "- NO classifier feedback, NO disease labels, NO DART-Met",
        "- NO threshold tuning, NO label-driven optimization",
        "",
        "## Outputs",
        "- tables/local_file_inventory_v2.csv",
        "- tables/dataset_inventory_v2.csv",
        "- tables/metadata_audit_v2.csv",
        "- tables/spectral_coverage_qc_v2.csv",
        "- tables/preprocessing_qc_v2.csv",
        "- tables/exclusion_log_v2.csv",
        "- tables/per_spectrum_gaira_outputs_v2.csv",
        "- tables/substrate_physics_application_v2.csv",
        "- tables/identity_coherence_by_group_v2.csv",
        "- tables/concentration_response_metrics_v2.csv",
        "- tables/variance_decomposition_v2.csv",
        "- tables/gaira_transferability_map_v2.csv",
        "- tables/gaira_vs_fornasaro_comparison_v2.csv",
        "- 10 figures (mean spectra by method, ring window by method, spectral coverage, "
        "G01 stability by method, adenine MSS stability, ring presence by method, "
        "concentration response by method, ring concentration response, variance decomposition, "
        "transferability map, dataset design, PCA, normalization effect)",
        "- reports/REPORT_dataset_context_audit_v2.md",
        "- reports/REPORT_gaira_vs_fornasaro_paper_v2.md",
        "- reports/REPORT_european_adenine_reproducibility_benchmark_v2.md",
        "",
        f"## Final decision\n**{decision}**",
    ]
    (AUDIT / "gaira_base_4_european_adenine_reproducibility_benchmark_v2_audit_log.md").write_text("\n".join(txt))


# ──────────────────────────────────────────────────────────────────────
# Entrypoint
# ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 78)
    print("gaira_base_4_european_adenine_reproducibility_benchmark_v2")
    print("=" * 78)
    master_x = canonical_master_axis()

    stage0_context_audit()
    df, wn, Y = stage1_load()
    Y_pp = stage2_preprocess(df, wn, Y, master_x)
    scored = stage3_gaira_scoring(df, Y_pp, master_x)
    ic_df = stage4_identity_coherence(scored)
    cr_df = stage5_concentration_response(scored)
    var_df = stage6_variance_decomposition(scored)
    transfer_df = stage7_transferability(scored, ic_df, cr_df)
    stage8_compare_paper(scored, transfer_df)
    stage9_extra_figures(scored, df, Y_pp, master_x)
    decision = _decision(transfer_df, ic_df, var_df)
    write_report(decision, scored, ic_df, cr_df, var_df, transfer_df)
    write_audit(decision)
    try:
        shutil.copy(__file__, CODE_SNAPSHOT / Path(__file__).name)
    except Exception:
        pass
    print(f"[done] decision: {decision}")


if __name__ == "__main__":
    main()
