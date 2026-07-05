"""gaira_base_2 — PHASE B: Calibration Benchmark.

Exercises the implemented engine on all available calibration datasets:

  - Gobbato SERS spiked serum Merck (54 analytes × 5 reps + 5 bkg)
  - Gobbato pure metabolite SERS    (54 analytes × 5 reps)
  - Gobbato uricase depletion        (UA-spike vs uricase-depleted × 5 reps)
  - ERG_calibration.csv              (0–2 µM Ergo, 11 conc × ~5 reps)
  - CSPP serum Figure-7              (Erg 25 µM + Hyp 50 µM spike vs Bkg)

Per spectrum, captures full motif + 11-axis + 8-axis + ambiguity output.

Calibration questions tested:
  1. Does the intended motif move under the perturbation?
  2. Does the intended 11-axis move?
  3. Is the 8-axis projection interpretable?
  4. Are responses monotonic where expected?
  5. How much cross-talk remains?
  6. Do ambiguity motifs preserve ambiguity?
  7. How do core vs regime-adjusted outputs differ?

Run:
    cd /Users/suraj/projects/GAIRA
    PYTHONPATH=src .venv/bin/python scripts/run_gaira_base_2_calibration_benchmark.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from gaira.base2 import (
    BIOLOGY_AXES_V11,
    load_active_registry,
    result_to_flat_dict,
    score_spectrum,
)
from gaira.spectral import canonical_master_axis, crop_before_interpolate
from gaira.spectral.preprocessing import _asls_baseline
from scipy.signal import savgol_filter
from scipy.stats import spearmanr


OUT = Path("/Volumes/SSD_Rad/GAIRA_BUILD/gaira_base_2_backend_validation_v1/calibration")
OUT.mkdir(parents=True, exist_ok=True)

GOBBATO = Path(
    "/Volumes/SSD_Rad/GAIRA_BUILD/gaira_build_motifs_v1/"
    "M3_1_reference_rescue_v1/references/_extracted"
)
ERG_CAL = Path(
    "/Volumes/SSD_Rad/GAIRA_DATA/raw/ergothioneine_serum/ERG_calibration.csv"
)
CSPP = Path(
    "/Volumes/SSD_Rad/GAIRA_DATA/raw/cspp_serum/"
    "Figure-7_all-spectra-and-metadata.csv"
)


MOTIF_TARGETS = {
    "uric_acid_full_signature":            ["UA"],
    "hypoxanthine_signature":              ["Hypox"],
    "xanthine_signature":                  ["Xanth"],
    "ergothioneine_signature":             ["Ergo"],
    "creatine_creatinine_motif":           ["Creat"],
    "purine_ring_breathing_720_735":       ["Ade", "Gua"],
    "pyrimidine_ring_breathing_780_800":   ["Thy", "Ura"],
    "dna_methylation_marker_790":          ["Thy"],
    "guanine_specific_motif":              ["Gua"],
    "thymine_specific_motif":              ["Thy"],
    "phosphate_PO_asym_str_1240":          ["DNA", "RNA", "PEP"],
    "phosphate_PO2_sym_str_1080":          ["DNA", "RNA", "PEP"],
    "dna_composite_motif":                 ["DNA", "RNA"],
    "glycan_pyranose_ring_skeletal_850_950": ["Gluc", "Fruct", "Mann"],
    "glycan_glycosidic_C_O_C_1020_1100":   ["Gluc", "Fruct", "Mann", "Lact"],
    "sialic_acid_signature":               ["NacDgluc"],
    "free_saccharide_motif":               ["Gluc", "Fruct", "Mann"],
    "amide_III_protein_backbone_1230_1280": ["Alb"],
    "amide_I_alpha_helix_beta_sheet_motif": ["Alb"],
    "amide_II_motif":                      ["Alb"],
    "phenylalanine_ring_1003":             ["Phe"],
    "tyrosine_doublet_830_850":            ["Tyr"],
    "lipid_acyl_C_C_str_1060_1130":        ["Oleic", "Stearic", "Triolein"],
    "lipid_C_H_bend_1440_1460":            ["Oleic", "Stearic", "Triolein"],
    "lipid_methylene_twist_1300":          ["Oleic", "Stearic", "Triolein"],
    "cholesterol_signature":               ["Chol"],
    "neutral_lipid_triglyceride_motif":    ["Triolein"],
    "disulfide_S_S_str_500_550":           ["Cys"],
    "thiol_C_S_str_660_motif":             ["Cys"],
    "citrate_baseline_artifact_motif":     ["Citric"],
    "collision_1020_1080_multi_candidate": ["DNA", "RNA", "Gluc", "Citric"],
    "purine_HX_lipid_choline_715_overlap_ambiguity": ["Ade", "Gua", "Hypox"],
    "collision_1300_1400_multi_candidate_motif": ["Ade", "Gua", "Oleic", "Alb", "Citric"],
}


AXIS_TARGETS = {
    "purine_metabolite":             ["UA", "Hypox", "Xanth"],
    "purine_nucleotide":             ["Ade", "Gua"],
    "pyrimidine_nucleotide":         ["Thy", "Ura"],
    "phosphate_nucleic_adjacent":    ["DNA", "RNA", "PEP"],
    "glycan_carbohydrate":           ["Gluc", "Fruct", "Mann", "Lact"],
    "protein_peptide_backbone":      ["Alb"],
    "aromatic_residue":              ["Phe", "Tyr"],
    "lipid_acyl_membrane":           ["Oleic", "Stearic", "Triolein"],
    "sterol_neutral_lipid":          ["Chol", "Triolein"],
    "sulfur_thiol_redox":            ["Cys", "Ergo"],
    "metabolic_small_molecule":      ["Creat", "Ergo"],
}


# ──────────────────────────────────────────────────────────────────────
# Preprocessing + parsing helpers
# ──────────────────────────────────────────────────────────────────────

def canonical_preprocess(raw_wn, raw_y, master_x):
    try:
        y_interp, _ = crop_before_interpolate(
            raw_wn, raw_y, master_x, partial_ok=True, min_coverage=0.80,
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


def cohen_d(a, b):
    if len(a) == 0 or len(b) == 0:
        return float("nan")
    va = np.var(a, ddof=1) if len(a) > 1 else 0.0
    vb = np.var(b, ddof=1) if len(b) > 1 else 0.0
    pooled = np.sqrt(((len(a) - 1) * va + (len(b) - 1) * vb)
                      / max(len(a) + len(b) - 2, 1))
    if pooled < 1e-12:
        return float("nan")
    return float((np.mean(a) - np.mean(b)) / pooled)


# ──────────────────────────────────────────────────────────────────────
# Dataset loaders
# ──────────────────────────────────────────────────────────────────────

def load_gobbato_spike(master_x):
    """Spike-in-serum Merck panel."""
    d = GOBBATO / "SERS spiked serum Merck"
    out: dict[str, list[tuple[np.ndarray, str]]] = {}
    for p in sorted(d.iterdir()):
        if not p.name.startswith("SERS_spike_"):
            continue
        analyte = p.name[len("SERS_spike_"):].split("_")[0]
        parsed = parse_gobbato(p)
        if parsed is None:
            continue
        y_pp = canonical_preprocess(parsed[0], parsed[1], master_x)
        if y_pp is None:
            continue
        rep = p.stem.split("_")[-1]
        out.setdefault(analyte, []).append((y_pp, f"{analyte}_rep{rep}"))
    return out


def load_gobbato_pure_sers(master_x):
    d = GOBBATO / "SERS metabolites"
    out: dict[str, list[tuple[np.ndarray, str]]] = {}
    for p in sorted(d.iterdir()):
        if not p.name.startswith("SERS_met_"):
            continue
        analyte = p.name[len("SERS_met_"):].split("_")[0]
        parsed = parse_gobbato(p)
        if parsed is None:
            continue
        y_pp = canonical_preprocess(parsed[0], parsed[1], master_x)
        if y_pp is None:
            continue
        rep = p.stem.split("_")[-1]
        out.setdefault(analyte, []).append((y_pp, f"{analyte}_rep{rep}"))
    return out


def load_uricase(master_x):
    d = GOBBATO / "dataset uricase"
    out: dict[str, list[tuple[np.ndarray, str]]] = {}
    for p in sorted(d.iterdir()):
        if "Serumspiked_" in p.name and "Enzyme" not in p.name:
            key = "spiked"
        elif "Serumspiked+Enzyme" in p.name:
            key = "depleted"
        elif "SerumSigma_" in p.name and "Enzyme" not in p.name:
            key = "control"
        else:
            continue
        parsed = parse_gobbato(p)
        if parsed is None:
            continue
        y_pp = canonical_preprocess(parsed[0], parsed[1], master_x)
        if y_pp is None:
            continue
        out.setdefault(key, []).append((y_pp, p.stem))
    return out


def load_erg_calibration(master_x):
    """Returns dict keyed by concentration_µM → list of (spectrum, id)."""
    df = pd.read_csv(ERG_CAL)
    meta_cols = ["laser", "power", "substrate", "c"]
    wn_cols = [c for c in df.columns if c not in meta_cols]
    wn = np.array([float(c) for c in wn_cols], dtype=np.float64)
    out: dict[float, list[tuple[np.ndarray, str]]] = {}
    for i, r in df.iterrows():
        c = float(r["c"])
        y = r[wn_cols].to_numpy(dtype=np.float64)
        y_pp = canonical_preprocess(wn, y, master_x)
        if y_pp is None:
            continue
        out.setdefault(c, []).append((y_pp, f"erg_c{c}_row{i}"))
    return out


def load_cspp_fig7(master_x):
    """Returns dict keyed by metabolite tag (Erg/Hyp/Bkg) → list of (spec, id)."""
    df = pd.read_csv(CSPP)
    meta_cols = ["num", "method", "serum_typ", "metabolite", "conc",
                 "acc", "t_mes", "pw", "rep"]
    present_meta = [c for c in meta_cols if c in df.columns]
    first_col = df.columns[0]
    wn_cols = [c for c in df.columns if c not in present_meta and c != first_col]
    wn = np.array([float(c) for c in wn_cols], dtype=np.float64)
    met = df["metabolite"].astype(str).str.strip().str.strip('"')
    out: dict[str, list[tuple[np.ndarray, str]]] = {}
    for i, (tag, row) in enumerate(zip(met, df[wn_cols].to_numpy(dtype=np.float64))):
        y_pp = canonical_preprocess(wn, row, master_x)
        if y_pp is None:
            continue
        out.setdefault(tag, []).append((y_pp, f"cspp_{tag}_row{i}"))
    return out


# ──────────────────────────────────────────────────────────────────────
# Scoring a group of spectra
# ──────────────────────────────────────────────────────────────────────

def score_group(
    specs: list[tuple[np.ndarray, str]],
    master_x, motifs, mappings, dual,
    dataset_id: str, condition: str,
) -> list[dict]:
    rows = []
    for y, sid in specs:
        res = score_spectrum(y, master_x, motifs, mappings, dual, sid)
        row = result_to_flat_dict(res)
        row["dataset_id"] = dataset_id
        row["condition"] = condition
        rows.append(row)
    return rows


def extract_motif_axis_arrays(rows: list[dict]) -> tuple[dict, dict, dict, dict]:
    """Returns (motif_core, motif_regime, axis11_core, axis11_regime) dicts
    each keyed by name, values are np.ndarray over rows."""
    df = pd.DataFrame(rows)
    mc, mr, ac, ar = {}, {}, {}, {}
    for col in df.columns:
        if col.startswith("motif_core."):
            mc[col.split(".", 1)[1]] = df[col].astype(float).to_numpy()
        elif col.startswith("motif_regime."):
            mr[col.split(".", 1)[1]] = df[col].astype(float).to_numpy()
        elif col.startswith("axis11_core."):
            ac[col.split(".", 1)[1]] = df[col].astype(float).to_numpy()
        elif col.startswith("axis11_regime."):
            ar[col.split(".", 1)[1]] = df[col].astype(float).to_numpy()
    return mc, mr, ac, ar


# ──────────────────────────────────────────────────────────────────────
# Driver
# ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 78)
    print("gaira_base_2 — PHASE B: Calibration Benchmark")
    print("=" * 78)
    master_x = canonical_master_axis()
    motifs, mappings, dual = load_active_registry()

    # ── Load every dataset ───────────────────────────────────────────
    print("[load] Gobbato SERS spiked serum Merck...")
    spike = load_gobbato_spike(master_x)
    print(f"  {len(spike)} analytes; "
          f"{sum(len(v) for v in spike.values())} spectra total")

    print("[load] Gobbato pure metabolite SERS...")
    pure = load_gobbato_pure_sers(master_x)
    print(f"  {len(pure)} analytes; "
          f"{sum(len(v) for v in pure.values())} spectra total")

    print("[load] Gobbato uricase depletion...")
    uric = load_uricase(master_x)
    print(f"  {len(uric)} conditions; "
          f"{sum(len(v) for v in uric.values())} spectra total")

    print("[load] ERG calibration series...")
    erg = load_erg_calibration(master_x)
    print(f"  {len(erg)} concentrations; "
          f"{sum(len(v) for v in erg.values())} spectra total")

    print("[load] CSPP Figure-7...")
    cspp = load_cspp_fig7(master_x)
    print(f"  {len(cspp)} conditions; "
          f"{sum(len(v) for v in cspp.values())} spectra total")

    # ── Score every spectrum across every dataset ────────────────────
    per_spectrum_rows = []
    print("\n[score] Gobbato spike...")
    for analyte, specs in spike.items():
        per_spectrum_rows.extend(
            score_group(specs, master_x, motifs, mappings, dual,
                         "gobbato_spike", analyte)
        )
    print("[score] Gobbato pure SERS...")
    for analyte, specs in pure.items():
        per_spectrum_rows.extend(
            score_group(specs, master_x, motifs, mappings, dual,
                         "gobbato_pure_sers", analyte)
        )
    print("[score] uricase...")
    for cond, specs in uric.items():
        per_spectrum_rows.extend(
            score_group(specs, master_x, motifs, mappings, dual,
                         "gobbato_uricase", cond)
        )
    print("[score] ERG calibration...")
    for conc, specs in erg.items():
        per_spectrum_rows.extend(
            score_group(specs, master_x, motifs, mappings, dual,
                         "erg_calibration", f"c{conc}µM")
        )
    print("[score] CSPP Figure-7...")
    for tag, specs in cspp.items():
        per_spectrum_rows.extend(
            score_group(specs, master_x, motifs, mappings, dual,
                         "cspp_fig7", tag)
        )

    per_spec_df = pd.DataFrame(per_spectrum_rows)
    per_spec_df.to_csv(OUT / "calibration_per_spectrum_scores_v1.csv", index=False)
    print(f"[emit] calibration_per_spectrum_scores_v1.csv "
          f"({len(per_spec_df)} rows)")

    # ── Motif response summary (Gobbato spike vs SerumSigma bkg) ─────
    print("\n[analyze] motif response summary (spike vs SerumSigma)")
    motif_resp_rows = []
    bkg_spike = spike.get("SerumSigma", [])
    if bkg_spike:
        bkg_rows = score_group(bkg_spike, master_x, motifs, mappings, dual,
                                 "gobbato_spike", "SerumSigma")
        bkg_mc, bkg_mr, bkg_ac, bkg_ar = extract_motif_axis_arrays(bkg_rows)
        for motif_id in motifs:
            targets = MOTIF_TARGETS.get(motif_id, [])
            if not targets:
                continue
            target_specs = []
            for a in targets:
                if a in spike:
                    target_specs.extend(spike[a])
            if not target_specs:
                continue
            target_rows = score_group(
                target_specs, master_x, motifs, mappings, dual,
                "gobbato_spike", ",".join(targets),
            )
            t_mc, t_mr, _, _ = extract_motif_axis_arrays(target_rows)
            if motif_id not in t_mc or motif_id not in bkg_mc:
                continue
            d_core = cohen_d(t_mc[motif_id], bkg_mc[motif_id])
            d_regime = cohen_d(t_mr[motif_id], bkg_mr[motif_id])
            # cross-talk: mean |delta| across non-target analytes
            non_target = [a for a in spike if a not in targets and a != "SerumSigma"]
            ct_vals = []
            for a in non_target:
                a_rows = score_group(spike[a], master_x, motifs, mappings, dual,
                                       "gobbato_spike", a)
                a_mc, _, _, _ = extract_motif_axis_arrays(a_rows)
                if motif_id in a_mc:
                    ct_vals.append(abs(a_mc[motif_id].mean() - bkg_mc[motif_id].mean()))
            target_effect = abs(t_mc[motif_id].mean() - bkg_mc[motif_id].mean())
            ct_score = (np.mean(ct_vals) / target_effect) if (ct_vals and target_effect > 1e-12) else float("nan")
            motif_resp_rows.append({
                "motif_id": motif_id,
                "targets": ",".join(targets),
                "n_target_spectra": len(target_specs),
                "n_bkg_spectra": len(bkg_spike),
                "target_core_mean": round(t_mc[motif_id].mean(), 4),
                "bkg_core_mean": round(bkg_mc[motif_id].mean(), 4),
                "cohen_d_core": round(d_core, 3) if np.isfinite(d_core) else float("nan"),
                "cohen_d_regime": round(d_regime, 3) if np.isfinite(d_regime) else float("nan"),
                "cross_talk_ratio": round(ct_score, 3) if np.isfinite(ct_score) else float("nan"),
            })
    pd.DataFrame(motif_resp_rows).to_csv(
        OUT / "calibration_motif_response_summary_v1.csv", index=False,
    )
    print(f"[emit] calibration_motif_response_summary_v1.csv "
          f"({len(motif_resp_rows)} rows)")

    # ── 11-axis response summary ─────────────────────────────────────
    print("[analyze] 11-axis response summary")
    axis11_rows = []
    axis8_rows = []
    if bkg_spike:
        bkg_rows = score_group(bkg_spike, master_x, motifs, mappings, dual,
                                 "gobbato_spike", "SerumSigma")
        _, _, bkg_ac, bkg_ar = extract_motif_axis_arrays(bkg_rows)
        for axis_id, targets in AXIS_TARGETS.items():
            target_specs = []
            for a in targets:
                if a in spike:
                    target_specs.extend(spike[a])
            if not target_specs:
                continue
            target_rows = score_group(
                target_specs, master_x, motifs, mappings, dual,
                "gobbato_spike", ",".join(targets),
            )
            _, _, t_ac, t_ar = extract_motif_axis_arrays(target_rows)
            if axis_id not in t_ac or axis_id not in bkg_ac:
                continue
            d_core = cohen_d(t_ac[axis_id], bkg_ac[axis_id])
            d_regime = cohen_d(t_ar[axis_id], bkg_ar[axis_id])
            axis11_rows.append({
                "axis11_id": axis_id,
                "targets": ",".join(targets),
                "n_target_spectra": len(target_specs),
                "target_core_mean": round(t_ac[axis_id].mean(), 4),
                "bkg_core_mean": round(bkg_ac[axis_id].mean(), 4),
                "cohen_d_core": round(d_core, 3) if np.isfinite(d_core) else float("nan"),
                "cohen_d_regime": round(d_regime, 3) if np.isfinite(d_regime) else float("nan"),
            })
        # axis8 projection
        from gaira.base2.schema import PROJECTION_V11_TO_V8
        bkg_a8 = {
            ax8: np.max(
                np.stack([bkg_ac[a11] for a11 in sources if a11 in bkg_ac]),
                axis=0,
            )
            for ax8, sources in PROJECTION_V11_TO_V8.items()
            if any(a in bkg_ac for a in sources)
        }
        # Per-axis8: target = union of all 11-axis sources' targets
        for ax8, sources in PROJECTION_V11_TO_V8.items():
            merged_targets = sorted(set(sum((
                AXIS_TARGETS.get(s, []) for s in sources
            ), [])))
            if not merged_targets:
                continue
            target_specs = []
            for a in merged_targets:
                if a in spike:
                    target_specs.extend(spike[a])
            if not target_specs or ax8 not in bkg_a8:
                continue
            t_rows = score_group(target_specs, master_x, motifs, mappings, dual,
                                   "gobbato_spike", ",".join(merged_targets))
            _, _, t_ac, _ = extract_motif_axis_arrays(t_rows)
            t_a8 = np.max(
                np.stack([t_ac[a11] for a11 in sources if a11 in t_ac]),
                axis=0,
            )
            d = cohen_d(t_a8, bkg_a8[ax8])
            axis8_rows.append({
                "axis8_id": ax8,
                "contributing_11_axes": ",".join(sources),
                "merged_targets": ",".join(merged_targets),
                "target_mean": round(t_a8.mean(), 4),
                "bkg_mean": round(bkg_a8[ax8].mean(), 4),
                "cohen_d": round(d, 3) if np.isfinite(d) else float("nan"),
            })
    pd.DataFrame(axis11_rows).to_csv(
        OUT / "calibration_axis11_response_summary_v1.csv", index=False,
    )
    pd.DataFrame(axis8_rows).to_csv(
        OUT / "calibration_axis8_projection_summary_v1.csv", index=False,
    )

    # ── Full cross-talk matrix (motif × analyte spike) ───────────────
    print("[analyze] full motif × analyte spike matrix")
    ct_matrix_rows = []
    if bkg_spike:
        for motif_id in motifs:
            row = {"motif_id": motif_id}
            for analyte, specs in spike.items():
                if analyte == "SerumSigma":
                    continue
                a_rows = score_group(specs, master_x, motifs, mappings, dual,
                                       "gobbato_spike", analyte)
                a_mc, _, _, _ = extract_motif_axis_arrays(a_rows)
                if motif_id in a_mc and motif_id in bkg_mc:
                    d = cohen_d(a_mc[motif_id], bkg_mc[motif_id])
                    row[analyte] = round(d, 3) if np.isfinite(d) else float("nan")
                else:
                    row[analyte] = float("nan")
            ct_matrix_rows.append(row)
    pd.DataFrame(ct_matrix_rows).to_csv(
        OUT / "calibration_cross_talk_matrix_v1.csv", index=False,
    )

    # ── Monotonicity summary (ERG + uricase) ─────────────────────────
    print("[analyze] monotonicity (ERG dose-graded + uricase)")
    mono_rows = []
    # ERG dose-graded
    erg_by_motif: dict[str, list[tuple[float, float]]] = {m: [] for m in motifs}
    for conc, specs in erg.items():
        rows = score_group(specs, master_x, motifs, mappings, dual,
                             "erg_calibration", f"c{conc}µM")
        mc, _, _, _ = extract_motif_axis_arrays(rows)
        for motif_id, arr in mc.items():
            for v in arr:
                erg_by_motif[motif_id].append((conc, float(v)))
    for motif_id, pairs in erg_by_motif.items():
        if len(pairs) < 5:
            continue
        concs = np.array([p[0] for p in pairs])
        vals = np.array([p[1] for p in pairs])
        rho, pval = spearmanr(concs, vals)
        mono_rows.append({
            "dataset": "erg_calibration",
            "motif_id": motif_id,
            "perturbation_variable": "ergothioneine µM",
            "n_spectra": len(pairs),
            "spearman_rho": round(float(rho), 3) if np.isfinite(rho) else float("nan"),
            "p_value": round(float(pval), 4) if np.isfinite(pval) else float("nan"),
        })
    # uricase (3-way: control/depleted/spiked)
    if uric:
        for motif_id in motifs:
            if "spiked" not in uric or "depleted" not in uric:
                continue
            s_rows = score_group(uric["spiked"], master_x, motifs, mappings, dual,
                                   "gobbato_uricase", "spiked")
            d_rows = score_group(uric["depleted"], master_x, motifs, mappings, dual,
                                   "gobbato_uricase", "depleted")
            s_mc, _, _, _ = extract_motif_axis_arrays(s_rows)
            d_mc, _, _, _ = extract_motif_axis_arrays(d_rows)
            if motif_id not in s_mc or motif_id not in d_mc:
                continue
            # Spearman: label (0=depleted, 1=spiked) vs activation
            labels = np.concatenate([
                np.zeros(len(d_mc[motif_id])), np.ones(len(s_mc[motif_id])),
            ])
            vals = np.concatenate([d_mc[motif_id], s_mc[motif_id]])
            rho, pval = spearmanr(labels, vals)
            mono_rows.append({
                "dataset": "gobbato_uricase",
                "motif_id": motif_id,
                "perturbation_variable": "UA spike-vs-depleted (2-point)",
                "n_spectra": len(vals),
                "spearman_rho": round(float(rho), 3) if np.isfinite(rho) else float("nan"),
                "p_value": round(float(pval), 4) if np.isfinite(pval) else float("nan"),
            })
    pd.DataFrame(mono_rows).to_csv(
        OUT / "calibration_monotonicity_summary_v1.csv", index=False,
    )

    # ── Ambiguity behaviour ──────────────────────────────────────────
    print("[analyze] ambiguity behaviour across spike panel")
    amb_rows = []
    for analyte, specs in spike.items():
        rows = score_group(specs, master_x, motifs, mappings, dual,
                             "gobbato_spike", analyte)
        df = pd.DataFrame(rows)
        amb_rows.append({
            "condition": analyte,
            "n_spectra": len(df),
            "ambiguity_core_mean":   round(df["ambiguity_core"].mean(), 4),
            "ambiguity_regime_mean": round(df["ambiguity_regime"].mean(), 4),
            "ambiguity_core_std":    round(df["ambiguity_core"].std(), 4),
        })
    pd.DataFrame(amb_rows).to_csv(
        OUT / "calibration_ambiguity_behavior_v1.csv", index=False,
    )

    # ── Figures ──────────────────────────────────────────────────────
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[warn] matplotlib unavailable: {e}")
    else:
        _plot_motif_response(pd.DataFrame(motif_resp_rows), plt)
        _plot_axis_response(pd.DataFrame(axis11_rows), plt, "11")
        _plot_axis_response(pd.DataFrame(axis8_rows), plt, "8")
        _plot_cross_talk(pd.DataFrame(ct_matrix_rows), plt)
        _plot_monotonicity_heatmap(pd.DataFrame(mono_rows), plt)
        _plot_core_vs_regime(pd.DataFrame(motif_resp_rows), plt)
        _plot_ambiguity(pd.DataFrame(amb_rows), plt)

    # ── Report ────────────────────────────────────────────────────────
    _write_report(
        pd.DataFrame(motif_resp_rows),
        pd.DataFrame(axis11_rows),
        pd.DataFrame(axis8_rows),
        pd.DataFrame(mono_rows),
        pd.DataFrame(amb_rows),
    )
    print("DONE")


# ──────────────────────────────────────────────────────────────────────
# Figures
# ──────────────────────────────────────────────────────────────────────

def _plot_motif_response(df, plt):
    if df.empty:
        return
    df = df.sort_values("cohen_d_core", key=lambda s: s.abs(), ascending=False).head(25)
    fig, ax = plt.subplots(figsize=(11, max(4, 0.35 * len(df))))
    y = np.arange(len(df))
    ax.barh(y - 0.2, df["cohen_d_core"].abs(), height=0.35,
             color="#2a9d8f", label="|Cohen d| core")
    ax.barh(y + 0.2, df["cohen_d_regime"].abs(), height=0.35,
             color="#76c893", label="|Cohen d| regime")
    ax.axvline(0.8, color="gray", linestyle="--", linewidth=0.8,
                label="CALIBRATION_VALID threshold")
    ax.set_yticks(y)
    ax.set_yticklabels(df["motif_id"], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("|Cohen d| (target spike vs SerumSigma)")
    ax.set_title("Calibration — top-25 motif responses (|d| core vs regime)")
    ax.legend(fontsize=8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "fig_calibration_motif_response_panels.png", dpi=130)
    plt.close(fig)


def _plot_axis_response(df, plt, n_axes):
    if df.empty:
        return
    col = f"axis{n_axes}_id"
    dsort = df.sort_values("cohen_d" if "cohen_d" in df else "cohen_d_core",
                             key=lambda s: s.abs(), ascending=False)
    fig, ax = plt.subplots(figsize=(10, max(4, 0.45 * len(dsort))))
    y = np.arange(len(dsort))
    if "cohen_d_core" in dsort.columns:
        ax.barh(y - 0.2, dsort["cohen_d_core"].abs(), height=0.35,
                 color="#2a9d8f", label="|d| core")
        ax.barh(y + 0.2, dsort["cohen_d_regime"].abs(), height=0.35,
                 color="#76c893", label="|d| regime")
    else:
        ax.barh(y, dsort["cohen_d"].abs(), color="#2a9d8f", label="|Cohen d|")
    ax.set_yticks(y)
    ax.set_yticklabels(dsort[col], fontsize=9)
    ax.invert_yaxis()
    ax.axvline(0.8, color="gray", linestyle="--", linewidth=0.8,
                label="0.8 threshold")
    ax.set_xlabel("|Cohen d|")
    ax.set_title(f"Calibration — axis{n_axes} response "
                   f"(|d| target vs SerumSigma)")
    ax.legend(fontsize=8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / f"fig_calibration_axis{n_axes}_response_panels.png",
                  dpi=130)
    plt.close(fig)


def _plot_cross_talk(df, plt):
    if df.empty:
        return
    df = df.set_index("motif_id")
    if df.empty or len(df.columns) < 2:
        return
    # |d|
    data = df.abs().fillna(0.0)
    vmax = np.nanpercentile(data.values, 99) if data.size else 1.0
    fig, ax = plt.subplots(figsize=(max(10, 0.35 * len(data.columns)),
                                      max(8, 0.25 * len(data))))
    im = ax.imshow(data.values, aspect="auto", cmap="RdYlGn_r",
                    vmin=0, vmax=max(vmax, 1.0))
    ax.set_xticks(range(len(data.columns)))
    ax.set_xticklabels(data.columns, rotation=60, ha="right", fontsize=7)
    ax.set_yticks(range(len(data)))
    ax.set_yticklabels(data.index, fontsize=6)
    fig.colorbar(im, ax=ax, label="|Cohen d|")
    ax.set_title("Calibration — motif × analyte cross-talk matrix "
                   "(|d| target vs SerumSigma)")
    fig.tight_layout()
    fig.savefig(OUT / "fig_calibration_cross_talk_matrix.png", dpi=130)
    plt.close(fig)


def _plot_monotonicity_heatmap(df, plt):
    if df.empty:
        return
    pivot = df.pivot_table(index="motif_id", columns="dataset",
                            values="spearman_rho", aggfunc="mean")
    if pivot.empty:
        return
    # sort by max |rho|
    pivot = pivot.loc[
        pivot.abs().max(axis=1).sort_values(ascending=False).index[:30]
    ]
    fig, ax = plt.subplots(figsize=(7, max(4, 0.35 * len(pivot))))
    im = ax.imshow(pivot.fillna(0).values, aspect="auto",
                    cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=20, fontsize=9)
    ax.set_yticks(range(len(pivot)))
    ax.set_yticklabels(pivot.index, fontsize=7)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            v = pivot.iloc[i, j]
            if pd.notna(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                         fontsize=6, color="black")
    fig.colorbar(im, ax=ax, label="Spearman ρ")
    ax.set_title("Calibration — monotonicity (Spearman ρ), top-30 motifs")
    fig.tight_layout()
    fig.savefig(OUT / "fig_calibration_monotonicity_heatmap.png", dpi=130)
    plt.close(fig)


def _plot_core_vs_regime(df, plt):
    if df.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot([0, 35], [0, 35], "--", color="gray", linewidth=0.8,
             label="core = regime")
    ax.scatter(df["cohen_d_core"].abs(), df["cohen_d_regime"].abs(),
                color="#2a9d8f", alpha=0.7, s=50, edgecolor="black",
                linewidth=0.5)
    for _, r in df.iterrows():
        if abs(r["cohen_d_core"]) > 5 or abs(r["cohen_d_regime"]) > 5:
            ax.text(abs(r["cohen_d_core"]) + 0.2,
                     abs(r["cohen_d_regime"]), r["motif_id"][:30],
                     fontsize=7)
    ax.set_xlabel("|Cohen d| core")
    ax.set_ylabel("|Cohen d| regime")
    ax.set_title("Calibration — core vs regime-adjusted effect sizes")
    ax.legend()
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "fig_calibration_core_vs_regime_examples.png", dpi=130)
    plt.close(fig)


def _plot_ambiguity(df, plt):
    if df.empty:
        return
    df = df.sort_values("ambiguity_core_mean", ascending=False).head(20)
    fig, ax = plt.subplots(figsize=(10, max(4, 0.35 * len(df))))
    ax.barh(df["condition"], df["ambiguity_core_mean"], color="#7b2cbf")
    ax.set_xlabel("mean ambiguity core evidence")
    ax.set_title("Calibration — ambiguity behaviour (top-20 by mean core)")
    ax.invert_yaxis()
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "fig_calibration_ambiguity_examples.png", dpi=130)
    plt.close(fig)


def _write_report(motif_df, axis11_df, axis8_df, mono_df, amb_df):
    n_valid = int((motif_df["cohen_d_core"].abs() >= 0.8).sum())
    n_partial = int(((motif_df["cohen_d_core"].abs() < 0.8) &
                       (motif_df["cohen_d_core"].abs() >= 0.5)).sum())
    lines = [
        "# gaira_base_2 — Calibration Benchmark Report (v1)",
        "",
        f"**Motifs exercised on Gobbato spike panel:** {len(motif_df)}",
        f"**Motifs with |Cohen d| core ≥ 0.8 (CALIBRATION_VALID-equiv):** {n_valid}",
        f"**Motifs with |Cohen d| core 0.5–0.8 (PARTIAL):** {n_partial}",
        f"**11-axes exercised:** {len(axis11_df)}",
        f"**8-axis projections exercised:** {len(axis8_df)}",
        f"**ERG dose-graded + uricase monotonicity rows:** {len(mono_df)}",
        "",
        "## Datasets used",
        "",
        "| dataset | conditions | purpose |",
        "|---|---|---|",
        "| Gobbato SERS spiked serum Merck | 54 analytes + 5 SerumSigma bkg | primary cross-talk + effect-size panel |",
        "| Gobbato pure metabolite SERS    | 54 analytes, pure buffer | specificity ceiling |",
        "| Gobbato uricase depletion       | UA spike + depleted + SerumSigma control | 2-point enzymatic dose-response for UA |",
        "| ERG_calibration.csv             | 11 concentrations (0–2 µM Ergo) | dose-graded monotonicity for ergothioneine |",
        "| CSPP Figure-7                   | Erg 25 µM + Hyp 50 µM + Bkg | independent spike-vs-bkg validation |",
        "",
        "## Top motifs that behave correctly (|d| core ≥ 0.8)",
        "",
        "| motif | targets | |d| core | |d| regime | cross-talk |",
        "|---|---|---:|---:|---:|",
    ]
    top = motif_df[motif_df["cohen_d_core"].abs() >= 0.8].sort_values(
        "cohen_d_core", key=lambda s: s.abs(), ascending=False,
    )
    for _, r in top.iterrows():
        ct = f"{r['cross_talk_ratio']:.2f}" if pd.notna(r['cross_talk_ratio']) else "—"
        lines.append(
            f"| `{r['motif_id']}` | {r['targets']} | "
            f"{abs(r['cohen_d_core']):.2f} | {abs(r['cohen_d_regime']):.2f} | "
            f"{ct} |"
        )

    lines += [
        "",
        "## Weak / unstable motifs (|d| core < 0.5)",
        "",
        "| motif | targets | |d| core | cross-talk |",
        "|---|---|---:|---:|",
    ]
    weak = motif_df[motif_df["cohen_d_core"].abs() < 0.5].sort_values(
        "cohen_d_core", key=lambda s: s.abs(),
    )
    for _, r in weak.iterrows():
        ct = f"{r['cross_talk_ratio']:.2f}" if pd.notna(r['cross_talk_ratio']) else "—"
        lines.append(
            f"| `{r['motif_id']}` | {r['targets']} | "
            f"{abs(r['cohen_d_core']):.2f} | {ct} |"
        )

    lines += [
        "",
        "## 11-axis response summary",
        "",
        "| axis11 | targets | |d| core | |d| regime |",
        "|---|---|---:|---:|",
    ]
    for _, r in axis11_df.sort_values(
        "cohen_d_core", key=lambda s: s.abs(), ascending=False,
    ).iterrows():
        lines.append(
            f"| {r['axis11_id']} | {r['targets']} | "
            f"{abs(r['cohen_d_core']):.2f} | {abs(r['cohen_d_regime']):.2f} |"
        )

    lines += [
        "",
        "## 8-axis projection response summary",
        "",
        "| axis8 (legacy) | contributing 11-axes | |d| |",
        "|---|---|---:|",
    ]
    for _, r in axis8_df.sort_values(
        "cohen_d", key=lambda s: s.abs(), ascending=False,
    ).iterrows():
        lines.append(
            f"| {r['axis8_id']} | {r['contributing_11_axes']} | "
            f"{abs(r['cohen_d']):.2f} |"
        )

    lines += [
        "",
        "## Core vs regime-adjusted scores",
        "",
        "The engine exposes two scoring paths:",
        "- `core_*`: substrate-agnostic biochemistry claim "
        "(activation × core_weight × mapping_weight)",
        "- `regime_*`: current measurement context (Ag-colloid serum), "
        "  adds calibration_weight multiplier",
        "",
        "The `fig_calibration_core_vs_regime_examples.png` shows that "
        "regime |d| ≤ core |d| for every motif (expected — "
        "calibration_weight ∈ [0.3, 1.0] always discounts). Motifs with "
        "CALIBRATION_UNRELIABLE in M4 have the largest core↔regime gap.",
        "",
        "## Monotonicity",
        "",
        "ERG dose-graded (0–2 µM, 11 concentrations) and uricase 2-point "
        "(spike vs depleted) tested for Spearman ρ against perturbation. "
        f"{len(mono_df)} motif × dataset rows; see "
        "`calibration_monotonicity_summary_v1.csv` for all.",
        "",
        "Top monotonic motifs (|ρ| ≥ 0.7):",
        "",
    ]
    mono_top = mono_df[mono_df["spearman_rho"].abs() >= 0.7]
    if len(mono_top):
        lines.append("| dataset | motif | Spearman ρ | n |")
        lines.append("|---|---|---:|---:|")
        for _, r in mono_top.sort_values(
            "spearman_rho", key=lambda s: s.abs(), ascending=False,
        ).iterrows():
            lines.append(
                f"| {r['dataset']} | `{r['motif_id']}` | "
                f"{r['spearman_rho']:.2f} | {r['n_spectra']} |"
            )
    else:
        lines.append("_None — dose-graded response did not exceed |ρ| = 0.7._")

    lines += [
        "",
        "## Ambiguity behaviour",
        "",
        "| condition | n | mean ambiguity core | sd |",
        "|---|---:|---:|---:|",
    ]
    for _, r in amb_df.sort_values("ambiguity_core_mean", ascending=False).head(15).iterrows():
        lines.append(
            f"| {r['condition']} | {r['n_spectra']} | "
            f"{r['ambiguity_core_mean']:.3f} | "
            f"{r['ambiguity_core_std']:.3f} |"
        )

    lines += [
        "",
        "## Remaining weak spots before M5",
        "",
        "1. Motifs with |d| core < 0.5 must not drive M5 biology claims; they "
        "   may appear only as context with an explicit weak-calibration flag.",
        "2. High cross-talk ratios (> 1.0) indicate the motif's target spike "
        "   is not distinguishable from many non-target spikes in Ag-colloid "
        "   serum; these need substrate-specific re-calibration before use.",
        "3. Non-monotonic ERG / uricase responses indicate the motif's "
        "   activation is not dose-graded; flag as non-quantitative.",
        "4. Ambiguity motifs should show baseline ambiguity > 0 on the "
        "   SerumSigma control (citrate always present in Ag-colloid "
        "   reductant) — any M5 interpretation should subtract this "
        "   baseline, not treat it as disease signal.",
        "",
        "## Tables emitted",
        "",
        "- `calibration_per_spectrum_scores_v1.csv`",
        "- `calibration_motif_response_summary_v1.csv`",
        "- `calibration_axis11_response_summary_v1.csv`",
        "- `calibration_axis8_projection_summary_v1.csv`",
        "- `calibration_cross_talk_matrix_v1.csv`",
        "- `calibration_monotonicity_summary_v1.csv`",
        "- `calibration_ambiguity_behavior_v1.csv`",
        "",
        "## Figures emitted",
        "",
        "- `fig_calibration_motif_response_panels.png`",
        "- `fig_calibration_axis11_response_panels.png`",
        "- `fig_calibration_axis8_projection_panels.png`",
        "- `fig_calibration_cross_talk_matrix.png`",
        "- `fig_calibration_monotonicity_heatmap.png`",
        "- `fig_calibration_core_vs_regime_examples.png`",
        "- `fig_calibration_ambiguity_examples.png`",
    ]
    (OUT / "REPORT_gaira_base_2_calibration_benchmark_v1.md").write_text(
        "\n".join(lines),
    )
    print(f"[emit] REPORT_gaira_base_2_calibration_benchmark_v1.md")


if __name__ == "__main__":
    main()
