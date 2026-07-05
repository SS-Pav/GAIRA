"""Derive lightweight demo assets for the GAIRA polished Streamlit demo.

Inputs (authoritative):
  - Raman physics atlas:   GAIRA/config/spectral_anchor_windows_v1.csv
  - Grounding corpus:      RamanBioLib (/Volumes/SSD_Rad/.../ramanbiolib)
  - Calibration v3:        /Volumes/SSD_Rad/.../gaira_calibration_eval_v3/tables
  - Regression dataset:    /Volumes/SSD_Rad/.../ergothioneine_serum/ERG_calibration.csv
  - GAIRA pipeline:        gaira.spectral.{preprocessing, window_panel, bsv_projection}

Outputs (derived, live only under streamlit_apps/gaira_demo/data):
  - atlas_explorer.csv
  - grounding_corpus_summary.csv
  - grounding_molecule_index.csv
  - grounding_molecule_spectra.parquet
  - grounding_molecule_bsv.csv
  - calibration_conditions.csv
  - calibration_delta_bsv.csv
  - ergothioneine_dose_response.csv
  - ergothioneine_bsv_per_concentration.csv
  - ergothioneine_spectra_mean.parquet (for optional sanity checks)

Run:
    cd /Users/suraj/projects/GAIRA
    PYTHONPATH=src .venv/bin/python streamlit_apps/gaira_demo/build_demo_assets.py
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from gaira.spectral.bsv_projection import project_to_bsv  # noqa: E402
from gaira.spectral.window_panel import BSV_COMPONENTS, WINDOW_DEFS, extract_window_features  # noqa: E402
from gaira.calibration.preprocessing import preprocess_calibration  # noqa: E402


ATLAS_CSV = ROOT / "config" / "spectral_anchor_windows_v1.csv"
RAMANBIO_DIR = Path(
    "/Volumes/SSD_Rad/GAIRA_DATA/raw/ramanbiolib/ramanbiolib-main/ramanbiolib/db"
)
CAL_V3_TABLES = Path(
    "/Volumes/SSD_Rad/GAIRA_DATA/processed/gaira_calibration_eval_v3/tables"
)
ERG_CSV = Path("/Volumes/SSD_Rad/GAIRA_DATA/raw/ergothioneine_serum/ERG_calibration.csv")

OUT = Path(__file__).parent / "data"
OUT.mkdir(parents=True, exist_ok=True)


# ────────────────────────────────────────────────────────────────
# 1. Atlas explorer
# ────────────────────────────────────────────────────────────────

def build_atlas_explorer() -> pd.DataFrame:
    atlas = pd.read_csv(ATLAS_CSV)
    atlas["n_supporting_sources"] = (
        atlas["supporting_source_ids"].fillna("").str.split(";").map(
            lambda xs: len([x for x in xs if x.strip()])
        )
    )
    atlas["candidate_axes_list"] = atlas["candidate_axes"].fillna("").apply(
        lambda s: [x.strip() for x in s.split(";") if x.strip()]
    )
    atlas["n_candidate_axes"] = atlas["candidate_axes_list"].map(len)
    atlas["has_companion"] = atlas["n_candidate_axes"] > 1
    atlas["width_cm1"] = (atlas["end_cm1"] - atlas["start_cm1"]).round(1)
    atlas["display_label"] = atlas.apply(
        lambda r: f"{int(r['central_cm1'])} cm⁻¹  [{r['primary_axis']}]", axis=1
    )
    atlas["range_label"] = atlas.apply(
        lambda r: f"{int(r['start_cm1'])}–{int(r['end_cm1'])}", axis=1
    )

    keep_cols = [
        "window_id", "primary_axis", "classification",
        "start_cm1", "end_cm1", "central_cm1", "width_cm1",
        "range_label", "display_label",
        "candidate_axes", "n_candidate_axes", "has_companion",
        "ambiguity_score", "locality_score",
        "source_count", "n_supporting_sources",
        "matrix_distribution", "substrate_distribution",
        "priority_tags", "notes",
    ]
    atlas_out = atlas[keep_cols].copy()
    atlas_out.to_csv(OUT / "atlas_explorer.csv", index=False)
    return atlas_out


# ────────────────────────────────────────────────────────────────
# 2. Grounding corpus + pure-molecule spectra + BSV
# ────────────────────────────────────────────────────────────────

def _parse_list(x: str) -> list:
    if pd.isna(x):
        return []
    try:
        return list(ast.literal_eval(x))
    except Exception:
        return []


# coarse family grouping for readable selectors
FAMILY_ORDER = [
    "Proteins", "AminoAcids", "Lipids", "Saccharides",
    "NucleicAcids", "PrimaryMetabolites", "Pigments", "Vitamins", "Other",
]

def _family(raw_type: str) -> str:
    if not isinstance(raw_type, str):
        return "Other"
    t = raw_type.split("/")[0]
    if t in FAMILY_ORDER:
        return t
    return "Other"


def build_grounding_assets() -> dict:
    meta = pd.read_csv(RAMANBIO_DIR / "metadata_db.csv")
    spec = pd.read_csv(RAMANBIO_DIR / "raman_spectra_db.csv")

    # Parse array columns
    spec["wn_list"] = spec["wavenumbers"].map(_parse_list)
    spec["int_list"] = spec["intensity"].map(_parse_list)
    spec = spec[spec["wn_list"].map(len) > 0].reset_index(drop=True)

    # All spectra share wn=[450..1800] per metadata; use the first as reference
    wn_ref = np.asarray(spec.iloc[0]["wn_list"], dtype=float)
    keep_ids = []
    rows_spec = []
    X_rows = []
    for _, row in spec.iterrows():
        wn_i = np.asarray(row["wn_list"], dtype=float)
        y_i = np.asarray(row["int_list"], dtype=float)
        if len(wn_i) != len(wn_ref) or not np.allclose(wn_i, wn_ref, atol=0.5):
            # resample
            y_i = np.interp(wn_ref, wn_i, y_i)
        keep_ids.append(int(row["id"]))
        rows_spec.append({"id": int(row["id"]), "component": row["component"]})
        X_rows.append(y_i)
    X = np.asarray(X_rows, dtype=float)

    # Min-max per spectrum for consistent display
    X_disp = X.copy()
    for i in range(X_disp.shape[0]):
        mn, mx = X_disp[i].min(), X_disp[i].max()
        if mx > mn:
            X_disp[i] = (X_disp[i] - mn) / (mx - mn)
        else:
            X_disp[i] = 0.0

    # BSV projection from already-preprocessed spectra
    feats = extract_window_features(X_disp, wn_ref)
    bsv = project_to_bsv(feats)

    # Join meta to the kept ids
    meta_idx = meta.set_index("id").reindex(keep_ids)
    meta_idx = meta_idx.reset_index(drop=False).rename(columns={"index": "id"})
    meta_idx["family"] = meta_idx["type"].map(_family)

    molecule_index = pd.DataFrame({
        "id": keep_ids,
        "component": [r["component"] for r in rows_spec],
        "type": meta_idx["type"].values,
        "family": meta_idx["family"].values,
        "sample_substrate": meta_idx["sample_substrate"].values,
        "laser_wavelength": meta_idx["laser_wavelength"].values,
    })

    # Spectra parquet: long columns
    spectra_df = pd.DataFrame({
        "wavenumber": np.tile(wn_ref, len(keep_ids)),
        "intensity_norm": X_disp.reshape(-1),
        "id": np.repeat(keep_ids, len(wn_ref)),
    })
    try:
        spectra_df.to_parquet(OUT / "grounding_molecule_spectra.parquet", index=False)
    except Exception:
        spectra_df.to_csv(OUT / "grounding_molecule_spectra.csv", index=False)

    # BSV
    bsv_df = pd.DataFrame(bsv, columns=BSV_COMPONENTS)
    bsv_df.insert(0, "id", keep_ids)
    bsv_df.insert(1, "component", [r["component"] for r in rows_spec])
    bsv_df.insert(2, "family", meta_idx["family"].values)
    bsv_df["dominant_axis"] = bsv_df[BSV_COMPONENTS].idxmax(axis=1)
    bsv_df["dominant_weight"] = bsv_df[BSV_COMPONENTS].max(axis=1)
    bsv_df.to_csv(OUT / "grounding_molecule_bsv.csv", index=False)
    molecule_index.to_csv(OUT / "grounding_molecule_index.csv", index=False)

    # Corpus summary
    summary_rows = [
        ("n_molecule_spectra", int(len(keep_ids))),
        ("n_metadata_entries", int(len(meta))),
        ("n_atlas_bands", int(pd.read_csv(ATLAS_CSV).shape[0])),
        ("n_atlas_axes", int(pd.read_csv(ATLAS_CSV)["primary_axis"].nunique())),
        ("wavenumber_min_cm1", float(wn_ref.min())),
        ("wavenumber_max_cm1", float(wn_ref.max())),
    ]
    summary = pd.DataFrame(summary_rows, columns=["metric", "value"])
    summary.to_csv(OUT / "grounding_corpus_summary.csv", index=False)

    # Category breakdown
    fam_counts = molecule_index["family"].value_counts().reset_index()
    fam_counts.columns = ["family", "n_molecules"]
    fam_counts.to_csv(OUT / "grounding_family_counts.csv", index=False)

    # Axis coverage: how many atlas bands support each primary axis
    atlas = pd.read_csv(ATLAS_CSV)
    axis_counts = (
        atlas.groupby(["primary_axis", "classification"])["window_id"].count()
        .unstack(fill_value=0).reset_index()
    )
    axis_counts.to_csv(OUT / "atlas_axis_coverage.csv", index=False)

    return {
        "n_kept": len(keep_ids),
        "wn_range": (float(wn_ref.min()), float(wn_ref.max())),
    }


# ────────────────────────────────────────────────────────────────
# 3. Calibration: reuse v3 outputs into a compact demo table
# ────────────────────────────────────────────────────────────────

def build_calibration_assets() -> dict:
    contrasts = pd.read_csv(CAL_V3_TABLES / "calibration_contrast_summary_v3.csv")
    delta = pd.read_csv(CAL_V3_TABLES / "calibration_delta_bsv_v3.csv")

    # Tidy conditions
    keep_cols = [
        "contrast_id", "display_name", "sael_contrast_id",
        "sael_status", "sael_overall_confidence",
        "n_control", "n_perturbed", "n_testable_axes", "testable_axes",
        "confidence_weighted_score",
        "n_high_conf_agree", "n_moderate_conf_agree", "n_low_conf_agree",
        "n_disagree", "n_mixed_resolved", "n_mixed_flat",
        "n_flat", "n_not_testable", "overall_label",
    ]
    contrasts_out = contrasts[keep_cols].copy()
    contrasts_out.to_csv(OUT / "calibration_conditions.csv", index=False)

    # Reorder delta rows so axes follow canonical order per contrast
    delta["_axis_order"] = delta["axis"].map(
        {c: i for i, c in enumerate(BSV_COMPONENTS)}
    ).fillna(len(BSV_COMPONENTS)).astype(int)
    delta = delta.sort_values(["contrast_id", "_axis_order"]).drop(columns="_axis_order")
    delta.to_csv(OUT / "calibration_delta_bsv.csv", index=False)

    return {"n_contrasts": len(contrasts_out), "n_axis_rows": len(delta)}


# ────────────────────────────────────────────────────────────────
# 4. Regression: Ergothioneine titration → per-concentration BSV
# ────────────────────────────────────────────────────────────────

_ERG_META_COLS = {"laser", "power", "substrate", "c"}


def build_ergothioneine_dose_response() -> dict:
    df = pd.read_csv(ERG_CSV)
    wn_cols = [c for c in df.columns if c not in _ERG_META_COLS]
    wn_raw = np.asarray([float(c) for c in wn_cols], dtype=float)
    X_raw = df[wn_cols].to_numpy(dtype=float)
    conc = df["c"].to_numpy(dtype=float)

    # Preprocess with GAIRA calibration pipeline (AsLS + SG + L2)
    pp = preprocess_calibration(X_raw, wn_raw, crop_range=(400.0, 1800.0))
    X = pp.X
    wn = pp.wavenumbers

    # Window features → BSV (per-spectrum, then averaged per concentration)
    feats = extract_window_features(X, wn)
    bsv = project_to_bsv(feats)

    # Per-concentration mean BSV + std
    rows = []
    mean_spec_rows = []
    baseline_mask = conc == 0.0
    baseline_bsv = bsv[baseline_mask].mean(axis=0)

    conc_levels = sorted(set(float(c) for c in conc))
    for c_val in conc_levels:
        mask = conc == c_val
        bsv_c = bsv[mask]
        mean_bsv = bsv_c.mean(axis=0)
        std_bsv = bsv_c.std(axis=0)
        delta_bsv = mean_bsv - baseline_bsv
        commit = (np.abs(delta_bsv) > 0.005).sum()  # simple commit fraction
        row = {
            "concentration_uM": c_val,
            "n_spectra": int(mask.sum()),
            "radar_area": float(np.sum(mean_bsv)),
            "commit_axes": int(commit),
        }
        for i, comp in enumerate(BSV_COMPONENTS):
            row[f"bsv_{comp}"] = float(mean_bsv[i])
            row[f"bsv_std_{comp}"] = float(std_bsv[i])
            row[f"delta_bsv_{comp}"] = float(delta_bsv[i])
        rows.append(row)

        mean_spec = X[mask].mean(axis=0)
        for j, wn_j in enumerate(wn):
            mean_spec_rows.append({
                "concentration_uM": c_val,
                "wavenumber": float(wn_j),
                "intensity_l2norm": float(mean_spec[j]),
            })

    out_df = pd.DataFrame(rows).sort_values("concentration_uM").reset_index(drop=True)
    out_df.to_csv(OUT / "ergothioneine_bsv_per_concentration.csv", index=False)

    # Dose-response summary (tidy long format for plotting)
    long_rows = []
    for _, r in out_df.iterrows():
        for comp in BSV_COMPONENTS:
            long_rows.append({
                "concentration_uM": r["concentration_uM"],
                "axis": comp,
                "bsv_mean": r[f"bsv_{comp}"],
                "bsv_std": r[f"bsv_std_{comp}"],
                "delta_bsv": r[f"delta_bsv_{comp}"],
            })
    long_df = pd.DataFrame(long_rows)
    long_df.to_csv(OUT / "ergothioneine_dose_response.csv", index=False)

    # Mean spectra per conc (for optional view)
    mean_spec_df = pd.DataFrame(mean_spec_rows)
    try:
        mean_spec_df.to_parquet(OUT / "ergothioneine_spectra_mean.parquet", index=False)
    except Exception:
        mean_spec_df.to_csv(OUT / "ergothioneine_spectra_mean.csv", index=False)

    return {"n_conc_levels": len(conc_levels), "baseline_conc_uM": 0.0}


# ────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────

def main():
    print(f"[assets] writing to {OUT}")
    atlas = build_atlas_explorer()
    print(f"  atlas_explorer: {len(atlas)} bands")
    ground = build_grounding_assets()
    print(f"  grounding: {ground}")
    cal = build_calibration_assets()
    print(f"  calibration: {cal}")
    erg = build_ergothioneine_dose_response()
    print(f"  ergothioneine: {erg}")
    print("[assets] done")


if __name__ == "__main__":
    main()
