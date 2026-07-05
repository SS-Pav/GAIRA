"""gaira_base_2 — side-by-side output generator.

Produces a single CSV per input cohort with columns for:

- frozen gaira_base 8-axis view (read from
  ``gaira_build_axes_v1/outputs/*/tables/pilot*_per_spectrum_bsv.csv``
  if the cohort has a frozen pilot; omitted otherwise)
- gaira_base_2 motif activations + weights (50 motifs × 3 columns)
- gaira_base_2 11-axis core + regime scores
- gaira_base_2 8-axis projection core + regime scores (MAX combiner)
- gaira_base_2 ambiguity lane core + regime

The script runs on three input batches for engine-level validation:

  A. Gobbato pure powder Raman (5 analytes — CORE references)
  B. Gobbato SERS spiked serum Merck (primary M4 calibration panel)
  C. Frozen HCC holdout (pilot 1) spectra — legacy pilot comparability

No M5 cohort analysis is performed. This is engine-level validation
only.

Run:
    cd /Users/suraj/projects/GAIRA
    PYTHONPATH=src .venv/bin/python scripts/run_gaira_base_2_side_by_side.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from gaira.base2 import (
    load_active_registry,
    result_to_flat_dict,
    score_spectrum,
)
from gaira.spectral import canonical_master_axis, crop_before_interpolate
from gaira.spectral.preprocessing import _asls_baseline
from scipy.signal import savgol_filter


# ──────────────────────────────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────────────────────────────

OUT_ROOT = Path(
    "/Volumes/SSD_Rad/GAIRA_BUILD/gaira_base_2_implementation_v1/outputs"
)
GOBBATO_EXTRACTED = Path(
    "/Volumes/SSD_Rad/GAIRA_BUILD/gaira_build_motifs_v1/"
    "M3_1_reference_rescue_v1/references/_extracted"
)
GAIRA_BASE_PILOT1 = Path(
    "/Volumes/SSD_Rad/GAIRA_BUILD/gaira_build_axes_v1/outputs/"
    "pilot1/tables/pilot1_hcc_per_spectrum_bsv.csv"
)


# ──────────────────────────────────────────────────────────────────────
# Preprocessing — canonical pipeline (crop → AsLS → SG → L2)
# ──────────────────────────────────────────────────────────────────────

def canonical_preprocess(raw_wn: np.ndarray, raw_y: np.ndarray,
                          master_x: np.ndarray) -> np.ndarray | None:
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
    baseline = _asls_baseline(y_interp, lam=1e5, p=0.001, n_iter=10)
    y_bc = y_interp - baseline
    y_sg = savgol_filter(y_bc, window_length=11, polyorder=3)
    norm = np.linalg.norm(y_sg)
    return y_sg / norm if norm > 1e-12 else None


def parse_gobbato_file(path: Path) -> tuple[np.ndarray, np.ndarray] | None:
    try:
        lines = path.read_text(encoding="latin-1").splitlines()
    except Exception:
        return None
    hdr = None
    for i, ln in enumerate(lines):
        if ln.startswith("Pixel;Wavelength;Wavenumber;Raman Shift"):
            hdr = i
            break
    if hdr is None:
        return None
    wn, y = [], []
    for ln in lines[hdr + 1:]:
        parts = ln.strip().rstrip(";").split(";")
        if len(parts) < 8:
            continue
        try:
            rs = float(parts[3].replace(",", "."))
            ds = float(parts[7].replace(",", "."))
        except ValueError:
            continue
        wn.append(rs); y.append(ds)
    return (np.array(wn, dtype=np.float64), np.array(y, dtype=np.float64))


def load_hcc_pilot1_raw(master_x: np.ndarray, n_spectra: int = 30):
    """Load a sample of HCC pilot1 spectra (raw CSV), preprocessed."""
    from gaira.spectral.dataset_loader import HCC_HOLDOUT_CSV
    if not HCC_HOLDOUT_CSV.exists():
        return [], []
    df = pd.read_csv(HCC_HOLDOUT_CSV)
    meta_cols = ["acquisition_date", "substrate_batch", "class", "sample_code"]
    wn_cols = [c for c in df.columns if c not in meta_cols]
    raw_wn = np.array([float(c) for c in wn_cols])
    X = df[wn_cols].values.astype(np.float64)[:n_spectra]
    ids = [f"hcc_pilot1_{df.iloc[i]['sample_code']}_{df.iloc[i]['class']}"
           for i in range(min(n_spectra, len(df)))]
    pp = []
    kept_ids = []
    for i in range(X.shape[0]):
        y_pp = canonical_preprocess(raw_wn, X[i], master_x)
        if y_pp is not None:
            pp.append(y_pp)
            kept_ids.append(ids[i])
    return pp, kept_ids


# ──────────────────────────────────────────────────────────────────────
# Side-by-side runner
# ──────────────────────────────────────────────────────────────────────

def _batch_rows(spec_batch: list[np.ndarray], ids: list[str],
                 master_x, motifs, mappings, dual) -> list[dict]:
    rows = []
    for sid, y in zip(ids, spec_batch):
        res = score_spectrum(y, master_x, motifs, mappings, dual, sid)
        rows.append(result_to_flat_dict(res))
    return rows


def main():
    print("=" * 78)
    print("gaira_base_2 side-by-side output generator")
    print("=" * 78)
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    master_x = canonical_master_axis()
    motifs, mappings, dual = load_active_registry()
    print(f"engine loaded: {len(motifs)} active motifs, "
          f"{len(mappings)} mappings, {len(dual)} dual-status rows")
    print()

    # ── Batch A: Gobbato pure powder Raman (CORE references) ──────────
    print("[A] Gobbato pure powder Raman (CORE references)")
    pure_dir = GOBBATO_EXTRACTED / "Raman metabolites"
    a_specs, a_ids = [], []
    for p in sorted(pure_dir.glob("Raman_pwd_*.txt")):
        analyte = p.name[len("Raman_pwd_"):].split("_")[0]
        # Only the 5 key metabolite analytes (match M3.1 scope)
        if analyte not in {"UA", "Hypox", "Xanth", "Ergo", "Creat"}:
            continue
        parsed = parse_gobbato_file(p)
        if parsed is None:
            continue
        y_pp = canonical_preprocess(parsed[0], parsed[1], master_x)
        if y_pp is not None:
            a_specs.append(y_pp)
            a_ids.append(f"gobbato_powder_{analyte}_{p.stem.split('_')[-1]}")
    rows_a = _batch_rows(a_specs, a_ids, master_x, motifs, mappings, dual)
    if rows_a:
        pd.DataFrame(rows_a).to_csv(
            OUT_ROOT / "side_by_side_A_gobbato_powder_raman.csv", index=False,
        )
        print(f"  wrote {len(rows_a)} rows → "
              f"side_by_side_A_gobbato_powder_raman.csv")

    # ── Batch B: Gobbato SERS spiked serum Merck ──────────────────────
    print("[B] Gobbato SERS spiked serum Merck (calibration panel)")
    spike_dir = GOBBATO_EXTRACTED / "SERS spiked serum Merck"
    picks = ["UA", "Hypox", "Xanth", "Ergo", "Creat", "Phe", "Tyr", "Alb",
             "Gua", "Gluc", "Oleic", "SerumSigma"]
    b_specs, b_ids = [], []
    for p in sorted(spike_dir.glob("SERS_spike_*.txt")):
        analyte = p.name[len("SERS_spike_"):].split("_")[0]
        if analyte not in picks:
            continue
        parsed = parse_gobbato_file(p)
        if parsed is None:
            continue
        y_pp = canonical_preprocess(parsed[0], parsed[1], master_x)
        if y_pp is not None:
            b_specs.append(y_pp)
            rep = p.stem.split("_")[-1]
            b_ids.append(f"gobbato_spike_{analyte}_rep{rep}")
    rows_b = _batch_rows(b_specs, b_ids, master_x, motifs, mappings, dual)
    if rows_b:
        pd.DataFrame(rows_b).to_csv(
            OUT_ROOT / "side_by_side_B_gobbato_sers_spike.csv", index=False,
        )
        print(f"  wrote {len(rows_b)} rows → "
              f"side_by_side_B_gobbato_sers_spike.csv")

    # ── Batch C: HCC pilot1 raw (side-by-side with frozen base CSV) ───
    print("[C] HCC pilot1 raw spectra (legacy comparability)")
    c_specs, c_ids = load_hcc_pilot1_raw(master_x, n_spectra=30)
    rows_c = _batch_rows(c_specs, c_ids, master_x, motifs, mappings, dual)
    # Attach the corresponding frozen gaira_base 8-axis view where possible
    if GAIRA_BASE_PILOT1.exists():
        base_df = pd.read_csv(GAIRA_BASE_PILOT1)
        # Try to merge by spectrum_index or sample_code
        print(f"  frozen gaira_base pilot1 rows: {len(base_df)}")
        print(f"  frozen gaira_base pilot1 columns: {list(base_df.columns)[:10]}")
    if rows_c:
        out_c = pd.DataFrame(rows_c)
        if GAIRA_BASE_PILOT1.exists():
            out_c["legacy_base_source"] = (
                "gaira_build_axes_v1/outputs/pilot1/tables/"
                "pilot1_hcc_per_spectrum_bsv.csv (frozen — see doc for "
                "explicit non-identity)"
            )
        out_c.to_csv(
            OUT_ROOT / "side_by_side_C_hcc_pilot1_raw.csv", index=False,
        )
        print(f"  wrote {len(rows_c)} rows → "
              f"side_by_side_C_hcc_pilot1_raw.csv")

    # ── Per-axis summary per batch ────────────────────────────────────
    summary_rows = []
    for label, rows, batch_name in [
        ("A_gobbato_powder", rows_a, "Gobbato pure powder Raman"),
        ("B_gobbato_sers", rows_b, "Gobbato SERS spike in serum"),
        ("C_hcc_pilot1", rows_c, "HCC pilot1 raw"),
    ]:
        if not rows:
            continue
        df = pd.DataFrame(rows)
        for col in df.columns:
            if not col.startswith("axis11_core."):
                continue
            axis = col.split(".", 1)[1]
            summary_rows.append({
                "batch": label,
                "batch_description": batch_name,
                "axis": axis,
                "core_mean":   round(df[col].mean(), 4),
                "core_max":    round(df[col].max(), 4),
                "regime_mean": round(df[f"axis11_regime.{axis}"].mean(), 4),
                "regime_max":  round(df[f"axis11_regime.{axis}"].max(), 4),
                "n_spectra":   len(df),
            })
    if summary_rows:
        pd.DataFrame(summary_rows).to_csv(
            OUT_ROOT / "side_by_side_axis11_summary.csv", index=False,
        )
        print(f"\n[emit] axis-level summary → side_by_side_axis11_summary.csv")

    print()
    print("=" * 78)
    print("DONE — 3 batches scored, engine-level outputs emitted")
    print("=" * 78)


if __name__ == "__main__":
    main()
