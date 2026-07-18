"""GAIRA Demo v3 — deterministic global-coordinate calibration builder.

Assembles a FROZEN, versioned global-coordinate calibration from three roles:

  Role A  Biochemical anchors  — the 202 reference-analyte family mapping +
          curated motif/MSS definitions define axis MEANING (fixed in engine).
          (The 202-molecule table is 8-axis legacy; it is used for coverage /
          ontology grounding, NOT for center/scale.)
  Role B  Calibration behaviour — adenine (6 live Ag-SERS concentrations) and
          ergothioneine (55 live Ag-SERS spectra, 11 concentrations) projected
          through the UNCHANGED V2 engine.
  Role C  Biological range — serum-liver (212 patients) + EV-diabetes (63
          samples), projected label-free, to estimate realistic ranges and
          nuisance variation. These do NOT define axis meaning.

Calibration = robust per-axis standardization fit on the pooled, LABEL-FREE
reference/calibration population:
    center_j = median(raw_bsv_j)                 (over pooled population)
    scale_j  = 1.4826 * MAD(raw_bsv_j)  (floored) (robust ~sigma)
    global_j = (raw_j - center_j) / scale_j
Robust stats prevent a single high-dynamic-range axis (e.g. redox G10) from
dominating by numeric scale, while preserving real relative biological
variation (an off-reference sample still exceeds +/-1). Labels are never used.

Deterministic: build twice -> identical numeric content (build_timestamp is
stored separately and excluded from the content hash).

Outputs (data/generated/):
  global_coordinate_calibration_v1.json
  global_coordinate_reference_samples_v1.csv
  global_coordinate_build_manifest_v1.json

Usage:  python tools/build_global_coordinate_reference.py [--timestamp ISO]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

DEMO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(DEMO_ROOT))

from gaira_core import config as cfg                       # noqa: E402
from gaira_core import global_coordinates as gc            # noqa: E402
from gaira_core.ontology import load_ontology              # noqa: E402

OUT_DIR = cfg.GENERATED_DIR
OUT_DIR.mkdir(parents=True, exist_ok=True)

CAL_VERSION = "v1"
# Scale floor = a minimum meaningful reference spread (BSV units). Axes whose
# robust reference spread is below this are treated as noise-level: without a
# floor, thinly-grounded axes (e.g. Purine-nuc/Pyrimidine, reference MAD ~1e-3)
# would explode any small deviation into huge z-scores and dominate global
# variance purely from near-zero spread. 0.02 = 2% BSV. Well-grounded axes
# (glycan/protein/aromatic/sterol/redox/metabolite) keep their real spread.
SCALE_FLOOR = 0.02
CLIP = 4.0                  # display clip in robust-sigma units
Q_LOW, Q_HIGH = 2.5, 97.5   # reference quantiles (percent)
WN_GRID = np.linspace(cfg.WAVENUMBER_MIN, cfg.WAVENUMBER_MAX, cfg.WAVENUMBER_N)


def _sha256(path: Path) -> str:
    if not path.exists():
        return "MISSING"
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ── spectrum readers (label-free projection; labels carried but unused in fit) ──

def _interp(wn, y):
    order = np.argsort(wn)
    return np.interp(WN_GRID, np.asarray(wn)[order], np.asarray(y)[order],
                     left=0.0, right=0.0)


def read_serum():
    p = cfg.LIVER_PATIENT_TABLES / "patient_level_mean_spectra.csv"
    if not p.exists():
        return [], p
    sp = pd.read_csv(p)
    wn_cols = [c for c in sp.columns if c.startswith("wn_")]
    wn = np.array([int(c[3:]) for c in wn_cols], float)
    rows = []
    for _, r in sp.iterrows():
        y = np.clip(r[wn_cols].to_numpy(float), 0, None)
        rows.append(dict(dataset="serum_liver", sample_id=str(r["sample_id"]),
                         label=str(r.get("class_label_display", "")),
                         matrix="serum", substrate="Ag colloid SERS",
                         wn=wn, y=y, role="biological_range"))
    return rows, p


def read_ev():
    p = cfg.EV_DIABETES_TABLES / "sample_query_spectra.csv"
    if not p.exists():
        return [], p
    sp = pd.read_csv(p)
    rows = []
    for _, r in sp.iterrows():
        try:
            wn = np.asarray(json.loads(r["wavenumbers_json"]), float)
            y = np.asarray(json.loads(r["intensity_json"]), float)
        except Exception:
            continue
        rows.append(dict(dataset="ev_diabetes", sample_id=str(r["sample_id"]),
                         label=str(r.get("class_label", "")),
                         matrix="extracellular_vesicle", substrate="Ag colloid SERS",
                         wn=WN_GRID, y=np.clip(_interp(wn, y), 0, None),
                         role="biological_range"))
    return rows, p


def read_adenine():
    from gaira_core.data_loader import _ADENINE_FILES, _read_adenine_csv, _crop_and_interp
    d = cfg.ADENINE_RAW_DIR
    rows, files = [], []
    for fname, label, ng in _ADENINE_FILES:
        p = d / fname
        if not p.exists():
            continue
        parsed = _read_adenine_csv(p)
        if parsed is None:
            continue
        wn, y = _crop_and_interp(*parsed)
        if not np.any(y):
            continue
        rows.append(dict(dataset="adenine", sample_id=fname, label=label,
                         matrix="reference_calibration", substrate="Ag colloid SERS",
                         wn=wn, y=y, role="calibration_behavior",
                         concentration_ng_mL=ng))
        files.append(p)
    return rows, files


def read_ergothioneine():
    p = cfg.GAIRA_DATA_VOLUME / "raw" / "ergothioneine_serum" / "ERG_calibration.csv"
    if not p.exists():
        return [], p
    df = pd.read_csv(p)
    meta = ["laser", "power", "substrate", "c"]
    wn_cols = [c for c in df.columns if c not in meta]
    wn = np.array([float(c) for c in wn_cols], float)
    rows = []
    for i, r in df.iterrows():
        y = _interp(wn, r[wn_cols].to_numpy(float))
        rows.append(dict(dataset="ergothioneine", sample_id=f"erg_{i}",
                         label=f"{r['c']}uM", matrix="serum_calibration",
                         substrate="Ag colloid SERS",
                         wn=WN_GRID, y=np.clip(y, 0, None), role="calibration_behavior",
                         concentration_uM=float(r["c"])))
    return rows, p


def project(rows):
    out = []
    for r in rows:
        bsv = gc.raw_bsv_from_spectrum(r["wn"], r["y"], substrate=r["substrate"],
                                       domain=r["matrix"], sample_id=r["sample_id"])
        rec = {k: r[k] for k in r if k not in ("wn", "y")}
        for a in cfg.BSV_AXES:
            rec[f"raw_{a}"] = float(bsv[a])
        rec["_raw_bsv"] = bsv
        out.append(rec)
    return out


def fit_calibration(projected):
    """Robust per-axis center/scale over the pooled, label-free population."""
    M = np.array([[p[f"raw_{a}"] for a in cfg.BSV_AXES] for p in projected], float)
    center = np.median(M, axis=0)
    mad = np.median(np.abs(M - center), axis=0) * 1.4826
    scale = np.maximum(mad, SCALE_FLOOR)
    q_low = np.percentile(M, Q_LOW, axis=0)
    q_high = np.percentile(M, Q_HIGH, axis=0)
    return (dict(zip(cfg.BSV_AXES, center.tolist())),
            dict(zip(cfg.BSV_AXES, scale.tolist())),
            dict(zip(cfg.BSV_AXES, q_low.tolist())),
            dict(zip(cfg.BSV_AXES, q_high.tolist())))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--timestamp", default=None,
                    help="ISO timestamp to stamp (default: now). Excluded from content hash.")
    args = ap.parse_args()

    onto = load_ontology()

    serum, serum_p = read_serum()
    ev, ev_p = read_ev()
    adenine, adenine_files = read_adenine()
    erg, erg_p = read_ergothioneine()

    all_rows = serum + ev + adenine + erg
    if not all_rows:
        print("ERROR: no spectra available (is GAIRA_DATA mounted?). Not writing artifacts.")
        return 2

    projected = project(all_rows)
    # Fit center/scale on the BIOLOGICAL population only (the "normal reference
    # range"). Calibration titration spectra (adenine, ergothioneine) are
    # projected through the SAME frozen transform but are NOT used to set the
    # reference range — by design their extremes (e.g. high-dose ergothioneine
    # redox) may exceed the biological range. Labels are never used.
    fit_pop = [p for p in projected if p.get("role") == "biological_range"]
    if len(fit_pop) < 20:
        fit_pop = projected  # degraded fallback: too few biological spectra
    center, scale, q_low, q_high = fit_calibration(fit_pop)

    calib = gc.GlobalCalibration(
        ontology_version=onto.version, calibration_version=CAL_VERSION,
        axes=tuple(cfg.BSV_AXES), center=center, scale=scale, q_low=q_low, q_high=q_high,
        clip=CLIP, transform="robust_z_median_mad", scale_floor=SCALE_FLOOR,
        n_reference_analytes=202,
        n_calibration_spectra=len(adenine) + len(erg),
        n_biological_spectra=len(serum) + len(ev),
        matrix_composition={}, substrate_composition={}, limitations="")

    # attach global coordinates to every reference sample
    ref_records = []
    for p in projected:
        g = gc.to_global(p["_raw_bsv"], calib)
        rec = {k: v for k, v in p.items() if k != "_raw_bsv"}
        for a in cfg.BSV_AXES:
            rec[f"global_{a}"] = g[a]["unbounded"]
            rec[f"globaldisp_{a}"] = g[a]["display"]
        ref_records.append(rec)
    ref_df = pd.DataFrame(ref_records)

    matrix_comp = ref_df["matrix"].value_counts().to_dict()
    substrate_comp = ref_df["substrate"].value_counts().to_dict()

    # ── calibration artifact (content = numeric; timestamp excluded) ──
    calibration_content = {
        "ontology_version": onto.version,
        "calibration_version": CAL_VERSION,
        "axes": list(cfg.BSV_AXES),
        "transform": "robust_z_median_mad",
        "formula": "global_j = (raw_bsv_j - center_j) / scale_j ; display = clip(global, -clip, clip)",
        "scale_floor": SCALE_FLOOR,
        "clip": CLIP,
        "reference_quantile_low_pct": Q_LOW,
        "reference_quantile_high_pct": Q_HIGH,
        "axis_center": center,
        "axis_scale": scale,
        "reference_q_low": q_low,
        "reference_q_high": q_high,
        "n_reference_analytes": 202,
        "fit_population_role": "biological_range",
        "n_fit_population": len(fit_pop),
        "n_calibration_spectra": len(adenine) + len(erg),
        "n_biological_spectra": len(serum) + len(ev),
        "n_total_reference_population": len(all_rows),
        "matrix_composition": matrix_comp,
        "substrate_composition": substrate_comp,
        "population_by_dataset": ref_df["dataset"].value_counts().to_dict(),
        "population_by_role": ref_df["role"].value_counts().to_dict(),
        "labels_used_in_fit": False,
        "limitations": (
            "Fit population is 100% Ag-colloid SERS biological spectra (serum+EV); "
            "center/scale reflect that biological Ag-SERS range. Raman-regime "
            "samples project off-distribution. Axis meaning is fixed by the engine's "
            "motif/MSS definitions, not by this population. Three legacy split families "
            "(purine/lipid/redox) are not independently grounded (see ontology)."),
    }
    content_hash = hashlib.sha256(
        json.dumps(calibration_content, sort_keys=True).encode()).hexdigest()
    calibration_out = dict(calibration_content)
    calibration_out["content_sha256"] = content_hash
    calibration_out["build_timestamp"] = args.timestamp or "UNSET"

    (OUT_DIR / "global_coordinate_calibration_v1.json").write_text(
        json.dumps(calibration_out, indent=2))

    # ── reference samples CSV (labels stored for LATER comparison, not fit) ──
    drop = [c for c in ref_df.columns if c.startswith("raw_") is False and c.startswith("global") is False
            and c not in ("dataset", "sample_id", "label", "matrix", "substrate", "role",
                          "concentration_ng_mL", "concentration_uM")]
    ref_df.drop(columns=[c for c in drop if c in ref_df.columns], errors="ignore")\
          .to_csv(OUT_DIR / "global_coordinate_reference_samples_v1.csv", index=False)

    # ── build manifest ──
    manifest = {
        "ontology_version": onto.version,
        "calibration_version": CAL_VERSION,
        "build_timestamp": args.timestamp or "UNSET",
        "code_module": "tools/build_global_coordinate_reference.py",
        "calibration_content_sha256": content_hash,
        "source_files": {
            "serum_mean_spectra": {"path": str(serum_p), "sha256": _sha256(serum_p), "n": len(serum)},
            "ev_sample_spectra": {"path": str(ev_p), "sha256": _sha256(ev_p), "n": len(ev)},
            "ergothioneine": {"path": str(erg_p), "sha256": _sha256(erg_p), "n": len(erg)},
            "adenine_files": [{"path": str(p), "sha256": _sha256(p)} for p in adenine_files],
        },
        "reference_population": {
            "total": len(all_rows), "biological": len(serum) + len(ev),
            "calibration": len(adenine) + len(erg),
            "by_dataset": ref_df["dataset"].value_counts().to_dict(),
        },
        "determinism": "raw projection + robust stats are deterministic; timestamp excluded from content hash",
    }
    (OUT_DIR / "global_coordinate_build_manifest_v1.json").write_text(
        json.dumps(manifest, indent=2))

    print(f"OK — reference population n={len(all_rows)} "
          f"(serum {len(serum)}, ev {len(ev)}, adenine {len(adenine)}, erg {len(erg)})")
    print(f"content_sha256={content_hash}")
    print("center:", {a: round(center[a], 4) for a in cfg.BSV_AXES})
    print("scale :", {a: round(scale[a], 4) for a in cfg.BSV_AXES})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
